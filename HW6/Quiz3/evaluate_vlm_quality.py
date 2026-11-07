import argparse
import csv
import json
import gc
import re
import time
from pathlib import Path
from collections import defaultdict
from coco_scoring import parse_predictions, extract_objects, summarize, VERSION

import torch
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--adapter", default="outputs/vlm_dpo")
parser.add_argument("--output-dir", type=Path, default=Path("results/vlm_quality"))
parser.add_argument("--data-root", type=Path, default=Path("data/vlm"))
parser.add_argument("--limit", type=int, default=0, help="0 uses all target images")
args = parser.parse_args()
if args.limit < 0:
    parser.error("limit must be nonnegative")
if not (Path(args.adapter) / "adapter_config.json").is_file():
    parser.error("adapter directory must contain adapter_config.json")
if args.output_dir.exists() and any(args.output_dir.iterdir()):
    parser.error("Choose an empty output directory")
MODEL_NAME = args.model
ADAPTER = args.adapter
OUTPUT_DIR = args.output_dir
IMAGE_ROOT = args.data_root / "images"
CONDITIONS = ("original", "dark_0.5", "resolution_0.25", "blur_2")

def transform(image, condition):
    if condition == "original":
        return image.copy()
    if condition == "dark_0.5":
        return ImageEnhance.Brightness(image).enhance(0.5)
    if condition == "resolution_0.25":
        w, h = image.size
        return image.resize((max(1, w // 4), max(1, h // 4)), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)
    if condition == "blur_2":
        return image.filter(ImageFilter.GaussianBlur(radius=2))
    raise ValueError(condition)

MAX_PIXELS = 512 * 28 * 28
MAX_NEW_TOKENS = 100

torch.manual_seed(42)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

device = "cuda"

# COCO category names
def load_split(split):
    annotation_file = args.data_root / "annotations" / f"{split}.json"
    with annotation_file.open(encoding="utf-8") as f:
        coco = json.load(f)

    categories = {
        x["id"]: x["name"] for x in coco["categories"]
    }

    image_objects = defaultdict(set)
    for ann in coco["annotations"]:
        image_objects[ann["image_id"]].add(
            categories[ann["category_id"]]
        )

    samples = []
    for img in coco["images"]:
        samples.append({
            "filename": img["file_name"],
            "image_id": img["id"],
            "image_path": IMAGE_ROOT / split / img["file_name"],
            "ground_truth": sorted(
                image_objects[img["id"]]
            ),
        })

    return samples, sorted(set(categories.values()))


def load_model(use_adapter):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    if use_adapter:
        model = PeftModel.from_pretrained(
            model, ADAPTER
        )

    model.eval()
    return model


def run_inference(model, processor, sample, categories):
    with Image.open(sample["image_path"]) as source:
        image = transform(source.convert("RGB"), sample["condition"])

    prompt = (
        "請仔細觀察圖片，辨識圖片中實際可見的物件。"
        "請只輸出 COCO 80 類別中的英文名稱，以逗號分隔。"
        "只列出有明確視覺證據的物件，不要猜測。"
        "若無法確認，請不要列出。"
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    input_length = inputs["input_ids"].shape[1]
    generated = output[:, input_length:]

    answer = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    predicted = parse_predictions(answer, categories)

    del image, inputs, output, generated

    return answer, predicted, latency


def evaluate_model(model_name, use_adapter, processor, splits):
    print("\n" + "=" * 60)
    print("Evaluating:", model_name)
    print("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = load_model(use_adapter)

    all_results = {}
    summary = {}

    for split, samples in splits.items():
        print(f"\nEvaluating {split}: {len(samples)} images")

        predictions = []
        total_latency = 0.0
        tp = fp = fn = 0

        for sample in tqdm(samples, desc=f"{model_name}-{split}"):
            answer, predicted, latency = run_inference(
                model, processor, sample, categories
            )

            gt = set(sample["ground_truth"])
            pred = set(predicted)

            current_tp = len(gt & pred)
            current_fp = len(pred - gt)
            current_fn = len(gt - pred)

            tp += current_tp
            fp += current_fp
            fn += current_fn
            total_latency += latency

            predictions.append({
                "image": sample["filename"],
                "ground_truth": sorted(gt),
                "predicted": sorted(pred),
                "raw_answer": answer,
                **extract_objects(answer, categories),
                "tp": current_tp,
                "fp": current_fp,
                "fn": current_fn,
                "latency": round(latency, 4),
            })

        all_results[split] = predictions
        summary[split] = summarize(predictions)
        (OUTPUT_DIR / f"{'dpo' if use_adapter else 'base'}_{split}.json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")


        print(json.dumps(
            summary[split], indent=2, ensure_ascii=False
        ))

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    result = {
        "model": model_name,
        "scoring_version": VERSION,
        "peak_gpu_memory_gb": round(peak_memory, 3),
        "splits": summary,
        "predictions": all_results,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# Load evaluation data
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

splits = {}
categories = None

samples, categories = load_split("target_eval")
samples = sorted(samples, key=lambda x: x["filename"])
if args.limit:
    samples = samples[:args.limit]
if not samples:
    raise ValueError("No target images")
for sample in samples:
    if not sample["image_path"].is_file():
        raise FileNotFoundError(sample["image_path"])
for condition in CONDITIONS:
    splits[condition] = [dict(sample, condition=condition) for sample in samples]

# Save transform previews and experiment settings before starting inference.
with Image.open(samples[0]["image_path"]) as source:
    original = source.convert("RGB")
for condition in CONDITIONS:
    transform(original, condition).save(OUTPUT_DIR / f"preview_{condition}.png")
(OUTPUT_DIR / "settings.json").write_text(json.dumps({
    "model": MODEL_NAME, "adapter": str(Path(ADAPTER).resolve()),
    "images": [r["filename"] for r in samples], "conditions": list(CONDITIONS),
    "brightness_factor": 0.5, "resolution_scale": 0.25, "blur_radius": 2,
    "resize_filter": "bilinear", "scoring_version": VERSION,
    "max_pixels": MAX_PIXELS, "min_pixels": 256 * 28 * 28,
    "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
    "note": "Original COCO annotations are shared across conditions. Scores measure robustness to corruption."
}, ensure_ascii=False, indent=2), encoding="utf-8")

print("COCO categories:", len(categories))

# ============================================================
# Processor
# ============================================================

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=256 * 28 * 28,
    max_pixels=MAX_PIXELS,
)

# ============================================================
# Base model
# ============================================================

base_result = evaluate_model(
    "Base Qwen2.5-VL-3B",
    False,
    processor,
    splits,
)

with open(OUTPUT_DIR / "base_predictions.json", "w", encoding="utf-8") as f:
    json.dump(base_result, f, ensure_ascii=False, indent=2)

# ============================================================
# DPO model
# ============================================================

dpo_result = evaluate_model(
    "DPO Qwen2.5-VL-3B",
    True,
    processor,
    splits,
)

with open(OUTPUT_DIR / "dpo_predictions.json", "w", encoding="utf-8") as f:
    json.dump(dpo_result, f, ensure_ascii=False, indent=2)

# ============================================================
# Comparison
# ============================================================

comparison = {}

for split in splits:
    comparison[split] = {}

    for metric in [
        "precision",
        "recall",
        "f1",
        "hallucination_rate",
        "average_latency_sec",
        "images_with_unmatched_predicted_categories_rate",
        "empty_prediction_rate",
    ]:
        base_value = base_result["splits"][split][metric]
        dpo_value = dpo_result["splits"][split][metric]

        comparison[split][metric] = {
            "base": base_value,
            "dpo": dpo_value,
            "change": round(dpo_value - base_value, 4) if dpo_value is not None and base_value is not None else None,
        }

comparison["peak_gpu_memory_gb"] = {
    "base": base_result["peak_gpu_memory_gb"],
    "dpo": dpo_result["peak_gpu_memory_gb"],
}

with open(OUTPUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 65)
print("BASE vs DPO COMPARISON")
print("=" * 65)

for split, metrics in comparison.items():
    if split == "peak_gpu_memory_gb":
        continue

    print(f"\n{split}")
    for metric, values in metrics.items():
        if any(values[k] is None for k in ('base', 'dpo', 'change')):
            print(f"{metric}: undefined (empty denominator)")
            continue
        print(
            f"{metric:24s} "
            f"Base={values['base']:.4f} "
            f"DPO={values['dpo']:.4f} "
            f"Change={values['change']:+.4f}"
        )

print("\nPeak GPU memory:")
print(comparison["peak_gpu_memory_gb"])

print("\nEvaluation completed.")
print("Results saved to:", OUTPUT_DIR)



with (OUTPUT_DIR / "quality_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["condition", "metric", "base", "dpo", "dpo_minus_base",
                     "base_minus_original", "dpo_minus_original"])
    for condition in CONDITIONS:
        for metric, values in comparison[condition].items():
            original_values = comparison["original"][metric]
            delta = lambda a, b: a-b if a is not None and b is not None else None
            writer.writerow([condition, metric, values["base"], values["dpo"], values["change"],
                             delta(values["base"], original_values["base"]),
                             delta(values["dpo"], original_values["dpo"])])
print("Table saved to:", OUTPUT_DIR / "quality_summary.csv")
