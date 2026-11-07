import json
import gc
import re
import time
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER = "outputs/vlm_dpo"
OUTPUT_DIR = Path("results/vlm_dpo")
IMAGE_ROOT = Path("data/vlm/images")

MAX_PIXELS = 512 * 28 * 28
MAX_NEW_TOKENS = 100

torch.manual_seed(42)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

device = "cuda"

# COCO category names
def load_split(split):
    annotation_file = Path(f"data/vlm/annotations/{split}.json")
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


def parse_predictions(text, categories):
    text = text.lower()
    found = []

    for category in categories:
        pattern = (
            r"(?<![a-z])"
            + re.escape(category.lower())
            + r"(?![a-z])"
        )
        if re.search(pattern, text):
            found.append(category)

    return sorted(set(found))


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
    image = Image.open(sample["image_path"]).convert("RGB")

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
                "tp": current_tp,
                "fp": current_fp,
                "fn": current_fn,
                "latency": round(latency, 4),
            })

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )
        hallucination_rate = fp / (tp + fp) if tp + fp else 0
        avg_latency = total_latency / len(samples)

        all_results[split] = predictions
        summary[split] = {
            "images": len(samples),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "average_latency_sec": round(avg_latency, 4),
        }

        print(json.dumps(
            summary[split], indent=2, ensure_ascii=False
        ))

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    result = {
        "model": model_name,
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

for split in ["target_eval", "general_eval"]:
    samples, current_categories = load_split(split)
    splits[split] = samples
    categories = current_categories

    print(f"{split}: {len(samples)} images")

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
    ]:
        base_value = base_result["splits"][split][metric]
        dpo_value = dpo_result["splits"][split][metric]

        comparison[split][metric] = {
            "base": base_value,
            "dpo": dpo_value,
            "change": round(dpo_value - base_value, 4),
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
