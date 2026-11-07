
import gc
import json
import time
import argparse

from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = BASE_DIR / "benchmark"
IMAGE_DIR = BENCHMARK_DIR

QUESTION_FILE = BENCHMARK_DIR / "questions.jsonl"
RESULT_DIR = BASE_DIR / "results"

MODELS = {
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "smolvlm2": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}

MAX_NEW_TOKENS = 128
IMAGE_SIZE = (336, 336)

# 正式實驗前必須確認所有題目的 Ground Truth
REQUIRE_VERIFIED = True


# ============================================================
# Utility
# ============================================================

def load_questions():
    if not QUESTION_FILE.exists():
        raise FileNotFoundError(
            f"Question file not found: {QUESTION_FILE}"
        )

    questions = []

    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if not questions:
        raise ValueError("No benchmark questions found.")

    if REQUIRE_VERIFIED:
        pending = [
            q["id"]
            for q in questions
            if q.get("ground_truth_verified") is not True
        ]

        if pending:
            raise ValueError(
                "Ground Truth verification is incomplete. "
                f"Pending question IDs: {pending}"
            )

    return questions


def load_image(image_path):
    path = IMAGE_DIR / image_path

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    image.thumbnail(IMAGE_SIZE)

    return image


def get_gpu_memory():
    if not torch.cuda.is_available():
        return {
            "allocated_gb": None,
            "reserved_gb": None,
        }

    return {
        "allocated_gb": round(
            torch.cuda.memory_allocated() / 1024**3, 3
        ),
        "reserved_gb": round(
            torch.cuda.memory_reserved() / 1024**3, 3
        ),
    }


# ============================================================
# Model loading
# ============================================================

def load_model(model_name):
    model_id = MODELS[model_name]

    print(f"\nLoading model: {model_id}")

    torch.cuda.empty_cache()
    gc.collect()

    start = time.time()

    if model_name == "qwen2.5-vl":
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=128 * 128,
            max_pixels=336 * 336,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )

    elif model_name == "smolvlm2":
        processor = AutoProcessor.from_pretrained(model_id)

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    loading_time = time.time() - start

    print(f"Loading time: {loading_time:.2f}s")
    print("GPU memory:", get_gpu_memory())

    return processor, model, loading_time


# ============================================================
# Inference
# ============================================================

def generate_qwen(processor, model, image, prompt):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]

    answer = processor.batch_decode(
        output_ids[:, input_length:],
        skip_special_tokens=True,
    )[0]

    return answer.strip()


def generate_smolvlm(processor, model, image, prompt):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]

    answer = processor.decode(
        output_ids[0][input_length:],
        skip_special_tokens=True,
    )

    return answer.strip()


# ============================================================
# Benchmark runner
# ============================================================

def run_model(model_name, questions):

    processor, model, loading_time = load_model(model_name)

    output_dir = RESULT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "predictions.jsonl"

    print(f"\nRunning {model_name}")
    print(f"Number of questions: {len(questions)}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with open(output_file, "w", encoding="utf-8") as f:

        for index, question in enumerate(questions, start=1):

            image = load_image(question["image"])
            prompt = question["prompt"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.time()

            if model_name == "qwen2.5-vl":
                answer = generate_qwen(
                    processor, model, image, prompt
                )
            else:
                answer = generate_smolvlm(
                    processor, model, image, prompt
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latency = time.time() - start

            memory = get_gpu_memory()

            peak_memory = (
                round(
                    torch.cuda.max_memory_allocated() / 1024**3,
                    3,
                )
                if torch.cuda.is_available()
                else None
            )

            result = {
                "id": question["id"],
                "image": question["image"],
                "category": question["category"],
                "prompt": prompt,
                "answer": answer,
                "latency_seconds": round(latency, 3),
                "gpu_memory_allocated_gb": memory["allocated_gb"],
                "gpu_memory_reserved_gb": memory["reserved_gb"],
                "peak_gpu_memory_gb": peak_memory,
            }

            f.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            f.flush()

            print(
                f"[{index:02d}/{len(questions)}] "
                f"ID={question['id']} | "
                f"Latency={latency:.2f}s | "
                f"Peak GPU={peak_memory}GB"
            )

            print("Answer:", answer)

    summary = {
        "model": MODELS[model_name],
        "loading_time_seconds": round(loading_time, 3),
        "num_questions": len(questions),
        "average_latency_seconds": None,
    }

    with open(output_file, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    summary["average_latency_seconds"] = round(
        sum(r["latency_seconds"] for r in results) / len(results),
        3,
    )

    with open(
        output_dir / "summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSummary:", summary)
    print("Saved:", output_file)

    # Release GPU memory before loading next model
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Released {model_name} GPU memory.")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["qwen2.5-vl", "smolvlm2", "all"],
        default="all",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment.")

    print("GPU:", torch.cuda.get_device_name(0))

    questions = load_questions()

    print(f"Loaded {len(questions)} benchmark questions.")

    selected_models = (
        list(MODELS.keys())
        if args.model == "all"
        else [args.model]
    )

    for model_name in selected_models:
        run_model(model_name, questions)

    print("\nAll benchmark experiments completed!")


if __name__ == "__main__":
    main()
