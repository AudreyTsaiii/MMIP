import json
import time
import threading
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-1.7B"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "benchmark" / "questions.jsonl"
OUTPUT_PATH = BASE_DIR / "results" / "cot_fp16.jsonl"
SUMMARY_PATH = BASE_DIR / "results" / "cot_fp16_summary.json"

# Original Baseline System Prompt + CoT instruction
SYSTEM_PROMPT = (
    "你是一個可靠且專業的 AI 助理。"
    "請使用繁體中文回答，內容應正確、清楚並符合使用者的要求。"
    "若題目要求特定格式，請嚴格遵守。"
    "請仔細閱讀題目，先確認題目要求與限制。"
    "若題目涉及計算或推理，請先完成必要的分析與檢查。"
    "最後提供清楚、正確且符合題目格式的答案。"
    "請呈現簡潔的解題依據，避免不必要的冗長說明。"
)

MAX_NEW_TOKENS = 384
SAMPLE_INTERVAL = 0.05


def load_questions():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_answer(model, tokenizer, question):
    prompt = format_prompt(tokenizer, question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return answer.strip(), elapsed


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    print(f"Loading model: {MODEL_NAME}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    torch.cuda.synchronize()
    load_time = time.perf_counter() - load_start
    model_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)

    print(f"Model loading time: {load_time:.2f} sec")
    print(f"GPU memory after loading: {model_memory_gb:.3f} GB")

    # Warm-up: exclude the first generation from benchmark results.
    print("Running warm-up...")
    _ = generate_answer(model, tokenizer, "請簡單介紹機器學習。")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    questions = load_questions()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    memory_samples = []

    print(f"\nRunning CoT benchmark: {len(questions)} questions")

    for index, item in enumerate(questions, start=1):
        samples = []
        stop_sampling = threading.Event()

        def sample_memory():
            while not stop_sampling.is_set():
                samples.append(torch.cuda.memory_allocated() / (1024 ** 3))
                time.sleep(SAMPLE_INTERVAL)

        sampler = threading.Thread(target=sample_memory)
        sampler.start()

        try:
            answer, elapsed = generate_answer(
                model,
                tokenizer,
                item["question"],
            )
        finally:
            stop_sampling.set()
            sampler.join()

        memory_samples.extend(samples)

        result = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answer": answer,
            "reference_answer": item["reference_answer"],
            "criteria": item["criteria"],
            "latency_seconds": round(elapsed, 4),
            "sampled_avg_memory_gb": (
                round(sum(samples) / len(samples), 4) if samples else None
            ),
        }
        results.append(result)

        print(
            f"[{index:02d}/{len(questions)}] "
            f"{item['id']} | {elapsed:.2f} sec | "
            f"{len(answer)} characters"
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    latencies = [r["latency_seconds"] for r in results]
    average_latency = sum(latencies) / len(latencies)

    summary = {
        "model": MODEL_NAME,
        "precision": "FP16",
        "prompt_strategy": "Chain-of-Thought (CoT)",
        "decoding": "Greedy Decoding",
        "enable_thinking": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_questions": len(results),
        "model_load_time_seconds": round(load_time, 4),
        "model_memory_after_loading_gb": round(model_memory_gb, 4),
        "average_latency_seconds": round(average_latency, 4),
        "total_generation_time_seconds": round(sum(latencies), 4),
        "average_sampled_gpu_memory_gb": (
            round(sum(memory_samples) / len(memory_samples), 4)
            if memory_samples else None
        ),
        "peak_gpu_memory_gb": round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 4
        ),
        "memory_sampling_interval_seconds": SAMPLE_INTERVAL,
        "memory_measurement": "PyTorch allocated GPU memory",
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== FP16 CoT Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nPredictions saved to: {OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
