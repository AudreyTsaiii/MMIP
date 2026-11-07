
import json
import time
import threading
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-1.7B"

BASE_DIR = Path(__file__).resolve().parent
QUESTION_PATH = BASE_DIR / "benchmark" / "questions.jsonl"
RESULT_DIR = BASE_DIR / "results"

OUTPUT_PATH = RESULT_DIR / "quant_int4_cot.jsonl"
SUMMARY_PATH = RESULT_DIR / "quant_int4_cot_summary.json"

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
MEMORY_SAMPLE_INTERVAL = 0.05


# --------------------------------------------------
# Utility
# --------------------------------------------------

def load_questions():
    with QUESTION_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tokenizer, question):
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


class GPUMemorySampler:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        while not self.stop_event.is_set():
            memory = (
                torch.cuda.memory_allocated(0) / (1024 ** 3)
            )
            self.samples.append(memory)
            time.sleep(self.interval)

    def start(self):
        self.samples = []
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._sample)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

        if not self.samples:
            return 0.0

        return sum(self.samples) / len(self.samples)


# --------------------------------------------------
# Model
# --------------------------------------------------

def load_model():
    print(f"Loading model: {MODEL_NAME}")
    print("Quantization: INT4 NF4 + Double Quantization")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    start_time = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.eval()
    torch.cuda.synchronize()

    load_time = time.perf_counter() - start_time

    memory_after_loading = (
        torch.cuda.memory_allocated(0) / (1024 ** 3)
    )

    print(f"Model loading time: {load_time:.2f} seconds")
    print(
        f"GPU memory after loading: "
        f"{memory_after_loading:.4f} GB"
    )

    return tokenizer, model, load_time, memory_after_loading


# --------------------------------------------------
# Inference
# --------------------------------------------------

def generate_answer(model, tokenizer, question):
    prompt = build_prompt(tokenizer, question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_attention_mask=True,
    ).to("cuda")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)

    sampler = GPUMemorySampler(MEMORY_SAMPLE_INTERVAL)
    sampler.start()

    start_time = time.perf_counter()

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        latency = time.perf_counter() - start_time

    finally:
        average_memory = sampler.stop()

    peak_memory = (
        torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return {
        "answer": answer,
        "latency_seconds": round(latency, 4),
        "output_tokens": len(new_tokens),
        "answer_chars": len(answer),
        "average_gpu_memory_gb": round(average_memory, 4),
        "peak_gpu_memory_gb": round(peak_memory, 4),
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer, model, load_time, memory_after_loading = (
        load_model()
    )

    questions = load_questions()

    if not questions:
        raise ValueError("No benchmark questions found.")

    print(f"\nNumber of questions: {len(questions)}")

    # Warm-up: exclude the first generation from benchmark results.
    print("\nRunning warm-up...")
    generate_answer(
        model,
        tokenizer,
        questions[0]["question"],
    )

    torch.cuda.synchronize()
    print("Warm-up completed.\n")

    results = []
    latencies = []
    average_memories = []
    peak_memories = []
    total_generation_time = 0.0

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for index, item in enumerate(questions, start=1):

            print(
                f"[{index:02d}/{len(questions)}] "
                f"{item['id']} - {item['category']}"
            )

            result = generate_answer(
                model,
                tokenizer,
                item["question"],
            )

            record = {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": result["answer"],
                "reference_answer": item["reference_answer"],
                "criteria": item["criteria"],
                **{
                    key: value
                    for key, value in result.items()
                    if key != "answer"
                },
            }

            results.append(record)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            latencies.append(result["latency_seconds"])
            average_memories.append(
                result["average_gpu_memory_gb"]
            )
            peak_memories.append(
                result["peak_gpu_memory_gb"]
            )

            total_generation_time += result["latency_seconds"]

            print(
                f"  Latency: {result['latency_seconds']:.2f}s | "
                f"Average Memory: "
                f"{result['average_gpu_memory_gb']:.4f} GB | "
                f"Peak Memory: "
                f"{result['peak_gpu_memory_gb']:.4f} GB"
            )

    summary = {
        "model": MODEL_NAME,
        "precision": "INT4",
        "quantization": "4-bit NF4 with double quantization",
        "prompt_strategy": "Chain-of-Thought (CoT)",
        "decoding": "Greedy Decoding",
        "enable_thinking": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_questions": len(results),
        "model_load_time_seconds": round(load_time, 4),
        "model_memory_after_loading_gb": round(
            memory_after_loading, 4
        ),
        "average_latency_seconds": round(
            sum(latencies) / len(latencies), 4
        ),
        "total_generation_time_seconds": round(
            total_generation_time, 4
        ),
        "average_sampled_gpu_memory_gb": round(
            sum(average_memories) / len(average_memories), 4
        ),
        "peak_gpu_memory_gb": round(max(peak_memories), 4),
        "memory_sampling_interval_seconds": (
            MEMORY_SAMPLE_INTERVAL
        ),
        "memory_measurement": "PyTorch allocated GPU memory",
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== INT4 + CoT Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
