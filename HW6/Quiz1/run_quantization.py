import argparse
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

SYSTEM_PROMPT = (
    "你是一個可靠且專業的 AI 助理。"
    "請使用繁體中文回答，內容應正確、清楚並符合使用者的要求。"
    "若題目要求特定格式，請嚴格遵守。"
)

MAX_NEW_TOKENS = 384
MEMORY_SAMPLE_INTERVAL = 0.05


# --------------------------------------------------
# Utility
# --------------------------------------------------

def load_questions(path):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


class GPUMemorySampler:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        while not self.stop_event.is_set():
            memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
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

def load_model(bits):
    print(f"Loading model: {MODEL_NAME}")
    print(f"Quantization: INT{bits}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if bits == 8:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError("bits must be 4 or 8")

    start_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.eval()
    torch.cuda.synchronize()

    load_time = time.time() - start_time
    memory_after_loading = (
        torch.cuda.memory_allocated(0) / (1024 ** 3)
    )

    print(f"Model loading time: {load_time:.2f} seconds")
    print(f"GPU memory after loading: {memory_after_loading:.4f} GB")

    return tokenizer, model, load_time, memory_after_loading


# --------------------------------------------------
# Inference
# --------------------------------------------------

def generate_answer(model, tokenizer, question):
    prompt = build_prompt(tokenizer, question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(0)

    sampler = GPUMemorySampler(MEMORY_SAMPLE_INTERVAL)
    sampler.start()

    start_time = time.time()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    latency = time.time() - start_time

    average_memory = sampler.stop()

    peak_memory = (
        torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return {
        "answer": answer,
        "latency_seconds": latency,
        "output_tokens": len(new_tokens),
        "answer_chars": len(answer),
        "average_gpu_memory_gb": average_memory,
        "peak_gpu_memory_gb": peak_memory,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        required=True,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer, model, load_time, memory_after_loading = (
        load_model(args.bits)
    )

    questions = load_questions(QUESTION_PATH)

    print(f"\nNumber of questions: {len(questions)}")

    # Warm-up
    print("\nRunning warm-up...")
    generate_answer(model, tokenizer, questions[0]["question"])

    torch.cuda.synchronize()
    print("Warm-up completed.\n")

    output_path = RESULT_DIR / f"quant_int{args.bits}.jsonl"
    summary_path = RESULT_DIR / f"quant_int{args.bits}_summary.json"

    results = []
    latencies = []
    average_memories = []
    peak_memories = []
    total_generation_time = 0.0

    with open(output_path, "w", encoding="utf-8") as f:
        for index, item in enumerate(questions, start=1):
            print(
                f"[{index}/{len(questions)}] "
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
                "reference_answer": item["reference_answer"],
                "criteria": item["criteria"],
                **result,
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
        "precision": f"INT{args.bits}",
        "quantization": (
            "8-bit" if args.bits == 8
            else "4-bit NF4 with double quantization"
        ),
        "prompt_strategy": "Baseline Prompt",
        "decoding": "Greedy Decoding",
        "enable_thinking": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_questions": len(results),
        "model_load_time_seconds": load_time,
        "model_memory_after_loading_gb": memory_after_loading,
        "average_latency_seconds": sum(latencies) / len(latencies),
        "total_generation_time_seconds": total_generation_time,
        "average_sampled_gpu_memory_gb": (
            sum(average_memories) / len(average_memories)
        ),
        "peak_gpu_memory_gb": max(peak_memories),
        "memory_sampling_interval_seconds": MEMORY_SAMPLE_INTERVAL,
        "memory_measurement": "PyTorch allocated GPU memory",
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Quantization Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
