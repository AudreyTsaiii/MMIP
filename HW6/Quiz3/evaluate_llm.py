
import argparse
import json
import re
import gc
from pathlib import Path
from collections import defaultdict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# --------------------------------------------------
# Evaluation rules
# --------------------------------------------------

ABSTAIN_MARKERS = [
    "未說明",
    "未提供",
    "沒有說明",
    "沒有提供",
    "無法確認",
    "無法判斷",
    "無法得知",
    "無法確定",
    "資訊不足",
]

REFUND_CHANNELS = [
    "購物金",
    "信用卡",
    "銀行帳戶",
    "現金",
    "原付款方式",
    "匯款",
]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tokenizer, user_prompt):
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )


def generate_answer(model, tokenizer, prompt, max_new_tokens=96):
    inputs = build_prompt(tokenizer, prompt)

    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    input_length = inputs["input_ids"].shape[1]

    answer = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    ).strip()

    return answer


# --------------------------------------------------
# General ability evaluation
# --------------------------------------------------

def extract_choice(response, options):
    # Prefer an explicit answer label, such as "答案：C"
    patterns = [
        r"(?:答案|選項|answer)\s*[:：是為]?\s*([ABCD])\b",
        r"^\s*([ABCD])\s*[.、，：:)]",
        r"^\s*([ABCD])\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback: match the option text
    for i, option in enumerate(options):
        if option and option in response:
            return "ABCD"[i]

    return None


def evaluate_general(model, tokenizer, dataset):
    results = []
    category_results = defaultdict(lambda: {"correct": 0, "total": 0})

    correct = 0

    for i, item in enumerate(dataset):
        options = item["options"]

        option_text = "\n".join(
            f"{'ABCD'[j]}. {option}"
            for j, option in enumerate(options)
        )

        prompt = (
            "請回答以下選擇題。"
            "請只輸出正確選項的英文字母 A、B、C 或 D，不需要解釋。\n\n"
            f"題目：{item['prompt']}\n\n"
            f"選項：\n{option_text}"
        )

        response = generate_answer(
            model, tokenizer, prompt, max_new_tokens=16
        )

        predicted = extract_choice(response, options)
        expected = item["answer"].strip().upper()

        is_correct = predicted == expected

        correct += int(is_correct)

        category = item["category"]
        category_results[category]["total"] += 1
        category_results[category]["correct"] += int(is_correct)

        results.append({
            "id": item["id"],
            "category": category,
            "expected": expected,
            "predicted": predicted,
            "response": response,
            "correct": is_correct,
        })

        print(
            f"[General {i + 1}/{len(dataset)}] "
            f"{item['id']} | expected={expected} "
            f"predicted={predicted} | "
            f"{'Correct' if is_correct else 'Wrong'}"
        )

    accuracy = correct / len(dataset) if dataset else 0

    category_accuracy = {
        category: values["correct"] / values["total"]
        for category, values in category_results.items()
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(dataset),
        "category_accuracy": category_accuracy,
        "predictions": results,
    }


# --------------------------------------------------
# Target ability evaluation
# --------------------------------------------------

def evaluate_target_answer(response, category, expected_answer):
    response = response.strip()

    has_abstention = any(
        marker in response for marker in ABSTAIN_MARKERS
    )

    mentions_refund_channel = any(
        channel in response for channel in REFUND_CHANNELS
    )

    if category == "stated":
        # The answer must contain the policy's specified refund method.
        expected_keywords = [
            keyword
            for keyword in REFUND_CHANNELS
            if keyword in expected_answer
        ]

        is_correct = (
            bool(expected_keywords)
            and all(keyword in response for keyword in expected_keywords)
            and not has_abstention
        )

    elif category == "unstated":
        # The answer must acknowledge missing information,
        # without inventing a refund destination.
        is_correct = has_abstention and not mentions_refund_channel

    else:
        raise ValueError(f"Unknown target category: {category}")

    return is_correct


def evaluate_target(model, tokenizer, dataset):
    results = []
    category_results = defaultdict(lambda: {"correct": 0, "total": 0})

    correct = 0

    for i, item in enumerate(dataset):
        response = generate_answer(
            model,
            tokenizer,
            item["prompt"],
            max_new_tokens=96,
        )

        is_correct = evaluate_target_answer(
            response,
            item["category"],
            item["answer"],
        )

        correct += int(is_correct)

        category = item["category"]
        category_results[category]["total"] += 1
        category_results[category]["correct"] += int(is_correct)

        results.append({
            "id": item["id"],
            "category": category,
            "expected": item["answer"],
            "response": response,
            "correct": is_correct,
        })

        print(
            f"[Target {i + 1}/{len(dataset)}] "
            f"{item['id']} | category={category} | "
            f"{'Correct' if is_correct else 'Wrong'}"
        )

    accuracy = correct / len(dataset) if dataset else 0

    category_accuracy = {
        category: values["correct"] / values["total"]
        for category, values in category_results.items()
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(dataset),
        "category_accuracy": category_accuracy,
        "predictions": results,
    }


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model(base_model, tokenizer, adapter_path=None):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )

    model.eval()
    return model


# --------------------------------------------------
# Main
# --------------------------------------------------

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument(
        "--general_eval",
        default="data/general_eval.jsonl",
    )
    parser.add_argument(
        "--target_eval",
        default="data/target_eval.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="results/evaluation",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    general_data = load_jsonl(args.general_eval)
    target_data = load_jsonl(args.target_eval)

    print(f"General evaluation questions: {len(general_data)}")
    print(f"Target evaluation questions: {len(target_data)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    summaries = {}

    # Evaluate Base Model first, then DPO Model.
    models_to_evaluate = [
        ("base", None),
        ("dpo", args.adapter_path),
    ]

    for model_name, adapter_path in models_to_evaluate:
        print("\n" + "=" * 60)
        print(f"Evaluating {model_name.upper()} Model")
        print("=" * 60)

        model = load_model(
            args.base_model,
            tokenizer,
            adapter_path,
        )

        general_result = evaluate_general(
            model, tokenizer, general_data
        )

        target_result = evaluate_target(
            model, tokenizer, target_data
        )

        summaries[model_name] = {
            "general": {
                "accuracy": general_result["accuracy"],
                "correct": general_result["correct"],
                "total": general_result["total"],
                "category_accuracy": general_result["category_accuracy"],
            },
            "target": {
                "accuracy": target_result["accuracy"],
                "correct": target_result["correct"],
                "total": target_result["total"],
                "category_accuracy": target_result["category_accuracy"],
            },
        }

        save_json(
            general_result["predictions"],
            output_dir / f"{model_name}_general_predictions.json",
        )

        save_json(
            target_result["predictions"],
            output_dir / f"{model_name}_target_predictions.json",
        )

        print("\nGeneral Accuracy:",
              f"{general_result['accuracy']:.2%}")
        print("Target Accuracy:",
              f"{target_result['accuracy']:.2%}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    save_json(summaries, output_dir / "summary.json")

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    for task in ["general", "target"]:
        base_acc = summaries["base"][task]["accuracy"]
        dpo_acc = summaries["dpo"][task]["accuracy"]

        print(f"\n{task.upper()}")
        print(f"Base Model: {base_acc:.2%}")
        print(f"DPO Model:  {dpo_acc:.2%}")
        print(f"Difference: {(dpo_acc - base_acc) * 100:+.2f} percentage points")

        if task == "target":
            for category in ["stated", "unstated"]:
                base_cat = summaries["base"][task]["category_accuracy"].get(category, 0)
                dpo_cat = summaries["dpo"][task]["category_accuracy"].get(category, 0)

                print(
                    f"  {category}: "
                    f"Base={base_cat:.2%}, "
                    f"DPO={dpo_cat:.2%}"
                )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
