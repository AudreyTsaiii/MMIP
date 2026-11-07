
import json
import os
import random
import re
import time
from pathlib import Path

from google import genai
from google.genai import types


# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "results"

INPUT_PATHS = {
    "FP16": RESULT_DIR / "baseline_fp16.jsonl",
    "INT8": RESULT_DIR / "quant_int8.jsonl",
    "INT4": RESULT_DIR / "quant_int4.jsonl",
}

OUTPUT_PATH = RESULT_DIR / "quant_judge_results.jsonl"
SUMMARY_PATH = RESULT_DIR / "quant_judge_summary.json"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
RANDOM_SEED = 42
MAX_RETRIES = 3

METRICS = [
    "correctness",
    "completeness",
    "instruction_following",
    "overall",
]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, results):
    with path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Cannot parse Gemini response:\n{text}")


def validate_scores(data):
    if "scores" not in data:
        raise ValueError("Missing scores")

    for label in ["A", "B", "C"]:
        if label not in data["scores"]:
            raise ValueError(f"Missing scores for {label}")

        for metric in METRICS:
            score = data["scores"][label].get(metric)

            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not 1 <= score <= 5
            ):
                raise ValueError(
                    f"Invalid score: {label}.{metric}={score}"
                )

    if data.get("winner") not in ["A", "B", "C", "Tie"]:
        raise ValueError(f"Invalid winner: {data.get('winner')}")

    return data


def build_judge_prompt(question, reference, criteria, answers):
    return f"""
你是一位嚴謹且公平的 AI 模型評估員。
請使用繁體中文理解題目與評分標準。

請評估以下三個模型對同一道題目的回答。

【題目】
{question}

【參考答案】
{reference}

【評分標準】
{criteria}

【回答 A】
{answers["A"]}

【回答 B】
{answers["B"]}

【回答 C】
{answers["C"]}

請分別為 A、B、C 評分，每個面向均為 1 至 5 分：

1. correctness：正確性。
2. completeness：完整性。
3. instruction_following：指令遵循。
4. overall：整體品質。

評分原則：
- 5 分：表現優秀，幾乎完全符合要求。
- 4 分：大致正確完整，只有輕微缺失。
- 3 分：部分符合要求，但有明顯缺失。
- 2 分：存在多項錯誤或重要遺漏。
- 1 分：嚴重錯誤，或幾乎沒有回答題目。
- 不要因為回答比較長就給比較高的分數。
- 不要因為回答有條列或推理過程就自動加分。
- 如果題目要求特定格式，應嚴格依照格式評分。
- 參考答案是評分依據，但其他正確方法也應給予合理分數。
- 請獨立評估三個回答，不要因為回答順序或模型名稱而偏袒任何一方。
- 最後依據整體品質判斷 winner。
- 如果整體品質相當，請選擇 Tie。

winner 請填：
- "A"：回答 A 整體最好
- "B"：回答 B 整體最好
- "C"：回答 C 整體最好
- "Tie"：三者整體品質相當

請只輸出合法 JSON，不要輸出 Markdown 或其他文字。
格式如下：
{{
  "scores": {{
    "A": {{
      "correctness": 1,
      "completeness": 1,
      "instruction_following": 1,
      "overall": 1
    }},
    "B": {{
      "correctness": 1,
      "completeness": 1,
      "instruction_following": 1,
      "overall": 1
    }},
    "C": {{
      "correctness": 1,
      "completeness": 1,
      "instruction_following": 1,
      "overall": 1
    }}
  }},
  "winner": "A",
  "reason": "簡短說明判斷理由"
}}
""".strip()


def judge_one(client, items, rng):
    # Randomize the order of the three answers.
    labels = ["A", "B", "C"]

    models = ["FP16", "INT8", "INT4"]
    rng.shuffle(models)

    model_to_label = {
        model: label
        for model, label in zip(models, labels)
    }

    label_to_model = {
        label: model
        for model, label in model_to_label.items()
    }

    answers = {
        label: items[model]["answer"]
        for label, model in label_to_model.items()
    }

    reference_item = items["FP16"]

    prompt = build_judge_prompt(
        question=reference_item["question"],
        reference=reference_item["reference_answer"],
        criteria=reference_item["criteria"],
        answers=answers,
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

            data = validate_scores(parse_json(response.text))

            scores = {
                label_to_model[label]: data["scores"][label]
                for label in labels
            }

            winner = data["winner"]
            mapped_winner = (
                "Tie" if winner == "Tie"
                else label_to_model[winner]
            )

            return {
                "id": reference_item["id"],
                "category": reference_item["category"],
                "scores": scores,
                "winner": mapped_winner,
                "reason": data.get("reason", ""),
                "answer_order": models,
            }

        except Exception as exc:
            last_error = exc
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {exc}")

            if attempt < MAX_RETRIES:
                wait_seconds = (
                    65 if "429" in str(exc)
                    else 10 * attempt
                )
                print(f"  Waiting {wait_seconds} seconds...")
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Gemini evaluation failed for {reference_item['id']}: "
        f"{last_error}"
    )


def calculate_summary(results):
    summary = {
        "model": GEMINI_MODEL,
        "num_questions": len(results),
        "scores": {},
        "wins": {
            "FP16": 0,
            "INT8": 0,
            "INT4": 0,
            "Tie": 0,
        },
    }

    models = ["FP16", "INT8", "INT4"]

    for model in models:
        summary["scores"][model] = {}

        for metric in METRICS:
            values = [
                row["scores"][model][metric]
                for row in results
            ]

            summary["scores"][model][metric] = round(
                sum(values) / len(values), 4
            )

    for row in results:
        summary["wins"][row["winner"]] += 1

    summary["category_scores"] = {}

    categories = sorted(set(row["category"] for row in results))

    for category in categories:
        category_results = [
            row for row in results
            if row["category"] == category
        ]

        summary["category_scores"][category] = {}

        for model in models:
            values = [
                row["scores"][model]["overall"]
                for row in category_results
            ]

            summary["category_scores"][category][model] = round(
                sum(values) / len(values), 4
            )

    return summary


def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    # Load all three model results.
    datasets = {
        model: load_jsonl(path)
        for model, path in INPUT_PATHS.items()
    }

    data_by_model = {
        model: {item["id"]: item for item in data}
        for model, data in datasets.items()
    }

    if not datasets["FP16"]:
        raise ValueError("No benchmark results found.")

    expected_ids = set(data_by_model["FP16"])

    for model in ["INT8", "INT4"]:
        if set(data_by_model[model]) != expected_ids:
            raise ValueError(
                f"Question IDs do not match for {model}."
            )

    for question_id in expected_ids:
        reference = data_by_model["FP16"][question_id]

        for model in ["INT8", "INT4"]:
            item = data_by_model[model][question_id]

            if item["question"] != reference["question"]:
                raise ValueError(
                    f"Question mismatch: {question_id}, {model}"
                )

    client = genai.Client(api_key=api_key)
    rng = random.Random(RANDOM_SEED)

    # Resume from previous results.
    results = []

    if OUTPUT_PATH.exists():
        results = load_jsonl(OUTPUT_PATH)

    completed_ids = {row["id"] for row in results}

    print(f"Gemini Judge model: {GEMINI_MODEL}")
    print(f"Total questions: {len(expected_ids)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {len(expected_ids) - len(completed_ids)}")

    fp16_data = datasets["FP16"]

    for index, reference_item in enumerate(fp16_data, start=1):
        question_id = reference_item["id"]

        if question_id in completed_ids:
            print(f"Skipping {question_id} (already evaluated)")
            continue

        items = {
            model: data_by_model[model][question_id]
            for model in ["FP16", "INT8", "INT4"]
        }

        print(f"\n[{index:02d}/{len(fp16_data)}] Evaluating {question_id}")

        # Stay below Gemini free-tier request limit.
        time.sleep(6)

        result = judge_one(client, items, rng)
        results.append(result)
        completed_ids.add(question_id)

        for model in ["FP16", "INT8", "INT4"]:
            print(
                f"  {model} Overall: "
                f"{result['scores'][model]['overall']}/5"
            )

        print(f"  Winner: {result['winner']}")

        # Save after every successful question.
        save_jsonl(OUTPUT_PATH, results)

    summary = calculate_summary(results)

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Quantization Judge Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetailed results: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
