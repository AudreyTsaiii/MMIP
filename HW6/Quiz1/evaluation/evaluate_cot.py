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

BASELINE_PATH = RESULT_DIR / "baseline_fp16.jsonl"
COT_PATH = RESULT_DIR / "cot_fp16.jsonl"

OUTPUT_PATH = RESULT_DIR / "judge_results.jsonl"
SUMMARY_PATH = RESULT_DIR / "judge_summary.json"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
RANDOM_SEED = 42
MAX_RETRIES = 3


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_json(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Cannot parse Gemini response as JSON:\n{text}")


def validate_scores(data):
    required = ["correctness", "completeness",
                "instruction_following", "overall"]

    for answer_key in ["answer_a", "answer_b"]:
        if answer_key not in data:
            raise ValueError(f"Missing {answer_key}")

        for metric in required:
            score = data[answer_key].get(metric)
            if not isinstance(score, (int, float)) or not 1 <= score <= 5:
                raise ValueError(
                    f"Invalid score: {answer_key}.{metric}={score}"
                )

    if data.get("winner") not in ["A", "B", "Tie"]:
        raise ValueError(f"Invalid winner: {data.get('winner')}")

    return data


def build_judge_prompt(question, reference, criteria, answer_a, answer_b):
    return f"""
你是一位嚴謹且公平的 AI 模型評估員。
請使用繁體中文理解題目與評分標準。

請根據以下資料，評估兩個回答。

【題目】
{question}

【參考答案】
{reference}

【評分標準】
{criteria}

【回答 A】
{answer_a}

【回答 B】
{answer_b}

請分別為 A 和 B 評分，每個面向均為 1 至 5 分：

1. correctness：正確性。是否包含正確資訊，有無錯誤或不合理內容。
2. completeness：完整性。是否涵蓋題目要求的重點。
3. instruction_following：指令遵循。是否遵守題目要求的格式、長度及限制。
4. overall：整體品質。綜合考量以上面向。

評分原則：
- 5 分：表現優秀，幾乎完全符合要求。
- 4 分：大致正確完整，只有輕微缺失。
- 3 分：部分符合要求，但有明顯缺失。
- 2 分：存在多項錯誤或重要遺漏。
- 1 分：嚴重錯誤，或幾乎沒有回答題目。
- 不要因為回答比較長就給比較高的分數。
- 不要因為回答有條列或推理過程就自動加分。
- 如果題目要求特定格式，應嚴格依照格式評分。
- 參考答案是評分依據，但若回答使用其他正確方法，也應給予合理分數。
- 先獨立評估兩個回答，再判斷哪個整體品質較好。

winner 請填：
- "A"：回答 A 整體較好
- "B"：回答 B 整體較好
- "Tie"：兩者整體品質相當

請只輸出合法 JSON，不要輸出 Markdown 或其他文字。
格式如下：
{{
  "answer_a": {{
    "correctness": 1,
    "completeness": 1,
    "instruction_following": 1,
    "overall": 1
  }},
  "answer_b": {{
    "correctness": 1,
    "completeness": 1,
    "instruction_following": 1,
    "overall": 1
  }},
  "winner": "A",
  "reason": "簡短說明判斷理由"
}}
""".strip()


def judge_one(client, item_base, item_cot, rng):
    # Randomize answer order to reduce position bias.
    answers = [
        ("baseline", item_base["answer"]),
        ("cot", item_cot["answer"]),
    ]
    rng.shuffle(answers)

    label_a, answer_a = answers[0]
    label_b, answer_b = answers[1]

    prompt = build_judge_prompt(
        question=item_base["question"],
        reference=item_base["reference_answer"],
        criteria=item_base["criteria"],
        answer_a=answer_a,
        answer_b=answer_b,
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

            # Map randomized A/B scores back to original methods.
            scores = {
                label_a: data["answer_a"],
                label_b: data["answer_b"],
            }

            winner = data["winner"]
            if winner == "A":
                mapped_winner = label_a
            elif winner == "B":
                mapped_winner = label_b
            else:
                mapped_winner = "Tie"

            return {
                "id": item_base["id"],
                "category": item_base["category"],
                "baseline_scores": scores["baseline"],
                "cot_scores": scores["cot"],
                "winner": mapped_winner,
                "reason": data.get("reason", ""),
                "answer_order": [label_a, label_b],
            }

        except Exception as exc:
            last_error = exc
            print(
                f"  Attempt {attempt}/{MAX_RETRIES} failed: {exc}"
            )
            if attempt < MAX_RETRIES:
                if "429" in str(exc):
                    wait_seconds = 65
                else:
                    wait_seconds = 10 * attempt

                print(f"  Waiting {wait_seconds} seconds before retry...")
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Gemini evaluation failed for {item_base['id']}: {last_error}"
    )


def calculate_summary(results):
    metrics = [
        "correctness",
        "completeness",
        "instruction_following",
        "overall",
    ]

    summary = {
        "model": GEMINI_MODEL,
        "num_questions": len(results),
        "baseline": {},
        "cot": {},
        "comparison": {},
        "wins": {
            "baseline": 0,
            "cot": 0,
            "tie": 0,
        },
    }

    for method in ["baseline", "cot"]:
        key = f"{method}_scores"

        for metric in metrics:
            values = [r[key][metric] for r in results]
            summary[method][metric] = round(
                sum(values) / len(values), 4
            )

    for metric in metrics:
        base_score = summary["baseline"][metric]
        cot_score = summary["cot"][metric]

        summary["comparison"][metric] = {
            "difference_cot_minus_baseline": round(
                cot_score - base_score, 4
            )
        }

    for result in results:
        if result["winner"] == "baseline":
            summary["wins"]["baseline"] += 1
        elif result["winner"] == "cot":
            summary["wins"]["cot"] += 1
        else:
            summary["wins"]["tie"] += 1

    summary["category_scores"] = {}

    categories = sorted(set(r["category"] for r in results))

    for category in categories:
        category_results = [
            r for r in results if r["category"] == category
        ]

        summary["category_scores"][category] = {}

        for method in ["baseline", "cot"]:
            values = [
                r[f"{method}_scores"]["overall"]
                for r in category_results
            ]
            summary["category_scores"][category][method] = round(
                sum(values) / len(values), 4
            )

    return summary


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Please export GEMINI_API_KEY first."
        )

    baseline = load_jsonl(BASELINE_PATH)
    cot = load_jsonl(COT_PATH)

    baseline_by_id = {item["id"]: item for item in baseline}
    cot_by_id = {item["id"]: item for item in cot}

    if set(baseline_by_id) != set(cot_by_id):
        raise ValueError("Baseline and CoT question IDs do not match.")

    if not baseline:
        raise ValueError("No benchmark results found.")

    client = genai.Client(api_key=api_key)
    rng = random.Random(RANDOM_SEED)

    # Resume from previously saved results.
    results = []
    if OUTPUT_PATH.exists():
        results = load_jsonl(OUTPUT_PATH)

    completed_ids = {r["id"] for r in results}

    print(f"Gemini Judge model: {GEMINI_MODEL}")
    print(f"Questions to evaluate: {len(baseline)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {len(baseline) - len(completed_ids)}")

    for index, item in enumerate(baseline, start=1):
        if item["id"] in completed_ids:
            print(f"Skipping {item['id']} (already evaluated)")
            continue

        cot_item = cot_by_id[item["id"]]

        # Stay below the Gemini free-tier requests-per-minute limit.
        time.sleep(6)

        if item["question"] != cot_item["question"]:
            raise ValueError(f"Question mismatch: {item['id']}")

        print(f"\n[{index:02d}/{len(baseline)}] Evaluating {item['id']}")

        result = judge_one(client, item, cot_item, rng)
        results.append(result)
        completed_ids.add(item["id"])

        print(
            f"  Baseline overall: "
            f"{result['baseline_scores']['overall']}/5"
        )
        print(
            f"  CoT overall: "
            f"{result['cot_scores']['overall']}/5"
        )
        print(f"  Winner: {result['winner']}")

        # Save after each question so completed work is preserved.
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = calculate_summary(results)

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Gemini Judge Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetailed results: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
