import json
import csv
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "Qwen2.5-VL-3B": ROOT / "results/qwen2.5-vl/predictions.jsonl",
    "SmolVLM2-2.2B": ROOT / "results/smolvlm2/predictions.jsonl",
}

QUESTIONS = ROOT / "benchmark/questions.jsonl"
OUTPUT_DIR = ROOT / "evaluation"


# --------------------------------------------------
# 1. Answer aliases
# --------------------------------------------------

ALIASES = {
    "滑板運動場景": [
        "滑板運動場景", "滑板运动场景",
        "滑板", "skateboarding", "skateboard"
    ],
    "棒球": [
        "棒球", "baseball"
    ],
    "野生動物與草原場景": [
        "野生動物與草原場景",
        "野生动物与草原场景",
        "wildlife and grassland scene"
    ],
    "野生動物場景": [
        "野生動物場景",
        "野生动物场景",
        "wildlife scene"
    ],
    "飛機": [
        "飛機", "飞机", "airplane", "aeroplane", "plane"
    ],
    "沙發": [
        "沙發", "沙发", "sofa", "couch"
    ],
    "桌子": [
        "桌子", "桌", "table", "desk",
        "咖啡桌", "咖啡桌子", "coffee table",
        "電腦桌", "电脑桌", "computer desk"
    ],
    "貓": [
        "貓", "猫", "cat", "kitten"
    ],
    "書本": [
        "書本", "书本", "書籍", "书籍",
        "book", "books"
    ],
    "吹風機": [
        "吹風機", "吹风机", "hair dryer", "hairdryer"
    ],
    "披薩": [
        "披薩", "披萨", "pizza"
    ],
    "2": [
        "2", "兩匹", "两匹",
        "兩匹馬", "两匹马",
        "兩隻馬", "两只马",
        "two horses", "2 horses"
    ],
    "公車": [
        "公車", "巴士", "公交車", "公交车",
        "bus"
    ],
    "泰迪熊": [
        "泰迪熊", "絨毛熊", "绒毛熊",
        "玩偶熊", "teddy bear", "teddy"
    ],
    "滑雪板": [
        "滑雪板", "snowboard"
    ],
    "白色": [
        "白色", "white"
    ],
    "香蕉": [
        "香蕉", "banana"
    ],
    "橘色": [
        "橘色", "橙色", "orange"
    ],
}


# --------------------------------------------------
# 2. Basic functions
# --------------------------------------------------

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()

    text = re.sub(
        r"""[\s，。！？、；：,.!?;:'"（）()\[\]【】]""",
        "",
        text
    )

    return text


def get_aliases(target):
    """
    Return acceptable forms of a Ground Truth concept.
    """
    target = str(target)

    aliases = [target]
    aliases.extend(ALIASES.get(target, []))

    return list({
        normalize(alias)
        for alias in aliases
        if normalize(alias)
    })


def contains_alias(answer, alias):
    """
    Check whether an alias appears in the answer.
    Numeric aliases use digit boundaries to avoid
    accidental matches such as 2 inside 2024.
    """
    if not alias:
        return False

    if alias.isdigit():
        return re.search(
            rf"(?<!\d){re.escape(alias)}(?!\d)",
            answer
        ) is not None

    return alias in answer


def target_matched(answer, target):
    """
    Check whether one Ground Truth target is mentioned.
    """
    answer = normalize(answer)

    return any(
        contains_alias(answer, alias)
        for alias in get_aliases(target)
    )


# --------------------------------------------------
# 3. Category-specific evaluation
# --------------------------------------------------

def evaluate_answer(category, answer, ground_truth):
    """
    Return:
        correct: strict question-level correctness
        matched_targets: successfully matched targets
        target_count: number of required targets
        recall: matched target ratio
        note: evaluation explanation
    """

    if isinstance(ground_truth, list):
        targets = ground_truth
    else:
        targets = [ground_truth]

    targets = [str(x) for x in targets]

    matched_targets = [
        target
        for target in targets
        if target_matched(answer, target)
    ]

    matched_count = len(matched_targets)
    target_count = len(targets)

    recall = (
        matched_count / target_count
        if target_count else 0
    )

    if category == "object_recognition":
        # All required objects must be identified.
        correct = (
            matched_count == target_count
            and target_count > 0
        )

        note = (
            f"辨識到 {matched_count}/{target_count} 個指定物件"
        )

    else:
        # Other categories accept any valid answer variant.
        correct = matched_count > 0

        note = (
            "符合可接受答案"
            if correct else "未匹配可接受答案"
        )

    return {
        "correct": correct,
        "matched_targets": matched_targets,
        "matched_count": matched_count,
        "target_count": target_count,
        "recall": recall,
        "note": note,
    }


# --------------------------------------------------
# 4. Evaluate one model
# --------------------------------------------------

def evaluate_model(questions, predictions, model_name):

    question_map = {q["id"]: q for q in questions}
    prediction_map = {p["id"]: p for p in predictions}

    if set(question_map) != set(prediction_map):
        raise ValueError(
            f"{model_name}: 題目 ID 與預測結果不一致"
        )

    records = []
    category_results = defaultdict(list)

    total_matched_objects = 0
    total_required_objects = 0

    latencies = []
    gpu_peaks = []

    for qid in sorted(question_map):

        q = question_map[qid]
        p = prediction_map[qid]

        if q["image"] != p["image"] or q["prompt"] != p["prompt"]:
            raise ValueError(
                f"{model_name}: ID={qid} 的題目或圖片不一致"
            )

        category = q["category"]
        answer = p["answer"]

        if category == "image_captioning":

            correct = None
            ground_truth = ""
            references = q.get("reference_captions", [])

            matched_targets = []
            matched_count = 0
            target_count = 0
            recall = None
            note = "使用 Caption 評估指標另行評估"

        else:

            ground_truth = q.get("ground_truth")

            if ground_truth is None:
                raise ValueError(
                    f"ID={qid} 缺少 Ground Truth"
                )

            references = []

            result = evaluate_answer(
                category,
                answer,
                ground_truth
            )

            correct = result["correct"]
            matched_targets = result["matched_targets"]
            matched_count = result["matched_count"]
            target_count = result["target_count"]
            recall = result["recall"]
            note = result["note"]

            category_results[category].append(int(correct))

            if category == "object_recognition":
                total_matched_objects += matched_count
                total_required_objects += target_count

        latencies.append(float(p["latency_seconds"]))
        gpu_peaks.append(float(p["peak_gpu_memory_gb"]))

        records.append({
            "model": model_name,
            "id": qid,
            "category": category,
            "image": q["image"],
            "prompt": q["prompt"],
            "ground_truth": json.dumps(
                ground_truth,
                ensure_ascii=False
            ),
            "reference_captions": json.dumps(
                references,
                ensure_ascii=False
            ),
            "answer": answer,
            "matched_targets": json.dumps(
                matched_targets,
                ensure_ascii=False
            ),
            "matched_count": matched_count,
            "target_count": target_count,
            "object_recall": (
                "" if recall is None else round(recall, 4)
            ),
            "auto_correct": (
                "" if correct is None else int(correct)
            ),
            "evaluation_note": note,
            "latency_seconds": p["latency_seconds"],
            "peak_gpu_memory_gb": p["peak_gpu_memory_gb"],
        })

    evaluated = sum(
        len(scores)
        for scores in category_results.values()
    )

    correct_count = sum(
        sum(scores)
        for scores in category_results.values()
    )

    category_accuracy = {
        category: sum(scores) / len(scores)
        for category, scores in category_results.items()
    }

    category_accuracy["image_captioning"] = None

    object_recall = (
        total_matched_objects / total_required_objects
        if total_required_objects else None
    )

    summary = {
        "model": model_name,
        "total_questions": len(records),
        "accuracy_evaluated_questions": evaluated,
        "correct": correct_count,
        "overall_accuracy": (
            correct_count / evaluated if evaluated else 0
        ),
        "category_accuracy": category_accuracy,
        "object_recognition": {
            "matched_objects": total_matched_objects,
            "required_objects": total_required_objects,
            "object_recall": object_recall,
        },
        "average_latency_seconds": (
            sum(latencies) / len(latencies)
        ),
        "average_peak_gpu_memory_gb": (
            sum(gpu_peaks) / len(gpu_peaks)
        ),
        "maximum_peak_gpu_memory_gb": max(gpu_peaks),
    }

    return summary, records


# --------------------------------------------------
# 5. Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_jsonl(QUESTIONS)

    if len(questions) != 20:
        raise ValueError(
            f"預期 20 題，實際取得 {len(questions)} 題"
        )

    if not all(
        q.get("ground_truth_verified") is True
        for q in questions
    ):
        raise ValueError(
            "仍有 Ground Truth 尚未人工確認"
        )

    all_summaries = []
    all_records = []

    for model_name, path in FILES.items():

        predictions = load_jsonl(path)

        if len(predictions) != len(questions):
            raise ValueError(
                f"{model_name}: 預測筆數不等於題目數"
            )

        summary, records = evaluate_model(
            questions,
            predictions,
            model_name
        )

        all_summaries.append(summary)
        all_records.extend(records)

        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        print(
            f"Accuracy: {summary['correct']}/"
            f"{summary['accuracy_evaluated_questions']} "
            f"({summary['overall_accuracy']:.2%})"
        )

        print(
            f"Average Latency: "
            f"{summary['average_latency_seconds']:.3f}s"
        )

        print(
            f"Average Peak GPU Memory: "
            f"{summary['average_peak_gpu_memory_gb']:.3f} GB"
        )

        print(
            f"Maximum Peak GPU Memory: "
            f"{summary['maximum_peak_gpu_memory_gb']:.3f} GB"
        )

        print("\nPer-category Accuracy:")

        for category, accuracy in summary["category_accuracy"].items():

            if accuracy is None:
                print(
                    f"  {category}: N/A "
                    f"(待進行 Caption 評估)"
                )
            else:
                print(f"  {category}: {accuracy:.2%}")

        obj = summary["object_recognition"]

        print(
            "\nObject Recognition Recall: "
            f"{obj['matched_objects']}/"
            f"{obj['required_objects']} "
            f"({obj['object_recall']:.2%})"
        )

    # Save JSON summary
    with (
        OUTPUT_DIR / "metrics_summary.json"
    ).open("w", encoding="utf-8") as f:

        json.dump(
            all_summaries,
            f,
            ensure_ascii=False,
            indent=2
        )

    # Save detailed question evaluation
    with (
        OUTPUT_DIR / "question_evaluation.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(all_records[0].keys())
        )

        writer.writeheader()
        writer.writerows(all_records)

    # Save model comparison
    with (
        OUTPUT_DIR / "comparison.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        fields = [
            "model",
            "overall_accuracy",
            "accuracy_evaluated_questions",
            "average_latency_seconds",
            "average_peak_gpu_memory_gb",
            "maximum_peak_gpu_memory_gb",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for summary in all_summaries:
            writer.writerow({
                key: summary[key]
                for key in fields
            })

    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("Saved:")
    print(" - evaluation/metrics_summary.json")
    print(" - evaluation/question_evaluation.csv")
    print(" - evaluation/comparison.csv")


if __name__ == "__main__":
    main()