import json
import csv
import re
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


ROOT = Path(__file__).resolve().parents[1]

QUESTIONS = ROOT / "benchmark/questions.jsonl"

FILES = {
    "Qwen2.5-VL-3B": ROOT / "results/qwen2.5-vl/predictions.jsonl",
    "SmolVLM2-2.2B": ROOT / "results/smolvlm2/predictions.jsonl",
}

OUTPUT_DIR = ROOT / "evaluation"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text):
    """
    Simple English tokenizer.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l(candidate, references):
    """
    Calculate ROUGE-L F1 against multiple references.
    Use the best matching reference.
    """

    def lcs_length(a, b):
        previous = [0] * (len(b) + 1)

        for x in a:
            current = [0]

            for j, y in enumerate(b):
                if x == y:
                    current.append(previous[j] + 1)
                else:
                    current.append(
                        max(previous[j + 1], current[-1])
                    )

            previous = current

        return previous[-1]

    best_score = 0.0

    for reference in references:
        if not candidate or not reference:
            continue

        lcs = lcs_length(candidate, reference)

        precision = lcs / len(candidate)
        recall = lcs / len(reference)

        if precision + recall == 0:
            score = 0.0
        else:
            score = (
                2 * precision * recall
                / (precision + recall)
            )

        best_score = max(best_score, score)

    return best_score


def evaluate_caption(candidate_text, reference_texts):
    candidate = tokenize(candidate_text)
    references = [
        tokenize(ref)
        for ref in reference_texts
    ]

    if not candidate or not any(references):
        return {
            "BLEU-1": 0.0,
            "BLEU-4": 0.0,
            "METEOR": 0.0,
            "ROUGE-L": 0.0,
        }

    smoothing = SmoothingFunction().method1

    bleu1 = sentence_bleu(
        references,
        candidate,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothing
    )

    bleu4 = sentence_bleu(
        references,
        candidate,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing
    )

    meteor = meteor_score(
        references,
        candidate
    )

    rouge = rouge_l(
        candidate,
        references
    )

    return {
        "BLEU-1": bleu1,
        "BLEU-4": bleu4,
        "METEOR": meteor,
        "ROUGE-L": rouge,
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_jsonl(QUESTIONS)

    caption_questions = [
        q for q in questions
        if q["category"] == "image_captioning"
    ]

    if len(caption_questions) != 4:
        raise ValueError(
            f"預期 4 題 Captioning，實際取得 {len(caption_questions)} 題"
        )

    all_records = []
    summaries = []

    for model_name, prediction_path in FILES.items():

        predictions = load_jsonl(prediction_path)
        prediction_map = {
            p["id"]: p for p in predictions
        }

        model_scores = {
            metric: []
            for metric in [
                "BLEU-1",
                "BLEU-4",
                "METEOR",
                "ROUGE-L",
            ]
        }

        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        for q in caption_questions:

            qid = q["id"]

            if qid not in prediction_map:
                raise ValueError(
                    f"{model_name}: 缺少 ID={qid} 的預測"
                )

            prediction = prediction_map[qid]

            if q["image"] != prediction["image"]:
                raise ValueError(
                    f"{model_name}: ID={qid} 圖片不一致"
                )

            references = q.get("reference_captions", [])

            if not references:
                raise ValueError(
                    f"ID={qid} 沒有 Reference Captions"
                )

            scores = evaluate_caption(
                prediction["answer"],
                references
            )

            for metric, value in scores.items():
                model_scores[metric].append(value)

            record = {
                "model": model_name,
                "id": qid,
                "image": q["image"],
                "answer": prediction["answer"],
                "reference_captions": json.dumps(
                    references,
                    ensure_ascii=False
                ),
                **scores,
            }

            all_records.append(record)

            print(f"\nID: {qid}")

            for metric, value in scores.items():
                print(f"{metric}: {value:.4f}")

        summary = {
            "model": model_name,
            "num_questions": len(caption_questions),
        }

        print("\nAverage Scores:")

        for metric, values in model_scores.items():

            average = sum(values) / len(values)

            summary[metric] = average

            print(f"{metric}: {average:.4f}")

        summaries.append(summary)

    # Save detailed results
    with (
        OUTPUT_DIR / "caption_evaluation.csv"
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
        OUTPUT_DIR / "caption_comparison.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(summaries[0].keys())
        )

        writer.writeheader()
        writer.writerows(summaries)

    with (
        OUTPUT_DIR / "caption_metrics.json"
    ).open("w", encoding="utf-8") as f:

        json.dump(
            summaries,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 60)
    print("Caption evaluation completed!")
    print("Saved:")
    print(" - evaluation/caption_evaluation.csv")
    print(" - evaluation/caption_comparison.csv")
    print(" - evaluation/caption_metrics.json")


if __name__ == "__main__":
    main()