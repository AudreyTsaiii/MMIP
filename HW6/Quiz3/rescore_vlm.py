"""Rescore saved VLM answers against COCO labels without loading a model."""
import argparse
import json
from pathlib import Path
from coco_scoring import VERSION, extract_objects, summarize


def rescore(result, annotation_root):
    summaries = {}
    for split, rows in result["predictions"].items():
        coco = json.loads((annotation_root / f"{split}.json").read_text(encoding="utf-8"))
        categories = sorted({r["name"] for r in coco["categories"]})
        category_by_id = {r["id"]: r["name"] for r in coco["categories"]}
        objects = {r["id"]: set() for r in coco["images"]}
        for ann in coco["annotations"]:
            objects[ann["image_id"]].add(category_by_id[ann["category_id"]])
        truth = {r["file_name"]: objects[r["id"]] for r in coco["images"]}
        if len({r["image"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate images in {split}")
        for row in rows:
            gt = truth[row["image"]]
            extracted = extract_objects(row["raw_answer"], categories)
            pred = set(extracted["predicted"])
            row.update(extracted, ground_truth=sorted(gt), tp=len(gt & pred), fp=len(pred-gt), fn=len(gt-pred),
                false_positives=sorted(pred-gt), false_negatives=sorted(gt-pred))
        summaries[split] = summarize(rows)
    result.update(splits=summaries, scoring_version=VERSION,
        scoring_note="Micro category-set scores against COCO annotations. Unmatched categories require visual review; not verified hallucinations. Timing and memory are retained from original inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/vlm_dpo", type=Path)
    parser.add_argument("--output-dir", default="results/vlm_dpo_rescored", type=Path)
    parser.add_argument("--annotations", default="data/vlm/annotations", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose an empty output directory.")
    results = [rescore(json.loads((args.input_dir / f"{name}_predictions.json").read_text(encoding="utf-8")), args.annotations) for name in ("base", "dpo")]
    base, dpo = results
    if base["predictions"].keys() != dpo["predictions"].keys():
        parser.error("Base and DPO splits differ")
    comparison = {}
    for split in base["splits"]:
        if {r["image"] for r in base["predictions"][split]} != {r["image"] for r in dpo["predictions"][split]}:
            parser.error(f"Base and DPO image sets differ in {split}")
        comparison[split] = {}
        for metric in ("precision", "recall", "f1", "hallucination_rate", "images_with_unmatched_predicted_categories_rate", "empty_prediction_rate", "average_latency_sec"):
            a, b = base["splits"][split][metric], dpo["splits"][split][metric]
            comparison[split][metric] = {"base": a, "dpo": b, "change": b-a if a is not None and b is not None else None}
    comparison["peak_gpu_memory_gb"] = {name: result.get("peak_gpu_memory_gb") for name, result in zip(("base", "dpo"), results)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in zip(("base", "dpo"), results):
        (args.output_dir / f"{name}_predictions.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
