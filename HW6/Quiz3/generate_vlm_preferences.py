import json
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path("data/vlm")
ANNOTATION_FILE = ROOT / "annotations/train.json"
CSV_FILE = ROOT / "train_review_verified.csv"
IMAGE_DIR = ROOT / "images/train"
OUTPUT = ROOT / "preferences.jsonl"

# Load COCO annotations
with open(ANNOTATION_FILE, encoding="utf-8") as f:
    coco = json.load(f)

categories = {x["id"]: x["name"] for x in coco["categories"]}
images = {x["file_name"]: x for x in coco["images"]}

objects_by_image = defaultdict(list)
for ann in coco["annotations"]:
    name = categories[ann["category_id"]]
    objects_by_image[ann["image_id"]].append(name)

# Load human-verified absent objects
with open(CSV_FILE, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

records = []
missing_images = []
empty_objects = []
overlaps = []

for row in rows:
    filename = row["image"].strip()
    image_path = IMAGE_DIR / filename

    if filename not in images or not image_path.exists():
        missing_images.append(filename)
        continue

    image_info = images[filename]
    image_id = image_info["id"]

    annotated = objects_by_image[image_id]
    counts = defaultdict(int)
    for name in annotated:
        counts[name] += 1

    annotated_text = ", ".join(
        f"{name} ({count})" for name, count in sorted(counts.items())
    )

    absent_objects = [
        x.strip()
        for x in row["verified_absent_objects"].split(";")
        if x.strip()
    ]

    if not absent_objects:
        empty_objects.append(filename)
        continue

    for negative in absent_objects:
        if negative in counts:
            overlaps.append((filename, negative))

        prompt = (
            "請仔細觀察圖片，辨識圖片中實際可見的物件。"
            "只描述有視覺證據支持的物件，不要猜測或加入不存在的物件。"
        )

        chosen = (
            f"圖片中可辨識的物件包括：{annotated_text}。"
            "我只列出有標註支持的物件，不對未確認的物件作推測。"
        )

        rejected = (
            f"圖片中可辨識的物件包括：{annotated_text}，"
            f"另外還有 {negative}。"
        )

        records.append({
            "image": str(image_path),
            "image_filename": filename,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "negative_object": negative,
            "annotated_objects": dict(counts),
            "category": "visual_hallucination"
        })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("=" * 55)
print("VLM DPO preference generation completed")
print("=" * 55)
print("Source images:", len(rows))
print("Generated preference pairs:", len(records))
print("Missing images:", len(missing_images))
print("Empty negative records:", len(empty_objects))
print("Negative objects overlapping COCO labels:", len(overlaps))
print("Output:", OUTPUT)

if missing_images:
    print("Missing:", missing_images[:10])

if empty_objects:
    print("Empty:", empty_objects[:10])

if overlaps:
    print("\nWARNING: These negative objects also occur in COCO annotations:")
    for item in overlaps[:20]:
        print(item)

print("\nFirst 2 records:")
for record in records[:2]:
    print(json.dumps(record, ensure_ascii=False, indent=2))

assert len(records) == 360, (
    f"Expected 360 pairs, generated {len(records)}. "
    "Please inspect the CSV and image paths."
)
assert not missing_images, "Some images are missing."

print("\nSUCCESS: 360 preference pairs generated.")
