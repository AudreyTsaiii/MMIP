
import json
import random
import requests

from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).parent
ANNOTATION_DIR = BASE_DIR / "annotations"
IMAGE_DIR = BASE_DIR / "images"

INSTANCE_FILE = ANNOTATION_DIR / "instances_val2017.json"
CAPTION_FILE = ANNOTATION_DIR / "captions_val2017.json"

NUM_IMAGES = 20
SEED = 42

random.seed(SEED)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

print("Loading COCO annotations...")

with open(INSTANCE_FILE, encoding="utf-8") as f:
    instances = json.load(f)

with open(CAPTION_FILE, encoding="utf-8") as f:
    captions = json.load(f)

# Category ID -> category name
category_names = {
    c["id"]: c["name"]
    for c in instances["categories"]
}

# Image ID -> image metadata
image_info = {
    img["id"]: img
    for img in instances["images"]
}

# Image ID -> object annotations
image_objects = defaultdict(list)

for ann in instances["annotations"]:
    image_objects[ann["image_id"]].append(ann)

# Image ID -> captions
image_captions = defaultdict(list)

for ann in captions["annotations"]:
    image_captions[ann["image_id"]].append(ann["caption"])

# Candidate filtering
candidates = []

for image_id, objects in image_objects.items():
    info = image_info[image_id]

    categories = {
        ann["category_id"]
        for ann in objects
    }

    # Avoid images with too few or too many objects
    if not (3 <= len(objects) <= 12):
        continue

    # Require at least two object categories
    if len(categories) < 2:
        continue

    if image_id not in image_captions:
        continue

    candidates.append({
        "id": image_id,
        "info": info,
        "objects": objects,
        "categories": categories,
        "captions": image_captions[image_id],
    })

print("Candidate images:", len(candidates))

# Greedy selection for category diversity
selected = []
covered_categories = set()
remaining = candidates.copy()

while len(selected) < NUM_IMAGES and remaining:
    def score(item):
        new_categories = item["categories"] - covered_categories

        # Prefer images introducing new categories
        diversity = len(new_categories)

        # Prefer moderate object counts
        object_score = -abs(len(item["objects"]) - 6)

        return diversity * 10 + object_score + random.random() * 0.01

    best = max(remaining, key=score)

    selected.append(best)
    covered_categories.update(best["categories"])
    remaining.remove(best)

# Download selected images
metadata = []

for index, item in enumerate(selected, start=1):
    image_id = item["id"]
    info = item["info"]

    filename = f"coco_{image_id:012d}.jpg"
    image_path = IMAGE_DIR / filename

    if not image_path.exists():
        url = info["coco_url"]
        print(f"Downloading {index}/{NUM_IMAGES}: {filename}")

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        image_path.write_bytes(response.content)

    object_counts = Counter(
        category_names[ann["category_id"]]
        for ann in item["objects"]
    )

    metadata.append({
        "image_id": image_id,
        "image": f"images/{filename}",
        "width": info["width"],
        "height": info["height"],
        "objects": dict(object_counts),
        "captions": item["captions"],
    })

    print(
        f"{index:02d}. {filename} | "
        f"Objects: {dict(object_counts)}"
    )

# Save metadata
output_file = BASE_DIR / "coco_candidates.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# Create contact sheet
thumb_width = 240
thumb_height = 180
label_height = 35
columns = 4
rows = (len(metadata) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (columns * thumb_width, rows * (thumb_height + label_height)),
    "white",
)

draw = ImageDraw.Draw(sheet)

for i, item in enumerate(metadata):
    img = Image.open(BASE_DIR / item["image"]).convert("RGB")
    img.thumbnail((thumb_width, thumb_height))

    x = (i % columns) * thumb_width
    y = (i // columns) * (thumb_height + label_height)

    sheet.paste(img, (x, y))

    label = f'{i + 1}. ID: {item["image_id"]}'
    draw.text((x + 5, y + thumb_height + 5), label, fill="black")

sheet.save(BASE_DIR / "contact_sheet.jpg")

print("\nDone!")
print("Metadata:", output_file)
print("Contact sheet:", BASE_DIR / "contact_sheet.jpg")
print("Selected images:", len(metadata))
