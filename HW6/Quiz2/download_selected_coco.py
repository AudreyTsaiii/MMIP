import json
import random
import urllib.request
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ANNOTATION_FILE = Path("benchmark/annotations/instances_val2017.json")
IMAGE_DIR = Path("benchmark/images")
OUTPUT_DIR = Path("benchmark/selected")

TOTAL_IMAGES = 200
TRAIN_COUNT = 120
TARGET_COUNT = 50
GENERAL_COUNT = 30

SEED = 42
BASE_URL = "http://images.cocodataset.org/val2017/"

random.seed(SEED)


# --------------------------------------------------
# Load COCO annotations
# --------------------------------------------------
with open(ANNOTATION_FILE, encoding="utf-8") as f:
    coco = json.load(f)

annotations_by_image = {}

for ann in coco["annotations"]:
    image_id = ann["image_id"]
    annotations_by_image.setdefault(image_id, []).append(
        ann["category_id"]
    )

categories = {
    cat["id"]: cat["name"]
    for cat in coco["categories"]
}

# Select images with at least two annotated categories.
candidates = []

for img in coco["images"]:
    category_ids = set(
        annotations_by_image.get(img["id"], [])
    )

    if len(category_ids) >= 2:
        candidates.append(img)

if len(candidates) < TOTAL_IMAGES:
    raise RuntimeError(
        f"Not enough candidate images: {len(candidates)}"
    )

selected = random.sample(candidates, TOTAL_IMAGES)

train_images = selected[:TRAIN_COUNT]
target_images = selected[
    TRAIN_COUNT:TRAIN_COUNT + TARGET_COUNT
]
general_images = selected[
    TRAIN_COUNT + TARGET_COUNT:
]

splits = {
    "train": train_images,
    "target_eval": target_images,
    "general_eval": general_images,
}

# --------------------------------------------------
# Download selected images only
# --------------------------------------------------
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_image(img):
    filename = img["file_name"]
    destination = IMAGE_DIR / filename
    url = BASE_URL + filename

    if destination.exists():
        return filename, "already_exists"

    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(
                request, timeout=30
            ) as response:
                content = response.read()

            destination.write_bytes(content)
            return filename, "downloaded"

        except Exception as e:
            if attempt == 2:
                return filename, f"failed: {e}"
            time.sleep(2)


print(f"Selected {len(selected)} images.")
print("Starting downloads...")

results = []

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(download_image, img)
        for img in selected
    ]

    for future in as_completed(futures):
        results.append(future.result())

downloaded = [
    filename for filename, status in results
    if status in ("downloaded", "already_exists")
]

failed = [
    (filename, status)
    for filename, status in results
    if status.startswith("failed")
]

# --------------------------------------------------
# Save split manifest and filtered annotations
# --------------------------------------------------
selected_ids = {img["id"] for img in selected}

for split_name, images in splits.items():
    split_ids = {img["id"] for img in images}

    split_annotations = [
        ann for ann in coco["annotations"]
        if ann["image_id"] in split_ids
    ]

    output = {
        "images": images,
        "annotations": split_annotations,
        "categories": coco["categories"],
    }

    with open(
        OUTPUT_DIR / f"{split_name}.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

# Save complete selection manifest
with open(
    OUTPUT_DIR / "selected_images.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "seed": SEED,
            "splits": {
                name: [img["file_name"] for img in images]
                for name, images in splits.items()
            },
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\n===== Download Summary =====")
print("Selected:", len(selected))
print("Downloaded / existing:", len(downloaded))
print("Failed:", len(failed))
print("Train:", len(train_images))
print("Target evaluation:", len(target_images))
print("General evaluation:", len(general_images))

if failed:
    print("\nFailed files:")
    for filename, error in failed:
        print(filename, error)

print("\nAnnotations saved to:", OUTPUT_DIR)
