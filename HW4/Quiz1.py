from __future__ import annotations
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from torchvision.datasets import STL10


STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck",]


# 固定 PyTorch 的隨機種子
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 只切分資料索引
def split_indices(n: int, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n)
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)
    return np.asarray(train_idx), np.asarray(val_idx)


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def prepare_imdb(root: Path, seed: int) -> None:
    print("\n[1/3] Loading IMDb dataset from Hugging Face...")
    out = root / "imdb"
    make_dir(out)
    dataset = load_dataset("stanfordnlp/imdb")

    train_full = dataset["train"]
    test_ds = dataset["test"]
    split = train_full.train_test_split(test_size=0.20, seed=seed, stratify_by_column="label")
    train_ds, val_ds = split["train"], split["test"]

    summary = [
        {"split": "train", "samples": len(train_ds)},
        {"split": "validation", "samples": len(val_ds)},
        {"split": "test", "samples": len(test_ds)},
    ]
    save_summary(summary, out / "split_summary.csv")

    label_names = {0: "negative", 1: "positive"}
    distribution_rows = []
    for split_name, split_ds in [("train", train_ds), ("validation", val_ds), ("test", test_ds)]:
        counts = Counter(int(x) for x in split_ds["label"])
        for label_id in sorted(label_names):
            distribution_rows.append({
                "split": split_name,
                "label_id": label_id,
                "label": label_names[label_id],
                "count": counts.get(label_id, 0),
            })
    save_summary(distribution_rows, out / "label_distribution.csv")

    examples = []
    for split_name, split_ds in [("train", train_ds), ("validation", val_ds), ("test", test_ds)]:
        for i in range(min(5, len(split_ds))):
            row = split_ds[i]
            examples.append({
                "split": split_name,
                "label": label_names[int(row["label"])],
                "text": re.sub(r"\s+", " ", str(row["text"])).strip(),
            })
    pd.DataFrame(examples).to_csv(out / "text_examples.csv", index=False, encoding="utf-8-sig")

    # Save split indices to make the exact split reproducible.
    split_index_rows = []
    for split_name, indices in [
        ("train", split["train"]._indices["indices"] if getattr(split["train"], "_indices", None) is not None else range(len(train_ds))),
        ("validation", split["test"]._indices["indices"] if getattr(split["test"], "_indices", None) is not None else range(len(val_ds))),
    ]:
        for idx in indices:
            split_index_rows.append({"split": split_name, "original_train_index": int(idx)})
    pd.DataFrame(split_index_rows).to_csv(out / "train_validation_indices.csv", index=False, encoding="utf-8-sig")

    print(f"IMDb: train={len(train_ds):,}, validation={len(val_ds):,}, test={len(test_ds):,}")
    print(f"Saved statistics and examples to: {out}")


def prepare_stl10(root: Path, seed: int, sample_count: int = 20) -> None:
    print("\n[2/3] Downloading/loading STL-10...")
    out = root / "stl10"
    image_dir = out / "sample_images"
    make_dir(image_dir)

    raw_root = root / "raw"
    train_full = STL10(root=str(raw_root), split="train", download=True)
    test_ds = STL10(root=str(raw_root), split="test", download=True)

    train_idx, val_idx = split_indices(len(train_full), test_size=0.10, seed=seed)
    split_indices_dict = {
        "train": train_idx.tolist(),
        "validation": val_idx.tolist(),
        "test": list(range(len(test_ds))),
    }
    save_json(split_indices_dict, out / "split_indices.json")

    summary = [
        {"split": "train", "samples": len(train_idx)},
        {"split": "validation", "samples": len(val_idx)},
        {"split": "test", "samples": len(test_ds)},
    ]
    save_summary(summary, out / "split_summary.csv")

    distribution_rows = []
    for split_name, indices, dataset in [
        ("train", train_idx, train_full),
        ("validation", val_idx, train_full),
        ("test", np.arange(len(test_ds)), test_ds),
    ]:
        labels = np.asarray(dataset.labels)[indices]
        counts = Counter(int(x) for x in labels)
        for label_id, label_name in enumerate(STL10_CLASSES):
            distribution_rows.append({
                "split": split_name,
                "label_id": label_id,
                "label": label_name,
                "count": counts.get(label_id, 0),
            })
    save_summary(distribution_rows, out / "label_distribution.csv")

    # Save sample images with labels and a contact sheet.
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(train_full), size=min(sample_count, len(train_full)), replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(12, 9))
    axes = np.asarray(axes).reshape(-1)
    image_manifest = []
    for plot_i, idx in enumerate(selected):
        image, label = train_full[int(idx)]
        label_name = STL10_CLASSES[int(label)]
        image_path = image_dir / f"stl10_train_{int(idx):05d}_{label_name}.png"
        image.save(image_path)
        axes[plot_i].imshow(image)
        axes[plot_i].set_title(label_name)
        axes[plot_i].axis("off")
        image_manifest.append({
            "original_index": int(idx),
            "label_id": int(label),
            "label": label_name,
            "image_path": str(image_path.relative_to(root)),
        })
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle("STL-10 training examples", fontsize=16)
    fig.tight_layout()
    fig.savefig(out / "sample_grid.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    save_summary(image_manifest, out / "sample_images.csv")

    print(f"STL-10: train={len(train_idx):,}, validation={len(val_idx):,}, test={len(test_ds):,}")
    print(f"Saved statistics and sample images to: {out}")


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def image_identifier(image_value: Any, row: dict[str, Any], row_index: int) -> str:
    # Prefer identifiers that refer to the image itself. A generic "id" may
    # identify a caption row instead, which could leak the same image across splits.
    for key in ("image_id", "img_id", "filename", "file_name", "image_name"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    if isinstance(image_value, dict):
        path = image_value.get("path")
        if path:
            return Path(str(path)).name
    pil_filename = getattr(image_value, "filename", None)
    if pil_filename:
        return Path(str(pil_filename)).name
    # Fallback: same image rows should normally expose an image id or path.
    # If not, each row is treated as a separate image to avoid accidental merging.
    return f"row_{row_index:06d}"


def image_to_pil(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and Path(value["path"]).is_file():
            try:
                return Image.open(value["path"]).convert("RGB")
            except Exception:
                pass
        if value.get("bytes"):
            import io
            try:
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            except Exception:
                pass
    return None


def normalize_caption(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        for key in ("caption", "text", "raw"):
            if key in value:
                return str(value[key]).strip()
    return str(value).strip()



def collect_flickr_rows(dataset_split: Any, split_name: str) -> list[dict[str, Any]]:
    columns = list(dataset_split.column_names)
    image_col = first_existing(columns, ["image", "img", "photo", "picture"])
    caption_col = first_existing(columns, ["caption", "captions", "text", "sentence", "description"])
    caption_cols = [c for c in columns if re.fullmatch(r"caption_\d+", c, re.IGNORECASE)]
    id_col = first_existing(columns, ["image_id", "img_id", "filename", "file_name", "image_name"])

    if image_col is None or (caption_col is None and not caption_cols):
        raise ValueError(
            f"Cannot identify image/caption columns in Flickr8k split '{split_name}'. "
            f"Columns found: {columns}."
        )

    rows = []
    for i, row in enumerate(dataset_split):
        image_value = row[image_col]
        image_id = (
            str(row[id_col])
            if id_col and row.get(id_col) is not None
            else image_identifier(image_value, row, i)
        )
        image = image_to_pil(image_value)

        if caption_cols:
            captions = [row[c] for c in caption_cols]
        else:
            captions = [row[caption_col]]

        for caption_value in captions:
            caption = normalize_caption(caption_value)
            if not caption:
                continue
            rows.append({
                "source_split": split_name,
                "image_id": image_id,
                "caption": caption,
                "image": image,
            })

    return rows


def split_flickr_by_image(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    image_ids = sorted({row["image_id"] for row in rows})
    if len(image_ids) < 3:
        raise ValueError("Flickr8k dataset has too few unique image IDs to create train/validation/test splits.")
    train_ids, temp_ids = train_test_split(image_ids, test_size=0.20, random_state=seed, shuffle=True)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed, shuffle=True)
    split_id_sets = {
        "train": set(train_ids),
        "validation": set(val_ids),
        "test": set(test_ids),
    }
    return {
        split_name: [row for row in rows if row["image_id"] in ids]
        for split_name, ids in split_id_sets.items()
    }


def prepare_flickr8k(root: Path, seed: int, sample_count: int = 20) -> None:
    print("\n[3/3] Loading Flickr8k from Hugging Face...")
    out = root / "flickr8k"
    image_dir = out / "sample_images"
    make_dir(image_dir)

    dataset = load_dataset("jxie/flickr8k")
    all_rows: list[dict[str, Any]] = []
    for split_name in dataset.keys():
        all_rows.extend(collect_flickr_rows(dataset[split_name], split_name))

    # If the repository contains official split names, keep those splits.
    keys_lower = {key.lower() for key in dataset.keys()}
    if {"train", "validation", "test"}.issubset(keys_lower):
        key_map = {key.lower(): key for key in dataset.keys()}
        split_rows = {}
        for target_name, source_name in [
            ("train", key_map["train"]),
            ("validation", key_map["validation"]),
            ("test", key_map["test"]),
        ]:
            split_rows[target_name] = [
                row for row in all_rows if row["source_split"] == source_name
            ]
        split_method = "preserved Hugging Face train/validation/test splits"
    else:
        split_rows = split_flickr_by_image(all_rows, seed)
        split_method = "random split by unique image ID (80% train / 10% validation / 10% test)"

    summary = []
    split_manifest_rows = []
    for split_name, rows in split_rows.items():
        unique_images = len({row["image_id"] for row in rows})
        summary.append({
            "split": split_name,
            "image_caption_pairs": len(rows),
            "unique_images": unique_images,
        })
        for row in rows:
            split_manifest_rows.append({
                "split": split_name,
                "image_id": row["image_id"],
                "caption": row["caption"],
                "source_split": row["source_split"],
            })
    save_summary(summary, out / "split_summary.csv")
    pd.DataFrame(split_manifest_rows).to_csv(out / "captions.csv", index=False, encoding="utf-8-sig")
    save_json({"split_method": split_method, "seed": seed}, out / "split_info.json")

    # Save sample captions.
    caption_examples = []
    for split_name, rows in split_rows.items():
        for row in rows[:5]:
            caption_examples.append({
                "split": split_name,
                "image_id": row["image_id"],
                "caption": row["caption"],
            })
    pd.DataFrame(caption_examples).to_csv(out / "caption_examples.csv", index=False, encoding="utf-8-sig")

    # Save a contact sheet of available decoded images.
    candidates = [row for row in all_rows if row["image"] is not None]
    # Keep one row per image ID for the image grid.
    unique_rows = {}
    for row in candidates:
        unique_rows.setdefault(row["image_id"], row)
    candidates = list(unique_rows.values())
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:sample_count]

    if selected:
        cols = 4
        rows_n = int(np.ceil(len(selected) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(14, rows_n * 3.2))
        axes = np.asarray(axes, dtype=object).reshape(-1)
        image_manifest = []
        for i, row in enumerate(selected):
            image = row["image"]
            image_path = image_dir / f"flickr8k_{i:03d}.jpg"
            image.save(image_path, quality=95)
            axes[i].imshow(image)
            caption = row["caption"]
            axes[i].set_title(caption[:90] + ("..." if len(caption) > 90 else ""), fontsize=8)
            axes[i].axis("off")
            image_manifest.append({
                "image_id": row["image_id"],
                "caption": row["caption"],
                "image_path": str(image_path.relative_to(root)),
            })
        for ax in axes[len(selected):]:
            ax.axis("off")
        fig.suptitle("Flickr8k image-caption examples", fontsize=16)
        fig.tight_layout()
        fig.savefig(out / "sample_grid.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        save_summary(image_manifest, out / "sample_images.csv")
    else:
        print(
            "WARNING: no decoded Flickr8k images were available from the dataset rows. "
            "Caption CSV and statistics were saved; inspect the dataset's image feature/schema."
        )

    print(f"Flickr8k split method: {split_method}")
    print("Flickr8k summary:")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Saved statistics, captions and sample images to: {out}")


def main() -> None:
    output_dir = "./data_quiz1"
    seed = 42
    flickr_samples = 20

    root = Path(output_dir).resolve()
    make_dir(root)
    set_seed(seed)

    print(f"Output directory: {root}")
    prepare_imdb(root, seed)
    prepare_stl10(root, seed)
    prepare_flickr8k(root, seed, sample_count=flickr_samples)

    print("\nAll Quiz 1 dataset preparation steps finished.")
    print(f"Output root: {root}")


if __name__ == "__main__":
    main()
