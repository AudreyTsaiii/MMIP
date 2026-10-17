
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


ROOT_DIR = Path(__file__).resolve().parent
RESULT_DIR = ROOT_DIR / "results_quiz2"
CACHE_DIR = RESULT_DIR / "feature_cache"

DATASET_NAME = "jxie/flickr8k"
TEXT_MODEL_NAME = "distilbert-base-uncased"

CAPTION_COLUMNS = [f"caption_{i}" for i in range(5)]
NUM_CLASSES = 2
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FlickrImageDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        image = image.convert("RGB")
        return self.transform(image)


def get_captions(hf_dataset):
    captions = []

    for row in hf_dataset:
        for column in CAPTION_COLUMNS:
            captions.append(str(row[column]))

    return captions


def build_pairs(num_images):
    """
    每張圖片有五個正樣本和五個負樣本。

    正樣本：圖片 i + 圖片 i 自己的 caption
    負樣本：圖片 i + 下一張圖片的 caption
    """

    positive_image_ids = np.repeat(
        np.arange(num_images), len(CAPTION_COLUMNS)
    )
    caption_ids = np.arange(num_images * len(CAPTION_COLUMNS))

    negative_image_ids = positive_image_ids.copy()
    negative_caption_ids = (
        (positive_image_ids // len(CAPTION_COLUMNS) + 1)
        % num_images
    ) * len(CAPTION_COLUMNS) + (
        caption_ids % len(CAPTION_COLUMNS)
    )

    image_ids = np.concatenate([
        positive_image_ids,
        negative_image_ids,
    ])

    caption_ids = np.concatenate([
        caption_ids,
        negative_caption_ids,
    ])

    labels = np.concatenate([
        np.ones(len(positive_image_ids), dtype=np.int64),
        np.zeros(len(negative_image_ids), dtype=np.int64),
    ])

    return (
        torch.tensor(image_ids, dtype=torch.long),
        torch.tensor(caption_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# ==================== Image Feature Extraction ====================

def load_image_backbone(backbone_name, device):
    if backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
        transform = weights.transforms()
        feature_dim = 512

    elif backbone_name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        model.heads = nn.Identity()
        transform = weights.transforms()
        feature_dim = 768

    else:
        raise ValueError(f"Unknown image backbone: {backbone_name}")

    model = model.to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, transform, feature_dim


@torch.inference_mode()
def extract_image_features(dataset, backbone_name, device, batch_size, split, cache_dir):
    cache_path = cache_dir / f"image_{backbone_name}_{split}_{len(dataset)}.pt"

    if cache_path.exists():
        print(f"Loading cached image features: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    model, transform, _ = load_image_backbone(backbone_name, device)

    image_dataset = FlickrImageDataset(dataset, transform)
    loader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    features = []

    print(f"Extracting image features using {backbone_name}...")

    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        features.append(output.cpu())

    features = torch.cat(features, dim=0).float()
    torch.save(features, cache_path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return features


# ==================== Text Feature Extraction ====================

@torch.inference_mode()
def extract_text_features(captions, device, batch_size, split, cache_dir):
    cache_path = cache_dir / f"text_distilbert_{split}_{len(captions)}.pt"

    if cache_path.exists():
        print(f"Loading cached text features: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print("Extracting text features using DistilBERT...")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    features = []

    for start in range(0, len(captions), batch_size):
        batch_captions = captions[start:start + batch_size]

        tokens = tokenizer(
            batch_captions,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        tokens = {
            key: value.to(device)
            for key, value in tokens.items()
        }

        output = model(**tokens)

        # 使用 [CLS] 位置的向量代表整段文字
        text_features = output.last_hidden_state[:, 0, :]
        features.append(text_features.cpu())

    features = torch.cat(features, dim=0).float()
    torch.save(features, cache_path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return features


# ==================== Fusion Classifier ====================

class FusionClassifier(nn.Module):
    def __init__(self, image_dim, text_dim=768):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(image_dim + text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, image_features, text_features):
        fused_features = torch.cat(
            [image_features, text_features], dim=1
        )
        return self.classifier(fused_features)


# ==================== Training and Evaluation ====================

def evaluate(model, image_features, text_features, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.inference_mode():
        for image_ids, caption_ids, labels in loader:
            image_batch = image_features[image_ids].to(device)
            text_batch = text_features[caption_ids].to(device)
            labels = labels.to(device)

            logits = model(image_batch, text_batch)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return {
        "loss": float(total_loss / len(loader.dataset)),
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "precision": float(precision_score(
            all_labels, all_predictions, zero_division=0
        )),
        "recall": float(recall_score(
            all_labels, all_predictions, zero_division=0
        )),
        "f1": float(f1_score(
            all_labels, all_predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(
            all_labels, all_probabilities
        )),
    }


def train(args):
    set_seed(SEED)
    RESULT_DIR_MODEL = RESULT_DIR / args.image_backbone
    CACHE_DIR = RESULT_DIR_MODEL / "feature_cache"

    RESULT_DIR_MODEL.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Image backbone: {args.image_backbone}")

    # Load official train / validation / test splits
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    required_splits = ["train", "validation", "test"]
    for split in required_splits:
        if split not in dataset:
            raise ValueError(f"Dataset is missing split: {split}")

    feature_sets = {}
    pair_sets = {}

    for split in required_splits:
        hf_split = dataset[split]
        captions = get_captions(hf_split)

        print(f"\nPreparing {split} split...")
        print(f"Images: {len(hf_split)}")
        print(f"Captions: {len(captions)}")

        image_features = extract_image_features(
            hf_split, args.image_backbone, device,
            args.feature_batch_size, split, CACHE_DIR
        )

        text_features = extract_text_features(
            captions, device, args.feature_batch_size, split, CACHE_DIR
        )

        image_ids, caption_ids, labels = build_pairs(len(hf_split))

        feature_sets[split] = (image_features, text_features)
        pair_sets[split] = TensorDataset(
            image_ids, caption_ids, labels
        )

        print(
            f"Pairs: {len(labels)} | "
            f"Match: {(labels == 1).sum().item()} | "
            f"Not Match: {(labels == 0).sum().item()}"
        )

    image_dim = feature_sets["train"][0].shape[1]

    model = FusionClassifier(image_dim=image_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        pair_sets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        pair_sets["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        pair_sets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    history = []
    best_val_f1 = -1.0
    best_path = RESULT_DIR_MODEL / f"{args.image_backbone}_best.pth"
    history_path = RESULT_DIR_MODEL / f"{args.image_backbone}_history.json"
    result_path = RESULT_DIR_MODEL / f"{args.image_backbone}_results.json"

    print("\n========== Training ==========")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        train_images, train_texts = feature_sets["train"]

        for image_ids, caption_ids, labels in train_loader:
            image_batch = train_images[image_ids].to(device)
            text_batch = train_texts[caption_ids].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(image_batch, text_batch)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

        train_loss = total_loss / total_samples

        val_metrics = evaluate(
            model,
            *feature_sets["validation"],
            val_loader,
            device,
            criterion,
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_backbone": args.image_backbone,
                    "image_feature_dim": image_dim,
                    "text_model": TEXT_MODEL_NAME,
                    "best_epoch": epoch + 1,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    # Save training history
    history_path = RESULT_DIR_MODEL / f"{args.image_backbone}_history.json"
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)

    # Load best validation checkpoint and evaluate test set once
    checkpoint = torch.load(
        best_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model,
        *feature_sets["test"],
        test_loader,
        device,
        criterion,
    )

    results = {
        "image_backbone": args.image_backbone,
        "text_backbone": TEXT_MODEL_NAME,
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics,
    }

    result_path = RESULT_DIR_MODEL / f"{args.image_backbone}_results.json"
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\n========== Final Test Results ==========")
    print(f"Best epoch: {checkpoint['best_epoch']}")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nModel saved to: {best_path}")
    print(f"Results saved to: {result_path}")


def compare_backbones():
    backbones = ["resnet18", "vit_b_16"]
    rows = []

    for backbone in backbones:
        result_path = (
            RESULT_DIR
            / backbone
            / f"{backbone}_results.json"
        )

        if not result_path.exists():
            print(f"Missing result file: {result_path}")
            continue

        with open(result_path, "r", encoding="utf-8") as file:
            results = json.load(file)

        metrics = results["test_metrics"]

        rows.append({
            "Image Backbone": backbone,
            "Text Backbone": results["text_backbone"],
            "Best Epoch": results["best_epoch"],
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"],
        })

    if not rows:
        print("No experiment results found.")
        return

    import pandas as pd

    df = pd.DataFrame(rows)

    print("\n========== Backbone Comparison ==========")
    print(df.round(4).to_string(index=False))

    output_path = RESULT_DIR / "backbone_comparison.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal Image-Text Matching Classification"
    )
    parser.add_argument(
        "--image-backbone",
        choices=["resnet18", "vit_b_16"],
        default="resnet18",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    train(args)
    compare_backbones()
