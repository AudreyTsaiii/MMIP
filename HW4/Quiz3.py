
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (vit_b_16, ViT_B_16_Weights,resnet18, ResNet18_Weights)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# ==================== Configuration ====================
SEED = 42
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_CLASSES = 10
NUM_WORKERS = 0

DATA_DIR = Path("./data_quiz1/raw")
OUTPUT_DIR = Path("./results_quiz3")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, eval_transform


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    labels = np.asarray(all_labels)
    preds = np.asarray(all_preds)
    probs = np.asarray(all_probs)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels, preds),
        "macro_auc": roc_auc_score(
            labels, probs, multi_class="ovr", average="macro",
            labels=np.arange(NUM_CLASSES),
        ),
    }
    return metrics, labels, preds, probs


def build_model(model_name):
    if model_name == "ViT":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(
            model.heads.head.in_features, NUM_CLASSES
        )
    else:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model.to(DEVICE)


def train_model(model_name, train_loader, val_loader):
    print(f"\n========== Training {model_name} ==========")
    model = build_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )

    best_val_acc = -1.0
    history = []
    model_path = OUTPUT_DIR / f"{model_name.lower()}_best.pth"

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion)
        elapsed = time.time() - start_time

        history.append({
            "model": model_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_auc": val_metrics["macro_auc"],
            "epoch_seconds": elapsed,
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-AUC: {val_metrics['macro_auc']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), model_path)

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / f"{model_name.lower()}_training_history.csv",
        index=False, encoding="utf-8-sig"
    )

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading STL-10...")

    train_transform, eval_transform = get_transforms()

    train_full = datasets.STL10(
        root=str(DATA_DIR), split="train",
        download=True, transform=train_transform,
    )
    val_full = datasets.STL10(
        root=str(DATA_DIR), split="train",
        download=False, transform=eval_transform,
    )
    test_ds = datasets.STL10(
        root=str(DATA_DIR), split="test",
        download=True, transform=eval_transform,
    )

    indices = np.arange(len(train_full))
    labels = np.asarray(train_full.labels)

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.1,
        random_state=SEED,
        shuffle=True,
        stratify=labels,
    )

    split_info = {
        "seed": SEED,
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "test_indices": list(range(len(test_ds))),
    }
    with (OUTPUT_DIR / "split_indices.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(split_info, f)

    train_ds = Subset(train_full, train_idx.tolist())
    val_ds = Subset(val_full, val_idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(
        f"Train: {len(train_ds)} | Validation: {len(val_ds)} | "
        f"Test: {len(test_ds)}"
    )

    results = []
    all_predictions = []

    for model_name in ["ViT", "ResNet18"]:
        model = train_model(model_name, train_loader, val_loader)
        criterion = nn.CrossEntropyLoss()
        metrics, labels, preds, probs = evaluate(
            model, test_loader, criterion
        )

        results.append({"model": model_name, **metrics})

        print(f"\n{model_name} Test Results:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        for i in range(len(labels)):
            record = {
                "model": model_name,
                "true_label_id": int(labels[i]),
                "true_label": CLASSES[int(labels[i])],
                "predicted_label_id": int(preds[i]),
                "predicted_label": CLASSES[int(preds[i])],
            }
            for class_id, class_name in enumerate(CLASSES):
                record[f"prob_{class_name}"] = float(probs[i, class_id])
            all_predictions.append(record)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(all_predictions).to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False, encoding="utf-8-sig",
    )

    print("\nFinal model comparison:")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
