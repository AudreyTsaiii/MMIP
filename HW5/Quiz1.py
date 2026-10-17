
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (resnet18, ResNet18_Weights,vit_b_16, ViT_B_16_Weights,)
from sklearn.model_selection import train_test_split


SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
EMBED_DIM = 256
TEMPERATURE = 0.1
NUM_WORKERS = 0

DATA_DIR = Path("C:/Users/User/Downloads/ImageProcessing/HW4/data_quiz1/raw/stl10_binary")
OUTPUT_DIR = Path("./results_quiz1")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])


def build_backbone(model_name):
    if model_name == "ResNet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        feature_dim = model.heads.head.in_features
        model.heads = nn.Identity()

    model = model.to(DEVICE)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model, feature_dim


class RetrievalModel(nn.Module):
    def __init__(self, backbone, feature_dim):
        super().__init__()
        self.backbone = backbone
        self.embedding = nn.Sequential(nn.Linear(feature_dim, 512),nn.ReLU(),nn.Linear(512, EMBED_DIM))

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)

        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)


def supervised_contrastive_loss(embeddings, labels):
    similarity = torch.matmul(embeddings, embeddings.T)
    logits = similarity / TEMPERATURE

    batch_size = labels.size(0)
    self_mask = torch.eye(batch_size, device=labels.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    positive_mask = positive_mask & ~self_mask

    logits = logits.masked_fill(self_mask, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0

    if not valid.any():
        return embeddings.sum() * 0.0

    mean_positive_log_prob = ((log_prob * positive_mask.float()).sum(dim=1)/ positive_count.clamp(min=1))

    return -mean_positive_log_prob[valid].mean()


@torch.no_grad()
def extract_features(backbone, loader):
    backbone.eval()
    features_list = []
    labels_list = []

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(DEVICE)
        features = backbone(images)

        features_list.append(features.cpu())
        labels_list.append(labels)

        if batch_idx % 100 == 0:
            print(f"  Extracted {batch_idx} batches")

    features = torch.cat(features_list)
    labels = torch.cat(labels_list)

    print(f"Feature extraction: {len(labels)} images, "f"{time.time() - start_time:.1f}s")
    return features, labels


def make_feature_loader(dataset, indices, batch_size):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def train_embedding(model_name, train_features, train_labels,
                    val_features, val_labels):
    print(f"\n========== Training {model_name} embedding ==========")

    backbone, feature_dim = build_backbone(model_name)
    model = RetrievalModel(backbone, feature_dim).to(DEVICE)

    optimizer = torch.optim.AdamW(model.embedding.parameters(),lr=LEARNING_RATE, weight_decay=1e-4)

    train_features = train_features.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    val_features = val_features.to(DEVICE)
    val_labels = val_labels.to(DEVICE)

    best_val_loss = float("inf")
    history = []

    model_path = OUTPUT_DIR / f"{model_name.lower()}_best.pth"

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        permutation = torch.randperm(len(train_features), device=DEVICE)
        total_loss = 0.0
        total_samples = 0

        for start_idx in range(0, len(permutation), BATCH_SIZE):
            idx = permutation[start_idx:start_idx + BATCH_SIZE]
            batch_features = train_features[idx]
            batch_labels = train_labels[idx]

            optimizer.zero_grad()
            embeddings = F.normalize(model.embedding(batch_features), p=2, dim=1)
            loss = supervised_contrastive_loss(embeddings, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            total_samples += len(idx)

        train_loss = total_loss / total_samples

        model.eval()
        with torch.no_grad():
            val_embeddings = F.normalize(model.embedding(val_features), p=2, dim=1)
            val_loss = supervised_contrastive_loss(val_embeddings, val_labels).item()

        elapsed = time.time() - start
        history.append({
            "model": model_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_seconds": elapsed,
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.embedding.state_dict(), model_path)
            print("  Saved best embedding.")

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / f"{model_name.lower()}_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model.embedding.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    return model, backbone


@torch.no_grad()
def embed_features(model, features):
    model.eval()
    outputs = []

    for start_idx in range(0, len(features), BATCH_SIZE):
        batch = features[start_idx:start_idx + BATCH_SIZE].to(DEVICE)
        embeddings = model.embedding(batch)
        outputs.append(F.normalize(embeddings, p=2, dim=1).cpu())

    return torch.cat(outputs)


def retrieval_metrics(query_embeddings, query_labels,
                      database_embeddings, database_labels,
                      query_indices=None, top_k=5):
    # Cosine similarity because embeddings are L2-normalized.
    similarities = query_embeddings @ database_embeddings.T
    similarities = similarities.numpy()

    query_labels = query_labels.numpy()
    database_labels = database_labels.numpy()

    # Remove self-matches when querying against the same dataset.
    if query_indices is not None:
        for i, database_idx in enumerate(query_indices):
            similarities[i, database_idx] = -np.inf

    recall_at_1 = []
    recall_at_5 = []
    ap_at_5 = []

    for i in range(len(query_labels)):
        ranked = np.argsort(-similarities[i])[:top_k]
        relevant = (database_labels[ranked] == query_labels[i]).astype(np.float32)

        recall_at_1.append(float(relevant[:1].sum() > 0))
        recall_at_5.append(float(relevant.sum() > 0))

        precisions = []
        for rank, is_relevant in enumerate(relevant, start=1):
            if is_relevant:
                precisions.append(relevant[:rank].sum() / rank)

        total_relevant = np.sum(database_labels == query_labels[i])
        denominator = min(total_relevant, top_k)

        ap_at_5.append(
            sum(precisions) / denominator
            if denominator > 0 else 0.0
        )

    return {
        "Recall@1": float(np.mean(recall_at_1)),
        "Recall@5": float(np.mean(recall_at_5)),
        "mAP@5": float(np.mean(ap_at_5)),
    }


def show_retrieval_results(
    model_name, query_images, query_labels,
    database_images, database_labels, embeddings,
    output_path, num_queries=5, top_k=5,
):
    similarities = embeddings[:num_queries] @ embeddings.T
    similarities = similarities.numpy()
    labels = query_labels.numpy()
    db_labels = database_labels.numpy()

    fig, axes = plt.subplots(num_queries, top_k + 1,figsize=(2.0 * (top_k + 1), 2.4 * num_queries))

    for row in range(num_queries):
        query_idx = row
        scores = similarities[row].copy()
        scores[query_idx] = -np.inf
        top_indices = np.argsort(-scores)[:top_k]

        axes[row, 0].imshow(query_images[query_idx])
        axes[row, 0].set_title(f"Query\n{CLASSES[int(labels[query_idx])]}")
        axes[row, 0].axis("off")

        for col, db_idx in enumerate(top_indices, start=1):
            axes[row, col].imshow(database_images[int(db_idx)])
            match = db_labels[int(db_idx)] == labels[query_idx]
            axes[row, col].set_title(
                f"{CLASSES[int(db_labels[db_idx])]}\n"
                f"{'Match' if match else 'Non-match'}"
            )
            axes[row, col].axis("off")

    fig.suptitle(f"{model_name} Image Retrieval", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading STL-10...")

    transform = get_transform()
    train_dataset = datasets.STL10(
        root=str(DATA_DIR.parent),
        split="train",
        download=False,
        transform=transform
    )

    test_dataset = datasets.STL10(
        root=str(DATA_DIR.parent),
        split="test",
        download=False,
        transform=transform
    )
    test_images = np.transpose(test_dataset.data, (0, 2, 3, 1))

    labels = np.asarray(train_dataset.labels)
    indices = np.arange(len(train_dataset))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=1000,
        random_state=SEED,
        stratify=labels
    )

    CLASSES = [
        "airplane", "bird", "car", "cat", "deer",
        "dog", "horse", "monkey", "ship", "truck"
    ]

    print(f"Train: {len(train_idx)}")
    print(f"Validation: {len(val_idx)}")
    print(f"Test: {len(test_dataset)}")
    with open(OUTPUT_DIR / "split_indices.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "train_indices": train_idx.tolist(),
            "validation_indices": val_idx.tolist(),
            "test_size": len(test_dataset),
        }, f)

    train_loader = make_feature_loader(train_dataset, train_idx, BATCH_SIZE)
    val_loader = make_feature_loader(train_dataset, val_idx, BATCH_SIZE)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    results = []

    for model_name in ["ResNet18", "ViT"]:
        print(f"\n========== {model_name} ==========")
        backbone, feature_dim = build_backbone(model_name)

        train_features, train_labels = extract_features(backbone, train_loader)
        val_features, val_labels = extract_features(backbone, val_loader)
        test_features, test_labels = extract_features(backbone, test_loader)

        model, _ = train_embedding( model_name,train_features, train_labels,val_features, val_labels,)

        test_embeddings = embed_features(model, test_features)

        # Evaluate retrieval among test images.
        # The first 1,000 test images are queries; all test images
        # form the retrieval database.
        num_queries = min(1000, len(test_embeddings))
        query_embeddings = test_embeddings[:num_queries]
        query_labels = test_labels[:num_queries]

        metrics = retrieval_metrics(
            query_embeddings,
            query_labels,
            test_embeddings,
            test_labels,
            query_indices=np.arange(num_queries),
            top_k=5,
        )

        # Measure query-to-database retrieval latency.
        start = time.perf_counter()
        _ = query_embeddings @ test_embeddings.T
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        retrieval_seconds = time.perf_counter() - start
        metrics["retrieval_ms_per_query"] = (retrieval_seconds / num_queries * 1000)
        metrics["model"] = model_name
        results.append(metrics)

        print(f"\n{model_name} Test Retrieval Results:")
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                print(f"{key}: {value:.4f}")

        show_retrieval_results(
            model_name,
            test_images,
            test_labels,
            test_images,
            test_labels,
            test_embeddings,
            OUTPUT_DIR / f"{model_name.lower()}_retrieval.png",
            num_queries=5,
            top_k=5,
        )

        torch.save(model.embedding.state_dict(), OUTPUT_DIR / f"{model_name.lower()}_embedding.pth")

        del model, backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv",index=False, encoding="utf-8-sig")

    print("\n========== Final Comparison ==========")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
