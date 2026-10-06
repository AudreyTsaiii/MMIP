import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from torchvision.models import ResNet18_Weights

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data"
SPLIT_DIR = "./data_split"
RESULT_DIR = "./results_quiz2"
PLAIN_DIR = os.path.join(RESULT_DIR, "plain_cnn")
RESNET_DIR = os.path.join(RESULT_DIR, "resnet18")

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 10

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLAIN_DIR, exist_ok=True)
os.makedirs(RESNET_DIR, exist_ok=True)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("=" * 70)
print("Quiz 2: CIFAR-10 CNN Image Classification")
print("=" * 70)
print("Device:", DEVICE)

def build_transform(img_size=32, imagenet_normalize=False):
    transforms = [
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ]

    if imagenet_normalize:
        transforms.append(
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

    return T.Compose(transforms)


def build_loaders(img_size=32, batch_size=64, imagenet_normalize=False):
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=build_transform(img_size, imagenet_normalize=imagenet_normalize)
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=build_transform(img_size, imagenet_normalize=imagenet_normalize)
    )

    folds = np.load(os.path.join(SPLIT_DIR, "cifar10_folds.npz"))
    train_idx = folds["fold0_train"]
    val_idx = folds["fold0_val"]

    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_set)}")
    print(f"Validation: {len(val_set)}")
    print(f"Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# 先保留一個預設的 dataloader，讓後續函式可以直接使用
plain_train_loader, plain_val_loader, plain_test_loader = build_loaders(img_size=32, batch_size=BATCH_SIZE, imagenet_normalize=False)


class PlainCNN(nn.Module):
    def __init__(self, num_classes=10, img_size=32):
        super().__init__()
        self.img_size = img_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        reduced_size = img_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced_size * reduced_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# def create_resnet18(num_classes=10):
#     model = resnet18(weights=None)

#     # CIFAR-10 是 32x32，因此修改 ResNet18 的第一層
#     model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#     model.maxpool = nn.Identity()
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model

def create_resnet18(num_classes=10, pretrained=True, img_size=32):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # CIFAR-10 的圖片較小，通常不需要太早的 downsample；img_size 會真實影響輸入資料大小
    if img_size <= 32:
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def topk_accuracy(outputs, targets, ks=(1, 5)):
    max_k = max(ks)
    _, pred = outputs.topk(max_k, dim=1)
    pred = pred.t()

    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = []

    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum()
        results.append((correct_k / targets.size(0)).item())

    return results


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0
    total_samples = 0
    total_top1 = 0
    total_top5 = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        top1, top5 = topk_accuracy(outputs, labels)
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size

    return (total_loss / total_samples, total_top1 / total_samples, total_top5 / total_samples)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    total_samples = 0
    total_top1 = 0
    total_top5 = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        top1, top5 = topk_accuracy(outputs, labels)
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size

    return (total_loss / total_samples, total_top1 / total_samples, total_top5 / total_samples)


def train_model(model, learning_rate, epochs, model_name, train_loader, val_loader):
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_top1": [],
        "val_top1": [],
        "train_top5": [],
        "val_top5": []
    }

    best_val_top1 = 0
    output_dir = get_model_output_dir(model_name)
    model_path = os.path.join(output_dir, f"{model_name}_best.pth")

    print("\n" + "=" * 70)
    print(f"Training: {model_name}")
    print("=" * 70)
    print(f"Parameters: {count_parameters(model):,}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_top1, train_top5 = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_top1"].append(train_top1)
        history["val_top1"].append(val_top1)
        history["train_top5"].append(train_top5)
        history["val_top5"].append(val_top5)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Top-1: {train_top1:.4f} | "
            f"Val Top-1: {val_top1:.4f} | "
            f"Val Top-5: {val_top5:.4f}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(model.state_dict(), model_path)

    elapsed = time.time() - start_time

    print(f"\nTraining time: {elapsed / 60:.2f} minutes")
    print(f"Best Validation Top-1: {best_val_top1:.4f}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    return model, history

def get_model_output_dir(model_name):
    name = model_name.lower()
    if "resnet" in name:
        output_dir = RESNET_DIR
    else:
        output_dir = PLAIN_DIR

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_history(history, model_name):
    epochs = range(1, len(history["train_loss"]) + 1)
    output_dir = get_model_output_dir(model_name)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss.png"))
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_top1"], label="Train Top-1")
    plt.plot(epochs, history["val_top1"], label="Validation Top-1")
    plt.plot(epochs, history["val_top5"], label="Validation Top-5")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_accuracy.png"))
    plt.show()

@torch.no_grad()
def predict_test(model, model_name, test_loader):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.concatenate(all_probabilities, axis=0)

    test_top1 = (y_true == y_pred).mean()

    top5_correct = 0

    for i in range(len(y_true)):
        top5 = np.argsort(y_prob[i])[-5:]
        if y_true[i] in top5:
            top5_correct += 1

    test_top5 = top5_correct / len(y_true)

    print("\n" + "=" * 70)
    print(f"{model_name} Test Results")
    print("=" * 70)
    print(f"Test Top-1 Accuracy: {test_top1:.4f}")
    print(f"Test Top-5 Accuracy: {test_top5:.4f}")

    rows = []

    for i in range(len(y_true)):
        top5_indices = np.argsort(y_prob[i])[-5:][::-1]

        rows.append({
            "image_id": i,
            "true_label": CLASS_NAMES[y_true[i]],
            "predicted_label": CLASS_NAMES[y_pred[i]],
            "prediction_probability": float(y_prob[i][y_pred[i]]),
            "top5_predictions": ",".join(CLASS_NAMES[j] for j in top5_indices)
        })

    prediction_df = pd.DataFrame(rows)

    output_dir = get_model_output_dir(model_name)
    prediction_path = os.path.join(output_dir, f"{model_name}_test_predictions.csv")
    prediction_df.to_csv(prediction_path, index=False)

    print(f"Prediction saved: {prediction_path}")

    return y_true, y_pred, y_prob, test_top1, test_top5


def plot_roc_curve(y_true, y_prob, model_name):
    y_true_binary = label_binarize(y_true, classes=np.arange(NUM_CLASSES))

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    macro_auc = roc_auc_score(y_true_binary, y_prob, average="macro")

    plt.figure(figsize=(9, 7))

    for i in range(NUM_CLASSES):
        plt.plot(fpr[i], tpr[i], label=f"{CLASS_NAMES[i]} (AUC={roc_auc[i]:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()

    output_dir = get_model_output_dir(model_name)
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc.png"))
    plt.show()

    print(f"{model_name} Macro-AUC: {macro_auc:.4f}")

    auc_df = pd.DataFrame({"Class": CLASS_NAMES, "AUC": [roc_auc[i] for i in range(NUM_CLASSES)]})

    auc_df.loc[len(auc_df)] = ["Macro-AUC", macro_auc]
    auc_df.to_csv(os.path.join(output_dir, f"{model_name}_auc.csv"), index=False)

    return macro_auc


def run_baseline():
    results = []

    plain_train_loader, plain_val_loader, plain_test_loader = build_loaders(img_size=32, batch_size=BATCH_SIZE, imagenet_normalize=False)
    plain_model = PlainCNN(num_classes=NUM_CLASSES, img_size=32)
    plain_model, plain_history = train_model(plain_model, LEARNING_RATE, EPOCHS, "plain_cnn", plain_train_loader, plain_val_loader)
    plot_history(plain_history, "plain_cnn")

    criterion = nn.CrossEntropyLoss()
    _, val_top1, val_top5 = evaluate(plain_model, plain_val_loader, criterion)
    y_true, _, y_prob, test_top1, test_top5 = predict_test(plain_model, "plain_cnn", plain_test_loader)
    macro_auc = plot_roc_curve(y_true, y_prob, "plain_cnn")

    results.append({
        "Model": "Plain CNN",
        "Parameters": count_parameters(plain_model),
        "Val Top-1": val_top1,
        "Val Top-5": val_top5,
        "Test Top-1": test_top1,
        "Test Top-5": test_top5,
        "Macro-AUC": macro_auc
    })

    resnet_train_loader, resnet_val_loader, resnet_test_loader = build_loaders(img_size=32, batch_size=BATCH_SIZE, imagenet_normalize=True)
    resnet_model = create_resnet18(num_classes=NUM_CLASSES, pretrained=True, img_size=32)

    resnet_model, resnet_history = train_model(resnet_model, LEARNING_RATE, EPOCHS, "resnet18", resnet_train_loader, resnet_val_loader)
    plot_history(resnet_history, "resnet18")

    _, val_top1, val_top5 = evaluate(resnet_model, resnet_val_loader, criterion)
    y_true, _, y_prob, test_top1, test_top5 = predict_test(resnet_model, "resnet18", resnet_test_loader)
    macro_auc = plot_roc_curve(y_true, y_prob, "resnet18")

    results.append({
        "Model": "ResNet18",
        "Parameters": count_parameters(resnet_model),
        "Val Top-1": val_top1,
        "Val Top-5": val_top5,
        "Test Top-1": test_top1,
        "Test Top-5": test_top5,
        "Macro-AUC": macro_auc
    })

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("Baseline Model Comparison")
    print("=" * 70)
    print(result_df.to_string(index=False))

    result_df.to_csv(os.path.join(RESULT_DIR, "baseline_comparison.csv"), index=False)


def run_hyperparameter_experiments():
    results = []

    experiments = [
        ("Plain CNN", "plain", 1e-4),
        ("Plain CNN", "plain", 1e-3),
        ("Plain CNN", "plain", 1e-2),
        ("ResNet18", "resnet", 1e-4),
        ("ResNet18", "resnet", 1e-3),
        ("ResNet18", "resnet", 1e-2)
    ]

    for i, (model_name, model_type, lr) in enumerate(experiments, 1):
        print("\n" + "#" * 70)
        print(f"Experiment {i}/{len(experiments)} | {model_name} | lr={lr}")
        print("#" * 70)

        use_imagenet_normalize = model_type == "resnet"
        model_train_loader, model_val_loader, model_test_loader = build_loaders(
            img_size=32,
            batch_size=BATCH_SIZE,
            imagenet_normalize=use_imagenet_normalize
        )
        model = PlainCNN(num_classes=NUM_CLASSES, img_size=32) if model_type == "plain" else create_resnet18(num_classes=NUM_CLASSES, pretrained=True, img_size=32)
        exp_name = f"exp_{i}_{model_type}_lr_{lr}"

        model, history = train_model(model, lr, EPOCHS, exp_name, model_train_loader, model_val_loader)

        criterion = nn.CrossEntropyLoss()
        _, val_top1, val_top5 = evaluate(model, model_val_loader, criterion)

        best_epoch = np.argmax(history["val_top1"]) + 1

        results.append({
            "Experiment": i,
            "Model": model_name,
            "Learning Rate": lr,
            "Best Epoch": best_epoch,
            "Val Top-1": val_top1,
            "Val Top-5": val_top5,
            "Parameters": count_parameters(model)
        })

    result_df = pd.DataFrame(results)
    print("Hyperparameter Experiment Results")
    print(result_df.to_string(index=False))
    result_df.to_csv(os.path.join(RESULT_DIR, "hyperparameter_results.csv"),index=False)
 
if __name__ == "__main__":
    run_baseline()
    run_hyperparameter_experiments()