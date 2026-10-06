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

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data"
SPLIT_DIR = "./data_split"
RESULT_DIR = "./results_quiz3"

PLAIN_DIR = os.path.join(RESULT_DIR, "plain_cnn")
RESNET_DIR = os.path.join(RESULT_DIR, "resnet18")
KERNEL_DIR = os.path.join(RESULT_DIR, "kernel_visualization")
XAI_DIR = os.path.join(RESULT_DIR, "xai")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLAIN_DIR, exist_ok=True)
os.makedirs(RESNET_DIR, exist_ok=True)
os.makedirs(KERNEL_DIR, exist_ok=True)
os.makedirs(XAI_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 20
PLAIN_LR = 1e-3
RESNET_LR = 1e-4
NUM_CLASSES = 10

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print("=" * 70)
print("Quiz 3: Improving Generalization Ability")
print("=" * 70)
print("Device:", DEVICE)


def build_transform(img_size=32, augmentation=False, imagenet_normalize=False):
    transforms = []

    if augmentation:
        transforms.append(T.RandomCrop(img_size, padding=4))
        transforms.append(T.RandomHorizontalFlip())
    elif img_size != 32:
        transforms.append(T.Resize((img_size, img_size)))

    transforms.append(T.ToTensor())

    if imagenet_normalize:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def build_loaders(img_size=32, batch_size=64, augmentation=False, imagenet_normalize=False):
    train_transform = build_transform(img_size=img_size, augmentation=augmentation, imagenet_normalize=imagenet_normalize)
    eval_transform = build_transform(img_size=img_size, augmentation=False, imagenet_normalize=imagenet_normalize)

    train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=eval_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=eval_transform)

    folds = np.load(os.path.join(SPLIT_DIR, "cifar10_folds.npz"))
    train_idx = folds["fold0_train"]
    val_idx = folds["fold0_val"]

    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_set)}")
    print(f"Validation: {len(val_set)}")
    print(f"Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


class PlainCNN(nn.Module):
    def __init__(self, num_classes=10, img_size=32):
        super().__init__()

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


def create_resnet18(num_classes=10, pretrained=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
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

    total_loss = 0.0
    total_samples = 0
    total_top1 = 0.0
    total_top5 = 0.0

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

    return total_loss / total_samples, total_top1 / total_samples, total_top5 / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_top1 = 0.0
    total_top5 = 0.0

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

    return total_loss / total_samples, total_top1 / total_samples, total_top5 / total_samples


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

    best_val_top1 = 0.0

    if "resnet" in model_name.lower():
        output_dir = RESNET_DIR
    else:
        output_dir = PLAIN_DIR

    model_path = os.path.join(output_dir, f"{model_name}_best.pth")

    print("\n" + "=" * 70)
    print(f"Training: {model_name}")
    print("=" * 70)
    print(f"Learning Rate: {learning_rate}")
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

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Top-1: {train_top1:.4f} | Val Top-1: {val_top1:.4f} | Val Top-5: {val_top5:.4f}")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(model.state_dict(), model_path)

    elapsed = time.time() - start_time

    print(f"\nTraining time: {elapsed / 60:.2f} minutes")
    print(f"Best Validation Top-1: {best_val_top1:.4f}")
    print(f"Best model saved to: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    return model, history


def plot_history(history, model_name, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

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
def predict_test(model, model_name, test_loader, output_dir):
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

    prediction_path = os.path.join(output_dir, f"{model_name}_test_predictions.csv")
    prediction_df.to_csv(prediction_path, index=False)

    print(f"Prediction saved to: {prediction_path}")

    return y_true, y_pred, y_prob, test_top1, test_top5


def calculate_macro_auc(y_true, y_prob):
    y_true_binary = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
    return roc_auc_score(y_true_binary, y_prob, average="macro")


def run_experiment(model_type, augmentation):
    if model_type == "plain":
        model_name = "plain_cnn_aug" if augmentation else "plain_cnn_no_aug"
        train_loader, val_loader, test_loader = build_loaders(img_size=32, batch_size=BATCH_SIZE, augmentation=augmentation, imagenet_normalize=False)
        model = PlainCNN(num_classes=NUM_CLASSES, img_size=32)
        output_dir = PLAIN_DIR
        learning_rate = PLAIN_LR

    else:
        model_name = "resnet18_aug" if augmentation else "resnet18_no_aug"
        train_loader, val_loader, test_loader = build_loaders(img_size=32, batch_size=BATCH_SIZE, augmentation=augmentation, imagenet_normalize=True)
        model = create_resnet18(num_classes=NUM_CLASSES, pretrained=True)
        output_dir = RESNET_DIR
        learning_rate = RESNET_LR

    model, history = train_model(model, learning_rate, EPOCHS, model_name, train_loader, val_loader)

    plot_history(history, model_name, output_dir)
    y_true, y_pred, y_prob, test_top1, test_top5 = predict_test(model, model_name, test_loader, output_dir)
    macro_auc = calculate_macro_auc(y_true, y_prob)

    print(f"Test Macro-AUC: {macro_auc:.4f}")

    return {
        "Model": model_name,
        "Augmentation": augmentation,
        "Parameters": count_parameters(model),
        "Best Val Top-1": max(history["val_top1"]),
        "Best Val Top-5": max(history["val_top5"]),
        "Test Top-1": test_top1,
        "Test Top-5": test_top5,
        "Macro-AUC": macro_auc,
        "model": model,
        "history": history,
        "test_loader": test_loader
    }


def visualize_kernels(model):
    model.eval()

    first_conv = model.features[0]
    weights = first_conv.weight.detach().cpu()

    kernel_indices = [0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    for ax, kernel_idx in zip(axes, kernel_indices):
        kernel = weights[kernel_idx].permute(1, 2, 0).numpy()

        kernel_min = kernel.min()
        kernel_max = kernel.max()

        kernel = (kernel - kernel_min) / (kernel_max - kernel_min + 1e-8)

        ax.imshow(kernel)
        ax.set_title(f"Kernel {kernel_idx}")
        ax.axis("off")

    plt.tight_layout()

    output_path = os.path.join(KERNEL_DIR, "plain_cnn_two_kernels.png")
    plt.savefig(output_path, dpi=200)
    plt.show()

    print("\n" + "=" * 70)
    print("Kernel Visualization")
    print("=" * 70)
    print(f"Kernel shape: {weights.shape}")
    print(f"Saved to: {output_path}")


def generate_saliency(model, image, class_index):
    model.eval()

    image = image.clone().detach().to(DEVICE)
    image.requires_grad_(True)

    model.zero_grad()

    output = model(image)
    score = output[:, class_index].sum()
    score.backward()

    gradient = image.grad.detach().abs()

    saliency = gradient.max(dim=1)[0]
    saliency = saliency[0].cpu().numpy()

    saliency_min = saliency.min()
    saliency_max = saliency.max()

    saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)

    return saliency, output.detach()


def denormalize_image(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image.cpu() * std + mean
    image = image.clamp(0, 1)

    return image


def run_saliency(model, test_loader, num_images=6):
    model.eval()

    selected = 0

    for images, labels in test_loader:
        for i in range(images.size(0)):
            if selected >= num_images:
                break

            image = images[i:i + 1].to(DEVICE)
            label = labels[i].item()

            with torch.no_grad():
                output = model(image)
                prediction = output.argmax(dim=1).item()

            saliency, _ = generate_saliency(model, image, prediction)

            original = denormalize_image(images[i]).permute(1, 2, 0).numpy()

            original_tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0)

            original_display = torch.nn.functional.interpolate(
                original_tensor,
                size=(64, 64),
                mode="nearest"
            )[0].permute(1, 2, 0).numpy()

            saliency_display = torch.nn.functional.interpolate(
                torch.from_numpy(saliency).unsqueeze(0).unsqueeze(0),
                size=(64, 64),
                mode="bilinear",
                align_corners=False
            )[0, 0].numpy()

            fig, axes = plt.subplots(1, 3, figsize=(6, 2))

            axes[0].imshow(original_display)
            axes[0].set_title(f"Original\nTrue: {CLASS_NAMES[label]}")
            axes[0].axis("off")

            axes[1].imshow(saliency_display, cmap="jet")
            axes[1].set_title(f"Saliency Map\nPred: {CLASS_NAMES[prediction]}")
            axes[1].axis("off")

            axes[2].imshow(original_display)
            axes[2].imshow(saliency_display, cmap="jet", alpha=0.45)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.tight_layout()

            output_path = os.path.join(XAI_DIR, f"saliency_{selected + 1:02d}.png")
            plt.savefig(output_path, dpi=200)
            plt.show()
            plt.close()

            print(f"Saliency {selected + 1}: True={CLASS_NAMES[label]}, Pred={CLASS_NAMES[prediction]}")
            print(f"Saved to: {output_path}")

            selected += 1

        if selected >= num_images:
            break


def save_comparison(results):
    table = []

    for result in results:
        table.append({
            "Model": result["Model"],
            "Augmentation": result["Augmentation"],
            "Parameters": result["Parameters"],
            "Best Val Top-1": result["Best Val Top-1"],
            "Best Val Top-5": result["Best Val Top-5"],
            "Test Top-1": result["Test Top-1"],
            "Test Top-5": result["Test Top-5"],
            "Macro-AUC": result["Macro-AUC"]
        })

    df = pd.DataFrame(table)

    print("\n" + "=" * 70)
    print("Quiz 3 Comparison")
    print("=" * 70)
    print(df.to_string(index=False))

    output_path = os.path.join(RESULT_DIR, "quiz3_comparison.csv")
    df.to_csv(output_path, index=False)

    print(f"\nComparison saved to: {output_path}")

    return df


def main():
    results = []

    print("\n\n")
    print("=" * 70)
    print("Experiment 1: Plain CNN WITHOUT Data Augmentation")
    print("=" * 70)

    plain_no_aug = run_experiment(model_type="plain", augmentation=False)
    results.append(plain_no_aug)

    print("\n\n")
    print("=" * 70)
    print("Experiment 2: Plain CNN WITH Data Augmentation")
    print("=" * 70)

    plain_aug = run_experiment(model_type="plain", augmentation=True)
    results.append(plain_aug)

    print("\n\n")
    print("=" * 70)
    print("Experiment 3: ResNet18 WITHOUT Data Augmentation")
    print("=" * 70)

    resnet_no_aug = run_experiment(model_type="resnet", augmentation=False)
    results.append(resnet_no_aug)

    print("\n\n")
    print("=" * 70)
    print("Experiment 4: ResNet18 WITH Data Augmentation")
    print("=" * 70)

    resnet_aug = run_experiment(model_type="resnet", augmentation=True)
    results.append(resnet_aug)

    comparison_df = save_comparison(results)

    print("\n\n")
    print("=" * 70)
    print("Advanced Part 1: Kernel Visualization")
    print("=" * 70)

    visualize_kernels(plain_aug["model"])

    print("\n\n")
    print("=" * 70)
    print("Advanced Part 2: Saliency Map XAI")
    print("=" * 70)

    run_saliency(resnet_aug["model"], resnet_aug["test_loader"], num_images=6)

    print("\n\n")
    print("=" * 70)
    print("Quiz 3 Finished")
    print("=" * 70)

    print(f"All results saved under: {RESULT_DIR}")


if __name__ == "__main__":
    main()