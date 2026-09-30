"""
Quiz 2：深度學習信用卡違約預測
資料集：Default of Credit Card Clients Dataset (UCI_Credit_Card.csv)

流程：
1. 讀檔、切分 train/val、Feature Scaling
2. 基礎 MLP，訓練 100 epochs，畫 Training/Validation Loss
3. 展示 validation 一筆 sample 的預測
4. 進階：
   - Dropout
   - Early Stopping
   - Improved = Dropout + Early Stopping + AdamW
5. 比較不同模型的 Accuracy / Precision / Recall / F1-score
"""

import copy
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# ============================================================
# 0. 基本設定
# ============================================================

torch.manual_seed(42)

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


# ============================================================
# 1. 讀檔、切分、Feature Scaling
# ============================================================

df = pd.read_csv("UCI_Credit_Card.csv")

print("data shape:", df.shape)
print("positive rate:", df["default.payment.next.month"].mean())

X = df.drop(columns=["ID", "default.payment.next.month"])
y = df["default.payment.next.month"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.BCEWithLogitsLoss()

print("input dimension:", X_train_t.shape[1])
print("training samples:", len(train_ds))
print("validation samples:", len(X_val_t))


# ============================================================
# 2. 模型定義
# ============================================================

class MLP(nn.Module):
    """基礎 MLP：兩層 hidden layer，無正則化。"""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class MLPDropout(nn.Module):
    """加入 Dropout 的 MLP。"""

    def __init__(self, input_dim, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(p),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class ImprovedMLP(nn.Module):
    """Improved = Dropout + Early Stopping + AdamW。"""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. 共用 Training Loop
# ============================================================

def train_model(model, optimizer, epochs=EPOCHS, early_stopping=False, patience=5, verbose_name=""):
    train_losses, val_losses = [], []
    best_val_loss, best_state, epochs_no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        train_loss = epoch_loss / len(train_ds)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f"[{verbose_name}] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"[{verbose_name}] Early stopping at epoch {epoch}, best_val_loss={best_val_loss:.4f}")
                break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses


# ============================================================
# 4. Model Evaluation
# ============================================================

def get_metrics(model):
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(X_val_t)).numpy().flatten()

    pred = (probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_val, pred),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0)
    }


# ============================================================
# 5. Baseline MLP
# ============================================================

print("\n" + "=" * 60)
print("1. Baseline MLP")
print("=" * 60)

model = MLP(X_train_t.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses, val_losses = train_model(
    model, optimizer, epochs=EPOCHS, verbose_name="Baseline"
)


# ------------------------------------------------------------
# Validation sample prediction
# ------------------------------------------------------------

model.eval()
sample_x, sample_y = X_val_t[0:1], y_val.iloc[0]

with torch.no_grad():
    prob = torch.sigmoid(model(sample_x)).item()

pred_label = int(prob >= 0.5)

print("\nSample prediction")
print(f"True label: {sample_y}")
print(f"Predicted probability of default: {prob:.4f}")
print(f"Predicted label: {pred_label}")


# ============================================================
# 6. Dropout
# ============================================================

print("\n" + "=" * 60)
print("2. Dropout")
print("=" * 60)

model_dropout = MLPDropout(X_train_t.shape[1], p=0.3)
optimizer_dropout = torch.optim.Adam(model_dropout.parameters(), lr=LEARNING_RATE)

train_losses_dropout, val_losses_dropout = train_model(
    model_dropout, optimizer_dropout, epochs=EPOCHS, verbose_name="Dropout"
)


# ============================================================
# 7. Early Stopping
# ============================================================

print("\n" + "=" * 60)
print("3. Early Stopping")
print("=" * 60)

model_es = MLP(X_train_t.shape[1])
optimizer_es = torch.optim.Adam(model_es.parameters(), lr=LEARNING_RATE)

train_losses_es, val_losses_es = train_model(
    model_es,
    optimizer_es,
    epochs=EPOCHS,
    early_stopping=True,
    patience=5,
    verbose_name="EarlyStopping"
)


# ============================================================
# 8. Improved Model
#    Dropout + Early Stopping + AdamW
# ============================================================

print("\n" + "=" * 60)
print("4. Improved MLP")
print("Dropout + Early Stopping + AdamW")
print("=" * 60)

model_improved = ImprovedMLP(X_train_t.shape[1])
optimizer_improved = torch.optim.AdamW(model_improved.parameters(), lr=LEARNING_RATE)

train_losses_improved, val_losses_improved = train_model(
    model_improved,
    optimizer_improved,
    epochs=EPOCHS,
    early_stopping=True,
    patience=5,
    verbose_name="Improved"
)


# ============================================================
# 9. Model Comparison
# ============================================================

comparison = pd.DataFrame([
    {"Model": "Baseline MLP", **get_metrics(model)},
    {"Model": "Dropout", **get_metrics(model_dropout)},
    {"Model": "Early Stopping", **get_metrics(model_es)},
    {"Model": "Improved", **get_metrics(model_improved)}
])

print("\n" + "=" * 60)
print("Model Comparison")
print("=" * 60)
print(comparison.to_string(index=False))


# ============================================================
# 10. Training / Validation Loss Comparison
# ============================================================

plt.figure(figsize=(10, 7))

plt.plot(train_losses, label="Baseline - Train", linestyle="--")
plt.plot(val_losses, label="Baseline - Val")

plt.plot(train_losses_dropout, label="Dropout - Train", linestyle="--")
plt.plot(val_losses_dropout, label="Dropout - Val")

plt.plot(train_losses_es, label="Early Stopping - Train", linestyle="--")
plt.plot(val_losses_es, label="Early Stopping - Val")

plt.plot(train_losses_improved, label="Improved - Train", linestyle="--")
plt.plot(val_losses_improved, label="Improved - Val")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("quiz2_loss_comparison.png", dpi=150)
plt.show()

print("\nSaved plot to quiz2_loss_comparison.png")


# ============================================================
# 11. F1-score Comparison
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(comparison["Model"], comparison["f1"])

plt.xlabel("Model")
plt.ylabel("F1-score")
plt.title("F1-score Comparison")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("quiz2_f1_comparison.png", dpi=150)
plt.show()

print("Saved plot to quiz2_f1_comparison.png")