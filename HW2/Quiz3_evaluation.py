"""
Quiz 3：模型表現評估
資料集：Default of Credit Card Clients Dataset (UCI_Credit_Card.csv)

基礎：
1. 延續 Quiz 2 的信用卡違約預測資料集與 MLP 模型
2. 取得 MLP 對 Validation Dataset 的 Prediction Probability
3. 繪製 ROC Curve
4. 計算 AUC
5. 將 AUC 標示於 ROC Curve

進階：
1. 使用相同 Training / Validation Dataset
2. 使用 Logistic Regression 建立第二個分類模型
3. 取得 Logistic Regression 的 Prediction Probability
4. 將 MLP 與 Logistic Regression ROC Curve 畫在同一張圖
5. 計算兩個模型的 AUC
6. 比較兩個模型的正負樣本區分能力
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt


# ============================================================
# 1. 設定
# ============================================================

torch.manual_seed(42)

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


# ============================================================
# 2. 讀取資料
# ============================================================

df = pd.read_csv("UCI_Credit_Card.csv")

print("=" * 60)
print("Dataset Information")
print("=" * 60)

print("Data shape:", df.shape)
print("Positive rate:", df["default.payment.next.month"].mean())


# ============================================================
# 3. Features / Label
# ============================================================

X = df.drop(columns=["ID", "default.payment.next.month"])
y = df["default.payment.next.month"]


# ============================================================
# 4. Train / Validation Split
#    與 Quiz 2 完全相同
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# 5. Feature Scaling
#    只使用 Training Dataset fit
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# ============================================================
# 6. Convert to PyTorch Tensor
# ============================================================

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(
    y_val.values,
    dtype=torch.float32
).view(-1, 1)


# ============================================================
# 7. DataLoader
# ============================================================

train_ds = TensorDataset(X_train_t, y_train_t)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# 8. MLP Model
#    延續 Quiz 2 的 Baseline MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 9. 建立 MLP
# ============================================================

input_dim = X_train_t.shape[1]

model = MLP(input_dim)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 10. Training MLP
# ============================================================

print("\n" + "=" * 60)
print("Training MLP")
print("=" * 60)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0

    for xb, yb in train_loader:

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)

    train_loss = epoch_loss / len(train_ds)

    train_losses.append(train_loss)


    # Validation Loss
    model.eval()

    with torch.no_grad():

        val_logits = model(X_val_t)

        val_loss = criterion(
            val_logits,
            y_val_t
        ).item()

    val_losses.append(val_loss)


    if epoch % 10 == 0:

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )


# ============================================================
# 11. MLP Validation Prediction Probability
# ============================================================

model.eval()

with torch.no_grad():

    mlp_logits = model(X_val_t)

    mlp_probs = torch.sigmoid(
        mlp_logits
    ).numpy().flatten()


# ============================================================
# 12. MLP ROC Curve
# ============================================================

mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(
    y_val,
    mlp_probs
)

mlp_auc = roc_auc_score(
    y_val,
    mlp_probs
)


print("\n" + "=" * 60)
print("MLP ROC / AUC")
print("=" * 60)

print(f"MLP AUC: {mlp_auc:.4f}")


# ============================================================
# 13. Logistic Regression
#     使用完全相同的 Training / Validation Dataset
# ============================================================

print("\n" + "=" * 60)
print("Training Logistic Regression")
print("=" * 60)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 14. Logistic Regression Prediction Probability
# ============================================================

lr_probs = lr_model.predict_proba(
    X_val_scaled
)[:, 1]


# ============================================================
# 15. Logistic Regression ROC Curve
# ============================================================

lr_fpr, lr_tpr, lr_thresholds = roc_curve(
    y_val,
    lr_probs
)

lr_auc = roc_auc_score(
    y_val,
    lr_probs
)


print("\n" + "=" * 60)
print("Logistic Regression ROC / AUC")
print("=" * 60)

print(f"Logistic Regression AUC: {lr_auc:.4f}")


# ============================================================
# 16. Model Comparison
# ============================================================

print("\n" + "=" * 60)
print("Model Comparison")
print("=" * 60)

comparison = pd.DataFrame({
    "Model": [
        "MLP",
        "Logistic Regression"
    ],
    "AUC": [
        mlp_auc,
        lr_auc
    ]
})

print(comparison.to_string(index=False))


# ============================================================
# 17. ROC Curve Comparison
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    mlp_fpr,
    mlp_tpr,
    label=f"MLP (AUC = {mlp_auc:.4f})"
)

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_auc:.4f})"
)

# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")

plt.title("ROC Curve Comparison")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "quiz3_roc_comparison.png",
    dpi=150
)

plt.show()


print("\nROC curve saved to:")
print("quiz3_roc_comparison.png")


# ============================================================
# 18. Training / Validation Loss
#     可選：保留 Quiz 2 的 MLP 訓練結果
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("MLP Training vs Validation Loss")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "quiz3_mlp_loss.png",
    dpi=150
)

plt.show()


print("MLP loss plot saved to:")
print("quiz3_mlp_loss.png")


# ============================================================
# 19. Summary
# ============================================================

print("\n" + "=" * 60)
print("Final Summary")
print("=" * 60)

print(f"MLP AUC:                 {mlp_auc:.4f}")
print(f"Logistic Regression AUC: {lr_auc:.4f}")

if mlp_auc > lr_auc:
    print("MLP has a higher AUC in this validation dataset.")
elif lr_auc > mlp_auc:
    print("Logistic Regression has a higher AUC in this validation dataset.")
else:
    print("Both models have the same AUC.")