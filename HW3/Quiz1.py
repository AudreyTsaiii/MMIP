import os
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "./data"
SPLIT_DIR = "./data_split"
os.makedirs(SPLIT_DIR, exist_ok=True)

print("torch:", torch.__version__, "| torchvision:", torchvision.__version__)

to_tensor = T.Compose([T.ToTensor()])

train_full = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=to_tensor)
test_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=to_tensor)

CLASS_NAMES = train_full.classes
print("訓練集大小:", len(train_full))
print("測試集大小:", len(test_set))
print("類別:", CLASS_NAMES)

# 檢查每個類別的數量
train_labels = np.array(train_full.targets)
test_labels = np.array(test_set.targets)

print(f"{'Class':<12}{'Train':>8}{'Test':>8}")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<12}{(train_labels == i).sum():>8}{(test_labels == i).sum():>8}")

# 每個類別各取 5 張圖，確認資料長相正常、標籤沒有錯位
fig, axes = plt.subplots(10, 5, figsize=(4, 8))

for class_idx, class_name in enumerate(CLASS_NAMES):
    idxs = np.where(train_labels == class_idx)[0][:5]

    for col, idx in enumerate(idxs):
        img, _ = train_full[idx]

        h, w = img.shape[1], img.shape[2]

        fig, ax = plt.subplots(figsize=(w / 100, h / 100))

        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis("off")

plt.show()

plt.tight_layout()
plt.show()
print(train_full[0][0].shape)


N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

folds = {}
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_labels)), train_labels)):
    folds[f"fold{fold_idx}_train"] = train_idx
    folds[f"fold{fold_idx}_val"] = val_idx
    print(f"Fold {fold_idx}: train={len(train_idx)}  val={len(val_idx)}")

# 驗證每一折確實是 stratified: 檢查 fold0 的 val set 裡每個類別的比例，應該要接近 10% 各類別
fold0_val_labels = train_labels[folds["fold0_val"]]
print(f"{'Class':<12}{'Fold0-Val Count':>16}{'比例':>10}")
for i, name in enumerate(CLASS_NAMES):
    count = (fold0_val_labels == i).sum()
    print(f"{name:<12}{count:>16}{count/len(fold0_val_labels):>10.1%}")

# 確認 train/val 完全沒有重疊
overlap = set(folds["fold0_train"]).intersection(set(folds["fold0_val"]))
assert len(overlap) == 0, "train/val 有重疊，切分有誤！"
print("\nfold0 train/val 無重疊，切分正確。")


# 存下所有 fold 的 index，Quiz 2, Quiz 3 直接讀取
np.savez(os.path.join(SPLIT_DIR, "cifar10_folds.npz"), **folds)

meta = {
    "n_folds": N_FOLDS,
    "default_fold": 0,
    "seed": SEED,
    "class_names": CLASS_NAMES,
    "train_pool_size": len(train_full),
    "test_size": len(test_set),
}
with open(os.path.join(SPLIT_DIR, "split_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("已存檔：")
print(f"  {SPLIT_DIR}/cifar10_folds.npz: 5 fold 的 train/val index")
print(f"  {SPLIT_DIR}/split_meta.json: 切分設定")
