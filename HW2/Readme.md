# 機器學習分類任務 — Quiz 1 / Quiz 2 / Quiz 3

本專案包含三份作業：二元分類任務（Bank Marketing Dataset）、深度學習信用卡違約預測（Default of Credit Card Clients Dataset）、以及模型表現評估（ROC / AUC）。以下依作業要求逐項說明方法與實驗結果。

## 環境需求

```
python >= 3.10
pandas
numpy
scikit-learn
torch          # Quiz 2、Quiz 3 的 MLP 使用 PyTorch
matplotlib
```

## 檔案結構

```
.
├── Quiz1_ML.py                  # Quiz 1：Logistic Regression + Random Forest
├── Quiz2_DP.py                # Quiz 2：Baseline MLP + Dropout + Early Stopping + Improved
├── Quiz3_ecaluation.py                 # Quiz 3：MLP vs Logistic Regression 的 ROC / AUC 比較
├── quiz2_loss_comparison.png    # Quiz 2：Training / Validation Loss 曲線比較
├── quiz2_f1_comparison.png      # Quiz 2：四個模型 F1 比較
├── quiz3_roc_comparison.png     # Quiz 3：MLP 與 Logistic Regression 的 ROC 曲線
├── quiz3_mlp_loss.png           # Quiz 3：MLP 訓練過程 Loss 曲線
└── README.md
```

---

## Quiz 1：機器學習分類任務

### 資料集

使用 Kaggle 的 **Bank Marketing Dataset**，目標欄位為 `deposit`（客戶是否申辦定存，yes/no 轉換為 1/0）。資料包含年齡、工作、婚姻狀況、教育程度、餘額、貸款狀況等 17 個原始欄位，其中類別型欄位以 one-hot encoding 轉換為數值特徵，數值型欄位（age、balance、day、duration、campaign、pdays、previous）以 `StandardScaler` 做 Feature Scaling。資料以 `train_test_split(test_size=0.2, stratify=y)` 切分為 Training / Validation Dataset，Validation 共 2,233 筆（實際類別分布：0 類 1,175 筆、1 類 1,058 筆，接近平衡）。

### 基礎部分：Logistic Regression

在 Threshold = 0.5 下的結果：

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.5 | 0.8253 | 0.8275 | 0.7977 | 0.8123 |

Confusion Matrix：

|  | Pred 0 | Pred 1 |
|---|---|---|
| **True 0** | 999 | 176 |
| **True 1** | 214 | 844 |

調整 Threshold 至 0.3 後重新計算：

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.3 | 0.8218 | 0.7550 | 0.9234 | 0.8308 |

|  | Pred 0 | Pred 1 |
|---|---|---|
| **True 0** | 858 | 317 |
| **True 1** | 81 | 977 |

可以看到 threshold 從 0.5 降到 0.3 之後，模型更容易把樣本判成正類，Recall 從 0.7977 上升到 0.9234，但 Precision 從 0.8275 下降到 0.7550，是典型的 precision–recall trade-off；F1 反而因此小幅提升（0.8123 → 0.8308），代表對這份資料而言，0.3 附近的門檻讓兩者的平衡更好。完整掃描 0.1～0.9 的結果如下：

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.1 | 0.6413 | 0.5695 | 0.9953 | 0.7245 |
| 0.2 | 0.7761 | 0.6862 | 0.9716 | 0.8044 |
| 0.3 | 0.8218 | 0.7550 | 0.9234 | 0.8308 |
| 0.4 | 0.8334 | 0.8014 | 0.8620 | 0.8306 |
| 0.5 | 0.8253 | 0.8275 | 0.7977 | 0.8123 |
| 0.6 | 0.8065 | 0.8486 | 0.7202 | 0.7791 |
| 0.7 | 0.7824 | 0.8734 | 0.6323 | 0.7336 |
| 0.8 | 0.7425 | 0.9005 | 0.5132 | 0.6538 |
| 0.9 | 0.6771 | 0.9181 | 0.3497 | 0.5065 |

### 進階部分：Random Forest 與模型比較

使用 Random Forest 建立第二個模型，並用同樣的方式掃描 threshold 找出最佳 F1 的門檻（Threshold = 0.4）：

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.4 | 0.8607 | 0.8005 | 0.9405 | 0.8648 |

|  | Pred 0 | Pred 1 |
|---|---|---|
| **True 0** | 927 | 248 |
| **True 1** | 63 | 995 |

**兩模型在各自最佳 Threshold 下的比較：**

| Model | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.3 | 0.8218 | 0.7550 | 0.9234 | 0.8308 |
| Random Forest | 0.4 | 0.8607 | 0.8005 | 0.9405 | 0.8648 |

**Precision / Recall 差異與錯誤類型討論：**

Random Forest 在 Precision 和 Recall 上都優於 Logistic Regression（Precision 0.8005 vs 0.7550；Recall 0.9405 vs 0.9234），顯示 Random Forest 對這份資料中的非線性關係掌握得更好，整體分類能力較強。

從 Confusion Matrix 來看，兩個模型在各自選定的門檻下，都是 **False Positive（把本來不會申辦的客戶誤判成會申辦）明顯多於 False Negative（把本來會申辦的客戶誤判成不會）**：Logistic Regression 的 FP=317 遠大於 FN=81；Random Forest 的 FP=248 也遠大於 FN=63。這是因為兩者的門檻都刻意調低（0.3、0.4）以拉高 Recall、盡量不漏掉真正會申辦定存的客戶，代價就是多了一些「白跑一趟」的行銷誤判。但即使如此，Random Forest 在絕對數量上的兩種錯誤（FP=248、FN=63）都比 Logistic Regression（FP=317、FN=81）少，說明它不是單純靠更極端的門檻取得優勢，而是模型本身的區分能力較佳。

---

## Quiz 2：深度學習信用卡違約預測

### 資料集

使用 Kaggle 的 **Default of Credit Card Clients Dataset**（`UCI_Credit_Card.csv`），共 30,000 筆、25 個欄位（`ID`、23 個特徵、目標欄位 `default.payment.next.month`）。正類（下期違約）比例為 0.2212，屬於中度不平衡資料。丟棄 `ID` 後，以 `train_test_split(test_size=0.2, stratify=y)` 切分，Training 24,000 筆、Validation 6,000 筆，特徵以 `StandardScaler` 做 Feature Scaling。

### 模型架構與超參數

| 超參數 | 設定 | 說明 |
|---|---|---|
| Hidden Layers / Neurons | 23 → 64 → 32 → 1 | 兩層隱藏層，輸入 23 維對應特徵數，容量足以捕捉特徵間關係但不至於過大 |
| Activation | ReLU | 隱藏層之間的非線性 |
| Loss Function | `BCEWithLogitsLoss` | 內建 sigmoid，數值穩定，適合二元分類 |
| Optimizer | Adam（Improved 版本改用 AdamW） | Adam 能自動調整每個參數的學習率，收斂較快；AdamW 將 weight decay 與梯度更新解耦，正則化效果更穩定 |
| Learning Rate | 1e-3 | Adam 常用預設值 |
| Batch Size | 64 | 兼顧訓練速度與梯度穩定性 |
| Epochs | ≥ 50（實際訓練至 90+ epoch 以觀察完整的 overfitting 走勢） | 滿足作業「至少 50 epochs」要求 |

### 基礎部分：Baseline MLP

Training / Validation Loss（節錄）：

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 0 | 0.5001 | 0.4578 |
| 10 | 0.4226 | 0.4365 |
| 20 | 0.4146 | 0.4388 |
| 30 | 0.4089 | 0.4437 |
| 40 | 0.4029 | 0.4481 |
| 50 | 0.3974 | 0.4508 |
| 60 | 0.3924 | 0.4577 |
| 70 | 0.3888 | 0.4633 |
| 80 | 0.3835 | 0.4674 |
| 90 | 0.3805 | 0.4694 |

完整曲線見 `quiz2_loss_comparison.png`。

**Validation 單筆樣本預測展示：** True label = 0，模型輸出違約機率 = 0.0810，判定為 0（正確）。

### 進階部分：改善策略與模型比較

除了作業要求的「至少一種」改善策略，本專案額外比較了三種做法，並加碼嘗試三者合併：

- **Dropout**（p=0.3，加在兩層隱藏層之後）
- **Early Stopping**（`patience=5`，還原至 val loss 最低時的權重）
- **Improved**：Dropout + Early Stopping + AdamW 三者合併

四個模型在同一份 Validation Dataset 上的比較：

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline MLP | 0.8098 | 0.6336 | 0.3323 | 0.4360 |
| Dropout | 0.8195 | 0.6778 | 0.3504 | 0.4620 |
| Early Stopping | 0.8187 | 0.6526 | 0.3851 | 0.4844 |
| Improved（Dropout+ES+AdamW） | 0.8185 | 0.6630 | 0.3647 | 0.4706 |

### 觀察與討論

**1. 是否改善 Overfitting？**
Baseline MLP 呈現典型的 overfitting 走勢：Train Loss 從 0.5001 一路單調下降到 0.3805，但 Val Loss 在 epoch 10 附近觸底（0.4365）後就持續回升到 0.4694，Train/Val 的差距從 epoch 0 的 0.0139 擴大到 epoch 90 的 0.0889，代表模型在後段訓練中逐漸記憶 training set、喪失泛化能力。加入 Dropout 之後，Val Loss 全程維持在 0.431～0.435 之間、幾乎不再上升，Overfitting 明顯被抑制。

**2. Validation Loss 是否下降？**
三種改善策略的最終／最佳 Val Loss 都低於 Baseline 在相同訓練階段的數值：Dropout 在 epoch 90 時 Val Loss 為 0.4348（Baseline 同期為 0.4694）；Early Stopping 與 Improved 都在 Val Loss 觸底（0.4341）時停止，同樣遠低於 Baseline 持續訓練後的數值。

**3. 模型是否變得更穩定？**
Dropout 讓 Val Loss 曲線全程平坦，是最穩定的版本；Early Stopping 則是用「及早喊停」的方式避免進入不穩定的後段訓練，僅需 13 個 epoch 即可達到接近最佳的 Val Loss。值得一提的是，**Improved（三者合併）並沒有比 Early Stopping 單獨使用更好**：兩者的 `best_val_loss` 一樣是 0.4341，但 Improved 的 F1（0.4706）反而略低於 Early Stopping 單獨使用的 F1（0.4844）。這說明在這份資料與這組超參數下，Early Stopping 本身已經攔住了大部分的 overfitting，疊加 Dropout 後在提前停止的機制下訓練步數更少（21 epoch vs 13 epoch，但學習率調整方式不同），模型可能還沒有機會學到足夠的區分能力就被停止，並非「疊加的正則化策略越多越好」，需要針對個別任務調整。

---

## Quiz 3：模型表現評估

### ROC Curve 與 AUC 的意義

ROC（Receiver Operating Characteristic）曲線是把分類模型在**所有可能的 threshold** 下的表現畫成一條曲線：X 軸是 False Positive Rate（FPR = FP / (FP + TN)），Y 軸是 True Positive Rate（TPR = Recall = TP / (TP + FN)）。曲線越靠左上角，代表模型在各種門檻下都能同時維持高 TPR、低 FPR，也就是對正負樣本的區分能力越強。

AUC（Area Under Curve）是 ROC 曲線下方的面積，數值介於 0.5（等同隨機猜測）到 1（完美區分）之間。AUC 的優點是**不依賴任何特定的 threshold 選擇**，可以視為模型「跨所有可能門檻的整體排序能力」的單一指標，適合用來做模型間的整體比較。

### 基礎部分：MLP 的 ROC / AUC

沿用 Quiz 2 相同的 Training / Validation Dataset 與 Baseline MLP，取得 Validation 上的預測機率後繪製 ROC 曲線（見 `quiz3_roc_comparison.png`），並計算：

**MLP AUC = 0.7468**

### 進階部分：與 Logistic Regression 比較

使用相同的 Training / Validation Dataset 建立 Logistic Regression 作為第二個模型，取得其預測機率並將兩條 ROC 曲線畫在同一張圖上，分別標示 AUC：

| Model | AUC |
|---|---|
| MLP | 0.7468 |
| Logistic Regression | 0.7076 |

**比較與結論：** MLP 的 AUC（0.7468）高於 Logistic Regression（0.7076），代表 MLP 在整條 ROC 曲線上大致位於 Logistic Regression 的左上方，對正負樣本（是否違約）的區分能力較好。這與 Quiz 1 觀察到的現象一致——在 Bank Marketing 資料集中，非線性的 Random Forest 也優於線性的 Logistic Regression——顯示這兩份資料中的目標變數都與特徵之間存在非線性關係，能夠建模非線性關係的演算法（Random Forest、MLP）普遍比純線性模型（Logistic Regression）表現更好。

---

## 總結

三份作業共通的觀察是：**能捕捉特徵間非線性關係的模型（Random Forest、MLP）在 Precision、Recall、F1、AUC 等指標上，普遍優於線性的 Logistic Regression**；而在 Quiz 2 的深度學習模型訓練過程中，**正則化策略需要針對任務調整、並非疊加越多越好**，這點透過 Early Stopping 單獨使用優於三種策略合併使用的結果得到印證。