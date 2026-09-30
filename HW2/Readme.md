# Machine Learning & Deep Learning Quizzes

本專案包含三個 Quiz，依序涵蓋：

1. **Quiz 1：機器學習分類任務**
2. **Quiz 2：深度學習信用卡違約預測**
3. **Quiz 3：模型表現評估**

使用的資料集主要為 **Default of Credit Card Clients Dataset**，並透過 Scikit-Learn 與 PyTorch 實作 Machine Learning 與 Deep Learning Classification Models。

---

# Quiz 1：機器學習分類任務

## 1.1 Overview

Quiz 1 的目標是完成一個 Binary Classification Task，使用 Scikit-Learn 建立 Machine Learning Classification Model。

本次使用兩種 Machine Learning Classification Algorithms：

* Logistic Regression
* Random Forest

並進行：

* Training / Validation Split
* Feature Scaling
* Prediction Probability
* Classification Threshold 調整
* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-Score

---

## 1.2 Dataset

Quiz 1 使用一份二元分類資料集，Target Variable 為：

```text
deposit
```

Validation Dataset：

```text
Validation size: 2233
```

Actual class distribution：

```text
Class 0: 1175
Class 1: 1058
```

---

# 1.3 Model 1：Logistic Regression

使用 Logistic Regression 建立第一個 Machine Learning Classification Model。

## Threshold = 0.5

Confusion Matrix：

```text
[[999, 176],
 [214, 844]]
```

Evaluation：

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8253 |
| Precision | 0.8275 |
| Recall    | 0.7977 |
| F1-Score  | 0.8123 |

---

## Threshold = 0.3

將 Classification Threshold 從 0.5 調整為 0.3：

Confusion Matrix：

```text
[[858, 317],
 [81, 977]]
```

Evaluation：

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8218 |
| Precision | 0.7550 |
| Recall    | 0.9234 |
| F1-Score  | 0.8308 |

降低 Threshold 後，模型會更容易將樣本判斷為 Positive，因此 Recall 明顯提高，但 Precision 下降。

---

## Logistic Regression Threshold Results

| Threshold | Accuracy | Precision | Recall |     F1 |
| --------: | -------: | --------: | -----: | -----: |
|       0.1 |   0.6413 |    0.5695 | 0.9953 | 0.7245 |
|       0.2 |   0.7761 |    0.6862 | 0.9716 | 0.8044 |
|       0.3 |   0.8218 |    0.7550 | 0.9234 | 0.8308 |
|       0.4 |   0.8334 |    0.8014 | 0.8620 | 0.8306 |
|       0.5 |   0.8253 |    0.8275 | 0.7977 | 0.8123 |
|       0.6 |   0.8065 |    0.8486 | 0.7202 | 0.7791 |
|       0.7 |   0.7824 |    0.8734 | 0.6323 | 0.7336 |
|       0.8 |   0.7425 |    0.9005 | 0.5132 | 0.6538 |
|       0.9 |   0.6771 |    0.9181 | 0.3497 | 0.5065 |

本次選擇 Threshold = 0.3 作為 Logistic Regression 的主要結果。

---

# 1.4 Model 2：Random Forest

第二個 Machine Learning Model 使用 Random Forest。

Validation Dataset 同樣為 2233 筆資料。

Actual class distribution：

```text
Class 0: 1175
Class 1: 1058
```

---

## Threshold = 0.5

Confusion Matrix：

```text
[[979, 196],
 [116, 942]]
```

Evaluation：

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8603 |
| Precision | 0.8278 |
| Recall    | 0.8904 |
| F1-Score  | 0.8579 |

---

## Random Forest Threshold Results

| Threshold | Accuracy | Precision | Recall |     F1 |
| --------: | -------: | --------: | -----: | -----: |
|       0.1 |   0.7062 |    0.6178 | 0.9962 | 0.7627 |
|       0.2 |   0.8065 |    0.7150 | 0.9839 | 0.8282 |
|       0.3 |   0.8383 |    0.7587 | 0.9660 | 0.8499 |
|       0.4 |   0.8607 |    0.8005 | 0.9405 | 0.8648 |
|       0.5 |   0.8603 |    0.8278 | 0.8904 | 0.8579 |
|       0.6 |   0.8415 |    0.8431 | 0.8176 | 0.8301 |
|       0.7 |   0.7976 |    0.8677 | 0.6758 | 0.7598 |
|       0.8 |   0.7062 |    0.8972 | 0.4291 | 0.5806 |
|       0.9 |   0.5996 |    0.9457 | 0.1645 | 0.2802 |

本次選擇 Threshold = 0.4 作為 Random Forest 的主要結果。

---

# 1.5 Quiz 1 Model Comparison

使用各模型調整後的 Threshold：

| Model               | Threshold | Accuracy | Precision | Recall |     F1 |
| ------------------- | --------: | -------: | --------: | -----: | -----: |
| Logistic Regression |       0.3 |   0.8218 |    0.7550 | 0.9234 | 0.8308 |
| Random Forest       |       0.4 |   0.8607 |    0.8005 | 0.9405 | 0.8648 |

從結果可以觀察到，兩個模型在 Precision 與 Recall 之間存在 trade-off。

Logistic Regression 在 Threshold = 0.3 時提高 Recall，使模型能辨識較多 Positive Samples，但同時產生較多 False Positive。

Random Forest 在 Threshold = 0.4 時，其 Precision 與 Recall 均維持在較高的水準。

---

# Quiz 2：深度學習信用卡違約預測

## 2.1 Overview

Quiz 2 使用指定的：

**Default of Credit Card Clients Dataset**

並使用 PyTorch 建立 Multi-Layer Perceptron（MLP）進行信用卡違約預測。

實驗包含：

* Feature Scaling
* MLP
* Training / Validation Loss
* Validation Sample Prediction
* Dropout
* Early Stopping
* AdamW
* Accuracy
* Precision
* Recall
* F1-Score

---

# 2.2 Dataset

資料集共有：

```text
Data shape: (30000, 25)
```

其中：

* Total samples：30,000
* Original columns：25
* ID：移除
* Features：23
* Target：`default.payment.next.month`

Positive rate：

```text
0.2212
```

也就是約：

```text
22.12%
```

Positive Class：

```text
1 = Default
```

Negative Class：

```text
0 = No Default
```

---

# 2.3 Train / Validation Split

使用：

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

因此：

| Dataset            | Samples |
| ------------------ | ------: |
| Training Dataset   |  24,000 |
| Validation Dataset |   6,000 |

使用 `stratify=y` 維持 Training 與 Validation Dataset 的類別比例。

---

# 2.4 Feature Scaling

使用 `StandardScaler`：

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

Scaler 只在 Training Dataset 上進行 `fit`，再使用相同的 scaler 對 Validation Dataset 進行 transformation。

---

# 2.5 Baseline MLP

Baseline MLP 架構：

```text
Input: 23 features
       │
       ▼
Linear(23 → 64)
       │
     ReLU
       │
       ▼
Linear(64 → 32)
       │
     ReLU
       │
       ▼
Linear(32 → 1)
```

Model：

```python
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
```

---

# 2.6 MLP Hyperparameters

| Hyperparameter |             Value |
| -------------- | ----------------: |
| Epochs         |               100 |
| Batch Size     |                64 |
| Learning Rate  |             0.001 |
| Optimizer      |              Adam |
| Loss Function  | BCEWithLogitsLoss |
| Hidden Layers  |                 2 |
| Hidden Neurons |            64, 32 |
| Random Seed    |                42 |

---

# 2.7 Baseline MLP Training Result

Training：

```text
Epoch 0:
Train Loss = 0.5001
Validation Loss = 0.4578

Epoch 10:
Train Loss = 0.4226
Validation Loss = 0.4365

Epoch 20:
Train Loss = 0.4146
Validation Loss = 0.4388

Epoch 30:
Train Loss = 0.4089
Validation Loss = 0.4437

Epoch 40:
Train Loss = 0.4029
Validation Loss = 0.4481

Epoch 50:
Train Loss = 0.3974
Validation Loss = 0.4508

Epoch 60:
Train Loss = 0.3924
Validation Loss = 0.4577

Epoch 70:
Train Loss = 0.3888
Validation Loss = 0.4633

Epoch 80:
Train Loss = 0.3835
Validation Loss = 0.4674

Epoch 90:
Train Loss = 0.3805
Validation Loss = 0.4694
```

可以觀察到：

* Training Loss 持續下降
* Validation Loss 在約 Epoch 10 後開始上升

這表示模型後期開始出現 **Overfitting**。

---

# 2.8 Validation Sample Prediction

實際使用 Validation Dataset 中的一筆資料進行預測。

結果：

```text
True label: 0

Predicted probability of default:
0.0810

Predicted label:
0
```

模型預測此 Sample 發生信用卡違約的 Probability 為：

```text
8.10%
```

因此在 Threshold = 0.5 時，預測結果為：

```text
No Default (0)
```

---

# 2.9 Model Improvement

為改善 Baseline MLP，在 Quiz 2 中測試：

1. Dropout
2. Early Stopping
3. Improved MLP

Improved MLP 使用：

```text
Dropout
+
Early Stopping
+
AdamW
```

其中 AdamW 本次沒有額外設定 `weight_decay`。

---

# 2.10 Dropout

Dropout 使用：

```text
p = 0.3
```

Model architecture：

```text
Linear(23 → 64)
      ↓
    ReLU
      ↓
Dropout(0.3)
      ↓
Linear(64 → 32)
      ↓
    ReLU
      ↓
Dropout(0.3)
      ↓
Linear(32 → 1)
```

Training result：

```text
Epoch 0:
Train Loss = 0.4890
Validation Loss = 0.4547

Epoch 10:
Train Loss = 0.4359
Validation Loss = 0.4356

Epoch 20:
Train Loss = 0.4326
Validation Loss = 0.4348

Epoch 30:
Train Loss = 0.4269
Validation Loss = 0.4313

Epoch 40:
Train Loss = 0.4267
Validation Loss = 0.4322

Epoch 50:
Train Loss = 0.4247
Validation Loss = 0.4333

Epoch 60:
Train Loss = 0.4234
Validation Loss = 0.4335

Epoch 70:
Train Loss = 0.4236
Validation Loss = 0.4321

Epoch 80:
Train Loss = 0.4222
Validation Loss = 0.4354

Epoch 90:
Train Loss = 0.4216
Validation Loss = 0.4348
```

相較於 Baseline，Validation Loss 沒有持續上升，顯示 Dropout 對降低 Overfitting 有一定幫助。

---

# 2.11 Early Stopping

Early Stopping 使用：

```text
Patience = 5
```

Training result：

```text
Epoch 0:
Train Loss = 0.4722
Validation Loss = 0.4524

Epoch 10:
Train Loss = 0.4235
Validation Loss = 0.4397

Early stopping at Epoch 13

Best Validation Loss = 0.4341
```

Early Stopping 會在 Validation Loss 長時間沒有改善時停止 Training，並恢復 Validation Loss 最低時的模型權重。

---

# 2.12 Improved MLP

Improved MLP：

```text
Dropout + Early Stopping + AdamW
```

Training result：

```text
Epoch 0:
Train Loss = 0.4984
Validation Loss = 0.4578

Epoch 10:
Train Loss = 0.4361
Validation Loss = 0.4385

Epoch 20:
Train Loss = 0.4323
Validation Loss = 0.4348

Early stopping at Epoch 21

Best Validation Loss = 0.4341
```

---

# 2.13 Quiz 2 Model Comparison

在 Validation Dataset 上使用 Threshold = 0.5：

| Model          | Accuracy | Precision | Recall |     F1 |
| -------------- | -------: | --------: | -----: | -----: |
| Baseline MLP   |   0.8098 |    0.6336 | 0.3323 | 0.4360 |
| Dropout        |   0.8195 |    0.6778 | 0.3504 | 0.4620 |
| Early Stopping |   0.8187 |    0.6526 | 0.3851 | 0.4844 |
| Improved       |   0.8185 |    0.6630 | 0.3647 | 0.4706 |

---

# 2.14 Quiz 2 Observation

Baseline MLP 的 Training Loss 持續下降，但是 Validation Loss 在後期開始上升，顯示模型有 Overfitting 現象。

加入 Dropout 後：

* Validation Loss 整體較穩定
* Accuracy 提升
* Precision 提升
* Recall 提升
* F1-Score 提升

Early Stopping 則可以在 Validation Loss 不再改善時停止 Training。

本次結果中：

```text
Baseline F1 = 0.4360
Early Stopping F1 = 0.4844
```

Early Stopping 的 F1-Score 高於 Baseline。

Improved MLP 雖然加入 Dropout、Early Stopping 與 AdamW，但本次實驗的 F1-Score：

```text
Improved F1 = 0.4706
```

略低於單獨使用 Early Stopping 的結果。

因此，本次實驗顯示不同模型改善策略的效果會受到資料集、模型架構與 Hyperparameters 影響，並非加入越多策略就一定能得到更好的 Validation Performance。

---

# Quiz 3：模型表現評估

## 3.1 Overview

Quiz 3 延續 Quiz 2 的信用卡違約預測資料集與 MLP。

本次主要評估：

* Prediction Probability
* ROC Curve
* AUC

另外加入 Machine Learning Classification Model：

* Logistic Regression

比較 MLP 與 Logistic Regression 的正負樣本區分能力。

---

# 3.2 MLP Prediction Probability

MLP 對 Validation Dataset 輸出 Logit，再使用 Sigmoid 轉換成 Probability：

```python
with torch.no_grad():

    mlp_logits = model(X_val_t)

    mlp_probs = torch.sigmoid(
        mlp_logits
    ).numpy().flatten()
```

`mlp_probs` 為每筆 Validation Sample 發生信用卡違約的 Prediction Probability。

---

# 3.3 ROC Curve

ROC Curve（Receiver Operating Characteristic Curve）用來觀察模型在不同 Classification Threshold 下的分類能力。

ROC Curve：

* X-axis：False Positive Rate (FPR)
* Y-axis：True Positive Rate (TPR)

公式：

```text
TPR = TP / (TP + FN)

FPR = FP / (FP + TN)
```

透過改變 Classification Threshold，可以得到不同的 FPR 與 TPR 組合，形成 ROC Curve。

---

# 3.4 AUC

AUC（Area Under the ROC Curve）代表 ROC Curve 下方的面積。

AUC 可以用來衡量模型區分 Positive Samples 與 Negative Samples 的能力。

一般而言：

```text
AUC ≈ 1
→ 具有很好的正負樣本區分能力

AUC ≈ 0.5
→ 接近隨機分類
```

因此 AUC 越接近 1，通常代表模型具有較好的整體正負樣本區分能力。

---

# 3.5 MLP AUC

MLP 在 Quiz 3 Validation Dataset 上：

```text
MLP AUC = 0.7468
```

實際結果：

```text
0.746762
```

---

# 3.6 Second Model：Logistic Regression

Quiz 3 進階部分使用課程中介紹過的 Machine Learning Classification Algorithm：

**Logistic Regression**

使用與 Quiz 2 完全相同的：

* Training Dataset
* Validation Dataset
* Feature Scaling

建立 Logistic Regression：

```python
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(
    X_train_scaled,
    y_train
)
```

Validation Prediction Probability：

```python
lr_probs = lr_model.predict_proba(
    X_val_scaled
)[:, 1]
```

---

# 3.7 Logistic Regression AUC

Logistic Regression 在相同 Validation Dataset 上：

```text
Logistic Regression AUC = 0.7076
```

實際結果：

```text
0.707636
```

---

# 3.8 ROC / AUC Comparison

| Model               |        AUC |
| ------------------- | ---------: |
| MLP                 | **0.7468** |
| Logistic Regression | **0.7076** |

ROC Curve：

```text
quiz3_roc_comparison.png
```

圖中包含：

* MLP ROC Curve
* Logistic Regression ROC Curve
* Random Classifier R
