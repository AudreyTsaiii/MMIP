# HW4 — Deep Learning: Sentiment Analysis, Image Classification & Image Captioning

本專案完成四個 Quiz，涵蓋資料集準備、文字情緒分類、影像分類，以及影像描述生成。使用 PyTorch、Torchvision、Hugging Face Datasets 與 Transformers 建立實驗流程，並透過分類指標、BLEU-4、Gemini 評分及生成時間比較模型表現。

## 1. 專案架構（Project Structure）

```text
HW4/
├── Quiz1.py
├── Quiz2.py
├── Quiz3.py
├── Quiz4.py
├── Readme.md
│
├── data_quiz1/
│   ├── imdb/
│   │   ├── split_summary.csv
│   │   ├── label_distribution.csv
│   │   ├── text_examples.csv
│   │   └── train_validation_indices.csv
│   │
│   ├── stl10/
│   │   ├── split_indices.json
│   │   ├── split_summary.csv
│   │   ├── label_distribution.csv
│   │   ├── sample_images.csv
│   │   ├── sample_grid.png
│   │   └── sample_images/
│   │
│   ├── flickr8k/
│       ├── split_summary.csv
│       ├── captions.csv
│       ├── split_info.json
│       ├── caption_examples.csv
│       ├── sample_images.csv
│       ├── sample_grid.png
│       └── sample_images/
│
├── results_quiz2/
│   ├── vocab.json
│   ├── rnn_training_history.csv
│   ├── lstm_training_history.csv
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   └── loss_comparison.png
│
├── results_quiz3/
│   ├── split_indices.json
│   ├── vit_training_history.csv
│   ├── resnet18_training_history.csv
│   ├── model_comparison.csv
│   └── test_predictions.csv
│
└── results_quiz4/
    ├── training_history.csv
    ├── caption_predictions.csv
    └── method_comparison.csv
```

**目錄說明:**

- `Quiz1.py`:下載與準備 IMDb、STL-10、Flickr8k 三種資料集，產生資料分割資訊、類別分布、範例圖片及文字描述。
- `Quiz2.py`:建立 RNN 與 LSTM 情緒分類模型，完成訓練、驗證、測試及結果比較。
- `Quiz3.py`:微調 ViT 與 ResNet18，完成 STL-10 影像分類，並比較 Accuracy 與 Macro-AUC。
- `Quiz4.py`:微調 BLIP 影像描述模型，比較 Greedy Search 與 Beam Search，使用 BLEU-4 及 Gemini 評估結果。
- `data_quiz1/`:存放 Quiz 1 產生的資料集統計、分割資訊與範例。
- `results_quiz2/`、`results_quiz3/`、`results_quiz4/`:分別存放模型權重、訓練紀錄、測試預測及評估結果。

以上為主要檔案架構；實際產生的檔案會依照執行流程與是否已完成訓練而有所不同。BLIP 模型權重、下載的原始資料集及其他大型檔案不一定會包含在 Git repository 中。

---

## 2. 環境需求（Requirements）

### 使用套件

| 套件 | 用途 |
|---|---|
| Python | 執行環境 |
| PyTorch | 神經網路建立與訓練 |
| Torchvision | STL-10 資料集、影像轉換、ViT 與 ResNet18 |
| Hugging Face Datasets | 下載及讀取 IMDb、Flickr8k |
| Transformers | 載入 BLIP 影像描述模型 |
| Scikit-learn | 資料切分及模型評估 |
| NumPy、Pandas | 資料處理與結果輸出 |
| Matplotlib | 繪製訓練曲線及資料集範例 |
| Pillow | 影像讀取與處理 |
| NLTK | BLEU-4 計算 |
| Google Gen AI SDK | 使用 Gemini 評估影像描述 |

### 建立虛擬環境

Windows PowerShell:

```powershell
cd HW4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

安裝 PyTorch 時，請依照電腦是否有 NVIDIA GPU，選擇合適的 CUDA 或 CPU 版本。接著安裝其他套件:

```powershell
pip install torchvision datasets transformers scikit-learn numpy pandas matplotlib pillow nltk google-genai
```

請確認 PyTorch 與 Torchvision 版本相容，並依實際硬體環境安裝對應的版本。

---

## 3. Quiz 1 — Dataset Preparation

### 3.1 作業目標

依照作業要求，準備三種深度學習任務所需的資料集:

1. Sentiment Analysis:文字情緒分類。
2. Image Classification:影像分類。
3. Image Captioning:影像與文字描述配對。

### 3.2 使用資料集

| 資料集 | 任務 | 資料內容 |
|---|---|---|
| IMDb | Sentiment Analysis | 電影評論及二元情緒標籤 |
| STL-10 | Image Classification | 10 類 RGB 影像 |
| Flickr8k | Image Captioning | 影像及對應的文字描述 |

#### IMDb

使用 Hugging Face 上的 `stanfordnlp/imdb`。

- 原始訓練集:25,000 筆。
- 驗證集:由原始訓練集切分 20%，共 5,000 筆。
- 實際訓練集:20,000 筆。
- 測試集:25,000 筆。
- 類別:Negative、Positive。

採用固定 random seed 與分層切分（stratified split），保留獨立測試集，避免使用測試資料進行訓練。

產生的主要檔案:

- `split_summary.csv`:各資料集分割的樣本數。
- `label_distribution.csv`:各分割的類別分布。
- `text_examples.csv`:各分割的評論範例。
- `train_validation_indices.csv`:訓練集與驗證集的索引紀錄。

#### STL-10

使用 Torchvision 的 `STL10`，包含 10 個影像類別:

`airplane`、`bird`、`car`、`cat`、`deer`、`dog`、`horse`、`monkey`、`ship`、`truck`。

- 原始訓練集:5,000 張。
- 訓練集:4,500 張。
- 驗證集:500 張。
- 測試集:8,000 張。

訓練集依照固定 seed 隨機切分 90%／10%，測試集使用資料集提供的官方測試集。

主要輸出:

- `split_indices.json`:訓練、驗證與測試索引。
- `split_summary.csv`:各分割的影像數量。
- `label_distribution.csv`:各類別的樣本分布。
- `sample_images.csv`:範例圖片的索引、標籤及路徑。
- `sample_grid.png`:20 張訓練影像的排列展示。
- `sample_images/`:個別範例圖片。

#### Flickr8k

使用 Hugging Face 上的 `jxie/flickr8k`，準備影像描述生成任務所需的影像與文字描述配對。

程式會自動辨識資料集中的影像、caption 及影像識別欄位，並處理可解碼的影像。

若資料集提供 `train`、`validation`、`test` 三個官方分割，則保留官方分割；否則依照唯一影像 ID 隨機切分為 80% 訓練集、10% 驗證集及 10% 測試集。

切分時以影像 ID 為單位，而非直接隨機切分 caption rows，以降低同一張影像同時出現在不同資料集的風險。

主要輸出:

- `split_summary.csv`:各分割的影像描述配對數與唯一影像數。
- `captions.csv`:影像 ID、文字描述及來源分割。
- `split_info.json`:分割方式與 random seed。
- `caption_examples.csv`:各分割的文字描述範例。
- `sample_images.csv`:範例影像及 caption。
- `sample_grid.png`:影像與文字描述的排列展示。
- `sample_images/`:個別範例圖片。


### 3.3 執行方式

```powershell
python Quiz1.py
```

執行後會依序準備 IMDb、STL-10 與 Flickr8k，並將統計資料及範例輸出至 `data_quiz1/`。

---

## 4. Quiz 2 — Sentiment Analysis

### 4.1 作業目標

使用 IMDb 電影評論完成二元情緒分類，建立 RNN 與 LSTM 模型，並透過測試集的分類指標比較兩者表現。

### 4.2 資料前處理

1. 將文字轉換成小寫。
2. 使用正規表示式進行 tokenization。
3. 根據訓練集建立詞彙表，最大詞彙量為 30,000。
4. 移除出現次數少於 2 次的詞彙。
5. 使用 `<PAD>` 處理序列長度差異，使用 `<UNK>` 表示未知詞彙。
6. 每筆評論最多保留 256 個 token。
7. 使用 `pad_sequence` 將同一 batch 的序列補齊。
8. 使用 `pack_padded_sequence` 忽略補齊部分，減少 padding 對循環神經網路的影響。

詞彙表只使用訓練資料建立，避免驗證集與測試集的文字資訊洩漏至訓練流程。

### 4.3 模型架構

兩個模型共用 Embedding 與全連接分類層，但**循環層結構與 Dropout 比例並不相同**（RNN 為 0.3、LSTM 為 0.5），比較結果時需一併考慮（見 4.5 結果分析）。

**RNN**

```text
Input Tokens
     ↓
Embedding (128)
     ↓
RNN (Hidden Size = 128)
     ↓
Last Hidden State
     ↓
Dropout (0.3)
     ↓
Fully Connected (128 → 2)
     ↓
Negative / Positive
```

**LSTM**

```text
Input Tokens
     ↓
Embedding (128)
     ↓
LSTM (Hidden Size = 128)
     ↓
Last Hidden State
     ↓
Dropout (0.5)
     ↓
Fully Connected (128 → 2)
     ↓
Negative / Positive
```

RNN 使用循環神經網路學習序列資訊；LSTM 則透過輸入閘、遺忘閘及輸出閘控制資訊流動，改善長序列學習時的記憶與梯度傳遞問題。

### 4.4 訓練設定

| 參數 | 設定 |
|---|---:|
| Random seed | 42 |
| Batch size | 64 |
| Maximum epochs | 15 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | Cross Entropy Loss |
| Embedding dimension | 128 |
| Hidden dimension | 128 |
| Maximum sequence length | 256 |
| Early stopping patience | 3 |
| Dropout | RNN 0.3／LSTM 0.5 |

模型以 Validation Loss 選擇最佳權重，若連續 3 個 epoch 未改善則提前停止訓練。

### 4.5 實驗結果

#### 訓練過程

| 項目 | RNN | LSTM |
|---|---:|---:|
| 實際訓練 epoch 數 | 10（Early Stopping） | 9（Early Stopping） |
| 最佳 epoch（Validation Loss 最低） | 7 | 6 |
| 最佳 Validation Loss | 0.6018 | 0.3714 |
| 最佳 epoch 的 Validation Accuracy | 72.70% | 84.90% |
| 最佳 epoch 的 Validation F1 | 72.23% | 84.84% |

#### 測試集結果

| Metric | RNN | LSTM |
|---|---:|---:|
| Loss | 0.6149 | 0.3978 |
| Accuracy | 71.98% | 83.72% |
| Precision | 72.88% | 84.44% |
| Recall | 70.03% | 82.67% |
| F1-score | 71.43% | 83.55% |

**結果分析:**

- LSTM 的 Accuracy 比 RNN 高出約 11.74 個百分點，Loss 較低，Precision、Recall 與 F1-score 也都較高。
- RNN 的訓練過程不穩定:Validation Accuracy 在 50%～73% 間波動，且第 7 epoch 後 Training Loss 由 0.4530 回升至 0.5828，這是單純 RNN 在長序列上常見的梯度問題。
- LSTM 在第 6 epoch 達到最低 Validation Loss 後，Training Loss 持續下降（至 0.1074），但 Validation Loss 回升，顯示之後開始過擬合；因此以 Validation Loss 選出第 6 epoch 的權重。（第 7 epoch 的 Validation Accuracy 為 85.66%，略高於第 6 epoch，但本實驗以 Validation Loss 作為選模依據。）
- **比較的限制:** 兩個模型的 Dropout 不同（0.3 與 0.5），因此 11.74 個百分點的差距並非完全來自 RNN 與 LSTM 的結構差異；此外結果僅為單一 random seed。結論應解讀為「在本次設定下 LSTM 明顯優於 RNN」，而非嚴格的結構消融實驗。

### 4.6 輸出檔案

| 檔案 | 說明 |
|---|---|
| `vocab.json` | 訓練時建立的詞彙表 |
| `rnn_training_history.csv` | RNN 每個 epoch 的 Loss、Accuracy 與 F1 |
| `lstm_training_history.csv` | LSTM 每個 epoch 的 Loss、Accuracy 與 F1 |
| `model_comparison.csv` | RNN 與 LSTM 的測試集比較 |
| `test_predictions.csv` | 每筆測試評論的真實標籤、預測標籤與類別機率（測試集推論結果） |
| `loss_comparison.png` | RNN 與 LSTM 的 Training Loss、Validation Loss 曲線 |

### 4.7 執行方式

```powershell
python Quiz2.py
```

---

## 5. Quiz 3 — Vision Transformer

### 5.1 作業目標

使用 STL-10 進行影像分類，微調 Vision Transformer（ViT）與 ResNet18，並比較兩者的分類準確率及 Macro-AUC。

### 5.2 資料前處理

訓練集與驗證集來自 STL-10 官方訓練集（固定 seed 42，依類別分層切分 90%／10%）:

- 訓練集:4,500 張。
- 驗證集:500 張。
- 測試集:8,000 張。

為符合預訓練模型的輸入規格，使用以下影像轉換。

**Training Transform**

1. `RandomResizedCrop(224)`:隨機裁切並調整至 224 × 224。
2. `RandomHorizontalFlip()`:隨機水平翻轉。
3. `ToTensor()`:轉換成 Tensor。
4. `Normalize()`:依照 ImageNet 的 mean 與 standard deviation 正規化。

**Validation / Test Transform**

1. `Resize(256)`。
2. `CenterCrop(224)`。
3. `ToTensor()`。
4. `Normalize()`。

訓練時使用隨機影像增強；驗證及測試時使用固定的影像轉換，以保持評估的一致性。

### 5.3 模型架構

#### Vision Transformer（ViT-B/16）

使用 Torchvision 的 `vit_b_16`，載入預訓練權重 `ViT_B_16_Weights.DEFAULT`。

```text
Input Image (224 × 224)
          ↓
Patch Embedding (16 × 16 patches)
          ↓
Transformer Encoder
          ↓
Classification Head
          ↓
10 STL-10 Classes
```

將原始分類頭替換為輸出維度為 10 的全連接層，以符合 STL-10 的類別數量。

#### ResNet18

使用 Torchvision 的 `resnet18`，載入預訓練權重 `ResNet18_Weights.DEFAULT`。

```text
Input Image (224 × 224)
          ↓
Convolutional Stem
          ↓
Residual Blocks
          ↓
Global Average Pooling
          ↓
Fully Connected Layer
          ↓
10 STL-10 Classes
```

將原始全連接分類層替換成 10 類輸出。

ViT 使用 Transformer 的自注意力機制學習影像區塊之間的關係；ResNet18 則利用卷積與殘差連接提取多層次影像特徵。

> **微調範圍:** 兩個模型皆為**全部參數微調**（`model.parameters()` 全部交給 AdamW 更新），僅將最後的分類層替換成 10 類輸出（ViT 的 `heads.head`、ResNet18 的 `fc`）。沒有凍結任何層，也未使用 mixed precision。

### 5.4 訓練設定

| 參數 | 設定 |
|---|---:|
| Random seed | 42 |
| Batch size | 8 |
| Maximum epochs | 5 |
| Learning rate | 0.0001 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Loss function | Cross Entropy Loss |
| Input resolution | 224 × 224 |
| Number of classes | 10 |

每個 epoch 後評估驗證集，以 Validation Accuracy 最高的模型作為最佳模型並儲存權重。兩個模型皆完整訓練 5 個 epoch（無 Early Stopping）。

### 5.5 實驗結果

#### 訓練過程

| 項目 | ViT | ResNet18 |
|---|---:|---:|
| 最佳 epoch（Validation Accuracy 最高） | 4 | 4 |
| 最佳 Validation Accuracy | 88.80% | 92.80% |
| 最佳 epoch 的 Validation Loss | 0.3773 | 0.2007 |
| 最佳 epoch 的 Validation Macro-AUC | 0.9929 | 0.9975 |
| 5 個 epoch 總訓練時間 | 約 875 秒 | 約 130 秒 |

#### 測試集結果

| Metric | ViT | ResNet18 |
|---|---:|---:|
| Test Loss | 0.3329 | 0.2263 |
| Accuracy | 89.20% | 92.38% |
| Macro-AUC | 0.9944 | 0.9964 |

**結果分析:**

- ResNet18 的 Accuracy 比 ViT 高約 3.18 個百分點，Test Loss 也較低；兩者的 Macro-AUC 都高於 0.99，差距僅 0.002，顯示兩個模型對 STL-10 十個類別都有良好的區分能力。
- ViT 的訓練較不穩定:第 2 epoch 的 Validation Accuracy 從 88.0% 掉到 81.8%，之後才回升；ResNet18 的 Validation Accuracy 則一直維持在 90.6%～92.8%。
- ResNet18 的總訓練時間約為 ViT 的 1/7（約 130 秒對 875 秒），在小規模資料集與短訓練週期下更有效率。ViT-B/16 參數量較大，通常需要更多資料或更長的訓練才能發揮優勢，這可能是其表現較低的原因之一（此為合理推測，本實驗未做額外驗證）。
- 以上為單一 random seed 的結果，ResNet18 優於 ViT 的結論僅適用於本次設定。

Macro-AUC 採用 One-vs-Rest（OvR）方式計算各類別的 ROC-AUC，再對十個類別取平均，使每個類別具有相同權重。

### 5.6 輸出檔案

| 檔案 | 說明 |
|---|---|
| `split_indices.json` | 訓練、驗證與測試集索引 |
| `vit_training_history.csv` | ViT 每個 epoch 的訓練及驗證指標 |
| `resnet18_training_history.csv` | ResNet18 每個 epoch 的訓練及驗證指標 |
| `model_comparison.csv` | 兩個模型的測試集結果 |
| `test_predictions.csv` | 每張測試影像的真實類別、預測類別及各類別機率（測試集推論結果） |

### 5.7 執行方式

```powershell
python Quiz3.py
```

---

## 6. Quiz 4 — Image Captioning

### 6.1 作業目標

使用 Flickr8k 微調影像描述模型，讓模型根據輸入影像產生自然語言描述，並完成以下實驗:

1. 訓練及儲存影像描述模型。
2. 使用 Gemini 評估生成描述與影像內容的一致性。
3. 使用 BLEU-4 量化生成文字與參考描述的相似程度。
4. 比較 Greedy Search 與 Beam Search 的生成時間與描述品質。

### 6.2 使用資料集與前處理

資料集使用 Hugging Face 上的 `jxie/flickr8k`，每筆資料包含一張影像及最多 5 句文字描述（`caption_0` ～ `caption_4`）。

**資料切分（`split_dataset()`）:**

- 若資料集提供官方 `test` 分割:測試集使用官方 `test`，並將 `train` 以固定 seed 隨機切分為 90% 訓練集與 10% 驗證集。
- 若沒有官方 `test` 分割:以影像 row 為單位隨機切分為 80% 訓練、10% 驗證、10% 測試。
- 切分在展開 caption 之前以影像 row 進行，避免同一張影像出現在不同分割。
- 此切分邏輯在 `Quiz4.py` 內獨立執行，並未讀取 `data_quiz1/flickr8k/` 的分割檔。

**樣本與標籤:**

- 訓練時，每個 epoch 每張影像出現一次，目標文字從該影像的可用 caption 中隨機選擇一句。
- 驗證時固定使用 `caption_0` 計算 Validation Loss。

使用 BLIP Processor 進行影像與文字前處理:

- 影像轉換成模型所需的 Pixel Values。
- Caption 轉換成 Token IDs。
- 使用固定長度 32 進行 padding 與 truncation。
- 將 padding token 對應的 Label 設為 `-100`，使其不參與 Cross Entropy Loss 計算。

### 6.3 模型架構:BLIP

使用預訓練模型 `Salesforce/blip-image-captioning-base`。

BLIP（Bootstrapping Language-Image Pre-training）是一種視覺與語言多模態模型，可將影像特徵與文字生成結合，產生描述影像內容的句子。

```text
Input Image
     ↓
Image Processor
     ↓
Vision Encoder
     ↓
Visual Features
     ↓
Text Decoder
     ↓
Generated Caption
```

訓練時使用影像與參考 caption 計算 Language Modeling Loss，並透過反向傳播更新**全部模型參數**（`model.parameters()`）。

### 6.4 訓練設定與訓練結果

| 參數 | 設定 |
|---|---:|
| Random seed | 42 |
| Batch size | 4 |
| Epochs | 2 |
| Learning rate | 0.00005 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Maximum text length | 32 |
| Model | BLIP Image Captioning Base |
| Mixed precision | CUDA 環境啟用 FP16（autocast + GradScaler） |

每個 epoch 結束後計算 Validation Loss，並儲存 Validation Loss 最低的模型與 Processor；訓練結束後重新載入該最佳模型進行評估。

**訓練紀錄（`training_history.csv`）:**

| Epoch | Training Loss | Validation Loss | 時間（秒） |
|---:|---:|---:|---:|
| 1 | 2.3389 | 2.1106 | 445.3 |
| 2 | 2.0829 | 2.1325 | 496.4 |

第 2 epoch 的 Training Loss 持續下降，但 Validation Loss 由 2.1106 小幅回升至 2.1325，因此**最終儲存並用於評估的是第 1 epoch 的模型**。這顯示在此設定下，增加 epoch 已沒有明顯助益，後續可考慮降低學習率或加入 Early Stopping。


### 6.5 生成方法

本實驗比較兩種文字生成策略。

**Greedy Search**

每一步選擇目前機率最高的 Token，直到生成結束。此方法較簡單，通常生成速度較快，但每一步的局部最佳選擇不一定能形成整體最佳句子。

**Beam Search（num_beams = 3）**

每一步保留多個候選序列，再從中選擇較佳的生成結果。Beam Search 可以考慮多條可能的生成路徑，但通常需要較多計算時間。

為了公平比較，兩種方法使用相同的模型與影像，並設定 `max_new_tokens=30`、`do_sample=False`。

### 6.6 評估設定與指標

**評估範圍:**

- 以固定 seed（42）將測試集索引打亂後，取前 `EVAL_IMAGES = 200` 張影像進行評估（若測試集少於 200 張則使用全部）。
- Gemini 評分僅針對上述影像中的前 `GEMINI_IMAGES = 16` 張，以控制 API 呼叫量與費用。

#### BLEU-4

- 使用 NLTK 的 `corpus_bleu`，為 **corpus-level** 計算。
- 採用 **multi-reference**:每張影像使用其全部可用的參考 caption（最多 5 句）。
- 權重為 (0.25, 0.25, 0.25, 0.25)，即 BLEU-4。
- Tokenization 為轉小寫後以空白切分（`lower().split()`），不另外移除標點，因此標點黏在字詞上時可能降低 n-gram 比對的命中率。
- 使用 `SmoothingFunction().method1` 處理零重疊的情況。

分數越高通常代表生成文字與參考描述在用詞及片語上越接近，但 BLEU 不一定能完整反映語意正確性或影像內容的一致性。

#### Gemini 評分

使用 `GEMINI_MODEL` 所設定的模型（目前為 `gemini-3.5-flash-lite`），同時提供影像與兩種生成描述（Caption A 為 Greedy、Caption B 為 Beam），請 Gemini 依影像內容評分 1～5 分:

| 分數 | 評估標準 |
|---|---|
| 5 | 描述準確且與影像內容相關 |
| 4 | 大致準確，只有少量細節遺漏 |
| 3 | 部分正確，但遺漏或誤述重要資訊 |
| 2 | 大部分描述不準確 |
| 1 | 描述與影像無關 |

Gemini 同時回傳簡短理由，並以平均分數比較不同生成方法。評分失敗（例如 API 額度用盡或回傳格式錯誤）的影像會記為空值，不納入平均。

#### Generation Time

記錄每張影像呼叫 `model.generate()` 到生成完成的時間（不含影像前處理與 decode），並計算平均生成時間。CUDA 環境會在計時前後同步 GPU，以減少非同步執行對計時結果的影響。

### 6.7 實驗結果

| Metric | Greedy Search | Beam Search（3） |
|---|---:|---:|
| BLEU-4 | 0.2220 | 0.2526 |
| 平均生成時間 | 0.3637 秒 | 0.4585 秒 |
| Gemini 平均評分 | 4.0625 / 5 | 3.7500 / 5 |

Gemini 評分樣本數 N = 16（兩種方法皆為同 16 張影像；平均分數 4.0625 = 65/16、3.7500 = 60/16）。生成時間會隨執行環境與當下負載而變動，不同次執行的絕對值可能不同，應著重兩種方法的相對差異。

**結果分析:**

- **BLEU-4:** Beam Search 為 0.2526，高於 Greedy Search 的 0.2220，表示其生成文字與參考描述具有較高的 n-gram 重疊程度。
- **生成速度:** Greedy Search 平均每張影像約 0.364 秒；Beam Search 約 0.458 秒，平均生成時間增加約 26.1%。
- **Gemini 評分:** Greedy Search 平均 4.0625 分，高於 Beam Search 的 3.7500 分，差距約 0.31 分。但僅評 16 張影像，且 LLM 評分本身有隨機性，這個差距不宜視為顯著差異；可以確定的是，BLEU-4 較高的 Beam Search 在 Gemini 評分上並未領先，顯示兩種指標衡量的面向不同。

整體而言，Beam Search 在 BLEU-4 上表現較佳，但生成時間較長；Greedy Search 生成較快，且在 16 張影像的 Gemini 評分上較高。兩種指標衡量的面向不同，因此不能只根據單一指標判定哪種方法全面較佳。

以上 Gemini 分數僅為 16 張影像的 LLM 評分平均，不應直接視為完整測試集或人工標註的結果；BLEU-4 與生成時間為 200 張評估影像的結果。

### 6.8 輸出檔案

| 檔案 | 說明 |
|---|---|
| `training_history.csv` | 每個 epoch 的 Training Loss、Validation Loss 與訓練時間 |
| `caption_predictions.csv` | 每張評估影像的參考描述、Greedy / Beam 生成結果、生成時間及 Gemini 評分與理由 |
| `method_comparison.csv` | BLEU-4、平均生成時間及 Gemini 平均分數的比較 |

### 6.9 執行方式

```powershell
python Quiz4.py
```
### 6.10 Gemini API 設定

若要啟用 Gemini 評分，需先取得 Google Gemini API Key，並設定環境變數。

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
python Quiz4.py
```

請將 `YOUR_API_KEY` 替換成自己的 API Key，勿將金鑰寫入程式碼或上傳至 GitHub。

程式目前設定 `GEMINI_IMAGES = 16`，代表最多對 16 張影像進行 Gemini 評分；若沒有設定 `GEMINI_API_KEY`，程式會略過 Gemini 評估，但仍會計算 BLEU-4 與生成時間。遇到 API 額度用盡（429 / RESOURCE_EXHAUSTED）時會停止重試，該張影像的評分記為空值。

---

## 7. 實驗結果總覽

| Quiz | 任務 | 模型／方法 | 主要結果 |
|---|---|---|---|
| Quiz 1 | Dataset Preparation | IMDb、STL-10、Flickr8k | 完成資料集準備及統計輸出 |
| Quiz 2 | Sentiment Analysis | RNN | Accuracy 71.98%，F1 71.43% |
| Quiz 2 | Sentiment Analysis | LSTM | Accuracy 83.72%，F1 83.55% |
| Quiz 3 | Image Classification | ViT | Accuracy 89.20%，Macro-AUC 0.9944 |
| Quiz 3 | Image Classification | ResNet18 | Accuracy 92.38%，Macro-AUC 0.9964 |
| Quiz 4 | Image Captioning | Greedy Search | BLEU-4 0.2220，平均 0.3637 秒，Gemini 4.0625 / 5 |
| Quiz 4 | Image Captioning | Beam Search（3） | BLEU-4 0.2526，平均 0.4585 秒，Gemini 3.7500 / 5 |

## 8. Conclusion

本專案透過四個 Quiz 實作不同的深度學習任務，從資料集整理、文字序列建模、影像特徵學習到多模態文字生成，建立完整的實驗流程。

在情緒分類中，LSTM 在本次設定下優於 RNN，但兩者的 Dropout 設定不同，且 RNN 訓練不穩定，差距不應全部歸因於模型結構；在 STL-10 影像分類中，ResNet18 的 Accuracy 與 Macro-AUC 均略高於 ViT，且訓練時間遠短於 ViT；在影像描述任務中，Beam Search 提升 BLEU-4 但需要更多生成時間，Greedy Search 的 Gemini 平均評分較高（4.06 對 3.75，但僅 16 張影像）。

這些結果說明，不同模型與推論策略在分類能力、文字相似度、內容正確性及運算效率之間可能存在取捨，需要透過多種評估指標進行整體比較。由於所有實驗皆為單一 seed，結論僅適用於本次設定。