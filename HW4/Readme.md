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
│   │   ├── split_summary.csv
│   │   ├── captions.csv
│   │   ├── split_info.json
│   │   ├── caption_examples.csv
│   │   ├── sample_images.csv
│   │   ├── sample_grid.png
│   │   └── sample_images/
│   │
│   └── raw/
│       └── STL-10 原始資料集
│
├── results_quiz2/
│   ├── vocab.json
│   ├── rnn_best.pth
│   ├── lstm_best.pth
│   ├── rnn_training_history.csv
│   ├── lstm_training_history.csv
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   └── loss_comparison.png
│
├── results_quiz3/
│   ├── split_indices.json
│   ├── vit_best.pth
│   ├── resnet18_best.pth
│   ├── vit_training_history.csv
│   ├── resnet18_training_history.csv
│   ├── model_comparison.csv
│   └── test_predictions.csv
│
└── results_quiz4/
    ├── blip_finetuned/
    │   ├── config.json
    │   ├── model weights
    │   ├── preprocessor_config.json
    │   └── tokenizer / processor files
    ├── training_history.csv
    ├── caption_predictions.csv
    └── method_comparison.csv
```

**目錄說明：**

- `Quiz1.py`：下載與準備 IMDb、STL-10、Flickr8k 三種資料集，產生資料分割資訊、類別分布、範例圖片及文字描述。
- `Quiz2.py`：建立 RNN 與 LSTM 情緒分類模型，完成訓練、驗證、測試及結果比較。
- `Quiz3.py`：微調 ViT 與 ResNet18，完成 STL-10 影像分類，並比較 Accuracy 與 Macro-AUC。
- `Quiz4.py`：訓練或載入 BLIP 影像描述模型，比較 Greedy Search 與 Beam Search，使用 BLEU-4 及 Gemini 評估結果。
- `data_quiz1/`：存放 Quiz 1 產生的資料集統計、分割資訊與範例。
- `results_quiz2/`、`results_quiz3/`、`results_quiz4/`：分別存放模型權重、訓練紀錄、測試預測及評估結果。

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

Windows PowerShell：

```powershell
cd HW4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

安裝 PyTorch 時，請依照電腦是否有 NVIDIA GPU，選擇合適的 CUDA 或 CPU 版本。接著安裝其他套件：

```powershell
pip install torchvision datasets transformers scikit-learn numpy pandas matplotlib pillow nltk google-genai
```

請確認 PyTorch 與 Torchvision 版本相容，並依實際硬體環境安裝對應的版本。

---

## 3. Quiz 1 — Dataset Preparation

### 3.1 作業目標

依照作業要求，準備三種深度學習任務所需的資料集：

1. Sentiment Analysis：文字情緒分類。
2. Image Classification：影像分類。
3. Image Captioning：影像與文字描述配對。

### 3.2 使用資料集

| 資料集 | 任務 | 資料內容 |
|---|---|---|
| IMDb | Sentiment Analysis | 電影評論及二元情緒標籤 |
| STL-10 | Image Classification | 10 類 RGB 影像 |
| Flickr8k | Image Captioning | 影像及對應的文字描述 |

#### IMDb

使用 Hugging Face 上的 `stanfordnlp/imdb`。

- 原始訓練集：25,000 筆。
- 驗證集：由原始訓練集切分 20%，共 5,000 筆。
- 實際訓練集：20,000 筆。
- 測試集：25,000 筆。
- 類別：Negative、Positive。

採用固定 random seed 與分層切分（stratified split），保留獨立測試集，避免使用測試資料進行訓練。

產生的主要檔案：

- `split_summary.csv`：各資料集分割的樣本數。
- `label_distribution.csv`：各分割的類別分布。
- `text_examples.csv`：各分割的評論範例。
- `train_validation_indices.csv`：訓練集與驗證集的索引紀錄。

#### STL-10

使用 Torchvision 的 `STL10`，包含 10 個影像類別：

`airplane`、`bird`、`car`、`cat`、`deer`、`dog`、`horse`、`monkey`、`ship`、`truck`。

- 原始訓練集: 5,000 張。
- 訓練集: 4,500 張。
- 驗證集: 500 張。
- 測試集: 8,000 張。

訓練集依照固定 seed 隨機切分 90%／10%，測試集使用資料集提供的官方測試集。

主要輸出: 

- `split_indices.json`：訓練、驗證與測試索引。
- `split_summary.csv`：各分割的影像數量。
- `label_distribution.csv`：各類別的樣本分布。
- `sample_images.csv`：範例圖片的索引、標籤及路徑。
- `sample_grid.png`：20 張訓練影像的排列展示。
- `sample_images/`：個別範例圖片。

#### Flickr8k

使用 Hugging Face 上的 `jxie/flickr8k`，準備影像描述生成任務所需的影像與文字描述配對。

程式會自動辨識資料集中的影像、caption 及影像識別欄位，並處理可解碼的影像。

若資料集提供 `train`、`validation`、`test` 三個官方分割，則保留官方分割；否則依照唯一影像 ID 隨機切分為 80% 訓練集、10% 驗證集及 10% 測試集。

切分時以影像 ID 為單位，而非直接隨機切分 caption rows，以降低同一張影像同時出現在不同資料集的風險。

主要輸出: 

- `split_summary.csv`: 各分割的影像描述配對數與唯一影像數。
- `captions.csv`: 影像 ID、文字描述及來源分割。
- `split_info.json`: 分割方式與 random seed。
- `caption_examples.csv`: 各分割的文字描述範例。
- `sample_images.csv`: 範例影像及 caption。
- `sample_grid.png`: 影像與文字描述的排列展示。
- `sample_images/`: 個別範例圖片。

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

兩個模型共用 Embedding 與全連接分類層，主要差異在循環神經網路的結構。

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

模型以 Validation Loss 選擇最佳權重，若連續 3 個 epoch 未改善則提前停止訓練。

### 4.5 測試結果

以下為測試集實驗結果：

| Metric | RNN | LSTM |
|---|---:|---:|
| Loss | 0.6149 | 0.3978 |
| Accuracy | 71.98% | 83.72% |
| Precision | 72.88% | 84.44% |
| Recall | 70.03% | 82.67% |
| F1-score | 71.43% | 83.55% |

**結果分析: **

LSTM 的 Accuracy 比 RNN 高出約 11.74 個百分點，且 Loss 較低，Precision、Recall 與 F1-score 均有改善。結果顯示，在本次 IMDb 情緒分類實驗中，LSTM 比單純 RNN 更能有效處理文字序列資訊。

### 4.6 輸出檔案

| 檔案 | 說明 |
|---|---|
| `vocab.json` | 訓練時建立的詞彙表 |
| `rnn_best.pth` | RNN 最佳模型權重 |
| `lstm_best.pth` | LSTM 最佳模型權重 |
| `rnn_training_history.csv` | RNN 每個 epoch 的 Loss、Accuracy 與 F1 |
| `lstm_training_history.csv` | LSTM 每個 epoch 的 Loss、Accuracy 與 F1 |
| `model_comparison.csv` | RNN 與 LSTM 的測試集比較 |
| `test_predictions.csv` | 每筆測試評論的真實標籤、預測標籤與類別機率 |
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

訓練集與驗證集來自 STL-10 官方訓練集: 

- 訓練集: 4,500 張。
- 驗證集: 500 張。
- 測試集: 8,000 張。

為符合預訓練模型的輸入規格，使用以下影像轉換。

**Training Transform**

1. `RandomResizedCrop(224)`: 隨機裁切並調整至 224 × 224。
2. `RandomHorizontalFlip()`: 隨機水平翻轉。
3. `ToTensor()`: 轉換成 Tensor。
4. `Normalize()`: 依照 ImageNet 的 mean 與 standard deviation 正規化。

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

每個 epoch 後評估驗證集，以 Validation Accuracy 最高的模型作為最佳模型並儲存權重。

### 5.5 測試結果

| Metric | ViT | ResNet18 |
|---|---:|---:|
| Test Loss | 0.3329 | 0.2263 |
| Accuracy | 89.20% | 92.38% |
| Macro-AUC | 0.9944 | 0.9964 |

**結果分析：**

ResNet18 的 Accuracy 比 ViT 高約 3.18 個百分點，且 Test Loss 較低。兩個模型的 Macro-AUC 均高於 0.99，顯示模型對 STL-10 十個類別具有良好的區分能力；其中 ResNet18 在本次測試中整體表現較佳。

Macro-AUC 採用 One-vs-Rest（OvR）方式計算各類別的 ROC-AUC，再對十個類別取平均，使每個類別具有相同權重。

### 5.6 輸出檔案

| 檔案 | 說明 |
|---|---|
| `split_indices.json` | 訓練、驗證與測試集索引 |
| `vit_best.pth` | ViT 最佳模型權重 |
| `resnet18_best.pth` | ResNet18 最佳模型權重 |
| `vit_training_history.csv` | ViT 每個 epoch 的訓練及驗證指標 |
| `resnet18_training_history.csv` | ResNet18 每個 epoch 的訓練及驗證指標 |
| `model_comparison.csv` | 兩個模型的測試集結果 |
| `test_predictions.csv` | 每張測試影像的真實類別、預測類別及各類別機率 |

### 5.7 執行方式

```powershell
python Quiz3.py
```

---

## 6. Quiz 4 — Image Captioning

### 6.1 作業目標

使用 Flickr8k 訓練影像描述模型，讓模型根據輸入影像產生自然語言描述，並完成以下實驗：

1. 訓練及儲存影像描述模型。
2. 使用 Gemini 評估生成描述與影像內容的一致性。
3. 使用 BLEU-4 量化生成文字與參考描述的相似程度。
4. 比較 Greedy Search 與 Beam Search 的生成時間與描述品質。

### 6.2 使用資料集與前處理

資料集使用 Hugging Face 上的 `jxie/flickr8k`。

每筆資料包含一張影像及對應的文字描述。模型訓練時從同一張影像的可用 caption 中隨機選擇一個描述作為目標文字。

使用 BLIP Processor 進行影像與文字前處理：

- 影像轉換成模型所需的 Pixel Values。
- Caption 轉換成 Token IDs。
- 使用固定長度 32 進行 padding 與 truncation。
- 將 padding token 對應的 Label 設為 `-100`，使其不參與 Cross Entropy Loss 計算。

訓練資料以影像 row 為單位，而非先將同一張影像的多個 caption 展開成互相獨立的影像樣本。

### 6.3 模型架構：BLIP

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

訓練時使用影像與參考 caption 計算 Language Modeling Loss，並透過反向傳播更新模型參數。

### 6.4 訓練設定

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
| Mixed precision | CUDA 環境啟用 FP16 |

每個 epoch 結束後計算 Validation Loss，並儲存 Validation Loss 最低的模型與 Processor。

**注意：** `Quiz4.py` 的訓練函式為 `train_model()`，但目前正式執行的 `main()` 會直接載入 `results_quiz4/blip_finetuned/` 中已儲存的模型，再執行評估。若尚未建立模型權重，需先執行訓練流程，才能執行目前的評估流程。

### 6.5 生成方法

本實驗比較兩種文字生成策略。

**Greedy Search**

每一步選擇目前機率最高的 Token，直到生成結束。此方法較簡單，通常生成速度較快，但每一步的局部最佳選擇不一定能形成整體最佳句子。

**Beam Search（num_beams = 3）**

每一步保留多個候選序列，再從中選擇較佳的生成結果。Beam Search 可以考慮多條可能的生成路徑，但通常需要較多計算時間。

為了公平比較，兩種方法使用相同的模型與影像，並設定 `max_new_tokens=30`、`do_sample=False`。

### 6.6 評估指標

#### BLEU-4

BLEU-4 用於比較生成描述與參考描述之間的 n-gram 重疊程度，最高階數為 4，並使用 smoothing function 處理零重疊的情況。

分數越高通常代表生成文字與參考描述在用詞及片語上越接近，但 BLEU 不一定能完整反映語意正確性或影像內容的一致性。

#### Gemini 評分

使用 Gemini 對影像與兩種生成描述進行評估，評分範圍為 1～5 分：

| 分數 | 評估標準 |
|---|---|
| 5 | 描述準確且與影像內容相關 |
| 4 | 大致準確，只有少量細節遺漏 |
| 3 | 部分正確，但遺漏或誤述重要資訊 |
| 2 | 大部分描述不準確 |
| 1 | 描述與影像無關 |

Gemini 同時回傳簡短理由，並以平均分數比較不同生成方法。

#### Generation Time

記錄每張影像從呼叫 `model.generate()` 到生成完成的時間，並計算平均生成時間。CUDA 環境會在計時前後同步 GPU，以減少非同步執行對計時結果的影響。

### 6.7 實驗結果

以下為目前已取得的測試結果：

| Metric | Greedy Search | Beam Search（3） |
|---|---:|---:|
| BLEU-4 | 0.2220 | 0.2526 |
| 平均生成時間 | 0.1724 秒 | 0.2352 秒 |
| Gemini 平均評分 | 3.79 / 5 | 3.68 / 5 |

**結果分析：**

- **BLEU-4：** Beam Search 為 0.2526，高於 Greedy Search 的 0.2220，表示其生成文字與參考描述具有較高的 n-gram 重疊程度。
- **生成速度：** Greedy Search 平均每張影像約 0.172 秒；Beam Search 約 0.235 秒。Beam Search 的平均生成時間約增加 36.3%。
- **Gemini 評分：** Greedy Search 平均 3.79 分，略高於 Beam Search 的 3.68 分。這表示 BLEU-4 較高不一定代表影像內容描述一定更準確。

整體而言，Beam Search 在 BLEU-4 上表現較佳，但 Greedy Search 生成速度較快，且 Gemini 平均評分略高。兩種指標衡量的面向不同，因此不能只根據單一指標判定哪種方法全面較佳。

以上 Gemini 分數是目前已成功取得的評分樣本平均值，不應直接視為完整測試集的人工標註結果。

### 6.8 輸出檔案

| 檔案 | 說明 |
|---|---|
| `blip_finetuned/` | 儲存最佳 BLIP 模型權重與 Processor 設定 |
| `training_history.csv` | 每個 epoch 的 Training Loss、Validation Loss 與訓練時間 |
| `caption_predictions.csv` | 每張測試影像的參考描述、Greedy / Beam 生成結果、生成時間及 Gemini 評分與理由 |
| `method_comparison.csv` | BLEU-4、平均生成時間及 Gemini 平均分數的比較 |

### 6.9 執行方式

```powershell
python Quiz4.py
```

目前的 `main()` 會載入既有的微調模型，對測試資料進行描述生成與評估。

若要從頭訓練模型，需先執行程式中的 `split_dataset()`、`train_model()` 及後續評估流程；若尚未訓練並儲存模型，直接執行目前的 `main()` 會因找不到模型目錄而失敗。

### 6.10 Gemini API 設定

若要啟用 Gemini 評分，需先取得 Google Gemini API Key，並設定環境變數。

Windows PowerShell：

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
python Quiz4.py
```

請將 `YOUR_API_KEY` 替換成自己的 API Key，勿將金鑰寫入程式碼或上傳至 GitHub。

程式目前設定 `GEMINI_IMAGES = 16`，代表最多對 16 張影像進行 Gemini 評分；如果沒有設定 `GEMINI_API_KEY`，程式會略過 Gemini 評估，但仍會計算 BLEU-4 與生成時間。

---

## 7. 實驗結果總覽

| Quiz | 任務 | 模型／方法 | 主要結果 |
|---|---|---|---|
| Quiz 1 | Dataset Preparation | IMDb、STL-10、Flickr8k | 完成資料集準備及統計輸出 |
| Quiz 2 | Sentiment Analysis | RNN | Accuracy 71.98%，F1 71.43% |
| Quiz 2 | Sentiment Analysis | LSTM | Accuracy 83.72%，F1 83.55% |
| Quiz 3 | Image Classification | ViT | Accuracy 89.20%，Macro-AUC 0.9944 |
| Quiz 3 | Image Classification | ResNet18 | Accuracy 92.38%，Macro-AUC 0.9964 |
| Quiz 4 | Image Captioning | Greedy Search | BLEU-4 0.2220，平均 0.1724 秒 |
| Quiz 4 | Image Captioning | Beam Search（3） | BLEU-4 0.2526，平均 0.2352 秒 |

## 8. 重現實驗注意事項

1. **固定隨機種子：** 使用 seed 42 盡可能提高實驗的可重現性；GPU、套件版本及運算環境仍可能造成數值差異。
2. **資料集下載：** 首次執行需要網路連線下載 Hugging Face 或 Torchvision 的資料集及預訓練權重。
3. **模型權重：** `.pth` 檔案及 BLIP 預訓練／微調權重可能佔用大量空間，可依提交規定決定是否上傳。
4. **訓練與推論：** Quiz 2、Quiz 3 會訓練模型；Quiz 4 從頭訓練、儲存最佳權重，再執行測試集評估。
5. **硬體差異：** ViT、ResNet18 與 BLIP 的訓練時間會依 GPU、CUDA、記憶體及 batch size 改變。
6. **評估公平性：** 測試集應保留作為最終評估資料；模型選擇應依據驗證集指標完成。
7. **Git 管理：** 建議將虛擬環境、快取、下載的大型資料集及不必要的模型權重排除於版本控制之外，並保留程式碼、README 與必要的實驗結果。

## 9. Conclusion

本專案透過四個 Quiz 實作不同的深度學習任務，從資料集整理、文字序列建模、影像特徵學習到多模態文字生成，建立完整的實驗流程。

在情緒分類中，LSTM 優於本次實驗的 RNN；在 STL-10 影像分類中，ResNet18 的 Accuracy 與 Macro-AUC 均略高於 ViT；在影像描述任務中，Beam Search 提升 BLEU-4，但需要更多生成時間，而 Greedy Search 的 Gemini 平均評分略高。

這些結果說明，不同模型與推論策略在分類能力、文字相似度、內容正確性及運算效率之間可能存在取捨，需要透過多種評估指標進行整體比較。
