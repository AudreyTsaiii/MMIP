# Deep Learning HW: Image Retrieval, Multimodal Classification & CLIP Image Captioning

本專案完成三個 Quiz:影像檢索(CNN 與 ViT Backbone)、影像與文字的多模態分類(更換影像 Backbone),以及以 CLIP 作為 Image Encoder 搭配 GPT-2 文字 Decoder 的 Image Captioning,並與前一次作業的 BLIP 比較 BLEU-4、BERTScore 與生成時間。

## 1. 專案架構(Project Structure)

```text
HW/
├── Quiz1.py
├── Quiz2.py
├── Quiz3.py
├── Readme.md
│
├── results_quiz1/
│   ├── split_indices.json
│   ├── resnet18_history.csv
│   ├── resnet18_retrieval.png
│   ├── vit_history.csv
│   ├── vit_retrieval.png
│   └── model_comparison.csv
│
├── results_quiz2/
│   ├── resnet18/
│   │   ├── resnet18_history.json
│   │   ├── resnet18_results.json
│   │   └── feature_cache/
│   ├── vit_b_16/
│   │   ├── vit_b_16_history.json
│   │   ├── vit_b_16_results.json
│   │   └── feature_cache/
│   └── backbone_comparison.csv
│
└── results_quiz3_clip/
    ├── clip_gpt2/
    │   ├── image_processor/
    │   └── tokenizer/
    ├── cache/
    ├── training_history.csv / training_history.json
    ├── caption_predictions.csv
    ├── caption_comparison.csv
    ├── backbone_comparison.csv
    └── comparison_results.json
```

**目錄說明:**

- `Quiz1.py`:以 ResNet18 與 ViT 作為 Backbone,訓練 embedding 層完成影像檢索。
- `Quiz2.py`:影像與文字的多模態配對分類,比較不同影像 Backbone。
- `Quiz3.py`:CLIP + GPT-2 影像描述模型,並與 Week 4 的 BLIP 比較。
- `results_quiz1/`、`results_quiz2/`、`results_quiz3_clip/`:各 Quiz 的模型權重、訓練紀錄與評估結果。

---

## 2. 環境需求(Requirements)

| 套件 | 用途 |
|---|---|
| Python | 執行環境 |
| PyTorch、Torchvision | 模型建立、STL-10、ResNet18 與 ViT-B/16 |
| Transformers | DistilBERT、CLIP、GPT-2 |
| Hugging Face Datasets | 讀取 Flickr8k |
| Scikit-learn | 資料切分與分類指標 |
| NumPy、Pandas、Matplotlib | 資料處理、結果輸出與檢索結果圖 |
| Pillow | 影像處理 |
| NLTK | BLEU-4 |
| bert-score | BERTScore |

```powershell
pip install torchvision transformers datasets scikit-learn numpy pandas matplotlib pillow nltk bert-score
```

PyTorch 請依硬體環境(CUDA 或 CPU)選擇對應版本安裝。

---

## 3. Quiz 1 — 影像檢索(Image Retrieval)

### 3.1 作業目標

1. 自行準備影像檢索資料集。
2. 建立並訓練影像檢索模型。
3. 完成檢索推論並展示檢索結果。
4. 分別使用 CNN(ResNet18)與 Vision Transformer(ViT-B/16)作為 Backbone,並比較檢索表現。

### 3.2 資料集

使用 STL-10(10 類 RGB 影像),以類別標籤作為相關性判斷(同類別即視為相關):

| 分割 | 影像數量 | 說明 |
|---|---:|---|
| Train | 4,000 | 由官方訓練集依類別分層切分(seed 42) |
| Validation | 1,000 | 同上 |
| Test | 8,000 | 官方測試集 |

影像轉換:`Resize(224, 224)`、`ToTensor()`、ImageNet 正規化,不使用資料增強。

### 3.3 模型與訓練流程

Backbone 使用 ImageNet 預訓練權重並**完全凍結**(`requires_grad = False`),先一次性萃取所有影像特徵,再只訓練後方的 embedding 層。

```text
Image (224 × 224)
  ↓
Frozen Backbone
  ResNet18 → 512 維特徵
  ViT-B/16 → 768 維特徵
  ↓
Embedding Layer(訓練)
  Linear(特徵維度 → 512) → ReLU → Linear(512 → 256)
  ↓
L2 Normalize → 256 維 Embedding
  ↓
Cosine Similarity Search → Top-K 檢索結果
```

**損失函數:** Supervised Contrastive Loss(temperature = 0.1),同一批次中同類別影像互為正樣本,其餘為負樣本。

| 參數 | 設定 |
|---|---:|
| Random seed | 42 |
| Batch size | 64 |
| Epochs | 5 |
| Learning rate | 0.001 |
| Optimizer | AdamW(weight decay 0.0001) |
| Embedding dimension | 256 |
| Temperature | 0.1 |

以 Validation Loss 最低的 epoch 作為最佳 embedding:

| 項目 | ResNet18 | ViT |
|---|---:|---:|
| 最佳 epoch | 3 | 2 |
| 最佳 Validation Loss | 4.9707 | 4.7280 |

**關於 Validation Loss 高於 Training Loss:** 訓練時每個 batch 為 64 張,驗證時是將 1,000 張驗證影像一次計算。Supervised Contrastive Loss 的分母包含 batch 內所有負樣本,批次越大,損失的基準值越高(差距約為 ln(1000/64) ≈ 2.75,與實際觀察到的差距相符)。因此 Validation Loss 的絕對值不能與 Training Loss 直接比較,只用於在不同 epoch 之間選擇最佳 embedding。

特徵萃取時間(Backbone 前向傳遞):

| 資料 | ResNet18 | ViT |
|---|---:|---:|
| Train(4,000 張) | 9.2 秒 | 46.5 秒 |
| Validation(1,000 張) | 1.9 秒 | 12.7 秒 |
| Test(8,000 張) | 14.9 秒 | 141.7 秒 |

### 3.4 檢索評估設定

- **Query 與 Gallery:** 測試集的前 1,000 張影像作為 Query,全部 8,000 張測試影像作為檢索資料庫;檢索時排除 Query 自己(self-match)。
- **相似度:** Embedding 已做 L2 正規化,以內積(等同 cosine similarity)排序。
- **Recall@K:** 前 K 個結果中至少有一張同類別影像的比例(hit rate)。
- **mAP@5:** 前 5 個結果的 Average Precision,分母為 min(該類別總相關影像數, 5),再對所有 Query 取平均。
- **retrieval_ms_per_query:** 1,000 個 Query 對 8,000 張影像的相似度矩陣乘法時間除以 Query 數;不含特徵萃取與排序。

### 3.5 實驗結果

| Metric | ResNet18 | ViT |
|---|---:|---:|
| Recall@1 | 93.40% | 98.00% |
| Recall@5 | 97.70% | 99.00% |
| mAP@5 | 0.9255 | 0.9730 |
| 檢索時間(ms / query) | 0.0223 | 0.0172 |

**結果分析:**

- ViT 的 Recall@1 比 ResNet18 高 4.6 個百分點,mAP@5 高約 0.048,檢索品質三項指標都較佳。
- 兩個模型的 Embedding 維度(256)與資料庫大小(8,000)相同,檢索時間量測的是同樣大小的矩陣乘法,因此 0.0223 與 0.0172 的差異屬於量測誤差,不代表其中一個模型的檢索比較快。
- 兩者真正的成本差異在特徵萃取:ViT 萃取 8,000 張測試影像需 141.7 秒,約為 ResNet18(14.9 秒)的 9.5 倍。ViT 以較高的特徵萃取成本換取較佳的檢索品質。
- Backbone 完全凍結,只訓練 embedding 層,且訓練僅 5 個 epoch、每個 epoch 約 0.2 秒;所有結果為單一 random seed。
- 〔建議補充:未訓練 embedding(直接以 Backbone 原始特徵做 cosine 檢索)的 baseline,以證明 embedding 訓練的效益〕

**檢索結果展示:** `results_quiz1/resnet18_retrieval.png` 與 `results_quiz1/vit_retrieval.png` 各展示 5 張 Query 影像與其 Top-5 檢索結果(標示 Match / Non-match)。

### 3.6 執行方式

```powershell
python Quiz1.py
```

程式使用 `download=False` 讀取 STL-10,需先在 `DATA_DIR` 指定的路徑下備妥 STL-10 資料集(與前一次作業相同);`DATA_DIR` 目前為寫死的 Windows 路徑,在其他電腦執行時需修改。

---

## 4. Quiz 2 — 多模態深度學習神經網路

### 4.1 作業目標

1. 自行準備包含影像與文字的多模態資料集。
2. 使用影像與文字 Backbone 萃取特徵,融合特徵後建立分類模型。
3. 完成訓練並進行多模態分類推論。
4. 更換影像 Backbone 重新訓練,並比較不同 Backbone 組合的表現。

### 4.2 資料集與任務

使用 Hugging Face 的 `jxie/flickr8k`,並採用其官方 `train`、`validation`、`test` 三個分割。任務為**影像-文字配對分類(Image-Text Matching)**:判斷一段文字是否描述了該影像,為二元分類。

每張影像產生 10 組影像-文字配對:

| 類型 | 組成 | 標籤 | 數量 |
|---|---|---|---:|
| 正樣本 | 影像 i + 影像 i 自己的 5 句 caption | 1(Match) | 5 |
| 負樣本 | 影像 i + 下一張影像(i+1)的 5 句 caption | 0(Not Match) | 5 |

因此每個分割中正負樣本各佔 50%,配對總數為「影像數 × 10」:

| 分割 | 影像數 | Caption 數 | 配對數 | Match | Not Match |
|---|---:|---:|---:|---:|---:|
| Train | 6,000 | 30,000 | 60,000 | 30,000 | 30,000 |
| Validation | 1,000 | 5,000 | 10,000 | 5,000 | 5,000 |
| Test | 1,000 | 5,000 | 10,000 | 5,000 | 5,000 |

### 4.3 模型架構

兩個 Backbone 皆使用預訓練權重並**凍結**,先萃取並快取特徵,只訓練融合後的分類器。

```text
Image → Image Backbone (ResNet18: 512 維 / ViT-B/16: 768 維) ─┐
                                                              ├→ Concatenation → Linear(→512) → ReLU
Caption → DistilBERT [CLS] 向量 (768 維, max_length 64) ──────┘      → Dropout(0.3) → Linear(512 → 2)
```

比較的 Backbone 組合:

| 組合 | 影像 Backbone | 文字 Backbone |
|---|---|---|
| A | ResNet18 | DistilBERT(`distilbert-base-uncased`) |
| B | ViT-B/16 | DistilBERT(`distilbert-base-uncased`) |

兩個組合的文字 Backbone 相同,只更換影像 Backbone,因此差異可歸因於影像特徵。

| 參數 | 設定 |
|---|---:|
| Random seed | 42 |
| Batch size | 256 |
| Epochs | 10 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | Cross Entropy Loss |
| 最佳模型選擇依據 | Validation F1 |

### 4.4 實驗結果

最佳 epoch 以 Validation F1 選出,測試集只在最後以最佳 checkpoint 評估一次:

| Metric | ResNet18 + DistilBERT | ViT-B/16 + DistilBERT |
|---|---:|---:|
| Best Epoch | 9 | 10 |
| Accuracy | 83.17% | 85.75% |
| Precision | 77.78% | 81.77% |
| Recall | 92.88% | 92.02% |
| F1-score | 84.66% | 86.59% |
| ROC-AUC | 0.9022 | 0.9254 |

**結果分析:**

- ViT 組合在 Accuracy(高 2.58 個百分點)、Precision(高約 4.0 個百分點)、F1(高約 1.9 個百分點)與 ROC-AUC(高約 0.023)上較佳;ResNet18 組合的 Recall 略高(92.88% 對 92.02%)。
- 兩個組合的 Recall 都明顯高於 Precision,表示模型傾向將配對判為 Match,誤判為 Match 的比例較高。
- ResNet18 以 Validation F1 選出的最佳 epoch 為第 9;但其 Validation Loss 與 ROC-AUC 在第 10 epoch 仍是最佳,且 Validation F1 在各 epoch 間有波動(例如第 9 epoch 0.8515、第 10 epoch 0.8471),顯示以 F1 選模帶有一些雜訊。ViT 的最佳 epoch 為第 10,等於設定的最大 epoch。兩者的訓練可能都尚未完全收斂,更長的訓練可能進一步提升結果。
- 負樣本是以「下一張影像的 caption」建構,屬於固定的簡單負樣本,模型表現不代表在較難的配對(例如相似場景)上的能力。
- 即使固定 seed,重新執行 ResNet18 時結果仍有些微差異(Accuracy 由 83.53% 變為 83.17%,最佳 epoch 由 10 變為 9),應來自 GPU 運算的非決定性。ViT 與 ResNet18 的差距大於這個變動,但整體仍為單一 seed 的結果。

### 4.5 執行方式

```powershell
python Quiz2.py --image-backbone resnet18
python Quiz2.py --image-backbone vit_b_16
```

每次執行結束後會比對現有的結果檔,並輸出 `results_quiz2/backbone_comparison.csv`;兩個 Backbone 都跑完後,比較表才會同時包含兩列。

---

## 5. Quiz 3 — CLIP Image Captioning

### 5.1 作業目標

1. 使用 Week 4 的 Flickr8k 影像描述資料集完成資料準備。
2. 使用 CLIP 作為 Image Encoder,搭配 GPT-2 文字 Decoder 建立 Image Captioning 模型。
3. 完成訓練並產生影像描述。
4. 比較 Week 4 的 BLIP 與本次 CLIP-GPT2 的 BLEU-4 與 BERTScore。

### 5.2 資料集

沿用 Week 4 的切分方式:若資料集有官方 `test` 分割則直接使用,並將 `train` 以固定 seed 切分為 90% 訓練與 10% 驗證。訓練時每張影像隨機選一句 caption 作為目標;驗證時固定使用 `caption_0`。

| 分割 | 影像數量 |
|---|---:|
| Train | 5,400 |
| Validation | 600 |
| Test | 1,000(評估時取其中 200 張) |

### 5.3 模型架構

```text
Image → CLIP Vision Encoder(凍結)→ pooler_output (768 維)
      → Linear(768 → 768) + Tanh → 1 個 prefix embedding
      → [prefix, caption token embeddings] → GPT-2 → Caption
```

- Image Encoder:`openai/clip-vit-base-patch32` 的 `CLIPVisionModel`,參數凍結。
- Decoder:`gpt2`(124M),所有參數皆參與訓練;影像僅以**單一個 prefix embedding** 輸入 Decoder。
- 損失:僅計算 caption token(含結尾的 EOS)的 Language Modeling Loss,prefix 與 padding 位置不計入。

| 參數 | 設定 |
|---|---:|
| Random seed | 42 |
| Batch size | 4 |
| Epochs | 2 |
| Learning rate | 0.00005 |
| Optimizer | AdamW(weight decay 0.0001) |
| Maximum text length | 32 |
| Mixed precision | CUDA 環境啟用 FP16 |
| 最佳模型選擇依據 | Validation Loss |

**訓練紀錄(`training_history.csv`):**

| Epoch | Training Loss | Validation Loss | 時間(秒) |
|---:|---:|---:|---:|
| 1 | 2.9172 | 2.4911 | 395.1 |
| 2 | 2.5451 | 2.3802 | 7034.7 |

兩個 epoch 的 Validation Loss 持續下降,因此最佳模型為第 2 epoch。第 2 epoch 的耗時明顯高於第 1 epoch,可能受執行環境影響(例如電腦休眠或 GPU 資源不足),與模型表現無關。

### 5.4 評估設定

- **評估影像:** 以固定 seed 打亂測試集索引後取前 200 張,與 Week 4 的 BLIP 使用相同的 200 張影像(以 `row_index` 對齊)。
- **生成設定:** Greedy Search 與 Beam Search(num_beams = 3),`max_new_tokens = 30`,不使用取樣。CLIP-GPT2 另外使用 `repetition_penalty = 1.1`(Week 4 的 BLIP 沒有)。
- **解碼實作:** 由於 HuggingFace `generate()` 搭配 `inputs_embeds` 在本環境下會產生退化輸出(重複的單一 token、沒有空格),CLIP-GPT2 改用自行實作的解碼函式:每一步將「影像 prefix + 已生成 token」整段送入 GPT-2 取下一個 token 的機率,與訓練時的計算方式一致。Beam Search 以累積 log 機率除以序列長度作為排序分數。此實作沒有使用 KV cache。
- **使用的模型:** 直接載入訓練時儲存的最佳 checkpoint(第 2 epoch),不重新訓練。
- **BLEU-4:** 與 Week 4 相同:corpus-level、multi-reference(每張影像最多 5 句參考)、`lower().split()` 斷詞、smoothing method1。
- **BERTScore-F1:** 使用 `bert-score`(`lang="en"`、不做 baseline rescale);每句生成描述與該影像的所有參考描述各計算一次 F1,取最大值作為該影像的分數,再對所有影像取平均。
- **平均生成時間:** 讀取各自 `caption_predictions.csv` 中記錄的生成秒數,包含影像編碼與文字生成,不含影像前處理。

### 5.5 實驗結果

| 模型 | BLEU-4 | BERTScore-F1 | 平均生成時間 |
|---|---:|---:|---:|
| BLIP(Greedy) | 0.2220 | 0.9287 | 0.3121 秒 |
| BLIP(Beam) | 0.2526 | 0.9301 | 0.4084 秒 |
| CLIP-GPT2(Greedy) | 0.1661 | 0.9286 | 0.2110 秒 |
| CLIP-GPT2(Beam) | 0.1901 | 0.9303 | 0.2688 秒 |

評估影像為相同的 200 張測試影像;BERTScore 未做 baseline rescale。

**結果分析:**

- **BLEU-4:** BLIP 高於 CLIP-GPT2,Greedy 差 0.056、Beam 差 0.063,CLIP-GPT2 約低 25%,表示 BLIP 生成的描述在用詞與片語上與參考描述的重疊程度較高。
- **BERTScore-F1:** 兩個模型幾乎相同(Greedy 為 0.9287 對 0.9286,Beam 為 0.9301 對 0.9303,差距不超過 0.0002),表示兩者生成描述的語意與參考描述的相似度相近。由於未做 baseline rescale,BERTScore 的數值集中在 0.93 附近,區分能力有限,這樣的微小差距不宜過度解讀。
- **兩項指標合併來看:** CLIP-GPT2 的語意相似度與 BLIP 相當,但用詞與參考描述的重疊較少,可能是生成了語意相近但措辭不同的描述(此為推測,未逐句檢查)。
- **Beam Search vs Greedy Search:** 兩個模型的 Beam Search 都使 BLEU-4 提升(BLIP +0.0306、CLIP-GPT2 +0.0240),BERTScore 只有些微提升,而生成時間分別增加約 30.9% 與 27.4%。
- **生成時間:** CLIP-GPT2 的平均生成時間低於 BLIP。但兩者的實作不同(BLIP 使用 `generate()` 與 KV cache,CLIP-GPT2 為自行實作且沒有 KV cache),且 BLIP 的時間取自 Week 4 預測檔,為不同次的執行紀錄;此外 BLIP 與 CLIP-GPT2 的影像編碼器規模與輸入解析度也不同。因此時間只能作為參考,不宜視為兩種架構的嚴格比較。
- 以上結果為單一 random seed、200 張測試影像。

### 5.6 執行方式

```powershell
python Quiz3.py
```

行為由 `Quiz3.py` 中的兩個設定決定:

| 設定 | 行為 |
|---|---|
| `EVALUATE_ONLY = True` | 略過訓練與生成,直接讀取 `results_quiz3_clip/caption_predictions.csv` 的既有預測,再與 Week 4 比較並計算 BLEU-4 與 BERTScore |
| `EVALUATE_ONLY = False`、`RETRAIN = False` | 若已有 `clip_gpt2_best.pth` 就載入並重新生成預測;沒有才會訓練(目前使用的設定) |
| `EVALUATE_ONLY = False`、`RETRAIN = True` | 重新訓練模型後再生成與評估 |

程式會讀取 Week 4 的 `results_quiz4/caption_predictions.csv`,若找不到會列出已檢查的路徑並停止。

---

## 6. 實驗結果總覽

| Quiz | 任務 | 模型/方法 | 主要結果 |
|---|---|---|---|
| Quiz 1 | 影像檢索 | ResNet18 | Recall@1 93.40%,mAP@5 0.9255 |
| Quiz 1 | 影像檢索 | ViT | Recall@1 98.00%,mAP@5 0.9730 |
| Quiz 2 | 多模態配對分類 | ResNet18 + DistilBERT | Accuracy 83.17%,F1 84.66% |
| Quiz 2 | 多模態配對分類 | ViT-B/16 + DistilBERT | Accuracy 85.75%,F1 86.59% |
| Quiz 3 | Image Captioning | BLIP(Beam) | BLEU-4 0.2526,BERTScore-F1 0.9301 |
| Quiz 3 | Image Captioning | CLIP-GPT2(Beam) | BLEU-4 0.1901,BERTScore-F1 0.9303 |



## 7. Conclusion

在影像檢索中,ViT 的 Recall@1 與 mAP@5 都高於 ResNet18,但特徵萃取成本約為 ResNet18 的 9.5 倍;在多模態配對分類中,以 ViT 取代 ResNet18 作為影像 Backbone,Accuracy、F1 與 ROC-AUC 都有提升。

在 Image Captioning 中,以凍結的 CLIP 搭配單一 prefix 與 GPT-2 的模型,BERTScore-F1 與 BLIP 相當(約 0.93),但 BLEU-4 約低 25%;兩個模型的 Beam Search 都優於 Greedy Search,並需要多約 27% 至 31% 的生成時間。

前兩個 Quiz 的 Backbone 皆為凍結,且所有結論皆為單一 random seed 的結果。