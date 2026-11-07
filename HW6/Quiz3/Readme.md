# HW6 Quiz3：使用 DPO 微調 LLM 與 VLM

## 1. 專案介紹

本專案分為基礎與進階兩個部分，使用 Direct Preference Optimization（DPO）對開源大型語言模型（LLM）及視覺語言模型（VLM）進行微調，觀察模型在指定優化目標上的表現變化。

- **基礎實驗：** 微調 LLM，使模型能夠遵循指定政策，避免在資訊不足時自行捏造答案。
- **進階實驗：** 微調 VLM，使模型在圖片物件辨識時，減少描述圖片中不存在或缺乏視覺證據支持的物件，並測試不同影像品質對模型表現的影響。

透過 Base Model 與 DPO Model 的比較，分析微調前後的模型效能、準確性及推論成本。

---

# Part 1. 基礎：DPO 微調 LLM

## 2. 優化目標

本實驗希望改善 LLM 的政策遵循能力，讓模型在回答問題時：

1. 僅根據題目提供的政策內容回答。
2. 當政策有明確提供答案時，正確回答問題。
3. 當政策沒有提供相關資訊時，明確表示資訊不足，而不是自行推測。
4. 當問題包含錯誤前提時，不應直接接受錯誤前提並產生不實資訊。

## 3. 訓練資料集

使用自行建立的繁體中文 Preference Dataset，共 120 組偏好資料。

| 類別 | 數量 | 訓練目標 |
|---|---:|---|
| Stated | 40 | 政策有明確提供資訊時，選擇正確回答 |
| Unstated | 40 | 政策沒有提供資訊時，選擇明確拒絕推測的回答 |
| False Premise | 40 | 面對錯誤前提時，不接受不正確的假設 |
| **總計** | **120** | |

每組資料包含：

- Prompt：政策內容及使用者問題。
- Chosen：符合政策、資訊正確的回答。
- Rejected：違反政策或包含未經支持資訊的回答。

## 4. 通用能力 Benchmark

另外準備 100 題與政策遵循目標無直接關係的繁體中文選擇題，用來觀察 DPO 微調是否影響模型原本的通用能力。

| 類別 | 題數 |
|---|---:|
| 算術（Arithmetic） | 20 |
| 邏輯（Logic） | 20 |
| 閱讀理解（Reading） | 20 |
| 語文（Language） | 20 |
| 數列（Sequence） | 20 |
| **總計** | **100** |

此 Benchmark 與 Preference Dataset 分開建立，避免直接使用訓練資料評估模型。

## 5. 模型與訓練設定

| 項目 | 設定 |
|---|---|
| Base Model | Qwen/Qwen2.5-1.5B-Instruct |
| 微調方法 | DPO |
| 微調技術 | LoRA / QLoRA |
| Quantization | 4-bit NF4 |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| Learning Rate | 5e-5 |
| Epochs | 1 |
| Gradient Accumulation | 8 |

使用 4-bit NF4 量化與 LoRA 降低微調所需的 GPU 記憶體。

## 6. LLM 評估結果

### 6.1 優化目標：政策遵循能力

以 20 題 Target Evaluation 測試模型對政策資訊的掌握程度。

| 指標 | Base Model | DPO Model |
|---|---:|---:|
| Target Overall Accuracy | 65% | **75%** |
| Stated Accuracy | 80% | **100%** |
| Correct Abstention Rate | 50% | 50% |

DPO 微調後，Target Overall Accuracy 從 65% 提升至 75%，增加 10 個百分點。

在 Stated 類型中，DPO Model 的正確率提升至 100%，顯示模型對政策明確提供的資訊有更好的遵循能力。

然而，Correct Abstention Rate 仍為 50%，代表模型在政策未提供資訊時，仍有改善空間。

### 6.2 通用能力 Benchmark

使用獨立建立的 100 題繁體中文選擇題，比較 Base Model 與 DPO Model 的通用能力。

| 類別 | Base Model | DPO Model |
|---|---:|---:|
| Arithmetic | 20% | 20% |
| Logic | 25% | 25% |
| Reading | 100% | 100% |
| Language | 95% | 95% |
| Sequence | 30% | 30% |
| **Overall Accuracy** | **54%** | **54%** |

Base Model 與 DPO Model 均答對 54 題，整體準確率皆為 54%，沒有觀察到通用 Benchmark 的整體性能變化。

各類別的準確率也完全相同，表示本次微調主要針對指定的政策遵循任務進行優化，並未明顯改變模型在這組通用測試題上的表現。

此結果顯示，DPO 微調在改善特定任務能力的同時，沒有造成這組通用 Benchmark 的整體準確率下降。
---

# Part 2. 進階：DPO 微調 VLM

## 7. 優化目標

本實驗將優化目標延伸至視覺語言模型（VLM）。

希望模型在觀察圖片及辨識物件時：

1. 僅描述圖片中實際可見的物件。
2. 避免加入圖片中不存在的物件。
3. 減少缺乏視覺證據支持的推測。
4. 在不同影像品質條件下，維持較穩定的物件辨識能力。

本實驗使用 COCO 2017 Dataset 作為物件類別與標註來源，並以人工檢查方式建立訓練用的負樣本。

## 8. VLM 訓練資料集

從 COCO 2017 Dataset 選取 120 張圖片作為訓練資料。

每張圖片經過人工檢查，確認其已標註物件及未出現的候選物件，再建立 Preference Pairs。

| 資料內容 | 數量 |
|---|---:|
| 訓練圖片 | 120 |
| 每張圖片建立的偏好資料 | 3 組 |
| Preference Pairs | **360** |
| 不重複訓練圖片 | 120 |

每組 Preference Pair 包含：

- Image：輸入圖片。
- Prompt：要求模型辨識圖片中實際可見的物件。
- Chosen：只描述有標註支持的物件。
- Rejected：在正確物件描述中，額外加入人工確認不存在的候選物件。

透過 Chosen 與 Rejected 的比較，讓模型學習避免產生缺乏視覺證據的物件描述。

### 資料集切分

| Dataset | 圖片數量 | 用途 |
|---|---:|---|
| Train | 120 | DPO 訓練 |
| Target Evaluation | 50 | 評估指定優化目標 |
| General Evaluation | 30 | 評估其他一般圖片上的表現 |

所有資料集均使用不同圖片，避免訓練圖片與測試圖片重疊。

## 9. 模型與訓練設定

| 項目 | 設定 |
|---|---|
| Base Model | Qwen/Qwen2.5-VL-3B-Instruct |
| 微調方法 | DPO |
| Quantization | 4-bit NF4 |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| Learning Rate | 5e-5 |
| Epochs | 1 |
| Gradient Accumulation | 8 |
| Beta | 0.1 |
| Gradient Checkpointing | Enabled |
| Optimizer Steps | 45 |

使用 4-bit NF4 量化及 LoRA 降低 GPU 記憶體使用量，並透過 Gradient Checkpointing 降低訓練時的記憶體需求。

本次訓練共使用 360 組 Preference Pairs，完成 45 個 Optimizer Steps。

## 10. VLM 評估指標

使用 COCO 80 個物件類別作為評估標準，將模型預測結果與 COCO Annotation 比較。

採用以下指標：

- **Precision：** 模型預測的物件類別中，符合 COCO 標註的比例。
- **Recall：** COCO 標註的物件類別中，成功被模型辨識的比例。
- **F1-score：** Precision 與 Recall 的調和平均。
- **Hallucination Rate：** 預測類別中未出現在 COCO 標註中的比例。
- **Average Latency：** 每張圖片的平均推論時間。
- **Empty Prediction Rate：** 沒有成功解析出任何 COCO 類別的圖片比例。

其中：

\[
Precision = \frac{TP}{TP+FP}
\]

\[
Recall = \frac{TP}{TP+FN}
\]

\[
F1 = \frac{2 \times Precision \times Recall}{Precision+Recall}
\]

\[
Hallucination\ Rate = \frac{FP}{TP+FP}
\]

注意：此處的 Hallucination Rate 是依照 COCO Annotation 定義的未匹配預測比例。COCO 未標註某個物件，不代表該物件一定不存在於圖片中，因此不能將所有 FP 直接視為經人工確認的視覺幻覺。

## 11. VLM DPO 評估結果

### 11.1 Target Evaluation

使用 50 張獨立圖片比較 Base Model 與 DPO Model。

| 指標 | Base Model | DPO Model | 變化 |
|---|---:|---:|---:|
| Precision | 86.32% | **88.50%** | +2.18 pp |
| Recall | 45.56% | **55.56%** | +10.00 pp |
| F1-score | 59.64% | **68.26%** | +8.62 pp |
| Hallucination Rate | 13.68% | **11.50%** | -2.18 pp |
| Empty Prediction Rate | 4.00% | **2.00%** | -2.00 pp |
| Average Latency | 0.658s | 1.192s | +0.534s |

在 Target Evaluation 中，DPO Model 的 Precision、Recall 及 F1-score 均有所提升。

其中 Recall 增加 10 個百分點，F1-score 增加 8.62 個百分點，表示微調後模型能辨識出更多符合 COCO 標註的物件。

同時，依 COCO 標註計算的 Hallucination Rate 從 13.68% 降至 11.50%，顯示在目標測試集上，模型的預測與標註一致性有所改善。

### 11.2 General Evaluation

使用另外 30 張圖片，觀察模型在一般圖片上的表現。

| 指標 | Base Model | DPO Model | 變化 |
|---|---:|---:|---:|
| Precision | 88.10% | 81.97% | -6.13 pp |
| Recall | 41.57% | **56.18%** | +14.61 pp |
| F1-score | 56.49% | **66.67%** | +10.18 pp |
| Hallucination Rate | 11.90% | 18.03% | +6.13 pp |
| Empty Prediction Rate | 16.67% | **10.00%** | -6.67 pp |
| Average Latency | 0.655s | 1.275s | +0.620s |

在 General Evaluation 中，DPO Model 的 Recall 與 F1-score 仍然提升，但 Precision 下降，Hallucination Rate 上升。

這代表 DPO 微調提高了模型的物件辨識覆蓋率，但也可能使模型在一般圖片中產生更多未被 COCO 標註支持的預測。

因此，DPO 的改善並非在所有資料集上都能同時提高 Precision 與 Recall。

---

# Part 3. 影像品質與穩健性測試

## 12. 影像品質調整

為了觀察影像品質改變對模型辨識能力的影響，額外建立三種影像處理條件，並與原始圖片進行比較。

所有條件均使用相同的 50 張 Target Evaluation 圖片。

| 條件 | 說明 |
|---|---|
| Original | 原始圖片 |
| Dark (0.5) | 降低圖片亮度 |
| Resolution (0.25) | 將圖片解析度縮小至原始尺寸的 25% |
| Blur (2) | 使用 Gaussian Blur，設定 blur 強度為 2 |

## 13. 不同影像品質下的評估結果

### 13.1 F1-score

| Image Condition | Base | DPO | Improvement |
|---|---:|---:|---:|
| Original | 59.64% | **68.26%** | +8.62 pp |
| Dark (0.5) | 59.18% | **67.80%** | +8.62 pp |
| Resolution (0.25) | 51.97% | **58.91%** | +6.94 pp |
| Blur (2) | 49.81% | **55.15%** | +5.34 pp |

DPO Model 在四種影像條件下的 F1-score 均高於 Base Model。

即使圖片亮度降低、解析度下降或受到模糊處理，DPO Model 仍維持較高的整體辨識表現。

### 13.2 Recall

| Image Condition | Base | DPO | Improvement |
|---|---:|---:|---:|
| Original | 45.56% | **55.56%** | +10.00 pp |
| Dark (0.5) | 43.89% | **55.56%** | +11.67 pp |
| Resolution (0.25) | 36.67% | **45.00%** | +8.33 pp |
| Blur (2) | 35.56% | **41.67%** | +6.11 pp |

在四種影像條件下，DPO Model 的 Recall 均有所提升，表示模型能找出更多標註中的物件。

### 13.3 Precision 與 Hallucination Rate

| Image Condition | Base Precision | DPO Precision | DPO Hallucination Rate |
|---|---:|---:|---:|
| Original | 86.32% | **88.50%** | 11.50% |
| Dark (0.5) | **90.80%** | 86.96% | 13.04% |
| Resolution (0.25) | **89.19%** | 85.26% | 14.74% |
| Blur (2) | **83.12%** | 81.52% | 18.48% |

雖然 DPO Model 在所有影像條件下都有較高的 F1-score 與 Recall，但在 Dark、Resolution 及 Blur 條件下，Precision 均低於 Base Model。

其中 Blur 條件下的 Hallucination Rate 最高，達到 18.48%。

這顯示模型在影像品質下降時，雖然能維持較好的物件辨識覆蓋率，但仍可能產生更多未被 COCO 標註支持的預測。

## 14. 推論效能比較

| Image Condition | Base Latency | DPO Latency |
|---|---:|---:|
| Original | 0.658s | 1.192s |
| Dark (0.5) | 0.618s | 1.278s |
| Resolution (0.25) | 0.380s | 0.896s |
| Blur (2) | 0.453s | 1.011s |

DPO Model 的推論時間高於 Base Model，表示本次微調在提升辨識能力的同時，也增加了推論成本。

## 15. GPU 記憶體使用量

| Model | Peak GPU Memory |
|---|---:|
| Base Model | 3.496 GB |
| DPO Model | 2.759 GB |

以上為本次評估程序記錄的推論峰值 GPU 記憶體，實際數值會受到模型載入方式、量化設定及執行環境影響。

---

# Part 4. 結論

## 16. 實驗總結

### 基礎實驗：LLM DPO

透過 120 組 Preference Pairs 對 Qwen2.5-1.5B-Instruct 進行 DPO 微調。

Target Evaluation 的整體準確率由 65% 提升至 75%，Stated Accuracy 由 80% 提升至 100%。

結果顯示 DPO 能改善模型對明確政策資訊的遵循能力。不過 Correct Abstention Rate 仍為 50%，代表模型面對資訊不足的問題時，仍有改善空間。

通用 Benchmark 則用來觀察微調是否影響原有能力，實驗結果顯示通用表現沒有明顯提升。

### 進階實驗：VLM DPO

透過 360 組 Preference Pairs 對 Qwen2.5-VL-3B-Instruct 進行 DPO 微調。

在 Target Evaluation 中：

- Precision 提升 2.18 個百分點。
- Recall 提升 10.00 個百分點。
- F1-score 提升 8.62 個百分點。
- 依 COCO 標註計算的 Hallucination Rate 降低 2.18 個百分點。

在 General Evaluation 中，Recall 與 F1-score 也有提升，但 Precision 下降，表示模型的物件辨識覆蓋率提高時，可能伴隨更多未匹配預測。

### 影像品質實驗

在 Original、Dark、Resolution 及 Blur 四種影像條件下，DPO Model 的 F1-score 均高於 Base Model。

其中：

- Original：F1 提升 8.62 個百分點。
- Dark：F1 提升 8.62 個百分點。
- Resolution：F1 提升 6.94 個百分點。
- Blur：F1 提升 5.34 個百分點。

結果顯示 DPO 微調後的模型在不同影像品質條件下，均具有較好的整體物件辨識表現。

不過，影像品質下降時，DPO Model 的 Precision 並未同步提升，且推論時間較長。因此，仍需要在辨識覆蓋率、預測精確度及推論效率之間取得平衡。

