# HW6 Quiz2 — 視覺多模態大模型（Visual Multimodal Large Models）

## 1. Project Overview

本作業旨在比較兩個開源視覺多模態大型模型（Vision-Language Models, VLMs）在不同視覺理解任務上的表現。

本實驗選用以下兩個模型：

- **Qwen2.5-VL-3B-Instruct**
- **SmolVLM2-2.2B-Instruct**

自行設計 20 道視覺理解 Benchmark，涵蓋影像分類、物件辨識、視覺問答、圖片描述及屬性與空間關係理解。使用相同的 Prompt 與測試圖片進行推論，再透過規則式量化指標及商用 LLM（Gemini）評估兩個模型的表現。

## 2. Models

| Model | Hugging Face Model | Precision |
|---|---|---|
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | FP16 |
| SmolVLM2-2.2B | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | FP16 |

### Experimental Environment

| Item | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 |
| GPU Memory | 10 GB |
| Python | 3.10.21 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.57.6 |

## 3. Visual Understanding Benchmark

本實驗自行設計 20 道與視覺理解相關的題目，使用 COCO 2017 Validation Images 作為圖片來源，並搭配人工檢查的 Ground Truth。

共分成五種任務，每種類別各 4 題。

| Category | Questions | Description |
|---|---:|---|
| Image Classification | 4 | 根據圖片辨識主要場景或活動類別 |
| Object Recognition | 4 | 辨識圖片中的物件及其數量或組成 |
| Visual Question Answering | 4 | 根據圖片內容回答指定問題 |
| Image Captioning | 4 | 根據圖片生成自然語言描述 |
| Attribute & Spatial Understanding | 4 | 理解物件的顏色、屬性及空間關係 |
| **Total** | **20** | |

### Benchmark Design

- 每道題目均包含圖片、Prompt 及對應的 Ground Truth 或 Reference Captions。
- 除 Image Captioning 使用英文描述外，其餘題目以繁體中文設計。
- 兩個 VLM 使用相同的題目、圖片及 Prompt，確保比較條件一致。
- 所有題目的 Ground Truth 均經過人工檢查。

### Dataset Structure

```text
benchmark/
├── images/
├── annotations/
│   ├── captions_val2017.json
│   └── instances_val2017.json
├── questions.jsonl
├── review_benchmark.html
└── review_questions.py
```

## 4. Model Inference

兩個模型分別載入並執行 20 道 Benchmark，記錄模型載入時間、平均推論延遲及 GPU Memory 使用量。

### Inference Performance

| Metric | Qwen2.5-VL-3B | SmolVLM2-2.2B |
|---|---:|---:|
| Model Loading Time | 6.723 s | 6.857 s |
| Average Latency | **0.440 s** | 0.781 s |
| Average Peak GPU Memory | 7.033 GB | **4.606 GB** |
| Maximum Peak GPU Memory | 7.033 GB | 4.611 GB |

### Observation

- Qwen2.5-VL-3B 的平均推論延遲較低，推論速度較快。
- SmolVLM2-2.2B 使用較少的 GPU Memory，對硬體資源的需求較低。
- 兩個模型皆成功在 RTX 3080 上完成 20 道題目的推論。

## 5. Quantitative Evaluation

本實驗使用兩種評估方式：

1. Rule-based Evaluation：根據預先定義的 Ground Truth 與同義詞對照規則，自動評估模型答案。
2. Caption Evaluation：使用文字生成評估指標，比較生成描述與人工參考描述的相似程度。

### 5.1 Rule-based Evaluation

Image Captioning 題目不適合直接使用一般分類 Accuracy，因此此部分只計算其餘 16 題的 Accuracy。

| Category | Qwen2.5-VL-3B | SmolVLM2-2.2B |
|---|---:|---:|
| Image Classification | 75.00% | 50.00% |
| Object Recognition | 75.00% | 0.00% |
| Visual Question Answering | 75.00% | 25.00% |
| Attribute & Spatial Understanding | 75.00% | 25.00% |
| **Overall Accuracy** | **75.00% (12/16)** | **25.00% (4/16)** |

另外，Object Recognition 也計算物件層級的 Recall：

| Metric | Qwen2.5-VL-3B | SmolVLM2-2.2B |
|---|---:|---:|
| Object Recall | 85.71% (6/7) | 28.57% (2/7) |

**Evaluation Method:**

- 以人工確認的 Ground Truth 作為答案依據。
- 使用 Alias Mapping 處理中英文、繁簡中文及常見同義詞差異。
- Object Recognition 的題目正確率要求題目中的目標物件全部符合；Object Recall 則計算成功辨識的目標物件比例。
- Image Captioning 不納入上述 Accuracy 計算。

**Observation:**

Qwen2.5-VL-3B 在四個非圖片描述類別中皆取得較高的 Accuracy，整體 Accuracy 為 75%，高於 SmolVLM2-2.2B 的 25%。

### 5.2 Image Captioning Evaluation

針對 4 道 Image Captioning 題目，每張圖片皆提供 5 個 Reference Captions，並使用以下指標評估生成結果：

- **BLEU-1**：衡量 Unigram 的文字重疊程度。
- **BLEU-4**：衡量 1-gram 至 4-gram 的文字重疊程度。
- **METEOR**：考慮詞彙匹配及同義詞等因素。
- **ROUGE-L**：根據最長共同子序列（Longest Common Subsequence）衡量文字相似度。

| Metric | Qwen2.5-VL-3B | SmolVLM2-2.2B |
|---|---:|---:|
| BLEU-1 | 0.3070 | **0.4957** |
| BLEU-4 | 0.0685 | **0.1337** |
| METEOR | 0.4087 | **0.4864** |
| ROUGE-L | 0.2500 | **0.3937** |

**Observation:**

SmolVLM2-2.2B 在四項 Caption Evaluation 指標中皆取得較高分數，代表其生成描述與參考描述具有較高的文字重疊程度。

需要注意，Caption Metrics 主要衡量文字相似度，不完全代表描述內容的事實正確性或視覺理解能力。此外，本實驗採用每張圖片分別計算 BLEU，再取四張圖片的平均值，並非官方 COCO Corpus-level 評估結果。

## 6. Commercial LLM Evaluation — Gemini LLM-as-a-Judge

除了使用規則式量化指標，本實驗進一步使用 Google Gemini API 作為 LLM-as-a-Judge，評估兩個 VLM 的生成答案。

**Judge Model:** `gemini-3.5-flash-lite`

### Evaluation Method

1. 將兩個 VLM 對同一道題目的答案匿名標記為 A 與 B。
2. 隨機決定兩個模型的 A/B 順序，降低固定答案位置可能造成的偏差。
3. 提供原始圖片、題目、Ground Truth 或 Reference Captions，以及兩個模型的答案。
4. Gemini 分別對兩個答案進行 1–5 分評分。
5. 評估面向包括：
   - Correctness：答案正確性。
   - Visual Grounding：答案是否有圖片內容作為依據。
   - Completeness：答案是否完整。
   - Overall：整體表現。
6. Gemini 同時判斷兩個答案的偏好，結果分為 Qwen、SmolVLM2 或 Tie。

### Evaluation Results

| Metric | Qwen2.5-VL-3B | SmolVLM2-2.2B |
|---|---:|---:|
| Overall | **4.20** | 3.05 |
| Correctness | **4.25** | 3.05 |
| Visual Grounding | **4.25** | 3.20 |
| Completeness | **4.40** | 3.05 |

### Model Preference

| Preference | Questions | Percentage |
|---|---:|---:|
| Qwen2.5-VL-3B | 10 | 50% |
| SmolVLM2-2.2B | 6 | 30% |
| Tie | 4 | 20% |
| **Total** | **20** | **100%** |

**Observation:**

Gemini 在四項評分指標中皆給予 Qwen2.5-VL-3B 較高的平均分數，並在 20 道題目中有 10 題偏好 Qwen。

此結果與 Rule-based Evaluation 的整體趨勢一致，顯示 Qwen 在本次 Benchmark 的非圖片描述任務中具有較好的答案正確性、視覺依據及完整性。

## 7. Overall Comparison

整合推論效能、規則式量化指標、Caption Metrics 及 Gemini 評估結果：

| Evaluation | Better Model |
|---|---|
| Average Latency | Qwen2.5-VL-3B |
| GPU Memory Usage | SmolVLM2-2.2B |
| Rule-based Accuracy | Qwen2.5-VL-3B |
| Object Recall | Qwen2.5-VL-3B |
| BLEU-1 | SmolVLM2-2.2B |
| BLEU-4 | SmolVLM2-2.2B |
| METEOR | SmolVLM2-2.2B |
| ROUGE-L | SmolVLM2-2.2B |
| Gemini Overall | Qwen2.5-VL-3B |

### Discussion

根據本次實驗結果：

- **Qwen2.5-VL-3B-Instruct** 在影像分類、物件辨識、視覺問答及屬性與空間理解等任務中取得較佳的整體表現，並具有較低的推論延遲。
- **SmolVLM2-2.2B-Instruct** 在 Image Captioning 的文字重疊指標上表現較佳，且 GPU Memory 使用量較低。
- Gemini LLM-as-a-Judge 的結果與 Rule-based Evaluation 大致一致，進一步提供不同角度的模型比較。

整體而言，Qwen2.5-VL-3B 在本次 Benchmark 的多數視覺理解任務中表現較好，而 SmolVLM2-2.2B 在圖片描述及資源使用效率方面具有優勢。

## 8. Limitations

本實驗仍有以下限制：

1. **Benchmark Size：** 本次僅設計 20 道題目，每個類別只有 4 題，樣本數有限，結果不宜直接推廣至所有視覺理解任務。
2. **Rule-based Evaluation：** 以人工定義的 Alias Mapping 進行答案比對，可能無法涵蓋所有語意相同但文字不同的答案。
3. **Caption Metrics：** BLEU、METEOR 及 ROUGE-L 主要衡量文字重疊程度，無法完整評估圖片描述的語意品質及事實正確性。
4. **LLM-as-a-Judge：** Gemini 的評分可能受到模型本身判斷方式影響，因此不能視為絕對客觀的 Ground Truth。
5. **Benchmark Distribution：** 測試圖片及題目類型有限，無法代表所有真實世界的視覺場景。

## 9. Project Structure

```text
Quiz2/
├── benchmark/
│   ├── images/
│   ├── annotations/
│   ├── questions.jsonl
│   ├── review_benchmark.html
│   └── review_questions.py
│
├── scripts/
│   ├── test_qwen_vl.py
│   ├── test_smolvlm.py
│   └── run_benchmark.py
│
├── results/
│   ├── qwen2.5-vl/
│   │   └── predictions.jsonl
│   ├── smolvlm2/
│   │   └── predictions.jsonl
│   └── comparison/
│
├── evaluation/
│   ├── evaluate.py
│   ├── evaluate_caption.py
│   ├── evaluate_gemini.py
│   ├── metrics_summary.json
│   ├── question_evaluation.csv
│   ├── comparison.csv
│   ├── caption_evaluation.csv
│   ├── caption_comparison.csv
│   ├── caption_metrics.json
│   ├── gemini_judge_per_question.csv
│   ├── gemini_judge_comparison.csv
│   └── gemini_judge_results.jsonl
│
└── update_benchmark.py
```

## 10. How to Run

### 10.1 Environment Setup

```bash
source /home/yen/dl-work/.venv/bin/activate
```

### 10.2 Run Both VLMs

```bash
python scripts/run_benchmark.py
```

### 10.3 Rule-based Evaluation

```bash
python evaluation/evaluate.py
```

### 10.4 Caption Evaluation

```bash
python evaluation/evaluate_caption.py
```

### 10.5 Gemini LLM-as-a-Judge

先設定 Gemini API Key：

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

執行評估：

```bash
python evaluation/evaluate_gemini.py
```

Gemini 評估結果支援逐題保存及中斷後續跑。

## 11. Conclusion

成功部署並比較兩個開源視覺多模態大型模型，並自行設計 20 道視覺理解 Benchmark，涵蓋五種不同任務。

實驗結果顯示，Qwen2.5-VL-3B 在非圖片描述任務的 Accuracy 達到 75%，且 Gemini 整體評分為 4.20/5；SmolVLM2-2.2B 則在 Caption Evaluation 的四項指標中皆取得較高分數，同時具有較低的 GPU Memory 使用量。

透過相同 Prompt、量化評估指標及商用 LLM-as-a-Judge，從不同面向比較兩個 VLM 的視覺理解能力與推論效能，完成本次視覺多模態大模型的實驗要求。