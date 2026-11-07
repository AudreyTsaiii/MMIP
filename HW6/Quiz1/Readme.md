# HW6 Quiz 1 — Large Language Model

## 1. Project Overview

本作業使用開源大型語言模型 **Qwen/Qwen3-1.7B**，在 NVIDIA RTX 3080 GPU 上進行部署與推論實驗。

實驗主要分為四個部分：

1. **LLM Deployment**：部署並執行開源 LLM。
2. **Instruction Benchmark**：設計 20 道自訂題目，評估模型在不同能力面向的表現。
3. **Prompt Engineering**：使用 CoT-style Prompting，並透過 Gemini 作為 LLM-as-a-Judge 評估 Prompt 優化效果。
4. **Model Quantization**：比較 FP16、INT8、INT4 三種數值精度的記憶體使用量、推論速度與回答品質，最後結合 INT4 與 CoT Prompt 進行進階實驗。

## 2. Environment

| 項目 | 設定 |
|---|---|
| Model | Qwen/Qwen3-1.7B |
| GPU | NVIDIA GeForce RTX 3080 |
| Framework | PyTorch |
| Transformers | 4.57.6 |
| PEFT | 0.17.1 |
| BitsAndBytes | 0.50.2 |
| Evaluation Model | Gemini 3.5 Flash Lite |
| Programming Language | Python 3.10.21 |

### Model Configuration

- Model: `Qwen/Qwen3-1.7B`
- Decoding Strategy: Greedy Decoding
- `do_sample=False`
- `max_new_tokens=384`
- `enable_thinking=False`
- Batch size: 1

所有實驗皆使用相同的 20 道 Benchmark 題目，並採用 Greedy Decoding，以降低生成隨機性對實驗比較的影響。

## 3. Instruction Benchmark

### 3.1 Benchmark Design

設計 20 道自訂 Instruction Benchmark，涵蓋五種不同能力，每個類別各四題。

| Category | Number of Questions | Evaluation Focus |
|---|---:|---|
| Knowledge | 4 | 知識理解與事實回答 |
| Reasoning | 4 | 邏輯推理與問題分析 |
| Instruction Following | 4 | 遵循指令與指定輸出格式 |
| Summarization | 4 | 文章理解與摘要能力 |
| Problem Solving | 4 | 問題解決與應用能力 |
| **Total** | **20** | |

### 3.2 Evaluation Method

使用 Gemini 3.5 Flash Lite 作為 LLM-as-a-Judge，對模型回答進行評估。

評分面向：

- **Correctness**：答案的正確性。
- **Completeness**：答案的完整性。
- **Instruction Following**：是否遵循題目要求。
- **Overall**：整體回答品質。

每個面向採用 1–5 分評分，並比較不同 Prompt 與模型精度的回答表現。

> LLM-as-a-Judge 的評分結果是模型輔助評估，不代表人工標註的絕對正確率。

## 4. Prompt Engineering

### 4.1 Baseline Prompt

Baseline 使用一般的 System Prompt，要求模型以繁體中文回答，並依照題目要求提供答案。

### 4.2 CoT-style Prompting

在 Baseline 的基礎上，加入分析與檢查的指示，引導模型在回答前先確認題目要求，並在涉及計算或推理時完成必要分析。

使用的 System Prompt：

```python
SYSTEM_PROMPT = (
    "你是一個可靠且專業的 AI 助理。"
    "請使用繁體中文回答，內容應正確、清楚並符合使用者的要求。"
    "若題目要求特定格式，請嚴格遵守。"
    "請仔細閱讀題目，先確認題目要求與限制。"
    "若題目涉及計算或推理，請先完成必要的分析與檢查。"
    "最後提供清楚、正確且符合題目格式的答案。"
    "請呈現簡潔的解題依據，避免不必要的冗長說明。"
)
```

本實驗設定 `enable_thinking=False`，因此屬於透過 Prompt 引導模型進行分析的 CoT-style Prompting，而非啟用 Qwen3 原生 Thinking Mode。

### 4.3 Evaluation Results

使用 Gemini 3.5 Flash Lite 比較 FP16 Baseline 與 FP16 CoT 的回答品質。

| Evaluation Metric | Baseline | CoT | Difference |
|---|---:|---:|---:|
| Correctness | 4.25 | 4.40 | +0.15 |
| Completeness | 3.95 | 4.35 | +0.40 |
| Instruction Following | 3.80 | 4.20 | +0.40 |
| **Overall** | **3.90** | **4.15** | **+0.25** |

比較結果：

- CoT 在四項整體評估指標中皆有提升。
- Completeness 與 Instruction Following 均提升 0.40 分。
- Overall 從 3.90 提升至 4.15。
- 20 題中，CoT 有 6 題勝出、Baseline 有 1 題勝出，另外 13 題評分相同。

各類別的 Overall 表現：

| Category | Baseline | CoT |
|---|---:|---:|
| Knowledge | 4.25 | 4.75 |
| Reasoning | 3.25 | 4.25 |
| Instruction Following | 4.75 | 4.25 |
| Summarization | 4.75 | 4.75 |
| Problem Solving | 2.50 | 2.75 |

其中 Reasoning 類別提升 1.00 分，顯示加入分析指示對推理題目有較明顯的幫助。不過 Instruction Following 類別下降 0.50 分，代表 Prompt 優化並非對所有能力面向都有一致的改善。

### 4.4 Inference Performance

| Metric | Baseline FP16 | CoT FP16 |
|---|---:|---:|
| Average Latency | 2.8677 s | 2.5816 s |
| Total Generation Time | 57.35 s | 51.63 s |
| Average GPU Memory | 3.2428 GB | 3.2480 GB |
| Peak GPU Memory | 3.2691 GB | 3.3097 GB |

CoT 的平均推論時間較 Baseline 低，但 GPU 記憶體使用量略微增加。

由於生成回答長度也會影響推論時間，因此不能單純將延遲下降解讀為 CoT 本身能加速推論。

## 5. Model Quantization

### 5.1 Quantization Methods

使用 BitsAndBytes 調整模型數值精度，並比較以下三種設定：

| Precision | Configuration |
|---|---|
| FP16 | Original half-precision model |
| INT8 | 8-bit quantization |
| INT4 | 4-bit NF4 quantization with double quantization |

INT4 使用 NF4（NormalFloat 4-bit）量化，並啟用 Double Quantization，以進一步降低量化常數所需的記憶體。

### 5.2 GPU Memory and Inference Performance

所有模型皆使用 Baseline Prompt，並執行相同的 20 道 Benchmark 題目。

| Metric | FP16 | INT8 | INT4 |
|---|---:|---:|---:|
| Model Memory After Loading | 3.2058 GB | 1.8954 GB | 1.2573 GB |
| Average GPU Memory | 3.2428 GB | 1.9286 GB | 1.2898 GB |
| Peak GPU Memory | 3.2691 GB | 2.0174 GB | 1.3206 GB |
| Average Latency | 2.8677 s | 12.0913 s | 3.8384 s |
| Total Generation Time | 57.35 s | 241.83 s | 76.77 s |

### 5.3 Memory Comparison

以 FP16 作為基準：

| Precision | Average GPU Memory | Memory Reduction |
|---|---:|---:|
| FP16 | 3.2428 GB | — |
| INT8 | 1.9286 GB | 40.53% |
| INT4 | 1.2898 GB | 60.22% |

INT4 相較於 FP16，平均 GPU 記憶體使用量降低約 60.22%，是三種設定中記憶體使用量最低的配置。

### 5.4 Inference Speed Comparison

INT8 與 INT4 的平均推論時間都高於 FP16：

- INT8 平均延遲約為 12.09 秒。
- INT4 平均延遲約為 3.84 秒。
- FP16 平均延遲約為 2.87 秒。

本次實驗中，量化雖然有效降低 GPU 記憶體使用量，但沒有帶來推論速度的提升。量化後的實際速度仍會受到 GPU 硬體、量化運算核心及推論實作方式影響。

### 5.5 Gemini Evaluation

使用 Gemini 3.5 Flash Lite 比較 FP16、INT8、INT4 在相同 20 道題目上的回答品質。

| Evaluation Metric | FP16 | INT8 | INT4 |
|---|---:|---:|---:|
| Correctness | 4.20 | 3.95 | 4.05 |
| Completeness | 3.95 | 3.70 | 3.75 |
| Instruction Following | 3.70 | 3.65 | 3.45 |
| **Overall** | **3.80** | **3.60** | **3.50** |

評估結果顯示：

- FP16 在四項整體評估指標中皆取得最高分。
- INT8 與 INT4 的 Overall 分數分別為 3.60 與 3.50。
- INT4 的 Correctness 為 4.05，略高於 INT8 的 3.95，但 Completeness 與 Instruction Following 較低。
- 20 題的比較結果中，FP16 勝出 5 題、INT8 勝出 2 題、INT4 勝出 4 題，另有 9 題評分相同。

整體而言，FP16 在回答品質與推論速度方面表現較佳；INT4 則具有明顯的記憶體優勢。

## 6. Advanced Experiment: INT4 + CoT

### 6.1 Experiment Design

為符合進階要求，將前述 CoT-style Prompting 與 INT4 量化模型結合。

實驗設定：

| Item | Configuration |
|---|---|
| Model | Qwen/Qwen3-1.7B |
| Precision | INT4 |
| Quantization | NF4 + Double Quantization |
| Prompt Strategy | CoT-style Prompting |
| Decoding | Greedy Decoding |
| `enable_thinking` | False |
| `max_new_tokens` | 384 |
| Benchmark | 20 Questions |

本實驗將 INT4 Baseline 與 INT4 + CoT 進行比較，觀察在低精度量化模型中加入優化 Prompt 後，對推論速度及 GPU 記憶體使用量的影響。

### 6.2 INT4 + CoT Results

| Metric | INT4 Baseline | INT4 + CoT |
|---|---:|---:|
| Model Memory After Loading | 1.2573 GB | 1.2573 GB |
| Average GPU Memory | 1.2898 GB | 1.2932 GB |
| Peak GPU Memory | 1.3206 GB | 1.3597 GB |
| Average Latency | 3.8384 s | 3.0848 s |
| Total Generation Time | 76.77 s | 61.70 s |

### 6.3 Comparison and Analysis

與 INT4 Baseline 相比：

- Average GPU Memory 從 1.2898 GB 增加至 1.2932 GB，增加約 0.0034 GB。
- Peak GPU Memory 從 1.3206 GB 增加至 1.3597 GB。
- Average Latency 從 3.8384 秒下降至 3.0848 秒，降低約 19.6%。

CoT Prompt 並未明顯增加平均 GPU 記憶體使用量，而本次實驗的平均推論時間也有所下降。

若與 FP16 Baseline 比較：

| Metric | FP16 Baseline | INT4 + CoT |
|---|---:|---:|
| Average GPU Memory | 3.2428 GB | 1.2932 GB |
| Peak GPU Memory | 3.2691 GB | 1.3597 GB |
| Average Latency | 2.8677 s | 3.0848 s |

INT4 + CoT 相較於 FP16 Baseline：

- 平均 GPU 記憶體降低約 60.1%。
- Peak GPU Memory 降低約 58.4%。
- 平均推論時間增加約 7.6%。

因此，INT4 + CoT 在本次實驗中能以接近 FP16 的推論時間，大幅降低 GPU 記憶體需求。

> 注意：以上結果是本次 20 題 Benchmark 的實驗觀察，並不代表 CoT Prompt 在所有情況下都能降低推論時間。不同回答長度也會影響延遲。

## 7. Overall Discussion

綜合所有實驗結果：

### 7.1 Prompt Engineering

加入 CoT-style Prompting 後，FP16 的 Overall 評分從 3.90 提升至 4.15，尤其在 Reasoning 類別有較明顯的改善。

這表示透過適當的 Prompt 設計，可以在不重新訓練模型的情況下，改善部分回答品質。

### 7.2 Quantization

量化能有效降低 GPU 記憶體使用量：

- INT8 平均 GPU 記憶體降低約 40.53%。
- INT4 平均 GPU 記憶體降低約 60.22%。

不過，量化不一定能提升推論速度或維持完全相同的回答品質。本次實驗中 FP16 的整體評分最高。

### 7.3 Best Configuration

不同需求下有不同的選擇：

| Goal | Recommended Configuration | Reason |
|---|---|---|
| Best Overall Quality | FP16 + CoT | CoT 評估結果較佳 |
| Lowest GPU Memory | INT4 | 平均 GPU 記憶體最低 |
| Combined Advanced Requirement | INT4 + CoT | 結合 Prompt Optimization 與低精度量化 |
| Fastest Measured Inference | FP16 + CoT | 本次實驗平均延遲最低 |

整體而言，FP16 + CoT 在本次實驗中具有較好的回答品質與推論速度；INT4 則適合 GPU 記憶體有限的情境。INT4 + CoT 則展示了將 Prompt Optimization 與 Model Quantization 結合的可行性。

## 8. GPU Memory Measurement

本作業使用 PyTorch 的 `torch.cuda.memory_allocated()` 監測 GPU 記憶體使用量。

- 模型載入完成後，記錄 Model Memory After Loading。
- 推論過程中以 0.05 秒為間隔取樣。
- 記錄每題推論期間的平均記憶體使用量及 Peak GPU Memory。
- 透過 20 題 Benchmark 的結果計算整體平均推論記憶體使用量。

本作業所呈現的 GPU Memory 為 PyTorch allocated GPU memory，並非整張 GPU 的總使用量或 CUDA reserved memory。

## 9. Project Structure

```text
Quiz1/
├── benchmark/
│   └── questions.jsonl
│
├── evaluation/
│   ├── evaluate_cot.py
│   └── evaluate_quantization.py
│
├── results/
│   ├── baseline_fp16.jsonl
│   ├── baseline_fp16_summary.json
│   ├── cot_fp16.jsonl
│   ├── cot_fp16_summary.json
│   ├── quant_int8.jsonl
│   ├── quant_int8_summary.json
│   ├── quant_int4.jsonl
│   ├── quant_int4_summary.json
│   ├── judge_results.jsonl
│   ├── judge_summary.json
│   ├── quant_judge_results.jsonl
│   ├── quant_judge_summary.json
│   ├── quant_int4_cot.jsonl
│   └── quant_int4_cot_summary.json
│
├── run_baseline.py
├── run_cot.py
├── run_quantization.py
└── run_int4_cot.py
```

## 10. How to Run

### 10.1 Activate Environment

```bash
source /home/yen/dl-work/.venv/bin/activate
cd /home/yen/dl-work/ImageProcessing/HW6
```

### 10.2 Run FP16 Baseline

```bash
python Quiz1/run_baseline.py
```

### 10.3 Run FP16 + CoT

```bash
python Quiz1/run_cot.py
```

### 10.4 Run INT8 Quantization

```bash
python Quiz1/run_quantization.py --bits 8
```

### 10.5 Run INT4 Quantization

```bash
python Quiz1/run_quantization.py --bits 4
```

### 10.6 Run INT4 + CoT

```bash
python Quiz1/run_int4_cot.py
```

### 10.7 Run Gemini Evaluation

```bash
python Quiz1/evaluation/evaluate_cot.py
python Quiz1/evaluation/evaluate_quantization.py
```

執行 Gemini 評估前，需先設定 `GEMINI_API_KEY` 環境變數。

## 11. Conclusion

成功部署並執行 Qwen3-1.7B 開源大型語言模型，設計涵蓋五種能力面向的 20 道 Instruction Benchmark，並完成 Prompt Engineering、模型量化與 GPU 記憶體分析。

實驗結果顯示，CoT-style Prompting 能改善部分回答品質；INT8 與 INT4 量化能有效降低 GPU 記憶體使用量，但可能伴隨推論速度或回答品質的變化。

最後透過 INT4 + CoT 實驗，成功將優化後的 Prompt 與低精度量化模型結合，並測量推論過程中的 GPU 記憶體使用量。

整體完成了基本與進階要求，並透過不同模型精度與 Prompt 策略的比較，分析模型回答品質、推論速度及 GPU 記憶體使用量之間的取捨。
