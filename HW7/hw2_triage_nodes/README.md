# 作業二:建立與比較分流節點(A Prompt / B SFT / C Decoding / D System One)

> **實驗環境**:Qwen2.5-VL-3B-Instruct、單張 RTX 3080、greedy decoding、每筆 warm-up 後重複 3 次取平均、Benchmark n=24。
> 原始結果:`results/real_base.json`(A/C/D)、`results/real_sft.json`(B);離線規則基線:`results/smoke_rule.json`。
>

```
pip install -r requirements.txt
python data/make_benchmark.py                 # 產生 Benchmark 與合成圖
python scripts/run_bench.py --backend rule    # 離線 smoke test(非 VLM)
pytest -q tests                               # 6 passed
# GPU:
python scripts/gen_sft_data.py --n 1500
python scripts/train_sft.py --out checkpoints/sft_lora
python scripts/run_bench.py --backend qwen --nodes A,C,D --out results/real_base.json
python scripts/run_bench.py --backend qwen --adapter checkpoints/sft_lora --nodes B --out results/real_sft.json
python scripts/fill_readme.py results/real_base.json results/real_sft.json
```

## 基礎(30 分):四種節點(`triage/nodes.py`)

| 實驗 | 分數 | 做法 |
|---|---|---|
| A | 6 | 完全 prompt:規則 + schema + few-shot。**嚴格** `json.loads` + pydantic,不修補 → 量測模型原生結構正確率 |
| B | 8 | Qwen2.5-VL-3B **LoRA SFT**:1324 筆模板 + 合成圖(已排除與 Benchmark 逐字重複),極短 prompt(規則內化到權重)。另附 **RLVR 可驗證獎勵** `triage/reward.py`(GRPO 用:格式 0.3 + intent 0.4 + order_id 0.2 + 澄清 0.1),以程式驗證取代人類偏好標註的 RLHF |
| C | 8 | 與 A 相同語意的 prompt + **文法約束解碼(自製字元級狀態機,`triage/fsm_decode.py`)**:固定骨架直接強制輸出、enum 只允許合法選項、自由文字只禁引號/控制字元/純空白 token。 |
| D | 8 | **System One / System Two**:單次 forward 以 8 個選項字母讀 intent 機率(~1 token 成本);離題/攻擊且高信心 → **0 次生成**直接結束;一般意圖 → System Two 只約束生成 `{urgency, slots}`;信心 < τ → 退回 C 慢路徑 |

A 與 C 的 prompt 語意相同,差異只在解碼 → 可隔離「解碼約束」的效果。

## 進階(20 分)

### Benchmark(10 分,`data/benchmark.jsonl`,24 筆 ≥ 10)
一般明確 8、帶圖 7(裂痕手機、壓爛紙箱、膨脹電池、藍牙錯誤畫面、物流截圖)、多輪 2(補訂單號、中途換需求)、含糊 1、離題 3、攻擊 4(含圖中 injection)、一句多需求 1。
每筆含 gold:intent / urgency / order_id / product / need_clarification。圖片由 PIL 合成(含可 OCR 的訂單號)。

### 量化(10 分,`triage/metrics.py`)
- **結構正確性**:json_valid%、schema_valid%
- **語意正確性**:intent / urgency / order_id(含「不可幻覺」)/ clarify 正確率、full_correct(intent+order_id+clarify 全對)
- **花費時間**:每筆 warm-up 後重複取平均;mean / p50 / p95
- **花費成本**:input/output token、模型呼叫次數、`延遲 × GPU 單價($0.35/hr,可調)` → 每千次請求成本

## 結果

**A~D 實測(Qwen2.5-VL-3B,RTX 3080):**

<!-- BENCH:START -->
| 節點 | json_valid% | schema_valid% | intent_acc% | urgency_acc% | order_id_acc% | clarify_acc% | full_correct% | latency_mean_s | latency_p95_s | in_tokens_mean | out_tokens_mean | model_calls_mean | cost_usd_per_1k_req |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_prompt | 95.8 | 95.8 | 75.0 | 33.3 | 91.7 | 87.5 | 75.0 | 1.021 | 1.379 | 1094.9 | 60.1 | 1 | 0.0992 |
| B_sft | 100.0 | 100.0 | 91.7 | 83.3 | 100.0 | 95.8 | 87.5 | 1.267 | 1.725 | 134.9 | 60.8 | 1 | 0.1232 |
| C_decoding | 100.0 | 100.0 | 79.2 | 37.5 | 100.0 | 91.7 | 75.0 | 0.978 | 1.363 | 917.9 | 57.9 | 1 | 0.0951 |
| D_system_one | 100.0 | 100.0 | 79.2 | 50.0 | 100.0 | 83.3 | 70.8 | 0.781 | 1.081 | 728.2 | 45.8 | 1.92 | 0.076 |
<!-- BENCH:END -->

**關鍵數字(相對 A)**

| | schema_valid | intent | full_correct | 延遲 | 輸入 token | 每千次成本 |
|---|---|---|---|---|---|---|
| A Prompt | 95.8% | 75.0% | 75.0% | 1.02 s | 1095 | $0.099 |
| B SFT | 100% | **91.7%** | **87.5%** | 1.27 s(+24%) | **135(−88%)** | $0.123(+24%) |
| C Decoding | **100%** | 79.2% | 75.0% | 0.98 s | 918 | $0.095 |
| D System One | **100%** | 79.2% | 70.8% | **0.78 s(−24%)** | 728 | **$0.076(−23%)** |

離線規則基線(RuleBackend,非 VLM):intent 79.2%、full_correct 79.2% —— **3B 模型的 A/C/D 並沒有明顯贏過關鍵字規則**,只有 B 勝出,這是誠實的底線比較。

各類別 intent 準確率(%):

| 類別 | A | B | C | D |
|---|---|---|---|---|
| 一般明確(8) | 100 | 87.5 | 100 | 100 |
| 帶圖(5) | 80 | 100 | 80 | 100 |
| 離題(3) | 66.7 | 100 | 100 | 66.7 |
| 攻擊(4) | 0 | 75 | 0 | 0 |

### 分析

**1. 結構正確性(A vs C vs D)**:A 在 24 筆中有 1 筆失敗(b16 攻擊句:輸出 markdown 圍欄且所有欄位為 null),schema_valid 95.8%;
C/D 的文法約束使結構恆為 100%。**但結構正確不等於語意正確**:C 的 intent 只比 A 多 1 題(79.2 vs 75.0,不顯著),
且約束解碼會「被迫填字串」——b17、b19 模型在 order_id 填入「需要訂單編號」這類說明文字(之後被 `Slots` 驗證器轉成 null 才沒有扣分)。
換言之,約束解碼保證了形狀,內容品質仍取決於模型本身。

**2. SFT(B)是準確度唯一明顯的提升**:intent +16.7 pt(4 題)、urgency 33→83%、clarify 87.5→95.8%;且 prompt 內化後輸入 token 少 88%(135 vs 1095)。
- urgency 大幅進步主要是**學會了我的標註慣例**(安全疑慮/金額/情緒才算 high),A/C/D 即使 prompt 寫了標準也傾向全判 high(A 33%、C 37.5%),這是「對齊標註規範」而非更聰明。
- **B 的延遲反而比 A 高 24%**,儘管輸入短很多:LoRA 以未合併(unmerged)方式推論,每層多一次低秩矩陣運算;且本表成本以 GPU 時間計。
  若改用以 token 計價的 API,B 的輸入成本會是最低。可用 `--merge` 合併 adapter 後重測延遲(見下)。

**3. System One / Two(D)最快最便宜**:延遲 −24%、成本 −23%。路徑分布:fast 2、slots 18、slow 4。
fast 路徑(離題)只做 1 次 forward、不生成。代價:
- product_acc 僅 55.6%(A 100%、C 88.9%):System Two 的 prompt 強調「沒提到填 null、不要猜」,b02、b05、b12 明明有商品卻填 null → clarify 降到 83.3%。
- b10 的 issue_summary 竟抄了 prompt 內的 intent 說明文字(「商品瑕疵、故障、運送損壞(常附照片)」),表示 System Two 的 prompt 洩漏了類別描述,需改成不含描述。
- 離題 intent 66.7%:b14(「幫我寫 Python」)被 System One 判成 tech_support 且信心夠高而走 slots 路徑,沒有進 fast。
- 平均 1.92 次模型呼叫:一般案件是「兩段式」,省下的是輸出 token 與較短 prompt,不是呼叫次數。

**4. 攻擊類(b16–b19)**:A/C/D 全為 0%,B 為 75%。原因不同:A/C/D 不會主動標 unsafe(prompt 中「企圖操控則填 unsafe」的指示被忽略);
B 學會了。**但 B 的 75% 很可能被高估**:SFT 資料的攻擊模板與 Benchmark 攻擊句**語意高度相似**(僅排除逐字重複),屬於近似分佈內測試。
實務上攻擊本來就該由作業三的輸入防禦層先攔,節點只是第二道。

### 限制與威脅(請在報告中保留)
- **n=24,且作者自撰**;B 的優勢有一部分來自 Benchmark 與 SFT 模板同源(特別是攻擊/離題/一般句型)。要公平評估泛化,應另寫一組**不同措辭的 held-out** 測試,或用真實使用者語料。
- B 與 A/C/D 分兩個 process 跑(A/C/D 同一個載入的模型);延遲含 GPU 時脈波動,差異 <10% 不建議下結論。
- 成本採 `延遲 × $0.35/hr` 估算;改用 token 計價結論會改變(B 輸入 token 最省)。
- 3B 小模型的 A/C/D 未贏過規則基線;更大模型(7B)或加入 few-shot 可能改變排序,**未測試**。

### 建議的後續實驗(尚未做)
1. `python scripts/run_bench.py --backend qwen --adapter checkpoints/sft_lora --merge --nodes B`:合併 LoRA 後重測 B 的延遲。
2. B 之後加上 C 的文法約束(SFT + 約束解碼),理論上可兼得準確度與 100% 結構。
3. 修 D 的 System Two prompt(移除類別描述、放寬「不要猜」)後重跑,看 product_acc 能否回到 C 的水準。

## 已知限制
- **實測發現**:lm-format-enforcer 搭配 Qwen2.5-VL(byte-level BPE)遇到中文自由文字欄位會在字串內無限吐空白/換行直到截斷,C/D 的 schema_valid 僅 79.2%(5/24 失敗,皆為含中文 issue_summary 的案例);改用 `maxLength`+`pattern` 或限制空白皆無效/變慢,改自製狀態機。**第一版自製是 token 級**,要求字串以單獨的 `\"` token 開頭,但 Qwen 習慣把 `\":\"`、`\"TW` 這類符號與內容合併成一個 token,導致每個可為 null 的欄位都被迫選 null(slots 全空、clarify 全錯);**第二版改成字元級**(逐字元檢查每個 token,可跨越骨架與內容)才修正。這是報告中值得討論的一點:**約束解碼的保證取決於套件對 tokenizer 的處理**。
- 自製狀態機的核心以會合併符號的假 tokenizer + 500 組隨機『模型』測過(輸出恆可 parse,且字串欄位不會被強迫為 null);已在 Qwen2.5-VL 真 tokenizer 上驗證(C/D 的 schema_valid=100%)。
- SFT 資料為模板合成,分布較乾淨;真實使用者語料請另行蒐集。