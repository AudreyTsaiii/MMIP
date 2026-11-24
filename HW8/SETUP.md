# MMIP 第十週作業：多輪對話 RAG、VideoRAG、RAG 評估、GraphRAG

LLM、Embedding 與多模態理解都使用 **Google Gemini API**（免費額度就能跑）。沒有設定 key 時會自動退回離線模式（TF-IDF + 抽取式回答），只能用來確認流程跑得通，數字沒有參考價值。

| 作業 | 內容 | 主要程式 |
|---|---|---|
| 一 | 多輪對話 RAG：自製「數位影像處理」知識庫與 QA、Chunk/Overlap/Top-K 穩定度掃描、5 種 Prompting 比較、Gradio UI | `common/rag.py`、`hw1/` |
| 二 | 多輪對話 VideoRAG：影片檢索 → K-means 關鍵影格 → 影格選取 → 多模態生成 | `hw2/videorag.py` |
| 三 | 用 RAGAS 評估作業一；設計 VideoRAG 的四層評估 | `common/ragas_metrics.py`、`hw3/` |
| 四 | Self-RAG 論文知識圖譜、24 題 Benchmark、GraphRAG（local/global）與 RAG 的比較 | `hw4/` |

## 1. 安裝（Windows / macOS / Linux 都可以，需要 Python 3.10 以上）

```bash
python -m venv .venv
# Windows：.venv\Scripts\activate    macOS/Linux：source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt   # 選用：CLIP 影格向量與本機 Whisper，可以略過
```

## 2. 設定 API key

把 `.env.example` 複製成 `.env`，填入 `GEMINI_API_KEY=...`。`.env` 已經列在 `.gitignore`，**不會被 push 到 GitHub**。

## 3. 錄作業二的影片

把自己錄的影片放進 `data/videos/`（mp4、mov、mkv、webm 都可以）：
- **有說話的影片** 2～3 支：例如講解一個影像處理概念，每支 1～3 分鐘。
- **沒說話的操作影片** 2～3 支：例如螢幕錄影操作修圖軟體、程式或網頁，每支 1～3 分鐘。

影片太大，預設不會進 repo。要繳交的話請上傳雲端硬碟，並在報告中附上連結。

## 4. 一鍵跑完全部實驗並產生報告

```bash
python run_all.py
```

依序執行：QA 擴充 → 參數掃描 → Prompt 比較 → 多輪展示 → RAGAS 評估 → GraphRAG 建圖與評估 → 影片建庫 → 影片出題 → 產生 `REPORT.md`。

- 第一次跑到影片時，Gemini 會先看完整影片並出題，寫到 `data/video_qa.json`。**請打開這個檔案檢查題目和時間點**，有錯就修正（這是報告裡的人工檢核步驟），然後執行 `python run_all.py --only video` 完成影片評估，最後再跑一次 `python make_report.py`。
- 影片還沒錄好的話，可以先跑 `python run_all.py --skip-video`。
- 免費額度有每分鐘請求上限，預設每分鐘只送 10 個請求，全部跑完大約 1～2 小時。所有呼叫都有快取（`.cache/`），中途因為額度用完而中斷，隔天重跑會從斷點接續。付費帳號可以在 `.env` 設定 `GEMINI_RPM=60` 來加速。

## 5. 開啟介面（作業一進階的 UI，也用來展示作業二、四）

```bash
python app.py        # 瀏覽器開 http://127.0.0.1:7860
```

請截三張圖放到 `docs/`，報告會引用：`ui_rag.png`（多輪對話＋Reference）、`ui_video.png`（影片問答＋影格）、`ui_graph.png`（GraphRAG）。

## 6. Push 到 GitHub

```bash
git init
git add .
git status            # 確認清單裡沒有 .env
git commit -m "MMIP week10 RAG homework"
git branch -M main
git remote add origin https://github.com/AudreyTsaiii/MMIP.git
git push -u origin main
```

## 檔案結構

```
data/kb_image_processing.md     作業一知識庫（20 篇）
data/qa_dataset.jsonl           手寫 120 組 QA（正式問句/口語問句/答案/原文證據句）
data/qa_generated.jsonl         5W1H 擴充 QA（run_all 產生）
data/multiturn_dialogs.json     10 段 × 3 輪多輪測試對話
data/selfrag_paper.md           作業四論文內容（放 data/selfrag.pdf 會改用完整 PDF）
data/graphrag_benchmark.json    作業四 24 題 Benchmark
data/video_qa.json              作業二/三影片題目（run_all 產生，需人工檢查）
common/backends.py              Gemini / 離線後端、快取、限速
common/rag.py                   多輪 RAG 核心（切塊、混合檢索、Query Rewriting、Prompt）
common/ragas_metrics.py         RAGAS 指標實作
hw1/  sweep.py  prompt_compare.py  demo_multiturn.py  augment_qa.py  app.py
hw2/  videorag.py
hw3/  eval_rag.py  eval_video.py
hw4/  graphrag.py  evaluate.py
app.py  run_all.py  make_report.py
results/                        所有實驗輸出（CSV、圖、逐題 JSONL）
REPORT.md                       自動產生的報告
```
