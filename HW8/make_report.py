"""讀取 results/ 的實驗結果，自動產生 REPORT.md（重跑實驗後再執行一次即可更新數字）。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common.backends import ROOT
from common.rag import PROMPTS, load_docs, load_index_qa, load_qa

R = ROOT / "results"


def md(df: pd.DataFrame, floatfmt=".3f") -> str:
    try:
        return df.to_markdown(floatfmt=floatfmt)
    except Exception:
        return "```\n" + df.to_string() + "\n```"


def rd(p):
    p = R / p
    return pd.read_csv(p, index_col=0) if p.exists() else None


def shot(name, label):
    p = ROOT / "docs" / f"{name}.png"
    return f"![{label}](docs/{name}.png)\n" if p.exists() else f"> 📸 請把 {label} 截圖存成 `docs/{name}.png` 後重跑 make_report.py\n"


def size_curve(full: pd.DataFrame) -> str:
    g = full.groupby("chunk_size")["stability"].max()
    best = int(g.idxmax())
    lo, hi = int(g.index.min()), int(g.index.max())
    txt = "- 各 Chunk Size 的最佳穩定分數：" + "、".join(f"{k} 字 {v:.3f}" for k, v in g.items()) + "。"
    if best not in (lo, hi):
        txt += (f"最佳落在中間的 {best} 字：太小（{lo} 字）會把定義與條件切開，口語問句最不穩定；"
                f"太大（{hi} 字）會跨主題，在固定預算下只能放 1~2 塊，兩端都比中間差。")
    elif best == lo:
        txt += f"最佳為最小的 {lo} 字，表示本知識庫的答案多為短句，細切能讓檢索更精準。"
    else:
        txt += f"最佳為最大的 {hi} 字，表示答案常需要較長上下文。"
    return txt + "\n"


def section_hw1(out):
    docs, qa, iqa = load_docs(), load_qa(), load_index_qa()
    cfg = json.loads((R / "hw1_best_config.json").read_text(encoding="utf-8")) if (R / "hw1_best_config.json").exists() else {}
    gen = len(iqa) - len(qa)
    from common.rag import build_units

    n_chunks = len([u for u in build_units(docs, [], cfg.get("chunk_size", 400), cfg.get("overlap", 0)) if u.kind == "chunk"])
    n_emb = n_chunks + 2 * len(iqa)
    out += ["## 作業一：多輪對話 RAG\n",
            "### 1-1 自製主題 QA Dataset（5 分）\n",
            f"- 主題：**數位影像處理**（與本課程 MMIP 相關），自行撰寫 {len(docs)} 篇講義式知識文件（`data/kb_image_processing.md`，"
            f"共 {sum(len(d['text']) for d in docs.values()):,} 字），涵蓋取樣量化、色彩空間、濾波、邊緣偵測、頻率域、形態學、分割、特徵、"
            "壓縮、復原、CNN、物件偵測、語意分割、ViT/CLIP、資料增強、品質評估。",
            f"- 手寫 QA：**{len(qa)} 組**（`data/qa_dataset.jsonl`），每組包含 正式問句、口語化問句、標準答案、"
            "**原文證據句**（程式自動檢查必為原文子字串）。",
            f"- 5W1H 擴充 QA：**{gen} 組**（`data/qa_generated.jsonl`，依講義第 12 頁建議由 Gemini 以 5W1H 角度產生，"
            "證據句不在原文者自動丟棄）。" if gen else "- 5W1H 擴充 QA：設定 API key 後執行 `python -m hw1.augment_qa` 產生。",
            f"- **索引內 Embedding 總數 = {n_emb}**（文件 Chunk {n_chunks} + QA 問題向量 {len(iqa)} + QA 答案向量 {len(iqa)}），"
            f"超過門檻 200。Embedding 模型：`{cfg.get('embedder', '?')}`。",
            "- 檢索單位帶有座標資訊（文件編號、起訖字元位置、是否跨文件），回答時可精確指出 Reference。\n",
            "範例：\n", "| 編號 | 正式問句 | 口語化問句 | 標準答案 |", "|---|---|---|---|"]
    for q in qa[:: len(qa) // 5][:5]:
        out.append(f"| {q['id']} | {q['question']} | {q['question_colloquial']} | {q['answer']} |")
    demo = R / "hw1_multiturn_demo.md"
    out += ["\n### 1-2 多輪對話 Context 經過 RAG 得到合理回覆與 Reference（10 分）\n",
            "流程（講義第 8、9 頁兩階段）：\n",
            "```\n使用者追問 ──► Query Rewriting（LLM 依最近 3 輪對話補齊代名詞/省略的主詞）\n"
            "           ──► 混合檢索：Dense Embedding + BM25（字元 n-gram）以 RRF 融合，QA 以 qa_id 去重\n"
            "           ──► Top-K 參考資料（Chunk 與 QA，皆附出處座標）\n"
            "           ──► System Prompt（角色、只能依參考資料、每句標 [編號]）+ 先前對話 + 參考資料 + 問題 ──► LLM\n"
            "           ──► 回答 + Reference（UI 標示哪些 Reference 被實際引用）\n```\n"]
    if demo.exists():
        t = demo.read_text(encoding="utf-8")
        hit = [l for l in t.splitlines() if "命中率" in l]
        out.append(f"10 段 × 3 輪測試對話（第 2、3 輪都含「它、那、其中」等指涉）完整紀錄見 `results/hw1_multiturn_demo.md`。{hit[0].lstrip('- ') if hit else ''}\n")
        body = t.split("## 對話 M01")[1].split("## 對話 M02")[0] if "## 對話 M01" in t else ""
        out += ["節錄（對話 M01）：\n", body.strip(), ""]
    sweep = sorted(R.glob("hw1_sweep_*.csv"))
    sweep = [p for p in sweep if "topk" not in p.name and "perdoc" not in p.name]
    out += ["### 1-3 最穩定的 Chunk Size、Overlap、Top-K（5 分）\n",
            "- 評分採講義第 10 頁的「繁中檢索能力評測」：正確段落排第 1 名得 1 分、第 2 名 1/2 分 … 前五名沒有得 0 分（MRR@5）。"
            "「正確段落」＝Chunk 覆蓋標準證據句 ≥80%。",
            "- 只看 MRR@5 會偏好越大塊越好（大 Chunk 一定含答案，但送進 LLM 的雜訊與 token 也最多），因此再加上 **固定 800 字的 Context 預算**："
            "K_budget = 800 / 平均 Chunk 長度，只計前 K_budget 名（MRR@budget）。",
            "- 「穩定」：同一組參數在 **正式問句、口語問句、多輪追問（改寫後）** 三種提問風格下的平均要高、標準差要小，"
            "**穩定分數 = mean − std**；另以 bootstrap 估 95% 信賴區間。",
            "- 知識庫以「連續講義」方式切塊（像切 PDF），所以大 Chunk 會跨越兩個主題而被稀釋。",
            f"- 掃描範圍：Chunk Size ∈ {{100,150,200,300,400,600}} 字 × Overlap ∈ {{0,10,25,50}}% × Top-K ∈ {{1..10}}。\n"]
    if sweep:
        df = pd.read_csv(sweep[-1]).head(8)
        cols = ["chunk_size", "overlap", "n_chunks", "K_budget", "MRRb_formal", "MRRb_colloquial", "MRRb_multiturn",
                "mean", "std", "stability", "ci95_low", "ci95_high"]
        out += [f"穩定分數前 8 名（Embedder：`{cfg.get('embedder')}`，完整結果 `{sweep[-1].name}`）：\n",
                df[cols].to_markdown(index=False, floatfmt=".3f"), "",
                f"![sweep]({'results/' + sweep[-1].with_suffix('.png').name})\n"]
        full = pd.read_csv(sweep[-1])
        b = full.iloc[0]
        gain = (full["MRR5_multiturn"] - full["MRR5_multiturn_no_rewrite"]).mean()
        out += [f"**結論：Chunk Size = {cfg.get('chunk_size')}、Overlap = {cfg.get('overlap')}、Top-K = {cfg.get('top_k')}。**",
                f"- 此設定 MRR@budget 平均 {b['mean']:.3f}、三種問法間標準差 {b['std']:.3f}，為所有組合中最穩定。",
                f"- Top-K 依 Recall@K 曲線取「達到 Recall@10 的 95% 的最小 K」：" +
                "、".join(f"K={x['K']}: {x['recall_at_K']:.3f}" for x in cfg.get("recall_curve", [])) + "。",
                (f"- 多輪追問若不做 Query Rewriting，MRR@5 平均下降 {gain:.3f}（所有參數組合平均），證明改寫對多輪 RAG 是必要的。"
                 if gain > 0 else f"- 多輪 Query Rewriting 對檢索 MRR@5 的平均影響為 {gain:+.3f}。"),
                size_curve(full)]
    pc = rd("hw1_prompt_compare.csv")
    out += ["### 1-4 最合適的 Prompting Technique（5 分）\n",
            "在相同檢索設定下比較 5 種 Prompt（`common/rag.py` 的 `PROMPTS`）："
            "**Plain**（只給資料，baseline）、**Grounded**（角色＋只能依資料＋引用格式）、**CoT**（先列證據再回答）、"
            "**Few-shot**（示範一個引用式回答）、**Step-Back**（講義第 16~20 頁，先抽象出背景問題並一起檢索）。"
            "以作業三的 RAGAS 評審打分，綜合分數 = Faithfulness、Answer Relevancy、Answer Correctness 平均"
            "（Context 類指標由檢索決定，五者相同）。\n"]
    if pc is not None:
        keep = [c for c in ["faithfulness", "answer_relevancy", "answer_correctness", "multiturn_correctness",
                            "avg_answer_chars", "score", "judge"] if c in pc]
        out += [md(pc[keep]), "", f"**結論：最合適的是 `{cfg.get('technique', pc.index[0])}`。**"
                if pc["judge"].iloc[0] == "llm" else "（離線模式數字無意義，請設定 API key 後重跑。）", ""]
    out += ["### 1-5 User Interface（5 分）\n",
            "`python app.py` 啟動 Gradio 介面（第一個分頁）：",
            "- 左側聊天室支援多輪對話；右側即時顯示 **多輪改寫後的查詢**、每筆 **Reference 卡片**（出處座標、檢索分數、是否被回答引用）。",
            "- 可即時切換 Prompting Technique、Top-K、是否啟用多輪改寫，一鍵清除對話；頁首顯示目前模型與索引 Embedding 數。",
            "- 同一介面另有 VideoRAG（顯示選出的影格畫廊與時間戳）與 GraphRAG（顯示路由、種子實體與子圖）分頁。\n",
            shot("ui_rag", "作業一 UI")]


def section_hw2(out):
    cat = R / "video_store" / "catalog.json"
    out += ["## 作業二：多輪對話 VideoRAG\n",
            "### 2-1 自錄 Video Corpus（5 分）\n"]
    if cat.exists():
        c = json.loads(cat.read_text(encoding="utf-8"))
        out += ["| 影片 | 長度 (秒) | 是否有說話 |", "|---|---|---|"] + \
               [f"| {x['file']} | {x['duration']:.0f} | {'有（講解）' if x['has_speech'] else '沒有（操作）'} |" for x in c] + [""]
    else:
        out.append("（請把自錄影片放進 `data/videos/` 後執行 `python -m hw2.videorag --ingest`）\n")
    out += ["### 2-2 參考 VideoRAG 實作多輪對話聊天機器人（10 分）\n",
            "依 Jeong et al., *VideoRAG: Retrieval-Augmented Generation over Video Corpus*（Findings of ACL 2025，講義第 56~63 頁）改編：\n",
            "| 論文步驟 | 本作業做法（`hw2/videorag.py`） |", "|---|---|",
            "| 影片表示 | 1 fps 抽影格；語音轉逐字稿（本機 Whisper 或 Gemini 音訊理解，靜音影片自動標記為 silent） |",
            "| Frame Selection（CLIP → K-means → 代表影格） | 影格 → CLIP 向量（無 CLIP 時用色彩直方圖+縮圖）→ K-means，每分鐘約 8 群，取最接近群中心的影格；"
            "再由 Gemini 為關鍵影格寫畫面描述（含畫面文字、操作） |",
            "| Action 1：找到相關影片 | 影片切成 20 秒片段（逐字稿＋畫面描述）做 Embedding；以片段最高分代表影片分數排序影片 |",
            "| Action 2：選出最相關影格（論文以 MLP 對 40 組隨機組合打分） | 沒有訓練資料，改為「畫面描述/CLIP 與問題的相似度」＋ MMR 去重，挑 4 張 |",
            "| Action 3：生成 | 影格圖片＋逐字稿片段＋對話歷史 → Gemini 多模態生成，每個陳述標註 [影片 mm:ss] |",
            "| （新增）多輪對話 | 與作業一相同的 Query Rewriting，讓「那下一步呢？」能找到正確片段 |", "",
            "沒有說話的操作影片，逐字稿為空，完全依靠關鍵影格的畫面描述與影格圖片回答；有說話的影片則語音與畫面互補。"
            "模態消融結果見作業三進階。\n", shot("ui_video", "作業二 UI")]


def section_hw3(out):
    out += ["## 作業三：評估 RAG 系統\n", "### 3-1 以 RAGAS 評估作業一（10 分）\n",
            "依 RAGAS 論文（講義第 66~69 頁）實作於 `common/ragas_metrics.py`：Faithfulness、Answer Relevancy、Context Relevancy 三項 "
            "reference-free 指標，加上利用自製 QA 標準答案的 Context Precision、Context Recall、Answer Correctness。"
            "評估集：20 題口語問句（單輪）＋ 10 段 × 3 輪多輪對話（共 50 筆），多輪題以改寫後的獨立問題交給評審。\n"]
    cfg = json.loads((R / "hw1_best_config.json").read_text(encoding="utf-8")) if (R / "hw1_best_config.json").exists() else {}
    t = cfg.get("technique", "grounded")
    s, s2 = rd(f"hw3_ragas_summary_{t}.csv"), rd(f"hw3_ragas_summary_{t}_norewrite.csv")
    if s is not None:
        out += [f"最佳設定（chunk={cfg.get('chunk_size')}、overlap={cfg.get('overlap')}、top-k={cfg.get('top_k')}、prompt=`{t}`）：\n", md(s), ""]
    if s2 is not None and "multiturn" in s2.index and s is not None:
        comp = pd.DataFrame({"有改寫": s.loc["multiturn"], "無改寫": s2.loc["multiturn"]})
        comp["差異"] = comp["有改寫"] - comp["無改寫"]
        out += ["多輪對話消融（關閉 Query Rewriting）：\n", md(comp), ""]
    out += ["解讀：Faithfulness 反映回答是否忠於檢索資料（幻覺程度）；Context Recall 高代表檢索幾乎都拿到了答案所需資訊；"
            "Context Relevancy 天生偏低，因為它計算的是「必要句子 / 全部句子」，Top-K 段落中本來就有大量和問題無關的句子，"
            "這也是講義第 69 頁指出不同 Context Relevance 定義差異的原因。逐題明細在 `results/hw3_ragas_*.jsonl`。\n",
            "### 3-2 延伸 RAG 評估手段來評估 VideoRAG（10 分）\n",
            "文字 RAGAS 只看文字 Context；VideoRAG 多了「找錯影片、找錯時間、挑錯影格」三種失敗，而且答案可能只存在畫面中。"
            "因此設計四層評估（`hw3/eval_video.py`）：\n",
            "| 層級 | 指標 | 對應的 RAGAS 概念 |", "|---|---|---|",
            "| A. 影片檢索 | Video Hit@1、Video MRR | Context Precision（影片層級） |",
            "| B. 時間定位 | Segment Hit@K、Best tIoU（取回片段與標準時間區間的時間 IoU） | Context Recall（時間軸） |",
            "| C. 影格選取 | Frame Temporal Precision（影格落在標準區間 ±5 秒）、Frame Relevance Precision（多模態 LLM 看圖判斷影格是否有用） | 視覺版 Context Precision / Relevance |",
            "| D. 生成 | Multimodal Faithfulness（評審同時看逐字稿與影格判斷陳述是否被支持）、Answer Relevancy、Answer Correctness | Faithfulness 等 |", "",
            "標準答案由 Gemini 看 **完整影片**（不經過我們的抽影格流程，避免同一套偏差）出題並標時間區間，再經人工檢查修正（`data/video_qa.json`）。"
            "另做兩種分析：**模態消融**（完整 / 只有語音 / 只有畫面）與 **依題目所需模態、影片類型（有/無說話）、對話輪次分組**。\n"]
    for name, title in [("hw3_video_overall.csv", "模態消融（依 mode）"), ("hw3_video_by_needs.csv", "依題目所需模態（full）"),
                        ("hw3_video_by_videotype.csv", "依影片是否有說話（full）"), ("hw3_video_by_turn.csv", "依對話輪次（full）")]:
        d = rd(name)
        if d is not None:
            out += [f"{title}：\n", md(d), ""]


def hw4_reading() -> str:
    p = R / "hw4_benchmark_answers.jsonl"
    if not p.exists():
        return ""
    df = pd.read_json(p, lines=True)
    m = "judge_score" if "judge_score" in df else "keypoint_recall"
    t = df.pivot_table(index="type", columns="system", values=m, aggfunc="mean")
    if not {"graphrag", "vanilla_rag"} <= set(t.columns):
        return ""
    parts = []
    for typ in ["local", "multihop", "global"]:
        if typ in t.index:
            d = t.loc[typ, "graphrag"] - t.loc[typ, "vanilla_rag"]
            parts.append(f"{typ} 題 GraphRAG {'高於' if d > 0.02 else '低於' if d < -0.02 else '約等於'} Vanilla RAG（{d:+.3f}）")
    txt = f"解讀（以 {m}）：" + "；".join(parts) + "。"
    if "global" in t.index and t.loc["global", "graphrag"] - t.loc["global", "vanilla_rag"] > 0.02:
        txt += ("Global 題的差距最能說明 GraphRAG 的價值：社群摘要涵蓋整篇論文，而 Top-K 段落只看得到局部，"
                "正是講義第 48~52 頁所說「RAG 回答不了」的問題型態。")
    if "local" in t.index and t.loc["local", "graphrag"] < t.loc["local", "vanilla_rag"]:
        txt += "Local 題 Vanilla RAG 較好，因為答案就在一段原文裡，抽成三元組反而可能遺漏細節（例如數字的上下文）。"
    return txt


def section_hw4(out):
    summ = json.loads((R / "hw4_eval_summary.json").read_text(encoding="utf-8")) if (R / "hw4_eval_summary.json").exists() else {}
    bench = json.loads((ROOT / "data" / "graphrag_benchmark.json").read_text(encoding="utf-8"))
    from collections import Counter

    cnt = Counter(b["type"] for b in bench)
    out += ["## 作業四：GraphRAG\n", "### 4-1 將 AI 論文 Graph 化（5 分）\n",
            "論文：**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**（Asai et al., ICLR 2024，arXiv:2310.11511，講義第 31~44 頁）。\n",
            "建圖流程（`hw4/graphrag.py`）：論文切成 1,200 字段落 → LLM 依指定 schema 抽取實體（Method/Model/Token/Dataset/Metric/Person/Organization/Result…）"
            "與關係（trained_on、outperforms、scores、has_value、affiliated_with…，數字結果寫進關係描述）→ 名稱正規化後合併成 networkx 有向多重圖 "
            "→ Louvain 社群偵測 → LLM 為每個社群寫摘要（Microsoft GraphRAG 的 community report）。可匯出 GraphML（Gephi）與 Cypher（Neo4j，講義第 54 頁）。\n",
            f"- 節點 **{summ.get('nodes', '?')}**、邊 **{summ.get('edges', '?')}**、社群 **{summ.get('communities', '?')}**（建圖模型：`{summ.get('llm', '?')}`）\n",
            "![graph](results/hw4_graph.png)\n",
            "### 4-2 Benchmark（5 分）\n",
            f"`data/graphrag_benchmark.json` 共 **{len(bench)} 題**：local {cnt['local']} 題（單一事實/數字）、multihop {cnt['multihop']} 題"
            f"（需串兩個以上實體，例如「哪位作者不在 UW、屬於哪個機構」）、global {cnt['global']} 題（整體性問題，例如講義第 48 頁的「請用 5W1H 整理這篇文章」）。"
            "每題有標準答案與關鍵詞清單（可含同義詞）。\n",
            "### 4-3 實作 GraphRAG 並評估（10 分）\n",
            "- **Local search**（講義第 53 頁）：問題 → 名稱比對＋Embedding 找種子實體 → 沿關係走 2 步 → 以關係描述與問題相似度保留 40 條邊 → 子圖三元組＋實體描述＋來源段落交給 LLM。",
            "- **Global search**：各社群摘要先 map（抽出相關重點並打分）再 reduce（整合成答案）。",
            "- **Router**：LLM 判斷問題屬於 local 或 global。",
            "- 比較對象：No-RAG（LLM 直接回答）、Vanilla RAG（論文切 500 字、混合檢索 Top-5）。",
            "- 指標：Key-point Recall（關鍵詞命中率，不依賴 LLM）、LLM-Judge Correctness（0~5 分正規化）、Router Accuracy。\n"]
    bt, ov = rd("hw4_eval_by_type.csv"), rd("hw4_eval_overall.csv")
    if ov is not None:
        out += ["整體：\n", md(ov), ""]
    p = R / "hw4_eval_by_type.csv"
    if p.exists():
        out += ["依題型：\n", "```\n" + pd.read_csv(p, header=[0, 1], index_col=0).round(3).to_string() + "\n```\n"]
    out += [shot("ui_graph", "作業四 GraphRAG UI"), f"Router Accuracy：**{summ.get('router_accuracy', '?')}**\n",
            hw4_reading(), "逐題答案與評審理由見 `results/hw4_benchmark_answers.jsonl`。\n"]


def main():
    from common.backends import get_llm

    llm = get_llm()
    out = ["# MMIP 第十週作業：RAG / VideoRAG / RAG 評估 / GraphRAG\n",
           f"> 實驗模型：`{llm.name}`。所有數字由 `python run_all.py` 產生，本報告由 `make_report.py` 依 `results/` 自動生成。\n",
           "| 作業 | 項目 | 配分 | 本報告章節 |", "|---|---|---|---|",
           "| 一 | 自製 QA Dataset（Embedding ≥ 200） | 5 | 1-1 |", "| 一 | 多輪對話 RAG + Reference | 10 | 1-2 |",
           "| 一 | 最穩定 Chunk/Overlap/Top-K | 5 | 1-3 |", "| 一 | 最合適 Prompting | 5 | 1-4 |", "| 一 | User Interface | 5 | 1-5 |",
           "| 二 | 自錄 Video Corpus | 5 | 2-1 |", "| 二 | 多輪 VideoRAG | 10 | 2-2 |",
           "| 三 | RAGAS 評估作業一 | 10 | 3-1 |", "| 三 | VideoRAG 評估方法 | 10 | 3-2 |",
           "| 四 | 論文 Graph 化 | 5 | 4-1 |", "| 四 | ≥20 題 Benchmark | 5 | 4-2 |", "| 四 | GraphRAG 實作與評估 | 10 | 4-3 |", ""]
    section_hw1(out)
    section_hw2(out)
    section_hw3(out)
    section_hw4(out)
    out += ["## 重現方式\n", "```bash\npip install -r requirements.txt\ncp .env.example .env   # 填入 GEMINI_API_KEY\n"
            "python run_all.py       # 全部實驗 + 產生報告\npython app.py           # 開啟 UI\n```\n"]
    (ROOT / "REPORT.md").write_text("\n".join(out), encoding="utf-8")
    print("REPORT.md 已更新")


if __name__ == "__main__":
    main()
