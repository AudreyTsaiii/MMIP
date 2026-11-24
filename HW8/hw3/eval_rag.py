"""作業三基礎 (10)：用 RAGAS 評估作業一的 RAG 系統。

評估集（全部來自自製 QA Dataset，所以有標準答案）：
  - 單輪：口語化問題（與索引中的正式問句措辭不同，避免「背答案」）
  - 多輪：data/multiturn_dialogs.json 的 10 段 × 3 輪對話（後兩輪都含代名詞/省略）
指標：Faithfulness、Answer Relevancy、Context Relevancy、Context Precision、Context Recall、Answer Correctness
（定義見 common/ragas_metrics.py）

用法：
  python -m hw3.eval_rag                       # 用目前最佳設定評估
  python -m hw3.eval_rag --technique cot --n 30
  python -m hw3.eval_rag --no-rewrite          # 消融：關掉多輪改寫
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.backends import ROOT, get_embedder, get_llm
from common.rag import ChatRAG, RAGIndex, build_units, load_docs, load_index_qa, load_qa
from common.ragas_metrics import METRICS, evaluate_sample

OUT = ROOT / "results"


def run_eval(technique="grounded", n_single=30, chunk_size=None, overlap=None, top_k=None,
             rewrite=True, tag=None, seed=0, llm=None, embedder=None, verbose=True, n_dialogs=10):
    cfg = json.loads((OUT / "hw1_best_config.json").read_text(encoding="utf-8"))
    chunk_size = chunk_size or cfg["chunk_size"]
    overlap = cfg["overlap"] if overlap is None else overlap
    top_k = top_k or cfg["top_k"]
    docs, qa = load_docs(), load_qa()
    qa_by_id = {q["id"]: q for q in qa}
    llm = llm or get_llm()
    emb = embedder or get_embedder()
    index = RAGIndex(build_units(docs, load_index_qa(), chunk_size, overlap), emb)
    bot = ChatRAG(index, llm, top_k=top_k, technique=technique, rewrite=rewrite)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(qa), size=min(n_single, len(qa)), replace=False)
    rows = []
    # 單輪
    for i in pick:
        q = qa[i]
        bot.reset()
        t = bot.ask(q["question_colloquial"])
        rows.append(dict(setting="single", dialog="-", turn=1, qa_id=q["id"], question=q["question_colloquial"],
                         rewritten=t.rewritten, answer=t.answer, ground_truth=q["answer"],
                         contexts=[r["text"] for r in t.refs], refs=[r["ref"] for r in t.refs]))
    # 多輪
    for d in json.loads((ROOT / "data" / "multiturn_dialogs.json").read_text(encoding="utf-8"))[:n_dialogs]:
        bot.reset()
        for k, turn in enumerate(d["turns"]):
            t = bot.ask(turn["q"])
            g = qa_by_id[turn["gold"]]
            rows.append(dict(setting="multiturn", dialog=d["id"], turn=k + 1, qa_id=g["id"], question=turn["q"],
                             rewritten=t.rewritten, answer=t.answer, ground_truth=g["answer"],
                             contexts=[r["text"] for r in t.refs], refs=[r["ref"] for r in t.refs]))
    for r in rows:
        # 多輪的後續問題含代名詞，評審要看「改寫後的獨立問題」才公平
        m = evaluate_sample(dict(question=r["rewritten"], answer=r["answer"], contexts=r["contexts"],
                                 ground_truth=r["ground_truth"]), llm, emb)
        r.update(m)
    df = pd.DataFrame(rows)
    tag = tag or f"{technique}_c{chunk_size}_o{overlap}_k{top_k}{'' if rewrite else '_norewrite'}"
    df.to_json(OUT / f"hw3_ragas_{tag}.jsonl", orient="records", lines=True, force_ascii=False)
    summ = df.groupby("setting")[[m for m in METRICS if m in df]].mean().round(4)
    summ.loc["overall"] = df[[m for m in METRICS if m in df]].mean().round(4)
    if verbose:
        print(f"== {tag} (judge={df.get('judge', pd.Series(['?'])).iloc[0]}) ==")
        print(summ.to_string())
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--technique", default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--no-rewrite", action="store_true")
    a = ap.parse_args()
    cfg = json.loads((OUT / "hw1_best_config.json").read_text(encoding="utf-8"))
    tech = a.technique or cfg.get("technique", "grounded")
    df, summ = run_eval(tech, a.n, rewrite=not a.no_rewrite)
    summ.to_csv(OUT / f"hw3_ragas_summary_{tech}{'_norewrite' if a.no_rewrite else ''}.csv")


if __name__ == "__main__":
    main()
