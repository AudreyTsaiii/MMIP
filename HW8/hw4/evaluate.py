"""作業四 (5)+(10)：用 24 題 Benchmark（data/graphrag_benchmark.json）評估 GraphRAG。

題型：local（10 題，單一事實/數字）、multihop（8 題，需要串兩個以上實體）、global（6 題，整體摘要/5W1H）
比較系統：
  1. No-RAG：LLM 直接回答（只在有 API key 時跑）
  2. Vanilla RAG：論文切塊 + 混合檢索 Top-5（作業一的方法）
  3. GraphRAG：本作業（auto 路由 local/global）
指標：
  - Key-point Recall：標準答案關鍵詞（可含同義詞 a|b）出現在回答中的比例（不需 LLM、可重現）
  - LLM-Judge Correctness：LLM 依標準答案給 0~5 分再除以 5（有 API key 時）
  - Router Accuracy：GraphRAG 路由結果與題型（global vs. local/multihop）是否一致
"""
from __future__ import annotations

import json
import re

import pandas as pd

from common.backends import ROOT, get_embedder, get_llm
from common.rag import RAGIndex, Unit, chunk_text
from hw4.graphrag import GRAPH_JSON, GraphRAG, load_paper

OUT = ROOT / "results"

JUDGE = """請依【標準答案】為【系統回答】的正確性與完整性打 0~5 分（5=完全正確且完整，0=錯誤或沒回答）。只輸出 JSON：{{"score": 分數, "reason": "一句話"}}
【問題】{q}
【標準答案】{gt}
【系統回答】{a}"""

RAG_PROMPT = """你是論文問答助手。只能根據【參考資料】回答，使用繁體中文，數字要精確。資訊不足就說不知道。
【參考資料】
{ctx}
【問題】{q}
回答："""


def keypoint_recall(ans: str, keys: list[str]) -> float:
    a = ans.lower()
    return sum(any(alt.lower() in a for alt in k.split("|")) for k in keys) / len(keys)


def main():
    llm, emb = get_llm(), get_embedder()
    bench = json.loads((ROOT / "data" / "graphrag_benchmark.json").read_text(encoding="utf-8"))
    paper = load_paper(str(ROOT / "data" / "selfrag.pdf") if (ROOT / "data" / "selfrag.pdf").exists() else None)
    g = GraphRAG.load(llm=llm, embedder=emb) if GRAPH_JSON.exists() else GraphRAG.build(paper, llm, emb)
    units = [Unit(f"p{i}", "chunk", "paper", "Self-RAG", paper[s:e], paper[s:e], s, e)
             for i, (s, e) in enumerate(chunk_text(paper, 500, 100))]
    rag = RAGIndex(units, get_embedder())
    rows = []
    for b in bench:
        res = {}
        hits = rag.search(b["q"], 5)
        ctx = "\n".join(f"[{i + 1}] {u.text}" for i, (u, _) in enumerate(hits))
        res["vanilla_rag"] = llm.generate(RAG_PROMPT.format(ctx=ctx, q=b["q"]) if llm.online else f"{ctx}\n【問題】{b['q']}")
        gr = g.ask(b["q"])
        res["graphrag"] = gr["answer"]
        if llm.online:
            res["no_rag"] = llm.generate(f"請用繁體中文回答這個關於 Self-RAG 論文（Asai et al., 2023）的問題：{b['q']}")
        for sysname, ans in res.items():
            row = dict(id=b["id"], type=b["type"], system=sysname, question=b["q"], answer=ans.strip(),
                       keypoint_recall=keypoint_recall(ans, b["key"]))
            if sysname == "graphrag":
                row["route"] = gr["mode"]
                row["route_ok"] = (gr["mode"] == "global") == (b["type"] == "global")
            if llm.online:
                j = llm.generate_json(JUDGE.format(q=b["q"], gt=b["a"], a=ans)) or {}
                row["judge_score"] = float(j.get("score", 0)) / 5
                row["judge_reason"] = j.get("reason", "")
            rows.append(row)
        print(b["id"], {r["system"]: round(r["keypoint_recall"], 2) for r in rows[-len(res):]})
    df = pd.DataFrame(rows)
    df.to_json(OUT / "hw4_benchmark_answers.jsonl", orient="records", lines=True, force_ascii=False)
    metrics = ["keypoint_recall"] + (["judge_score"] if "judge_score" in df else [])
    by_type = df.pivot_table(index="system", columns="type", values=metrics, aggfunc="mean").round(3)
    overall = df.groupby("system")[metrics].mean().round(3)
    route_acc = df[df.system == "graphrag"]["route_ok"].mean()
    by_type.to_csv(OUT / "hw4_eval_by_type.csv")
    overall.to_csv(OUT / "hw4_eval_overall.csv")
    summary = dict(judge="llm" if llm.online else "keypoint-only", llm=llm.name,
                   nodes=g.G.number_of_nodes(), edges=g.G.number_of_edges(), communities=len(g.summaries),
                   router_accuracy=round(float(route_acc), 3))
    (OUT / "hw4_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(by_type.to_string())
    print(overall.to_string())
    print(summary)


if __name__ == "__main__":
    main()
