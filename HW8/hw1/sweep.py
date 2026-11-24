"""作業一進階 (5)：找出表現最穩定的 Chunk Size、Overlap Size、Top-K。

評分方式採用講義第 10 頁的「繁中檢索能力評測」：正確段落排第 1 名得 1 分、第 2 名 1/2 分 … 第 5 名 1/5 分，
前五筆沒找到 0 分（= MRR@5）。「正確段落」定義：Chunk 覆蓋標準答案證據句 80% 以上的字元。

只看 MRR@5 會偏好「越大塊越好」（大 Chunk 幾乎一定包含答案，但塞給 LLM 的雜訊也最多），
因此再加上「固定 Context 預算」：每次最多送 BUDGET 字給 LLM，K_budget = BUDGET / 平均 chunk 長度，
只計算前 K_budget 名的倒數排名（MRR@budget）。參數選擇以 MRR@budget 的穩定分數為準。

「穩定」的定義：同一組參數在三種提問風格（正式問句 / 口語問句 / 多輪對話追問）下的分數
平均要高、標準差要小 → 穩定分數 = mean − std。另外用 bootstrap 估 95% 信賴區間。

用法：python -m hw1.sweep [--embedder tfidf|gemini|BAAI/bge-m3]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.backends import ROOT, get_embedder, get_llm
from common.rag import ChatRAG, RAGIndex, Turn, build_units, doc_offsets, load_docs, load_qa

SIZES = [100, 150, 200, 300, 400, 600]
OV_RATIOS = [0.0, 0.1, 0.25, 0.5]
KS = [1, 2, 3, 4, 5, 6, 8, 10]
OUT = ROOT / "results"


def covers(unit, ev_span, thr=0.8) -> bool:
    s, e = ev_span  # 全域字元座標
    inter = max(0, min(unit.meta["gend"], e) - max(unit.meta["gstart"], s))
    return inter >= thr * (e - s)


def build_query_sets(qa, docs, llm):
    """回傳 {set_name: [(query, gold_qa_id), ...]}"""
    sets = {"formal": [(q["question"], q["id"]) for q in qa],
            "colloquial": [(q["question_colloquial"], q["id"]) for q in qa]}
    dialogs = json.loads((ROOT / "data" / "multiturn_dialogs.json").read_text(encoding="utf-8"))
    qa_by_id = {q["id"]: q for q in qa}
    # 多輪：用與正式系統相同的 Query Rewriting 產生獨立問句（歷史中的助教回覆用標準答案）
    dummy = ChatRAG(index=None, llm=llm)
    mt = []
    for d in dialogs:
        dummy.reset()
        for t in d["turns"]:
            rq = dummy.rewrite_query(t["q"])
            mt.append((rq, t["gold"]))
            dummy.history.append(Turn(t["q"], rq, qa_by_id[t["gold"]]["answer"], []))
    sets["multiturn"] = mt
    sets["multiturn_no_rewrite"] = [(t["q"], t["gold"]) for d in dialogs for t in d["turns"]]
    return sets


def evaluate(index, qsets, ev_spans, kmax=10):
    """回傳每個 query set 的 rank 陣列（找不到 = inf）。"""
    ranks = {}
    for name, qs in qsets.items():
        r = []
        for q, gold in qs:
            hits = index.search(q, kmax, kinds=("chunk",))
            rk = next((i + 1 for i, (u, _) in enumerate(hits) if covers(u, ev_spans[gold])), np.inf)
            r.append(rk)
        ranks[name] = np.array(r, dtype=float)
    return ranks


def mrr5(r, cut=5):
    return float(np.mean(np.where(r <= cut, 1.0 / r, 0.0)))


BUDGET = 800  # 每次送進 LLM 的參考資料字數預算（控制成本、避免 lost-in-the-middle）


def bootstrap_ci(r, B=1000, seed=0, cut=5):
    rng = np.random.default_rng(seed)
    vals = [mrr5(r[rng.integers(0, len(r), len(r))], cut) for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--per-doc", action="store_true", help="每篇文件各自切塊（預設把整份知識庫當成連續講義）")
    args = ap.parse_args()
    docs, qa = load_docs(), load_qa()
    llm = get_llm()
    off = doc_offsets(docs)
    ev_spans = {}
    for q in qa:
        s = off[q["doc_id"]] + docs[q["doc_id"]]["text"].index(q["evidence"])
        ev_spans[q["id"]] = (s, s + len(q["evidence"]))
    qsets = build_query_sets(qa, docs, llm)
    main_sets = ["formal", "colloquial", "multiturn"]
    rows, krows = [], []
    emb_name = None
    for size in SIZES:
        for ratio in OV_RATIOS:
            ov = int(size * ratio)
            units = build_units(docs, qa, size, ov, use_qa=False, stream=not args.per_doc)
            emb = get_embedder(args.embedder)
            emb_name = emb.name
            idx = RAGIndex(units, emb)
            ranks = evaluate(idx, qsets, ev_spans)
            avg_len = np.mean([u.end - u.start for u in units])
            kb = max(1, int(round(BUDGET / avg_len)))
            per = {k: mrr5(v) for k, v in ranks.items()}
            perb = {k: mrr5(v, kb) for k, v in ranks.items()}
            vals = [perb[k] for k in main_sets]
            allr = np.concatenate([ranks[k] for k in main_sets])
            lo, hi = bootstrap_ci(allr, cut=kb)
            rows.append(dict(chunk_size=size, overlap=ov, overlap_ratio=ratio, n_chunks=idx.n_embeddings,
                             avg_chunk_len=round(avg_len, 1), K_budget=kb,
                             **{f"MRR5_{k}": round(v, 4) for k, v in per.items()},
                             **{f"MRRb_{k}": round(v, 4) for k, v in perb.items()},
                             mean=round(np.mean(vals), 4), std=round(np.std(vals), 4),
                             stability=round(np.mean(vals) - np.std(vals), 4),
                             ci95_low=round(lo, 4), ci95_high=round(hi, 4)))
            for K in KS:
                krows.append(dict(chunk_size=size, overlap=ov, K=K, MRR_at_K=round(mrr5(allr, K), 4),
                                  recall_at_K=round(float(np.mean(allr <= K)), 4),
                                  ctx_tokens=K * size))
            print(rows[-1])
    df = pd.DataFrame(rows).sort_values("stability", ascending=False)
    kdf = pd.DataFrame(krows)
    best = df.iloc[0]
    bk = kdf[(kdf.chunk_size == best.chunk_size) & (kdf.overlap == best.overlap)].sort_values("K")
    # Top-K：取 Recall@K 達到 Recall@10 的 95% 的最小 K（再多放只是增加雜訊與 token）
    rmax = bk.recall_at_K.max()
    best_k = int(bk[bk.recall_at_K >= 0.95 * rmax].K.min())
    best_k = max(best_k, int(best.K_budget))
    tag = emb_name.replace("/", "_") + ("_perdoc" if args.per_doc else "")
    df.to_csv(OUT / f"hw1_sweep_{tag}.csv", index=False)
    kdf.to_csv(OUT / f"hw1_sweep_topk_{tag}.csv", index=False)
    cfg = dict(embedder=emb_name, chunk_size=int(best.chunk_size), overlap=int(best.overlap), top_k=best_k,
               stability=float(best.stability), mean_mrr5=float(best["mean"]), std=float(best["std"]),
               recall_curve=bk[["K", "recall_at_K"]].to_dict("records"))
    old = OUT / "hw1_best_config.json"
    if args.per_doc:
        old = OUT / "hw1_best_config_perdoc.json"
    if old.exists():
        cfg["technique"] = json.loads(old.read_text(encoding="utf-8")).get("technique", "grounded")
    old.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(df.head(10).to_string())
    print("BEST", cfg)
    plot(df, kdf, best, tag)


def plot(df, kdf, best, tag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "WenQuanYi Zen Hei", "DejaVu Sans"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for ratio in OV_RATIOS:
        d = df[df.overlap_ratio == ratio].sort_values("chunk_size")
        ax[0].errorbar(d.chunk_size, d["mean"], yerr=d["std"], marker="o", capsize=3, label=f"overlap {int(ratio*100)}%")
    ax[0].set_xlabel("Chunk size (chars)")
    ax[0].set_ylabel(f"MRR within {BUDGET}-char budget (mean ± std, 3 query styles)")
    ax[0].legend()
    ax[0].set_title("Chunk size / overlap")
    d = kdf[(kdf.chunk_size == best.chunk_size) & (kdf.overlap == best.overlap)]
    ax[1].plot(d.K, d.recall_at_K, marker="o")
    ax[1].set_xlabel("Top-K")
    ax[1].set_ylabel("Recall@K")
    ax[1].set_title(f"Top-K (chunk={int(best.chunk_size)}, overlap={int(best.overlap)})")
    fig.tight_layout()
    fig.savefig(OUT / f"hw1_sweep_{tag}.png", dpi=150)


if __name__ == "__main__":
    main()
