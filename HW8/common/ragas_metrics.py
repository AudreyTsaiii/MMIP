"""RAGAS 評估指標（依 Es et al., EACL 2024 Demo 論文定義實作，講義第 66~69 頁）。

無參考答案（reference-free，RAGAS 論文原始三項）：
  - Faithfulness 忠實度   ：把回答拆成原子陳述，判斷每一句能否由 Context 推得 → 被支持的比例
  - Answer Relevance 回答相關性：由回答反推 n 個問題，與原問題做 Embedding cosine 平均
  - Context Relevance 脈絡相關性：Context 中「回答問題必要的句子數 / 總句數」
有參考答案（ragas 套件後續加入的指標，用我們自製 QA Dataset 的標準答案）：
  - Context Precision@K：每個檢索結果是否對產生標準答案有用，計算平均精確率（排名越前面越好）
  - Context Recall    ：標準答案中的陳述，有多少比例可以在 Context 找到
  - Answer Correctness：回答與標準答案的陳述層級 F1

為了節省免費 API 額度，每一筆樣本只呼叫一次 LLM 評審（一次輸出所有判斷的 JSON），
answer relevance 的 cosine 則用 Embedding 計算。沒有 API key 時退回「字元 bigram 重疊」的近似版本，
結果會標註 judge=lexical，僅供流程測試。
"""
from __future__ import annotations

import re

import numpy as np

from .backends import LLM, Embedder, char_tokens

JUDGE_PROMPT = """你是嚴格的 RAG 系統評審。請根據以下資料，輸出一個 JSON 物件（不要輸出其他文字）。

【問題】{q}
【參考資料（已編號）】
{ctx}
【系統回答】{a}
【標準答案】{gt}
【參考資料的句子清單】
{sents}

請輸出：
{{
 "answer_statements": [{{"s": "把系統回答拆成的原子陳述（每句只含一個事實）", "supported": true/false（能否完全由參考資料推得）}}],
 "generated_questions": ["根據系統回答反推出的問題（3 個，不要看原問題）"],
 "noncommittal": true/false（系統回答是否含糊其辭、拒答或說資料沒有提到）,
 "relevant_sentence_ids": [回答【問題】真正必要的句子編號（來自句子清單）],
 "context_useful": [每筆參考資料依序是否有助於得出標準答案，true/false],
 "gt_statements": [{{"s": "把標準答案拆成的原子陳述", "attributed": true/false（能否在參考資料中找到）}}],
 "answer_vs_gt": {{"tp": 回答中與標準答案一致的陳述數, "fp": 回答中標準答案沒有的陳述數, "fn": 標準答案中回答遺漏的陳述數}}
}}"""


def split_sents(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[。！？；\n])", text) if len(s.strip()) > 3]


def _avg_precision(useful: list[bool]) -> float:
    hits, score = 0, 0.0
    for i, u in enumerate(useful):
        if u:
            hits += 1
            score += hits / (i + 1)
    return score / hits if hits else 0.0


def ragas_llm(sample: dict, llm: LLM, embedder: Embedder) -> dict:
    """sample: {question, answer, contexts: [str], ground_truth}"""
    ctxs = sample["contexts"]
    ctx = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(ctxs))
    sents = [s for c in ctxs for s in split_sents(c)]
    sl = "\n".join(f"({i}) {s}" for i, s in enumerate(sents))
    j = llm.generate_json(JUDGE_PROMPT.format(q=sample["question"], ctx=ctx, a=sample["answer"],
                                              gt=sample.get("ground_truth", ""), sents=sl))
    if not isinstance(j, dict):
        return {"judge": "llm-failed"}
    st = j.get("answer_statements") or []
    faith = np.mean([bool(x.get("supported")) for x in st]) if st else 0.0
    gq = [q for q in (j.get("generated_questions") or []) if isinstance(q, str)]
    if gq and not j.get("noncommittal"):
        V = embedder.embed([sample["question"]] + gq, kind="query")
        arel = float(np.mean(V[1:] @ V[0]))
    else:
        arel = 0.0
    rel_ids = {int(i) for i in (j.get("relevant_sentence_ids") or []) if str(i).isdigit() and int(i) < len(sents)}
    crel = len(rel_ids) / max(1, len(sents))
    useful = [bool(u) for u in (j.get("context_useful") or [])][:len(ctxs)]
    cprec = _avg_precision(useful)
    gts = j.get("gt_statements") or []
    crec = np.mean([bool(x.get("attributed")) for x in gts]) if gts else 0.0
    c = j.get("answer_vs_gt") or {}
    tp, fp, fn = (float(c.get(k, 0) or 0) for k in ("tp", "fp", "fn"))
    corr = tp / (tp + 0.5 * (fp + fn)) if tp + fp + fn else 0.0
    return dict(judge="llm", faithfulness=float(faith), answer_relevancy=arel, context_relevancy=crel,
                context_precision=cprec, context_recall=float(crec), answer_correctness=corr)


def _overlap(a: str, b: str) -> float:
    ta, tb = set(char_tokens(a, (2,))), set(char_tokens(b, (2,)))
    return len(ta & tb) / max(1, len(ta))


def ragas_lexical(sample: dict) -> dict:
    """離線近似版（無 LLM）：以字元 bigram 重疊取代 LLM 判斷。"""
    ctx_all = " ".join(sample["contexts"])
    ans = re.sub(r"\[\d+\]", "", sample["answer"])
    st = split_sents(ans) or [ans]
    faith = np.mean([_overlap(s, ctx_all) > 0.6 for s in st])
    arel = _overlap(sample["question"], ans)
    sents = [s for c in sample["contexts"] for s in split_sents(c)]
    gt = sample.get("ground_truth", "")
    crel = np.mean([_overlap(gt, s) > 0.3 for s in sents]) if sents else 0.0
    useful = [_overlap(gt, c) > 0.3 for c in sample["contexts"]]
    gts = split_sents(gt) or [gt]
    crec = np.mean([_overlap(s, ctx_all) > 0.6 for s in gts])
    p, r = _overlap(ans, gt), _overlap(gt, ans)
    corr = 2 * p * r / (p + r) if p + r else 0.0
    return dict(judge="lexical", faithfulness=float(faith), answer_relevancy=float(arel), context_relevancy=float(crel),
                context_precision=_avg_precision(useful), context_recall=float(crec), answer_correctness=float(corr))


METRICS = ["faithfulness", "answer_relevancy", "context_relevancy", "context_precision", "context_recall",
           "answer_correctness"]


def evaluate_sample(sample: dict, llm: LLM, embedder: Embedder) -> dict:
    return ragas_llm(sample, llm, embedder) if llm.online else ragas_lexical(sample)
