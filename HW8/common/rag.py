"""作業一：多輪對話 RAG 核心。

流程（對應講義第 8、9、25 頁）：
  第一階段 數位化：文件 → Chunk（含座標：doc_id / 起訖字元位置）→ Embedding
                    QA Dataset → 把 Q 轉成 Embedding（講義第 12 頁：將文本轉成 QA 形式）
  第二階段 RAG：多輪 History + Query → Query Rewriting（補代名詞）→ 混合檢索（Dense + BM25, RRF 融合）
                → Top-K 參考資料 + System Prompt → LLM → 回覆 + Reference
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .backends import ROOT, LLM, Embedder, char_tokens, get_embedder, get_llm

DATA = ROOT / "data"


# ---------------------------------------------------------------- data
def load_docs(path: Path = DATA / "kb_image_processing.md") -> dict:
    kb = path.read_text(encoding="utf-8")
    pat = r"## \[(D\d+)\] ([^\n]+)\n(.+?)(?=\n## |\Z)"
    return {m.group(1): {"title": m.group(2), "text": m.group(3).strip()} for m in re.finditer(pat, kb, re.S)}


def load_qa(path: Path = DATA / "qa_dataset.jsonl") -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_index_qa() -> list[dict]:
    """建索引用的 QA：手寫 120 組 + （若有）Gemini 以 5W1H 擴充的 QA。評估的標準答案只用手寫的。"""
    qa = load_qa()
    gen = DATA / "qa_generated.jsonl"
    return qa + (load_qa(gen) if gen.exists() else [])


@dataclass
class Unit:
    """檢索單位：文件 Chunk 或 QA 配對。"""
    uid: str
    kind: str  # "chunk" | "qa"
    doc_id: str
    title: str
    text: str  # 給 LLM 看的內容
    embed_text: str  # 拿去做 Embedding 的內容
    start: int = 0
    end: int = 0
    meta: dict = field(default_factory=dict)

    def ref(self) -> str:
        if self.kind == "chunk":
            extra = "、".join(d for d in self.meta.get("docs", []) if d != self.doc_id)
            extra = f"（跨 {extra}）" if extra else ""
            return f"{self.doc_id}《{self.title}》第 {self.start}–{self.end} 字{extra}"
        return f"{self.meta.get('qa_id')}（QA，出自 {self.doc_id}《{self.title}》）"


def chunk_text(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """以字元為單位的滑動視窗切塊；結尾盡量對齊句號，避免把句子切斷太多。"""
    assert 0 <= overlap < size
    spans, s, n = [], 0, len(text)
    while s < n:
        e = min(n, s + size)
        if e < n:
            cut = max(text.rfind(p, s + int(size * 0.6), e) for p in "。；！？")
            if cut > 0:
                e = cut + 1
        spans.append((s, e))
        if e >= n:
            break
        s = max(e - overlap, s + 1)
    return spans


def doc_offsets(docs: dict) -> dict:
    """stream 模式下每篇文件在整份語料中的起始位置（文件間以換行分隔）。"""
    off, pos = {}, 0
    for did, d in docs.items():
        off[did] = pos
        pos += len(d["text"]) + 1
    return off


def build_units(docs: dict, qa: list[dict] | None, size: int, overlap: int, use_qa: bool = True,
                stream: bool = True) -> list[Unit]:
    """stream=True：把整份知識庫當成一本連續的講義來切（像切 PDF 一樣，Chunk 可能跨越兩個主題），
    每個 Chunk 仍記錄它主要屬於哪一篇文件與字元座標。stream=False：每篇文件各自切。"""
    units = []
    if stream:
        off = doc_offsets(docs)
        full = "\n".join(d["text"] for d in docs.values())
        starts = sorted((o, did) for did, o in off.items())
        for i, (s, e) in enumerate(chunk_text(full, size, overlap)):
            # 依重疊字數決定主要文件
            cover = {}
            for j, (o, did) in enumerate(starts):
                oe = starts[j + 1][0] if j + 1 < len(starts) else len(full)
                inter = max(0, min(e, oe) - max(s, o))
                if inter:
                    cover[did] = inter
            did = max(cover, key=cover.get)
            body = full[s:e].strip()
            ls = max(0, s - off[did])
            units.append(Unit(f"c{i:04d}", "chunk", did, docs[did]["title"], body, body, ls, ls + (e - s),
                              meta={"gstart": s, "gend": e, "docs": list(cover)}))
    else:
        off = doc_offsets(docs)
        for did, d in docs.items():
            for i, (s, e) in enumerate(chunk_text(d["text"], size, overlap)):
                body = d["text"][s:e]
                units.append(Unit(f"{did}-c{i}", "chunk", did, d["title"], body, f"{d['title']}：{body}", s, e,
                                  meta={"gstart": off[did] + s, "gend": off[did] + e, "docs": [did]}))
    if use_qa and qa:
        # 雙向量 QA 索引：同一組 QA 存兩個 Embedding —— 問題向量（講義第 12 頁：把 Q 轉成 Embedding）
        # 與答案向量（使用者的問法跟答案用詞較像時也能命中），檢索時以 qa_id 去重。
        for q in qa:
            for kind, key in (("qa", q["question"]), ("qa_ans", q["answer"])):
                units.append(Unit(f"{q['id']}{'' if kind == 'qa' else '-a'}", "qa", q["doc_id"], q["doc_title"],
                                  f"Q：{q['question']} A：{q['answer']}", key,
                                  meta={"qa_id": q["id"], "evidence": q.get("evidence", ""), "vector": kind}))
    return units


# ---------------------------------------------------------------- BM25 (char bigram)
class BM25:
    def __init__(self, texts: list[str]):
        from rank_bm25 import BM25Okapi

        self.bm = BM25Okapi([char_tokens(t, (1, 2)) for t in texts])

    def scores(self, q: str) -> np.ndarray:
        return np.asarray(self.bm.get_scores(char_tokens(q, (1, 2))), dtype=np.float32)


# ---------------------------------------------------------------- index
class RAGIndex:
    def __init__(self, units: list[Unit], embedder: Embedder | None = None, hybrid: bool = True):
        self.units = units
        self.emb = embedder or get_embedder()
        texts = [u.embed_text for u in units]
        self.emb.fit(texts)
        self.M = self.emb.embed(texts, kind="doc")
        self.bm25 = BM25(texts) if hybrid else None

    @property
    def n_embeddings(self) -> int:
        return int(self.M.shape[0])

    def search(self, query: str, k: int = 5, rrf_k: int = 60, kinds=("chunk", "qa")) -> list[tuple[Unit, float]]:
        qv = self.emb.embed([query], kind="query")[0]
        dense = self.M @ qv
        mask = np.array([u.kind in kinds for u in self.units])
        order_d = [i for i in np.argsort(-dense) if mask[i]]
        if self.bm25 is None:
            seen, out = set(), []
            for i in order_d:
                key = self.units[i].meta.get("qa_id") or self.units[i].uid
                if key not in seen:
                    seen.add(key)
                    out.append((self.units[i], float(dense[i])))
                if len(out) == k:
                    break
            return out
        sparse = self.bm25.scores(query)
        order_s = [i for i in np.argsort(-sparse) if mask[i]]
        fused: dict[int, float] = {}
        for rank, i in enumerate(order_d[:100]):
            fused[i] = fused.get(i, 0) + 1 / (rrf_k + rank + 1)
        for rank, i in enumerate(order_s[:100]):
            fused[i] = fused.get(i, 0) + 1 / (rrf_k + rank + 1)
        out, seen = [], set()
        for i in sorted(fused, key=lambda i: -fused[i]):
            key = self.units[i].meta.get("qa_id") or self.units[i].uid
            if key in seen:
                continue
            seen.add(key)
            out.append((self.units[i], fused[i]))
            if len(out) == k:
                break
        return out


# ---------------------------------------------------------------- prompting techniques
SYSTEM = ("你是「數位影像處理」課程的助教。只能根據【參考資料】回答，使用繁體中文。"
          "每個事實陳述後面要用 [編號] 標註出處。如果參考資料不足以回答，請直接說「參考資料中沒有提到」，不要編造。")

PROMPTS = {
    # 1. 直接把資料丟給模型，沒有額外規則（baseline）
    "plain": dict(system=None, template="{ctx}\n\n【問題】{q}\n請回答："),
    # 2. 角色 + 規則 + 引用格式（grounded）
    "grounded": dict(system=SYSTEM, template="{hist}【參考資料】\n{ctx}\n\n【問題】{q}\n請用 2~4 句話回答，並標註 [編號]。"),
    # 3. Chain-of-Thought：先列出相關證據再回答
    "cot": dict(system=SYSTEM, template=("{hist}【參考資料】\n{ctx}\n\n【問題】{q}\n"
                                         "請依照以下格式作答：\n證據：逐條列出會用到的參考資料編號與關鍵句\n"
                                         "回答：根據證據用 2~4 句話回答，並標註 [編號]。")),
    # 4. Few-shot：示範一個好的引用式回答
    "fewshot": dict(system=SYSTEM, template=("範例\n【參考資料】\n[1] 中值濾波器以鄰域像素的中位數取代中心像素，對椒鹽雜訊特別有效。\n"
                                             "【問題】黑白點雜訊要怎麼去除？\n回答：可以使用中值濾波器，它以鄰域中位數取代中心像素，對椒鹽雜訊特別有效[1]。\n\n"
                                             "{hist}【參考資料】\n{ctx}\n\n【問題】{q}\n回答：")),
    # 5. Step-Back：先抽象出背景問題，兩個問題一起檢索（講義第 16~20 頁）
    "stepback": dict(system=SYSTEM, template=("{hist}【背景問題】{sb}\n【參考資料】\n{ctx}\n\n【問題】{q}\n"
                                              "請先根據背景原理思考，再用 2~4 句話回答原問題，並標註 [編號]。")),
}

REWRITE_PROMPT = ("以下是使用者與助教的對話紀錄，以及使用者最新的提問。請把最新提問改寫成一個「不需要看對話紀錄也能理解」的完整繁體中文問題："
                  "補上被省略的主詞、把代名詞（它、這個、那、其）換成具體名稱。只輸出改寫後的問題。\n\n"
                  "【對話紀錄】\n{hist}\n【最新提問】{q}\n改寫：")
STEPBACK_PROMPT = ("請把下面這個具體問題「往後退一步」，改寫成一個詢問背後通用原理或概念的問題，只輸出問題本身。\n"
                   "例：『JPEG 品質設 10 為什麼會有方塊？』→『JPEG 壓縮的流程與失真來源是什麼？』\n問題：{q}\n抽象問題：")

PRONOUNS = re.compile(r"^(那|它|這|其|他|該|此)|它|這個|那個|這種|那種|其|呢[？?]?$|還有|另外|差在|跟.{0,4}比")


@dataclass
class Turn:
    user: str
    rewritten: str
    answer: str
    refs: list


class ChatRAG:
    """多輪對話 RAG 聊天機器人。"""

    def __init__(self, index: RAGIndex, llm: LLM | None = None, top_k: int = 5,
                 technique: str = "grounded", rewrite: bool = True, max_hist: int = 3):
        self.index, self.llm = index, llm or get_llm()
        self.top_k, self.technique, self.rewrite, self.max_hist = top_k, technique, rewrite, max_hist
        self.history: list[Turn] = []

    def reset(self):
        self.history = []

    def _hist_text(self) -> str:
        return "\n".join(f"使用者：{t.user}\n助教：{t.answer}" for t in self.history[-self.max_hist:])

    def rewrite_query(self, q: str) -> str:
        if not self.rewrite or not self.history:
            return q
        if self.llm.online:
            out = self.llm.generate(REWRITE_PROMPT.format(hist=self._hist_text(), q=q)).strip()
            return out.splitlines()[0].strip() if out else q
        # 離線版：偵測到省略/代名詞時，把上一輪改寫後的問題當作主題前綴
        if PRONOUNS.search(q) or len(q) < 10:
            return f"{self.history[-1].rewritten}；{q}"
        return q

    def retrieve(self, q: str, k: int | None = None):
        return self.index.search(q, k or self.top_k)

    def ask(self, q: str) -> Turn:
        sq = self.rewrite_query(q)
        tech = PROMPTS[self.technique]
        sb = ""
        hits = self.retrieve(sq)
        if self.technique == "stepback":
            sb = self.llm.generate(STEPBACK_PROMPT.format(q=sq)).strip() if self.llm.online else sq
            seen = {u.uid for u, _ in hits}
            hits += [(u, s) for u, s in self.retrieve(sb, max(2, self.top_k // 2)) if u.uid not in seen]
        ctx = "\n".join(f"[{i + 1}] {u.text}" for i, (u, _) in enumerate(hits))
        hist = f"【先前對話】\n{self._hist_text()}\n\n" if self.history else ""
        prompt = tech["template"].format(ctx=ctx, q=sq, hist=hist, sb=sb)
        ans = self.llm.generate(prompt, system=tech["system"]).strip()
        refs = [dict(n=i + 1, uid=u.uid, kind=u.kind, ref=u.ref(), text=u.text, score=round(s, 4))
                for i, (u, s) in enumerate(hits)]
        turn = Turn(q, sq, ans, refs)
        self.history.append(turn)
        return turn


def build_default(chunk_size=200, overlap=50, top_k=5, technique="grounded", embedder=None, llm=None) -> ChatRAG:
    cfg_path = ROOT / "results" / "hw1_best_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        chunk_size, overlap, top_k = cfg["chunk_size"], cfg["overlap"], cfg["top_k"]
        technique = cfg.get("technique", technique)
    units = build_units(load_docs(), load_index_qa(), chunk_size, overlap, stream=True)
    return ChatRAG(RAGIndex(units, embedder), llm, top_k=top_k, technique=technique)
