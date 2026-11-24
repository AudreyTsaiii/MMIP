"""作業四：GraphRAG（論文：Self-RAG, arXiv:2310.11511）。

(5) Graph 化：論文切塊 → LLM 抽取 (實體, 類型, 描述) 與 (頭, 關係, 尾, 描述) → 合併成 networkx 知識圖譜
    → Louvain 社群偵測 → LLM 為每個社群寫摘要（Microsoft GraphRAG 的 community report 概念）
    無 API key 時退回規則式抽取：以專有名詞/縮寫字典當實體、同句共現當關係。
(10) 回答問題：
    - Local search（講義第 53 頁）：Query → 找種子實體（名稱比對 + Embedding）→ 沿關係多走 1~2 步
      → 把子圖的三元組 + 實體描述 + 來源段落交給 LLM
    - Global search（講義第 48 頁「請用 5W1H 整理 XXX」這類 RAG 答不好的問題）：
      對社群摘要做 map-reduce
    - Router：判斷問題要走 local 還是 global
輸出：results/hw4_graph.graphml（可匯入 Gephi）、results/hw4_graph.cypher（可匯入 Neo4j）、results/hw4_graph.png

用法：python -m hw4.graphrag --build      # 建圖
     python -m hw4.graphrag --ask "IsSup 有哪些輸出值？"
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from common.backends import ROOT, get_embedder, get_llm
from common.rag import chunk_text

OUT = ROOT / "results"
GRAPH_JSON = OUT / "hw4_graph.json"

EXTRACT_PROMPT = """你要把一篇 AI 論文的段落轉成知識圖譜。請從【段落】抽取實體與關係，輸出 JSON（不要其他文字）：
{{"entities":[{{"name":"實體名稱（英文原名，統一大小寫，如 Self-RAG、IsSup、GPT-4、PopQA）","type":"Method|Model|Component|Token|Dataset|Metric|Person|Organization|Concept|Result|Hyperparameter","description":"一句話描述（繁體中文，保留數字）"}}],
 "relations":[{{"source":"實體","relation":"簡短動詞片語（英文 snake_case，如 trained_on、outperforms、has_value、part_of、uses、evaluated_by、scores、proposed_by、affiliated_with）","target":"實體","description":"一句話說明此關係（保留具體數字）"}}]}}
規則：數字結果要寫成關係（例如 Self-RAG 13B -scores-> PopQA，描述寫 55.8）；同一個東西用同一個名稱。

【段落】
{text}"""

COMMUNITY_PROMPT = """以下是知識圖譜中一個社群的實體與關係。請用繁體中文寫 3~5 句的摘要，說明這群實體共同描述了論文的哪個面向，保留重要數字。

{body}

摘要："""

ANSWER_PROMPT = """你是論文問答助手。只能根據【知識圖譜資訊】回答，使用繁體中文，數字要精確。資訊不足就說不知道。

【知識圖譜資訊】
{ctx}

【問題】{q}
回答："""

MAP_PROMPT = """根據以下社群摘要，列出與問題有關的重點（條列，每點一句，附重要性 1-10 分，格式「(分數) 重點」）。若無關請只輸出 NONE。
【社群摘要】{summary}
【問題】{q}"""

REDUCE_PROMPT = """以下是從論文知識圖譜各個社群整理出的重點（括號內為重要性）。請整合成完整、有條理的繁體中文回答。

{points}

【問題】{q}
回答："""

ROUTER_PROMPT = """判斷以下關於論文的問題，是要查「特定事實/數字/少數實體之間的關係」(local)，還是要「整體摘要、整理、比較、列出全部」(global)。只輸出 local 或 global。
問題：{q}"""

# 規則式抽取用的詞典（離線模式）
TERMS = r"(Self-RAG|RAG|Retrieve|IsRel|IsSup|IsUse|critic|generator|retriever|GPT-4|ChatGPT|Llama ?2(?:-chat)?(?: ?\d+B)?|Alpaca|Contriever(?:-MS MARCO)?|GTR-XXL|PopQA|TriviaQA|PubHealth|ARC-Challenge|ASQA|ALCE|FactScore|MAUVE|SAIL|Toolformer|CoVE|perplexity\.ai|Llama2-FT|Open-Instruct|PPO|RLHF|vLLM|beam search|reflection tokens?|critique tokens?|University of Washington|Allen Institute for AI|IBM Research AI|Akari Asai|Zeqiu Wu|Yizhong Wang|Avirup Sil|Hannaneh Hajishirzi|Stability AI|LATS|No Retriever|No Critic|Retrieve top 1|Hard constraints)"


def load_paper(pdf: str | None = None) -> str:
    if pdf:
        from pypdf import PdfReader

        txt = "\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
        cut = txt.find("\nReferences")
        return txt[:cut] if cut > 0 else txt
    return (ROOT / "data" / "selfrag_paper.md").read_text(encoding="utf-8")


def norm(name: str) -> str:
    n = re.sub(r"\s+", " ", name.strip())
    n = re.sub(r"(?i)^self[- ]?rag$", "Self-RAG", n)
    n = re.sub(r"(?i)^reflection tokens?$", "reflection token", n)
    n = re.sub(r"(?i)^critique tokens?$", "critique token", n)
    n = re.sub(r"(?i)^llama ?2", "Llama2", n)
    return n


class GraphRAG:
    def __init__(self, G: nx.MultiDiGraph, chunks: list[str], summaries: dict, llm=None, embedder=None):
        self.G, self.chunks, self.summaries = G, chunks, summaries
        self.llm = llm or get_llm()
        self.emb = embedder or get_embedder()
        self.nodes = list(G.nodes)
        texts = [f"{n}：{'；'.join(G.nodes[n].get('desc', [])[:3])}" for n in self.nodes]
        self.emb.fit(texts + chunks)
        self.NV = self.emb.embed(texts)

    # ------------------------------------------------------------ build
    @classmethod
    def build(cls, text: str, llm=None, embedder=None, chunk_size=1200, overlap=150):
        llm = llm or get_llm()
        chunks = [text[s:e] for s, e in chunk_text(text, chunk_size, overlap)]
        G = nx.MultiDiGraph()

        def add_node(n, typ="Concept", desc="", cid=None):
            n = norm(n)
            if not n:
                return None
            if n not in G:
                G.add_node(n, type=typ, desc=[], chunks=[])
            if desc and desc not in G.nodes[n]["desc"]:
                G.nodes[n]["desc"].append(desc)
            if cid is not None and cid not in G.nodes[n]["chunks"]:
                G.nodes[n]["chunks"].append(cid)
            return n

        for cid, ch in enumerate(chunks):
            if llm.online:
                j = llm.generate_json(EXTRACT_PROMPT.format(text=ch)) or {}
                for e in j.get("entities", []):
                    add_node(e.get("name", ""), e.get("type", "Concept"), e.get("description", ""), cid)
                for r in j.get("relations", []):
                    s, t = add_node(r.get("source", ""), cid=cid), add_node(r.get("target", ""), cid=cid)
                    if s and t and s != t:
                        G.add_edge(s, t, relation=r.get("relation", "related_to"), desc=r.get("description", ""), chunk=cid)
            else:  # 規則式：同一句出現的詞典實體兩兩相連
                for sent in re.split(r"(?<=[.;:])\s+|\n", ch):
                    ents = list(dict.fromkeys(norm(m) for m in re.findall(TERMS, sent)))
                    for e in ents:
                        add_node(e, desc=sent.strip()[:200], cid=cid)
                    for i in range(len(ents)):
                        for k in range(i + 1, len(ents)):
                            G.add_edge(ents[i], ents[k], relation="co_occurs", desc=sent.strip()[:300], chunk=cid)
        # 社群 + 摘要
        UG = nx.Graph(G)
        comms = nx.community.louvain_communities(UG, seed=42) if UG.number_of_edges() else [set(UG.nodes)]
        summaries = {}
        for ci, c in enumerate(sorted(comms, key=len, reverse=True)):
            for n in c:
                G.nodes[n]["community"] = ci
            sub = G.subgraph(c)
            body = "\n".join(f"- {n}（{G.nodes[n]['type']}）：{'；'.join(G.nodes[n]['desc'][:2])}" for n in list(c)[:40])
            body += "\n" + "\n".join(f"- {u} -{d['relation']}-> {v}：{d['desc']}" for u, v, d in list(sub.edges(data=True))[:60])
            if len(c) < 2:
                continue
            summaries[ci] = (llm.generate(COMMUNITY_PROMPT.format(body=body)).strip() if llm.online
                             else f"社群 {ci}：" + "、".join(sorted(c)[:15]) + "。" +
                             " ".join(dict.fromkeys(d['desc'] for _, _, d in list(sub.edges(data=True))[:6])))
        return cls(G, chunks, summaries, llm, embedder)

    def save(self, path: Path = GRAPH_JSON):
        data = nx.node_link_data(self.G, edges="links")
        path.write_text(json.dumps(dict(graph=data, chunks=self.chunks, summaries=self.summaries,
                                        built_with=self.llm.name), ensure_ascii=False, indent=1), encoding="utf-8")
        self.export()

    @classmethod
    def load(cls, path: Path = GRAPH_JSON, llm=None, embedder=None):
        d = json.loads(path.read_text(encoding="utf-8"))
        G = nx.node_link_graph(d["graph"], directed=True, multigraph=True, edges="links")
        return cls(G, d["chunks"], {int(k): v for k, v in d["summaries"].items()}, llm, embedder)

    def export(self):
        H = nx.MultiDiGraph()
        for n, a in self.G.nodes(data=True):
            H.add_node(n, type=a.get("type", ""), desc=" | ".join(a.get("desc", [])), community=a.get("community", -1))
        for u, v, a in self.G.edges(data=True):
            H.add_edge(u, v, relation=a.get("relation", ""), desc=a.get("desc", ""))
        nx.write_graphml(H, OUT / "hw4_graph.graphml")
        esc = lambda s: str(s).replace("\\", "\\\\").replace("'", "\\'")
        lines = [f"MERGE (:Entity {{name:'{esc(n)}', type:'{esc(a.get('type',''))}', community:{a.get('community',-1)}, "
                 f"description:'{esc(' | '.join(a.get('desc', [])))}'}});" for n, a in self.G.nodes(data=True)]
        for u, v, a in self.G.edges(data=True):
            rel = re.sub(r"\W", "_", a.get("relation", "RELATED")).upper() or "RELATED"
            lines.append(f"MATCH (a:Entity {{name:'{esc(u)}'}}),(b:Entity {{name:'{esc(v)}'}}) "
                         f"MERGE (a)-[:{rel} {{description:'{esc(a.get('desc',''))}'}}]->(b);")
        (OUT / "hw4_graph.cypher").write_text("\n".join(lines), encoding="utf-8")
        self.plot()

    def plot(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        UG = nx.Graph(self.G)
        top = sorted(UG.degree, key=lambda x: -x[1])[:45]
        H = UG.subgraph([n for n, _ in top])
        pos = nx.spring_layout(H, seed=3, k=0.9)
        col = [self.G.nodes[n].get("community", 0) for n in H]
        plt.figure(figsize=(14, 10))
        nx.draw_networkx_edges(H, pos, alpha=0.25)
        nx.draw_networkx_nodes(H, pos, node_color=col, cmap="tab20", node_size=[120 + 40 * UG.degree(n) for n in H])
        nx.draw_networkx_labels(H, pos, font_size=8)
        plt.title(f"Self-RAG knowledge graph (top-45 nodes by degree; |V|={self.G.number_of_nodes()}, |E|={self.G.number_of_edges()})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(OUT / "hw4_graph.png", dpi=140)
        plt.close()

    # ------------------------------------------------------------ query
    def route(self, q: str) -> str:
        if self.llm.online:
            r = self.llm.generate(ROUTER_PROMPT.format(q=q)).strip().lower()
            return "global" if "global" in r else "local"
        return "global" if re.search(r"整理|總結|摘要|5W1H|pipeline|差異|完整|概述|列出|哪些.*(實驗|任務|機制)", q) else "local"

    def seeds(self, q: str, k: int = 6) -> list[str]:
        ql = q.lower()
        hit = [n for n in self.nodes if len(n) > 2 and n.lower() in ql]
        sims = self.NV @ self.emb.embed([q], kind="query")[0]
        for i in np.argsort(-sims):
            if len(hit) >= k:
                break
            if self.nodes[i] not in hit:
                hit.append(self.nodes[i])
        return hit[:k]

    def local_context(self, q: str, hops: int = 2, max_edges: int = 40) -> tuple[str, list[str]]:
        seeds = self.seeds(q)
        frontier, seen = set(seeds), set(seeds)
        edges = []
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                for u, v, d in list(self.G.out_edges(n, data=True)) + list(self.G.in_edges(n, data=True)):
                    edges.append((u, v, d))
                    o = v if u == n else u
                    if o not in seen:
                        nxt.add(o)
            seen |= nxt
            frontier = nxt
        # 以「關係描述與問題的相似度」排序，只保留最相關的邊（避免子圖爆炸）
        uniq = list({(u, v, d.get("relation"), d.get("desc")): (u, v, d) for u, v, d in edges}.values())
        if uniq:
            ev = self.emb.embed([f"{u} {d.get('relation')} {v} {d.get('desc','')}" for u, v, d in uniq])
            s = ev @ self.emb.embed([q], kind="query")[0]
            s += np.array([0.15 * ((u in seeds) + (v in seeds)) for u, v, _ in uniq])
            uniq = [uniq[i] for i in np.argsort(-s)[:max_edges]]
        ent = "\n".join(f"- {n}（{self.G.nodes[n].get('type','')}）：{'；'.join(self.G.nodes[n].get('desc', [])[:2])}" for n in seeds)
        rel = "\n".join(f"- ({u}) -[{d.get('relation')}]-> ({v})：{d.get('desc','')}" for u, v, d in uniq)
        cids = [c for c, _ in sorted(((d.get("chunk"), 1) for _, _, d in uniq[:8] if d.get("chunk") is not None),
                                      key=lambda x: x[0])]
        cids = list(dict.fromkeys(cids))[:2]
        src = "\n".join(f"[段落{c}] {self.chunks[c]}" for c in cids)
        return f"## 種子實體\n{ent}\n\n## 關係（{hops}-hop 子圖）\n{rel}\n\n## 來源段落\n{src}", seeds

    def ask(self, q: str, mode: str = "auto") -> dict:
        mode = self.route(q) if mode == "auto" else mode
        if mode == "global":
            pts = []
            for ci, s in self.summaries.items():
                if self.llm.online:
                    r = self.llm.generate(MAP_PROMPT.format(summary=s, q=q)).strip()
                    if r and not r.startswith("NONE"):
                        pts += [l for l in r.splitlines() if l.strip()]
                else:
                    pts.append(s)
            pts.sort(key=lambda l: -int(re.search(r"\((\d+)\)", l).group(1)) if re.search(r"\((\d+)\)", l) else 0)
            ctx = "\n".join(pts[:40])
            ans = self.llm.generate(REDUCE_PROMPT.format(points=ctx, q=q)) if self.llm.online else \
                self.llm.generate(f"[1] {ctx}\n【問題】{q}")
            return dict(mode="global", answer=ans.strip(), context=ctx, seeds=[])
        ctx, seeds = self.local_context(q)
        prompt = ANSWER_PROMPT.format(ctx=ctx, q=q) if self.llm.online else \
            "\n".join(f"[{i + 1}] {l}" for i, l in enumerate(ctx.splitlines()) if l.startswith("-") or l.startswith("[段落")) + f"\n【問題】{q}"
        return dict(mode="local", answer=self.llm.generate(prompt).strip(), context=ctx, seeds=seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--pdf", default=None, help="論文 PDF 路徑（例如 data/selfrag.pdf）")
    ap.add_argument("--ask", default=None)
    a = ap.parse_args()
    if a.build or not GRAPH_JSON.exists():
        pdf = a.pdf or (str(ROOT / "data" / "selfrag.pdf") if (ROOT / "data" / "selfrag.pdf").exists() else None)
        g = GraphRAG.build(load_paper(pdf))
        g.save()
        print(f"graph: {g.G.number_of_nodes()} nodes, {g.G.number_of_edges()} edges, {len(g.summaries)} communities")
    if a.ask:
        g = GraphRAG.load()
        r = g.ask(a.ask)
        print(r["mode"], r["seeds"])
        print(r["answer"])


if __name__ == "__main__":
    main()
