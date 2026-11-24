"""作業一基礎 (10)：展示多輪對話 Context 經過 RAG 後得到合理回覆與 Reference。

把 data/multiturn_dialogs.json 的 10 段對話實際跑一遍，輸出 results/hw1_multiturn_demo.md：
每一輪列出 使用者原句 → 改寫後的獨立問題 → 回答 → Reference（出處座標），並標示 Reference 是否命中標準答案來源。
"""
from __future__ import annotations

import json

from common.backends import ROOT
from common.rag import build_default, load_qa


def main():
    bot = build_default()
    qa = {q["id"]: q for q in load_qa()}
    dialogs = json.loads((ROOT / "data" / "multiturn_dialogs.json").read_text(encoding="utf-8"))
    lines = [f"# 作業一：多輪對話 RAG 展示\n",
             f"- LLM：`{bot.llm.name}`｜Embedder：`{bot.index.emb.name}`｜索引 Embedding 數：**{bot.index.n_embeddings}**",
             f"- 設定：top_k={bot.top_k}，Prompting=`{bot.technique}`，多輪 Query Rewriting=開\n"]
    hit_total = n = 0
    for d in dialogs:
        bot.reset()
        lines.append(f"## 對話 {d['id']}\n")
        for k, t in enumerate(d["turns"]):
            r = bot.ask(t["q"])
            gold = qa[t["gold"]]
            hit = any(x["uid"].startswith(gold["id"]) or (x["kind"] == "chunk" and gold["evidence"] in x["text"]) for x in r.refs)
            hit_total += hit
            n += 1
            lines.append(f"**第 {k + 1} 輪｜使用者：** {t['q']}  ")
            if r.rewritten != t["q"]:
                lines.append(f"🔁 改寫：{r.rewritten}  ")
            lines.append(f"**助教：** {r.answer}\n")
            lines.append("<details><summary>Reference（" + ("✅ 命中標準答案來源" if hit else "❌ 未命中") + "）</summary>\n")
            for x in r.refs:
                lines.append(f"- [{x['n']}] {x['ref']}（score {x['score']}）：{x['text'][:120]}…")
            lines.append("\n</details>\n")
    lines.insert(3, f"- **Reference 命中率：{hit_total}/{n} = {hit_total / n:.1%}**\n")
    (ROOT / "results" / "hw1_multiturn_demo.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Reference 命中 {hit_total}/{n}")


if __name__ == "__main__":
    main()
