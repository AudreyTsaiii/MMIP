"""作業一基礎 (5)：擴充 QA Dataset，讓 Embedding 數量超過門檻（200）。

手寫 120 組 QA（data/qa_dataset.jsonl，同時是評估用的標準答案）之外，
依講義第 12 頁「將文本轉換成 QA 形式，並試著加入 5W1H 元素使 Q 更貼近使用情境」，
讓 Gemini 對每篇文件產生 5W1H 風格的問答，並檢查證據句確實出自原文（避免幻覺）。
輸出 data/qa_generated.jsonl，建索引時會自動併入（評估仍只用手寫的 120 題）。

用法：python -m hw1.augment_qa --per-doc 6
"""
from __future__ import annotations

import argparse
import json

from common.backends import ROOT, get_llm
from common.rag import load_docs

PROMPT = """以下是「數位影像處理」課程的一篇講義。請產生 {n} 組問答，讓學生在實際情境中會問的問題能被檢索到。
規則：
1. 問題要涵蓋 5W1H（What/Why/When/Where/Who/How）不同角度，並且有一半寫成學生的口語化情境，例如「我拍的夜景很暗要怎麼調亮？」
2. 答案 1~2 句，只能根據講義內容。
3. evidence 必須「一字不差」複製講義中支持答案的一段文字（10~40 字）。
4. 不要和這些既有問題重複：{existing}
輸出 JSON 陣列：[{{"w":"5W1H 類別","question":"...","answer":"...","evidence":"..."}}]

【講義標題】{title}
【講義內容】{text}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=6)
    a = ap.parse_args()
    llm = get_llm()
    assert llm.online, "需要 GEMINI_API_KEY"
    docs = {k: v for k, v in load_docs().items()}
    seed = [json.loads(l) for l in (ROOT / "data" / "qa_dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    out, kept, dropped = [], 0, 0
    for did, d in docs.items():
        existing = "；".join(q["question"] for q in seed if q["doc_id"] == did)
        items = llm.generate_json(PROMPT.format(n=a.per_doc, existing=existing, title=d["title"], text=d["text"])) or []
        for it in items:
            if not isinstance(it, dict) or it.get("evidence", "") not in d["text"] or not it.get("question"):
                dropped += 1
                continue
            kept += 1
            out.append(dict(id=f"G{len(out) + 1:03d}", doc_id=did, doc_title=d["title"], question=it["question"],
                            question_colloquial="", answer=it["answer"], evidence=it["evidence"],
                            w=it.get("w", ""), generated=True))
    p = ROOT / "data" / "qa_generated.jsonl"
    p.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
    print(f"產生 {kept} 組（證據不在原文而丟棄 {dropped} 組）→ {p}")


if __name__ == "__main__":
    main()
