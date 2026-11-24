"""作業一進階 (5)：User Interface。

python -m hw1.app   → 瀏覽器開 http://127.0.0.1:7860
功能：多輪對話、顯示 Query Rewriting 結果、每則回答的 Reference（含出處座標與檢索分數）、
      可即時切換 Prompting Technique / Top-K / 是否改寫，並可一鍵清除對話。
"""
from __future__ import annotations

import html
import re

import gradio as gr

from common.backends import has_gemini
from common.rag import PROMPTS, build_default

BOT = build_default()

TECH_LABEL = {"grounded": "Grounded（角色+規則+引用）", "cot": "Chain-of-Thought", "fewshot": "Few-shot",
              "stepback": "Step-Back", "plain": "Plain（baseline）"}

CSS = """
.ref-card{border:1px solid #d0d7de;border-radius:8px;padding:8px 10px;margin:6px 0;background:#f8fafc;font-size:14px}
.ref-card b{color:#1f4e79}
.ref-meta{color:#57606a;font-size:12px}
.rewrite{background:#fff8e1;border-left:4px solid #f0b400;padding:6px 10px;border-radius:4px}
"""


def render_refs(turn) -> str:
    used = set(int(n) for n in re.findall(r"\[(\d+)\]", turn.answer))
    cards = []
    for r in turn.refs:
        mark = "✅ 已引用" if r["n"] in used else "未引用"
        cards.append(f"<div class='ref-card'><b>[{r['n']}] {html.escape(r['ref'])}</b> "
                     f"<span class='ref-meta'>{r['kind']}｜score={r['score']}｜{mark}</span><br>"
                     f"{html.escape(r['text'])}</div>")
    rw = ""
    if turn.rewritten != turn.user:
        rw = f"<div class='rewrite'>🔁 多輪改寫：<b>{html.escape(turn.rewritten)}</b></div>"
    return rw + "".join(cards)


def chat(msg, history, technique, top_k, rewrite):
    if not msg.strip():
        return history, "", gr.update()
    BOT.technique, BOT.top_k, BOT.rewrite = technique, int(top_k), rewrite
    turn = BOT.ask(msg)
    history = history + [{"role": "user", "content": msg}, {"role": "assistant", "content": turn.answer}]
    return history, "", render_refs(turn)


def reset():
    BOT.reset()
    return [], "", ""


def build():
    status = ("🟢 Gemini（" + BOT.llm.name + "）" if has_gemini() else "🟡 離線模式（未設定 GEMINI_API_KEY，回答為抽取式）")
    with gr.Blocks(title="影像處理助教 RAG") as demo:
        gr.Markdown(f"## 🖼️ 數位影像處理 助教 — 多輪對話 RAG\n{status}｜索引內 Embedding 數：**{BOT.index.n_embeddings}**"
                    f"（Embedder：{BOT.index.emb.name}）")
        with gr.Row():
            with gr.Column(scale=3):
                cb = gr.Chatbot(height=520, label="對話")
                with gr.Row():
                    box = gr.Textbox(placeholder="輸入問題，例如：Canny 邊緣偵測的流程是什麼？", scale=5, show_label=False)
                    send = gr.Button("送出", variant="primary", scale=1)
                gr.Examples(["Canny 邊緣偵測的流程是什麼？", "其中非極大值抑制是做什麼的？", "那 JPEG 2000 改了什麼？",
                             "照片有很多黑白小點要用什麼濾波？", "CLIP 怎麼做零樣本分類？"], inputs=box)
            with gr.Column(scale=2):
                with gr.Accordion("⚙️ 設定", open=True):
                    tech = gr.Dropdown([(v, k) for k, v in TECH_LABEL.items()], value=BOT.technique, label="Prompting Technique")
                    topk = gr.Slider(1, 10, value=BOT.top_k, step=1, label="Top-K")
                    rw = gr.Checkbox(value=True, label="多輪 Query Rewriting（補代名詞）")
                    clr = gr.Button("🗑️ 清除對話")
                gr.Markdown("### 📚 Reference")
                refs = gr.HTML()
        send.click(chat, [box, cb, tech, topk, rw], [cb, box, refs])
        box.submit(chat, [box, cb, tech, topk, rw], [cb, box, refs])
        clr.click(reset, None, [cb, box, refs])
    return demo


if __name__ == "__main__":
    build().launch(css=CSS, theme=gr.themes.Soft())
