"""整合介面：python app.py → http://127.0.0.1:7860
分頁 1：作業一 多輪對話 RAG（影像處理助教）
分頁 2：作業二 多輪對話 VideoRAG（需先 python -m hw2.videorag --ingest）
分頁 3：作業四 GraphRAG（Self-RAG 論文問答，需先 python -m hw4.graphrag --build）
"""
from __future__ import annotations

import html

import gradio as gr

from common.backends import ROOT, has_gemini
from hw1 import app as rag_app
from hw2.videorag import STORE

VR = None
GR = None


def video_chat(msg, history):
    global VR
    if not msg.strip():
        return history, "", "", []
    if VR is None:
        from hw2.videorag import VideoRAG

        VR = VideoRAG()
    out = VR.ask(msg)
    history = history + [{"role": "user", "content": msg}, {"role": "assistant", "content": out["answer"]}]
    rw = f"<div class='rewrite'>🔁 多輪改寫：<b>{html.escape(out['rewritten'])}</b></div>" if out["rewritten"] != msg else ""
    refs = "".join(f"<div class='ref-card'><b>🎬 {html.escape(r['video'])}　{r['start']}–{r['end']}</b> "
                   f"<span class='ref-meta'>{r['modality']}｜score={r['score']}</span></div>" for r in out["refs"])
    rank = "、".join(f"{v}({s})" for v, s in out["video_ranking"][:3])
    gallery = [(str(ROOT / f["path"]), f"{f['file']} @ {int(f['t'])//60:02d}:{int(f['t'])%60:02d}") for f in out["frames"]]
    return history, "", rw + f"<div class='ref-meta'>影片排名：{html.escape(rank)}</div>" + refs, gallery


def video_reset():
    if VR is not None:
        VR.reset()
    return [], "", "", []


def graph_ask(q, mode):
    global GR
    if GR is None:
        from hw4.graphrag import GraphRAG

        GR = GraphRAG.load()
    r = GR.ask(q, mode)
    return r["answer"], f"**路由：{r['mode']}**｜種子實體：{', '.join(r['seeds'])}\n\n```\n{r['context'][:4000]}\n```"


def build():
    status = "🟢 Gemini 已連線" if has_gemini() else "🟡 離線模式（未設定 GEMINI_API_KEY）"
    with gr.Blocks(title="MMIP Week10 RAG 作業") as demo:
        gr.Markdown(f"# MMIP 第十週：RAG / VideoRAG / GraphRAG　{status}")
        with gr.Tab("作業一：多輪 RAG"):
            inner = rag_app.build()
            inner.render() if hasattr(inner, "render") else None
        with gr.Tab("作業二：VideoRAG"):
            if not (STORE / "catalog.json").exists():
                gr.Markdown("⚠️ 尚未建立影片庫：把影片放進 `data/videos/` 後執行 `python -m hw2.videorag --ingest`")
            with gr.Row():
                with gr.Column(scale=3):
                    vcb = gr.Chatbot(height=460, label="影片對話")
                    vbox = gr.Textbox(placeholder="問影片內容，例如：影片中怎麼新增圖層？", show_label=False)
                    vclr = gr.Button("🗑️ 清除對話")
                with gr.Column(scale=2):
                    gr.Markdown("### 🎞️ 選出的影格（Frame Selection）")
                    gal = gr.Gallery(columns=2, height=300)
                    gr.Markdown("### 📚 Reference")
                    vrefs = gr.HTML()
            vbox.submit(video_chat, [vbox, vcb], [vcb, vbox, vrefs, gal])
            vclr.click(video_reset, None, [vcb, vbox, vrefs, gal])
        with gr.Tab("作業四：GraphRAG"):
            with gr.Row():
                gq = gr.Textbox(label="關於 Self-RAG 論文的問題", value="請用 5W1H 整理這篇論文說了什麼。", scale=4)
                gm = gr.Radio(["auto", "local", "global"], value="auto", label="模式", scale=1)
            gbtn = gr.Button("查詢", variant="primary")
            gans = gr.Markdown()
            with gr.Accordion("檢索到的子圖 / 社群摘要", open=False):
                gctx = gr.Markdown()
            if (ROOT / "results" / "hw4_graph.png").exists():
                gr.Image(str(ROOT / "results" / "hw4_graph.png"), label="知識圖譜（度數前 45 的節點）")
            gbtn.click(graph_ask, [gq, gm], [gans, gctx])
    return demo


if __name__ == "__main__":
    build().launch(css=rag_app.CSS, theme=gr.themes.Soft())
