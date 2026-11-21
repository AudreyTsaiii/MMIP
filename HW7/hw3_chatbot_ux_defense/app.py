"""Gradio 多模態多輪客服分流機器人。
  python app.py                                 # 離線規則後端(無 GPU 也能體驗對話流程/防禦/UX)
  TRIAGE_BACKEND=qwen TRIAGE_NODE=D python app.py   # 真 VLM + System One 分流
  TRIAGE_BACKEND=qwen TRIAGE_NODE=B TRIAGE_ADAPTER=checkpoints/sft_lora python app.py
"""
import os
import time

import gradio as gr

from triage.backends import RuleBackend, get_backend
from triage.dialog import Bot, Session, reading_delay
from triage.nodes import NodeA, NodeB, NodeC, NodeD, SafeNode

KIND, NODE, ADAPTER = os.getenv("TRIAGE_BACKEND", "rule"), os.getenv("TRIAGE_NODE", "D"), os.getenv("TRIAGE_ADAPTER")
MODEL = os.getenv("TRIAGE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")   # 可填本機路徑(離線環境)


def build_bot() -> Bot:
    fallback = NodeA(RuleBackend())
    if KIND == "rule":
        return Bot(fallback)
    be = get_backend("qwen", model_id=MODEL, adapter=ADAPTER) if NODE == "B" else get_backend("qwen", model_id=MODEL)
    primary = {"A": NodeA, "B": NodeB, "C": NodeC, "D": NodeD}[NODE](be)
    return Bot(SafeNode(primary, fallback))


BOT = build_bot()
PACE_DEFAULT = float(os.getenv("TRIAGE_PACE", "1.0"))
WELCOME = "您好!我是小3C客服。可以協助訂單、物流、退換貨、維修與發票。有商品照片也可以直接上傳喔。"


def _qr(quick):
    quick = (quick or [])[:6]
    return [gr.update(value=(quick[i] if i < len(quick) else ""), visible=i < len(quick)) for i in range(6)]


def submit(msg, history, session, pace_on):
    text = (msg or {}).get("text", "") if isinstance(msg, dict) else (msg or "")
    files = [f if isinstance(f, str) else f.get("path") for f in ((msg or {}).get("files", []) if isinstance(msg, dict) else [])]
    history = list(history or [])
    for f in files:
        history.append({"role": "user", "content": {"path": f}})
    if text:
        history.append({"role": "user", "content": text})
    yield history, session, gr.update(value=None, interactive=False), *_qr([]), {}
    reply = BOT.respond(session, text, files)
    for i, b in enumerate(reply.bubbles):
        if pace_on:                                   # 依「上一則」長度估閱讀時間,再出下一則,避免訊息轟炸
            if i > 0:
                history.append({"role": "assistant", "content": "⋯"})
                yield history, session, gr.update(), *_qr([]), {}
                time.sleep(reading_delay(reply.bubbles[i - 1]) * PACE_DEFAULT)
                history.pop()
            else:
                time.sleep(0.4)
        history.append({"role": "assistant", "content": b})
        yield history, session, gr.update(), *_qr([]), {}
    dbg = {"debug": reply.debug, "tickets": [t.__dict__ for t in BOT.dispatcher.tickets if t.id in session.ticket_ids], "pending": len(session.pending)}
    yield history, session, gr.update(value=None, interactive=True), *_qr(reply.quick_replies), dbg


with gr.Blocks(title="小3C 客服分流機器人") as demo:
    gr.Markdown("## 🛍️ 小3C 客服\n多輪、可上傳照片;背後會把需求結構化並分流到對應部門。")
    chat = gr.Chatbot(value=[{"role": "assistant", "content": WELCOME}], height=460, label="對話")
    with gr.Row():
        qbtn = [gr.Button(visible=False, size="sm") for _ in range(6)]
    box = gr.MultimodalTextbox(file_types=["image"], placeholder="輸入問題,或上傳商品/包裹照片…", show_label=False)
    pace = gr.Checkbox(value=True, label="依閱讀時間分段送出(UX)")
    with gr.Accordion("🔧 內部狀態(開發者檢視:分流結果 / 防禦 / 工單)", open=False):
        dbg = gr.JSON()
    state = gr.State(Session())
    outs = [chat, state, box, *qbtn, dbg]
    box.submit(submit, [box, chat, state, pace], outs)
    for b in qbtn:
        b.click(lambda v: {"text": v, "files": []}, b, box, queue=False).then(submit, [box, chat, state, pace], outs)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))