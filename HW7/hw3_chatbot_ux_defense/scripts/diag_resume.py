"""診斷多輪『離題後拉回』為何真模型只有 33%。  python scripts/diag_resume.py [--backend rule|qwen]
逐案印出:開案句 → 插話 → 訂單編號 三步各自的 gate / 回覆 / 最終工單。"""
import argparse, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from triage.backends import get_backend  # noqa: E402
from triage.dialog import Bot, Session  # noqa: E402
from triage.nodes import NodeA  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="rule"); a = ap.parse_args()
node = NodeA(get_backend(a.backend))
OPEN = [("我想查物流", "logistics", "TW20261108456"), ("我想退貨", "return_exchange", "TW20261030118"), ("我要問發票", "invoice", "TW20261102777")]
INTR = ["今天天氣如何", "幫我用 Python 寫個函式", "推薦一個食譜", "下週股票會漲嗎", "講個笑話"]
for o, intent, oid in OPEN:
    for it in INTR:
        b = Bot(node, use_guard=False, offtopic_policy=True); s = Session()
        r1 = b.respond(s, o); r2 = b.respond(s, it); r3 = b.respond(s, oid)
        ts = b.dispatcher.tickets
        ok = len(ts) == 1 and ts[0].intent == intent and ts[0].slots.get("order_id") == oid
        print(f"{'OK ' if ok else 'BAD'} | {o} → {it} → {oid}")
        print(f"     step1 gate={r1.debug.get('gate')!s:16} 回:{(r1.bubbles[0] if r1.bubbles else '')[:40]!r}")
        print(f"     step2 gate={r2.debug.get('gate')!s:16} 回:{(r2.bubbles[0] if r2.bubbles else '')[:40]!r}")
        print(f"     step3 gate={r3.debug.get('gate')!s:16} 回:{(r3.bubbles[0] if r3.bubbles else '')[:40]!r}")
        print(f"     tickets={[(t.intent, t.slots.get('order_id')) for t in ts]}")