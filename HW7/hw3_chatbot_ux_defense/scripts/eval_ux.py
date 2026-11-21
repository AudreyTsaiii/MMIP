"""作業三-基礎(3):UX 優化「話題銜接」。  python scripts/eval_ux.py
基線:處理完一件事只問「還有其他問題嗎?」,其餘需求要使用者自己重提、重填訂單編號。
優化:同訊息多需求/中途插入新需求 → pending 佇列;處理完主動銜接、沿用已知訂單編號(只需點「是」)。
以腳本化的模擬使用者量測(不是真人實驗):使用者會回答機器人問的問題;若機器人沒主動接續,才自己重提。"""
import json, pathlib, statistics as st, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from triage.backends import RuleBackend  # noqa: E402
from triage.dialog import Bot, Session  # noqa: E402
from triage.nodes import NodeA  # noqa: E402

RE = {"invoice": "我還有發票要處理", "logistics": "我還想查物流", "return_exchange": "我還要退貨", "defect_repair": "我的商品壞了"}
D = [  # (opening, goals, oid, product, interject)
 ("我想退貨,另外發票也要改統編,訂單 TW20261103999", ["return_exchange", "invoice"], "TW20261103999", "耳機", None),
 ("訂單 TW20261104001 物流一直沒更新,而且我還想開發票", ["logistics", "invoice"], "TW20261104001", "耳機", None),
 ("耳機壞了想退貨,訂單 TW20261104002", ["return_exchange", "defect_repair"], "TW20261104002", "耳機", None),
 ("我要換貨,另外幫我查一下物流 TW20261104003", ["return_exchange", "logistics"], "TW20261104003", "耳機", None),
 ("發票要開統編,還有包裹沒收到 訂單TW20261104004", ["invoice", "logistics"], "TW20261104004", "耳機", None),
 ("手機螢幕裂了要報修,順便問發票 TW20261104005", ["defect_repair", "invoice"], "TW20261104005", "手機", None),
 ("想退貨也想查物流,還要發票,單號 TW20261104006", ["return_exchange", "logistics", "invoice"], "TW20261104006", "耳機", None),
 ("筆電壞了,同時想問發票和退款,訂單TW20261104007", ["defect_repair", "invoice", "return_exchange"], "TW20261104007", "筆電", None),
 ("我想查物流", ["logistics", "invoice"], "TW20261104008", "耳機", "對了,發票也要處理"),
 ("我想退貨", ["return_exchange", "defect_repair"], "TW20261104009", "耳機", "還有我的耳機壞了"),
 ("我要問發票", ["invoice", "logistics"], "TW20261104010", "耳機", "另外物流也沒更新"),
 ("我想換貨", ["return_exchange", "logistics"], "TW20261104011", "耳機", "我還要查包裹到哪"),
]

def simulate(topic_transition: bool, opening, goals, oid, product, interject, cap=16):
    bot = Bot(NodeA(RuleBackend()), use_guard=False, topic_transition=topic_transition); s = Session()
    msg, turns, reraise, oid_given, interjected = opening, 0, 0, 1 if oid in opening else 0, False
    done = lambda: set(goals) <= {t.intent for t in bot.dispatcher.tickets}  # noqa: E731
    while turns < cap:
        turns += 1
        r = bot.respond(s, msg); text = " ".join(r.bubbles)
        if done(): break
        asked_q = any(k in text for k in ("訂單編號", "哪一項商品", "說明遇到的狀況", "同一筆訂單"))
        if interject and not interjected and asked_q:
            interjected, msg = True, interject
        elif "同一筆訂單" in text: msg = "是"
        elif "訂單編號" in text: msg = oid; oid_given += 1
        elif "哪一項商品" in text: msg = product
        elif "說明遇到的狀況" in text: msg = "螢幕出現問題,畫面異常"
        else:
            left = [g for g in goals if g not in {t.intent for t in bot.dispatcher.tickets}]
            if not left: break
            msg = RE[left[0]]; reraise += 1
    return {"solved": done(), "user_turns": turns, "user_reraises": reraise, "order_id_typed": oid_given}

out = {}
for name, tt in (("baseline_no_transition", False), ("with_topic_transition", True)):
    rs = [simulate(tt, *d) for d in D]
    out[name] = {"dialogues": len(rs), "all_issues_solved%": round(100 * sum(r["solved"] for r in rs) / len(rs), 1),
                 "avg_user_turns": round(st.mean(r["user_turns"] for r in rs), 2),
                 "avg_user_reraises": round(st.mean(r["user_reraises"] for r in rs), 2),
                 "avg_order_id_typed": round(st.mean(r["order_id_typed"] for r in rs), 2),
                 "per_dialogue_turns": [r["user_turns"] for r in rs]}
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(ROOT / "results/ux_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
