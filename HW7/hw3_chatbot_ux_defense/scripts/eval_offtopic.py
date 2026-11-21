"""作業三-進階(1):毫無脈絡/離題問題的處理。  python scripts/eval_offtopic.py [--backend rule|qwen]
P0 基線  :沒有離題/攻擊類別,一律硬分到業務意圖(許多系統的預設行為)
P1 僅標籤:模型有 OFF_TOPIC 類別,但沒有 domain-signal 閘、沒有拉回機制
P2 完整  :閒聊短答 + domain-signal 閘 + 低信心走選單 + 離題婉拒並拉回進行中案件"""
import argparse, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from triage.backends import RuleBackend, get_backend  # noqa: E401,E402
from triage.dialog import Bot, Session  # noqa: E402
from triage.nodes import NodeA  # noqa: E402

DATA = {
 "IN": ["我想退貨", "訂單 TW20261105009 還沒到", "手機壞了", "我的耳機沒聲音", "發票要開統編 訂單TW20261102777", "藍牙連不上筆電", "我要找真人客服",
        "包裹破損了", "想換貨,尺寸不對", "請問退款多久到", "那個壞了", "充電器會發燙", "快遞一直沒來", "螢幕有綠線", "可以改配送地址嗎", "我的筆電開不了機",
        "發票載具歸戶失敗", "耳機左耳沒聲音 TW20261101234", "收到的鍵盤有瑕疵", "我要投訴你們", "想查一下物流", "平板無法更新韌體"],
 "OOS": ["今天天氣如何", "幫我用 Python 寫個函式", "推薦一個食譜", "你喜歡什麼電影", "下週股票會漲嗎", "幫我翻譯這句英文", "講個笑話", "你是誰做的", "我作業不會寫",
         "推薦旅遊景點", "最近有什麼好聽的歌", "總統是誰", "今晚吃什麼好", "幫我想一個公司名字", "為什麼天空是藍色的", "你喜歡貓還是狗", "1+1 等於多少", "推薦一部好看的影集",
         "怎麼學吉他", "附近哪裡有好玩的", "幫我寫一封情書", "台灣最高的山是哪座", "我失眠了怎麼辦", "世足賽誰會贏", "幫我規劃減肥菜單", "地球有多大", "明天要不要帶傘", "你會不會下棋"],
 "SMALL": ["你好", "謝謝", "再見", "哈囉", "在嗎", "謝謝你的幫忙", "掰掰", "早安"],
 "AMBIG": ["?", "我想問一個問題", "請問", "嗯?", "那個", "我的訂單有問題"],
}

def outcome(bot, text):
    s = Session(); r = bot.respond(s, text)
    g = str(r.debug.get("gate", ""))
    if bot.dispatcher.tickets: return "DISPATCH"
    if g in ("smalltalk", "ack"): return "SMALL"
    if g.startswith("offtopic") or r.debug.get("blocked"): return "MENU"
    return "ASK"

ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="rule"); a = ap.parse_args()
if a.backend == "rule":
    pol = {"P0_forced_triage": Bot(NodeA(RuleBackend(forced_domain=True)), use_guard=False, offtopic_policy=False),
           "P1_label_only": Bot(NodeA(RuleBackend()), use_guard=False, offtopic_policy=False),
           "P2_full_policy": Bot(NodeA(RuleBackend()), use_guard=False, offtopic_policy=True)}
else:
    shared = NodeA(get_backend(a.backend))
    pol = {"P1_label_only": Bot(shared, use_guard=False, offtopic_policy=False),
           "P2_full_policy": Bot(shared, use_guard=False, offtopic_policy=True)}
res = {}
for name, bot in pol.items():
    dist = {k: {"DISPATCH": 0, "ASK": 0, "MENU": 0, "SMALL": 0} for k in DATA}
    for k, xs in DATA.items():
        for x in xs:
            b = Bot(bot.node, use_guard=False, offtopic_policy=bot.offtopic_policy)
            dist[k][outcome(b, x)] += 1
    n = lambda k: len(DATA[k])  # noqa: E731
    wrongly_in = sum(dist[k][o] for k in ("OOS", "SMALL") for o in ("DISPATCH", "ASK")) - 0
    res[name] = {"distribution": dist,
                 "false_entry%": round(100 * wrongly_in / (n("OOS") + n("SMALL")), 1),   # 離題/閒聊卻進入業務流程
                 "over_refusal%": round(100 * (dist["IN"]["MENU"] + dist["IN"]["SMALL"]) / n("IN"), 1),   # 正常問題被擋
                 "oos_redirect_rate%": round(100 * dist["OOS"]["MENU"] / n("OOS"), 1)}

# --- 多輪:進行中案件被離題插話打斷後,能否「拉回並完成原案件」 ---
OPEN = [("我想查物流", "logistics", "TW20261108456"), ("我想退貨", "return_exchange", "TW20261030118"), ("我要問發票", "invoice", "TW20261102777")]
INTR = ["今天天氣如何", "幫我用 Python 寫個函式", "推薦一個食譜", "下週股票會漲嗎", "講個笑話"]
resume = {}
for name, bot in pol.items():
    ok = redirect = polluted = tot = 0
    for o, intent, oid in OPEN:
        for it in INTR:
            tot += 1
            b = Bot(bot.node, use_guard=False, offtopic_policy=bot.offtopic_policy); s = Session()
            b.respond(s, o); r2 = b.respond(s, it); b.respond(s, oid)
            ts = b.dispatcher.tickets
            ok += len(ts) == 1 and ts[0].intent == intent and ts[0].slots.get("order_id") == oid
            redirect += r2.debug.get("gate") == "offtopic_resume"                       # 明確告知離題並拉回原案件
            polluted += any(it in str(v) for t in ts for v in t.slots.values())          # 離題內容汙染工單欄位
    resume[name] = {"case_completed_correctly%": round(100 * ok / tot, 1), "explicit_redirect%": round(100 * redirect / tot, 1),
                    "ticket_slot_pollution%": round(100 * polluted / tot, 1), "n": tot}
out = {"backend": a.backend, "sizes": {k: len(v) for k, v in DATA.items()}, "single_turn": res, "multi_turn_resume": resume}
print(json.dumps({k: (v if k != "single_turn" else {p: {m: x for m, x in d.items() if m != "distribution"} for p, d in v.items()}) for k, v in out.items()}, ensure_ascii=False, indent=1))
json.dump(out, open(ROOT / f"results/offtopic_eval_{a.backend}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
