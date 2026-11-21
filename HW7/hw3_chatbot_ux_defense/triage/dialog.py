"""多輪對話管理(狀態機)+ UX 策略。

重點設計
1. 使用者可見的回覆 100% 由程式模板產生;LLM 只輸出結構化資料 → 結構性防禦 prompt injection。
2. 槽位累積(跨輪合併);一次只追問一個缺漏欄位;圖片只保留最近 2 張進 context,其餘濃縮成 image_findings。
3. UX-話題銜接:同一則訊息/對話中出現多個需求 → 進 pending 佇列;處理完一個主動銜接下一個,並沿用已知訂單編號。
4. 離題:閒聊短答、離題婉拒+拉回進行中的案件、無脈絡片段改問選單,而不是亂派工。
5. 攻擊:guard 攔截 → 不進 history(避免污染上下文)→ strike 計數,第 3 次起冷卻並轉真人。
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .backends import KEYWORDS, Message
from .guard import DOMAIN_HINT, InputGuard, OutputGuard, Verdict
from .nodes import Node
from .router import Dispatcher, Ticket
from .schema import ASK_TEMPLATE, ORDER_ID_RE, REQUIRED_SLOTS, Intent

INTENT_ZH = {
    Intent.RETURN_EXCHANGE: "退換貨", Intent.LOGISTICS: "物流查詢", Intent.DEFECT_REPAIR: "報修/瑕疵處理",
    Intent.INVOICE: "發票", Intent.TECH_SUPPORT: "技術支援", Intent.HUMAN_HANDOFF: "真人服務",
}
MENU = ["查物流", "退換貨", "商品故障/報修", "發票", "找真人"]
MENU_TO_TEXT = {"查物流": "我想查物流", "退換貨": "我想退換貨", "商品故障/報修": "我的商品壞了", "發票": "我想問發票", "找真人": "我要找真人客服"}
SWITCH_CUE = re.compile(r"(算了|改成|不要了|還是|其實|換成|改為|不查了)")
YES = re.compile(r"^(是|對|對的|沒錯|是的|好|好的|可以|嗯|yes|y|ok)[!!。~]*$", re.I)
NO = re.compile(r"^(不是|不對|不用|不|否|no|n|不要)[!!。~]*$", re.I)
SMALLTALK = [
    (re.compile(r"^(你好|您好|哈囉|嗨|hi|hello|早安|午安|晚安|在嗎|在不在)[!!。~\s]*$", re.I), "greet"),
    (re.compile(r"^(謝謝|感謝|多謝|謝啦|thx|thanks)"), "thanks"),
    (re.compile(r"^(再見|掰掰|拜拜|bye|沒有了|沒事了|沒其他了|不用了)"), "bye"),
]
TPL = {
    "greet": ["您好!我是小3C客服。可以協助訂單、物流、退換貨、維修與發票,請問今天需要什麼協助?"],
    "thanks": ["不客氣!還有其他需要我協助的嗎?"],
    "bye": ["好的,祝您使用愉快,有需要隨時找我!"],
    "empty": ["我沒有收到內容喔,請輸入問題,或上傳商品/包裹照片。"],
    "menu": ["這個部分我沒辦法協助,我主要處理 3C 商品的訂單、物流、退換貨、維修與發票。請問想處理哪一項?"],
    "menu_blank": ["可以多說一點嗎?我可以協助訂單、物流、退換貨、維修與發票,請問想處理哪一項?"],
    "offtopic_resume": ["這部分我沒辦法協助。回到剛剛的「{case}」,{ask}"],
    "offtopic_many": ["看起來您可能需要的不是我們客服能協助的內容。若有訂單相關問題隨時告訴我,也可以為您轉接真人客服。"],
    "block1": ["這個要求我沒辦法處理。若是訂單、物流、退換貨或維修問題,請直接告訴我。"],
    "block2": ["我無法依照這類指示操作,請描述您遇到的商品或訂單問題。再次提醒,我只能處理客服相關需求。"],
    "block_lock": ["因為多次收到無法處理的要求,為保護您的帳號與系統安全,後續將由真人客服協助您。"],
    "abuse_block": ["很抱歉讓您不愉快。如果是商品或訂單有狀況,請告訴我細節,我會盡快幫您處理;需要的話也可以轉真人客服。"],
    "deescalate": ["非常抱歉造成您的困擾,我會優先處理。"],
    "bad_image": ["這張圖片我無法讀取(格式或大小不符),可以換一張 JPG/PNG、8MB 以內的照片嗎?"],
    "fail": ["系統暫時無法判斷您的需求,已為您轉交真人客服,不需要重複說明。"],
    "dispatched": ["已為您建立「{dept}」工單 #{tid},預計 {sla} 小時內回覆。"],
    "dispatched_queued": ["已受理並保存您的需求(單號 #{tid}),目前下游系統忙碌,恢復後會自動送出,不需要重新說明。"],
    "next_confirm": ["接著處理您剛提到的「{case}」。要沿用同一筆訂單 {oid} 嗎?"],
    "next_ask": ["接著處理您剛提到的「{case}」。{ask}"],
    "anything_else": ["還有其他需要我協助的嗎?"],
    "defer": ["好的,我先把「{case}」記下來,等目前的「{cur}」處理完就接著幫您處理。"],
    "confirm_order": ["請問是同一筆訂單 {oid} 嗎?"],
}


def reading_delay(text: str, cps: float = 8.0, lo: float = 0.8, hi: float = 6.0) -> float:
    """依字數估計使用者閱讀時間(中文約 8 字/秒);夾在 [lo, hi] 秒。"""
    return max(lo, min(hi, 0.4 + len(text) / cps))


def split_bubbles(text: str, max_len: int = 60) -> list[str]:
    """長回覆切成最多 3 個氣泡(以句號/問號斷句),每個氣泡獨立計算閱讀時間。"""
    if len(text) <= max_len:
        return [text]
    parts = [p for p in re.split(r"(?<=[。!?!?])", text) if p.strip()]
    bubbles, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > max_len and len(bubbles) < 2:
            bubbles.append(cur); cur = p
        else:
            cur += p
    return bubbles + [cur] if cur else bubbles


@dataclass
class Case:
    intent: Intent
    slots: dict = field(default_factory=lambda: {"order_id": None, "product": None, "issue_summary": None, "image_findings": None})
    urgency: str = "low"
    ticket: Optional[Ticket] = None
    asked: int = 0
    awaiting: Optional[tuple] = None   # ("confirm_order", oid)

    @property
    def collecting(self) -> bool:
        return self.ticket is None

    def missing(self) -> list[str]:
        out = []
        for k in REQUIRED_SLOTS[self.intent]:
            v = self.slots.get(k)
            if k == "issue_summary":      # 太短(例如只回「手機」)不算有描述問題
                if (v and len(v) >= 3) or self.slots.get("image_findings"):
                    continue
            elif v:
                continue
            out.append(k)
        return out

    def merge(self, new: dict):
        for k, v in new.items():
            if not v:
                continue
            if k == "issue_summary" and self.slots.get(k) and len(v) <= 6:
                continue                      # 避免「手機」這類短回答覆蓋原本的問題描述
            if k == "image_findings" and self.slots.get(k) and v not in self.slots[k]:
                self.slots[k] = (self.slots[k] + ";" + v)[:80]
            else:
                self.slots[k] = v


@dataclass
class Session:
    history: list[Message] = field(default_factory=list)
    active: Optional[Case] = None
    pending: deque = field(default_factory=deque)       # deque[Case]
    last_order_id: Optional[str] = None
    strikes: int = 0
    offtopic_streak: int = 0
    locked: bool = False
    turn: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=6))   # 此對話最近訊息(洪水偵測用)
    ticket_ids: list = field(default_factory=list)                    # 此對話建立的工單(UI 面板只顯示自己的)


@dataclass
class BotReply:
    bubbles: list[str]
    quick_replies: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def domain_signal(text: str, has_image: bool) -> bool:
    if has_image or ORDER_ID_RE.search(text) or DOMAIN_HINT.search(text):
        return True
    return any(k in text for i, ks in KEYWORDS.items() if i != Intent.OFF_TOPIC for k in ks)


def secondary_intents(text: str, primary: Intent) -> list[Intent]:
    """同一句話提到的其他需求(詞彙法;生產環境可改由模型輸出)。"""
    return [i for i, ks in KEYWORDS.items()
            if i not in (primary, Intent.OFF_TOPIC, Intent.HUMAN_HANDOFF) and any(k in text for k in ks)]


def ordered_intents(text: str) -> list[Intent]:
    """文字中出現的業務意圖,依最早出現的關鍵字位置排序(「先提到的先處理」)。"""
    pos = {}
    for i, ks in KEYWORDS.items():
        if i in (Intent.OFF_TOPIC, Intent.HUMAN_HANDOFF):
            continue
        hits = [text.find(k) for k in ks if k in text]
        if hits:
            pos[i] = min(hits)
    return sorted(pos, key=pos.get)


class Bot:
    def __init__(self, node: Node, guard: Optional[InputGuard] = None, dispatcher: Optional[Dispatcher] = None,
                 topic_transition: bool = True, offtopic_policy: bool = True, use_guard: bool = True):
        self.node, self.dispatcher = node, dispatcher or Dispatcher()
        self.guard = guard or InputGuard(semantic=True)
        self.out_guard = OutputGuard()
        self.topic_transition, self.offtopic_policy, self.use_guard = topic_transition, offtopic_policy, use_guard

    # ---------- helpers ----------
    def _t(self, key: str, s: Session, **kw) -> str:
        opts = TPL[key]
        return opts[s.turn % len(opts)].format(**kw)

    def _reply(self, s: Session, texts: list[str], quick=None, **dbg) -> BotReply:
        safe = []
        for t in texts:
            ok, why = self.out_guard.check(t)
            safe.append(t if ok else TPL["fail"][0])
            if not ok:
                dbg["output_blocked"] = why
        return BotReply([b for t in safe for b in split_bubbles(t)], quick or [], dbg)

    def _ask(self, case: Case) -> tuple[str, list[str]]:
        m = case.missing()
        k = m[0]
        quick = ["手機", "耳機", "筆電", "其他"] if k == "product" else []
        return ASK_TEMPLATE[k], quick

    def _window(self, s: Session) -> list[Message]:
        w = s.history[-6:]
        with_img = [i for i, m in enumerate(w) if m.images]
        keep = set(with_img[-2:])                     # 只保留最近 2 張圖,控制 token 成本
        return [Message(m.role, m.text, m.images if i in keep else [], m.image_caption) for i, m in enumerate(w)]

    # ---------- main ----------
    def respond(self, s: Session, text: str, images: Optional[list[str]] = None) -> BotReply:
        s.turn += 1
        text, images = (text or "").strip(), images or []
        if s.locked:
            return self._reply(s, [TPL["block_lock"][0]], blocked="locked")
        if not text and not images:
            return self._reply(s, [self._t("empty", s)])

        # ---- 1. 防禦 ----
        verdict, ocr = Verdict(text=text), ""
        if self.use_guard:
            verdict = self.guard.check_text(text, recent=s.recent)
            for p in images:
                iv, o = self.guard.check_image(p)
                ocr += o
                if iv.action == "block" and verdict.action != "block":
                    verdict = iv
                    verdict.reasons.append("image")
            text = verdict.text
        if verdict.action == "block":
            return self._on_attack(s, verdict)
        prefix = [self._t("deescalate", s)] if verdict.action == "warn" else []

        # ---- 2. 閒聊 ----
        for rx, key in ([] if images else SMALLTALK):
            if rx.match(text) and (key in ("greet", "thanks") or not (s.active and s.active.collecting)):
                s.offtopic_streak = 0
                return self._reply(s, [self._t(key, s)], MENU if key == "greet" else [], gate="smalltalk")
        if s.active and s.active.awaiting and s.active.awaiting[0] == "confirm_order":
            r = self._handle_confirm(s, text)
            if r:
                return r

        # ---- 3. 離題/無脈絡 gate ----
        sig = domain_signal(text, bool(images))
        off_cue = any(k in text for k in KEYWORDS[Intent.OFF_TOPIC])
        collecting = bool(s.active and s.active.collecting)
        if self.offtopic_policy and off_cue and not sig:
            return self._offtopic(s)
        s.history.append(Message("user", text, images, ocr.strip() or None))
        out = self.node.triage(self._window(s))
        r = out.result
        if r is None:
            return self._handoff(s, "node_failed")
        ack = bool(YES.match(text) or NO.match(text))
        multi = self.topic_transition and len(ordered_intents(text)) >= 2
        if self.offtopic_policy and not collecting and (
                r.intent == Intent.OFF_TOPIC or (r.confidence < 0.3 and not multi) or (not sig and (r.confidence < 0.45 or ack))):
            s.history.pop()
            if ack:   # 沒有進行中的案件時的「好/嗯」只是應答,不能被歷史脈絡誤判成新案件
                return self._reply(s, [self._t("anything_else", s)], ["查物流", "退換貨", "發票", "沒有了"], gate="ack")
            return self._offtopic(s, blank=(r.intent != Intent.OFF_TOPIC))
        if r.intent == Intent.UNSAFE:
            s.history.pop()
            return self._on_attack(s, Verdict("block", "model_unsafe", 1.0, ["node"], text))
        if r.intent == Intent.OFF_TOPIC:
            if collecting and len(text) <= 12:
                r.intent = s.active.intent       # 「好」「嗯」之類在補槽時的短答,不當離題
            else:
                return self._offtopic(s)
        s.offtopic_streak = 0
        explicit = ordered_intents(text)
        if collecting and not explicit:
            r.intent = s.active.intent           # 補槽中的短答(「耳機」「是」)沒有新需求線索 → 維持原案件,避免被歷史脈絡誤判成換題
        elif self.topic_transition and len(explicit) >= 2 and (r.confidence < 0.5 or r.intent not in explicit):
            r.intent = explicit[0]               # 一句話多需求且模型沒有明確主意圖 → 先處理最先提到的

        # ---- 4. 案件管理 ----
        notes: list[str] = []
        new_slots = r.slots.model_dump()
        # 槽位須「有出處」:模型填的訂單編號若沒在使用者的話/圖片 OCR 出現過,就是幻覺(常見:抄提示詞範例),一律視為沒給 → 走追問
        if new_slots.get("order_id"):
            seen = {m.upper() for h in s.history if h.role == "user"
                    for t in (h.text, h.image_caption or "") for m in re.findall(r"[A-Z]{2}\d{8,12}", re.sub(r"\s+", "", t.upper()) if t else "")}
            if new_slots["order_id"].upper() not in seen:
                new_slots["order_id"] = None
                r.slots.order_id = None
        if new_slots.get("order_id"):
            s.last_order_id = new_slots["order_id"]
        sec = secondary_intents(text, r.intent) if self.topic_transition else []
        if s.active is None or not s.active.collecting:
            s.active = Case(r.intent, urgency=r.urgency)
        elif r.intent != s.active.intent and r.intent != Intent.HUMAN_HANDOFF:
            if SWITCH_CUE.search(text):
                s.active = Case(r.intent, urgency=r.urgency)           # 使用者主動換題:放棄舊案
            elif self.topic_transition:
                stub = Case(r.intent, urgency=r.urgency); stub.merge(new_slots)
                s.pending.append(stub)
                notes.append(self._t("defer", s, case=INTENT_ZH[r.intent], cur=INTENT_ZH[s.active.intent]))
                new_slots = {}                                          # 新需求的槽位不要污染目前案件
            else:
                s.active = Case(r.intent, urgency=r.urgency)
        c = s.active
        if new_slots:
            c.merge(new_slots)
        if r.urgency == "high":
            c.urgency = "high"
        for i in sec:
            if not any(p.intent == i for p in s.pending) and i != c.intent:
                st = Case(i, urgency="low")
                if c.slots.get("order_id"):
                    st.slots["order_id"] = c.slots["order_id"]
                s.pending.append(st)
                notes.append(self._t("defer", s, case=INTENT_ZH[i], cur=INTENT_ZH[c.intent]))

        # ---- 5. 補槽 or 派工 ----
        return self._advance(s, prefix, notes, dbg={"path": out.path, "intent": c.intent.value, "conf": r.confidence})

    # ---------- state transitions ----------
    def _advance(self, s: Session, prefix: list[str], notes: list[str], dbg: dict) -> BotReply:
        c = s.active
        if c.intent == Intent.HUMAN_HANDOFF:
            return self._handoff(s, "user_request", prefix)
        m = c.missing()
        if m == ["order_id"] and s.last_order_id and c.awaiting is None and c.asked == 0 and self.topic_transition:
            c.awaiting = ("confirm_order", s.last_order_id)
            return self._reply(s, prefix + notes + [self._t("confirm_order", s, oid=s.last_order_id)], ["是", "不是,我另外提供"], **dbg)
        if m:
            c.asked += 1
            if c.asked > 3:
                return self._handoff(s, "too_many_asks", prefix)
            q, quick = self._ask(c)
            return self._reply(s, prefix + notes + [q], quick, **dbg)
        t = self.dispatcher.dispatch(c.intent, c.urgency, c.slots)
        s.ticket_ids.append(t.id)
        c.ticket = t
        key = "dispatched" if t.status == "open" else "dispatched_queued"
        msgs = prefix + notes + [self._t(key, s, dept=t.department, tid=t.id, sla=t.sla_hours)]
        if c.urgency == "high" and c.intent == Intent.DEFECT_REPAIR:
            msgs.append("若有冒煙、發燙或膨脹,請先停止使用並遠離易燃物。")
        s.history.append(Message("assistant", msgs[-1]))
        return self._next_topic(s, msgs, dbg)

    def _next_topic(self, s: Session, msgs: list[str], dbg: dict) -> BotReply:
        if self.topic_transition and s.pending:
            nxt = s.pending.popleft()
            if not nxt.slots.get("order_id") and s.last_order_id:
                nxt.slots["order_id"] = s.last_order_id
            s.active = nxt
            if nxt.slots.get("order_id") and not nxt.missing():
                nxt.awaiting = ("confirm_order", nxt.slots["order_id"])
                msgs.append(self._t("next_confirm", s, case=INTENT_ZH[nxt.intent], oid=nxt.slots["order_id"]))
                return self._reply(s, msgs, ["是", "不是,我另外提供"], pending_left=len(s.pending), **dbg)
            q, quick = self._ask(nxt)
            nxt.asked += 1
            msgs.append(self._t("next_ask", s, case=INTENT_ZH[nxt.intent], ask=q))
            return self._reply(s, msgs, quick, pending_left=len(s.pending), **dbg)
        msgs.append(self._t("anything_else", s))
        return self._reply(s, msgs, ["查物流", "退換貨", "發票", "沒有了"], **dbg)

    def _handle_confirm(self, s: Session, text: str) -> Optional[BotReply]:
        c = s.active
        _, oid = c.awaiting
        if YES.match(text):
            c.awaiting = None; c.slots["order_id"] = oid
            return self._advance(s, [], [], {"path": "confirm_yes"})
        if NO.match(text):
            c.awaiting = None; c.slots["order_id"] = None; c.asked += 1
            q, quick = self._ask(c)
            return self._reply(s, [q], quick, path="confirm_no")
        c.awaiting = None                      # 使用者直接給了新資訊 → 取消確認,照常處理
        return None

    def _offtopic(self, s: Session, blank: bool = False) -> BotReply:
        s.offtopic_streak += 1
        if s.offtopic_streak >= 3:
            return self._reply(s, [self._t("offtopic_many", s)], ["找真人"] + MENU[:2], gate="offtopic_many")
        if s.active and s.active.collecting:
            q, quick = self._ask(s.active) if s.active.missing() else ("請問還需要補充什麼嗎?", [])
            return self._reply(s, [self._t("offtopic_resume", s, case=INTENT_ZH.get(s.active.intent, "目前案件"), ask=q)], quick, gate="offtopic_resume")
        return self._reply(s, [self._t("menu_blank" if blank else "menu", s)], MENU, gate="offtopic_menu")

    def _on_attack(self, s: Session, v: Verdict) -> BotReply:
        s.strikes += 1
        dbg = {"blocked": v.category, "reasons": v.reasons, "strikes": s.strikes}
        if v.category == "abuse":
            return self._reply(s, [self._t("abuse_block", s)], ["找真人"], **dbg) if s.strikes < 3 else self._lock(s, dbg)
        if v.category in ("bad_image", "image_too_large"):
            return self._reply(s, [self._t("bad_image", s)], **dbg)
        if s.strikes >= 3:
            return self._lock(s, dbg)
        return self._reply(s, [self._t("block1" if s.strikes == 1 else "block2", s)], MENU[:3], **dbg)

    def _lock(self, s: Session, dbg: dict) -> BotReply:
        s.locked = True
        t = self.dispatcher.dispatch(Intent.UNSAFE, "low", {"issue_summary": f"strikes={s.strikes}"})
        s.ticket_ids.append(t.id)
        return self._reply(s, [TPL["block_lock"][0]], **dbg, ticket=t.id)

    def _handoff(self, s: Session, why: str, prefix: Optional[list[str]] = None) -> BotReply:
        c = s.active
        slots = dict(c.slots) if c else {}
        t = self.dispatcher.dispatch(Intent.HUMAN_HANDOFF, c.urgency if c else "medium", slots)
        s.ticket_ids.append(t.id)
        if c:
            c.ticket = t
        msg = (TPL["fail"][0] if why == "node_failed" else
               f"好的,已為您轉接「{t.department}」,單號 #{t.id}。您剛才提供的資訊都已帶給同仁,不需要重複說明。")
        return self._reply(s, (prefix or []) + [msg], handoff=why)