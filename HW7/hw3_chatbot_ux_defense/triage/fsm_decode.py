"""自製「字元級」文法約束解碼(取代 lm-format-enforcer)。

為什麼自己寫:
 1) lm-format-enforcer + Qwen2.5-VL(byte-level BPE)在中文自由文字欄位會一路吐空白直到截斷。
 2) 我的第一版自製狀態機是「token 級」,要求字串以單獨的 `"` token 開頭;但 Qwen 習慣把 `":"`、`"TW` 這類符號與
    文字合併成一個 token,單獨的 `"` 對模型很不自然 → 每個可為 null 的欄位都被迫選 null(slots 全空)。
所以改成「字元級」:對每個 token 檢查其「每個字元」是否合法(可跨越骨架與內容),完全尊重模型原本的分詞。

輸出為緊湊 JSON(骨架固定),項目:
  lit   固定骨架      enum  只允許合法選項      nstr  可為 null 的字串(null 或 "…",以 " 結束)
自由字串內禁止:反斜線、控制字元、純空白 token;以字元數上限收尾(超過只允許收引號)。
核心 FSM 不依賴 torch;torch 的 LogitsProcessor 在 backends.py。
"""
from __future__ import annotations

from collections import defaultdict

from .schema import Intent

BAD = set("\\\n\r\t\x0b\x0c")
DONE = ("done",)


def full_plan() -> list:
    q = lambda s: f'"{s}"'  # noqa: E731
    return [("lit", '{"intent":'), ("enum", [q(i.value) for i in Intent]),
            ("lit", ',"urgency":'), ("enum", [q("low"), q("medium"), q("high")]),
            ("lit", ',"slots":{"order_id":'), ("nstr", 16),
            ("lit", ',"product":'), ("nstr", 12),
            ("lit", ',"issue_summary":'), ("nstr", 40),
            ("lit", ',"image_findings":'), ("nstr", 40),
            ("lit", '},"need_clarification":'), ("enum", ["true", "false"]),
            ("lit", ',"confidence":'), ("enum", ["0.1", "0.3", "0.5", "0.7", "0.9"]),
            ("lit", "}")]


def slots_plan() -> list:   # 實驗 D 的 System Two:intent 已知,只生成 urgency + slots
    q = lambda s: f'"{s}"'  # noqa: E731
    return [("lit", '{"urgency":'), ("enum", [q("low"), q("medium"), q("high")]),
            ("lit", ',"slots":{"order_id":'), ("nstr", 16),
            ("lit", ',"product":'), ("nstr", 12),
            ("lit", ',"issue_summary":'), ("nstr", 40),
            ("lit", ',"image_findings":'), ("nstr", 40),
            ("lit", "}}")]


def plan_for_schema(schema: dict) -> list:
    return full_plan() if "intent" in schema.get("properties", {}) else slots_plan()


class FSM:
    def __init__(self, texts: list[str], special: set[int], eos_ids: set[int], plan: list):
        """texts[i] = token i 解碼後的文字。"""
        self.texts, self.special, self.eos = texts, set(special), set(eos_ids)
        self.items = []
        for kind, arg in plan:
            if kind == "enum":
                for a in arg:
                    for b in arg:
                        if a is not b and b.startswith(a):
                            raise ValueError("enum 選項互為前綴")
            self.items.append((kind, arg))
        self.by_first = defaultdict(list)
        for i, t in enumerate(texts):
            if t and i not in self.special:
                self.by_first[t[0]].append(i)
        self._cache: dict = {}
        self.reset()

    # ---------- 字元級自動機 ----------
    def _opts(self, i):
        kind, arg = self.items[i]
        return arg if kind == "enum" else ["null", '"']

    def _enter(self, i):
        if i >= len(self.items):
            return DONE
        kind = self.items[i][0]
        if kind == "lit":
            return (i, 0, None, False, 0)
        return (i, 0, tuple(range(len(self._opts(i)))), False, 0)

    def reset(self):
        self.state = self._enter(0)

    def step_char(self, st, ch):
        if st == DONE:
            return None
        i, pos, cands, in_str, n = st
        kind, arg = self.items[i]
        if in_str:
            if ch == '"':
                return self._enter(i + 1)
            if ch in BAD:
                return None
            return (i, pos, cands, True, n + 1)
        if kind == "lit":
            if ch != arg[pos]:
                return None
            pos += 1
            return self._enter(i + 1) if pos == len(arg) else (i, pos, None, False, 0)
        opts = self._opts(i)
        cands = [k for k in cands if len(opts[k]) > pos and opts[k][pos] == ch]
        if not cands:
            return None
        pos += 1
        done = [k for k in cands if len(opts[k]) == pos]
        if done:
            if kind == "nstr" and done[0] == 1:
                return (i, pos, None, True, 0)
            return self._enter(i + 1)
        return (i, pos, tuple(cands), False, 0)

    def _consume(self, st, text):
        for ch in text:
            st = self.step_char(st, ch)
            if st is None:
                return None
        return st

    # ---------- 對外 API ----------
    def _key(self, st):
        if st == DONE:
            return st
        i, pos, cands, in_str, n = st
        if in_str:   # 字數只分「未滿 / 已滿」兩種,以便快取
            return (i, pos, cands, True, self.items[i][1] if n >= self.items[i][1] else 0)
        return st

    def allowed(self):
        """回傳 (快取鍵, 允許的 token id list)。"""
        st = self.state
        key = self._key(st)
        if key in self._cache:
            return key, self._cache[key]
        if st == DONE:
            ids = sorted(self.eos)
        else:
            rep = key if key != DONE else st
            in_str = rep[3]
            firsts = [c for c in self.by_first if self.step_char(rep, c) is not None]
            if in_str and rep[4] > 0:        # 字數已滿:只允許以引號開頭(收尾)的 token
                firsts = [c for c in firsts if c == '"']
            ids = []
            for c in firsts:
                for t in self.by_first[c]:
                    s = self.texts[t]
                    if in_str and not s.strip():
                        continue                      # 字串內不允許純空白 token
                    if self._consume(rep, s) is not None:
                        ids.append(t)
        self._cache[key] = ids
        return key, ids

    def advance(self, tok: int):
        if self.state == DONE:
            return
        nxt = self._consume(self.state, self.texts[tok])
        if nxt is None:          # 理論上不會發生(processor 已遮罩);保守處理
            raise ValueError(f"token {tok}={self.texts[tok]!r} 不在允許集合")
        self.state = nxt