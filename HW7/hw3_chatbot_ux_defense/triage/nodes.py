"""四種分流節點實作:A Prompt / B SFT / C Constrained Decoding / D System One 雙系統。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from .backends import Backend, GenResult, Message
from .prompts import (SYSTEM_ONE_OPTIONS, SYSTEM_SHORT, system_full, system_one_prompt,
                      system_two_slots)
from .schema import (Intent, Slots, TriageResult, constrained_schema, constrained_slots_schema,
                     parse_strict)


@dataclass
class NodeOutput:
    result: Optional[TriageResult]
    raw: str = ""
    valid_json: bool = False
    valid_schema: bool = False
    in_tokens: int = 0
    out_tokens: int = 0
    seconds: float = 0.0
    path: str = ""                 # D:走了哪條路 fast / slots / slow
    calls: int = 0                 # 模型 forward/generate 次數
    error: str = ""


def _finish(raw: str, gens: list[GenResult], path: str = "") -> NodeOutput:
    o = NodeOutput(None, raw, path=path, calls=len(gens),
                   in_tokens=sum(g.in_tokens for g in gens), out_tokens=sum(g.out_tokens for g in gens),
                   seconds=sum(g.seconds for g in gens))
    try:
        json.loads(raw)
        o.valid_json = True
    except Exception as e:  # noqa: BLE001
        o.error = f"json: {e}"
        return o
    try:
        o.result = parse_strict(raw)
        o.valid_schema = True
    except (ValidationError, ValueError) as e:
        o.error = f"schema: {str(e)[:120]}"
    return o


class Node:
    name = "base"

    def triage(self, history: list[Message]) -> NodeOutput:  # pragma: no cover
        raise NotImplementedError


class NodeA(Node):
    """實驗 A:只靠 prompt(schema + 規則 + few-shot)。嚴格解析,不修補 → 量測原生結構正確率。"""
    name = "A_prompt"

    def __init__(self, backend: Backend, retries: int = 0, max_new_tokens: int = 300):
        self.b, self.retries, self.mnt = backend, retries, max_new_tokens
        self.system = system_full()

    def triage(self, history):
        gens, raw = [], ""
        for _ in range(self.retries + 1):
            g = self.b.generate(self.system, history, self.mnt)
            gens.append(g); raw = g.text
            out = _finish(raw, gens)
            if out.valid_schema:
                return out
        return out


class NodeB(Node):
    """實驗 B:SFT(LoRA)後的模型 + 極短 prompt;規則已內化到權重。需傳入載入 adapter 的 backend。"""
    name = "B_sft"

    def __init__(self, backend: Backend, max_new_tokens: int = 200):
        self.b, self.mnt = backend, max_new_tokens

    def triage(self, history):
        g = self.b.generate(SYSTEM_SHORT, history, self.mnt)
        return _finish(g.text, [g])


class NodeC(Node):
    """實驗 C:與 A 相同的 prompt 語意,但解碼階段以 JSON-schema 文法限制可選 token。"""
    name = "C_decoding"

    def __init__(self, backend: Backend, max_new_tokens: int = 200):
        self.b, self.mnt = backend, max_new_tokens
        self.schema = constrained_schema()
        self.system = system_full(self.schema)

    def triage(self, history):
        g = self.b.generate(self.system, history, self.mnt, json_schema=self.schema)
        return _finish(g.text, [g])


def _s2_schema() -> dict:
    return {"type": "object",
            "properties": {"urgency": {"type": "string", "enum": ["low", "medium", "high"]},
                           "slots": constrained_slots_schema()},
            "required": ["urgency", "slots"]}


class NodeD(Node):
    """實驗 D:System One / System Two 分流。

    System One:單次 forward,在單一 token 上讀出 8 類 intent 的機率(~1 token 成本)。
    - OFF_TOPIC / UNSAFE 且夠有信心 → 直接結束,完全不呼叫生成(最便宜、最快、也最不易被 injection 影響)。
    - 一般意圖且信心 ≥ tau → System Two:intent 已知,只生成 {urgency, slots}(文法約束,輸出短)。
    - 信心 < tau → slow path:退回 NodeC 完整約束生成(讓大模型自己判斷)。
    need_clarification / department / missing_slots 全部由程式推導。
    """
    name = "D_system_one"

    def __init__(self, backend: Backend, tau: float = 0.6, max_new_tokens: int = 160):
        self.b, self.tau, self.mnt = backend, tau, max_new_tokens
        self.sys1 = system_one_prompt()
        self.slow = NodeC(backend)
        self.s2_schema = _s2_schema()

    def triage(self, history):
        probs, g1 = self.b.classify(self.sys1, history, SYSTEM_ONE_OPTIONS)
        intent = max(probs, key=probs.get)
        conf = probs[intent]
        if conf < self.tau:
            out = self.slow.triage(history)
            out.path, out.in_tokens, out.seconds = "slow", out.in_tokens + g1.in_tokens, out.seconds + g1.seconds
            out.calls += 1
            return out
        if intent in (Intent.OFF_TOPIC, Intent.UNSAFE):
            r = TriageResult(intent=intent, urgency="low", slots=Slots(), need_clarification=False, confidence=conf)
            return NodeOutput(r, r.model_dump_json(), True, True, g1.in_tokens, 1, g1.seconds, "fast", 1)
        g2 = self.b.generate(system_two_slots(intent, self.s2_schema), history, self.mnt, json_schema=self.s2_schema)
        try:
            d = json.loads(g2.text)
            r = TriageResult(intent=intent, urgency=d["urgency"], slots=Slots(**d["slots"]),
                             need_clarification=False, confidence=round(conf, 2))
            r.need_clarification = bool(r.missing_slots)
            o = NodeOutput(r, r.model_dump_json(), True, True)
        except Exception as e:  # noqa: BLE001
            o = NodeOutput(None, g2.text, error=f"s2: {str(e)[:100]}")
        o.in_tokens, o.out_tokens = g1.in_tokens + g2.in_tokens, 1 + g2.out_tokens
        o.seconds, o.path, o.calls = g1.seconds + g2.seconds, "slots", 2
        return o


class SafeNode(Node):
    """產品用:primary 失敗(JSON/schema 無效)時依序降級 → RuleBackend → 轉真人。"""
    name = "safe"

    def __init__(self, primary: Node, fallback: Node):
        self.primary, self.fallback = primary, fallback

    def triage(self, history):
        t0 = time.perf_counter()
        try:
            out = self.primary.triage(history)
        except Exception as e:  # noqa: BLE001  (OOM、逾時…)
            out = NodeOutput(None, error=f"primary_exc: {e}")
        if out.valid_schema:
            return out
        fb = self.fallback.triage(history)
        fb.path = f"fallback({out.error[:30]})"
        fb.seconds += time.perf_counter() - t0 if fb.seconds == 0 else 0
        return fb
