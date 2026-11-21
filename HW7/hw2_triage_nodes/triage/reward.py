"""可驗證獎勵(Verifiable Reward)—— 實驗 B 的 RL 延伸(GRPO / RLVR)。
SFT 之後若仍有邊界案例(例如多意圖、含糊句),可用此函式當 reward 做 GRPO(例如 TRL 的 GRPOTrainer)。
不需要人類偏好標註,因為「結構對不對」「意圖對不對」都能程式化驗證 —— 這是 RLHF 在本任務的實用替代。"""
from __future__ import annotations

import json

from .schema import parse_strict


def triage_reward(completion: str, gold: dict) -> float:
    """0~1.0:格式 0.3 + intent 0.4 + order_id 0.2 + 澄清判斷 0.1;外加過長懲罰。"""
    try:
        json.loads(completion)
    except Exception:  # noqa: BLE001
        return 0.0
    r = 0.1  # 合法 JSON
    try:
        res = parse_strict(completion)
    except Exception:  # noqa: BLE001
        return r
    r += 0.2
    r += 0.4 * (res.intent.value == gold["intent"])
    r += 0.2 * ((res.slots.order_id or None) == gold["order_id"])
    r += 0.1 * (bool(res.missing_slots) == gold["need_clarification"])
    if len(completion) > 400:
        r -= 0.1
    return max(0.0, min(1.0, r))
