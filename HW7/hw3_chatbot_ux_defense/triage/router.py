"""下游派工(模擬)。真實系統換成工單系統 API / 訊息佇列即可。"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from .schema import DEPARTMENT, Intent

SLA_HOURS = {"high": 1, "medium": 8, "low": 24}


@dataclass
class Ticket:
    id: str
    department: str
    intent: str
    urgency: str
    slots: dict
    created: float = field(default_factory=time.time)
    status: str = "open"          # open | queued(下游暫時不可用,稍後重送)

    @property
    def sla_hours(self) -> int:
        return SLA_HOURS[self.urgency]


class Dispatcher:
    def __init__(self, fail: bool = False):
        self._n = itertools.count(1)
        self.tickets: list[Ticket] = []
        self.fail = fail            # 角落情境:下游 API 掛掉 → 不丟失工單,標記 queued 並稍後重送
        self.outbox: list[Ticket] = []

    def dispatch(self, intent: Intent, urgency: str, slots: dict) -> Ticket:
        t = Ticket(f"T{next(self._n):05d}", DEPARTMENT[intent], intent.value, urgency, dict(slots))
        if self.fail:
            t.status = "queued"
            self.outbox.append(t)
        self.tickets.append(t)
        return t

    def flush(self) -> int:
        n = 0
        for t in list(self.outbox):
            t.status = "open"; self.outbox.remove(t); n += 1
        return n
