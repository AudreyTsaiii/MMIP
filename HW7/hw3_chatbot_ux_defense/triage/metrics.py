"""Benchmark 載入、評分、彙整。"""
from __future__ import annotations

import json
import pathlib
import statistics as st
from collections import defaultdict

from .backends import Message
from .nodes import NodeOutput

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def load_benchmark(path: pathlib.Path | None = None) -> list[dict]:
    path = path or DATA / "benchmark.jsonl"
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def to_history(turns: list[dict]) -> list[Message]:
    out = []
    for t in turns:
        imgs = [str(DATA / t["image"])] if t.get("image") else []
        out.append(Message(t["role"], t["text"], imgs, t.get("image_caption")))
    return out


def score_one(o: NodeOutput, gold: dict) -> dict:
    s = {"json_valid": o.valid_json, "schema_valid": o.valid_schema,
         "intent": False, "urgency": False, "order_id": False, "product": None, "clar": False}
    r = o.result
    if r is not None:
        s["intent"] = r.intent.value == gold["intent"]
        s["urgency"] = r.urgency == gold["urgency"]
        s["order_id"] = (r.slots.order_id or None) == gold["order_id"]
        if gold["product"]:
            p = r.slots.product or ""
            s["product"] = bool(p) and (gold["product"] in p or p in gold["product"])
        s["clar"] = bool(r.missing_slots) == gold["need_clarification"]
    s["full"] = s["intent"] and s["order_id"] and s["clar"]
    return s


def _pct(xs):
    xs = [x for x in xs if x is not None]
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def aggregate(rows: list[dict], outs: list[NodeOutput], cats: list[str], gpu_price_per_hr: float) -> dict:
    lat = sorted(o.seconds for o in outs)
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    by_cat = defaultdict(list)
    for r, c in zip(rows, cats):
        by_cat[c].append(r["intent"])
    mean_s = st.mean(lat)
    return {
        "n": len(rows),
        "json_valid%": _pct([r["json_valid"] for r in rows]),
        "schema_valid%": _pct([r["schema_valid"] for r in rows]),
        "intent_acc%": _pct([r["intent"] for r in rows]),
        "urgency_acc%": _pct([r["urgency"] for r in rows]),
        "order_id_acc%": _pct([r["order_id"] for r in rows]),
        "product_acc%": _pct([r["product"] for r in rows]),
        "clarify_acc%": _pct([r["clar"] for r in rows]),
        "full_correct%": _pct([r["full"] for r in rows]),
        "latency_mean_s": round(mean_s, 3), "latency_p50_s": round(lat[len(lat) // 2], 3),
        "latency_p95_s": round(p95, 3),
        "in_tokens_mean": round(st.mean(o.in_tokens for o in outs), 1),
        "out_tokens_mean": round(st.mean(o.out_tokens for o in outs), 1),
        "model_calls_mean": round(st.mean(o.calls for o in outs), 2),
        "cost_usd_per_1k_req": round(mean_s * 1000 * gpu_price_per_hr / 3600, 4),
        "intent_acc_by_category%": {c: _pct(v) for c, v in by_cat.items()},
        "path_dist": {p: sum(1 for o in outs if o.path == p) for p in {o.path for o in outs}},
    }


def markdown_table(results: dict[str, dict]) -> str:
    cols = ["json_valid%", "schema_valid%", "intent_acc%", "urgency_acc%", "order_id_acc%", "clarify_acc%",
            "full_correct%", "latency_mean_s", "latency_p95_s", "in_tokens_mean", "out_tokens_mean",
            "model_calls_mean", "cost_usd_per_1k_req"]
    head = "| 節點 | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1)
    lines = [head]
    for name, agg in results.items():
        lines.append(f"| {name} | " + " | ".join(str(agg[c]) for c in cols) + " |")
    return "\n".join(lines)
