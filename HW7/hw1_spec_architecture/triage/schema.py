"""結構輸出 Spec:分流節點的唯一輸出格式。

設計原則
1. 模型只負責「語意判斷」(intent / urgency / slots / 是否需追問)。
2. 部門 (department)、必填欄位 (required slots)、缺漏欄位 (missing_slots) 一律由程式
   以確定性規則推導,不讓模型生成 → 少生成 token、少出錯、也無法被 prompt injection 竄改路由。
"""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Intent(str, Enum):
    RETURN_EXCHANGE = "return_exchange"  # 退換貨 / 退款
    LOGISTICS = "logistics"              # 物流查詢、延遲、遺失
    DEFECT_REPAIR = "defect_repair"      # 商品瑕疵 / 故障報修 / 運送損壞(通常附照片)
    INVOICE = "invoice"                  # 發票、收據、統編
    TECH_SUPPORT = "tech_support"        # 使用/設定/相容性等技術問題
    HUMAN_HANDOFF = "human_handoff"      # 使用者要求真人 / 高度情緒 / 法律糾紛
    OFF_TOPIC = "off_topic"              # 與本客服無關(閒聊、寫作業、天氣…)
    UNSAFE = "unsafe"                    # 攻擊:injection、越獄、辱罵、惡意圖片


# 意圖 → 下游部門(確定性映射)
DEPARTMENT: dict[Intent, str] = {
    Intent.RETURN_EXCHANGE: "售後退換貨組",
    Intent.LOGISTICS: "物流客服組",
    Intent.DEFECT_REPAIR: "維修與品保組",
    Intent.INVOICE: "財務發票組",
    Intent.TECH_SUPPORT: "技術支援組",
    Intent.HUMAN_HANDOFF: "真人客服主管",
    Intent.OFF_TOPIC: "(不派工)",
    Intent.UNSAFE: "資安/風控(僅記錄)",
}

# 各意圖「建單所需」的必填欄位
REQUIRED_SLOTS: dict[Intent, list[str]] = {
    Intent.RETURN_EXCHANGE: ["order_id", "issue_summary"],
    Intent.LOGISTICS: ["order_id"],
    Intent.DEFECT_REPAIR: ["product", "issue_summary"],  # 有圖時 image_findings 會補足
    Intent.INVOICE: ["order_id"],
    Intent.TECH_SUPPORT: ["product", "issue_summary"],
    Intent.HUMAN_HANDOFF: [],
    Intent.OFF_TOPIC: [],
    Intent.UNSAFE: [],
}

# 缺欄位時的「單一追問」模板(一次只問一件事,降低使用者負擔)
ASK_TEMPLATE: dict[str, str] = {
    "order_id": "方便提供訂單編號嗎?(例如 TW20261105009)",
    "product": "請問是哪一項商品呢?(例如:手機、耳機、筆電)",
    "issue_summary": "可以簡單說明遇到的狀況嗎?",
}

ORDER_ID_RE = re.compile(r"\b[A-Z]{2}\d{8,12}\b")


class Slots(BaseModel):
    order_id: Optional[str] = Field(None, description="訂單編號,格式如 TW20261105009;沒提到就 null")
    product: Optional[str] = Field(None, description="商品名稱/類別;沒提到就 null")
    issue_summary: Optional[str] = Field(None, description="一句話描述問題(<=40字)")
    image_findings: Optional[str] = Field(None, description="從使用者照片看到的重點(<=40字);沒有圖就 null")

    @field_validator("order_id")
    @classmethod
    def _check_order_id(cls, v):
        if v is None:
            return v
        v = v.strip().upper()
        return v if ORDER_ID_RE.fullmatch(v) else None  # 不合格式視為沒抓到,而非讓整份輸出失敗


class TriageResult(BaseModel):
    intent: Intent
    urgency: Literal["low", "medium", "high"] = Field(
        description="high=安全疑慮/金額大/情緒激動;medium=影響使用;low=一般詢問"
    )
    slots: Slots
    need_clarification: bool = Field(description="必填資訊不足、需要追問時為 true")
    confidence: float = Field(ge=0.0, le=1.0)

    # ---- 以下全部是推導欄位,不由模型生成 ----
    @property
    def department(self) -> str:
        return DEPARTMENT[self.intent]

    @property
    def missing_slots(self) -> list[str]:
        out = []
        for k in REQUIRED_SLOTS[self.intent]:
            if not getattr(self.slots, k):
                if k == "issue_summary" and self.slots.image_findings:
                    continue
                out.append(k)
        return out

    def next_question(self) -> Optional[str]:
        m = self.missing_slots
        return ASK_TEMPLATE[m[0]] if m else None


def result_json_schema() -> dict:
    """給 constrained decoding / prompt 使用的 JSON Schema(只含模型需生成的欄位)。"""
    return TriageResult.model_json_schema()


def slots_json_schema() -> dict:
    return Slots.model_json_schema()


def parse_strict(text: str) -> TriageResult:
    """嚴格解析:不做任何修補。失敗即拋例外 → 用來量測「模型原生」結構正確率。"""
    return TriageResult.model_validate_json(text.strip())


def parse_lenient(text: str) -> TriageResult:
    """寬鬆解析:剝除 ```json 圍欄、取第一個 {...}。僅供產品線 fallback,不計入實驗 A 的原生正確率。"""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no json object")
    return TriageResult.model_validate(json.loads(m.group(0)))


# ---------- 給 constrained decoding 用的「手寫扁平 schema」 ----------
# 為什麼不直接用 pydantic 產生的 schema:它含 $defs/$ref,各 grammar 引擎支援度不一;
# 手寫版additionally 把 order_id 用 pattern 鎖格式、字串加 maxLength、confidence 用離散 enum,
# 以「限制可生成空間」換取更短、更穩定的輸出。
_NULLABLE = lambda s: {"anyOf": [s, {"type": "null"}]}  # noqa: E731

_SLOT_PROPS = {
    "order_id": _NULLABLE({"type": "string", "pattern": "^[A-Z]{2}[0-9]{8,12}$"}),
    "product": _NULLABLE({"type": "string", "maxLength": 20}),
    "issue_summary": _NULLABLE({"type": "string", "maxLength": 60}),
    "image_findings": _NULLABLE({"type": "string", "maxLength": 60}),
}


def constrained_slots_schema() -> dict:
    return {"type": "object", "properties": dict(_SLOT_PROPS), "required": list(_SLOT_PROPS)}


def constrained_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": [i.value for i in Intent]},
            "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
            "slots": constrained_slots_schema(),
            "need_clarification": {"type": "boolean"},
            "confidence": {"enum": [0.1, 0.3, 0.5, 0.7, 0.9]},
        },
        "required": ["intent", "urgency", "slots", "need_clarification", "confidence"],
    }
