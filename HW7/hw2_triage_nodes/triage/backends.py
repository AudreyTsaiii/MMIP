"""模型後端抽象。

- RuleBackend  :純規則(關鍵字 + regex),零依賴、毫秒級。用途:(1) CI/離線可跑;(2) 產品上的最終 fallback;
                 (3) 評測離題/防禦等「與模型無關」的系統層設計。它不是 VLM,不看圖片(只讀 image_caption)。
- QwenVLBackend:真實 VLM(Qwen2.5-VL),支援 LoRA adapter、constrained decoding、單次 forward 分類。
"""
from __future__ import annotations

import json
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .prompts import wrap_user
from .schema import ORDER_ID_RE, Intent


@dataclass
class Message:
    role: str                                   # "user" | "assistant"
    text: str = ""
    images: list = field(default_factory=list)  # 路徑或 PIL.Image
    image_caption: Optional[str] = None         # 供 RuleBackend / 測試使用的圖片描述(OCR 或 caption)


@dataclass
class GenResult:
    text: str
    in_tokens: int = 0
    out_tokens: int = 0
    seconds: float = 0.0


class Backend(ABC):
    name = "base"

    @abstractmethod
    def generate(self, system: str, history: list[Message], max_new_tokens: int = 256,
                 json_schema: Optional[dict] = None) -> GenResult: ...

    @abstractmethod
    def classify(self, system: str, history: list[Message], options: dict[str, Intent]
                 ) -> tuple[dict[Intent, float], GenResult]: ...


# ----------------------------------------------------------------------------------
# RuleBackend
# ----------------------------------------------------------------------------------
KEYWORDS: dict[Intent, list[str]] = {
    Intent.RETURN_EXCHANGE: ["退貨", "退款", "換貨", "退錢", "不想要", "想退", "退回", "鑑賞期", "換新", "退費"],
    Intent.LOGISTICS: ["物流", "配送", "還沒到", "沒收到", "快遞", "包裹", "到貨", "出貨", "追蹤", "宅配",
                       "延遲", "寄到", "送到", "還沒送", "改地址", "超商取貨", "貨到哪"],
    Intent.DEFECT_REPAIR: ["壞了", "壞掉", "故障", "瑕疵", "破損", "裂", "無法開機", "不能開機", "開不了機",
                           "沒聲音", "冒煙", "膨脹", "維修", "報修", "刮傷", "凹", "漏液", "死機", "黑屏", "花屏",
                           "破掉", "碎", "發燙", "過熱", "綠線", "條紋", "閃屏", "當機", "充不進", "沒電", "耗電"],
    Intent.INVOICE: ["發票", "統編", "收據", "載具", "折讓", "報帳"],
    Intent.TECH_SUPPORT: ["怎麼設定", "如何設定", "怎麼用", "連線", "藍牙", "配對", "連不上", "相容", "韌體",
                          "驅動", "更新", "app", "App", "重設", "怎麼連", "支援"],
    Intent.HUMAN_HANDOFF: ["真人", "人工", "客服人員", "主管", "消保", "投訴", "消基會", "律師", "提告", "找你們老闆"],
    Intent.OFF_TOPIC: ["天氣", "笑話", "寫程式", "作業", "股票", "食譜", "翻譯", "你是誰", "總統", "彩券", "電影",
                       "午餐", "python", "Python", "詩", "星座", "加密貨幣", "旅遊", "考試", "論文", "唱歌"],
}
PRODUCTS = ["無線耳機", "耳機", "手機", "筆電", "平板", "滑鼠", "鍵盤", "充電器", "螢幕", "電視", "相機", "手錶",
            "行動電源", "路由器", "音響", "喇叭", "電腦", "充電線", "保護殼", "智慧手環", "掃地機器人"]
HIGH_URGENCY = ["冒煙", "膨脹", "著火", "起火", "燙", "爆", "觸電", "氣死", "很生氣", "太誇張", "投訴", "消保", "律師"]
ASK_HUMAN_RE = re.compile(r"(真人|人工|客服人員|主管)")


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


class RuleBackend(Backend):
    name = "rule"

    def __init__(self, forced_domain: bool = False):
        self.forced_domain = forced_domain   # True:模擬「沒有離題/攻擊類別、一律硬分到業務意圖」的基線

    def _user_text(self, history: list[Message]) -> tuple[str, str]:
        users = [m for m in history if m.role == "user"]
        last = users[-1] if users else Message("user", "")
        last_text = (last.text + " " + (last.image_caption or "")).strip()
        prev = " ".join((m.text + " " + (m.image_caption or "")) for m in users[:-1])
        return last_text, prev

    def _scores(self, text: str) -> dict[Intent, float]:
        sc = {i: 0.0 for i in Intent}
        for intent, kws in KEYWORDS.items():
            sc[intent] = float(sum(1 for k in kws if k in text))
        return sc

    def classify(self, system, history, options):
        last, prev = self._user_text(history)
        sc = self._scores(last)
        if sum(sc.values()) == 0 and prev:
            # 最新訊息沒有任何線索(例如只回了訂單編號)→ 繼承先前脈絡,但信心較低
            sc = {k: v * 0.8 for k, v in self._scores(prev).items()}
        labels = [options[k] for k in options]
        vec = [sc[l] * 2.2 if not (self.forced_domain and l in (Intent.OFF_TOPIC, Intent.UNSAFE)) else -50.0
               for l in labels]
        probs = dict(zip(labels, _softmax(vec)))
        return probs, GenResult("", 0, 0, 0.0)

    def generate(self, system, history, max_new_tokens=256, json_schema=None):
        from .prompts import SYSTEM_ONE_OPTIONS
        t0 = time.perf_counter()
        probs, _ = self.classify(system, history, SYSTEM_ONE_OPTIONS)
        intent = max(probs, key=probs.get)
        conf = probs[intent]
        last, prev = self._user_text(history)
        full = last + " " + prev
        m = ORDER_ID_RE.search(full)
        product = next((p for p in PRODUCTS if p in full), None)
        summary = re.sub(ORDER_ID_RE, "", last).strip()[:40] or None
        caption = next((h.image_caption for h in reversed(history) if h.role == "user" and h.image_caption), None)
        urgency = "high" if any(k in full for k in HIGH_URGENCY) else ("medium" if intent in
                  (Intent.DEFECT_REPAIR, Intent.RETURN_EXCHANGE) else "low")
        slots = {"order_id": m.group(0) if m else None, "product": product,
                 "issue_summary": summary, "image_findings": caption}
        from .schema import REQUIRED_SLOTS
        need = any(not slots[k] for k in REQUIRED_SLOTS[intent] if not (k == "issue_summary" and caption))
        out = {"intent": intent.value, "urgency": urgency, "slots": slots,
               "need_clarification": need, "confidence": round(conf, 2)}
        txt = json.dumps(out, ensure_ascii=False)
        return GenResult(txt, 0, len(txt) // 2, time.perf_counter() - t0)


# ----------------------------------------------------------------------------------
# QwenVLBackend(需 GPU;沙盒內未執行,請在 Colab / 本機驗證)
# ----------------------------------------------------------------------------------
class QwenVLBackend(Backend):
    name = "qwen"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", adapter: Optional[str] = None,
                 max_pixels: int = 448 * 448):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=64 * 28 * 28, max_pixels=max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.name = "qwen+lora" if adapter else "qwen"

    # -- helpers --
    def build_inputs(self, system: str, history: list[Message], add_generation_prompt=True):
        from PIL import Image
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}]
        pil = []
        for m in history:
            content = []
            for im in m.images:
                pil.append(im if hasattr(im, "size") else Image.open(im).convert("RGB"))
                content.append({"type": "image"})
            content.append({"type": "text", "text": wrap_user(m.text) if m.role == "user" else m.text})
            msgs.append({"role": m.role, "content": content})
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
        return prompt, pil

    def _sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def _fsm_processor(self, schema: dict, prompt_len: int):
        """建立自製「字元級」文法約束的 LogitsProcessor(見 triage/fsm_decode.py)。batch=1。"""
        import torch
        from transformers import LogitsProcessor
        from .fsm_decode import FSM, plan_for_schema

        tok = self.proc.tokenizer
        if not hasattr(self, "_texts"):      # 只算一次(約數秒~十幾秒):每個 token 的文字
            self._texts = [tok.decode([i]) for i in range(len(tok))]
            self._special = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
            eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
            self._eos = {e for e in eos if e is not None}
            self._mask_cache = {}            # (fsm_key) → 允許集合的 bool mask;跨請求共用
        fsm = FSM(self._texts, self._special, self._eos, plan_for_schema(schema))
        if not hasattr(self, "_fsm_shared"):
            self._fsm_shared = {}
        # 同一 plan 共用 FSM 的 allowed() 快取(最耗時的是第一次掃詞表)
        pk = json.dumps(plan_for_schema(schema), ensure_ascii=False)
        fsm._cache = self._fsm_shared.setdefault(pk, fsm._cache)
        masks = self._mask_cache.setdefault(pk, {})

        class Proc(LogitsProcessor):
            def __init__(self):
                self.seen = 0

            def __call__(self, input_ids, scores):
                n = input_ids.shape[1] - prompt_len
                while self.seen < n:
                    fsm.advance(int(input_ids[0, prompt_len + self.seen]))
                    self.seen += 1
                key, ids = fsm.allowed()
                m = masks.get(key)
                if m is None or m.shape[-1] != scores.shape[-1]:
                    m = torch.full((scores.shape[-1],), float("-inf"), device=scores.device, dtype=scores.dtype)
                    m[torch.tensor(ids, device=scores.device)] = 0
                    masks[key] = m
                return scores + m

        return Proc()

    # -- API --
    def generate(self, system, history, max_new_tokens=256, json_schema=None):
        prompt, pil = self.build_inputs(system, history)
        inputs = self.proc(text=[prompt], images=pil or None, return_tensors="pt").to(self.model.device)
        kw = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if json_schema is not None:
            from transformers import LogitsProcessorList
            kw["logits_processor"] = LogitsProcessorList([self._fsm_processor(json_schema, inputs["input_ids"].shape[1])])
        self._sync(); t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **kw)
        self._sync(); dt = time.perf_counter() - t0
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = self.proc.batch_decode(gen, skip_special_tokens=True)[0]
        return GenResult(text, int(inputs["input_ids"].shape[1]), int(gen.shape[1]), dt)

    def classify(self, system, history, options):
        prompt, pil = self.build_inputs(system, history)
        inputs = self.proc(text=[prompt], images=pil or None, return_tensors="pt").to(self.model.device)
        ids = [self.proc.tokenizer.encode(k, add_special_tokens=False)[0] for k in options]
        self._sync(); t0 = time.perf_counter()
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        self._sync(); dt = time.perf_counter() - t0
        p = self.torch.softmax(logits[ids].float(), dim=-1).tolist()
        return dict(zip(options.values(), p)), GenResult("", int(inputs["input_ids"].shape[1]), 1, dt)


def get_backend(kind: str, **kw) -> Backend:
    if kind == "rule":
        return RuleBackend(**kw)
    if kind == "qwen":
        return QwenVLBackend(**kw)
    raise ValueError(kind)