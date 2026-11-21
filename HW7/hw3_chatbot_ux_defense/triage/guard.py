"""輸入/輸出防禦層(Defense in depth)。

層次(越前面越便宜):
  L0 資源限制     長度、重複洪水、圖片大小/格式
  L1 正規化       NFKC 全形→半形、移除零寬/控制字元、去除字母間夾雜的分隔符、leetspeak、base64 解碼
  L2 規則偵測     injection / 洩漏 system prompt / 角色扮演越獄 / 偽造 system 標籤 / 越權 / 外洩個資 / 辱罵
  L3 圖片         重新編碼(去 EXIF/隱藏資料)+ OCR,圖中文字與使用者文字走同一套偵測(影像 injection)
  L4 結構性隔離   (在 nodes/dialog)LLM 只輸出結構化資料、使用者可見回覆全由程式模板產生、
                  department 由 enum 映射 → 即使前面全被繞過,injection 也無法讓系統「說出」或「做出」任意內容
  L5 輸出檢查     canary 外洩、回覆內不得含未授權承諾、回覆長度/HTML 轉義
"""
from __future__ import annotations

import base64
import binascii
import html
import io
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import deque
from typing import Optional
from dataclasses import dataclass, field

from .prompts import CANARY
from .schema import ORDER_ID_RE

MAX_CHARS = 1000
MAX_IMG_BYTES = 8 * 1024 * 1024
MAX_IMG_SIDE = 6000

_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿­‪‫‬‭‮"), None)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})

# (category, weight, regex)。weight>=1 單獨即攔截;0.5 需要和其他訊號疊加
_R = lambda s: re.compile(s, re.I)  # noqa: E731
PATTERNS: list[tuple[str, float, re.Pattern]] = [
    ("override", 1.0, _R(r"(忽略|無視|无视|忘記|忘记|不要管|別管|别管|跳過|跳过|覆蓋|覆盖|取消|停止遵守|不要遵守|不用遵守)"
                         r".{0,8}(指令|指示|規則|规则|提示詞?|設定|设定|限制|prompt|instruction|rule)")),
    ("override", 1.0, _R(r"(ignore|disregard|forget|override|bypass)\s*(all|any|the|your|every)?\s*(previous|prior|above|earlier|former|system)?"
                         r".{0,15}(instruction|rule|prompt|guideline|constraint|restriction)")),
    ("override", 0.6, _R(r"(以上|上述|前面|之前|先前).{0,6}(所有|全部|一切).{0,4}(都|通通)?.{0,3}(作廢|無效|无效|不算)")),
    ("leak", 1.0, _R(r"(印出|輸出|输出|顯示|显示|告訴我|告诉我|說出|洩漏|泄露|重複|重复|複述|复述|列出|reveal|print|show|repeat|leak|dump|display)"
                     r".{0,14}(system\s*prompt|系統提示|系统提示|你的(設定|规则|規則|指令|提示詞?|prompt)|initial\s*prompt|原始(指令|提示)|內部規則)")),
    ("leak", 1.0, _R(r"(system\s*prompt|系統提示詞|系统提示词)")),
    ("leak", 1.0, _R(r"(show|tell|reveal|print|repeat|give|display|dump)\b.{0,14}\b(your|the)\s+(system\s+|initial\s+|original\s+|hidden\s+|internal\s+)?"
                     r"(instructions?|rules|prompt|guidelines|configuration)")),
    ("roleplay", 1.0, _R(r"(你現在是|你现在是|從現在起你是|从现在起你是|假裝你是|假装你是|扮演|act as|pretend (to be|you are)|you are now)"
                         r".{0,16}(沒有.{0,2}(限制|規則|规则|約束|约束|道德)|无限制|無限制|不受.{0,4}(限制|約束|约束)|管理員|管理员|開發者|开发者|root|admin|DAN|developer|unrestricted|no (rules|restrictions))")),
    ("jailbreak", 1.0, _R(r"\b(DAN|jailbreak|developer\s*mode|do anything now)\b|越獄|越狱|解除(限制|封印)")),
    ("fake_role", 1.0, _R(r"^\s*(system|assistant|developer|管理員|管理员|系統|系统)\s*[::>\]]")),
    ("tag_escape", 1.0, _R(r"</?\s*(user_message|system|assistant|instructions?)\s*>")),
    ("privilege", 1.0, _R(r"(核准|批准|通過|通过|同意|approve).{0,8}(所有|全部|任何|all)\s*.{0,6}(退款|退貨|退货|訂單|订单|refund|order)")),
    ("privilege", 1.0, _R(r"(intent|department|urgency)\s*[=:=]\s*['\"]?\w+|refund\s*approved|直接(給我|给我)?(退款|全額退款)")),
    ("exfil", 1.0, _R(r"(所有|全部|其他|別人|别人|任何).{0,6}(客戶|客户|用戶|用户|會員|会员|訂單|订单).{0,6}(資料|资料|個資|个资|資訊|信息|清單|名單|名单|電話|地址)")),
    ("exfil", 1.0, _R(r"(輸出|输出|給我|给我|告訴我|告诉我|提供|說出|show|give|tell|reveal).{0,6}(管理員|管理员|內部|内部|後台|后台).{0,6}(密碼|密码|api\s*key|金鑰|token)")),
    ("exfil", 0.6, _R(r"(管理員|管理员|內部|内部|後台|后台).{0,4}(密碼|密码|權限|权限|api\s*key|金鑰|token)")),
]
ABUSE = _R(r"(白癡|白痴|智障|笨蛋|廢物|废物|垃圾(?!郵件|邮件|桶|袋|信|訊息|簡訊|分類|车|車)|去死|王八|他媽|他妈|幹你|干你|混蛋|腦殘|脑残|fuck|shit|idiot|stupid|dumb)")
DOMAIN_HINT = _R(r"(訂單|订单|退貨|退货|退款|換貨|發票|物流|配送|快遞|壞|故障|維修|瑕疵|裂|沒收到|還沒到|耳機|手機|筆電|保固|統編|\bTW\d+)")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(_ZW)
    t = "".join(ch for ch in t if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"[ \t]+", " ", t).strip()


def variants(text: str, ocr: bool = False) -> list[str]:
    """回傳用於比對的多種還原版本:原文、去分隔符(防 i g n o r e / 忽 略)、leetspeak、base64 解碼。"""
    n = normalize(text)
    squeezed = re.sub(r"[\s\.\-_*|/\\,~`'\"]+", "", n)
    out = [n, squeezed, squeezed.lower().translate(_LEET)]
    if ocr:  # OCR 常把大寫 I 讀成 l / | / 1、並吃掉空格:額外還原一版再比對
        out.append(re.sub(r"[l|1!]", "i", squeezed.lower()))
    for m in re.finditer(r"[A-Za-z0-9+/]{24,}={0,2}", n):
        try:
            dec = base64.b64decode(m.group(0) + "=" * (-len(m.group(0)) % 4), validate=True).decode("utf-8")
            if dec.isprintable() or "\n" in dec:
                out.append("[b64]" + dec)
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass
    return list(dict.fromkeys(out))


@dataclass
class Verdict:
    action: str = "allow"          # allow | warn | block
    category: str = ""             # 命中類別
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    text: str = ""                 # 淨化後文字(可交給後續流程)


class InputGuard:
    _SEM = None  # 語意分類器為 process-level 單例(訓練約 2 秒)

    def __init__(self, enable_normalize: bool = True, enable_flood: bool = True, semantic: bool = False,
                 sem_block: float = 0.5):
        self.enable_normalize, self.enable_flood = enable_normalize, enable_flood
        self.semantic, self.sem_block = semantic, sem_block
        if semantic and InputGuard._SEM is None:
            from .semantic_guard import SemanticGuard
            InputGuard._SEM = SemanticGuard()
        self.recent: deque[str] = deque(maxlen=6)

    def check_text(self, text: str, source: str = "user", recent: Optional[deque] = None) -> Verdict:
        raw = text or ""
        v = Verdict(text=normalize(raw)[:MAX_CHARS] if self.enable_normalize else raw[:MAX_CHARS])
        if len(raw) > MAX_CHARS * 3:
            v.action, v.category, v.reasons = "block", "flood", [f"length={len(raw)}"]
            return v
        forms = variants(raw, ocr=(source == "image")) if self.enable_normalize else [raw]
        best: dict[str, float] = {}
        for form in forms:
            for cat, w, rx in PATTERNS:
                if rx.search(form):
                    best[cat] = max(best.get(cat, 0), w + (0.2 if form.startswith("[b64]") else 0))
        score = sum(best.values())
        if best:
            v.category = max(best, key=best.get)
            v.reasons = [f"{c}:{w}" for c, w in best.items()]
        v.score = score
        abusive = bool(ABUSE.search(" ".join(forms)))
        if abusive:
            v.reasons.append("abuse")
        if score >= 1.0:
            v.action = "block"
            if source != "user":
                v.category = f"{source}_{v.category}"
        elif abusive:
            # 帶有實際問題的情緒性用語 → 放行但標記降溫;純辱罵 → 攔截
            v.category = "abuse"
            v.action = "warn" if DOMAIN_HINT.search(raw) else "block"
        elif score >= 0.5:
            v.action, v.category = "warn", v.category or "suspicious"
        if self.semantic and v.action != "block" and v.text:
            p = InputGuard._SEM.prob(v.text)
            if p >= self.sem_block:
                v.action, v.category = "block", "semantic"
                v.reasons.append(f"semantic:{p:.2f}")
                v.score = max(v.score, p)
        if self.enable_flood and source == "user":
            key = re.sub(r"\s+", "", v.text)
            hist = self.recent if recent is None else recent     # 洪水偵測的「最近訊息」須屬於各自對話,不可跨使用者共用
            hist.append(key)
            if len(key) >= 6 and sum(1 for k in hist if k == key) >= 3:
                v.action, v.category = "block", "flood"
                v.reasons.append("repeat>=3")
        return v

    # ---- 圖片 ----
    def check_image(self, path: str) -> tuple[Verdict, str]:
        """回傳 (verdict, ocr_text)。圖片內文字視為「不可信資料」,與使用者文字走同一套偵測。"""
        from PIL import Image
        v = Verdict()
        try:
            import os
            if os.path.getsize(path) > MAX_IMG_BYTES:
                return Verdict("block", "image_too_large", 1.0, ["size"]), ""
            im = Image.open(path)
            im.verify()
            im = Image.open(path)
            if max(im.size) > MAX_IMG_SIDE:
                return Verdict("block", "image_too_large", 1.0, ["dim"]), ""
            clean = io.BytesIO()
            Image.open(path).convert("RGB").save(clean, format="PNG")  # 重新編碼:丟棄 EXIF / 附加資料
        except Exception as e:  # noqa: BLE001
            return Verdict("block", "bad_image", 1.0, [str(e)[:60]]), ""
        ocr = ocr_text(path)
        tv = self.check_text(ocr, source="image") if ocr.strip() else v
        return tv, ocr


_RAPID = None


def _rapid_ocr(path: str) -> str:
    """純 pip 的備援 OCR(rapidocr-onnxruntime,模型內建,中英文),無 tesseract 時使用。"""
    global _RAPID
    try:
        if _RAPID is None:
            from rapidocr_onnxruntime import RapidOCR
            _RAPID = RapidOCR()
        res, _ = _RAPID(path)
        return "\n".join(r[1] for r in (res or []))
    except Exception:  # noqa: BLE001
        return ""


def ocr_text(path: str) -> str:
    if not shutil.which("tesseract"):
        return _rapid_ocr(path)
    try:
        langs = "chi_tra+eng"
        r = subprocess.run(["tesseract", path, "stdout", "-l", langs, "--psm", "6"], capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            r = subprocess.run(["tesseract", path, "stdout", "-l", "eng", "--psm", "6"], capture_output=True, text=True, timeout=20)
        return r.stdout
    except Exception:  # noqa: BLE001
        return _rapid_ocr(path)


class OutputGuard:
    """使用者可見文字的最後一道關。"""
    FORBIDDEN = re.compile(r"(已(為您)?(核准|批准)全部|system prompt|<user_message>|</?system>)", re.I)

    def check(self, reply: str) -> tuple[bool, str]:
        if CANARY in reply:
            return False, "canary_leak"
        if self.FORBIDDEN.search(reply):
            return False, "forbidden_phrase"
        if len(reply) > 1200:
            return False, "too_long"
        return True, ""

    @staticmethod
    def echo_safe(user_text: str, limit: int = 30) -> str:
        """回述使用者內容(例如問題摘要)時:截斷、轉義、移除可能成為指令的片段。"""
        t = normalize(user_text)
        t = ORDER_ID_RE.sub("", t).strip()
        return html.escape(t[:limit])