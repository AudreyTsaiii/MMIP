"""LLM 與 Embedding 後端。

- 設定環境變數 GEMINI_API_KEY 時使用 Google Gemini（生成 + Embedding + 多模態）。
- 沒有 key 時自動退回離線模式：TF-IDF 字元 n-gram Embedding + 抽取式回答，
  讓整個流程（建索引、參數掃描、UI）在沒有網路的情況下也能跑。
所有 LLM / Embedding 呼叫都有磁碟快取（.cache/），重跑實驗不會重複花 API 額度。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DB = ROOT / ".cache" / "calls.sqlite"


def _load_dotenv(path: Path = ROOT / ".env"):
    """讀取專案根目錄的 .env（GEMINI_API_KEY=...），不覆蓋已存在的環境變數。.env 已列入 .gitignore。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = re.split(r"\s+#", v, maxsplit=1)[0]  # 去掉行尾註解，例如 GEMINI_MODEL=xxx   # 說明
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
for _s in (sys.stdout, sys.stderr):  # Windows 終端機預設 cp950，印中文/符號會出錯
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "")  # 留空 = 自動挑選帳號可用的最新 Flash 模型
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "")
GEMINI_RPM = float(os.environ.get("GEMINI_RPM", "14"))  # 免費額度每分鐘請求數上限，付費帳號可調高
_last_call = [0.0]


def _throttle():
    gap = 60.0 / max(GEMINI_RPM, 0.1)
    wait = _last_call[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _version_key(name: str):
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", name)[:2]]
    return nums + [0] * (2 - len(nums))


def resolve_models() -> tuple[str, str]:
    """回傳 (生成模型, Embedding 模型)。有設定環境變數就用設定值，否則從帳號可用模型中挑選。"""
    gen, emb = GEMINI_MODEL, GEMINI_EMBED_MODEL
    if gen and emb:
        return gen, emb
    hit = CACHE.get("resolved_models")
    if hit:
        return gen or hit[0], emb or hit[1]
    from google import genai

    client = genai.Client()  # 要存在變數裡：臨時物件被回收後 client 會關閉，分頁列表就會失敗
    names = [m.name.split("/")[-1] for m in client.models.list()]
    flash = [n for n in names if "flash" in n and not re.search(r"lite|image|tts|live|audio|thinking|exp|preview", n)]
    flash = flash or [n for n in names if "flash" in n and not re.search(r"image|tts|live|audio", n)]
    g = max(flash, key=_version_key) if flash else "gemini-2.5-flash"
    embs = [n for n in names if "embedding" in n and "gemini" in n] or [n for n in names if "embedding" in n]
    e = "gemini-embedding-001" if "gemini-embedding-001" in names else (embs[0] if embs else "gemini-embedding-001")
    CACHE.put("resolved_models", [g, e])
    return gen or g, emb or e


def has_gemini() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


# ---------------------------------------------------------------- cache
class _Cache:
    def __init__(self, path: Path = CACHE_DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def get(self, k):
        r = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v, ensure_ascii=False)))
        self.db.commit()


CACHE = _Cache()


# ---------------------------------------------------------------- tokenization (中文字元 n-gram)
def char_tokens(text: str, n=(1, 2)) -> list[str]:
    text = text.lower()
    # 英數字當成一個 token，中文字以字元 n-gram 表示
    words = re.findall(r"[a-z0-9]+|[一-鿿]", text)
    toks = []
    for k in n:
        if k == 1:
            toks += words
        else:
            toks += ["".join(words[i:i + k]) for i in range(len(words) - k + 1)]
    return toks


# ---------------------------------------------------------------- embedders
class Embedder:
    name = "base"
    dim = None

    def fit(self, texts):  # 只有 TF-IDF 需要
        return self

    def embed(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    """離線 Embedding：字元 1~2 gram TF-IDF（L2 normalize 後內積 = cosine）。"""

    name = "tfidf-char12"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(analyzer=lambda s: char_tokens(s), sublinear_tf=True)
        self.fitted = False

    def fit(self, texts):
        self.vec.fit(texts)
        self.fitted = True
        return self

    def embed(self, texts, kind="doc"):
        if not self.fitted:
            self.fit(texts)
        m = self.vec.transform(texts)
        from sklearn.preprocessing import normalize

        return normalize(m).toarray().astype(np.float32)


class GeminiEmbedder(Embedder):
    def __init__(self, dim: int = 768):
        from google import genai

        self.client = genai.Client()
        self.name = resolve_models()[1]
        self.dim = dim

    def embed(self, texts, kind="doc"):
        from google.genai import types

        task = "RETRIEVAL_DOCUMENT" if kind == "doc" else "RETRIEVAL_QUERY"
        out: list = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            v = CACHE.get(CACHE.key("emb", self.name, self.dim, task, t))
            if v is None:
                todo.append(i)
            else:
                out[i] = v
        for s in range(0, len(todo), 90):
            idx = todo[s:s + 90]
            for attempt in range(6):
                try:
                    _throttle()
                    r = self.client.models.embed_content(
                        model=self.name,
                        contents=[texts[i] for i in idx],
                        config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
                    )
                    break
                except Exception as e:  # 免費額度 429 時退避重試
                    if attempt == 5:
                        raise
                    time.sleep(min(60, 5 * 2 ** attempt))
            for i, e in zip(idx, r.embeddings):
                out[i] = list(e.values)
                CACHE.put(CACHE.key("emb", self.name, self.dim, task, texts[i]), out[i])
        m = np.array(out, dtype=np.float32)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


class STEmbedder(Embedder):
    """本機 sentence-transformers 模型（例如 BAAI/bge-m3），需能連到 HuggingFace。"""

    def __init__(self, model="BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self.m = SentenceTransformer(model)

    def embed(self, texts, kind="doc"):
        return self.m.encode(texts, normalize_embeddings=True, batch_size=32).astype(np.float32)


def get_embedder(name: str | None = None) -> Embedder:
    name = name or os.environ.get("EMBEDDER", "auto")
    if name == "auto":
        name = "gemini" if has_gemini() else "tfidf"
    if name == "gemini":
        return GeminiEmbedder()
    if name == "tfidf":
        return TfidfEmbedder()
    return STEmbedder(name)


# ---------------------------------------------------------------- LLMs
class LLM:
    name = "base"
    online = False

    def generate(self, prompt: str, system: str | None = None, images: list | None = None,
                 json_mode: bool = False, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def generate_json(self, prompt, system=None, images=None):
        txt = self.generate(prompt, system=system, images=images, json_mode=True)
        m = re.search(r"\{.*\}|\[.*\]", txt, re.S)
        try:
            return json.loads(m.group(0) if m else txt)
        except Exception:
            return None


class GeminiLLM(LLM):
    """GEMINI_MODEL 可以填多個模型（逗號分隔），例如
       GEMINI_MODEL=gemini-3.5-flash-lite,gemini-3.1-flash-lite
    免費額度是「每個模型各自計算」，某個模型的每日額度用完（429 + per day）時，會自動換下一個，
    並記錄在快取裡，之後的步驟直接跳過已用完的模型，直到額度恢復。"""
    online = True

    def __init__(self, model: str | None = None):
        from google import genai

        self.client = genai.Client()
        spec = model or GEMINI_MODEL
        self.chain = [m.strip() for m in spec.split(",") if m.strip()] if spec else [resolve_models()[0]]
        self.name = self._pick()

    def _pick(self) -> str:
        now = time.time()
        for m in self.chain:
            until = CACHE.get(f"exhausted:{m}")
            if not until or until < now:
                return m
        raise RuntimeError(f"所有模型的每日免費額度都用完了：{self.chain}。請明天再重跑（已完成的部分有快取）。")

    def _mark_exhausted(self, msg: str):
        m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?", msg)
        secs = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60) if m and (m.group(1) or m.group(2)) else 12 * 3600
        CACHE.put(f"exhausted:{self.name}", time.time() + secs)
        old = self.name
        self.name = self._pick()
        print(f"[quota] {old} 今日額度用完，改用 {self.name}", flush=True)

    def _part(self, img):
        from google.genai import types

        if isinstance(img, (str, Path)):
            data = Path(img).read_bytes()
            mime = "image/png" if str(img).endswith(".png") else "image/jpeg"
            return types.Part.from_bytes(data=data, mime_type=mime)
        return img

    def generate(self, prompt, system=None, images=None, json_mode=False, temperature=0.0):
        from google.genai import types

        img_keys = [hashlib.md5(Path(i).read_bytes()).hexdigest() if isinstance(i, (str, Path)) else
                    hashlib.md5(str(i).encode()).hexdigest() for i in (images or [])]
        for m in self.chain:  # 任何一個模型答過就直接用快取
            hit = CACHE.get(CACHE.key("gen", m, system, prompt, img_keys, json_mode, temperature))
            if hit is not None:
                return hit
        contents_imgs = [self._part(i) for i in (images or [])]
        attempt = rotations = 0
        while True:
            gemma = "gemma" in self.name  # Gemma 不支援 system instruction 與 JSON mode，改寫進 prompt
            cfg = types.GenerateContentConfig(
                system_instruction=None if gemma else system,
                temperature=temperature,
                response_mime_type="application/json" if (json_mode and not gemma) else None,
            )
            p = (f"{system}\n\n{prompt}" if gemma and system else prompt)
            if gemma and json_mode:
                p += "\n只輸出 JSON。"
            try:
                _throttle()
                r = self.client.models.generate_content(model=self.name, contents=contents_imgs + [p], config=cfg)
                txt = r.text or ""
                break
            except Exception as e:
                msg = str(e)
                if ("429" in msg and re.search(r"per ?day|PerDay", msg, re.I)) or "404" in msg:
                    self._mark_exhausted(msg)  # 額度用完或模型不存在：換下一個模型，不算重試次數
                    continue
                attempt += 1
                if "503" in msg and attempt >= 2 and len(self.chain) > 1 and rotations < 2 * len(self.chain):  # 模型忙碌：暫時換下一個
                    print(f"[busy] {self.name} 忙碌中，暫時改用下一個模型", flush=True)
                    i = self.chain.index(self.name)
                    self.chain = self.chain[i + 1:] + self.chain[:i + 1]
                    self.name = self._pick()
                    attempt, rotations = 0, rotations + 1
                    continue
                if attempt >= 7:
                    raise
                time.sleep(min(60, 4 * 2 ** attempt))
        CACHE.put(CACHE.key("gen", self.name, system, prompt, img_keys, json_mode, temperature), txt)
        return txt


class OfflineLLM(LLM):
    """沒有 API key 時的替代品：從 prompt 內的【參考資料】挑出與問題最相關的句子當回答。
    僅用於讓流程可以離線跑通，不代表真正的 LLM 品質。"""

    name = "offline-extractive"

    def generate(self, prompt, system=None, images=None, json_mode=False, temperature=0.0):
        q = re.search(r"【問題】\s*(.+)", prompt)
        q = q.group(1) if q else prompt[-200:]
        ctx = re.findall(r"\[(\d+)\]\s*(.+)", prompt)
        if not ctx:
            return "（離線模式）無法回答。"
        qt = set(char_tokens(q, (2,)))
        best = []
        for n, text in ctx:
            for s in re.split(r"(?<=[。！？；])", text):
                if len(s) < 4:
                    continue
                st = set(char_tokens(s, (2,)))
                best.append((len(qt & st) / (len(qt) + 1e-9), n, s.strip()))
        best.sort(reverse=True)
        top = [b for b in best[:2] if b[0] > 0] or best[:1]
        return "".join(f"{s}[{n}]" for _, n, s in top)


def get_llm(name: str | None = None) -> LLM:
    name = name or os.environ.get("LLM", "auto")
    if name == "auto":
        name = "gemini" if has_gemini() else "offline"
    return GeminiLLM() if name == "gemini" else OfflineLLM()
