"""作業二：多輪對話 VideoRAG（改編自 Jeong et al., "VideoRAG: Retrieval-Augmented Generation over Video Corpus",
Findings of ACL 2025；講義第 56~63 頁）。

影片庫：data/videos/ 底下自己錄的影片（有說話的講解影片 + 沒說話的操作影片）。

離線建庫（ingest）
  1. 以 1 fps 抽影格
  2. 語音轉文字：Gemini（音訊直接丟給模型，要求輸出帶時間戳的逐字稿）；本機有 openai-whisper 時也可用
     → 沒有語音的影片逐字稿為空，系統自動標記為 silent，之後完全依靠畫面資訊
  3. Frame Selection（講義第 60 頁）：影格 → 視覺向量（CLIP；沒有 CLIP 時用色彩直方圖+縮圖）→ K-means
     → 每群取最接近中心的影格當關鍵影格
  4. 關鍵影格描述：Gemini 看圖寫出「畫面內容、操作步驟、畫面上的文字」
  5. 依時間切成片段（預設 20 秒），每段 = 該時段逐字稿 + 該時段關鍵影格描述 → Embedding
線上問答（對應 VideoRAG 的三個動作）
  Action 1 Retrieval：Query（多輪改寫後）→ 片段檢索 → 依影片聚合分數 → 取最相關影片與片段
  Action 2 Frame Selection：在取回的片段附近，以「影格描述/CLIP 與問題的相似度」+ MMR 多樣性挑出 N 張影格
           （論文用 MLP 對隨機 40 組影格組合打分；我們沒有訓練資料，改用相似度 + MMR，報告中說明）
  Action 3 Generation：影格圖片 + 逐字稿片段 + 對話歷史 → Gemini 多模態生成，回覆附 Reference（影片名稱 + 時間戳）

用法：python -m hw2.videorag --ingest          # 建庫（影片放 data/videos/）
     python -m hw2.videorag --ask "影片裡怎麼新增圖層？"
     python -m hw2.app                        # 聊天 UI
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.backends import ROOT, get_embedder, get_llm
from common.rag import REWRITE_PROMPT

VIDEOS = ROOT / "data" / "videos"
STORE = ROOT / "results" / "video_store"
SEG_SEC = 20

ASR_PROMPT = """請把這段音訊逐字轉成繁體中文逐字稿，並切成 5~15 秒的片段。輸出 JSON 陣列：
[{"start": 起始秒數, "end": 結束秒數, "text": "內容"}]
如果整段沒有人說話（只有背景音、音樂或靜音），請輸出 []。"""

CAPTION_PROMPT = """這是一段影片在 {t} 秒的畫面。請用繁體中文 2~3 句描述：畫面中有什麼、正在進行什麼操作或動作、畫面上可讀到的文字（按鈕、選單、標題、程式碼）。只描述看得到的內容。"""

ANSWER_SYS = ("你是影片助教。只能根據提供的【影片片段】（逐字稿與影格圖片）回答，使用繁體中文。"
              "每個陳述後面用 [影片名稱 mm:ss] 標註出處。若影片中沒有相關內容，請直接說明。")


def fmt_t(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------- visual features
class VisualEncoder:
    """CLIP（sentence-transformers 的 clip-ViT-B-32）可用時用 CLIP，否則用色彩直方圖 + 16x16 灰階縮圖。"""

    def __init__(self):
        self.clip = None
        try:
            from sentence_transformers import SentenceTransformer

            self.clip = SentenceTransformer("clip-ViT-B-32")
            self.name = "clip-ViT-B-32"
        except Exception:
            self.name = "colorhist+thumb"

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        if self.clip is not None:
            from PIL import Image

            v = self.clip.encode([Image.open(p).convert("RGB") for p in paths], batch_size=32, normalize_embeddings=True)
            return np.asarray(v, dtype=np.float32)
        import cv2

        feats = []
        for p in paths:
            img = cv2.imread(str(p))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
            h /= h.sum() + 1e-9
            th = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (16, 16)).flatten() / 255.0
            f = np.concatenate([h * 4, th - th.mean()])
            feats.append(f / (np.linalg.norm(f) + 1e-9))
        return np.asarray(feats, dtype=np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray | None:
        if self.clip is None:
            return None
        return np.asarray(self.clip.encode(texts, normalize_embeddings=True), dtype=np.float32)


# ---------------------------------------------------------------- ingest
def extract_frames(video: Path, outdir: Path, fps: float = 1.0) -> list[tuple[float, Path]]:
    import cv2

    outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(vfps / fps)))
    frames, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / vfps
            h, w = img.shape[:2]
            if max(h, w) > 768:
                s = 768 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            p = outdir / f"f_{int(t):05d}.jpg"
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append((t, p))
        i += 1
    cap.release()
    return frames


def transcribe(video: Path, workdir: Path, llm) -> list[dict]:
    audio = workdir / "audio.mp3"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:  # pip install imageio-ffmpeg 會附帶 ffmpeg 執行檔（Windows 不用另外安裝）
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                        "-b:a", "48k", str(audio)], check=False, capture_output=True)
    if not audio.exists() or audio.stat().st_size < 2000:
        return []  # 沒有音軌
    try:  # 本機 whisper 優先（免費、不佔額度）
        import whisper

        m = whisper.load_model("small")
        r = m.transcribe(str(audio), language="zh", initial_prompt="以下是繁體中文的句子。")
        segs = [dict(start=s["start"], end=s["end"], text=s["text"].strip()) for s in r["segments"]]
        # whisper 在靜音時常產生幻覺短句，用 no_speech_prob 過濾
        segs = [s for s, raw in zip(segs, r["segments"]) if raw.get("no_speech_prob", 0) < 0.6 and s["text"]]
        return segs
    except Exception:
        pass
    if not llm.online:
        return []
    from google.genai import types

    part = types.Part.from_bytes(data=audio.read_bytes(), mime_type="audio/mp3")
    out = llm.generate_json(ASR_PROMPT, images=[part])
    return [dict(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]).strip())
            for s in (out or []) if isinstance(s, dict) and str(s.get("text", "")).strip()]


def select_keyframes(feats: np.ndarray, n_key: int) -> list[int]:
    from sklearn.cluster import KMeans

    n_key = min(n_key, len(feats))
    km = KMeans(n_clusters=n_key, n_init=10, random_state=0).fit(feats)
    idx = []
    for c in range(n_key):
        members = np.where(km.labels_ == c)[0]
        d = np.linalg.norm(feats[members] - km.cluster_centers_[c], axis=1)
        idx.append(int(members[np.argmin(d)]))
    return sorted(set(idx))


def ingest(videos_dir: Path = VIDEOS, key_per_min: int = 8, max_key: int = 64):
    llm, venc = get_llm(), VisualEncoder()
    STORE.mkdir(parents=True, exist_ok=True)
    catalog = []
    for v in sorted(p for p in videos_dir.glob("*") if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}):
        vid = re.sub(r"\W+", "_", v.stem)
        wd = STORE / vid
        print("ingest", v.name)
        frames = extract_frames(v, wd / "frames")
        if not frames:
            continue
        dur = frames[-1][0] + 1
        feats = venc.encode_images([p for _, p in frames])
        np.save(wd / "frame_feats.npy", feats)
        n_key = int(np.clip(round(dur / 60 * key_per_min), 4, max_key))
        keys = select_keyframes(feats, n_key)
        transcript = transcribe(v, wd, llm)
        caps = []
        for i in keys:
            t, p = frames[i]
            cap = llm.generate(CAPTION_PROMPT.format(t=int(t)), images=[p]).strip() if llm.online else ""
            caps.append(dict(frame=i, t=t, path=str(p.relative_to(ROOT)), caption=cap))
        meta = dict(video_id=vid, file=v.name, duration=dur, has_speech=bool(transcript), transcript=transcript,
                    keyframes=caps, n_frames=len(frames), visual_encoder=venc.name,
                    frames=[dict(t=t, path=str(p.relative_to(ROOT))) for t, p in frames])
        (wd / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        catalog.append(dict(video_id=vid, file=v.name, duration=dur, has_speech=meta["has_speech"]))
    (STORE / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done:", catalog)


# ---------------------------------------------------------------- index & chat
@dataclass
class Segment:
    video_id: str
    file: str
    start: float
    end: float
    speech: str
    visual: str
    keyframes: list = field(default_factory=list)

    @property
    def text(self):
        s = f"【影片】{self.file}（{fmt_t(self.start)}–{fmt_t(self.end)}）\n"
        if self.speech:
            s += f"【逐字稿】{self.speech}\n"
        if self.visual:
            s += f"【畫面】{self.visual}"
        return s


class VideoRAG:
    def __init__(self, llm=None, embedder=None, seg_sec: int = SEG_SEC, n_frames: int = 4, top_seg: int = 3,
                 use_frames: bool = True, use_speech: bool = True):
        self.llm, self.emb = llm or get_llm(), embedder or get_embedder()
        self.n_frames, self.top_seg, self.use_frames, self.use_speech = n_frames, top_seg, use_frames, use_speech
        self.venc = None
        self.metas = {m["video_id"]: json.loads((STORE / m["video_id"] / "meta.json").read_text(encoding="utf-8"))
                      for m in json.loads((STORE / "catalog.json").read_text(encoding="utf-8"))}
        self.segs: list[Segment] = []
        for vid, m in self.metas.items():
            for s in np.arange(0, m["duration"], seg_sec):
                e = min(m["duration"], s + seg_sec)
                sp = " ".join(x["text"] for x in m["transcript"] if x["start"] < e and x["end"] > s) if use_speech else ""
                kf = [k for k in m["keyframes"] if s <= k["t"] < e]
                vis = " ".join(f"({fmt_t(k['t'])}) {k['caption']}" for k in kf if k["caption"]) if use_frames else ""
                self.segs.append(Segment(vid, m["file"], float(s), float(e), sp, vis, kf))
        texts = [s.text for s in self.segs]
        self.emb.fit(texts + [k["caption"] for m in self.metas.values() for k in m["keyframes"]])
        self.SV = self.emb.embed(texts)
        self.history = []

    def reset(self):
        self.history = []

    # Action 1
    def retrieve(self, q: str):
        qv = self.emb.embed([q], kind="query")[0]
        sims = self.SV @ qv
        vid_score = {}
        for s, sc in zip(self.segs, sims):
            vid_score[s.video_id] = max(vid_score.get(s.video_id, -1), sc)
        videos = sorted(vid_score, key=lambda v: -vid_score[v])
        order = np.argsort(-sims)[: self.top_seg]
        return videos, vid_score, [(self.segs[i], float(sims[i])) for i in order]

    # Action 2
    def select_frames(self, q: str, segs: list) -> list[dict]:
        cands = []
        for s, _ in segs:
            m = self.metas[s.video_id]
            cands += [dict(k, video_id=s.video_id, file=s.file) for k in m["keyframes"]
                      if s.start - SEG_SEC / 2 <= k["t"] < s.end + SEG_SEC / 2]
        if not cands:
            return []
        caps = [c["caption"] or c["file"] for c in cands]
        rel = self.emb.embed(caps) @ self.emb.embed([q], kind="query")[0]
        if self.venc is None:
            self.venc = VisualEncoder()
        tv = self.venc.encode_text([q])
        if tv is not None:  # CLIP 文字→影像相似度
            iv = self.venc.encode_images([ROOT / c["path"] for c in cands])
            rel = 0.5 * rel + 0.5 * (iv @ tv[0])
        feats = self.emb.embed(caps)
        chosen = []
        while len(chosen) < min(self.n_frames, len(cands)):  # MMR：相關且彼此不重複
            best, bs = None, -1e9
            for i in range(len(cands)):
                if i in chosen:
                    continue
                red = max((float(feats[i] @ feats[j]) for j in chosen), default=0)
                sc = 0.7 * rel[i] - 0.3 * red
                if sc > bs:
                    best, bs = i, sc
            chosen.append(best)
        return [dict(cands[i], score=round(float(rel[i]), 3)) for i in sorted(chosen, key=lambda i: cands[i]["t"])]

    def rewrite(self, q):
        if not self.history:
            return q
        hist = "\n".join(f"使用者：{h['user']}\n助教：{h['answer']}" for h in self.history[-3:])
        if self.llm.online:
            return self.llm.generate(REWRITE_PROMPT.format(hist=hist, q=q)).strip().splitlines()[0]
        return f"{self.history[-1]['rewritten']}；{q}"

    # Action 3
    def ask(self, q: str) -> dict:
        sq = self.rewrite(q)
        videos, vscore, segs = self.retrieve(sq)
        frames = self.select_frames(sq, segs) if self.use_frames else []
        ctx = "\n\n".join(s.text for s, _ in segs)
        flist = "\n".join(f"圖{i + 1}：{f['file']} {fmt_t(f['t'])}" for i, f in enumerate(frames))
        hist = "\n".join(f"使用者：{h['user']}\n助教：{h['answer']}" for h in self.history[-3:])
        prompt = (f"{'【先前對話】' + chr(10) + hist + chr(10) * 2 if hist else ''}【影片片段】\n{ctx}\n\n"
                  f"【附上的影格】\n{flist or '（無）'}\n\n【問題】{sq}\n請回答並標註 [影片名稱 mm:ss]。")
        if self.llm.online:
            ans = self.llm.generate(prompt, system=ANSWER_SYS, images=[ROOT / f["path"] for f in frames])
        else:
            ans = self.llm.generate("\n".join(f"[{i + 1}] {s.text}" for i, (s, _) in enumerate(segs)) + f"\n【問題】{sq}")
        refs = [dict(video=s.file, start=fmt_t(s.start), end=fmt_t(s.end), score=round(sc, 3),
                     modality=("speech+visual" if s.speech and s.visual else "speech" if s.speech else "visual"))
                for s, sc in segs]
        out = dict(user=q, rewritten=sq, answer=ans.strip(), refs=refs, frames=frames,
                   video_ranking=[(v, round(float(vscore[v]), 3)) for v in videos])
        self.history.append(out)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--videos", default=str(VIDEOS))
    ap.add_argument("--ask", default=None)
    a = ap.parse_args()
    if a.ingest:
        ingest(Path(a.videos))
    if a.ask:
        r = VideoRAG().ask(a.ask)
        print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
