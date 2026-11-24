"""作業三進階 (10)：延伸 RAGAS，評估 VideoRAG。

文字 RAG 的 RAGAS 只看「文字 Context」。VideoRAG 多了三個會出錯的環節：找錯影片、找錯時間點、挑錯影格，
而且答案可能來自畫面而非語音。因此把評估拆成四層：

  A. 影片檢索（Action 1）  Video Hit@1、Video MRR
  B. 時間定位             Segment Hit@K（取回片段與標準時間區間有重疊）、Best tIoU（時間 IoU）
  C. 影格選取（Action 2）  Frame Temporal Precision（影格落在標準區間 ±5 秒內的比例）
                          Frame Relevance Precision（LLM 看圖判斷影格是否有助回答 → 視覺版 Context Precision）
  D. 生成（Action 3）      Multimodal Faithfulness（評審同時看逐字稿與影格，判斷每個陳述是否被支持）
                          Answer Relevancy、Answer Correctness（同 RAGAS）
再加上兩個分析：
  - 模態消融：full（語音+影格）/ speech-only（不給影格）/ visual-only（不給逐字稿）
  - 依題目所需模態（speech / visual）與影片類型（有說話 / 沒說話）分組

標準答案：python -m hw3.eval_video --make-qa 讓 Gemini 看「完整影片」（不經過我們的抽影格流程）出題，
產生 data/video_qa.json，請自行檢查修改後再評估（這也是報告中的人工檢核步驟）。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common.backends import ROOT, get_embedder, get_llm
from common.ragas_metrics import split_sents
from hw2.videorag import STORE, VIDEOS, VideoRAG

OUT = ROOT / "results"
QA_PATH = ROOT / "data" / "video_qa.json"

MAKE_QA = """你是出題老師。請仔細看完這部影片，出 {n} 題可以用影片內容回答的繁體中文問題，作為影片問答系統的測驗。
要求：一半題目的答案來自「說話內容」，一半來自「畫面」（操作步驟、畫面文字、物體）；若影片沒有人說話，全部出畫面題。
其中 2 題要有「追問」（followup），追問要用代名詞指涉前一題，例如「那它下一步是什麼？」。
輸出 JSON 陣列：[{{"q":"問題","answer":"標準答案","start":答案出現的起始秒數,"end":結束秒數,"needs":"speech 或 visual","followup":{{"q":"追問","answer":"答案","start":秒,"end":秒,"needs":"..."}} 或 null}}]"""

FRAME_JUDGE = """問題：{q}
依序附上 {n} 張系統選出的影格。請逐張判斷該影格對回答問題是否有幫助，只輸出 JSON：{{"useful": [true/false, ...共 {n} 個]}}"""

MM_JUDGE = """你是嚴格的多模態 RAG 評審。附上的圖片是系統選出的影格。請輸出 JSON：
{{"answer_statements":[{{"s":"系統回答中的原子陳述","supported":true/false（能否由逐字稿或附圖推得）}}],
 "generated_questions":["由系統回答反推的 3 個問題"],
 "noncommittal": true/false,
 "answer_vs_gt":{{"tp":數,"fp":數,"fn":數}}}}
【問題】{q}
【逐字稿與畫面描述】
{ctx}
【系統回答】{a}
【標準答案】{gt}"""


def make_qa(n=6):
    from google import genai

    llm = get_llm()
    assert llm.online, "出題需要 GEMINI_API_KEY"
    client = genai.Client()
    allqa = []
    for v in sorted(p for p in VIDEOS.glob("*") if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}):
        f = client.files.upload(file=str(v))
        while f.state and f.state.name == "PROCESSING":
            time.sleep(3)
            f = client.files.get(name=f.name)
        r = client.models.generate_content(model=llm.name, contents=[f, MAKE_QA.format(n=n)],
                                           config={"response_mime_type": "application/json"})
        import re

        vid = re.sub(r"\W+", "_", v.stem)
        for k, item in enumerate(json.loads(r.text)):
            item.update(id=f"{vid}_{k}", video_id=vid)
            allqa.append(item)
        print(v.name, "ok")
    QA_PATH.write_text(json.dumps(allqa, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"寫入 {QA_PATH}（{len(allqa)} 題），請人工檢查後再跑評估")


def tiou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def judge_answer(llm, emb, q, out, gt):
    ctx = "\n".join(f"{r['video']} {r['start']}-{r['end']}" for r in out["refs"])
    ctx = out.get("_ctx", ctx)
    j = llm.generate_json(MM_JUDGE.format(q=q, ctx=ctx, a=out["answer"], gt=gt),
                          images=[ROOT / f["path"] for f in out["frames"]]) or {}
    st = j.get("answer_statements") or []
    faith = float(np.mean([bool(x.get("supported")) for x in st])) if st else 0.0
    gq = [x for x in (j.get("generated_questions") or []) if isinstance(x, str)]
    arel = float(np.mean(emb.embed([q] + gq, kind="query")[1:] @ emb.embed([q], kind="query")[0])) \
        if gq and not j.get("noncommittal") else 0.0
    c = j.get("answer_vs_gt") or {}
    tp, fp, fn = (float(c.get(k, 0) or 0) for k in ("tp", "fp", "fn"))
    corr = tp / (tp + 0.5 * (fp + fn)) if tp + fp + fn else 0.0
    return dict(mm_faithfulness=faith, answer_relevancy=arel, answer_correctness=corr)


def evaluate(modes=("full", "speech_only", "visual_only")):
    llm, emb = get_llm(), get_embedder()
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    catalog = {c["video_id"]: c for c in json.loads((STORE / "catalog.json").read_text(encoding="utf-8"))}
    rows = []
    for mode in modes:
        vr = VideoRAG(llm, emb, use_frames=mode != "speech_only", use_speech=mode != "visual_only")
        for item in qa:
            vr.reset()
            chain = [item] + ([dict(item["followup"], id=item["id"] + "_f", video_id=item["video_id"])]
                              if item.get("followup") else [])
            for turn_i, it in enumerate(chain):
                out = vr.ask(it["q"])
                gold = (float(it["start"]), float(it["end"]))
                ranking = [v for v, _ in out["video_ranking"]]
                rr = 1 / (ranking.index(it["video_id"]) + 1) if it["video_id"] in ranking else 0
                segs = [(s, sc) for s, sc in vr.retrieve(out["rewritten"])[2]]
                seg_hit = any(s.video_id == it["video_id"] and tiou((s.start, s.end), gold) > 0 for s, _ in segs)
                best_tiou = max([tiou((s.start, s.end), gold) for s, _ in segs if s.video_id == it["video_id"]] or [0])
                fr = out["frames"]
                ftp = float(np.mean([f["video_id"] == it["video_id"] and gold[0] - 5 <= f["t"] <= gold[1] + 5
                                     for f in fr])) if fr else np.nan
                row = dict(mode=mode, id=it["id"], turn=turn_i + 1, video_id=it["video_id"],
                           video_has_speech=catalog[it["video_id"]]["has_speech"], needs=it.get("needs", "?"),
                           question=it["q"], rewritten=out["rewritten"], answer=out["answer"],
                           video_hit1=float(ranking[:1] == [it["video_id"]]), video_mrr=rr,
                           segment_hit=float(seg_hit), best_tiou=best_tiou, frame_temporal_precision=ftp)
                if llm.online:
                    if fr and mode == "full":  # 一次呼叫判斷全部影格（省額度）
                        j = llm.generate_json(FRAME_JUDGE.format(q=out["rewritten"], n=len(fr)),
                                              images=[ROOT / f["path"] for f in fr]) or {}
                        u = j.get("useful")
                        useful = ([bool(x) for x in u] if isinstance(u, list) else [bool(u)] * len(fr))[:len(fr)]
                        row["frame_relevance_precision"] = float(np.mean(useful)) if useful else np.nan
                    out["_ctx"] = "\n".join(s.text for s, _ in segs)
                    row.update(judge_answer(llm, emb, out["rewritten"], out, it["answer"]))
                rows.append(row)
        print(mode, "done")
    df = pd.DataFrame(rows)
    df.to_json(OUT / "hw3_video_eval.jsonl", orient="records", lines=True, force_ascii=False)
    num = [c for c in ["video_hit1", "video_mrr", "segment_hit", "best_tiou", "frame_temporal_precision",
                       "frame_relevance_precision", "mm_faithfulness", "answer_relevancy", "answer_correctness"] if c in df]
    overall = df.groupby("mode")[num].mean().round(3)
    by_need = df[df["mode"] == "full"].groupby("needs")[num].mean().round(3)
    by_type = df[df["mode"] == "full"].groupby("video_has_speech")[num].mean().round(3)
    by_turn = df[df["mode"] == "full"].groupby("turn")[num].mean().round(3)
    overall.to_csv(OUT / "hw3_video_overall.csv")
    by_need.to_csv(OUT / "hw3_video_by_needs.csv")
    by_type.to_csv(OUT / "hw3_video_by_videotype.csv")
    by_turn.to_csv(OUT / "hw3_video_by_turn.csv")
    for name, t in [("overall (by mode)", overall), ("by needed modality", by_need), ("by video has_speech", by_type),
                    ("by turn", by_turn)]:
        print("==", name)
        print(t.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-qa", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    if a.make_qa:
        make_qa(a.n)
    else:
        evaluate()
