"""一鍵執行全部實驗並產生報告 REPORT.md。

    python run_all.py              # 全部
    python run_all.py --skip-video # 影片還沒錄好時先跳過作業二
    python run_all.py --only hw4   # 只跑某一份作業（hw1 / hw3 / hw4 / video / report）

所有 Gemini 呼叫都會快取在 .cache/，中途因額度用完中斷時，隔天直接重跑會從斷點接續、不重複花額度。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from common.backends import ROOT, has_gemini

PY = sys.executable


def step(name, args):
    print(f"\n{'=' * 70}\n▶ {name}\n{'=' * 70}", flush=True)
    t = time.time()
    r = subprocess.run([PY, "-m", *args], cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    ok = r.returncode == 0
    print(f"{'✔' if ok else '✘'} {name}（{time.time() - t:.0f}s）", flush=True)
    return dict(step=name, ok=ok, seconds=round(time.time() - t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    online = has_gemini()
    print("後端：", "Gemini（已讀到 GEMINI_API_KEY）" if online else "離線模式（沒有 GEMINI_API_KEY，結果僅供流程測試）")
    log = []
    want = lambda k: a.only in (None, k)
    if want("hw1"):
        if online and not (ROOT / "data" / "qa_generated.jsonl").exists():
            log.append(step("作業一｜5W1H 擴充 QA Dataset", ["hw1.augment_qa"]))
        log.append(step("作業一｜Chunk / Overlap / Top-K 參數掃描", ["hw1.sweep"]))
        log.append(step("作業一｜Prompting Technique 比較", ["hw1.prompt_compare"]))
        log.append(step("作業一｜多輪對話展示", ["hw1.demo_multiturn"]))
    if want("hw3"):
        log.append(step("作業三｜RAGAS 評估（最佳設定）", ["hw3.eval_rag", "--n", "20"]))
        log.append(step("作業三｜RAGAS 消融：關閉多輪改寫", ["hw3.eval_rag", "--n", "0", "--no-rewrite"]))
    if want("hw4"):
        log.append(step("作業四｜建立知識圖譜", ["hw4.graphrag", "--build"]))
        log.append(step("作業四｜Benchmark 評估", ["hw4.evaluate"]))
    if want("video") and not a.skip_video:
        vids = [p for p in (ROOT / "data" / "videos").glob("*") if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}]
        if not vids:
            print("⚠️ data/videos/ 沒有影片，跳過作業二與影片評估")
        else:
            log.append(step("作業二｜影片庫建庫（抽影格、轉逐字稿、關鍵影格）", ["hw2.videorag", "--ingest"]))
            qa = ROOT / "data" / "video_qa.json"
            if not qa.exists() and online:
                log.append(step("作業三進階｜讓 Gemini 看完整影片出題", ["hw3.eval_video", "--make-qa"]))
                print("📝 已產生 data/video_qa.json，請檢查題目與時間點是否正確，修改後再跑：python run_all.py --only video")
            elif qa.exists():
                log.append(step("作業三進階｜VideoRAG 評估", ["hw3.eval_video"]))
    log.append(step("產生報告 REPORT.md", ["make_report"]))
    (ROOT / "results" / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [x["step"] for x in log if not x["ok"]]
    print("\n全部完成" if not bad else f"\n以下步驟失敗（多半是 API 額度用完，稍後重跑即可接續）：{bad}")


if __name__ == "__main__":
    main()
