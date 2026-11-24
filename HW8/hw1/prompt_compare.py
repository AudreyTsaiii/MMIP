"""作業一進階 (5)：找出最合適的 Prompting Technique。

在相同的檢索設定（sweep 選出的 chunk/overlap/top-k）下，比較 5 種 Prompting：
  plain / grounded / cot / fewshot / stepback（定義見 common/rag.py 的 PROMPTS）
以 RAGAS 指標（作業三的評估器）打分，綜合分數 = Faithfulness、Answer Relevancy、Answer Correctness 的平均
（這三項直接受 Prompt 影響；Context 類指標主要由檢索決定，只列出供參考）。
選出最佳技術後寫回 results/hw1_best_config.json，UI 預設就會用它。

用法：python -m hw1.prompt_compare --n 8 --dialogs 3（單輪 8 題 + 多輪 3 段×3 輪 = 17 題／技術，配合免費額度）
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common.backends import ROOT
from common.rag import PROMPTS
from hw3.eval_rag import run_eval

OUT = ROOT / "results"
KEY = ["faithfulness", "answer_relevancy", "answer_correctness"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--dialogs", type=int, default=3)
    ap.add_argument("--techniques", default=",".join(PROMPTS))
    a = ap.parse_args()
    rows = []
    for tech in a.techniques.split(","):
        df, summ = run_eval(tech, a.n, tag=f"prompt_{tech}", n_dialogs=a.dialogs)
        o = summ.loc["overall"].to_dict()
        o.update(technique=tech, judge=df["judge"].iloc[0],
                 avg_answer_chars=round(df["answer"].str.len().mean(), 1),
                 multiturn_correctness=summ.loc["multiturn", "answer_correctness"] if "multiturn" in summ.index else None)
        o["score"] = round(sum(o[k] for k in KEY) / len(KEY), 4)
        rows.append(o)
    res = pd.DataFrame(rows).set_index("technique").sort_values("score", ascending=False)
    res.to_csv(OUT / "hw1_prompt_compare.csv")
    print(res.to_string())
    best = res.index[0]
    if res["judge"].iloc[0] != "llm":
        print("離線模式（lexical judge）無法真正比較 Prompt，不更新設定。請設定 GEMINI_API_KEY 後重跑。")
        return
    cfgp = OUT / "hw1_best_config.json"
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    cfg["technique"] = best
    cfg["prompt_scores"] = res["score"].to_dict()
    cfgp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST technique:", best)


if __name__ == "__main__":
    main()
