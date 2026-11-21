"""實驗 A~D 量化比較。
  # 真實 VLM(Colab T4/L4):
  python scripts/run_bench.py --backend qwen --nodes A,C,D --out results/real_base.json
  python scripts/run_bench.py --backend qwen --adapter checkpoints/sft_lora --nodes B --out results/real_sft.json
  # 離線 smoke test(RuleBackend,非 VLM,只驗證流程):
  python scripts/run_bench.py --backend rule --nodes A,B,C,D --out results/smoke_rule.json
"""
import argparse
import gc
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from triage.backends import get_backend  # noqa: E402
from triage.metrics import aggregate, load_benchmark, markdown_table, score_one, to_history  # noqa: E402
from triage.nodes import NodeA, NodeB, NodeC, NodeD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="rule")
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
ap.add_argument("--adapter", default=None)
ap.add_argument("--nodes", default="A,B,C,D")
ap.add_argument("--repeat", type=int, default=3, help="每筆重複次數,延遲取平均(第1次為 warm-up 不計)")
ap.add_argument("--gpu-price", type=float, default=0.35, help="USD/hr,Colab T4 等級雲端 GPU 約 0.35")
ap.add_argument("--out", default="results/bench.json")
a = ap.parse_args()

bench = load_benchmark()
want = a.nodes.split(",")
results, detail = {}, {}


def run(node, label):
    rows, outs, cats = [], [], []
    for item in bench:
        hist = to_history(item["turns"])
        node.triage(hist)  # warm-up(載入 kernel / cache)
        runs = [node.triage(hist) for _ in range(a.repeat)]
        o = runs[-1]
        o.seconds = sum(r.seconds for r in runs) / len(runs)
        rows.append(score_one(o, item["gold"])); outs.append(o); cats.append(item["category"])
        detail.setdefault(label, []).append({"id": item["id"], "raw": o.raw[:300], "path": o.path,
                                              "error": o.error, **rows[-1]})
    results[label] = aggregate(rows, outs, cats, a.gpu_price)
    print(f"[{label}] done  schema_valid={results[label]['schema_valid%']}%  "
          f"intent={results[label]['intent_acc%']}%  lat={results[label]['latency_mean_s']}s")


kw = {"model_id": a.model} if a.backend == "qwen" else {}
base = get_backend(a.backend, **kw)
if "A" in want: run(NodeA(base), "A_prompt")
if "C" in want: run(NodeC(base), "C_decoding")
if "D" in want: run(NodeD(base), "D_system_one")
if "B" in want:
    if a.backend == "qwen":
        if not a.adapter:
            print("skip B: 需要 --adapter");
        else:
            del base; gc.collect()
            import torch; torch.cuda.empty_cache()
            run(NodeB(get_backend("qwen", model_id=a.model, adapter=a.adapter)), "B_sft")
    else:
        run(NodeB(base), "B_sft")

pathlib.Path(a.out).parent.mkdir(exist_ok=True, parents=True)
json.dump({"backend": a.backend, "model": a.model, "adapter": a.adapter, "results": results, "detail": detail},
          open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n" + markdown_table(results))
