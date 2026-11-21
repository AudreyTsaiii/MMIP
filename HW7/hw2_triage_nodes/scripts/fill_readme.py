"""把 GPU 實驗結果填回 README。  python scripts/fill_readme.py results/real_base.json results/real_sft.json"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from triage.metrics import markdown_table  # noqa: E402
res = {}
for p in sys.argv[1:]:
    res.update(json.load(open(p, encoding="utf-8"))["results"])
order = [k for k in ("A_prompt", "B_sft", "C_decoding", "D_system_one") if k in res]
tbl = markdown_table({k: res[k] for k in order})
r = ROOT / "README.md"; s = r.read_text(encoding="utf-8")
a, b = s.index("<!-- BENCH:START -->"), s.index("<!-- BENCH:END -->")
r.write_text(s[:a] + "<!-- BENCH:START -->\n" + tbl + "\n" + s[b:], encoding="utf-8")
print(tbl)
