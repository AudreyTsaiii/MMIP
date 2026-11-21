#!/usr/bin/env bash
# Colab(T4/L4)一鍵跑完實驗 A~D。  !bash scripts/colab_run.sh
set -e
pip -q install torch transformers accelerate peft lm-format-enforcer qwen-vl-utils pydantic pillow scikit-learn gradio
python data/make_benchmark.py
python scripts/gen_sft_data.py --n 1500
python scripts/train_sft.py --out checkpoints/sft_lora
python scripts/run_bench.py --backend qwen --nodes A,C,D --out results/real_base.json
python scripts/run_bench.py --backend qwen --adapter checkpoints/sft_lora --nodes B --out results/real_sft.json
python scripts/fill_readme.py results/real_base.json results/real_sft.json
python scripts/eval_guard.py --backend qwen
python scripts/eval_offtopic.py --backend qwen
