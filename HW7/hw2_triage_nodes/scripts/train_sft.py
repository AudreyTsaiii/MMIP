"""實驗 B:對 Qwen2.5-VL 做 LoRA SFT,讓它「原生」輸出分流 JSON。需 GPU(Colab T4 可跑 3B,~25 分鐘/2 epoch)。
  python scripts/gen_sft_data.py --n 1500
  python scripts/train_sft.py --out checkpoints/sft_lora
只對 assistant 的目標 JSON 計算 loss(prompt 部分 label=-100)。batch=1 + 梯度累積,避免 VLM 變長影像 padding 問題。"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
ap.add_argument("--out", default="checkpoints/sft_lora")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--accum", type=int, default=8)
a = ap.parse_args()

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402

from triage.backends import Message, QwenVLBackend  # noqa: E402
from triage.metrics import DATA  # noqa: E402
from triage.prompts import SYSTEM_SHORT  # noqa: E402

bk = QwenVLBackend(a.model, max_pixels=448 * 448)
proc, model = bk.proc, bk.model
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.print_trainable_parameters()


class DS(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ex = self.rows[i]
        hist = [Message(t["role"], t["text"], [str(DATA / t["image"])] if t.get("image") else []) for t in ex["turns"]]
        prompt, pil = bk.build_inputs(SYSTEM_SHORT, hist)
        full = prompt + ex["target"] + "<|im_end|>"
        enc = proc(text=[full], images=pil or None, return_tensors="pt")
        n_prompt = proc(text=[prompt], images=pil or None, return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone(); labels[:, :n_prompt] = -100
        item = {k: v for k, v in enc.items()}; item["labels"] = labels
        return item


def collate(batch):  # batch size 固定 1
    return {k: v for k, v in batch[0].items()}


args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=1, gradient_accumulation_steps=a.accum,
                         num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
                         logging_steps=10, save_strategy="no", fp16=not torch.cuda.is_bf16_supported(),
                         bf16=torch.cuda.is_bf16_supported(), remove_unused_columns=False, report_to=[])
Trainer(model=model, args=args, train_dataset=DS(DATA / "sft/train.jsonl"), data_collator=collate).train()
model.save_pretrained(a.out)
print("saved LoRA adapter ->", a.out)
