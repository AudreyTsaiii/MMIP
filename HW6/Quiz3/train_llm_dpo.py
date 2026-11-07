import os
import torch
import yaml

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_PATH = "data/preferences.jsonl"
OUTPUT_DIR = "results/checkpoints/qwen25-1.5b-dpo"

MAX_LENGTH = 512
SEED = 42


# ============================================================
# 2. Check GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(
    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
), "GB")


# ============================================================
# 3. Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================================================
# 4. Load model with 4-bit quantization
# ============================================================

print("\nLoading 4-bit model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)


# ============================================================
# 5. LoRA configuration
# ============================================================

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


# ============================================================
# 6. Load preference dataset
# ============================================================

print("\nLoading preference dataset...")

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)

def format_prompt(example):
    example["prompt"] = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": example["prompt"],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return example


dataset = dataset.map(format_prompt)

required_columns = {"prompt", "chosen", "rejected"}

if not required_columns.issubset(set(dataset.column_names)):
    raise ValueError(
        f"Dataset must contain columns: {required_columns}"
    )

print("Total preference pairs:", len(dataset))

if len(dataset) < 100:
    raise ValueError("At least 100 preference pairs are required.")


# ============================================================
# 7. DPO configuration
# ============================================================

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,

    # Training
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,

    # DPO
    beta=0.1,
    loss_type=["sigmoid"],

    # Memory
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },

    # Sequence length
    max_length=MAX_LENGTH,

    # Logging and saving
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,

    # Other
    optim="adamw_torch",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    seed=SEED,
    report_to="none",
    remove_unused_columns=False,
)


# ============================================================
# 8. Initialize DPOTrainer
# ============================================================

print("\nInitializing DPOTrainer...")

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)


# ============================================================
# 9. Train
# ============================================================

print("\nStarting DPO training...")

trainer.train()

# Save LoRA adapter
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining completed.")
print("Model saved to:", OUTPUT_DIR)