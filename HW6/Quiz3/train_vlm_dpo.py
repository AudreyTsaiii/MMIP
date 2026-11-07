import json
import gc
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATA_FILE = Path("data/vlm/preferences.jsonl")
OUTPUT_DIR = Path("outputs/vlm_dpo")

EPOCHS = 1
LEARNING_RATE = 5e-5
BETA = 0.1
GRAD_ACCUM = 8
LOG_INTERVAL = 8

MAX_PIXELS = 512 * 28 * 28

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = torch.device("cuda")

# ============================================================
# Load dataset
# ============================================================

with DATA_FILE.open(encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

assert len(data) == 360, f"Expected 360 pairs, got {len(data)}"

print(f"Loaded {len(data)} preference pairs.")

# ============================================================
# Processor
# ============================================================

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=256 * 28 * 28,
    max_pixels=MAX_PIXELS,
)

# ============================================================
# 4-bit QLoRA model
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("\nLoading Qwen2.5-VL-3B...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager",
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

lora_config = LoraConfig(
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
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()

# ============================================================
# Prepare multimodal inputs
# ============================================================

def prepare_example(row, answer):
    image_path = Path(row["image"])
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path

    image = Image.open(image_path).convert("RGB")

    user_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": row["prompt"]},
    ]

    prompt_messages = [
        {"role": "user", "content": user_content}
    ]

    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    )

    full_inputs = processor(
        text=[full_text],
        images=[image],
        return_tensors="pt",
    )

    prompt_length = prompt_inputs["input_ids"].shape[1]

    inputs = {
        k: v.to(device)
        for k, v in full_inputs.items()
        if isinstance(v, torch.Tensor)
    }

    labels = inputs["input_ids"].clone()
    labels[:, :prompt_length] = -100

    inputs["labels"] = labels

    if (labels != -100).sum().item() == 0:
        raise RuntimeError("No answer tokens available for DPO loss.")

    return inputs


def get_log_probability(row, answer):
    inputs = prepare_example(row, answer)

    labels = inputs.pop("labels")

    outputs = model(**inputs, use_cache=False)

    logits = outputs.logits[:, :-1, :]
    targets = labels[:, 1:]

    mask = targets != -100
    safe_targets = targets.masked_fill(~mask, 0)

    token_logps = F.log_softmax(
        logits.float(), dim=-1
    ).gather(
        dim=-1,
        index=safe_targets.unsqueeze(-1),
    ).squeeze(-1)

    result = (token_logps * mask).sum(dim=-1)

    del outputs, logits, token_logps, inputs, labels
    return result


# ============================================================
# Precompute reference model log probabilities
# ============================================================

print("\nPrecomputing frozen reference model log probabilities...")

reference_cache = {}

def get_reference(row, answer, cache_key):
    if cache_key in reference_cache:
        return reference_cache[cache_key]

    with torch.no_grad():
        value = get_log_probability(row, answer).detach().cpu()

    reference_cache[cache_key] = value
    return value


model.eval()

with model.disable_adapter():
    for row in tqdm(data, desc="Reference model"):
        filename = row["image_filename"]

        get_reference(
            row,
            row["chosen"],
            (filename, "chosen"),
        )

        get_reference(
            row,
            row["rejected"],
            (filename, row["negative_object"]),
        )

print("Reference cache entries:", len(reference_cache))

gc.collect()
torch.cuda.empty_cache()

# ============================================================
# DPO training
# ============================================================

trainable_params = [
    p for p in model.parameters() if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

print("\nStarting DPO training...")
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Beta:", BETA)
print("Gradient accumulation:", GRAD_ACCUM)

model.train()
optimizer.zero_grad(set_to_none=True)

global_step = 0
running_loss = 0.0

for epoch in range(EPOCHS):
    order = torch.randperm(len(data)).tolist()
    progress = tqdm(order, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for step, idx in enumerate(progress, start=1):
        row = data[idx]
        filename = row["image_filename"]

        chosen_logp = get_log_probability(
            row, row["chosen"]
        )

        rejected_logp = get_log_probability(
            row, row["rejected"]
        )

        ref_chosen = reference_cache[
            (filename, "chosen")
        ].to(device)

        ref_rejected = reference_cache[
            (filename, row["negative_object"])
        ].to(device)

        chosen_reward = chosen_logp - ref_chosen
        rejected_reward = rejected_logp - ref_rejected

        loss = -F.logsigmoid(
            BETA * (chosen_reward - rejected_reward)
        ).mean()

        (loss / GRAD_ACCUM).backward()

        running_loss += loss.item()

        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params, 1.0
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            avg_loss = running_loss / GRAD_ACCUM
            running_loss = 0.0

            progress.set_postfix(
                loss=f"{avg_loss:.4f}",
                step=global_step,
                gpu=f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
            )

        del chosen_logp, rejected_logp, loss

    if len(data) % GRAD_ACCUM != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

# ============================================================
# Save adapter
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model.config.use_cache = True
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 55)
print("VLM DPO training completed!")
print("=" * 55)
print("Adapter:", OUTPUT_DIR)
print("Preference pairs:", len(data))
print("Epochs:", EPOCHS)
print("Optimizer steps:", global_step)
print(
    "Peak GPU memory:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB",
)
