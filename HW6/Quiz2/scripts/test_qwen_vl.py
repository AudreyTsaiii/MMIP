
import time
import requests
import torch

from io import BytesIO
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Limit image resolution to reduce visual tokens.
MIN_PIXELS = 128 * 128
MAX_PIXELS = 336 * 336

print("Loading model:", MODEL_ID)
print("GPU:", torch.cuda.get_device_name(0))

torch.cuda.empty_cache()

start_time = time.time()

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)

model.eval()

load_time = time.time() - start_time

print(f"Model loading time: {load_time:.2f}s")
print(
    "GPU memory after loading:",
    round(torch.cuda.memory_allocated() / 1024**3, 3),
    "GB",
)

# Download test image
image_url = (
    "https://huggingface.co/datasets/"
    "huggingface/documentation-images/resolve/main/"
    "p-blog/candy.JPG"
)

response = requests.get(image_url, timeout=30)
response.raise_for_status()

image = Image.open(BytesIO(response.content)).convert("RGB")

# Explicitly resize image to limit visual tokens.
image.thumbnail((448, 448))

prompt = "請用繁體中文描述這張圖片中的主要物件。"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

print("Input prepared.")
print(
    "GPU memory before inference:",
    round(torch.cuda.memory_allocated() / 1024**3, 3),
    "GB",
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

torch.cuda.synchronize()
start_time = time.time()

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

torch.cuda.synchronize()
latency = time.time() - start_time

input_length = inputs["input_ids"].shape[1]

answer = processor.decode(
    generated_ids[0][input_length:],
    skip_special_tokens=True,
)

print("\n========== Qwen2.5-VL Result ==========")
print("Prompt:", prompt)
print("Answer:", answer)
print(f"Latency: {latency:.2f}s")
print(
    "Peak GPU memory:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    "GB",
)
print("=======================================")
