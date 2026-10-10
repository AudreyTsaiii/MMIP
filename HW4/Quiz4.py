
import os
import re
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from PIL import Image


SEED = 42
MODEL_NAME = "Salesforce/blip-image-captioning-base"
DATASET_NAME = "jxie/flickr8k"

DATA_DIR = Path("./data_quiz1")
OUTPUT_DIR = Path("./results_quiz4")
MODEL_DIR = OUTPUT_DIR / "blip_finetuned"

BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 5e-5
MAX_LENGTH = 32
NUM_WORKERS = 0

# Evaluate this many test images to control runtime.
EVAL_IMAGES = 200
GEMINI_IMAGES = 16
GEMINI_MODEL = "gemini-3.5-flash-lite"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAPTION_COLUMNS = [f"caption_{i}" for i in range(5)]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_image(value):
    """Convert a Hugging Face image value to a PIL RGB image."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            import io
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    raise ValueError("Unable to decode an image in the dataset.")


def get_captions(row):
    return [
        str(row[c]).strip()
        for c in CAPTION_COLUMNS
        if c in row and row[c] is not None and str(row[c]).strip()
    ]


class CaptionDataset(Dataset):
    def __init__(self, source, items, processor, training=False):
        """
        items:
          training=True: [(row_index, caption), ...]
          training=False: [row_index, ...], using caption_0 for loss
        """
        self.source = source
        self.items = items
        self.processor = processor
        self.training = training

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        row_index = item
        row = self.source[int(row_index)]

        if self.training:
            captions = get_captions(row)
            caption = random.choice(captions) if captions else ""
        else:
            captions = get_captions(row)
            caption = captions[0] if captions else ""
        image = get_image(row["image"])

        encoded = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def split_dataset():
    """
    Use the official test split if available.
    Otherwise split image rows into train/validation/test.
    Split image rows BEFORE expanding the five captions to prevent leakage.
    """
    loaded = load_dataset("jxie/flickr8k", download_mode="reuse_dataset_if_exists")

    if isinstance(loaded, DatasetDict):
        if "train" in loaded:
            base = loaded["train"]
        else:
            base = loaded[next(iter(loaded.keys()))]

        if "test" in loaded and loaded["test"] is not base:
            official_test = loaded["test"]
            train_idx, val_idx = train_test_split(
                np.arange(len(base)),
                test_size=0.1,
                random_state=SEED,
            )
            return base, list(train_idx), list(val_idx), official_test, list(
                range(len(official_test))
            )

    else:
        base = loaded

    all_indices = np.arange(len(base))
    train_idx, temp_idx = train_test_split(
        all_indices, test_size=0.2, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED
    )

    return (
        base,
        list(train_idx),
        list(val_idx),
        base,
        list(test_idx),
    )


def train_model(processor, train_source, train_indices, val_source, val_indices):
    # train_items = []
    # for idx in train_indices:
    #     row = train_source[idx]
    #     for caption in get_captions(row):
    #         train_items.append((idx, caption))
    train_items = list(train_indices)

    train_dataset = CaptionDataset(
        train_source, train_items, processor, training=True
    )
    val_dataset = CaptionDataset(
        val_source, val_indices, processor, training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val_loss = float("inf")
    history = []

    for epoch in range(EPOCHS):
        start_time = time.perf_counter()
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / max(1, len(train_loader))

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    outputs = model(**batch)
                val_loss_sum += outputs.loss.item()

        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        elapsed = time.perf_counter() - start_time

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch_seconds": elapsed,
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            processor.save_pretrained(MODEL_DIR)
            print("  Saved best model.")

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv", index=False
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)
    best_model.to(DEVICE)
    best_model.eval()
    return best_model


def generate_caption(model, processor, image, num_beams):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=num_beams,
            do_sample=False,
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption, elapsed


def calculate_bleu(references, predictions):
    refs_tokenized = [
        [ref.lower().split() for ref in image_refs]
        for image_refs in references
    ]
    preds_tokenized = [caption.lower().split() for caption in predictions]

    return corpus_bleu(
        refs_tokenized,
        preds_tokenized,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )



def gemini_judge(client, image, caption_greedy, caption_beam, max_retries=3):
    prompt = f"""
You are evaluating an image-captioning system.
Inspect the provided image and evaluate each caption based on whether it
accurately describes visible image content.

Caption A (Greedy Search): {caption_greedy}
Caption B (Beam Search): {caption_beam}

Give each caption a score from 1 to 5:
5 = accurate and relevant
4 = mostly accurate, minor omissions
3 = partly accurate, some important details missing or incorrect
2 = mostly inaccurate
1 = unrelated to the image

Return ONLY valid JSON in this format:
{{"A_score": 1, "B_score": 1,
  "A_reason": "short explanation",
  "B_reason": "short explanation"}}
Do not judge writing style; judge image-content consistency.
"""

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt, image],
            )

            response_text = (response.text or "").strip()
            match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)

            if not match:
                raise ValueError(
                    f"Gemini did not return valid JSON: {response_text}"
                )

            result = json.loads(match.group(0))

            if not all(key in result for key in
                       ["A_score", "B_score", "A_reason", "B_reason"]):
                raise ValueError("Gemini JSON is missing required fields.")

            if result["A_score"] not in range(1, 6) or \
               result["B_score"] not in range(1, 6):
                raise ValueError("Gemini scores must be integers from 1 to 5.")

            return result

        except Exception as exc:
            error_text = str(exc)

            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                print("Gemini quota exceeded. Stop retrying.")
                break

            if attempt < max_retries:
                wait_seconds = 2 ** (attempt + 1)
                print(f"Gemini evaluation failed: {exc}")
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                print(f"Gemini evaluation failed after retries: {exc}")

    return {
        "A_score": None,
        "B_score": None,
        "A_reason": "Evaluation failed",
        "B_reason": "Evaluation failed",
    }


def evaluate_models(model, processor, test_source, test_indices):
    rng = random.Random(SEED)
    selected_indices = list(test_indices)
    rng.shuffle(selected_indices)
    selected_indices = selected_indices[:min(EVAL_IMAGES, len(selected_indices))]

    records = []
    references = []
    greedy_predictions = []
    beam_predictions = []
    greedy_times = []
    beam_times = []

    print(f"\nEvaluating {len(selected_indices)} test images...")

    for count, idx in enumerate(selected_indices, start=1):
        row = test_source[idx]
        image = get_image(row["image"])
        refs = get_captions(row)
        if not refs:
            continue

        greedy_caption, greedy_time = generate_caption(
            model, processor, image, num_beams=1
        )
        beam_caption, beam_time = generate_caption(
            model, processor, image, num_beams=3
        )

        references.append(refs)
        greedy_predictions.append(greedy_caption)
        beam_predictions.append(beam_caption)
        greedy_times.append(greedy_time)
        beam_times.append(beam_time)

        records.append({
            "row_index": idx,
            "references": " || ".join(refs),
            "greedy_caption": greedy_caption,
            "greedy_seconds": greedy_time,
            "beam_caption": beam_caption,
            "beam_seconds": beam_time,
        })

        if count % 20 == 0:
            print(f"Processed {count}/{len(selected_indices)} images.")

    if not records:
        raise RuntimeError("No valid test images/captions were found.")

    bleu_greedy = calculate_bleu(references, greedy_predictions)
    bleu_beam = calculate_bleu(references, beam_predictions)

    # Gemini evaluates a smaller subset to limit API calls.
    gemini_scores_greedy = []
    gemini_scores_beam = []

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        from google import genai
        client = genai.Client(api_key=api_key)
        gemini_count = min(GEMINI_IMAGES, len(records))
        print(f"\nGemini judging {gemini_count} images...")

        for i in range(gemini_count):
            idx = records[i]["row_index"]
            image = get_image(test_source[idx]["image"])
            result = gemini_judge(
                client,
                image,
                records[i]["greedy_caption"],
                records[i]["beam_caption"],
            )
            records[i]["gemini_greedy_score"] = result["A_score"]
            records[i]["gemini_beam_score"] = result["B_score"]
            records[i]["gemini_greedy_reason"] = result["A_reason"]
            records[i]["gemini_beam_reason"] = result["B_reason"]

            if result["A_score"] is not None:
                gemini_scores_greedy.append(float(result["A_score"]))
            if result["B_score"] is not None:
                gemini_scores_beam.append(float(result["B_score"]))
    else:
        print("\nGEMINI_API_KEY not set; skipping Gemini evaluation.")

    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / "caption_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = [
        {
            "method": "Greedy Search",
            "BLEU_4": bleu_greedy,
            "avg_generation_seconds": float(np.mean(greedy_times)),
            "Gemini_avg_score": (
                float(np.mean(gemini_scores_greedy))
                if gemini_scores_greedy else np.nan
            ),
        },
        {
            "method": "Beam Search (num_beams=3)",
            "BLEU_4": bleu_beam,
            "avg_generation_seconds": float(np.mean(beam_times)),
            "Gemini_avg_score": (
                float(np.mean(gemini_scores_beam))
                if gemini_scores_beam else np.nan
            ),
        },
    ]

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        OUTPUT_DIR / "method_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== Method Comparison ==========")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Loading dataset: {DATASET_NAME}")

    train_source, train_idx, val_idx, test_source, test_idx = split_dataset()
    print(
        f"Train images: {len(train_idx)} | "
        f"Validation images: {len(val_idx)} | "
        f"Test images: {len(test_idx)}"
    )

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = train_model(
        processor,
        train_source,
        train_idx,
        train_source,
        val_idx,
    )

    evaluate_models(model, processor, test_source, test_idx)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# def main():
#     set_seed(SEED)
#     print(f"Device: {DEVICE}")
#     print(f"Loading dataset: {DATASET_NAME}")

#     _, _, _, test_source, test_idx = split_dataset()

#     print(f"Test images: {len(test_idx)}")

#     print(f"Loading saved model from: {MODEL_DIR}")
#     processor = BlipProcessor.from_pretrained(MODEL_DIR)
#     model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)
#     model.to(DEVICE)
#     model.eval()

#     evaluate_models(model, processor, test_source, test_idx)

#     del model
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
