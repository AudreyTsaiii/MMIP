import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModel,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
)
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


# ==================== Configuration ====================

SEED = 42

DATASET_NAME = "jxie/flickr8k"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TEXT_MODEL_NAME = "gpt2"

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "results_quiz3_clip"
MODEL_DIR = OUTPUT_DIR / "clip_gpt2"
CACHE_DIR = OUTPUT_DIR / "cache"
EVALUATE_ONLY = False
RETRAIN = False

WEEK4_CANDIDATES = [
    ROOT_DIR / "results_quiz4",
    ROOT_DIR.parent / "HW4" / "results_quiz4",
    ROOT_DIR.parent / "results_quiz4",
]

BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 5e-5
MAX_LENGTH = 32
MAX_NEW_TOKENS = 30
NUM_WORKERS = 0

# Use the same evaluation size as Week 4.
EVAL_IMAGES = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAPTION_COLUMNS = [f"caption_{i}" for i in range(5)]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Reproducibility ====================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== Dataset Utilities ====================

def get_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            import io
            return Image.open(
                io.BytesIO(value["bytes"])
            ).convert("RGB")

        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")

    raise ValueError("Unable to decode an image in the dataset.")


def get_captions(row):
    return [
        str(row[column]).strip()
        for column in CAPTION_COLUMNS
        if column in row
        and row[column] is not None
        and str(row[column]).strip()
    ]


def split_dataset():
    """
    Follow the Week 4 split:
    - Split the original train set into 90% train and 10% validation.
    - Use the official test split if available.
    - Split image rows before expanding captions.
    """
    loaded = load_dataset(
        DATASET_NAME,
        download_mode="reuse_dataset_if_exists",
    )

    if isinstance(loaded, DatasetDict):
        if "train" in loaded:
            base = loaded["train"]
        else:
            base = loaded[next(iter(loaded.keys()))]

        if "test" in loaded:
            official_test = loaded["test"]

            train_idx, val_idx = train_test_split(
                np.arange(len(base)),
                test_size=0.1,
                random_state=SEED,
            )

            return (
                base,
                list(train_idx),
                list(val_idx),
                official_test,
                list(range(len(official_test))),
            )

    else:
        base = loaded

    all_indices = np.arange(len(base))

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=SEED,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=SEED,
    )

    return (
        base,
        list(train_idx),
        list(val_idx),
        base,
        list(test_idx),
    )


# ==================== Caption Dataset ====================

class CaptionDataset(Dataset):
    def __init__(
        self,
        source,
        indices,
        image_processor,
        tokenizer,
        training=False,
    ):
        self.source = source
        self.indices = indices
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.training = training

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row_index = int(self.indices[index])
        row = self.source[row_index]

        image = get_image(row["image"])
        captions = get_captions(row)

        if not captions:
            caption = ""
        elif self.training:
            caption = random.choice(captions)
        else:
            caption = captions[0]

        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        # Reserve one position for EOS.
        token_ids = self.tokenizer(
            caption,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH - 1,
        )["input_ids"]

        token_ids = token_ids + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(token_ids)

        padding_length = MAX_LENGTH - len(token_ids)

        token_ids += (
            [self.tokenizer.pad_token_id] * padding_length
        )
        attention_mask += [0] * padding_length

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(
                token_ids, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long
            ),
        }


# ==================== CLIP + GPT-2 Model ====================

class CLIPGPT2Captioner(nn.Module):
    def __init__(self):
        super().__init__()

        # Frozen CLIP vision encoder.
        self.image_encoder = CLIPVisionModel.from_pretrained(
            CLIP_MODEL_NAME
        )

        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False

        self.image_encoder.eval()

        image_dim = self.image_encoder.config.hidden_size
        text_dim = GPT2LMHeadModel.from_pretrained(
            TEXT_MODEL_NAME
        ).config.hidden_size

        # Map the image representation to GPT-2's hidden dimension.
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, text_dim),
            nn.Tanh(),
        )

        # Pretrained autoregressive text decoder.
        self.decoder = GPT2LMHeadModel.from_pretrained(
            TEXT_MODEL_NAME
        )

    def train(self, mode=True):
        super().train(mode)

        # Keep the frozen image encoder in evaluation mode.
        self.image_encoder.eval()
        return self

    def encode_prefix(self, pixel_values):
        with torch.no_grad():
            image_output = self.image_encoder(
                pixel_values=pixel_values
            )
            image_features = image_output.pooler_output

        return self.image_projection(image_features).unsqueeze(1)

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
    ):
        image_prefix = self.encode_prefix(pixel_values)

        # Convert caption token IDs to GPT-2 embeddings.
        text_embeddings = self.decoder.transformer.wte(
            input_ids
        )

        # Sequence: [image prefix, caption tokens].
        inputs_embeds = torch.cat(
            [image_prefix, text_embeddings],
            dim=1,
        )

        prefix_mask = torch.ones(
            attention_mask.size(0),
            1,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        full_attention_mask = torch.cat(
            [prefix_mask, attention_mask],
            dim=1,
        )

        # Ignore the image prefix and padding when calculating loss.
        text_labels = input_ids.clone()
        text_labels[attention_mask == 0] = -100

        prefix_labels = torch.full(
            (input_ids.size(0), 1),
            -100,
            dtype=torch.long,
            device=input_ids.device,
        )

        labels = torch.cat(
            [prefix_labels, text_labels],
            dim=1,
        )

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )


# ==================== Training and Validation ====================

def run_epoch(model, loader, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    use_amp = DEVICE.type == "cuda"

    for batch in loader:
        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(**batch)
                loss = outputs.loss

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = batch["input_ids"].size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


def train_model(
    image_processor,
    tokenizer,
    train_source,
    train_indices,
    val_source,
    val_indices,
):
    train_dataset = CaptionDataset(
        train_source,
        train_indices,
        image_processor,
        tokenizer,
        training=True,
    )

    val_dataset = CaptionDataset(
        val_source,
        val_indices,
        image_processor,
        tokenizer,
        training=False,
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

    model = CLIPGPT2Captioner().to(DEVICE)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    best_val_loss = float("inf")
    best_path = MODEL_DIR / "clip_gpt2_best.pth"
    history = []

    print("\n========== Training CLIP + GPT-2 ==========")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(
        "Trainable parameters:",
        sum(p.numel() for p in trainable_parameters),
    )

    for epoch in range(EPOCHS):
        start_time = time.perf_counter()

        train_loss = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
        )

        val_loss = run_epoch(
            model,
            val_loader,
        )

        elapsed = time.perf_counter() - start_time

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_seconds": elapsed,
        }
        history.append(record)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # CLIP is frozen, so only save the projection and decoder.
            torch.save(
                {
                    "image_projection":
                        model.image_projection.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "best_epoch": epoch + 1,
                    "val_loss": val_loss,
                },
                best_path,
            )

            print("  Saved best model.")

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=2, ensure_ascii=False)

    # Reload the best validation checkpoint.
    checkpoint = torch.load(
        best_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.image_projection.load_state_dict(
        checkpoint["image_projection"]
    )
    model.decoder.load_state_dict(checkpoint["decoder"])

    image_processor.save_pretrained(
        MODEL_DIR / "image_processor"
    )
    tokenizer.save_pretrained(
        MODEL_DIR / "tokenizer"
    )

    print(f"\nBest epoch: {checkpoint['best_epoch']}")
    print(f"Model saved to: {best_path}")

    return model


# ==================== Caption Generation ====================

def apply_repetition_penalty(logits, generated, penalty=1.1):
    if generated:
        ids = torch.tensor(sorted(set(generated)), device=logits.device)
        scores = logits[ids]
        logits[ids] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits


@torch.inference_mode()
def decode_caption(model, prefix, tokenizer, num_beams,
                   max_new_tokens=MAX_NEW_TOKENS,
                   repetition_penalty=1.1, length_penalty=1.0):
    eos = tokenizer.eos_token_id
    beams = [([], 0.0)]      # (token_ids, cumulative log-prob)
    finished = []            # (token_ids, normalized score)

    for _ in range(max_new_tokens):
        if len(beams[0][0]) > 0:
            ids = torch.tensor([b[0] for b in beams], device=DEVICE)
            embeds = torch.cat(
                [prefix.expand(len(beams), -1, -1),
                 model.decoder.transformer.wte(ids)],
                dim=1,
            )
        else:
            embeds = prefix

        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=DEVICE)
        logits = model.decoder(
            inputs_embeds=embeds, attention_mask=mask, use_cache=False
        ).logits[:, -1].float()

        candidates = []
        for i, (tokens, score) in enumerate(beams):
            row = apply_repetition_penalty(
                logits[i].clone(), tokens, repetition_penalty
            )
            log_probs = torch.log_softmax(row, dim=-1)
            top_lp, top_ids = log_probs.topk(num_beams)
            for lp, tok in zip(top_lp.tolist(), top_ids.tolist()):
                candidates.append((tokens + [tok], score + lp))

        candidates.sort(key=lambda c: c[1], reverse=True)

        beams = []
        for tokens, score in candidates:
            if tokens[-1] == eos:
                finished.append(
                    (tokens[:-1], score / (len(tokens) ** length_penalty))
                )
            else:
                beams.append((tokens, score))
            if len(beams) == num_beams:
                break

        if len(finished) >= num_beams or not beams:
            break

    for tokens, score in beams:
        finished.append((tokens, score / (max(len(tokens), 1) ** length_penalty)))

    return max(finished, key=lambda f: f[1])[0]


@torch.inference_mode()
def generate_caption(model, image_processor, tokenizer, image, num_beams=1):
    pixel_values = image_processor(
        images=image, return_tensors="pt"
    )["pixel_values"].to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    prefix = model.encode_prefix(pixel_values)
    token_ids = decode_caption(model, prefix, tokenizer, num_beams)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    caption = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return caption, elapsed


# ==================== BLEU and BERTScore ====================

def calculate_bleu(references, predictions):
    refs_tokenized = [
        [ref.lower().split() for ref in image_refs]
        for image_refs in references
    ]

    preds_tokenized = [
        caption.lower().split()
        for caption in predictions
    ]

    return float(corpus_bleu(
        refs_tokenized,
        preds_tokenized,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    ))


def calculate_bertscore_all(references, systems):
    """
    Each generated caption is compared against all five references.
    The maximum F1 across references is used for each image.
    BERTScore is calculated in one call for all systems.
    """
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise ImportError(
            "Please install BERTScore first: pip install bert-score"
        ) from exc

    candidates = []
    reference_sentences = []
    pair_positions = []

    for system_name, predictions in systems.items():
        for image_index, prediction in enumerate(predictions):
            for reference in references[image_index]:
                candidates.append(prediction)
                reference_sentences.append(reference)
                pair_positions.append((system_name, image_index))

    print("\nCalculating BERTScore F1...")

    _, _, f1_scores = bert_score(
        candidates,
        reference_sentences,
        lang="en",
        batch_size=32,
        device=str(DEVICE),
        verbose=True,
        rescale_with_baseline=False,
    )

    f1_values = f1_scores.detach().cpu().tolist()

    grouped_scores = {
        name: [[] for _ in references]
        for name in systems
    }

    for (system_name, image_index), score in zip(
        pair_positions, f1_values
    ):
        grouped_scores[system_name][image_index].append(score)

    # Max-reference F1 per image, then average over images.
    per_image_scores = {
        name: [
            max(scores) if scores else 0.0
            for scores in image_scores
        ]
        for name, image_scores in grouped_scores.items()
    }

    return {
        name: float(np.mean(scores))
        for name, scores in per_image_scores.items()
    }, per_image_scores


# ==================== Test Caption Generation ====================

def evaluate_clip_model(
    model,
    image_processor,
    tokenizer,
    test_source,
    test_indices,
):
    selected_indices = list(test_indices)
    random.Random(SEED).shuffle(selected_indices)
    selected_indices = selected_indices[
        :min(EVAL_IMAGES, len(selected_indices))
    ]

    records = []

    print(
        f"\nGenerating captions for "
        f"{len(selected_indices)} test images..."
    )

    for count, index in enumerate(selected_indices, start=1):
        row = test_source[int(index)]
        image = get_image(row["image"])
        references = get_captions(row)

        if not references:
            continue

        greedy_caption, greedy_seconds = generate_caption(
            model,
            image_processor,
            tokenizer,
            image,
            num_beams=1,
        )

        beam_caption, beam_seconds = generate_caption(
            model,
            image_processor,
            tokenizer,
            image,
            num_beams=3,
        )

        records.append({
            "row_index": int(index),
            "references": " || ".join(references),
            "greedy_caption": greedy_caption,
            "greedy_seconds": greedy_seconds,
            "beam_caption": beam_caption,
            "beam_seconds": beam_seconds,
        })

        if count % 20 == 0:
            print(
                f"Processed {count}/{len(selected_indices)} images."
            )

    if not records:
        raise RuntimeError("No valid test images/captions were found.")

    prediction_path = OUTPUT_DIR / "caption_predictions.csv"

    pd.DataFrame(records).to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nCaption predictions saved to: {prediction_path}")

    return pd.DataFrame(records)


# ==================== Compare with Week 4 BLIP ====================

def find_week4_predictions():
    for directory in WEEK4_CANDIDATES:
        path = directory / "caption_predictions.csv"
        if path.exists():
            return path

    checked_paths = "\n".join(
        str(directory / "caption_predictions.csv")
        for directory in WEEK4_CANDIDATES
    )

    raise FileNotFoundError(
        "Cannot find the Week 4 BLIP caption_predictions.csv.\n"
        "Checked these paths:\n" + checked_paths
    )


def parse_references(value):
    if pd.isna(value):
        return []

    return [
        caption.strip()
        for caption in str(value).split(" || ")
        if caption.strip()
    ]

def load_trained_model():
    model = CLIPGPT2Captioner().to(DEVICE)
    checkpoint = torch.load(
        MODEL_DIR / "clip_gpt2_best.pth",
        map_location=DEVICE,
        weights_only=True,
    )
    model.image_projection.load_state_dict(checkpoint["image_projection"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.eval()
    return model

def compare_with_week4(clip_df):
    week4_path = find_week4_predictions()
    print(f"\nLoading Week 4 results: {week4_path}")

    blip_df = pd.read_csv(week4_path)

    required_columns = {
        "row_index",
        "references",
        "greedy_caption",
        "beam_caption",
    }

    if not required_columns.issubset(blip_df.columns):
        raise ValueError(
            "Week 4 caption_predictions.csv is missing required columns."
        )

    # Compare only the same test images.
    merged = clip_df.merge(
        blip_df,
        on="row_index",
        how="inner",
        suffixes=("_clip", "_blip"),
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No common test images found between Week 4 and Quiz 3."
        )

    references = [
        parse_references(value)
        for value in merged["references_clip"]
    ]

    valid_indices = [
        index
        for index, refs in enumerate(references)
        if refs
    ]

    merged = merged.iloc[valid_indices].reset_index(drop=True)
    references = [references[index] for index in valid_indices]

    systems = {
        "BLIP Greedy": merged["greedy_caption_blip"].fillna("").tolist(),
        "BLIP Beam": merged["beam_caption_blip"].fillna("").tolist(),
        "CLIP-GPT2 Greedy": merged["greedy_caption_clip"].fillna("").tolist(),
        "CLIP-GPT2 Beam": merged["beam_caption_clip"].fillna("").tolist(),
    }

    bert_means, bert_per_image = calculate_bertscore_all(
        references,
        systems,
    )

    rows = []

    
    timing_columns = {
        "BLIP Greedy": "greedy_seconds_blip",
        "BLIP Beam": "beam_seconds_blip",
        "CLIP-GPT2 Greedy": "greedy_seconds_clip",
        "CLIP-GPT2 Beam": "beam_seconds_clip",
    }


    for name, predictions in systems.items():
        bleu = calculate_bleu(references, predictions)

        timing_column = timing_columns[name]
        if timing_column in merged.columns:
            avg_seconds = float(
                pd.to_numeric(
                    merged[timing_column],
                    errors="coerce",
                ).mean()
            )
        else:
            avg_seconds = float("nan")

        rows.append({
            "Model": name,
            "Num Images": len(references),
            "BLEU-4": bleu,
            "BERTScore-F1": bert_means[name],
            "Avg Generation Seconds": avg_seconds,
        })

        merged[f"{name}_BERTScore_F1"] = bert_per_image[name]

    comparison_df = pd.DataFrame(rows)

    comparison_path = OUTPUT_DIR / "backbone_comparison.csv"
    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    detailed_path = OUTPUT_DIR / "caption_comparison.csv"
    merged.to_csv(
        detailed_path,
        index=False,
        encoding="utf-8-sig",
    )

    results = {
        "dataset": DATASET_NAME,
        "clip_model": CLIP_MODEL_NAME,
        "text_decoder": TEXT_MODEL_NAME,
        "num_test_images": len(references),
        "bert_score_reference_method": "maximum F1 across references",
        "metrics": rows,
    }

    with open(
        OUTPUT_DIR / "comparison_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\n========== Week 4 vs Quiz 3 ==========")
    print(comparison_df.round(4).to_string(index=False))

    print(f"\nComparison saved to: {comparison_path}")
    print(f"Detailed captions saved to: {detailed_path}")


# def main():
#     set_seed(SEED)

#     print(f"Device: {DEVICE}")
#     print(f"Dataset: {DATASET_NAME}")
#     print(f"CLIP Encoder: {CLIP_MODEL_NAME}")
#     print(f"Text Decoder: {TEXT_MODEL_NAME}")

#     print("\nLoading dataset...")
#     (
#         train_source,
#         train_indices,
#         val_indices,
#         test_source,
#         test_indices,
#     ) = split_dataset()

#     print(
#         f"Train images: {len(train_indices)} | "
#         f"Validation images: {len(val_indices)} | "
#         f"Test images: {len(test_indices)}"
#     )

#     image_processor = CLIPImageProcessor.from_pretrained(
#         CLIP_MODEL_NAME
#     )

#     tokenizer = GPT2TokenizerFast.from_pretrained(
#         TEXT_MODEL_NAME
#     )

#     # GPT-2 has no separate padding token by default.
#     tokenizer.pad_token = tokenizer.eos_token

#     model = train_model(
#         image_processor,
#         tokenizer,
#         train_source,
#         train_indices,
#         train_source,
#         val_indices,
#     )

#     clip_df = evaluate_clip_model(
#         model,
#         image_processor,
#         tokenizer,
#         test_source,
#         test_indices,
#     )

#     compare_with_week4(clip_df)

#     del model

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     print("\nQuiz 3 completed.")


def main():
    set_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"CLIP Encoder: {CLIP_MODEL_NAME}")
    print(f"Text Decoder: {TEXT_MODEL_NAME}")

    prediction_path = OUTPUT_DIR / "caption_predictions.csv"

    if EVALUATE_ONLY:
        # Reuse existing CLIP-GPT2 predictions.
        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Saved predictions not found: {prediction_path}\n"
                "Set EVALUATE_ONLY = False to run the full pipeline."
            )

        print("\nEvaluation-only mode: skipping training and caption generation.")
        clip_df = pd.read_csv(prediction_path)
        print(f"Loaded {len(clip_df)} saved caption predictions.")

    else:
        # Full training and evaluation pipeline.
        print("\nLoading dataset...")
        (
            train_source,
            train_indices,
            val_indices,
            test_source,
            test_indices,
        ) = split_dataset()

        print(
            f"Train images: {len(train_indices)} | "
            f"Validation images: {len(val_indices)} | "
            f"Test images: {len(test_indices)}"
        )

        image_processor = CLIPImageProcessor.from_pretrained(
            CLIP_MODEL_NAME
        )
        tokenizer = GPT2TokenizerFast.from_pretrained(TEXT_MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        # 
        if (MODEL_DIR / "clip_gpt2_best.pth").exists() and not RETRAIN:
            model = load_trained_model()
        else:
            model = train_model(
                image_processor, tokenizer,
                train_source, train_indices, train_source, val_indices,
            )

        clip_df = evaluate_clip_model(
            model,
            image_processor,
            tokenizer,
            test_source,
            test_indices,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compare saved predictions and recalculate BLEU-4 / BERTScore.
    compare_with_week4(clip_df)

    print("\nQuiz 3 completed.")


if __name__ == "__main__":
    main()
