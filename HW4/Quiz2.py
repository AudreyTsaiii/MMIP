
import json
import random
import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
MAX_VOCAB_SIZE = 30000
MIN_FREQ = 2
MAX_LENGTH = 256
EMBED_DIM = 128
HIDDEN_DIM = 128
RNN_DROPOUT = 0.3
LSTM_DROPOUT = 0.5
PATIENCE = 3

OUTPUT_DIR = Path("./results_quiz2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text):
    return re.findall(r"\b\w+\b|[^\w\s]", text.lower())


def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in counter.most_common(MAX_VOCAB_SIZE - 2):
        if count >= MIN_FREQ:
            vocab[word] = len(vocab)
    return vocab


class IMDbDataset(Dataset):
    def __init__(self, dataset, vocab):
        self.sequences = []
        self.labels = []

        for item in dataset:
            tokens = tokenize(item["text"])[:MAX_LENGTH]
            ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
            self.sequences.append(torch.tensor(ids or [vocab["<UNK>"]], dtype=torch.long))
            self.labels.append(int(item["label"]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, model_type):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

        if model_type == "RNN":
            self.recurrent = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
            self.dropout = nn.Dropout(RNN_DROPOUT)
        else:
            self.recurrent = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
            self.dropout = nn.Dropout(LSTM_DROPOUT)

        self.fc = nn.Linear(HIDDEN_DIM, 2)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.recurrent(packed)

        if isinstance(hidden, tuple):  # LSTM returns (hidden, cell)
            hidden = hidden[0]

        last_hidden = hidden[-1]
        return self.fc(self.dropout(last_hidden))


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for x, lengths, labels in loader:
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            logits = model(x, lengths)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)



def train_model(model_type, train_loader, val_loader, vocab_size):
    set_seed(SEED)
    print(f"\n========== Training {model_type} ==========")
    model = SentimentModel(vocab_size, model_type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_path = OUTPUT_DIR / f"{model_type.lower()}_best.pth"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for x, lengths, labels in train_loader:
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion)

        history.append({
            "model": model_type,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print("  Validation Loss improved. Model saved.")
        else:
            patience_counter += 1
            print(f"  No improvement: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / f"{model_type.lower()}_training_history.csv",
        index=False,
        encoding="utf-8-sig"
    )

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    print(f"Best validation Loss: {best_val_loss:.4f}")
    return model



def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    print("Loading IMDb dataset...")
    dataset = load_dataset("stanfordnlp/imdb")
    split = dataset["train"].train_test_split(test_size=0.2, seed=SEED, stratify_by_column="label")
    train_raw = split["train"]
    val_raw = split["test"]
    test_raw = dataset["test"]

    print(
        f"Train: {len(train_raw):,} | "
        f"Validation: {len(val_raw):,} | Test: {len(test_raw):,}"
    )

    vocab = build_vocab(train_raw["text"])
    with (OUTPUT_DIR / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    train_ds = IMDbDataset(train_raw, vocab)
    val_ds = IMDbDataset(val_raw, vocab)
    test_ds = IMDbDataset(test_raw, vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    results = []
    predictions = []

    for model_type in ["RNN", "LSTM"]:
        model = train_model(model_type, train_loader, val_loader, len(vocab))
        criterion = nn.CrossEntropyLoss()
        metrics, labels, preds, probs = evaluate(model, test_loader, criterion)

        result = {"model": model_type, **metrics}
        results.append(result)
        print(f"\n{model_type} test results:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        for i in range(len(labels)):
            predictions.append({
                "model": model_type,
                "true_label": int(labels[i]),
                "predicted_label": int(preds[i]),
                "negative_probability": float(probs[i][0]),
                "positive_probability": float(probs[i][1]),
            })

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    print("\nModel comparison:")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")

    RESULT_DIR = Path("./results_quiz2")

    rnn = pd.read_csv(RESULT_DIR / "rnn_training_history.csv")
    lstm = pd.read_csv(RESULT_DIR / "lstm_training_history.csv")

    print("RNN columns:", rnn.columns.tolist())
    print("LSTM columns:", lstm.columns.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, history, name in [(axes[0], rnn, "RNN"), (axes[1], lstm, "LSTM"),]:
        ax.plot(history["train_loss"], marker="o", label="Training Loss")
        ax.plot(history["val_loss"], marker="o", label="Validation Loss")
        ax.set_title(f"{name} Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(range(len(history)))
        ax.set_xticklabels(range(1, len(history) + 1))
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()