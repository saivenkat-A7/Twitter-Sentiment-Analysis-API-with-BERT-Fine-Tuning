

import os
import gc
import json
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME   = os.environ.get("MODEL_NAME",   "distilbert-base-uncased")
MODEL_OUTPUT = os.environ.get("MODEL_OUTPUT", "model_output")
DATA_DIR     = os.environ.get("PROCESSED_DATA_DIR", "data/processed")
RESULTS_DIR  = os.environ.get("RESULTS_DIR", "results")

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-5))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE",    8))    # default 8 (memory safe)
NUM_EPOCHS    = int(os.environ.get("NUM_EPOCHS",    3))
MAX_LEN       = int(os.environ.get("MAX_LEN",       128))  # reduced from 256 → saves ~50% RAM
NUM_LABELS    = 2


# ── Dataset ───────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Free GPU/CPU memory after each batch
            del input_ids, attention_mask, labels, outputs
            gc.collect()

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    avg_loss = total_loss / len(loader)
    return acc, prec, rec, f1, avg_loss


# ── Training Loop ─────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(MODEL_OUTPUT, exist_ok=True)
    os.makedirs(RESULTS_DIR,  exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading preprocessed data …")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    if args.sample_size:
        train_df = train_df.sample(n=min(args.sample_size, len(train_df)), random_state=42)
        test_df  = test_df.sample(n=min(args.sample_size // 5, len(test_df)),  random_state=42)

    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # Tokenizer + Model
    print(f"Loading model: {MODEL_NAME} …")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(device)

    # Free memory after model load
    gc.collect()

    # Datasets / Loaders
    train_dataset = SentimentDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LEN
    )
    test_dataset  = SentimentDataset(
        test_df["text"].tolist(),  test_df["label"].tolist(),  tokenizer, MAX_LEN
    )

    # pin_memory=False and num_workers=0 to reduce RAM usage
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # Optimizer + Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    history = []
    best_f1 = 0.0
    start   = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            # Explicitly free batch tensors
            del input_ids, attention_mask, labels, outputs, loss
            gc.collect()

            if step % 50 == 0:
                print(f"  Epoch {epoch}/{NUM_EPOCHS}  Step {step}/{len(train_loader)}"
                      f"  Loss: {running_loss / step:.4f}")

        # Validation
        acc, prec, rec, f1, val_loss = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}"
              f"  f1={f1:.4f}  val_loss={val_loss:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": round(running_loss / len(train_loader), 4),
            "val_loss":   round(val_loss, 4),
            "accuracy":   round(acc, 4),
            "f1":         round(f1, 4),
        })

        if f1 >= best_f1:
            best_f1 = f1
            model.save_pretrained(MODEL_OUTPUT)
            tokenizer.save_pretrained(MODEL_OUTPUT)
            print(f"  ✓ Model saved (best f1={best_f1:.4f})")

        gc.collect()

    elapsed = time.time() - start

    # Final metrics
    acc, prec, rec, f1, _ = evaluate(model, test_loader, device)

    metrics = {
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1_score":  round(float(f1),   4),
    }
    run_summary = {
        "hyperparameters": {
            "model_name":    MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "batch_size":    BATCH_SIZE,
            "num_epochs":    NUM_EPOCHS,
        },
        "final_metrics": {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
        },
        "training_time_seconds": round(elapsed, 2),
        "history": history,
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"),     "w") as f:
        json.dump(metrics,     f, indent=2)
    with open(os.path.join(RESULTS_DIR, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"\nMetrics   → {RESULTS_DIR}/metrics.json")
    print(f"Summary   → {RESULTS_DIR}/run_summary.json")
    print(f"Artifacts → {MODEL_OUTPUT}/")
    print("Training complete ✓")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for sentiment analysis")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Limit rows used (e.g. 2000 for quick testing)")
    args = parser.parse_args()
    train(args)