import os
import argparse

import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model_output")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
MAX_LEN    = int(os.environ.get("MAX_LEN",   256))
LABEL_MAP  = {0: "negative", 1: "positive"}


# ── Inference Helper ──────────────────────────────────────────────────────────
def predict_batch(texts: list[str], model, tokenizer, device) -> list[dict]:
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids      = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs      = torch.softmax(logits, dim=1).cpu().numpy()
    pred_ids   = probs.argmax(axis=1)

    return [
        {"sentiment": LABEL_MAP[int(pid)], "confidence": round(float(probs[i, pid]), 4)}
        for i, pid in enumerate(pred_ids)
    ]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Batch sentiment prediction")
    parser.add_argument("--input-file",  required=True, help="Path to input CSV (needs 'text' column)")
    parser.add_argument("--output-file", required=True, help="Path to write output CSV")
    parser.add_argument("--model-path",  default=MODEL_PATH, help="Directory with model artifacts")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path} …")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_path)
    model     = DistilBertForSequenceClassification.from_pretrained(args.model_path).to(device)
    model.eval()

    # Load data
    df = pd.read_csv(args.input_file)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a column named 'text'.")

    texts = df["text"].fillna("").astype(str).tolist()
    print(f"Running inference on {len(texts):,} rows …")

    results: list[dict] = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk   = texts[i : i + BATCH_SIZE]
        results.extend(predict_batch(chunk, model, tokenizer, device))
        print(f"  Processed {min(i + BATCH_SIZE, len(texts)):,}/{len(texts):,}")

    df["sentiment"]  = [r["sentiment"]  for r in results]
    df["confidence"] = [r["confidence"] for r in results]

    df.to_csv(args.output_file, index=False)
    print(f"Predictions saved → {args.output_file}  ✓")


if __name__ == "__main__":
    main()