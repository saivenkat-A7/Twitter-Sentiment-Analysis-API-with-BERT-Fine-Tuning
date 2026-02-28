import os
import re
import argparse
import pandas as pd
from datasets import load_dataset

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = os.environ.get("PROCESSED_DATA_DIR", "data/processed")


# ── Text Cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove HTML tags, URLs, special characters and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s'!?.,]", " ", text)  # keep useful punctuation
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


# ── Main ──────────────────────────────────────────────────────────────────────
def main(output_dir: str = OUTPUT_DIR, sample_size: int | None = None):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading IMDB dataset from Hugging Face …")
    dataset = load_dataset("imdb")

    train_df = pd.DataFrame(dataset["train"])
    test_df  = pd.DataFrame(dataset["test"])

    # Optional down-sampling for quick iteration
    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        test_df  = test_df.sample(n=min(sample_size // 5, len(test_df)),  random_state=42)

    print(f"Raw train size : {len(train_df):,}")
    print(f"Raw test  size : {len(test_df):,}")

    # Clean text
    print("Cleaning text …")
    train_df["text"]  = train_df["text"].apply(clean_text)
    test_df["text"]   = test_df["text"].apply(clean_text)

    # Rename label column so it matches the required schema
    train_df = train_df[["text", "label"]]
    test_df  = test_df[["text", "label"]]

    # Remove empty rows after cleaning
    train_df = train_df[train_df["text"].str.len() > 5].reset_index(drop=True)
    test_df  = test_df[test_df["text"].str.len()  > 5].reset_index(drop=True)

    # Save
    train_path = os.path.join(output_dir, "train.csv")
    test_path  = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"Saved {len(train_df):,} training rows  → {train_path}")
    print(f"Saved {len(test_df):,}  test     rows  → {test_path}")
    print("Preprocessing complete ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IMDB sentiment data")
    parser.add_argument("--output-dir",  default=OUTPUT_DIR, help="Directory for processed CSVs")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Limit dataset size (useful for fast local testing)")
    args = parser.parse_args()
    main(output_dir=args.output_dir, sample_size=args.sample_size)