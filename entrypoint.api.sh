#!/bin/bash
set -e

echo "============================================"
echo "  Twitter Sentiment Analysis — API Service  "
echo "============================================"


# ── Step 1: Preprocess data ───────────────────────────────────────────────────
if [ ! -f "$PROCESSED_DATA_DIR/train.csv" ] || [ ! -f "$PROCESSED_DATA_DIR/test.csv" ]; then
    echo ""
    echo "[1/3] Preprocessing data..."
    python scripts/preprocess.py \
        --output-dir "$PROCESSED_DATA_DIR" \
        ${SAMPLE_SIZE:+--sample-size $SAMPLE_SIZE}
    echo "[1/3] Preprocessing complete ✓"
else
    echo "[1/3] Preprocessed data already exists — skipping ✓"
fi

# ── Step 2: Train model ───────────────────────────────────────────────────────
if [ ! -f "$MODEL_OUTPUT/config.json" ]; then
    echo ""
    echo "[2/3] Training model..."
    echo "      Sample size : ${SAMPLE_SIZE:-full dataset}"
    echo "      Batch size  : ${BATCH_SIZE:-16}"
    echo "      Epochs      : ${NUM_EPOCHS:-3}"
    echo ""

    # Run training — if killed (OOM), print helpful message
    python scripts/train.py \
        ${SAMPLE_SIZE:+--sample-size $SAMPLE_SIZE} || {
        echo ""
        echo "╔══════════════════════════════════════════════════════╗"
        echo "║  ❌  Training was killed — likely OUT OF MEMORY      ║"
        echo "║                                                      ║"
        echo "║  Fix options:                                        ║"
        echo "║  1. Increase Docker memory (Settings > Resources)   ║"
        echo "║  2. Set SAMPLE_SIZE=1000 in your .env file          ║"
        echo "║  3. Set BATCH_SIZE=4 in your .env file              ║"
        echo "╚══════════════════════════════════════════════════════╝"
        exit 1
    }

    echo "[2/3] Training complete ✓"
else
    echo "[2/3] Model artifacts already exist — skipping training ✓"
fi

# ── Step 3: Start API ─────────────────────────────────────────────────────────
echo ""
echo "[3/3] Starting FastAPI server on port ${API_PORT:-8000}..."
exec uvicorn src.api:app \
    --host 0.0.0.0 \
    --port "${API_PORT:-8000}" \
    --workers 1
