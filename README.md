#  Twitter Sentiment Analysis API with BERT Fine-Tuning

A production-ready sentiment analysis service built with **DistilBERT** (fine-tuned on IMDB reviews), served via **FastAPI**, visualised with **Streamlit**, and fully containerised with **Docker**.



## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  docker-compose                       │
│                                                      │
│  ┌─────────────────┐     ┌──────────────────────┐   │
│  │  Streamlit UI   │────▶│   FastAPI Backend     │   │
│  │  (port 8501)    │     │   (port 8000)         │   │
│  └─────────────────┘     └──────────┬───────────┘   │
│                                     │                │
│                           ┌─────────▼──────────┐    │
│                           │  DistilBERT Model  │    │
│                           │  (model_output/)   │    │
│                           └────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

**Data flow:**
1. User enters text in the Streamlit UI
2. UI sends `POST /predict` to the FastAPI backend
3. FastAPI tokenises text → runs inference with DistilBERT → returns `{sentiment, confidence}`
4. UI renders result + live charts (Pandas + Matplotlib)

---

##  Model Choice & Rationale

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `bert-base-uncased` | 440 MB | baseline | 93% |
| **`distilbert-base-uncased`**  | **265 MB** | **60% faster** | **97% of BERT** |
| `roberta-base` | 500 MB | similar to BERT | 94–95% |

**DistilBERT** was chosen because:
- ~40% smaller model size → faster Docker image builds and lower memory footprint
- 60% faster inference → better API latency in production
- Retains 97% of BERT's NLU capability (Sanh et al., 2019)
- Ideal for binary classification tasks like sentiment analysis

**Dataset:** IMDB movie reviews (50,000 samples, balanced positive/negative) from Hugging Face Datasets.

---

##  Project Structure

```
twitter-sentiment/
├── data/
│   ├── raw/                  # Raw downloaded data (gitignored)
│   ├── processed/            # Cleaned train.csv & test.csv
│   └── unseen/               # CSV files for batch prediction
├── model_output/             # Fine-tuned model artifacts (gitignored)
├── results/                  # metrics.json & run_summary.json
├── scripts/
│   ├── preprocess.py         # Data cleaning & splitting
│   ├── train.py              # BERT fine-tuning
│   └── batch_predict.py      # Bulk inference on CSV
├── src/
│   ├── api.py                # FastAPI application
│   └── ui.py                 # Streamlit web interface
├── tests/
│   └── test_api.py           # pytest API tests
├── Dockerfile.api            # API container definition
├── Dockerfile.ui             # UI container definition
├── docker-compose.yml        # Service orchestration
├── requirements.txt          # Training dependencies
├── requirements.api.txt      # API-only dependencies
├── requirements.ui.txt       # UI-only dependencies
├── .env.example              # Environment variable template
└── README.md
```

---

##  Quick Start

> **Prerequisites:** Docker ≥ 24, Docker Compose v2, Python 3.11+

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd twitter-sentiment

# 2. Set up environment
cp .env.example .env

# 3. Install training dependencies
pip install -r requirements.txt

# 4. Preprocess data
python scripts/preprocess.py

# 5. Train the model 
python scripts/train.py

# 6. Build & launch all services
docker-compose up --build -d

# 7. Open the UI
open http://localhost:8501

# 8. Or call the API directly
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely fantastic!"}'
```

---

##  Step-by-Step Setup

### Step 1 — Environment

```bash
cp .env.example .env
# Edit .env if you want to change ports or model hyperparameters
```

### Step 2 — Preprocess Data

```bash
# Full IMDB dataset (25K train, 25K test — recommended)
python scripts/preprocess.py

# Quick test run with a small sample
python scripts/preprocess.py --sample-size 2000
```

Output: `data/processed/train.csv` and `data/processed/test.csv`

### Step 3 — Train the Model

```bash
# Full training (recommended for production)
python scripts/train.py

# Quick iteration with fewer samples
python scripts/train.py --sample-size 2000
```

Outputs:
- `model_output/` — model weights, config, tokenizer files
- `results/metrics.json` — accuracy, precision, recall, F1
- `results/run_summary.json` — hyperparameters + training history

> The model must be trained **before** building the Docker image. `Dockerfile.api` copies `model_output/` into the image.

### Step 4 — Launch Services

```bash
# Build and start all services in detached mode
docker-compose up --build -d

# Check that both containers are healthy
docker ps

# View logs
docker-compose logs -f api
docker-compose logs -f ui
```

Wait ~60 seconds for the API to load the model. Status becomes `(healthy)` once ready.

---

## 📡 API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

### `GET /health`

Liveness probe — returns 200 when the API is operational.

**Response:**
```json
{
  "status": "ok",
  "uptime_secs": 142.3,
  "model_loaded": true
}
```

---

### `POST /predict`

Run sentiment analysis on a single text string.

**Request:**
```json
{ "text": "I love this product!" }
```

**Response (200):**
```json
{
  "sentiment": "positive",
  "confidence": 0.9871,
  "text": "I love this product!"
}
```

**Error responses:**
| Code | Reason |
|------|--------|
| 422 | Empty text, whitespace-only, or text > 10,000 chars |
| 503 | Model not yet loaded |

**Example with curl:**
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "The customer service was terrible."}'
```

**Python example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Absolutely loved the experience!"}
)
print(response.json())
# {'sentiment': 'positive', 'confidence': 0.9934, 'text': 'Absolutely loved the experience!'}
```

---

##  Streamlit UI

Open `http://localhost:8501` in your browser.

**Features:**
-  **Predict tab** — Enter text, see sentiment + confidence gauge chart
-  **History & Charts tab** — Pie chart, confidence histogram, trend over time, CSV download
-  **Model Performance tab** — Metrics bar chart, training loss curves, hyperparameter table

---

##  Batch Prediction

```bash
# Prepare input CSV (must have a 'text' column)
cat > data/unseen/tweets.csv << EOF
text
"I absolutely love the new update!"
"This is the worst experience ever."
"Pretty average, nothing special."
EOF

# Run batch inference
python scripts/batch_predict.py \
    --input-file  data/unseen/tweets.csv \
    --output-file results/predictions.csv

# View results
cat results/predictions.csv
```

**Output CSV format:**

| text | sentiment | confidence |
|------|-----------|------------|
| I absolutely love the new update! | positive | 0.9921 |
| This is the worst experience ever. | negative | 0.9874 |
| Pretty average, nothing special. | negative | 0.6123 |

---

##  Running Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests cover: health check (200), predict response schema, sentiment label validity, confidence range, empty/whitespace/oversized text error handling.

---

##  Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `8000` | Host port for the FastAPI service |
| `UI_PORT` | `8501` | Host port for the Streamlit service |
| `MODEL_PATH` | `/app/model_output` | Path to model artifacts inside the container |
| `MAX_LEN` | `256` | Maximum tokenisation length |
| `API_URL` | `http://api:8000` | API URL used by the UI (internal Docker network) |
| `MODEL_NAME` | `distilbert-base-uncased` | Hugging Face model identifier for training |
| `LEARNING_RATE` | `2e-5` | AdamW learning rate |
| `BATCH_SIZE` | `10` | Training & inference batch size |
| `NUM_EPOCHS` | `2` | Number of training epochs |
| `PROCESSED_DATA_DIR` | `data/processed` | Directory for train/test CSVs |
| `MODEL_OUTPUT` | `model_output` | Directory to save fine-tuned model |
| `RESULTS_DIR` | `results` | Directory for metrics JSON files |

---

##  Results & Metrics

After training, inspect metrics:

```bash
cat results/metrics.json
# {
#   "accuracy":  0.9312,
#   "precision": 0.9287,
#   "recall":    0.9341,
#   "f1_score":  0.9314
# }

cat results/run_summary.json
# Includes hyperparameters, per-epoch history, and training time
```

The **Model Performance** tab in the Streamlit UI renders these as interactive charts.

---

##  Troubleshooting

**Model files missing when building Docker image:**
```bash
# Make sure training ran successfully first
ls model_output/   # should contain config.json, tokenizer files, model.safetensors
```

**API container unhealthy:**
```bash
docker-compose logs api   # check for model loading errors
```


**Out of memory during training:**
```bash
# Reduce batch size in .env
BATCH_SIZE=10
# Or train on a sample
python scripts/train.py --sample-size 1000
```

---

## Video Demo
https://youtu.be/hRrOjVLocP8

