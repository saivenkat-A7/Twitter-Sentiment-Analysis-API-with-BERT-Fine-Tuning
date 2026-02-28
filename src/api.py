import os
import time
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model_output")
MAX_LEN    = int(os.environ.get("MAX_LEN", 256))
LABEL_MAP  = {0: "negative", 1: "positive"}

# ── Global state ──────────────────────────────────────────────────────────────
_state: dict = {}


# ── Lifespan (load model once at startup) ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Loading model from {MODEL_PATH} …")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    _state["model"]     = model
    _state["tokenizer"] = tokenizer
    _state["device"]    = device
    _state["start_time"] = time.time()
    log.info(f"Model ready on {device} ✓")
    yield
    _state.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="Fine-tuned DistilBERT model for binary sentiment classification",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text field must not be empty")
        if len(v) > 10_000:
            raise ValueError("text exceeds maximum length of 10 000 characters")
        return v


class PredictResponse(BaseModel):
    sentiment:  str
    confidence: float
    text:       str


class HealthResponse(BaseModel):
    status:       str
    uptime_secs:  float
    model_loaded: bool


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health():
    """Liveness probe — returns 200 when the API is operational."""
    return {
        "status":       "ok",
        "uptime_secs":  round(time.time() - _state.get("start_time", time.time()), 2),
        "model_loaded": "model" in _state,
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    """Run sentiment analysis on a single text string."""
    try:
        model     = _state["model"]
        tokenizer = _state["tokenizer"]
        device    = _state["device"]
    except KeyError:
        raise HTTPException(status_code=503, detail="Model not loaded yet, please retry.")

    encoding = tokenizer(
        req.text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_id    = int(probs.argmax())
    sentiment  = LABEL_MAP[pred_id]
    confidence = round(float(probs[pred_id]), 4)

    log.info(f"Prediction: {sentiment} ({confidence:.4f}) | text[:80]='{req.text[:80]}'")
    return PredictResponse(sentiment=sentiment, confidence=confidence, text=req.text)


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)