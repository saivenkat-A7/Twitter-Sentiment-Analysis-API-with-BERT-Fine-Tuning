

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """Create test client with mocked model."""
    # Mock the model state so we don't need real model files
    mock_tokenizer = MagicMock()
    mock_model     = MagicMock()

    # Simulate tokenizer output
    mock_tokenizer.return_value = {
        "input_ids":      torch.zeros(1, 256, dtype=torch.long),
        "attention_mask": torch.ones(1, 256, dtype=torch.long),
    }

    # Simulate model logits → positive (class 1) with high confidence
    mock_logits = torch.tensor([[0.1, 3.5]])
    mock_output = MagicMock()
    mock_output.logits = mock_logits
    mock_model.return_value = mock_output

    with patch("src.api.DistilBertTokenizerFast.from_pretrained", return_value=mock_tokenizer), \
         patch("src.api.DistilBertForSequenceClassification.from_pretrained", return_value=mock_model):
        from src.api import app, _state
        _state["model"]      = mock_model
        _state["tokenizer"]  = mock_tokenizer
        _state["device"]     = torch.device("cpu")
        _state["start_time"] = 0.0
        with TestClient(app) as c:
            yield c


# ── Health Check ──────────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body_has_status_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_body_has_model_loaded(self, client):
        body = client.get("/health").json()
        assert body["model_loaded"] is True

    def test_health_has_uptime(self, client):
        body = client.get("/health").json()
        assert "uptime_secs" in body
        assert isinstance(body["uptime_secs"], float)


# ── Predict Endpoint ──────────────────────────────────────────────────────────
class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json={"text": "I love this product!"})
        assert resp.status_code == 200

    def test_predict_response_has_required_keys(self, client):
        body = client.post("/predict", json={"text": "Great experience overall."}).json()
        assert "sentiment"  in body
        assert "confidence" in body
        assert "text"       in body

    def test_predict_sentiment_is_valid_class(self, client):
        body = client.post("/predict", json={"text": "This is wonderful!"}).json()
        assert body["sentiment"] in {"positive", "negative"}

    def test_predict_confidence_in_range(self, client):
        body = client.post("/predict", json={"text": "Mediocre at best."}).json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_predict_empty_text_returns_422(self, client):
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_predict_whitespace_only_returns_422(self, client):
        resp = client.post("/predict", json={"text": "   "})
        assert resp.status_code == 422

    def test_predict_missing_text_field_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_long_text_returns_200(self, client):
        long_text = "This is a test. " * 300   # ~4800 chars
        resp = client.post("/predict", json={"text": long_text})
        assert resp.status_code == 200

    def test_predict_text_too_long_returns_422(self, client):
        too_long = "a" * 10_001
        resp = client.post("/predict", json={"text": too_long})
        assert resp.status_code == 422

    def test_predict_returns_input_text(self, client):
        text = "Unique test string 12345"
        body = client.post("/predict", json={"text": text}).json()
        assert body["text"] == text