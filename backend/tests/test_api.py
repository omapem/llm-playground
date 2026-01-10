"""Tests for FastAPI routes."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_tokenization_health_endpoint(client):
    """Test tokenization service health endpoint."""
    response = client.get("/api/v1/tokenization/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_encode_endpoint(client):
    """Test tokenization encoding endpoint."""
    payload = {
        "text": "hello world",
        "tokenizer_type": "bpe",
    }

    response = client.post("/api/v1/tokenization/encode", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "original_text" in data
    assert "tokens" in data
    assert "token_ids" in data
    assert "summary" in data
    assert data["original_text"] == "hello world"
    assert len(data["tokens"]) > 0


def test_encode_invalid_text(client):
    """Test encoding with empty text."""
    payload = {
        "text": "",
        "tokenizer_type": "bpe",
    }

    response = client.post("/api/v1/tokenization/encode", json=payload)
    # Should fail validation
    assert response.status_code == 422


def test_encode_invalid_tokenizer_type(client):
    """Test encoding with invalid tokenizer type."""
    payload = {
        "text": "hello",
        "tokenizer_type": "invalid_type",
    }

    response = client.post("/api/v1/tokenization/encode", json=payload)
    assert response.status_code == 500


def test_compare_endpoint(client):
    """Test tokenizer comparison endpoint."""
    payload = {
        "text": "hello world",
        "tokenizer1": "bpe",
        "tokenizer2": "bpe",
    }

    response = client.post("/api/v1/tokenization/compare", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "text" in data
    assert "tokenizer1_tokens" in data
    assert "tokenizer2_tokens" in data
    assert "difference" in data


def test_train_endpoint(client):
    """Test tokenizer training endpoint."""
    payload = {
        "texts": ["hello world", "testing one two three"],
        "vocab_size": 300,
        "tokenizer_type": "bpe",
        "min_frequency": 1,
    }

    response = client.post("/api/v1/tokenization/train", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "vocab_size" in data
    assert "texts_processed" in data


def test_train_no_texts(client):
    """Test training with no texts."""
    payload = {
        "texts": [],
        "vocab_size": 300,
        "tokenizer_type": "bpe",
    }

    response = client.post("/api/v1/tokenization/train", json=payload)
    # Should fail validation
    assert response.status_code == 422


def test_coverage_endpoint(client):
    """Test vocabulary coverage analysis endpoint."""
    payload = {
        "texts": ["hello world", "testing one two three"],
        "tokenizer_type": "bpe",
    }

    response = client.post("/api/v1/tokenization/coverage", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "total_tokens" in data
    assert "unique_tokens" in data
    assert "unknown_tokens" in data
    assert "coverage" in data
    assert "oov_rate" in data


def test_cost_estimate_endpoint(client):
    """Test cost estimation endpoint."""
    payload = {
        "text": "hello world test text",
        "tokenizer_type": "bpe",
        "cost_per_token": 0.001,
    }

    response = client.post("/api/v1/tokenization/estimate-cost", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "token_count" in data
    assert "cost_per_token" in data
    assert "estimated_cost" in data
    assert "characters" in data


def test_cost_estimate_custom_price(client):
    """Test cost estimation with custom price."""
    payload = {
        "text": "hello world",
        "tokenizer_type": "bpe",
        "cost_per_token": 0.01,
    }

    response = client.post("/api/v1/tokenization/estimate-cost", json=payload)
    assert response.status_code == 200

    data = response.json()
    # Estimated cost should be token_count * cost_per_token
    assert data["estimated_cost"] == data["token_count"] * 0.01
