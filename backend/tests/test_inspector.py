"""Tests for tokenization inspector."""

import pytest

from app.tokenization.bpe import BPETokenizer
from app.tokenization.inspector import TokenizationInspector, Token


@pytest.fixture
def tokenizer():
    """Create and train a tokenizer for testing."""
    tok = BPETokenizer(vocab_size=500, special_tokens=["[UNK]", "[CLS]"])
    texts = ["hello world", "testing one two three", "the quick brown fox"]
    tok.train(texts, vocab_size=300)
    return tok


@pytest.fixture
def inspector(tokenizer):
    """Create inspector for tokenizer."""
    return TokenizationInspector(tokenizer)


def test_inspector_initialization(inspector):
    """Test inspector initializes correctly."""
    assert inspector.tokenizer is not None


def test_inspect_basic(inspector):
    """Test basic inspection."""
    text = "hello world"
    viz = inspector.inspect(text)

    assert viz.original_text == text
    assert len(viz.tokens) > 0
    assert len(viz.token_ids) > 0
    assert viz.summary is not None


def test_token_dataclass(inspector):
    """Test Token dataclass creation."""
    viz = inspector.inspect("hello")

    assert len(viz.tokens) > 0
    token = viz.tokens[0]
    assert isinstance(token, Token)
    assert token.id >= 0
    assert isinstance(token.text, str)
    assert isinstance(token.start, int)
    assert isinstance(token.end, int)


def test_summary_statistics(inspector):
    """Test summary statistics computation."""
    text = "the quick brown fox jumps over the lazy dog"
    viz = inspector.inspect(text)

    summary = viz.summary
    assert "token_count" in summary
    assert "character_count" in summary
    assert summary["character_count"] == len(text)
    assert summary["token_count"] > 0


def test_special_token_detection(inspector):
    """Test detection of special tokens."""
    text = "hello"
    viz = inspector.inspect(text)

    # At least check that detection method exists
    for token in viz.tokens:
        assert isinstance(token.special, bool)


def test_visualization_to_dict(inspector):
    """Test converting visualization to dictionary."""
    text = "hello world"
    viz = inspector.inspect(text)
    viz_dict = viz.to_dict()

    assert "original_text" in viz_dict
    assert "tokens" in viz_dict
    assert "token_ids" in viz_dict
    assert "summary" in viz_dict
    assert isinstance(viz_dict["tokens"], list)


def test_vocabulary_coverage_analysis(inspector, tokenizer):
    """Test vocabulary coverage analysis."""
    texts = [
        "hello world",
        "testing one two",
        "the quick brown fox",
    ]

    coverage = inspector.analyze_vocabulary_coverage(texts)

    assert "total_tokens" in coverage
    assert "unique_tokens" in coverage
    assert "unknown_tokens" in coverage
    assert "coverage" in coverage
    assert "oov_rate" in coverage

    # Check value ranges
    assert coverage["coverage"] >= 0
    assert coverage["coverage"] <= 1
    assert coverage["oov_rate"] >= 0
    assert coverage["oov_rate"] <= 1


def test_cost_estimation(inspector):
    """Test cost estimation."""
    text = "hello world test"
    estimate = inspector.estimate_cost(text, cost_per_token=0.001)

    assert "token_count" in estimate
    assert "cost_per_token" in estimate
    assert "estimated_cost" in estimate
    assert "characters" in estimate
    assert estimate["characters"] == len(text)
    assert estimate["estimated_cost"] > 0


def test_cost_estimation_zero_tokens(inspector):
    """Test cost estimation with empty text."""
    estimate = inspector.estimate_cost("")

    assert estimate["token_count"] == 0
    assert estimate["estimated_cost"] == 0


def test_tokenizer_comparison(inspector, tokenizer):
    """Test comparing two tokenizers."""
    from app.tokenization.bpe import BPETokenizer

    other_tokenizer = BPETokenizer(vocab_size=300)
    texts = ["hello world", "testing"]
    other_tokenizer.train(texts, vocab_size=200)

    text = "hello world"
    comparison = inspector.compare_tokenizers(text, other_tokenizer)

    assert "text" in comparison
    assert "self" in comparison
    assert "other" in comparison
    assert "difference" in comparison
    assert comparison["text"] == text
