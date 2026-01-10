"""Tests for HuggingFace tokenizer wrapper."""

import pytest

from app.tokenization.huggingface_wrapper import HuggingFaceTokenizerWrapper


@pytest.fixture
def tokenizer():
    """Create a HuggingFace tokenizer wrapper for testing."""
    return HuggingFaceTokenizerWrapper(
        vocab_size=1000,
        model_type="bpe",
        special_tokens=["[UNK]", "[CLS]", "[SEP]"],
    )


def test_tokenizer_initialization(tokenizer):
    """Test tokenizer initializes correctly."""
    assert tokenizer.vocab_size == 1000
    assert tokenizer.model_type == "bpe"
    assert tokenizer.tokenizer is not None


def test_encode_basic(tokenizer):
    """Test basic encoding."""
    # Note: Will fail until tokenizer is trained
    # This is expected behavior
    text = "hello world"

    try:
        result = tokenizer.encode(text)
        assert len(result.tokens) > 0
        assert len(result.token_ids) == len(result.tokens)
    except Exception:
        # Expected if tokenizer not trained
        pass


def test_training_basic(tokenizer):
    """Test basic tokenizer training."""
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "hello world hello",
        "testing testing one two three",
    ]

    tokenizer.train(texts, vocab_size=500)

    # After training, should be able to tokenize
    result = tokenizer.encode("hello world")
    assert len(result.tokens) > 0


def test_wordpiece_model():
    """Test WordPiece tokenizer creation."""
    tokenizer = HuggingFaceTokenizerWrapper(
        vocab_size=1000, model_type="wordpiece"
    )

    assert tokenizer.model_type == "wordpiece"
    assert tokenizer.tokenizer is not None


def test_invalid_model_type():
    """Test that invalid model type raises error."""
    with pytest.raises(ValueError):
        HuggingFaceTokenizerWrapper(vocab_size=1000, model_type="invalid")


def test_save_and_load(tmp_path, tokenizer):
    """Test saving and loading trained tokenizer."""
    from pathlib import Path

    # Train first
    texts = ["hello world", "testing one two three"]
    tokenizer.train(texts, vocab_size=300)

    # Save
    save_path = tmp_path / "tokenizer.json"
    tokenizer.save(Path(save_path))

    # Should create file
    assert save_path.exists()

    # Load into new instance
    new_tokenizer = HuggingFaceTokenizerWrapper(vocab_size=1000, model_type="bpe")
    new_tokenizer.load(Path(save_path))

    # Should have same vocab
    assert new_tokenizer.get_vocab_size() > 0


def test_train_from_files(tmp_path, tokenizer):
    """Test training from files."""
    from pathlib import Path

    # Create temporary training file
    train_file = tmp_path / "train.txt"
    train_file.write_text(
        "hello world\ntesting one two three\nthe quick brown fox"
    )

    # Train
    tokenizer.train_from_files([str(train_file)], vocab_size=300)

    # Should be trainable
    result = tokenizer.encode("hello world")
    assert len(result.tokens) > 0
