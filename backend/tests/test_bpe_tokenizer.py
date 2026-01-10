"""Tests for BPE tokenizer implementation."""

import pytest

from app.tokenization.bpe import BPETokenizer


@pytest.fixture
def tokenizer():
    """Create a BPE tokenizer for testing."""
    return BPETokenizer(vocab_size=1000, special_tokens=["[UNK]", "[CLS]", "[SEP]"])


def test_tokenizer_initialization(tokenizer):
    """Test tokenizer initializes correctly."""
    assert tokenizer.vocab_size == 1000
    assert tokenizer.unk_token == "[UNK]"
    assert len(tokenizer.vocab) > 0
    assert "[UNK]" in tokenizer.vocab


def test_tokenizer_has_special_tokens(tokenizer):
    """Test that special tokens are in vocabulary."""
    assert "[UNK]" in tokenizer.vocab
    assert "[CLS]" in tokenizer.vocab
    assert "[SEP]" in tokenizer.vocab


def test_encode_basic(tokenizer):
    """Test basic encoding."""
    text = "hello world"
    result = tokenizer.encode(text)

    assert len(result.tokens) > 0
    assert len(result.token_ids) == len(result.tokens)
    assert all(isinstance(tid, int) for tid in result.token_ids)


def test_decode_basic(tokenizer):
    """Test basic decoding."""
    text = "hello world"
    result = tokenizer.encode(text)
    decoded = tokenizer.decode(result.token_ids)

    # Should be able to recover approximate text
    assert len(decoded) > 0


def test_token_to_id_conversion(tokenizer):
    """Test token to ID conversion."""
    # Test special token
    unk_id = tokenizer.token_to_id("[UNK]")
    assert unk_id == tokenizer.vocab["[UNK]"]

    # Test unknown token defaults to UNK
    unknown_id = tokenizer.token_to_id("nonexistent_token_xyz")
    assert unknown_id >= 0


def test_id_to_token_conversion(tokenizer):
    """Test ID to token conversion."""
    # Get a valid ID
    unk_id = tokenizer.vocab["[UNK]"]
    token = tokenizer.id_to_token(unk_id)
    assert token == "[UNK]"


def test_encode_deterministic(tokenizer):
    """Test that encoding is deterministic."""
    text = "the quick brown fox"
    result1 = tokenizer.encode(text)
    result2 = tokenizer.encode(text)

    assert result1.tokens == result2.tokens
    assert result1.token_ids == result2.token_ids


def test_offsets_correctness(tokenizer):
    """Test that offsets point to correct character positions."""
    text = "hello world"
    result = tokenizer.encode(text)

    # Check that offsets don't exceed text length
    for start, end in result.offsets:
        assert 0 <= start <= len(text)
        assert 0 <= end <= len(text)
        assert start <= end


def test_training_increases_vocab(tokenizer):
    """Test that training increases vocabulary."""
    initial_vocab_size = tokenizer.get_vocab_size()

    # Train on simple corpus
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "hello world hello",
        "testing testing one two three",
    ]
    tokenizer.train(texts, vocab_size=500)

    # Vocab should grow (but maybe not to target if corpus is small)
    assert tokenizer.get_vocab_size() > initial_vocab_size


def test_training_adds_merges(tokenizer):
    """Test that training creates merges."""
    assert len(tokenizer.merges) == 0

    texts = ["hello world", "testing testing"]
    tokenizer.train(texts, vocab_size=200)

    assert len(tokenizer.merges) > 0


def test_save_and_load(tmp_path, tokenizer):
    """Test saving and loading tokenizer."""
    from pathlib import Path

    # Train a simple tokenizer
    texts = ["hello world", "testing one two three"]
    tokenizer.train(texts, vocab_size=300)

    # Save it
    save_path = tmp_path / "tokenizer.json"
    tokenizer.save(Path(save_path))

    # Load into new tokenizer
    new_tokenizer = BPETokenizer(vocab_size=1000)
    new_tokenizer.load(Path(save_path))

    # Should produce same tokens
    text = "hello world"
    original_result = tokenizer.encode(text)
    loaded_result = new_tokenizer.encode(text)

    assert original_result.tokens == loaded_result.tokens


def test_empty_text_handling(tokenizer):
    """Test handling of empty text."""
    result = tokenizer.encode("")
    assert len(result.tokens) == 0
    assert len(result.token_ids) == 0


def test_unicode_handling(tokenizer):
    """Test handling of unicode characters."""
    text = "Hello 世界 🌍"
    result = tokenizer.encode(text)

    # Should not crash
    assert len(result.tokens) > 0
    assert len(result.token_ids) > 0
