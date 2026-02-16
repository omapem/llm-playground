"""Tests for SFT dataset processing."""

import json
import pytest
from datasets import Dataset
from app.sft.dataset import SFTDatasetProcessor
from app.sft.templates import AlpacaTemplate, ChatTemplate


@pytest.fixture
def sample_alpaca_data():
    """Sample Alpaca-format data."""
    return [
        {"instruction": "What is ML?", "input": "", "output": "ML is machine learning"},
        {"instruction": "Explain AI", "input": "", "output": "AI is artificial intelligence"},
        {"instruction": "Translate", "input": "Hello", "output": "Bonjour"},
    ]


@pytest.fixture
def sample_chat_data():
    """Sample chat-format data."""
    return [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence"},
            ]
        },
    ]


def test_format_dataset_alpaca(sample_alpaca_data):
    """Test formatting Alpaca dataset with template."""
    processor = SFTDatasetProcessor()
    template = AlpacaTemplate()

    # Create dataset from list
    dataset = Dataset.from_list(sample_alpaca_data)

    formatted = processor.format_dataset(dataset, template)

    assert len(formatted) == 3
    assert "text" in formatted.column_names
    assert "### Instruction:" in formatted[0]["text"]
    assert "What is ML?" in formatted[0]["text"]
    assert "### Response:" in formatted[0]["text"]


def test_format_dataset_chat(sample_chat_data):
    """Test formatting chat dataset with template."""
    processor = SFTDatasetProcessor()
    template = ChatTemplate("chatml")

    dataset = Dataset.from_list(sample_chat_data)

    formatted = processor.format_dataset(dataset, template)

    assert len(formatted) == 2
    assert "text" in formatted.column_names
    assert "<|im_start|>user" in formatted[0]["text"]
    assert "Hello" in formatted[0]["text"]


def test_load_dataset_from_memory(sample_alpaca_data):
    """Test loading dataset from in-memory data."""
    processor = SFTDatasetProcessor()
    dataset = Dataset.from_list(sample_alpaca_data)

    # Processor should work with already-loaded datasets
    template = AlpacaTemplate()
    formatted = processor.format_dataset(dataset, template)

    assert len(formatted) == 3


def test_format_dataset_removes_original_columns(sample_alpaca_data):
    """Test formatting removes original columns."""
    processor = SFTDatasetProcessor()
    template = AlpacaTemplate()
    dataset = Dataset.from_list(sample_alpaca_data)

    formatted = processor.format_dataset(dataset, template)

    # Should only have 'text' column
    assert formatted.column_names == ["text"]
    assert "instruction" not in formatted.column_names
    assert "output" not in formatted.column_names


def test_format_dataset_custom_text_column(sample_alpaca_data):
    """Test formatting with custom text column name."""
    processor = SFTDatasetProcessor()
    template = AlpacaTemplate()
    dataset = Dataset.from_list(sample_alpaca_data)

    formatted = processor.format_dataset(dataset, template, text_column="prompt")

    assert "prompt" in formatted.column_names
    assert "text" not in formatted.column_names


def test_load_dataset_with_validation_split(sample_alpaca_data, tmp_path):
    """Test loading dataset with validation split."""
    processor = SFTDatasetProcessor()

    # Save sample data to temporary file
    json_file = tmp_path / "test_split_dataset.json"
    with open(json_file, "w") as f:
        json.dump(sample_alpaca_data, f)

    # Load with validation split
    train, val = processor.load_dataset(str(json_file), validation_split=0.3)

    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)
    assert len(train) + len(val) == 3  # Total should match original


def test_load_dataset_invalid_validation_split(tmp_path):
    """Test load_dataset raises error for invalid validation split."""
    processor = SFTDatasetProcessor()

    # Create a temporary dataset file
    json_file = tmp_path / "test_dataset.json"
    with open(json_file, "w") as f:
        json.dump([{"text": "test"}], f)

    with pytest.raises(ValueError, match="validation_split must be"):
        processor.load_dataset(str(json_file), validation_split=1.5)


def test_load_and_format_combines_operations(sample_alpaca_data, tmp_path):
    """Test load_and_format combines loading and formatting."""
    processor = SFTDatasetProcessor()
    template = AlpacaTemplate()

    # Save sample data to temporary file
    json_file = tmp_path / "test_dataset.json"
    with open(json_file, "w") as f:
        json.dump(sample_alpaca_data, f)

    # Load and format in one step
    formatted = processor.load_and_format(str(json_file), template)

    assert len(formatted) == 3
    assert "text" in formatted.column_names
    assert "### Instruction:" in formatted[0]["text"]


def test_load_dataset_unsupported_format(tmp_path):
    """Test load_dataset raises error for unsupported file format."""
    processor = SFTDatasetProcessor()

    # Create a file with unsupported extension
    csv_file = tmp_path / "test_dataset.csv"
    csv_file.write_text("col1,col2\nval1,val2")

    with pytest.raises(ValueError, match="Unsupported file format"):
        processor.load_dataset(str(csv_file))
