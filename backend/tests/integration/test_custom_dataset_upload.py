"""Integration tests for custom dataset upload support."""

import json
import tempfile
from pathlib import Path

import pytest

from app.data.loaders import CustomDatasetLoader


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def temp_upload_dir():
    """Create a temporary directory for dataset uploads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def loader(temp_upload_dir):
    """Create a CustomDatasetLoader with a temporary upload directory."""
    return CustomDatasetLoader(upload_dir=temp_upload_dir)


@pytest.fixture
def sample_text_data():
    """Sample text-format records."""
    return [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Machine learning models require large datasets."},
        {"text": "Transformers have revolutionized NLP tasks."},
    ]


@pytest.fixture
def sample_instruction_data():
    """Sample instruction-format records."""
    return [
        {
            "instruction": "Summarize the following text.",
            "input": "Large language models are neural networks trained on text.",
            "output": "LLMs are text-trained neural networks.",
        },
        {
            "instruction": "Translate to French.",
            "input": "Hello world",
            "output": "Bonjour le monde",
        },
    ]


# ── CustomDatasetLoader Tests ─────────────────────────────────────────


class TestCustomDatasetLoader:
    """Tests for the CustomDatasetLoader class."""

    def test_load_json_file(self, loader, sample_text_data, tmp_path):
        """Test loading a valid JSON file (array of objects)."""
        file_path = tmp_path / "data.json"
        file_path.write_text(json.dumps(sample_text_data))

        records = loader.load_from_file(str(file_path))
        assert len(records) == 3
        assert records[0]["text"] == "The quick brown fox jumps over the lazy dog."

    def test_load_jsonl_file(self, loader, sample_text_data, tmp_path):
        """Test loading a valid JSONL file (one JSON object per line)."""
        file_path = tmp_path / "data.jsonl"
        lines = [json.dumps(record) for record in sample_text_data]
        file_path.write_text("\n".join(lines))

        records = loader.load_from_file(str(file_path))
        assert len(records) == 3
        assert records[1]["text"] == "Machine learning models require large datasets."

    def test_validate_text_format(self, loader, sample_text_data):
        """Test validation of text-only format (requires 'text' field)."""
        assert loader.validate_format(sample_text_data) is True

    def test_validate_instruction_format(self, loader, sample_instruction_data):
        """Test validation of instruction format (instruction/input/output)."""
        assert loader.validate_format(sample_instruction_data) is True

    def test_validate_rejects_empty_data(self, loader):
        """Test that validation rejects empty datasets."""
        assert loader.validate_format([]) is False

    def test_validate_rejects_missing_fields(self, loader):
        """Test that validation rejects records without required fields."""
        bad_data = [{"title": "No text field", "content": "Missing text key"}]
        assert loader.validate_format(bad_data) is False

    def test_sanitize_text(self, loader):
        """Test text cleaning: strip whitespace, normalize spaces."""
        assert loader.sanitize_text("  hello   world  ") == "hello world"
        assert loader.sanitize_text("tabs\there\t") == "tabs here"
        assert loader.sanitize_text("line\nbreak") == "line break"

    def test_sanitize_preserves_normal_text(self, loader):
        """Test that sanitize doesn't alter well-formed text."""
        normal = "This is a normal sentence."
        assert loader.sanitize_text(normal) == normal

    def test_load_rejects_invalid_json(self, loader, tmp_path):
        """Test that loading invalid JSON raises ValueError."""
        file_path = tmp_path / "bad.json"
        file_path.write_text("not valid json {{{")

        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load_from_file(str(file_path))

    def test_save_and_list_datasets(self, loader, sample_text_data, temp_upload_dir):
        """Test saving a dataset and listing uploaded datasets."""
        dataset_id = loader.save_dataset(sample_text_data, filename="train.json")
        assert dataset_id is not None

        datasets = loader.list_datasets()
        assert len(datasets) == 1
        assert datasets[0]["dataset_id"] == dataset_id
        assert datasets[0]["record_count"] == 3

    def test_get_dataset_details(self, loader, sample_text_data):
        """Test getting details of a specific uploaded dataset."""
        dataset_id = loader.save_dataset(sample_text_data, filename="my_data.json")

        details = loader.get_dataset(dataset_id)
        assert details is not None
        assert details["dataset_id"] == dataset_id
        assert details["record_count"] == 3
        assert details["format"] == "text"
        assert details["filename"] == "my_data.json"

    def test_get_nonexistent_dataset(self, loader):
        """Test that getting a nonexistent dataset returns None."""
        assert loader.get_dataset("nonexistent-id") is None

    def test_get_dataset_rejects_path_traversal(self, loader):
        """Test that get_dataset blocks path traversal attempts."""
        assert loader.get_dataset("../../etc/passwd") is None
        assert loader.get_dataset("../app/main.py") is None
        assert loader.get_dataset("..") is None

    def test_sanitize_records(self, loader):
        """Test that sanitize_records cleans all text fields."""
        data = [
            {"text": "  hello   world  \n newline "},
            {"text": "normal text"},
        ]
        sanitized = loader.sanitize_records(data)
        assert sanitized[0]["text"] == "hello world newline"
        assert sanitized[1]["text"] == "normal text"

    def test_sanitize_records_instruction_format(self, loader):
        """Test that sanitize_records cleans instruction format fields."""
        data = [
            {
                "instruction": "  summarize \t this  ",
                "input": "  some  \n  text  ",
                "output": "  result  ",
            },
        ]
        sanitized = loader.sanitize_records(data)
        assert sanitized[0]["instruction"] == "summarize this"
        assert sanitized[0]["input"] == "some text"
        assert sanitized[0]["output"] == "result"

    def test_save_dataset_sanitizes_filename(self, loader, sample_text_data):
        """Test that save_dataset strips path components from filename."""
        dataset_id = loader.save_dataset(sample_text_data, filename="../../etc/evil.json")
        details = loader.get_dataset(dataset_id)
        assert details is not None
        assert details["filename"] == "evil.json"

    def test_load_rejects_empty_file(self, loader, tmp_path):
        """Test that loading an empty file raises ValueError."""
        file_path = tmp_path / "empty.json"
        file_path.write_text("")

        with pytest.raises(ValueError, match="file is empty"):
            loader.load_from_file(str(file_path))

    def test_detect_format_text(self, loader, sample_text_data):
        """Test format detection for text-format data."""
        fmt = loader.detect_format(sample_text_data)
        assert fmt == "text"

    def test_detect_format_instruction(self, loader, sample_instruction_data):
        """Test format detection for instruction-format data."""
        fmt = loader.detect_format(sample_instruction_data)
        assert fmt == "instruction"


class TestUploadEndpoint:
    """Tests for the /datasets/upload API endpoint."""

    @pytest.fixture
    def client(self):
        """Create a FastAPI test client."""
        from fastapi.testclient import TestClient
        from app.main import app

        return TestClient(app)

    def test_upload_json_file(self, client, sample_text_data, tmp_path):
        """Test uploading a valid JSON file via the API."""
        file_path = tmp_path / "upload.json"
        file_path.write_text(json.dumps(sample_text_data))

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/data/datasets/upload",
                files={"file": ("upload.json", f, "application/json")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "dataset_id" in data
        assert data["record_count"] == 3
        assert data["format"] == "text"

    def test_upload_rejects_invalid_json(self, client, tmp_path):
        """Test that the API rejects invalid JSON files."""
        file_path = tmp_path / "bad.json"
        file_path.write_text("not json!!")

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/data/datasets/upload",
                files={"file": ("bad.json", f, "application/json")},
            )

        assert response.status_code == 400

    def test_upload_rejects_invalid_format(self, client, tmp_path):
        """Test that the API rejects data without required fields."""
        bad_data = [{"no_text_field": "value"}]
        file_path = tmp_path / "bad_format.json"
        file_path.write_text(json.dumps(bad_data))

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/data/datasets/upload",
                files={"file": ("bad_format.json", f, "application/json")},
            )

        assert response.status_code == 400

    def test_get_uploaded_dataset_by_id(self, client, sample_text_data, tmp_path):
        """Test getting a specific uploaded dataset via the API."""
        file_path = tmp_path / "data.json"
        file_path.write_text(json.dumps(sample_text_data))

        with open(file_path, "rb") as f:
            upload_resp = client.post(
                "/api/data/datasets/upload",
                files={"file": ("data.json", f, "application/json")},
            )

        dataset_id = upload_resp.json()["dataset_id"]
        response = client.get(f"/api/data/datasets/uploaded/{dataset_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
        assert data["record_count"] == 3

    def test_get_nonexistent_dataset_api(self, client):
        """Test that GET for nonexistent dataset returns 404."""
        response = client.get("/api/data/datasets/uploaded/00000000-0000-4000-8000-000000000000")
        assert response.status_code == 404

    def test_upload_path_traversal_blocked(self, client):
        """Test that path traversal in dataset_id is blocked."""
        response = client.get("/api/data/datasets/uploaded/../../etc/passwd")
        assert response.status_code in (404, 422)

    def test_list_uploaded_datasets(self, client, sample_text_data, tmp_path):
        """Test listing uploaded datasets via the API."""
        # Upload a file first
        file_path = tmp_path / "data.json"
        file_path.write_text(json.dumps(sample_text_data))

        with open(file_path, "rb") as f:
            client.post(
                "/api/data/datasets/upload",
                files={"file": ("data.json", f, "application/json")},
            )

        response = client.get("/api/data/datasets/uploaded")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert len(data["datasets"]) >= 1
