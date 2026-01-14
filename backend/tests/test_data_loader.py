"""Tests for data collection and loading functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.data.loaders import DatasetLoader, load_dataset


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_init_valid_dataset(self):
        """Test DatasetLoader initialization with valid dataset name."""
        loader = DatasetLoader(dataset_name="wikitext", dataset_config="wikitext-103-v1")
        assert loader.dataset_name == "wikitext"
        assert loader.dataset_config == "wikitext-103-v1"

    def test_init_with_split(self):
        """Test DatasetLoader initialization with split specification."""
        loader = DatasetLoader(
            dataset_name="openwebtext",
            dataset_config="default",
            split="train"
        )
        assert loader.split == "train"

    def test_supported_datasets(self):
        """Test that loader knows about supported datasets."""
        loader = DatasetLoader("wikitext", "wikitext-103-v1")
        supported = loader.get_supported_datasets()
        assert "wikitext" in supported
        assert "openwebtext" in supported
        assert "c4" in supported

    @patch('datasets.load_dataset')
    def test_load_wikitext_dataset(self, mock_load):
        """Test loading WikiText dataset."""
        # Mock the HF datasets.load_dataset
        mock_dataset = MagicMock()
        mock_load.return_value = mock_dataset

        loader = DatasetLoader("wikitext", "wikitext-103-v1")
        result = loader.load()

        mock_load.assert_called_once()
        assert result == mock_dataset

    @patch('datasets.load_dataset')
    def test_load_with_split(self, mock_load):
        """Test loading specific split of a dataset."""
        mock_dataset = MagicMock()
        mock_load.return_value = mock_dataset

        loader = DatasetLoader("wikitext", "wikitext-103-v1", split="train")
        result = loader.load()

        assert result == mock_dataset

    @patch('datasets.load_dataset')
    def test_load_subset_of_data(self, mock_load):
        """Test loading a subset of data."""
        mock_dataset = MagicMock()
        mock_load.return_value = mock_dataset

        loader = DatasetLoader("wikitext", "wikitext-103-v1", max_examples=1000)
        result = loader.load()

        assert result == mock_dataset
        assert loader.max_examples == 1000

    @patch('datasets.load_dataset')
    def test_load_invalid_dataset(self, mock_load):
        """Test error handling for invalid dataset."""
        mock_load.side_effect = ValueError("Dataset not found")

        loader = DatasetLoader("invalid_dataset", "config")
        with pytest.raises(ValueError):
            loader.load()

    def test_get_text_column(self):
        """Test identifying text column in dataset."""
        loader = DatasetLoader("wikitext", "wikitext-103-v1")

        # Each dataset has expected text columns
        text_col = loader.get_text_column()
        assert isinstance(text_col, str)

    @patch('datasets.load_dataset')
    def test_stream_dataset(self, mock_load):
        """Test streaming large datasets without loading full into memory."""
        mock_dataset = MagicMock()
        mock_load.return_value = mock_dataset

        loader = DatasetLoader("openwebtext", "default", streaming=True)
        result = loader.load()

        assert result == mock_dataset

    def test_dataset_info(self):
        """Test getting information about available dataset configurations."""
        loader = DatasetLoader("wikitext", "wikitext-103-v1")
        info = loader.get_dataset_info()

        assert "name" in info
        assert "description" in info


class TestLoadDatasetHelper:
    """Tests for load_dataset helper function."""

    @patch('app.data.loaders.DatasetLoader')
    def test_load_dataset_simple(self, mock_loader_class):
        """Test simple dataset loading with helper function."""
        mock_loader = MagicMock()
        mock_dataset = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_loader_class.return_value = mock_loader

        result = load_dataset("wikitext", "wikitext-103-v1")

        assert result == mock_dataset

    @patch('app.data.loaders.DatasetLoader')
    def test_load_dataset_with_kwargs(self, mock_loader_class):
        """Test load_dataset with additional keyword arguments."""
        mock_loader = MagicMock()
        mock_dataset = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_loader_class.return_value = mock_loader

        result = load_dataset(
            "openwebtext",
            "default",
            split="train",
            max_examples=5000
        )

        assert result == mock_dataset

    @patch('app.data.loaders.DatasetLoader')
    def test_load_dataset_invalid_args(self, mock_loader_class):
        """Test error handling with invalid arguments."""
        mock_loader_class.side_effect = ValueError("Invalid configuration")

        with pytest.raises(ValueError):
            load_dataset("invalid", "bad_config")


class TestDatasetLoaderIntegration:
    """Integration tests for actual dataset loading (slow, optional)."""

    @pytest.mark.slow
    def test_load_wikitext_real(self):
        """Test loading real WikiText dataset (requires network and storage)."""
        loader = DatasetLoader("wikitext", "wikitext-103-v1", max_examples=100)
        dataset = loader.load()

        assert dataset is not None
        assert len(dataset) > 0

    @pytest.mark.slow
    def test_load_openwebtext_subset(self):
        """Test loading OpenWebText subset."""
        loader = DatasetLoader(
            "openwebtext",
            "default",
            max_examples=50,
            streaming=True
        )
        dataset = loader.load()

        assert dataset is not None
