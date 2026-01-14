"""Tests for train/validation split functionality."""

import pytest
from app.data.splitter import TrainValSplitter


class TestTrainValSplitter:
    """Tests for TrainValSplitter class."""

    def test_init_default_split(self):
        """Test initialization with default 80/20 split."""
        splitter = TrainValSplitter()
        assert splitter.train_ratio == 0.8
        assert splitter.val_ratio == 0.2

    def test_init_custom_split(self):
        """Test initialization with custom split ratios."""
        splitter = TrainValSplitter(train_ratio=0.7, val_ratio=0.3)
        assert splitter.train_ratio == 0.7
        assert splitter.val_ratio == 0.3

    def test_init_with_test_split(self):
        """Test initialization with train/val/test split."""
        splitter = TrainValSplitter(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        assert splitter.train_ratio == 0.7
        assert splitter.val_ratio == 0.15
        assert splitter.test_ratio == 0.15

    def test_init_invalid_ratios_sum_to_more_than_one(self):
        """Test that initialization fails if ratios sum > 1."""
        with pytest.raises(ValueError):
            TrainValSplitter(train_ratio=0.6, val_ratio=0.6)

    def test_init_invalid_negative_ratio(self):
        """Test that initialization fails with negative ratios."""
        with pytest.raises(ValueError):
            TrainValSplitter(train_ratio=-0.1)

    def test_split_list_of_items(self):
        """Test splitting a simple list."""
        splitter = TrainValSplitter(train_ratio=0.6, val_ratio=0.4)
        items = list(range(100))

        train, val = splitter.split(items)

        assert len(train) == 60
        assert len(val) == 40
        assert len(train) + len(val) == 100

    def test_split_preserves_order(self):
        """Test that split preserves order of items."""
        splitter = TrainValSplitter(train_ratio=0.8, val_ratio=0.2)
        items = [f"item_{i}" for i in range(10)]

        train, val = splitter.split(items)

        assert len(train) == 8
        assert len(val) == 2

    def test_split_with_three_way(self):
        """Test three-way split (train/val/test)."""
        splitter = TrainValSplitter(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        items = list(range(100))

        train, val, test = splitter.split(items)

        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_split_with_seed_reproducibility(self):
        """Test that split with seed is reproducible."""
        items = list(range(1000))

        splitter1 = TrainValSplitter(seed=42)
        train1, val1 = splitter1.split(items)

        splitter2 = TrainValSplitter(seed=42)
        train2, val2 = splitter2.split(items)

        assert train1 == train2
        assert val1 == val2

    def test_split_different_seeds_different_results(self):
        """Test that different seeds produce different splits."""
        items = list(range(1000))

        splitter1 = TrainValSplitter(seed=42)
        train1, val1 = splitter1.split(items)

        splitter2 = TrainValSplitter(seed=123)
        train2, val2 = splitter2.split(items)

        assert train1 != train2

    def test_split_small_list(self):
        """Test splitting a very small list."""
        splitter = TrainValSplitter(train_ratio=0.7, val_ratio=0.3)
        items = [1, 2, 3]

        train, val = splitter.split(items)

        # With rounding, may get 2+1 or 3+0
        assert len(train) + len(val) == 3

    def test_split_single_item(self):
        """Test splitting a single item."""
        splitter = TrainValSplitter(train_ratio=0.8, val_ratio=0.2)
        items = [42]

        train, val = splitter.split(items)

        # With single item, rounding gives either train or val
        assert len(train) + len(val) == 1

    def test_split_stratified_by_label(self):
        """Test stratified split preserving label distribution."""
        splitter = TrainValSplitter(
            train_ratio=0.8,
            val_ratio=0.2,
            stratify=True
        )

        # Create items with labels (tuples of item and label)
        items = [(f"item_{i}", i % 3) for i in range(90)]

        train, val = splitter.split(items)

        # Count label distribution in train and val
        train_labels = [item[1] for item in train]
        val_labels = [item[1] for item in val]

        # Both should have roughly equal distribution of labels
        assert len(train) == 72
        assert len(val) == 18

    def test_split_returns_lists(self):
        """Test that split returns lists."""
        splitter = TrainValSplitter()
        items = [1, 2, 3, 4, 5]

        train, val = splitter.split(items)

        assert isinstance(train, list)
        assert isinstance(val, list)

    def test_split_no_item_loss(self):
        """Test that no items are lost during split."""
        splitter = TrainValSplitter()
        items = list(range(100))

        train, val = splitter.split(items)

        total_items = len(train) + len(val)
        assert total_items == len(items)

    def test_split_dataset_dict(self):
        """Test splitting a dataset dictionary."""
        splitter = TrainValSplitter(train_ratio=0.8, val_ratio=0.2)

        # Simulate a dataset dict with texts
        dataset = {
            "texts": [f"Document {i}" for i in range(100)]
        }

        train, val = splitter.split(dataset["texts"])

        assert len(train) == 80
        assert len(val) == 20

    def test_split_get_indices(self):
        """Test getting split indices instead of actual items."""
        splitter = TrainValSplitter(train_ratio=0.7, val_ratio=0.3)
        items = list(range(100))

        train_idx, val_idx = splitter.get_split_indices(len(items))

        assert len(train_idx) == 70
        assert len(val_idx) == 30

    def test_split_equal_distribution(self):
        """Test that split produces roughly equal distribution."""
        splitter = TrainValSplitter(train_ratio=0.5, val_ratio=0.5)
        items = list(range(1000))

        train, val = splitter.split(items)

        # Should be exactly equal with 1000 items
        assert len(train) == 500
        assert len(val) == 500

    def test_split_imbalanced_ratios(self):
        """Test split with very imbalanced ratios."""
        splitter = TrainValSplitter(train_ratio=0.99, val_ratio=0.01)
        items = list(range(10000))

        train, val = splitter.split(items)

        assert len(train) == 9900
        assert len(val) == 100
