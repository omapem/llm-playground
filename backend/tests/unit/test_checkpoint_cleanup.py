"""Tests for CheckpointCleaner - quality-based checkpoint cleanup.

Tests validate that CheckpointCleaner correctly tracks checkpoints by
validation loss and removes the worst-performing checkpoints while
keeping the N best.
"""

import os
import pytest
from pathlib import Path

from app.training.checkpoint_cleaner import CheckpointCleaner


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Create a temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return str(checkpoint_dir)


def _create_fake_checkpoint(checkpoint_dir: str, step: int) -> str:
    """Create a fake checkpoint file for testing.

    Args:
        checkpoint_dir: Directory to create the file in
        step: Step number for the checkpoint filename

    Returns:
        Path to the created checkpoint file
    """
    path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    with open(path, "w") as f:
        f.write(f"fake checkpoint at step {step}")
    return path


class TestRegisterCheckpoint:
    """Tests for registering checkpoints with the cleaner."""

    def test_register_checkpoint(self, tmp_checkpoint_dir):
        """Registering a checkpoint adds it to the tracked list."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=3)
        path = _create_fake_checkpoint(tmp_checkpoint_dir, step=100)

        cleaner.register(checkpoint_path=path, val_loss=2.5, step=100)

        tracked = cleaner.get_best_checkpoints()
        assert len(tracked) == 1
        assert tracked[0]["path"] == path
        assert tracked[0]["val_loss"] == 2.5
        assert tracked[0]["step"] == 100

    def test_register_multiple_checkpoints(self, tmp_checkpoint_dir):
        """Registering multiple checkpoints tracks all of them."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=5)

        for step, val_loss in [(100, 2.5), (200, 1.8), (300, 3.1)]:
            path = _create_fake_checkpoint(tmp_checkpoint_dir, step)
            cleaner.register(checkpoint_path=path, val_loss=val_loss, step=step)

        tracked = cleaner.get_best_checkpoints()
        assert len(tracked) == 3


class TestGetBestCheckpoints:
    """Tests for retrieving best checkpoints sorted by validation loss."""

    def test_get_best_checkpoints_sorted(self, tmp_checkpoint_dir):
        """Returns checkpoints sorted by validation loss ascending (best first)."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=5)

        entries = [(100, 2.5), (200, 1.2), (300, 3.1), (400, 0.8)]
        for step, val_loss in entries:
            path = _create_fake_checkpoint(tmp_checkpoint_dir, step)
            cleaner.register(checkpoint_path=path, val_loss=val_loss, step=step)

        best = cleaner.get_best_checkpoints()

        # Should be sorted ascending by val_loss
        assert len(best) == 4
        assert best[0]["val_loss"] == 0.8
        assert best[0]["step"] == 400
        assert best[1]["val_loss"] == 1.2
        assert best[1]["step"] == 200
        assert best[2]["val_loss"] == 2.5
        assert best[2]["step"] == 100
        assert best[3]["val_loss"] == 3.1
        assert best[3]["step"] == 300


class TestCleanup:
    """Tests for checkpoint cleanup (removing worst checkpoints)."""

    def test_cleanup_removes_worst(self, tmp_checkpoint_dir):
        """Cleanup keeps N best checkpoints and removes the rest."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=2)

        # Register 4 checkpoints with different val_loss values
        entries = [(100, 2.5), (200, 1.2), (300, 3.1), (400, 0.8)]
        paths = {}
        for step, val_loss in entries:
            path = _create_fake_checkpoint(tmp_checkpoint_dir, step)
            paths[step] = path
            cleaner.register(checkpoint_path=path, val_loss=val_loss, step=step)

        removed = cleaner.cleanup()

        # Should remove the 2 worst (highest val_loss): step 100 (2.5) and step 300 (3.1)
        assert len(removed) == 2
        assert paths[300] in removed  # val_loss=3.1 (worst)
        assert paths[100] in removed  # val_loss=2.5 (second worst)

        # The removed files should no longer exist on disk
        assert not os.path.exists(paths[300])
        assert not os.path.exists(paths[100])

        # The kept files should still exist
        assert os.path.exists(paths[200])  # val_loss=1.2 (second best)
        assert os.path.exists(paths[400])  # val_loss=0.8 (best)

    def test_cleanup_dry_run(self, tmp_checkpoint_dir):
        """dry_run=True reports what would be removed without deleting files."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=1)

        entries = [(100, 2.5), (200, 1.2)]
        paths = {}
        for step, val_loss in entries:
            path = _create_fake_checkpoint(tmp_checkpoint_dir, step)
            paths[step] = path
            cleaner.register(checkpoint_path=path, val_loss=val_loss, step=step)

        removed = cleaner.cleanup(dry_run=True)

        # Should report the worst checkpoint for removal
        assert len(removed) == 1
        assert paths[100] in removed  # val_loss=2.5 is worse than 1.2

        # But all files should still exist on disk
        assert os.path.exists(paths[100])
        assert os.path.exists(paths[200])

    def test_cleanup_no_action_within_limit(self, tmp_checkpoint_dir):
        """No deletion when number of checkpoints is within the limit."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=5)

        entries = [(100, 2.5), (200, 1.2), (300, 0.9)]
        for step, val_loss in entries:
            path = _create_fake_checkpoint(tmp_checkpoint_dir, step)
            cleaner.register(checkpoint_path=path, val_loss=val_loss, step=step)

        removed = cleaner.cleanup()

        # No checkpoints should be removed (3 < 5)
        assert removed == []

        # All files should still exist
        for step, _ in entries:
            path = os.path.join(tmp_checkpoint_dir, f"checkpoint_step_{step}.pt")
            assert os.path.exists(path)

    def test_cleanup_exactly_at_limit(self, tmp_checkpoint_dir):
        """No deletion when number of checkpoints equals the limit."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=3)

        entries = [(100, 2.5), (200, 1.2), (300, 0.9)]
        for step, val_loss in entries:
            path = _create_fake_checkpoint(tmp_checkpoint_dir, step)
            cleaner.register(checkpoint_path=path, val_loss=val_loss, step=step)

        removed = cleaner.cleanup()

        assert removed == []

    def test_cleanup_handles_already_deleted_files(self, tmp_checkpoint_dir):
        """Cleanup gracefully handles checkpoints that were already deleted."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=1)

        path1 = _create_fake_checkpoint(tmp_checkpoint_dir, step=100)
        path2 = _create_fake_checkpoint(tmp_checkpoint_dir, step=200)
        cleaner.register(checkpoint_path=path1, val_loss=2.5, step=100)
        cleaner.register(checkpoint_path=path2, val_loss=1.0, step=200)

        # Manually delete the file before cleanup
        os.remove(path1)

        # Cleanup should not raise an error
        removed = cleaner.cleanup()
        assert path1 in removed


class TestGetBestCheckpoint:
    """Tests for getting the single best checkpoint."""

    def test_get_best_checkpoint(self, tmp_checkpoint_dir):
        """Returns the path to the checkpoint with the lowest validation loss."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=5)

        entries = [(100, 2.5), (200, 0.8), (300, 1.5)]
        for step, val_loss in entries:
            path = _create_fake_checkpoint(tmp_checkpoint_dir, step)
            cleaner.register(checkpoint_path=path, val_loss=val_loss, step=step)

        best_path = cleaner.get_best_checkpoint()

        expected_path = os.path.join(tmp_checkpoint_dir, "checkpoint_step_200.pt")
        assert best_path == expected_path

    def test_get_best_checkpoint_single(self, tmp_checkpoint_dir):
        """Returns the only checkpoint when there is just one."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=3)

        path = _create_fake_checkpoint(tmp_checkpoint_dir, step=100)
        cleaner.register(checkpoint_path=path, val_loss=1.5, step=100)

        assert cleaner.get_best_checkpoint() == path


class TestNaNHandling:
    """Tests for NaN/Inf validation loss handling."""

    def test_register_skips_nan_val_loss(self, tmp_checkpoint_dir):
        """NaN validation loss is silently skipped."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=3)
        path = _create_fake_checkpoint(tmp_checkpoint_dir, step=100)

        cleaner.register(checkpoint_path=path, val_loss=float("nan"), step=100)

        assert len(cleaner.get_best_checkpoints()) == 0

    def test_register_skips_inf_val_loss(self, tmp_checkpoint_dir):
        """Inf validation loss is silently skipped."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=3)
        path = _create_fake_checkpoint(tmp_checkpoint_dir, step=100)

        cleaner.register(checkpoint_path=path, val_loss=float("inf"), step=100)

        assert len(cleaner.get_best_checkpoints()) == 0

    def test_best_checkpoint_ignores_nan_entries(self, tmp_checkpoint_dir):
        """get_best_checkpoint returns valid entry even if NaN was attempted."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=3)

        path1 = _create_fake_checkpoint(tmp_checkpoint_dir, step=100)
        path2 = _create_fake_checkpoint(tmp_checkpoint_dir, step=200)

        cleaner.register(checkpoint_path=path1, val_loss=float("nan"), step=100)
        cleaner.register(checkpoint_path=path2, val_loss=1.5, step=200)

        assert cleaner.get_best_checkpoint() == path2


class TestEmptyCleaner:
    """Tests for handling empty state."""

    def test_empty_cleaner_get_best_checkpoints(self, tmp_checkpoint_dir):
        """get_best_checkpoints returns empty list when no checkpoints registered."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=3)

        assert cleaner.get_best_checkpoints() == []

    def test_empty_cleaner_get_best_checkpoint(self, tmp_checkpoint_dir):
        """get_best_checkpoint returns None when no checkpoints registered."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=3)

        assert cleaner.get_best_checkpoint() is None

    def test_empty_cleaner_cleanup(self, tmp_checkpoint_dir):
        """cleanup returns empty list when no checkpoints registered."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=3)

        removed = cleaner.cleanup()
        assert removed == []

    def test_empty_cleaner_cleanup_dry_run(self, tmp_checkpoint_dir):
        """cleanup with dry_run returns empty list when no checkpoints registered."""
        cleaner = CheckpointCleaner(tmp_checkpoint_dir, save_total_limit=3)

        removed = cleaner.cleanup(dry_run=True)
        assert removed == []
