"""Quality-based checkpoint cleanup for training.

Complements CheckpointManager's recency-based rotation by tracking
checkpoints by validation loss and retaining only the N best-performing
checkpoints. This prevents unbounded disk usage during long training runs
while ensuring the highest-quality checkpoints are preserved.
"""

import logging
import math
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointCleaner:
    """Track and clean up checkpoints by validation loss.

    Keeps the N best checkpoints (lowest validation loss) and removes
    the rest. Complements CheckpointManager's recency-based rotation
    by adding quality-based retention.

    Args:
        checkpoint_dir: Directory where checkpoints are stored
        save_total_limit: Maximum number of best checkpoints to keep

    Example:
        >>> cleaner = CheckpointCleaner('./checkpoints', save_total_limit=3)
        >>> cleaner.register('checkpoint_step_100.pt', val_loss=2.5, step=100)
        >>> cleaner.register('checkpoint_step_200.pt', val_loss=1.2, step=200)
        >>> removed = cleaner.cleanup()
    """

    def __init__(self, checkpoint_dir: str, save_total_limit: int = 3) -> None:
        """Initialize checkpoint cleaner.

        Args:
            checkpoint_dir: Directory where checkpoints are stored
            save_total_limit: Maximum number of best checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_total_limit = save_total_limit
        self._tracked: List[Dict] = []

    def register(self, checkpoint_path: str, val_loss: float, step: int) -> None:
        """Register a checkpoint with its validation loss.

        Checkpoints with NaN or Inf validation loss are silently skipped
        to prevent corrupted sorting in cleanup.

        Args:
            checkpoint_path: Full path to the checkpoint file
            val_loss: Validation loss for this checkpoint
            step: Training step when this checkpoint was saved
        """
        if math.isnan(val_loss) or math.isinf(val_loss):
            logger.warning(
                f"Skipping checkpoint step={step} with invalid val_loss={val_loss}"
            )
            return

        self._tracked.append({
            "path": checkpoint_path,
            "val_loss": val_loss,
            "step": step,
        })
        logger.debug(
            f"Registered checkpoint step={step}, val_loss={val_loss:.4f}, "
            f"path={checkpoint_path}"
        )

    def get_best_checkpoints(self) -> List[Dict]:
        """Get the N best checkpoints sorted by validation loss (ascending).

        Returns:
            List of checkpoint dicts with 'path', 'val_loss', and 'step' keys,
            sorted by val_loss ascending (best first). Returns all tracked
            checkpoints, not just those within the save_total_limit.
        """
        return sorted(self._tracked, key=lambda x: x["val_loss"])

    def cleanup(self, dry_run: bool = False) -> List[str]:
        """Remove checkpoints beyond the save_total_limit.

        Keeps the N best checkpoints (lowest validation loss) and removes
        the rest from disk.

        Args:
            dry_run: If True, report what would be removed without
                     actually deleting files

        Returns:
            List of paths that were removed (or would be removed in dry_run)
        """
        if len(self._tracked) <= self.save_total_limit:
            return []

        # Sort by val_loss ascending (best first)
        sorted_checkpoints = sorted(self._tracked, key=lambda x: x["val_loss"])

        # Keep the best N, mark the rest for removal
        to_keep = sorted_checkpoints[:self.save_total_limit]
        to_remove = sorted_checkpoints[self.save_total_limit:]

        removed_paths = []
        for checkpoint in to_remove:
            path = checkpoint["path"]
            removed_paths.append(path)

            if not dry_run:
                try:
                    os.remove(path)
                    logger.info(
                        f"Removed checkpoint: step={checkpoint['step']}, "
                        f"val_loss={checkpoint['val_loss']:.4f}, path={path}"
                    )
                except FileNotFoundError:
                    logger.warning(
                        f"Checkpoint already deleted: {path}"
                    )

        if not dry_run:
            # Update internal tracking to only keep the survivors
            self._tracked = list(to_keep)

            kept_steps = [c["step"] for c in to_keep]
            logger.info(
                f"Checkpoint cleanup complete: kept {len(to_keep)} best "
                f"(steps {kept_steps}), removed {len(removed_paths)}"
            )

        return removed_paths

    def get_best_checkpoint(self) -> Optional[str]:
        """Get the single best checkpoint path (lowest validation loss).

        Returns:
            Path to the best checkpoint, or None if no checkpoints are tracked
        """
        if not self._tracked:
            return None

        best = min(self._tracked, key=lambda x: x["val_loss"])
        return best["path"]
