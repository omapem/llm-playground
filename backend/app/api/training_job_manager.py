"""Training job manager for background training execution."""

import threading
import uuid
from typing import Dict, Optional, Any
from datetime import datetime

from app.training import Trainer, TrainingConfig


class TrainingJob:
    """Represents a single training job."""

    def __init__(self, job_id: str, config: TrainingConfig, trainer: Trainer):
        """Initialize training job."""
        self.job_id = job_id
        self.config = config
        self.trainer = trainer
        self.status = "pending"
        self.thread: Optional[threading.Thread] = None
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None

    def start(self):
        """Start training job in background thread."""
        def _train():
            try:
                self.status = "running"
                self.started_at = datetime.now()
                self.trainer.train()
                self.status = "completed"
                self.completed_at = datetime.now()
            except Exception as e:
                self.status = "failed"
                self.error = str(e)
                self.completed_at = datetime.now()

        self.thread = threading.Thread(target=_train, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop training job."""
        if self.status == "running":
            self.status = "stopped"
            # Note: Threading doesn't support clean stopping,
            # so we just mark it as stopped
            # In production, you'd want a more robust solution

    def get_status(self) -> Dict[str, Any]:
        """Get job status information."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "current_step": self.trainer.current_step,
            "max_steps": self.config.max_steps,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


class TrainingJobManager:
    """Manager for training jobs."""

    def __init__(self):
        """Initialize job manager."""
        self.jobs: Dict[str, TrainingJob] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, config: TrainingConfig, dataset=None) -> str:
        """Create a new training job.

        Args:
            config: Training configuration
            dataset: Optional training dataset (uses mock if None)

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())

        # Create mock dataset if none provided
        if dataset is None:
            from torch.utils.data import Dataset
            import torch

            class MockDataset(Dataset):
                def __init__(self, size=100, seq_len=128):
                    self.size = size
                    self.seq_len = seq_len

                def __len__(self):
                    return self.size

                def __getitem__(self, idx):
                    return torch.randint(0, 50257, (self.seq_len,))

            dataset = MockDataset()

        # Create trainer
        trainer = Trainer(config=config, train_dataset=dataset)

        # Create job
        job = TrainingJob(job_id, config, trainer)
        self.jobs[job_id] = job

        return job_id

    def start_job(self, job_id: str) -> None:
        """Start a training job.

        Args:
            job_id: Job ID to start
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        job.start()

    def stop_job(self, job_id: str) -> None:
        """Stop a training job.

        Args:
            job_id: Job ID to stop
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        job.stop()

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID.

        Args:
            job_id: Job ID

        Returns:
            Training job or None if not found
        """
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[TrainingJob]:
        """List all training jobs.

        Returns:
            List of all training jobs
        """
        return list(self.jobs.values())

    def save_config(self, name: str, config_dict: Dict[str, Any]) -> str:
        """Save a training configuration.

        Args:
            name: Config name
            config_dict: Configuration dictionary

        Returns:
            Config ID
        """
        config_id = str(uuid.uuid4())
        self.configs[config_id] = {
            "name": name,
            "config": config_dict,
            "created_at": datetime.now().isoformat(),
        }
        return config_id

    def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get a configuration by ID.

        Args:
            config_id: Config ID

        Returns:
            Configuration dictionary or None if not found
        """
        return self.configs.get(config_id)

    def list_configs(self) -> list[Dict[str, Any]]:
        """List all saved configurations.

        Returns:
            List of all configurations
        """
        return [
            {"config_id": cid, **config}
            for cid, config in self.configs.items()
        ]

    def delete_config(self, config_id: str) -> bool:
        """Delete a configuration.

        Args:
            config_id: Config ID to delete

        Returns:
            True if deleted, False if not found
        """
        if config_id in self.configs:
            del self.configs[config_id]
            return True
        return False


# Global job manager instance
job_manager = TrainingJobManager()
