"""Training job manager for background training execution."""

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Any
from datetime import datetime

import psutil
import torch

from app.training import Trainer, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits for training job management.

    Args:
        max_concurrent_jobs: Maximum number of concurrent training jobs
        max_gpu_memory_gb: Maximum GPU memory usage per job (in GB)
        max_cpu_percent: Maximum total CPU usage percentage
    """
    max_concurrent_jobs: int = 2
    max_gpu_memory_gb: float = 20.0
    max_cpu_percent: float = 80.0


class TrainingJob:
    """Represents a single training job."""

    def __init__(self, job_id: str, config: TrainingConfig, trainer: Trainer):
        """Initialize training job."""
        self.job_id = job_id
        self.config = config
        self.trainer = trainer
        self._status = "pending"
        self._status_lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None

    @property
    def status(self) -> str:
        """Get current job status (thread-safe)."""
        with self._status_lock:
            return self._status

    @status.setter
    def status(self, value: str):
        """Set job status (thread-safe)."""
        with self._status_lock:
            self._status = value

    def start(self, cancellation_event: Optional[threading.Event] = None, job_manager=None):
        """Start training job in background thread.

        Args:
            cancellation_event: Optional event to signal cancellation
            job_manager: Optional reference to job manager for cleanup
        """
        def _train():
            try:
                self.status = "running"
                self.started_at = datetime.now()
                self.trainer.train(cancellation_event=cancellation_event)

                # Only mark as completed if not cancelled
                if cancellation_event and cancellation_event.is_set():
                    self.status = "cancelled"
                else:
                    self.status = "completed"
                self.completed_at = datetime.now()
            except Exception as e:
                self.status = "failed"
                self.error = str(e)
                self.completed_at = datetime.now()
            finally:
                # Clean up the cancellation event after job completes
                if job_manager and self.job_id in job_manager.cancellation_flags:
                    del job_manager.cancellation_flags[self.job_id]

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

    def __init__(self, resource_limits: Optional[ResourceLimits] = None):
        """Initialize job manager.

        Args:
            resource_limits: Optional resource limits for job management.
                           Defaults to ResourceLimits() if not provided.
        """
        self.jobs: Dict[str, TrainingJob] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.cancellation_flags: Dict[str, threading.Event] = {}
        self.resource_limits = resource_limits if resource_limits is not None else ResourceLimits()

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

    def can_start_job(self) -> bool:
        """Check if a new job can be started within resource limits.

        Returns:
            True if resources are available, False otherwise
        """
        # Check concurrent jobs limit
        running_jobs = sum(
            1 for job in self.jobs.values()
            if job.status in ["starting", "running"]
        )
        if running_jobs >= self.resource_limits.max_concurrent_jobs:
            logger.warning(
                f"Cannot start job: {running_jobs} jobs running "
                f"(max: {self.resource_limits.max_concurrent_jobs})"
            )
            return False

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > self.resource_limits.max_cpu_percent:
            logger.warning(
                f"Cannot start job: CPU usage {cpu_percent:.1f}% "
                f"exceeds limit {self.resource_limits.max_cpu_percent}%"
            )
            return False

        # Check GPU memory (if GPU available)
        if torch.cuda.is_available():
            gpu_memory_bytes = torch.cuda.memory_allocated()
            gpu_memory_gb = gpu_memory_bytes / (1024**3)
            if gpu_memory_gb > self.resource_limits.max_gpu_memory_gb:
                logger.warning(
                    f"Cannot start job: GPU memory usage {gpu_memory_gb:.1f}GB "
                    f"exceeds limit {self.resource_limits.max_gpu_memory_gb}GB"
                )
                return False

        return True

    def start_job(self, job_id: str) -> None:
        """Start a training job.

        Args:
            job_id: Job ID to start

        Raises:
            ValueError: If job not found
            RuntimeError: If resource limits would be exceeded
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        # Check resource limits before starting
        if not self.can_start_job():
            raise RuntimeError(
                "Resource limits exceeded. Cannot start job. "
                "Try stopping other jobs or increasing resource limits."
            )

        # Create cancellation event
        cancellation_event = threading.Event()
        self.cancellation_flags[job_id] = cancellation_event

        job = self.jobs[job_id]
        job.start(cancellation_event=cancellation_event, job_manager=self)

    def stop_job(self, job_id: str) -> None:
        """Stop a training job.

        Args:
            job_id: Job ID to stop
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        job.stop()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if job was cancelled, False if already completed/failed

        Raises:
            ValueError: If job does not exist
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]

        # Can only cancel running or pending jobs
        if job.status in ["completed", "failed", "cancelled"]:
            return False

        # Set status to cancelling
        job.status = "cancelling"

        # Set the cancellation event
        if job_id in self.cancellation_flags:
            self.cancellation_flags[job_id].set()

        return True

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
