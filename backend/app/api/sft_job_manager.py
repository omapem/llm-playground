"""SFT job manager for background SFT training execution.

Manages SFT training jobs running in background threads with
status tracking, configuration management, and lifecycle control.
"""

import threading
import uuid
from typing import Dict, Optional, Any, List
from datetime import datetime
import logging

from app.sft import SFTConfig, SFTTrainer, SFTCallback

logger = logging.getLogger(__name__)


class SFTJob:
    """Represents a single SFT training job.

    Tracks job status, execution thread, and training progress
    for a supervised fine-tuning task.

    Args:
        job_id: Unique job identifier
        config: SFT configuration
        trainer: SFT trainer instance

    Attributes:
        job_id: Unique identifier
        config: Training configuration
        trainer: Trainer instance
        status: Current job status (pending/running/completed/failed/stopped)
        thread: Background thread running training
        created_at: Job creation timestamp
        started_at: Training start timestamp
        completed_at: Training completion timestamp
        error: Error message if failed
        result: Training results if completed
    """

    def __init__(self, job_id: str, config: SFTConfig, trainer: SFTTrainer):
        """Initialize SFT job.

        Args:
            job_id: Unique job identifier
            config: SFT configuration
            trainer: SFT trainer instance
        """
        self.job_id = job_id
        self.config = config
        self.trainer = trainer
        self.status = "pending"
        self.thread: Optional[threading.Thread] = None
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None

    def start(self):
        """Start SFT training job in background thread.

        Launches training in a daemon thread to avoid blocking.
        Updates job status based on training outcome.
        """
        def _train():
            try:
                self.status = "running"
                self.started_at = datetime.now()
                logger.info(f"Starting SFT job {self.job_id}")

                self.result = self.trainer.train()

                self.status = "completed"
                self.completed_at = datetime.now()
                logger.info(f"SFT job {self.job_id} completed successfully")

            except Exception as e:
                self.status = "failed"
                self.error = str(e)
                self.completed_at = datetime.now()
                logger.error(f"SFT job {self.job_id} failed: {e}")

        self.thread = threading.Thread(target=_train, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop training job (mark as stopped).

        Note: Threading doesn't support clean stopping,
        so we just mark it as stopped. In production,
        you'd want a more robust solution with proper cleanup.
        """
        if self.status == "running":
            self.status = "stopped"
            logger.info(f"SFT job {self.job_id} stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get job status information.

        Returns:
            Dict with job status, timestamps, error, and results

        Example:
            >>> job.get_status()
            {
                'job_id': 'abc-123',
                'status': 'completed',
                'created_at': '2024-01-01T12:00:00',
                'started_at': '2024-01-01T12:00:01',
                'completed_at': '2024-01-01T12:30:00',
                'error': None,
                'result': {'train_loss': 0.5, ...}
            }
        """
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "result": self.result,
        }


class SFTJobManager:
    """Manager for SFT training jobs.

    Handles job creation, tracking, status queries, and
    configuration management for supervised fine-tuning tasks.

    Attributes:
        jobs: Dictionary mapping job IDs to SFTJob instances
        configs: Dictionary mapping config IDs to saved configurations

    Example:
        >>> manager = SFTJobManager()
        >>> config = SFTConfig(base_model='gpt2', ...)
        >>> job_id = manager.create_job(config)
        >>> manager.start_job(job_id)
        >>> status = manager.get_job_status(job_id)
    """

    def __init__(self):
        """Initialize job manager with empty job and config stores."""
        self.jobs: Dict[str, SFTJob] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}

    def create_job(
        self,
        config: SFTConfig,
        callbacks: Optional[List[SFTCallback]] = None
    ) -> str:
        """Create a new SFT training job.

        Args:
            config: SFT configuration
            callbacks: Optional list of training callbacks

        Returns:
            Job ID (UUID string)

        Example:
            >>> config = SFTConfig(base_model='gpt2', dataset_name='alpaca', ...)
            >>> job_id = manager.create_job(config)
        """
        job_id = str(uuid.uuid4())

        # Create trainer with config and callbacks
        trainer = SFTTrainer(config, callbacks=callbacks)

        # Create and store job
        job = SFTJob(job_id, config, trainer)
        self.jobs[job_id] = job

        logger.info(f"Created SFT job {job_id}")
        return job_id

    def start_job(self, job_id: str):
        """Start a job by ID.

        Args:
            job_id: Job identifier

        Raises:
            ValueError: If job not found

        Example:
            >>> manager.start_job('abc-123')
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        self.jobs[job_id].start()

    def stop_job(self, job_id: str):
        """Stop a job by ID.

        Args:
            job_id: Job identifier

        Raises:
            ValueError: If job not found

        Example:
            >>> manager.stop_job('abc-123')
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        self.jobs[job_id].stop()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status by ID.

        Args:
            job_id: Job identifier

        Returns:
            Dict with job status, timestamps, and results

        Raises:
            ValueError: If job not found

        Example:
            >>> status = manager.get_job_status('abc-123')
            >>> print(status['status'])  # 'completed'
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        return self.jobs[job_id].get_status()

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs with their statuses.

        Returns:
            List of job status dictionaries

        Example:
            >>> jobs = manager.list_jobs()
            >>> print(f"Total jobs: {len(jobs)}")
        """
        return [job.get_status() for job in self.jobs.values()]

    def save_config(self, name: str, config: Dict[str, Any]) -> str:
        """Save a configuration for later use.

        Args:
            name: Human-readable config name
            config: Configuration dictionary

        Returns:
            Config ID (UUID string)

        Example:
            >>> config_dict = {'base_model': 'gpt2', 'dataset_name': 'alpaca'}
            >>> config_id = manager.save_config('my_config', config_dict)
        """
        config_id = str(uuid.uuid4())
        self.configs[config_id] = {
            "name": name,
            "config": config,
        }

        logger.info(f"Saved config '{name}' with ID {config_id}")
        return config_id

    def get_config(self, config_id: str) -> Dict[str, Any]:
        """Get a saved configuration by ID.

        Args:
            config_id: Configuration identifier

        Returns:
            Dict with 'name' and 'config' keys

        Raises:
            ValueError: If config not found

        Example:
            >>> config = manager.get_config('xyz-789')
            >>> print(config['name'])  # 'my_config'
        """
        if config_id not in self.configs:
            raise ValueError(f"Config {config_id} not found")

        return self.configs[config_id]

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all saved configurations.

        Returns:
            List of config dictionaries with 'name' and 'config' keys

        Example:
            >>> configs = manager.list_configs()
            >>> for cfg in configs:
            ...     print(cfg['name'])
        """
        return list(self.configs.values())


# Global job manager instance
sft_job_manager = SFTJobManager()
