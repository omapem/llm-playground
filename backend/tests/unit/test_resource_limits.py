"""Tests for resource limit functionality in training job manager."""

import tempfile
import time
from unittest.mock import patch, MagicMock

import pytest
import torch

from app.api.training_job_manager import TrainingJobManager, ResourceLimits
from app.training.config import TrainingConfig
from app.transformer import TransformerConfig


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config(temp_checkpoint_dir):
    """Create sample training configuration."""
    model_config = TransformerConfig(
        vocab_size=50257,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        max_position_embeddings=128,
    )
    return TrainingConfig(
        model_config=model_config,
        batch_size=4,
        learning_rate=1e-3,
        max_steps=10,
        warmup_steps=2,
        checkpoint_dir=temp_checkpoint_dir,
        save_steps=5,
        logging_steps=2,
    )


class TestResourceLimits:
    """Tests for ResourceLimits dataclass."""

    def test_resource_limits_initialization(self):
        """Test ResourceLimits initialization with default values."""
        limits = ResourceLimits()
        assert limits.max_concurrent_jobs == 2
        assert limits.max_gpu_memory_gb == 20.0
        assert limits.max_cpu_percent == 80.0

    def test_resource_limits_custom_values(self):
        """Test ResourceLimits initialization with custom values."""
        limits = ResourceLimits(
            max_concurrent_jobs=5,
            max_gpu_memory_gb=40.0,
            max_cpu_percent=90.0,
        )
        assert limits.max_concurrent_jobs == 5
        assert limits.max_gpu_memory_gb == 40.0
        assert limits.max_cpu_percent == 90.0


class TestCanStartJob:
    """Tests for can_start_job() method."""

    def test_can_start_job_within_limits(self, sample_config):
        """Test can_start_job returns True when within all limits."""
        limits = ResourceLimits(
            max_concurrent_jobs=2,
            max_gpu_memory_gb=20.0,
            max_cpu_percent=80.0,
        )
        job_manager = TrainingJobManager(resource_limits=limits)

        # Mock resource checks to return values within limits
        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("torch.cuda.is_available", return_value=True):
                # Mock mem_get_info to return (free_memory, total_memory)
                # 25GB free, 40GB total (more than the 20GB limit)
                with patch("torch.cuda.mem_get_info", return_value=(25 * 1024**3, 40 * 1024**3)):
                    assert job_manager.can_start_job() is True

    def test_cannot_start_job_exceeding_concurrent_jobs(self, sample_config):
        """Test can_start_job returns False when max concurrent jobs exceeded."""
        limits = ResourceLimits(max_concurrent_jobs=2)
        job_manager = TrainingJobManager(resource_limits=limits)

        # Create and start 2 jobs
        job_id1 = job_manager.create_job(sample_config)
        job_id2 = job_manager.create_job(sample_config)

        # Set jobs to running status (simulating started jobs)
        job_manager.jobs[job_id1].status = "running"
        job_manager.jobs[job_id2].status = "running"

        # Mock resource checks
        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("torch.cuda.is_available", return_value=False):
                # Should return False because we already have 2 running jobs
                assert job_manager.can_start_job() is False

    def test_cannot_start_job_exceeding_cpu_limit(self, sample_config):
        """Test can_start_job returns False when CPU limit exceeded."""
        limits = ResourceLimits(max_cpu_percent=80.0)
        job_manager = TrainingJobManager(resource_limits=limits)

        # Mock CPU usage above limit
        with patch("psutil.cpu_percent", return_value=85.0):
            with patch("torch.cuda.is_available", return_value=False):
                assert job_manager.can_start_job() is False

    def test_cannot_start_job_exceeding_gpu_memory_limit(self, sample_config):
        """Test can_start_job returns False when GPU memory limit exceeded."""
        limits = ResourceLimits(max_gpu_memory_gb=20.0)
        job_manager = TrainingJobManager(resource_limits=limits)

        # Mock GPU memory - only 15GB free (less than 20GB required)
        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("torch.cuda.is_available", return_value=True):
                # Mock mem_get_info to return (free_memory, total_memory)
                # 15GB free, 40GB total (less than the 20GB limit)
                with patch("torch.cuda.mem_get_info", return_value=(15 * 1024**3, 40 * 1024**3)):
                    assert job_manager.can_start_job() is False

    def test_can_start_job_counts_only_running_jobs(self, sample_config):
        """Test can_start_job only counts jobs in running/starting/cancelling status."""
        limits = ResourceLimits(max_concurrent_jobs=2)
        job_manager = TrainingJobManager(resource_limits=limits)

        # Create 4 jobs with different statuses
        job_id1 = job_manager.create_job(sample_config)
        job_id2 = job_manager.create_job(sample_config)
        job_id3 = job_manager.create_job(sample_config)
        job_id4 = job_manager.create_job(sample_config)

        # One running, one completed, one pending, one cancelling
        job_manager.jobs[job_id1].status = "running"
        job_manager.jobs[job_id2].status = "completed"
        job_manager.jobs[job_id3].status = "pending"
        job_manager.jobs[job_id4].status = "cancelling"

        # Mock resource checks
        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("torch.cuda.is_available", return_value=False):
                # Should return False because 2 jobs are consuming resources (running + cancelling)
                assert job_manager.can_start_job() is False

    def test_can_start_job_handles_no_gpu(self, sample_config):
        """Test can_start_job works correctly when GPU not available."""
        limits = ResourceLimits(max_gpu_memory_gb=20.0)
        job_manager = TrainingJobManager(resource_limits=limits)

        # Mock no GPU available
        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("torch.cuda.is_available", return_value=False):
                # Should return True, GPU check should be skipped
                assert job_manager.can_start_job() is True


class TestStartJobWithLimits:
    """Tests for start_job() with resource limit enforcement."""

    def test_start_job_queues_when_limits_exceeded(self, sample_config):
        """Test start_job queues the job when resource limits exceeded."""
        limits = ResourceLimits(max_concurrent_jobs=1)
        job_manager = TrainingJobManager(resource_limits=limits)

        # Create and start one job
        job_id1 = job_manager.create_job(sample_config)
        job_manager.jobs[job_id1].status = "running"

        # Try to start another job
        job_id2 = job_manager.create_job(sample_config)

        # Mock resource checks
        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("torch.cuda.is_available", return_value=False):
                job_manager.start_job(job_id2)

                # Job should be queued, not errored
                job = job_manager.get_job(job_id2)
                assert job.status == "queued"

    def test_start_job_succeeds_when_within_limits(self, sample_config):
        """Test start_job succeeds when within resource limits."""
        limits = ResourceLimits(max_concurrent_jobs=2)
        job_manager = TrainingJobManager(resource_limits=limits)

        # Create job
        job_id = job_manager.create_job(sample_config)

        # Mock resource checks
        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("torch.cuda.is_available", return_value=False):
                # Should not raise error
                job_manager.start_job(job_id)
                time.sleep(0.5)
                job = job_manager.get_job(job_id)
                assert job.status in ["running", "completed"]

    def test_start_job_queues_with_cpu_limit_exceeded(self, sample_config):
        """Test start_job queues when CPU limit exceeded."""
        limits = ResourceLimits(max_cpu_percent=80.0)
        job_manager = TrainingJobManager(resource_limits=limits)

        job_id = job_manager.create_job(sample_config)

        # Mock CPU usage above limit
        with patch("psutil.cpu_percent", return_value=85.0):
            with patch("torch.cuda.is_available", return_value=False):
                job_manager.start_job(job_id)

                # Job should be queued
                job = job_manager.get_job(job_id)
                assert job.status == "queued"


class TestJobManagerWithResourceLimits:
    """Tests for TrainingJobManager with resource limits."""

    def test_job_manager_accepts_resource_limits(self):
        """Test TrainingJobManager accepts resource_limits parameter."""
        limits = ResourceLimits(
            max_concurrent_jobs=5,
            max_gpu_memory_gb=40.0,
            max_cpu_percent=90.0,
        )
        job_manager = TrainingJobManager(resource_limits=limits)

        assert job_manager.resource_limits == limits
        assert job_manager.resource_limits.max_concurrent_jobs == 5

    def test_job_manager_default_resource_limits(self):
        """Test TrainingJobManager uses default ResourceLimits if not provided."""
        job_manager = TrainingJobManager()

        assert job_manager.resource_limits is not None
        assert isinstance(job_manager.resource_limits, ResourceLimits)
        assert job_manager.resource_limits.max_concurrent_jobs == 2


class TestGPUMemoryMonitoring:
    """Tests for GPU memory monitoring in Trainer."""

    def test_gpu_memory_logging_on_gpu(self, sample_config):
        """Test GPU memory is logged when CUDA is available."""
        from app.training import Trainer
        from torch.utils.data import Dataset
        import torch
        from unittest.mock import MagicMock

        class MockDataset(Dataset):
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return torch.randint(0, 50257, (128,))

        # Create trainer
        trainer = Trainer(sample_config, MockDataset())

        # Mock logger to capture GPU memory logs
        with patch("app.training.trainer.logger") as mock_logger:
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.memory_allocated", return_value=5 * 1024**3):  # 5GB
                    with patch("torch.cuda.memory_reserved", return_value=6 * 1024**3):  # 6GB
                        # This should trigger GPU memory logging
                        trainer._log_gpu_memory(step=100)

                        # Check that logger.info was called with GPU memory info
                        mock_logger.info.assert_called()
                        call_args = str(mock_logger.info.call_args)
                        assert "GPU Memory" in call_args or "5.0" in call_args

    def test_gpu_memory_logging_on_cpu(self, sample_config):
        """Test GPU memory logging is skipped when CUDA not available."""
        from app.training import Trainer
        from torch.utils.data import Dataset
        import torch

        class MockDataset(Dataset):
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return torch.randint(0, 50257, (128,))

        # Create trainer
        trainer = Trainer(sample_config, MockDataset())

        # Mock logger to verify no GPU logs
        with patch("app.training.trainer.logger") as mock_logger:
            with patch("torch.cuda.is_available", return_value=False):
                # This should not log GPU memory
                trainer._log_gpu_memory(step=100)

                # Logger should not be called
                mock_logger.info.assert_not_called()
