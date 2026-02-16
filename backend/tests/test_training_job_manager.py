"""Tests for training job manager."""

import tempfile
import time

import pytest
from app.api.training_job_manager import TrainingJobManager, TrainingJob, ResourceLimits
from app.training.config import TrainingConfig
from app.transformer import TransformerConfig


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def job_manager():
    """Create fresh job manager for each test."""
    # Use higher limits to avoid resource constraints in tests
    # Set CPU to 100% to effectively disable CPU checks (system load varies)
    limits = ResourceLimits(max_concurrent_jobs=5, max_cpu_percent=100.0)
    return TrainingJobManager(resource_limits=limits)


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


class TestTrainingJobManager:
    """Tests for TrainingJobManager."""

    def test_create_job(self, job_manager, sample_config):
        """Test creating a training job."""
        job_id = job_manager.create_job(sample_config)
        assert job_id is not None
        assert job_id in job_manager.jobs

    def test_get_job(self, job_manager, sample_config):
        """Test retrieving a job."""
        job_id = job_manager.create_job(sample_config)
        job = job_manager.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id

    def test_get_nonexistent_job(self, job_manager):
        """Test retrieving nonexistent job."""
        job = job_manager.get_job("nonexistent")
        assert job is None

    def test_list_jobs(self, job_manager, sample_config):
        """Test listing all jobs."""
        job_id1 = job_manager.create_job(sample_config)
        job_id2 = job_manager.create_job(sample_config)

        jobs = job_manager.list_jobs()
        assert len(jobs) == 2
        job_ids = [job.job_id for job in jobs]
        assert job_id1 in job_ids
        assert job_id2 in job_ids

    def test_start_job(self, job_manager, sample_config):
        """Test starting a job."""
        job_id = job_manager.create_job(sample_config)
        job_manager.start_job(job_id)

        job = job_manager.get_job(job_id)
        assert job.status in ["running", "completed"]

    def test_start_nonexistent_job(self, job_manager):
        """Test starting nonexistent job."""
        with pytest.raises(ValueError):
            job_manager.start_job("nonexistent")

    def test_stop_job(self, job_manager, sample_config):
        """Test stopping a job."""
        job_id = job_manager.create_job(sample_config)
        job_manager.start_job(job_id)
        time.sleep(0.5)
        job_manager.stop_job(job_id)

        job = job_manager.get_job(job_id)
        assert job.status in ["stopped", "completed"]

    def test_stop_nonexistent_job(self, job_manager):
        """Test stopping nonexistent job."""
        with pytest.raises(ValueError):
            job_manager.stop_job("nonexistent")

    def test_save_config(self, job_manager):
        """Test saving a configuration."""
        config_dict = {"batch_size": 8, "learning_rate": 1e-3}
        config_id = job_manager.save_config("test_config", config_dict)

        assert config_id is not None
        assert config_id in job_manager.configs

    def test_get_config(self, job_manager):
        """Test retrieving a configuration."""
        config_dict = {"batch_size": 8, "learning_rate": 1e-3}
        config_id = job_manager.save_config("test_config", config_dict)

        config = job_manager.get_config(config_id)
        assert config is not None
        assert config["name"] == "test_config"
        assert config["config"] == config_dict

    def test_get_nonexistent_config(self, job_manager):
        """Test retrieving nonexistent configuration."""
        config = job_manager.get_config("nonexistent")
        assert config is None

    def test_list_configs(self, job_manager):
        """Test listing all configurations."""
        config_id1 = job_manager.save_config("config1", {"batch_size": 8})
        config_id2 = job_manager.save_config("config2", {"batch_size": 16})

        configs = job_manager.list_configs()
        assert len(configs) == 2
        config_ids = [c["config_id"] for c in configs]
        assert config_id1 in config_ids
        assert config_id2 in config_ids

    def test_delete_config(self, job_manager):
        """Test deleting a configuration."""
        config_id = job_manager.save_config("test_config", {"batch_size": 8})

        assert job_manager.delete_config(config_id) is True
        assert job_manager.get_config(config_id) is None

    def test_delete_nonexistent_config(self, job_manager):
        """Test deleting nonexistent configuration."""
        assert job_manager.delete_config("nonexistent") is False


class TestTrainingJob:
    """Tests for TrainingJob class."""

    def test_job_initialization(self, sample_config):
        """Test job initialization."""
        from app.training import Trainer
        from torch.utils.data import Dataset
        import torch

        class MockDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return torch.randint(0, 50257, (128,))

        trainer = Trainer(sample_config, MockDataset())
        job = TrainingJob("test_id", sample_config, trainer)

        assert job.job_id == "test_id"
        assert job.status == "pending"
        assert job.config == sample_config
        assert job.trainer == trainer

    def test_job_get_status(self, sample_config):
        """Test getting job status."""
        from app.training import Trainer
        from torch.utils.data import Dataset
        import torch

        class MockDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return torch.randint(0, 50257, (128,))

        trainer = Trainer(sample_config, MockDataset())
        job = TrainingJob("test_id", sample_config, trainer)

        status = job.get_status()
        assert status["job_id"] == "test_id"
        assert status["status"] == "pending"
        assert status["current_step"] == 0
        assert status["max_steps"] == sample_config.max_steps

    def test_job_start(self, sample_config):
        """Test starting a job."""
        from app.training import Trainer
        from torch.utils.data import Dataset
        import torch

        class MockDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return torch.randint(0, 50257, (128,))

        trainer = Trainer(sample_config, MockDataset())
        job = TrainingJob("test_id", sample_config, trainer)

        job.start()
        time.sleep(0.5)  # Give thread time to start

        assert job.status in ["running", "completed"]
        assert job.thread is not None

    def test_job_stop(self, sample_config):
        """Test stopping a job."""
        from app.training import Trainer
        from torch.utils.data import Dataset
        import torch

        class MockDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return torch.randint(0, 50257, (128,))

        trainer = Trainer(sample_config, MockDataset())
        job = TrainingJob("test_id", sample_config, trainer)

        job.start()
        time.sleep(0.5)
        job.stop()

        assert job.status in ["stopped", "completed"]


class TestJobManagerIntegration:
    """Integration tests for job manager."""

    def test_complete_workflow(self, job_manager, sample_config):
        """Test complete job workflow."""
        # Create job
        job_id = job_manager.create_job(sample_config)
        assert job_id is not None

        # Start job
        job_manager.start_job(job_id)
        job = job_manager.get_job(job_id)
        assert job.status in ["running", "completed"]

        # Wait for some training
        time.sleep(2)

        # Get status
        status = job.get_status()
        assert status["current_step"] >= 0

        # Stop job
        job_manager.stop_job(job_id)
        assert job.status in ["stopped", "completed"]

    def test_multiple_concurrent_jobs(self, job_manager, sample_config):
        """Test running multiple jobs concurrently."""
        # Create multiple jobs
        job_ids = []
        for _ in range(3):
            job_id = job_manager.create_job(sample_config)
            job_ids.append(job_id)

        # Start all jobs
        for job_id in job_ids:
            job_manager.start_job(job_id)

        # Verify all running
        time.sleep(1)
        for job_id in job_ids:
            job = job_manager.get_job(job_id)
            assert job.status in ["running", "completed"]

        # Stop all jobs
        for job_id in job_ids:
            job_manager.stop_job(job_id)
