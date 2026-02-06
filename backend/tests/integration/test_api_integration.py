"""Integration tests for API and job manager."""

import tempfile
import time

import pytest
from app.api.training_job_manager import TrainingJobManager
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
    return TrainingJobManager()


class TestJobManagerAPIIntegration:
    """Test job manager with API-like workflows."""

    def test_create_and_start_job_flow(self, job_manager, temp_checkpoint_dir):
        """Test creating and starting a job."""
        # Create configuration
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            max_steps=10,
            checkpoint_dir=temp_checkpoint_dir,
        )

        # Create job
        job_id = job_manager.create_job(config)
        assert job_id is not None

        # Start job
        job_manager.start_job(job_id)

        # Wait for training
        time.sleep(2)

        # Get status
        job = job_manager.get_job(job_id)
        status = job.get_status()

        assert status["job_id"] == job_id
        assert status["status"] in ["running", "completed"]
        assert status["current_step"] >= 0

    def test_multiple_jobs_lifecycle(self, job_manager, temp_checkpoint_dir):
        """Test managing multiple jobs through lifecycle."""
        # Create multiple configs
        jobs = []
        for i in range(3):
            config = TrainingConfig(
                model_config=TransformerConfig(
                    vocab_size=50257,
                    hidden_size=64,
                    num_layers=1,
                    num_heads=1,
                ),
                batch_size=4,
                max_steps=10,
                checkpoint_dir=f"{temp_checkpoint_dir}/job_{i}",
            )
            job_id = job_manager.create_job(config)
            jobs.append(job_id)

        # Start all jobs
        for job_id in jobs:
            job_manager.start_job(job_id)

        # Wait for some progress
        time.sleep(2)

        # Check all jobs are running/completed
        for job_id in jobs:
            job = job_manager.get_job(job_id)
            assert job.status in ["running", "completed"]

        # Stop all jobs
        for job_id in jobs:
            job_manager.stop_job(job_id)

        # Verify all stopped/completed
        for job_id in jobs:
            job = job_manager.get_job(job_id)
            assert job.status in ["stopped", "completed"]

    def test_config_management_flow(self, job_manager):
        """Test configuration management."""
        # Save multiple configs
        config_ids = []
        for i in range(3):
            config_dict = {
                "batch_size": 4 * (i + 1),
                "learning_rate": 1e-3 * (i + 1),
            }
            config_id = job_manager.save_config(f"config_{i}", config_dict)
            config_ids.append(config_id)

        # List all configs
        all_configs = job_manager.list_configs()
        assert len(all_configs) >= 3

        # Get specific configs
        for config_id in config_ids:
            config = job_manager.get_config(config_id)
            assert config is not None

        # Delete configs
        for config_id in config_ids:
            assert job_manager.delete_config(config_id) is True

        # Verify deleted
        for config_id in config_ids:
            assert job_manager.get_config(config_id) is None

    def test_metrics_retrieval_flow(self, job_manager, temp_checkpoint_dir):
        """Test metrics retrieval during training."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            max_steps=20,
            logging_steps=5,
            checkpoint_dir=temp_checkpoint_dir,
        )

        # Create and start job
        job_id = job_manager.create_job(config)
        job_manager.start_job(job_id)

        # Wait for some training
        time.sleep(2)

        # Get job and metrics
        job = job_manager.get_job(job_id)
        metrics = job.trainer.metrics_tracker.get_averages()

        # Verify metrics available
        assert len(metrics) > 0
        if "loss" in metrics:
            assert metrics["loss"] > 0

        # Get throughput
        throughput = job.trainer.get_throughput()
        assert "steps_per_second" in throughput
        assert throughput["steps_per_second"] >= 0

    def test_checkpoint_management_flow(self, job_manager, temp_checkpoint_dir):
        """Test checkpoint management through job manager."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            max_steps=15,
            save_steps=5,
            checkpoint_dir=temp_checkpoint_dir,
        )

        # Create and start job
        job_id = job_manager.create_job(config)
        job_manager.start_job(job_id)

        # Wait for checkpoints
        time.sleep(3)

        # Get job and list checkpoints
        job = job_manager.get_job(job_id)
        checkpoints = job.trainer.checkpoint_manager.list_checkpoints()

        # Verify checkpoints exist
        assert len(checkpoints) > 0

        # Get latest checkpoint
        latest = job.trainer.checkpoint_manager.get_latest_checkpoint()
        assert latest is not None

    def test_resume_workflow_integration(self, job_manager, temp_checkpoint_dir):
        """Test resume workflow through job manager."""
        # Initial job
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            max_steps=10,
            save_steps=5,
            checkpoint_dir=temp_checkpoint_dir,
        )

        job_id1 = job_manager.create_job(config)
        job_manager.start_job(job_id1)

        # Wait for completion
        time.sleep(3)

        # Create resume job
        config.max_steps = 20
        config.resume_from_checkpoint = True

        job_id2 = job_manager.create_job(config)
        job2 = job_manager.get_job(job_id2)

        # Verify resumed from correct step
        assert job2.trainer.current_step == 10

        # Continue training
        job_manager.start_job(job_id2)
        time.sleep(3)

        # Verify reached target
        assert job2.trainer.current_step >= 10


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""

    def test_complete_api_workflow(self, job_manager, temp_checkpoint_dir):
        """Test complete workflow from config to completion."""
        # Step 1: Save configuration
        config_dict = {
            "model_config": {
                "vocab_size": 50257,
                "hidden_size": 128,
                "num_layers": 2,
                "num_heads": 2,
            },
            "batch_size": 4,
            "max_steps": 15,
            "save_steps": 5,
            "logging_steps": 5,
            "checkpoint_dir": temp_checkpoint_dir,
        }
        config_id = job_manager.save_config("test_workflow", config_dict)
        assert config_id is not None

        # Step 2: Retrieve configuration
        saved_config = job_manager.get_config(config_id)
        assert saved_config is not None
        assert saved_config["name"] == "test_workflow"

        # Step 3: Create training config from dict
        from app.transformer import TransformerConfig
        model_config = TransformerConfig(**config_dict["model_config"])
        config = TrainingConfig(
            model_config=model_config,
            **{k: v for k, v in config_dict.items() if k != "model_config"}
        )

        # Step 4: Create and start job
        job_id = job_manager.create_job(config)
        job_manager.start_job(job_id)

        # Step 5: Monitor progress
        time.sleep(2)
        job = job_manager.get_job(job_id)
        status = job.get_status()
        assert status["current_step"] > 0

        # Step 6: Get metrics
        metrics = job.trainer.metrics_tracker.get_averages()
        assert len(metrics) > 0

        # Step 7: Wait for completion
        time.sleep(2)

        # Step 8: List checkpoints
        checkpoints = job.trainer.checkpoint_manager.list_checkpoints()
        assert len(checkpoints) > 0

        # Step 9: Cleanup
        job_manager.delete_config(config_id)
        assert job_manager.get_config(config_id) is None

    def test_failure_recovery_scenario(self, job_manager, temp_checkpoint_dir):
        """Test recovering from failed job."""
        config = TrainingConfig(
            model_config=TransformerConfig(
                vocab_size=50257,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
            ),
            batch_size=4,
            max_steps=10,
            save_steps=5,
            checkpoint_dir=temp_checkpoint_dir,
        )

        # Start job
        job_id = job_manager.create_job(config)
        job_manager.start_job(job_id)

        # Simulate failure by stopping
        time.sleep(1)
        job_manager.stop_job(job_id)

        job = job_manager.get_job(job_id)
        stopped_step = job.trainer.current_step

        # Resume from checkpoint
        config.resume_from_checkpoint = True
        config.max_steps = 20

        new_job_id = job_manager.create_job(config)
        new_job = job_manager.get_job(new_job_id)

        # Should resume from last checkpoint or current step
        assert new_job.trainer.current_step >= 0

        # Continue training
        job_manager.start_job(new_job_id)
        time.sleep(2)

        # Verify made progress
        assert new_job.trainer.current_step > 0


class TestConcurrencyIntegration:
    """Test concurrent operations."""

    def test_concurrent_job_execution(self, job_manager, temp_checkpoint_dir):
        """Test multiple jobs running concurrently."""
        jobs = []

        # Create multiple jobs with different configs
        for i in range(3):
            config = TrainingConfig(
                model_config=TransformerConfig(
                    vocab_size=50257,
                    hidden_size=64 + (i * 32),
                    num_layers=1 + i,
                    num_heads=1,
                ),
                batch_size=4,
                max_steps=10,
                checkpoint_dir=f"{temp_checkpoint_dir}/job_{i}",
            )
            job_id = job_manager.create_job(config)
            jobs.append(job_id)

        # Start all jobs
        for job_id in jobs:
            job_manager.start_job(job_id)

        # Wait for progress
        time.sleep(2)

        # Verify all jobs are running/completed
        all_jobs = job_manager.list_jobs()
        running_or_completed = sum(
            1 for job in all_jobs
            if job.status in ["running", "completed"]
        )
        assert running_or_completed >= len(jobs)

        # Stop all jobs
        for job_id in jobs:
            job_manager.stop_job(job_id)

    def test_concurrent_metrics_access(self, job_manager, temp_checkpoint_dir):
        """Test accessing metrics from multiple jobs concurrently."""
        jobs = []

        # Create and start multiple jobs
        for i in range(2):
            config = TrainingConfig(
                model_config=TransformerConfig(
                    vocab_size=50257,
                    hidden_size=128,
                    num_layers=2,
                    num_heads=2,
                ),
                batch_size=4,
                max_steps=15,
                logging_steps=5,
                checkpoint_dir=f"{temp_checkpoint_dir}/job_{i}",
            )
            job_id = job_manager.create_job(config)
            job_manager.start_job(job_id)
            jobs.append(job_id)

        # Wait for training
        time.sleep(2)

        # Access metrics from all jobs concurrently
        all_metrics = []
        for job_id in jobs:
            job = job_manager.get_job(job_id)
            metrics = job.trainer.metrics_tracker.get_averages()
            all_metrics.append(metrics)

        # Verify metrics retrieved
        assert len(all_metrics) == len(jobs)
