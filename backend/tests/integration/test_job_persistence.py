"""Integration tests for job persistence with SQLite."""

import json
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from app.training.config import TrainingConfig
from app.transformer import TransformerConfig
from app.api.persistence import JobDatabase


class SimpleDataset(Dataset):
    """Simple dataset for integration testing."""

    def __init__(self, size: int = 100, seq_len: int = 128, vocab_size: int = 50257):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randint(0, self.vocab_size, (self.seq_len,))


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_jobs.db")


@pytest.fixture
def db(temp_db_path):
    """Create a JobDatabase instance with a temporary database."""
    return JobDatabase(db_path=temp_db_path)


@pytest.fixture
def sample_config():
    """Create a sample TrainingConfig for testing."""
    return TrainingConfig(
        model_config=TransformerConfig(
            vocab_size=50257,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
            max_position_embeddings=128,
        ),
        batch_size=4,
        learning_rate=1e-3,
        max_steps=20,
        warmup_steps=5,
    )


@pytest.fixture
def sample_config_json(sample_config):
    """Serialize a sample config to JSON."""
    return json.dumps(sample_config.to_dict())


class TestJobDatabase:
    """Tests for the JobDatabase class."""

    def test_create_job_record(self, db, sample_config_json):
        """Test saving a new job record to the database."""
        job_id = "test-job-001"
        db.save_job(job_id=job_id, config_json=sample_config_json, status="pending")

        # Verify the job was saved by loading it
        record = db.load_job(job_id)
        assert record is not None
        assert record["job_id"] == job_id
        assert record["status"] == "pending"
        assert record["config_json"] == sample_config_json
        assert record["created_at"] is not None
        assert record["updated_at"] is not None
        assert record["error_message"] is None
        assert record["metrics_json"] is None

    def test_load_job_record(self, db, sample_config_json):
        """Test retrieving a saved job from the database."""
        job_id = "test-job-002"
        db.save_job(job_id=job_id, config_json=sample_config_json, status="pending")

        record = db.load_job(job_id)
        assert record is not None
        assert record["job_id"] == job_id

        # Verify config can be deserialized back
        config_dict = json.loads(record["config_json"])
        assert config_dict["batch_size"] == 4
        assert config_dict["learning_rate"] == 1e-3

    def test_load_nonexistent_job(self, db):
        """Test loading a job that doesn't exist returns None."""
        record = db.load_job("nonexistent-job")
        assert record is None

    def test_update_job_status(self, db, sample_config_json):
        """Test updating job status transitions."""
        job_id = "test-job-003"
        db.save_job(job_id=job_id, config_json=sample_config_json, status="pending")

        # Update to running
        db.update_job_status(job_id, "running")
        record = db.load_job(job_id)
        assert record["status"] == "running"

        # Update to completed
        db.update_job_status(job_id, "completed")
        record = db.load_job(job_id)
        assert record["status"] == "completed"

    def test_update_job_status_with_error(self, db, sample_config_json):
        """Test updating job status with an error message."""
        job_id = "test-job-004"
        db.save_job(job_id=job_id, config_json=sample_config_json, status="running")

        db.update_job_status(job_id, "failed", error_message="Out of memory")
        record = db.load_job(job_id)
        assert record["status"] == "failed"
        assert record["error_message"] == "Out of memory"

    def test_update_job_status_with_metrics(self, db, sample_config_json):
        """Test updating job status with final metrics."""
        job_id = "test-job-005"
        db.save_job(job_id=job_id, config_json=sample_config_json, status="running")

        metrics = json.dumps({"loss": 0.5, "perplexity": 1.65})
        db.update_job_status(job_id, "completed", metrics_json=metrics)
        record = db.load_job(job_id)
        assert record["status"] == "completed"
        assert record["metrics_json"] == metrics

    def test_update_job_updates_timestamp(self, db, sample_config_json):
        """Test that updating a job changes the updated_at timestamp."""
        job_id = "test-job-006"
        db.save_job(job_id=job_id, config_json=sample_config_json, status="pending")
        record_before = db.load_job(job_id)

        # Small delay to ensure different timestamp
        time.sleep(0.05)

        db.update_job_status(job_id, "running")
        record_after = db.load_job(job_id)
        assert record_after["updated_at"] >= record_before["updated_at"]

    def test_list_jobs(self, db, sample_config_json):
        """Test listing all jobs."""
        db.save_job(job_id="job-a", config_json=sample_config_json, status="pending")
        db.save_job(job_id="job-b", config_json=sample_config_json, status="running")
        db.save_job(job_id="job-c", config_json=sample_config_json, status="completed")

        all_jobs = db.list_jobs()
        assert len(all_jobs) == 3

    def test_list_jobs_with_status_filter(self, db, sample_config_json):
        """Test listing jobs filtered by status."""
        db.save_job(job_id="job-a", config_json=sample_config_json, status="pending")
        db.save_job(job_id="job-b", config_json=sample_config_json, status="running")
        db.save_job(job_id="job-c", config_json=sample_config_json, status="completed")
        db.save_job(job_id="job-d", config_json=sample_config_json, status="running")

        running_jobs = db.list_jobs(status="running")
        assert len(running_jobs) == 2
        assert all(j["status"] == "running" for j in running_jobs)

        completed_jobs = db.list_jobs(status="completed")
        assert len(completed_jobs) == 1

    def test_list_jobs_empty(self, db):
        """Test listing jobs when database is empty."""
        jobs = db.list_jobs()
        assert jobs == []

    def test_mark_running_as_failed_on_startup(self, temp_db_path, sample_config_json):
        """Test that running jobs are marked as failed when a new DB instance is created."""
        # First instance: create jobs with various statuses
        db1 = JobDatabase(db_path=temp_db_path)
        db1.save_job(job_id="job-running-1", config_json=sample_config_json, status="running")
        db1.save_job(job_id="job-running-2", config_json=sample_config_json, status="running")
        db1.save_job(job_id="job-completed", config_json=sample_config_json, status="completed")
        db1.save_job(job_id="job-pending", config_json=sample_config_json, status="pending")
        db1.save_job(job_id="job-failed", config_json=sample_config_json, status="failed")

        # Simulate server restart: create new instance with mark_running_failed=True
        db2 = JobDatabase(db_path=temp_db_path, mark_running_as_failed=True)

        # Previously running jobs should now be failed
        job1 = db2.load_job("job-running-1")
        assert job1["status"] == "failed"
        assert job1["error_message"] == "Server restarted"

        job2 = db2.load_job("job-running-2")
        assert job2["status"] == "failed"
        assert job2["error_message"] == "Server restarted"

        # Other jobs should be unchanged
        assert db2.load_job("job-completed")["status"] == "completed"
        assert db2.load_job("job-pending")["status"] == "pending"
        assert db2.load_job("job-failed")["status"] == "failed"

    def test_thread_safety(self, db, sample_config_json):
        """Test concurrent access to the database doesn't corrupt data."""
        num_threads = 10
        errors = []

        def create_and_update(thread_id):
            try:
                job_id = f"thread-job-{thread_id}"
                db.save_job(job_id=job_id, config_json=sample_config_json, status="pending")
                db.update_job_status(job_id, "running")
                db.update_job_status(job_id, "completed")
                record = db.load_job(job_id)
                assert record["status"] == "completed"
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=create_and_update, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread safety errors: {errors}"

        # Verify all jobs exist and are completed
        all_jobs = db.list_jobs()
        assert len(all_jobs) == num_threads
        assert all(j["status"] == "completed" for j in all_jobs)

    def test_save_duplicate_job_id_raises(self, db, sample_config_json):
        """Test that saving a job with a duplicate ID raises an error."""
        job_id = "duplicate-job"
        db.save_job(job_id=job_id, config_json=sample_config_json, status="pending")
        with pytest.raises(Exception):
            db.save_job(job_id=job_id, config_json=sample_config_json, status="pending")


class TestJobManagerIntegration:
    """Tests for TrainingJobManager integration with JobDatabase."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for DB and checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_jobs.db")
            checkpoint_dir = str(Path(tmpdir) / "checkpoints")
            yield db_path, checkpoint_dir

    def test_job_manager_saves_to_db(self, temp_dirs, sample_config):
        """Test that TrainingJobManager saves new jobs to the database."""
        from app.api.training_job_manager import TrainingJobManager, ResourceLimits

        db_path, checkpoint_dir = temp_dirs
        db = JobDatabase(db_path=db_path)
        manager = TrainingJobManager(
            resource_limits=ResourceLimits(max_concurrent_jobs=2),
            db=db,
        )

        sample_config.checkpoint_dir = checkpoint_dir
        job_id = manager.create_job(sample_config)

        # Verify job was persisted to DB
        record = db.load_job(job_id)
        assert record is not None
        assert record["job_id"] == job_id
        assert record["status"] == "pending"

    def test_job_manager_updates_status_on_completion(self, temp_dirs, sample_config):
        """Test that TrainingJobManager updates DB status when job completes."""
        from app.api.training_job_manager import TrainingJobManager, ResourceLimits

        db_path, checkpoint_dir = temp_dirs
        db = JobDatabase(db_path=db_path)
        manager = TrainingJobManager(
            resource_limits=ResourceLimits(
                max_concurrent_jobs=2,
                max_cpu_percent=100.0,
            ),
            db=db,
        )

        # Use minimal config so training finishes quickly
        sample_config.max_steps = 5
        sample_config.checkpoint_dir = checkpoint_dir
        sample_config.save_steps = 100  # Don't save checkpoints during test
        sample_config.logging_steps = 100
        sample_config.mixed_precision = None
        sample_config.gradient_checkpointing = False

        job_id = manager.create_job(sample_config)
        manager.start_job(job_id)

        # Wait for the job to complete
        job = manager.get_job(job_id)
        for _ in range(60):  # Wait up to 30 seconds
            if job.status in ("completed", "failed"):
                break
            time.sleep(0.5)

        # Wait for watcher thread to update DB
        time.sleep(1.0)

        # Verify DB was updated
        record = db.load_job(job_id)
        assert record is not None
        assert record["status"] in ("completed", "failed")

    def test_job_manager_updates_status_on_failure(self, temp_dirs, sample_config):
        """Test that TrainingJobManager updates DB when job fails."""
        from unittest.mock import patch
        from app.api.training_job_manager import TrainingJobManager, ResourceLimits

        db_path, checkpoint_dir = temp_dirs
        db = JobDatabase(db_path=db_path)
        manager = TrainingJobManager(
            resource_limits=ResourceLimits(
                max_concurrent_jobs=2,
                max_cpu_percent=100.0,
            ),
            db=db,
        )

        sample_config.max_steps = 5
        sample_config.checkpoint_dir = checkpoint_dir
        sample_config.save_steps = 100
        sample_config.logging_steps = 100
        sample_config.mixed_precision = None
        sample_config.gradient_checkpointing = False

        job_id = manager.create_job(sample_config)

        # Patch trainer.train to raise an error, forcing failure
        job = manager.get_job(job_id)
        with patch.object(job.trainer, 'train', side_effect=RuntimeError("OOM error")):
            manager.start_job(job_id)

            # Wait for the job to finish
            for _ in range(60):
                if job.status in ("completed", "failed"):
                    break
                time.sleep(0.5)

        assert job.status == "failed"
        assert job.error == "OOM error"

        # Wait for watcher thread to update DB
        time.sleep(1.0)
        record = db.load_job(job_id)
        assert record is not None
        assert record["status"] == "failed"
        assert record["error_message"] == "OOM error"

    def test_job_manager_works_without_db(self, temp_dirs, sample_config):
        """Test that TrainingJobManager still works without a database (backward compat)."""
        from app.api.training_job_manager import TrainingJobManager, ResourceLimits

        _, checkpoint_dir = temp_dirs
        manager = TrainingJobManager(
            resource_limits=ResourceLimits(max_concurrent_jobs=2),
        )

        sample_config.checkpoint_dir = checkpoint_dir
        job_id = manager.create_job(sample_config)
        assert job_id is not None
        job = manager.get_job(job_id)
        assert job is not None
