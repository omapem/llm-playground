"""Unit tests for SFT API routes and job manager.

Tests both the SFTJobManager and FastAPI routes for background
SFT training execution and management.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import time

from app.main import app
from app.api.sft_job_manager import SFTJobManager, SFTJob
from app.sft import SFTConfig


# Test client
client = TestClient(app)


class TestSFTJobManager:
    """Tests for SFTJobManager."""

    def setup_method(self):
        """Create fresh job manager for each test."""
        self.manager = SFTJobManager()

    def test_init(self):
        """Test job manager initialization."""
        assert isinstance(self.manager.jobs, dict)
        assert isinstance(self.manager.configs, dict)
        assert len(self.manager.jobs) == 0
        assert len(self.manager.configs) == 0

    def test_create_job(self):
        """Test job creation."""
        config = SFTConfig(
            base_model="gpt2",
            dataset_name="tatsu-lab/alpaca",
            dataset_format="alpaca",
            max_steps=10,
            output_dir="/tmp/test_sft",
        )

        job_id = self.manager.create_job(config)

        assert isinstance(job_id, str)
        assert len(job_id) > 0
        assert job_id in self.manager.jobs

        job = self.manager.jobs[job_id]
        assert isinstance(job, SFTJob)
        assert job.job_id == job_id
        assert job.config == config
        assert job.status == "pending"
        assert job.thread is None

    def test_create_job_with_callbacks(self):
        """Test job creation with callbacks."""
        from app.sft import SFTCallback

        class MockCallback(SFTCallback):
            def on_train_begin(self, trainer, **kwargs):
                pass

        config = SFTConfig(
            base_model="gpt2",
            dataset_name="tatsu-lab/alpaca",
            dataset_format="alpaca",
            max_steps=10,
            output_dir="/tmp/test_sft",
        )
        callbacks = [MockCallback()]

        job_id = self.manager.create_job(config, callbacks=callbacks)

        assert job_id in self.manager.jobs
        job = self.manager.jobs[job_id]
        assert job.trainer is not None

    def test_start_job(self):
        """Test starting a job."""
        config = SFTConfig(
            base_model="gpt2",
            dataset_name="tatsu-lab/alpaca",
            dataset_format="alpaca",
            max_steps=10,
            output_dir="/tmp/test_sft",
        )

        job_id = self.manager.create_job(config)
        self.manager.start_job(job_id)

        job = self.manager.jobs[job_id]
        assert job.status in ["running", "completed", "failed"]
        assert job.thread is not None
        assert job.started_at is not None

    def test_start_job_not_found(self):
        """Test starting non-existent job raises error."""
        with pytest.raises(ValueError, match="Job .* not found"):
            self.manager.start_job("nonexistent-id")

    def test_stop_job(self):
        """Test stopping a job."""
        config = SFTConfig(
            base_model="gpt2",
            dataset_name="tatsu-lab/alpaca",
            dataset_format="alpaca",
            max_steps=10,
            output_dir="/tmp/test_sft",
        )

        job_id = self.manager.create_job(config)
        self.manager.start_job(job_id)

        # Give job time to start
        time.sleep(0.1)

        self.manager.stop_job(job_id)

        job = self.manager.jobs[job_id]
        assert job.status == "stopped"

    def test_stop_job_not_found(self):
        """Test stopping non-existent job raises error."""
        with pytest.raises(ValueError, match="Job .* not found"):
            self.manager.stop_job("nonexistent-id")

    def test_get_job_status(self):
        """Test getting job status."""
        config = SFTConfig(
            base_model="gpt2",
            dataset_name="tatsu-lab/alpaca",
            dataset_format="alpaca",
            max_steps=10,
            output_dir="/tmp/test_sft",
        )

        job_id = self.manager.create_job(config)
        status = self.manager.get_job_status(job_id)

        assert isinstance(status, dict)
        assert status["job_id"] == job_id
        assert status["status"] == "pending"
        assert status["created_at"] is not None
        assert status["started_at"] is None
        assert status["completed_at"] is None
        assert status["error"] is None
        assert status["result"] is None

    def test_get_job_status_not_found(self):
        """Test getting status of non-existent job raises error."""
        with pytest.raises(ValueError, match="Job .* not found"):
            self.manager.get_job_status("nonexistent-id")

    def test_list_jobs_empty(self):
        """Test listing jobs when none exist."""
        jobs = self.manager.list_jobs()
        assert isinstance(jobs, list)
        assert len(jobs) == 0

    def test_list_jobs(self):
        """Test listing multiple jobs."""
        config1 = SFTConfig(
            base_model="gpt2",
            dataset_name="tatsu-lab/alpaca",
            dataset_format="alpaca",
            max_steps=10,
            output_dir="/tmp/test_sft1",
        )
        config2 = SFTConfig(
            base_model="gpt2",
            dataset_name="tatsu-lab/alpaca",
            dataset_format="alpaca",
            max_steps=10,
            output_dir="/tmp/test_sft2",
        )

        job_id1 = self.manager.create_job(config1)
        job_id2 = self.manager.create_job(config2)

        jobs = self.manager.list_jobs()

        assert len(jobs) == 2
        assert all(isinstance(job, dict) for job in jobs)
        job_ids = [job["job_id"] for job in jobs]
        assert job_id1 in job_ids
        assert job_id2 in job_ids

    def test_save_config(self):
        """Test saving a configuration."""
        config_dict = {
            "base_model": "gpt2",
            "dataset_name": "tatsu-lab/alpaca",
            "dataset_format": "alpaca",
            "max_steps": 10,
        }

        config_id = self.manager.save_config("test_config", config_dict)

        assert isinstance(config_id, str)
        assert len(config_id) > 0
        assert config_id in self.manager.configs

        saved = self.manager.configs[config_id]
        assert saved["name"] == "test_config"
        assert saved["config"] == config_dict

    def test_get_config(self):
        """Test getting a saved configuration."""
        config_dict = {
            "base_model": "gpt2",
            "dataset_name": "tatsu-lab/alpaca",
            "dataset_format": "alpaca",
        }

        config_id = self.manager.save_config("test_config", config_dict)
        retrieved = self.manager.get_config(config_id)

        assert retrieved["name"] == "test_config"
        assert retrieved["config"] == config_dict

    def test_get_config_not_found(self):
        """Test getting non-existent config raises error."""
        with pytest.raises(ValueError, match="Config .* not found"):
            self.manager.get_config("nonexistent-id")

    def test_list_configs_empty(self):
        """Test listing configs when none exist."""
        configs = self.manager.list_configs()
        assert isinstance(configs, list)
        assert len(configs) == 0

    def test_list_configs(self):
        """Test listing multiple configs."""
        config1 = {"base_model": "gpt2", "dataset_name": "alpaca"}
        config2 = {"base_model": "llama2", "dataset_name": "openassistant"}

        self.manager.save_config("config1", config1)
        self.manager.save_config("config2", config2)

        configs = self.manager.list_configs()

        assert len(configs) == 2
        assert all(isinstance(c, dict) for c in configs)
        names = [c["name"] for c in configs]
        assert "config1" in names
        assert "config2" in names


class TestSFTJob:
    """Tests for SFTJob."""

    def test_init(self):
        """Test job initialization."""
        config = SFTConfig(
            base_model="gpt2",
            dataset_name="tatsu-lab/alpaca",
            dataset_format="alpaca",
            max_steps=10,
            output_dir="/tmp/test_sft",
        )

        from app.sft import SFTTrainer
        trainer = SFTTrainer(config)

        job = SFTJob("test-job-id", config, trainer)

        assert job.job_id == "test-job-id"
        assert job.config == config
        assert job.trainer == trainer
        assert job.status == "pending"
        assert job.thread is None
        assert isinstance(job.created_at, datetime)
        assert job.started_at is None
        assert job.completed_at is None
        assert job.error is None
        assert job.result is None

    def test_get_status(self):
        """Test getting job status."""
        config = SFTConfig(
            base_model="gpt2",
            dataset_name="tatsu-lab/alpaca",
            dataset_format="alpaca",
            max_steps=10,
            output_dir="/tmp/test_sft",
        )

        from app.sft import SFTTrainer
        trainer = SFTTrainer(config)
        job = SFTJob("test-job-id", config, trainer)

        status = job.get_status()

        assert isinstance(status, dict)
        assert status["job_id"] == "test-job-id"
        assert status["status"] == "pending"
        assert status["created_at"] is not None
        assert status["started_at"] is None
        assert status["completed_at"] is None
        assert status["error"] is None
        assert status["result"] is None


class TestSFTAPIRoutes:
    """Tests for SFT API routes."""

    def test_start_training(self):
        """Test starting SFT training via API."""
        response = client.post(
            "/api/sft/train",
            json={
                "config": {
                    "base_model": "gpt2",
                    "dataset_name": "tatsu-lab/alpaca",
                    "dataset_format": "alpaca",
                    "max_steps": 10,
                    "output_dir": "/tmp/test_sft",
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_start_training_invalid_config(self):
        """Test starting training with invalid config."""
        response = client.post(
            "/api/sft/train",
            json={
                "config": {
                    # Missing required fields
                    "base_model": "gpt2",
                }
            }
        )

        assert response.status_code == 400

    def test_get_job_status(self):
        """Test getting job status via API."""
        # Start a job first
        start_response = client.post(
            "/api/sft/train",
            json={
                "config": {
                    "base_model": "gpt2",
                    "dataset_name": "tatsu-lab/alpaca",
                    "dataset_format": "alpaca",
                    "max_steps": 10,
                    "output_dir": "/tmp/test_sft",
                }
            }
        )
        job_id = start_response.json()["job_id"]

        # Get status
        response = client.get(f"/api/sft/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "created_at" in data

    def test_get_job_status_not_found(self):
        """Test getting status of non-existent job."""
        response = client.get("/api/sft/jobs/nonexistent-id")

        assert response.status_code == 404

    def test_list_jobs(self):
        """Test listing jobs via API."""
        # Start a couple jobs
        client.post(
            "/api/sft/train",
            json={
                "config": {
                    "base_model": "gpt2",
                    "dataset_name": "tatsu-lab/alpaca",
                    "dataset_format": "alpaca",
                    "max_steps": 10,
                    "output_dir": "/tmp/test_sft1",
                }
            }
        )
        client.post(
            "/api/sft/train",
            json={
                "config": {
                    "base_model": "gpt2",
                    "dataset_name": "tatsu-lab/alpaca",
                    "dataset_format": "alpaca",
                    "max_steps": 10,
                    "output_dir": "/tmp/test_sft2",
                }
            }
        )

        # List jobs
        response = client.get("/api/sft/jobs")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert len(data["jobs"]) >= 2

    def test_stop_job(self):
        """Test stopping a job via API."""
        # Start a job
        start_response = client.post(
            "/api/sft/train",
            json={
                "config": {
                    "base_model": "gpt2",
                    "dataset_name": "tatsu-lab/alpaca",
                    "dataset_format": "alpaca",
                    "max_steps": 10,
                    "output_dir": "/tmp/test_sft",
                }
            }
        )
        job_id = start_response.json()["job_id"]

        # Stop it
        response = client.post(f"/api/sft/jobs/{job_id}/stop")

        assert response.status_code == 200
        assert "message" in response.json()

    def test_stop_job_not_found(self):
        """Test stopping non-existent job."""
        response = client.post("/api/sft/jobs/nonexistent-id/stop")

        assert response.status_code == 404

    def test_save_config(self):
        """Test saving config via API."""
        response = client.post(
            "/api/sft/configs",
            json={
                "name": "test_config",
                "config": {
                    "base_model": "gpt2",
                    "dataset_name": "tatsu-lab/alpaca",
                    "dataset_format": "alpaca",
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "config_id" in data
        assert data["name"] == "test_config"

    def test_list_configs(self):
        """Test listing configs via API."""
        # Save a couple configs
        client.post(
            "/api/sft/configs",
            json={
                "name": "config1",
                "config": {"base_model": "gpt2"}
            }
        )
        client.post(
            "/api/sft/configs",
            json={
                "name": "config2",
                "config": {"base_model": "llama2"}
            }
        )

        # List configs
        response = client.get("/api/sft/configs")

        assert response.status_code == 200
        configs = response.json()
        assert isinstance(configs, list)
        assert len(configs) >= 2

    def test_get_config(self):
        """Test getting config via API."""
        # Save a config
        save_response = client.post(
            "/api/sft/configs",
            json={
                "name": "test_config",
                "config": {"base_model": "gpt2"}
            }
        )
        config_id = save_response.json()["config_id"]

        # Get it
        response = client.get(f"/api/sft/configs/{config_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_config"
        assert data["config"]["base_model"] == "gpt2"

    def test_get_config_not_found(self):
        """Test getting non-existent config."""
        response = client.get("/api/sft/configs/nonexistent-id")

        assert response.status_code == 404
