"""Tests for training API endpoints."""

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.training.config import TrainingConfig
from app.transformer import TransformerConfig


client = TestClient(app)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config_dict(temp_checkpoint_dir):
    """Create sample training configuration."""
    return {
        "model_config": {
            "vocab_size": 50257,
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 2,
            "max_position_embeddings": 128,
        },
        "batch_size": 4,
        "learning_rate": 1e-3,
        "max_steps": 10,
        "warmup_steps": 2,
        "checkpoint_dir": temp_checkpoint_dir,
        "save_steps": 5,
        "logging_steps": 2,
    }


class TestTrainingConfigEndpoints:
    """Tests for training configuration endpoints."""

    def test_create_config(self, sample_config_dict):
        """Test creating a new training config."""
        response = client.post("/api/training/configs", json={
            "name": "test_config",
            "config": sample_config_dict,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_config"
        assert "config_id" in data

    def test_list_configs(self, sample_config_dict):
        """Test listing all configs."""
        # Create a config first
        client.post("/api/training/configs", json={
            "name": "list_test",
            "config": sample_config_dict,
        })

        response = client.get("/api/training/configs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_config_by_id(self, sample_config_dict):
        """Test retrieving config by ID."""
        # Create config
        create_response = client.post("/api/training/configs", json={
            "name": "get_test",
            "config": sample_config_dict,
        })
        config_id = create_response.json()["config_id"]

        # Get config
        response = client.get(f"/api/training/configs/{config_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "get_test"

    def test_get_nonexistent_config(self):
        """Test getting config that doesn't exist."""
        response = client.get("/api/training/configs/nonexistent")
        assert response.status_code == 404

    def test_delete_config(self, sample_config_dict):
        """Test deleting a config."""
        # Create config
        create_response = client.post("/api/training/configs", json={
            "name": "delete_test",
            "config": sample_config_dict,
        })
        config_id = create_response.json()["config_id"]

        # Delete config
        response = client.delete(f"/api/training/configs/{config_id}")
        assert response.status_code == 200

        # Verify it's gone
        get_response = client.get(f"/api/training/configs/{config_id}")
        assert get_response.status_code == 404


class TestTrainingJobEndpoints:
    """Tests for training job management endpoints."""

    def test_start_training_job(self, sample_config_dict):
        """Test starting a new training job."""
        response = client.post("/api/training/jobs/start", json={
            "config": sample_config_dict,
            "dataset_path": None,  # Will use mock dataset
        })
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "running"

    def test_get_job_status(self, sample_config_dict):
        """Test getting training job status."""
        # Start job
        start_response = client.post("/api/training/jobs/start", json={
            "config": sample_config_dict,
            "dataset_path": None,
        })
        job_id = start_response.json()["job_id"]

        # Get status
        response = client.get(f"/api/training/jobs/{job_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "current_step" in data

    def test_get_nonexistent_job_status(self):
        """Test getting status of nonexistent job."""
        response = client.get("/api/training/jobs/nonexistent/status")
        assert response.status_code == 404

    def test_stop_training_job(self, sample_config_dict):
        """Test stopping a training job."""
        # Start job
        start_response = client.post("/api/training/jobs/start", json={
            "config": sample_config_dict,
            "dataset_path": None,
        })
        job_id = start_response.json()["job_id"]

        # Stop job
        response = client.post(f"/api/training/jobs/{job_id}/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["stopped", "completed"]

    def test_list_jobs(self, sample_config_dict):
        """Test listing all training jobs."""
        # Start a job
        client.post("/api/training/jobs/start", json={
            "config": sample_config_dict,
            "dataset_path": None,
        })

        response = client.get("/api/training/jobs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0


class TestMetricsEndpoints:
    """Tests for training metrics endpoints."""

    def test_get_job_metrics(self, sample_config_dict):
        """Test retrieving training metrics."""
        # Start job
        start_response = client.post("/api/training/jobs/start", json={
            "config": sample_config_dict,
            "dataset_path": None,
        })
        job_id = start_response.json()["job_id"]

        # Wait a bit for metrics to accumulate
        import time
        time.sleep(1)

        # Get metrics
        response = client.get(f"/api/training/jobs/{job_id}/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data

    def test_get_metrics_history(self, sample_config_dict):
        """Test retrieving metrics history."""
        # Start job
        start_response = client.post("/api/training/jobs/start", json={
            "config": sample_config_dict,
            "dataset_path": None,
        })
        job_id = start_response.json()["job_id"]

        # Get metrics history
        response = client.get(f"/api/training/jobs/{job_id}/metrics/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_throughput(self, sample_config_dict):
        """Test retrieving training throughput."""
        # Start job
        start_response = client.post("/api/training/jobs/start", json={
            "config": sample_config_dict,
            "dataset_path": None,
        })
        job_id = start_response.json()["job_id"]

        # Get throughput
        response = client.get(f"/api/training/jobs/{job_id}/metrics/throughput")
        assert response.status_code == 200
        data = response.json()
        assert "steps_per_second" in data
        assert "tokens_per_second" in data


class TestCheckpointEndpoints:
    """Tests for checkpoint management endpoints."""

    def test_list_checkpoints(self, sample_config_dict):
        """Test listing checkpoints for a job."""
        # Start job with checkpoint saving
        start_response = client.post("/api/training/jobs/start", json={
            "config": sample_config_dict,
            "dataset_path": None,
        })
        job_id = start_response.json()["job_id"]

        # Wait for training to create checkpoints
        import time
        time.sleep(3)

        # List checkpoints
        response = client.get(f"/api/training/jobs/{job_id}/checkpoints")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_latest_checkpoint(self, sample_config_dict):
        """Test getting latest checkpoint."""
        # Start job
        start_response = client.post("/api/training/jobs/start", json={
            "config": sample_config_dict,
            "dataset_path": None,
        })
        job_id = start_response.json()["job_id"]

        # Wait for checkpoint
        import time
        time.sleep(3)

        # Get latest checkpoint
        response = client.get(f"/api/training/jobs/{job_id}/checkpoints/latest")
        assert response.status_code == 200
        data = response.json()
        if data:  # May be None if no checkpoints yet
            assert "step" in data
            assert "path" in data

    def test_download_checkpoint(self, sample_config_dict):
        """Test downloading a checkpoint file."""
        # Start job
        start_response = client.post("/api/training/jobs/start", json={
            "config": sample_config_dict,
            "dataset_path": None,
        })
        job_id = start_response.json()["job_id"]

        # Wait for checkpoint
        import time
        time.sleep(3)

        # Get checkpoint list
        checkpoints_response = client.get(f"/api/training/jobs/{job_id}/checkpoints")
        checkpoints = checkpoints_response.json()

        if checkpoints:
            checkpoint_path = checkpoints[0]["path"]
            # Download checkpoint
            response = client.get(f"/api/training/jobs/{job_id}/checkpoints/download",
                                params={"checkpoint_path": checkpoint_path})
            assert response.status_code == 200


class TestResumeEndpoints:
    """Tests for training resumption endpoints."""

    def test_resume_from_checkpoint(self, sample_config_dict):
        """Test resuming training from checkpoint."""
        # Start initial job
        start_response = client.post("/api/training/jobs/start", json={
            "config": sample_config_dict,
            "dataset_path": None,
        })
        job_id = start_response.json()["job_id"]

        # Wait for completion
        import time
        time.sleep(3)

        # Stop job
        client.post(f"/api/training/jobs/{job_id}/stop")

        # Resume from checkpoint
        response = client.post(f"/api/training/jobs/{job_id}/resume")
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["resumed_from_step"] >= 0


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self):
        """Test basic health check."""
        response = client.get("/api/training/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_training_status_overview(self):
        """Test getting training system status."""
        response = client.get("/api/training/status")
        assert response.status_code == 200
        data = response.json()
        assert "active_jobs" in data
        assert "total_jobs" in data
