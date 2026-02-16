"""Integration tests for SFT API end-to-end workflows.

Tests complete user workflows from job creation through monitoring
and configuration management via the REST API.
"""

import pytest
from fastapi.testclient import TestClient
import time

from app.main import app


client = TestClient(app)


class TestSFTAPIIntegration:
    """End-to-end integration tests for SFT API."""

    def test_complete_training_workflow(self):
        """Test complete workflow: create job → start → monitor → check results."""
        # 1. Start training job
        response = client.post(
            "/api/sft/train",
            json={
                "config": {
                    "base_model": "gpt2",
                    "dataset_name": "tatsu-lab/alpaca",
                    "dataset_format": "alpaca",
                    "max_steps": 5,  # Very short for testing
                    "output_dir": "/tmp/test_sft_integration",
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        job_id = data["job_id"]
        assert data["status"] == "running"

        # 2. Check job appears in list
        response = client.get("/api/sft/jobs")
        assert response.status_code == 200
        jobs = response.json()["jobs"]
        job_ids = [job["job_id"] for job in jobs]
        assert job_id in job_ids

        # 3. Get job status
        response = client.get(f"/api/sft/jobs/{job_id}")
        assert response.status_code == 200
        status = response.json()
        assert status["job_id"] == job_id
        assert status["status"] in ["running", "completed", "failed"]
        assert status["created_at"] is not None
        assert status["started_at"] is not None

        # 4. Wait a bit for job to potentially complete
        time.sleep(0.5)

        # 5. Check final status
        response = client.get(f"/api/sft/jobs/{job_id}")
        assert response.status_code == 200
        final_status = response.json()
        assert final_status["status"] in ["running", "completed", "failed", "stopped"]

    def test_config_management_workflow(self):
        """Test workflow: save config → list configs → retrieve config → use for job."""
        # 1. Save configuration
        config_data = {
            "base_model": "gpt2",
            "dataset_name": "tatsu-lab/alpaca",
            "dataset_format": "alpaca",
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
        }

        response = client.post(
            "/api/sft/configs",
            json={
                "name": "test_integration_config",
                "config": config_data
            }
        )

        assert response.status_code == 200
        save_data = response.json()
        config_id = save_data["config_id"]
        assert save_data["name"] == "test_integration_config"

        # 2. List configs and verify it appears
        response = client.get("/api/sft/configs")
        assert response.status_code == 200
        configs = response.json()
        config_names = [cfg["name"] for cfg in configs]
        assert "test_integration_config" in config_names

        # 3. Retrieve config by ID
        response = client.get(f"/api/sft/configs/{config_id}")
        assert response.status_code == 200
        retrieved = response.json()
        assert retrieved["name"] == "test_integration_config"
        assert retrieved["config"]["base_model"] == "gpt2"
        assert retrieved["config"]["lora_r"] == 16

        # 4. Use config to start a job
        full_config = retrieved["config"].copy()
        full_config["max_steps"] = 5
        full_config["output_dir"] = "/tmp/test_config_job"

        response = client.post(
            "/api/sft/train",
            json={"config": full_config}
        )

        assert response.status_code == 200
        job_data = response.json()
        assert "job_id" in job_data
        assert job_data["status"] == "running"

    def test_job_stop_workflow(self):
        """Test workflow: start job → verify running → stop → verify stopped."""
        # 1. Start job
        response = client.post(
            "/api/sft/train",
            json={
                "config": {
                    "base_model": "gpt2",
                    "dataset_name": "tatsu-lab/alpaca",
                    "dataset_format": "alpaca",
                    "max_steps": 1000,  # Long enough to stop
                    "output_dir": "/tmp/test_stop_job",
                }
            }
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # 2. Give job time to start
        time.sleep(0.2)

        # 3. Verify job is running
        response = client.get(f"/api/sft/jobs/{job_id}")
        assert response.status_code == 200
        status_before = response.json()
        assert status_before["status"] == "running"

        # 4. Stop job
        response = client.post(f"/api/sft/jobs/{job_id}/stop")
        assert response.status_code == 200
        assert "stopped" in response.json()["message"]

        # 5. Verify job is stopped
        response = client.get(f"/api/sft/jobs/{job_id}")
        assert response.status_code == 200
        status_after = response.json()
        assert status_after["status"] == "stopped"

    def test_multiple_concurrent_jobs(self):
        """Test handling multiple jobs running concurrently."""
        job_ids = []

        # Start 3 concurrent jobs
        for i in range(3):
            response = client.post(
                "/api/sft/train",
                json={
                    "config": {
                        "base_model": "gpt2",
                        "dataset_name": "tatsu-lab/alpaca",
                        "dataset_format": "alpaca",
                        "max_steps": 5,
                        "output_dir": f"/tmp/test_concurrent_{i}",
                    }
                }
            )

            assert response.status_code == 200
            job_ids.append(response.json()["job_id"])

        # Verify all jobs exist
        response = client.get("/api/sft/jobs")
        assert response.status_code == 200
        all_job_ids = [job["job_id"] for job in response.json()["jobs"]]

        for job_id in job_ids:
            assert job_id in all_job_ids

        # Verify each job has status
        for job_id in job_ids:
            response = client.get(f"/api/sft/jobs/{job_id}")
            assert response.status_code == 200
            status = response.json()
            assert status["status"] in ["running", "completed", "failed"]

    def test_error_handling_invalid_config(self):
        """Test API error handling for invalid configurations."""
        # Missing required field
        response = client.post(
            "/api/sft/train",
            json={
                "config": {
                    "base_model": "gpt2",
                    # Missing dataset_name and dataset_format
                }
            }
        )

        assert response.status_code == 400
        assert "detail" in response.json()

    def test_error_handling_nonexistent_job(self):
        """Test API error handling for nonexistent jobs."""
        # Get status of non-existent job
        response = client.get("/api/sft/jobs/nonexistent-job-id")
        assert response.status_code == 404

        # Stop non-existent job
        response = client.post("/api/sft/jobs/nonexistent-job-id/stop")
        assert response.status_code == 404

    def test_error_handling_nonexistent_config(self):
        """Test API error handling for nonexistent configs."""
        response = client.get("/api/sft/configs/nonexistent-config-id")
        assert response.status_code == 404

    def test_job_status_timestamps(self):
        """Test that job timestamps are properly tracked."""
        # Start job
        response = client.post(
            "/api/sft/train",
            json={
                "config": {
                    "base_model": "gpt2",
                    "dataset_name": "tatsu-lab/alpaca",
                    "dataset_format": "alpaca",
                    "max_steps": 5,
                    "output_dir": "/tmp/test_timestamps",
                }
            }
        )

        job_id = response.json()["job_id"]

        # Check timestamps
        response = client.get(f"/api/sft/jobs/{job_id}")
        status = response.json()

        # created_at should always exist
        assert status["created_at"] is not None

        # started_at should exist after job starts
        assert status["started_at"] is not None

        # completed_at may or may not exist depending on timing
        # but it should be None or a valid timestamp
        assert status["completed_at"] is None or isinstance(status["completed_at"], str)

    def test_config_persistence_across_requests(self):
        """Test that saved configs persist across multiple API calls."""
        # Save config
        response = client.post(
            "/api/sft/configs",
            json={
                "name": "persistent_config",
                "config": {"base_model": "gpt2"}
            }
        )

        config_id = response.json()["config_id"]

        # Retrieve multiple times
        for _ in range(3):
            response = client.get(f"/api/sft/configs/{config_id}")
            assert response.status_code == 200
            assert response.json()["name"] == "persistent_config"

    def test_job_list_pagination_ready(self):
        """Test that job listing returns properly structured data for pagination."""
        # Create multiple jobs
        for i in range(5):
            client.post(
                "/api/sft/train",
                json={
                    "config": {
                        "base_model": "gpt2",
                        "dataset_name": "tatsu-lab/alpaca",
                        "dataset_format": "alpaca",
                        "max_steps": 5,
                        "output_dir": f"/tmp/test_list_{i}",
                    }
                }
            )

        # Get job list
        response = client.get("/api/sft/jobs")
        assert response.status_code == 200

        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)
        assert len(data["jobs"]) >= 5

        # Verify each job has required fields
        for job in data["jobs"]:
            assert "job_id" in job
            assert "status" in job
            assert "created_at" in job


class TestSFTAPIConsistency:
    """Tests to ensure API consistency and standards."""

    def test_all_endpoints_return_json(self):
        """Test that all endpoints return proper JSON responses."""
        # List jobs
        response = client.get("/api/sft/jobs")
        assert response.headers["content-type"] == "application/json"

        # List configs
        response = client.get("/api/sft/configs")
        assert response.headers["content-type"] == "application/json"

    def test_error_responses_have_detail(self):
        """Test that error responses include detail field."""
        # Invalid config
        response = client.post(
            "/api/sft/train",
            json={"config": {"invalid": "config"}}
        )
        assert "detail" in response.json()

        # Not found
        response = client.get("/api/sft/jobs/invalid")
        assert "detail" in response.json()

    def test_success_responses_have_expected_fields(self):
        """Test that success responses contain expected fields."""
        # Save config response
        response = client.post(
            "/api/sft/configs",
            json={"name": "test", "config": {"base_model": "gpt2"}}
        )
        data = response.json()
        assert "config_id" in data
        assert "name" in data

        # Start job response
        response = client.post(
            "/api/sft/train",
            json={
                "config": {
                    "base_model": "gpt2",
                    "dataset_name": "tatsu-lab/alpaca",
                    "dataset_format": "alpaca",
                    "max_steps": 5,
                    "output_dir": "/tmp/test_fields",
                }
            }
        )
        data = response.json()
        assert "job_id" in data
        assert "status" in data
