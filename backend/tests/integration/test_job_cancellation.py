"""Integration tests for training job cancellation.

Tests the cancellation functionality across the full stack:
- TrainingJobManager cancellation flags
- Cancellation event propagation through the training loop
- API endpoint for job cancellation
- Thread-safe cleanup
"""

import time
import threading
import pytest
from app.training import TrainingConfig
from app.transformer import TransformerConfig
from app.api.training_job_manager import TrainingJobManager


def wait_for_job_status(job, expected_status, timeout=10.0):
    """Wait for job to reach expected status.

    Args:
        job: Job to monitor
        expected_status: Status to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        True if status reached, False if timeout
    """
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        if job.status == expected_status:
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def job_manager():
    """Create a fresh job manager for each test."""
    return TrainingJobManager()


@pytest.fixture
def training_config():
    """Create a minimal training config for testing."""
    model_config = TransformerConfig(
        vocab_size=50257,
        max_position_embeddings=128,
        num_layers=2,
        num_heads=2,
        hidden_size=128,
        intermediate_size=512,
    )

    return TrainingConfig(
        model_config=model_config,
        max_steps=1000,  # Long enough that cancellation can interrupt
        batch_size=2,
        logging_steps=10,
        save_steps=100,
        checkpoint_dir="/tmp/test_checkpoints",
    )


def test_cancel_running_job(job_manager, training_config):
    """Test cancelling a running training job.

    This test verifies:
    1. Job starts and reaches 'running' state
    2. cancel_job() can be called while job is running
    3. Job status transitions to 'cancelling' and then 'cancelled'
    4. Training loop exits cleanly
    """
    # Create and start job
    job_id = job_manager.create_job(training_config)
    job_manager.start_job(job_id)

    # Wait for job to start running
    time.sleep(1.0)
    job = job_manager.get_job(job_id)
    assert job is not None
    assert job.status == "running"

    # Cancel the job
    job_manager.cancel_job(job_id)

    # Job should transition to cancelling
    assert job.status in ["cancelling", "cancelled"]

    # Wait for cancellation to complete
    assert wait_for_job_status(job, "cancelled", timeout=10.0), \
        f"Job did not reach cancelled status in time, current status: {job.status}"

    # Thread should be finished
    assert job.thread is not None
    assert not job.thread.is_alive()


def test_cancel_nonexistent_job(job_manager):
    """Test attempting to cancel a non-existent job.

    Should raise ValueError with appropriate message.
    """
    with pytest.raises(ValueError, match="Job .* not found"):
        job_manager.cancel_job("nonexistent-job-id")


def test_cancel_already_completed_job(job_manager, training_config):
    """Test attempting to cancel an already completed job.

    This test verifies:
    1. Completed jobs cannot be cancelled
    2. Attempting to cancel returns False or raises appropriate error
    3. Job status remains 'completed'
    """
    # Create job
    job_id = job_manager.create_job(training_config)
    job = job_manager.get_job(job_id)

    # Manually mark job as completed (simulating completion)
    job.status = "completed"
    job.completed_at = time.time()

    # Attempt to cancel - should return False since it's already completed
    result = job_manager.cancel_job(job_id)
    assert result is False

    # Status should remain completed
    assert job.status == "completed"

    # Test with failed status too
    job.status = "failed"
    result = job_manager.cancel_job(job_id)
    assert result is False
    assert job.status == "failed"


def test_api_cancel_endpoint(training_config):
    """Test the API endpoint for job cancellation.

    This test verifies:
    1. POST /api/training/jobs/{job_id}/cancel endpoint exists
    2. Returns 200 with updated status
    3. Returns 404 for non-existent jobs
    4. Properly cancels the job
    """
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.training_job_manager import job_manager

    client = TestClient(app)

    # Create and start job using the global job_manager
    job_id = job_manager.create_job(training_config)
    job_manager.start_job(job_id)

    # Wait for job to start
    time.sleep(1.0)

    # Test cancelling via API
    response = client.post(f"/api/training/jobs/{job_id}/cancel")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["cancelling", "cancelled"]

    # Wait for cancellation
    job = job_manager.get_job(job_id)
    assert wait_for_job_status(job, "cancelled", timeout=10.0), \
        f"Job did not reach cancelled status in time, current status: {job.status}"

    # Test cancelling non-existent job
    response = client.post("/api/training/jobs/nonexistent-job/cancel")
    assert response.status_code == 404


def test_cancellation_event_propagation(job_manager, training_config):
    """Test that cancellation event is properly propagated to training loop.

    This test verifies:
    1. Cancellation event is created when job starts
    2. Event is passed through to Trainer.train()
    3. Training loop checks the event periodically
    4. Loop exits when event is set
    """
    # Create and start job
    job_id = job_manager.create_job(training_config)

    # Verify cancellation_event exists in job_manager
    assert hasattr(job_manager, "cancellation_flags")
    assert job_id not in job_manager.cancellation_flags

    # Start job - should create cancellation event
    job_manager.start_job(job_id)

    # Wait for job to start
    time.sleep(1.0)

    # Cancellation event should now exist
    assert job_id in job_manager.cancellation_flags
    event = job_manager.cancellation_flags[job_id]
    assert isinstance(event, threading.Event)
    assert not event.is_set()

    # Cancel job
    job_manager.cancel_job(job_id)

    # Event should be set
    assert event.is_set()

    # Wait for job to exit
    job = job_manager.get_job(job_id)
    assert wait_for_job_status(job, "cancelled", timeout=10.0), \
        f"Job did not reach cancelled status in time, current status: {job.status}"


def test_cancellation_cleans_up_event(job_manager, training_config):
    """Test that cancellation event is cleaned up after job completes.

    This verifies proper resource cleanup to prevent memory leaks.
    """
    # Create and start job
    job_id = job_manager.create_job(training_config)
    job_manager.start_job(job_id)

    # Wait for job to start
    time.sleep(1.0)

    # Event should exist
    assert job_id in job_manager.cancellation_flags

    # Cancel job
    job_manager.cancel_job(job_id)

    # Wait for cancellation to complete
    job = job_manager.get_job(job_id)
    assert wait_for_job_status(job, "cancelled", timeout=10.0), \
        f"Job did not reach cancelled status in time, current status: {job.status}"

    # Event should be cleaned up to prevent memory leaks
    assert job_id not in job_manager.cancellation_flags, \
        "Cancellation event should be cleaned up after job completion"


def test_multiple_cancellations_safe(job_manager, training_config):
    """Test that calling cancel_job multiple times is safe.

    This verifies idempotent behavior.
    """
    # Create and start job
    job_id = job_manager.create_job(training_config)
    job_manager.start_job(job_id)

    # Wait for job to start
    time.sleep(1.0)

    # Cancel multiple times
    job_manager.cancel_job(job_id)
    job_manager.cancel_job(job_id)
    job_manager.cancel_job(job_id)

    # Should not raise errors
    job = job_manager.get_job(job_id)
    assert wait_for_job_status(job, "cancelled", timeout=10.0), \
        f"Job did not reach cancelled status in time, current status: {job.status}"


def test_cancellation_during_checkpoint(job_manager, training_config):
    """Test cancelling a job during checkpoint save.

    This verifies that cancellation is safe even during I/O operations.
    """
    # Configure frequent checkpointing
    training_config.save_steps = 5

    # Create and start job
    job_id = job_manager.create_job(training_config)
    job_manager.start_job(job_id)

    # Wait for job to start and potentially hit a checkpoint
    time.sleep(1.5)

    # Cancel job
    job_manager.cancel_job(job_id)

    # Wait for cancellation
    job = job_manager.get_job(job_id)
    assert wait_for_job_status(job, "cancelled", timeout=10.0), \
        f"Job did not reach cancelled status in time, current status: {job.status}"


def test_no_memory_leak_multiple_jobs(job_manager, training_config):
    """Test that cancellation_flags doesn't grow indefinitely with multiple jobs.

    This verifies proper resource cleanup to prevent memory leaks.
    """
    job_ids = []

    # Start 3 jobs and cancel them all
    for i in range(3):
        job_id = job_manager.create_job(training_config)
        job_ids.append(job_id)
        job_manager.start_job(job_id)

    # Wait for all jobs to start
    time.sleep(1.5)

    # All jobs should have events
    for job_id in job_ids:
        assert job_id in job_manager.cancellation_flags

    # Cancel all jobs
    for job_id in job_ids:
        job_manager.cancel_job(job_id)

    # Wait for all cancellations to complete
    for job_id in job_ids:
        job = job_manager.get_job(job_id)
        assert wait_for_job_status(job, "cancelled", timeout=10.0), \
            f"Job {job_id} did not reach cancelled status in time"

    # All events should be cleaned up
    for job_id in job_ids:
        assert job_id not in job_manager.cancellation_flags, \
            f"Cancellation event for job {job_id} should be cleaned up after completion"

    # cancellation_flags should be empty
    assert len(job_manager.cancellation_flags) == 0, \
        f"cancellation_flags should be empty but has {len(job_manager.cancellation_flags)} entries"
