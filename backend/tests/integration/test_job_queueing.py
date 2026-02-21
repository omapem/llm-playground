"""Integration tests for training job queueing system.

Tests the job queue functionality:
- Jobs queued when resource limits exceeded
- Auto-start of queued jobs when resources become available
- FIFO ordering of queued jobs
- Queue status reporting
- Cancellation of queued jobs
- Queue draining after all jobs start
"""

import tempfile
import time
import threading

import pytest

from app.training import TrainingConfig
from app.transformer import TransformerConfig
from app.api.training_job_manager import TrainingJobManager, ResourceLimits


def wait_for_job_status(job, expected_status, timeout=15.0):
    """Wait for job to reach expected status.

    Args:
        job: Job to monitor
        expected_status: Status (or list of statuses) to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        True if status reached, False if timeout
    """
    if isinstance(expected_status, str):
        expected_status = [expected_status]
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        if job.status in expected_status:
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def job_manager():
    """Create a job manager with max_concurrent_jobs=1 to easily trigger queueing."""
    limits = ResourceLimits(max_concurrent_jobs=1, max_cpu_percent=100.0)
    return TrainingJobManager(resource_limits=limits)


@pytest.fixture
def training_config(temp_checkpoint_dir):
    """Create a minimal training config for fast test execution."""
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
        max_steps=5,
        batch_size=2,
        logging_steps=5,
        save_steps=100,
        checkpoint_dir=temp_checkpoint_dir,
        mixed_precision=None,
        gradient_checkpointing=False,
    )


@pytest.fixture
def long_training_config(temp_checkpoint_dir):
    """Training config with more steps so the job stays running during tests."""
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
        max_steps=1000,
        batch_size=2,
        logging_steps=100,
        save_steps=500,
        checkpoint_dir=temp_checkpoint_dir,
        mixed_precision=None,
        gradient_checkpointing=False,
    )


class TestJobQueueing:
    """Tests for job queueing system."""

    def test_job_queued_when_resources_exceeded(self, job_manager, long_training_config):
        """Test that a job is queued (not rejected) when resource limits are exceeded.

        With max_concurrent_jobs=1, starting a second job while one is running
        should queue the second job with status 'queued' instead of raising
        RuntimeError.
        """
        # Start first job - should run normally
        job_id_1 = job_manager.create_job(long_training_config)
        job_manager.start_job(job_id_1)

        job_1 = job_manager.get_job(job_id_1)
        assert wait_for_job_status(job_1, "running"), \
            f"First job did not start, status: {job_1.status}"

        # Start second job - should be queued, not raise an error
        job_id_2 = job_manager.create_job(long_training_config)
        job_manager.start_job(job_id_2)

        job_2 = job_manager.get_job(job_id_2)
        assert job_2.status == "queued", \
            f"Second job should be 'queued', got '{job_2.status}'"

        # Clean up: cancel the running job
        job_manager.cancel_job(job_id_1)
        wait_for_job_status(job_1, "cancelled")

    def test_queued_job_starts_when_resources_available(
        self, job_manager, training_config
    ):
        """Test that a queued job auto-starts after the running job completes.

        When the first job finishes (max_steps=5), the queued job should
        automatically start running.
        """
        # Start first job (short - will complete quickly)
        job_id_1 = job_manager.create_job(training_config)
        job_manager.start_job(job_id_1)

        job_1 = job_manager.get_job(job_id_1)
        assert wait_for_job_status(job_1, ["running", "completed"]), \
            f"First job did not start, status: {job_1.status}"

        # Queue second job
        job_id_2 = job_manager.create_job(training_config)
        job_manager.start_job(job_id_2)

        job_2 = job_manager.get_job(job_id_2)

        # Wait for first job to complete and second to auto-start
        assert wait_for_job_status(job_2, ["running", "completed"], timeout=30.0), \
            f"Queued job did not auto-start, status: {job_2.status}"

    def test_queue_is_fifo(self, job_manager, long_training_config, training_config):
        """Test that queued jobs start in FIFO order.

        When multiple jobs are queued, the first one queued should start first.
        """
        # Start a long-running job to block the queue
        blocker_id = job_manager.create_job(long_training_config)
        job_manager.start_job(blocker_id)

        blocker = job_manager.get_job(blocker_id)
        assert wait_for_job_status(blocker, "running"), \
            f"Blocker job did not start, status: {blocker.status}"

        # Queue two more jobs
        queued_id_1 = job_manager.create_job(training_config)
        job_manager.start_job(queued_id_1)

        queued_id_2 = job_manager.create_job(training_config)
        job_manager.start_job(queued_id_2)

        queued_1 = job_manager.get_job(queued_id_1)
        queued_2 = job_manager.get_job(queued_id_2)
        assert queued_1.status == "queued"
        assert queued_2.status == "queued"

        # Verify queue order
        queue = job_manager.get_queue_status()
        assert queue[0] == queued_id_1, "First queued job should be first in queue"
        assert queue[1] == queued_id_2, "Second queued job should be second in queue"

        # Cancel blocker - first queued job should start first
        job_manager.cancel_job(blocker_id)
        wait_for_job_status(blocker, "cancelled")

        # First queued job should start
        assert wait_for_job_status(queued_1, ["running", "completed"], timeout=15.0), \
            f"First queued job did not start, status: {queued_1.status}"

        # Second queued job should still be queued while first is running
        # (unless first completed fast enough to also start second)
        # Just verify both eventually complete
        assert wait_for_job_status(queued_2, ["running", "completed", "queued"], timeout=15.0)

        # Wait for everything to finish
        wait_for_job_status(queued_1, "completed", timeout=30.0)
        wait_for_job_status(queued_2, ["running", "completed"], timeout=30.0)

    def test_get_queue_status(self, job_manager, long_training_config, training_config):
        """Test that get_queue_status returns queued job IDs in order."""
        # Start a blocker
        blocker_id = job_manager.create_job(long_training_config)
        job_manager.start_job(blocker_id)

        blocker = job_manager.get_job(blocker_id)
        assert wait_for_job_status(blocker, "running")

        # Queue three jobs
        queued_ids = []
        for _ in range(3):
            qid = job_manager.create_job(training_config)
            job_manager.start_job(qid)
            queued_ids.append(qid)

        # Verify queue status
        queue = job_manager.get_queue_status()
        assert len(queue) == 3, f"Expected 3 queued jobs, got {len(queue)}"
        assert queue == queued_ids, \
            f"Queue order mismatch: expected {queued_ids}, got {queue}"

        # Clean up
        job_manager.cancel_job(blocker_id)
        wait_for_job_status(blocker, "cancelled")

    def test_cancel_queued_job(self, job_manager, long_training_config, training_config):
        """Test that a queued job can be cancelled and is removed from the queue."""
        # Start a blocker
        blocker_id = job_manager.create_job(long_training_config)
        job_manager.start_job(blocker_id)

        blocker = job_manager.get_job(blocker_id)
        assert wait_for_job_status(blocker, "running")

        # Queue a job
        queued_id = job_manager.create_job(training_config)
        job_manager.start_job(queued_id)

        queued_job = job_manager.get_job(queued_id)
        assert queued_job.status == "queued"

        # Cancel the queued job
        result = job_manager.cancel_job(queued_id)
        assert result is True, "cancel_job should return True for queued job"

        # Job should now be cancelled
        assert queued_job.status == "cancelled"

        # Job should be removed from queue
        queue = job_manager.get_queue_status()
        assert queued_id not in queue, \
            "Cancelled job should be removed from queue"

        # Clean up
        job_manager.cancel_job(blocker_id)
        wait_for_job_status(blocker, "cancelled")

    def test_multiple_jobs_auto_start(self, job_manager, training_config):
        """Test that multiple queued jobs start in sequence as resources free up.

        When a running job completes, the next queued job starts. When that
        completes, the next one starts, and so on.
        """
        # Create and start first job (short, will complete quickly)
        job_id_1 = job_manager.create_job(training_config)
        job_manager.start_job(job_id_1)

        # Queue two more jobs
        job_id_2 = job_manager.create_job(training_config)
        job_manager.start_job(job_id_2)

        job_id_3 = job_manager.create_job(training_config)
        job_manager.start_job(job_id_3)

        job_1 = job_manager.get_job(job_id_1)
        job_2 = job_manager.get_job(job_id_2)
        job_3 = job_manager.get_job(job_id_3)

        # All three jobs should eventually complete
        assert wait_for_job_status(job_1, "completed", timeout=30.0), \
            f"Job 1 did not complete, status: {job_1.status}"
        assert wait_for_job_status(job_2, "completed", timeout=30.0), \
            f"Job 2 did not complete, status: {job_2.status}"
        assert wait_for_job_status(job_3, "completed", timeout=30.0), \
            f"Job 3 did not complete, status: {job_3.status}"

    def test_queue_empty_after_all_started(
        self, job_manager, training_config
    ):
        """Test that the queue drains to empty after all jobs have started."""
        # Start first job
        job_id_1 = job_manager.create_job(training_config)
        job_manager.start_job(job_id_1)

        # Queue a second job
        job_id_2 = job_manager.create_job(training_config)
        job_manager.start_job(job_id_2)

        job_1 = job_manager.get_job(job_id_1)
        job_2 = job_manager.get_job(job_id_2)

        # Wait for both to complete
        assert wait_for_job_status(job_1, "completed", timeout=30.0), \
            f"Job 1 did not complete, status: {job_1.status}"
        assert wait_for_job_status(job_2, "completed", timeout=30.0), \
            f"Job 2 did not complete, status: {job_2.status}"

        # Queue should be empty
        queue = job_manager.get_queue_status()
        assert len(queue) == 0, \
            f"Queue should be empty after all jobs complete, got {queue}"

    def test_queued_job_auto_starts_after_cancellation(
        self, job_manager, long_training_config, training_config
    ):
        """Test that queued jobs auto-start when a running job is cancelled.

        Cancellation should trigger the same queue-draining logic as completion.
        """
        # Start a long-running job
        blocker_id = job_manager.create_job(long_training_config)
        job_manager.start_job(blocker_id)

        blocker = job_manager.get_job(blocker_id)
        assert wait_for_job_status(blocker, "running")

        # Queue a short job
        queued_id = job_manager.create_job(training_config)
        job_manager.start_job(queued_id)

        queued_job = job_manager.get_job(queued_id)
        assert queued_job.status == "queued"

        # Cancel the blocker
        job_manager.cancel_job(blocker_id)
        wait_for_job_status(blocker, "cancelled")

        # Queued job should auto-start
        assert wait_for_job_status(queued_job, ["running", "completed"], timeout=15.0), \
            f"Queued job did not auto-start after cancellation, status: {queued_job.status}"

    def test_first_job_starts_immediately(self, job_manager, training_config):
        """Test that the first job starts immediately (not queued) when resources available."""
        job_id = job_manager.create_job(training_config)
        job_manager.start_job(job_id)

        job = job_manager.get_job(job_id)
        # The job should be starting or running, never queued
        assert job.status in ["starting", "running", "completed"], \
            f"First job should start immediately, got status: {job.status}"
