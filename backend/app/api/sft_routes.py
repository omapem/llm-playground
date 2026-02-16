"""FastAPI routes for SFT training management.

Provides REST API endpoints for supervised fine-tuning job management,
including job creation, status monitoring, and configuration storage.
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

from app.sft import SFTConfig
from .sft_job_manager import sft_job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sft", tags=["sft"])


# Request/Response Models
class JobStartRequest(BaseModel):
    """Request model for starting an SFT job.

    Attributes:
        config: Dictionary containing SFT configuration parameters
    """

    config: Dict[str, Any]


class JobStartResponse(BaseModel):
    """Response model for job start.

    Attributes:
        job_id: Unique identifier for the started job
        status: Initial job status (typically 'running')
    """

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Response model for job status.

    Attributes:
        job_id: Unique job identifier
        status: Current job status
        created_at: ISO timestamp of job creation
        started_at: ISO timestamp of training start (or None)
        completed_at: ISO timestamp of completion (or None)
        error: Error message if failed (or None)
        result: Training results if completed (or None)
    """

    job_id: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]
    result: Optional[Dict[str, Any]]


class JobListResponse(BaseModel):
    """Response model for listing jobs.

    Attributes:
        jobs: List of job status responses
    """

    jobs: List[JobStatusResponse]


class ConfigSaveRequest(BaseModel):
    """Request model for saving a config.

    Attributes:
        name: Human-readable configuration name
        config: Configuration dictionary
    """

    name: str
    config: Dict[str, Any]


class ConfigSaveResponse(BaseModel):
    """Response model for config save.

    Attributes:
        config_id: Unique identifier for saved config
        name: Configuration name
    """

    config_id: str
    name: str


# Job Management Endpoints
@router.post("/train", response_model=JobStartResponse)
async def start_training(request: JobStartRequest):
    """Start a new SFT training job.

    Creates a background job for supervised fine-tuning with the
    provided configuration. Training runs asynchronously and can
    be monitored via status endpoints.

    Args:
        request: Job start request with SFT configuration

    Returns:
        Job ID and initial status

    Raises:
        HTTPException: 400 if configuration is invalid

    Example:
        POST /api/sft/train
        {
            "config": {
                "base_model": "gpt2",
                "dataset_name": "tatsu-lab/alpaca",
                "dataset_format": "alpaca",
                "max_steps": 100,
                "output_dir": "./outputs/sft_run"
            }
        }

        Response:
        {
            "job_id": "abc-123-def-456",
            "status": "running"
        }
    """
    try:
        # Parse and validate config
        config = SFTConfig(**request.config)

        # Create and start job
        job_id = sft_job_manager.create_job(config)
        sft_job_manager.start_job(job_id)

        logger.info(f"Started SFT training job {job_id}")
        return JobStartResponse(job_id=job_id, status="running")

    except Exception as e:
        logger.error(f"Failed to start SFT training: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of an SFT training job.

    Retrieves current status, timestamps, and results for a
    specific training job.

    Args:
        job_id: Job identifier

    Returns:
        Job status with timestamps and results

    Raises:
        HTTPException: 404 if job not found

    Example:
        GET /api/sft/jobs/abc-123-def-456

        Response:
        {
            "job_id": "abc-123-def-456",
            "status": "completed",
            "created_at": "2024-01-01T12:00:00",
            "started_at": "2024-01-01T12:00:01",
            "completed_at": "2024-01-01T12:30:00",
            "error": null,
            "result": {"train_loss": 0.5, "eval_loss": 0.6}
        }
    """
    try:
        status = sft_job_manager.get_job_status(job_id)
        return JobStatusResponse(**status)

    except ValueError as e:
        logger.error(f"Job not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all SFT training jobs.

    Returns a list of all jobs with their current statuses,
    timestamps, and results.

    Returns:
        List of job statuses

    Example:
        GET /api/sft/jobs

        Response:
        {
            "jobs": [
                {
                    "job_id": "abc-123",
                    "status": "completed",
                    "created_at": "2024-01-01T12:00:00",
                    ...
                },
                {
                    "job_id": "def-456",
                    "status": "running",
                    "created_at": "2024-01-01T13:00:00",
                    ...
                }
            ]
        }
    """
    jobs = sft_job_manager.list_jobs()
    return JobListResponse(jobs=jobs)


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """Stop a running SFT training job.

    Marks a job as stopped. Note that due to threading limitations,
    this doesn't immediately kill the training process but prevents
    further progress tracking.

    Args:
        job_id: Job identifier

    Returns:
        Success message

    Raises:
        HTTPException: 404 if job not found

    Example:
        POST /api/sft/jobs/abc-123/stop

        Response:
        {
            "message": "Job abc-123 stopped"
        }
    """
    try:
        sft_job_manager.stop_job(job_id)
        logger.info(f"Stopped SFT job {job_id}")
        return {"message": f"Job {job_id} stopped"}

    except ValueError as e:
        logger.error(f"Job not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))


# Configuration Endpoints
@router.post("/configs", response_model=ConfigSaveResponse)
async def save_config(request: ConfigSaveRequest):
    """Save an SFT configuration for later use.

    Stores a configuration template that can be retrieved and
    reused for future training jobs.

    Args:
        request: Config save request with name and configuration

    Returns:
        Config ID and name

    Example:
        POST /api/sft/configs
        {
            "name": "alpaca_qlora_gpt2",
            "config": {
                "base_model": "gpt2",
                "dataset_name": "tatsu-lab/alpaca",
                "use_qlora": true,
                "lora_r": 16
            }
        }

        Response:
        {
            "config_id": "xyz-789",
            "name": "alpaca_qlora_gpt2"
        }
    """
    config_id = sft_job_manager.save_config(request.name, request.config)
    logger.info(f"Saved config '{request.name}' with ID {config_id}")
    return ConfigSaveResponse(config_id=config_id, name=request.name)


@router.get("/configs")
async def list_configs():
    """List all saved SFT configurations.

    Returns all stored configuration templates with their
    names and IDs.

    Returns:
        List of configs with names and configurations

    Example:
        GET /api/sft/configs

        Response:
        [
            {
                "name": "alpaca_qlora_gpt2",
                "config": {"base_model": "gpt2", ...}
            },
            {
                "name": "openassistant_lora_llama2",
                "config": {"base_model": "llama2-7b", ...}
            }
        ]
    """
    return sft_job_manager.list_configs()


@router.get("/configs/{config_id}")
async def get_config(config_id: str):
    """Get a saved configuration by ID.

    Retrieves a previously saved configuration template.

    Args:
        config_id: Configuration identifier

    Returns:
        Config name and configuration data

    Raises:
        HTTPException: 404 if config not found

    Example:
        GET /api/sft/configs/xyz-789

        Response:
        {
            "name": "alpaca_qlora_gpt2",
            "config": {
                "base_model": "gpt2",
                "dataset_name": "tatsu-lab/alpaca",
                "use_qlora": true
            }
        }
    """
    try:
        return sft_job_manager.get_config(config_id)

    except ValueError as e:
        logger.error(f"Config not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
