"""FastAPI routes for training management."""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.training import TrainingConfig
from app.transformer import TransformerConfig
from .training_job_manager import get_job_manager


router = APIRouter(prefix="/api/training", tags=["training"])


# Request/Response Models
class ConfigCreateRequest(BaseModel):
    """Request model for creating a config."""

    name: str
    config: Dict[str, Any]


class ConfigResponse(BaseModel):
    """Response model for config operations."""

    name: str
    config_id: str


class JobStartRequest(BaseModel):
    """Request model for starting a training job."""

    config: Dict[str, Any]
    dataset_path: Optional[str] = None


class JobStartResponse(BaseModel):
    """Response model for job start."""

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Response model for job status."""

    job_id: str
    status: str
    current_step: int
    max_steps: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


class JobListResponse(BaseModel):
    """Response model for listing jobs."""

    jobs: List[JobStatusResponse]


class MetricsResponse(BaseModel):
    """Response model for metrics."""

    metrics: Dict[str, float]


class ThroughputResponse(BaseModel):
    """Response model for throughput."""

    steps_per_second: float
    tokens_per_second: float


class CheckpointInfo(BaseModel):
    """Checkpoint information."""

    step: int
    path: str


class ResumeResponse(BaseModel):
    """Response model for resume operation."""

    job_id: str
    resumed_from_step: int


# Configuration Endpoints
@router.post("/configs", response_model=ConfigResponse)
async def create_config(request: ConfigCreateRequest):
    """Create a new training configuration."""
    config_id = get_job_manager().save_config(request.name, request.config)
    return ConfigResponse(name=request.name, config_id=config_id)


@router.get("/configs", response_model=List[Dict[str, Any]])
async def list_configs():
    """List all saved configurations."""
    return get_job_manager().list_configs()


@router.get("/configs/{config_id}")
async def get_config(config_id: str):
    """Get a configuration by ID."""
    config = get_job_manager().get_config(config_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Config not found")
    return config


@router.delete("/configs/{config_id}")
async def delete_config(config_id: str):
    """Delete a configuration."""
    if not get_job_manager().delete_config(config_id):
        raise HTTPException(status_code=404, detail="Config not found")
    return {"status": "deleted"}


# Training Job Endpoints
@router.post("/jobs/start", response_model=JobStartResponse)
async def start_training_job(request: JobStartRequest):
    """Start a new training job."""
    # Convert config dict to TrainingConfig
    config_dict = request.config.copy()

    # Extract and convert model_config
    model_config_dict = config_dict.pop("model_config")
    model_config = TransformerConfig(**model_config_dict)

    # Create TrainingConfig
    config = TrainingConfig(model_config=model_config, **config_dict)

    # Create and start job
    job_id = get_job_manager().create_job(config)
    get_job_manager().start_job(job_id)

    return JobStartResponse(job_id=job_id, status="running")


@router.get("/jobs")
async def list_jobs():
    """List all training jobs."""
    jobs = get_job_manager().list_jobs()
    return [job.get_status() for job in jobs]


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get training job status."""
    job = get_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(**job.get_status())


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """Stop a training job."""
    try:
        get_job_manager().stop_job(job_id)
        job = get_job_manager().get_job(job_id)
        return {"status": job.status if job else "stopped"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running training job.

    Sends a cancellation signal to the training loop, which will
    exit gracefully at the next cancellation check (every 10 steps).
    """
    try:
        cancelled = get_job_manager().cancel_job(job_id)
        job = get_job_manager().get_job(job_id)
        return {"status": job.status if job else "cancelled", "cancelled": cancelled}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Metrics Endpoints
@router.get("/jobs/{job_id}/metrics", response_model=MetricsResponse)
async def get_job_metrics(job_id: str):
    """Get current training metrics."""
    job = get_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    metrics = job.trainer.metrics_tracker.get_averages()
    return MetricsResponse(metrics=metrics)


@router.get("/jobs/{job_id}/metrics/history")
async def get_metrics_history(job_id: str):
    """Get metrics history."""
    job = get_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Return raw metrics from tracker
    # (In production, you'd want to persist this to a database)
    metrics_data = []
    for metric_name, values in job.trainer.metrics_tracker.metrics.items():
        metrics_data.append({
            "metric": metric_name,
            "values": list(values),
        })

    return metrics_data


@router.get("/jobs/{job_id}/metrics/throughput", response_model=ThroughputResponse)
async def get_job_throughput(job_id: str):
    """Get training throughput."""
    job = get_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    throughput = job.trainer.get_throughput()
    return ThroughputResponse(**throughput)


# Checkpoint Endpoints
@router.get("/jobs/{job_id}/checkpoints", response_model=List[CheckpointInfo])
async def list_checkpoints(job_id: str):
    """List checkpoints for a job."""
    job = get_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    checkpoints = job.trainer.checkpoint_manager.list_checkpoints()
    return [CheckpointInfo(**cp) for cp in checkpoints]


@router.get("/jobs/{job_id}/checkpoints/latest")
async def get_latest_checkpoint(job_id: str):
    """Get latest checkpoint for a job."""
    job = get_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    latest = job.trainer.checkpoint_manager.get_latest_checkpoint()
    if latest is None:
        return None

    # Extract step from filename
    import os
    filename = os.path.basename(latest)
    step = int(filename.split("_")[-1].replace(".pt", ""))

    return CheckpointInfo(step=step, path=latest)


@router.get("/jobs/{job_id}/checkpoints/download")
async def download_checkpoint(job_id: str, checkpoint_path: str = Query(...)):
    """Download a checkpoint file."""
    job = get_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Verify checkpoint exists and belongs to this job
    import os
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    return FileResponse(
        checkpoint_path,
        media_type="application/octet-stream",
        filename=os.path.basename(checkpoint_path),
    )


# Resume Endpoint
@router.post("/jobs/{job_id}/resume", response_model=ResumeResponse)
async def resume_job(job_id: str):
    """Resume training from latest checkpoint."""
    job = get_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Create new config with resume enabled
    config = job.config
    config.resume_from_checkpoint = True

    # Create new job
    new_job_id = get_job_manager().create_job(config)
    new_job = get_job_manager().get_job(new_job_id)

    # Get the step we're resuming from
    resumed_step = new_job.trainer.current_step

    # Start the job
    get_job_manager().start_job(new_job_id)

    return ResumeResponse(job_id=new_job_id, resumed_from_step=resumed_step)


# Health Endpoints
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/status")
async def training_status():
    """Get training system status."""
    jobs = get_job_manager().list_jobs()
    active_jobs = sum(1 for job in jobs if job.status == "running")

    return {
        "active_jobs": active_jobs,
        "total_jobs": len(jobs),
        "jobs_by_status": {
            "running": sum(1 for job in jobs if job.status == "running"),
            "completed": sum(1 for job in jobs if job.status == "completed"),
            "failed": sum(1 for job in jobs if job.status == "failed"),
            "stopped": sum(1 for job in jobs if job.status == "stopped"),
            "pending": sum(1 for job in jobs if job.status == "pending"),
        },
    }
