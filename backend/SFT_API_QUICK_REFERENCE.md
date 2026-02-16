# SFT API Quick Reference

Fast reference for SFT API endpoints and job manager usage.

## Endpoints

### Start Training
```bash
POST /api/sft/train
Body: {"config": {...}}
→ {"job_id": "...", "status": "running"}
```

### Get Job Status
```bash
GET /api/sft/jobs/{job_id}
→ {"job_id": "...", "status": "...", "created_at": "...", ...}
```

### List Jobs
```bash
GET /api/sft/jobs
→ {"jobs": [...]}
```

### Stop Job
```bash
POST /api/sft/jobs/{job_id}/stop
→ {"message": "Job ... stopped"}
```

### Save Config
```bash
POST /api/sft/configs
Body: {"name": "...", "config": {...}}
→ {"config_id": "...", "name": "..."}
```

### List Configs
```bash
GET /api/sft/configs
→ [{"name": "...", "config": {...}}, ...]
```

### Get Config
```bash
GET /api/sft/configs/{config_id}
→ {"name": "...", "config": {...}}
```

## Python Usage

### Start Job Programmatically
```python
from app.api.sft_job_manager import sft_job_manager
from app.sft import SFTConfig

config = SFTConfig(
    base_model="gpt2",
    dataset_name="tatsu-lab/alpaca",
    dataset_format="alpaca",
    max_steps=100
)

job_id = sft_job_manager.create_job(config)
sft_job_manager.start_job(job_id)
```

### Monitor Job
```python
status = sft_job_manager.get_job_status(job_id)
print(f"Status: {status['status']}")

if status['status'] == 'completed':
    print(f"Results: {status['result']}")
```

### List All Jobs
```python
jobs = sft_job_manager.list_jobs()
for job in jobs:
    print(f"{job['job_id']}: {job['status']}")
```

### Save Configuration
```python
config_dict = {
    "base_model": "gpt2",
    "dataset_name": "tatsu-lab/alpaca",
    "use_lora": True
}

config_id = sft_job_manager.save_config("my_config", config_dict)
```

## Status Values

- `pending`: Job created, not started
- `running`: Training in progress
- `completed`: Training finished successfully
- `failed`: Training failed (check `error` field)
- `stopped`: Job manually stopped

## Config Example

```json
{
  "base_model": "gpt2",
  "dataset_name": "tatsu-lab/alpaca",
  "dataset_format": "alpaca",
  "max_steps": 100,
  "output_dir": "./outputs/sft",
  "use_lora": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "learning_rate": 2e-4,
  "batch_size": 4
}
```

## Testing

```bash
# Run API tests
pytest tests/unit/test_sft_api.py -v

# Run with coverage
pytest tests/unit/test_sft_api.py --cov=app.api.sft_job_manager --cov=app.api.sft_routes
```

## Error Codes

- `400`: Invalid configuration
- `404`: Job or config not found
- `500`: Internal server error

## Files

- Job Manager: `backend/app/api/sft_job_manager.py`
- API Routes: `backend/app/api/sft_routes.py`
- Tests: `backend/tests/unit/test_sft_api.py`
- Main App: `backend/app/main.py`
