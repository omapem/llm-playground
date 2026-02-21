"""FastAPI routes for data collection and preparation."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from loguru import logger

from fastapi import UploadFile, File

from app.data.loaders import DatasetLoader, load_dataset, CustomDatasetLoader
from app.data.cleaning import DataCleaner
from app.data.splitter import TrainValSplitter
from app.data.stats import DataStats

router = APIRouter(prefix="/api/data", tags=["data"])


# ============================================================================
# Request/Response Models
# ============================================================================


class LoadDatasetRequest(BaseModel):
    """Request to load a dataset."""

    dataset_name: str = Field(..., description="Name of the dataset (wikitext, openwebtext, c4)")
    dataset_config: str = Field(..., description="Configuration/variant of the dataset")
    split: Optional[str] = Field(None, description="Dataset split (train, validation, test)")
    max_examples: Optional[int] = Field(None, description="Maximum number of examples to load")
    streaming: bool = Field(False, description="Whether to stream the dataset")


class DatasetInfoResponse(BaseModel):
    """Response with dataset information."""

    name: str
    config: str
    description: str
    text_column: str
    supported_configs: List[str]


class CleaningRequest(BaseModel):
    """Request to clean documents."""

    documents: List[str] = Field(..., description="Documents to clean")
    min_length: int = Field(50, description="Minimum document length")
    remove_duplicates: bool = Field(True, description="Remove duplicate documents")
    remove_pii: bool = Field(True, description="Remove PII from documents")
    allowed_languages: Optional[List[str]] = Field(None, description="Allowed languages (en, es, fr)")


class CleaningResponse(BaseModel):
    """Response after cleaning documents."""

    cleaned_documents: List[str]
    original_count: int
    cleaned_count: int
    duplicate_rate: float
    stats: Dict[str, Any]


class SplitRequest(BaseModel):
    """Request to split dataset."""

    items: List[Any] = Field(..., description="Items to split")
    train_ratio: float = Field(0.8, description="Proportion for training")
    val_ratio: float = Field(0.2, description="Proportion for validation")
    test_ratio: float = Field(0.0, description="Proportion for testing")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class SplitResponse(BaseModel):
    """Response after splitting data."""

    train_size: int
    val_size: int
    test_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float


class StatsRequest(BaseModel):
    """Request to compute dataset statistics."""

    documents: List[str] = Field(..., description="Documents to analyze")
    compute_language_dist: bool = Field(False, description="Compute language distribution")


class StatsResponse(BaseModel):
    """Response with dataset statistics."""

    num_documents: int
    total_characters: int
    total_words: int
    vocabulary_size: int
    avg_doc_length: float
    min_doc_length: int
    max_doc_length: int
    duplicate_rate: float
    language_distribution: Optional[Dict[str, int]] = None


# ============================================================================
# Dataset Loading Endpoints
# ============================================================================


@router.get("/datasets/supported")
async def get_supported_datasets() -> Dict[str, Dict[str, Any]]:
    """Get list of supported datasets.

    Returns:
        Dictionary mapping dataset names to their configurations
    """
    loader = DatasetLoader("wikitext", "wikitext-103-v1")
    return loader.get_supported_datasets()


@router.post("/datasets/info")
async def get_dataset_info(request: LoadDatasetRequest) -> DatasetInfoResponse:
    """Get information about a specific dataset.

    Args:
        request: Dataset information request

    Returns:
        Dataset information

    Raises:
        HTTPException: If dataset not found
    """
    try:
        loader = DatasetLoader(
            dataset_name=request.dataset_name,
            dataset_config=request.dataset_config
        )
        info = loader.get_dataset_info()

        return DatasetInfoResponse(
            name=request.dataset_name,
            config=request.dataset_config,
            description=info.get("description", ""),
            text_column=info.get("text_column", "text"),
            supported_configs=info.get("configs", [])
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/datasets/load")
async def load_data(request: LoadDatasetRequest) -> Dict[str, Any]:
    """Load a dataset from HuggingFace Hub.

    Args:
        request: Dataset loading request

    Returns:
        Dataset information and stats

    Raises:
        HTTPException: If dataset loading fails
    """
    try:
        loader = DatasetLoader(
            dataset_name=request.dataset_name,
            dataset_config=request.dataset_config,
            split=request.split,
            max_examples=request.max_examples,
            streaming=request.streaming,
        )

        dataset = loader.load()
        stats = loader.get_stats()

        return {
            "status": "success",
            "message": f"Loaded {stats.get('total_examples', 0)} examples",
            "dataset_stats": stats,
        }
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Data Cleaning Endpoints
# ============================================================================


@router.post("/clean", response_model=CleaningResponse)
async def clean_documents(request: CleaningRequest) -> CleaningResponse:
    """Clean a batch of documents.

    Args:
        request: Cleaning request with documents and options

    Returns:
        Cleaned documents and statistics

    Raises:
        HTTPException: If cleaning fails
    """
    try:
        cleaner = DataCleaner(
            min_length=request.min_length,
            remove_duplicates=request.remove_duplicates,
            remove_pii_enabled=request.remove_pii,
            allowed_languages=request.allowed_languages,
        )

        cleaned = cleaner.clean_documents(request.documents)
        stats = cleaner.get_cleaning_stats()

        return CleaningResponse(
            cleaned_documents=cleaned,
            original_count=len(request.documents),
            cleaned_count=len(cleaned),
            duplicate_rate=stats.get("duplicate_rate", 0.0),
            stats=stats,
        )
    except Exception as e:
        logger.error(f"Cleaning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Data Splitting Endpoints
# ============================================================================


@router.post("/split", response_model=SplitResponse)
async def split_data(request: SplitRequest) -> SplitResponse:
    """Split data into train/validation/test sets.

    Args:
        request: Split request with items and ratios

    Returns:
        Split statistics

    Raises:
        HTTPException: If split fails
    """
    try:
        splitter = TrainValSplitter(
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
            test_ratio=request.test_ratio,
            seed=request.seed,
        )

        if request.test_ratio > 0:
            train, val, test = splitter.split(request.items)
            test_size = len(test)
        else:
            train, val = splitter.split(request.items)
            test_size = 0

        return SplitResponse(
            train_size=len(train),
            val_size=len(val),
            test_size=test_size,
            train_ratio=len(train) / len(request.items) if request.items else 0,
            val_ratio=len(val) / len(request.items) if request.items else 0,
            test_ratio=test_size / len(request.items) if request.items else 0,
        )
    except Exception as e:
        logger.error(f"Split failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dataset Statistics Endpoints
# ============================================================================


@router.post("/stats", response_model=StatsResponse)
async def compute_statistics(request: StatsRequest) -> StatsResponse:
    """Compute statistics for a dataset.

    Args:
        request: Statistics request with documents

    Returns:
        Dataset statistics

    Raises:
        HTTPException: If computation fails
    """
    try:
        stats = DataStats(
            documents=request.documents,
            compute_language_dist=request.compute_language_dist,
        )
        stats.compute()

        summary = stats.get_summary()

        return StatsResponse(
            num_documents=summary["num_documents"],
            total_characters=summary["total_characters"],
            total_words=summary["total_words"],
            vocabulary_size=summary["vocabulary_size"],
            avg_doc_length=summary["avg_doc_length"],
            min_doc_length=summary["min_doc_length"],
            max_doc_length=summary["max_doc_length"],
            duplicate_rate=summary["duplicate_rate"],
            language_distribution=stats.get_language_distribution() if request.compute_language_dist else None,
        )
    except Exception as e:
        logger.error(f"Statistics computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stats/report")
async def get_statistics_report(request: StatsRequest) -> Dict[str, str]:
    """Get a detailed statistics report.

    Args:
        request: Statistics request with documents

    Returns:
        Formatted statistics report

    Raises:
        HTTPException: If computation fails
    """
    try:
        stats = DataStats(
            documents=request.documents,
            compute_language_dist=request.compute_language_dist,
        )
        stats.compute()

        return {
            "report": stats.get_detailed_report()
        }
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Pipeline Endpoints
# ============================================================================


@router.post("/pipeline")
async def run_data_pipeline(
    dataset_name: str = Query(..., description="Dataset to load"),
    dataset_config: str = Query(..., description="Dataset configuration"),
    max_examples: Optional[int] = Query(None, description="Max examples to load"),
    min_doc_length: int = Query(50, description="Minimum document length"),
    remove_duplicates: bool = Query(True, description="Remove duplicates"),
    train_ratio: float = Query(0.8, description="Training split ratio"),
    val_ratio: float = Query(0.2, description="Validation split ratio"),
) -> Dict[str, Any]:
    """Run complete data pipeline: load → clean → split → stats.

    Args:
        dataset_name: Name of dataset to load
        dataset_config: Dataset configuration
        max_examples: Maximum examples to load
        min_doc_length: Minimum document length for cleaning
        remove_duplicates: Whether to remove duplicates
        train_ratio: Training set ratio
        val_ratio: Validation set ratio

    Returns:
        Pipeline results with stats

    Raises:
        HTTPException: If pipeline fails
    """
    try:
        logger.info(f"Starting data pipeline for {dataset_name}")

        # Load dataset
        loader = DatasetLoader(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            max_examples=max_examples,
        )
        dataset = loader.load()

        # For now, just return that pipeline started
        # Full implementation would process the entire pipeline
        return {
            "status": "started",
            "message": "Data pipeline execution started",
            "dataset_name": dataset_name,
            "config": dataset_config,
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Custom Dataset Upload Endpoints
# ============================================================================

# Module-level loader (uses default upload directory)
_custom_loader = CustomDatasetLoader()


MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a custom dataset file (JSON or JSONL).

    The file must contain records with either a "text" field or
    "instruction"/"output" fields. Text fields are sanitized
    (whitespace normalized) before storage. Maximum file size: 50 MB.

    Args:
        file: Uploaded file (JSON array or JSONL)

    Returns:
        Upload result with dataset_id, record_count, and format

    Raises:
        HTTPException 400: If file contains invalid JSON or unrecognized format
        HTTPException 413: If file exceeds size limit
    """
    import tempfile
    import os

    # Sanitize the filename suffix for temp file creation
    safe_suffix = ".json"
    if file.filename:
        _, ext = os.path.splitext(os.path.basename(file.filename))
        if ext.lower() in {".json", ".jsonl"}:
            safe_suffix = ext.lower()

    tmp_path: Optional[str] = None
    try:
        # Read with size limit
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
            )

        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=safe_suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Load and validate
        data = _custom_loader.load_from_file(tmp_path)

        if not _custom_loader.validate_format(data):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid dataset format. Each record must have a 'text' "
                    "field or 'instruction'/'output' fields."
                ),
            )

        # Sanitize text fields before saving
        data = _custom_loader.sanitize_records(data)

        fmt = _custom_loader.detect_format(data)
        dataset_id = _custom_loader.save_dataset(data, filename=file.filename or "upload.json")

        return {
            "dataset_id": dataset_id,
            "record_count": len(data),
            "format": fmt,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process file: {e}")
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)


@router.get("/datasets/uploaded")
async def list_uploaded_datasets() -> Dict[str, Any]:
    """List all uploaded custom datasets.

    Returns:
        Dictionary with a "datasets" key containing metadata list
    """
    datasets_list = _custom_loader.list_datasets()
    return {"datasets": datasets_list}


@router.get("/datasets/uploaded/{dataset_id}")
async def get_uploaded_dataset(dataset_id: str) -> Dict[str, Any]:
    """Get details of a specific uploaded dataset.

    Args:
        dataset_id: The unique dataset identifier

    Returns:
        Dataset metadata

    Raises:
        HTTPException 404: If dataset not found
    """
    details = _custom_loader.get_dataset(dataset_id)
    if details is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return details
