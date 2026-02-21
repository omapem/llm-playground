"""Job persistence with SQLite using SQLAlchemy ORM.

Provides durable storage for training job records so that job history
survives server restarts. On startup, any jobs left in "running" state
are automatically marked as "failed" since the background threads that
were executing them no longer exist.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import String, Text, DateTime, create_engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger(__name__)


# ── SQLAlchemy ORM models ──────────────────────────────────────────────


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""
    pass


class JobRecord(Base):
    """Persistent record of a training job.

    Attributes:
        job_id: Unique identifier (primary key).
        config_json: Serialized TrainingConfig (JSON string).
        status: Current job status (pending, running, completed, failed, cancelled).
        created_at: Timestamp when the job was created.
        updated_at: Timestamp of the most recent status change.
        error_message: Error details when status is "failed".
        metrics_json: Serialized final metrics (JSON string).
    """

    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a plain dictionary.

        Returns:
            Dictionary with all job record fields.
        """
        return {
            "job_id": self.job_id,
            "config_json": self.config_json,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "error_message": self.error_message,
            "metrics_json": self.metrics_json,
        }


# ── JobDatabase ────────────────────────────────────────────────────────


class JobDatabase:
    """Thread-safe SQLite-backed storage for training job records.

    Args:
        db_path: Path to the SQLite database file.
            Defaults to ``./data/jobs.db``.
        mark_running_as_failed: If ``True``, any jobs with status "running"
            will be set to "failed" with message "Server restarted" when
            the database is opened. This handles recovery after an
            unclean server shutdown.
    """

    def __init__(
        self,
        db_path: str = "./data/jobs.db",
        mark_running_as_failed: bool = False,
    ) -> None:
        # Ensure parent directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        # Create engine with check_same_thread=False for multi-threaded access
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )

        # Create tables if they don't exist
        Base.metadata.create_all(self._engine)

        # Recovery: mark stale "running" jobs as failed
        if mark_running_as_failed:
            self._mark_running_as_failed()

    # ── public API ─────────────────────────────────────────────────────

    def save_job(
        self,
        job_id: str,
        config_json: str,
        status: str = "pending",
    ) -> None:
        """Persist a new job record.

        Args:
            job_id: Unique job identifier.
            config_json: Serialized TrainingConfig.
            status: Initial status (default ``"pending"``).

        Raises:
            IntegrityError: If a job with the same ``job_id`` already exists.
        """
        now = datetime.now(timezone.utc)
        record = JobRecord(
            job_id=job_id,
            config_json=config_json,
            status=status,
            created_at=now,
            updated_at=now,
        )
        with Session(self._engine) as session:
            session.add(record)
            session.commit()

    def load_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load a single job record by its ID.

        Args:
            job_id: The job identifier to look up.

        Returns:
            Dictionary of job fields, or ``None`` if not found.
        """
        with Session(self._engine) as session:
            record = session.get(JobRecord, job_id)
            if record is None:
                return None
            return record.to_dict()

    def update_job_status(
        self,
        job_id: str,
        status: str,
        error_message: Optional[str] = None,
        metrics_json: Optional[str] = None,
    ) -> None:
        """Update the status (and optionally error / metrics) of a job.

        Args:
            job_id: The job to update.
            status: New status value.
            error_message: Optional error details (typically for ``"failed"``).
            metrics_json: Optional serialized metrics (typically for ``"completed"``).
        """
        now = datetime.now(timezone.utc)
        values: Dict[str, Any] = {
            "status": status,
            "updated_at": now,
        }
        if error_message is not None:
            values["error_message"] = error_message
        if metrics_json is not None:
            values["metrics_json"] = metrics_json

        with Session(self._engine) as session:
            stmt = (
                update(JobRecord)
                .where(JobRecord.job_id == job_id)
                .values(**values)
            )
            result = session.execute(stmt)
            session.commit()
            if result.rowcount == 0:
                logger.warning(f"update_job_status: no record found for job_id={job_id}")

    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all job records, optionally filtered by status.

        Args:
            status: If provided, only return jobs with this status.

        Returns:
            List of job record dictionaries.
        """
        with Session(self._engine) as session:
            stmt = select(JobRecord)
            if status is not None:
                stmt = stmt.where(JobRecord.status == status)
            records = session.scalars(stmt).all()
            return [r.to_dict() for r in records]

    # ── private helpers ────────────────────────────────────────────────

    def _mark_running_as_failed(self) -> None:
        """Mark all "running" jobs as "failed" with a restart message.

        Called during startup to handle jobs that were interrupted by a
        server shutdown.
        """
        now = datetime.now(timezone.utc)
        with Session(self._engine) as session:
            stmt = (
                update(JobRecord)
                .where(JobRecord.status == "running")
                .values(
                    status="failed",
                    error_message="Server restarted",
                    updated_at=now,
                )
            )
            result = session.execute(stmt)
            session.commit()
            if result.rowcount > 0:
                logger.info(
                    f"Marked {result.rowcount} previously running job(s) as failed "
                    f"due to server restart."
                )
