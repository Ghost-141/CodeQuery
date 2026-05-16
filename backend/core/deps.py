from typing import Generator

from sqlalchemy.orm import Session

from backend.core.database import SessionLocal
from backend.core.qdrant_client import qdrant_client


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session for FastAPI dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_qdrant():
    """Return Qdrant client for FastAPI dependency injection."""
    return qdrant_client
