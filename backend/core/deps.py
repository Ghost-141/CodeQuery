from typing import Generator

from backend.core.database import SessionLocal
from backend.core.qdrant_client import qdrant_client


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_qdrant():
    return qdrant_client
