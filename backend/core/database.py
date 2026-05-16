import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.core.config import settings

Base = declarative_base()

engine = create_engine(
    settings.db_path,
    connect_args=(
        {"check_same_thread": False} if settings.db_path.startswith("sqlite") else {}
    ),
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Repo(Base):
    __tablename__ = "repos"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    url = Column(String, nullable=False)
    name = Column(String)
    status = Column(String, default="pending")
    error_message = Column(Text, nullable=True)
    collection_name = Column(String, nullable=False)
    local_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    if settings.db_path.startswith("sqlite"):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def init_db():
    Base.metadata.create_all(bind=engine)
