from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.api.v1 import v1_router
from backend.core.database import init_db

from backend.core.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting up Codebase Q&A Agent backend")
        init_db()
        logger.info("Backend startup complete")
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}")
        raise RuntimeError(f"Cannot start application")
    yield
    logger.info("Shutting down Codebase Q&A Agent backend")


app = FastAPI(title="Codebase Q&A Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)
