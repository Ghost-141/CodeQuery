import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.v1 import v1_router
from backend.core.config import settings
from backend.core.database import SessionLocal, init_db, Repo

from backend.core.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(title="Codebase Q&A Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)


@app.on_event("startup")
def on_startup():
    logger.info("Starting up Codebase Q&A Agent backend")
    init_db()
    logger.info("Backend startup complete")
