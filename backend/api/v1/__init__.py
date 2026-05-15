from fastapi import APIRouter

from backend.api.v1.repos import router as repos_router
from backend.api.v1.sessions import router as sessions_router
from backend.api.v1.chat import router as chat_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(repos_router)
v1_router.include_router(sessions_router)
v1_router.include_router(chat_router)
