from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.core.deps import get_db
from backend.schemas.models import RepoCreate, RepoResponse
from backend.services.repo_service import RepoService

router = APIRouter(prefix="/repos", tags=["repos"])


def get_repo_service(db: Session = Depends(get_db)) -> RepoService:
    return RepoService(db)


@router.post("", response_model=RepoResponse)
def create_repo(
    payload: RepoCreate, repo_service: RepoService = Depends(get_repo_service)
):
    repo = repo_service.create_repo(payload.url)
    return repo


@router.get("", response_model=list[RepoResponse])
def list_repos(repo_service: RepoService = Depends(get_repo_service)):
    return repo_service.list_repos()
