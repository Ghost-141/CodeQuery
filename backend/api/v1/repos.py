from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.core.database import get_db_session
from backend.schemas.models import RepoCreate, RepoResponse
from backend.services.repo_service import RepoService

router = APIRouter(prefix="/repos", tags=["repos"])


def get_repo_service(db: Session = Depends(get_db_session)) -> RepoService:
    return RepoService(db)


@router.post("", response_model=RepoResponse)
def create_repo(payload: RepoCreate, repo_service: RepoService = Depends(get_repo_service)):
    repo = repo_service.create_repo(payload.url)
    return repo


@router.get("", response_model=list[RepoResponse])
def list_repos(repo_service: RepoService = Depends(get_repo_service)):
    return repo_service.list_repos()


@router.get("/{repo_id}", response_model=RepoResponse)
def get_repo(repo_id: str, repo_service: RepoService = Depends(get_repo_service)):
    repo = repo_service.get_repo(repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repo not found")
    return repo


@router.get("/{repo_id}/status")
def get_repo_status(repo_id: str, repo_service: RepoService = Depends(get_repo_service)):
    repo = repo_service.get_repo(repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repo not found")
    return {"repo_id": repo.id, "status": repo.status, "error_message": repo.error_message}


@router.delete("/{repo_id}")
def delete_repo(repo_id: str, repo_service: RepoService = Depends(get_repo_service)):
    deleted = repo_service.delete_repo(repo_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Repo not found")
    return {"detail": "Repo deleted"}
