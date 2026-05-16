from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.core.deps import get_db
from backend.core.logger import setup_logger
from backend.core.qdrant_client import qdrant_client
from backend.schemas.models import RepoCreate, RepoResponse
from backend.services.repo_service import RepoService

router = APIRouter(prefix="/repos", tags=["repos"])

logger = setup_logger(__name__)


def get_repo_service(db: Session = Depends(get_db)) -> RepoService:
    return RepoService(db)


@router.post("", response_model=RepoResponse)
def create_repo(
    payload: RepoCreate, repo_service: RepoService = Depends(get_repo_service)
):
    try:
        if not payload.url or not payload.url.strip():
            raise HTTPException(status_code=400, detail="URL is required")

        # Validate URL format
        if not any(
            host in payload.url
            for host in [
                "github.com",
                "gitlab.com",
            ]
        ):
            raise HTTPException(
                status_code=400,
                detail="Only GitHub/GitLab URLs are supported",
            )

        repo = repo_service.create_repo(payload.url)
        if not repo:
            raise HTTPException(status_code=500, detail="Failed to create repository")

        return repo
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create repository")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("", response_model=list[RepoResponse])
def list_repos(repo_service: RepoService = Depends(get_repo_service)):
    repos = repo_service.list_repos()

    # Auto-cleanup: remove repos whose Qdrant collections no longer exist
    alive_repos = []
    for repo in repos:
        if repo.status == "ready":
            try:
                qdrant_client.get_collection(repo.collection_name)
                alive_repos.append(repo)
            except Exception:
                logger.warning(
                    f"Collection '{repo.collection_name}' missing for repo {repo.id} ({repo.name}). Auto-deleting."
                )
                repo_service.delete_repo(repo.id)
        else:
            alive_repos.append(repo)

    return alive_repos


@router.delete("/{repo_id}")
def delete_repo(repo_id: str, repo_service: RepoService = Depends(get_repo_service)):
    repo = repo_service.get_repo(repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repo not found")

    repo_service.delete_repo(repo_id)
    return {"detail": "Repo deleted"}
