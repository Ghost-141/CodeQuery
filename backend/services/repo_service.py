import os
import shutil
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.logger import setup_logger
from backend.core.database import Repo
from backend.services.indexer.worker import run_indexing_task

logger = setup_logger(__name__)


class RepoService:
    def __init__(self, db: Session):
        self.db = db

    def create_repo(self, url: str) -> Repo:
        repo_id = str(uuid.uuid4())
        repo_name = url.rstrip("/").split("/")[-1]
        local_path = str(settings.repos_dir / repo_id)
        collection_name = f"repo_{repo_id}"

        try:
            repo = Repo(
                id=repo_id,
                url=url,
                name=repo_name,
                status="pending",
                collection_name=collection_name,
                local_path=local_path,
            )
            self.db.add(repo)
            self.db.commit()
            self.db.refresh(repo)

        except Exception as e:
            self.db.rollback()
            logger.exception("Failed to create repo in database")
            raise RuntimeError(f"Database error: {str(e)}")

        # Trigger background indexing
        run_indexing_task(repo_id, url, local_path, collection_name)

        return repo

    def get_repo(self, repo_id: str) -> Optional[Repo]:
        return self.db.query(Repo).filter(Repo.id == repo_id).first()

    def list_repos(self) -> list[Repo]:
        return self.db.query(Repo).order_by(Repo.created_at.desc()).all()

    def update_status(
        self, repo_id: str, status: str, error_message: Optional[str] = None
    ) -> Optional[Repo]:
        repo = self.get_repo(repo_id)
        if not repo:
            return None
        repo.status = status
        if error_message is not None:
            repo.error_message = error_message
        repo.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(repo)
        return repo

    def delete_repo(self, repo_id: str) -> bool:
        repo = self.get_repo(repo_id)
        if not repo:
            return False
        # Clean up local files

        if os.path.exists(repo.local_path):
            shutil.rmtree(repo.local_path, ignore_errors=True)
        self.db.delete(repo)
        self.db.commit()
        return True
