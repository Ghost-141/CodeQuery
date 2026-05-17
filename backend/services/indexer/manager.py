import os
import shutil
from pathlib import Path
import subprocess
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


def clone_repo(url: str, local_path: str) -> str:
    if os.path.exists(local_path):
        shutil.rmtree(local_path, ignore_errors=True)
    os.makedirs(local_path, exist_ok=True)

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "-b", "main", url, local_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
    except subprocess.CalledProcessError:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "-b", "master", url, local_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, local_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
    return local_path


def list_source_files(repo_path: str):
    repo_path = Path(repo_path)
    # Common source and documentation extensions
    allowed_extensions = {
        ".py",
        ".md",
        ".txt",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".yaml",
        ".yml",
        ".json",
    }

    for root, dirs, files in os.walk(repo_path):
        # Skip hidden dirs and common non-source dirs
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d
            not in {
                "__pycache__",
                "node_modules",
                "build",
                "dist",
                ".git",
                "venv",
                ".venv",
            }
        ]
        for f in files:
            path = Path(root) / f
            # Skip binary files by checking extension and a small buffer
            if path.suffix.lower() in allowed_extensions:
                if not _is_binary(str(path)):
                    yield str(path)


def _is_binary(file_path: str) -> bool:
    """Check if a file is binary by looking for null bytes in the first 1KB."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except PermissionError:
        logger.warning(f"Permission denied reading: {file_path}")
        return True
    except Exception as e:
        logger.warning(f"Error checking {file_path}: {e}")
        return True
