from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM Provider: "groq" or "ollama"
    llm_provider: str = "groq"

    # Groq
    groq_api_key: str = ""

    # Ollama (local LLM)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama-3.1-70b-versatile"

    # Paths
    repos_dir: Path = Path.home() / ".codequery" / "repos"
    db_path: str = "sqlite:///./app.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
