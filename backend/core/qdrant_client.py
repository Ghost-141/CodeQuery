from qdrant_client import QdrantClient
from tenacity import retry, stop_after_attempt, wait_exponential
from backend.core.config import settings
from backend.core.logger import setup_logger

logger = setup_logger(__name__)

_qdrant_client = None


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def _create_qdrant_client() -> QdrantClient:
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        client.get_collections()

        client.set_model("sentence-transformers/all-MiniLM-L6-v2")
        client.set_sparse_model("Qdrant/bm25")

        return client

    except Exception as e:
        raise RuntimeError(f"Cannot connect to Qdrant: {str(e)}") from e


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client

    if _qdrant_client is None:
        _qdrant_client = _create_qdrant_client()

    return _qdrant_client


# Initialize with same models as embedder for integrated query support if needed
qdrant_client = get_qdrant_client()
qdrant_client.set_model("sentence-transformers/all-MiniLM-L6-v2")
qdrant_client.set_sparse_model("Qdrant/bm25")
