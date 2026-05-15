from qdrant_client import QdrantClient

from backend.core.config import settings

qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

# Initialize with same models as embedder for integrated query support if needed
qdrant_client.set_model("sentence-transformers/all-MiniLM-L6-v2")
qdrant_client.set_sparse_model("Qdrant/bm25")
