from fastembed import TextEmbedding, SparseTextEmbedding
from typing import List, Dict, Any
from backend.core.config import settings
from backend.core.logger import setup_logger

logger = setup_logger(__name__)

_dense_model = None
_sparse_model = None


def get_dense_model():
    global _dense_model
    if _dense_model is None:
        try:
            logger.info(f"Loading dense model: {settings.embedding_model}")
            _dense_model = TextEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.error(f"Failed to load dense model: {e}")
            raise RuntimeError(f"Cannot load embedding model: {str(e)}")
    return _dense_model


def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        try:
            logger.info("Loading sparse model: BM25")
            _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        except Exception as e:
            logger.error(f"Failed to load sparse model: {e}")
            raise RuntimeError(f"Cannot load embedding model: {str(e)}")
    return _sparse_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate dense embeddings for a list of texts."""
    model = get_dense_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


def embed_text(text: str) -> List[float]:
    """Generate a dense embedding for a single text."""
    return embed_texts([text])[0]


def embed_sparse_texts(texts: List[str]) -> List[Dict[str, Any]]:
    """Generate sparse (BM25) embeddings for a list of texts."""
    model = get_sparse_model()
    embeddings = list(model.embed(texts))
    return [
        {"indices": e.indices.tolist(), "values": e.values.tolist()} for e in embeddings
    ]


def embed_sparse_text(text: str) -> Dict[str, Any]:
    """Generate a sparse (BM25) embedding for a single text."""
    return embed_sparse_texts([text])[0]
