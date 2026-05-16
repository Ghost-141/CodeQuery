import json
from typing import Type, Optional, List, Dict, Any
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from qdrant_client import models

from backend.core.qdrant_client import qdrant_client
from backend.core.logger import setup_logger
from backend.services.indexer.embedder import embed_text, embed_sparse_text
from backend.services.indexer.reranker import rerank_chunks

logger = setup_logger(__name__)


def search_code(query: str, collection_name: str, top_k: int = 10) -> str:
    """Hybrid search using dense + sparse vectors, then re-rank."""
    if not collection_name:
        return "Error: collection_name is required"

    top_k = min(top_k, 10)
    logger.info(f"Performing hybrid search for query: {query}")

    try:
        dense_vector = embed_text(query)
        sparse_dict = embed_sparse_text(query)
        sparse_vector = models.SparseVector(
            indices=sparse_dict["indices"], values=sparse_dict["values"]
        )

        fetch_limit = top_k * 3

        dense_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            using="dense",
            limit=fetch_limit,
            with_payload=True,
        ).points

        sparse_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=sparse_vector,
            using="sparse",
            limit=fetch_limit,
            with_payload=True,
        ).points

        # Collect all unique chunks from both searches (no fusion, just union)
        seen_ids = set()
        chunks = []

        for point in dense_results + sparse_results:
            if point.id in seen_ids:
                continue
            seen_ids.add(point.id)
            payload = point.payload or {}
            chunks.append(
                {
                    "file_path": payload.get("file_path", ""),
                    "start_line": payload.get("start_line", 0),
                    "end_line": payload.get("end_line", 0),
                    "node_type": payload.get("node_type", ""),
                    "name": payload.get("name", ""),
                    "score": getattr(point, "score", 0),
                    "content": payload.get("content", "")[:4000],
                    "parent_name": payload.get("parent_name", ""),
                    "hierarchy_path": payload.get("hierarchy_path", ""),
                }
            )

        # Re-rank with cross-encoder for relevance ordering
        chunks = rerank_chunks(query, chunks)

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}. Falling back to dense-only.")
        try:
            dense_vector = embed_text(query)
            response = qdrant_client.query_points(
                collection_name=collection_name,
                query=dense_vector,
                using="dense",
                limit=top_k,
                with_payload=True,
            )
            chunks = []
            for r in response.points:
                payload = r.payload or {}
                chunks.append(
                    {
                        "file_path": payload.get("file_path", ""),
                        "start_line": payload.get("start_line", 0),
                        "end_line": payload.get("end_line", 0),
                        "node_type": payload.get("node_type", ""),
                        "name": payload.get("name", ""),
                        "score": getattr(r, "score", 0),
                        "content": payload.get("content", "")[:4000],
                        "parent_name": payload.get("parent_name", ""),
                        "hierarchy_path": payload.get("hierarchy_path", ""),
                    }
                )
            chunks = rerank_chunks(query, chunks)
        except Exception as e2:
            logger.error(f"Fallback search also failed: {e2}")
            return "Error: Search failed"

    # Debug output
    print("\n" + "=" * 60)
    print(f"DEBUG: HYBRID CHUNKS FOR QUERY: '{query}'")
    for i, c in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(f"File: {c['file_path']} (lines {c['start_line']}-{c['end_line']})")
        print(f"Score: {c.get('rerank_score', c['score']):.4f}")
    print("=" * 60 + "\n")

    return json.dumps(chunks, indent=2)


class SearchCodeInput(BaseModel):
    query: str = Field(
        description="Semantic search query to find relevant code chunks."
    )
    top_k: int = Field(default=10, description="Number of results to return (max 10).")


class SearchCodeTool(BaseTool):
    name: str = "search_code"
    description: str = (
        "Semantic search over the indexed codebase using hybrid search (dense + sparse vectors). "
        "Returns matching chunks with file path and line numbers."
    )
    args_schema: Type[BaseModel] = SearchCodeInput

    # Internal state injected at instantiation
    collection_name: str

    def _run(self, query: str, top_k: int = 10) -> str:
        return search_code(
            query=query, collection_name=self.collection_name, top_k=top_k
        )
