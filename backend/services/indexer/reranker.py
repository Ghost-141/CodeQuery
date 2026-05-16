"""Fast re-ranker using exact-name boost only (no slow cross-encoder)."""

from typing import List, Dict, Any
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


def rerank_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fast re-rank chunks by exact name match boost.
    
    Skips slow cross-encoder (5-15s on CPU) and uses instant string matching.
    Cross-encoder is disabled to keep responses under 1 minute.
    
    Args:
        query: The user's search query
        chunks: List of chunk dicts with "name", "hierarchy_path" keys
    
    Returns:
        Chunks sorted with exact matches first
    """
    if not chunks:
        return chunks
    
    query_lower = query.lower().strip()
    query_parts = query_lower.replace("class", " ").replace("function", " ").replace("def", " ").split()
    
    boosted = []
    remaining = []
    
    for chunk in chunks:
        chunk_name = (chunk.get("name", "") or "").lower()
        hierarchy = (chunk.get("hierarchy_path", "") or "").lower()
        
        name_match = (
            query_lower == chunk_name or
            query_lower in hierarchy or
            any(part == chunk_name for part in query_parts if len(part) > 2)
        )
        
        if name_match:
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = 1.0
            chunk_copy["boost_reason"] = f"exact_name_match:{chunk.get('name')}"
            boosted.append(chunk_copy)
        else:
            remaining.append(chunk)
    
    # Sort remaining by original vector score (no cross-encoder)
    remaining.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    result = boosted + remaining
    
    logger.info(
        f"Fast re-ranked {len(result)} chunks for query '{query[:50]}...'. "
        f"Exact matches boosted: {len(boosted)}"
    )
    
    return result
