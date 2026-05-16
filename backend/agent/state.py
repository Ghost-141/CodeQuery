from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

def add_citations(left: list, right: list) -> list:
    """Reducer to append new citations and ensure uniqueness."""
    if not left:
        return right
    if not right:
        return left
    
    # Simple list append - deduping can happen in the UI or extraction
    return left + right

def add_iterations(left: int, right: int) -> int:
    """Reducer to count total iterations."""
    return left + right

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    repo_id: str
    repo_local_path: str
    collection_name: str
    reasoning_enabled: bool
    citations: Annotated[list[dict], add_citations]
    iterations: Annotated[int, add_iterations]
