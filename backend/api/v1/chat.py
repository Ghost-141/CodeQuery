import json
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from backend.agent.graph import build_graph
from backend.core.database import Repo
from backend.core.deps import get_db
from backend.core.logger import setup_logger
from backend.schemas.models import ChatRequest

router = APIRouter(prefix="/chat", tags=["chat"])
logger = setup_logger(__name__)


async def chat_stream(
    repo_id: str, message: str, db: Session, session_id: Optional[str] = None
) -> AsyncGenerator[dict, None]:

    repo = db.query(Repo).filter(Repo.id == repo_id).first()

    if not repo:
        yield {"event": "error", "data": json.dumps({"detail": "Repo not found"})}
        return

    if repo.status != "ready":
        yield {
            "event": "error",
            "data": json.dumps({"detail": f"Repo is not ready. Status: {repo.status}"}),
        }
        return

    graph = build_graph()
    
    # Use existing session_id for multi-turn, or create new one
    thread_id = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    is_new_thread = not session_id
    logger.info(f"Using thread_id={thread_id} (new={is_new_thread})")

    # Check if this thread already has state (multi-turn)
    existing_messages = []
    if session_id:
        try:
            existing_state = await graph.aget_state(config)
            if existing_state and existing_state.values:
                existing_messages = existing_state.values.get("messages", [])
                logger.info(f"Found existing thread with {len(existing_messages)} messages")
        except Exception:
            pass  # New thread, start fresh

    # Append new message to existing history
    all_messages = existing_messages + [HumanMessage(content=message)]
    
    state_input = {
        "messages": all_messages,
        "repo_id": repo.id,
        "repo_local_path": repo.local_path,
        "collection_name": repo.collection_name,
        "reasoning_enabled": True,
        "iterations": 0,
        "citations": [],
    }

    try:
        logger.info(
            f"Invoking graph for repo_id={repo_id} with {len(all_messages)} total messages"
        )

        final_state = await graph.ainvoke(state_input, config)

        logger.info(f"Graph execution complete for thread_id={thread_id}")
    except Exception as exc:
        logger.exception(f"Error in graph execution for repo_id={repo_id}")
        error_msg = str(exc)
        if "rate_limit" in error_msg.lower() or "too large" in error_msg.lower():
            error_msg = "Request too large. The context exceeded the model's limit. Try reducing input tokens."
        yield {"event": "error", "data": json.dumps({"detail": error_msg})}
        return

    messages = final_state.get("messages", [])

    logger.info(f"Received {len(messages)} messages from graph")

    if not messages:
        logger.warning(f"No messages returned from graph for repo_id={repo_id}")
        yield {
            "event": "error",
            "data": json.dumps({"detail": "No response generated."}),
        }
        return

    # Find the last AIMessage with actual content (skip tool-call-only messages)
    last_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
            last_msg = msg
            break

    if not last_msg:
        logger.warning(
            "No AIMessage with content found. Using last message as fallback."
        )
        last_msg = messages[-1]

    final_content = str(last_msg.content).strip()
    logger.info(f"FINAL AGENT OUTPUT: {final_content[:100]}...")

    final_metadata = (
        last_msg.response_metadata if hasattr(last_msg, "response_metadata") else {}
    )
    logger.info(f"Response metadata: {list(final_metadata.keys())}")

    if not final_content:
        final_content = "I processed your request but couldn't generate a response. Please try rephrasing your question."

    # Stream response word-by-word for UX
    words = final_content.split(" ")
    for i, word in enumerate(words):
        token = word + (" " if i < len(words) - 1 else "")
        yield {"event": "token", "data": json.dumps({"token": token})}

    # Emit citations
    if "citations" in final_metadata:
        yield {
            "event": "citations",
            "data": json.dumps({"citations": final_metadata["citations"]}),
        }

    yield {"event": "done", "data": json.dumps({"content": final_content, "thread_id": thread_id})}


@router.post("")
def chat(payload: ChatRequest, repo_id: str, db: Session = Depends(get_db)):
    repo = db.query(Repo).filter(Repo.id == repo_id).first()
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Repo not found"
        )

    async def event_generator():
        async for event in chat_stream(
            repo.id, payload.message, db, session_id=payload.session_id
        ):
            yield event

    return EventSourceResponse(event_generator())
