import json
from functools import lru_cache
from typing import Any, Dict, List, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from backend.agent.state import AgentState
from backend.agent.tools import (
    ListDirectoryTool,
    ReadFileTool,
    SearchCodeTool,
    SummarizeModuleTool,
)

from backend.core.llm import get_llm
from backend.core.config import settings
from backend.core.logger import setup_logger

logger = setup_logger(__name__)

_is_ollama = getattr(settings, "llm_provider", "groq").lower() == "ollama"
MAX_TOOL_RESPONSE_LENGTH = 20000 if _is_ollama else 8000
MAX_MESSAGES = 10 if _is_ollama else 5

SYSTEM_PROMPT = """You are an expert codebase Q&A assistant. You answer questions about code architecture, functions, and file relationships using ONLY the indexed codebase.

TOOL CALLING RULES — CRITICAL:
- You MUST call tools using the tool_call mechanism. NEVER describe tool calls in your response text.
- If you need to search, call search_code. If you need to read a file, call read_file. Do NOT say "I will search" — just CALL the tool.
- You have MAXIMUM 3 tool rounds. Use them wisely: 1 search → 1-2 read_file → answer.

MULTI-TURN CONTEXT:
- Check the conversation history BEFORE deciding to search. If previous tool results already contain the answer, answer directly WITHOUT searching again.
- Only search for NEW topics or when the user asks for something not covered in previous results.
- If the user asks a follow-up like "explain more" or "what about X", use existing context if possible, or search only for the new aspect.

RESPONSE RULES:
1. NO HALLUCINATION: If tools don't provide the answer, say "I don't have enough information." Do not invent code.
2. STRUCTURED CITATIONS: Provide citations as file path + line numbers for every claim.
3. CONCISE & TECHNICAL: Be precise. Use technical terms correctly.
4. DEEP ANALYSIS: For architecture questions, explore multiple directories, read key files, trace relationships.

IMPORTANT: SEARCH RESULTS ARE TRUNCATED
- search_code returns code snippets, but they may be truncated (first ~4000 chars only).
- If search shows only the class signature or docstring, use read_file to get the FULL implementation.
- NEVER say "I don't have enough information" if the search found the right file — just read it with read_file.
"""


@lru_cache(maxsize=128)
def get_tools_for_repo(repo_local_path: str, collection_name: str):
    """Instantiate tools with repo-specific context."""
    return [
        SearchCodeTool(collection_name=collection_name),
        ReadFileTool(repo_local_path=repo_local_path),
        ListDirectoryTool(repo_local_path=repo_local_path),
        SummarizeModuleTool(collection_name=collection_name),
    ]


def _build_reasoning_trace(
    response: AIMessage, state: AgentState
) -> List[Dict[str, Any]]:
    """Build or extend the reasoning trace for the current step."""
    if not state.get("reasoning_enabled", True):
        return []

    new_step = {
        "thought": (
            f"I am calling {len(response.tool_calls)} tools to gather information."
            if response.tool_calls
            else "I have enough information to answer."
        ),
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"]}
            for tc in (response.tool_calls or [])
        ],
    }

    trace = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.response_metadata:
            prev_trace = msg.response_metadata.get("reasoning_trace", [])
            if prev_trace:
                trace = list(prev_trace)
                break

    trace.append(new_step)
    return trace


def agent_node(state: AgentState):
    """Agent node: invoke LLM with tools and attach metadata."""
    tools = get_tools_for_repo(state["repo_local_path"], state["collection_name"])
    llm = get_llm().bind_tools(tools)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"][
        -MAX_MESSAGES:
    ]

    response = llm.invoke(messages)

    if not response.response_metadata:
        response.response_metadata = {}

    # In this new flow, citations are extracted during tool execution and stored in state
    response.response_metadata["citations"] = state.get("citations", [])
    response.response_metadata["reasoning_trace"] = _build_reasoning_trace(
        response, state
    )

    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Route to tools if last message has tool calls, else end.
    Max 3 tool rounds to keep response under 1 minute."""
    last_msg = state["messages"][-1]
    iterations = state.get("iterations", 0)

    if iterations >= 3:
        logger.warning(f"Max iterations ({iterations}) reached, forcing end.")
        return "end"

    return "tools" if isinstance(last_msg, AIMessage) and last_msg.tool_calls else "end"


def tool_node(state: AgentState):
    """Execute tool calls from the last AIMessage."""
    tools_list = get_tools_for_repo(state["repo_local_path"], state["collection_name"])
    tools_map = {t.name: t for t in tools_list}

    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {"messages": []}

    tool_messages = []
    new_citations = []

    for tc in last_msg.tool_calls:
        tool_obj = tools_map.get(tc["name"])
        if not tool_obj:
            content = f"Error: Tool '{tc['name']}' not found."
        else:
            try:
                # 1. Execute the tool
                raw_result = tool_obj.invoke(tc["args"])

                # 2. Extract citations BEFORE truncation
                if tc["name"] in ("search_code", "summarize_module"):
                    try:
                        data = json.loads(raw_result)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "file_path" in item:
                                    new_citations.append(
                                        {
                                            "file": item["file_path"],
                                            "start_line": item.get("start_line"),
                                            "end_line": item.get("end_line"),
                                        }
                                    )
                    except:
                        pass
                elif tc["name"] in ("read_file", "list_directory"):
                    path = tc["args"].get("path")
                    if path:
                        new_citations.append({"file": path})

                # 3. Safe truncation for the message history
                content = raw_result
                if isinstance(content, str) and len(content) > MAX_TOOL_RESPONSE_LENGTH:
                    content = content[:MAX_TOOL_RESPONSE_LENGTH] + "\n...[truncated]"

            except Exception as e:
                logger.error(f"Error executing tool {tc['name']}: {e}")
                content = f"Error: {e}"

        tool_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))

    return {"messages": tool_messages, "citations": new_citations, "iterations": 1}
