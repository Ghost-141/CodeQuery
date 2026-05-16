import json
import logging
from functools import lru_cache
from typing import Any, Dict, List, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

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

# Groq: 4K tool response, 6 message history (strict 6K TPM limit)
# Ollama: 20K tool response, 12 message history (large local context)
_is_ollama = getattr(settings, "llm_provider", "groq").lower() == "ollama"
MAX_TOOL_RESPONSE_LENGTH = 20000 if _is_ollama else 4000
MAX_MESSAGES = 12 if _is_ollama else 6

SYSTEM_PROMPT = """You are an expert codebase Q&A assistant. You answer questions about code architecture, functions, and file relationships using ONLY the indexed codebase.

TOOL CALLING RULES — CRITICAL:
- You MUST call tools using the tool_call mechanism. NEVER describe tool calls in your response text.
- If you need to search, call search_code. If you need to read a file, call read_file. Do NOT say "I will search" — just CALL the tool.
- You have MAXIMUM 3 tool rounds. Use them wisely: 1 search → 1-2 read_file → answer.

MULTI-TURN CONTEXT — FOLLOW-UP QUESTIONS:
- Check the conversation history BEFORE deciding to search.
- If the user uses pronouns like "it", "this", "that", "the class", "the function" — they are referring to the PREVIOUSLY DISCUSSED topic. Use the previous tool results and your previous answer. Do NOT search again.
- Example: If you just explained LGBMRanker and user asks "how to implement it?" → "it" means LGBMRanker. Use the read_file results already in history.
- Only search for COMPLETELY NEW topics not mentioned in the conversation.
- If unsure what "it" refers to, use your previous answer as context.

SEARCH RESULTS ARE TRUNCATED — THIS IS NORMAL:
- search_code returns the FIRST ~800 chars of each code chunk (only 5 chunks max).
- Your job: find the right file from search results, then call read_file to see the FULL implementation.
- If search results show the correct file path, you HAVE enough information to proceed. Call read_file immediately.

RESPONSE RULES:
1. NO HALLUCINATION: Do not invent code. Only describe what you see in the files.
2. STRUCTURED CITATIONS: Provide citations as file path + line numbers for every claim.
3. CONCISE & TECHNICAL: Be precise. Use technical terms correctly.
4. DEEP ANALYSIS: For architecture questions, explore multiple directories, read key files, trace relationships.

WHEN TO SAY "I don't have enough information":
- ONLY say this if search_code returns ZERO results (no matching files found).
- If search_code finds files but content looks incomplete, call read_file for the full file. Do NOT say "I don't have enough information".
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
