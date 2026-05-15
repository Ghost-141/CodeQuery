import json
from functools import lru_cache
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from backend.agent.state import AgentState
from backend.agent.tools import list_directory, read_file, search_code, summarize_module
from backend.core.config import settings

MAX_TOOL_RESPONSE_LENGTH = 1000
MAX_MESSAGES = 6

SYSTEM_PROMPT = """You are an expert codebase Q&A assistant. You help users understand codebases by answering questions about architecture, function behavior, file relationships, and usage examples.

CORE RULES:
1. SEARCH BEFORE SPEAKING: Always use search_code or list_directory to gather evidence before answering.
2. NO HALLUCINATION: If the tools don't provide the answer, say "I don't have enough information." Do not invent code or behavior.
3. STRUCTURED CITATIONS: You MUST provide citations for your claims. A citation consists of a file path and, if possible, line numbers.
4. OUT-OF-SCOPE: If a question is not about the indexed codebase, politely decline to answer.
5. CONCISE & TECHNICAL: Be precise. Use technical terms correctly.
6. DEEP ANALYSIS: For architecture questions, explore multiple directories, read key files, and trace relationships between components. Don't just list files - explain how they work together.

Tool usage guidelines:
- Use list_directory to understand the project structure. Explore nested directories when relevant.
- Use search_code for semantic search and finding definitions. Search for patterns like "class", "def", "import" to find key components.
- Use read_file to examine specific implementation details. Read entire files when understanding architecture.
- Use summarize_module for high-level overviews of classes or modules.
- For architecture questions: Start with root directory, identify main packages, trace imports, find entry points, and map component relationships.
"""


def get_llm():
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
    )


@lru_cache(maxsize=128)
def _get_tools(repo_local_path: str, collection_name: str):
    """Bind repo_local_path and collection_name to tools (cached per repo)."""
    return [
        t.bind(repo_local_path=repo_local_path, collection_name=collection_name)
        for t in (list_directory, read_file, search_code, summarize_module)
    ]


def _extract_citations(state: AgentState) -> list[dict]:
    """Extract file citations from ToolMessages in the conversation state."""
    citations = []
    seen = set()

    # Build a map of tool_call_id -> (tool_name, args) from AIMessages
    tool_call_map = {}
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_call_map[tc["id"]] = (tc["name"], tc.get("args", {}))

    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            continue

        # Try to extract from search_code JSON results
        try:
            for item in json.loads(msg.content):
                if isinstance(item, dict) and "file_path" in item:
                    key = f"{item['file_path']}:{item.get('start_line')}-{item.get('end_line')}"
                    if key not in seen:
                        citations.append({
                            "file": item["file_path"],
                            "start_line": item.get("start_line"),
                            "end_line": item.get("end_line"),
                        })
                        seen.add(key)
        except (json.JSONDecodeError, TypeError):
            pass

        # For read_file/list_directory, extract path from tool call args
        if msg.tool_call_id and msg.tool_call_id in tool_call_map:
            name, args = tool_call_map[msg.tool_call_id]
            path = args.get("path")
            if path and name in ("read_file", "list_directory"):
                key = f"{path}:{name}"
                if key not in seen:
                    citations.append({"file": path})
                    seen.add(key)

    return citations


def _build_reasoning_trace(response, state: AgentState) -> list[dict]:
    """Build reasoning trace from response and previous messages."""
    if not state.get("reasoning_enabled", True):
        return []

    has_tool_calls = getattr(response, "tool_calls", None)

    step = {
        "thought": (
            f"I need to use {len(response.tool_calls)} tool(s) to gather more information."
            if has_tool_calls
            else "I have enough information to provide a final answer."
        ),
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"]}
            for tc in (response.tool_calls or [])
        ] if has_tool_calls else [],
    }

    # Collect traces from previous AI messages
    trace = [
        step
        for msg in state["messages"]
        if isinstance(msg, AIMessage)
        for step in (msg.response_metadata or {}).get("reasoning_trace", [])
    ]
    trace.append(step)

    return trace


def agent_node(state: AgentState):
    """Agent node: invoke LLM with tools and attach metadata."""
    tools = _get_tools(state["repo_local_path"], state["collection_name"])
    llm = get_llm().bind_tools(tools)

    # Keep only recent messages to avoid context explosion
    recent = state["messages"][-MAX_MESSAGES:] if len(state["messages"]) > MAX_MESSAGES else state["messages"]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + recent

    response = llm.invoke(messages)

    # Ensure response_metadata exists
    if not getattr(response, "response_metadata", None):
        response.response_metadata = {}

    # Attach citations and reasoning trace
    response.response_metadata["citations"] = _extract_citations(state)
    response.response_metadata["reasoning_trace"] = _build_reasoning_trace(response, state)

    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Route to tools if last message has tool calls, else end."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def tool_node(state: AgentState):
    """Execute tool calls from the last AIMessage."""
    tools = {t.name: t for t in _get_tools(state["repo_local_path"], state["collection_name"])}
    last = state["messages"][-1]

    if not getattr(last, "tool_calls", None):
        return {"messages": []}

    tool_messages = []
    for tc in last.tool_calls:
        name = tc["name"]
        result = tools[name].invoke(tc["args"]) if name in tools else f"Error: Tool {name} not found."
        if isinstance(result, str) and len(result) > MAX_TOOL_RESPONSE_LENGTH:
            result = result[:MAX_TOOL_RESPONSE_LENGTH] + "\n...[truncated]"
        tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    return {"messages": tool_messages}
