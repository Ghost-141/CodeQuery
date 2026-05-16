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
- You have EXACTLY 3 tool rounds total. Round 1: search. Round 2: read files or search again. Round 3: ANSWER with text, do NOT call tools.
- On round 3, your response MUST be plain text answering the user. If you call a tool on round 3, the system will discard it.

MULTI-TURN CONTEXT:
- Check the conversation history BEFORE deciding to search. If previous tool results already contain the answer, answer directly WITHOUT searching again.
- Only search for NEW topics or when the user asks for something not covered in previous results.
- If the user asks a follow-up like "explain more" or "what about X", use existing context if possible, or search only for the new aspect.

RESPONSE RULES:
1. NO HALLUCINATION: If tools don't provide the answer, say "I don't have enough information." Do not invent code.
2. NEVER USE PLACEHOLDER NAMES: Do not use generic names like "TestClassName", "SomeClass", "ExampleFunction", or "MyModule". Only use exact class names, function names, and file paths found in the tool results.
3. STRUCTURED CITATIONS: Provide citations as file path + line numbers for every claim.
4. CONCISE & TECHNICAL: Be precise. Use technical terms correctly.
5. DEEP ANALYSIS: For architecture questions, explore multiple directories, read key files, trace relationships.

FINDING RELATIONSHIPS AND USAGE EXAMPLES:
- To find WHO CALLS a function: search for "function_name(" including the parenthesis. This matches call sites because function calls include "(". Example: search_code("fit(") finds all callers of fit().
- To find WHAT A FUNCTION CALLS: read the function's implementation with read_file and trace its internal calls.
- To find CLASS INHERITANCE: child classes are defined as "class Child(ParentClass):". Search for "(ParentClass)" to find all subclasses.
- To find USAGE EXAMPLES: search for "ClassName(" or "function_name(" — call sites in test files and documentation are natural usage examples. Combine with read_file on test files.
- To trace IMPORT RELATIONSHIPS: search for "from .module import" or "import module_name".
- For relationship queries, CALL search_code MULTIPLE TIMES with different patterns, then read the most relevant files.

IMPORTANT: SEARCH RESULTS ARE TRUNCATED
- search_code returns code snippets, but they may be truncated (first ~4000 chars only).
- If search shows only the class signature or docstring, use read_file to get the FULL implementation.
- NEVER say "I don't have enough information" if the search found the right file — just read it with read_file.

MANDATORY WORKFLOW — FOLLOW EXACTLY:
When you receive search_code results, you MUST:
1. Read the "name" and "content" fields in EACH search result.
2. Use ONLY the exact names that appear in those fields. If the class is named "TestTqdmLoggingHandler", you MUST write "TestTqdmLoggingHandler", never "TestClassName".
3. If a search result shows a file_path, that file EXISTS. NEVER claim it does not exist. If you need details, call read_file on that exact path.
4. If the content is truncated and you cannot see the full name, call read_file on the file_path before answering.
5. Do NOT describe what you WOULD do — actually call read_file if you need more information.

EXAMPLES OF BAD vs GOOD OUTPUT:
BAD: "The TestClassName class in tests/tests_contrib_logging.py has methods..."
GOOD: "The TestTqdmLoggingHandler class in tests/tests_contrib_logging.py:12 has methods test_should_call_tqdm_write and test_should_call_handle_error_if_exception_was_thrown (tests/tests_contrib_logging.py:15-28)."

BAD: "The file does not exist, so I cannot read it."
GOOD: [Calls read_file("tests/tests_contrib_logging.py") to get the exact content.]
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
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"][
        -MAX_MESSAGES:
    ]

    current_iter = state.get("iterations", 0)
    if current_iter >= 2:
        logger.info(f"Iteration {current_iter}: reminding model to answer")
        messages.append(
            SystemMessage(
                content="REMINDER: You are on your FINAL tool round. After this, you MUST provide a text answer. Do NOT call another tool."
            )
        )

    try:
        llm = get_llm().bind_tools(tools)
        response = llm.invoke(messages)
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        response = AIMessage(content=f"Error invoking LLM: {e}")

    if not response.response_metadata:
        response.response_metadata = {}

    # citations are extracted during tool execution and stored in state
    state_citations = state.get("citations", [])
    response.response_metadata["citations"] = state_citations
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
                if tc["name"] == "search_code":
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
                    except json.JSONDecodeError:
                        pass  # search_code error strings are not JSON, that's fine
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

    logger.info(f"tool_node extracted {len(new_citations)} citations")
    return {"messages": tool_messages, "citations": new_citations, "iterations": 1}
