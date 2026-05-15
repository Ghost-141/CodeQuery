from langgraph.graph import END, StateGraph
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

from backend.agent.nodes import agent_node, should_continue, tool_node
from backend.agent.state import AgentState
from backend.core.config import settings

memory = AsyncSqliteSaver.from_conn_string(settings.db_path.replace("sqlite:///", ""))


def build_graph():
    """Build and compile the agent workflow graph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=memory)
