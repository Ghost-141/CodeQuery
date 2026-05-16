import json
import time
import logging
from typing import List, Dict

import requests
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE = "http://127.0.0.1:8000/api/v1"


def init_session_state():
    """Initialize Streamlit session state - no session history tracking."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexing_repo_ids" not in st.session_state:
        st.session_state.indexing_repo_ids = set()
    if "selected_repo_id" not in st.session_state:
        st.session_state.selected_repo_id = None
    if "last_poll_time" not in st.session_state:
        st.session_state.last_poll_time = 0


def status_emoji(status: str) -> str:
    """Return emoji for repo status."""
    return {
        "pending": "⏳",
        "cloning": "📥",
        "parsing": "🔍",
        "embedding": "🧠",
        "ready": "✅",
        "failed": "❌",
    }.get(status, "❓")


def render_citations(citations: List[Dict]):
    """Render citation links at the bottom of assistant messages."""
    if not citations:
        return
    st.markdown("---")
    st.markdown("📎 **Sources**")
    for cite in citations:
        file_path = cite.get("file") or cite.get("file_path") or "Unknown"
        start = cite.get("start_line") or "?"
        end = cite.get("end_line") or "?"
        st.markdown(f"- `{file_path}:{start}-{end}`")


def handle_polling():
    """Check for indexing status and refresh if needed."""
    if st.session_state.indexing_repo_ids:
        time.sleep(2)

        try:
            r = requests.get(f"{API_BASE}/repos", timeout=10)
            r.raise_for_status()
            repos = r.json()
        except Exception as e:
            logger.error(f"Failed to fetch repos: {e}")
            return

        finished_ids = set()

        for rid in list(st.session_state.indexing_repo_ids):
            repo_data = next((r for r in repos if r["id"] == rid), None)
            if repo_data:
                if repo_data["status"] in ["ready", "failed"]:
                    finished_ids.add(rid)

        st.session_state.indexing_repo_ids -= finished_ids
        st.rerun()


def main():
    st.set_page_config(
        page_title="Codebase Q&A Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("🤖 CodeQuery")

        # 1. Index New Repo
        with st.expander("➕ Index New Repository", expanded=False):
            with st.form("index_repo_form", clear_on_submit=True):
                repo_url = st.text_input(
                    "GitHub URL", placeholder="https://github.com/user/repo"
                )
                if st.form_submit_button("Start Indexing", use_container_width=True):
                    if repo_url:
                        try:
                            r = requests.post(
                                f"{API_BASE}/repos", json={"url": repo_url}, timeout=10
                            )
                            r.raise_for_status()
                            repo = r.json()
                            st.session_state.indexing_repo_ids.add(repo["id"])
                            st.success("Indexing started!")
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Failed to create repo: {e}")
                            st.error("Failed to start indexing.")

        st.divider()

        # 2. Repository List
        st.subheader("Repositories")
        try:
            r = requests.get(f"{API_BASE}/repos", timeout=10)
            r.raise_for_status()
            repos = r.json()
        except Exception as e:
            logger.error(f"Failed to fetch repos: {e}")
            repos = []
        if not repos:
            st.info("No repositories indexed yet.")
        else:
            for repo in repos:
                emoji = status_emoji(repo["status"])
                name = repo.get("name") or repo["url"].split("/")[-1]

                with st.container(border=True):
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        st.write(f"{emoji} **{name}**")
                    with col2:
                        if repo["status"] == "ready":
                            if st.button(
                                "💬", key=f"chat_btn_{repo['id']}", help="New Chat"
                            ):
                                st.session_state.selected_repo_id = repo["id"]
                                st.session_state.messages = []
                                st.rerun()

                    # Enhanced status display
                    if repo["status"] == "ready":
                        st.caption("✅ Indexed and ready")
                    elif repo["status"] == "failed":
                        st.error(f"Error: {repo.get('error_message', 'Unknown')}")
                    else:
                        st.info(f"Status: **{repo['status'].capitalize()}**...")
                        if repo["status"] in ["cloning", "parsing", "embedding"]:
                            st.progress(
                                {"cloning": 0.3, "parsing": 0.6, "embedding": 0.9}.get(
                                    repo["status"], 0.1
                                ),
                                text=f"Currently {repo['status']}...",
                            )
                        st.session_state.indexing_repo_ids.add(repo["id"])

        st.divider()

    # Main Area
    if not st.session_state.selected_repo_id:
        st.title("Welcome to CodeQuery Agent!")
        st.markdown("""
        Select a repository from the sidebar to start chatting.
        I can help you understand code, find bugs, and explain architectural patterns.
        """)

        ready_repos = [r for r in repos if r["status"] == "ready"]
        if ready_repos:
            st.subheader("Quick Start")
            cols = st.columns(min(len(ready_repos), 3))
            for i, repo in enumerate(ready_repos):
                with cols[i % 3]:
                    name = repo.get("name") or repo["url"].split("/")[-1]
                    if st.button(
                        f"Chat about {name}",
                        key=f"qs_{repo['id']}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_repo_id = repo["id"]
                        st.session_state.messages = []
                        st.rerun()

        handle_polling()
        return

    # Chat Interface
    selected_repo = next(
        (r for r in repos if r["id"] == st.session_state.selected_repo_id), None
    )
    if selected_repo:
        repo_name = selected_repo.get("name") or selected_repo["url"].split("/")[-1]
        st.header(f"Chat: {repo_name}")

    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_citations(msg.get("citations"))

    # Chat Input
    if prompt := st.chat_input("Ask a question about the code..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_content = ""
            reasoning_trace = []
            citations = []

            with st.status("Agent is working...", expanded=True) as status:
                url = f"{API_BASE}/chat?repo_id={st.session_state.selected_repo_id}"
                payload = {"message": prompt}

                try:
                    with requests.post(
                        url, json=payload, stream=True, timeout=120
                    ) as r:
                        if r.status_code != 200:
                            try:
                                error_detail = r.json().get("detail", "Unknown error")
                            except:
                                error_detail = r.text or "Unknown error"
                            logger.error(f"Chat stream failed: {error_detail}")
                            st.error(f"**Agent Error:** {error_detail}")
                            status.update(label="Error occurred", state="error")
                        else:
                            current_event = "message"
                            for line in r.iter_lines():
                                if not line:
                                    continue

                                decoded = line.decode("utf-8")

                                if decoded.startswith("event: "):
                                    current_event = decoded[7:].strip()
                                elif decoded.startswith("data: "):
                                    try:
                                        data = json.loads(decoded[6:])

                                        if current_event == "token":
                                            token = data.get("token", "")
                                            full_content += token
                                            response_placeholder.markdown(
                                                full_content + "▌"
                                            )
                                        elif current_event == "reasoning":
                                            reasoning_trace = data.get(
                                                "reasoning_trace", []
                                            )
                                            if reasoning_trace:
                                                latest_thought = reasoning_trace[
                                                    -1
                                                ].get("thought", "Thinking...")
                                                logger.info(
                                                    f"AGENT THOUGHT: {latest_thought}"
                                                )
                                                status.update(
                                                    label=f"Agent: {latest_thought}",
                                                    state="running",
                                                )
                                        elif current_event == "citations":
                                            citations = data.get("citations", [])
                                        elif current_event == "error":
                                            detail = (
                                                data.get("detail")
                                                if isinstance(data, dict)
                                                else str(data)
                                            )
                                            logger.error(f"AGENT ERROR: {detail}")
                                            st.error(f"**Agent Error:** {detail}")
                                            status.update(
                                                label="Error occurred", state="error"
                                            )
                                            break
                                        elif current_event == "done":
                                            logger.info(
                                                f"AGENT STREAM DONE. Final content length: {len(full_content)}"
                                            )
                                            status.update(
                                                label="Response complete",
                                                state="complete",
                                                expanded=False,
                                            )
                                            break
                                    except json.JSONDecodeError:
                                        logger.warning(
                                            f"Failed to decode SSE data: {decoded[6:]}"
                                        )
                except Exception as e:
                    logger.exception("Streaming error")
                    st.error(f"**Stream Error:** {str(e)}")
                    status.update(label="Error occurred", state="error")

                # Final render of components
                response_placeholder.markdown(full_content)
                render_citations(citations)

            # Save to state
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_content,
                    "reasoning_trace": reasoning_trace,
                    "citations": citations,
                }
            )
            st.rerun()

    handle_polling()


if __name__ == "__main__":
    main()
