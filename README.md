# Codebase Q&A Agent

An intelligent agent that answers natural language questions about any public GitHub repository. Built with FastAPI, LangGraph, Qdrant, Groq, and Streamlit.


## Features

- **Dynamic Repo Indexing**: Paste any GitHub URL in the UI and the agent clones, parses, embeds, and indexes it automatically.
- **Structure-Aware Chunking**: Uses Tree-sitter to split code at function/class boundaries.
- **Agentic Q&A**: LangGraph-powered agent dynamically chooses between search, read, list, and summarize tools.
- **Streaming Responses**: Real-time token streaming with source citations.
- **Persistent Memory**: SQLite stores sessions; LangGraph `AsyncSqliteSaver` checkpoints agent state for true multi-turn memory.



## Tech Stack

| Component | Choice |
|-----------|--------|
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | `all-MiniLM-L6-v2` (local) |
| Vector DB | Qdrant (local Docker) |
| Agent Framework | LangGraph + AsyncSqliteSaver |
| Backend | FastAPI (port 8000) |
| Frontend | Streamlit (port 8501) |
| Code Parser | Tree-sitter |



## Project Setup

### Prerequisites

- python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- Docker (for Qdrant Vector Database)
- [Groq API key](https://console.groq.com).

### Installation

1. Clone the respository
    ```bash
    git clone https://github.com/Ghost-141/CodeQuery.git
    cd CodeQuery 
    ```

2. Create & Activate Environment

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate     # Windows
    pip install uv
    uv sync
    ```


### Environment Configuration:

Create a `.env` file in the root directory. Paste the followings in that file:

```bash
# Groq API Key - get one at https://console.groq.com
GROQ_API_KEY=your_groq_api_key

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6335

# FastAPI
API_HOST=0.0.0.0
API_PORT=8000

# Models
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama-3.1-8b-instant
```

### Run the Project

- Run backend
    ```bash
    rav run backend
    ```

- Run Qdrant Vector Database
    ```bash
    rav run db
    ```
- Run frontend
    ```bash
    rav run frontend
    ```

Qdrant Vector Database will be available at `http://localhost:6333`.

The UI will open at `http://localhost:8501`.


## Project Structure

```
├── backend
│   ├── agent
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── tools
│   │       ├── list_directory.py
│   │       ├── read_file.py
│   │       ├── search_code.py
│   │       └── summarize_module.py
│   ├── api
│   │   └── v1
│   │       ├── chat.py
│   │       └── repos.py
│   ├── core
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── deps.py
│   │   ├── logger.py
│   │   └── qdrant_client.py
│   ├── main.py
│   ├── schemas
│   │   └── models.py
│   └── services
│       ├── indexer
│       │   ├── chunker.py
│       │   ├── embedder.py
│       │   ├── manager.py
│       │   └── worker.py
│       └── repo_service.py
├── docker-compose.yml
├── frontend
│   ├── app.py
├── LICENSE
├── pyproject.toml
├── rav.yaml
├── README.md
└── uv.lock
```


## AI Usage

This project was developed with AI assistance for:
- **Code Generation**: Boilerplate code, API endpoints, and utility functions
- **Documentation**: README and inline code comments

All AI-generated code was reviewed, tested, and integrated manually to match with project requiremets.