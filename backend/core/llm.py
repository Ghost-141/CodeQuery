"""Unified LLM provider supporting Groq and Ollama backends."""

from langchain_core.language_models.chat_models import BaseChatModel
from backend.core.config import settings
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


def get_llm(temperature: float = 0.1) -> BaseChatModel:
    """
    Initialize the LLM based on the configured provider.

    Provider is selected via LLM_PROVIDER env var (default: 'groq').

    Groq: Uses ChatGroq with cloud API. Good for fast responses, limited context.
    Ollama: Uses ChatOllama with local models. Good for large context, privacy.

    Args:
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)

    Returns:
        A LangChain chat model instance ready for bind_tools()
    """
    provider = getattr(settings, "llm_provider", "groq").lower()

    if provider == "ollama":
        return _get_ollama_llm(temperature)
    elif provider == "groq":
        return _get_groq_llm(temperature)
    else:
        logger.warning(f"Unknown LLM provider '{provider}', falling back to groq")
        return _get_groq_llm(temperature)


def _get_ollama_llm(temperature: float) -> BaseChatModel:
    """Initialize ChatOllama with local model."""
    from langchain_ollama import ChatOllama

    model = getattr(settings, "ollama_model", "llama3.2:latest")
    base_url = getattr(settings, "ollama_base_url", "http://localhost:11434")

    logger.info(f"Using Ollama LLM: {model} at {base_url}")

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=8192,
    )


def _get_groq_llm(temperature: float) -> BaseChatModel:
    """Initialize ChatGroq with cloud API."""
    from langchain_groq import ChatGroq

    logger.info(f"Using Groq LLM: {settings.llm_model}")

    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
    )
