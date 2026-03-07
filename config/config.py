import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Config:
    """Configuration for all model providers"""

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")

    # Anthropic
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Ollama
    OLLAMA_IP = os.getenv("OLLAMA_IP", "localhost")
    OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")

    @classmethod
    def ollama_url(cls) -> str:
        return f"http://{cls.OLLAMA_IP}:{cls.OLLAMA_PORT}"


# Model definitions with their providers
MODELS = {
    # OpenAI
    "gpt-3.5-turbo": "openai",
    "gpt-4": "openai",
    "gpt-4-turbo": "openai",
    "gpt-4o-mini": "openai",

    # Anthropic
    "claude-3-opus": "anthropic",
    "claude-3-sonnet": "anthropic",

    # Ollama
    "qwen2.5:3b": "ollama",
    "llama3.1:8b": "ollama",
}


def get_model_provider(model_name: str) -> str:
    """Get the provider for a given model"""
    return MODELS.get(model_name)
