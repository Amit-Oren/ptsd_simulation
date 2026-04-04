import re
import importlib
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from config.config import Config, get_model_provider


def build_llm(model_name: str, temperature: float = 0.7):
    """Instantiate the right LLM class based on the model's provider."""
    provider = get_model_provider(model_name)
    if provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=SecretStr(Config.OPENAI_API_KEY or ""))
    if provider == "anthropic":
        return ChatAnthropic(model_name=model_name, temperature=temperature, api_key=SecretStr(Config.ANTHROPIC_API_KEY or ""), timeout=60, stop=None)
    if provider == "ollama":
        return ChatOllama(model=model_name, temperature=temperature, base_url=Config.ollama_url())
    raise ValueError(f"Unknown provider for model '{model_name}'. Add it to config/config.py.")


def load_agent_class(class_name: str):
    """Dynamically import an agent class by name from the agents package."""
    module_name = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    module = importlib.import_module(f"agents.{module_name}")
    return getattr(module, class_name)
