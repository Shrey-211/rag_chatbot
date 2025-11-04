"""Configuration management using environment variables and YAML."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    """Application configuration.

    Loads from environment variables and optional YAML config file.
    Environment variables take precedence over YAML.
    """

    # LLM Configuration
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama2", env="OLLAMA_MODEL")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")

    # Embedding Configuration
    embedding_provider: str = Field(default="local", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")

    # Vector Store Configuration
    vectorstore_provider: str = Field(default="chroma", env="VECTORSTORE_PROVIDER")
    vectorstore_persist_path: str = Field(default="./data/chroma", env="VECTORSTORE_PERSIST_PATH")
    vectorstore_collection_name: str = Field(
        default="rag_documents", env="VECTORSTORE_COLLECTION_NAME"
    )

    # Retrieval Configuration
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")

    # Rate Limiting & Retry
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="RETRY_DELAY")
    request_timeout: float = Field(default=30.0, env="REQUEST_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


_config_instance: Optional[Config] = None
_yaml_config: Optional[Dict[str, Any]] = None


def load_yaml_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    global _yaml_config

    if _yaml_config is not None:
        return _yaml_config

    if not Path(config_path).exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Expand environment variables in YAML
        config_data = _expand_env_vars(config_data)

        _yaml_config = config_data
        logger.info(f"Loaded configuration from {config_path}")
        return config_data

    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}


def _expand_env_vars(config: Any) -> Any:
    """Recursively expand environment variables in config.

    Args:
        config: Configuration value (dict, list, str, etc.)

    Returns:
        Configuration with env vars expanded
    """
    if isinstance(config, dict):
        return {k: _expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Expand ${VAR} or ${VAR:default}
        if config.startswith("${") and config.endswith("}"):
            var_spec = config[2:-1]
            if ":" in var_spec:
                var_name, default = var_spec.split(":", 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(var_spec, config)
    return config


def get_config(config_path: str = "config.yaml") -> Config:
    """Get configuration singleton.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config instance
    """
    global _config_instance

    if _config_instance is None:
        # Load YAML first
        load_yaml_config(config_path)

        # Create Config instance (env vars override YAML)
        _config_instance = Config()

        logger.info(
            f"Configuration loaded: LLM={_config_instance.llm_provider}, "
            f"Embedding={_config_instance.embedding_provider}, "
            f"VectorStore={_config_instance.vectorstore_provider}"
        )

    return _config_instance


def get_yaml_config() -> Dict[str, Any]:
    """Get YAML configuration dictionary.

    Returns:
        YAML config dictionary
    """
    if _yaml_config is None:
        load_yaml_config()
    return _yaml_config or {}

