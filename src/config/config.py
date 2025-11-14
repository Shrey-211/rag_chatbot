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
    ollama_model: str = Field(default="llama3.2:1b", env="OLLAMA_MODEL")
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
    min_relevance_score: float = Field(default=0.0, env="MIN_RELEVANCE_SCORE")
    enable_reranking: bool = Field(default=False, env="ENABLE_RERANKING")

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
        yaml_config = load_yaml_config(config_path)

        # Extract values from YAML to override defaults
        # Only override if not set in environment variables
        config_overrides = {}
        
        # LLM configuration
        if yaml_config.get("llm"):
            llm = yaml_config["llm"]
            if not os.getenv("LLM_PROVIDER"):
                config_overrides["llm_provider"] = llm.get("provider", "ollama")
            
            if llm.get("ollama") and not os.getenv("OLLAMA_BASE_URL"):
                config_overrides["ollama_base_url"] = llm["ollama"].get("base_url", "http://localhost:11434")
            if llm.get("ollama") and not os.getenv("OLLAMA_MODEL"):
                config_overrides["ollama_model"] = llm["ollama"].get("model", "llama3.2:1b")
            
            if llm.get("openai") and not os.getenv("OPENAI_MODEL"):
                config_overrides["openai_model"] = llm["openai"].get("model", "gpt-3.5-turbo")
        
        # Embedding configuration
        if yaml_config.get("embedding"):
            emb = yaml_config["embedding"]
            if not os.getenv("EMBEDDING_PROVIDER"):
                config_overrides["embedding_provider"] = emb.get("provider", "local")
            
            if emb.get("local") and not os.getenv("EMBEDDING_MODEL"):
                config_overrides["embedding_model"] = emb["local"].get("model", "all-MiniLM-L6-v2")
            if emb.get("local") and not os.getenv("EMBEDDING_DEVICE"):
                config_overrides["embedding_device"] = emb["local"].get("device", "cpu")
        
        # Vector store configuration
        if yaml_config.get("vectorstore"):
            vs = yaml_config["vectorstore"]
            if not os.getenv("VECTORSTORE_PROVIDER"):
                config_overrides["vectorstore_provider"] = vs.get("provider", "chroma")
            
            if vs.get("chroma") and not os.getenv("VECTORSTORE_PERSIST_PATH"):
                config_overrides["vectorstore_persist_path"] = vs["chroma"].get("persist_directory", "./data/chroma")
            if vs.get("chroma") and not os.getenv("VECTORSTORE_COLLECTION_NAME"):
                config_overrides["vectorstore_collection_name"] = vs["chroma"].get("collection_name", "rag_documents")
        
        # Retrieval configuration
        if yaml_config.get("retrieval"):
            retr = yaml_config["retrieval"]
            if not os.getenv("CHUNK_SIZE"):
                config_overrides["chunk_size"] = retr.get("chunk_size", 500)
            if not os.getenv("CHUNK_OVERLAP"):
                config_overrides["chunk_overlap"] = retr.get("chunk_overlap", 50)
            if not os.getenv("TOP_K_RESULTS"):
                config_overrides["top_k_results"] = retr.get("top_k", 5)
            if not os.getenv("MIN_RELEVANCE_SCORE"):
                config_overrides["min_relevance_score"] = retr.get("min_relevance_score", 0.0)
            if not os.getenv("ENABLE_RERANKING"):
                config_overrides["enable_reranking"] = retr.get("enable_reranking", False)
        
        # API configuration
        if yaml_config.get("api"):
            api = yaml_config["api"]
            if not os.getenv("API_HOST"):
                config_overrides["api_host"] = api.get("host", "0.0.0.0")
            if not os.getenv("API_PORT"):
                config_overrides["api_port"] = api.get("port", 8000)
            if not os.getenv("API_RELOAD"):
                config_overrides["api_reload"] = api.get("reload", True)
        
        # Retry configuration
        if yaml_config.get("llm", {}).get("retry"):
            retry = yaml_config["llm"]["retry"]
            if not os.getenv("MAX_RETRIES"):
                config_overrides["max_retries"] = retry.get("max_attempts", 3)
            if not os.getenv("RETRY_DELAY"):
                config_overrides["retry_delay"] = retry.get("delay", 1.0)
            if not os.getenv("REQUEST_TIMEOUT"):
                config_overrides["request_timeout"] = retry.get("timeout", 30.0)

        # Create Config instance with YAML overrides (env vars still take precedence)
        _config_instance = Config(**config_overrides)

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

