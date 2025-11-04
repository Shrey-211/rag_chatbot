"""Ollama LLM adapter implementation."""

import logging
from typing import Any, Callable, Dict, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.adapters.llm.base import LLMAdapter, LLMResponse

logger = logging.getLogger(__name__)


class OllamaAdapter(LLMAdapter):
    """Adapter for Ollama local LLM API.

    Supports both local and Docker-based Ollama deployments.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        timeout: float = 30.0,
        max_retries: int = 3,
        **default_params,
    ):
        """Initialize Ollama adapter.

        Args:
            base_url: Ollama API base URL
            model: Model name to use
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **default_params: Default generation parameters
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_params = default_params
        self.client = httpx.Client(timeout=timeout)
        
        # Validate model exists on initialization
        self._validate_model()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated text

        Raises:
            httpx.HTTPError: If API request fails
        """
        params = {**self.default_params, **kwargs}

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.get("temperature", 0.7),
                "num_predict": params.get("max_tokens", 1024),
                "top_p": params.get("top_p", 0.9),
                "top_k": params.get("top_k", 40),
            },
        }

        try:
            response = self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                text=data.get("response", ""),
                model=self.model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0)
                    + data.get("eval_count", 0),
                },
                metadata={
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "eval_duration": data.get("eval_duration"),
                },
            )
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def stream_generate(
        self, prompt: str, callback: Callable[[str], None], **kwargs
    ) -> LLMResponse:
        """Generate streaming response using Ollama.

        Args:
            prompt: Input prompt
            callback: Function to call with each token
            **kwargs: Generation parameters

        Returns:
            LLMResponse with complete text
        """
        params = {**self.default_params, **kwargs}

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": params.get("temperature", 0.7),
                "num_predict": params.get("max_tokens", 1024),
            },
        }

        full_text = []
        total_metadata = {}

        try:
            with self.client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        import json

                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            full_text.append(chunk)
                            callback(chunk)

                        # Capture metadata from the final chunk
                        if data.get("done", False):
                            total_metadata = {
                                "prompt_eval_count": data.get("prompt_eval_count", 0),
                                "eval_count": data.get("eval_count", 0),
                                "total_duration": data.get("total_duration"),
                            }

            complete_text = "".join(full_text)
            return LLMResponse(
                text=complete_text,
                model=self.model,
                usage={
                    "prompt_tokens": total_metadata.get("prompt_eval_count", 0),
                    "completion_tokens": total_metadata.get("eval_count", 0),
                    "total_tokens": total_metadata.get("prompt_eval_count", 0)
                    + total_metadata.get("eval_count", 0),
                },
                metadata=total_metadata,
            )
        except httpx.HTTPError as e:
            logger.error(f"Ollama streaming error: {e}")
            raise

    def token_usage(self, response: LLMResponse) -> Dict[str, int]:
        """Extract token usage from response.

        Args:
            response: LLM response

        Returns:
            Token usage dictionary
        """
        return response.usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def health_check(self) -> bool:
        """Check Ollama service health.

        Returns:
            True if service is healthy
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information.

        Returns:
            Model information dictionary
        """
        try:
            response = self.client.post(
                f"{self.base_url}/api/show", json={"name": self.model}
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")

        return {"model": self.model, "base_url": self.base_url}

    def _validate_model(self) -> None:
        """Validate that the specified model exists in Ollama.
        
        Raises:
            ValueError: If model is not found or Ollama is not accessible
        """
        try:
            # Get list of available models
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            available_models = [m["name"] for m in data.get("models", [])]
            
            if not available_models:
                raise ValueError(
                    f"No models found in Ollama at {self.base_url}. "
                    f"Please pull a model first:\n"
                    f"  ollama pull {self.model}\n"
                    f"Or use a different LLM provider in config.yaml:\n"
                    f"  llm:\n"
                    f"    provider: 'mock'  # For testing\n"
                    f"    # or provider: 'openai' with your API key"
                )
            
            # Check if requested model exists
            model_found = any(
                self.model in model or model.startswith(self.model + ":")
                for model in available_models
            )
            
            if not model_found:
                available_list = "\n  ".join(available_models)
                raise ValueError(
                    f"Model '{self.model}' not found in Ollama.\n"
                    f"Available models:\n  {available_list}\n\n"
                    f"To fix this:\n"
                    f"  1. Pull the model: ollama pull {self.model}\n"
                    f"  2. Or use one of the available models above\n"
                    f"  3. Or switch to a different provider in config.yaml"
                )
            
            logger.info(f"Ollama model '{self.model}' validated successfully")
            
        except httpx.ConnectError:
            raise ValueError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Please ensure Ollama is running:\n"
                f"  - Check if Ollama is installed: ollama --version\n"
                f"  - Start Ollama if needed (it usually runs automatically)\n"
                f"  - Or use a different provider in config.yaml"
            )
        except httpx.HTTPError as e:
            if "404" not in str(e):
                raise ValueError(
                    f"Ollama API error at {self.base_url}: {e}\n"
                    f"Please check your Ollama installation."
                )

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()

