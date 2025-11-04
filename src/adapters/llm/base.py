"""Base interface for LLM adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class LLMResponse:
    """Standardized LLM response format."""

    text: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters.

    This interface ensures all LLM providers have consistent methods for
    generation, streaming, token usage tracking, and health checks.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters
                     (temperature, max_tokens, etc.)

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    def stream_generate(
        self, prompt: str, callback: Callable[[str], None], **kwargs
    ) -> LLMResponse:
        """Generate a streaming response from the LLM.

        Args:
            prompt: The input prompt
            callback: Function called with each token/chunk
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with full generated text and metadata

        Raises:
            Exception: If streaming fails
        """
        pass

    @abstractmethod
    def token_usage(self, response: LLMResponse) -> Dict[str, int]:
        """Extract token usage information from response.

        Args:
            response: The LLM response object

        Returns:
            Dictionary with token counts (prompt_tokens, completion_tokens, total_tokens)
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the LLM service is available and healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information (name, version, etc.)
        """
        pass

