"""OpenAI LLM adapter implementation."""

import logging
from typing import Any, Callable, Dict

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.adapters.llm.base import LLMAdapter, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI Chat Completions API.

    Supports GPT-3.5, GPT-4, and other OpenAI models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        timeout: float = 30.0,
        max_retries: int = 3,
        **default_params,
    ):
        """Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., gpt-3.5-turbo, gpt-4)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **default_params: Default generation parameters
        """
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_params = default_params
        self.client = OpenAI(api_key=api_key, timeout=timeout)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated text

        Raises:
            openai.OpenAIError: If API request fails
        """
        params = {**self.default_params, **kwargs}

        messages = params.get("messages")
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1024),
                top_p=params.get("top_p", 1.0),
                frequency_penalty=params.get("frequency_penalty", 0.0),
                presence_penalty=params.get("presence_penalty", 0.0),
            )

            return LLMResponse(
                text=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                },
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def stream_generate(
        self, prompt: str, callback: Callable[[str], None], **kwargs
    ) -> LLMResponse:
        """Generate streaming response using OpenAI.

        Args:
            prompt: Input prompt
            callback: Function to call with each token
            **kwargs: Generation parameters

        Returns:
            LLMResponse with complete text
        """
        params = {**self.default_params, **kwargs}

        messages = params.get("messages")
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        full_text = []
        model_name = self.model

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1024),
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_text.append(content)
                    callback(content)
                if chunk.model:
                    model_name = chunk.model

            complete_text = "".join(full_text)

            # Note: Token usage is not available in streaming mode
            # We could estimate it, but for now we return 0
            return LLMResponse(
                text=complete_text,
                model=model_name,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={"streaming": True},
            )
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
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
        """Check OpenAI service health.

        Returns:
            True if service is healthy
        """
        try:
            # Simple test with minimal token usage
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": "hi"}], max_tokens=5
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information.

        Returns:
            Model information dictionary
        """
        return {
            "model": self.model,
            "provider": "openai",
            "supports_streaming": True,
        }

