"""Mock LLM adapter for testing."""

import time
from typing import Any, Callable, Dict

from src.adapters.llm.base import LLMAdapter, LLMResponse


class MockLLMAdapter(LLMAdapter):
    """Mock LLM adapter for testing purposes.

    Returns predefined responses without making actual API calls.
    """

    def __init__(
        self,
        response_text: str = "This is a mock response.",
        delay: float = 0.1,
        fail_health_check: bool = False,
    ):
        """Initialize mock adapter.

        Args:
            response_text: Default response text
            delay: Simulated response delay in seconds
            fail_health_check: If True, health check will fail
        """
        self.response_text = response_text
        self.delay = delay
        self.fail_health_check = fail_health_check
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response.

        Args:
            prompt: Input prompt (logged but not used)
            **kwargs: Ignored

        Returns:
            Mock LLMResponse
        """
        time.sleep(self.delay)
        self.call_count += 1

        # Simple logic to make responses somewhat relevant
        if "?" in prompt:
            response_text = f"Mock answer to your question. {self.response_text}"
        else:
            response_text = self.response_text

        return LLMResponse(
            text=response_text,
            model="mock-model",
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split()),
            },
            metadata={"call_count": self.call_count, "prompt_length": len(prompt)},
        )

    def stream_generate(
        self, prompt: str, callback: Callable[[str], None], **kwargs
    ) -> LLMResponse:
        """Generate mock streaming response.

        Args:
            prompt: Input prompt
            callback: Function to call with each token
            **kwargs: Ignored

        Returns:
            Mock LLMResponse
        """
        time.sleep(self.delay / 10)
        self.call_count += 1

        # Simulate streaming by sending words one at a time
        words = self.response_text.split()
        for word in words:
            callback(word + " ")
            time.sleep(self.delay / len(words))

        return LLMResponse(
            text=self.response_text,
            model="mock-model",
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(words),
                "total_tokens": len(prompt.split()) + len(words),
            },
            metadata={
                "call_count": self.call_count,
                "streaming": True,
            },
        )

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
        """Check mock service health.

        Returns:
            False if fail_health_check is True, otherwise True
        """
        return not self.fail_health_check

    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information.

        Returns:
            Model information dictionary
        """
        return {
            "model": "mock-model",
            "provider": "mock",
            "call_count": self.call_count,
        }

