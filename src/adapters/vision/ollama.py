"""Ollama vision adapter for multimodal document processing."""

import base64
import logging
from pathlib import Path
from typing import Optional

import httpx

from src.adapters.vision.base import VisionAdapter, VisionResponse

logger = logging.getLogger(__name__)


class OllamaVisionAdapter(VisionAdapter):
    """Ollama vision adapter for processing images with vision models.
    
    Supports models like llama3.2-vision, llava, bakllava, etc.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2-vision:11b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 300.0,
    ):
        """Initialize Ollama vision adapter.
        
        Args:
            base_url: Ollama API base URL
            model: Vision model name (e.g., llama3.2-vision:11b, llava:7b)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds (default 300s for large images)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        logger.info(f"Initialized Ollama Vision Adapter: model={model}, base_url={base_url}")
    
    def analyze_image(self, image_path: str, prompt: str) -> VisionResponse:
        """Analyze an image with Ollama vision model.
        
        Args:
            image_path: Path to image file
            prompt: Text prompt for analysis
            
        Returns:
            VisionResponse with extracted information
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Read and encode image as base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            logger.info(f"Analyzing image {path.name} with {self.model}...")
            
            # Prepare request
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
            
            # Make request
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
            
            # Extract response text
            response_text = result.get("response", "")
            
            logger.info(f"✓ Vision analysis complete ({len(response_text)} chars)")
            
            return VisionResponse(
                text=response_text,
                model=self.model,
                metadata={
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0),
                    "total_duration": result.get("total_duration", 0),
                }
            )
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during vision analysis: {e}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Ollama vision service is available.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                # Check if vision model is available
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                if self.model in model_names:
                    return True
                else:
                    logger.warning(f"Vision model '{self.model}' not found in Ollama. Available models: {model_names}")
                    logger.warning(f"Pull the model with: ollama pull {self.model}")
                    return False
                    
        except Exception as e:
            logger.error(f"Vision health check failed: {e}")
            return False

