"""Base interface for vision adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class VisionResponse:
    """Response from vision model."""
    
    text: str
    model: str
    metadata: Dict[str, Any]


class VisionAdapter(ABC):
    """Abstract base class for vision models that can process images."""
    
    @abstractmethod
    def analyze_image(self, image_path: str, prompt: str) -> VisionResponse:
        """Analyze an image with a text prompt.
        
        Args:
            image_path: Path to image file
            prompt: Text prompt for analysis
            
        Returns:
            VisionResponse with analysis results
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if vision service is available.
        
        Returns:
            True if healthy, False otherwise
        """
        pass

