"""
Image processing agent for handling image queries.
"""
from typing import Any, Dict, Optional
from .base_agent import BaseAgent
import logging
import base64
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageAgent(BaseAgent):
    """Agent specialized in processing images."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the image agent."""
        super().__init__("ImageAgent", config)
        self.supported_formats = self.config.get("supported_formats", ["jpg", "jpeg", "png", "gif", "webp"])
        self.max_size_mb = self.config.get("max_size_mb", 10)
    
    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Process image input.
        
        Args:
            input_data: Image file path, base64 string, or bytes
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing processed image results
        """
        if not self.validate_input(input_data):
            return {"error": "Invalid input data"}
        
        self.logger.info("Processing image")
        
        # Determine input type and process accordingly
        if isinstance(input_data, str):
            # Could be file path or base64
            if Path(input_data).exists():
                image_path = input_data
                self.logger.info(f"Processing image from file: {image_path}")
            else:
                # Assume base64
                self.logger.info("Processing image from base64 string")
        elif isinstance(input_data, bytes):
            self.logger.info("Processing image from bytes")
        else:
            return {"error": "Unsupported input type"}
        
        # Placeholder for actual image processing
        return {
            "agent": self.name,
            "status": "success",
            "message": "Image processed successfully",
        }
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate image input.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if input_data is None:
            return False
        
        if isinstance(input_data, str):
            # Check if it's a valid file path
            path = Path(input_data)
            if path.exists():
                # Check file extension
                ext = path.suffix[1:].lower()
                return ext in self.supported_formats
            # Assume it's base64 or URL (basic validation)
            return len(input_data) > 0
        
        if isinstance(input_data, bytes):
            return len(input_data) > 0
        
        return False

