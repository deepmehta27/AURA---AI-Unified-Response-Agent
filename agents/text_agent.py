"""
Text processing agent for handling text queries.
"""
from typing import Any, Dict, Optional
from .base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class TextAgent(BaseAgent):
    """Agent specialized in processing text queries."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the text agent."""
        super().__init__("TextAgent", config)
        self.max_length = self.config.get("max_length", 10000)
    
    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Process text input.
        
        Args:
            input_data: Text string or dictionary with text field
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing processed text results
        """
        if not self.validate_input(input_data):
            return {"error": "Invalid input data"}
        
        # Extract text from input
        if isinstance(input_data, dict):
            text = input_data.get("text", "")
        else:
            text = str(input_data)
        
        # Process text (placeholder for actual implementation)
        self.logger.info(f"Processing text of length {len(text)}")
        
        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length]
            self.logger.warning(f"Text truncated to {self.max_length} characters")
        
        return {
            "agent": self.name,
            "input_length": len(text),
            "processed_text": text,
            "status": "success",
        }
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate text input.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if input_data is None:
            return False
        
        if isinstance(input_data, dict):
            return "text" in input_data and bool(input_data["text"])
        
        return bool(str(input_data).strip())

