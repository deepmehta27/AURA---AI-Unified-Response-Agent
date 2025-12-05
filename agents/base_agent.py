"""
Base Agent for AURA
Abstract class that all specialized agents inherit from
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config.settings import settings
from utils.logger import logger


class BaseAgent(ABC):
    """Abstract base class for all AURA agents"""
    
    def __init__(self, name: str, description: str):
        """
        Initialize base agent
        
        Args:
            name: Agent name
            description: Agent description/purpose
        """
        self.name = name
        self.description = description
        self.model = settings.get('llm.model_name', 'gpt-5-mini')
        self.max_tokens = settings.get('llm.max_tokens', 2000)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=settings.openai_api_key)
        
        logger.info(f"Initialized {self.name} agent with model: {self.model}")
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and return response
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Response dictionary
        """
        pass
    
    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        verbosity: str = "medium",
        reasoning_effort: str = "medium",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None  # Optional, only for non-reasoning models
    ) -> str:
        """
        Call OpenAI API with GPT-5 mini support
        
        Args:
            messages: List of message dictionaries
            response_format: Optional response format specification
            verbosity: Reasoning verbosity level ("low", "medium", "high")
            reasoning_effort: Amount of reasoning effort ("low", "medium", "high")
            max_tokens: Override default max tokens
            temperature: Temperature for non-reasoning models (ignored for GPT-5/o1/o3)
            
        Returns:
            Response text from OpenAI
        """
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "store": False
            }
            
            # GPT-5 and reasoning models (o1, o3) configuration
            if "gpt-5" in self.model.lower() or "o1" in self.model.lower() or "o3" in self.model.lower():
                # Use max_completion_tokens for reasoning models
                params["max_completion_tokens"] = max_tokens or self.max_tokens
                params["verbosity"] = verbosity
                params["reasoning_effort"] = reasoning_effort
                # Note: GPT-5/reasoning models don't support temperature
                
            else:
                # Older models (GPT-4, GPT-3.5) use max_tokens
                params["max_tokens"] = max_tokens or self.max_tokens
                # Only add temperature for non-reasoning models
                if temperature is not None:
                    params["temperature"] = temperature
            
            # Add response format if specified
            if response_format:
                params["response_format"] = response_format
            
            # Make API call
            response = self.client.chat.completions.create(**params)
            
            # Check finish reason for token limit issues
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                logger.warning(
                    f"{self.name}: Response truncated due to token limit. "
                    f"Consider increasing max_tokens (current: {max_tokens or self.max_tokens})"
                )
            
            result = response.choices[0].message.content
            
            logger.info(f"{self.name} processed request successfully with {self.model}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific OpenAI errors
            if "max_tokens" in error_msg.lower() or "token limit" in error_msg.lower():
                logger.error(f"{self.name}: Token limit exceeded. Try reducing input size or increasing max_tokens.")
                raise Exception(
                    "Response too long. Please try a shorter query or break it into multiple requests."
                )
            
            if "temperature" in error_msg.lower() and "gpt-5" in self.model.lower():
                logger.error(f"{self.name}: GPT-5 models don't support temperature parameter.")
                raise Exception("Model configuration error. Please check model parameters.")
            
            logger.error(f"Error in {self.name} OpenAI call: {error_msg}")
            raise
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt for the agent
        
        Returns:
            System prompt string
        """
        return f"""You are {self.name}, an AI agent specialized in {self.description}.

Your role is to provide accurate, helpful, and detailed responses based on your specialization.

Guidelines:
- Be precise and factual
- Cite sources when referencing documents
- If uncertain, acknowledge limitations
- Format responses clearly
- Focus on your area of expertise

Respond professionally and concisely."""
    
    def get_info(self) -> Dict[str, str]:
        """Get agent information"""
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "max_tokens": self.max_tokens
        }
