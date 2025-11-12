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
        self.model = settings.get('llm.model_name', 'gpt-5-mini')  # Default to GPT-5 mini
        self.max_tokens = settings.get('llm.max_tokens', 2000)  # Increased for GPT-5
        
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
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call OpenAI API with GPT-5 mini support
        
        Args:
            messages: List of message dictionaries
            response_format: Optional response format specification
            verbosity: Reasoning verbosity level ("low", "medium", "high")
            reasoning_effort: Amount of reasoning effort ("low", "medium", "high")
            max_tokens: Override default max tokens
            
        Returns:
            Response text from OpenAI
        """
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "store": False
            }
            
            # GPT-5 and reasoning models use max_completion_tokens instead of max_tokens
            if "gpt-5" in self.model or "o1" in self.model or "o3" in self.model:
                params["max_completion_tokens"] = max_tokens or self.max_tokens
                params["verbosity"] = verbosity
                params["reasoning_effort"] = reasoning_effort
            else:
                # Older models use max_tokens
                params["max_tokens"] = max_tokens or self.max_tokens
            
            if response_format:
                params["response_format"] = response_format
            
            response = self.client.chat.completions.create(**params)
            
            result = response.choices[0].message.content
            
            logger.info(f"{self.name} processed request successfully with {self.model}")
            return result
            
        except Exception as e:
            logger.error(f"Error in {self.name} OpenAI call: {str(e)}")
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
