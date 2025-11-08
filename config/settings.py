"""
Configuration loader for AURA
Loads settings from config.yaml and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings loader"""
    
    def __init__(self):
        self.config_path = Path(__file__).parent / "config.yaml"
        self.config = self._load_config()
        self._validate_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise Exception(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing configuration file: {e}")
    
    def _validate_env_vars(self):
        """Validate that required environment variables are set"""
        required_vars = [
            'OPENAI_API_KEY',
            'PINECONE_API_KEY',
            'PINECONE_ENVIRONMENT',
            'SUPABASE_URL',
            'SUPABASE_KEY',
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise Exception(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                f"Please set them in your .env file"
            )
    
    # API Keys
    @property
    def openai_api_key(self) -> str:
        return os.getenv('OPENAI_API_KEY')
    
    @property
    def pinecone_api_key(self) -> str:
        return os.getenv('PINECONE_API_KEY')
    
    @property
    def pinecone_environment(self) -> str:
        return os.getenv('PINECONE_ENVIRONMENT')
    
    @property
    def supabase_url(self) -> str:
        return os.getenv('SUPABASE_URL')
    
    @property
    def supabase_key(self) -> str:
        return os.getenv('SUPABASE_KEY')
    
    @property
    def supabase_db_password(self) -> str:
        return os.getenv('SUPABASE_DB_PASSWORD')
    
    # Configuration getters
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.config.get('llm', {})
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get specific agent configuration"""
        agents = self.config.get('agents', {})
        return agents.get(agent_name, {})
    
    def get_pinecone_config(self) -> Dict[str, Any]:
        """Get Pinecone configuration"""
        return self.config.get('pinecone', {})
    
    def get_supabase_config(self) -> Dict[str, Any]:
        """Get Supabase configuration"""
        return self.config.get('supabase', {})
    
    def get_rabbitmq_config(self) -> Dict[str, Any]:
        """Get RabbitMQ configuration"""
        return self.config.get('rabbitmq', {})

# Global settings instance
settings = Settings()
