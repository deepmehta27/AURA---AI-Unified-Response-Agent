"""
API Dependencies
Shared utilities, authentication, and agent management
"""

from fastapi import HTTPException, Header
from typing import Optional
import os

# Agent imports (lazy loaded when needed)
_text_agent = None
_image_agent = None
_audio_agent = None
_orchestrator = None

def get_text_agent():
    """Lazy load Text Agent"""
    global _text_agent
    if _text_agent is None:
        from agents.text_agent import text_agent
        _text_agent = text_agent
    return _text_agent

def get_image_agent():
    """Lazy load Image Agent"""
    global _image_agent
    if _image_agent is None:
        from agents.image_agent import image_agent
        _image_agent = image_agent
    return _image_agent

def get_audio_agent():
    """Lazy load Audio Agent"""
    global _audio_agent
    if _audio_agent is None:
        from agents.audio_agent import audio_agent
        _audio_agent = audio_agent
    return _audio_agent

def get_orchestrator():
    """Lazy load Orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        from agents.orchestrator import orchestrator
        _orchestrator = orchestrator
    return _orchestrator

# Optional: API Key Authentication
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Verify API key from environment variable.
    Uncomment this in routes if you want authentication.
    """
    expected_key = os.getenv("AURA_API_KEY")
    
    # Skip validation if no key is set (development mode)
    if not expected_key:
        return True
    
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    
    return True

# File validation helper
def validate_file_size(file_size: int, max_size_mb: int = 50):
    """Validate uploaded file size"""
    max_bytes = max_size_mb * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_size_mb}MB"
        )
    return True

def validate_file_extension(filename: str, allowed_extensions: list):
    """Validate file extension"""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    return True
