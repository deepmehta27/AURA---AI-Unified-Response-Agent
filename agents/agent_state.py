"""
Agent State Management for LangGraph
Defines the state structure for multi-agent workflows
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add


class AgentState(TypedDict):
    """State passed between agents in the workflow"""
    
    # Input
    query: str
    image_path: Optional[str]
    audio_path: Optional[str]
    document_path: Optional[str]
    
    # Classification
    query_type: Optional[str]  # "text", "image", "audio", "multi_modal"
    intent: Optional[str]  # "search", "analyze", "process", "question"
    
    # Processing
    current_agent: Optional[str]
    processing_steps: Annotated[List[str], add]  # Track workflow
    
    # Results
    text_response: Optional[str]
    image_analysis: Optional[Dict[str, Any]]
    audio_transcript: Optional[str]
    retrieved_docs: Optional[List[Dict[str, Any]]]
    
    # Metadata
    confidence: Optional[float]
    error: Optional[str]
    metadata: Dict[str, Any]
    
    # Final output
    final_response: Optional[str]
    sources: Optional[List[Dict[str, Any]]]
