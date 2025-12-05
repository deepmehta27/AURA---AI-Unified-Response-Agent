from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BaseResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query text to process")
    use_rag: bool = Field(default=True, description="Whether to use RAG for context")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Chat history for context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is AURA?",
                "use_rag": True,
                "history": None
            }
        }

class AgentResponse(BaseResponse):
    response: str
    query_type: Optional[str] = None
    agents_used: Optional[List[str]] = None
    sources: Optional[List[Dict[str, Any]]] = None
