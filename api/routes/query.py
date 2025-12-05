from fastapi import APIRouter, HTTPException, Depends
from api.schemas import QueryRequest, AgentResponse
from api.dependencies import get_orchestrator
from utils.logger import logger

router = APIRouter()

@router.post("/", response_model=AgentResponse)
async def process_query(
    request: QueryRequest,
    orchestrator = Depends(get_orchestrator)
):
    """
    Process a text query using the Orchestrator.
    """
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Build input parameters
        input_params = {
            "query": request.query,
            "use_rag": request.use_rag
        }
        
        # Only add history if it's not None and not empty
        if request.history and len(request.history) > 0:
            input_params["chat_history"] = request.history
        
        # Process with orchestrator
        result = orchestrator.process(**input_params)
        
        # Check if there was an error
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error occurred")
            logger.error(f"Query processing failed: {error_msg}")
            
            # Return error in proper format
            return AgentResponse(
                success=False,
                response="",
                error=error_msg,
                metadata=result.get("metadata")
            )

        return AgentResponse(
            success=result["success"],
            response=result.get("response", ""),
            error=result.get("error"),
            query_type=result.get("metadata", {}).get("query_type"),
            agents_used=result.get("metadata", {}).get("agents_used"),
            metadata=result.get("metadata")
        )

    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
