"""Test Text Agent"""

from agents.text_agent import text_agent
from utils.logger import logger
from utils.pinecone_store import pinecone_store

def test_simple_query():
    """Test simple query without RAG"""
    logger.info("Testing simple query...")
    
    result = text_agent.process({
        "query": "What is AURA?",
        "use_rag": False
    })
    
    if result["success"]:
        logger.info(f"Response: {result['response']}")
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False

def test_rag_query():
    """Test query with RAG"""
    logger.info("Testing RAG query...")
    
    result = text_agent.process({
        "query": "What are the features of AURA?",
        "use_rag": True,
        "top_k": 3
    })
    
    if result["success"]:
        logger.info(f"Response: {result['response']}")
        logger.info(f"Sources retrieved: {result['metadata']['documents_retrieved']}")
        
        if result["sources"]:
            logger.info("Top source:")
            logger.info(f"  ID: {result['sources'][0]['id']}")
            logger.info(f"  Score: {result['sources'][0]['score']:.4f}")
        
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False

def test_batch_analysis():
    """Test batch document analysis"""
    logger.info("Testing batch analysis...")
    
    docs = [
        "AURA is an intelligent document processing platform.",
        "AURA uses AI agents for automation.",
        "AURA supports PDF, DOCX, and image files."
    ]
    
    result = text_agent.analyze_document_batch(docs, analysis_type="summary")
    
    if result["success"]:
        logger.info(f"Analysis: {result['analysis']}")
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False

if __name__ == "__main__":
    logger.info("Starting Text Agent tests...")
    logger.info("="*50)
    
    tests = [
        ("Simple Query", test_simple_query),
        ("RAG Query", test_rag_query),
        ("Batch Analysis", test_batch_analysis)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        try:
            if test_func():
                logger.info(f"[PASS] {test_name}")  # Removed emoji
            else:
                logger.error(f"[FAIL] {test_name}")  # Removed emoji
        except Exception as e:
            logger.error(f"[ERROR] {test_name}: {str(e)}")
    
    logger.info("\n" + "="*50)
    logger.info("All tests completed!")
