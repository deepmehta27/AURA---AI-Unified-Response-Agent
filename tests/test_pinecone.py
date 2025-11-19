"""Test Pinecone and Embedding utilities"""

import time  # Add this import
from utils.pinecone_store import pinecone_store
from utils.embedding_generator import embedding_generator
from utils.logger import logger

def test_embeddings():
    """Test embedding generation"""
    logger.info("Testing embedding generation...")
    
    text = "This is a test document for AURA"
    embedding = embedding_generator.generate_embedding(text)
    
    logger.info(f"Generated embedding dimension: {len(embedding)}")
    logger.info(f"First 5 values: {embedding[:5]}")
    
    return len(embedding) == 384

def test_pinecone():
    """Test Pinecone operations"""
    logger.info("Testing Pinecone operations...")
    
    # Test upsert
    success = pinecone_store.upsert_document(
        doc_id="test-doc-1",
        text="AURA is a multimodal AI agent system for customer support",
        metadata={"source": "test", "type": "demo"}
    )
    
    if success:
        logger.info("Document upserted successfully")
    
    # IMPORTANT: Wait for Pinecone to index the document
    logger.info("Waiting 5 seconds for Pinecone to index the document...")
    time.sleep(5)
    
    # Test query
    results = pinecone_store.query("AI agent system", top_k=3)
    logger.info(f"Query returned {len(results)} results")
    
    if results:
        for i, result in enumerate(results, 1):
            logger.info(f"Result {i}: Score={result['score']:.4f}, Text={result['text'][:50]}...")
    else:
        logger.warning("No results found - Pinecone may need more time to index")
    
    # Test stats
    stats = pinecone_store.get_stats()
    logger.info(f"Index stats: {stats}")
    
    return success

if __name__ == "__main__":
    logger.info("Starting tests...")
    
    if test_embeddings():
        logger.info("[PASS] Embedding test passed")
    else:
        logger.error("[FAIL] Embedding test failed")
    
    if test_pinecone():
        logger.info("[PASS] Pinecone test passed")
    else:
        logger.error("[FAIL] Pinecone test failed")
    
    logger.info("All tests completed!")
