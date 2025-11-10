"""Test document processor"""

from pathlib import Path
from utils.document_processor import document_processor
from utils.logger import logger
from utils.pinecone_store import pinecone_store

def create_test_files():
    """Create sample test files"""
    test_dir = Path("data/sample")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a test text file
    test_txt = test_dir / "test_document.txt"
    with open(test_txt, 'w') as f:
        f.write("""
AURA - AI Unified Response Agent

AURA is a cutting-edge multimodal AI system designed for intelligent customer support.

Key Features:
- Text processing with natural language understanding
- Image analysis and OCR capabilities
- Audio transcription and processing
- Retrieval-Augmented Generation (RAG) for accurate responses
- Multi-agent orchestration for complex queries

Built with enterprise-grade infrastructure including Pinecone, Supabase, and RabbitMQ.
        """)
    
    logger.info(f"Created test file: {test_txt}")
    return str(test_txt)

def test_text_extraction():
    """Test text extraction from different formats"""
    logger.info("Testing text extraction...")
    
    test_file = create_test_files()
    
    # Test processing
    result = document_processor.process_file(test_file)
    
    if result.get("success"):
        logger.info(f"Successfully extracted {len(result['text'])} characters")
        logger.info(f"Metadata: {result['metadata']}")
        return True
    else:
        logger.error(f"Extraction failed: {result.get('error')}")
        return False

def test_chunking():
    """Test text chunking"""
    logger.info("Testing text chunking...")
    
    text = "This is a test. " * 100  # Create long text
    chunks = document_processor.chunk_text(text, chunk_size=100, overlap=20)
    
    logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
    logger.info(f"First chunk: {chunks[0][:50]}...")
    
    return len(chunks) > 1

def test_process_and_store():
    """Test full pipeline: process, upload, and store"""
    logger.info("Testing full processing pipeline...")
    
    test_file = create_test_files()
    
    # Process and store
    result = document_processor.process_and_store(test_file)
    
    if result.get("success"):
        logger.info(f"Document ID: {result['document_id']}")
        logger.info(f"Chunks stored: {result['chunks_stored']}")
        logger.info(f"Supabase URL: {result.get('supabase_url')}")
        
        # Test query
        logger.info("Testing search...")
        search_results = pinecone_store.query("AI customer support system", top_k=3)
        logger.info(f"Search returned {len(search_results)} results")
        
        if search_results:
            logger.info(f"Top result score: {search_results[0]['score']:.4f}")
        
        return True
    else:
        logger.error(f"Pipeline failed: {result.get('error')}")
        return False

if __name__ == "__main__":
    logger.info("Starting document processor tests...")
    
    tests = [
        ("Text Extraction", test_text_extraction),
        ("Text Chunking", test_chunking),
        ("Full Pipeline", test_process_and_store)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                logger.info(f"[PASS] {test_name}")
            else:
                logger.error(f"[FAIL] {test_name}")
        except Exception as e:
            logger.error(f"[ERROR] {test_name}: {str(e)}")
    
    logger.info("\nAll tests completed!")
