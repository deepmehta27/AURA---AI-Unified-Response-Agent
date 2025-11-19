"""Test configuration setup"""

from config.settings import settings
from utils.logger import logger

def test_configuration():
    """Test that configuration loads correctly"""
    try:
        logger.info("Testing AURA configuration...")
        
        # Test config loading
        app_name = settings.get('app.name')
        logger.info(f"App name: {app_name}")
        
        # Test LLM config
        llm_config = settings.get_llm_config()
        logger.info(f"LLM Model: {llm_config.get('model_name')}")
        
        # Test Pinecone config
        pinecone_config = settings.get_pinecone_config()
        logger.info(f"Pinecone Index: {pinecone_config.get('index_name')}")
        
        # Test environment variables (Windows-friendly)
        logger.info(f"OpenAI API Key: {'[SET]' if settings.openai_api_key else '[MISSING]'}")
        logger.info(f"Pinecone API Key: {'[SET]' if settings.pinecone_api_key else '[MISSING]'}")
        logger.info(f"Supabase URL: {'[SET]' if settings.supabase_url else '[MISSING]'}")
        
        logger.info("Configuration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_configuration()
