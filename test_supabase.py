"""Test Supabase client"""

import os
from pathlib import Path
from utils.supabase_client import supabase_client
from utils.logger import logger

def test_supabase_storage():
    """Test Supabase storage operations"""
    logger.info("Testing Supabase storage...")
    
    # Create a test file
    test_content = b"This is a test file for AURA Supabase integration"
    test_filename = "test/test_document.txt"
    
    try:
        # Test upload
        logger.info("Testing file upload...")
        public_url = supabase_client.upload_bytes(
            file_data=test_content,
            destination_path=test_filename,
            content_type="text/plain"
        )
        
        if public_url:
            logger.info(f"Upload successful! URL: {public_url}")
        else:
            logger.error("Upload failed")
            return False
        
        # Test list files
        logger.info("Testing list files...")
        files = supabase_client.list_files("test")
        logger.info(f"Found {len(files)} files in 'test' folder")
        
        # Test download
        logger.info("Testing file download...")
        downloaded_data = supabase_client.download_bytes(test_filename)
        
        if downloaded_data:
            logger.info(f"Downloaded {len(downloaded_data)} bytes")
            logger.info(f"Content matches: {downloaded_data == test_content}")
        
        # Test delete
        logger.info("Testing file deletion...")
        deleted = supabase_client.delete_file(test_filename)
        
        if deleted:
            logger.info("File deleted successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Supabase test failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting Supabase tests...")
    
    if test_supabase_storage():
        logger.info("[PASS] Supabase test passed")
    else:
        logger.error("[FAIL] Supabase test failed")
