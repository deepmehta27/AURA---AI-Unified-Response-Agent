"""Test Image Agent"""

from agents.image_agent import image_agent
from utils.logger import logger
from PIL import Image, ImageDraw
from pathlib import Path
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def create_test_image():
    """Create a test image with text"""
    # Create test directory
    Path("data/test_images").mkdir(parents=True, exist_ok=True)
    
    # Create image with text
    img = Image.new('RGB', (800, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Add text
    text = "AURA - Intelligent Document Processing\nAI Agent System\nVersion 1.0"
    draw.text((50, 100), text, fill=(0, 0, 0))
    
    # Add colored rectangle
    draw.rectangle([50, 250, 300, 350], fill=(52, 152, 219), outline=(0, 0, 0), width=2)
    draw.text((60, 280), "Test Image", fill=(255, 255, 255))
    
    # Save
    image_path = "data/test_images/test_document.png"
    img.save(image_path)
    logger.info(f"Created test image: {image_path}")
    
    return image_path


def test_describe_image():
    """Test image description"""
    logger.info("Testing image description...")
    
    image_path = create_test_image()
    
    result = image_agent.process({
        "image_path": image_path,
        "analysis_type": "describe"
    })
    
    if result["success"]:
        logger.info(f"Description: {result['response'][:200]}...")
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False


def test_ocr():
    """Test OCR extraction with multiple engines"""
    logger.info("Testing multi-engine OCR...")
    
    image_path = create_test_image()
    
    result = image_agent.process({
        "image_path": image_path,
        "analysis_type": "ocr",
        "ocr_engine": "all"  # Use all available engines
    })
    
    if result["success"]:
        logger.info(f"Extracted text:\n{result['response'][:300]}...")
        
        # Check which engines were used
        engines_used = result['metadata'].get('engines_used', [])
        logger.info(f"OCR Engines used: {', '.join(engines_used)}")
        
        # Show individual results if available
        if 'individual_results' in result['metadata']:
            for engine, engine_result in result['metadata']['individual_results'].items():
                confidence = engine_result.get('confidence', 0)
                logger.info(f"  - {engine}: confidence {confidence:.1f}%")
        
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False


def test_ocr_single_engine():
    """Test OCR with single engine (GPT Vision)"""
    logger.info("Testing single-engine OCR (GPT Vision only)...")
    
    image_path = create_test_image()
    
    result = image_agent.process({
        "image_path": image_path,
        "analysis_type": "ocr",
        "ocr_engine": "gpt_vision"  # Only GPT Vision
    })
    
    if result["success"]:
        logger.info(f"GPT Vision OCR:\n{result['response'][:200]}...")
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False


def test_question_answering():
    """Test answering questions about image"""
    logger.info("Testing question answering...")
    
    image_path = create_test_image()
    
    result = image_agent.process({
        "image_path": image_path,
        "analysis_type": "question",
        "query": "What text is visible in this image?"
    })
    
    if result["success"]:
        logger.info(f"Answer: {result['response'][:200]}...")
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False


def test_analyze():
    """Test custom image analysis"""
    logger.info("Testing custom analysis...")
    
    image_path = create_test_image()
    
    result = image_agent.process({
        "image_path": image_path,
        "analysis_type": "analyze",
        "query": "Identify the colors and layout structure of this image"
    })
    
    if result["success"]:
        logger.info(f"Analysis: {result['response'][:200]}...")
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False


if __name__ == "__main__":
    logger.info("Starting Image Agent tests...")
    logger.info("="*50)
    
    tests = [
        ("Image Description", test_describe_image),
        ("Multi-Engine OCR", test_ocr),
        ("Single Engine OCR", test_ocr_single_engine),
        ("Question Answering", test_question_answering),
        ("Custom Analysis", test_analyze)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        try:
            if test_func():
                logger.info(f"[PASS] {test_name}")
            else:
                logger.error(f"[FAIL] {test_name}")
        except Exception as e:
            logger.error(f"[ERROR] {test_name}: {str(e)}")
    
    logger.info("\n" + "="*50)
    logger.info("All tests completed!")
