"""Test LangGraph Orchestrator"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import orchestrator
from utils.logger import logger
from PIL import Image, ImageDraw
import numpy as np
import wave


def create_test_image():
    """Create a test image with text"""
    Path("data/test_images").mkdir(parents=True, exist_ok=True)
    
    img = Image.new('RGB', (800, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    text = "AURA Multi-Agent System\nText, Image, Audio Processing\nVersion 1.0"
    draw.text((50, 100), text, fill=(0, 0, 0))
    
    draw.rectangle([50, 250, 300, 350], fill=(52, 152, 219), outline=(0, 0, 0), width=2)
    draw.text((60, 280), "Orchestrator Test", fill=(255, 255, 255))
    
    image_path = "data/test_images/orchestrator_test.png"
    img.save(image_path)
    logger.info(f"Created test image: {image_path}")
    
    return image_path


def create_test_audio():
    """Create a test audio file"""
    Path("data/test_audio").mkdir(parents=True, exist_ok=True)
    
    sample_rate = 16000
    duration = 2
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    audio_path = "data/test_audio/orchestrator_test.wav"
    
    with wave.open(audio_path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    logger.info(f"Created test audio: {audio_path}")
    return audio_path


def test_text_query():
    """Test pure text query routing"""
    logger.info("Testing text query routing...")
    
    result = orchestrator.process(
        query="What are the key features of AURA?",
        use_rag=True
    )
    
    if result["success"]:
        logger.info(f"Response: {result['response'][:200]}...")
        logger.info(f"Query Type: {result['metadata']['query_type']}")
        logger.info(f"Intent: {result['metadata']['intent']}")
        logger.info(f"Agents Used: {result['metadata']['agents_used']}")
        logger.info(f"Steps: {result['metadata']['processing_steps']}")
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False


def test_image_query():
    """Test image query routing"""
    logger.info("Testing image query routing...")
    
    image_path = create_test_image()
    
    result = orchestrator.process(
        query="What text is in this image?",
        image_path=image_path
    )
    
    if result["success"]:
        logger.info(f"Response: {result['response'][:200]}...")
        logger.info(f"Query Type: {result['metadata']['query_type']}")
        logger.info(f"Agents Used: {result['metadata']['agents_used']}")
        logger.info(f"Steps: {result['metadata']['processing_steps']}")
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False


def test_audio_query():
    """Test audio query routing"""
    logger.info("Testing audio query routing...")
    
    # Check if real audio exists
    real_audio = Path(r"E:\projects\AURA\sample-0.mp3")
    
    if real_audio.exists():
        logger.info("Using real audio file")
        audio_path = str(real_audio)
    else:
        logger.info("Using test beep audio")
        audio_path = create_test_audio()
    
    result = orchestrator.process(
        query="Transcribe this audio",
        audio_path=audio_path
    )
    
    if result["success"]:
        logger.info(f"Response: {result['response'][:200]}...")
        logger.info(f"Query Type: {result['metadata']['query_type']}")
        logger.info(f"Agents Used: {result['metadata']['agents_used']}")
        logger.info(f"Steps: {result['metadata']['processing_steps']}")
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False


def test_multi_modal_text_image():
    """Test multi-modal query (text + image)"""
    logger.info("Testing multi-modal query (text + image)...")
    
    image_path = create_test_image()
    
    result = orchestrator.process(
        query="Describe this image and tell me what AURA is based on your knowledge",
        image_path=image_path
    )
    
    if result["success"]:
        logger.info(f"Response: {result['response'][:300]}...")
        logger.info(f"Query Type: {result['metadata']['query_type']}")
        logger.info(f"Agents Used: {result['metadata']['agents_used']}")
        logger.info(f"Steps: {result['metadata']['processing_steps']}")
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False


def test_multi_modal_text_audio():
    """Test multi-modal query (text + audio)"""
    logger.info("Testing multi-modal query (text + audio)...")
    
    real_audio = Path(r"E:\projects\AURA\sample-0.mp3")
    
    if real_audio.exists():
        audio_path = str(real_audio)
        
        result = orchestrator.process(
            query="Transcribe this audio and then explain what AURA is",
            audio_path=audio_path
        )
        
        if result["success"]:
            logger.info(f"Response: {result['response'][:300]}...")
            logger.info(f"Query Type: {result['metadata']['query_type']}")
            logger.info(f"Agents Used: {result['metadata']['agents_used']}")
            logger.info(f"Steps: {result['metadata']['processing_steps']}")
            return True
        else:
            logger.error(f"Failed: {result.get('error')}")
            return False
    else:
        logger.info("Real audio not found, skipping test")
        return True


def test_classification_accuracy():
    """Test query classification accuracy"""
    logger.info("Testing classification accuracy...")
    
    test_cases = [
        {
            "query": "What is machine learning?",
            "expected_type": "text",
            "expected_intent": "question"
        },
        {
            "query": "Analyze this document and summarize it",
            "expected_type": "text",
            "expected_intent": "analyze"
        },
        {
            "query": "Search for information about AI",
            "expected_type": "text",
            "expected_intent": "search"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for case in test_cases:
        result = orchestrator.process(
            query=case["query"]
        )
        
        if result["success"]:
            actual_type = result["metadata"]["query_type"]
            actual_intent = result["metadata"]["intent"]
            
            type_match = actual_type == case["expected_type"]
            intent_match = actual_intent == case["expected_intent"]
            
            if type_match and intent_match:
                passed += 1
                logger.info(f"✓ Query: '{case['query'][:40]}...'")
                logger.info(f"  Type: {actual_type} (expected: {case['expected_type']})")
                logger.info(f"  Intent: {actual_intent} (expected: {case['expected_intent']})")
            else:
                logger.warning(f"✗ Query: '{case['query'][:40]}...'")
                logger.warning(f"  Type: {actual_type} (expected: {case['expected_type']})")
                logger.warning(f"  Intent: {actual_intent} (expected: {case['expected_intent']})")
    
    logger.info(f"Classification accuracy: {passed}/{total} ({passed/total*100:.1f}%)")
    return passed == total


def test_error_handling():
    """Test orchestrator error handling"""
    logger.info("Testing error handling...")
    
    # Test with invalid image path
    result = orchestrator.process(
        query="Analyze this image",
        image_path="nonexistent_image.png"
    )
    
    # Should handle gracefully
    if not result["success"]:
        logger.info("✓ Correctly handled invalid image path")
        return True
    else:
        logger.warning("✗ Did not catch invalid image path")
        return False


def test_workflow_tracking():
    """Test workflow step tracking"""
    logger.info("Testing workflow tracking...")
    
    result = orchestrator.process(
        query="What is AURA?",
        use_rag=True
    )
    
    if result["success"]:
        steps = result["metadata"]["processing_steps"]
        logger.info(f"Workflow steps tracked: {len(steps)}")
        for i, step in enumerate(steps, 1):
            logger.info(f"  {i}. {step}")
        
        # Verify expected steps
        has_classification = any("Classified" in step for step in steps)
        has_processing = any("processed" in step.lower() for step in steps)
        
        if has_classification and has_processing:
            logger.info("✓ All expected workflow steps present")
            return True
        else:
            logger.warning("✗ Some workflow steps missing")
            return False
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False


if __name__ == "__main__":
    logger.info("Starting Orchestrator tests...")
    logger.info("="*60)
    
    tests = [
        ("Text Query Routing", test_text_query),
        ("Image Query Routing", test_image_query),
        ("Audio Query Routing", test_audio_query),
        ("Multi-Modal (Text + Image)", test_multi_modal_text_image),
        ("Multi-Modal (Text + Audio)", test_multi_modal_text_audio),
        ("Classification Accuracy", test_classification_accuracy),
        ("Error Handling", test_error_handling),
        ("Workflow Tracking", test_workflow_tracking)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}\n")
        
        try:
            if test_func():
                logger.info(f"[PASS] {test_name}")
                passed += 1
            else:
                logger.error(f"[FAIL] {test_name}")
                failed += 1
        except Exception as e:
            logger.error(f"[ERROR] {test_name}: {str(e)}")
            failed += 1
    
    logger.info("\n" + "="*60)
    logger.info("Test Summary")
    logger.info("="*60)
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total:  {passed + failed}")
    logger.info(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
    logger.info("="*60)
    logger.info("All tests completed!")
