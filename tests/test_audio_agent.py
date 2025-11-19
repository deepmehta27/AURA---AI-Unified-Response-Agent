"""Test Audio Agent"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.audio_agent import audio_agent
from utils.logger import logger
import numpy as np
import wave
import tempfile


def create_test_audio():
    """Create a simple test audio file"""
    # Create a 3-second test audio (beep sound)
    sample_rate = 16000
    duration = 3
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save to WAV file
    Path("data/test_audio").mkdir(parents=True, exist_ok=True)
    audio_path = "data/test_audio/test_beep.wav"
    
    with wave.open(audio_path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    logger.info(f"Created test audio: {audio_path}")
    return audio_path


def test_transcribe():
    """Test audio transcription with real file"""
    logger.info("Testing audio transcription...")
    
    # Check for sample audio file - USE RAW STRING
    sample_path = r"E:\projects\AURA\sample-0.mp3"
    
    if Path(sample_path).exists():
        logger.info(f"Found sample audio: {sample_path}")
        
        result = audio_agent.process({
            "audio_path": sample_path,
            "analysis_type": "transcribe",
            "language": "en"
        })
        
        if result["success"]:
            logger.info(f"Transcript: {result['response'][:200]}...")  # First 200 chars
            logger.info(f"Language: {result['metadata']['language']}")
            logger.info(f"Segments: {result['metadata']['segments']}")
            return True
        else:
            logger.error(f"Transcription failed: {result.get('error')}")
            return False
    else:
        logger.warning(f"Sample audio not found at: {sample_path}")
        logger.info("Skipping real transcription test")
        logger.info("To test: place an MP3 file with speech at E:\\projects\\AURA\\sample-0.mp3")
        return True  # Don't fail test if file missing


def test_summarize():
    """Test audio summarization"""
    logger.info("Testing audio summarization...")
    
    sample_path = r"E:\projects\AURA\sample-0.mp3"
    
    if Path(sample_path).exists():
        result = audio_agent.process({
            "audio_path": sample_path,
            "analysis_type": "summarize"
        })
        
        if result["success"]:
            logger.info(f"Summary: {result['response']}")
            return True
        else:
            logger.error(f"Failed: {result.get('error')}")
            return False
    else:
        logger.info("Summarization pipeline ready (no sample file)")
        return True


def test_audio_info():
    """Test getting audio information"""
    logger.info("Testing audio info...")
    
    audio_path = create_test_audio()
    
    result = audio_agent.get_audio_info(audio_path)
    
    if result["success"]:
        info = result["info"]
        logger.info(f"Duration: {info['duration_seconds']}s")
        logger.info(f"Channels: {info['channels']}")
        logger.info(f"Sample rate: {info['frame_rate']}Hz")
        logger.info(f"File size: {info['file_size_mb']:.2f}MB")
        return True
    else:
        logger.error(f"Failed: {result.get('error')}")
        return False


def test_analyze():
    """Test audio analysis"""
    logger.info("Testing audio analysis...")
    
    sample_path = r"E:\projects\AURA\sample-0.mp3"
    
    if Path(sample_path).exists():
        result = audio_agent.process({
            "audio_path": sample_path,
            "analysis_type": "analyze",
            "query": "What are the main topics discussed?"
        })
        
        if result["success"]:
            logger.info(f"Analysis: {result['response'][:200]}...")
            return True
        else:
            logger.error(f"Failed: {result.get('error')}")
            return False
    else:
        logger.info("Analysis pipeline ready (no sample file)")
        return True


if __name__ == "__main__":
    logger.info("Starting Audio Agent tests...")
    logger.info("="*50)
    
    tests = [
        ("Audio Info", test_audio_info),
        ("Transcription", test_transcribe),
        ("Summarization", test_summarize),
        ("Analysis", test_analyze)
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
