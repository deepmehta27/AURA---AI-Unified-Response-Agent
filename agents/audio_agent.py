"""
Audio Agent for AURA
Handles audio transcription, analysis, and processing
"""

from typing import Dict, Any, Optional
import base64
from pathlib import Path
from agents.base_agent import BaseAgent
from utils.logger import logger
import tempfile
import os
import shutil

# Configure ffmpeg for Windows
if os.name == 'nt':  # Windows only
    ffmpeg_locations = [
        r"C:\Program Files\ffmpeg-8.0-essentials_build\bin",
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin"
    ]
    
    for ffmpeg_path in ffmpeg_locations:
        if os.path.exists(os.path.join(ffmpeg_path, 'ffmpeg.exe')):
            # Add to PATH if not already there
            if ffmpeg_path not in os.environ.get('PATH', ''):
                os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
            logger.info(f"ffmpeg configured: {ffmpeg_path}")
            
            # Verify ffmpeg is accessible
            if shutil.which('ffmpeg'):
                logger.info("ffmpeg is accessible")
            break

# Audio processing libraries
try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.info("Whisper imported successfully")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available - install with: pip install openai-whisper")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not available - install with: pip install pydub")


class AudioAgent(BaseAgent):
    """Agent specialized in audio processing and transcription"""
    
    def __init__(self):
        super().__init__(
            name="Audio Agent",
            description="audio transcription, analysis, and processing"
        )
        
        # Check ffmpeg availability
        self.ffmpeg_available = shutil.which('ffmpeg') is not None
        if self.ffmpeg_available:
            logger.info("ffmpeg is available for audio processing")
        else:
            logger.warning("ffmpeg not found - some audio formats may not work")
        
        # Initialize Whisper model
        self._init_whisper()
        
        logger.info("Audio Agent initialized")
    
    def _init_whisper(self):
        """Initialize Whisper model"""
        if WHISPER_AVAILABLE:
            try:
                # Use base model (faster, good quality)
                # Options: tiny, base, small, medium, large
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper base model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self.whisper_model = None
        else:
            self.whisper_model = None
            logger.warning("Whisper not available - transcription will be limited")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process audio with various analysis types
        
        Args:
            input_data: {
                "audio_path": str (path to audio file),
                "audio_url": str (URL to audio),
                "audio_bytes": bytes (raw audio data),
                "analysis_type": str (transcribe, analyze, summarize, translate),
                "language": str (source language, optional),
                "translate_to": str (target language for translation)
            }
            
        Returns:
            {
                "success": bool,
                "response": str (transcript/analysis),
                "metadata": dict
            }
        """
        try:
            analysis_type = input_data.get("analysis_type", "transcribe")
            
            # Get audio data
            audio_path = self._get_audio_path(input_data)
            
            if not audio_path:
                return {
                    "success": False,
                    "error": "No audio provided or invalid audio format"
                }
            
            logger.info(f"Processing audio with analysis type: {analysis_type}")
            
            # Route to appropriate method
            if analysis_type == "transcribe":
                return self._transcribe_audio(audio_path, input_data)
            elif analysis_type == "analyze":
                return self._analyze_audio(audio_path, input_data)
            elif analysis_type == "summarize":
                return self._summarize_audio(audio_path, input_data)
            elif analysis_type == "translate":
                return self._translate_audio(audio_path, input_data)
            else:
                return {
                    "success": False,
                    "error": f"Unknown analysis type: {analysis_type}"
                }
            
        except Exception as e:
            logger.error(f"Error in AudioAgent.process: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_audio_path(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Get audio file path, handling various input types"""
        try:
            # From file path
            if "audio_path" in input_data:
                path = Path(input_data["audio_path"])
                if path.exists():
                    return str(path)
                else:
                    logger.error(f"Audio file not found: {path}")
                    return None
            
            # From bytes - save to temp file
            elif "audio_bytes" in input_data:
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                )
                temp_file.write(input_data["audio_bytes"])
                temp_file.close()
                logger.info(f"Audio saved to temp file: {temp_file.name}")
                return temp_file.name
            
            # From URL - would need to download (not implemented yet)
            elif "audio_url" in input_data:
                logger.error("URL audio download not yet implemented")
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting audio path: {str(e)}")
            return None
    
    def _transcribe_audio(self, audio_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        try:
            if not self.whisper_model:
                return {
                    "success": False,
                    "error": "Whisper model not available"
                }
            
            # Check if file exists
            if not Path(audio_path).exists():
                return {
                    "success": False,
                    "error": f"Audio file not found: {audio_path}"
                }
            
            # Check ffmpeg for MP3/other formats
            file_ext = Path(audio_path).suffix.lower()
            if file_ext in ['.mp3', '.m4a', '.aac', '.ogg'] and not self.ffmpeg_available:
                return {
                    "success": False,
                    "error": f"ffmpeg required for {file_ext} files. Please ensure ffmpeg is in PATH."
                }
            
            # Get optional language
            language = input_data.get("language", None)
            
            logger.info(f"Transcribing audio: {audio_path}")
            
            try:
                # Transcribe with Whisper
                result = self.whisper_model.transcribe(
                    audio_path,
                    language=language,
                    fp16=False  # Use fp32 for CPU
                )
            except Exception as e:
                error_msg = str(e)
                if "ffmpeg" in error_msg.lower():
                    return {
                        "success": False,
                        "error": "ffmpeg error. Ensure ffmpeg is in PATH and try again."
                    }
                raise
            
            transcript = result["text"]
            detected_language = result.get("language", "unknown")
            
            logger.info(f"Transcription complete (language: {detected_language})")
            logger.info(f"Transcript length: {len(transcript)} characters")
            
            return {
                "success": True,
                "response": transcript,
                "analysis_type": "transcribe",
                "metadata": {
                    "language": detected_language,
                    "audio_path": audio_path,
                    "model": "whisper-base",
                    "segments": len(result.get("segments", []))
                }
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_audio(self, audio_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audio content with GPT"""
        try:
            # First transcribe
            transcript_result = self._transcribe_audio(audio_path, input_data)
            
            if not transcript_result["success"]:
                return transcript_result
            
            transcript = transcript_result["response"]
            
            # Analyze with GPT
            analysis_prompt = input_data.get(
                "query",
                "Analyze this audio transcript. Identify key topics, sentiment, and main points."
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing audio transcripts."
                },
                {
                    "role": "user",
                    "content": f"""Audio Transcript:
{transcript}

Analysis Task: {analysis_prompt}

Provide a detailed analysis."""
                }
            ]
            
            response = self._call_openai(
                messages=messages,
                max_tokens=1000,
                reasoning_effort="medium",
                verbosity="medium"
            )
            
            return {
                "success": True,
                "response": response,
                "analysis_type": "analyze",
                "metadata": {
                    "transcript": transcript,
                    "language": transcript_result["metadata"]["language"],
                    "model": self.model
                }
            }
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _summarize_audio(self, audio_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize audio content"""
        try:
            # First transcribe
            transcript_result = self._transcribe_audio(audio_path, input_data)
            
            if not transcript_result["success"]:
                return transcript_result
            
            transcript = transcript_result["response"]
            
            # Summarize with GPT
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at summarizing audio transcripts concisely."
                },
                {
                    "role": "user",
                    "content": f"""Summarize this audio transcript in 3-5 bullet points:

{transcript}

Provide a clear, concise summary of the main points."""
                }
            ]
            
            response = self._call_openai(
                messages=messages,
                max_tokens=500,
                reasoning_effort="low",
                verbosity="low"
            )
            
            return {
                "success": True,
                "response": response,
                "analysis_type": "summarize",
                "metadata": {
                    "transcript": transcript,
                    "language": transcript_result["metadata"]["language"],
                    "model": self.model
                }
            }
            
        except Exception as e:
            logger.error(f"Audio summarization failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _translate_audio(self, audio_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate audio to another language"""
        try:
            # First transcribe
            transcript_result = self._transcribe_audio(audio_path, input_data)
            
            if not transcript_result["success"]:
                return transcript_result
            
            transcript = transcript_result["response"]
            source_language = transcript_result["metadata"]["language"]
            target_language = input_data.get("translate_to", "English")
            
            # Translate with GPT
            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert translator. Translate from {source_language} to {target_language}."
                },
                {
                    "role": "user",
                    "content": f"""Translate this text to {target_language}:

{transcript}

Provide only the translation, maintaining the original meaning and tone."""
                }
            ]
            
            response = self._call_openai(
                messages=messages,
                max_tokens=2000,
                reasoning_effort="medium",
                verbosity="low"
            )
            
            return {
                "success": True,
                "response": response,
                "analysis_type": "translate",
                "metadata": {
                    "original_transcript": transcript,
                    "source_language": source_language,
                    "target_language": target_language,
                    "model": self.model
                }
            }
            
        except Exception as e:
            logger.error(f"Audio translation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get audio file information"""
        try:
            if not PYDUB_AVAILABLE:
                return {
                    "success": False,
                    "error": "pydub not available"
                }
            
            audio = AudioSegment.from_file(audio_path)
            
            return {
                "success": True,
                "info": {
                    "duration_seconds": len(audio) / 1000.0,
                    "channels": audio.channels,
                    "sample_width": audio.sample_width,
                    "frame_rate": audio.frame_rate,
                    "file_size_mb": os.path.getsize(audio_path) / (1024 * 1024)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting audio info: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
audio_agent = AudioAgent()