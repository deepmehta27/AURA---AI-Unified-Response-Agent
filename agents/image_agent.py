"""
Image Agent for AURA - Enhanced with Multi-Engine OCR
Handles image analysis, OCR, and visual understanding
"""

from typing import Dict, Any, Optional, List
import base64
from pathlib import Path
from agents.base_agent import BaseAgent
from utils.logger import logger
from PIL import Image
import io
import cv2
import numpy as np

# OCR engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    logger.info("pytesseract imported successfully")
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available - install with: pip install pytesseract")


class ImageAgent(BaseAgent):
    """Agent specialized in image processing with multi-engine OCR"""
    
    def __init__(self):
        super().__init__(
            name="Image Agent",
            description="image analysis, multi-engine OCR, and visual question answering"
        )
        self.vision_model = "gpt-5-mini"
        
        # Set Tesseract path directly (Windows PATH too long)
        if TESSERACT_AVAILABLE:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            logger.info("Tesseract path configured: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
        
        # Initialize OCR engines
        self._init_ocr_engines()
        
        logger.info(f"Image Agent initialized with vision model: {self.vision_model}")
    
    def _init_ocr_engines(self):
        """Initialize available OCR engines"""
        self.ocr_engines = {
            "tesseract": TESSERACT_AVAILABLE,
            "gpt_vision": True  # Fallback only
        }
        
        available_engines = [k for k, v in self.ocr_engines.items() if v]
        logger.info(f"Available OCR engines: {available_engines}")
        
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract not available - using GPT Vision as fallback")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image with various analysis types
        
        Args:
            input_data: {
                "image_path": str,
                "image_url": str,
                "image_bytes": bytes,
                "query": str,
                "analysis_type": str (describe, ocr, analyze, question),
                "ocr_engine": str (tesseract, gpt_vision, all)
            }
            
        Returns:
            {
                "success": bool,
                "response": str,
                "metadata": dict
            }
        """
        try:
            analysis_type = input_data.get("analysis_type", "describe")
            query = input_data.get("query", "")
            
            # Get image data
            image_data = self._get_image_data(input_data)
            
            if not image_data:
                return {
                    "success": False,
                    "error": "No image provided"
                }
            
            logger.info(f"Processing image with analysis type: {analysis_type}")
            
            # Route to appropriate method
            if analysis_type == "ocr":
                return self._process_ocr_advanced(image_data, input_data)
            elif analysis_type == "describe":
                return self._describe_image(image_data)
            elif analysis_type == "analyze":
                return self._analyze_image(image_data, query)
            elif analysis_type == "question":
                if not query:
                    return {"success": False, "error": "Query required"}
                return self._answer_question(image_data, query)
            else:
                return {"success": False, "error": f"Unknown analysis type: {analysis_type}"}
            
        except Exception as e:
            logger.error(f"Error in ImageAgent.process: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _get_image_data(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract and encode image data"""
        try:
            # From file path
            if "image_path" in input_data:
                path = Path(input_data["image_path"])
                if not path.exists():
                    logger.error(f"Image not found: {path}")
                    return None
                
                with open(path, "rb") as f:
                    image_bytes = f.read()
                
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                return {
                    "type": "file",
                    "base64": base64_image,
                    "path": str(path),
                    "bytes": image_bytes
                }
            
            # From URL
            elif "image_url" in input_data:
                return {
                    "type": "url",
                    "url": input_data["image_url"]
                }
            
            # From bytes
            elif "image_bytes" in input_data:
                base64_image = base64.b64encode(input_data["image_bytes"]).decode('utf-8')
                return {
                    "type": "bytes",
                    "base64": base64_image,
                    "bytes": input_data["image_bytes"]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting image data: {str(e)}")
            return None
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR"""
        try:
            # Read image
            img = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Threshold
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            logger.info("Image preprocessed for OCR")
            return thresh
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}, using original")
            return cv2.imread(image_path)
    
    def _process_ocr_advanced(self, image_data: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced OCR with Tesseract primary, GPT Vision fallback"""
        try:
            ocr_engine = input_data.get("ocr_engine", "all")
            results = {}
            
            # Priority 1: Tesseract (local, fast, free)
            if (ocr_engine == "all" or ocr_engine == "tesseract") and TESSERACT_AVAILABLE:
                if image_data.get("path"):
                    tesseract_result = self._ocr_tesseract(image_data["path"])
                    confidence = tesseract_result.get("confidence", 0)
                    
                    # Use Tesseract if confidence > 60%
                    if confidence > 60:
                        results["tesseract"] = tesseract_result
                        logger.info(f"Tesseract confident: {confidence:.1f}%")
                    elif confidence > 0:
                        # Low confidence but got something
                        results["tesseract"] = tesseract_result
                        logger.warning(f"Tesseract low confidence: {confidence:.1f}%")
            
            # Priority 2: GPT Vision (fallback)
            # Use if:
            # - Explicitly requested
            # - Tesseract unavailable
            # - Tesseract low confidence (<70%)
            use_gpt = False
            
            if ocr_engine == "gpt_vision":
                use_gpt = True
            elif not TESSERACT_AVAILABLE:
                use_gpt = True
                logger.info("Tesseract not available - using GPT Vision")
            elif len(results) == 0:
                use_gpt = True
                logger.info("Tesseract failed - using GPT Vision fallback")
            elif results.get("tesseract", {}).get("confidence", 0) < 70:
                use_gpt = True
                logger.info("Tesseract low confidence - using GPT Vision for validation")
            
            if use_gpt:
                gpt_result = self._ocr_gpt_vision(image_data)
                results["gpt_vision"] = gpt_result
            
            if len(results) == 0:
                return {"success": False, "error": "No OCR engines available"}
            
            # Smart combination
            combined_text = self._combine_ocr_smart(results)
            
            return {
                "success": True,
                "response": combined_text,
                "analysis_type": "ocr",
                "metadata": {
                    "engines_used": list(results.keys()),
                    "individual_results": results,
                    "model": self.vision_model,
                    "primary_engine": "tesseract" if "tesseract" in results else "gpt_vision"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in OCR: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _ocr_tesseract(self, image_path: str) -> Dict[str, Any]:
        """Run Tesseract OCR"""
        try:
            # Preprocess image
            processed = self._preprocess_image(image_path)
            
            # Run Tesseract
            text = pytesseract.image_to_string(processed)
            
            # Get confidence
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            logger.info(f"Tesseract OCR complete (confidence: {avg_confidence:.2f}%)")
            
            return {
                "text": text.strip(),
                "confidence": avg_confidence,
                "engine": "tesseract"
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def _ocr_gpt_vision(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run GPT-5 Vision OCR"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """DOCUMENT OCR EXTRACTION

                        TASK: Extract ALL text from this document image with business-document precision.

                        REQUIREMENTS:
                        1. Extract every word, number, and symbol
                        2. Preserve formatting (line breaks, spacing, structure)
                        3. Identify document structure (headers, sections, tables)
                        4. Flag uncertain text with [?]
                        5. Note any missing or illegible sections

                        OUTPUT FORMAT:
                        **Document Type:** [type]

                        **Full Text:**
                        [Complete extracted text with formatting preserved]

                        **Structured Data:**
                        - Date(s): [if present]
                        - ID/Reference Numbers: [if present]
                        - Amounts/Prices: [if present]
                        - Parties/Names: [if present]

                        **Tables** (if present):
                        [Table data in structured format]

                        **OCR Quality:** [confidence level]
                        **Issues:** [any unclear sections]

                        Be thorough - every word matters for document intelligence."""
                        },
                        self._build_image_content(image_data)
                    ]
                }
            ]
            
            text = self._call_vision_api(messages)
            
            return {
                "text": text,
                "confidence": 95,  # GPT-5 is generally very confident
                "engine": "gpt_vision"
            }
            
        except Exception as e:
            logger.error(f"GPT Vision OCR failed: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def _combine_ocr_smart(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Smart OCR combination: prefer Tesseract, validate with GPT"""
        
        # If only one engine, use it
        if len(results) == 1:
            engine_name = list(results.keys())[0]
            return results[engine_name]["text"]
        
        # Both available: prefer Tesseract if confident
        if "tesseract" in results:
            tess_conf = results["tesseract"].get("confidence", 0)
            tess_text = results["tesseract"]["text"]
            
            if tess_conf > 75:
                # High confidence - use Tesseract
                logger.info(f"Using Tesseract (confidence: {tess_conf:.1f}%)")
                if "gpt_vision" in results:
                    return f"{tess_text}\n\n[Validated with GPT-5 Vision]"
                return tess_text
            
            # Medium confidence - show both
            logger.info("Showing both Tesseract and GPT Vision results")
            combined = [
                f"[TESSERACT - {tess_conf:.1f}%]",
                tess_text,
                ""
            ]
            
            if "gpt_vision" in results:
                gpt_text = results["gpt_vision"]["text"]
                gpt_conf = results["gpt_vision"].get("confidence", 95)
                combined.extend([
                    f"[GPT VISION - {gpt_conf:.1f}%]",
                    gpt_text
                ])
            
            return "\n".join(combined)
        
        # Only GPT available
        return results["gpt_vision"]["text"]
    
    def _describe_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image description"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Document Image Analysis:

                            TASK: Identify and describe this document image for business intelligence.

                            OUTPUT:
                            **Document Type:** [invoice/contract/receipt/form/report/letter/diagram/other]

                            **Key Visual Elements:**
                            - Headers/Titles: [text]
                            - Logos/Branding: [if present]
                            - Layout: [single/multi-column, form, table, freeform]
                            - Notable Features: [stamps, signatures, barcodes]

                            **Text Preview:** [sample of visible text]

                            **Quality Assessment:**
                            - Scan Quality: [excellent/good/poor]
                            - Readability: [clear/moderate/difficult]
                            - Issues: [blur, skew, shadows, missing parts]

                            **Recommended Action:** [OCR/manual entry/rescan]

                            Focus on document characteristics, not artistic description."""
                        },
                        self._build_image_content(image_data)
                    ]
                }
            ]
            
            response = self._call_vision_api(messages)
            
            return {
                "success": True,
                "response": response,
                "analysis_type": "describe",
                "metadata": {
                    "image_source": image_data.get("type"),
                    "model": self.vision_model
                }
            }
            
        except Exception as e:
            logger.error(f"Error describing image: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _analyze_image(self, image_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Perform custom analysis"""
        try:
            analysis_prompt = query if query else """Deep Document Analysis:

            ANALYZE this document image for:
            1. Document Type & Purpose
            2. Key Information (dates, amounts, parties, terms)
            3. Document Structure (sections, clauses, fields)
            4. Legal/Financial Elements (signatures, totals, terms)
            5. Completeness (missing fields, stamps, signatures)
            6. Risks/Flags (anomalies, inconsistencies, concerns)

            Provide business intelligence, not just description."""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        self._build_image_content(image_data)
                    ]
                }
            ]
            
            response = self._call_vision_api(messages)
            
            return {
                "success": True,
                "response": response,
                "analysis_type": "analyze",
                "metadata": {
                    "query": analysis_prompt,
                    "model": self.vision_model
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _answer_question(self, image_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Answer question about image"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        self._build_image_content(image_data)
                    ]
                }
            ]
            
            response = self._call_vision_api(messages)
            
            return {
                "success": True,
                "response": response,
                "analysis_type": "question",
                "metadata": {
                    "query": query,
                    "model": self.vision_model
                }
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _build_image_content(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build image content for API"""
        if image_data["type"] == "url":
            return {
                "type": "image_url",
                "image_url": {"url": image_data["url"]}
            }
        else:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data['base64']}"
                }
            }
    
    def _call_vision_api(self, messages: list) -> str:
        """Call OpenAI Vision API"""
        try:
            params = {
                "model": self.vision_model,
                "messages": messages,
                "store": False,
                "max_completion_tokens": 1000,
                "verbosity": "medium",
                "reasoning_effort": "medium"
            }
            
            response = self.client.chat.completions.create(**params)
            
            result = response.choices[0].message.content
            logger.info("GPT-5 mini Vision API call successful")
            return result
            
        except Exception as e:
            logger.error(f"Error calling vision API: {str(e)}")
            raise


# Global instance
image_agent = ImageAgent()
