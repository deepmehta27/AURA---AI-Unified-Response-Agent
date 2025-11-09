"""
Embedding Generator for AURA
Generates vector embeddings from text using Sentence Transformers
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
from config.settings import settings
from utils.logger import logger

class EmbeddingGenerator:
    """Generate embeddings for text data"""
    
    def __init__(self):
        """Initialize the embedding model"""
        self.model_name = settings.get('embeddings.model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.dimension = settings.get('embeddings.dimension', 384)
        
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text string
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimension
            
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                logger.warning("Empty text list provided for embeddings")
                return []
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                logger.warning("No valid texts after filtering")
                return [[0.0] * self.dimension] * len(texts)
            
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=True)
            return embeddings.tolist()
        
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimension

# Global instance
embedding_generator = EmbeddingGenerator()
