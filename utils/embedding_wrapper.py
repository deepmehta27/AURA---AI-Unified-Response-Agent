"""
Embedding wrapper for LangChain compatibility
"""

from typing import List
from langchain_core.embeddings import Embeddings
from utils.embedding_generator import embedding_generator


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for Sentence Transformers to work with LangChain"""
    
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.model.encode([text])[0].tolist()


# Global instance
langchain_embeddings = SentenceTransformerEmbeddings(embedding_generator.model)
