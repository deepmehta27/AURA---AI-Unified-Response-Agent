"""
Pinecone Vector Store for AURA
Handles all interactions with Pinecone vector database
"""

from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from config.settings import settings
from utils.logger import logger
from utils.embedding_generator import embedding_generator

class PineconeStore:
    """Manage Pinecone vector database operations"""
    
    def __init__(self):
        """Initialize Pinecone connection"""
        self.api_key = settings.pinecone_api_key
        self.environment = settings.pinecone_environment
        self.index_name = settings.get('pinecone.index_name', 'aura-docs')
        self.dimension = settings.get('pinecone.dimension', 384)
        self.metric = settings.get('pinecone.metric', 'cosine')
        
        logger.info("Initializing Pinecone connection...")
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists, create if not
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=settings.get('pinecone.cloud', 'aws'),
                        region=self.environment
                    )
                )
                logger.info(f"Pinecone index '{self.index_name}' created successfully")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info("Successfully connected to Pinecone index")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def upsert_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Insert or update a document in Pinecone
        
        Args:
            doc_id: Unique document ID
            text: Document text content
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = embedding_generator.generate_embedding(text)
            
            # Prepare metadata
            meta = metadata or {}
            meta['text'] = text[:1000]  # Store first 1000 chars for reference
            
            # Upsert to Pinecone
            self.index.upsert(
                vectors=[(doc_id, embedding, meta)]
            )
            
            logger.info(f"Successfully upserted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting document {doc_id}: {str(e)}")
            return False
    
    def upsert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Insert or update multiple documents (batch)
        
        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'
            
        Returns:
            Number of successfully upserted documents
        """
        try:
            # Prepare texts for batch embedding
            texts = [doc['text'] for doc in documents]
            embeddings = embedding_generator.generate_embeddings(texts)
            
            # Prepare vectors for upsert
            vectors = []
            for i, doc in enumerate(documents):
                doc_id = doc['id']
                embedding = embeddings[i]
                metadata = doc.get('metadata', {})
                metadata['text'] = doc['text'][:1000]
                
                vectors.append((doc_id, embedding, metadata))
            
            # Batch upsert
            self.index.upsert(vectors=vectors)
            
            logger.info(f"Successfully upserted {len(documents)} documents")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error in batch upsert: {str(e)}")
            return 0
    
    def query(self, query_text: str, top_k: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Query Pinecone for similar documents
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of matching documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = embedding_generator.generate_embedding(query_text)
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter
            )
            
            # Format results
            matches = []
            for match in results.matches:
                matches.append({
                    'id': match.id,
                    'score': match.score,
                    'text': match.metadata.get('text', ''),
                    'metadata': match.metadata
                })
            
            logger.info(f"Query returned {len(matches)} results")
            return matches
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from Pinecone
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful
        """
        try:
            self.index.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}

# Global instance
pinecone_store = PineconeStore()
