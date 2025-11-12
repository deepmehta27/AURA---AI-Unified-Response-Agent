"""
Text Agent for AURA
Handles text-based queries, document Q&A, and retrieval-augmented generation
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from utils.pinecone_store import pinecone_store
from utils.logger import logger


class TextAgent(BaseAgent):
    """Agent specialized in text processing and document Q&A"""
    
    def __init__(self):
        super().__init__(
            name="Text Agent",
            description="document analysis, question answering, and information retrieval"
        )
        self.top_k = 5  # Number of documents to retrieve
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text query with RAG
        
        Args:
            input_data: {
                "query": str,
                "use_rag": bool (default: True),
                "top_k": int (optional),
                "filters": dict (optional)
            }
            
        Returns:
            {
                "success": bool,
                "response": str,
                "sources": list,
                "metadata": dict
            }
        """
        try:
            query = input_data.get("query", "")
            use_rag = input_data.get("use_rag", True)
            top_k = input_data.get("top_k", self.top_k)
            filters = input_data.get("filters")
            
            if not query:
                return {
                    "success": False,
                    "error": "No query provided"
                }
            
            logger.info(f"Processing text query: {query[:50]}...")
            
            # Retrieve relevant documents if RAG is enabled
            context = ""
            sources = []
            
            if use_rag:
                logger.info(f"Retrieving {top_k} relevant documents from Pinecone")
                retrieved_docs = pinecone_store.query(query, top_k=top_k, filter=filters)
                
                if retrieved_docs:
                    context = self._format_context(retrieved_docs)
                    sources = [
                        {
                            "id": doc["id"],
                            "score": doc["score"],
                            "text": doc["text"][:200] + "..."
                        }
                        for doc in retrieved_docs
                    ]
                    logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
                else:
                    logger.warning("No relevant documents found in knowledge base")
            
            # Build messages for OpenAI
            messages = [
                {"role": "system", "content": self._build_rag_system_prompt()},
                {"role": "user", "content": self._build_user_message(query, context)}
            ]
            
            # Get response from OpenAI (GPT-5 mini)
            response = self._call_openai(
                messages=messages,
                response_format={"type": "text"},
                reasoning_effort="medium",  # Leverage GPT-5's reasoning
                verbosity="medium"
            )
            
            return {
                "success": True,
                "response": response,
                "sources": sources,
                "metadata": {
                    "query": query,
                    "documents_retrieved": len(sources),
                    "rag_enabled": use_rag,
                    "model": self.model
                }
            }
            
        except Exception as e:
            logger.error(f"Error in TextAgent.process: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Document {i}] (Relevance: {doc['score']:.2f})")
            context_parts.append(doc["text"])
            context_parts.append("")  # Empty line between documents
        
        return "\n".join(context_parts)
    
    def _build_rag_system_prompt(self) -> str:
        """Build system prompt for RAG"""
        return """You are a Text Agent specialized in document analysis and question answering.

Your task is to answer user questions based on the provided context from retrieved documents.

Guidelines:
- Use information from the retrieved documents to answer questions
- If the context doesn't contain relevant information, say so clearly
- Cite document numbers when referencing information (e.g., "According to Document 2...")
- Be accurate and avoid hallucination
- If multiple documents provide different information, mention the differences
- Provide concise but complete answers

Remember: Only use information from the provided context. Do not make up information."""
    
    def _build_user_message(self, query: str, context: str) -> str:
        """Build user message with query and context"""
        if context:
            return f"""Context from relevant documents:

{context}

---

User Question: {query}

Please answer the question based on the context provided above."""
        else:
            return f"""No relevant documents were found in the knowledge base.

User Question: {query}

Please provide a general answer based on your knowledge, but clearly state that this is not based on the knowledge base."""
    
    def analyze_document_batch(self, documents: List[str], analysis_type: str = "summary") -> Dict[str, Any]:
        """
        Analyze multiple documents in batch
        
        Args:
            documents: List of document texts
            analysis_type: Type of analysis (summary, extract_key_points, compare)
            
        Returns:
            Analysis results
        """
        try:
            logger.info(f"Analyzing {len(documents)} documents with type: {analysis_type}")
            
            # Build batch analysis prompt
            doc_texts = "\n\n---\n\n".join([
                f"Document {i+1}:\n{doc[:1000]}"  # Limit to first 1000 chars per doc
                for i, doc in enumerate(documents)
            ])
            
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": f"""Analyze the following {len(documents)} documents.

Analysis Type: {analysis_type}

{doc_texts}

Provide a {analysis_type} of these documents."""}
            ]
            
            response = self._call_openai(
                messages=messages,
                max_tokens=4000,  # Increased for GPT-5 mini's larger capacity
                reasoning_effort="high"  # Use more reasoning for complex analysis
            )
            
            return {
                "success": True,
                "analysis": response,
                "documents_analyzed": len(documents),
                "analysis_type": analysis_type,
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
text_agent = TextAgent()
