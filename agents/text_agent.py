"""
Text Agent for AURA - Enhanced with LangChain
Handles text-based queries, document Q&A, and retrieval-augmented generation
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from utils.pinecone_store import pinecone_store
from utils.logger import logger

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils.embedding_wrapper import langchain_embeddings  
from pinecone import Pinecone
from config.settings import settings


class TextAgent(BaseAgent):
    """Agent specialized in text processing and document Q&A with LangChain"""
    
    def __init__(self):
        super().__init__(
            name="Text Agent",
            description="document analysis, question answering, and information retrieval"
        )
        self.top_k = 5
        
        # Initialize LangChain components
        self._init_langchain()
        
        logger.info("Text Agent initialized with LangChain RAG")
    
    def _init_langchain(self):
        """Initialize LangChain components"""
        try:
            # Initialize ChatOpenAI
            self.llm = ChatOpenAI(
                model="gpt-5-mini",
                max_tokens=2000,
                openai_api_key=settings.openai_api_key,
            )
            
            # Initialize Pinecone for LangChain
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.get('pinecone.index_name'))
            
            # Create LangChain vector store
            self.vectorstore = PineconeVectorStore(
                index=index,
                embedding=langchain_embeddings,
                text_key="text"
            )
            
            # Create conversation memory using ChatMessageHistory
            self.memory = ChatMessageHistory()
            self.memory_key = "chat_history"
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            
            # Create custom prompt
            self.prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a Text Agent specialized in document analysis and question answering.

    Use the following context from retrieved documents to answer the question:

    Context:
    {context}

    Question: {question}

    Guidelines:
    - Use information from the context to answer accurately
    - Cite document numbers when referencing information
    - If context doesn't contain relevant information, say so clearly
    - Be precise and avoid hallucination
    - Provide concise but complete answers

    Answer:"""
            )
            
            # Simple RAG chain using LCEL (LangChain Expression Language)
            
            def format_docs(docs):
                return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
            
            self.rag_chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("LangChain components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LangChain: {str(e)}")
            # Fallback to basic mode
            self.llm = None
            self.vectorstore = None
            self.rag_chain = None
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text query with LangChain RAG
        
        Args:
            input_data: {
                "query": str,
                "use_rag": bool (default: True),
                "top_k": int (optional),
                "use_memory": bool (default: False),
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
            use_memory = input_data.get("use_memory", False)
            
            if not query:
                return {
                    "success": False,
                    "error": "No query provided"
                }
            
            logger.info(f"Processing text query: {query[:50]}...")
            
            # Use LangChain RAG if available and enabled
            if use_rag and self.rag_chain:
                return self._process_with_langchain(query, use_memory)
            elif use_rag:
                # Fallback to custom RAG
                return self._process_with_custom_rag(query, input_data)
            else:
                # No RAG - direct query
                return self._process_direct(query)
            
        except Exception as e:
            logger.error(f"Error in TextAgent.process: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _process_with_langchain(self, query: str, use_memory: bool) -> Dict[str, Any]:
        """Process query using LangChain RAG chain"""
        try:
            logger.info("Using LangChain RAG chain")
            
            # Clear memory if not using it
            if not use_memory:
                self.memory.clear()
            
            # Query the chain
            response = self.rag_chain.invoke(query)
            
            # Get source documents from retriever
            source_docs = self.retriever.get_relevant_documents(query)
            
            # Extract sources
            sources = []
            for i, doc in enumerate(source_docs):
                sources.append({
                    "id": doc.metadata.get("id", f"doc-{i}"),
                    "score": doc.metadata.get("score", 0.0),
                    "text": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
            
            return {
                "success": True,
                "response": response,
                "sources": sources,
                "metadata": {
                    "query": query,
                    "documents_retrieved": len(sources),
                    "rag_enabled": True,
                    "method": "langchain",
                    "model": self.model,
                    "memory_used": use_memory
                }
            }
            
        except Exception as e:
            logger.error(f"Error in LangChain processing: {str(e)}")
            # Fallback to custom RAG
            return self._process_with_custom_rag(query, {"top_k": self.top_k})

    
    def _process_with_custom_rag(self, query: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback: Process with custom RAG (original implementation)"""
        try:
            logger.info("Using custom RAG (LangChain fallback)")
            
            top_k = input_data.get("top_k", self.top_k)
            filters = input_data.get("filters")
            
            # Retrieve documents
            retrieved_docs = pinecone_store.query(query, top_k=top_k, filter=filters)
            
            context = ""
            sources = []
            
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
            
            # Build messages
            messages = [
                {"role": "system", "content": self._build_rag_system_prompt()},
                {"role": "user", "content": self._build_user_message(query, context)}
            ]
            
            # Get response
            response = self._call_openai(
                messages=messages,
                response_format={"type": "text"},
                reasoning_effort="medium",
                verbosity="medium"
            )
            
            return {
                "success": True,
                "response": response,
                "sources": sources,
                "metadata": {
                    "query": query,
                    "documents_retrieved": len(sources),
                    "rag_enabled": True,
                    "method": "custom",
                    "model": self.model
                }
            }
            
        except Exception as e:
            logger.error(f"Error in custom RAG: {str(e)}")
            raise
    
    def _process_direct(self, query: str) -> Dict[str, Any]:
        """Process query without RAG"""
        try:
            logger.info("Processing without RAG")
            
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": query}
            ]
            
            response = self._call_openai(
                messages=messages,
                response_format={"type": "text"}
            )
            
            return {
                "success": True,
                "response": response,
                "sources": [],
                "metadata": {
                    "query": query,
                    "documents_retrieved": 0,
                    "rag_enabled": False,
                    "model": self.model
                }
            }
            
        except Exception as e:
            logger.error(f"Error in direct processing: {str(e)}")
            raise
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Document {i}] (Relevance: {doc['score']:.2f})")
            context_parts.append(doc["text"])
            context_parts.append("")
        
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
            
            # Use LangChain if available
            if self.llm:
                # Convert to LangChain documents
                langchain_docs = [Document(page_content=doc) for doc in documents]
                
                prompt = f"Provide a {analysis_type} of the following documents:\n\n"
                prompt += "\n\n---\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
                
                response = self.llm.invoke(prompt)
                analysis = response.content
            else:
                # Fallback to custom implementation
                doc_texts = "\n\n---\n\n".join([
                    f"Document {i+1}:\n{doc[:1000]}"
                    for i, doc in enumerate(documents)
                ])
                
                messages = [
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": f"""Analyze the following {len(documents)} documents.

Analysis Type: {analysis_type}

{doc_texts}

Provide a {analysis_type} of these documents."""}
                ]
                
                analysis = self._call_openai(
                    messages=messages,
                    max_tokens=4000,
                    reasoning_effort="high"
                )
            
            return {
                "success": True,
                "analysis": analysis,
                "documents_analyzed": len(documents),
                "analysis_type": analysis_type,
                "method": "langchain" if self.llm else "custom",
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")


# Global instance
text_agent = TextAgent()
