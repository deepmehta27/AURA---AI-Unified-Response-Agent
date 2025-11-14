"""
LangGraph-based Multi-Agent Orchestrator
Routes queries to appropriate agents and manages workflows
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from agents.agent_state import AgentState
from agents.text_agent import text_agent
from agents.image_agent import image_agent
from utils.logger import logger
from openai import OpenAI
from config.settings import settings


class AgentOrchestrator:
    """Orchestrates multiple agents using LangGraph"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.workflow = self._build_workflow()
        logger.info("AgentOrchestrator initialized with LangGraph")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        # Create state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes (agent functions)
        workflow.add_node("classify", self._classify_query)
        workflow.add_node("text_agent", self._call_text_agent)
        workflow.add_node("image_agent", self._call_image_agent)
        workflow.add_node("multi_modal", self._call_multi_modal)
        workflow.add_node("synthesize", self._synthesize_response)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        # Add conditional edges based on classification
        workflow.add_conditional_edges(
            "classify",
            self._route_query,
            {
                "text": "text_agent",
                "image": "image_agent",
                "multi_modal": "multi_modal",
                "error": END
            }
        )
        
        # Add edges to synthesis
        workflow.add_edge("text_agent", "synthesize")
        workflow.add_edge("image_agent", "synthesize")
        workflow.add_edge("multi_modal", "synthesize")
        
        # End after synthesis
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def _classify_query(self, state: AgentState) -> AgentState:
        """Classify the query type and intent"""
        try:
            query = state["query"]
            has_image = state.get("image_path") is not None
            has_audio = state.get("audio_path") is not None
            has_document = state.get("document_path") is not None
            
            logger.info(f"Classifying query: {query[:50]}...")
            
            # Build classification prompt
            classification_prompt = f"""Classify this user request:

Query: {query}
Has Image: {has_image}
Has Audio: {has_audio}
Has Document: {has_document}

Determine:
1. Query Type: text, image, audio, or multi_modal
2. Intent: search, analyze, process, or question

Respond in this exact format:
Type: <query_type>
Intent: <intent>
Reasoning: <brief reasoning>"""

            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": classification_prompt}],
                max_completion_tokens=150,
                verbosity="low",
                reasoning_effort="low",
                store=False
            )
            
            result = response.choices[0].message.content
            
            # Parse response
            query_type = "text"  # default
            intent = "question"  # default
            
            for line in result.split("\n"):
                if line.startswith("Type:"):
                    query_type = line.split(":")[1].strip().lower()
                elif line.startswith("Intent:"):
                    intent = line.split(":")[1].strip().lower()
            
            # Override if files present
            if has_image and not has_document:
                query_type = "image"
            elif has_image and (has_document or len(query) > 20):
                query_type = "multi_modal"
            
            state["query_type"] = query_type
            state["intent"] = intent
            state["processing_steps"] = [f"Classified as {query_type} with intent {intent}"]
            
            logger.info(f"Classification: Type={query_type}, Intent={intent}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            state["error"] = str(e)
            state["query_type"] = "error"
            return state
    
    def _route_query(self, state: AgentState) -> str:
        """Route to appropriate agent based on classification"""
        query_type = state.get("query_type", "text")
        
        if state.get("error"):
            return "error"
        
        routing = {
            "text": "text",
            "image": "image",
            "audio": "text",  # Will add audio agent later
            "multi_modal": "multi_modal"
        }
        
        route = routing.get(query_type, "text")
        logger.info(f"Routing to: {route}")
        return route
    
    def _call_text_agent(self, state: AgentState) -> AgentState:
        """Call text agent"""
        try:
            logger.info("Calling Text Agent...")
            
            result = text_agent.process({
                "query": state["query"],
                "use_rag": True,
                "top_k": 5
            })
            
            if result["success"]:
                state["text_response"] = result["response"]
                state["retrieved_docs"] = result.get("sources", [])
                state["metadata"]["documents_used"] = result["metadata"]["documents_retrieved"]
            else:
                state["error"] = result.get("error")
            
            state["processing_steps"].append("Text agent processed")
            state["current_agent"] = "text"
            
            return state
            
        except Exception as e:
            logger.error(f"Error calling text agent: {str(e)}")
            state["error"] = str(e)
            return state
    
    def _call_image_agent(self, state: AgentState) -> AgentState:
        """Call image agent"""
        try:
            logger.info("Calling Image Agent...")
            
            # Determine analysis type from intent
            analysis_type_map = {
                "search": "describe",
                "analyze": "analyze",
                "process": "ocr",
                "question": "question"
            }
            analysis_type = analysis_type_map.get(state.get("intent"), "describe")
            
            result = image_agent.process({
                "image_path": state.get("image_path"),
                "analysis_type": analysis_type,
                "query": state["query"] if analysis_type == "question" else ""
            })
            
            if result["success"]:
                state["image_analysis"] = {
                    "response": result["response"],
                    "analysis_type": result["analysis_type"],
                    "metadata": result["metadata"]
                }
            else:
                state["error"] = result.get("error")
            
            state["processing_steps"].append("Image agent processed")
            state["current_agent"] = "image"
            
            return state
            
        except Exception as e:
            logger.error(f"Error calling image agent: {str(e)}")
            state["error"] = str(e)
            return state
    
    def _call_multi_modal(self, state: AgentState) -> AgentState:
        """Handle multi-modal queries (text + image)"""
        try:
            logger.info("Processing multi-modal query...")
            
            # Call both agents
            state = self._call_image_agent(state)
            state = self._call_text_agent(state)
            
            state["processing_steps"].append("Multi-modal processing completed")
            state["current_agent"] = "multi_modal"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in multi-modal processing: {str(e)}")
            state["error"] = str(e)
            return state
    
    def _synthesize_response(self, state: AgentState) -> AgentState:
        """Synthesize final response from all agent outputs"""
        try:
            logger.info("Synthesizing final response...")
            
            # Collect all responses
            parts = []
            sources = []
            
            if state.get("text_response"):
                parts.append(f"**Document Analysis:**\n{state['text_response']}")
                if state.get("retrieved_docs"):
                    sources.extend(state["retrieved_docs"])
            
            if state.get("image_analysis"):
                img_response = state["image_analysis"]["response"]
                analysis_type = state["image_analysis"]["analysis_type"]
                parts.append(f"**Image Analysis ({analysis_type}):**\n{img_response}")
            
            if state.get("audio_transcript"):
                parts.append(f"**Audio Transcript:**\n{state['audio_transcript']}")
            
            # Combine responses
            if len(parts) > 1:
                # Multi-modal: synthesize with GPT
                synthesis_prompt = f"""Combine these analysis results into a coherent response to the user's question: "{state['query']}"

{chr(10).join(parts)}

Provide a unified, clear answer that integrates all information."""

                response = self.client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    max_completion_tokens=500,
                    verbosity="medium",
                    reasoning_effort="medium",
                    store=False
                )
                
                final_response = response.choices[0].message.content
            else:
                # Single agent: use direct response
                final_response = parts[0] if parts else "No response generated."
            
            state["final_response"] = final_response
            state["sources"] = sources
            state["processing_steps"].append("Response synthesized")
            
            logger.info("Synthesis complete")
            
            return state
            
        except Exception as e:
            logger.error(f"Error synthesizing response: {str(e)}")
            state["error"] = str(e)
            state["final_response"] = f"Error: {str(e)}"
            return state
    
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process a query through the agent workflow
        
        Args:
            query: User query
            **kwargs: Optional image_path, audio_path, document_path
            
        Returns:
            Final result dictionary
        """
        try:
            # Initialize state
            initial_state = AgentState(
                query=query,
                image_path=kwargs.get("image_path"),
                audio_path=kwargs.get("audio_path"),
                document_path=kwargs.get("document_path"),
                query_type=None,
                intent=None,
                current_agent=None,
                processing_steps=[],
                text_response=None,
                image_analysis=None,
                audio_transcript=None,
                retrieved_docs=None,
                confidence=None,
                error=None,
                metadata={},
                final_response=None,
                sources=None
            )
            
            # Run workflow
            logger.info(f"Starting workflow for query: {query[:50]}...")
            final_state = self.workflow.invoke(initial_state)
            
            # Return result
            return {
                "success": final_state.get("error") is None,
                "response": final_state.get("final_response"),
                "sources": final_state.get("sources", []),
                "metadata": {
                    "query_type": final_state.get("query_type"),
                    "intent": final_state.get("intent"),
                    "agents_used": final_state.get("current_agent"),
                    "processing_steps": final_state.get("processing_steps", []),
                    **final_state.get("metadata", {})
                },
                "error": final_state.get("error")
            }
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }


# Global instance
orchestrator = AgentOrchestrator()
