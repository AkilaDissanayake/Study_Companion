# chatbot.py
"""
LangGraph Application Brain and Orchestration Engine.

Defines the multi model pipeline structure, state attributes, processing nodes, 
and conditional flow control edges that execute the intelligent Study Companion agent.
This implements a Corrective RAG (CRAG) architecture using a local Cross-Encoder.
"""

import json
import math
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from sentence_transformers import CrossEncoder

from utils.tools import tools
from utils.prompts import (
    REWRITER_PROMPT, 
    CLASSIFIER_PROMPT, 
    SAFETY_PROMPT, 
    COMPOSER_PROMPT
)
from utils.token_manager import log_token_usage
from utils.logger import get_logger

# Initialize logger for debugging
logger = get_logger(__name__, "chatbot.log")

# Load the local Grader Model globally. 
# Doing this outside a function ensures it only loads into memory once when the server starts. 
# May have to upgrade later to more complex models if the grading accuracy is insufficient.
grader_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


#  Background Token Manager 
class TokenTrackingCallbackHandler(BaseCallbackHandler):
    """
    Listens passively for LLM completion events to decouple token tracking from core node logic.
    This guarantees tokens are logged to the database without slowing down the chatbot.
    """
    def __init__(self, user_id: str, model_name: str):
        self.user_id = user_id
        self.model_name = model_name

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            # Extract the hidden OpenAI metadata receipt
            llm_output = response.generations[0][0].message.response_metadata
            
            if "token_usage" in llm_output:
                prompt_tokens = llm_output["token_usage"].get("prompt_tokens", 0)
                completion_tokens = llm_output["token_usage"].get("completion_tokens", 0)
                
                # Send the numbers to the database logger
                log_token_usage(
                    user_id=self.user_id,
                    model_name=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
        except Exception as e:
            # If it fails, log the error but do not crash the chat
            logger.warning(f"Failed to get token usage with error {e}")
            

class AgentState(TypedDict):
    """
    The 'Clipboard' of the application. 
    Every node reads from and writes to this dictionary as it passes down the assembly line.
    """
    user_id: str
    chat_history: str
    raw_question: str
    rewritten_question: str
    subject: str
    detail_level: str
    needs_tools: bool
    is_safe: bool
    safety_reason: str
    
    # CRAG Retrieval State Variables
    context: str
    confidence_score: float
    is_answerable: bool
    status: str
    
    # Output Variables
    raw_data: str          
    final_response: str


#  Model Initialization 
# fast_llm: Temp 0 makes it a strict, predictable robot for routing and JSON tasks.
fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# tutor_llm: Temp 0.3 allows slight conversational variety for teaching without hallucinating.
tutor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# heavy_llm: Temp 0.7 allows complex reasoning when synthesizing custom tools (graphs/images).
heavy_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


#  Node Definitions 

def rewrite_node(state: AgentState):
    """Combines the chat history and the current question into one standalone query."""
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-rewriter")
    chain = REWRITER_PROMPT | fast_llm.with_config({"callbacks": [tracker]})
    
    response = chain.invoke({
        "chat_history": state.get("chat_history", ""),
        "raw_question": state["raw_question"]
    })
    return {"rewritten_question": response.content}


def classify_node(state: AgentState):
    """Analyzes the question to determine the subject, detail level, and if tools are needed."""
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-classifier")
    # Bind JSON format to force the LLM to return strict computer-readable data
    llm_json = fast_llm.bind(response_format={"type": "json_object"}).with_config({"callbacks": [tracker]})
    chain = CLASSIFIER_PROMPT | llm_json
    
    response = chain.invoke({"rewritten_question": state["rewritten_question"]})

    try:
        data = json.loads(response.content)
        return {
            "subject": data.get("subject", "General"),
            "detail_level": data.get("detail_level", "concise"),
            "needs_tools": data.get("needs_tools", False)
        }
    except json.JSONDecodeError:
        # Fallback defaults if the LLM messes up the JSON formatting
        return {"subject": "General", "detail_level": "detailed", "needs_tools": False}


def safety_node(state: AgentState):
    """Acts as a firewall to block inappropriate or malicious prompts."""
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-safety")
    llm_json = fast_llm.bind(response_format={"type": "json_object"}).with_config({"callbacks": [tracker]})
    chain = SAFETY_PROMPT | llm_json
    
    response = chain.invoke({"rewritten_question": state["rewritten_question"]})
    
    try:
        data = json.loads(response.content)
        return {
            "is_safe": data.get("is_safe", True),
            "safety_reason": data.get("reason", "safe")
        }
    except json.JSONDecodeError:
        return {"is_safe": True, "safety_reason": "default assumed safe"}


def retrieval_node(state: AgentState):
    """Pulls ground-truth facts from the ChromaDB vector database."""
    # Imported inside the function to prevent circular dependency errors with vdb_handler
    from utils.vdb_handler import search_vdb
    
    retrieved_text = search_vdb(
        user_id=state["user_id"], 
        subject=state["subject"], 
        query=state["rewritten_question"]
    )
    
    return {"context": retrieved_text}


def grade_context_node(state: AgentState):
    """Scores the retrieved text to ensure it actually contains the answer (Zero API cost)."""
    query = state['rewritten_question']
    context = state.get('context', '')
    
    # Run the fast local transformer model. Cast result to float for math operations.
    raw_score = float(grader_model.predict([query, context]))
    
    # Pass the raw logit through a Sigmoid function to convert it to a 0-100 percentage
    confidence_score = (1 / (1 + math.exp(-raw_score))) * 100
    
    # If the score is 70% or higher, it is safe to let the LLM answer.
    if confidence_score >= 70:
        is_answerable = True
        status = "CORRECT"
    else:
        is_answerable = False
        status = "INCORRECT"
        
    return {
        "confidence_score": round(confidence_score, 2),
        "is_answerable": is_answerable,
        "status": status
    }


def domain_tutor_node(state: AgentState):
    """Generates the educational answer based STRICTLY on the ChromaDB context."""
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-tutor")
    llm_tracked = tutor_llm.with_config({"callbacks": [tracker]})
    
    # Inject the database context so the LLM reads it before answering
    prompt = f"""
    Use the following retrieved context to answer the student's question accurately.
    Context: {state.get('context', 'No context available.')}
    
    Question: Provide a {state.get('detail_level', 'detailed')} explanation for this {state.get('subject', 'General')} topic: {state.get('rewritten_question')}
    """
    
    response = llm_tracked.invoke(prompt)
    return {"raw_data": response.content}


def tool_execution_node(state: AgentState):
    """Allows the heavy LLM to execute custom Python tools (like DrawGraph)."""
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-tools")
    # Bind the tools list so the LLM knows what functions it is allowed to trigger
    cloud_with_tools = heavy_llm.bind_tools(tools).with_config({"callbacks": [tracker]})
    
    response = cloud_with_tools.invoke(state["rewritten_question"])
    
    # Extract the tool output data
    raw_output = response.content if response.content else str(response.tool_calls)
    return {"raw_data": raw_output}


def answer_composer_node(state: AgentState):
    """Formats the final text and handles apologies if safety/grading failed."""
    
    # 1. Fast-Fail: The prompt was malicious
    if not state.get("is_safe", True):
        override_data = f"Safety Violation Flagged: {state.get('safety_reason')}. Reject the request respectfully."
        
    # 2. Fast-Fail: The database didn't have the answer
    elif not state.get("is_answerable", True):
        override_data = "System Note: The retrieved textbook context does not contain the answer. Politely apologize to the student and inform them that this information is not in their uploaded materials."
        
    # 3. Success: Format the raw data from the Tutor or Tools
    else:
        override_data = state.get("raw_data", "No data provided.")
        
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-composer")
    chain = COMPOSER_PROMPT | fast_llm.with_config({"callbacks": [tracker]})
    
    response = chain.invoke({
        "raw_data": override_data,
        "rewritten_question": state["rewritten_question"]
    })
    return {"final_response": response.content}


# --- Routing Edge Logic ---

def route_safety(state: AgentState):
    """Directs flow based on the safety check."""
    if not state.get("is_safe", True):
        return "answer_composer"  # Skip to the end and reject
    return "retrieval_node"       # Proceed to database search

def route_after_grading(state: AgentState):
    """Directs flow based on the CRAG context score and tool requirements."""
    if not state.get("is_answerable", False):
        return "answer_composer"  # Skip generation and apologize
    
    # If the context has the answer, decide who writes it
    if state.get("needs_tools"):
        return "tool_execution"
    return "domain_tutor"


# --- Workflow Graph Orchestration Settings ---
workflow = StateGraph(AgentState)

# 1. Register all nodes to the graph
workflow.add_node("rewriter", rewrite_node)
workflow.add_node("classifier", classify_node)
workflow.add_node("safety", safety_node)
workflow.add_node("retrieval_node", retrieval_node)
workflow.add_node("grader_node", grade_context_node)
workflow.add_node("tool_execution", tool_execution_node)
workflow.add_node("domain_tutor", domain_tutor_node)
workflow.add_node("answer_composer", answer_composer_node)

# 2. Define the Linear Edges (The guaranteed path)
workflow.set_entry_point("rewriter")
workflow.add_edge("rewriter", "classifier")
workflow.add_edge("classifier", "safety")

# 3. Define Conditional Edges (The Crossroads)
# Check Safety
workflow.add_conditional_edges(
    "safety",
    route_safety,
    {
        "answer_composer": "answer_composer",
        "retrieval_node": "retrieval_node"
    }
)

# After retrieval, grade the context
workflow.add_edge("retrieval_node", "grader_node")

# Check Grades and Needs
workflow.add_conditional_edges(
    "grader_node",
    route_after_grading,
    {
        "answer_composer": "answer_composer",  # INCORRECT path
        "tool_execution": "tool_execution",    # CORRECT + Tools needed
        "domain_tutor": "domain_tutor"         # CORRECT + Normal explanation
    }
)

# 4. Convergence (All successful paths end at the composer)
workflow.add_edge("tool_execution", "answer_composer")
workflow.add_edge("domain_tutor", "answer_composer")
workflow.add_edge("answer_composer", END)

# Compile the final application
ChatBot = workflow.compile()