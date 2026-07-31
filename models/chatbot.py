# chatbot.py
"""
LangGraph Application Brain and Orchestration Engine.

Defines the multi-model pipeline structure, state attributes, processing nodes, 
and conditional flow control edges that execute the intelligent Study Companion agent.
Implements full Corrective RAG (CRAG) with Web Search Fallback and Dynamic Context.
"""

import json
import math
import asyncio
import inspect
import re
import os
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from sentence_transformers import CrossEncoder
from langchain_community.tools import DuckDuckGoSearchRun
from utils.vdb_handler import search_vdb
from utils.tools import tools
from utils.json_handler import read_config
from utils.prompts import (
    REWRITER_PROMPT, 
    CLASSIFIER_PROMPT, 
    SAFETY_PROMPT,
    GREETING_PROMPT, 
    DOMAIN_TUTOR_PROMPT,
    COMPOSER_PROMPT
)
from utils.token_manager import log_token_usage
from utils.web_search import search_and_grade_web
from utils.logger import get_logger
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
logger = get_logger(__name__, "chatbot.log")

# Load the local Grader Model globally. 
logger.info(f"Initializing CRAG Grader Model on device: {EMBEDDING_DEVICE}")
grader_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=EMBEDDING_DEVICE)

class TokenTrackingCallbackHandler(BaseCallbackHandler):
    """Listens passively for LLM completion events to decouple token tracking."""
    def __init__(self, user_id: str, model_name: str):
        self.user_id = user_id
        self.model_name = model_name

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            llm_output = response.generations[0][0].message.response_metadata
            if "token_usage" in llm_output:
                prompt_tokens = llm_output["token_usage"].get("prompt_tokens", 0)
                completion_tokens = llm_output["token_usage"].get("completion_tokens", 0)
                
                log_token_usage(
                    user_id=self.user_id,
                    model_name=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
        except Exception as e:
            logger.warning(f"Failed to get token usage with error {e}")
            

class AgentState(TypedDict):
    """The 'Clipboard' of the application."""
    user_id: str
    chat_history: str
    raw_question: str
    rewritten_question: str
    subject: str
    detail_level: str
    needs_tools: bool
    needs_documents:bool
    is_safe: bool
    safety_reason: str
    
    # CRAG & Dynamic Context State Variables
    retrieved_chunks: list[str]  
    context: str                 
    confidence_score: float      
    is_answerable: bool          
    status: str                  # "CORRECT", "AMBIGUOUS", or "INCORRECT"
    
    # Output Variables
    raw_data: str          
    final_response: str


# --- Model Initialization ---
fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tutor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
heavy_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


# --- Node Definitions ---

async def rewrite_node(state: AgentState):

    logger.info(f"Rewriting user question for clarity: {state['raw_question']}")
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-rewriter")
    chain = REWRITER_PROMPT | fast_llm.with_config({"callbacks": [tracker]})
    
    response = await chain.ainvoke({
        "chat_history": state.get("chat_history", ""),
        "raw_question": state["raw_question"]
    })
    logger.debug(f"Rewritten question: {response.content}")
    return {"rewritten_question": response.content}


async def classify_node(state: AgentState):
    logger.info(f"Classifying user question.")
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-classifier")
    llm_json = fast_llm.bind(response_format={"type": "json_object"}).with_config({"callbacks": [tracker]})
    chain = CLASSIFIER_PROMPT | llm_json
    
    #Get the subject list
    user_id = state["user_id"]
    config_file = f"{user_id}.json"
    user_config = read_config(config_file, default_fallback={})
    subjects = list(user_config.get("subjects", []))

    if "General" not in subjects:
        subjects.append("General")
        
    subjects_string = ", ".join(subjects)
    # dynamically pull the names and descriptions of your actual tools!
    if tools:
        tools_desc = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    else:
        tools_desc = "No external tools currently available."
    
    # Inject both the question AND the tool list into the prompt
    response = await chain.ainvoke({
        "rewritten_question": state["rewritten_question"],
        "tools_description": tools_desc,
        "available_subjects": subjects_string
    })
    
    try:
        data = json.loads(response.content)
        logger.debug(f"Classification result: {data}")
        return {
            "subject": data.get("subject", "General"),
            "detail_level": data.get("detail_level", "concise"),
            "needs_tools": data.get("needs_tools", False),
            "needs_documents": data.get("needs_documents", True)
        }
    except json.JSONDecodeError:
        return {"subject": "General", "detail_level": "detailed", "needs_tools": False, "needs_documents": True}


async def safety_node(state: AgentState):

    logger.info(f"Performing safety check on question")
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-safety")
    llm_json = fast_llm.bind(response_format={"type": "json_object"}).with_config({"callbacks": [tracker]})
    chain = SAFETY_PROMPT | llm_json
    
    response = await chain.ainvoke({"rewritten_question": state["rewritten_question"]})
    try:
        data = json.loads(response.content)
        logger.debug(f"Safety check result: {data}")
        return {
            "is_safe": data.get("is_safe", True),
            "safety_reason": data.get("reason", "safe")
        }
    except json.JSONDecodeError:
        return {"is_safe": True, "safety_reason": "default assumed safe"}

async def greeting_node(state: AgentState):
    """Handles casual conversation and greetings natively without RAG."""
    logger.info(f"Routing to Greeting Node for input: {state.get('rewritten_question')}")
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-greeting")
    
    chain = GREETING_PROMPT | fast_llm.with_config({"callbacks": [tracker]})
    
    response = await chain.ainvoke({"rewritten_question": state["rewritten_question"]})
    
    logger.info(f"Greeting generated: {response.content}")
    return {"final_response": response.content}


async def retrieval_node(state: AgentState):
    """Pulls formatted context chunks (with sources) from the Vector DB."""
    
    logger.info(f"Retrieving context from Vector DB for question")
    if state.get("subject") == "General":
        logger.warning("Subject is 'General'; retrieval may yield broad results.")
        state["subject"] = "root"  
        
    # search_vdb now returns the strings PRE-FORMATTED with [Source: filename.pdf]
    res = await asyncio.to_thread(
        search_vdb,
        user_id=state["user_id"], 
        subject=state["subject"], 
        query=state["rewritten_question"]
    )
    
    chunks = res if isinstance(res, list) else [res] if res else []
    logger.debug(f"Retrieved chunks: {chunks}")
    
    return {"retrieved_chunks": chunks}


async def grade_context_node(state: AgentState):
    """Full CRAG Evaluator: Implements the 3-tier Correct/Ambiguous/Incorrect logic."""

    logger.info(f"Grading retrieved context")
    query = state['rewritten_question']
    raw_chunks = state.get('retrieved_chunks', [])
    
    # NEW: Create a synchronous helper function for the heavy math
    def _evaluate_chunks():
        filtered = []
        highest_score = 0.0
        
        for chunk in raw_chunks:
            if not chunk.strip():
                continue
                
            raw_score = float(grader_model.predict([query, chunk]))
            
            try:
                confidence = (1 / (1 + math.exp(-raw_score))) * 100
            except OverflowError:
                confidence = 0.0 if raw_score < 0 else 100.0
                
            if confidence > highest_score:
                highest_score = confidence
            
            if confidence >= 30.0:
                filtered.append(chunk)
                
        return filtered, highest_score

    # NEW: Run the heavy CPU math on a background thread!
    filtered_chunks, max_score = await asyncio.to_thread(_evaluate_chunks)
    
    logger.debug(f"Filtered chunks: {filtered_chunks} with max confidence score: {max_score}")
    
    # 3-Tier CRAG Decision Gate
    if max_score >= 80.0:
        status = "CORRECT"
        is_answerable = True
    elif max_score <= 30.0:
        status = "INCORRECT"
        is_answerable = False  
    else:
        status = "AMBIGUOUS"
        is_answerable = True
        
    final_context = "\n\n---\n\n".join(filtered_chunks)
    
    return {
        "context": final_context,
        "confidence_score": round(max_score, 2),
        "is_answerable": is_answerable,
        "status": status
    }

async def web_search_node(state: AgentState):
    """CRAG Fallback: Fetches and refines external knowledge when the database fails."""
    
    logger.info("Routing to external web search node.")
    
    # NEW: Push the blocking internet request to a background thread!
    formatted_web_data, is_rescued = await asyncio.to_thread(
        search_and_grade_web,
        query=state["rewritten_question"], 
        grader_model=grader_model
    )
    if state.get("status") == "AMBIGUOUS":
        new_context = state.get("context", "") + "\n\n" + formatted_web_data
    else:
        new_context = formatted_web_data
        
    return {
        "context": new_context,
        "is_answerable": is_rescued if state.get("status") == "INCORRECT" else True
    }


async def domain_tutor_node(state: AgentState):
    """Generates the educational answer based on the refined CRAG context."""
    logger.info(f"Generating educational answer for question: {state['rewritten_question']}")
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-tutor")
    
    # Bind the tracker to the LLM
    llm_tracked = tutor_llm.with_config({"callbacks": [tracker]})
    
    # Chain the prompt and the LLM together
    chain = DOMAIN_TUTOR_PROMPT | llm_tracked
    
    # Invoke the chain by passing the state variables as a dictionary
    response =  await chain.ainvoke({
        "context": state.get('context', 'No context available.'),
        "detail_level": state.get('detail_level', 'detailed'),
        "subject": state.get('subject', 'General'),
        "rewritten_question": state.get('rewritten_question', '')
    })
    
    return {"raw_data": response.content}


async def tool_execution_node(state: AgentState): #async put this out of main event loop
    """
    Executes multiple requested tools concurrently in parallel using asyncio.gather.
    Includes user_id injection for workspace management tools.
    """
    logger.info(f"Executing external tools for question: {state['rewritten_question']}")
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-tools")
    llm_with_tools = heavy_llm.bind_tools(tools).with_config({"callbacks": [tracker]})
    
    # Asynchronous LLM call
    response = await llm_with_tools.ainvoke(state["rewritten_question"])
    
    if not response.tool_calls:
        logger.info("LLM decided not to use a tool and provided a direct response.")
        return {"raw_data": response.content}
        
    tool_map = {tool.name: tool for tool in tools}
    
    async def run_single_tool(tool_call):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"].copy()
        
        #  Use LangChain's native args_schema instead of inspect
        tool_instance = tool_map.get(tool_name)
        if tool_instance:
            # Check if 'user_id' is a declared parameter in the tool's Pydantic schema
            if "user_id" in tool_instance.args_schema.__fields__:
                tool_args["user_id"] = state["user_id"]
                
        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
        
        if tool_instance:
            try:
                # Execute asynchronously
                result = await tool_instance.ainvoke(tool_args)
                return f"[Output from {tool_name}]:\n{result}"
            except Exception as e:
                logger.error(f"Tool {tool_name} execution failed: {e}")
                return f"[Error from {tool_name}]:\nExecution failed - {e}"
        else:
            return f"[Error]: Tool '{tool_name}' does not exist."
    # Execute all selected tools simultaneously across the event loop
    tool_outputs = await asyncio.gather(*[run_single_tool(call) for call in response.tool_calls])
    
    final_raw_output = "\n\n".join(tool_outputs)
    return {"raw_data": final_raw_output}


async def answer_composer_node(state: AgentState):
    """Formats the final text natively using the fast LLM, handling standard URLs."""
    logger.info(f"Composing final answer for question: {state['rewritten_question']}")
    
    if not state.get("is_safe", True):
        override_data = f"Safety Violation Flagged: {state.get('safety_reason')}. Reject the request respectfully."
        
    elif not state.get("is_answerable", True):
        override_data = "System Note: Both the textbook database and the external web search failed to find an answer. Politely apologize to the student."
        
    else:
        # override_data now contains clean strings like "![Graph](http://localhost...)"
        override_data = state.get("raw_data", "No data provided.")
        
    logger.debug(f"Final raw data to compose: {override_data}")
    # Because we use standard URLs now, the fast LLM handles this easily and cheaply.
    tracker = TokenTrackingCallbackHandler(state["user_id"], "fast-composer")
    chain = COMPOSER_PROMPT | fast_llm.with_config({"callbacks": [tracker]})
    
    # Send the raw data directly to the LLM
    response = await chain.ainvoke({
        "raw_data": override_data,
        "rewritten_question": state["rewritten_question"]
    })
    
    logger.info("Final answer generated successfully.")
    
    return {"final_response": response.content}

# --- Routing Edge Logic ---
#Async not needed since simple operations, no blocking I/O
def route_safety(state: AgentState):
    """Decides if we need to bypass the database and go straight to tools or answer composition."""

    logger.info(f"Routing after safety check: is_safe={state.get('is_safe')}, needs_tools={state.get('needs_tools')}, needs_documents={state.get('needs_documents')}")
    if not state.get("is_safe", True):
        return "answer_composer"

    if not state.get("needs_documents") and not state.get("needs_tools"):
        logger.info("Conversational query detected. Routing to greeting_node.")
        return "greeting_node"

    if state.get("needs_tools") and not state.get("needs_documents"):
        return "tool_execution"
    return "retrieval_node"

def route_after_grading(state: AgentState):
    """Decides if we need to supplement/replace the database with a Web Search."""

    logger.info(f"Routing after context grading: status={state.get('status')}, needs_tools={state.get('needs_tools')}")
    status = state.get("status", "INCORRECT")
    
    # If the database was INCORRECT or AMBIGUOUS, hit the web
    if status in ["INCORRECT", "AMBIGUOUS"]:
        return "web_search_node"
    
    # If the database was PERFECT, jump straight to execution
    if state.get("needs_tools"):
        return "tool_execution"
    return "domain_tutor"

def route_execution(state: AgentState):
    """Directs traffic to tools or tutor after contexts are finalized."""

    logger.info(f"Routing after context finalization: needs_tools={state.get('needs_tools')}")
    if state.get("needs_tools"):
        return "tool_execution"
    return "domain_tutor"


# --- Workflow Graph Orchestration ---
workflow = StateGraph(AgentState)

#  Register all nodes
workflow.add_node("rewriter", rewrite_node)
workflow.add_node("classifier", classify_node)
workflow.add_node("safety", safety_node)
workflow.add_node("greeting_node", greeting_node)
workflow.add_node("retrieval_node", retrieval_node)
workflow.add_node("grader_node", grade_context_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("tool_execution", tool_execution_node)
workflow.add_node("domain_tutor", domain_tutor_node)
workflow.add_node("answer_composer", answer_composer_node)

# Linear Entry Flow
workflow.set_entry_point("rewriter")
workflow.add_edge("rewriter", "classifier")
workflow.add_edge("classifier", "safety")

#  Safety Check Router
workflow.add_conditional_edges(
    "safety",
    route_safety,
    {   "greeting_node": "greeting_node",
        "answer_composer": "answer_composer",
        "tool_execution": "tool_execution",
        "retrieval_node": "retrieval_node"
    }
)

workflow.add_edge("greeting_node", END)
workflow.add_edge("retrieval_node", "grader_node")

#  CRAG Router
workflow.add_conditional_edges(
    "grader_node",
    route_after_grading,
    {
        "web_search_node": "web_search_node",  # INCORRECT / AMBIGUOUS
        "tool_execution": "tool_execution",    # CORRECT + Tools
        "domain_tutor": "domain_tutor"         # CORRECT + Normal
    }
)

#  Post-Web Search Router
workflow.add_conditional_edges(
    "web_search_node",
    route_execution,
    {
        "tool_execution": "tool_execution",
        "domain_tutor": "domain_tutor"
    }
)

#  Convergence
workflow.add_edge("tool_execution", "answer_composer")
workflow.add_edge("domain_tutor", "answer_composer")
workflow.add_edge("answer_composer", END)

ChatBot = workflow.compile()
# Visualization Export
try:
    with open("crag_architecture.png", "wb") as f:
        f.write(ChatBot.get_graph().draw_mermaid_png())
    logger.info("Successfully saved LangGraph architecture as crag_architecture.png")
except Exception as e:
    logger.warning(f"Could not save graph PNG: {e}")