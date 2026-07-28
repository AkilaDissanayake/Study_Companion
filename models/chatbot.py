# chatbot.py
"""
LangGraph Application Brain and Orchestration Engine.

Defines the multi-model pipeline structure, state attributes, processing nodes, 
and conditional flow control edges that execute the intelligent Study Companion agent.
Implements full Corrective RAG (CRAG) with Web Search Fallback and Dynamic Context.
"""

import json
import math
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
from utils.logger import get_logger

logger = get_logger(__name__, "chatbot.log")

# Load the local Grader Model globally. 
grader_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


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

def rewrite_node(state: AgentState):

    logger.info(f"Rewriting user question for clarity: {state['raw_question']}")
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-rewriter")
    chain = REWRITER_PROMPT | fast_llm.with_config({"callbacks": [tracker]})
    
    response = chain.invoke({
        "chat_history": state.get("chat_history", ""),
        "raw_question": state["raw_question"]
    })
    logger.debug(f"Rewritten question: {response.content}")
    return {"rewritten_question": response.content}


def classify_node(state: AgentState):
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
    response = chain.invoke({
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


def safety_node(state: AgentState):

    logger.info(f"Performing safety check on question")
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-safety")
    llm_json = fast_llm.bind(response_format={"type": "json_object"}).with_config({"callbacks": [tracker]})
    chain = SAFETY_PROMPT | llm_json
    
    response = chain.invoke({"rewritten_question": state["rewritten_question"]})
    try:
        data = json.loads(response.content)
        logger.debug(f"Safety check result: {data}")
        return {
            "is_safe": data.get("is_safe", True),
            "safety_reason": data.get("reason", "safe")
        }
    except json.JSONDecodeError:
        return {"is_safe": True, "safety_reason": "default assumed safe"}

def greeting_node(state: AgentState):
    """Handles casual conversation and greetings natively without RAG."""
    logger.info(f"Routing to Greeting Node for input: {state.get('rewritten_question')}")
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-greeting")
    
    chain = GREETING_PROMPT | fast_llm.with_config({"callbacks": [tracker]})
    
    response = chain.invoke({"rewritten_question": state["rewritten_question"]})
    
    logger.info(f"Greeting generated: {response.content}")
    return {"final_response": response.content}


def retrieval_node(state: AgentState):
    """Pulls raw context chunks from the Vector DB."""
    
    logger.info(f"Retrieving context from Vector DB for question")
    if state.get("subject") == "General":
        logger.warning("Subject is 'General'; retrieval may yield broad results.")
        state["subject"] = "root"  # Ensure we have a default subject for the DB query
    res = search_vdb(
        user_id=state["user_id"], 
        subject=state["subject"], 
        query=state["rewritten_question"]
    )
    chunks = res if isinstance(res, list) else [res] if res else []
    logger.debug(f"Retrieved chunks: {chunks}")
    return {"retrieved_chunks": chunks}


def grade_context_node(state: AgentState):
    """Full CRAG Evaluator: Implements the 3-tier Correct/Ambiguous/Incorrect logic."""

    logger.info(f"Grading retrieved context")
    query = state['rewritten_question']
    raw_chunks = state.get('retrieved_chunks', [])
    
    filtered_chunks = []
    max_score = 0.0
    
    for chunk in raw_chunks:
        if not chunk.strip():
            continue
            
        raw_score = float(grader_model.predict([query, chunk]))
        
        # Mathematical safeguard to prevent Python OverflowErrors on extreme logits
        try:
            confidence = (1 / (1 + math.exp(-raw_score))) * 100
        except OverflowError:
            confidence = 0.0 if raw_score < 0 else 100.0
            
        if confidence > max_score:
            max_score = confidence
        
        # Keep chunks that have at least some relevance (Ambiguous or Correct)
        if confidence >= 30.0:
            filtered_chunks.append(chunk)
    logger.debug(f"Filtered chunks: {filtered_chunks} with max confidence score: {max_score}")
    # 3-Tier CRAG Decision Gate
    if max_score >= 80.0:
        status = "CORRECT"
        is_answerable = True
    elif max_score <= 30.0:
        status = "INCORRECT"
        is_answerable = False  # Will be flipped to True if Web Search rescues it
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


def web_search_node(state: AgentState):
    """CRAG Fallback: Fetches and refines external knowledge when the database fails."""

    logger.info(f"Performing external web search for question")
    search_tool = DuckDuckGoSearchRun()
    
    query = state["rewritten_question"]
    
    # 1. Domain Restriction: Force the search engine to use reliable academic sources
    academic_query = f"{query} site:edu OR site:wikipedia.org OR site:scholar.google.com"
    
    try:
        # Execute the search
        raw_web_results = search_tool.invoke(academic_query)
        
        if not raw_web_results or "error" in raw_web_results.lower()[:20]:
            raise ValueError("No results found via DuckDuckGo")
            
        # 2. CRAG Knowledge Refinement (Decompose-then-Recompose)
        # DuckDuckGo strings are usually separated by '...' or newlines. We split them into chunks.
        web_snippets = raw_web_results.split("...")
        refined_snippets = []
        logger.debug(f"Raw web snippets: {web_snippets}")
        for snippet in web_snippets:
            if len(snippet.strip()) < 15:
                continue  # Skip empty or tiny fragments
                
            # Grade the web snippet using our local Cross-Encoder!
            raw_score = float(grader_model.predict([query, snippet.strip()]))
            try:
                confidence = (1 / (1 + math.exp(-raw_score))) * 100
            except OverflowError:
                confidence = 0.0 if raw_score < 0 else 100.0
                
            # Keep web snippets that are at least somewhat relevant (e.g., > 40%)
            if confidence >= 40.0:
                refined_snippets.append(snippet.strip())
        
        # If the grader threw everything away, the search was a failure
        if not refined_snippets:
            raise ValueError("Web search returned irrelevant noise.")
            
        # Recompose the surviving, high-quality facts
        formatted_web_data = "[Verified External Web Knowledge]:\n" + "\n- ".join(refined_snippets)
        is_rescued = True
        
    except Exception as e:
        logger.warning(f"Web search failed or was rejected by grader: {e}")
        formatted_web_data = "[External Web Search Failed]"
        is_rescued = False
    
    # CRAG AMBIGUOUS Logic: Combine database and web
    if state.get("status") == "AMBIGUOUS":
        new_context = state.get("context", "") + "\n\n" + formatted_web_data
    # CRAG INCORRECT Logic: Discard database entirely
    else:
        new_context = formatted_web_data
        
    return {
        "context": new_context,
        "is_answerable": is_rescued if state.get("status") == "INCORRECT" else True
    }


def domain_tutor_node(state: AgentState):
    """Generates the educational answer based on the refined CRAG context."""
    logger.info(f"Generating educational answer for question: {state['rewritten_question']}")
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-tutor")
    
    # Bind the tracker to the LLM
    llm_tracked = tutor_llm.with_config({"callbacks": [tracker]})
    
    # Chain the prompt and the LLM together
    chain = DOMAIN_TUTOR_PROMPT | llm_tracked
    
    # Invoke the chain by passing the state variables as a dictionary
    response = chain.invoke({
        "context": state.get('context', 'No context available.'),
        "detail_level": state.get('detail_level', 'detailed'),
        "subject": state.get('subject', 'General'),
        "rewritten_question": state.get('rewritten_question', '')
    })
    
    return {"raw_data": response.content}


def tool_execution_node(state: AgentState):
    logger.info(f"Executing external tools for question: {state['rewritten_question']}")
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-tools")
    llm_with_tools = heavy_llm.bind_tools(tools).with_config({"callbacks": [tracker]})
    
    # Ask the LLM which tool it wants to run and with what arguments
    #LangChain automatically takes the names and descriptions of  Python functions, turns them into a massive JSON schema
    response = llm_with_tools.invoke(state["rewritten_question"])
    
    # If the LLM decided it didn't need a tool after all and just answered
    if not response.tool_calls:
        logger.info("LLM decided not to use a tool and provided a direct response.")
        return {"raw_data": response.content}
        
    #  Create a dictionary to easily look up our actual Python tool functions by name
    tool_map = {tool.name: tool for tool in tools}
    
    tool_outputs = []
    
    #  Iterate over the tools the LLM selected and actually execute them
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
        
        if tool_name in tool_map:
            try:
                # EXECUTION HAPPENS HERE: We invoke the actual LangChain tool
                result = tool_map[tool_name].invoke(tool_args)
                tool_outputs.append(f"[Output from {tool_name}]:\n{result}")
                logger.info(f"Tool {tool_name} executed successfully.")
            except Exception as e:
                logger.error(f"Tool {tool_name} execution failed: {e}")
                tool_outputs.append(f"[Error from {tool_name}]:\nExecution failed - {e}")
        else:
            logger.warning(f"LLM tried to call a non-existent tool: {tool_name}")
            tool_outputs.append(f"[Error]: Tool '{tool_name}' does not exist.")
            
    #  Combine the results of all executed tools into a single string for the Composer
    final_raw_output = "\n\n".join(tool_outputs)
    
    return {"raw_data": final_raw_output}


def answer_composer_node(state: AgentState):
    """Formats the final text and handles apologies if all data retrievals failed."""

    logger.info(f"Composing final answer for question: {state['rewritten_question']}")
    if not state.get("is_safe", True):
        override_data = f"Safety Violation Flagged: {state.get('safety_reason')}. Reject the request respectfully."
        
    # Triggered only if the DB failed AND the Web Search crashed
    elif not state.get("is_answerable", True):
        override_data = "System Note: Both the textbook database and the external web search failed to find an answer. Politely apologize to the student."
        
    else:
        override_data = state.get("raw_data", "No data provided.")
        
    tracker = TokenTrackingCallbackHandler(state["user_id"], "gpt-4o-mini-composer")
    chain = COMPOSER_PROMPT | fast_llm.with_config({"callbacks": [tracker]})
    
    response = chain.invoke({
        "raw_data": override_data,
        "rewritten_question": state["rewritten_question"]
    })
    logger.info(f"Final answer generated: {response.content}")
    return {"final_response": response.content}


# --- Routing Edge Logic ---

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