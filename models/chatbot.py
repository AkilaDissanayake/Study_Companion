#chatbot.py
"""
LangGraph Application Brain and Orchestration Engine.

Defines the multi-model pipeline structure, state attributes, processing nodes, 
and conditional flow control edges that execute your intelligent Study Companion agent.
"""

import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from utils.prompts import REWRITER_PROMPT, CLASSIFIER_PROMPT
from utils.token_manager import log_token_usage


class AgentState(TypedDict):
    """
    Type schema tracking variables passing globally between operational graph nodes.

    Key-Value Mappings:
        user_id (str): Verified identification token of the active user session.
        chat_history (str): Flattened structural past dialogue log context.
        raw_question (str): Incoming user text string straight from the browser frontend.
        rewritten_question (str): De-contextualized independent query output string.
        subject (str): The classified area of study (e.g. 'Math', 'Physics').
        detail_level (str): The structural depth tag used for logic routing paths.
        final_response (str): Concrete completion output to render on user screens.
    """
    user_id: str
    chat_history: str
    raw_question: str
    rewritten_question: str
    subject: str
    detail_level: str
    final_response: str


# --- Model Engine Allocations ---
fast_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
local_llm = ChatOllama(model="llama3", temperature=0.3)
cloud_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


def rewrite_node(state: AgentState):
    """
    Condense current question attributes and past conversational history into an isolated query.

    Args:
        state (AgentState): Dynamic execution state dictionary mapping.

    Returns:
        dict: Updated state entry containing the standalone transformed 'rewritten_question'.
    """
    response = fast_llm.invoke(
        REWRITER_PROMPT.format(
            chat_history=state.get("chat_history", ""),
            raw_question=state["raw_question"]
        )
    )
    
    if response.response_metadata and "token_usage" in response.response_metadata:
        tokens = response.response_metadata["token_usage"]
        log_token_usage(
            user_id=state["user_id"],
            model_name="gpt-3.5-turbo-rewriter",
            prompt_tokens=tokens.get("prompt_tokens", 0),
            completion_tokens=tokens.get("completion_tokens", 0)
        )
        
    return {"rewritten_question": response.content}


def classify_node(state: AgentState):
    """
    Deconstruct the query via structural JSON processing to isolate topics and routing properties.

    Args:
        state (AgentState): Dynamic execution state dictionary mapping.

    Returns:
        dict: Extracted classifications updating 'subject' and 'detail_level' keys.
    """
    response = fast_llm.bind(response_format={"type": "json_object"}).invoke(
        CLASSIFIER_PROMPT.format(rewritten_question=state["rewritten_question"])
    )
    
    if response.response_metadata and "token_usage" in response.response_metadata:
        tokens = response.response_metadata["token_usage"]
        log_token_usage(
            user_id=state["user_id"],
            model_name="gpt-3.5-turbo-classifier",
            prompt_tokens=tokens.get("prompt_tokens", 0),
            completion_tokens=tokens.get("completion_tokens", 0)
        )

    try:
        data = json.loads(response.content)
        return {
            "subject": data.get("subject", "General"),
            "detail_level": data.get("detail_level", "concise")
        }
    except json.JSONDecodeError:
        return {"subject": "General", "detail_level": "detailed"}


def local_llm_node(state: AgentState):
    """
    Forward concise execution goals to a local open-source instance to minimize operation overhead.

    Args:
        state (AgentState): Dynamic execution state dictionary mapping.

    Returns:
        dict: Operational generation response assigned to the 'final_response' state context.
    """
    response = local_llm.invoke(state["rewritten_question"])
    
    prompt_tokens = response.response_metadata.get("prompt_eval_count", 0)
    completion_tokens = response.response_metadata.get("eval_count", 0)
    
    log_token_usage(
        user_id=state["user_id"],
        model_name="llama3-local",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )
    return {"final_response": response.content}


def cloud_llm_node(state: AgentState):
    """
    Route rigorous study execution requirements to premium external commercial inference endpoints.

    Args:
        state (AgentState): Dynamic execution state dictionary mapping.

    Returns:
        dict: Deep text extraction payload assigned to the 'final_response' state context.
    """
    response = cloud_llm.invoke(f"Provide a detailed explanation for: {state['rewritten_question']}")
    
    if response.response_metadata and "token_usage" in response.response_metadata:
        tokens = response.response_metadata["token_usage"]
        log_token_usage(
            user_id=state["user_id"],
            model_name="gpt-4o",
            prompt_tokens=tokens.get("prompt_tokens", 0),
            completion_tokens=tokens.get("completion_tokens", 0)
        )
    return {"final_response": response.content}


def route_question(state: AgentState):
    """
    Conditional evaluation switchboard choosing downstream workflow target nodes.

    Args:
        state (AgentState): Current data state dictionary mapping.

    Returns:
        str: target string representing the chosen downstream processing graph block.
    """
    if state["detail_level"] == "detailed":
        return "cloud_llm_node"
    return "local_llm_node"


# --- Workflow Graph Orchestration Settings ---
workflow = StateGraph(AgentState)

workflow.add_node("rewriter", rewrite_node)
workflow.add_node("classifier", classify_node)
workflow.add_node("local_llm_node", local_llm_node)
workflow.add_node("cloud_llm_node", cloud_llm_node)

workflow.set_entry_point("rewriter")
workflow.add_edge("rewriter", "classifier")
workflow.add_conditional_edges(
    "classifier",
    route_question,
    {
        "cloud_llm_node": "cloud_llm_node",
        "local_llm_node": "local_llm_node"
    }
)
workflow.add_edge("local_llm_node", END)
workflow.add_edge("cloud_llm_node", END)

ChatBot= workflow.compile()