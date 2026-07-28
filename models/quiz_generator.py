import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from utils.prompts import QUIZ_DRAFTER_PROMPT, QUIZ_CRITIC_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__, "quiz_graph.log")

# --- 1. Define the State ---
class QuizState(TypedDict):
    chat_history: str
    draft_quiz: dict      # The JSON output from the drafter
    critique: str         # The verbal feedback from the critic
    retry_count: int      
    final_quiz: dict      # The verified, accepted output


# --- 2. Initialize Models ---
# We use standard temperature for drafting to allow creativity in options
drafter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind(response_format={"type": "json_object"})
# We use 0 temperature for the critic so it is purely deterministic and strict
critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0) 


# --- 3. Define the Nodes ---
def draft_quiz_node(state: QuizState):
    """The Actor: Generates or refines the quiz."""
    logger.info(f"Drafting quiz. Attempt: {state.get('retry_count', 0) + 1}")
    
    chain = QUIZ_DRAFTER_PROMPT | drafter_llm
    
    response = chain.invoke({
        "history": state["chat_history"],
        "critique": state.get("critique", "None. This is the first attempt.")
    })
    
    try:
        draft = json.loads(response.content)
    except json.JSONDecodeError:
        draft = {"title": "Error", "questions": []}
        
    return {
        "draft_quiz": draft,
        # Increment retry count on every pass through this node
        "retry_count": state.get("retry_count", 0) + 1 
    }

def verify_facts_node(state: QuizState):
    """The Evaluator: Checks for hallucinations."""
    logger.info("Critiquing quiz against source text...")
    
    chain = QUIZ_CRITIC_PROMPT | critic_llm
    
    # Convert JSON back to string so the critic can read it easily
    draft_str = json.dumps(state["draft_quiz"], indent=2)
    
    response = chain.invoke({
        "history": state["chat_history"],
        "draft_quiz": draft_str
    })
    
    critique = response.content.strip()
    logger.debug(f"Critic Assessment: {critique}")
    
    return {"critique": critique}


# --- 4. Define the Router ---
def route_evaluation(state: QuizState):
    """Decides whether to accept the quiz, retry, or abort."""
    
    if state["critique"] == "PASS":
        logger.info("Quiz verified successfully. No hallucinations.")
        return "accept"
        
    if state["retry_count"] >= 3:
        logger.warning("Max retries hit. Accepting flawed quiz or aborting.")
        # Fallback: if we loop 3 times and still fail, just take what we have to prevent infinite loops
        return "accept" 
        
    logger.info("Hallucination detected. Routing back to Drafter for revision.")
    return "revise"


# --- 5. Build the Graph ---
workflow = StateGraph(QuizState)

workflow.add_node("drafter", draft_quiz_node)
workflow.add_node("critic", verify_facts_node)

workflow.set_entry_point("drafter")
workflow.add_edge("drafter", "critic")

workflow.add_conditional_edges(
    "critic",
    route_evaluation,
    {
        "accept": END,       # Map 'accept' string to END
        "revise": "drafter"  # Map 'revise' string back to drafter
    }
)

# Compile the engine
QuizGeneratorAgent = workflow.compile()