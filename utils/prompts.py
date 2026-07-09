# prompts.py
"""This module contains prompt templates for the Study Companion app.
This uses system and human messages to create structured prompts for the LLM to process user questions."""

from langchain_core.prompts import ChatPromptTemplate

# --- Node 1: Question Rewriter ---
REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant for a Student.
    Your task is to look at the user's latest question and the preceding chat history, 
    and rewrite the user's question so that it can be understood completely on its own.
    
    If the user's question makes sense on its own, return it with corrected grammar and punctuation.
    Do NOT answer the question."""),
    
    ("human", """Chat History:
    {chat_history}
    
    User's Latest Question: {raw_question}
    
    Standalone Question:""")
])


# --- Node 2: Parameter Extractor (Classifier) ---
CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent routing assistant for a Study Companion app.
    Analyze the user question and extract two parameters:
    1. 'subject': The general academic subject of the question (e.g., Physics, History, Math, General).
    2. 'detail_level': Determine if the user needs a "detailed" explanation or a "concise" answer. 
       - If they ask to "explain", "how", "why", or ask a complex multi-part question, output "detailed".
       - If they ask for a definition, a fact, a simple calculation, or a direct answer, output "concise".
    
    You MUST output your response as a strict JSON object with no additional text.
    Format: {{"subject": "string", "detail_level": "detailed" | "concise"}}"""),
    
    ("human", "{rewritten_question}")
])