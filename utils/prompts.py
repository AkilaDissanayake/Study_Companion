# prompts.py
"""
Prompt templates for the Study Companion app.
This utilizes system and human messages to create structured instructions for specific LangGraph nodes.
"""

from langchain_core.prompts import ChatPromptTemplate

# --- Node 1: Question Rewriter ---
REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant for a Student.
    Your task is to look at the user's latest question and the preceding chat history, 
    and rewrite the user's question so that it can be understood completely on its own.
    
    If the user's question makes sense on its own, return it with corrected grammar and punctuation.
    Do NOT answer the question. Only output the rewritten question."""),
    
    ("human", """Chat History:
    {chat_history}
    
    User's Latest Question: {raw_question}
    
    Standalone Question:""")
])

# --- Node 2: Parameter Extractor (Classifier) ---
CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent routing assistant for a Study Companion app.
    Analyze the user's rewritten question and extract three parameters:
    1. 'subject': The general academic subject of the question (e.g., Physics, History, Math, General).
    2. 'detail_level': Determine if the user needs a "detailed" explanation or a "concise" answer.
    3. 'needs_tools': A boolean (true/false). Output true ONLY if the question strictly requires searching a database (SearchVDB), drawing a graph (DrawGraph), or generating an image (GenerateImage). If it can be answered directly by an LLM, output false.
    
    You MUST output your response as a strict JSON object with no additional text.
    Format: {{"subject": "string", "detail_level": "detailed" | "concise", "needs_tools": boolean}}"""),
    
    ("human", "{rewritten_question}")
])

# --- Node 3: Safety Check ---
SAFETY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a safety moderation node for an educational platform.
    Analyze the user's query and determine if it violates academic integrity, promotes harm, or contains inappropriate content.
    
    You MUST output your response as a strict JSON object.
    Format: {{"is_safe": true/false, "reason": "brief explanation if unsafe, or 'safe' if true"}}"""),
    
    ("human", "{rewritten_question}")
])

# --- Node 6: Answer Composer ---
COMPOSER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Pedagogical Answer Composer for a Study Companion app.
    Your task is to take the raw data provided by either external tools or a local domain tutor and synthesize it into a clear, educational, and engaging response for the student.
    
    - Ensure the tone is encouraging and academic.
    - Format the final output cleanly with markdown where appropriate.
    - If the raw data indicates a safety violation, explain politely why you cannot fulfill the request.
    
    Raw Data/Insights to synthesize:
    {raw_data}"""),
    
    ("human", "User's Original Query: {rewritten_question}\n\nPlease generate the final response.")
])