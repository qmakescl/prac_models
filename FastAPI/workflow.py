# from https://pub.towardsai.net/building-ai-workflows-with-fastapi-and-langgraph-step-by-step-guide-599937ab84f3

# workflow.py
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o")  # You can switch to gpt-4o-mini for cheaper calls

# Define state
def answer_question(state: dict) -> dict:
    user_input = state["user_input"]
    response = llm.invoke([HumanMessage(content=user_input)])
    return {"answer": response.content}

# Build the graph
workflow = StateGraph(dict)
workflow.add_node("answer", answer_question)
workflow.add_edge(START, "answer")
workflow.add_edge("answer", END)
graph = workflow.compile()


# Error Handling
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def safe_invoke_llm(message):
    return llm.invoke([HumanMessage(content=message)])

def answer_question(state: dict) -> dict:
    user_input = state["user_input"]
    try:
        response = safe_invoke_llm(user_input)
        return {"answer": response.content}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
    

# Input Validation

from pydantic import BaseModel, constr

class RequestData(BaseModel):
    user_input: constr(min_length=1, max_length=500)  # limit input size

# Logging 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def answer_question(state: dict) -> dict:
    logger.info(f"Received input: {state['user_input']}")
    response = safe_invoke_llm(state['user_input'])
    logger.info("LLM response generated")
    return {"answer": response.content}
