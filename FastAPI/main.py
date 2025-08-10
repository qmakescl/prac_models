# from https://pub.towardsai.net/building-ai-workflows-with-fastapi-and-langgraph-step-by-step-guide-599937ab84f3

# main.py
from fastapi import FastAPI
from workflow import graph, RequestData

app = FastAPI()

@app.post("/run")
async def run_workflow(data: RequestData):
    result = graph.invoke({"user_input": data.user_input})
    return {"result": result["answer"]}