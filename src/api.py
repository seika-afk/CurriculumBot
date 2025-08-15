from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import BaseModel ,Field

from main import query

app=FastAPI()

class userInput(BaseModel):

    question :Annotated[str,Field(...,description="Question input'd by user.")]

@app.post("/query")
def answer_query(data : userInput):
    answer=query(data.question)
    return JSONResponse(status_code=200,content={'Output':answer})
