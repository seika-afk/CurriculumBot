from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import BaseModel, Field

# import your function from main.py
from main import query

# create FastAPI app instance
app = FastAPI(
    title="Query API",
    description="API to answer user questions using `query` function",
    version="1.0.0"
)

# request body model
class UserInput(BaseModel):
    question: Annotated[str, Field(..., description="Question input'd by user.")]

# POST endpoint
@app.post("/query")
def answer_query(data: UserInput):
    try:
        answer = query(data.question)   # call function
        return JSONResponse(status_code=200, content={"Output": answer})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Error": str(e)})
