from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from utils.pdf_loader import download_pdf_from_url, extract_text_from_pdf
from rag.vector_store import get_answer_from_chunks

# Create the FastAPI app
app = FastAPI()

# Define the data models for the request and response
class QueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Create a simple test endpoint
@app.post("/ask", response_model=QueryResponse)
async def hackrx_run(payload: QueryRequest):
    # For this test, we are just returning a fake answer
    # to prove the endpoint is working.
    fake_answers = [f"This is a test answer for: {q}" for q in payload.questions]
    return {"answers": fake_answers}