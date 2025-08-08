from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from utils.pdf_loader import download_pdf_from_url, extract_text_from_pdf
from rag.vector_store import get_answer_from_chunks

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
app = FastAPI()

class QueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]
class QueryResponse(BaseModel):
    answers: List[str]

@app.get("/")
def health_check():
    return {"status": "ok"}

# --- API Endpoint with NO SECURITY for this test ---
@app.post("/ask", response_model=QueryResponse)
async def hackrx_run(payload: QueryRequest): # NOTE: Security parameters removed
    # The authorization checks are temporarily disabled for this test.

    all_chunks = []
    for url in payload.documents:
        pdf_file_stream = download_pdf_from_url(url)
        if pdf_file_stream:
            chunks = extract_text_from_pdf(pdf_file_stream)
            all_chunks.extend(chunks)

    answers = []
    for question in payload.questions:
        answer = get_answer_from_chunks(question, all_chunks)
        answers.append(answer)
    return {"answers": answers}