print("--- Checkpoint 1: Starting script ---")

from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from utils.pdf_loader import download_pdf_from_url, extract_text_from_pdf
from rag.vector_store import get_answer_from_chunks

print("--- Checkpoint 2: Imports complete ---")

load_dotenv()
API_KEY = os.getenv("API_KEY")
print(f"--- Checkpoint 3: API Key is {'SET' if API_KEY else 'NOT SET'} ---")

app = FastAPI()
print("--- Checkpoint 4: FastAPI app created ---")

class QueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

print("--- Checkpoint 5: Pydantic models defined ---")

@app.post("/ask", response_model=QueryResponse)
async def hackrx_run(
    request: Request,
    payload: QueryRequest,
    authorization: Optional[str] = Header(None)
):
    print("--- Checkpoint 6: /ask route function was CALLED ---")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split("Bearer ")[-1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

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

print("--- Checkpoint 7: Route /ask has been defined ---")