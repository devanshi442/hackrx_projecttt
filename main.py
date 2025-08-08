from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from utils.pdf_loader import download_pdf_from_url, extract_text_from_pdf
from rag.vector_store import get_answer_from_chunks

load_dotenv()

API_KEY = os.getenv("API_KEY")

app = FastAPI()

class QueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QueryResponse)
@app.post("/ask")
async def hackrx_run(
    request: Request,
    payload: QueryRequest,
    authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split("Bearer ")[-1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    all_chunks = []
    for url in payload.documents:
        # Step 1: Download the PDF from the URL
        pdf_file_stream = download_pdf_from_url(url)


        # Step 2: Extract text chunks
        if pdf_file_stream:
            chunks = extract_text_from_pdf(pdf_file_stream)
            all_chunks.extend(chunks)

    answers = []
    for question in payload.questions:
        answer = get_answer_from_chunks(question, all_chunks)
        answers.append(answer)

    return {"answers": answers}
