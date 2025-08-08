from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from utils.pdf_loader import download_pdf_from_url, extract_text_from_pdf
from rag.vector_store import get_answer_from_chunks

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Create the FastAPI app instance
app = FastAPI()

# --- Pydantic Models ---
# Defines the expected structure of the incoming request data
class QueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]

# Defines the structure of the data the API will send back
class QueryResponse(BaseModel):
    answers: List[str]


# --- API Endpoint ---
# This single decorator creates the /ask URL for your function
@app.post("/ask", response_model=QueryResponse)
async def hackrx_run(
    request: Request,
    payload: QueryRequest,
    authorization: Optional[str] = Header(None)
):
    # 1. Security Check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split("Bearer ")[-1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    # 2. Core Logic to process documents
    all_chunks = []
    for url in payload.documents:
        pdf_file_stream = download_pdf_from_url(url)
        if pdf_file_stream:
            chunks = extract_text_from_pdf(pdf_file_stream)
            all_chunks.extend(chunks)

    # 3. Get answers for the questions
    answers = []
    for question in payload.questions:
        answer = get_answer_from_chunks(question, all_chunks)
        answers.append(answer)

    # 4. Return the final response
    return {"answers": answers}