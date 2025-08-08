# utils/pdf_loader.py

import requests
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def download_pdf_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            raise Exception(f"Failed to download PDF. Status code: {response.status_code}")
    except Exception as e:
        raise Exception(f"Error downloading PDF: {str(e)}")

def extract_text_from_pdf(file_stream):
    try:
        reader = PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def process_pdf(url):
    file_stream = download_pdf_from_url(url)
    text = extract_text_from_pdf(file_stream)
    chunks = split_text(text)
    return chunks
