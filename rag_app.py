import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from fastapi import FastAPI, Query
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from starlette.responses import JSONResponse
import subprocess

app = FastAPI()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = None
pdf_path = "/Users/sj1212/Downloads/2025_MandateForLeadership_FULL.pdf"  # Dedicated PDF document


def extract_text_from_pdf():
    """Extract text from a dedicated PDF file."""
    if not os.path.exists(pdf_path):
        return "Error: PDF document not found."
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text


def create_faiss_index(text):
    """Create a FAISS vector index from extracted text."""
    global vector_store
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(texts, embeddings)


import openai
import os

# Get OpenAI API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def query_openai(context, query):
    """Query OpenAI GPT-4 instead of Ollama."""
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Use "gpt-3.5-turbo" for cheaper queries
        messages=[
            {"role": "system", "content": "You are an AI assistant answering questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        api_key=OPENAI_API_KEY
    )
    
    return response["choices"][0]["message"]["content"]


def query_document(query, custom_context=None):
    """Retrieve relevant chunks and generate a response using LLaMA 3.2. Stick to the information in the document and focus on numbers, names, and places."""
    if custom_context:
        # Use the provided custom context
        return query_llama_ollama(custom_context, query)
    
    # Default behavior using vector store
    if vector_store is None:
        return "No document processed yet."
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return query_openai(context, query)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_document()
    yield

app = FastAPI(lifespan=lifespan)
def load_document():
    """Load and process the dedicated document at startup."""
    text = extract_text_from_pdf()
    if text != "Error: PDF document not found.":
        create_faiss_index(text)
        print("PDF document successfully loaded into FAISS index.")
    else:
        print(text)


@app.get("/query/")
def query_api(query: str = Query(..., description="Enter your question")):
    """Handle user queries."""
    answer = query_document(query)
    return JSONResponse(content={"answer": answer})
