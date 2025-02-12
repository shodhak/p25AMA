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


def query_llama_ollama(context, query):
    """Query the local LLaMA 3.2 model via Ollama three times and synthesize the answers."""
    enhanced_query = "Add information from other sources to enhance the answer. Make sure to mention when you use outside sources" + query
    prompt = f"Context: {context}\n\nQuestion: {enhanced_query}\nAnswer:"
    
    # Generate 3 different answers
    answers = []
    for _ in range(3):
        result = subprocess.run(["ollama", "run", "llama3", prompt], capture_output=True, text=True)
        answers.append(result.stdout.strip())
    
    # Create synthesis prompt
    synthesis_prompt = f"""Here are three answers to the question "{query}":

1: {answers[0]}
2: {answers[1]}
3: {answers[2]}

Please synthesize these three answers into one comprehensive, accurate response that combines the best insights from all three answers."""

    # Get synthesized answer
    final_result = subprocess.run(["ollama", "run", "llama3", synthesis_prompt], 
                                capture_output=True, text=True)
    return final_result.stdout.strip()


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
    return query_llama_ollama(context, query)


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
