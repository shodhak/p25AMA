import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from fastapi import FastAPI, Query
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from starlette.responses import JSONResponse
import openai

app = FastAPI()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = None
pdf_path = "2025_MandateForLeadership_FULL.pdf"  # Dedicated PDF document
FAISS_INDEX_PATH = "faiss_index"


def extract_text_from_pdf():
    """Extract text from a dedicated PDF file."""
    if not os.path.exists(pdf_path):
        return "Error: PDF document not found."
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text


def create_faiss_index(text):
    """Create FAISS vector index from extracted text and save it."""
    global vector_store
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    texts = text_splitter.split_text(text)
    print(f"Total chunks created: {len(texts)}")
    
    try:
        vector_store = FAISS.from_texts(texts, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)  # Save FAISS index locally
        print("✅ FAISS index created and saved successfully!")
    except Exception as e:
        print(f"❌ Error creating FAISS index: {e}")


def load_document():
    """Load FAISS index if available; otherwise, process the document."""
    global vector_store

    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading existing FAISS index...")
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("✅ FAISS index loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
    else:
        print("⚠️ FAISS index not found, processing the document...")
        text = extract_text_from_pdf()
        if text != "Error: PDF document not found.":
            create_faiss_index(text)
        else:
            print(text)

@app.get("/query/")
def query_api(query: str = Query(..., description="Enter your question")):
    """Handle user queries."""
    print(f"Received query: {query}")  # Debugging line

    try:
        if vector_store is None:
            return JSONResponse(content={"error": "FAISS index is not loaded."}, status_code=500)
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        answer = query_openai(context, query)
        print(f"Generated answer: {answer}")  # Debugging line
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


import openai

def query_openai(context, query):
    """Query OpenAI GPT-4o using the new API format."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key is missing."

    client = openai.OpenAI(api_key=OPENAI_API_KEY)  # Create an OpenAI client

    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI assistant answering questions based on provided context."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"OpenAI API error: {e}"

load_document()  # Ensure FAISS index is loaded on startup

@app.get("/")
def root():
    return {"message": "FastAPI is running! Go to /docs to test the API."}

