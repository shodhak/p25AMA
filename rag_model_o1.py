import PyPDF2
import torch
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from datasets import Dataset
import faiss
import numpy as np

# Step 1: Extract text from PDF in chunks
def extract_text_in_chunks(pdf_path, chunk_size=10):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        chunks = []
        current_chunk = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            current_chunk += page.extract_text()
            if (page_num + 1) % chunk_size == 0:  # Process every `chunk_size` pages
                chunks.append(current_chunk)
                current_chunk = ""
        if current_chunk:  # Add the remaining text
            chunks.append(current_chunk)
        return chunks

# Step 2: Preprocess text into passages
def preprocess_text(text):
    passages = text.split('\n\n')  # Split by double newlines
    passages = [p.strip() for p in passages if p.strip()]  # Remove empty passages
    return passages

# Step 3: Build the RAG model
def build_rag_model(passages):
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", passages=passages)
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, retriever, model, device

# Step 4: Generate response for a query
def generate_response(query, model, retriever, tokenizer, device):
    inputs = tokenizer(query, return_tensors="pt").to(device)
    question_hidden_states = model.question_encoder(input_ids=inputs["input_ids"]).last_hidden_state
    retrieved_docs = retriever(input_ids=inputs["input_ids"], question_hidden_states=question_hidden_states, return_tensors="pt").to(device)
    generated_ids = model.generate(input_ids=inputs["input_ids"], context_input_ids=retrieved_docs["context_input_ids"])
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Step 5: Main function to run the pipeline
def main():
    pdf_path = "/Users/sj1212/Documents/SINF-2024-0471_Proof_hi.pdf"  # Replace with your PDF file path

    chunks = extract_text_in_chunks(pdf_path, chunk_size=10)
    print(f"Extracted {len(chunks)} chunks from PDF.")

    all_passages = []
    for chunk in chunks:
        passages = preprocess_text(chunk)
        all_passages.extend(passages)
    print(f"Preprocessed text into {len(all_passages)} passages.")

    tokenizer, retriever, model, device = build_rag_model(all_passages)
    print("RAG model built successfully.")

    query = "What is the main topic of the document?"
    response = generate_response(query, model, retriever, tokenizer, device)
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()