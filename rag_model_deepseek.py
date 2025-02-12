import deepseek
import torch
from PyPDF2 import PdfFileReader, PdfFileWriter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
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
    # Load a retriever (e.g., Sentence Transformers for embeddings)
    retriever = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode passages into embeddings
    passage_embeddings = retriever.encode(passages, convert_to_tensor=True)

    # Build a FAISS index for efficient retrieval
    index = faiss.IndexFlatL2(passage_embeddings.shape[1])
    index.add(passage_embeddings.cpu().numpy())

    # Load a generator (e.g., BART or T5)
    generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    generator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

    return retriever, index, generator_tokenizer, generator_model

# Step 4: Generate response for a query
def generate_response(query, retriever, index, generator_tokenizer, generator_model, passages, top_k=3):
    # Encode the query
    query_embedding = retriever.encode(query, convert_to_tensor=True).cpu().numpy()

    # Retrieve top-k relevant passages
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    retrieved_passages = [passages[i] for i in indices[0]]

    # Combine retrieved passages into context
    context = " ".join(retrieved_passages)

    # Generate response using the generator model
    inputs = generator_tokenizer(context, return_tensors="pt", max_length=512, truncation=True)
    generated_ids = generator_model.generate(inputs["input_ids"], max_length=150)
    response = generator_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return response

# Step 5: Main function to run the pipeline
def main():
    # Path to your PDF file
    pdf_path = "/Users/sj1212/Documents/SINF-2024-0471_Proof_hi.pdf"  # Replace with your PDF file path

    # Step 1: Extract text from PDF in chunks
    chunks = extract_text_in_chunks(pdf_path, chunk_size=10)
    print(f"Extracted {len(chunks)} chunks from PDF.")

    # Step 2: Preprocess text into passages
    all_passages = []
    for chunk in chunks:
        passages = preprocess_text(chunk)
        all_passages.extend(passages)
    print(f"Preprocessed text into {len(all_passages)} passages.")

    # Step 3: Build the RAG model
    retriever, index, generator_tokenizer, generator_model = build_rag_model(all_passages)
    print("RAG model built successfully.")

    # Step 4: Generate response for a query
    query = "What is the main topic of the document?"
    response = generate_response(query, retriever, index, generator_tokenizer, generator_model, all_passages)
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()