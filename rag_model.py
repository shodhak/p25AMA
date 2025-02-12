import PyPDF2
import torch
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

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
    # Load pre-trained RAG tokenizer
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")

    # Initialize the retriever with in-memory passages
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-base",
        index_name="exact",  # Use in-memory exact search (no FAISS)
        passages=passages,  # Pass the list of passages directly
        use_dummy_dataset=True  # Avoid loading external datasets
    )

    # Load pre-trained RAG generator
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

    # Move model to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, retriever, model, device

# Step 4: Generate response for a query
def generate_response(query, model, retriever, tokenizer, device):
    # Tokenize the query
    inputs = tokenizer(query, return_tensors="pt").to(device)

    # Encode the query to generate question hidden states
    question_hidden_states = model.question_encoder(inputs["input_ids"]).last_hidden_state

    # Retrieve relevant passages using the question hidden states
    retrieved_docs = retriever(question_hidden_states=question_hidden_states, return_tensors="pt").to(device)

    # Generate response using the RAG model
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        context_input_ids=retrieved_docs["input_ids"],
        context_attention_mask=retrieved_docs["attention_mask"]
    )
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

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
    tokenizer, retriever, model, device = build_rag_model(all_passages)
    print("RAG model built successfully.")

    # Step 4: Generate response for a query
    query = "What is the main topic of the document?"
    response = generate_response(query, model, retriever, tokenizer, device)
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()