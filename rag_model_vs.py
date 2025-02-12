import os
import PyPDF2
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

class RagModelFromPDF:
    def __init__(self, pdf_folder_path, model_name='facebook/rag-sequence-nq'):
        self.pdf_folder_path = pdf_folder_path
        self.tokenizer = RagTokenizer.from_pretrained(model_name)
        self.retriever = RagRetriever.from_pretrained(model_name, index_name="exact", use_dummy_dataset=True)
        self.model = RagSequenceForGeneration.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.corpus = self._load_pdfs()

    def _load_pdfs(self):
        corpus = []
        for filename in os.listdir(self.pdf_folder_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.pdf_folder_path, filename)
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ''
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text()
                    corpus.append(text)
        return corpus

    def generate_answer(self, question):
        inputs = self.tokenizer(question, return_tensors='pt').to(self.device)
        generated_ids = self.model.generate(input_ids=inputs['input_ids'], num_beams=5, max_length=50)
        answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer

def extract_title_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        # Extract the title from the document metadata
        title = reader.metadata.get('/Title', 'No title found')
        return title

# Example usage:
pdf_folder_path = '/Users/sj1212/Documents/RAG_model/pdf_files'
rag_model = RagModelFromPDF(pdf_folder_path)
question = "What is the main topic of the documents?"
answer = rag_model.generate_answer(question)
print(answer)

# Example usage:
pdf_path = '/Users/sj1212/Documents/RAG_model/pdf_files/test.test.pdf'  # Replace with your PDF file path
title = extract_title_from_pdf(pdf_path)
print(f"Title: {title}")