import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load PDF file
pdf_file = '/Users/sj1212/Documents/SINF-2024-0471_Proof_hi.pdf'
with open(pdf_file, 'rb') as f:
    pdf = PyPDF2.PdfReader(f)

# Extract text from PDF
text = ''
for page in range(len(pdf.pages)):
    text += pdf.pages[page].extract_text()

# Preprocess text data
nltk.download('punkt')
tokenized_text = word_tokenize(text)
stop_words = nltk.corpus.stopwords.words('english')
filtered_text = [word for word in tokenized_text if word.lower() not in stop_words]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform text data
X = vectorizer.fit_transform(filtered_text)

# Define the RagModel class (assuming it's a simple dictionary-based model)
class RagModel:
    def __init__(self, vectorizer, num_terms):
        self.vectorizer = vectorizer
        self.num_terms = num_terms

    def predict(self, X):
        # For simplicity, assume that each entity is represented by its TF-IDF vector
        return [entity_vector for entity_vector in self.vectorizer.transform([entity]) for entity in self.vectorizer.vocabulary_]

# Train RagModel (using a simple example)
rag_model = RagModel(vectorizer, num_terms=1000)

# Define the knowledge graph
kg = {
    'Entity 1': ['word1', 'word2'],
    'Entity 2': ['word3', 'word4']
}

# Query interface
def query(kg):
    query_text = input("Enter your question: ")
    tokens = word_tokenize(query_text.lower())
    
    # Find matching entities
    matches = []
    for entity, words in kg.items():
        if any(token in words for token in tokens):
            matches.append((entity, len([word for word in words if word in tokens])))
    
    # Print output
    print("Output:")
    for match in sorted(matches, key=lambda x: x[1], reverse=True):
        print(f"Entity: {match[0]}, Count: {match[1]}")

# Run query function
query(kg)