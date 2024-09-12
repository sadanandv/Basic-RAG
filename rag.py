from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import faiss
import pdfplumber
import nltk
from nltk.corpus import wordnet

import warnings
warnings.filterwarnings("ignore")

# Ensure that NLTK WordNet is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load documents from a PDF file
file_path = 'data.pdf'
documents = []
with pdfplumber.open(file_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            documents.extend([line.strip() for line in text.split('\n') if line.strip()])

# Load tokenizer and model for embedding
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def embed_text(text):
    # Ensure truncation and padding are properly handled
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach()  # Use CLS token for document representation

# Embed all documents and prepare them for FAISS
embeddings = torch.stack([embed_text(doc) for doc in documents])
embeddings = embeddings.squeeze()
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.numpy())

def expand_query(query):
    # Expanding the query using WordNet synonyms
    expanded_query = [query]  # Start with the original query
    for word in query.split():
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())  # Add synonyms found
        expanded_query.extend(synonyms)
    return " ".join(expanded_query)

def retrieve_documents(query, k=5):
    # Transform and expand the query
    transformed_query = expand_query(query)
    query_embedding = embed_text(transformed_query).unsqueeze(0).numpy().reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    
    # Rank documents based on a new criterion: here just using distance, can be more complex
    ranked_docs = sorted([(distances[0][i], documents[indices[0][i]]) for i in range(len(distances[0]))], key=lambda x: x[0])
    return [doc for _, doc in ranked_docs]

# Enter your query here
query = str(input("Enter Query"))
retrieved_docs = retrieve_documents(query)
print("Retrieved Documents:", [doc for doc in retrieved_docs])

# Summarization pipeline using BART from Facebook
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Dynamic adjustment of max_length based on input length
input_length = len(" ".join(retrieved_docs).split())  # Count of words in input
max_length = min(200, max(50, input_length // 2))  # No less than half of the input length, and no less than 50

# Generate summary with dynamically adjusted length
summary = summarizer(" ".join(retrieved_docs), max_length=max_length, min_length=max(50, max_length // 2), do_sample=False)
print("Generated Summary:", summary[0]['summary_text'])

