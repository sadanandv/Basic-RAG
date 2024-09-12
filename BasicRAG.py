from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import faiss
import pdfplumber

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

def retrieve_documents(query, k=3):
    # Retrieve documents based on the query embedding
    query_embedding = embed_text(query).unsqueeze(0).numpy().reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# Define your query here
query = "Manchester United plays good football"
retrieved_docs = retrieve_documents(query)
print("Retrieved Documents:", retrieved_docs)

# Summarization pipeline using BART from Facebook
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Dynamic adjustment of max_length based on input length
input_length = len(" ".join(retrieved_docs).split())  # Count of words in input
max_length = min(200, max(50, input_length // 2))  # No less than half of the input length, and no less than 50

# Generate summary with dynamically adjusted length
summary = summarizer(" ".join(retrieved_docs), max_length=max_length, min_length=max(30, max_length // 2), do_sample=False)
print("Generated Summary:", summary[0]['summary_text'])
