# Retrieval-Augmented Generation (RAG) Implementation

This repository contains a Python-based implementation of a Retrieval-Augmented Generation (RAG) pipeline using BERT embeddings, FAISS for efficient similarity search, and BART for document summarization.

## Overview

The primary objective of this implementation is to combine document retrieval and summarization into a single pipeline. It embeds documents using BERT-based embeddings, retrieves the most relevant documents based on a user query, and then summarizes the retrieved content using the BART model.

## Features

- **Document Embedding:** Embeds documents using a pre-trained BERT model (`bert-base-uncased`) from Hugging Face.
- **Efficient Document Search:** Uses FAISS for similarity-based search to retrieve the most relevant documents based on a query.
- **Query Expansion:** Utilizes WordNet to expand the input query with synonyms, improving search accuracy.
- **Dynamic Summarization:** Uses Facebook's BART model to summarize the retrieved documents. The length of the summary is dynamically adjusted based on the length of the input.

## Installation

To run this code, you will need to install the following dependencies:

```bash
pip install torch transformers faiss-cpu pdfplumber nltk
```

Additionally, ensure that the wordnet corpus is available for NLTK:

```bash
python -c "import nltk; nltk.download('wordnet')"
```

## Usage
1. Load a PDF document: The code extracts text from a PDF file and processes it for embedding and retrieval.

2. Document Embedding: The `embed_text` function tokenizes and generates embeddings for each document using BERT's `CLS` token.

3. Query Expansion: The `expand_query` function uses WordNet to expand the user's query by adding synonyms of each word in the query.

4. Document Retrieval: The `retrieve_documents` function retrieves the top k most relevant documents based on the expanded query using FAISS.

5. Summarization: After retrieving the documents, the BART model is used to generate a concise summary of the retrieved content. The summary length is dynamically adjusted based on the size of the retrieved content.

## Example Code Execution
You can define your own query and retrieve documents relevant to that query. The code below shows an example query and prints the summary of the retrieved documents:

```python
# Define your query here
query = "Does Manchester United play good football?"
retrieved_docs = retrieve_documents(query)
print("Retrieved Documents:", [doc for doc in retrieved_docs])

# Generate summary
summary = summarizer(" ".join(retrieved_docs), max_length=max_length, min_length=max(30, max_length // 2), do_sample=False)
print("Generated Summary:", summary[0]['summary_text'])
```

### Sample Output
```
Retrieved Documents: [<retrieved document 1>, <retrieved document 2>, ...]
Generated Summary: Manchester United has a long-standing reputation in football, characterized by its strategic gameplay and a strong lineup...
```

## File Structure
- `main.py`: The main script that implements the RAG pipeline.
- `test.pdf`: A sample PDF document used for retrieval purposes.
- `requirements`.txt: List of required dependencies.

##Future Enhancements
- Improve the query expansion mechanism for more robust results.
- Integrate additional models for different NLP tasks like question answering or sentiment analysis.
- Support larger document collections and implement more sophisticated ranking algorithms. 
