# RAG Project

This project is a simple retrieval-augmented generation pipeline built on the Superstore sales dataset.
It turns raw transactions plus generated business summaries into vector-searchable documents, stores them in
ChromaDB, and answers analytics questions through a local Ollama model.

## What It Does

- Loads the sample Superstore CSV from `data/`
- Converts rows into natural-language transaction documents
- Generates higher-level summary documents for years, months, categories, regions, and top performers
- Chunks and embeds those documents with `sentence-transformers/all-MiniLM-L6-v2`
- Stores vectors in a persistent Chroma collection
- Retrieves relevant context and asks an Ollama model to answer the question

## Project Structure

```text
src/
  main.py
  rag_project/
    config.py
    data.py
    documents.py
    rag.py
    vector_store.py
data/
  Sample - Superstore.csv
```

## Setup

1. Create a virtual environment.
2. Install dependencies.
3. Make sure Ollama is installed and your chosen model is available locally.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
```

You can customize paths and model names with environment variables from `.env.example`.

## Commands

Inspect the dataset:

```bash
python3 src/main.py inspect
```

Build or rebuild the vector store:

```bash
python3 src/main.py ingest --rebuild
```

Ask a question:

```bash
python3 src/main.py ask "Which region has the best sales performance?"
```

Ask a question and print the retrieved context:

```bash
python3 src/main.py ask "Which category is the most profitable?" --show-context
```

## Notes

- The first ingest run will download the sentence-transformer model if it is not already cached.
- `ask` expects that the Chroma collection has already been created.
- The answer quality depends heavily on the generated summary documents and the local Ollama model you choose.
