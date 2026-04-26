from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    data_path: str = os.getenv("RAG_DATA_PATH", "data/Sample - Superstore.csv")
    chroma_path: str = os.getenv("RAG_CHROMA_PATH", "chroma_db")
    collection_name: str = os.getenv("RAG_COLLECTION_NAME", "sales_rag")
    embedding_model_name: str = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    ollama_model: str = os.getenv("RAG_OLLAMA_MODEL", "llama3.2:3b")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    retrieval_count: int = int(os.getenv("RAG_RETRIEVAL_COUNT", "8"))


def get_settings() -> Settings:
    return Settings()
