from __future__ import annotations

from typing import Iterable

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from rag_project.config import Settings


def chunk_documents(documents: Iterable[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(list(documents))


def load_embedding_model(settings: Settings) -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


def embed_documents(model: SentenceTransformer, chunks: list[Document]):
    return model.encode([chunk.page_content for chunk in chunks], show_progress_bar=False)


def get_collection(settings: Settings) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_or_create_collection(name=settings.collection_name)


def rebuild_collection(
    settings: Settings,
    chunks: list[Document],
    embeddings,
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_path)
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=settings.collection_name)
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk_{index}" for index in range(len(chunks))]
    embeddings_list = embeddings.tolist()

    batch_size = 512
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            documents=documents[start:end],
            embeddings=embeddings_list[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    return collection


def retrieve_context(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    question: str,
    limit: int,
) -> list[str]:
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=limit)
    return results["documents"][0]
