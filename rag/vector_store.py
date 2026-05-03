# =============================================================
# rag/vector_store.py
# -------------------------------------------------------------
# This module handles all vector database operations.
#
# What it does:
#   1. Loads the embedding model (all-MiniLM-L6-v2)
#   2. Initializes ChromaDB collection
#   3. Converts text chunks to vectors and stores them
#   4. Retrieves relevant chunks for a given query
#
# Why ChromaDB?
#   ChromaDB is lightweight, easy to set up, and runs locally.
#   No external server needed — perfect for this project.
#
# Why all-MiniLM-L6-v2?
#   It is fast, lightweight, and produces 384-dimensional vectors.
#   Good balance between speed and accuracy for this use case.
# =============================================================

import chromadb
import uuid
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    BATCH_SIZE,
    TOP_K_RESULTS
)


# -------------------------------------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------------------------------------

def load_embedding_model():
    """
    Loads the sentence transformer embedding model.

    The embedding model converts text into vectors (numbers).
    Similar texts produce similar vectors — this is how
    ChromaDB finds relevant chunks for a query.

    Returns:
        embedder (SentenceTransformer): loaded embedding model
    """

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Embedding model loaded successfully")
    print(f"Vector dimensions: {embedder.get_sentence_embedding_dimension()}")

    return embedder


# -------------------------------------------------------------
# INITIALIZE CHROMADB
# -------------------------------------------------------------

def init_chromadb():
    """
    Initializes a ChromaDB client and creates a collection.

    A collection in ChromaDB is like a table in a database.
    It stores text chunks, their vectors, and metadata together.

    We use cosine similarity to measure how similar two vectors
    are. Cosine similarity measures the angle between vectors:
        - Score of 1.0 means identical meaning
        - Score of 0.0 means completely different meaning

    Returns:
        collection (chromadb.Collection): ChromaDB collection
    """

    print("Initializing ChromaDB...")
    client = chromadb.Client()

    # Create collection with cosine similarity
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"ChromaDB collection created: {COLLECTION_NAME}")

    return collection


# -------------------------------------------------------------
# STORE CHUNKS
# -------------------------------------------------------------

def store_chunks(collection, chunks, embedder):
    """
    Converts text chunks to vectors and stores them in ChromaDB.

    Why batches?
        Processing all 10,049 chunks at once would use too much
        memory. Batches of 500 keep memory usage stable.

    Each chunk is stored with:
        - id       : unique identifier
        - embedding: 384-dimensional vector
        - document : original text
        - metadata : type, year, month, category, region

    Args:
        collection : ChromaDB collection
        chunks     : list of chunk dicts from data_preparation
        embedder   : loaded SentenceTransformer model
    """

    print(f"\nStoring {len(chunks)} chunks into ChromaDB...")
    print(f"Batch size: {BATCH_SIZE}")

    total_batches = (len(chunks) - 1) // BATCH_SIZE + 1

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Storing batches"):
        batch = chunks[i:i + BATCH_SIZE]

        # Extract texts, ids, and metadata from batch
        texts     = [c['text'] for c in batch]
        ids       = [str(uuid.uuid4()) for _ in batch]
        metadatas = [{k: v for k, v in c.items() if k != 'text'} for c in batch]

        # Convert texts to vectors
        embeddings = embedder.encode(texts).tolist()

        # Store in ChromaDB
        collection.add(
            ids        =ids,
            embeddings =embeddings,
            documents  =texts,
            metadatas  =metadatas
        )

    print(f"All chunks stored successfully")
    print(f"Total chunks in ChromaDB: {collection.count()}")


# -------------------------------------------------------------
# RETRIEVE
# -------------------------------------------------------------

def retrieve(query, collection, embedder, n_results=None, filters=None):
    """
    Retrieves the most relevant chunks for a given query.

    How it works:
        1. Convert query text to a vector
        2. ChromaDB finds the closest vectors using cosine similarity
        3. Returns the text of the most similar chunks

    Args:
        query      : the question or search text
        collection : ChromaDB collection
        embedder   : loaded SentenceTransformer model
        n_results  : number of chunks to retrieve (default: TOP_K_RESULTS)
        filters    : optional metadata filter dict
                     example: {"type": "regional_summary"}

    Returns:
        list of relevant text chunks
    """

    if n_results is None:
        n_results = TOP_K_RESULTS

    # Convert query to vector
    query_embedding = embedder.encode(query).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=filters
    )

    return results['documents'][0]