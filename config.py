# =============================================================
# config.py
# -------------------------------------------------------------
# Central configuration file for the RAG Sales Analysis project.
# All settings, paths, and model parameters are defined here.
# To change any setting, update this file only — no need to
# touch other files.
# =============================================================

import os

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the dataset CSV file
DATA_PATH = os.path.join(ROOT_DIR, "data", "Superstore.csv")

# Path to save generated chunks
CHUNKS_SAVE_PATH = os.path.join(ROOT_DIR, "results", "chunks.json")

# Path to save RAG query results
RESULTS_SAVE_PATH = os.path.join(ROOT_DIR, "results", "rag_results.json")

# -------------------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------------------

# Model used to convert text chunks into vectors
# all-MiniLM-L6-v2 is lightweight, fast, and accurate
# Output: 384-dimensional vectors
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# -------------------------------------------------------------
# VECTOR DATABASE
# -------------------------------------------------------------

# Name of the ChromaDB collection
COLLECTION_NAME = "superstore_rag"

# Number of chunks to retrieve per query
TOP_K_RESULTS = 10

# Batch size for storing chunks into ChromaDB
# 500 is safe for memory — adjust if needed
BATCH_SIZE = 500

# -------------------------------------------------------------
# LLM (Large Language Model)
# -------------------------------------------------------------

# Ollama model name — must be pulled before running
# Run: ollama pull llama3.2:3b
LLM_MODEL = "llama3.2:3b"

# Maximum words expected in LLM response
MAX_RESPONSE_WORDS = 150

# -------------------------------------------------------------
# CHUNKING
# -------------------------------------------------------------

# Minimum character length for a chunk to be stored
# Chunks shorter than this are skipped
MIN_CHUNK_LENGTH = 50

# -------------------------------------------------------------
# ANALYSIS QUERIES
# -------------------------------------------------------------

# The 5 required analytical queries for the project
ANALYSIS_QUERIES = [
    "What is the sales trend over the 4-year period from 2014 to 2017?",
    "Which product category generates the most revenue and profit?",
    "Which region has the best sales performance?",
    "Which months show the highest sales? Is there seasonality?",
    "Compare Technology vs Furniture sales and profit performance."
]