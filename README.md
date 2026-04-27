# RAG-Based Sales Data Analysis

A Retrieval-Augmented Generation (RAG) system for analyzing
retail sales data. Built for the Data Warehouse Course at the
University of Helsinki.

---

## Project Overview

This project builds a RAG pipeline that answers analytical
questions about a superstore sales dataset (2014-2017).
Instead of writing manual queries, users ask natural language
questions and the system retrieves relevant data and generates
accurate answers using a local LLM.

---

## System Architecture

    Dataset (CSV)
        |
        v
    Data Preparation        - converts rows to text chunks
        |
        v
    Vector Database         - stores chunks as vectors (ChromaDB)
        |
        v
    RAG Pipeline            - retrieves relevant chunks + asks LLM
        |
        v
    Evaluation              - compares answers against ground truth

---

## Project Structure

    RAG-Project/
    |
    |-- data/
    |   |-- superstore.csv              # dataset (download manually)
    |
    |-- rag/
    |   |-- __init__.py
    |   |-- data_preparation.py         # load, clean, chunk data
    |   |-- vector_store.py             # ChromaDB setup and retrieval
    |   |-- rag_pipeline.py             # RAG pipeline and LLM integration
    |   |-- evaluator.py                # accuracy evaluation
    |
    |-- tests/
    |   |-- test_data_preparation.py
    |   |-- test_vector_store.py
    |   |-- test_rag_pipeline.py
    |   |-- test_evaluator.py
    |
    |-- results/
    |   |-- chunks.json                 # generated chunks (auto created)
    |   |-- rag_results.json            # query results (auto created)
    |
    |-- main.py                         # entry point
    |-- config.py                       # all settings
    |-- requirements.txt                # dependencies
    |-- README.md                       # this file

---

## Requirements

- Python 3.10+
- Ollama installed and running
- 8GB RAM minimum
- 5GB free disk space

---

## Installation

1. Clone the repository

    git clone https://github.com/yourusername/RAG-Project.git
    cd RAG-Project

2. Download the dataset

    Download from:
    https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

    Place the CSV file in the data/ folder:
    data/Sample - Superstore.csv

3. Create and activate virtual environment

    python3 -m venv venv
    source venv/bin/activate

4. Install dependencies

    pip install -r requirements.txt

5. Install Ollama

    curl -fsSL https://ollama.ai/install.sh | sh

6. Pull the LLM model

    ollama pull llama3.2:3b

---

## Usage

Run the full pipeline:

    python main.py

This will:
- Load and prepare the dataset
- Build the vector database
- Run 5 analytical queries
- Evaluate accuracy
- Save results to results/rag_results.json

Run individual tests:

    python tests/test_data_preparation.py
    python tests/test_vector_store.py
    python tests/test_rag_pipeline.py
    python tests/test_evaluator.py

---

## Analysis Queries

1. What is the sales trend over the 4-year period from 2014 to 2017?
2. Which product category generates the most revenue and profit?
3. Which region has the best sales performance?
4. Which months show the highest sales? Is there seasonality?
5. Compare Technology vs Furniture sales and profit performance.

---

## Technical Stack

    Component        Tool
    ---------------- ------------------------------
    Language         Python 3.12
    Vector Database  ChromaDB
    Embeddings       all-MiniLM-L6-v2
    LLM              Ollama + Llama 3.2 3B
    Data Processing  Pandas, NumPy
    RAG Framework    Custom implementation

---

## Dataset

    Name    : Superstore Sales Dataset
    Source  : Kaggle
    Records : 9,994 transactions
    Period  : 2014 - 2017
    Fields  : Sales, Profit, Region, Category, Customer, Product

---

## AI Tool Usage

This project was developed with assistance from Claude AI
(Anthropic) for code generation and guidance. All code was
reviewed, tested, debugged, and understood by the team.
Full disclosure is provided in the technical report.

---

## Team

    Member A : data preparation, chunking, technical report
    Member B : vector database, RAG pipeline, demo preparation

---

## Course

    Course     : Data Warehouse Course
    University : University of Helsinki
    Instructor : Jiaheng Lu