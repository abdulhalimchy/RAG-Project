# RAG-Based Sales Data Analysis

A Retrieval-Augmented Generation (RAG) system for analyzing
retail sales data. Built for the Data Warehouse Course at the
University of Helsinki.

---

## Project Overview

This project builds a RAG pipeline that answers analytical
questions about a superstore sales dataset (2014-2017).
Users ask natural language questions and the system retrieves
relevant data and generates accurate answers using a local LLM.

---

## System Architecture

    Dataset (CSV)
        |
        v
    Data Preparation        - Data Preparation – created row-level text chunks and aggregated summary chunks
        |
        v
    Vector Database         - stores chunks as vectors (ChromaDB)
        |
        v
    RAG Pipeline            - retrieves relevant chunks + asks LLM
        |
        v
    Interactive Mode        - user asks questions, system answers

---

## Project Structure

    RAG-Project/
    |
    |-- data/
    |   |-- Superstore.csv               # dataset (download manually)
    |
    |-- rag/
    |   |-- __init__.py
    |   |-- data_preparation.py          # load, clean, chunk data
    |   |-- vector_store.py              # ChromaDB setup and retrieval
    |   |-- rag_pipeline.py              # RAG pipeline and LLM integration
    |   |-- evaluator.py                 # accuracy evaluation (30 questions)
    |
    |-- results/
    |   |-- chunks.json                  # generated chunks (auto created)
    |   |-- evaluation.json              # evaluation results (auto created)
    |
    |-- main.py                          # entry point - interactive mode
    |-- config.py                        # all settings
    |-- requirements.txt                 # dependencies
    |-- README.md                        # this file

---

## Requirements

- Python 3.10+
- Ollama installed and running
- 8GB RAM minimum
- 5GB free disk space (Recommended)

---

## Installation

1. Clone the repository

    git clone https://github.com/abdulhalimchy/RAG-Project.git
    cd RAG-Project

2. Download the dataset

    Download from:
    https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

    Place the CSV file in the data/ folder:
    data/Superstore.csv

3. Create and activate virtual environment

    Linux / Ubuntu:
        python3 -m venv venv
        source venv/bin/activate

    Windows:
        python -m venv venv
        venv\Scripts\activate

4. Install dependencies

    pip install -r requirements.txt

5. Install Ollama

    curl -fsSL https://ollama.ai/install.sh | sh

6. Pull the LLM model

    ollama pull llama3.2:3b

---

## Usage

Run the interactive query mode:

    python main.py

This will:
- Load and prepare the dataset
- Build the vector database
- Start interactive mode where you can ask any question

Force rebuild chunks:

    python main.py --rebuild

Run evaluation against 30 ground truth questions:

    python -m rag.evaluator

    Calculates ground truth answers for 30 questions directly from
    the dataset and measures two things:
      - Retrieval accuracy: was the expected answer found in the
        retrieved chunks before the LLM?
      - Answer accuracy: did the LLM response contain the correct
        answer?
    Saves full report to results/evaluation.json.

Run evaluation without saving results:

    python -m rag.evaluator --nosave

    Same as above but prints results to terminal only.

---

## Technical Stack

    Component        Tool
    ---------------- ------------------------------
    Language         Python 3.10+
    Vector Database  ChromaDB
    Embeddings       all-MiniLM-L6-v2
    LLM              Ollama + Llama 3.2 3B
    Data Processing  Pandas
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

AI tools, mainly ChatGPT and Claude, were used as study and development assistants during the project. At first, we used AI to understand the basic concepts of RAG, including indexing, retrieval, generation, embeddings, vector databases, metadata, and the role of aggregated summaries.

AI was also used during development to support data preparation, converting tabular rows into natural language chunks, deciding useful aggregated summary types, choosing metadata fields, designing prompts, debugging errors, reviewing code, and improving/refactoring the project structure. We provided the project instructions, dataset details, and our implementation requirements when asking for AI assistance.

Some AI-generated suggestions needed modification. For example, we had to debug API-related issues, improve the prompt when the LLM gave vague answers, adjust the chunking and retrieval strategy, and handle vector database storage more efficiently. We reviewed and tested the generated code ourselves instead of using it directly.
Our main contribution was understanding the system, making architectural decisions, selecting useful chunk types, designing metadata, integrating the modules, testing the pipeline, evaluating outputs, and improving the prompt based on real results. Overall, AI tools helped us learn faster and improve the system, but the final implementation, testing, evaluation, and design decisions were completed by us.


---

## Course

    Course     : Data Warehouse Course
    University : University of Helsinki
    Instructor : Jiaheng Lu