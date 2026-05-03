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
    Data Preparation        - converts rows to text chunks
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
- 5GB free disk space

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
the dataset, runs them through the RAG pipeline, and compares
the results. Saves accuracy report to results/evaluation.json.

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

This project was developed with assistance from Claude AI for code generation in some cases and guidance. All code was
reviewed, tested, debugged, and understood by the team.
Full disclosure is provided in the technical report.

---

## Course

    Course     : Data Warehouse Course
    University : University of Helsinki
    Instructor : Jiaheng Lu