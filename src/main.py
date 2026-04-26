from __future__ import annotations

import argparse

from rag_project.config import get_settings
from rag_project.data import describe_data, load_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Superstore RAG project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inspect", help="Show dataset information")

    ingest_parser = subparsers.add_parser("ingest", help="Embed and store documents")
    ingest_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the Chroma collection before ingesting",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the vector store")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved documents before the final answer",
    )

    return parser


def run_inspect() -> None:
    settings = get_settings()
    df = load_data(settings.data_path)
    print(describe_data(df))


def run_ingest(rebuild: bool) -> None:
    try:
        from rag_project.documents import build_documents
        from rag_project.vector_store import (
            chunk_documents,
            embed_documents,
            get_collection,
            load_embedding_model,
            rebuild_collection,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing RAG dependencies. Install them with `pip install -r requirements.txt`."
        ) from exc

    settings = get_settings()
    df = load_data(settings.data_path)
    documents = build_documents(df)
    chunks = chunk_documents(documents, settings)
    embedding_model = load_embedding_model(settings)
    embeddings = embed_documents(embedding_model, chunks)

    if rebuild:
        collection = rebuild_collection(settings, chunks, embeddings)
    else:
        existing_collection = get_collection(settings)
        if existing_collection.count() > 0:
            print(
                f"Collection '{settings.collection_name}' already has "
                f"{existing_collection.count()} documents. Use --rebuild to refresh it."
            )
            return
        collection = rebuild_collection(settings, chunks, embeddings)

    print(
        f"Ingested {len(documents)} documents and stored {collection.count()} chunks "
        f"in '{settings.collection_name}'."
    )


def run_ask(question: str, show_context: bool) -> None:
    try:
        from rag_project.rag import answer_question
        from rag_project.vector_store import get_collection, load_embedding_model
    except ImportError as exc:
        raise SystemExit(
            "Missing RAG dependencies. Install them with `pip install -r requirements.txt`."
        ) from exc

    settings = get_settings()
    collection = get_collection(settings)
    if collection.count() == 0:
        raise SystemExit(
            "The vector store is empty. Run `python3 src/main.py ingest --rebuild` first."
        )

    embedding_model = load_embedding_model(settings)
    answer, context = answer_question(
        collection=collection,
        embedding_model=embedding_model,
        settings=settings,
        question=question,
    )

    if show_context:
        print("Retrieved context:\n")
        for index, item in enumerate(context, start=1):
            print(f"{index}. {item}\n")

    print(answer)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        run_inspect()
    elif args.command == "ingest":
        run_ingest(rebuild=args.rebuild)
    elif args.command == "ask":
        run_ask(question=args.question, show_context=args.show_context)


if __name__ == "__main__":
    main()
