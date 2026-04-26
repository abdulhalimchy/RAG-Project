from __future__ import annotations

from langchain_ollama import OllamaLLM

from rag_project.config import Settings
from rag_project.vector_store import retrieve_context


def build_prompt(question: str, context: list[str]) -> str:
    joined_context = "\n\n".join(context)
    return f"""
You are a sales data analyst answering questions about Superstore performance.

Use only the provided context.
If the answer involves performance, prioritize total sales and mention profit when it helps.
If the answer is not present in the context, say that clearly.

Context:
{joined_context}

Question:
{question}

Answer clearly and briefly:
""".strip()


def answer_question(
    collection,
    embedding_model,
    settings: Settings,
    question: str,
) -> tuple[str, list[str]]:
    context = retrieve_context(
        collection=collection,
        embedding_model=embedding_model,
        question=question,
        limit=settings.retrieval_count,
    )
    llm = OllamaLLM(model=settings.ollama_model)
    prompt = build_prompt(question=question, context=context)
    return llm.invoke(prompt), context
