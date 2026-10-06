from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import invoke_llm


def build_context(results: list[tuple[Document, float]]) -> str:
    context_parts = []

    for rank, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata

        context_parts.append(
            f"""
[Source {rank}]
Document: {metadata.get("source")}
Page: {metadata.get("page_start")}-{metadata.get("page_end")}
Section: {metadata.get("section")}
Subsection: {metadata.get("subsection")}

Content:
{doc.page_content}
""".strip()
        )

    return "\n\n".join(context_parts)


def generate_answer(
    question: str,
    context: str,
) -> str:
    system_prompt = """
You are a document knowledge assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not guess or invent facts, terms, conditions, dates, or numbers.
3. Answer in the same language as the user's question.
4. Keep the answer concise but complete.
5. When helpful, identify the relevant document or section using the labels
   provided in the context.
6. Use polite, friendly, easy-to-understand language while keeping factual
   statements precise.
""".strip()

    user_prompt = f"""
Question:
{question}

Context:
{context}
""".strip()

    response = invoke_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        purpose="generate an answer",
    )

    return response.content.strip()
