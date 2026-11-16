"""
Prompt templates and context formatting for grounded generation.
"""
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a helpful, professional document analysis assistant.
Your task is to answer user questions using the provided Document Context below.

GUIDELINES:
1. Ground your answer strictly in the provided Context. Do NOT fabricate or assume information not present in the document.
2. If the user asks to summarize, outline, or explain what the document is about, synthesize a clear, well-structured overview using the main topics, sections, and agenda present in the Context.
3. If the user asks about a specific fact, figure, or detail that is completely absent from the Context, truthfully state:
   "I could not find the answer in the provided document."
4. When possible, cite the relevant page numbers whenever referencing facts.
5. Format your answers clearly using markdown headings, bullet points, and concise language.
"""

USER_PROMPT_TEMPLATE = """Document Context:
----------------------------------------
{context}
----------------------------------------

User Question: {question}

Answer:"""


def get_rag_prompt_template() -> ChatPromptTemplate:
    """
    Build the ChatPromptTemplate for RAG generation.
    """
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE),
    ])


def format_context(
    docs_with_scores: List[Tuple[Document, float]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Format retrieved documents into context text and citation metadata.

    Returns:
        Tuple of (formatted_context_string, list_of_citation_dictionaries).
    """
    formatted_chunks: List[str] = []
    citations: List[Dict[str, Any]] = []

    for idx, (doc, score) in enumerate(docs_with_scores):
        source_id = idx + 1
        filename = doc.metadata.get("filename", "unknown_document")
        page = doc.metadata.get("page_number", doc.metadata.get("page", 1))
        chunk_idx = doc.metadata.get("chunk_index", source_id)

        header = f"[Source #{source_id}: {filename} (Page {page})]"
        chunk_body = f"{header}\n{doc.page_content.strip()}"
        formatted_chunks.append(chunk_body)

        citations.append({
            "source_id": source_id,
            "filename": filename,
            "page": int(page) if str(page).isdigit() else 1,
            "chunk_index": int(chunk_idx) if str(chunk_idx).isdigit() else source_id,
            "content": doc.page_content.strip(),
            "score": score,
        })

    context_str = "\n\n".join(formatted_chunks) if formatted_chunks else "No relevant context found."
    return context_str, citations
