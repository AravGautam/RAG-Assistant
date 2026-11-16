"""
Text cleaners for normalizing document text before chunking.
"""
import re
import unicodedata
from typing import List
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Clean and normalize raw extracted document text.

    Args:
        text: Raw text string from document loaders.

    Returns:
        Cleaned, normalized text string.
    """
    if not text:
        return ""

    # Normalize Unicode characters (e.g. ligature, accented characters)
    text = unicodedata.normalize("NFKC", text)

    # Replace null bytes and non-printable control characters (except newline, tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace 3 or more consecutive newlines with 2 newlines (preserve paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_documents(documents: List[Document]) -> List[Document]:
    """
    Clean and sanitize a list of Document objects in place.

    Args:
        documents: Raw documents.

    Returns:
        List of cleaned Document objects with empty documents removed.
    """
    cleaned: List[Document] = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        if cleaned_content:
            doc.page_content = cleaned_content
            cleaned.append(doc)
    return cleaned

