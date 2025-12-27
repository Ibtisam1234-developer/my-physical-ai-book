"""
RAG (Retrieval-Augmented Generation) pipeline for Physical AI platform.
"""

from .embeddings import generate_embedding, generate_embeddings_batch
from .retrieval import search_similar_chunks
from .ingestion import chunk_text, process_document_file, ingest_documentation_directory
from .prompts import build_rag_prompt, SYSTEM_PROMPT

__all__ = [
    "generate_embedding",
    "generate_embeddings_batch",
    "search_similar_chunks",
    "chunk_text",
    "process_document_file",
    "ingest_documentation_directory",
    "build_rag_prompt",
    "SYSTEM_PROMPT"
]
