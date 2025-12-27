"""
Document ingestion and chunking for RAG system.
This module processes the existing documentation content in the docs folder.
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import hashlib
import re
from dataclasses import dataclass

from app.config import settings
from app.rag.embeddings import generate_embeddings_batch
from app.config import get_qdrant_client


logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    text: str
    source_file: str
    section: str
    chunk_index: int
    metadata: Dict[str, Any]


def chunk_text(
    text: str,
    max_chunk_size: int = 1000,
    overlap: int = 200,
    preserve_paragraphs: bool = True
) -> List[DocumentChunk]:
    """
    Chunk text into overlapping segments while preserving semantic boundaries.

    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap: Overlapping characters between chunks
        preserve_paragraphs: Whether to preserve paragraph boundaries

    Returns:
        List of DocumentChunk objects
    """
    if not text or not text.strip():
        return []

    chunks = []

    # Split by paragraphs if requested
    if preserve_paragraphs:
        paragraphs = re.split(r'\n\s*\n+', text)
    else:
        # Split into sentences
        paragraphs = [text]

    current_chunk = ""
    chunk_index = 0

    for para_idx, paragraph in enumerate(paragraphs):
        # If adding this paragraph exceeds max size
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            # Finalize current chunk
            if current_chunk.strip():
                chunks.append(DocumentChunk(
                    text=current_chunk.strip(),
                    source_file="unknown",  # Will be set by caller
                    section=f"chunk_{chunk_index}",
                    chunk_index=chunk_index,
                    metadata={}
                ))

            # Start new chunk with overlap from previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + paragraph
            else:
                current_chunk = paragraph

            chunk_index += 1
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

    # Add final chunk if it exists
    if current_chunk.strip():
        chunks.append(DocumentChunk(
            text=current_chunk.strip(),
            source_file="unknown",  # Will be set by caller
            section=f"chunk_{chunk_index}",
            chunk_index=chunk_index,
            metadata={}
        ))

    return chunks


def extract_section_headers(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract section headers from markdown text to preserve context.

    Args:
        text: Markdown text to analyze

    Returns:
        List of (header_text, start_pos, end_pos) tuples
    """
    headers = []

    # Find markdown headers (#, ##, ###, etc.)
    header_pattern = r'^(#+)\s+(.+)$'

    lines = text.split('\n')
    current_pos = 0

    for line_num, line in enumerate(lines):
        match = re.match(header_pattern, line.strip())
        if match:
            header_level = len(match.group(1))
            header_text = match.group(2)

            # Find the section boundary (until next header of same or higher level)
            start_pos = current_pos
            end_pos = current_pos + len(line) + 1  # +1 for newline

            # Look for next header of same or higher level
            for j in range(line_num + 1, len(lines)):
                next_line = lines[j]
                next_match = re.match(header_pattern, next_line.strip())
                if next_match and len(next_match.group(1)) <= header_level:
                    break
                end_pos += len(next_line) + 1  # +1 for newline

            headers.append((header_text, start_pos, end_pos))

        current_pos += len(line) + 1  # +1 for newline

    return headers


def chunk_markdown_with_sections(
    markdown_text: str,
    max_chunk_size: int = 1000,
    overlap: int = 200
) -> List[DocumentChunk]:
    """
    Chunk markdown text while preserving section context.

    Args:
        markdown_text: Markdown content to chunk
        max_chunk_size: Maximum characters per chunk
        overlap: Overlapping characters between chunks

    Returns:
        List of DocumentChunk objects with section information
    """
    # Extract section headers to maintain context
    headers = extract_section_headers(markdown_text)

    # Split document into sections
    sections = []
    start_pos = 0

    for header_text, header_start, header_end in headers:
        if header_start > start_pos:
            # Add content before this header
            section_content = markdown_text[start_pos:header_start].strip()
            if section_content:
                sections.append(("Previous Content", section_content))

        # Add the section starting from this header
        section_end = header_end
        # Find end of this section (until next header or end of document)
        for next_header_start in [h[1] for h in headers[headers.index((header_text, header_start, header_end)):]]:
            if next_header_start > header_end:
                section_end = next_header_start
                break

        section_content = markdown_text[header_start:section_end].strip()
        if section_content:
            sections.append((header_text, section_content))

        start_pos = section_end

    # Add remaining content if any
    if start_pos < len(markdown_text):
        remaining = markdown_text[start_pos:].strip()
        if remaining:
            sections.append(("Remaining", remaining))

    # Now chunk each section
    all_chunks = []
    chunk_index = 0

    for section_title, section_content in sections:
        # Use regular chunking for each section
        section_chunks = chunk_text(
            section_content,
            max_chunk_size=max_chunk_size,
            overlap=overlap,
            preserve_paragraphs=True
        )

        # Update chunk information
        for i, chunk in enumerate(section_chunks):
            all_chunks.append(DocumentChunk(
                text=chunk.text,
                source_file="unknown",  # Will be set by caller
                section=section_title,
                chunk_index=chunk_index + i,
                metadata={"section_title": section_title}
            ))

        chunk_index += len(section_chunks)

    return all_chunks


async def process_document_file(
    file_path: Path,
    max_chunk_size: int = 1000
) -> List[DocumentChunk]:
    """
    Process a single document file and return chunks.

    Args:
        file_path: Path to document file
        max_chunk_size: Maximum chunk size in characters

    Returns:
        List of document chunks with embeddings
    """
    try:
        # Read file content
        content = file_path.read_text(encoding='utf-8')

        # Determine chunking method based on file type
        if file_path.suffix.lower() in ['.md', '.mdx']:
            chunks = chunk_markdown_with_sections(content, max_chunk_size=max_chunk_size)
        else:
            chunks = chunk_text(content, max_chunk_size=max_chunk_size)

        # Update source file information
        for chunk in chunks:
            chunk.source_file = str(file_path)
            chunk.metadata['file_path'] = str(file_path)
            chunk.metadata['file_name'] = file_path.name
            chunk.metadata['file_size'] = file_path.stat().st_size
            chunk.metadata['file_extension'] = file_path.suffix

        logger.info(f"Processed {file_path.name} into {len(chunks)} chunks")

        return chunks

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        raise


async def ingest_documentation_directory(
    docs_path: Path,
    max_chunk_size: int = 1000,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Ingest all documentation files from the docs directory into Qdrant.

    Args:
        docs_path: Path to documentation directory
        max_chunk_size: Maximum chunk size in characters
        batch_size: Number of chunks to process in each batch

    Returns:
        Statistics about ingestion
    """
    # Find all markdown files in the docs directory and subdirectories
    md_files = list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.mdx"))

    logger.info(f"Found {len(md_files)} documentation files to process")

    total_chunks = 0
    total_tokens = 0
    failed_files = []

    # Process each file
    for file_path in md_files:
        try:
            logger.info(f"Processing file: {file_path}")

            # Process document into chunks
            chunks = await process_document_file(file_path, max_chunk_size)

            # Generate embeddings for chunks
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await generate_embeddings_batch(chunk_texts, batch_size=10)

            # Prepare points for Qdrant
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                # Create unique UUID for this chunk (Qdrant requires integer or UUID IDs)
                import uuid
                content_hash = hashlib.md5(chunk.text.encode()).hexdigest()
                point_id = str(uuid.uuid4())  # Use UUID instead of string

                point = {
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "text": chunk.text,
                        "source_file": str(chunk.source_file),
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "original_id": f"{file_path.name}_{chunk.chunk_index}_{content_hash[:8]}"  # Keep original ID as metadata
                    }
                }
                points.append(point)

            # Insert into Qdrant
            qdrant_client = get_qdrant_client()

            # Create collection if it doesn't exist
            try:
                qdrant_client.get_collection(settings.QDRANT_COLLECTION_NAME)
            except:
                # Collection doesn't exist, create it
                from qdrant_client.http import models
                qdrant_client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=768,  # For Gemini text-embedding-004
                        distance=models.Distance.COSINE
                    )
                )

            # Upsert points to Qdrant
            qdrant_client.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=points
            )

            # Update statistics
            total_chunks += len(chunks)
            total_tokens += sum(len(chunk.text.split()) for chunk in chunks)

            logger.info(f"Ingested {len(chunks)} chunks from {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            failed_files.append(str(file_path))

    # Create ingestion summary
    summary = {
        "total_files_processed": len(md_files) - len(failed_files),
        "failed_files": len(failed_files),
        "total_chunks_ingested": total_chunks,
        "total_tokens": total_tokens,
        "failed_files_list": failed_files,
        "collection_name": settings.QDRANT_COLLECTION_NAME,
        "docs_directory": str(docs_path)
    }

    logger.info(f"Ingestion completed: {summary}")

    return summary


async def validate_ingestion(docs_path: Path) -> Dict[str, Any]:
    """
    Validate that documentation has been properly ingested.

    Args:
        docs_path: Path to documentation directory

    Returns:
        Validation results
    """
    qdrant_client = get_qdrant_client()

    try:
        # Get collection info
        collection_info = qdrant_client.get_collection(settings.QDRANT_COLLECTION_NAME)

        # Count total documents in docs folder
        all_docs = list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.mdx"))

        # Count unique source files in Qdrant
        scroll_result = qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=10000  # Assuming not more than 10k files
        )

        unique_sources = set()
        for point in scroll_result[0]:  # scroll_result returns (points, next_page_offset)
            source_file = point.payload.get("source_file", "")
            if source_file:
                unique_sources.add(source_file)

        validation = {
            "collection_exists": True,
            "total_vectors": collection_info.points_count,
            "docs_folder_count": len(all_docs),
            "ingested_sources_count": len(unique_sources),
            "ingestion_completeness": len(unique_sources) / len(all_docs) if all_docs else 0,
            "validation_passed": len(unique_sources) >= len(all_docs) * 0.9  # 90% threshold
        }

        return validation

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "collection_exists": False,
            "error": str(e),
            "validation_passed": False
        }


async def rebuild_vector_store(docs_path: Path) -> Dict[str, Any]:
    """
    Completely rebuild the vector store from documentation.

    Args:
        docs_path: Path to documentation directory

    Returns:
        Rebuild statistics
    """
    qdrant_client = get_qdrant_client()

    try:
        # Delete existing collection if it exists
        try:
            qdrant_client.delete_collection(settings.QDRANT_COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {settings.QDRANT_COLLECTION_NAME}")
        except:
            logger.info("Collection doesn't exist, creating new one")

        # Re-ingest all documentation
        ingestion_stats = await ingest_documentation_directory(docs_path)

        # Validate the rebuild
        validation = await validate_ingestion(docs_path)

        rebuild_result = {
            "status": "success",
            "ingestion_stats": ingestion_stats,
            "validation": validation,
            "timestamp": asyncio.get_event_loop().time()
        }

        logger.info("Vector store rebuild completed successfully")

        return rebuild_result

    except Exception as e:
        logger.error(f"Vector store rebuild failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }


# Main ingestion function to process the existing docs folder
async def main():
    """
    Main function to ingest existing documentation into RAG system.
    """
    docs_path = Path("../../../docs")  # Relative to backend/app/rag/

    if not docs_path.exists():
        logger.error(f"Docs directory not found at {docs_path}")
        return

    logger.info(f"Starting ingestion of documentation from {docs_path}")

    # Process all documentation
    result = await ingest_documentation_directory(docs_path)

    print("Ingestion completed!")
    print(f"Files processed: {result['total_files_processed']}")
    print(f"Chunks ingested: {result['total_chunks_ingested']}")
    print(f"Tokens processed: {result['total_tokens']}")

    # Validate the ingestion
    validation = await validate_ingestion(docs_path)
    print(f"Validation passed: {validation['validation_passed']}")
    print(f"Completeness: {validation['ingestion_completeness']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
