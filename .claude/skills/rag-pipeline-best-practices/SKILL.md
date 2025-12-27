---
name: rag-pipeline-best-practices
description: Best practices and configuration for RAG (Retrieval-Augmented Generation) pipelines using Gemini embeddings and Qdrant. Use when implementing document ingestion, embedding generation, vector search, or prompt augmentation for knowledge retrieval systems.
tags: [rag, embeddings, qdrant, gemini, retrieval, vector-search]
---

# RAG Pipeline Best Practices

## Core Pipeline Configuration

### Document Chunking Strategy
- **Chunk Size**: 1000-1500 tokens per chunk
- **Overlap**: 200 tokens between consecutive chunks
- **Rationale**:
  - Preserves semantic context across chunk boundaries
  - Ensures key information isn't split at edges
  - Balances retrieval precision with context completeness

### Chunking Implementation Guidelines
```
Chunk 1: [tokens 0-1500]
Chunk 2: [tokens 1300-2800]  ← 200 token overlap with Chunk 1
Chunk 3: [tokens 2600-4100]  ← 200 token overlap with Chunk 2
```

**Markdown-Specific Considerations**:
- Respect heading boundaries when possible
- Keep code blocks intact within chunks
- Preserve list item completeness
- Maintain link context with surrounding text

## Embedding Generation

### Gemini Configuration
- **Model**: `text-embedding-004` (Gemini's latest embedding model)
- **Output Dimension**: 768 (standard for text-embedding-004)
- **Batch Processing**: Process multiple chunks in single API calls for efficiency
- **Rate Limiting**: Implement exponential backoff for API quota management

### Embedding Best Practices
- **Consistency**: Use same model for both indexing and query embedding
- **Normalization**: Normalize embeddings if using cosine similarity
- **Caching**: Cache embeddings to avoid redundant API calls during testing
- **Metadata**: Store chunk metadata alongside embeddings (source file, section, timestamp)

## Vector Storage (Qdrant)

### Collection Configuration
```python
{
    "vector_size": 768,           # Matches Gemini embedding dimension
    "distance": "Cosine",          # Optimal for semantic similarity
    "on_disk_payload": True,       # For large collections
    "hnsw_config": {
        "m": 16,                   # Number of edges per node
        "ef_construct": 100        # Quality of index construction
    }
}
```

### Collection Schema
- **Vector Field**: 768-dimensional embedding
- **Payload Fields**:
  - `text`: Original chunk content
  - `source`: File path or document ID
  - `section`: Heading/section name
  - `chunk_index`: Position in original document
  - `metadata`: Additional context (author, date, topic tags)

### Indexing Strategy
- Use HNSW (Hierarchical Navigable Small World) for fast approximate search
- Configure `ef_construct` higher (100-200) for better recall
- Use payload indexing for fast metadata filtering

## Semantic Search

### Query Configuration
- **top_k**: 5-7 chunks per query
- **Rationale**:
  - 5-7 provides sufficient context without overwhelming the prompt
  - Balances relevance with token budget
  - Allows for diversity in retrieved results

### Search Parameters
```python
search_params = {
    "limit": 7,                    # Retrieve top 7 most similar
    "score_threshold": 0.7,        # Optional: filter low-relevance results
    "with_payload": True,          # Include metadata
    "with_vectors": False          # Don't return vectors (saves bandwidth)
}
```

### Metadata Filtering
Apply filters to narrow search scope:
```python
filter = {
    "must": [
        {"key": "topic", "match": {"value": "robotics"}},
        {"key": "date", "range": {"gte": "2024-01-01"}}
    ]
}
```

## Prompt Augmentation

### Standard Template
```
Answer the following question using only the context provided below. If the context doesn't contain enough information to answer the question, say so explicitly.

Context:
{context}

Question: {query}

Answer:
```

### Context Formatting
Format retrieved chunks with clear boundaries:
```
[Source 1: path/to/doc.md - Section: Introduction]
{chunk_1_text}

[Source 2: path/to/doc.md - Section: Implementation]
{chunk_2_text}

[Source 3: another/doc.md - Section: Best Practices]
{chunk_3_text}
```

### Advanced Augmentation Techniques
1. **Include Relevance Scores**: Show confidence levels
2. **Source Attribution**: Enable users to verify information
3. **Chunk Ordering**: Rank by relevance or document order
4. **Context Compression**: Summarize redundant chunks if token budget is tight

## Re-Ranking (Optional Enhancement)

### When to Use Re-Ranking
- Initial retrieval returns many results with similar scores
- Domain-specific relevance differs from semantic similarity
- Need to prioritize recency or authority of sources

### Re-Ranking Strategies
1. **Cross-Encoder Re-Ranking**: Use a separate model to score query-chunk pairs
2. **Metadata Boosting**: Increase scores for recent or authoritative sources
3. **Diversity Re-Ranking**: Ensure retrieved chunks cover different aspects
4. **LLM-as-Judge**: Use Gemini to score chunk relevance before augmentation

### Implementation
```python
# After initial vector search with top_k=15-20
initial_results = qdrant_search(query_embedding, limit=20)

# Re-rank using cross-encoder or metadata
reranked_results = rerank(query, initial_results, method="cross-encoder")

# Select top 5-7 after re-ranking
final_context = reranked_results[:7]
```

## Retrieval Testing & Validation

### Test Retrieval Recall
Create a test set of known questions with ground-truth documents:

```python
test_cases = [
    {
        "question": "How does bipedal locomotion work?",
        "expected_sources": ["robotics/locomotion.md", "robotics/gait-control.md"],
        "expected_chunks": [12, 34]  # Chunk IDs that should be retrieved
    },
    # ... more test cases
]
```

### Evaluation Metrics
1. **Recall@K**: Percentage of relevant chunks retrieved in top K results
2. **Precision@K**: Percentage of retrieved chunks that are relevant
3. **MRR (Mean Reciprocal Rank)**: Average position of first relevant result
4. **NDCG (Normalized Discounted Cumulative Gain)**: Weighted relevance scoring

### Testing Protocol
1. **Build Test Set**: 20-50 representative questions with ground truth
2. **Run Retrieval**: Execute search for each test question
3. **Calculate Metrics**: Measure Recall@5, Recall@7, MRR
4. **Iterate**: Adjust chunking, top_k, or re-ranking based on results
5. **Target**: Aim for >80% Recall@7 on test questions

### Debugging Poor Retrieval
- **Low Recall**: Increase chunk overlap, expand top_k, check embedding quality
- **Low Precision**: Add metadata filters, implement re-ranking, improve chunking
- **Missing Context**: Adjust chunk size, check for semantic gaps in corpus
- **Wrong Sources**: Verify embedding consistency, check for corpus contamination

## Production Checklist

- [ ] Document chunking preserves semantic boundaries
- [ ] 200-token overlap configured
- [ ] Qdrant collection uses 768 dimensions with Cosine distance
- [ ] Search retrieves 5-7 chunks per query
- [ ] Prompt template constrains answers to provided context
- [ ] Test set created with 20+ known questions
- [ ] Recall@7 measured and documented
- [ ] Re-ranking strategy evaluated (if retrieval quality insufficient)
- [ ] Error handling for empty results and API failures
- [ ] Logging for search queries and retrieval performance

## Common Pitfalls to Avoid

1. **Inconsistent Embedding Models**: Always use same model version for indexing and querying
2. **No Overlap**: Chunks without overlap lose context at boundaries
3. **Wrong Distance Metric**: Use Cosine for normalized embeddings, not Euclidean
4. **Over-Retrieval**: Retrieving too many chunks (>10) dilutes relevance and wastes tokens
5. **Under-Retrieval**: Too few chunks (<3) may miss critical context
6. **No Testing**: Deploy without measuring recall leads to poor user experience
7. **Ignoring Metadata**: Missing opportunities for filtering and attribution

## Integration with Gemini Generation

After retrieving and formatting context:

```python
# Augment prompt with retrieved context
augmented_prompt = f"""Answer using only this context:

{formatted_context}

Question: {user_query}"""

# Generate response with Gemini
response = gemini.generate(
    model="gemini-2.5-flash",
    prompt=augmented_prompt,
    max_tokens=1024,
    temperature=0.3  # Lower for factual accuracy
)
```

---

**Usage Note**: Apply these configurations when implementing or optimizing RAG pipelines. Adjust parameters based on your specific corpus characteristics and retrieval quality requirements. Always validate with real test questions before production deployment.
