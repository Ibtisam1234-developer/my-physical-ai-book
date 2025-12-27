---
name: qdrant-client-usage
description: Qdrant vector database client usage patterns including collection creation with vector configuration, point upsert with embeddings and payload, and semantic search operations. Use when implementing vector storage, similarity search, or working with embeddings in Qdrant.
tags: [qdrant, vector-database, embeddings, similarity-search, vector-storage]
---

# Qdrant Client Usage Patterns

## Installation and Setup

### Install Qdrant Client
```bash
pip install qdrant-client
```

### Initialize Client
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Local instance
client = QdrantClient(host="localhost", port=6333)

# Cloud instance
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key"
)

# In-memory instance (for testing)
client = QdrantClient(":memory:")
```

## Collection Management

### Create Collection
```python
from qdrant_client.models import Distance, VectorParams

# Create collection with cosine similarity
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(
        size=768,              # Embedding dimension (e.g., Gemini text-embedding-004)
        distance=Distance.COSINE  # Cosine similarity for semantic search
    )
)
```

### Distance Metrics
```python
# COSINE: For normalized embeddings (most common for text)
Distance.COSINE

# DOT: Dot product similarity
Distance.DOT

# EUCLID: Euclidean distance
Distance.EUCLID

# MANHATTAN: Manhattan distance
Distance.MANHATTAN
```

**When to use each**:
- **COSINE**: Semantic text search (normalized embeddings)
- **DOT**: Similarity with magnitude importance
- **EUCLID**: Geometric distance in embedding space
- **MANHATTAN**: Alternative to Euclidean for high-dimensional data

### Collection Configuration Options
```python
from qdrant_client.models import (
    VectorParams,
    Distance,
    OptimizersConfigDiff,
    HnswConfigDiff,
    QuantizationConfig,
    ScalarQuantization,
    ScalarType
)

client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
        on_disk=True  # Store vectors on disk for large collections
    ),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000  # Start indexing after this many vectors
    ),
    hnsw_config=HnswConfigDiff(
        m=16,              # Number of edges per node
        ef_construct=100   # Quality of index construction
    ),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantization(
            type=ScalarType.INT8,  # Reduce memory usage
            quantile=0.99
        )
    )
)
```

### Check Collection Exists
```python
# List all collections
collections = client.get_collections()
print([col.name for col in collections.collections])

# Check specific collection
try:
    info = client.get_collection(collection_name="docs")
    print(f"Collection exists: {info.vectors_count} vectors")
except Exception:
    print("Collection does not exist")
```

### Delete Collection
```python
client.delete_collection(collection_name="docs")
```

## Upserting Points

### Single Point Upsert
```python
from qdrant_client.models import PointStruct

# Upsert single point
client.upsert(
    collection_name="docs",
    points=[
        PointStruct(
            id=1,                          # Unique ID (int or UUID)
            vector=embedding,              # 768-dimensional vector
            payload={                      # Metadata
                "source": "docs/intro.md",
                "chunk_text": "Introduction to Physical AI...",
                "section": "Overview",
                "topic": "physical-ai"
            }
        )
    ]
)
```

### Batch Upsert
```python
from qdrant_client.models import PointStruct
from typing import List

def upsert_documents(
    client: QdrantClient,
    collection_name: str,
    embeddings: List[List[float]],
    texts: List[str],
    sources: List[str]
):
    """Batch upsert multiple documents"""
    points = [
        PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "source": source,
                "chunk_text": text,
                "chunk_index": idx
            }
        )
        for idx, (embedding, text, source) in enumerate(zip(embeddings, texts, sources))
    ]

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"Upserted batch {i // batch_size + 1}: {len(batch)} points")
```

### Using UUID as Point ID
```python
import uuid
from qdrant_client.models import PointStruct

point = PointStruct(
    id=str(uuid.uuid4()),  # UUID as string
    vector=embedding,
    payload={
        "source": "docs/locomotion.md",
        "chunk_text": "Bipedal walking requires...",
    }
)

client.upsert(collection_name="docs", points=[point])
```

### Update Payload Only
```python
# Update payload without changing vector
client.set_payload(
    collection_name="docs",
    payload={
        "topic": "robotics",
        "updated_at": "2024-01-01"
    },
    points=[1, 2, 3]  # Point IDs to update
)
```

## Searching Points

### Basic Semantic Search
```python
# Search for similar vectors
results = client.search(
    collection_name="docs",
    query_vector=embedding,  # Query embedding (768-dimensional)
    limit=5                  # Return top 5 results
)

# Process results
for result in results:
    print(f"Score: {result.score}")
    print(f"Source: {result.payload['source']}")
    print(f"Text: {result.payload['chunk_text']}")
    print(f"ID: {result.id}")
    print("---")
```

### Search with Score Threshold
```python
# Only return results with score > 0.7
results = client.search(
    collection_name="docs",
    query_vector=embedding,
    limit=10,
    score_threshold=0.7  # Filter low-relevance results
)
```

### Search with Metadata Filters
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Search only within specific topic
results = client.search(
    collection_name="docs",
    query_vector=embedding,
    limit=5,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="topic",
                match=MatchValue(value="robotics")
            )
        ]
    )
)
```

### Advanced Filtering
```python
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range
)

# Complex filter: topic AND date range
results = client.search(
    collection_name="docs",
    query_vector=embedding,
    limit=5,
    query_filter=Filter(
        must=[
            # Must match topic
            FieldCondition(
                key="topic",
                match=MatchValue(value="physical-ai")
            ),
            # Must be within date range
            FieldCondition(
                key="created_at",
                range=Range(
                    gte="2024-01-01",
                    lte="2024-12-31"
                )
            )
        ],
        should=[
            # Prefer these sources
            FieldCondition(
                key="source",
                match=MatchAny(any=["intro.md", "fundamentals.md"])
            )
        ]
    )
)
```

### Search with Payload Selection
```python
# Only return specific payload fields
results = client.search(
    collection_name="docs",
    query_vector=embedding,
    limit=5,
    with_payload=["source", "chunk_text"],  # Only these fields
    with_vectors=False  # Don't return vectors (saves bandwidth)
)
```

## Complete RAG Search Pipeline

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict
import google.generativeai as genai

class QdrantRAGClient:
    """RAG client using Qdrant for vector storage"""

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        genai.configure(api_key="your-api-key")

    def create_collection(self, collection_name: str, vector_size: int = 768):
        """Create Qdrant collection for embeddings"""
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {collection_name}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using Gemini"""
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return result['embedding']

    def upsert_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, str]]
    ):
        """Upsert documents with embeddings"""
        points = []

        for idx, doc in enumerate(documents):
            # Generate embedding
            embedding = self.embed_text(doc['text'])

            # Create point
            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "source": doc['source'],
                    "chunk_text": doc['text'],
                    "section": doc.get('section', ''),
                    "topic": doc.get('topic', '')
                }
            )
            points.append(point)

        # Batch upsert
        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Upserted {len(points)} documents")

    def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 5,
        topic_filter: str = None
    ) -> List[Dict]:
        """Search for relevant documents"""
        # Generate query embedding
        query_embedding = self.embed_text(query)

        # Build filter if topic specified
        query_filter = None
        if topic_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="topic",
                        match=MatchValue(value=topic_filter)
                    )
                ]
            )

        # Search
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
            with_vectors=False
        )

        # Format results
        return [
            {
                "score": result.score,
                "source": result.payload['source'],
                "text": result.payload['chunk_text'],
                "section": result.payload.get('section', ''),
            }
            for result in results
        ]

    def rag_query(
        self,
        collection_name: str,
        query: str,
        limit: int = 5
    ) -> str:
        """Retrieve context and generate answer"""
        # Search for relevant documents
        results = self.search(collection_name, query, limit)

        # Build context
        context = "\n\n".join([
            f"[{r['source']}]\n{r['text']}"
            for r in results
        ])

        # Generate answer with Gemini
        prompt = f"""Answer the following question using only the context provided below.

Context:
{context}

Question: {query}

Answer:"""

        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)

        return response.text

# Usage example
rag_client = QdrantRAGClient()

# Create collection
rag_client.create_collection("docs")

# Upsert documents
documents = [
    {
        "text": "Physical AI combines perception, actuation, and learning...",
        "source": "docs/intro.md",
        "topic": "physical-ai"
    },
    {
        "text": "Bipedal locomotion requires balance control and gait generation...",
        "source": "docs/locomotion.md",
        "topic": "robotics"
    }
]
rag_client.upsert_documents("docs", documents)

# Search
results = rag_client.search("docs", "How does bipedal walking work?", limit=5)

# RAG query
answer = rag_client.rag_query("docs", "What is Physical AI?")
print(answer)
```

## Common Operations

### Get Point by ID
```python
# Retrieve specific point
point = client.retrieve(
    collection_name="docs",
    ids=[1, 2, 3]
)
```

### Scroll Through All Points
```python
# Iterate through all points
offset = None
limit = 100

while True:
    records, offset = client.scroll(
        collection_name="docs",
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    if not records:
        break

    for record in records:
        print(record.id, record.payload)
```

### Count Points
```python
# Get collection statistics
info = client.get_collection(collection_name="docs")
print(f"Total vectors: {info.vectors_count}")
print(f"Indexed vectors: {info.indexed_vectors_count}")
```

### Delete Points
```python
# Delete by IDs
client.delete(
    collection_name="docs",
    points_selector=[1, 2, 3]
)

# Delete by filter
from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue

client.delete(
    collection_name="docs",
    points_selector=FilterSelector(
        filter=Filter(
            must=[
                FieldCondition(
                    key="topic",
                    match=MatchValue(value="outdated")
                )
            ]
        )
    )
)
```

## Best Practices

### Collection Configuration
- **size=768**: Use embedding dimension that matches your model (Gemini: 768, OpenAI: 1536)
- **distance=COSINE**: Standard for semantic text search
- **on_disk=True**: For large collections (>1M vectors)

### Upserting
- **Batch upsert**: Process 100-1000 points per batch for efficiency
- **Unique IDs**: Use sequential integers or UUIDs
- **Payload design**: Include all metadata needed for filtering and display

### Searching
- **limit=5-7**: Optimal for RAG context (balances relevance and token budget)
- **score_threshold**: Filter low-relevance results (typically 0.7+)
- **with_vectors=False**: Save bandwidth by not returning vectors
- **Filters**: Use metadata filters to narrow search scope

### Performance
- **HNSW indexing**: Configure `m=16, ef_construct=100` for balanced speed/accuracy
- **Quantization**: Use scalar quantization for memory reduction
- **Batch operations**: Always batch when processing multiple points

## Best Practices Checklist

- [ ] Use `Distance.COSINE` for semantic text search
- [ ] Set `size=768` for Gemini embeddings (or match your model)
- [ ] Batch upsert operations (100-1000 points per batch)
- [ ] Include comprehensive payload (source, text, metadata)
- [ ] Search with `limit=5-7` for RAG use cases
- [ ] Set `with_vectors=False` when returning results to save bandwidth
- [ ] Use filters to narrow search by topic, date, or source
- [ ] Configure HNSW for production collections
- [ ] Monitor collection size and consider quantization for large datasets
- [ ] Implement error handling for connection issues

---

**Usage Note**: Apply these patterns when working with Qdrant for vector storage and semantic search. Always match vector dimensions to your embedding model, use appropriate distance metrics, and leverage payload filtering for accurate retrieval.
