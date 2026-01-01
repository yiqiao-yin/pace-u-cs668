# Embeddings and Vector Representations

## Overview

Embeddings are dense vector representations of text that capture semantic meaning. In RAG systems, embeddings enable semantic search by finding documents similar in meaning to a query, even when they don't share exact words.

## What are Embeddings?

### Concept

```python
# Embeddings convert text to numbers
text = "Machine learning is fascinating"
embedding = [0.123, -0.456, 0.789, ...]  # 384-1536 dimensions

# Similar meanings → Similar vectors
"AI is interesting" → [0.125, -0.450, 0.785, ...]  # Close to above
"The weather is nice" → [0.892, 0.234, -0.567, ...]  # Far from above
```

### Properties

```python
embedding_properties = {
    "dense": "Every dimension has a value (vs sparse)",
    "fixed_size": "Same dimensions regardless of text length",
    "semantic": "Captures meaning, not just keywords",
    "comparable": "Can measure similarity between embeddings"
}
```

## Embedding Models

### Popular Models

```python
embedding_models = {
    # OpenAI
    "text-embedding-3-small": {"dims": 1536, "cost": "$0.02/1M tokens"},
    "text-embedding-3-large": {"dims": 3072, "cost": "$0.13/1M tokens"},

    # Open Source
    "all-MiniLM-L6-v2": {"dims": 384, "cost": "Free"},
    "all-mpnet-base-v2": {"dims": 768, "cost": "Free"},
    "bge-large-en-v1.5": {"dims": 1024, "cost": "Free"},

    # Cohere
    "embed-english-v3.0": {"dims": 1024, "cost": "$0.10/1M tokens"}
}
```

### Using Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Single text
embedding = model.encode("Hello, world!")
print(f"Shape: {embedding.shape}")  # (384,)

# Multiple texts (batched)
texts = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "The cat sat on the mat"
]
embeddings = model.encode(texts)
print(f"Shape: {embeddings.shape}")  # (3, 384)
```

### Using OpenAI Embeddings

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Hello, world!"
)

embedding = response.data[0].embedding
print(f"Dimensions: {len(embedding)}")  # 1536
```

## Semantic Similarity

### Cosine Similarity

```python
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (norm(a) * norm(b))

# Example
query = model.encode("What is machine learning?")
doc1 = model.encode("ML is a subset of artificial intelligence")
doc2 = model.encode("The recipe calls for two eggs")

sim1 = cosine_similarity(query, doc1)
sim2 = cosine_similarity(query, doc2)

print(f"Similarity to doc1: {sim1:.4f}")  # ~0.7
print(f"Similarity to doc2: {sim2:.4f}")  # ~0.1
```

### Other Distance Metrics

```python
from scipy.spatial.distance import euclidean, cityblock

def similarity_metrics(a, b):
    """Calculate various similarity/distance metrics"""
    return {
        "cosine": cosine_similarity(a, b),
        "euclidean": euclidean(a, b),  # Lower = more similar
        "manhattan": cityblock(a, b),   # Lower = more similar
        "dot_product": np.dot(a, b)     # Higher = more similar
    }
```

## Vector Databases

### ChromaDB

```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize
client = chromadb.Client()

# Use default embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = client.create_collection(
    name="documents",
    embedding_function=ef
)

# Add documents
collection.add(
    documents=[
        "Python is a programming language",
        "JavaScript runs in browsers",
        "Rust is known for memory safety"
    ],
    ids=["doc1", "doc2", "doc3"],
    metadatas=[
        {"topic": "python"},
        {"topic": "javascript"},
        {"topic": "rust"}
    ]
)

# Query
results = collection.query(
    query_texts=["What language is good for web?"],
    n_results=2
)
print(results)
```

### Pinecone

```python
import pinecone

# Initialize
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# Create index
pinecone.create_index(
    name="rag-index",
    dimension=384,
    metric="cosine"
)

# Connect to index
index = pinecone.Index("rag-index")

# Upsert vectors
index.upsert(vectors=[
    {
        "id": "doc1",
        "values": embedding1.tolist(),
        "metadata": {"text": "Original text", "source": "file.pdf"}
    },
    # ... more vectors
])

# Query
results = index.query(
    vector=query_embedding.tolist(),
    top_k=5,
    include_metadata=True
)
```

### FAISS

```python
import faiss
import numpy as np

# Prepare embeddings
embeddings = np.array(embeddings).astype('float32')
dimension = embeddings.shape[1]

# Create index
index = faiss.IndexFlatL2(dimension)  # L2 distance
# Or: faiss.IndexFlatIP(dimension) for inner product

# Add vectors
index.add(embeddings)

# Search
query = np.array([query_embedding]).astype('float32')
distances, indices = index.search(query, k=5)

print(f"Nearest neighbors: {indices}")
print(f"Distances: {distances}")
```

## Embedding Best Practices

### Preprocessing

```python
def preprocess_for_embedding(text):
    """Prepare text for embedding"""
    # Remove excessive whitespace
    text = " ".join(text.split())

    # Handle special characters
    text = text.replace("\n", " ")

    # Truncate if too long (model dependent)
    max_tokens = 512  # Typical limit
    words = text.split()
    if len(words) > max_tokens:
        text = " ".join(words[:max_tokens])

    return text
```

### Chunking for Embeddings

```python
def chunk_for_embeddings(text, chunk_size=500, overlap=50):
    """
    Chunk text into embedding-friendly sizes.
    Consider semantic boundaries.
    """
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        if current_length + sentence_length > chunk_size:
            chunks.append(". ".join(current_chunk) + ".")

            # Keep overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s.split()) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s.split())
                else:
                    break

            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks
```

### Caching Embeddings

```python
import hashlib
import pickle

class EmbeddingCache:
    def __init__(self, model, cache_dir="./embedding_cache"):
        self.model = model
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_embedding(self, text):
        # Create cache key
        key = hashlib.md5(text.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")

        # Check cache
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # Generate and cache
        embedding = self.model.encode(text)
        with open(cache_path, "wb") as f:
            pickle.dump(embedding, f)

        return embedding
```

## Exercises

1. Compare different embedding models on semantic similarity tasks
2. Build a simple vector search system with FAISS
3. Implement embedding caching for efficiency
4. Evaluate embedding quality with benchmark datasets

## Additional Resources

- [Sentence Transformers](https://www.sbert.net/)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
- [Chroma Documentation](https://docs.trychroma.com/)
