# Retrieval Techniques

## Overview

Effective retrieval is crucial for RAG system performance. This section covers various retrieval methods, from basic semantic search to advanced techniques like hybrid search, reranking, and query transformations.

## Basic Retrieval Methods

### Semantic Search

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def index(self, documents):
        """Index documents for search"""
        self.documents = documents
        self.embeddings = self.model.encode(documents)

    def search(self, query, k=5):
        """Search for relevant documents"""
        query_embedding = self.model.encode(query)

        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [
            {"document": self.documents[i], "score": similarities[i]}
            for i in top_indices
        ]
```

### Keyword Search (BM25)

```python
from rank_bm25 import BM25Okapi
import nltk

class BM25Search:
    def __init__(self):
        self.bm25 = None
        self.documents = []

    def index(self, documents):
        """Index documents using BM25"""
        self.documents = documents
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, k=5):
        """Search using BM25 scoring"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-k:][::-1]

        return [
            {"document": self.documents[i], "score": scores[i]}
            for i in top_indices
        ]
```

## Hybrid Search

### Combining Semantic and Keyword

```python
class HybridSearch:
    def __init__(self, alpha=0.5):
        self.semantic = SemanticSearch()
        self.keyword = BM25Search()
        self.alpha = alpha  # Weight for semantic vs keyword

    def index(self, documents):
        self.semantic.index(documents)
        self.keyword.index(documents)

    def search(self, query, k=5):
        """Combine semantic and keyword search"""
        # Get results from both
        semantic_results = self.semantic.search(query, k=k*2)
        keyword_results = self.keyword.search(query, k=k*2)

        # Normalize scores
        semantic_scores = self._normalize([r["score"] for r in semantic_results])
        keyword_scores = self._normalize([r["score"] for r in keyword_results])

        # Combine scores
        combined = {}
        for i, result in enumerate(semantic_results):
            doc = result["document"]
            combined[doc] = self.alpha * semantic_scores[i]

        for i, result in enumerate(keyword_results):
            doc = result["document"]
            if doc in combined:
                combined[doc] += (1 - self.alpha) * keyword_scores[i]
            else:
                combined[doc] = (1 - self.alpha) * keyword_scores[i]

        # Sort and return top k
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [{"document": doc, "score": score} for doc, score in sorted_docs[:k]]

    def _normalize(self, scores):
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
```

## Reranking

### Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, k=5):
        """Rerank documents using cross-encoder"""
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score each pair
        scores = self.model.predict(pairs)

        # Sort by score
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        return [
            {"document": doc, "score": score}
            for doc, score in ranked[:k]
        ]

# Usage
retriever = SemanticSearch()
retriever.index(documents)
reranker = Reranker()

# First stage: retrieve many candidates
candidates = retriever.search(query, k=20)

# Second stage: rerank top candidates
candidate_docs = [r["document"] for r in candidates]
final_results = reranker.rerank(query, candidate_docs, k=5)
```

### Cohere Rerank API

```python
import cohere

def cohere_rerank(query, documents, api_key, top_n=5):
    """Use Cohere's rerank API"""
    co = cohere.Client(api_key)

    results = co.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model="rerank-english-v2.0"
    )

    return [
        {"document": documents[r.index], "score": r.relevance_score}
        for r in results.results
    ]
```

## Query Processing

### Query Expansion

```python
def expand_query(query, llm):
    """Generate multiple versions of the query"""
    prompt = f"""
    Generate 3 alternative phrasings of this search query.
    Original: {query}

    Alternatives (one per line):
    """
    response = llm.generate(prompt)
    alternatives = response.strip().split("\n")

    return [query] + alternatives

# Usage
queries = expand_query("How do transformers work?", llm)
# ["How do transformers work?",
#  "Explain the transformer architecture",
#  "What is the mechanism behind transformer models?",
#  "Transformer neural network explanation"]

# Search with all queries
all_results = []
for q in queries:
    results = retriever.search(q, k=5)
    all_results.extend(results)

# Deduplicate and rerank
```

### HyDE (Hypothetical Document Embeddings)

```python
def hyde_retrieval(query, llm, retriever, k=5):
    """
    Generate a hypothetical answer, then search for similar documents.
    Often outperforms direct query search.
    """
    # Generate hypothetical document
    prompt = f"""
    Write a passage that would answer this question:
    {query}

    Passage:
    """
    hypothetical_doc = llm.generate(prompt)

    # Search using the hypothetical document
    return retriever.search(hypothetical_doc, k=k)
```

### Multi-Query Retrieval

```python
def multi_query_retrieval(query, llm, retriever, k=5):
    """Generate multiple queries and combine results"""
    # Generate diverse queries
    prompt = f"""
    Generate 3 different search queries to answer this question.
    Each query should approach the question from a different angle.

    Question: {query}

    Queries:
    1."""

    response = llm.generate(prompt)
    queries = [query] + parse_queries(response)

    # Retrieve with each query
    all_docs = set()
    doc_scores = {}

    for q in queries:
        results = retriever.search(q, k=k)
        for r in results:
            doc = r["document"]
            if doc not in doc_scores:
                doc_scores[doc] = []
            doc_scores[doc].append(r["score"])

    # Average scores for each document
    final_results = [
        {"document": doc, "score": sum(scores) / len(scores)}
        for doc, scores in doc_scores.items()
    ]

    return sorted(final_results, key=lambda x: x["score"], reverse=True)[:k]
```

## Maximum Marginal Relevance (MMR)

```python
def mmr_retrieval(query, retriever, k=5, lambda_param=0.5):
    """
    Balance relevance and diversity in results.
    Avoids returning very similar documents.
    """
    # Get more candidates than needed
    candidates = retriever.search(query, k=k*3)

    selected = []
    remaining = list(candidates)

    while len(selected) < k and remaining:
        best_score = float('-inf')
        best_doc = None
        best_idx = -1

        for i, candidate in enumerate(remaining):
            # Relevance to query
            relevance = candidate["score"]

            # Max similarity to already selected docs
            if selected:
                similarities = [
                    cosine_similarity(candidate["embedding"], s["embedding"])
                    for s in selected
                ]
                max_sim = max(similarities)
            else:
                max_sim = 0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_doc = candidate
                best_idx = i

        selected.append(best_doc)
        remaining.pop(best_idx)

    return selected
```

## Contextual Compression

```python
def compress_context(query, documents, llm):
    """Extract only relevant parts of documents"""
    compressed = []

    for doc in documents:
        prompt = f"""
        Extract only the parts relevant to answering: {query}

        Document:
        {doc}

        Relevant excerpt (or "Not relevant" if none):
        """
        response = llm.generate(prompt)

        if response.lower() != "not relevant":
            compressed.append(response)

    return compressed
```

## Exercises

1. Implement hybrid search combining BM25 and semantic search
2. Build a two-stage retrieval pipeline with reranking
3. Compare HyDE vs direct query search
4. Implement MMR for diverse results

## Additional Resources

- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Cross-Encoders vs Bi-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [HyDE Paper](https://arxiv.org/abs/2212.10496)
