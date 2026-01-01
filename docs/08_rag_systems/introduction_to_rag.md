# Introduction to RAG

## Overview

Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by retrieving relevant information from external knowledge sources. This allows models to access up-to-date information and domain-specific knowledge beyond their training data.

## What is RAG?

### The Problem RAG Solves

```python
# LLM limitations:
limitations = {
    "knowledge_cutoff": "No information after training date",
    "hallucination": "May generate plausible but incorrect information",
    "domain_knowledge": "Limited expertise in specialized fields",
    "proprietary_data": "Cannot access private/internal documents"
}

# RAG solution:
# Retrieve relevant context → Augment prompt → Generate response
```

### RAG Architecture

```
User Query
    ↓
[Query Encoding] → Vector representation
    ↓
[Retrieval] ← Vector Database (documents indexed)
    ↓
[Relevant Documents]
    ↓
[Augmented Prompt] = Query + Retrieved Context
    ↓
[LLM Generation]
    ↓
Response (grounded in retrieved documents)
```

## RAG vs Fine-tuning

### When to Use Each

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| Knowledge updates | Easy (update documents) | Requires retraining |
| Data requirements | Works with any documents | Needs training examples |
| Cost | Lower (no training) | Higher (GPU training) |
| Transparency | Can cite sources | Black box |
| Accuracy | Good with quality retrieval | Can be very high |
| Latency | Retrieval adds overhead | Direct inference |

### Decision Framework

```python
def choose_approach(use_case):
    """Decide between RAG and fine-tuning"""

    use_rag_if = [
        "Knowledge changes frequently",
        "Need to cite sources",
        "Limited training data",
        "Quick deployment needed",
        "Multiple knowledge domains"
    ]

    use_finetuning_if = [
        "Stable, unchanging knowledge",
        "Need specific style/behavior",
        "Have abundant training data",
        "Latency is critical",
        "Knowledge is implicit (not factual)"
    ]

    # Often: Use both together!
    # Fine-tune for behavior, RAG for knowledge
```

## Basic RAG Pipeline

### Step 1: Document Ingestion

```python
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
```

### Step 2: Create Embeddings

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Initialize embedding model
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

### Step 3: Retrieval

```python
# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Return top 4 documents
)

# Retrieve relevant documents
query = "What is the main finding of the study?"
relevant_docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}: {doc.page_content[:200]}...")
```

### Step 4: Generation

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Combine all docs into prompt
    retriever=retriever,
    return_source_documents=True
)

# Query
result = qa_chain({"query": "What is the main finding?"})
print(result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']}")
```

## Use Cases

### Question Answering

```python
# Answer questions from documents
use_case_qa = """
Build a system that:
1. Ingests company documentation
2. Allows employees to ask questions
3. Returns answers with citations
4. Highlights confidence levels
"""
```

### Chatbots with Knowledge

```python
# Customer support chatbot
use_case_support = """
Build a support bot that:
1. Knows all product documentation
2. Answers customer questions accurately
3. Escalates when uncertain
4. Tracks common issues
"""
```

### Research Assistants

```python
# Academic research assistant
use_case_research = """
Build a research tool that:
1. Searches academic papers
2. Summarizes relevant findings
3. Identifies connections between papers
4. Generates literature reviews
"""
```

## Simple RAG Implementation

```python
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class SimpleRAG:
    def __init__(self):
        # Initialize components
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("docs")
        self.llm = OpenAI()

    def add_documents(self, documents):
        """Add documents to the knowledge base"""
        embeddings = self.embedder.encode(documents).tolist()
        ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )

    def query(self, question, n_results=3):
        """Query the RAG system"""
        # Retrieve relevant documents
        query_embedding = self.embedder.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Build context
        context = "\n\n".join(results['documents'][0])

        # Generate response
        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": results['documents'][0]
        }

# Usage
rag = SimpleRAG()
rag.add_documents([
    "Python is a programming language created by Guido van Rossum.",
    "Python 3.12 was released in October 2023.",
    "Python is known for its simple and readable syntax."
])

result = rag.query("Who created Python?")
print(result["answer"])
```

## RAG Evaluation

### Metrics

```python
evaluation_metrics = {
    "retrieval_metrics": {
        "precision_at_k": "Relevant docs in top-k results",
        "recall": "Proportion of relevant docs retrieved",
        "mrr": "Mean Reciprocal Rank"
    },
    "generation_metrics": {
        "faithfulness": "Is answer grounded in retrieved docs?",
        "relevance": "Does answer address the question?",
        "groundedness": "Are claims supported by sources?"
    },
    "end_to_end": {
        "accuracy": "Correctness of final answer",
        "latency": "Time to generate response",
        "user_satisfaction": "Human evaluation scores"
    }
}
```

## Exercises

1. Build a simple RAG system for a set of documents
2. Compare different chunking strategies
3. Evaluate retrieval quality with different embedding models
4. Add source citations to generated responses

## Additional Resources

- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
