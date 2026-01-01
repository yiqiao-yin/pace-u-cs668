# Tokenization and Context Management

## Overview

Tokenization is the process of converting text into tokens that language models can process. Understanding tokenization and effectively managing context windows is essential for building efficient LLM applications.

## How Tokenization Works

### Token Basics

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, how are you doing today?"

# Tokenize
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing', 'Ġtoday', '?']

# Token IDs
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")
# [15496, 11, 703, 389, 345, 1804, 1909, 30]

# Decode back to text
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")
# "Hello, how are you doing today?"
```

### Tokenization Methods

```python
# Different tokenization approaches

# 1. Word-based (simple, large vocabulary)
text = "unhappiness"
word_tokens = ["unhappiness"]  # Single token

# 2. Character-based (small vocab, long sequences)
char_tokens = list("unhappiness")  # ['u', 'n', 'h', ...]

# 3. Subword (BPE, WordPiece) - Best balance
# Byte-Pair Encoding example
bpe_tokens = ["un", "happiness"]  # Common subwords
```

### Comparing Tokenizers

```python
from transformers import AutoTokenizer

models = ["gpt2", "bert-base-uncased", "meta-llama/Llama-2-7b-hf"]
text = "Tokenization is fascinating!"

for model_name in models:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.tokenize(text)
        print(f"{model_name}: {len(tokens)} tokens")
        print(f"  Tokens: {tokens}\n")
    except:
        print(f"Cannot load {model_name}")
```

## Token Counting

### Using tiktoken (OpenAI)

```python
import tiktoken

def count_tokens_openai(text, model="gpt-4"):
    """Count tokens for OpenAI models"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Example
text = "Hello, world! This is a test message."
print(f"GPT-4 tokens: {count_tokens_openai(text)}")
```

### Using Transformers

```python
from transformers import AutoTokenizer

def count_tokens_hf(text, model_name):
    """Count tokens using Hugging Face tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(text))

# Example
text = "Hello, world! This is a test message."
print(f"Llama tokens: {count_tokens_hf(text, 'meta-llama/Llama-2-7b-hf')}")
```

### Estimating Tokens

```python
def estimate_tokens(text):
    """
    Rough estimation for English text.
    Generally: 1 token ≈ 4 characters or ~0.75 words
    """
    char_estimate = len(text) / 4
    word_estimate = len(text.split()) / 0.75
    return int((char_estimate + word_estimate) / 2)

# More accurate for specific content types
def estimate_by_content(text, content_type="english"):
    multipliers = {
        "english": 0.25,  # tokens per character
        "code": 0.35,     # More tokens for code
        "chinese": 0.5,   # More for CJK characters
        "numbers": 0.4    # Numbers often split
    }
    return int(len(text) * multipliers.get(content_type, 0.25))
```

## Context Window Management

### Understanding Context Windows

```python
context_limits = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "claude-3-opus": 200000,
    "llama-2": 4096,
    "llama-3": 8192
}

def check_fits_context(text, model, buffer=100):
    """Check if text fits in model's context window"""
    tokens = count_tokens_openai(text, model)
    limit = context_limits.get(model, 4096)
    return tokens < (limit - buffer)
```

### Truncation Strategies

```python
import tiktoken

def truncate_text(text, max_tokens, model="gpt-4", strategy="end"):
    """
    Truncate text to fit token limit.
    Strategies: 'start', 'end', 'middle'
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    if strategy == "end":
        # Keep beginning, cut end
        tokens = tokens[:max_tokens]
    elif strategy == "start":
        # Cut beginning, keep end
        tokens = tokens[-max_tokens:]
    elif strategy == "middle":
        # Keep start and end, cut middle
        half = max_tokens // 2
        tokens = tokens[:half] + tokens[-half:]

    return encoding.decode(tokens)

# Example
long_text = "..." * 10000
truncated = truncate_text(long_text, 1000, strategy="middle")
```

### Sliding Window

```python
def sliding_window_chunks(text, window_size, overlap, model="gpt-4"):
    """
    Split text into overlapping chunks for processing.
    Useful for long document analysis.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + window_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append({
            "text": chunk_text,
            "start_token": start,
            "end_token": min(end, len(tokens))
        })
        start += window_size - overlap

    return chunks

# Example: Process a book
chunks = sliding_window_chunks(book_text, window_size=3000, overlap=200)
for i, chunk in enumerate(chunks):
    result = process_chunk(chunk["text"])
```

## Chunking Strategies

### Simple Chunking

```python
def chunk_by_tokens(text, chunk_size, model="gpt-4"):
    """Split text into chunks of specified token size"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))

    return chunks
```

### Semantic Chunking

```python
import re

def chunk_by_paragraphs(text, max_tokens, model="gpt-4"):
    """Chunk by paragraphs, respecting token limit"""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens_openai(para, model)

        if current_tokens + para_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
```

### Recursive Chunking

```python
def recursive_chunk(text, max_tokens, separators=None, model="gpt-4"):
    """
    Recursively split text using different separators.
    Tries to maintain semantic coherence.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if count_tokens_openai(text, model) <= max_tokens:
        return [text]

    separator = separators[0]
    remaining_separators = separators[1:]

    parts = text.split(separator)
    chunks = []
    current = ""

    for part in parts:
        if count_tokens_openai(current + separator + part, model) <= max_tokens:
            current = current + separator + part if current else part
        else:
            if current:
                chunks.append(current)
            if count_tokens_openai(part, model) > max_tokens and remaining_separators:
                # Recursively chunk with finer separator
                chunks.extend(recursive_chunk(part, max_tokens, remaining_separators, model))
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks
```

## Managing Long Documents

### Summarization Chain

```python
def process_long_document(document, query, max_chunk_tokens=3000):
    """
    Process long document by chunking and summarizing.
    """
    # Chunk the document
    chunks = recursive_chunk(document, max_chunk_tokens)

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarize(chunk, query)
        summaries.append(summary)

    # Combine summaries
    combined = "\n\n".join(summaries)

    # If combined is still too long, recursively process
    if count_tokens_openai(combined) > max_chunk_tokens:
        return process_long_document(combined, query, max_chunk_tokens)

    # Final answer
    return generate_final_answer(combined, query)
```

### Map-Reduce Pattern

```python
def map_reduce_process(documents, question):
    """
    Map-reduce pattern for processing multiple documents.
    """
    # Map: Process each document
    mapped_results = []
    for doc in documents:
        result = process_single_document(doc, question)
        mapped_results.append(result)

    # Reduce: Combine results
    combined = "\n".join(mapped_results)
    final_answer = synthesize_answers(combined, question)

    return final_answer
```

## Token Budgeting

### Budget Allocation

```python
class TokenBudget:
    def __init__(self, model, max_output_tokens=1000):
        self.model = model
        self.max_context = context_limits.get(model, 4096)
        self.max_output = max_output_tokens
        self.available_input = self.max_context - max_output_tokens

    def allocate(self, system_tokens, examples_tokens):
        """Allocate tokens for different parts of the prompt"""
        reserved = system_tokens + examples_tokens
        return {
            "system": system_tokens,
            "examples": examples_tokens,
            "user_content": self.available_input - reserved,
            "output": self.max_output
        }

    def fits(self, system, examples, user_content):
        """Check if all content fits"""
        total = (count_tokens_openai(system) +
                count_tokens_openai(examples) +
                count_tokens_openai(user_content))
        return total <= self.available_input

# Usage
budget = TokenBudget("gpt-4", max_output_tokens=500)
allocation = budget.allocate(system_tokens=100, examples_tokens=500)
print(f"Available for user content: {allocation['user_content']} tokens")
```

## Exercises

1. Compare tokenization across different models for the same text
2. Implement a smart document chunker that respects sentence boundaries
3. Build a token budget manager for conversation history
4. Create a sliding window processor for long documents

## Additional Resources

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- [Tiktoken Library](https://github.com/openai/tiktoken)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers)
