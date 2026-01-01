# Introduction to Large Language Models

## Overview

Large Language Models (LLMs) are deep learning models trained on massive amounts of text data that can understand and generate human-like text. They form the foundation of modern generative AI applications.

## What are LLMs?

### Definition

LLMs are neural networks with billions of parameters trained on vast text corpora to predict the next token in a sequence. This simple objective leads to emergent capabilities like:

- Natural language understanding
- Text generation and completion
- Question answering
- Code generation
- Reasoning and problem-solving

### Key Characteristics

```python
# LLM characteristics
characteristics = {
    "scale": "Billions of parameters (7B to 1T+)",
    "training_data": "Trillions of tokens from diverse sources",
    "architecture": "Transformer-based (decoder-only typical)",
    "training": "Self-supervised on next-token prediction",
    "capabilities": "Emergent abilities at scale"
}
```

## Transformer Architecture

### Core Concepts

The Transformer architecture, introduced in "Attention Is All You Need" (2017), is the foundation of modern LLMs.

```
Input Tokens → Embedding → Transformer Blocks → Output Probabilities
                               ↓
                    [Attention + Feed-Forward] × N layers
```

### Self-Attention Mechanism

```python
# Simplified attention mechanism
import torch
import torch.nn.functional as F

def attention(query, key, value):
    """
    Compute scaled dot-product attention
    Q, K, V: (batch, seq_len, d_model)
    """
    d_k = query.size(-1)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights
```

### Model Components

```
Transformer Block:
├── Multi-Head Self-Attention
│   ├── Query projection
│   ├── Key projection
│   ├── Value projection
│   └── Output projection
├── Layer Normalization
├── Feed-Forward Network
│   ├── Linear (d_model → 4*d_model)
│   ├── Activation (GeLU)
│   └── Linear (4*d_model → d_model)
└── Residual Connections
```

## Pre-training vs Fine-tuning

### Pre-training

The initial training phase on massive datasets:

```python
# Pre-training objective: Next token prediction
def pretraining_loss(model, text):
    """
    Given: "The cat sat on the"
    Target: "cat sat on the mat"
    """
    input_ids = tokenize(text[:-1])
    target_ids = tokenize(text[1:])

    logits = model(input_ids)
    loss = cross_entropy(logits, target_ids)

    return loss
```

### Fine-tuning

Adapting pre-trained models for specific tasks:

```python
# Types of fine-tuning
fine_tuning_methods = {
    "full_fine_tuning": "Update all parameters",
    "instruction_tuning": "Train on instruction-response pairs",
    "rlhf": "Reinforcement Learning from Human Feedback",
    "lora": "Low-Rank Adaptation (parameter-efficient)",
    "prefix_tuning": "Train prefix tokens only"
}
```

## Model Sizes and Scaling

### Parameter Counts

```python
# Common model sizes
model_sizes = {
    "Small": "1-7B parameters",
    "Medium": "7-30B parameters",
    "Large": "30-70B parameters",
    "Very Large": "70B-200B parameters",
    "Massive": "200B+ parameters (GPT-4, Claude)"
}
```

### Scaling Laws

```python
# Chinchilla scaling laws (simplified)
# Optimal training: tokens ≈ 20 × parameters

def estimate_training_tokens(num_params):
    """Estimate optimal training tokens based on Chinchilla laws"""
    return 20 * num_params

# Example: 7B model should train on ~140B tokens
optimal_tokens = estimate_training_tokens(7e9)
print(f"Optimal training: {optimal_tokens/1e9:.0f}B tokens")
```

### Compute Requirements

| Model Size | GPU Memory (FP16) | Training Compute |
|------------|-------------------|------------------|
| 7B | ~14GB | ~10^23 FLOPs |
| 13B | ~26GB | ~10^23.5 FLOPs |
| 70B | ~140GB | ~10^24.5 FLOPs |
| 175B | ~350GB | ~10^25 FLOPs |

## Context Windows and Token Limits

### Understanding Tokens

```python
# Tokenization example
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, how are you today?"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', 'Ġtoday', '?']

# Token count
token_ids = tokenizer.encode(text)
print(f"Token count: {len(token_ids)}")
```

### Context Window Sizes

```python
context_windows = {
    "GPT-3.5": "4,096 tokens",
    "GPT-4": "8,192 / 32,768 / 128,000 tokens",
    "Claude 3": "200,000 tokens",
    "Llama 2": "4,096 tokens",
    "Llama 3": "8,192 / 128,000 tokens",
    "Gemini Pro": "32,768 / 1,000,000 tokens"
}
```

### Managing Context

```python
def estimate_tokens(text, chars_per_token=4):
    """Rough token estimation (English text)"""
    return len(text) // chars_per_token

def truncate_to_context(text, max_tokens, tokenizer):
    """Truncate text to fit context window"""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = tokenizer.decode(tokens)
    return text
```

## Popular LLM Platforms

### OpenAI GPT Models

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Anthropic Claude

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)

print(response.content[0].text)
```

### Meta Llama (via Hugging Face)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "[INST] Explain quantum computing in simple terms. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Model Capabilities

### Core Abilities

```python
capabilities = {
    "text_generation": "Complete text, write stories, articles",
    "question_answering": "Answer questions based on context or knowledge",
    "summarization": "Condense long documents",
    "translation": "Translate between languages",
    "code_generation": "Write and explain code",
    "reasoning": "Logic, math, multi-step problems",
    "conversation": "Engage in dialogue",
    "analysis": "Sentiment, intent, entity extraction"
}
```

### Limitations

```python
limitations = {
    "hallucination": "May generate plausible but incorrect information",
    "knowledge_cutoff": "No knowledge after training date",
    "context_limit": "Cannot process beyond context window",
    "bias": "May reflect biases in training data",
    "no_true_understanding": "Pattern matching, not comprehension",
    "inconsistency": "May give different answers to same question"
}
```

## Exercises

1. Compare token counts for the same text across different tokenizers
2. Experiment with different models on the same prompt
3. Test the limits of context windows
4. Explore model behavior at different temperature settings

## Additional Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
