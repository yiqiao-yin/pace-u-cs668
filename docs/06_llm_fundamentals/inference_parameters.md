---
sidebar_position: 4
---

# Inference Parameters and Sampling

Understanding how to control LLM outputs through inference parameters is essential for getting consistent, high-quality results. This guide covers the key parameters that influence how language models generate text.

## Temperature

Temperature controls the randomness of the model's output by scaling the probability distribution over tokens.

```python
from openai import OpenAI

client = OpenAI()

# Low temperature (0.0-0.3): More deterministic, focused
response_low = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0.1
)

# Medium temperature (0.4-0.7): Balanced creativity
response_medium = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a tagline for a coffee shop"}],
    temperature=0.7
)

# High temperature (0.8-1.0+): More creative, diverse
response_high = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a creative poem about the moon"}],
    temperature=0.9
)
```

### Temperature Guidelines

| Use Case | Temperature | Behavior |
|----------|-------------|----------|
| Code generation | 0.0 - 0.2 | Precise, deterministic |
| Factual Q&A | 0.1 - 0.3 | Accurate, consistent |
| General conversation | 0.5 - 0.7 | Balanced |
| Creative writing | 0.7 - 0.9 | Varied, imaginative |
| Brainstorming | 0.8 - 1.0 | Highly diverse |

## Top-p (Nucleus Sampling)

Top-p sampling selects from the smallest set of tokens whose cumulative probability exceeds the threshold `p`.

```python
# Top-p = 0.9 means consider tokens that make up 90% of probability mass
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Describe a sunset"}],
    temperature=0.8,
    top_p=0.9  # Only consider tokens in the top 90% probability
)
```

### How Top-p Works

```python
# Conceptual example of nucleus sampling
def nucleus_sampling(logits, top_p=0.9):
    """
    Select tokens whose cumulative probability exceeds top_p
    """
    # Sort tokens by probability (descending)
    sorted_probs = sorted(probabilities, reverse=True)

    # Find cutoff where cumulative prob exceeds top_p
    cumulative = 0
    cutoff_index = 0
    for i, prob in enumerate(sorted_probs):
        cumulative += prob
        if cumulative >= top_p:
            cutoff_index = i
            break

    # Only sample from tokens up to cutoff
    return sample_from(sorted_probs[:cutoff_index + 1])
```

### Temperature vs Top-p

| Parameter | Effect | Best For |
|-----------|--------|----------|
| Temperature | Scales all probabilities | General creativity control |
| Top-p | Truncates vocabulary | Filtering unlikely tokens |

**Recommendation**: Use one or the other, not both. OpenAI recommends adjusting temperature OR top_p, not both simultaneously.

## Top-k Sampling

Top-k limits the model to only consider the k most likely tokens at each step.

```python
# Using Hugging Face transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_ids = tokenizer.encode("The future of AI is", return_tensors="pt")

# Generate with top-k sampling
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_k=50,  # Only consider top 50 tokens
    temperature=0.8
)

print(tokenizer.decode(output[0]))
```

### Comparison of Sampling Methods

```python
import numpy as np

def compare_sampling_methods(probs, k=10, p=0.9, temp=1.0):
    """
    Demonstrate different sampling approaches
    """
    # Apply temperature
    scaled_probs = np.power(probs, 1/temp)
    scaled_probs = scaled_probs / scaled_probs.sum()

    # Top-k: Keep only top k tokens
    top_k_indices = np.argsort(scaled_probs)[-k:]
    top_k_probs = scaled_probs.copy()
    top_k_probs[~np.isin(range(len(probs)), top_k_indices)] = 0
    top_k_probs = top_k_probs / top_k_probs.sum()

    # Top-p: Keep tokens until cumulative prob >= p
    sorted_indices = np.argsort(scaled_probs)[::-1]
    cumsum = np.cumsum(scaled_probs[sorted_indices])
    cutoff = np.searchsorted(cumsum, p) + 1
    top_p_indices = sorted_indices[:cutoff]
    top_p_probs = scaled_probs.copy()
    top_p_probs[~np.isin(range(len(probs)), top_p_indices)] = 0
    top_p_probs = top_p_probs / top_p_probs.sum()

    return {
        'temperature_scaled': scaled_probs,
        'top_k': top_k_probs,
        'top_p': top_p_probs
    }
```

## Frequency and Presence Penalties

These parameters help reduce repetition in generated text.

### Frequency Penalty

Penalizes tokens based on how often they've appeared in the text so far.

```python
# Higher frequency penalty = less repetition of common words
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a long paragraph about dogs"}],
    frequency_penalty=0.5  # Range: -2.0 to 2.0
)
```

### Presence Penalty

Penalizes tokens based on whether they've appeared at all (binary).

```python
# Higher presence penalty = more topic diversity
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "List creative business ideas"}],
    presence_penalty=0.6  # Range: -2.0 to 2.0
)
```

### Penalty Guidelines

| Penalty Type | Value | Effect |
|--------------|-------|--------|
| Frequency = 0 | No penalty | Default behavior |
| Frequency > 0 | Reduce repetition | Less word repetition |
| Frequency < 0 | Encourage repetition | More consistent vocabulary |
| Presence = 0 | No penalty | Default behavior |
| Presence > 0 | Encourage new topics | More diverse content |
| Presence < 0 | Stay on topic | More focused content |

## Max Tokens

Controls the maximum length of the generated response.

```python
# Limit response length
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=150  # Limit to ~150 tokens
)

# Check token usage
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

### Token Estimation

```python
import tiktoken

def estimate_tokens(text, model="gpt-4"):
    """Estimate token count for a given text"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Example usage
text = "Hello, how are you doing today?"
tokens = estimate_tokens(text)
print(f"Text: '{text}'")
print(f"Estimated tokens: {tokens}")
```

## Stop Sequences

Define custom strings that signal the model to stop generating.

```python
# Stop at specific patterns
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "List 3 fruits:\n1."}],
    stop=["\n4.", "###", "END"]  # Stop at any of these
)

# Useful for structured outputs
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "Generate a JSON object for a user profile. End with }"
    }],
    stop=["}"]  # Stop after closing brace
)
```

## Combining Parameters Effectively

### For Code Generation

```python
def generate_code(prompt):
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,      # Low for precision
        max_tokens=1000,      # Allow longer code
        stop=["```\n\n"],     # Stop after code block
        frequency_penalty=0   # Allow repeated patterns (loops, etc.)
    )
```

### For Creative Writing

```python
def generate_creative_text(prompt):
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,      # Higher for creativity
        top_p=0.95,           # Wide vocabulary
        max_tokens=500,
        presence_penalty=0.6,  # Encourage variety
        frequency_penalty=0.3  # Reduce repetition
    )
```

### For Factual Q&A

```python
def answer_question(question):
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}],
        temperature=0.2,       # Low for accuracy
        max_tokens=300,
        frequency_penalty=0,
        presence_penalty=0
    )
```

## Streaming Responses

For better user experience with long responses:

```python
# Stream response tokens as they're generated
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Best Practices

1. **Start with defaults** - Adjust only when needed
2. **Test systematically** - Change one parameter at a time
3. **Document your settings** - Track what works for your use case
4. **Consider your audience** - Deterministic for APIs, creative for content
5. **Monitor costs** - Higher max_tokens = higher cost

```python
# Configuration template
GENERATION_CONFIGS = {
    "precise": {
        "temperature": 0.1,
        "max_tokens": 500,
        "frequency_penalty": 0,
        "presence_penalty": 0
    },
    "balanced": {
        "temperature": 0.5,
        "max_tokens": 1000,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3
    },
    "creative": {
        "temperature": 0.9,
        "max_tokens": 1500,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.6
    }
}

def generate_with_config(prompt, config_name="balanced"):
    config = GENERATION_CONFIGS[config_name]
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        **config
    )
```

## Summary

| Parameter | Range | Default | Primary Use |
|-----------|-------|---------|-------------|
| temperature | 0-2 | 1 | Control randomness |
| top_p | 0-1 | 1 | Nucleus sampling |
| top_k | 1-vocab size | varies | Limit vocabulary |
| frequency_penalty | -2 to 2 | 0 | Reduce repetition |
| presence_penalty | -2 to 2 | 0 | Encourage diversity |
| max_tokens | 1-context limit | varies | Limit output length |
| stop | list of strings | None | Custom stop conditions |
