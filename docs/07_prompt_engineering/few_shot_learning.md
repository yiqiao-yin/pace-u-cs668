# Few-Shot Learning

## Overview

Few-shot learning is a prompting technique where you provide the model with a few examples of the desired input-output pattern before asking it to perform the task. This helps the model understand exactly what you want without fine-tuning.

## Zero-Shot vs Few-Shot

### Zero-Shot Prompting

```python
# No examples provided
zero_shot_prompt = """
Classify the following text as positive, negative, or neutral:

Text: "This product exceeded my expectations!"
Classification:
"""
# Model infers the task from instructions alone
```

### One-Shot Prompting

```python
# Single example provided
one_shot_prompt = """
Classify the following text as positive, negative, or neutral.

Example:
Text: "I love this product, it's amazing!"
Classification: positive

Now classify:
Text: "This product exceeded my expectations!"
Classification:
"""
```

### Few-Shot Prompting

```python
# Multiple examples provided
few_shot_prompt = """
Classify the following text as positive, negative, or neutral.

Examples:
Text: "I love this product, it's amazing!"
Classification: positive

Text: "The quality is terrible, very disappointed."
Classification: negative

Text: "The product works as described."
Classification: neutral

Now classify:
Text: "This product exceeded my expectations!"
Classification:
"""
```

## Selecting Good Examples

### Diversity

```python
# Good: Diverse examples covering different cases
examples = [
    {"input": "I love this!", "output": "positive"},           # Simple positive
    {"input": "Worst purchase ever.", "output": "negative"},    # Simple negative
    {"input": "It's okay, nothing special.", "output": "neutral"}, # Neutral
    {"input": "Great quality but slow shipping.", "output": "mixed"}, # Complex
]

# Bad: Similar examples that don't show range
bad_examples = [
    {"input": "I love it!", "output": "positive"},
    {"input": "I love this product!", "output": "positive"},
    {"input": "Amazing, I love it!", "output": "positive"},
]
```

### Quality

```python
# Good: Clear, unambiguous examples
quality_examples = [
    {
        "input": "Calculate 15% tip on $42.50",
        "output": "$6.38"
    },
    {
        "input": "Calculate 20% tip on $85.00",
        "output": "$17.00"
    }
]

# Bad: Inconsistent or incorrect examples
bad_examples = [
    {
        "input": "15% of 42.50",  # Different format
        "output": "about 6 dollars"  # Imprecise
    }
]
```

### Relevance

```python
def select_relevant_examples(query, examples, n=3):
    """Select examples most relevant to the query"""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode query and examples
    query_embedding = model.encode(query)
    example_embeddings = model.encode([e['input'] for e in examples])

    # Calculate similarities
    similarities = np.dot(example_embeddings, query_embedding)

    # Get top n indices
    top_indices = np.argsort(similarities)[-n:][::-1]

    return [examples[i] for i in top_indices]
```

## Formatting Examples

### Simple Format

```python
simple_format = """
Input: {input1}
Output: {output1}

Input: {input2}
Output: {output2}

Input: {new_input}
Output:"""
```

### Structured Format

```python
structured_format = """
### Example 1
**Input:** Write a haiku about spring
**Output:**
Cherry blossoms fall
Gentle rain awakens earth
New life emerges

### Example 2
**Input:** Write a haiku about winter
**Output:**
Silent snowflakes drift
Bare trees stand in frozen mist
World wrapped in white peace

### Your Turn
**Input:** Write a haiku about summer
**Output:**"""
```

### JSON Format

```python
json_format = """
I will provide examples of extracting structured data from text.

Example 1:
Input: "John Smith, born January 15, 1990, works as a software engineer at Google."
Output: {"name": "John Smith", "birth_date": "1990-01-15", "job": "software engineer", "company": "Google"}

Example 2:
Input: "Sarah Connor, DOB: March 3, 1985, is a data scientist at Meta."
Output: {"name": "Sarah Connor", "birth_date": "1985-03-03", "job": "data scientist", "company": "Meta"}

Now extract:
Input: "{input_text}"
Output:"""
```

## Balancing Example Quantity

### How Many Examples?

```python
# Guidelines for number of examples
num_examples_guide = {
    "simple_classification": "2-3 examples",
    "format_transformation": "2-4 examples",
    "complex_reasoning": "3-5 examples",
    "creative_tasks": "1-3 examples",
    "code_generation": "1-2 examples"
}

# More isn't always better - diminishing returns after 5-8 examples
# Consider token budget constraints
```

### Token Budget Consideration

```python
def optimize_examples(examples, max_tokens, query):
    """Select examples within token budget"""
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4")

    query_tokens = len(encoding.encode(query))
    output_buffer = 500  # Reserve for response
    available = max_tokens - query_tokens - output_buffer

    selected = []
    current_tokens = 0

    for example in examples:
        example_tokens = len(encoding.encode(str(example)))
        if current_tokens + example_tokens <= available:
            selected.append(example)
            current_tokens += example_tokens
        else:
            break

    return selected
```

## Dynamic Few-Shot Selection

### Similarity-Based Selection

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class DynamicFewShot:
    def __init__(self, examples):
        self.examples = examples
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode([e['input'] for e in examples])

    def select_examples(self, query, n=3):
        """Select n most similar examples to the query"""
        query_embedding = self.model.encode(query)
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-n:][::-1]
        return [self.examples[i] for i in top_indices]

    def create_prompt(self, query, n=3):
        """Create prompt with dynamically selected examples"""
        selected = self.select_examples(query, n)

        prompt = "Here are some examples:\n\n"
        for ex in selected:
            prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
        prompt += f"Input: {query}\nOutput:"

        return prompt

# Usage
examples = [
    {"input": "Paris is the capital of France", "output": "fact"},
    {"input": "The moon is made of cheese", "output": "fiction"},
    # ... more examples
]

selector = DynamicFewShot(examples)
prompt = selector.create_prompt("Water boils at 100 degrees Celsius")
```

### Category-Based Selection

```python
def select_by_category(query, examples, category_fn, n_per_category=1):
    """Select examples from each category"""
    categories = {}
    for ex in examples:
        cat = category_fn(ex)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(ex)

    selected = []
    for cat, cat_examples in categories.items():
        selected.extend(cat_examples[:n_per_category])

    return selected
```

## Few-Shot for Different Tasks

### Classification

```python
classification_prompt = """
Classify the customer support ticket into one of these categories:
- billing
- technical
- general
- urgent

Examples:
Ticket: "I was charged twice for my subscription"
Category: billing

Ticket: "The app crashes when I try to upload files"
Category: technical

Ticket: "How do I change my password?"
Category: general

Ticket: "My account was hacked and I can't access it"
Category: urgent

Ticket: "{new_ticket}"
Category:"""
```

### Data Extraction

```python
extraction_prompt = """
Extract product information from the description.

Example 1:
Description: "Apple iPhone 15 Pro, 256GB storage, Titanium Blue color, $999"
Extracted: {"product": "iPhone 15 Pro", "brand": "Apple", "storage": "256GB", "color": "Titanium Blue", "price": 999}

Example 2:
Description: "Samsung Galaxy S24 Ultra 512GB Phantom Black - $1199.99"
Extracted: {"product": "Galaxy S24 Ultra", "brand": "Samsung", "storage": "512GB", "color": "Phantom Black", "price": 1199.99}

Description: "{input_description}"
Extracted:"""
```

### Code Generation

```python
code_prompt = """
Write a Python function based on the description.

Example:
Description: Function that reverses a string
Code:
```python
def reverse_string(s: str) -> str:
    return s[::-1]
```

Example:
Description: Function that finds the maximum value in a list
Code:
```python
def find_max(lst: list) -> int:
    if not lst:
        raise ValueError("List is empty")
    return max(lst)
```

Description: {task_description}
Code:"""
```

## Best Practices

```python
best_practices = [
    "Use consistent formatting across all examples",
    "Include edge cases in examples",
    "Order examples from simple to complex",
    "Match example difficulty to expected queries",
    "Keep examples concise but complete",
    "Validate examples are correct before using",
    "Update examples based on failure cases"
]
```

## Exercises

1. Create a few-shot prompt for email classification
2. Implement dynamic example selection based on similarity
3. Compare zero-shot, one-shot, and few-shot performance
4. Build a few-shot system for a domain-specific task

## Additional Resources

- [Few-Shot Learners Paper](https://arxiv.org/abs/2005.14165)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
