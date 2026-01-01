# Training Data Preparation

## Overview

The quality and format of training data are crucial for successful fine-tuning. This section covers how to collect, clean, format, and validate data for fine-tuning language models.

## Data Collection

### Sources of Training Data

```python
data_sources = {
    "existing_conversations": "Customer support logs, chat histories",
    "human_annotation": "Manually created examples",
    "synthetic_generation": "LLM-generated examples",
    "public_datasets": "Hugging Face, Kaggle, etc.",
    "domain_documents": "Internal docs, manuals, guides",
    "user_feedback": "Corrected model outputs"
}
```

### Synthetic Data Generation

```python
from openai import OpenAI

def generate_training_examples(task_description, num_examples=100):
    """Generate synthetic training data using an LLM"""
    client = OpenAI()

    prompt = f"""
    Generate {num_examples} diverse training examples for this task:
    {task_description}

    Format each example as:
    INPUT: [input text]
    OUTPUT: [expected output]

    Make examples varied and representative of real use cases.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    return parse_examples(response.choices[0].message.content)
```

## Data Quality Assessment

### Quality Checks

```python
def assess_data_quality(examples):
    """Evaluate training data quality"""
    issues = []

    for i, example in enumerate(examples):
        # Check for empty or too short
        if len(example.get("input", "")) < 10:
            issues.append(f"Example {i}: Input too short")

        if len(example.get("output", "")) < 5:
            issues.append(f"Example {i}: Output too short")

        # Check for duplicates
        # (simplified - real implementation would use similarity)

        # Check for formatting issues
        if not example.get("input") or not example.get("output"):
            issues.append(f"Example {i}: Missing input or output")

        # Check for potential PII
        pii_patterns = ["@", "xxx-xx-xxxx", "password"]
        text = example.get("input", "") + example.get("output", "")
        for pattern in pii_patterns:
            if pattern in text.lower():
                issues.append(f"Example {i}: Possible PII detected")

    return issues

# Quality metrics
def compute_quality_metrics(examples):
    return {
        "total_examples": len(examples),
        "avg_input_length": sum(len(e["input"]) for e in examples) / len(examples),
        "avg_output_length": sum(len(e["output"]) for e in examples) / len(examples),
        "duplicate_rate": find_duplicate_rate(examples),
        "empty_rate": sum(1 for e in examples if not e["input"] or not e["output"]) / len(examples)
    }
```

## Data Formatting

### Instruction Format

```python
# Alpaca format
alpaca_format = {
    "instruction": "Summarize the following article",
    "input": "Long article text here...",
    "output": "Concise summary here..."
}

# ChatML format
chatml_format = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this article: ..."},
    {"role": "assistant", "content": "Here's the summary: ..."}
]

# Llama 2 format
llama2_format = """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Summarize this article: ... [/INST] Here's the summary: ...</s>"""
```

### Format Conversion Functions

```python
def to_alpaca_format(instruction, input_text, output_text):
    """Convert to Alpaca format"""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }

def to_chatml_format(system, user, assistant):
    """Convert to ChatML format"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]

def to_llama2_format(system, user, assistant):
    """Convert to Llama 2 chat format"""
    return f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] {assistant}</s>"""

def to_training_text(example, tokenizer):
    """Convert example to tokenizable text"""
    return tokenizer.apply_chat_template(
        example,
        tokenize=False,
        add_generation_prompt=False
    )
```

## Data Splitting

### Train/Validation/Test Split

```python
from sklearn.model_selection import train_test_split

def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train, validation, and test sets"""
    assert train_ratio + val_ratio + test_ratio == 1.0

    # First split: separate test set
    train_val, test = train_test_split(
        data,
        test_size=test_ratio,
        random_state=42
    )

    # Second split: separate validation from training
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_ratio,
        random_state=42
    )

    return {
        "train": train,
        "validation": val,
        "test": test
    }
```

### Stratified Splitting

```python
def stratified_split(data, label_key, train_ratio=0.8):
    """Split while maintaining label distribution"""
    labels = [example[label_key] for example in data]

    train, val = train_test_split(
        data,
        test_size=1-train_ratio,
        stratify=labels,
        random_state=42
    )

    return {"train": train, "validation": val}
```

## Data Augmentation

### Text Augmentation Techniques

```python
import random

def augment_text(text):
    """Simple text augmentation techniques"""
    augmented = []

    # Synonym replacement (simplified)
    augmented.append(replace_synonyms(text))

    # Random word deletion
    words = text.split()
    if len(words) > 5:
        idx = random.randint(0, len(words)-1)
        deleted = words[:idx] + words[idx+1:]
        augmented.append(" ".join(deleted))

    # Random word swap
    if len(words) > 2:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
        augmented.append(" ".join(words))

    return augmented

def augment_dataset(examples, augmentation_factor=2):
    """Augment dataset with variations"""
    augmented = []

    for example in examples:
        augmented.append(example)  # Keep original

        # Add augmented versions
        for _ in range(augmentation_factor - 1):
            aug_input = random.choice(augment_text(example["input"]))
            augmented.append({
                **example,
                "input": aug_input
            })

    return augmented
```

### Back-Translation

```python
def back_translate(text, source_lang="en", pivot_lang="de"):
    """Augment by translating to another language and back"""
    # Translate to pivot language
    translated = translate(text, source_lang, pivot_lang)

    # Translate back to source
    back_translated = translate(translated, pivot_lang, source_lang)

    return back_translated
```

## Handling Imbalanced Data

```python
from collections import Counter

def balance_dataset(examples, label_key):
    """Balance dataset by oversampling minority classes"""
    # Count labels
    label_counts = Counter(e[label_key] for e in examples)
    max_count = max(label_counts.values())

    balanced = []
    for label in label_counts:
        label_examples = [e for e in examples if e[label_key] == label]
        count = label_counts[label]

        # Oversample to match majority class
        while len(label_examples) < max_count:
            label_examples.append(random.choice([e for e in examples if e[label_key] == label]))

        balanced.extend(label_examples[:max_count])

    random.shuffle(balanced)
    return balanced
```

## Saving and Loading Data

### JSONL Format

```python
import json

def save_jsonl(data, filepath):
    """Save data in JSONL format (one JSON per line)"""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_jsonl(filepath):
    """Load JSONL data"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data
```

### Hugging Face Datasets

```python
from datasets import Dataset, DatasetDict

def create_hf_dataset(train_data, val_data):
    """Create Hugging Face dataset"""
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })

# Save to disk
dataset.save_to_disk("./my_dataset")

# Push to Hugging Face Hub
dataset.push_to_hub("username/my-dataset")
```

## Exercises

1. Create a synthetic dataset for a specific task
2. Implement data quality checks for your dataset
3. Convert data between different formats
4. Build a balanced dataset from imbalanced data

## Additional Resources

- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [Data Augmentation for NLP](https://github.com/makcedward/nlpaug)
