# Introduction to Fine-Tuning

## Overview

Fine-tuning is the process of adapting a pre-trained language model to a specific task or domain by training it on additional data. This allows you to customize model behavior while leveraging the knowledge gained during pre-training.

## What is Fine-Tuning?

### The Concept

```python
# Pre-training: Learn general language understanding
pretrained_model = train_on_internet_scale_data()  # Trillions of tokens

# Fine-tuning: Adapt to specific task/domain
finetuned_model = continue_training(
    pretrained_model,
    task_specific_data  # Thousands to millions of examples
)
```

### Benefits of Fine-Tuning

```python
benefits = {
    "task_specialization": "Optimize for specific use cases",
    "domain_knowledge": "Learn specialized vocabulary and concepts",
    "style_adaptation": "Match desired tone and format",
    "improved_accuracy": "Better performance on target task",
    "reduced_prompting": "Less need for complex prompts",
    "lower_latency": "Smaller prompts, faster inference"
}
```

## When to Fine-Tune

### Decision Framework

```python
def should_finetune(use_case):
    """Decide if fine-tuning is appropriate"""

    finetune_if = [
        "Consistent style/format needed across outputs",
        "Domain-specific knowledge required",
        "Prompting alone doesn't achieve desired quality",
        "Have sufficient high-quality training data (100+ examples)",
        "Task requires following specific patterns",
        "Need to reduce prompt complexity"
    ]

    avoid_finetuning_if = [
        "Knowledge changes frequently (use RAG instead)",
        "Limited training data",
        "Simple task achievable with prompting",
        "Need to cite sources (use RAG)",
        "Quick iteration needed"
    ]

    return evaluate_criteria(use_case, finetune_if, avoid_finetuning_if)
```

### Fine-Tuning vs Alternatives

| Approach | Best For | Data Needed | Flexibility |
|----------|----------|-------------|-------------|
| Prompting | Quick experiments | None | High |
| Few-shot | Pattern learning | 2-10 examples | Medium |
| RAG | Dynamic knowledge | Documents | High |
| Fine-tuning | Behavior change | 100-10000+ examples | Low |

## Types of Fine-Tuning

### Full Fine-Tuning

```python
# Update all model parameters
# High compute, highest adaptation potential

from transformers import AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# All parameters are trainable
for param in model.parameters():
    param.requires_grad = True

# Train with all parameters
trainer = Trainer(model=model, ...)
trainer.train()
```

### Parameter-Efficient Fine-Tuning (PEFT)

```python
# Update only a small subset of parameters
# Lower compute, often similar performance

methods = {
    "LoRA": "Add low-rank matrices to attention layers",
    "QLoRA": "LoRA with quantized base model",
    "Prefix Tuning": "Learn prefix tokens",
    "Prompt Tuning": "Learn soft prompt embeddings",
    "Adapters": "Add small trainable modules"
}
```

### Instruction Fine-Tuning

```python
# Train model to follow instructions

training_data = [
    {
        "instruction": "Summarize this article",
        "input": "Long article text...",
        "output": "Concise summary..."
    },
    {
        "instruction": "Translate to French",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous?"
    }
]
```

## Basic Fine-Tuning Workflow

### 1. Prepare Data

```python
from datasets import Dataset

# Format: instruction + input + output
training_data = [
    {"text": "<s>[INST] Classify: great product! [/INST] positive</s>"},
    {"text": "<s>[INST] Classify: terrible service [/INST] negative</s>"},
    # ... more examples
]

dataset = Dataset.from_list(training_data)
```

### 2. Load Model and Tokenizer

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### 3. Configure Training

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./finetuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    fp16=True
)
```

### 4. Train

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
```

### 5. Evaluate and Save

```python
# Evaluate
metrics = trainer.evaluate()
print(f"Eval loss: {metrics['eval_loss']}")

# Save model
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
```

## Fine-Tuning Platforms

### OpenAI Fine-Tuning

```python
from openai import OpenAI

client = OpenAI()

# Upload training file
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo",
    hyperparameters={
        "n_epochs": 3
    }
)

# Monitor progress
status = client.fine_tuning.jobs.retrieve(job.id)
print(f"Status: {status.status}")

# Use fine-tuned model
response = client.chat.completions.create(
    model=job.fine_tuned_model,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Hugging Face + PEFT

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

## Common Pitfalls

```python
pitfalls = {
    "overfitting": {
        "symptom": "Low training loss, poor test performance",
        "solution": "More data, regularization, early stopping"
    },
    "catastrophic_forgetting": {
        "symptom": "Model forgets general knowledge",
        "solution": "Lower learning rate, mix in general data"
    },
    "bad_data": {
        "symptom": "Model learns wrong patterns",
        "solution": "Clean and validate training data"
    },
    "wrong_format": {
        "symptom": "Model doesn't follow instructions",
        "solution": "Use correct prompt template"
    }
}
```

## Exercises

1. Prepare a fine-tuning dataset for a classification task
2. Fine-tune a small model using Hugging Face Trainer
3. Compare LoRA vs full fine-tuning performance
4. Evaluate a fine-tuned model against the base model

## Additional Resources

- [Hugging Face Fine-Tuning](https://huggingface.co/docs/transformers/training)
- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [PEFT Library](https://github.com/huggingface/peft)
