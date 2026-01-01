# Parameter-Efficient Fine-Tuning (PEFT)

## Overview

Parameter-Efficient Fine-Tuning (PEFT) methods allow fine-tuning large language models by updating only a small fraction of the parameters. This dramatically reduces memory requirements and training time while achieving comparable performance to full fine-tuning.

## Why PEFT?

### The Problem with Full Fine-Tuning

```python
# Full fine-tuning memory requirements
model_sizes = {
    "Llama-2-7B": {
        "parameters": 7e9,
        "model_memory": "14 GB (FP16)",
        "gradients": "14 GB",
        "optimizer_states": "28 GB (Adam)",
        "total": "~56 GB minimum"
    },
    "Llama-2-70B": {
        "parameters": 70e9,
        "total": "~560 GB minimum"
    }
}
```

### PEFT Benefits

```python
peft_benefits = {
    "memory_efficient": "Train 7B model on single 24GB GPU",
    "faster_training": "Update fewer parameters = faster",
    "prevent_forgetting": "Base model frozen, less catastrophic forgetting",
    "modular": "Swap adapters for different tasks",
    "storage_efficient": "Save only small adapter weights"
}
```

## LoRA (Low-Rank Adaptation)

### How LoRA Works

```python
"""
LoRA adds trainable low-rank matrices to frozen attention layers.

Original: y = Wx
LoRA:     y = Wx + BAx

Where:
- W: Original frozen weights (d × k)
- B: Trainable matrix (d × r)
- A: Trainable matrix (r × k)
- r: Rank (typically 8-64, much smaller than d and k)

Parameters: r × (d + k) << d × k
"""

import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha

        # Get dimensions
        d, k = original_layer.weight.shape

        # LoRA matrices
        self.A = nn.Parameter(torch.randn(rank, k) * 0.01)
        self.B = nn.Parameter(torch.zeros(d, rank))

        # Freeze original weights
        self.original.weight.requires_grad = False

    def forward(self, x):
        # Original output
        original_out = self.original(x)

        # LoRA output: (x @ A.T) @ B.T = x @ (BA).T
        lora_out = x @ self.A.T @ self.B.T

        # Scale by alpha/rank
        return original_out + (self.alpha / self.rank) * lora_out
```

### Using PEFT Library

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                           # Rank
    lora_alpha=32,                  # Scaling factor
    lora_dropout=0.05,              # Dropout
    target_modules=[                # Which layers to adapt
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Check parameter counts
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622%
```

### LoRA Hyperparameters

```python
lora_guidelines = {
    "rank_r": {
        "4-8": "Small tasks, limited data",
        "16-32": "Medium complexity tasks",
        "64+": "Complex tasks, lots of data"
    },
    "lora_alpha": {
        "rule": "Often set to 2 × r",
        "effect": "Higher = stronger adaptation"
    },
    "target_modules": {
        "attention_only": ["q_proj", "v_proj"],
        "all_attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "plus_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "dropout": {
        "0.0": "Large datasets",
        "0.05-0.1": "Medium datasets",
        "0.1-0.2": "Small datasets"
    }
}
```

## QLoRA (Quantized LoRA)

### 4-bit Quantization + LoRA

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Memory usage: ~6GB for 7B model (vs 56GB full fine-tuning)
```

## Training with PEFT

### Complete Training Script

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load and prepare dataset
dataset = load_dataset("your_dataset")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# Save LoRA adapter
model.save_pretrained("./lora_adapter")
```

### Loading and Using LoRA

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# Merge for faster inference (optional)
merged_model = model.merge_and_unload()

# Generate
inputs = tokenizer("Hello, my name is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Other PEFT Methods

### Prefix Tuning

```python
from peft import PrefixTuningConfig

config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,  # Learnable prefix length
    encoder_hidden_size=128
)

# Adds learnable tokens at the beginning of each layer
# Parameters: num_layers × num_virtual_tokens × hidden_size
```

### Prompt Tuning

```python
from peft import PromptTuningConfig

config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=10,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify the sentiment: "
)

# Only adds learnable embeddings at input
# Even fewer parameters than prefix tuning
```

### Adapter Layers

```python
from peft import AdaptionPromptConfig

config = AdaptionPromptConfig(
    adapter_layers=30,
    adapter_len=10
)

# Inserts small trainable modules between layers
```

## Comparing PEFT Methods

| Method | Parameters | Memory | Performance | Use Case |
|--------|------------|--------|-------------|----------|
| LoRA | ~0.1-1% | Low | High | General purpose |
| QLoRA | ~0.1% | Very Low | High | Limited GPU memory |
| Prefix Tuning | ~0.01% | Very Low | Medium | Simple tasks |
| Prompt Tuning | ~0.001% | Minimal | Lower | Quick experiments |
| Full Fine-Tuning | 100% | Very High | Highest | Unlimited resources |

## Best Practices

```python
peft_best_practices = [
    "Start with LoRA r=16, increase if needed",
    "Target attention projections first (q_proj, v_proj)",
    "Use QLoRA for memory constraints",
    "Set learning rate higher than full fine-tuning (1e-4 to 3e-4)",
    "Monitor for overfitting (small adapter = easier to overfit)",
    "Merge adapters for production inference speed",
    "Keep base model version for reproducibility"
]
```

## Exercises

1. Fine-tune a model with LoRA on a classification task
2. Compare different rank values (4, 16, 64)
3. Implement QLoRA for a larger model
4. Merge and deploy a LoRA model

## Additional Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
