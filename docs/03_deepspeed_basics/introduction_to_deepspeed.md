# Introduction to DeepSpeed

## Overview

DeepSpeed is a deep learning optimization library developed by Microsoft that enables training of large-scale models efficiently. It provides memory optimization, distributed training capabilities, and performance improvements that make training massive models feasible on limited hardware.

## Why DeepSpeed?

### The Challenge of Large Models

Modern deep learning models have grown exponentially in size:
- GPT-3: 175 billion parameters
- Megatron-Turing: 530 billion parameters
- Training these models requires enormous amounts of GPU memory and compute

### DeepSpeed Solutions

DeepSpeed addresses these challenges through:
1. **ZeRO (Zero Redundancy Optimizer)**: Reduces memory redundancy across GPUs
2. **Mixed Precision Training**: Reduces memory and increases speed
3. **Pipeline Parallelism**: Splits model across GPUs vertically
4. **Tensor Parallelism**: Splits individual layers across GPUs
5. **CPU/NVMe Offloading**: Uses CPU RAM and storage for overflow

## Installation and Setup

### Installing DeepSpeed

```bash
# Basic installation
pip install deepspeed

# With specific features
pip install deepspeed[1bit_adam]  # 1-bit Adam optimizer
pip install deepspeed[sparse_attn]  # Sparse attention

# Verify installation
ds_report
```

### System Requirements

```bash
# Check CUDA version
nvcc --version

# Check GPU availability
nvidia-smi

# DeepSpeed environment report
ds_report
```

### Environment Setup

```bash
# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0-3
export NCCL_DEBUG=INFO  # Debug NCCL communication
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
```

## Integration with PyTorch

### Basic Integration

```python
import torch
import deepspeed
from torch import nn

# Define your model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleModel()

# DeepSpeed configuration
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4
        }
    },
    "fp16": {
        "enabled": True
    }
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)
```

### Training Loop with DeepSpeed

```python
# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(model_engine.local_rank)
        labels = labels.to(model_engine.local_rank)

        # Forward pass
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        # Backward pass (DeepSpeed handles optimizer.zero_grad())
        model_engine.backward(loss)

        # Update weights
        model_engine.step()

        print(f"Loss: {loss.item()}")
```

## DeepSpeed Configuration File

### Using JSON Configuration

```json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 8,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 1000
        }
    },

    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "gradient_clipping": 1.0,

    "zero_optimization": {
        "stage": 2
    }
}
```

### Loading Configuration

```python
import deepspeed
import argparse

# Add DeepSpeed arguments
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# Initialize with config file
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters()
)
```

## Launching DeepSpeed Training

### Single-Node Multi-GPU

```bash
# Using deepspeed launcher
deepspeed --num_gpus=4 train.py --deepspeed --deepspeed_config ds_config.json

# Using torchrun
torchrun --nproc_per_node=4 train.py --deepspeed --deepspeed_config ds_config.json
```

### Multi-Node Training

```bash
# Create hostfile
# hostfile content:
# node1 slots=8
# node2 slots=8

# Launch across nodes
deepspeed --hostfile=hostfile train.py \
    --deepspeed \
    --deepspeed_config ds_config.json
```

## Key Concepts

### Batch Size Terminology

```python
# Understanding batch sizes in DeepSpeed
config = {
    "train_batch_size": 128,  # Global batch size across all GPUs
    "train_micro_batch_size_per_gpu": 8,  # Batch per GPU per step
    "gradient_accumulation_steps": 4  # Steps before weight update
}

# Relationship:
# train_batch_size = train_micro_batch_size_per_gpu *
#                    gradient_accumulation_steps *
#                    num_gpus
# 128 = 8 * 4 * 4 (with 4 GPUs)
```

### Mixed Precision Training

```python
# FP16 configuration
fp16_config = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,  # Dynamic loss scaling
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}

# BF16 configuration (better for training stability)
bf16_config = {
    "bf16": {
        "enabled": True
    }
}
```

## Comparison with Other Approaches

| Feature | PyTorch DDP | DeepSpeed | FSDP |
|---------|-------------|-----------|------|
| Data Parallelism | ✓ | ✓ | ✓ |
| Memory Optimization | - | ZeRO | Sharding |
| CPU Offloading | - | ✓ | ✓ |
| Activation Checkpointing | Manual | ✓ | ✓ |
| Easy Configuration | - | ✓ | - |

## When to Use DeepSpeed

### Good Use Cases
- Training models that don't fit on a single GPU
- Multi-GPU and multi-node training
- Need for memory optimization
- Large-scale production training

### Alternatives
- Single small model on one GPU: Use vanilla PyTorch
- Simple distributed training: Use PyTorch DDP
- Hugging Face models: Consider Accelerate library

## Basic Example: Complete Training Script

```python
import torch
import deepspeed
import argparse
from torch.utils.data import DataLoader, TensorDataset

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# Create model
model = torch.nn.Linear(100, 10)

# Create dummy data
X = torch.randn(1000, 100)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32)

# DeepSpeed config
ds_config = {
    "train_batch_size": 32,
    "optimizer": {"type": "Adam", "params": {"lr": 0.001}},
    "fp16": {"enabled": True}
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Training loop
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(10):
    for inputs, labels in dataloader:
        inputs = inputs.to(model_engine.local_rank)
        labels = labels.to(model_engine.local_rank)

        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()

    if model_engine.local_rank == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Exercises

1. Install DeepSpeed and run the environment report
2. Convert a PyTorch training script to use DeepSpeed
3. Experiment with different batch sizes and gradient accumulation
4. Compare training speed with and without mixed precision

## Additional Resources

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [DeepSpeed Examples](https://github.com/microsoft/DeepSpeedExamples)
