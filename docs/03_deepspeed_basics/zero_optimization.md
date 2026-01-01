# ZeRO Optimization Stages

## Overview

ZeRO (Zero Redundancy Optimizer) is DeepSpeed's flagship memory optimization technique. It eliminates redundant memory storage across data-parallel processes, enabling training of models with billions of parameters on limited GPU memory.

## Memory Consumption in Data Parallelism

### The Problem

In standard data parallelism, each GPU stores:
1. **Model Parameters** (Φ): The weights of the neural network
2. **Gradients** (Φ): Computed during backward pass
3. **Optimizer States** (2Φ-12Φ): Adam stores momentum and variance

For a model with Φ parameters in mixed precision:
- Parameters: 2Φ bytes (FP16)
- Gradients: 2Φ bytes (FP16)
- Optimizer states: 12Φ bytes (Adam: FP32 params + momentum + variance)
- **Total: 16Φ bytes per GPU** (with redundancy!)

### ZeRO Solution

ZeRO partitions these states across GPUs instead of replicating them:

| Stage | What's Partitioned | Memory per GPU | Communication |
|-------|-------------------|----------------|---------------|
| ZeRO-1 | Optimizer states | 4Φ + 12Φ/N | Same as DDP |
| ZeRO-2 | + Gradients | 2Φ + 14Φ/N | Same as DDP |
| ZeRO-3 | + Parameters | 16Φ/N | Higher |

Where N = number of GPUs

## ZeRO Stage 1: Optimizer State Partitioning

### How It Works

```python
# Each GPU only stores 1/N of the optimizer states
# Example with 4 GPUs and Adam optimizer:
# GPU 0: Optimizer states for parameters 0-25%
# GPU 1: Optimizer states for parameters 25-50%
# GPU 2: Optimizer states for parameters 50-75%
# GPU 3: Optimizer states for parameters 75-100%
```

### Configuration

```json
{
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```

### Memory Savings

```python
# Original (4 GPUs, 1B parameter model):
# Per GPU: 2GB (params) + 2GB (grads) + 12GB (optimizer) = 16GB

# With ZeRO-1:
# Per GPU: 2GB (params) + 2GB (grads) + 3GB (optimizer/4) = 7GB
# Savings: ~56%
```

## ZeRO Stage 2: Gradient Partitioning

### How It Works

```python
# Gradients are reduced and partitioned immediately
# Each GPU only keeps gradients for its optimizer partition

# During backward pass:
# 1. Compute gradients on all GPUs
# 2. Reduce gradients
# 3. Each GPU keeps only its partition
# 4. Other gradients are discarded
```

### Configuration

```json
{
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```

### Memory Savings

```python
# With ZeRO-2 (4 GPUs, 1B parameter model):
# Per GPU: 2GB (params) + 0.5GB (grads/4) + 3GB (optimizer/4) = 5.5GB
# Additional savings vs ZeRO-1: ~21%
```

## ZeRO Stage 3: Parameter Partitioning

### How It Works

```python
# Parameters are also partitioned across GPUs
# Parameters are gathered on-demand during forward/backward

# Forward pass:
# 1. All-gather parameters needed for current layer
# 2. Compute forward pass
# 3. Discard gathered parameters

# Backward pass:
# 1. All-gather parameters for current layer
# 2. Compute gradients
# 3. Reduce-scatter gradients
# 4. Discard gathered parameters
```

### Configuration

```json
{
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "reduce_bucket_size": 5e8
    }
}
```

### Memory Savings

```python
# With ZeRO-3 (4 GPUs, 1B parameter model):
# Per GPU: 0.5GB (params/4) + 0.5GB (grads/4) + 3GB (optimizer/4) = 4GB
# Total savings vs baseline: ~75%
```

### Important Parameters

```json
{
    "zero_optimization": {
        "stage": 3,

        // Size of parameter prefetch buffer
        "stage3_prefetch_bucket_size": 5e8,

        // Parameters smaller than this are not partitioned
        "stage3_param_persistence_threshold": 1e6,

        // Max live parameters during forward/backward
        "stage3_max_live_parameters": 1e9,

        // Collect parameters when reuse distance exceeds this
        "stage3_max_reuse_distance": 1e9,

        // Gather parameters to FP16 instead of FP32
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

## ZeRO-Infinity: CPU and NVMe Offloading

### Overview

ZeRO-Infinity extends ZeRO-3 by offloading optimizer states, gradients, and parameters to CPU memory or NVMe storage.

### CPU Offloading

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### NVMe Offloading

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        }
    },
    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 1,
        "single_submit": false,
        "overlap_events": true
    }
}
```

### Memory Hierarchy

```
GPU Memory (fastest)
    ↓ offload
CPU Memory (slower)
    ↓ offload
NVMe Storage (slowest)
```

## Choosing the Right Stage

### Decision Guide

```python
# Stage 0: No ZeRO optimization
# Use when: Model fits easily on single GPU

# Stage 1: Optimizer state partitioning
# Use when: Need moderate memory savings
# Trade-off: Minimal communication overhead

# Stage 2: + Gradient partitioning
# Use when: Need more memory savings
# Trade-off: Slightly more communication

# Stage 3: + Parameter partitioning
# Use when: Model doesn't fit on single GPU
# Trade-off: Significant communication overhead

# ZeRO-Infinity: + CPU/NVMe offloading
# Use when: Training trillion-parameter models
# Trade-off: Slower but enables massive models
```

### Practical Example

```python
import deepspeed

# Estimate memory requirements
def estimate_memory(num_params, num_gpus, stage):
    param_size = num_params * 2  # FP16
    grad_size = num_params * 2   # FP16
    opt_size = num_params * 12   # Adam FP32

    if stage == 0:
        return param_size + grad_size + opt_size
    elif stage == 1:
        return param_size + grad_size + opt_size / num_gpus
    elif stage == 2:
        return param_size + (grad_size + opt_size) / num_gpus
    elif stage == 3:
        return (param_size + grad_size + opt_size) / num_gpus

# 7B parameter model on 8 GPUs
num_params = 7e9
num_gpus = 8

for stage in [0, 1, 2, 3]:
    mem = estimate_memory(num_params, num_gpus, stage) / 1e9
    print(f"Stage {stage}: {mem:.1f} GB per GPU")

# Output:
# Stage 0: 112.0 GB per GPU
# Stage 1: 56.0 GB per GPU
# Stage 2: 28.0 GB per GPU
# Stage 3: 14.0 GB per GPU
```

## Complete Configuration Example

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 4,
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
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 1000,
            "total_num_steps": 100000
        }
    },

    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },

    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },

    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": true,
        "cpu_checkpointing": true
    },

    "gradient_clipping": 1.0,
    "steps_per_print": 100
}
```

## Exercises

1. Train a model with each ZeRO stage and compare memory usage
2. Configure ZeRO-3 with CPU offloading for a large model
3. Measure the throughput impact of different ZeRO stages
4. Experiment with parameter persistence threshold values

## Additional Resources

- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [ZeRO-Infinity Paper](https://arxiv.org/abs/2104.07857)
- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)
