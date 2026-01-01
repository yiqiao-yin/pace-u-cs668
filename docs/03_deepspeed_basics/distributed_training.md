# Distributed Training

## Overview

Distributed training is essential for training large models efficiently. DeepSpeed supports various distributed training strategies including data parallelism, pipeline parallelism, tensor parallelism, and combinations thereof. This section covers how to configure and execute distributed training at scale.

## Data Parallelism

### Basic Concept

In data parallelism, the model is replicated across GPUs, and each GPU processes a different portion of the data.

```python
# Standard PyTorch DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model, device_ids=[local_rank])
```

### DeepSpeed Data Parallelism

```python
import deepspeed

# DeepSpeed handles distribution automatically
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# Training is same as single GPU
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### Configuration

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 1,

    "data_parallel_size": 8,  // Automatically determined from num_gpus

    "communication_data_type": "fp16",

    "prescale_gradients": false,
    "gradient_predivide_factor": 1.0
}
```

## Multi-GPU Training

### Launching on Single Node

```bash
# Method 1: DeepSpeed launcher
deepspeed --num_gpus=4 train.py \
    --deepspeed \
    --deepspeed_config ds_config.json

# Method 2: torchrun
torchrun --nproc_per_node=4 train.py \
    --deepspeed \
    --deepspeed_config ds_config.json

# Method 3: Specify GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 train.py
```

### Training Script Structure

```python
import argparse
import torch
import deepspeed
from torch.utils.data import DataLoader, DistributedSampler

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Set device
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)

    # Create model
    model = create_model()

    # Create dataset with distributed sampler
    train_dataset = create_dataset()
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler
    )

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Important for shuffling

        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

if __name__ == '__main__':
    main()
```

## Multi-Node Training

### Hostfile Configuration

```bash
# Create hostfile
# Format: hostname slots=num_gpus

# hostfile
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
```

### SSH Setup

```bash
# Ensure passwordless SSH between nodes
ssh-keygen -t rsa
ssh-copy-id user@node1
ssh-copy-id user@node2
# ... for all nodes

# Verify connectivity
ssh node1 hostname
ssh node2 hostname
```

### Launching Multi-Node Training

```bash
# From master node
deepspeed --hostfile=hostfile \
    --master_addr=node1 \
    --master_port=29500 \
    train.py \
    --deepspeed \
    --deepspeed_config ds_config.json

# Or with specific node selection
deepspeed --hostfile=hostfile \
    --include="node1:0,1,2,3@node2:0,1,2,3" \
    train.py
```

### MPI-based Launching

```bash
# Using mpirun
mpirun -np 32 \
    --hostfile hostfile \
    -bind-to none \
    -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH \
    python train.py --deepspeed
```

## Pipeline Parallelism

### Concept

Pipeline parallelism splits the model vertically across GPUs, with each GPU handling different layers.

```python
# Example: 4 GPUs with pipeline parallelism
# GPU 0: Layers 1-10
# GPU 1: Layers 11-20
# GPU 2: Layers 21-30
# GPU 3: Layers 31-40
```

### Configuration

```json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 1,

    "pipeline": {
        "pipe_partitioned": true,
        "grad_partitioned": true,
        "stages": 4
    },

    "zero_optimization": {
        "stage": 1
    }
}
```

### Implementation

```python
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

# Define model as sequential layers
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)

# Convert to pipeline-parallel model
def get_layer_specs():
    return [
        LayerSpec(nn.Embedding, vocab_size, hidden_size),
        *[LayerSpec(TransformerLayer, hidden_size) for _ in range(num_layers)],
        LayerSpec(nn.Linear, hidden_size, vocab_size)
    ]

model = PipelineModule(
    layers=get_layer_specs(),
    num_stages=4,  # Number of pipeline stages
    partition_method='parameters'  # or 'type:TransformerLayer'
)

# Initialize with pipeline config
engine, _, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# Training with pipeline
for batch in dataloader:
    loss = engine.train_batch(data_iter=iter([batch]))
```

## Model Parallelism

### Tensor Parallelism

Splits individual layers across GPUs.

```python
# Example: Linear layer split across 4 GPUs
# Original: (hidden_size, hidden_size) = (4096, 4096)
# Each GPU: (4096, 1024) - processes 1/4 of output features

import deepspeed
from deepspeed.moe.layer import MoE

# DeepSpeed provides utilities for tensor parallel layers
# For transformers, use Megatron-DeepSpeed integration
```

### 3D Parallelism

Combining data, pipeline, and tensor parallelism:

```python
# 3D Parallelism example with 64 GPUs
# - 2-way tensor parallelism
# - 4-way pipeline parallelism
# - 8-way data parallelism

config = {
    "train_batch_size": 256,

    "tensor_parallel": {
        "tp_size": 2
    },

    "pipeline": {
        "stages": 4
    },

    # Data parallelism is automatic:
    # dp_size = 64 / (2 * 4) = 8
}
```

## Communication Optimization

### NCCL Configuration

```bash
# Environment variables for optimal NCCL performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_LEVEL=NVL  # NVLink if available
```

### Gradient Communication

```json
{
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    },

    "communication_data_type": "fp16",

    "fp16": {
        "enabled": true,
        "communication_reduce_bucket_size": 5e8
    }
}
```

### 1-bit Adam

Reduces communication by compressing gradients:

```json
{
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "freeze_step": 400,
            "cuda_aware": true,
            "comm_backend_name": "nccl"
        }
    }
}
```

## Checkpointing in Distributed Training

### Saving Checkpoints

```python
# DeepSpeed handles distributed checkpointing
if model_engine.local_rank == 0:
    # Only save on rank 0
    model_engine.save_checkpoint(
        save_dir='./checkpoints',
        tag=f'step_{step}'
    )

# Or use automatic checkpointing
config = {
    "checkpoint": {
        "tag": "my_checkpoint",
        "save_dir": "./checkpoints",
        "save_on_each_node": false
    }
}
```

### Loading Checkpoints

```python
# Load checkpoint
_, client_state = model_engine.load_checkpoint(
    load_dir='./checkpoints',
    tag='step_10000'
)

# Resume training
start_step = client_state.get('step', 0)
```

## Monitoring and Debugging

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

if model_engine.local_rank == 0:
    writer = SummaryWriter('runs/experiment')

    # Log metrics
    writer.add_scalar('Loss/train', loss.item(), step)
    writer.add_scalar('LR', model_engine.get_lr()[0], step)
```

### DeepSpeed Monitoring

```json
{
    "tensorboard": {
        "enabled": true,
        "output_path": "./tb_logs",
        "job_name": "training_run"
    },

    "steps_per_print": 100,
    "wall_clock_breakdown": true
}
```

## Exercises

1. Set up multi-GPU training on a single node
2. Configure pipeline parallelism for a transformer model
3. Implement checkpointing for fault-tolerant training
4. Measure communication overhead with different configurations

## Additional Resources

- [DeepSpeed Distributed Training](https://www.deepspeed.ai/getting-started/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Pipeline Parallelism Tutorial](https://www.deepspeed.ai/tutorials/pipeline/)
