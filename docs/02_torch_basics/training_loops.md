# Training Loops

## Overview

Training a neural network involves iteratively updating model parameters to minimize a loss function. This section covers the complete training workflow, including forward propagation, backward propagation, optimization, and techniques for improving training efficiency.

## The Training Loop

### Basic Structure

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup
model = YourModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    model.train()  # Set to training mode
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch} completed. Average Loss: {epoch_loss:.4f}')
```

## Forward and Backward Propagation

### Understanding the Flow

```python
# Forward pass: Input → Model → Output → Loss
inputs = torch.randn(32, 784)
targets = torch.randint(0, 10, (32,))

# Forward
outputs = model(inputs)  # Compute predictions
loss = criterion(outputs, targets)  # Compute loss

# Backward pass: Loss → Gradients → Parameters
optimizer.zero_grad()  # Reset gradients to zero
loss.backward()        # Compute gradients via chain rule
optimizer.step()       # Update parameters using gradients

# Access gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f'{name}: grad norm = {param.grad.norm():.4f}')
```

### Gradient Accumulation

For effective larger batch sizes without running out of memory:

```python
accumulation_steps = 4  # Effective batch size = 32 * 4 = 128
optimizer.zero_grad()

for i, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps  # Normalize loss
    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Optimizers

### Common Optimizers

```python
# Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# SGD with weight decay (L2 regularization)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam (Adaptive Moment Estimation)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (Adam with decoupled weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Adagrad
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
```

### Per-Parameter Optimization

```python
# Different learning rates for different parts of the model
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},  # Lower LR for pretrained
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # Higher LR for new layers
])

# Only optimize certain parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=0.001)
```

## Learning Rate Scheduling

### Built-in Schedulers

```python
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR,
    CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
)

# Step decay: multiply LR by gamma every step_size epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Multi-step decay
scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

# Exponential decay
scheduler = ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Reduce on plateau (based on validation metric)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# One cycle policy (recommended for fast training)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs
)
```

### Using Schedulers in Training

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        # Training step
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()

        # Step scheduler per batch (for OneCycleLR)
        scheduler.step()

    # Step scheduler per epoch (for most schedulers)
    # scheduler.step()

    # For ReduceLROnPlateau, pass the validation metric
    val_loss = validate(model, val_loader)
    # scheduler.step(val_loss)

    # Check current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}, LR: {current_lr:.6f}')
```

## Gradient Clipping

Prevent exploding gradients, especially in RNNs:

```python
# Clip by norm (most common)
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Full training step with gradient clipping
for inputs, targets in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs.to(device))
    loss = criterion(outputs, targets.to(device))
    loss.backward()

    # Clip gradients before optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

## Complete Training Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(val_loader), 100. * correct / total


def train(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f'Saved best model with val_acc: {val_acc:.2f}%')

    return history
```

## Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

# Usage
early_stopping = EarlyStopping(patience=10, mode='min')

for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f'Early stopping at epoch {epoch}')
        break
```

## Mixed Precision Training

Speed up training and reduce memory usage:

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # Backward pass with scaler
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Best Practices

1. **Always use model.train() and model.eval()**: Affects dropout and batch norm
2. **Zero gradients before backward**: Prevents gradient accumulation
3. **Move data to device**: Ensure inputs and targets are on the same device as model
4. **Use gradient clipping for RNNs**: Prevents exploding gradients
5. **Monitor both training and validation metrics**: Detect overfitting early
6. **Save checkpoints regularly**: Recover from crashes

## Exercises

1. Implement a training loop with validation and early stopping
2. Compare different optimizers on the same task
3. Experiment with learning rate schedules
4. Implement gradient accumulation for larger effective batch sizes

## Additional Resources

- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
