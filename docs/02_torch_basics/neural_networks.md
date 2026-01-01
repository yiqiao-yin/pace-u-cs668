# Building Neural Networks

## Overview

PyTorch provides the `torch.nn` module for building neural networks. This section covers the essential components: layers, activation functions, loss functions, and how to combine them into complete models.

## The nn.Module Class

All neural networks in PyTorch inherit from `nn.Module`. This base class provides essential functionality for parameter management, GPU transfer, and model serialization.

### Basic Structure

```python
import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Define forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create model instance
model = SimpleNetwork()
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

## Layer Types

### Linear (Fully Connected) Layers

```python
# Linear layer: y = xW^T + b
linear = nn.Linear(in_features=100, out_features=50)

# Access weights and bias
print(linear.weight.shape)  # [50, 100]
print(linear.bias.shape)    # [50]

# No bias
linear_no_bias = nn.Linear(100, 50, bias=False)
```

### Convolutional Layers

```python
# 2D Convolution (for images)
conv2d = nn.Conv2d(
    in_channels=3,      # RGB image
    out_channels=64,    # Number of filters
    kernel_size=3,      # 3x3 kernel
    stride=1,           # Step size
    padding=1           # Preserve spatial dimensions
)

# Input shape: (batch, channels, height, width)
x = torch.randn(32, 3, 224, 224)
output = conv2d(x)
print(output.shape)  # [32, 64, 224, 224]

# 1D Convolution (for sequences)
conv1d = nn.Conv1d(
    in_channels=128,
    out_channels=256,
    kernel_size=5,
    padding=2
)

# Transposed Convolution (upsampling)
conv_transpose = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
```

### Recurrent Layers

```python
# LSTM
lstm = nn.LSTM(
    input_size=128,     # Feature dimension
    hidden_size=256,    # Hidden state dimension
    num_layers=2,       # Stacked LSTM layers
    batch_first=True,   # Input shape: (batch, seq, features)
    dropout=0.2,        # Dropout between layers
    bidirectional=True  # Bidirectional LSTM
)

# Input shape: (batch, sequence_length, features)
x = torch.randn(32, 50, 128)
output, (hidden, cell) = lstm(x)
print(output.shape)  # [32, 50, 512] (256 * 2 for bidirectional)

# GRU
gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
output, hidden = gru(x)
```

### Normalization Layers

```python
# Batch Normalization (normalizes across batch dimension)
batch_norm_2d = nn.BatchNorm2d(num_features=64)  # For conv layers
batch_norm_1d = nn.BatchNorm1d(num_features=256)  # For linear layers

# Layer Normalization (normalizes across feature dimension)
layer_norm = nn.LayerNorm(normalized_shape=256)

# Instance Normalization (for style transfer)
instance_norm = nn.InstanceNorm2d(64)

# Group Normalization
group_norm = nn.GroupNorm(num_groups=8, num_channels=64)
```

### Pooling Layers

```python
# Max Pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average Pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Adaptive Pooling (output fixed size regardless of input)
adaptive_avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global average pooling
adaptive_max = nn.AdaptiveMaxPool2d(output_size=(7, 7))

x = torch.randn(32, 64, 224, 224)
output = adaptive_avg(x)
print(output.shape)  # [32, 64, 1, 1]
```

### Dropout Layers

```python
# Regular dropout
dropout = nn.Dropout(p=0.5)  # 50% dropout probability

# 2D Dropout (for conv layers)
dropout2d = nn.Dropout2d(p=0.2)

# Alpha Dropout (for SELU activation)
alpha_dropout = nn.AlphaDropout(p=0.1)

# Dropout is only active during training
model.train()  # Enable dropout
model.eval()   # Disable dropout
```

## Activation Functions

```python
# ReLU and variants
relu = nn.ReLU()
relu_inplace = nn.ReLU(inplace=True)  # Saves memory
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
prelu = nn.PReLU()  # Learnable slope
elu = nn.ELU(alpha=1.0)
selu = nn.SELU()  # Self-normalizing

# Sigmoid and Tanh
sigmoid = nn.Sigmoid()  # Output: (0, 1)
tanh = nn.Tanh()        # Output: (-1, 1)

# Softmax
softmax = nn.Softmax(dim=1)  # Probabilities that sum to 1
log_softmax = nn.LogSoftmax(dim=1)  # More numerically stable

# GELU (used in transformers)
gelu = nn.GELU()

# Swish/SiLU
silu = nn.SiLU()

# Functional versions (can be used directly)
import torch.nn.functional as F
x = torch.randn(10)
output = F.relu(x)
output = F.softmax(x, dim=0)
```

## Loss Functions

### Classification Losses

```python
# Cross Entropy Loss (combines LogSoftmax and NLLLoss)
criterion = nn.CrossEntropyLoss()
logits = torch.randn(32, 10)  # Raw scores for 10 classes
targets = torch.randint(0, 10, (32,))  # Class indices
loss = criterion(logits, targets)

# Binary Cross Entropy
bce = nn.BCELoss()  # Expects sigmoid output
bce_logits = nn.BCEWithLogitsLoss()  # More stable, expects raw logits
probs = torch.sigmoid(torch.randn(32, 1))
binary_targets = torch.randint(0, 2, (32, 1)).float()
loss = bce(probs, binary_targets)

# Negative Log Likelihood
nll = nn.NLLLoss()
log_probs = F.log_softmax(logits, dim=1)
loss = nll(log_probs, targets)
```

### Regression Losses

```python
# Mean Squared Error
mse = nn.MSELoss()
predictions = torch.randn(32, 1)
targets = torch.randn(32, 1)
loss = mse(predictions, targets)

# Mean Absolute Error (L1 Loss)
mae = nn.L1Loss()
loss = mae(predictions, targets)

# Smooth L1 Loss (Huber Loss)
smooth_l1 = nn.SmoothL1Loss()
loss = smooth_l1(predictions, targets)
```

### Other Losses

```python
# Contrastive Loss (for siamese networks)
# Triplet Loss (for metric learning)
triplet = nn.TripletMarginLoss(margin=1.0)
anchor = torch.randn(32, 128)
positive = torch.randn(32, 128)
negative = torch.randn(32, 128)
loss = triplet(anchor, positive, negative)

# KL Divergence
kl_div = nn.KLDivLoss(reduction='batchmean')
```

## Building Complete Models

### Sequential Model

```python
# Simple sequential model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

# With named layers
model = nn.Sequential(
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 10))
)
```

### CNN for Image Classification

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNN(num_classes=10)
x = torch.randn(32, 3, 224, 224)
output = model(x)
print(output.shape)  # [32, 10]
```

### Model with Skip Connections

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)  # Skip connection
        out = F.relu(out)
        return out
```

## Model Utilities

### Parameter Initialization

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)

# Common initialization methods
nn.init.xavier_uniform_(layer.weight)
nn.init.kaiming_normal_(layer.weight)
nn.init.orthogonal_(layer.weight)
nn.init.constant_(layer.bias, 0)
```

### Freezing Layers

```python
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Freeze specific layers
for param in model.conv_layers.parameters():
    param.requires_grad = False

# Unfreeze last layer
for param in model.fc_layers[-1].parameters():
    param.requires_grad = True
```

### Moving Models to GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Ensure inputs are also on the same device
x = x.to(device)
output = model(x)
```

## Best Practices

1. **Use BatchNorm after Conv layers**: Improves training stability
2. **Apply Dropout before final layers**: Prevents overfitting
3. **Initialize weights properly**: Use Kaiming for ReLU, Xavier for tanh/sigmoid
4. **Use nn.Sequential for simple architectures**: Cleaner code
5. **Always call model.train() and model.eval()**: Affects dropout and batch norm

## Exercises

1. Build a simple MLP for MNIST digit classification
2. Create a CNN with residual connections
3. Implement a custom activation function as nn.Module
4. Build an autoencoder architecture

## Additional Resources

- [PyTorch nn.Module Tutorial](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [PyTorch nn Documentation](https://pytorch.org/docs/stable/nn.html)
