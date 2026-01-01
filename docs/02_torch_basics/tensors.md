# Introduction to Tensors

## Overview

Tensors are the fundamental data structure in PyTorch. They are multi-dimensional arrays similar to NumPy arrays but with additional capabilities for GPU acceleration and automatic differentiation. Understanding tensors is essential for building deep learning models.

## What is a Tensor?

A tensor is a generalization of vectors and matrices to potentially higher dimensions:
- **0-D Tensor (Scalar)**: A single number
- **1-D Tensor (Vector)**: A sequence of numbers
- **2-D Tensor (Matrix)**: A table of numbers
- **3-D+ Tensor**: Higher-dimensional arrays (e.g., images, video)

## Creating Tensors

### From Python Data

```python
import torch

# From a list
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(tensor_1d)  # tensor([1, 2, 3, 4, 5])

# From nested lists (2D)
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor_2d.shape)  # torch.Size([2, 3])

# From NumPy array
import numpy as np
np_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(np_array)

# Back to NumPy
numpy_from_tensor = tensor_1d.numpy()
```

### Special Tensors

```python
# Zeros
zeros = torch.zeros(3, 4)  # 3x4 tensor of zeros

# Ones
ones = torch.ones(2, 3)  # 2x3 tensor of ones

# Random values
random_uniform = torch.rand(3, 3)  # Uniform distribution [0, 1)
random_normal = torch.randn(3, 3)  # Normal distribution (mean=0, std=1)
random_int = torch.randint(0, 10, (3, 3))  # Random integers [0, 10)

# Identity matrix
identity = torch.eye(4)  # 4x4 identity matrix

# Range
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Like another tensor
x = torch.tensor([[1, 2], [3, 4]])
zeros_like = torch.zeros_like(x)  # Same shape as x, filled with zeros
ones_like = torch.ones_like(x)
```

### Specifying Data Types

```python
# Common data types
float32_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)  # Default for floats
float64_tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
int32_tensor = torch.tensor([1, 2], dtype=torch.int32)
int64_tensor = torch.tensor([1, 2], dtype=torch.int64)  # Default for integers
bool_tensor = torch.tensor([True, False], dtype=torch.bool)

# Type conversion
x = torch.tensor([1, 2, 3])
x_float = x.float()  # Convert to float32
x_double = x.double()  # Convert to float64
x_int = x.int()  # Convert to int32

# Check type
print(x.dtype)  # torch.int64
```

## Tensor Operations

### Basic Arithmetic

```python
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

# Element-wise operations
add = a + b  # or torch.add(a, b)
sub = a - b  # or torch.sub(a, b)
mul = a * b  # or torch.mul(a, b)
div = a / b  # or torch.div(a, b)
pow_result = a ** 2  # or torch.pow(a, 2)

# In-place operations (modify tensor directly)
a.add_(b)  # a = a + b (note the underscore)
a.mul_(2)  # a = a * 2
```

### Matrix Operations

```python
# Matrix multiplication
A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Three ways to do matrix multiplication
result1 = torch.mm(A, B)  # Only for 2D matrices
result2 = torch.matmul(A, B)  # Works for higher dimensions
result3 = A @ B  # Operator shorthand

# Dot product (for 1D tensors)
v1 = torch.tensor([1, 2, 3], dtype=torch.float32)
v2 = torch.tensor([4, 5, 6], dtype=torch.float32)
dot = torch.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32

# Transpose
A_T = A.T  # or A.transpose(0, 1)

# Batch matrix multiplication
batch_A = torch.randn(10, 3, 4)  # 10 matrices of shape 3x4
batch_B = torch.randn(10, 4, 5)  # 10 matrices of shape 4x5
batch_result = torch.bmm(batch_A, batch_B)  # Shape: 10x3x5
```

### Reduction Operations

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# Sum
total = x.sum()  # Sum of all elements
row_sum = x.sum(dim=1)  # Sum along rows: [6, 15]
col_sum = x.sum(dim=0)  # Sum along columns: [5, 7, 9]

# Mean
mean_all = x.mean()
mean_row = x.mean(dim=1)

# Max and Min
max_val = x.max()
max_row, max_idx = x.max(dim=1)  # Returns values and indices

min_val = x.min()

# Other reductions
std = x.std()  # Standard deviation
var = x.var()  # Variance
prod = x.prod()  # Product of all elements
```

## Broadcasting

Broadcasting allows operations on tensors with different shapes.

```python
# Scalar broadcasting
a = torch.tensor([[1, 2], [3, 4]])
b = 10
result = a + b  # [[11, 12], [13, 14]]

# Vector broadcasting
a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape: 2x3
b = torch.tensor([10, 20, 30])  # Shape: 3
result = a + b  # [[11, 22, 33], [14, 25, 36]]

# Broadcasting rules:
# 1. Compare dimensions from right to left
# 2. Dimensions must be equal OR one of them must be 1
a = torch.randn(4, 3, 2)
b = torch.randn(3, 1)  # Broadcasts to 4, 3, 2
result = a + b
```

## Reshaping Tensors

```python
x = torch.arange(12)  # [0, 1, 2, ..., 11]

# Reshape
reshaped = x.reshape(3, 4)  # Shape: 3x4
reshaped = x.reshape(2, 2, 3)  # Shape: 2x2x3

# View (shares memory with original)
viewed = x.view(3, 4)

# -1 infers the dimension
auto_reshape = x.reshape(-1, 4)  # Shape: 3x4
auto_reshape = x.reshape(2, -1)  # Shape: 2x6

# Flatten
flattened = reshaped.flatten()  # Back to 1D

# Squeeze and Unsqueeze
x = torch.zeros(1, 3, 1, 4)
squeezed = x.squeeze()  # Remove dimensions of size 1: Shape 3x4
squeezed = x.squeeze(0)  # Remove only dimension 0: Shape 3x1x4

y = torch.zeros(3, 4)
unsqueezed = y.unsqueeze(0)  # Add dimension at position 0: Shape 1x3x4
unsqueezed = y.unsqueeze(-1)  # Add at the end: Shape 3x4x1

# Permute dimensions
x = torch.randn(2, 3, 4)
permuted = x.permute(2, 0, 1)  # Shape: 4x2x3
```

## GPU Acceleration with CUDA

```python
# Check CUDA availability
print(torch.cuda.is_available())  # True if GPU available
print(torch.cuda.device_count())  # Number of GPUs

# Move tensor to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

x = torch.randn(1000, 1000)
x_gpu = x.to(device)  # Move to GPU
# or
x_gpu = x.cuda()  # Move to default GPU

# Create tensor directly on GPU
y = torch.randn(1000, 1000, device=device)

# Move back to CPU
x_cpu = x_gpu.cpu()

# Check device
print(x_gpu.device)  # cuda:0
```

## Automatic Differentiation with Autograd

```python
# Enable gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Perform operations
y = x ** 2
z = y.sum()  # z = x[0]^2 + x[1]^2

# Compute gradients
z.backward()

# Access gradients (dz/dx)
print(x.grad)  # tensor([4., 6.]) = [2*x[0], 2*x[1]]

# Disable gradient tracking
with torch.no_grad():
    y = x * 2  # No gradients computed

# Detach from computation graph
x_detached = x.detach()

# Zero gradients (important in training loops)
x.grad.zero_()
```

### Gradient Example

```python
# Simple neural network forward pass
x = torch.randn(10, 3, requires_grad=True)  # Input
w = torch.randn(3, 5, requires_grad=True)   # Weights
b = torch.randn(5, requires_grad=True)      # Bias

# Forward pass
y = x @ w + b
loss = y.sum()

# Backward pass
loss.backward()

# Gradients are now available
print(w.grad.shape)  # Shape: 3x5
print(b.grad.shape)  # Shape: 5
```

## Common Tensor Functions

```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Activation functions
relu = torch.relu(x)  # [0, 0, 0, 1, 2]
sigmoid = torch.sigmoid(x)  # Values between 0 and 1
tanh = torch.tanh(x)  # Values between -1 and 1
softmax = torch.softmax(x, dim=0)  # Probabilities that sum to 1

# Mathematical functions
abs_val = torch.abs(x)
exp = torch.exp(x)
log = torch.log(torch.abs(x) + 1e-8)  # Add small value to avoid log(0)
sqrt = torch.sqrt(torch.abs(x))
sin = torch.sin(x)
cos = torch.cos(x)

# Clipping
clipped = torch.clamp(x, min=-1, max=1)  # Clip values to [-1, 1]

# Comparison
greater = x > 0  # Boolean tensor
where = torch.where(x > 0, x, torch.zeros_like(x))  # Conditional selection
```

## Best Practices

1. **Use appropriate data types**: `float32` is usually sufficient; `float16` for memory savings
2. **Avoid unnecessary CPU-GPU transfers**: Keep tensors on the same device
3. **Use in-place operations sparingly**: They can interfere with autograd
4. **Set requires_grad=False for inference**: Reduces memory usage
5. **Use torch.no_grad() during evaluation**: Speeds up inference

## Exercises

1. Create tensors of different shapes and perform basic operations
2. Implement matrix multiplication and verify with NumPy
3. Move tensors to GPU and compare computation speed
4. Use autograd to compute gradients for a simple function

## Additional Resources

- [PyTorch Tensor Tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Autograd Tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
