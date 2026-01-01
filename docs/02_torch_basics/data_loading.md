# Data Loading

## Overview

Efficient data loading is crucial for training deep learning models. PyTorch provides the `Dataset` and `DataLoader` classes to handle data loading, batching, shuffling, and parallel processing. This section covers how to use these tools effectively.

## Dataset and DataLoader

### Basic Concepts

```python
from torch.utils.data import Dataset, DataLoader
import torch

# Dataset: Represents your data
# DataLoader: Wraps a Dataset and provides batching, shuffling, parallel loading

# Using a built-in dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load training data
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Create DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Faster GPU transfer
)

# Iterate through data
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f'Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}')
    if batch_idx == 2:
        break
```

## Custom Dataset Creation

### Basic Custom Dataset

```python
from torch.utils.data import Dataset
import pandas as pd
import torch

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file: Path to the CSV file with annotations
            transform: Optional transform to apply to samples
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        """Return the total number of samples"""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample by index"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get features and label
        features = self.data.iloc[idx, :-1].values.astype('float32')
        label = self.data.iloc[idx, -1]

        # Convert to tensors
        features = torch.tensor(features)
        label = torch.tensor(label)

        # Apply transform
        if self.transform:
            features = self.transform(features)

        return features, label

# Usage
dataset = CustomDataset('data.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Image Dataset

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load labels from file
        with open(labels_file, 'r') as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            img_name, label = line.strip().split(',')
            self.samples.append((img_name, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

# Usage with transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = ImageDataset('images/', 'labels.txt', transform=transform)
```

### Text Dataset

```python
from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }
```

## Data Augmentation

### Image Augmentation

```python
from torchvision import transforms

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5)  # Cutout augmentation
])

# Validation/Test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### Advanced Augmentation with Albumentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class AlbumentationsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

# Define augmentation pipeline
transform = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.OneOf([
        A.GaussNoise(),
        A.GaussianBlur(),
        A.MotionBlur(),
    ], p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

## DataLoader Options

### Key Parameters

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,           # Number of samples per batch
    shuffle=True,            # Shuffle data at every epoch
    num_workers=4,           # Number of subprocesses for data loading
    pin_memory=True,         # Copy tensors to CUDA pinned memory
    drop_last=True,          # Drop last incomplete batch
    prefetch_factor=2,       # Number of batches loaded in advance per worker
    persistent_workers=True  # Keep worker processes alive between epochs
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=64,           # Can use larger batch for validation
    shuffle=False,           # No need to shuffle validation data
    num_workers=4,
    pin_memory=True
)
```

### Custom Collate Function

```python
def custom_collate_fn(batch):
    """
    Custom function to process a batch of samples.
    Useful for variable-length sequences or custom batching logic.
    """
    # batch is a list of (data, label) tuples
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad sequences to same length
    max_len = max(len(d) for d in data)
    padded_data = torch.zeros(len(data), max_len)
    for i, d in enumerate(data):
        padded_data[i, :len(d)] = torch.tensor(d)

    labels = torch.tensor(labels)

    return padded_data, labels

# Use custom collate function
loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)
```

## Batch Processing

### Handling Variable-Length Sequences

```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def collate_sequences(batch):
    # Sort by sequence length (descending)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    sequences = [torch.tensor(item[0]) for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    lengths = torch.tensor([len(s) for s in sequences])

    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)

    return padded, labels, lengths

# In the training loop
for padded, labels, lengths in dataloader:
    # Pack for efficient RNN processing
    packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=True)
    output, hidden = rnn(packed)
```

## Splitting Datasets

```python
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split

# Random split
dataset = CustomDataset('data.csv')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Stratified split (for classification)
indices = list(range(len(dataset)))
labels = [dataset[i][1] for i in indices]
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=42
)
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

# K-fold cross-validation
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    # Train model for this fold
```

## Weighted Sampling

Handle class imbalance:

```python
from torch.utils.data import WeightedRandomSampler

# Calculate class weights
labels = [dataset[i][1] for i in range(len(dataset))]
class_counts = [labels.count(i) for i in range(num_classes)]
class_weights = [1.0 / count for count in class_counts]

# Assign weight to each sample
sample_weights = [class_weights[label] for label in labels]
sample_weights = torch.tensor(sample_weights)

# Create sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Use sampler in DataLoader (don't use shuffle with sampler)
train_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=4
)
```

## Efficient Data Loading

### Memory-Mapped Files

```python
import numpy as np
from torch.utils.data import Dataset

class MemmapDataset(Dataset):
    def __init__(self, data_path, shape, dtype='float32'):
        # Load as memory-mapped file (doesn't load into RAM)
        self.data = np.memmap(data_path, dtype=dtype, mode='r', shape=shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])
```

### Caching Dataset

```python
class CachedDataset(Dataset):
    def __init__(self, base_dataset, cache_size=1000):
        self.base_dataset = base_dataset
        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        item = self.base_dataset[idx]

        if len(self.cache) < self.cache_size:
            self.cache[idx] = item

        return item
```

## Best Practices

1. **Use num_workers > 0**: Parallelize data loading
2. **Set pin_memory=True for GPU training**: Speeds up CPU to GPU transfer
3. **Use appropriate batch size**: Balance between memory and training speed
4. **Apply augmentation only to training data**: Keep validation/test consistent
5. **Use persistent_workers=True**: Avoids worker initialization overhead
6. **Profile your data loading**: Use torch.utils.bottleneck to identify bottlenecks

## Exercises

1. Create a custom dataset for images organized in folders by class
2. Implement data augmentation pipeline for a specific task
3. Create a DataLoader with weighted sampling for imbalanced data
4. Implement a dataset that loads data lazily from disk

## Additional Resources

- [PyTorch Data Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [DataLoader Documentation](https://pytorch.org/docs/stable/data.html)
- [Albumentations Library](https://albumentations.ai/)
