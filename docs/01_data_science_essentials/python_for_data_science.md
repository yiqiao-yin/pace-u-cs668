# Python for Data Science

## Overview

Python is the most widely used programming language for data science due to its simplicity, readability, and extensive ecosystem of scientific computing libraries. This section covers the essential Python libraries that form the backbone of any data science workflow.

## NumPy for Numerical Computing

NumPy (Numerical Python) is the fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions.

### Key Features

- **N-dimensional arrays**: Efficient storage and manipulation of numerical data
- **Broadcasting**: Arithmetic operations on arrays of different shapes
- **Vectorized operations**: Fast element-wise operations without explicit loops
- **Linear algebra**: Matrix operations, eigenvalues, and decompositions

### Basic Usage

```python
import numpy as np

# Creating arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Array operations
print(arr * 2)          # Element-wise multiplication
print(np.mean(arr))     # Statistical operations
print(np.dot(matrix, matrix.T))  # Matrix multiplication

# Creating special arrays
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
random = np.random.randn(5, 5)
```

## Pandas for Data Manipulation

Pandas provides high-performance, easy-to-use data structures and data analysis tools. The two primary data structures are Series (1-dimensional) and DataFrame (2-dimensional).

### Key Features

- **DataFrame**: Tabular data structure with labeled axes
- **Data I/O**: Read/write CSV, Excel, SQL, JSON, and more
- **Data cleaning**: Handle missing values, duplicates, and type conversions
- **Data transformation**: Filtering, grouping, merging, and reshaping

### Basic Usage

```python
import pandas as pd

# Creating a DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# Reading data
df = pd.read_csv('data.csv')

# Basic operations
print(df.head())           # First 5 rows
print(df.describe())       # Statistical summary
print(df['age'].mean())    # Column operations

# Filtering
adults = df[df['age'] >= 18]

# Grouping
grouped = df.groupby('department')['salary'].mean()
```

## Matplotlib for Visualization

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

### Basic Plotting

```python
import matplotlib.pyplot as plt

# Line plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Line Plot')
plt.show()

# Scatter plot
plt.scatter(x, y, c='red', marker='o')
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30, edgecolor='black')
plt.show()
```

## Seaborn for Statistical Visualization

Seaborn is built on top of Matplotlib and provides a high-level interface for drawing attractive statistical graphics.

### Key Features

- **Statistical plots**: Distribution, regression, and categorical plots
- **Aesthetics**: Beautiful default themes and color palettes
- **Integration**: Works seamlessly with Pandas DataFrames

### Basic Usage

```python
import seaborn as sns

# Distribution plot
sns.histplot(data=df, x='age', kde=True)

# Box plot
sns.boxplot(data=df, x='department', y='salary')

# Correlation heatmap
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')

# Pair plot for multiple variables
sns.pairplot(df, hue='category')
```

## Best Practices

1. **Use vectorized operations**: Avoid loops when possible; use NumPy/Pandas operations
2. **Memory management**: Use appropriate data types to reduce memory usage
3. **Method chaining**: Chain Pandas operations for cleaner code
4. **Reproducibility**: Set random seeds for reproducible results

## Exercises

1. Load a CSV file and compute basic statistics for numerical columns
2. Create a visualization showing the distribution of a variable
3. Filter and transform a DataFrame based on multiple conditions
4. Generate a correlation matrix and visualize it as a heatmap

## Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
