# Exploratory Data Analysis (EDA)

## Overview

Exploratory Data Analysis (EDA) is the process of analyzing datasets to summarize their main characteristics, often using visual methods. EDA helps data scientists understand the data, detect patterns, spot anomalies, and form hypotheses before formal modeling.

## Understanding Your Data

### Initial Data Inspection

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Basic information
print(df.shape)              # (rows, columns)
print(df.info())             # Data types and non-null counts
print(df.dtypes)             # Column data types
print(df.head())             # First 5 rows
print(df.tail())             # Last 5 rows
print(df.sample(5))          # Random 5 rows

# Column names
print(df.columns.tolist())

# Memory usage
print(df.memory_usage(deep=True))
```

### Statistical Summary

```python
# Numerical columns summary
print(df.describe())

# Include all columns
print(df.describe(include='all'))

# Custom percentiles
print(df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))

# For categorical columns
print(df['category'].value_counts())
print(df['category'].value_counts(normalize=True))  # Proportions
```

## Understanding Data Distributions

### Univariate Analysis

Analyzing one variable at a time.

```python
# Histogram for numerical variables
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
axes[0, 0].hist(df['numeric_col'], bins=30, edgecolor='black')
axes[0, 0].set_title('Histogram')

# Density plot
df['numeric_col'].plot(kind='density', ax=axes[0, 1])
axes[0, 1].set_title('Density Plot')

# Box plot
df.boxplot(column='numeric_col', ax=axes[1, 0])
axes[1, 0].set_title('Box Plot')

# Violin plot
sns.violinplot(data=df, y='numeric_col', ax=axes[1, 1])
axes[1, 1].set_title('Violin Plot')

plt.tight_layout()
plt.show()

# For categorical variables
plt.figure(figsize=(10, 6))
df['category'].value_counts().plot(kind='bar')
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

### Bivariate Analysis

Analyzing relationships between two variables.

```python
# Scatter plot (numerical vs numerical)
plt.figure(figsize=(10, 6))
plt.scatter(df['feature1'], df['feature2'], alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Feature 1 vs Feature 2')
plt.show()

# With color-coded categories
plt.figure(figsize=(10, 6))
for category in df['category'].unique():
    subset = df[df['category'] == category]
    plt.scatter(subset['feature1'], subset['feature2'], label=category, alpha=0.6)
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Box plot (categorical vs numerical)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='numeric_col')
plt.title('Distribution by Category')
plt.xticks(rotation=45)
plt.show()

# Bar plot with error bars
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='category', y='numeric_col', ci=95)
plt.title('Mean by Category (with 95% CI)')
plt.show()
```

## Identifying Patterns and Correlations

### Correlation Analysis

```python
# Compute correlation matrix
correlation_matrix = df.select_dtypes(include=[np.number]).corr()

# Heatmap visualization
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Find highly correlated pairs
def get_high_correlations(corr_matrix, threshold=0.7):
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    return pd.DataFrame(high_corr)

print(get_high_correlations(correlation_matrix))
```

### Pair Plots

```python
# Pair plot for multiple variables
sns.pairplot(df[['feature1', 'feature2', 'feature3', 'target']],
             hue='category', diag_kind='kde')
plt.suptitle('Pair Plot', y=1.02)
plt.show()
```

## Outlier Detection

### Visual Methods

```python
# Box plots for all numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
n_cols = len(numerical_cols)
n_rows = (n_cols + 2) // 3

fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    df.boxplot(column=col, ax=axes[i])
    axes[i].set_title(col)

# Hide empty subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()
```

### Statistical Methods

```python
# Z-score method
from scipy import stats

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    return df[z_scores > threshold]

# IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] < lower) | (df[column] > upper)]

# Count outliers per column
for col in numerical_cols:
    outliers = detect_outliers_iqr(df, col)
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
```

## Visual Analysis Techniques

### Distribution Comparison

```python
# Compare distributions across categories
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overlapping histograms
for category in df['category'].unique():
    subset = df[df['category'] == category]
    axes[0].hist(subset['numeric_col'], bins=30, alpha=0.5, label=category)
axes[0].legend()
axes[0].set_title('Distribution by Category')

# KDE plots
for category in df['category'].unique():
    subset = df[df['category'] == category]
    subset['numeric_col'].plot(kind='kde', ax=axes[1], label=category)
axes[1].legend()
axes[1].set_title('Density by Category')

plt.tight_layout()
plt.show()
```

### Time Series Analysis

```python
# For time-based data
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Line plot over time
plt.figure(figsize=(14, 6))
df['value'].plot()
plt.title('Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Rolling statistics
plt.figure(figsize=(14, 6))
df['value'].plot(label='Original', alpha=0.5)
df['value'].rolling(window=7).mean().plot(label='7-day MA')
df['value'].rolling(window=30).mean().plot(label='30-day MA')
plt.legend()
plt.title('Value with Moving Averages')
plt.show()

# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['value'], model='additive', period=30)
decomposition.plot()
plt.tight_layout()
plt.show()
```

## Missing Data Analysis

```python
# Missing data visualization
import missingno as msno  # pip install missingno

# Matrix plot
msno.matrix(df)
plt.title('Missing Data Matrix')
plt.show()

# Bar plot of missing values
msno.bar(df)
plt.title('Missing Data by Column')
plt.show()

# Heatmap of missing value correlations
msno.heatmap(df)
plt.title('Missing Data Correlation')
plt.show()

# Custom missing data summary
missing_summary = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Missing Percentage': df.isnull().sum() / len(df) * 100
}).sort_values('Missing Percentage', ascending=False)
print(missing_summary[missing_summary['Missing Count'] > 0])
```

## EDA Report Template

```python
def generate_eda_report(df):
    """Generate a comprehensive EDA report."""

    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 50)

    # Basic Info
    print("\n1. BASIC INFORMATION")
    print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Data Types
    print("\n2. DATA TYPES")
    print(df.dtypes.value_counts())

    # Missing Values
    print("\n3. MISSING VALUES")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col in missing.index:
            print(f"   {col}: {missing[col]} ({missing[col]/len(df)*100:.1f}%)")
    else:
        print("   No missing values!")

    # Numerical Summary
    print("\n4. NUMERICAL COLUMNS SUMMARY")
    print(df.describe().round(2))

    # Categorical Summary
    print("\n5. CATEGORICAL COLUMNS")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"\n   {col}:")
        print(f"   Unique values: {df[col].nunique()}")
        print(df[col].value_counts().head())

    return None

generate_eda_report(df)
```

## Best Practices

1. **Start simple**: Begin with basic statistics before complex visualizations
2. **Document findings**: Keep notes on interesting patterns and anomalies
3. **Be systematic**: Analyze all variables, not just the obvious ones
4. **Question assumptions**: Verify data quality before drawing conclusions
5. **Iterate**: EDA is an iterative process; revisit as you learn more

## Exercises

1. Perform complete EDA on a real-world dataset
2. Create a visualization dashboard summarizing key findings
3. Identify and document at least 5 interesting patterns in a dataset
4. Build an automated EDA function for your workflow

## Additional Resources

- [Python Graph Gallery](https://www.python-graph-gallery.com/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Pandas Visualization](https://pandas.pydata.org/docs/user_guide/visualization.html)
