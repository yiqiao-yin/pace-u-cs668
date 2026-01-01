# Data Preprocessing

## Overview

Data preprocessing is a critical step in the data science pipeline. Raw data is often messy, incomplete, and inconsistent. Proper preprocessing transforms raw data into a clean, structured format suitable for analysis and machine learning.

## Data Cleaning

### Handling Missing Values

Missing data is one of the most common issues in real-world datasets.

```python
import pandas as pd
import numpy as np

# Identify missing values
df = pd.read_csv('data.csv')
print(df.isnull().sum())  # Count missing values per column
print(df.isnull().sum() / len(df) * 100)  # Percentage missing

# Strategies for handling missing values

# 1. Remove rows with missing values
df_clean = df.dropna()

# 2. Remove columns with too many missing values
threshold = 0.5  # Remove if >50% missing
df_clean = df.dropna(axis=1, thresh=int(threshold * len(df)))

# 3. Fill with a constant
df['column'].fillna(0, inplace=True)

# 4. Fill with mean/median/mode
df['numeric_col'].fillna(df['numeric_col'].mean(), inplace=True)
df['numeric_col'].fillna(df['numeric_col'].median(), inplace=True)
df['categorical_col'].fillna(df['categorical_col'].mode()[0], inplace=True)

# 5. Forward/backward fill (for time series)
df['column'].fillna(method='ffill', inplace=True)  # Forward fill
df['column'].fillna(method='bfill', inplace=True)  # Backward fill

# 6. Interpolation
df['column'] = df['column'].interpolate(method='linear')
```

### Handling Duplicates

```python
# Identify duplicates
print(df.duplicated().sum())

# View duplicate rows
print(df[df.duplicated()])

# Remove duplicates
df_clean = df.drop_duplicates()

# Remove duplicates based on specific columns
df_clean = df.drop_duplicates(subset=['column1', 'column2'], keep='first')
```

### Handling Outliers

```python
# Identify outliers using IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]

# Remove outliers
df_clean = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]

# Cap outliers (winsorization)
df['column'] = df['column'].clip(lower=lower_bound, upper=upper_bound)

# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df['column']))
df_clean = df[z_scores < 3]  # Keep data within 3 standard deviations
```

## Feature Scaling and Normalization

Scaling features ensures that all variables contribute equally to the analysis.

### Standardization (Z-score Normalization)

Transforms data to have mean=0 and standard deviation=1.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['feature1', 'feature2']])

# To inverse transform
df_original = scaler.inverse_transform(df_scaled)
```

### Min-Max Normalization

Scales data to a fixed range, typically [0, 1].

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df_normalized = scaler.fit_transform(df[['feature1', 'feature2']])
```

### Robust Scaling

Uses median and IQR, making it robust to outliers.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df_robust = scaler.fit_transform(df[['feature1', 'feature2']])
```

### When to Use Each Method

| Method | Use Case |
|--------|----------|
| StandardScaler | When data is normally distributed |
| MinMaxScaler | When you need bounded values (e.g., neural networks) |
| RobustScaler | When data contains outliers |

## Feature Engineering

### Creating New Features

```python
# Mathematical transformations
df['log_feature'] = np.log1p(df['feature'])  # Log transform
df['sqrt_feature'] = np.sqrt(df['feature'])  # Square root

# Interaction features
df['feature_product'] = df['feature1'] * df['feature2']
df['feature_ratio'] = df['feature1'] / (df['feature2'] + 1e-8)

# Date/time features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Binning continuous variables
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100],
                         labels=['child', 'young', 'middle', 'senior', 'elderly'])
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['feature1', 'feature2']])
```

## Handling Categorical Variables

### Label Encoding

Converts categories to numerical labels (use for ordinal data).

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Mapping for reference
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
```

### One-Hot Encoding

Creates binary columns for each category (use for nominal data).

```python
# Using pandas
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')

# Using sklearn
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['category']])
```

### Target Encoding

Replaces categories with their target mean (use with caution - can cause leakage).

```python
# Simple target encoding
target_means = df.groupby('category')['target'].mean()
df['category_target_encoded'] = df['category'].map(target_means)
```

## Data Type Conversions

```python
# Convert data types
df['int_column'] = df['int_column'].astype(int)
df['float_column'] = df['float_column'].astype(float)
df['string_column'] = df['string_column'].astype(str)
df['category_column'] = df['category_column'].astype('category')

# Convert to datetime
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Reduce memory usage with appropriate types
df['small_int'] = df['small_int'].astype('int8')  # -128 to 127
df['medium_int'] = df['medium_int'].astype('int32')
```

## Preprocessing Pipeline

Creating a reproducible preprocessing pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Define preprocessing for numeric columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Fit and transform
X_processed = preprocessor.fit_transform(X)
```

## Best Practices

1. **Always split data before preprocessing**: Fit transformers only on training data
2. **Document your transformations**: Keep track of all preprocessing steps
3. **Handle leakage**: Don't use target information in feature engineering
4. **Validate assumptions**: Check distributions before and after transformations
5. **Save preprocessing objects**: For applying the same transformations to new data

## Exercises

1. Clean a dataset by handling missing values and duplicates
2. Apply different scaling methods and compare the results
3. Create meaningful features from datetime columns
4. Build a complete preprocessing pipeline for a dataset

## Additional Resources

- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
