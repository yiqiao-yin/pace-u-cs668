# Machine Learning Basics

## Overview

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. This section covers fundamental concepts that form the foundation for building and evaluating machine learning models.

## Supervised vs. Unsupervised Learning

### Supervised Learning

The algorithm learns from labeled training data to make predictions on new data.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Regression: Predict continuous values
X = df[['feature1', 'feature2']]
y = df['target']  # Continuous target

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X_new)

# Classification: Predict categorical labels
y = df['category']  # Categorical target

model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

### Unsupervised Learning

The algorithm finds patterns in data without labeled outputs.

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Clustering: Group similar data points
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Dimensionality Reduction: Reduce features while preserving information
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
```

### Comparison

| Aspect | Supervised | Unsupervised |
|--------|------------|--------------|
| Data | Labeled | Unlabeled |
| Goal | Predict outcomes | Find patterns |
| Examples | Regression, Classification | Clustering, Dimensionality Reduction |
| Evaluation | Compare predictions to actual | Internal metrics (silhouette, inertia) |

## Training, Validation, and Test Sets

Properly splitting data is crucial for building reliable models.

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Simple split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# For classification, use stratified split to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Train-Validation-Test Split

```python
# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: separate validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 × 0.8 = 0.2
)

# Result: 60% train, 20% validation, 20% test
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

### Purpose of Each Set

- **Training Set**: Used to train the model (learn parameters)
- **Validation Set**: Used to tune hyperparameters and prevent overfitting
- **Test Set**: Used only for final evaluation (never seen during training)

## Model Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # For binary classification

# Accuracy: Overall correctness
accuracy = accuracy_score(y_test, y_pred)

# Precision: Of predicted positives, how many are actually positive?
precision = precision_score(y_test, y_pred)

# Recall: Of actual positives, how many did we predict?
recall = recall_score(y_test, y_pred)

# F1 Score: Harmonic mean of precision and recall
f1 = f1_score(y_test, y_pred)

# ROC-AUC: Area under the ROC curve
roc_auc = roc_auc_score(y_test, y_proba)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Complete Classification Report
print(classification_report(y_test, y_pred))
```

### Visualizing Classification Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

y_pred = model.predict(X_test)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# R² Score (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)

# Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2%}")
```

### Visualizing Regression Results

```python
# Actual vs Predicted
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')

# Residuals
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()
```

## Cross-Validation Techniques

Cross-validation provides a more robust estimate of model performance.

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# 5-Fold Cross-Validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### Stratified K-Fold (for Classification)

```python
from sklearn.model_selection import StratifiedKFold

# Maintains class proportions in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
```

### Cross-Validation with Multiple Metrics

```python
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### Leave-One-Out Cross-Validation

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
# Useful for small datasets, but computationally expensive
```

## Overfitting and Underfitting

### Understanding the Problem

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Compare different polynomial degrees
degrees = [1, 3, 15]
plt.figure(figsize=(15, 4))

for i, degree in enumerate(degrees):
    plt.subplot(1, 3, i + 1)

    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    train_score = model.score(X_poly_train, y_train)
    test_score = model.score(X_poly_test, y_test)

    plt.scatter(X_train, y_train, alpha=0.5, label='Train')
    plt.scatter(X_test, y_test, alpha=0.5, label='Test')

    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    plt.plot(X_plot, model.predict(X_plot_poly), 'r-', linewidth=2)

    plt.title(f'Degree {degree}\nTrain R²: {train_score:.3f}, Test R²: {test_score:.3f}')
    plt.legend()

plt.tight_layout()
plt.show()
```

### Diagnosing the Problem

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| High train error, high test error | Underfitting | More complex model, more features |
| Low train error, high test error | Overfitting | Regularization, more data, simpler model |
| Low train error, low test error | Good fit | Keep current approach |

### Solutions

```python
from sklearn.linear_model import Ridge, Lasso

# Regularization to prevent overfitting
ridge = Ridge(alpha=1.0)  # L2 regularization
lasso = Lasso(alpha=0.1)  # L1 regularization

# Early stopping (for iterative models)
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.2,
    n_iter_no_change=10,  # Stop if no improvement for 10 iterations
    random_state=42
)
```

## Basic Workflow

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 1. Create preprocessing and model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# 2. Define hyperparameter grid
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}

# 3. Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 4. Best model and parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# 5. Evaluate on test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

## Exercises

1. Split a dataset and train a classification model, then evaluate using multiple metrics
2. Compare model performance using different cross-validation strategies
3. Diagnose overfitting/underfitting using learning curves
4. Build a complete ML pipeline with preprocessing and hyperparameter tuning

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
