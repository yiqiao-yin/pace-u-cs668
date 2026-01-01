# Statistical Foundations

## Overview

Statistics forms the mathematical backbone of data science. Understanding statistical concepts enables data scientists to make informed decisions, draw valid conclusions from data, and communicate findings effectively.

## Descriptive Statistics

Descriptive statistics summarize and describe the main features of a dataset.

### Measures of Central Tendency

```python
import numpy as np
import pandas as pd
from scipy import stats

data = [23, 25, 27, 29, 31, 33, 35, 37, 39, 41]

# Mean: Average value
mean = np.mean(data)  # 32.0

# Median: Middle value
median = np.median(data)  # 32.0

# Mode: Most frequent value
mode = stats.mode(data)
```

### Measures of Dispersion

```python
# Variance: Average squared deviation from mean
variance = np.var(data)

# Standard Deviation: Square root of variance
std_dev = np.std(data)

# Range: Difference between max and min
data_range = np.max(data) - np.min(data)

# Interquartile Range (IQR)
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
```

### Measures of Shape

```python
from scipy.stats import skew, kurtosis

# Skewness: Asymmetry of distribution
# Positive skew: tail extends right
# Negative skew: tail extends left
skewness = skew(data)

# Kurtosis: "Tailedness" of distribution
# High kurtosis: heavy tails
# Low kurtosis: light tails
kurt = kurtosis(data)
```

## Probability Distributions

### Normal (Gaussian) Distribution

The most important distribution in statistics, characterized by its bell shape.

```python
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate normal distribution
mu, sigma = 0, 1  # mean and standard deviation
x = np.linspace(-4, 4, 100)
y = norm.pdf(x, mu, sigma)

plt.plot(x, y)
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()

# Calculate probabilities
prob_less_than_1 = norm.cdf(1, mu, sigma)  # P(X < 1)
prob_between = norm.cdf(1) - norm.cdf(-1)  # P(-1 < X < 1) ≈ 68%
```

### Other Important Distributions

```python
from scipy.stats import binom, poisson, expon

# Binomial: Number of successes in n trials
# Example: Probability of 7 heads in 10 coin flips
p_binomial = binom.pmf(7, n=10, p=0.5)

# Poisson: Number of events in fixed interval
# Example: 5 customers arriving when average is 3
p_poisson = poisson.pmf(5, mu=3)

# Exponential: Time between events
# Example: Time until next customer
p_exponential = expon.pdf(2, scale=1/3)
```

## Hypothesis Testing

Hypothesis testing is a statistical method for making decisions based on data.

### The Hypothesis Testing Framework

1. **Null Hypothesis (H₀)**: The default assumption (no effect)
2. **Alternative Hypothesis (H₁)**: What we want to prove
3. **Significance Level (α)**: Threshold for rejecting H₀ (commonly 0.05)
4. **p-value**: Probability of observing results if H₀ is true

### Common Tests

```python
from scipy import stats

# T-test: Compare means of two groups
group1 = [23, 25, 28, 30, 32]
group2 = [30, 32, 35, 37, 40]

# Independent samples t-test
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"t-statistic: {t_stat}, p-value: {p_value}")

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(group1, 25)

# Chi-square test: Test for independence
observed = [[10, 20], [30, 40]]
chi2, p_value, dof, expected = stats.chi2_contingency(observed)
```

### Interpreting Results

```python
alpha = 0.05  # Significance level

if p_value < alpha:
    print("Reject null hypothesis - result is statistically significant")
else:
    print("Fail to reject null hypothesis - insufficient evidence")
```

## Statistical Inference

### Confidence Intervals

A confidence interval provides a range of plausible values for a parameter.

```python
import scipy.stats as stats

data = np.random.randn(100) * 10 + 50  # Sample data

# 95% Confidence Interval for the mean
confidence = 0.95
n = len(data)
mean = np.mean(data)
std_err = stats.sem(data)

ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=std_err)
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
```

### Effect Size

Effect size measures the magnitude of a difference, independent of sample size.

```python
# Cohen's d: Standardized difference between two means
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

d = cohens_d(group1, group2)
# |d| < 0.2: small, 0.2-0.8: medium, > 0.8: large
```

## Correlation Analysis

### Pearson Correlation

Measures linear relationship between two continuous variables.

```python
# Pearson correlation coefficient
r, p_value = stats.pearsonr(x, y)
# r ranges from -1 (perfect negative) to +1 (perfect positive)
```

### Spearman Correlation

Measures monotonic relationship (rank-based).

```python
# Spearman correlation (for non-linear relationships)
rho, p_value = stats.spearmanr(x, y)
```

## Common Pitfalls

1. **p-hacking**: Testing multiple hypotheses until finding significance
2. **Confusing correlation with causation**: Correlation ≠ causation
3. **Ignoring sample size**: Large samples can make small effects significant
4. **Multiple comparisons problem**: Adjust for multiple tests (Bonferroni correction)

## Exercises

1. Calculate descriptive statistics for a dataset and interpret the results
2. Perform a t-test to compare two groups and interpret the p-value
3. Create confidence intervals for a sample mean
4. Compute correlation between two variables and visualize the relationship

## Additional Resources

- [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability)
- [Seeing Theory - Visual Statistics](https://seeing-theory.brown.edu/)
- [StatQuest YouTube Channel](https://www.youtube.com/c/joshstarmer)
