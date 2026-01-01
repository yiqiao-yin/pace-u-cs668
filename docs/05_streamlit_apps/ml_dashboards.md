# Machine Learning Dashboards

## Overview

Streamlit is excellent for building interactive dashboards for machine learning models. This section covers how to create dashboards for model inference, performance visualization, and interactive model exploration.

## Model Inference Dashboard

### Basic Inference Interface

```python
import streamlit as st
import torch
from transformers import pipeline

st.title("Text Classification Dashboard")

# Cache the model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

# User input
text = st.text_area("Enter text to analyze:", height=100)

if st.button("Analyze"):
    if text:
        with st.spinner("Analyzing..."):
            result = model(text)[0]

        col1, col2 = st.columns(2)
        col1.metric("Sentiment", result['label'])
        col2.metric("Confidence", f"{result['score']:.2%}")
    else:
        st.warning("Please enter some text.")
```

### Image Classification Dashboard

```python
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

st.title("Image Classification")

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

@st.cache_data
def load_labels():
    import urllib
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    return urllib.request.urlopen(url).read().decode("utf-8").split("\n")

model = load_model()
labels = load_labels()

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)

    st.subheader("Top 5 Predictions")
    for prob, idx in zip(top5_prob, top5_indices):
        st.progress(float(prob))
        st.write(f"{labels[idx]}: {prob:.2%}")
```

## Model Performance Dashboard

### Classification Metrics

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report

st.title("Model Performance Dashboard")

# Simulated predictions
np.random.seed(42)
n_samples = 1000
y_true = np.random.randint(0, 4, n_samples)
y_pred = y_true.copy()
# Add some errors
errors = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
y_pred[errors] = (y_pred[errors] + np.random.randint(1, 4, len(errors))) % 4

classes = ['Class A', 'Class B', 'Class C', 'Class D']

# Metrics
col1, col2, col3, col4 = st.columns(4)
accuracy = np.mean(y_true == y_pred)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Total Samples", n_samples)
col3.metric("Correct", int(accuracy * n_samples))
col4.metric("Errors", int((1 - accuracy) * n_samples))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
fig = ff.create_annotated_heatmap(
    cm, x=classes, y=classes,
    colorscale='Blues',
    showscale=True
)
fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
st.plotly_chart(fig, use_container_width=True)

# Classification Report
st.subheader("Classification Report")
report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format("{:.2f}"))
```

### Performance Over Time

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Model Performance Over Time")

# Simulated performance data
dates = pd.date_range('2023-01-01', periods=100)
df = pd.DataFrame({
    'Date': dates,
    'Accuracy': 0.85 + np.random.randn(100) * 0.02,
    'Loss': 0.3 - np.linspace(0, 0.1, 100) + np.random.randn(100) * 0.02,
    'F1 Score': 0.82 + np.linspace(0, 0.05, 100) + np.random.randn(100) * 0.02
})

# Metric selection
metrics = st.multiselect(
    "Select metrics to display",
    ['Accuracy', 'Loss', 'F1 Score'],
    default=['Accuracy', 'Loss']
)

# Plot
fig = px.line(df, x='Date', y=metrics, title='Performance Metrics Over Time')
fig.update_layout(hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

# Statistics
st.subheader("Statistics")
st.dataframe(df[metrics].describe())
```

## A/B Testing Comparison

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

st.title("Model A/B Testing Dashboard")

# Simulated A/B test results
n_samples = 1000

model_a_accuracy = 0.85 + np.random.randn(n_samples) * 0.1
model_b_accuracy = 0.88 + np.random.randn(n_samples) * 0.08

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model A")
    st.metric("Mean Accuracy", f"{model_a_accuracy.mean():.2%}")
    st.metric("Std Dev", f"{model_a_accuracy.std():.4f}")

with col2:
    st.subheader("Model B")
    st.metric("Mean Accuracy", f"{model_b_accuracy.mean():.2%}")
    st.metric("Std Dev", f"{model_b_accuracy.std():.4f}")

# Distribution comparison
df = pd.DataFrame({
    'Accuracy': np.concatenate([model_a_accuracy, model_b_accuracy]),
    'Model': ['Model A'] * n_samples + ['Model B'] * n_samples
})

fig = px.histogram(df, x='Accuracy', color='Model', barmode='overlay',
                   title='Accuracy Distribution Comparison')
st.plotly_chart(fig, use_container_width=True)

# Statistical test
t_stat, p_value = stats.ttest_ind(model_a_accuracy, model_b_accuracy)

st.subheader("Statistical Test")
st.write(f"T-statistic: {t_stat:.4f}")
st.write(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    st.success("Difference is statistically significant (p < 0.05)")
else:
    st.info("Difference is not statistically significant")
```

## Feature Importance Display

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Feature Importance Dashboard")

# Simulated feature importance
features = ['Age', 'Income', 'Credit Score', 'Employment', 'Loan Amount',
            'Loan Term', 'Interest Rate', 'DTI Ratio', 'Home Ownership', 'Purpose']
importance = np.random.uniform(0.02, 0.2, len(features))
importance = importance / importance.sum()

df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values('Importance', ascending=True)

# Bar chart
fig = px.bar(df, x='Importance', y='Feature', orientation='h',
             title='Feature Importance', color='Importance',
             color_continuous_scale='Blues')
st.plotly_chart(fig, use_container_width=True)

# Cumulative importance
df_sorted = df.sort_values('Importance', ascending=False)
df_sorted['Cumulative'] = df_sorted['Importance'].cumsum()

fig = px.line(df_sorted, x='Feature', y='Cumulative',
              title='Cumulative Feature Importance')
fig.add_hline(y=0.8, line_dash="dash", annotation_text="80% threshold")
st.plotly_chart(fig, use_container_width=True)
```

## Interactive Model Exploration

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Interactive Model Explorer")

# Sidebar parameters
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)

# Generate data
@st.cache_data
def generate_data():
    X, y = make_classification(n_samples=1000, n_features=20,
                               n_informative=10, n_redundant=5,
                               random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = generate_data()

# Train model
@st.cache_data
def train_model(n_estimators, max_depth, min_samples_split):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return model, train_acc, test_acc

with st.spinner("Training model..."):
    model, train_acc, test_acc = train_model(n_estimators, max_depth, min_samples_split)

# Display results
col1, col2 = st.columns(2)
col1.metric("Training Accuracy", f"{train_acc:.2%}")
col2.metric("Test Accuracy", f"{test_acc:.2%}")

# Overfitting warning
if train_acc - test_acc > 0.1:
    st.warning("⚠️ Potential overfitting detected!")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': [f'Feature {i}' for i in range(20)],
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

st.subheader("Top 10 Feature Importance")
st.bar_chart(importance_df.set_index('Feature'))
```

## Real-time Prediction Monitor

```python
import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px

st.title("Real-time Prediction Monitor")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame(columns=['timestamp', 'prediction', 'confidence'])

# Placeholders
metrics_placeholder = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

# Simulate real-time predictions
if st.button("Start Monitoring"):
    for i in range(50):
        # Simulate new prediction
        new_pred = pd.DataFrame({
            'timestamp': [pd.Timestamp.now()],
            'prediction': [np.random.choice(['Positive', 'Negative'])],
            'confidence': [np.random.uniform(0.5, 1.0)]
        })

        st.session_state.predictions = pd.concat(
            [st.session_state.predictions, new_pred],
            ignore_index=True
        ).tail(100)

        df = st.session_state.predictions

        # Update metrics
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predictions", len(df))
            col2.metric("Avg Confidence", f"{df['confidence'].mean():.2%}")
            positive_pct = (df['prediction'] == 'Positive').mean()
            col3.metric("Positive Rate", f"{positive_pct:.2%}")

        # Update chart
        fig = px.line(df, x='timestamp', y='confidence',
                      color='prediction', title='Prediction Confidence Over Time')
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Update table
        table_placeholder.dataframe(df.tail(10))

        time.sleep(0.5)
```

## Exercises

1. Build a complete model training and evaluation dashboard
2. Create an interactive hyperparameter tuning interface
3. Build a real-time prediction monitoring system
4. Create a model comparison dashboard for multiple models

## Additional Resources

- [Streamlit ML Gallery](https://streamlit.io/gallery?category=machine-learning)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [MLflow + Streamlit](https://mlflow.org/)
