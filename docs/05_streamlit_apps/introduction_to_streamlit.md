# Introduction to Streamlit

## Overview

Streamlit is an open-source Python framework for building data science and machine learning web applications. It allows you to create interactive dashboards and tools with pure Python, without needing to learn HTML, CSS, or JavaScript.

## What is Streamlit?

Streamlit transforms Python scripts into shareable web apps in minutes. Key features include:

- **Pure Python**: No frontend knowledge required
- **Reactive**: Apps update automatically when data changes
- **Widgets**: Rich set of interactive components
- **Easy Deployment**: One-click deployment to Streamlit Cloud
- **Fast Development**: Hot-reload during development

## Installation and Setup

```bash
# Install Streamlit
pip install streamlit

# Verify installation
streamlit --version

# Run hello world demo
streamlit hello
```

## Your First Streamlit App

### Basic App Structure

Create a file named `app.py`:

```python
import streamlit as st

# Title
st.title("My First Streamlit App")

# Header and text
st.header("Welcome!")
st.write("This is a simple Streamlit application.")

# User input
name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name}!")
```

### Running the App

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## Streamlit Execution Model

### Understanding the Flow

```python
import streamlit as st

# Streamlit runs the entire script top-to-bottom
# every time the user interacts with the app

st.write("This runs every time!")

# Widgets automatically trigger reruns
button_clicked = st.button("Click me")
if button_clicked:
    st.write("Button was clicked!")

# The script runs again when button is clicked
```

### Script Execution

1. User opens app → Script runs
2. User interacts with widget → Script reruns
3. State is managed through session_state

## Basic Components

### Text Elements

```python
import streamlit as st

# Title and headers
st.title("Main Title")
st.header("Header")
st.subheader("Subheader")

# Text
st.text("Fixed-width text")
st.write("Write anything - text, data, plots")
st.markdown("**Bold** and *italic* text")
st.latex(r"E = mc^2")

# Code
st.code("""
def hello():
    print("Hello, World!")
""", language="python")

# Caption and divider
st.caption("This is a caption")
st.divider()
```

### Data Display

```python
import streamlit as st
import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NYC', 'LA', 'Chicago']
})

# Display dataframe
st.dataframe(df)  # Interactive table

# Static table
st.table(df)

# Metrics
st.metric(label="Temperature", value="70°F", delta="1.2°F")

# JSON
st.json({'name': 'John', 'age': 30})
```

### Input Widgets

```python
import streamlit as st

# Text input
name = st.text_input("Your name", value="", placeholder="Enter name...")

# Number input
age = st.number_input("Your age", min_value=0, max_value=150, value=25)

# Slider
temperature = st.slider("Temperature", 0.0, 1.0, 0.5, step=0.1)

# Select box
option = st.selectbox("Choose option", ["A", "B", "C"])

# Multi-select
options = st.multiselect("Select multiple", ["X", "Y", "Z"])

# Checkbox
agree = st.checkbox("I agree")

# Radio
choice = st.radio("Pick one", ["Option 1", "Option 2", "Option 3"])

# Button
if st.button("Submit"):
    st.write("Submitted!")

# File uploader
uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt"])
```

## Layout and Containers

### Columns

```python
import streamlit as st

# Create columns
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Column 1")
    st.write("Content for column 1")

with col2:
    st.header("Column 2")
    st.write("Content for column 2")

with col3:
    st.header("Column 3")
    st.write("Content for column 3")

# Columns with different widths
left, right = st.columns([2, 1])  # 2:1 ratio
```

### Expander

```python
import streamlit as st

with st.expander("Click to expand"):
    st.write("This content is hidden by default")
    st.image("image.jpg")
```

### Tabs

```python
import streamlit as st

tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

with tab1:
    st.write("Content of Tab 1")

with tab2:
    st.write("Content of Tab 2")

with tab3:
    st.write("Content of Tab 3")
```

### Sidebar

```python
import streamlit as st

# Add content to sidebar
st.sidebar.title("Sidebar")
st.sidebar.write("This is the sidebar")

# Sidebar widgets
option = st.sidebar.selectbox("Select", ["A", "B", "C"])
value = st.sidebar.slider("Value", 0, 100, 50)
```

### Container

```python
import streamlit as st

# Create a container
container = st.container()

# Add content later
container.write("This was added later")

# Or use context manager
with st.container():
    st.write("Content in container")
```

## State Management

### Session State

```python
import streamlit as st

# Initialize state
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Update state
if st.button("Increment"):
    st.session_state.counter += 1

st.write(f"Counter: {st.session_state.counter}")

# State persists across reruns
```

### Callback Functions

```python
import streamlit as st

def on_change():
    st.session_state.result = st.session_state.input_value.upper()

# Initialize
if 'result' not in st.session_state:
    st.session_state.result = ""

# Widget with callback
st.text_input(
    "Enter text",
    key="input_value",
    on_change=on_change
)

st.write(f"Result: {st.session_state.result}")
```

## Caching

### Cache Data

```python
import streamlit as st
import pandas as pd

# Cache data loading
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# This only runs once, then cached
df = load_data("large_file.csv")
st.dataframe(df)
```

### Cache Resources

```python
import streamlit as st
from transformers import pipeline

# Cache ML models
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()  # Cached across all users
```

### Cache Parameters

```python
@st.cache_data(ttl=3600)  # Time to live: 1 hour
def fetch_data():
    return get_data_from_api()

@st.cache_data(max_entries=100)  # Max cached items
def process_data(data):
    return expensive_computation(data)
```

## Running and Configuration

### Command Line Options

```bash
# Run app
streamlit run app.py

# Specify port
streamlit run app.py --server.port 8080

# Run without auto-opening browser
streamlit run app.py --server.headless true
```

### Configuration File

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
headless = true

[browser]
gatherUsageStats = false
```

## Best Practices

1. **Use caching**: Cache data loading and model initialization
2. **Organize with containers**: Use columns, tabs, and expanders
3. **Manage state**: Use session_state for persistent data
4. **Optimize performance**: Avoid recomputing on every rerun
5. **Keep it simple**: Streamlit works best for straightforward apps

## Exercises

1. Create a simple calculator app with input fields and buttons
2. Build a data viewer that uploads and displays CSV files
3. Create an app with sidebar navigation and multiple pages
4. Implement a counter using session state

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Community](https://discuss.streamlit.io/)
