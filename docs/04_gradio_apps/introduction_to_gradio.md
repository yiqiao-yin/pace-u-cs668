# Introduction to Gradio

## Overview

Gradio is a Python library that allows you to quickly create customizable web interfaces for machine learning models. With just a few lines of code, you can create interactive demos that anyone can use through a web browser.

## What is Gradio?

Gradio simplifies the process of:
- Creating web UIs for ML models
- Sharing demos with others
- Getting feedback on model performance
- Prototyping and testing models

### Key Features

- **Simple API**: Create interfaces with minimal code
- **Multiple Components**: Text, images, audio, video, and more
- **Easy Sharing**: Generate public URLs instantly
- **Hugging Face Integration**: Deploy directly to Spaces
- **Customizable**: CSS styling and layout options

## Installation and Setup

```bash
# Install Gradio
pip install gradio

# Verify installation
python -c "import gradio; print(gradio.__version__)"
```

## Your First Gradio App

### Hello World

```python
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

# Create interface
demo = gr.Interface(
    fn=greet,           # Function to call
    inputs="text",      # Input component
    outputs="text"      # Output component
)

# Launch the app
demo.launch()
```

Running this opens a web interface at `http://localhost:7860`.

### Understanding the Interface

```python
import gradio as gr

def process(text, number, option):
    return f"Text: {text}, Number: {number}, Option: {option}"

demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Textbox(label="Enter text"),
        gr.Number(label="Enter number"),
        gr.Dropdown(["Option A", "Option B", "Option C"], label="Select")
    ],
    outputs=gr.Textbox(label="Result"),
    title="My First App",
    description="This is a simple demo application."
)

demo.launch()
```

## Basic Interface Components

### Input Components

```python
import gradio as gr

# Text input
text_input = gr.Textbox(
    label="Name",
    placeholder="Enter your name",
    lines=1,
    max_lines=5
)

# Number input
number_input = gr.Number(
    label="Age",
    value=25,
    minimum=0,
    maximum=150
)

# Slider
slider_input = gr.Slider(
    label="Temperature",
    minimum=0,
    maximum=1,
    step=0.1,
    value=0.7
)

# Checkbox
checkbox_input = gr.Checkbox(
    label="I agree to the terms",
    value=False
)

# Radio buttons
radio_input = gr.Radio(
    choices=["Small", "Medium", "Large"],
    label="Size",
    value="Medium"
)

# Dropdown
dropdown_input = gr.Dropdown(
    choices=["Red", "Green", "Blue"],
    label="Color",
    multiselect=False
)
```

### Output Components

```python
# Text output
text_output = gr.Textbox(label="Response")

# Markdown output
markdown_output = gr.Markdown()

# JSON output
json_output = gr.JSON(label="Data")

# Label for classification
label_output = gr.Label(num_top_classes=5)

# Dataframe output
df_output = gr.Dataframe(headers=["Name", "Value"])
```

## Launching Your App

### Basic Launch

```python
demo.launch()
```

### Launch Options

```python
demo.launch(
    server_name="0.0.0.0",  # Allow external connections
    server_port=7860,        # Custom port
    share=True,              # Create public URL
    auth=("username", "password"),  # Basic authentication
    ssl_verify=False,        # SSL verification
    debug=True               # Enable debug mode
)
```

### Creating Shareable Links

```python
# Generate a public URL (valid for 72 hours)
demo.launch(share=True)

# Output: Running on public URL: https://xxxxx.gradio.live
```

## Multiple Inputs and Outputs

```python
import gradio as gr

def process_data(name, age, interests):
    # Process inputs
    greeting = f"Hello {name}!"
    age_group = "young" if age < 30 else "experienced"
    interest_list = interests.split(", ")

    return greeting, age_group, interest_list

demo = gr.Interface(
    fn=process_data,
    inputs=[
        gr.Textbox(label="Name"),
        gr.Number(label="Age"),
        gr.Textbox(label="Interests (comma-separated)")
    ],
    outputs=[
        gr.Textbox(label="Greeting"),
        gr.Textbox(label="Age Group"),
        gr.JSON(label="Interests")
    ]
)

demo.launch()
```

## Real ML Model Example

```python
import gradio as gr
from transformers import pipeline

# Load a pre-trained model
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = classifier(text)[0]
    return {result['label']: result['score']}

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        label="Enter text",
        placeholder="Type something to analyze..."
    ),
    outputs=gr.Label(label="Sentiment"),
    title="Sentiment Analysis",
    description="Analyze the sentiment of your text using a pre-trained model.",
    examples=[
        ["I love this product! It's amazing!"],
        ["This is terrible. I'm very disappointed."],
        ["The weather is nice today."]
    ]
)

demo.launch()
```

## Adding Examples

```python
import gradio as gr

def echo(text, number):
    return f"{text} x {number}"

demo = gr.Interface(
    fn=echo,
    inputs=[
        gr.Textbox(label="Text"),
        gr.Number(label="Number")
    ],
    outputs=gr.Textbox(label="Result"),
    examples=[
        ["Hello", 3],
        ["World", 5],
        ["Gradio", 10]
    ],
    cache_examples=True  # Cache example outputs for faster loading
)

demo.launch()
```

## Styling and Theming

### Built-in Themes

```python
import gradio as gr

demo = gr.Interface(
    fn=lambda x: x,
    inputs="text",
    outputs="text",
    theme=gr.themes.Soft()  # or Default(), Monochrome(), Glass()
)

demo.launch()
```

### Custom CSS

```python
import gradio as gr

custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.gr-button {
    background-color: #4CAF50;
    color: white;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Custom Styled App")
    input_text = gr.Textbox(label="Input")
    output_text = gr.Textbox(label="Output")
    btn = gr.Button("Submit")
    btn.click(fn=lambda x: x.upper(), inputs=input_text, outputs=output_text)

demo.launch()
```

## Error Handling

```python
import gradio as gr

def safe_divide(a, b):
    if b == 0:
        raise gr.Error("Cannot divide by zero!")
    return a / b

def with_warning(a, b):
    if b < 0:
        gr.Warning("Using negative number as divisor")
    return a / b

demo = gr.Interface(
    fn=safe_divide,
    inputs=[
        gr.Number(label="Numerator"),
        gr.Number(label="Denominator")
    ],
    outputs=gr.Number(label="Result")
)

demo.launch()
```

## Flagging and Feedback

```python
import gradio as gr

def classify(text):
    # Your classification logic
    return "positive"

demo = gr.Interface(
    fn=classify,
    inputs="text",
    outputs="text",
    flagging_mode="manual",  # or "auto" or "never"
    flagging_options=["Wrong", "Offensive", "Other"],
    flagging_dir="./flagged_data"  # Where to save flagged examples
)

demo.launch()
```

## Best Practices

1. **Clear Labels**: Always use descriptive labels for inputs/outputs
2. **Examples**: Provide examples to help users understand expected inputs
3. **Error Messages**: Handle errors gracefully with informative messages
4. **Documentation**: Include title and description explaining the app
5. **Testing**: Test with various inputs before sharing

## Exercises

1. Create a simple calculator interface with basic operations
2. Build a text-to-uppercase converter with character count output
3. Create an interface that accepts multiple inputs and returns a formatted result
4. Add examples and custom styling to your interface

## Additional Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [Gradio Guides](https://gradio.app/guides/)
- [Gradio GitHub](https://github.com/gradio-app/gradio)
