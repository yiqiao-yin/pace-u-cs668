# Input and Output Components

## Overview

Gradio provides a rich set of components for handling different types of inputs and outputs. This section covers the various component types available, including text, images, audio, video, and files.

## Text Components

### Textbox

```python
import gradio as gr

# Basic textbox
text_input = gr.Textbox(
    label="Enter text",
    placeholder="Type here...",
    lines=3,
    max_lines=10,
    value="Default text"
)

# Password field
password = gr.Textbox(
    label="Password",
    type="password"
)

# Read-only textbox
readonly = gr.Textbox(
    label="Result",
    interactive=False
)
```

### TextArea for Long Text

```python
# Multi-line text area
textarea = gr.Textbox(
    label="Long Text",
    lines=10,
    max_lines=20,
    show_copy_button=True
)
```

### Code Editor

```python
# Code input with syntax highlighting
code = gr.Code(
    label="Python Code",
    language="python",
    lines=10,
    value="def hello():\n    print('Hello, World!')"
)
```

## Numeric Components

### Number Input

```python
number = gr.Number(
    label="Enter a number",
    value=0,
    minimum=-100,
    maximum=100,
    step=1,
    precision=2
)
```

### Slider

```python
slider = gr.Slider(
    label="Temperature",
    minimum=0,
    maximum=2,
    value=0.7,
    step=0.1,
    info="Controls randomness"
)
```

## Selection Components

### Radio Buttons

```python
radio = gr.Radio(
    choices=["Option A", "Option B", "Option C"],
    label="Select one",
    value="Option A",
    info="Choose your preferred option"
)
```

### Checkbox

```python
# Single checkbox
checkbox = gr.Checkbox(
    label="I agree to the terms",
    value=False
)

# Checkbox group
checkbox_group = gr.CheckboxGroup(
    choices=["Feature 1", "Feature 2", "Feature 3"],
    label="Select features",
    value=["Feature 1"]
)
```

### Dropdown

```python
# Single select dropdown
dropdown = gr.Dropdown(
    choices=["Small", "Medium", "Large"],
    label="Size",
    value="Medium",
    filterable=True
)

# Multi-select dropdown
multi_dropdown = gr.Dropdown(
    choices=["Python", "JavaScript", "Rust", "Go"],
    label="Languages",
    multiselect=True,
    value=["Python"]
)
```

## Image Components

### Image Input

```python
import gradio as gr
from PIL import Image

# Image upload
image_input = gr.Image(
    label="Upload an image",
    type="pil",  # Returns PIL Image
    sources=["upload", "webcam", "clipboard"],
    height=300
)

# Image as numpy array
image_numpy = gr.Image(
    label="Image",
    type="numpy"  # Returns numpy array
)

# Image as file path
image_path = gr.Image(
    label="Image",
    type="filepath"
)
```

### Image Output

```python
def process_image(img):
    # img is a PIL Image
    # Process and return
    return img.rotate(90)

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil", label="Processed")
)
```

### Image Examples

```python
import gradio as gr
import numpy as np

def analyze_image(img):
    # Get image dimensions
    height, width = img.shape[:2]
    return f"Image size: {width}x{height}"

demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Textbox(),
    examples=[
        ["path/to/image1.jpg"],
        ["path/to/image2.png"]
    ]
)
```

## Audio Components

### Audio Input

```python
# Audio upload or record
audio_input = gr.Audio(
    label="Audio",
    type="filepath",  # or "numpy"
    sources=["upload", "microphone"]
)

# Audio as numpy array
audio_numpy = gr.Audio(
    type="numpy"  # Returns (sample_rate, audio_data)
)
```

### Audio Output

```python
import numpy as np

def generate_audio():
    # Generate a simple sine wave
    sample_rate = 44100
    duration = 2
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    return (sample_rate, audio.astype(np.float32))

demo = gr.Interface(
    fn=generate_audio,
    inputs=[],
    outputs=gr.Audio(type="numpy", label="Generated Audio")
)
```

## Video Components

### Video Input

```python
video_input = gr.Video(
    label="Upload video",
    sources=["upload", "webcam"]
)
```

### Video Output

```python
def process_video(video_path):
    # Process video and return path to processed video
    return video_path

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(),
    outputs=gr.Video(label="Processed Video")
)
```

## File Components

### File Upload

```python
# Single file
file_input = gr.File(
    label="Upload a file",
    file_types=[".pdf", ".txt", ".csv"]
)

# Multiple files
files_input = gr.File(
    label="Upload files",
    file_count="multiple"
)
```

### File Download

```python
import gradio as gr

def create_file(text):
    with open("output.txt", "w") as f:
        f.write(text)
    return "output.txt"

demo = gr.Interface(
    fn=create_file,
    inputs=gr.Textbox(label="Text to save"),
    outputs=gr.File(label="Download")
)
```

## DataFrame Components

### Dataframe Input/Output

```python
import gradio as gr
import pandas as pd

# Dataframe input
df_input = gr.Dataframe(
    headers=["Name", "Age", "City"],
    datatype=["str", "number", "str"],
    row_count=(3, "dynamic"),
    col_count=(3, "fixed")
)

# Function that processes dataframe
def analyze_df(df):
    return df.describe()

demo = gr.Interface(
    fn=analyze_df,
    inputs=gr.Dataframe(headers=["A", "B", "C"]),
    outputs=gr.Dataframe()
)
```

## Label and Classification

### Classification Output

```python
import gradio as gr

def classify(text):
    # Return confidence scores
    return {
        "positive": 0.8,
        "negative": 0.15,
        "neutral": 0.05
    }

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Text"),
    outputs=gr.Label(num_top_classes=3)
)
```

## JSON and HighlightedText

### JSON Output

```python
def get_data():
    return {
        "name": "John",
        "age": 30,
        "skills": ["Python", "ML"]
    }

demo = gr.Interface(
    fn=get_data,
    inputs=[],
    outputs=gr.JSON(label="User Data")
)
```

### Highlighted Text

```python
import gradio as gr

def highlight_entities(text):
    # Return list of (text, label) tuples
    return [
        ("Apple", "ORG"),
        (" announced a new ",  None),
        ("iPhone", "PRODUCT"),
        (" in ", None),
        ("California", "LOC")
    ]

demo = gr.Interface(
    fn=highlight_entities,
    inputs=gr.Textbox(),
    outputs=gr.HighlightedText(
        label="Named Entities",
        combine_adjacent=True,
        color_map={
            "ORG": "blue",
            "PRODUCT": "green",
            "LOC": "red"
        }
    )
)
```

## Gallery Component

```python
import gradio as gr

def generate_images():
    # Return list of images
    return [
        ("image1.jpg", "Caption 1"),
        ("image2.jpg", "Caption 2"),
        ("image3.jpg", "Caption 3")
    ]

demo = gr.Interface(
    fn=generate_images,
    inputs=[],
    outputs=gr.Gallery(
        label="Generated Images",
        columns=3,
        height="auto"
    )
)
```

## Plot Components

### Matplotlib Plots

```python
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

def create_plot(num_points):
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, num_points)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title("Sine Wave")
    return fig

demo = gr.Interface(
    fn=create_plot,
    inputs=gr.Slider(10, 100, value=50, label="Points"),
    outputs=gr.Plot(label="Graph")
)
```

### Plotly Plots

```python
import gradio as gr
import plotly.express as px
import pandas as pd

def create_plotly():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [10, 15, 13, 17]
    })
    fig = px.line(df, x='x', y='y', title='Interactive Plot')
    return fig

demo = gr.Interface(
    fn=create_plotly,
    inputs=[],
    outputs=gr.Plot()
)
```

## Component State

### Preserving State Between Calls

```python
import gradio as gr

def increment(count):
    return count + 1

with gr.Blocks() as demo:
    count = gr.State(0)  # Hidden state component
    display = gr.Number(label="Count")
    btn = gr.Button("Increment")

    btn.click(
        fn=increment,
        inputs=count,
        outputs=count
    ).then(
        fn=lambda x: x,
        inputs=count,
        outputs=display
    )
```

## Exercises

1. Create an image processing interface with before/after comparison
2. Build an audio transcription demo with audio input and text output
3. Create a data upload and visualization interface
4. Build a multi-modal interface accepting text, image, and audio

## Additional Resources

- [Gradio Component Gallery](https://www.gradio.app/docs/components)
- [Gradio Cookbook](https://www.gradio.app/guides/)
