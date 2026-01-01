# Building ML Model Interfaces

## Overview

Gradio excels at creating user-friendly interfaces for machine learning models. This section covers how to connect various types of ML models to Gradio interfaces, including image classifiers, text generators, and multi-modal applications.

## Image Classification Demo

### Basic Image Classifier

```python
import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
import urllib
labels = urllib.request.urlopen(LABELS_URL).read().decode("utf-8").split("\n")

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_image(image):
    if image is None:
        return {}

    # Preprocess
    input_tensor = preprocess(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)

    # Get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)

    return {labels[idx]: float(prob) for prob, idx in zip(top5_prob, top5_indices)}

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="Image Classifier",
    description="Upload an image to classify it using ResNet-50",
    examples=[
        ["examples/cat.jpg"],
        ["examples/dog.jpg"]
    ]
)

demo.launch()
```

### Image Classification with Hugging Face

```python
import gradio as gr
from transformers import pipeline

# Load classifier
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

def classify(image):
    results = classifier(image)
    return {r["label"]: r["score"] for r in results}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="ViT Image Classifier"
)

demo.launch()
```

## Text Generation Interface

### Using Hugging Face Transformers

```python
import gradio as gr
from transformers import pipeline

# Load text generation model
generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt, max_length, temperature):
    result = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1
    )
    return result[0]["generated_text"]

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt..."),
        gr.Slider(50, 500, value=100, label="Max Length"),
        gr.Slider(0.1, 2.0, value=0.7, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="Text Generator",
    description="Generate text using GPT-2"
)

demo.launch()
```

### Chat Interface

```python
import gradio as gr
from transformers import pipeline

chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

def chat(message, history):
    # Format history for the model
    conversation_history = ""
    for user_msg, bot_msg in history:
        conversation_history += f"User: {user_msg}\nBot: {bot_msg}\n"

    # Generate response
    from transformers import Conversation
    conv = Conversation(message)
    response = chatbot(conv)

    return response.generated_responses[-1]

demo = gr.ChatInterface(
    fn=chat,
    title="Chatbot",
    description="Chat with DialoGPT"
)

demo.launch()
```

## Object Detection Visualization

```python
import gradio as gr
from transformers import pipeline
from PIL import Image, ImageDraw

# Load object detection model
detector = pipeline("object-detection", model="facebook/detr-resnet-50")

def detect_objects(image):
    # Run detection
    results = detector(image)

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    for result in results:
        box = result["box"]
        label = result["label"]
        score = result["score"]

        # Draw rectangle
        draw.rectangle(
            [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
            outline="red",
            width=2
        )
        # Draw label
        draw.text((box["xmin"], box["ymin"] - 10), f"{label}: {score:.2f}", fill="red")

    return image

demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Object Detection"
)

demo.launch()
```

## Multi-Modal Applications

### Image Captioning

```python
import gradio as gr
from transformers import pipeline

# Load image captioning model
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def generate_caption(image):
    result = captioner(image)
    return result[0]["generated_text"]

demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Caption"),
    title="Image Captioning",
    examples=["examples/beach.jpg", "examples/city.jpg"]
)

demo.launch()
```

### Visual Question Answering

```python
import gradio as gr
from transformers import pipeline

# Load VQA model
vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")

def answer_question(image, question):
    result = vqa(image=image, question=question)
    return {r["answer"]: r["score"] for r in result[:5]}

demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(label="Question", placeholder="Ask a question about the image...")
    ],
    outputs=gr.Label(label="Answers"),
    title="Visual Question Answering"
)

demo.launch()
```

## Speech Recognition

```python
import gradio as gr
from transformers import pipeline

# Load ASR model
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")

def transcribe(audio):
    if audio is None:
        return ""
    result = transcriber(audio)
    return result["text"]

demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs=gr.Textbox(label="Transcription"),
    title="Speech Recognition",
    description="Record or upload audio to transcribe"
)

demo.launch()
```

## Model Comparison Interface

```python
import gradio as gr
from transformers import pipeline

# Load multiple models
model_a = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
model_b = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def compare_models(text):
    result_a = model_a(text)[0]
    result_b = model_b(text)[0]

    return (
        {result_a["label"]: result_a["score"]},
        {result_b["label"]: result_b["score"]}
    )

demo = gr.Interface(
    fn=compare_models,
    inputs=gr.Textbox(label="Enter text to analyze"),
    outputs=[
        gr.Label(label="Model A (DistilBERT)"),
        gr.Label(label="Model B (Multilingual BERT)")
    ],
    title="Model Comparison",
    description="Compare sentiment analysis from two different models"
)

demo.launch()
```

## Batch Processing

```python
import gradio as gr
from transformers import pipeline

classifier = pipeline("text-classification")

def batch_classify(texts):
    # Split texts by newline
    text_list = [t.strip() for t in texts.split("\n") if t.strip()]

    results = []
    for text in text_list:
        result = classifier(text)[0]
        results.append({
            "text": text[:50] + "...",
            "label": result["label"],
            "score": round(result["score"], 4)
        })

    return results

demo = gr.Interface(
    fn=batch_classify,
    inputs=gr.Textbox(
        label="Texts (one per line)",
        lines=10,
        placeholder="Enter multiple texts, one per line..."
    ),
    outputs=gr.JSON(label="Results"),
    title="Batch Classification"
)

demo.launch()
```

## Real-time Processing with Streaming

```python
import gradio as gr
from transformers import pipeline, TextIteratorStreamer
from threading import Thread

model_id = "gpt2"
pipe = pipeline("text-generation", model=model_id)

def generate_stream(prompt):
    streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True)

    generation_kwargs = dict(
        text_inputs=prompt,
        max_new_tokens=100,
        streamer=streamer
    )

    thread = Thread(target=pipe, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text

demo = gr.Interface(
    fn=generate_stream,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Textbox(label="Generated Text"),
    live=True
)

demo.launch()
```

## Best Practices for ML Interfaces

### Model Loading

```python
import gradio as gr

# Load model once at startup
model = load_heavy_model()

def predict(input_data):
    # Use pre-loaded model
    return model(input_data)

# Don't reload model for each prediction
demo = gr.Interface(fn=predict, ...)
```

### Error Handling

```python
def safe_predict(image):
    try:
        if image is None:
            raise gr.Error("Please upload an image first!")

        result = model(image)
        return result
    except Exception as e:
        raise gr.Error(f"Prediction failed: {str(e)}")
```

### Progress Indicators

```python
import gradio as gr

def long_process(data, progress=gr.Progress()):
    results = []
    for i, item in enumerate(data):
        progress(i / len(data), desc=f"Processing item {i+1}/{len(data)}")
        results.append(process_item(item))
    return results
```

## Exercises

1. Create an image classifier with custom trained model
2. Build a multi-model comparison interface
3. Implement a real-time object detection demo
4. Create a multimodal application combining text and images

## Additional Resources

- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Gradio + Transformers](https://www.gradio.app/guides/using-hugging-face-integrations)
- [Model Hub](https://huggingface.co/models)
