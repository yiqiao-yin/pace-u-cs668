# API Integration

## Overview

Most LLM providers offer APIs for programmatic access to their models. This section covers how to integrate with major LLM APIs, handle authentication, manage requests, and build robust applications.

## OpenAI API

### Setup

```bash
pip install openai
```

```python
from openai import OpenAI
import os

# Initialize client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

### Chat Completions

```python
from openai import OpenAI

client = OpenAI()

# Basic chat completion
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to calculate factorial."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

### Streaming Responses

```python
from openai import OpenAI

client = OpenAI()

# Stream responses
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain machine learning in detail."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Function Calling

```python
from openai import OpenAI
import json

client = OpenAI()

# Define functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if function was called
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(f"Function: {function_name}, Args: {arguments}")
```

## Anthropic Claude API

### Setup

```bash
pip install anthropic
```

```python
from anthropic import Anthropic
import os

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
```

### Messages API

```python
from anthropic import Anthropic

client = Anthropic()

# Basic message
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain the theory of relativity."}
    ]
)

print(response.content[0].text)
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
```

### System Prompts

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    system="You are a professional Python developer. Provide clean, well-documented code.",
    messages=[
        {"role": "user", "content": "Create a REST API using FastAPI."}
    ]
)

print(response.content[0].text)
```

### Streaming

```python
from anthropic import Anthropic

client = Anthropic()

with client.messages.stream(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story about AI."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Vision Capabilities

```python
from anthropic import Anthropic
import base64

client = Anthropic()

# Read image and encode
with open("image.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": "What's in this image?"
                }
            ]
        }
    ]
)

print(response.content[0].text)
```

## Hugging Face Inference API

### Setup

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="your_hf_token")
```

### Text Generation

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="meta-llama/Llama-2-70b-chat-hf")

response = client.text_generation(
    "Explain quantum entanglement:",
    max_new_tokens=200,
    temperature=0.7
)

print(response)
```

### Chat Completion

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="meta-llama/Llama-2-70b-chat-hf")

messages = [
    {"role": "user", "content": "What is machine learning?"}
]

response = client.chat_completion(
    messages=messages,
    max_tokens=500
)

print(response.choices[0].message.content)
```

## Rate Limiting and Quota Management

### Handling Rate Limits

```python
import time
from openai import OpenAI, RateLimitError

client = OpenAI()

def make_request_with_retry(messages, max_retries=3):
    """Make API request with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return response
        except RateLimitError:
            wait_time = 2 ** attempt
            print(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)

    raise Exception("Max retries exceeded")
```

### Token Counting

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    """Count tokens for a given text"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def count_message_tokens(messages, model="gpt-4"):
    """Count tokens for chat messages"""
    encoding = tiktoken.encoding_for_model(model)
    total = 0
    for message in messages:
        total += 4  # message overhead
        for key, value in message.items():
            total += len(encoding.encode(value))
    total += 2  # reply priming
    return total

# Example
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]
print(f"Token count: {count_message_tokens(messages)}")
```

## Cost Optimization

### Caching Responses

```python
import hashlib
import json

class LLMCache:
    def __init__(self):
        self.cache = {}

    def get_key(self, messages, model):
        """Generate cache key from request"""
        content = json.dumps({"messages": messages, "model": model})
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, messages, model):
        """Get cached response"""
        key = self.get_key(messages, model)
        return self.cache.get(key)

    def set(self, messages, model, response):
        """Cache response"""
        key = self.get_key(messages, model)
        self.cache[key] = response

# Usage
cache = LLMCache()

def cached_completion(messages, model="gpt-4"):
    cached = cache.get(messages, model)
    if cached:
        return cached

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    cache.set(messages, model, response)
    return response
```

### Model Selection Strategy

```python
def select_model(task_complexity, budget_mode=False):
    """Select appropriate model based on task"""
    if budget_mode:
        return "gpt-3.5-turbo"

    complexity_map = {
        "simple": "gpt-3.5-turbo",
        "medium": "gpt-4",
        "complex": "gpt-4-turbo"
    }
    return complexity_map.get(task_complexity, "gpt-4")
```

## Error Handling

### Robust API Client

```python
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustLLMClient:
    def __init__(self, max_retries=3, timeout=60):
        self.client = OpenAI()
        self.max_retries = max_retries
        self.timeout = timeout

    def complete(self, messages, model="gpt-4", **kwargs):
        """Make robust API call with error handling"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=self.timeout,
                    **kwargs
                )
                return response

            except RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Rate limit hit, waiting {wait}s")
                time.sleep(wait)

            except APIConnectionError:
                logger.error("Connection error, retrying...")
                time.sleep(1)

            except APIError as e:
                logger.error(f"API error: {e}")
                if e.status_code >= 500:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise Exception(f"Failed after {self.max_retries} attempts")

# Usage
client = RobustLLMClient()
response = client.complete([{"role": "user", "content": "Hello"}])
```

## Environment Setup

### Managing API Keys

```python
import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()

# Access keys
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")

# .env file format:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

## Exercises

1. Build a multi-provider LLM client that can switch between APIs
2. Implement a token budget manager
3. Create a response caching system with TTL
4. Build an async client for batch processing

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Hugging Face Inference API](https://huggingface.co/docs/api-inference)
