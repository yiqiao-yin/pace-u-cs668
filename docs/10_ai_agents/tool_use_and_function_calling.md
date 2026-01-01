# Tool Use and Function Calling

## Overview

Tools extend AI agents beyond text generation, allowing them to interact with the real world through APIs, databases, and external services. Function calling is the mechanism that enables structured tool invocation.

## What are Tools?

### Definition

```python
tool_concept = """
A tool is a function that an agent can call to:
- Retrieve information (search, database queries)
- Take actions (send emails, create files)
- Compute results (calculations, data processing)
- Interact with external services (APIs, web services)
"""
```

### Anatomy of a Tool

```python
class Tool:
    def __init__(self):
        self.name = "tool_name"
        self.description = "What this tool does"
        self.parameters = {
            "param1": {
                "type": "string",
                "description": "Description of param1"
            }
        }

    def run(self, **kwargs):
        """Execute the tool"""
        pass
```

## Function Calling with OpenAI

### Defining Functions

```python
from openai import OpenAI

client = OpenAI()

# Define available functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

### Handling Function Calls

```python
import json

def get_weather(location, unit="celsius"):
    """Actual implementation of weather lookup"""
    # In reality, call a weather API
    return {"temperature": 22, "unit": unit, "condition": "sunny"}

def search_web(query):
    """Actual implementation of web search"""
    return f"Search results for: {query}"

# Map function names to implementations
available_functions = {
    "get_weather": get_weather,
    "search_web": search_web
}

def run_agent(user_message):
    messages = [{"role": "user", "content": user_message}]

    # First call to get function decision
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # Check if model wants to call a function
    if message.tool_calls:
        # Execute each function call
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Call the function
            function_response = available_functions[function_name](**function_args)

            # Add function result to messages
            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(function_response)
            })

        # Get final response with function results
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        return final_response.choices[0].message.content

    return message.content

# Example
result = run_agent("What's the weather like in Tokyo?")
print(result)
```

## Tool Selection and Routing

### Dynamic Tool Selection

```python
def select_tools_for_query(query, all_tools, llm):
    """Select relevant tools based on the query"""
    tool_descriptions = "\n".join([
        f"- {t['function']['name']}: {t['function']['description']}"
        for t in all_tools
    ])

    prompt = f"""
    Given this user query: "{query}"

    Select which tools would be helpful from:
    {tool_descriptions}

    Return only the tool names, comma-separated.
    """

    response = llm(prompt)
    selected_names = [n.strip() for n in response.split(",")]

    return [t for t in all_tools if t['function']['name'] in selected_names]
```

### Tool Chaining

```python
def chain_tools(task, tools, llm):
    """Execute multiple tools in sequence"""
    steps = []
    context = ""

    while True:
        # Decide next tool
        prompt = f"""
        Task: {task}
        Previous steps: {steps}
        Current context: {context}

        What tool should be used next? Or is the task complete?
        """

        decision = llm(prompt)

        if "complete" in decision.lower():
            break

        # Parse and execute tool
        tool_name, tool_input = parse_tool_decision(decision)
        result = execute_tool(tool_name, tool_input, tools)

        steps.append(f"{tool_name}: {result}")
        context += f"\n{tool_name} result: {result}"

    return context
```

## Building Custom Tools

### Simple Tool Implementation

```python
class CalculatorTool:
    name = "calculator"
    description = "Perform mathematical calculations"

    def run(self, expression: str) -> str:
        try:
            # Safe evaluation (in production, use a proper math parser)
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

class WikipediaTool:
    name = "wikipedia"
    description = "Search Wikipedia for information"

    def run(self, query: str) -> str:
        import wikipedia
        try:
            return wikipedia.summary(query, sentences=3)
        except:
            return "No results found"

class CodeExecutorTool:
    name = "python"
    description = "Execute Python code and return the output"

    def run(self, code: str) -> str:
        import io
        import sys

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            exec(code)
            output = buffer.getvalue()
            return output if output else "Code executed successfully"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout
```

### API Tool Template

```python
import requests

class APITool:
    def __init__(self, name, base_url, api_key=None):
        self.name = name
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def get(self, endpoint, params=None):
        response = requests.get(
            f"{self.base_url}/{endpoint}",
            headers=self.headers,
            params=params
        )
        return response.json()

    def post(self, endpoint, data):
        response = requests.post(
            f"{self.base_url}/{endpoint}",
            headers=self.headers,
            json=data
        )
        return response.json()

# Example: Weather API Tool
class WeatherAPITool(APITool):
    name = "weather"
    description = "Get current weather for a location"

    def __init__(self, api_key):
        super().__init__("weather", "https://api.weatherapi.com/v1", api_key)

    def run(self, location: str) -> str:
        data = self.get("current.json", {"q": location})
        return f"Temperature: {data['current']['temp_c']}°C, " \
               f"Condition: {data['current']['condition']['text']}"
```

## Error Handling

### Robust Tool Execution

```python
import time
from functools import wraps

def with_retry(max_attempts=3, delay=1):
    """Decorator for retrying failed tool calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
            return f"Error after {max_attempts} attempts: {str(last_exception)}"
        return wrapper
    return decorator

class RobustTool:
    def __init__(self, tool, fallback=None):
        self.tool = tool
        self.fallback = fallback

    @with_retry(max_attempts=3)
    def run(self, **kwargs):
        try:
            return self.tool.run(**kwargs)
        except Exception as e:
            if self.fallback:
                return self.fallback(**kwargs)
            raise e
```

## Tool Composition

### Tool Chains

```python
class ToolChain:
    """Chain multiple tools together"""

    def __init__(self, tools):
        self.tools = tools

    def run(self, initial_input):
        result = initial_input
        for tool in self.tools:
            result = tool.run(result)
        return result

# Example: Search -> Summarize -> Translate
search_and_translate = ToolChain([
    SearchTool(),
    SummarizeTool(),
    TranslateTool(target_lang="es")
])

result = search_and_translate.run("Latest AI news")
```

### Parallel Tool Execution

```python
import asyncio

async def run_tools_parallel(tools, inputs):
    """Run multiple tools in parallel"""
    async def run_tool(tool, inp):
        return await asyncio.to_thread(tool.run, inp)

    tasks = [run_tool(tool, inp) for tool, inp in zip(tools, inputs)]
    results = await asyncio.gather(*tasks)
    return results

# Example
tools = [SearchTool(), WikipediaTool(), NewsAPITool()]
queries = ["AI", "machine learning", "deep learning"]
results = asyncio.run(run_tools_parallel(tools, queries))
```

## Exercises

1. Create a tool that queries a REST API
2. Implement function calling with Claude API
3. Build a tool chain for a specific workflow
4. Add error handling and fallbacks to tools

## Additional Resources

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/claude/docs/tool-use)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)
