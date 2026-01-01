# Introduction to AI Agents

## Overview

AI agents are autonomous systems that use language models to reason, plan, and execute multi-step tasks. Unlike simple chatbots that respond to single queries, agents can break down complex goals into actions, use tools, and adapt based on feedback.

## What are AI Agents?

### Definition

```python
agent_definition = """
An AI agent is a system that:
1. Perceives its environment (receives input/observations)
2. Reasons about the task (using an LLM)
3. Plans a sequence of actions
4. Executes actions (often using tools)
5. Observes results and adapts
"""
```

### Agent vs Chatbot

| Chatbot | Agent |
|---------|-------|
| Single-turn responses | Multi-step task execution |
| Stateless conversations | Maintains goal and progress |
| No external actions | Uses tools and APIs |
| Answers questions | Accomplishes tasks |
| User-driven | Goal-driven |

## Agent Architecture

### Core Components

```python
agent_components = {
    "reasoning_engine": {
        "description": "LLM that decides what to do",
        "functions": ["interpret_goal", "plan_actions", "reflect_on_results"]
    },
    "memory": {
        "short_term": "Current conversation and task context",
        "long_term": "Persistent knowledge and past experiences"
    },
    "planning": {
        "description": "Breaking goals into actionable steps",
        "strategies": ["task_decomposition", "dependency_ordering"]
    },
    "tools": {
        "description": "Functions the agent can call",
        "examples": ["search", "calculator", "code_execution", "api_calls"]
    },
    "execution": {
        "description": "Running tools and observing results",
        "includes": ["error_handling", "retries", "fallbacks"]
    }
}
```

### Basic Agent Loop

```python
class SimpleAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.memory = []

    def run(self, goal):
        """Main agent loop"""
        self.memory.append({"role": "user", "content": goal})

        while True:
            # Think: Decide what to do next
            action = self.think()

            if action["type"] == "finish":
                return action["output"]

            # Act: Execute the chosen action
            result = self.execute(action)

            # Observe: Add result to memory
            self.memory.append({
                "role": "observation",
                "content": f"Tool {action['tool']}: {result}"
            })

    def think(self):
        """Decide the next action"""
        prompt = self.build_prompt()
        response = self.llm(prompt)
        return self.parse_action(response)

    def execute(self, action):
        """Execute a tool"""
        tool = self.tools.get(action["tool"])
        if tool:
            return tool.run(action["input"])
        return f"Unknown tool: {action['tool']}"
```

## Agent Architectures

### ReAct (Reasoning + Acting)

```python
react_pattern = """
The agent alternates between:
1. Thought: Reasoning about the current state
2. Action: Choosing and executing a tool
3. Observation: Processing the result

Example:
User: What's the weather in Paris and should I bring an umbrella?

Thought: I need to find the current weather in Paris.
Action: weather_api(location="Paris")
Observation: Temperature: 15°C, Conditions: Rainy, Humidity: 85%

Thought: It's rainy in Paris. I should recommend an umbrella.
Action: finish("It's rainy in Paris (15°C). Yes, definitely bring an umbrella!")
"""
```

### Plan-and-Execute

```python
plan_execute_pattern = """
1. Create a plan upfront
2. Execute each step
3. Revise plan if needed

Example:
User: Research and summarize recent AI news

Plan:
1. Search for recent AI news articles
2. Read top 3 articles
3. Extract key points from each
4. Synthesize into summary

Execution:
[Step 1] Searching... Found 10 articles
[Step 2] Reading article 1... Done
[Step 2] Reading article 2... Done
...
[Step 4] Writing summary...

Output: [Summary of AI news]
"""
```

### Reflexion

```python
reflexion_pattern = """
After each attempt:
1. Evaluate success/failure
2. Reflect on what went wrong
3. Generate improved approach
4. Retry with new strategy

Example:
Attempt 1: Direct search failed
Reflection: Search query too broad, got irrelevant results
Improvement: Use more specific keywords
Attempt 2: Specific search succeeded
"""
```

## Real-World Agent Applications

### Research Assistant

```python
research_agent = """
Task: Research a topic comprehensively

Capabilities:
- Web search for information
- Read and summarize articles
- Identify key sources
- Synthesize findings
- Generate citations

Workflow:
1. Understand research question
2. Search for relevant sources
3. Read and extract information
4. Cross-reference facts
5. Write comprehensive summary
"""
```

### Code Assistant

```python
code_agent = """
Task: Help with coding tasks

Capabilities:
- Write code
- Execute code in sandbox
- Debug errors
- Search documentation
- Refactor code

Workflow:
1. Understand requirements
2. Plan implementation
3. Write code
4. Test and debug
5. Refine until working
"""
```

### Customer Support

```python
support_agent = """
Task: Handle customer inquiries

Capabilities:
- Search knowledge base
- Look up customer information
- Create support tickets
- Escalate to human
- Track resolution

Workflow:
1. Understand customer issue
2. Search for solutions
3. Provide answer or escalate
4. Follow up if needed
"""
```

## Simple Agent Implementation

```python
from openai import OpenAI

class Tool:
    def __init__(self, name, description, function):
        self.name = name
        self.description = description
        self.function = function

    def run(self, input_str):
        return self.function(input_str)

class BasicAgent:
    def __init__(self, tools):
        self.client = OpenAI()
        self.tools = tools
        self.tool_descriptions = self._format_tools()

    def _format_tools(self):
        return "\n".join([
            f"- {t.name}: {t.description}"
            for t in self.tools
        ])

    def run(self, task, max_iterations=10):
        messages = [
            {"role": "system", "content": f"""You are a helpful agent.
Available tools:
{self.tool_descriptions}

Respond in this format:
Thought: [your reasoning]
Action: [tool_name]
Action Input: [input for the tool]

Or if you have the final answer:
Thought: [your reasoning]
Final Answer: [your answer]"""},
            {"role": "user", "content": task}
        ]

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            output = response.choices[0].message.content

            if "Final Answer:" in output:
                return output.split("Final Answer:")[-1].strip()

            # Parse and execute action
            action, action_input = self._parse_action(output)
            if action:
                result = self._execute_action(action, action_input)
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": f"Observation: {result}"})

        return "Max iterations reached"

    def _parse_action(self, output):
        # Simple parsing - production would be more robust
        lines = output.split("\n")
        action = None
        action_input = None
        for line in lines:
            if line.startswith("Action:"):
                action = line.split("Action:")[-1].strip()
            if line.startswith("Action Input:"):
                action_input = line.split("Action Input:")[-1].strip()
        return action, action_input

    def _execute_action(self, action_name, action_input):
        for tool in self.tools:
            if tool.name == action_name:
                return tool.run(action_input)
        return f"Unknown tool: {action_name}"

# Example usage
def calculator(expression):
    try:
        return str(eval(expression))
    except:
        return "Error evaluating expression"

def search(query):
    return f"Search results for '{query}': [simulated results]"

agent = BasicAgent([
    Tool("calculator", "Evaluate math expressions", calculator),
    Tool("search", "Search the web", search)
])

result = agent.run("What is 25 * 17 + 13?")
print(result)
```

## Exercises

1. Build a simple ReAct agent with 2-3 tools
2. Implement memory to track conversation history
3. Add error handling and retry logic
4. Create an agent for a specific use case

## Additional Resources

- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
