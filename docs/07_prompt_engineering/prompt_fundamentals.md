# Fundamentals of Prompt Engineering

## Overview

Prompt engineering is the practice of designing and optimizing inputs to language models to achieve desired outputs. A well-crafted prompt can dramatically improve the quality, accuracy, and usefulness of LLM responses.

## What is Prompt Engineering?

### Definition

Prompt engineering is the art and science of:
- Structuring inputs to guide model behavior
- Providing context and constraints
- Eliciting specific types of responses
- Optimizing for quality and consistency

### Why It Matters

```python
# Poor prompt
response = llm("Write code")
# Result: Vague, possibly incorrect code

# Good prompt
response = llm("""
Write a Python function that:
- Takes a list of integers as input
- Returns the two numbers that sum to a target value
- Uses O(n) time complexity
- Includes docstring and type hints
""")
# Result: Precise, well-documented code
```

## Prompt Anatomy

### Key Components

```python
prompt_structure = """
[System Instructions]
You are a helpful coding assistant specializing in Python.

[Context]
The user is building a REST API using FastAPI.

[Task]
Create an endpoint that handles user authentication.

[Constraints]
- Use JWT tokens
- Include input validation
- Follow RESTful conventions

[Format]
Provide the code with comments explaining each section.

[Examples] (Optional)
Example endpoint:
@app.post("/login")
def login(credentials: LoginRequest):
    ...
"""
```

### Component Breakdown

1. **Instructions**: What the model should do or be
2. **Context**: Background information and domain
3. **Task**: The specific request or question
4. **Constraints**: Limitations and requirements
5. **Format**: How to structure the output
6. **Examples**: Sample inputs and outputs

## Clear vs Ambiguous Prompts

### Ambiguous Prompt

```python
# Unclear, open-ended
prompt = "Tell me about Python"

# Problems:
# - What aspect of Python?
# - What level of detail?
# - What format?
# - What audience?
```

### Clear Prompt

```python
# Specific, well-defined
prompt = """
Explain Python's Global Interpreter Lock (GIL) to a developer
who understands threading but is new to Python.

Include:
1. What the GIL is
2. Why it exists
3. How it affects multi-threaded programs
4. Common workarounds

Keep the explanation under 300 words.
"""
```

## Writing Effective Instructions

### Be Specific

```python
# Vague
"Summarize this article"

# Specific
"Summarize this article in 3 bullet points, focusing on the main findings and their implications for healthcare."
```

### Use Action Verbs

```python
action_verbs = {
    "analysis": ["analyze", "evaluate", "compare", "assess"],
    "creation": ["create", "generate", "write", "design"],
    "explanation": ["explain", "describe", "clarify", "define"],
    "extraction": ["extract", "identify", "list", "find"],
    "transformation": ["convert", "translate", "reformat", "simplify"]
}

# Example
"Analyze the following code and identify potential security vulnerabilities."
```

### Set the Persona

```python
personas = [
    "You are an expert Python developer with 10 years of experience.",
    "You are a patient teacher explaining concepts to beginners.",
    "You are a security auditor reviewing code for vulnerabilities.",
    "You are a technical writer creating clear documentation."
]

# Usage
prompt = f"""
{personas[0]}

Review this code and suggest improvements for better performance.
"""
```

## Output Formatting

### Structured Outputs

```python
# JSON format
prompt = """
Extract the following information from the text and return as JSON:
{
    "name": "string",
    "date": "YYYY-MM-DD",
    "amount": "number",
    "category": "string"
}

Text: {input_text}
"""

# Markdown format
prompt = """
Create a project plan with the following structure:

## Overview
Brief description

## Timeline
| Phase | Duration | Deliverables |
|-------|----------|--------------|

## Risks
- Risk 1
- Risk 2

## Next Steps
1. First step
2. Second step
"""
```

### Controlling Length

```python
length_controls = [
    "Respond in exactly 3 sentences.",
    "Keep your response under 100 words.",
    "Provide a one-paragraph summary.",
    "Give a detailed explanation (500-700 words).",
    "Answer in a single word or phrase."
]
```

## Constraints and Boundaries

### What to Include

```python
prompt = """
Explain machine learning. Include:
- A simple definition
- Three real-world examples
- Common misconceptions
- Resources for learning more
"""
```

### What to Avoid

```python
prompt = """
Write a product description. Avoid:
- Technical jargon
- Superlatives (best, greatest, etc.)
- Unverified claims
- Mentioning competitors
"""
```

### Conditional Logic

```python
prompt = """
Analyze the customer feedback.

If the sentiment is positive:
  - Summarize what customers liked
  - Suggest ways to amplify these aspects

If the sentiment is negative:
  - Identify the main complaints
  - Propose specific solutions
  - Draft a response template
"""
```

## Prompt Templates

### Basic Template

```python
from string import Template

basic_template = Template("""
You are a $role.

$context

Task: $task

Requirements:
$requirements

Please provide your response in $format format.
""")

# Usage
prompt = basic_template.substitute(
    role="data scientist",
    context="You are analyzing sales data for Q4 2023.",
    task="Identify trends and anomalies in the data.",
    requirements="- Use statistical methods\n- Provide confidence intervals",
    format="a structured report"
)
```

### Parameterized Template

```python
def create_analysis_prompt(topic, audience, depth, format_type):
    return f"""
Analyze {topic} for an audience of {audience}.

Depth: {depth}
Output Format: {format_type}

Consider:
- Key concepts and definitions
- Current state and trends
- Future implications
- Actionable insights

Begin your analysis:
"""

# Usage
prompt = create_analysis_prompt(
    topic="renewable energy adoption",
    audience="business executives",
    depth="comprehensive",
    format_type="executive summary with bullet points"
)
```

## Iterative Refinement

### The Refinement Process

```python
refinement_steps = """
1. Start with a basic prompt
2. Test with sample inputs
3. Analyze output quality
4. Identify issues:
   - Too verbose/brief?
   - Missing information?
   - Wrong format?
   - Hallucinations?
5. Modify prompt to address issues
6. Repeat until satisfactory
"""
```

### A/B Testing Prompts

```python
def test_prompts(prompts, test_inputs, evaluate_fn):
    """Compare different prompts on the same inputs"""
    results = {}

    for name, prompt in prompts.items():
        scores = []
        for input_text in test_inputs:
            full_prompt = prompt.format(input=input_text)
            response = get_llm_response(full_prompt)
            score = evaluate_fn(response, input_text)
            scores.append(score)
        results[name] = sum(scores) / len(scores)

    return results

# Example
prompts = {
    "simple": "Summarize: {input}",
    "detailed": "Provide a 3-sentence summary of: {input}",
    "structured": "Summarize in bullet points: {input}"
}

results = test_prompts(prompts, test_data, quality_scorer)
best_prompt = max(results, key=results.get)
```

## Common Pitfalls

```python
pitfalls = {
    "vagueness": "Use specific, concrete language",
    "overloading": "Don't ask for too much at once",
    "no_examples": "Provide examples for complex tasks",
    "no_constraints": "Set clear boundaries and limits",
    "assuming_knowledge": "Provide necessary context",
    "poor_formatting": "Structure prompts for readability"
}
```

## Exercises

1. Transform vague prompts into specific, effective ones
2. Create a prompt template library for common tasks
3. A/B test different prompt variations
4. Build a prompt that handles edge cases gracefully

## Additional Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design)
