# Chain-of-Thought Prompting

## Overview

Chain-of-Thought (CoT) prompting is a technique that encourages language models to break down complex problems into intermediate reasoning steps. This approach significantly improves performance on tasks requiring multi-step reasoning, math, and logic.

## What is Chain-of-Thought?

### Basic Concept

Instead of asking for a direct answer, CoT prompts the model to "think through" the problem step by step.

```python
# Without Chain-of-Thought
prompt = "What is 23 × 17?"
# Model might give wrong answer directly

# With Chain-of-Thought
prompt = """
What is 23 × 17?
Let's solve this step by step:
"""
# Model breaks down: 23 × 17 = 23 × (10 + 7) = 230 + 161 = 391
```

## Zero-Shot Chain-of-Thought

### The Magic Phrase

Simply adding "Let's think step by step" can dramatically improve reasoning.

```python
zero_shot_cot = """
Q: A store has 15 apples. They sell 6 and receive a shipment of 12 more.
Then they sell 8 more. How many apples do they have now?

A: Let's think step by step.
"""

# Model output:
# 1. Start with 15 apples
# 2. Sell 6: 15 - 6 = 9 apples
# 3. Receive 12: 9 + 12 = 21 apples
# 4. Sell 8: 21 - 8 = 13 apples
# Therefore, they have 13 apples.
```

### Variations

```python
cot_triggers = [
    "Let's think step by step.",
    "Let's break this down.",
    "Let's solve this systematically.",
    "Let's work through this carefully.",
    "First, let me understand the problem...",
    "I'll approach this step by step.",
    "Let's analyze this methodically."
]
```

## Few-Shot Chain-of-Thought

### Providing Reasoning Examples

```python
few_shot_cot = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger starts with 5 tennis balls.
He buys 2 cans, each with 3 balls.
2 cans × 3 balls = 6 new balls.
5 original + 6 new = 11 total balls.
The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and
bought 6 more, how many apples do they have?

A: The cafeteria starts with 23 apples.
They used 20 apples for lunch.
23 - 20 = 3 apples remaining.
They bought 6 more apples.
3 + 6 = 9 total apples.
The answer is 9.

Q: {new_question}

A:"""
```

### Complex Reasoning Example

```python
complex_cot = """
Q: A train travels from City A to City B at 60 mph and returns at 40 mph.
What is the average speed for the entire journey?

A: Let me work through this step by step.

Step 1: Define variables
- Let the distance between cities be d miles
- Speed going: 60 mph
- Speed returning: 40 mph

Step 2: Calculate time for each leg
- Time going = d / 60 hours
- Time returning = d / 40 hours

Step 3: Calculate total distance and time
- Total distance = 2d miles
- Total time = d/60 + d/40 = (2d + 3d) / 120 = 5d/120 = d/24 hours

Step 4: Calculate average speed
- Average speed = Total distance / Total time
- Average speed = 2d / (d/24) = 2d × 24/d = 48 mph

The answer is 48 mph.

Note: This is NOT simply (60 + 40) / 2 = 50 mph because
the train spends more time at the slower speed.

Q: {new_problem}

A:"""
```

## Multi-Step Reasoning

### Breaking Down Complex Tasks

```python
multi_step_prompt = """
I need to solve a complex problem. I'll break it into clear steps.

Problem: {problem}

Step 1: Understand the problem
- What are we trying to find?
- What information is given?
- What constraints exist?

Step 2: Identify the approach
- What method or formula applies?
- Are there multiple ways to solve this?

Step 3: Execute the solution
- Apply the chosen method
- Show all calculations

Step 4: Verify the answer
- Does the answer make sense?
- Can we check it another way?

Let me solve this:
"""
```

### Structured Reasoning Template

```python
def create_reasoning_prompt(problem, context=None):
    template = """
{context}

Problem: {problem}

Let me reason through this systematically:

1. **Understanding the Problem:**
   - Given information:
   - What we need to find:
   - Key constraints:

2. **Approach:**
   - Strategy:
   - Relevant concepts/formulas:

3. **Solution:**
   [Detailed step-by-step work]

4. **Answer:**
   [Final answer with units if applicable]

5. **Verification:**
   - Reasonableness check:
   - Alternative verification:
"""
    return template.format(
        context=context if context else "Solve the following problem.",
        problem=problem
    )
```

## Self-Consistency

### Multiple Reasoning Paths

```python
def self_consistency(prompt, model, n_samples=5):
    """
    Generate multiple reasoning chains and take majority vote.
    """
    answers = []

    for _ in range(n_samples):
        response = model(prompt, temperature=0.7)
        # Extract final answer
        answer = extract_answer(response)
        answers.append(answer)

    # Majority vote
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0]

    return {
        "final_answer": most_common[0],
        "confidence": most_common[1] / n_samples,
        "all_answers": answers
    }
```

### Example

```python
# Run multiple times with temperature > 0
responses = [
    "23 × 17 = 23 × 10 + 23 × 7 = 230 + 161 = 391",
    "23 × 17 = 20 × 17 + 3 × 17 = 340 + 51 = 391",
    "23 × 17 = 25 × 17 - 2 × 17 = 425 - 34 = 391"
]
# All paths lead to 391 → high confidence
```

## Tree of Thoughts

### Exploring Multiple Branches

```python
def tree_of_thoughts(problem, model, breadth=3, depth=3):
    """
    Explore multiple reasoning paths in a tree structure.
    """
    # Initial thoughts
    initial_prompt = f"""
    Problem: {problem}

    Generate {breadth} different initial approaches to solving this:
    """
    initial_thoughts = model(initial_prompt)
    thoughts = parse_thoughts(initial_thoughts)

    solutions = []

    for thought in thoughts:
        # Develop each thought
        current = thought
        for d in range(depth):
            develop_prompt = f"""
            Problem: {problem}
            Current approach: {current}

            Continue this reasoning with the next step:
            """
            current = model(develop_prompt)

            # Evaluate progress
            eval_prompt = f"""
            Problem: {problem}
            Solution so far: {current}

            Rate this approach (1-10) and explain:
            """
            evaluation = model(eval_prompt)

            # Prune low-rated branches
            if get_rating(evaluation) < 5:
                break

        solutions.append(current)

    # Return best solution
    return select_best(solutions)
```

## Application Examples

### Math Word Problems

```python
math_cot = """
Q: A rectangular garden is 3 times as long as it is wide.
If the perimeter is 64 meters, what are the dimensions?

Let me solve this step by step:

1. Define variables:
   - Let width = w meters
   - Length = 3w meters (3 times the width)

2. Set up equation using perimeter formula:
   - Perimeter = 2(length + width)
   - 64 = 2(3w + w)
   - 64 = 2(4w)
   - 64 = 8w

3. Solve for w:
   - w = 64 / 8 = 8 meters

4. Find length:
   - length = 3w = 3 × 8 = 24 meters

5. Verify:
   - Perimeter = 2(24 + 8) = 2(32) = 64 ✓

Answer: The garden is 8 meters wide and 24 meters long.
"""
```

### Logical Reasoning

```python
logic_cot = """
Q: All A are B. All B are C. Some C are D.
Is it true that some A are D?

Let me reason through this:

1. Analyze each statement:
   - "All A are B" means A ⊆ B
   - "All B are C" means B ⊆ C
   - "Some C are D" means C ∩ D ≠ ∅

2. Derive implications:
   - From A ⊆ B and B ⊆ C, we get A ⊆ C
   - This means all A are also C

3. Consider the question:
   - We need to determine if A ∩ D ≠ ∅
   - We know some C are D, and all A are C
   - But the D elements might be C elements that aren't A

4. Conclusion:
   - We CANNOT conclude that some A are D
   - The overlap between C and D might not include any A elements

Answer: No, we cannot determine that some A are D. It's possible but not certain.
"""
```

## Best Practices

```python
cot_best_practices = [
    "Start with 'Let's think step by step' for simple problems",
    "Provide detailed examples for complex reasoning",
    "Show intermediate calculations, not just final answers",
    "Include verification steps",
    "Use clear formatting with numbered steps",
    "Encourage explicit statement of assumptions",
    "Use self-consistency for higher confidence"
]
```

## Exercises

1. Apply CoT to a multi-step word problem
2. Implement self-consistency with majority voting
3. Create a Tree of Thoughts solver for a puzzle
4. Compare CoT vs direct prompting on reasoning tasks

## Additional Resources

- [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Paper](https://arxiv.org/abs/2203.11171)
- [Tree of Thoughts Paper](https://arxiv.org/abs/2305.10601)
