# Multi-Agent Systems

## Overview

Multi-agent systems involve multiple AI agents working together to accomplish complex tasks. By specializing agents for different roles and enabling them to collaborate, these systems can tackle problems too complex for a single agent.

## Why Multi-Agent Systems?

### Benefits

```python
benefits = {
    "specialization": "Each agent optimized for specific task",
    "parallelization": "Multiple agents work simultaneously",
    "robustness": "System continues if one agent fails",
    "complexity_handling": "Break down complex problems",
    "diverse_perspectives": "Different approaches to same problem"
}
```

### When to Use Multi-Agent

```python
use_cases = [
    "Complex tasks requiring multiple skills",
    "Simulation of organizations/teams",
    "Debate and consensus building",
    "Software development workflows",
    "Research and analysis pipelines"
]
```

## Agent Communication Patterns

### Hierarchical

```python
hierarchical_pattern = """
Manager Agent
    ├── Worker Agent 1
    ├── Worker Agent 2
    └── Worker Agent 3

- Manager receives task, delegates to workers
- Workers report back to manager
- Manager synthesizes results
"""

class ManagerAgent:
    def __init__(self, workers):
        self.workers = workers

    def delegate_task(self, task):
        # Analyze task and assign to appropriate workers
        assignments = self.analyze_and_assign(task)

        results = {}
        for worker_id, subtask in assignments.items():
            results[worker_id] = self.workers[worker_id].execute(subtask)

        # Synthesize results
        return self.synthesize(results)
```

### Peer-to-Peer

```python
peer_pattern = """
Agent A ←→ Agent B
    ↕           ↕
Agent C ←→ Agent D

- All agents can communicate with each other
- No central coordinator
- Agents negotiate and collaborate directly
"""

class PeerAgent:
    def __init__(self, agent_id, peers):
        self.id = agent_id
        self.peers = peers
        self.inbox = []

    def send_message(self, recipient_id, message):
        self.peers[recipient_id].receive_message(self.id, message)

    def receive_message(self, sender_id, message):
        self.inbox.append({"from": sender_id, "content": message})

    def broadcast(self, message):
        for peer_id in self.peers:
            self.send_message(peer_id, message)
```

### Pipeline

```python
pipeline_pattern = """
Agent A → Agent B → Agent C → Agent D → Output

- Each agent processes and passes to next
- Specialized stages
- Linear workflow
"""

class PipelineAgent:
    def __init__(self, agent, next_agent=None):
        self.agent = agent
        self.next_agent = next_agent

    def process(self, input_data):
        result = self.agent.execute(input_data)

        if self.next_agent:
            return self.next_agent.process(result)

        return result
```

## Role-Based Agent Teams

### Defining Roles

```python
team_roles = {
    "researcher": {
        "description": "Gathers and analyzes information",
        "tools": ["web_search", "read_document", "summarize"],
        "system_prompt": "You are a research specialist..."
    },
    "writer": {
        "description": "Creates written content",
        "tools": ["write_document", "edit_text"],
        "system_prompt": "You are a professional writer..."
    },
    "critic": {
        "description": "Reviews and provides feedback",
        "tools": ["analyze_text", "provide_feedback"],
        "system_prompt": "You are a critical reviewer..."
    },
    "coder": {
        "description": "Writes and debugs code",
        "tools": ["write_code", "execute_code", "debug"],
        "system_prompt": "You are an expert programmer..."
    }
}
```

### Team Implementation

```python
class AgentTeam:
    def __init__(self, roles_config):
        self.agents = {}
        for role, config in roles_config.items():
            self.agents[role] = self.create_agent(role, config)

    def create_agent(self, role, config):
        return Agent(
            name=role,
            system_prompt=config["system_prompt"],
            tools=config["tools"]
        )

    def collaborate(self, task):
        """Execute task through team collaboration"""
        # Phase 1: Research
        research = self.agents["researcher"].execute(
            f"Research this topic: {task}"
        )

        # Phase 2: Write draft
        draft = self.agents["writer"].execute(
            f"Write about: {task}\nResearch: {research}"
        )

        # Phase 3: Critical review
        feedback = self.agents["critic"].execute(
            f"Review this draft: {draft}"
        )

        # Phase 4: Revise
        final = self.agents["writer"].execute(
            f"Revise draft based on feedback:\nDraft: {draft}\nFeedback: {feedback}"
        )

        return final
```

## Multi-Agent Frameworks

### AutoGen Example

```python
from autogen import AssistantAgent, UserProxyAgent

# Create agents
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate fibonacci numbers"
)
```

### CrewAI Example

```python
from crewai import Agent, Task, Crew

# Define agents
researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert researcher with attention to detail",
    tools=[search_tool, scrape_tool]
)

writer = Agent(
    role="Writer",
    goal="Create engaging content",
    backstory="Skilled content writer",
    tools=[write_tool]
)

# Define tasks
research_task = Task(
    description="Research the topic: {topic}",
    agent=researcher
)

write_task = Task(
    description="Write an article based on research",
    agent=writer
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True
)

# Execute
result = crew.kickoff(inputs={"topic": "AI agents"})
```

## Debate and Consensus

### Multi-Agent Debate

```python
class DebateSystem:
    def __init__(self, agents, moderator):
        self.agents = agents
        self.moderator = moderator

    def debate(self, topic, rounds=3):
        positions = {}
        history = []

        # Initial positions
        for agent in self.agents:
            positions[agent.name] = agent.take_position(topic)

        # Debate rounds
        for round_num in range(rounds):
            for agent in self.agents:
                # Agent responds to other positions
                other_positions = {k: v for k, v in positions.items()
                                 if k != agent.name}
                response = agent.respond(topic, other_positions, history)
                history.append({
                    "agent": agent.name,
                    "round": round_num,
                    "response": response
                })
                positions[agent.name] = response

        # Moderator synthesizes
        conclusion = self.moderator.synthesize(topic, history)
        return conclusion
```

### Consensus Building

```python
class ConsensusSystem:
    def __init__(self, agents):
        self.agents = agents

    def reach_consensus(self, question, max_rounds=5):
        """Iterate until agents agree or max rounds reached"""
        proposals = {}
        round_num = 0

        while round_num < max_rounds:
            # Each agent proposes
            for agent in self.agents:
                proposals[agent.name] = agent.propose(
                    question,
                    other_proposals=proposals
                )

            # Check for consensus
            if self.check_consensus(proposals):
                return self.finalize_consensus(proposals)

            round_num += 1

        return self.voting_fallback(proposals)

    def check_consensus(self, proposals):
        """Check if proposals are sufficiently similar"""
        # Implement similarity check
        pass
```

## Software Development Team

```python
class DevTeam:
    def __init__(self):
        self.pm = Agent("Product Manager", pm_prompt)
        self.architect = Agent("Architect", architect_prompt)
        self.developer = Agent("Developer", developer_prompt)
        self.reviewer = Agent("Code Reviewer", reviewer_prompt)
        self.tester = Agent("QA Tester", tester_prompt)

    def develop_feature(self, requirement):
        # PM creates specification
        spec = self.pm.execute(f"Create spec for: {requirement}")

        # Architect designs solution
        design = self.architect.execute(f"Design solution for: {spec}")

        # Developer implements
        code = self.developer.execute(f"Implement: {design}")

        # Review loop
        while True:
            review = self.reviewer.execute(f"Review code: {code}")

            if "approved" in review.lower():
                break

            code = self.developer.execute(
                f"Fix issues: {review}\nOriginal code: {code}"
            )

        # Testing
        test_results = self.tester.execute(f"Test: {code}")

        return {
            "spec": spec,
            "design": design,
            "code": code,
            "review": review,
            "tests": test_results
        }
```

## Challenges and Solutions

### Coordination

```python
challenges = {
    "message_passing": "Ensure reliable agent communication",
    "shared_state": "Manage shared memory/context",
    "deadlock": "Prevent agents waiting on each other",
    "load_balancing": "Distribute work effectively",
    "error_propagation": "Handle failures gracefully"
}

solutions = {
    "message_queue": "Use async message queues",
    "central_state": "Shared database or memory store",
    "timeouts": "Add timeouts to all operations",
    "orchestrator": "Central coordinator for task assignment",
    "checkpointing": "Save state for recovery"
}
```

## Exercises

1. Build a two-agent debate system
2. Create a software development team simulation
3. Implement a peer-to-peer negotiation system
4. Build a hierarchical research team

## Additional Resources

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [Multi-Agent Papers](https://arxiv.org/search/?query=multi-agent+LLM)
