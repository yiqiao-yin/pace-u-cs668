---
sidebar_position: 1
slug: /intro
---

# CS668: Generative AI Course

Welcome to the comprehensive Generative AI and Machine Learning course! This course covers everything from data science fundamentals to advanced AI agent development.

## Course Overview

This course is designed to take you from foundational concepts to cutting-edge AI applications. Each module builds upon the previous ones, creating a comprehensive learning path.

## Course Modules

### Foundation Modules

| Module | Description | Prerequisites |
|--------|-------------|---------------|
| **01 - Data Science Essentials** | Python, statistics, data preprocessing, and ML basics | Python basics |
| **02 - PyTorch Basics** | Tensors, neural networks, training loops | Module 01 |
| **03 - DeepSpeed Basics** | Distributed training and memory optimization | Module 02 |

### Application Modules

| Module | Description | Prerequisites |
|--------|-------------|---------------|
| **04 - Gradio Apps** | Build interactive ML interfaces | Module 02 |
| **05 - Streamlit Apps** | Create data science dashboards | Module 01 |

### Generative AI Modules

| Module | Description | Prerequisites |
|--------|-------------|---------------|
| **06 - LLM Fundamentals** | Understanding large language models | Module 01 |
| **07 - Prompt Engineering** | Crafting effective prompts | Module 06 |
| **08 - RAG Systems** | Retrieval-augmented generation | Modules 06, 07 |
| **09 - Fine-Tuning** | Customizing LLMs for specific tasks | Modules 02, 06 |
| **10 - AI Agents** | Building autonomous AI systems | Modules 06, 07, 08 |

## Getting Started

### Prerequisites

Before starting this course, you should have:

- Basic Python programming knowledge
- Familiarity with command line / terminal
- A computer with Python 3.8+ installed
- (Optional) Access to a GPU for deep learning modules

### Environment Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install numpy pandas matplotlib scikit-learn
pip install torch torchvision
pip install transformers datasets
pip install gradio streamlit
pip install openai anthropic
```

## How to Use This Course

1. **Follow the module order** - Each module builds on previous concepts
2. **Complete the exercises** - Hands-on practice is essential
3. **Experiment** - Try modifying the examples to deepen understanding
4. **Build projects** - Apply what you learn to real-world problems

## Learning Outcomes

By the end of this course, you will be able to:

- Build and train deep learning models with PyTorch
- Create interactive web applications for ML models
- Work with large language models through APIs
- Design effective prompts for various tasks
- Build RAG systems for knowledge-grounded AI
- Fine-tune models for specific domains
- Create autonomous AI agents

## Support and Resources

- **GitHub Repository**: Access all code examples and materials
- **Video Tutorials**: Supplementary video content for each module
- **Community**: Connect with other learners

Let's begin your journey into Generative AI!

---

**Next Step**: Start with [Module 01 - Data Science Essentials](/docs/data_science_essentials/)
