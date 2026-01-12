// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    'syllabus',
    {
      type: 'category',
      label: '01 - Data Science Essentials',
      items: [
        'data_science_essentials/README',
        'data_science_essentials/python_for_data_science',
        'data_science_essentials/statistical_foundations',
        'data_science_essentials/data_preprocessing',
        'data_science_essentials/exploratory_data_analysis',
        'data_science_essentials/machine_learning_basics',
      ],
    },
    {
      type: 'category',
      label: '02 - PyTorch Basics',
      items: [
        'torch_basics/README',
        'torch_basics/tensors',
        'torch_basics/neural_networks',
        'torch_basics/training_loops',
        'torch_basics/data_loading',
      ],
    },
    {
      type: 'category',
      label: '03 - DeepSpeed Basics',
      items: [
        'deepspeed_basics/README',
        'deepspeed_basics/introduction_to_deepspeed',
        'deepspeed_basics/zero_optimization',
        'deepspeed_basics/distributed_training',
      ],
    },
    {
      type: 'category',
      label: '04 - Gradio Apps',
      items: [
        'gradio_apps/README',
        'gradio_apps/introduction_to_gradio',
        'gradio_apps/input_output_components',
        'gradio_apps/building_ml_interfaces',
      ],
    },
    {
      type: 'category',
      label: '05 - Streamlit Apps',
      items: [
        'streamlit_apps/README',
        'streamlit_apps/introduction_to_streamlit',
        'streamlit_apps/data_visualization',
        'streamlit_apps/ml_dashboards',
      ],
    },
    {
      type: 'category',
      label: '06 - LLM Fundamentals',
      items: [
        'llm_fundamentals/README',
        'llm_fundamentals/introduction_to_llms',
        'llm_fundamentals/api_integration',
        'llm_fundamentals/tokenization_and_context',
        'llm_fundamentals/inference_parameters',
        'llm_fundamentals/model_selection',
        'llm_fundamentals/safety_alignment_ethics',
      ],
    },
    {
      type: 'category',
      label: '07 - Prompt Engineering',
      items: [
        'prompt_engineering/README',
        'prompt_engineering/prompt_fundamentals',
        'prompt_engineering/few_shot_learning',
        'prompt_engineering/chain_of_thought',
      ],
    },
    {
      type: 'category',
      label: '08 - RAG Systems',
      items: [
        'rag_systems/README',
        'rag_systems/introduction_to_rag',
        'rag_systems/embeddings_and_vectors',
        'rag_systems/retrieval_techniques',
      ],
    },
    {
      type: 'category',
      label: '09 - Fine-Tuning',
      items: [
        'fine_tuning/README',
        'fine_tuning/introduction_to_finetuning',
        'fine_tuning/lora_and_peft',
        'fine_tuning/training_data_preparation',
      ],
    },
    {
      type: 'category',
      label: '10 - AI Agents',
      items: [
        'ai_agents/README',
        'ai_agents/introduction_to_agents',
        'ai_agents/tool_use_and_function_calling',
        'ai_agents/multi_agent_systems',
      ],
    },
    {
      type: 'category',
      label: 'Appendix',
      items: [
        'appendix_submission/README',
        'appendix_project_guidelines/README',
      ],
    },
  ],
};

export default sidebars;
