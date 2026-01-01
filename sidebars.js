// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: '01 - Data Science Essentials',
      items: [
        '01_data_science_essentials/README',
        '01_data_science_essentials/python_for_data_science',
        '01_data_science_essentials/statistical_foundations',
        '01_data_science_essentials/data_preprocessing',
        '01_data_science_essentials/exploratory_data_analysis',
        '01_data_science_essentials/machine_learning_basics',
      ],
    },
    {
      type: 'category',
      label: '02 - PyTorch Basics',
      items: [
        '02_torch_basics/README',
        '02_torch_basics/tensors',
        '02_torch_basics/neural_networks',
        '02_torch_basics/training_loops',
        '02_torch_basics/data_loading',
      ],
    },
    {
      type: 'category',
      label: '03 - DeepSpeed Basics',
      items: [
        '03_deepspeed_basics/README',
        '03_deepspeed_basics/introduction_to_deepspeed',
        '03_deepspeed_basics/zero_optimization',
        '03_deepspeed_basics/distributed_training',
      ],
    },
    {
      type: 'category',
      label: '04 - Gradio Apps',
      items: [
        '04_gradio_apps/README',
        '04_gradio_apps/introduction_to_gradio',
        '04_gradio_apps/input_output_components',
        '04_gradio_apps/building_ml_interfaces',
      ],
    },
    {
      type: 'category',
      label: '05 - Streamlit Apps',
      items: [
        '05_streamlit_apps/README',
        '05_streamlit_apps/introduction_to_streamlit',
        '05_streamlit_apps/data_visualization',
        '05_streamlit_apps/ml_dashboards',
      ],
    },
    {
      type: 'category',
      label: '06 - LLM Fundamentals',
      items: [
        '06_llm_fundamentals/README',
        '06_llm_fundamentals/introduction_to_llms',
        '06_llm_fundamentals/api_integration',
        '06_llm_fundamentals/tokenization_and_context',
      ],
    },
    {
      type: 'category',
      label: '07 - Prompt Engineering',
      items: [
        '07_prompt_engineering/README',
        '07_prompt_engineering/prompt_fundamentals',
        '07_prompt_engineering/few_shot_learning',
        '07_prompt_engineering/chain_of_thought',
      ],
    },
    {
      type: 'category',
      label: '08 - RAG Systems',
      items: [
        '08_rag_systems/README',
        '08_rag_systems/introduction_to_rag',
        '08_rag_systems/embeddings_and_vectors',
        '08_rag_systems/retrieval_techniques',
      ],
    },
    {
      type: 'category',
      label: '09 - Fine-Tuning',
      items: [
        '09_fine_tuning/README',
        '09_fine_tuning/introduction_to_finetuning',
        '09_fine_tuning/lora_and_peft',
        '09_fine_tuning/training_data_preparation',
      ],
    },
    {
      type: 'category',
      label: '10 - AI Agents',
      items: [
        '10_ai_agents/README',
        '10_ai_agents/introduction_to_agents',
        '10_ai_agents/tool_use_and_function_calling',
        '10_ai_agents/multi_agent_systems',
      ],
    },
  ],
};

export default sidebars;
