import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Start Learning
          </Link>
        </div>
      </div>
    </header>
  );
}

const modules = [
  {
    title: 'Data Science Essentials',
    description: 'Master the fundamentals of data science with Python, statistics, and machine learning basics.',
    link: '/docs/data_science_essentials/',
  },
  {
    title: 'PyTorch Basics',
    description: 'Learn to build and train neural networks using PyTorch.',
    link: '/docs/torch_basics/',
  },
  {
    title: 'DeepSpeed Basics',
    description: 'Scale your training with distributed computing and memory optimization.',
    link: '/docs/deepspeed_basics/',
  },
  {
    title: 'Gradio Apps',
    description: 'Create interactive web interfaces for your ML models.',
    link: '/docs/gradio_apps/',
  },
  {
    title: 'Streamlit Apps',
    description: 'Build data science dashboards and applications.',
    link: '/docs/streamlit_apps/',
  },
  {
    title: 'LLM Fundamentals',
    description: 'Understand large language models and how to use them.',
    link: '/docs/llm_fundamentals/',
  },
  {
    title: 'Prompt Engineering',
    description: 'Master the art of crafting effective prompts for LLMs.',
    link: '/docs/prompt_engineering/',
  },
  {
    title: 'RAG Systems',
    description: 'Build retrieval-augmented generation applications.',
    link: '/docs/rag_systems/',
  },
  {
    title: 'Fine-Tuning',
    description: 'Customize LLMs for your specific use cases.',
    link: '/docs/fine_tuning/',
  },
  {
    title: 'AI Agents',
    description: 'Create autonomous AI agents that can reason and act.',
    link: '/docs/ai_agents/',
  },
];

function ModuleCard({title, description, link}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="card margin-bottom--lg">
        <div className="card__header">
          <Heading as="h3">{title}</Heading>
        </div>
        <div className="card__body">
          <p>{description}</p>
        </div>
        <div className="card__footer">
          <Link className="button button--primary button--block" to={link}>
            Explore Module
          </Link>
        </div>
      </div>
    </div>
  );
}

function HomepageModules() {
  return (
    <section className={styles.features}>
      <div className="container">
        <Heading as="h2" className="text--center margin-bottom--lg">
          Course Modules
        </Heading>
        <div className="row">
          {modules.map((props, idx) => (
            <ModuleCard key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Comprehensive Generative AI and Machine Learning Course">
      <HomepageHeader />
      <main>
        <HomepageModules />
      </main>
    </Layout>
  );
}
