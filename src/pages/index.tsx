import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import ChatBot from '@site/src/components/ChatBot/ChatBot';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <div className="text--center margin-bottom--lg">
              <img
                src="/img/book.png"
                alt="Author Profile"
                className={styles.authorImage}
                style={{ width: '80px', height: '80px', borderRadius: '50%', margin: '0 auto 1rem auto', display: 'block' }}
              />
              <p className="hero__subtitle text--center">Created by Ibtisam - AI Engineer & Python Developer</p>
            </div>
            <Heading as="h1" className="hero__title text--center">
              {siteConfig.title}
            </Heading>
            <p className="hero__subtitle text--center">{siteConfig.tagline}</p>
            <div className={styles.buttons}>
              <Link
                className="button button--secondary button--lg margin-right--md"
                to="/docs/intro">
                Start Learning - 20 Weeks ⏱️
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/docs/intro">
                View Curriculum
              </Link>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function TechnologyStack() {
  return (
    <section className={styles.technologyStack}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <Heading as="h2" className="text--center margin-bottom--lg">
              Comprehensive Technology Stack
            </Heading>
          </div>
        </div>
        <div className="row">
          <div className="col col--4 margin-bottom--lg">
            <div className={styles.techCard}>
              <h3>Backend Infrastructure</h3>
              <ul>
                <li>ROS 2 Humble Hawksbill</li>
                <li>FastAPI</li>
                <li>SQLAlchemy</li>
                <li>Pydantic</li>
                <li>Qdrant</li>
                <li>NVIDIA Isaac Sim</li>
              </ul>
            </div>
          </div>
          <div className="col col--4 margin-bottom--lg">
            <div className={styles.techCard}>
              <h3>AI and Machine Learning</h3>
              <ul>
                <li>NVIDIA Isaac ROS</li>
                <li>Gemini API</li>
                <li>PyTorch</li>
                <li>Transformers</li>
                <li>TensorRT</li>
                <li>GPU Acceleration</li>
              </ul>
            </div>
          </div>
          <div className="col col--4 margin-bottom--lg">
            <div className={styles.techCard}>
              <h3>Frontend & Visualization</h3>
              <ul>
                <li>Docusaurus</li>
                <li>React 18</li>
                <li>TypeScript</li>
                <li>Three.js</li>
                <li>WebGL</li>
                <li>Modern UI/UX</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col col--4 margin-bottom--lg">
            <div className={styles.techCard}>
              <h3>Simulation & Physics</h3>
              <ul>
                <li>PhysX</li>
                <li>USD (Universal Scene Description)</li>
                <li>Omniverse</li>
                <li>Domain Randomization</li>
                <li>Gazebo</li>
                <li>Unity ML-Agents</li>
              </ul>
            </div>
          </div>
          <div className="col col--4 margin-bottom--lg">
            <div className={styles.techCard}>
              <h3>Core Modules</h3>
              <ul>
                <li>ROS 2 Fundamentals</li>
                <li>Robot Simulation</li>
                <li>NVIDIA Isaac Platform</li>
                <li>Vision-Language-Action Systems</li>
                <li>Humanoid Control</li>
                <li>Complete Integration</li>
              </ul>
            </div>
          </div>
          <div className="col col--4 margin-bottom--lg">
            <div className={styles.techCard}>
              <h3>Learning Outcomes</h3>
              <ul>
                <li>ROS 2 Mastery</li>
                <li>Simulation Expertise</li>
                <li>Isaac Platform</li>
                <li>VLA Systems</li>
                <li>GPU Acceleration</li>
                <li>Humanoid Robotics</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function CourseOverview() {
  return (
    <section className={styles.courseOverview}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <Heading as="h2" className="text--center margin-bottom--lg">
              About This Course
            </Heading>
            <p className="text--center margin-bottom--lg">
              This comprehensive course provides end-to-end training in Physical AI and Humanoid Robotics using NVIDIA's Isaac platform.
              Students learn to build complete AI-powered humanoid robot systems that can understand natural language,
              perceive their environment, and execute complex physical actions.
            </p>
          </div>
        </div>
        <div className="row margin-top--lg">
          <div className="col col--4">
            <div className={styles.courseCard}>
              <div className={styles.courseIcon}>
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
              </div>
              <h3>Duration</h3>
              <p>20 weeks (5 months)</p>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.courseCard}>
              <div className={styles.courseIcon}>
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                </svg>
              </div>
              <h3>Modules</h3>
              <p>4 comprehensive modules</p>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.courseCard}>
              <div className={styles.courseIcon}>
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                  <line x1="8" y1="21" x2="16" y2="21"></line>
                  <line x1="12" y1="17" x2="12" y2="21"></line>
                </svg>
              </div>
              <h3>Projects</h3>
              <p>6 integrated projects</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="Complete Course in Embodied Artificial Intelligence - Learn ROS 2, NVIDIA Isaac, VLA Systems, and Humanoid Robotics">
      <HomepageHeader />
      <main>
        <CourseOverview />
        <TechnologyStack />
      </main>
    </Layout>
  );
}
