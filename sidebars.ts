import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  mainSidebar: [
    {
      type: 'doc',
      id: 'intro/intro',
      label: 'Introduction to Physical AI',
    },

    {
      type: 'category',
      label: 'ROS 2 Fundamentals',
      collapsed: false,
      items: [
        'module-1-ros2/intro',
        'module-1-ros2/nodes-topics-services',
        'module-1-ros2/middleware-real-time-control',
        'module-1-ros2/python-ros-integration',
        'module-1-ros2/urdf-humanoid-robots',
      ],
    },

    {
      type: 'category',
      label: 'Simulation Environments',
      collapsed: false,
      items: [
        'module-2-simulation/intro',
        'module-2-simulation/physics-simulation',
        'module-2-simulation/sensor-simulation',
        'module-2-simulation/unity-rendering',
      ],
    },

    {
      type: 'category',
      label: 'NVIDIA Isaac',
      collapsed: false,
      items: [
        'module-3-nvidia-isaac/intro',
        'module-3-nvidia-isaac/isaac-sim',
        'module-3-nvidia-isaac/navigation-planning',
        'module-3-nvidia-isaac/synthetic-data-perception',
        'module-3-nvidia-isaac/vla-implementation-patterns',
        'module-3-nvidia-isaac/summary-next-steps',
      ],
    },

    {
      type: 'category',
      label: 'Vision-Language-Action (VLA)',
      collapsed: false,
      items: [
        'module-4-vla/intro',
        'module-4-vla/vla-models-architectures',
        'module-4-vla/llm-task-planning',
        'module-4-vla/vla-integration',
        'module-4-vla/whisper-voice-command',
      ],
    },

    {
      type: 'category',
      label: 'Capstone Project',
      collapsed: false,
      items: [
        'capstone-project/physical-ai-capstone',
        'capstone-project/system-architecture',
        'capstone-project/implementation-guide',
        'capstone-project/final-summary',
      ],
    },

    {
      type: 'category',
      label: 'Weekly Breakdown',
      collapsed: true,
      items: [
        'weekly-breakdown/week-1',
        'weekly-breakdown/week-2',
        'weekly-breakdown/week-3',
        'weekly-breakdown/week-4',
        'weekly-breakdown/week-5',
      ],
    },

    'course-completion',
    'conclusion/conclusion',
    'resources',
  ],
};

export default sidebars;
