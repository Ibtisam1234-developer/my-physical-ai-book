---
name: docusaurus-configuration
description: Configuration patterns and best practices for Docusaurus documentation sites using TypeScript, MDX v2, custom theming, and component swizzling. Use when setting up or modifying Docusaurus projects, implementing custom components, configuring navigation, or integrating authentication and API proxies.
tags: [docusaurus, typescript, mdx, react, documentation, theming]
---

# Docusaurus Configuration Best Practices

## Project Foundation

### Classic Template with TypeScript
Initialize Docusaurus with TypeScript support:

```bash
npx create-docusaurus@latest my-website classic --typescript
```

**Key Files**:
- `docusaurus.config.ts` - Main configuration file
- `tsconfig.json` - TypeScript compiler options
- `src/` - Custom React components and pages
- `docs/` - MDX documentation files
- `blog/` - Optional blog posts

## Configuration Architecture

### docusaurus.config.ts Structure
```typescript
import {Config} from '@docusaurus/types';
import {themes as prismThemes} from 'prism-react-renderer';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Learn about embodied intelligence and robotics',
  favicon: 'img/favicon.ico',

  url: 'https://your-domain.com',
  baseUrl: '/',

  organizationName: 'your-org',
  projectName: 'your-project',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/your-org/your-repo/tree/main/',
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/your-org/your-repo/tree/main/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    // Custom theme configuration (see below)
  },
};

export default config;
```

## Custom ThemeConfig

### Complete Theme Configuration
```typescript
themeConfig: {
  // Navbar configuration
  navbar: {
    title: 'Physical AI Book',
    logo: {
      alt: 'Physical AI Logo',
      src: 'img/logo.svg',
    },
    items: [
      {
        type: 'docSidebar',
        sidebarId: 'tutorialSidebar',
        position: 'left',
        label: 'Tutorial',
      },
      {
        to: '/docs/robotics/fundamentals',
        label: 'Robotics',
        position: 'left',
      },
      {
        to: '/docs/physical-ai/overview',
        label: 'Physical AI',
        position: 'left',
      },
      {
        to: '/blog',
        label: 'Blog',
        position: 'left'
      },
      {
        href: 'https://github.com/your-org/your-repo',
        label: 'GitHub',
        position: 'right',
      },
    ],
  },

  // Footer configuration
  footer: {
    style: 'dark',
    links: [
      {
        title: 'Docs',
        items: [
          {
            label: 'Tutorial',
            to: '/docs/intro',
          },
          {
            label: 'Robotics Fundamentals',
            to: '/docs/robotics/fundamentals',
          },
        ],
      },
      {
        title: 'Community',
        items: [
          {
            label: 'Stack Overflow',
            href: 'https://stackoverflow.com/questions/tagged/docusaurus',
          },
          {
            label: 'Discord',
            href: 'https://discordapp.com/invite/docusaurus',
          },
        ],
      },
      {
        title: 'More',
        items: [
          {
            label: 'Blog',
            to: '/blog',
          },
          {
            label: 'GitHub',
            href: 'https://github.com/your-org/your-repo',
          },
        ],
      },
    ],
    copyright: `Copyright © ${new Date().getFullYear()} Your Project. Built with Docusaurus.`,
  },

  // Code highlighting theme
  prism: {
    theme: prismThemes.github,
    darkTheme: prismThemes.dracula,
    additionalLanguages: ['python', 'bash', 'typescript', 'json'],
  },

  // Color mode configuration
  colorMode: {
    defaultMode: 'light',
    disableSwitch: false,
    respectPrefersColorScheme: true,
  },
} satisfies Preset.ThemeConfig,
```

## MDX v2 Configuration

### Interactive Code Blocks
Use MDX v2 for rich, interactive content:

**Basic Code Block**:
````mdx
```python
import numpy as np

def forward_kinematics(joint_angles):
    """Calculate end-effector position from joint angles"""
    return transformation_matrix @ joint_angles
```
````

**Using @theme/CodeBlock**:
```mdx
import CodeBlock from '@theme/CodeBlock';

<CodeBlock language="python" title="Forward Kinematics Example">
{`import numpy as np

def forward_kinematics(joint_angles):
    """Calculate end-effector position from joint angles"""
    return transformation_matrix @ joint_angles
`}
</CodeBlock>
```

**Interactive Tabs with Code**:
```mdx
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import CodeBlock from '@theme/CodeBlock';

<Tabs>
  <TabItem value="python" label="Python" default>
    <CodeBlock language="python">
      {`# Python implementation`}
    </CodeBlock>
  </TabItem>
  <TabItem value="cpp" label="C++">
    <CodeBlock language="cpp">
      {`// C++ implementation`}
    </CodeBlock>
  </TabItem>
</Tabs>
```

## Component Swizzling

### Swizzling Root for Auth Provider
Swizzle the Root component to wrap the entire app:

```bash
npm run swizzle @docusaurus/theme-classic Root -- --eject
```

**src/theme/Root.tsx**:
```typescript
import React from 'react';
import {AuthProvider} from '@/lib/auth';

export default function Root({children}) {
  return (
    <AuthProvider>
      {children}
    </AuthProvider>
  );
}
```

### Common Swizzled Components
1. **Root** - Global providers (auth, theme, state)
2. **Navbar** - Custom navigation logic
3. **Footer** - Custom footer content
4. **DocItem** - Custom doc page layout
5. **CodeBlock** - Enhanced code highlighting

### Swizzling Commands
```bash
# Safe wrapping (recommended)
npm run swizzle @docusaurus/theme-classic Root -- --wrap

# Full ejection (use sparingly)
npm run swizzle @docusaurus/theme-classic Root -- --eject

# List swizzleable components
npm run swizzle @docusaurus/theme-classic -- --list
```

## Custom Pages

### Creating Custom Pages
**src/pages/chatbot.tsx**:
```typescript
import React from 'react';
import Layout from '@theme/Layout';
import ChatbotComponent from '@site/src/components/Chatbot';

export default function ChatbotPage() {
  return (
    <Layout
      title="AI Chatbot"
      description="Ask questions about Physical AI and Robotics"
    >
      <main className="container margin-vert--lg">
        <h1>Physical AI Chatbot</h1>
        <ChatbotComponent />
      </main>
    </Layout>
  );
}
```

## Sidebar Configuration for Robotics Topics

### sidebars.ts Structure
```typescript
import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['intro', 'installation', 'quick-start'],
    },
    {
      type: 'category',
      label: 'Physical AI Fundamentals',
      collapsed: false,
      items: [
        'physical-ai/overview',
        'physical-ai/perception',
        'physical-ai/actuation',
        'physical-ai/learning',
      ],
    },
    {
      type: 'category',
      label: 'Humanoid Robotics',
      collapsed: false,
      items: [
        'robotics/fundamentals',
        {
          type: 'category',
          label: 'Locomotion',
          items: [
            'robotics/locomotion/bipedal-walking',
            'robotics/locomotion/balance-control',
            'robotics/locomotion/gait-generation',
          ],
        },
        {
          type: 'category',
          label: 'Manipulation',
          items: [
            'robotics/manipulation/grasp-planning',
            'robotics/manipulation/dexterous-hands',
          ],
        },
        {
          type: 'category',
          label: 'Sensor Fusion',
          items: [
            'robotics/sensors/imu',
            'robotics/sensors/cameras',
            'robotics/sensors/lidar',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Control Systems',
      items: [
        'control/whole-body-control',
        'control/inverse-kinematics',
        'control/model-predictive-control',
      ],
    },
  ],
};

export default sidebars;
```

## Development API Proxy

### package.json Proxy Configuration
For proxying API requests during development:

```json
{
  "name": "physical-ai-docs",
  "version": "0.0.0",
  "private": true,
  "scripts": {
    "docusaurus": "docusaurus",
    "start": "docusaurus start",
    "build": "docusaurus build",
    "swizzle": "docusaurus swizzle"
  },
  "proxy": "http://localhost:8000"
}
```

### API Client Example
**src/lib/api.ts**:
```typescript
const API_BASE_URL = process.env.NODE_ENV === 'production'
  ? 'https://api.your-domain.com'
  : '/api';  // Proxied to localhost:8000 in development

export async function searchDocs(query: string) {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query}),
  });

  if (!response.ok) {
    throw new Error('Search failed');
  }

  return response.json();
}
```

## Best Practices

- [ ] Use TypeScript for all custom components
- [ ] Configure MDX v2 for interactive documentation
- [ ] Organize sidebar by logical topic categories (robotics topics)
- [ ] Swizzle Root component for global providers (auth)
- [ ] Use `@theme/CodeBlock` for syntax-highlighted examples
- [ ] Set up API proxy in development via package.json
- [ ] Customize themeConfig for navbar links and branding
- [ ] Create custom pages for specialized features

---

**Usage Note**: Apply these patterns when building Docusaurus-based documentation sites with TypeScript, custom theming, authentication, and API integration.
