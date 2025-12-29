<!--
SYNC IMPACT REPORT
===================
Version Change: 1.1.0 → 1.2.0
Rationale: Added new principle VIII (Static Localization & Routing) establishing requirements for content consistency, auth-gated access, stateless navigation, and directionality for Urdu localization

Modified Principles: N/A
Added Sections:
  - Principle VIII: Static Localization & Routing (content consistency, auth-gated access, stateless navigation, directionality)

Removed Sections: N/A

Templates Requiring Updates:
  ✅ .specify/templates/plan-template.md - Constitution Check section aligns with all principles including new Static Localization & Routing
  ✅ .specify/templates/spec-template.md - Requirements align with UX, testing, personalization, and now localization principles
  ✅ .specify/templates/tasks-template.md - Task categorization reflects TDD, modularity, personalization, and localization principles

Follow-up TODOs: None
-->

# Physical AI & Humanoid Robotics Platform Constitution

## Core Principles

### I. Security (NON-NEGOTIABLE)

All authentication and data flows MUST follow OWASP guidelines. Secrets MUST never be exposed in code, logs, or client-side responses.

**Rationale**: Physical AI systems integrate with real-world robotics hardware and student data. Security breaches could compromise educational infrastructure, student privacy, and potentially physical safety in lab environments.

**Requirements**:
- Authentication endpoints must use industry-standard protocols (OAuth2, JWT)
- All secrets (API keys, database credentials, JWT secrets) stored in environment variables
- Passwords hashed with bcrypt (minimum cost factor 12)
- HTTPS/TLS required for all production traffic
- Regular security audits against OWASP Top 10
- Input validation and sanitization on all user inputs
- SQL injection prevention through parameterized queries
- XSS prevention through proper output encoding
- CSRF protection on state-changing operations

### II. Test-Driven Development (NON-NEGOTIABLE)

Every feature MUST be preceded by tests. Test coverage MUST exceed 80% for all production code.

**Rationale**: TDD ensures code correctness, facilitates refactoring, and documents expected behavior. Given the educational nature of this platform and its integration with physical robotics systems, reliability is paramount.

**Requirements**:
- **Backend/RAG**: pytest with pytest-asyncio, pytest-cov
- **Frontend**: Jest with React Testing Library
- Red-Green-Refactor cycle strictly enforced: Write failing test → Implement minimal code → Refactor
- Unit tests for all business logic
- Integration tests for API endpoints and database operations
- Mock external services (Gemini API, Qdrant) in tests
- Coverage threshold: 80% minimum (lines, branches, functions)
- Tests must run in CI/CD pipeline before deployment
- No code merged without passing tests

### III. User Experience

Frontend interfaces MUST be accessible, responsive, and interactive.

**Rationale**: Students and educators need intuitive, performant interfaces to interact with complex robotics concepts and simulations. Accessibility ensures inclusivity for learners with diverse needs.

**Requirements**:
- **Accessibility**: WCAG 2.1 AA compliance (semantic HTML, ARIA labels, keyboard navigation, screen reader support)
- **Responsiveness**: Mobile-first design, works on devices from 320px to 4K displays
- **Interactivity**: Real-time feedback for robotics simulations, streaming chat responses, loading states
- **Performance**: First Contentful Paint < 1.5s, Time to Interactive < 3.5s
- **Error Handling**: User-friendly error messages with actionable guidance
- **Documentation**: Interactive code examples with syntax highlighting and copy-to-clipboard
- **Dark Mode**: Respect user's preferred color scheme

### IV. Gemini Usage

All LLM and RAG pipelines MUST use Gemini models via OpenAI Agent SDK. Streaming support is REQUIRED for chat user experience.

**Rationale**: Gemini provides state-of-the-art embeddings and generation capabilities optimized for technical and educational content. Streaming ensures responsive chat interactions.

**Requirements**:
- **Embeddings**: `text-embedding-004` (768 dimensions) for all vector operations
- **Generation**: `gemini-2.0-flash-exp` for RAG responses
- **Streaming**: Implement Server-Sent Events (SSE) or WebSockets for chat responses
- **SDK**: Use OpenAI Agent SDK configured for Gemini endpoints
- **Rate Limiting**: Implement exponential backoff and request queuing
- **Error Handling**: Graceful degradation when API unavailable
- **Prompt Engineering**: System prompts must emphasize accuracy for Physical AI and robotics content
- **Citation**: RAG responses must cite source documents

### V. Modularity

Sub-agents (Project Manager, Frontend, Backend, RAG, Auth, Test, Deployment, Content) handle specialized tasks. No agent writes code outside its domain.

**Rationale**: Clear separation of concerns ensures expertise is applied correctly, reduces cognitive load, and prevents cross-domain errors (e.g., backend logic in frontend components).

**Agent Responsibilities**:
- **Project Manager**: Coordinates workflows, manages task dependencies, ensures constitution compliance
- **Frontend Specialist**: Docusaurus configuration, React components, UI/UX, client-side routing
- **Backend Specialist**: FastAPI endpoints, Pydantic models, async operations, business logic
- **RAG Specialist**: Document ingestion, embedding generation, Qdrant operations, prompt augmentation
- **Auth Specialist**: Authentication flows, JWT management, session handling, security best practices
- **Test Specialist**: Test generation, mocking, coverage analysis, CI/CD integration
- **Deployment Specialist**: Vercel/Railway configuration, environment variables, PR previews
- **Content Generation Specialist**: Educational content creation, robotics tutorials, curriculum design

**Enforcement**:
- Pull requests must be reviewed by appropriate specialist agent
- Cross-domain changes require multi-agent collaboration with explicit handoffs
- Code reviews verify domain boundaries are respected

### VI. Human Approval

Critical stages (production deployment, production data ingestion) REQUIRE explicit human approval.

**Rationale**: Automated systems must have human oversight for irreversible or high-impact operations. This prevents accidental data loss, service disruption, or deployment of untested code.

**Approval Gates**:
- **Production Deployment**: Human must review deployment checklist and approve promotion
- **Database Migrations**: Human must verify migration scripts before applying to production
- **Bulk Data Ingestion**: Human must approve document sets before indexing to production Qdrant
- **Secret Rotation**: Human must verify new secrets before updating production environment variables
- **Breaking Changes**: Human must review impact analysis and migration plan

**Process**:
- Agent proposes action with detailed summary and risk assessment
- Human reviews proposal and provides explicit approval/rejection
- Agent logs approval with timestamp and justification
- Automated actions proceed only after approval received

### VII. Identity & Personalization

User identity and personalization capabilities MUST use stateless authentication with mandatory profile fields and strict privacy controls.

**Rationale**: Physical AI education requires personalized learning paths based on student background (software/hardware experience). Stateless JWT authentication enables cross-service communication between Node/Python backends while maintaining security. Profile data must be captured at signup to enable immediate personalization without additional friction.

**Requirements**:

**Stateless Authentication**:
- Use RS256 JWT tokens for all cross-service communication
- Node.js server (Better Auth or custom) signs tokens with private key
- Python FastAPI backend verifies tokens using public key/JWKS endpoint
- Token payload MUST include: user_id, email, software_background, hardware_background
- Token expiration: 1 hour for access tokens, 30 days for refresh tokens
- Implement token refresh mechanism without requiring re-authentication
- No server-side session storage (enables horizontal scaling)

**Schema Strictness**:
- `software_background` and `hardware_background` fields are MANDATORY at signup
- Fields stored in Neon PostgreSQL `users` table with NOT NULL constraints
- Validation rules:
  - `software_background`: enum (beginner, intermediate, advanced, expert)
  - `hardware_background`: enum (none, hobbyist, student, professional)
- Signup flow MUST NOT proceed without both fields completed
- Fields immutable after initial creation (require explicit "edit profile" action to change)

**Personalization Persistence**:
- Personalized content (curriculum recommendations, difficulty adjustments) generated on-demand using LLM
- Cache personalized content in Neon DB to avoid redundant LLM API costs
- Cache structure: `personalized_content` table with columns:
  - `user_id` (foreign key to users table)
  - `content_type` (enum: curriculum_path, difficulty_level, recommended_resources)
  - `content_payload` (JSONB)
  - `generated_at` (timestamp)
  - `expires_at` (timestamp, default 7 days)
- Cache invalidation: on profile update or explicit user request
- RAG queries personalized by injecting background into system prompt

**Privacy & PII Handling**:
- `software_background` and `hardware_background` classified as PII (Personally Identifiable Information)
- MUST only use profile fields in LLM prompt context
- NEVER log profile fields in application logs, error messages, or analytics
- Anonymize user data in development/staging environments
- Database backups encrypted at rest
- Profile data access limited to: authentication service, personalization service, user profile management
- Audit log all access to profile data with timestamp and service identifier
- GDPR compliance: users can request profile data export and deletion

**Cross-Service Communication**:
- Node.js auth service exposes JWKS endpoint at `/.well-known/jwks.json`
- Python backend fetches public keys from JWKS endpoint on startup
- Public key cache refreshed every 24 hours or on verification failure
- JWT verification failures result in HTTP 401 with clear error message
- No fallback to weaker authentication methods

### VIII. Static Localization & Routing

Localization features MUST maintain content consistency, implement auth-gated access, preserve state during navigation, and support proper text directionality.

**Rationale**: The Physical AI platform serves a global audience including Urdu-speaking learners. Proper localization ensures educational content is accessible to diverse communities while maintaining security and user experience standards. Content consistency prevents broken links and 404 errors that would degrade the learning experience.

**Requirements**:

**Content Consistency**:
- For every file in `docs/`, a corresponding translated file MUST exist in `i18n/ur/docusaurus-plugin-content-docs/current/` to prevent 404 errors during routing
- Automated build process MUST verify translation completeness before deployment
- Missing translations MUST either show English fallback or redirect to main content page with clear language indication
- Translation synchronization scripts MUST be implemented to identify and flag missing translations
- Translation completeness metrics MUST be tracked and reported in CI/CD pipeline

**Auth-Gated Access**:
- The "Translate to Urdu" routing logic MUST verify the user's JWT session via Better-Auth before initiating the redirect
- Unauthenticated users attempting to access localized content MUST be redirected to login page
- JWT session verification MUST include token validity, expiration check, and proper audience validation
- Session state MUST be preserved during language switch operations
- Auth middleware MUST be properly configured for all localization routes

**Stateless Navigation**:
- Use the Docusaurus router for internal navigation to ensure the React state (and Auth session) is preserved across page transitions
- Navigation between localized content MUST maintain user context and session data
- Client-side routing MUST not break authentication state or require re-authentication
- URL parameters and query strings MUST be properly handled during localization transitions
- Back button functionality MUST work correctly between localized and non-localized content

**Directionality**:
- The Urdu docs MUST be served with `dir="rtl"` as configured in the Docusaurus i18n settings
- RTL styling MUST be implemented for all UI components to ensure proper text alignment and layout
- Bidirectional text handling MUST be tested and validated for mixed content
- CSS rules MUST be properly configured to handle RTL layouts without breaking existing components
- UI elements (buttons, navigation, forms) MUST be properly positioned for RTL languages

## Technology Stack Requirements

### Frontend
- **Framework**: Docusaurus (TypeScript)
- **UI Components**: React with custom components
- **Styling**: CSS Modules, Tailwind CSS (optional)
- **State Management**: React Context API or Zustand
- **Testing**: Jest + React Testing Library
- **Deployment**: Vercel (connected to Git repository)

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: Neon PostgreSQL (serverless)
- **Vector Database**: Qdrant (self-hosted or cloud)
- **ORM**: SQLAlchemy async
- **Authentication**: Better Auth or custom JWT implementation
- **Testing**: pytest + pytest-asyncio + pytest-cov
- **Server**: Gunicorn with Uvicorn workers
- **Deployment**: Railway (connected to Git repository)

### AI/ML
- **Embeddings**: Gemini `text-embedding-004`
- **Generation**: Gemini `gemini-2.0-flash-exp`
- **SDK**: OpenAI Agent SDK configured for Gemini
- **RAG**: Qdrant for vector storage, custom retrieval pipeline

### DevOps
- **Version Control**: Git + GitHub
- **CI/CD**: GitHub Actions
- **Preview Environments**: Vercel (frontend), Railway (backend)
- **Monitoring**: Vercel Analytics, Railway logs
- **Environment Variables**: Managed via platform dashboards (Vercel, Railway)

## Agent Responsibilities

### Development Workflow
1. **Specification Phase** (`/sp.specify`):
   - Project Manager creates feature specification
   - Content Specialist provides robotics domain expertise
   - Human reviews and approves specification

2. **Planning Phase** (`/sp.plan`):
   - Project Manager creates implementation plan
   - Relevant specialists (Frontend, Backend, RAG) provide technical input
   - Plan includes constitution compliance checks

3. **Task Generation** (`/sp.tasks`):
   - Project Manager breaks plan into actionable tasks
   - Tasks assigned to appropriate specialist agents
   - TDD requirements explicit in each task

4. **Implementation Phase** (`/sp.implement`):
   - Specialist agents execute tasks in their domain
   - Test Specialist creates tests before implementation (Red)
   - Specialist implements minimal code to pass tests (Green)
   - Specialist refactors while maintaining green tests (Refactor)

5. **Review Phase**:
   - Test Specialist verifies coverage >80%
   - Appropriate specialists review cross-domain interactions
   - Project Manager ensures constitution compliance

6. **Deployment Phase** (`/sp.git.commit_pr`):
   - Deployment Specialist prepares deployment configuration
   - Preview environments created for PR review
   - Human approves production deployment
   - Deployment Specialist executes deployment

### Cross-Agent Communication
- Agents must explicitly hand off context when crossing domain boundaries
- Handoff includes: completed work summary, remaining tasks, blockers, questions
- Project Manager tracks handoffs and ensures no tasks fall through gaps

## Core Values

### Accuracy in Physical AI and Humanoid Robotics Content
- All educational content verified by Content Generation Specialist
- Citations to authoritative sources (research papers, manufacturer docs)
- Regular updates to reflect latest advances in field
- Student-facing explanations technically accurate but pedagogically sound

### Separation of Responsibilities
- Each agent operates within defined domain boundaries
- No agent modifies code outside their expertise area
- Cross-domain concerns handled through explicit collaboration
- Domain violations flagged in code review

### Continuous Integration of AI with Robotics
- RAG system continuously ingests latest robotics documentation
- Simulation environments kept current with industry standards
- Real-world deployment considerations built into curriculum
- Student projects bridge theory and practice

## Governance

### Amendment Process
1. Proposed amendment drafted with rationale and impact analysis
2. Affected templates and guidance documents identified
3. Human review and approval of amendment
4. Version number incremented according to semantic versioning:
   - **MAJOR**: Backward-incompatible principle removals or redefinitions
   - **MINOR**: New principles or materially expanded guidance
   - **PATCH**: Clarifications, wording fixes, non-semantic refinements
5. Constitution and dependent artifacts updated atomically
6. Amendment logged in Sync Impact Report (HTML comment at top of constitution)

### Compliance Review
- All pull requests MUST include constitution compliance checklist
- Automated checks where possible (test coverage, linting, security scans)
- Human review for subjective principles (UX, modularity, domain boundaries)
- Project Manager ensures constitution adherence throughout development lifecycle

### Conflict Resolution
- Constitution supersedes all other project practices and documentation
- In case of ambiguity, consult Project Manager for interpretation
- Disputed interpretations escalated to human for final decision
- Decisions documented as precedents for future reference

### Living Document
- Constitution reviewed quarterly for relevance and effectiveness
- Amendments proposed based on team feedback and project evolution
- Historical versions maintained in Git history
- Breaking changes require migration plan and communication to all stakeholders

**Version**: 1.2.0 | **Ratified**: 2025-12-25 | **Last Amended**: 2025-12-28
