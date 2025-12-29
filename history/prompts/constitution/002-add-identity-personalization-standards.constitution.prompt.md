---
id: 002
title: Add Identity Personalization Standards
stage: constitution
date: 2025-12-27
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: user
command: /sp.constitution
labels: ["constitution", "authentication", "personalization", "privacy", "jwt", "pii"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - none
---

## Prompt

Standards for Identity & Personalization

Stateless Auth: Use RS256 JWTs for cross-service communication. The Node server signs the tokens, and the Python backend verifies them using a public key/JWKS.

Schema Strictness: Software and Hardware background fields are mandatory at signup and must be stored in the user table of Neon DB.

Persistence: Personalized content is generated on-demand but should be cached or stored in Neon DB to avoid redundant LLM costs.

Privacy: The software_background and hardware_background are PII (Personally Identifiable Information) and must only be used in the prompt context, never logge

## Response snapshot

Successfully updated the project constitution from version 1.0.0 to 1.1.0 by adding Principle VII: Identity & Personalization. This new principle establishes comprehensive standards for:

1. **Stateless Authentication**: RS256 JWT tokens for cross-service communication between Node.js and Python FastAPI backends, with public key verification via JWKS endpoint
2. **Schema Strictness**: Mandatory `software_background` and `hardware_background` fields at signup stored in Neon PostgreSQL with NOT NULL constraints and enum validation
3. **Personalization Persistence**: On-demand LLM-generated content cached in `personalized_content` table to reduce API costs, with 7-day expiration
4. **Privacy & PII Handling**: Classified profile fields as PII with strict rules preventing logging, requiring encryption at rest, and ensuring GDPR compliance
5. **Cross-Service Communication**: JWKS endpoint at `/.well-known/jwks.json` with 24-hour public key caching

The constitution file was updated with the new principle inserted between Principle VI (Human Approval) and the Technology Stack Requirements section. The Sync Impact Report was updated to reflect the minor version bump (MINOR: new principle added) and the last amended date was set to 2025-12-27.

## Outcome

- ✅ Impact: Added comprehensive Identity & Personalization standards to constitution as Principle VII, establishing authentication, user schema, and privacy requirements
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md (version 1.0.0 → 1.1.0)
- 🔁 Next prompts: Consider implementing the authentication system according to these standards, or documenting this decision in an ADR
- 🧠 Reflection: This principle provides clear, actionable requirements for implementing personalized learning experiences while maintaining security and privacy. The stateless JWT approach enables horizontal scaling, and mandatory profile fields at signup reduce friction in the user journey.

## Evaluation notes (flywheel)

- Failure modes observed: None - constitution update completed successfully with proper versioning
- Graders run and results (PASS/FAIL): Manual validation - PASS (no unresolved placeholders, proper version increment, coherent structure)
- Prompt variant (if applicable): Standard constitution amendment workflow
- Next experiment (smallest change to try): Validate that auth-specialist and backend-specialist agents understand and enforce these new requirements in future implementations
