---
id: 003
title: Standards Static Localization Routing
stage: constitution
date: 2025-12-28
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: 001-auth-personalization
user:
command: /sp.constitution
labels: [constitution, localization, routing, i18n]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
---

## Prompt

```
📜 sp.constitution
Standards for Static Localization & Routing

Content Consistency: For every file in docs/, a corresponding translated file must exist in i18n/ur/docusaurus-plugin-content-docs/current/ to prevent 404 errors during routing.

Auth-Gated Access: The "Translate to Urdu" routing logic must verify the user's JWT session via Better-Auth before initiating the redirect.

Stateless Navigation: Use the Docusaurus router for internal navigation to ensure the React state (and Auth session) is preserved across page transitions.

Directionality: The Urdu docs must be served with dir="rtl" as configured in the Docusaurus i18n settings.
```

## Response snapshot

The constitution has been updated to include a new principle VIII: Static Localization & Routing. The key changes include:

1. Added new principle VIII with requirements for content consistency, auth-gated access, stateless navigation, and directionality
2. Version updated from 1.1.0 to 1.2.0
3. Sync Impact Report updated to reflect the changes
4. Detailed requirements added for:
   - Content Consistency: Ensuring translated files exist to prevent 404 errors
   - Auth-Gated Access: JWT session verification for Urdu content access
   - Stateless Navigation: Using Docusaurus router to preserve auth state
   - Directionality: Proper RTL text support for Urdu content

## Outcome

- ✅ Impact: New localization principle added to constitution with detailed requirements
- 🧪 Tests: N/A (constitution update)
- 📁 Files: .specify/memory/constitution.md updated
- 🔁 Next prompts: Implementation of localization features based on new requirements
- 🧠 Reflection: Constitution now includes comprehensive localization standards for Urdu content

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Implement localization features according to new constitution requirements