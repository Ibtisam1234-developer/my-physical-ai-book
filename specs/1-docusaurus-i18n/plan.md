# Implementation Plan: Docusaurus i18n Localization

**Branch**: `1-docusaurus-i18n` | **Date**: 2025-12-28 | **Spec**: specs/1-docusaurus-i18n/spec.md
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of Docusaurus i18n localization feature to support English (LTR) and Urdu (RTL) locales with authentication-gated access. The solution includes configuring Docusaurus i18n, creating a translate button component, implementing proper routing between locales, and ensuring JWT-based authentication validation for accessing Urdu content.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: TypeScript/JavaScript for Docusaurus, Node.js for auth server
**Primary Dependencies**: Docusaurus, @docusaurus/core, @docusaurus/router, @better-auth/client
**Storage**: N/A (static site generation with i18n directories)
**Testing**: Jest with React Testing Library
**Target Platform**: Web browser, responsive design for all devices
**Project Type**: Web documentation site with localization
**Performance Goals**: Static content loads under 2 seconds, seamless locale switching
**Constraints**: Must maintain RTL layout for Urdu, preserve auth session during navigation
**Scale/Scope**: Support for multiple locales, initially English and Urdu

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

1. **Security (NON-NEGOTIABLE)**: Authentication verification required via Better-Auth before accessing localized content (FR-004, FR-007)
2. **Test-Driven Development (NON-NEGOTIABLE)**: Tests required for locale switching component and auth validation logic
3. **User Experience**: RTL support for Urdu content with proper text direction and layout (FR-008)
4. **Modularity**: Frontend Specialist handles Docusaurus configuration, Auth Specialist handles JWT validation
5. **Human Approval**: Critical changes to auth flow require approval
6. **Identity & Personalization**: Integration with existing JWT-based authentication system
7. **Static Localization & Routing**: Content consistency, auth-gated access, stateless navigation, directionality requirements met

## Project Structure

### Documentation (this feature)

```text
specs/1-docusaurus-i18n/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
# Web application structure
docusaurus.config.js
i18n/
├── ur/
│   └── docusaurus-plugin-content-docs/
│       └── current/
│           └── [urdu markdown files]
src/
├── components/
│   └── LocaleRouter/
│       ├── LocaleRouter.tsx
│       └── TranslateButton.tsx
└── theme/
    └── [custom theme files for RTL support]
```

**Structure Decision**: Web application with Docusaurus frontend and i18n directory structure for localized content. The LocaleRouter component handles locale switching with authentication validation.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |