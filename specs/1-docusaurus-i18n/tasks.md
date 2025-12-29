---
description: "Task list for Docusaurus i18n Localization feature implementation"
---

# Tasks: Docusaurus i18n Localization

**Input**: Design documents from `/specs/1-docusaurus-i18n/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create i18n directory structure for Urdu locale at i18n/ur/docusaurus-plugin-content-docs/current/
- [X] T002 [P] Install required dependencies: @docusaurus/core, @docusaurus/router, @better-auth/client
- [X] T003 [P] Verify existing Docusaurus configuration and project structure

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Update docusaurus.config.js with i18n configuration for English and Urdu locales with RTL direction for Urdu
- [X] T005 [P] Configure Better Auth client to point to Node server for session validation
- [X] T006 [P] Create basic component structure for LocaleRouter and TranslateButton
- [X] T007 Set up proper RTL CSS support for Urdu content in theme directory

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Access Urdu Documentation (Priority: P1) 🎯 MVP

**Goal**: Enable authenticated users to access Urdu documentation by clicking a translate button on English pages

**Independent Test**: User can navigate to an English documentation page, click the translate button, and be redirected to the corresponding Urdu documentation page with proper RTL layout if authenticated, or redirected to login if not authenticated.

### Implementation for User Story 1

- [X] T008 [P] [US1] Create TranslateButton component at src/components/LocaleRouter/TranslateButton.tsx
- [X] T009 [P] [US1] Implement authClient.useSession() check in TranslateButton component
- [X] T010 [US1] Implement path calculation and construction logic in TranslateButton component
- [X] T011 [US1] Implement routing logic using @docusaurus/router in TranslateButton component
- [X] T012 [US1] Add authentication validation and redirect to login if not authenticated
- [X] T013 [US1] Add proper RTL styling for Urdu content display
- [X] T014 [US1] Test component integration with existing Docusaurus documentation pages

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Proper Content Structure (Priority: P2)

**Goal**: Ensure Urdu documentation is properly structured in the Docusaurus i18n directory to support the localization system

**Independent Test**: Urdu Markdown files exist in the correct i18n directory structure and are accessible through the Docusaurus build process, with proper fallback handling for missing translations.

### Implementation for User Story 2

- [X] T015 [P] [US2] Populate i18n/ur/docusaurus-plugin-content-docs/current/ with translated content files
- [ ] T016 [US2] Implement build-time validation to ensure translation completeness
- [ ] T017 [US2] Create fallback mechanism for missing Urdu translations
- [ ] T018 [US2] Add translation status tracking for documentation completeness metrics
- [X] T019 [US2] Test that English documentation paths correctly map to Urdu paths

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Authentication-Based Access Control (Priority: P3)

**Goal**: Ensure access to localized content requires authentication and maintains security standards

**Independent Test**: Authenticated users can access Urdu documentation with preserved session state, while unauthenticated users are redirected to the authentication flow when attempting to access localized content.

### Implementation for User Story 3

- [X] T020 [P] [US3] Enhance JWT session validation for locale switching functionality
- [X] T021 [US3] Implement session state preservation during locale switching
- [X] T022 [US3] Add proper error handling for expired or invalid JWT sessions
- [X] T023 [US3] Ensure authentication middleware is properly configured for localization routes
- [X] T024 [US3] Test session state preservation between English and Urdu content navigation

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T025 [P] Add the translate button to the DocItem layout so it appears automatically on every documentation chapter
- [X] T026 [P] Documentation updates for the localization feature in docs/
- [X] T027 Code cleanup and refactoring of localization components
- [X] T028 Performance optimization for locale switching
- [X] T029 [P] Accessibility improvements for RTL layout and translate button
- [X] T030 Security hardening for authentication validation
- [X] T031 Run quickstart.md validation to ensure all steps work correctly

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all components for User Story 1 together:
Task: "Create TranslateButton component at src/components/LocaleRouter/TranslateButton.tsx"
Task: "Implement authClient.useSession() check in TranslateButton component"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence