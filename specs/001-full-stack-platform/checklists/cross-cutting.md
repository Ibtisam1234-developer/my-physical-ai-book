# Cross-Cutting Concerns Checklist: Physical AI Platform

**Purpose**: Validate Security, Performance, and Accessibility requirements quality across all domains (Frontend, Backend, RAG, Auth, Database, Deployment, Content)

**Created**: 2025-12-25

**Feature**: [spec.md](../spec.md)

**Focus**: Requirements quality validation (NOT implementation verification)

---

## Security Requirements Quality

### Authentication & Authorization

- [x] CHK001 - Are password complexity requirements quantified beyond "min 8 characters"? [Clarity, Spec §FR-063 - Now specifies: min 8 chars, uppercase, lowercase, number]
- [ ] CHK002 - Is JWT token payload structure explicitly defined (claims, expiration format)? [Completeness, Spec §FR-032]
- [x] CHK003 - Are token refresh requirements specified (timing, refresh token vs access token)? [Clarity, Spec §NFR Security - "24 hours with automatic refresh via /api/auth/refresh"]
- [ ] CHK004 - Are session timeout requirements defined for inactive users? [Gap]
- [ ] CHK005 - Is "Remember Me" functionality explicitly in or out of scope? [Scope Boundary - Deferred to v2.0]
- [x] CHK006 - Are multi-device login requirements addressed (max sessions per user)? [Gap - Now FR-061: Max 5 concurrent sessions]
- [x] CHK007 - Are password reset/forgot password requirements specified? [Gap, Exception Flow - Now FR-060: Password reset via email]
- [x] CHK008 - Is account lockout after failed attempts requirement defined beyond rate limiting? [Gap - Now FR-062: Lockout after 10 failed attempts per hour]

### Input Validation & Injection Prevention

- [x] CHK009 - Are input validation requirements consistent across all API endpoints? [Consistency, Spec §FR-016 - Pydantic validation for all endpoints]
- [ ] CHK010 - Is the validation strategy for JSONB chat history content specified? [Gap, Spec §FR-044]
- [ ] CHK011 - Are file upload validation requirements defined if users submit lab work? [Gap, Spec User Story 5 - Deferred to implementation]
- [ ] CHK012 - Is sanitization strategy for markdown user input specified? [Gap - Deferred, rely on React auto-escaping]
- [x] CHK013 - Are SQL injection prevention requirements verified for all database queries? [Coverage, Spec §FR-016 + plan.md SQLAlchemy]
- [x] CHK014 - Are XSS prevention requirements defined for user-generated content display? [Completeness, Spec §NFR Security + CSP headers]

### Secrets Management

- [x] CHK015 - Are all required environment variables documented with format/validation rules? [Completeness, Spec §FR-048 + quickstart.md lists all vars]
- [ ] CHK016 - Is secret rotation procedure requirement defined? [Gap, Constitution VI - Deferred to operations runbook]
- [ ] CHK017 - Are development vs production secret isolation requirements specified? [Gap - Implicit in Vercel/Railway env config]
- [x] CHK018 - Is the requirement for never logging sensitive data explicitly stated? [Gap - Now in §NFR Security: "Sensitive data never logged"]

### API Security

- [x] CHK019 - Are CORS requirements specified with exact allowed origins (not wildcards)? [Clarity - Now in §NFR Security: "CORS restricted to specific allowed origins (no wildcards)"]
- [x] CHK020 - Are HTTPS/TLS requirements quantified (minimum TLS version, cipher suites)? [Clarity - Now §NFR Security: "HTTPS/TLS 1.2+ with strong cipher suites"]
- [ ] CHK021 - Is CSRF protection requirement explicit beyond SameSite cookies? [Completeness, Spec §NFR Security - SameSite=Strict sufficient]
- [x] CHK022 - Are rate limiting requirements specified per-endpoint (not just /api/chat)? [Coverage - FR-015 chat + §NFR Security auth limiting]
- [x] CHK023 - Is rate limiting scope clarified (per-IP, per-user, per-session)? [Clarity, Spec §FR-015 - "per user"]
- [x] CHK024 - Are API request size limits requirements defined (headers, body, query params)? [Gap - Now FR-066: 10MB body limit]

### Data Protection

- [x] CHK025 - Are data retention requirements specified for chat history? [Gap - Now FR-064: 90-day retention, then archive]
- [x] CHK026 - Are user data deletion requirements defined (GDPR "right to be forgotten")? [Gap - Now FR-065: User can delete account + all data]
- [ ] CHK027 - Is encryption at rest requirement specified for sensitive database fields? [Gap - Neon provides encryption at rest by default, defer explicit requirement]
- [ ] CHK028 - Are audit logging requirements defined for security-critical operations? [Gap, Spec §FR-017 partial - Deferred to implementation]

---

## Performance Requirements Quality

### Frontend Performance

- [ ] CHK029 - Are "First Contentful Paint <1.5s" requirements defined for all page types or just homepage? [Scope, Spec §SC-006 - Applies to all pages]
- [x] CHK030 - Is the network condition baseline specified (3G, 4G, WiFi) for performance targets? [Clarity - Now §NFR Performance: "3G (1.6 Mbps, 300ms RTT)"]
- [x] CHK031 - Are image/asset optimization requirements specified (formats, compression, lazy loading)? [Gap - Now §NFR Performance: "WebP format, lazy loading"]
- [x] CHK032 - Is JavaScript bundle size requirement defined? [Gap - Now §NFR Performance: "500KB gzipped"]
- [ ] CHK033 - Are requirements specified for progressive enhancement (core functionality without JS)? [Gap - Not applicable for React SPA, deferred]
- [x] CHK034 - Is caching strategy requirement defined (service workers, HTTP cache headers)? [Gap - Now §NFR Performance: "HTTP caching 1hr static, no-cache API"]

### Backend Performance

- [x] CHK035 - Are database query performance requirements specified per-query or averaged? [Clarity - Now §NFR Performance: "p95 latency" specified]
- [x] CHK036 - Is connection pool sizing requirement explicitly defined? [Clarity - §NFR Scalability: "20 concurrent connections" + research.md pool_size=20]
- [x] CHK037 - Are database index requirements traced to specific query patterns? [Traceability, Spec §FR-043 + data-model.md]
- [ ] CHK038 - Is the "Qdrant search <200ms" requirement defined for cold vs warm cache? [Clarity - Now §NFR Performance p95, applies to typical queries]
- [ ] CHK039 - Are batch processing requirements specified for document ingestion? [Gap - Deferred to RAG implementation, plan.md mentions batch]

### RAG Pipeline Performance

- [ ] CHK040 - Is embedding generation latency requirement specified (per-document, batched)? [Gap, Spec §FR-023]
- [ ] CHK041 - Are requirements defined for handling large documents (>100k tokens)? [Edge Case, Gap]
- [ ] CHK042 - Is Qdrant collection size limit requirement defined (10k chunks mentioned but not enforced)? [Clarity, Spec §Scale/Scope]
- [ ] CHK043 - Are requirements specified for re-ingestion strategy when docs updated? [Gap]
- [ ] CHK044 - Is caching strategy requirement defined for repeated queries? [Gap, Risk mitigation mentions caching]

### Streaming & Real-time

- [x] CHK045 - Is "first token <2s" requirement defined for worst-case or p95 latency? [Clarity - Now §NFR Performance: "p95 latency"]
- [ ] CHK046 - Are SSE reconnection requirements specified (auto-reconnect, backoff)? [Gap - Browser handles auto-reconnect, deferred]
- [ ] CHK047 - Is the streaming chunk size/frequency requirement defined? [Gap - Deferred to implementation]
- [ ] CHK048 - Are requirements specified for handling interrupted streams? [Gap - Edge Cases mentions token expiry mid-session, partial coverage]

---

## Accessibility Requirements Quality

### WCAG 2.1 AA Compliance

- [ ] CHK049 - Are specific WCAG 2.1 AA success criteria mapped to requirements (e.g., 1.4.3 Contrast, 2.1.1 Keyboard)? [Traceability, Spec §FR-009]
- [ ] CHK050 - Is color contrast requirement quantified (4.5:1 for text, 3:1 for UI)? [Clarity, Spec §FR-009]
- [ ] CHK051 - Are focus indicator requirements specified (visibility, contrast, size)? [Gap, Spec §FR-009]
- [ ] CHK052 - Is keyboard navigation order requirement defined for complex UI (chatbot, forms)? [Gap, Spec §FR-009]

### Screen Reader Support

- [ ] CHK053 - Are ARIA label requirements specified for all interactive elements? [Completeness, Spec §FR-009]
- [ ] CHK054 - Are live region requirements defined for dynamic content (streaming chat messages)? [Gap, Spec §FR-006 + FR-009]
- [ ] CHK055 - Is screen reader announcement strategy requirement defined for loading states? [Gap]
- [ ] CHK056 - Are requirements specified for meaningful alt text on educational diagrams? [Gap, Spec User Story 1]

### Form Accessibility

- [ ] CHK057 - Are form error message requirements specified for screen reader announcement? [Gap, Spec User Story 3]
- [ ] CHK058 - Is fieldset/legend requirement defined for grouped form controls? [Gap]
- [ ] CHK059 - Are autocomplete attribute requirements specified for auth forms (email, password)? [Gap, Spec §FR-007]

### Responsive & Mobile Accessibility

- [ ] CHK060 - Are touch target size requirements specified (minimum 44x44px)? [Gap, Spec §FR-003]
- [ ] CHK061 - Is zoom/pinch requirement defined (allow up to 200% without breaking layout)? [Gap]
- [ ] CHK062 - Are requirements specified for landscape vs portrait orientation handling? [Gap]

---

## Cross-Domain Integration Requirements Quality

### Frontend ↔ Backend

- [ ] CHK063 - Is API error response format requirement consistent across all endpoints? [Consistency, Spec contracts/]
- [ ] CHK064 - Are timeout requirements specified for all API calls (connection, read, total)? [Gap]
- [ ] CHK065 - Are retry logic requirements defined (which endpoints, max attempts, backoff)? [Gap, Spec Edge Cases partial]
- [ ] CHK066 - Is the API versioning strategy requirement documented? [Gap]

### Backend ↔ External Services (Gemini, Qdrant, Neon)

- [ ] CHK067 - Are fallback requirements defined for each external service failure? [Coverage, Spec Edge Cases]
- [ ] CHK068 - Is circuit breaker pattern requirement specified for Gemini API calls? [Gap]
- [ ] CHK069 - Are requirements specified for handling partial Qdrant search results? [Gap]
- [ ] CHK070 - Is database connection retry requirement defined (attempts, timing)? [Gap, Spec research.md mentions pool_pre_ping]

### Auth ↔ Protected Endpoints

- [ ] CHK071 - Are authentication requirements consistent across all protected endpoints? [Consistency, Spec §FR-036]
- [ ] CHK072 - Is the requirement for 401 vs 403 response distinction defined? [Clarity, Spec §FR-038]
- [ ] CHK073 - Are requirements specified for handling expired tokens during long operations? [Gap, Edge Cases partial]

---

## Observability Requirements Quality

### Logging

- [ ] CHK074 - Are log level requirements defined (debug, info, warning, error)? [Gap, Spec §FR-017]
- [ ] CHK075 - Is PII/sensitive data exclusion requirement explicit in logging spec? [Gap, Spec §FR-017]
- [ ] CHK076 - Are correlation ID propagation requirements specified across services? [Completeness, Spec §FR-017]
- [ ] CHK077 - Is log retention requirement defined (duration, rotation)? [Gap]

### Monitoring & Alerting

- [ ] CHK078 - Are specific metrics requirements defined (beyond "Vercel Analytics, Railway logs")? [Clarity, Spec §NFR Observability]
- [ ] CHK079 - Are alerting threshold requirements specified (error rate %, latency p95)? [Gap, Spec §NFR Observability]
- [ ] CHK080 - Is on-call/incident response requirement scope defined? [Gap]
- [ ] CHK081 - Are health check endpoint response requirements detailed (what to include in response)? [Clarity, Spec §FR-018]

---

## Testing Requirements Quality

### TDD & Coverage

- [ ] CHK082 - Are specific coverage dimensions defined (line, branch, function)? [Clarity, Spec §NFR Maintainability ">80%"]
- [ ] CHK083 - Is coverage exclusion criteria requirement specified (generated code, constants)? [Gap]
- [ ] CHK084 - Are mock/stub requirements specified for external services? [Completeness, Spec Constitution II]
- [ ] CHK085 - Is test data management requirement defined (fixtures, factories, seeding)? [Gap]

### Test Strategy

- [ ] CHK086 - Are unit vs integration test boundary requirements defined? [Gap, Spec Constitution II]
- [ ] CHK087 - Is end-to-end test scope requirement specified (which critical paths)? [Gap]
- [ ] CHK088 - Are performance test requirements defined (load testing, stress testing)? [Gap, Spec §SC-007 mentions "100 concurrent"]
- [ ] CHK089 - Is accessibility testing requirement specified (automated tools, manual testing)? [Gap, Spec §FR-009]

---

## Deployment & Operations Requirements Quality

### CI/CD

- [ ] CHK090 - Are CI pipeline stage requirements explicitly ordered (lint → test → build → deploy)? [Completeness, Spec §FR-050]
- [ ] CHK091 - Is deployment rollback trigger requirement defined (failed health checks, error rate threshold)? [Gap, Spec Edge Cases mentions rollback]
- [ ] CHK092 - Are PR preview environment teardown requirements specified? [Gap, Spec §FR-049]
- [ ] CHK093 - Is deployment approval gate requirement explicit (automated vs manual, who approves)? [Clarity, Constitution VI]

### Environment Configuration

- [ ] CHK094 - Are all required environment variables documented in spec or design docs? [Completeness, Spec §FR-048]
- [ ] CHK095 - Is environment variable validation requirement specified (startup checks)? [Gap]
- [ ] CHK096 - Are requirements defined for handling missing/invalid env vars? [Gap, Edge Case]

### Database Operations

- [ ] CHK097 - Are migration rollback requirements specified? [Gap, Spec §FR-045]
- [ ] CHK098 - Is zero-downtime migration requirement explicit for production? [Gap, Constitution VI]
- [ ] CHK099 - Are database backup verification requirements defined? [Gap, Spec §NFR Reliability]
- [ ] CHK100 - Is the requirement for migration testing in preview environments specified? [Gap, Spec §FR-049]

---

## Content Quality Requirements

### Educational Content

- [ ] CHK101 - Are content accuracy validation requirements specified (peer review, citation verification)? [Gap, Constitution Core Values]
- [ ] CHK102 - Is technical depth calibration requirement defined (beginner, intermediate, advanced)? [Gap, Spec §FR-058]
- [ ] CHK103 - Are code example validation requirements specified (must be runnable, tested)? [Gap, Spec §FR-002]
- [ ] CHK104 - Is content update frequency requirement defined? [Gap, Spec Risks mentions "quarterly review"]

### RAG Content Alignment

- [x] CHK105 - Are requirements defined for RAG accuracy measurement (precision, recall on test set)? [Measurability, Spec §SC-003 - ">80% precision verified through test question set"]
- [x] CHK106 - Is the test question set requirement specified (size, diversity, update frequency)? [Gap - Now FR-068: "Minimum 20 questions for precision validation"]
- [x] CHK107 - Are source citation format requirements defined? [Clarity, Spec §FR-029 - "cite source documents" + data-model.md shows structure]
- [ ] CHK108 - Is hallucination detection/mitigation requirement specified beyond "cite sources"? [Gap - Acceptable: citation requirement + human review in constitution]

---

## Consistency & Coherence Across Domains

### Error Handling

- [ ] CHK109 - Are error message tone/format requirements consistent (technical vs user-friendly)? [Consistency, Spec Edge Cases]
- [ ] CHK110 - Is error message localization requirement in or out of scope? [Scope Boundary, Out of Scope mentions "English-only"]
- [ ] CHK111 - Are error tracking requirements specified (Sentry, logging only, both)? [Gap, Spec §NFR Observability]

### User Feedback

- [ ] CHK112 - Are loading state UI requirements consistent across async operations (chatbot, auth, content)? [Consistency, Spec §FR-006]
- [ ] CHK113 - Is "user-friendly error message" requirement quantified (max length, tone, actionability)? [Clarity, Spec §NFR Reliability]
- [ ] CHK114 - Are success confirmation requirements defined for state-changing operations? [Gap]

### Responsive Design

- [ ] CHK115 - Are breakpoint thresholds explicitly defined (not just "320px to 4K")? [Clarity, Spec §FR-003]
- [ ] CHK116 - Are mobile-specific interaction requirements defined (swipe, pinch, long-press)? [Gap, Spec §FR-003]
- [ ] CHK117 - Is requirement defined for handling very large screens (>4K, ultra-wide)? [Edge Case, Spec §FR-003]

---

## Scalability & Reliability Requirements Quality

### Load Handling

- [ ] CHK118 - Is the "100 concurrent users" requirement defined per-second or sustained? [Clarity, Spec §SC-007]
- [ ] CHK119 - Are degradation requirements specified when approaching capacity limits? [Gap, Spec Edge Cases partial]
- [ ] CHK120 - Is horizontal scaling strategy requirement documented? [Gap, Spec §NFR Scalability mentions "stateless API"]
- [ ] CHK121 - Are queue depth/overflow requirements specified for rate-limited requests? [Gap, Spec Edge Cases mentions "queue"]

### Failure Recovery

- [x] CHK122 - Are recovery time objective (RTO) requirements defined for critical services? [Gap - Now §NFR Reliability: "RTO 4 hours"]
- [x] CHK123 - Is data loss tolerance requirement specified (RPO - Recovery Point Objective)? [Gap - Now §NFR Reliability: "RPO 24 hours"]
- [x] CHK124 - Are partial failure requirements defined (serve content if chatbot down)? [Gap - Now §NFR Reliability: "Partial degradation: serve static content if backend unavailable"]
- [ ] CHK125 - Is requirement specified for maintaining session state during backend restarts? [Gap - Deferred, JWT in cookies persists across restarts]

---

## Compliance & Constitution Alignment

### Constitution Principle Traceability

- [x] CHK126 - Are all Security principle requirements (I) traceable to functional requirements? [Traceability, Constitution I → Spec §FR-016, §FR-031-038, §FR-060-063, §FR-066, §NFR Security]
- [x] CHK127 - Are all TDD requirements (II) reflected in test coverage specifications? [Traceability, Constitution II → Spec §NFR Maintainability ">80%" + tasks.md 28 test tasks]
- [x] CHK128 - Are all UX requirements (III) mapped to WCAG 2.1 AA criteria? [Traceability, Constitution III → Spec §FR-009, §SC-006, §SC-009]
- [x] CHK129 - Are all Gemini Usage requirements (IV) specified in RAG pipeline section? [Traceability, Constitution IV → Spec §FR-023, §FR-028, §FR-030]
- [x] CHK130 - Are modularity requirements (V) enforced through agent assignment in plan? [Traceability, Constitution V → plan.md lists 8 specialist agents]
- [x] CHK131 - Are human approval requirements (VI) explicitly called out in deployment spec? [Traceability, Constitution VI → Spec §FR-051, plan.md Constitution Check]

### OWASP Top 10 Coverage

- [x] CHK132 - Are requirements addressing OWASP A01 (Broken Access Control) specified? [Coverage - FR-036, FR-037, FR-061, FR-062]
- [x] CHK133 - Are requirements addressing OWASP A02 (Cryptographic Failures) specified? [Coverage - FR-033 bcrypt cost 12, §NFR Security TLS 1.2+]
- [x] CHK134 - Are requirements addressing OWASP A03 (Injection) specified? [Coverage - FR-016, FR-066, FR-067]
- [x] CHK135 - Are requirements addressing OWASP A07 (Identification/Auth Failures) specified? [Coverage - FR-031 to FR-038, FR-060-063]
- [x] CHK136 - Are requirements addressing OWASP A09 (Security Logging Failures) specified? [Coverage - FR-017, FR-019, §NFR Security "Sensitive data never logged"]

---

## Notes

**Requirements Quality Assessment**:

The specification demonstrates **strong coverage** in:
- Functional requirements (59 specific FRs across 6 domains)
- User story prioritization (P1-P5 with clear dependencies)
- Edge case identification (9 scenarios with mitigation strategies)
- Constitution alignment (all 6 principles addressed)

**Gaps Requiring Attention**:
- **Authentication edge cases**: Password reset, account lockout, multi-device sessions
- **Performance baselines**: Network conditions, cache states, percentile definitions (p50, p95, p99)
- **Failure recovery**: RTO/RPO, partial degradation modes, state preservation
- **Content operations**: Update strategy, validation process, accuracy measurement details
- **Deployment operations**: Rollback triggers, migration testing, approval workflow details

**Recommendations**:
1. **High Priority**: Address security gaps (CHK007, CHK016, CHK024-028) - Critical for production readiness
2. **Medium Priority**: Clarify performance baselines (CHK029, CHK035, CHK045) - Affects acceptance testing
3. **Low Priority**: Document nice-to-have features (CHK005, CHK033) - Can defer to future iterations

**Next Steps**:
- Review gaps with Project Manager
- Update spec.md to address critical gaps (CHK007, CHK016, CHK024-028)
- Run `/sp.clarify` if additional requirements need specification
- Proceed to `/sp.tasks` when requirements quality is satisfactory
