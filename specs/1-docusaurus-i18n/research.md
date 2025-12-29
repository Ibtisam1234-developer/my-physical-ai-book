# Research: Docusaurus i18n Localization

## Decision: Docusaurus i18n Configuration Approach
**Rationale**: Docusaurus provides built-in internationalization support through its i18n configuration. This approach leverages the existing Docusaurus infrastructure rather than building custom localization logic.

**Alternatives considered**:
- Custom localization solution: More complex, reinventing existing functionality
- Third-party i18n libraries: Would require integration with Docusaurus, potentially causing conflicts
- Server-side locale detection: Less responsive, requires additional server infrastructure

## Decision: Locale Directory Structure
**Rationale**: Using Docusaurus's recommended i18n directory structure (i18n/ur/docusaurus-plugin-content-docs/current/) ensures compatibility with the build system and follows established conventions.

**Alternatives considered**:
- Custom directory structure: Would require custom build configuration and potentially break Docusaurus functionality
- Flat file structure: Would not properly separate localized content

## Decision: Authentication Integration Method
**Rationale**: Using @better-auth/client directly in Docusaurus components allows for client-side session validation without additional server round-trips, providing a responsive user experience.

**Alternatives considered**:
- Server-side validation only: Would require additional API endpoints and slower redirects
- Custom JWT parsing: Would duplicate functionality already provided by Better Auth
- Session cookies: Would not integrate as seamlessly with the existing auth system

## Decision: RTL Implementation Strategy
**Rationale**: Using Docusaurus's built-in RTL support with CSS direction property ensures proper text rendering and layout for Urdu content.

**Alternatives considered**:
- Custom RTL CSS: Would be more complex and error-prone
- External RTL libraries: Would add unnecessary dependencies
- Manual text direction: Would not handle all layout considerations

## Decision: Locale Routing Component Architecture
**Rationale**: Building a dedicated LocaleRouter component with TranslateButton sub-component provides a clean, reusable solution that can be easily integrated into Docusaurus documentation pages.

**Alternatives considered**:
- Modifying existing Docusaurus theme components: Would be harder to maintain
- Global navigation solution: Would be less targeted to documentation pages
- Pure CSS/HTML solution: Would not handle authentication validation

## Technical Implementation Notes

### Docusaurus i18n Configuration
- Configure `i18n` object in `docusaurus.config.js` with `en` and `ur` locales
- Set `ur` locale with `direction: 'rtl'` property
- Ensure proper fallback behavior for missing translations

### File Structure
- Place Urdu markdown files in `i18n/ur/docusaurus-plugin-content-docs/current/`
- Maintain the same directory structure as English documentation
- Implement build-time validation to ensure translation completeness

### Auth Integration
- Use `authClient.useSession()` hook to check authentication state
- Implement redirect logic to login page if user is not authenticated
- Preserve session state during locale switching

### Routing Implementation
- Use `@docusaurus/router` for client-side navigation
- Calculate current path and construct target Urdu path
- Handle missing translations with appropriate fallbacks