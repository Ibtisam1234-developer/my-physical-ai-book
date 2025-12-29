# Quickstart: Docusaurus i18n Localization

## Overview
This guide provides a quick introduction to implementing Docusaurus i18n localization with Urdu support and authentication validation.

## Prerequisites
- Docusaurus project already set up
- Better Auth authentication system configured
- Urdu translation files ready

## Setup Steps

### 1. Configure i18n in docusaurus.config.ts
The configuration has been updated to include Urdu locale with RTL direction:

```typescript
i18n: {
  defaultLocale: 'en',
  locales: ['en', 'ur'],
  localeConfigs: {
    en: { label: 'English', direction: 'ltr' },
    ur: { label: 'Urdu (اردو)', direction: 'rtl' },
  },
},
```

### 2. Organize Urdu translation files
Urdu markdown files are located in:
```
i18n/ur/docusaurus-plugin-content-docs/current/
```

### 3. Authentication Integration
The auth client is implemented in `src/lib/auth-client.ts` and communicates with the auth server to validate user sessions before allowing locale switching.

### 4. Translate Button Component
The TranslateButton component at `src/components/LocaleRouter/TranslateButton.tsx`:
- Checks authentication state with `authClient.useSession()`
- Calculates current path and constructs target Urdu path
- Handles routing with `@docusaurus/router`
- Provides proper accessibility attributes

### 5. Layout Integration
The translation button is automatically injected into all documentation pages through the theme override at `src/theme/DocItem/Layout/index.js`.

### 6. RTL Styling
RTL support is implemented in `src/css/custom.css` with specific CSS rules for right-to-left layout.

## Testing
- Verify that authenticated users can access Urdu content using the translate button
- Confirm that unauthenticated users see an alert when attempting to access Urdu content
- Test that RTL layout works correctly for Urdu content
- Ensure locale switching preserves session state
- Validate that path mapping works correctly (e.g., `/docs/intro` ↔ `/ur/docs/intro`)

## Common Issues
- Missing translation files causing 404 errors
- Authentication state not being properly validated
- RTL styling not applying correctly
- Session state not being preserved during locale switching
- Path construction not mapping correctly between locales

## Running the Application
To test the localization feature:
1. Start the auth server: `cd auth-server && npm run dev`
2. Start the Docusaurus site: `npm run start`
3. Navigate to a documentation page
4. Click the translate button (if authenticated) to switch to Urdu