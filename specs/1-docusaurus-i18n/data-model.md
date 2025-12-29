# Data Model: Docusaurus i18n Localization

## Locale Configuration
**Description**: Represents the language and text direction settings for the Docusaurus site
- **Properties**:
  - `locale` (string): Language code (e.g., 'en', 'ur')
  - `label` (string): Display name for the locale (e.g., 'English', 'اردو')
  - `direction` (string): Text direction ('ltr' or 'rtl')
- **Validation**: Must be a valid language code, direction must be 'ltr' or 'rtl'
- **Relationships**: Configures the Docusaurus i18n system

## Authentication Session
**Description**: Represents the user's JWT-based authentication state
- **Properties**:
  - `userId` (string): Unique identifier for the authenticated user
  - `email` (string): User's email address
  - `expiresAt` (Date): Session expiration timestamp
  - `isValid` (boolean): Whether the session is currently valid
- **Validation**: Session must be valid and not expired
- **State transitions**: Active → Expired when token expires

## Documentation Path
**Description**: Represents the URL structure that maps between English and Urdu documentation pages
- **Properties**:
  - `englishPath` (string): Original English documentation path (e.g., '/docs/intro')
  - `urduPath` (string): Corresponding Urdu documentation path (e.g., '/ur/docs/intro')
  - `hasTranslation` (boolean): Whether a Urdu translation exists for this path
  - `fallbackPath` (string): Path to use if Urdu translation is missing
- **Validation**: Paths must follow proper URL format, Urdu path must begin with '/ur'
- **Relationships**: Maps between locales for the same content

## Translation Status
**Description**: Represents the status of translation completeness for documentation
- **Properties**:
  - `totalDocuments` (number): Total number of English documents
  - `translatedDocuments` (number): Number of documents with Urdu translations
  - `completionPercentage` (number): Percentage of documents translated
  - `missingTranslations` (array): List of paths without Urdu translations
- **Validation**: Completion percentage must be between 0 and 100
- **State transitions**: Changes as translations are added or removed