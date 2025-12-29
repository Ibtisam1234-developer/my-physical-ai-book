# Data Model: Authentication & Personalization System

**Date**: 2025-12-27 (Updated with better-auth schema notes)
**Branch**: `001-auth-personalization`
**Database**: Neon PostgreSQL (serverless)
**Auth Framework**: better-auth with neonAdapter (auto-generates base schema, custom fields added via additionalFields config)

---

## Entity Relationship Diagram

```
┌─────────────────────────┐
│ users                   │
├─────────────────────────┤
│ id (PK) UUID            │
│ email VARCHAR UNIQUE    │
│ password_hash VARCHAR   │
│ software_background ENUM│
│ hardware_background ENUM│
│ created_at TIMESTAMP    │
│ updated_at TIMESTAMP    │
└─────────────────────────┘
            │
            │ 1
            │
            │
            │ *
            ▼
┌─────────────────────────┐
│ personalized_content    │
├─────────────────────────┤
│ id (PK) UUID            │
│ user_id (FK) UUID       │
│ content_hash VARCHAR(64)│
│ content_type VARCHAR(50)│
│ content_payload JSONB   │
│ generated_at TIMESTAMP  │
│ expires_at TIMESTAMP    │
└─────────────────────────┘
```

**Relationship**: One user has many personalized content entries. Cascade delete: when user is deleted, all their personalized content is also deleted (GDPR compliance).

---

## Entity: users

### Purpose
Stores user account information for authentication. Created during signup (FR-001), read during signin (FR-004), and referenced by personalized_content for cache lookups.

**Implementation Note**: better-auth with neonAdapter auto-generates the base users table. Custom fields `softwareBackground` and `hardwareBackground` are added via `additionalFields` configuration (see research.md R2). The schema below represents the expected final structure after better-auth initialization.

### Schema

```sql
CREATE TYPE software_background_enum AS ENUM ('beginner', 'intermediate', 'advanced', 'expert');
CREATE TYPE hardware_background_enum AS ENUM ('none', 'hobbyist', 'student', 'professional');

CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,  -- bcrypt with cost factor 12
  software_background software_background_enum NOT NULL,
  hardware_background hardware_background_enum NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index for fast email lookups during signin
CREATE INDEX idx_users_email ON users(email);
```

### Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique user identifier, used in JWT claims (user_id) |
| `email` | VARCHAR(255) | NOT NULL, UNIQUE | User's email address, used for signin |
| `password_hash` | VARCHAR(255) | NOT NULL | bcrypt hash of password (cost factor 12 per FR-003) |
| `software_background` | ENUM | NOT NULL | User's software experience level: beginner \| intermediate \| advanced \| expert (per Constitution VII) |
| `hardware_background` | ENUM | NOT NULL | User's hardware experience level: none \| hobbyist \| student \| professional (per Constitution VII) |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Account creation timestamp |
| `updated_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Last profile update timestamp (for future profile editing feature) |

### Validation Rules (Application Layer)

These are enforced in the auth server before database insertion:

1. **email**: Must match RFC 5322 email format (validated via Fastify schema `format: 'email'`)
2. **password**: Minimum 8 characters (per User Story 1, Scenario 5)
3. **software_background**: Must be one of: `beginner`, `intermediate`, `advanced`, `expert` (per FR-002)
4. **hardware_background**: Must be one of: `none`, `hobbyist`, `student`, `professional` (per FR-002)

### State Transitions

Users table is effectively immutable after creation (per Assumption 3 in spec: "Profile editing is out of scope"). No state transitions.

Future enhancement (out of scope): `updated_at` would be modified if profile editing is implemented.

### PII Classification

**PII Fields** (per Constitution VII, FR-025):
- `software_background` (MUST NOT be logged)
- `hardware_background` (MUST NOT be logged)

**Non-PII Fields**:
- `id` (can be logged for audit purposes)
- `email` (can be logged for audit purposes with caution)
- `created_at`, `updated_at` (can be logged)

**Important**: `password_hash` is sensitive but not PII. It should not be logged, but for different security reasons (not constitution PII rules).

### Indexes

1. **Primary Key**: `id` (default B-tree index)
   - Used for: Foreign key references from personalized_content
   - Query pattern: `SELECT * FROM users WHERE id = $1`

2. **Unique Constraint**: `email` (automatically indexed)
   - Used for: Signin lookup, duplicate email detection during signup
   - Query pattern: `SELECT id, password_hash, software_background, hardware_background FROM users WHERE email = $1`

### Sample Data

```sql
INSERT INTO users (email, password_hash, software_background, hardware_background) VALUES
  ('student@example.com', '$2b$12$...', 'intermediate', 'hobbyist'),
  ('expert@example.com', '$2b$12$...', 'expert', 'professional');
```

---

## Entity: personalized_content

### Purpose
Caches LLM-generated personalized content to avoid redundant API calls. Enables 60% cost reduction (SC-006) by serving repeated personalization requests from cache instead of regenerating.

### Schema

```sql
CREATE TYPE content_type_enum AS ENUM ('curriculum_path', 'difficulty_level', 'recommended_resources');

CREATE TABLE personalized_content (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  content_hash VARCHAR(64) NOT NULL,  -- SHA-256 hex string of chapter markdown
  content_type content_type_enum NOT NULL DEFAULT 'curriculum_path',
  content_payload JSONB NOT NULL,     -- { "personalized_text": "...", "model": "gemini-2.0-flash-exp", "tokens": 1234 }
  generated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMP NOT NULL DEFAULT NOW() + INTERVAL '7 days',

  -- Composite unique constraint for cache deduplication
  CONSTRAINT uq_personalized_content UNIQUE (user_id, content_hash, content_type)
);

-- Index for fast cache lookups (automatically created by UNIQUE constraint)
-- Index: idx_personalized_content_user_id_content_hash_content_type

-- Index for cache cleanup query (delete expired entries)
CREATE INDEX idx_personalized_content_expires_at ON personalized_content(expires_at);

-- Index for user's personalized content history (future feature: "My Personalizations" page)
CREATE INDEX idx_personalized_content_user_id ON personalized_content(user_id);
```

### Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique identifier for cache entry |
| `user_id` | UUID | NOT NULL, FOREIGN KEY (users.id) ON DELETE CASCADE | User who requested personalization |
| `content_hash` | VARCHAR(64) | NOT NULL | SHA-256 hash of chapter markdown (hex string, 64 chars). Used for cache lookups (FR-019). |
| `content_type` | ENUM | NOT NULL, DEFAULT 'curriculum_path' | Type of personalized content: curriculum_path \| difficulty_level \| recommended_resources (per FR-021) |
| `content_payload` | JSONB | NOT NULL | Personalized content and metadata. Schema: `{ "personalized_text": string, "model": string, "tokens": number, "original_length": number }` |
| `generated_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | When content was generated (for analytics) |
| `expires_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() + INTERVAL '7 days' | Cache expiration timestamp (7 days per Constitution VII) |

### Validation Rules (Application Layer)

1. **content_hash**: Must be 64-character lowercase hexadecimal string (SHA-256 output)
2. **content_payload.personalized_text**: Must be non-empty string
3. **expires_at**: Must be > generated_at (enforced via DEFAULT + INTERVAL)

### State Transitions

Cache entries are immutable. No updates, only inserts and deletes.

**State Flow**:
```
[Non-existent] ---(generate personalized content)---> [Cached]
                                                         │
                                                         │ (7 days)
                                                         ▼
                                                      [Expired] ---(cron job)---> [Deleted]
```

**Cache Invalidation Events** (future enhancements, not in MVP):
- User profile update (if profile editing is implemented)
- Manual "clear cache" action by user

### JSONB Structure: content_payload

```json
{
  "personalized_text": "For someone with intermediate software skills and hobbyist hardware experience, ROS 2 Navigation Stack builds on concepts you may know from...",
  "model": "gemini-2.0-flash-exp",
  "tokens": 1234,
  "original_length": 5678,
  "generated_at_iso": "2025-12-27T12:34:56Z"
}
```

**Fields**:
- `personalized_text` (string, required): LLM-generated tailored explanation
- `model` (string, required): LLM model used (for debugging/analytics)
- `tokens` (number, optional): Token count from LLM API response (for cost tracking)
- `original_length` (number, optional): Character length of original chapter markdown
- `generated_at_iso` (string, optional): ISO 8601 timestamp (redundant with `generated_at`, but useful for JSONB queries)

### Indexes

1. **Primary Key**: `id` (default B-tree index)
   - Used for: Direct row access (rare, mostly for admin queries)

2. **Unique Constraint**: `(user_id, content_hash, content_type)` (composite B-tree index)
   - Used for: Cache hit lookups (most common query)
   - Query pattern:
     ```sql
     SELECT content_payload
     FROM personalized_content
     WHERE user_id = $1
       AND content_hash = $2
       AND content_type = 'curriculum_path'
       AND expires_at > NOW()
     LIMIT 1;
     ```

3. **Index**: `expires_at` (B-tree index)
   - Used for: Cache cleanup cron job
   - Query pattern:
     ```sql
     DELETE FROM personalized_content
     WHERE expires_at < NOW();
     ```

4. **Index**: `user_id` (B-tree index)
   - Used for: Future feature - user's personalization history page
   - Query pattern:
     ```sql
     SELECT content_type, content_payload->>'personalized_text' AS text, generated_at
     FROM personalized_content
     WHERE user_id = $1
       AND expires_at > NOW()
     ORDER BY generated_at DESC
     LIMIT 20;
     ```

### Sample Data

```sql
INSERT INTO personalized_content (user_id, content_hash, content_type, content_payload) VALUES
  (
    '123e4567-e89b-12d3-a456-426614174000',
    'a3d2e1f4b5c6d7e8f9a0b1c2d3e4f5g6h7i8j9k0l1m2n3o4p5q6r7s8t9u0v1w2x3y4',
    'curriculum_path',
    '{
      "personalized_text": "For intermediate software and hobbyist hardware background...",
      "model": "gemini-2.0-flash-exp",
      "tokens": 856,
      "original_length": 3245
    }'::jsonb
  );
```

---

## Database Migration Strategy

### Initial Setup (MVP)

```sql
-- migration_001_initial_schema.sql

BEGIN;

-- Create enums
CREATE TYPE software_background_enum AS ENUM ('beginner', 'intermediate', 'advanced', 'expert');
CREATE TYPE hardware_background_enum AS ENUM ('none', 'hobbyist', 'student', 'professional');
CREATE TYPE content_type_enum AS ENUM ('curriculum_path', 'difficulty_level', 'recommended_resources');

-- Create users table
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  software_background software_background_enum NOT NULL,
  hardware_background hardware_background_enum NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);

-- Create personalized_content table
CREATE TABLE personalized_content (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  content_hash VARCHAR(64) NOT NULL,
  content_type content_type_enum NOT NULL DEFAULT 'curriculum_path',
  content_payload JSONB NOT NULL,
  generated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMP NOT NULL DEFAULT NOW() + INTERVAL '7 days',
  CONSTRAINT uq_personalized_content UNIQUE (user_id, content_hash, content_type)
);

CREATE INDEX idx_personalized_content_expires_at ON personalized_content(expires_at);
CREATE INDEX idx_personalized_content_user_id ON personalized_content(user_id);

COMMIT;
```

### Rollback Script

```sql
-- rollback_001_initial_schema.sql

BEGIN;

DROP TABLE IF EXISTS personalized_content CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TYPE IF EXISTS content_type_enum CASCADE;
DROP TYPE IF EXISTS hardware_background_enum CASCADE;
DROP TYPE IF EXISTS software_background_enum CASCADE;

COMMIT;
```

### Migration Execution

**Local Development**:
```bash
psql $DATABASE_URL -f specs/001-auth-personalization/migrations/001_initial_schema.sql
```

**Production** (requires human approval per Constitution Principle VI):
1. Review migration script for correctness
2. Test on staging environment (Neon branch database)
3. Schedule maintenance window (if needed)
4. Run migration: `psql $PRODUCTION_DATABASE_URL -f 001_initial_schema.sql`
5. Verify tables created: `\dt` in psql
6. Verify indexes created: `\di` in psql

---

## Query Patterns & Performance

### Query 1: Signup (Insert User)

```sql
INSERT INTO users (email, password_hash, software_background, hardware_background)
VALUES ($1, $2, $3, $4)
RETURNING id, email, software_background, hardware_background;
```

**Expected Performance**: <10ms (simple insert with UNIQUE constraint check on email)

**Indexes Used**: `users_email_key` (unique constraint, checks for duplicate email)

---

### Query 2: Signin (Lookup User by Email)

```sql
SELECT id, password_hash, software_background, hardware_background
FROM users
WHERE email = $1;
```

**Expected Performance**: <5ms (index lookup on email, returns 1 row)

**Indexes Used**: `idx_users_email` (B-tree index on email column)

---

### Query 3: Cache Lookup (Check for Existing Personalized Content)

```sql
SELECT content_payload
FROM personalized_content
WHERE user_id = $1
  AND content_hash = $2
  AND content_type = 'curriculum_path'
  AND expires_at > NOW()
LIMIT 1;
```

**Expected Performance**: <10ms (composite unique index scan, checks 1-4 conditions)

**Indexes Used**: `uq_personalized_content` (composite unique index on user_id, content_hash, content_type)

**Cache Hit Rate Goal**: 60% (per SC-006)

---

### Query 4: Cache Insert (Store New Personalized Content)

```sql
INSERT INTO personalized_content (user_id, content_hash, content_type, content_payload)
VALUES ($1, $2, $3, $4)
ON CONFLICT (user_id, content_hash, content_type) DO NOTHING
RETURNING id;
```

**Expected Performance**: <15ms (insert with UNIQUE constraint check)

**Indexes Used**: `uq_personalized_content` (checks for duplicate before insert)

**Note**: `ON CONFLICT DO NOTHING` handles race condition where two requests for the same chapter arrive simultaneously. Only one will be inserted; the other will be ignored.

---

### Query 5: Cache Cleanup (Delete Expired Entries)

```sql
DELETE FROM personalized_content
WHERE expires_at < NOW();
```

**Expected Performance**: <100ms for 1,000 expired rows

**Indexes Used**: `idx_personalized_content_expires_at` (B-tree index for fast range scan)

**Execution Frequency**: Daily cron job at 3:00 AM UTC (low-traffic period)

---

## Data Retention & Privacy

### GDPR Compliance

**Right to Erasure ("Right to be Forgotten")**:
```sql
DELETE FROM users WHERE id = $1;
-- Automatically cascades to personalized_content due to ON DELETE CASCADE
```

**Data Export**:
```sql
SELECT
  u.email,
  u.software_background,
  u.hardware_background,
  u.created_at,
  pc.content_type,
  pc.content_payload->>'personalized_text' AS personalized_text,
  pc.generated_at
FROM users u
LEFT JOIN personalized_content pc ON u.id = pc.user_id
WHERE u.id = $1;
```

### Data Anonymization (Development/Staging)

For non-production environments, anonymize PII fields:

```sql
UPDATE users SET
  email = CONCAT('user', id::text, '@example.com'),
  software_background = (ARRAY['beginner', 'intermediate', 'advanced', 'expert'])[floor(random() * 4 + 1)],
  hardware_background = (ARRAY['none', 'hobbyist', 'student', 'professional'])[floor(random() * 4 + 1)];
```

---

## Backup & Recovery Strategy

**Neon PostgreSQL Automatic Backups**:
- Point-in-time recovery (PITR) enabled
- 7-day retention window
- Backups encrypted at rest (per FR-027)

**Manual Backup** (before major migrations):
```bash
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql
```

**Recovery Test** (quarterly):
1. Create Neon branch database from backup
2. Run application tests against branch
3. Verify data integrity
4. Document recovery time objective (RTO)

---

## Next Steps

Data model complete. Proceed to:
1. Generate API contracts (`contracts/auth-api.yaml`, `contracts/personalization-api.yaml`)
2. Generate quickstart guide (`quickstart.md`)
3. Update agent context with schema details
