/**
 * Database migration script for auth-server
 * Creates all tables required by better-auth
 */

import { neon } from "@neondatabase/serverless";

const DATABASE_URL = process.env.DATABASE_URL || "";

async function migrate() {
  if (!DATABASE_URL) {
    console.error("DATABASE_URL environment variable is required");
    process.exit(1);
  }

  const sql = neon(DATABASE_URL);

  try {
    console.log("Starting database migration...");

    // Create enum types for software_background
    await sql`
      DO $$ BEGIN
        CREATE TYPE software_background_enum AS ENUM ('beginner', 'intermediate', 'advanced', 'expert');
      EXCEPTION
        WHEN duplicate_object THEN null;
      END $$;
    `;
    console.log("Created/verified software_background_enum type");

    // Create enum types for hardware_background
    await sql`
      DO $$ BEGIN
        CREATE TYPE hardware_background_enum AS ENUM ('none', 'hobbyist', 'student', 'professional');
      EXCEPTION
        WHEN duplicate_object THEN null;
      END $$;
    `;
    console.log("Created/verified hardware_background_enum type");

    // Create users table (better-auth base schema)
    await sql`
      CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email VARCHAR(255) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        name VARCHAR(255),
        image VARCHAR(255),
        email_verified BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
      );
    `;
    console.log("Created/verified users table");

    // Create sessions table (better-auth requires this)
    await sql`
      CREATE TABLE IF NOT EXISTS sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token VARCHAR(255) NOT NULL UNIQUE,
        expires_at TIMESTAMP NOT NULL,
        ip_address VARCHAR(255),
        user_agent VARCHAR(255),
        created_at TIMESTAMP NOT NULL DEFAULT NOW()
      );
    `;
    console.log("Created/verified sessions table");

    // Create accounts table (for OAuth, optional but good to have)
    await sql`
      CREATE TABLE IF NOT EXISTS accounts (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        account_id VARCHAR(255) NOT NULL,
        provider_id VARCHAR(255) NOT NULL,
        access_token VARCHAR(255),
        refresh_token VARCHAR(255),
        access_token_expires_at TIMESTAMP,
        refresh_token_expires_at TIMESTAMP,
        password VARCHAR(255),
        scope VARCHAR(255),
        id_token VARCHAR(255),
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        UNIQUE(account_id, provider_id)
      );
    `;
    console.log("Created/verified accounts table");

    // Create verification table (for email verification)
    await sql`
      CREATE TABLE IF NOT EXISTS verification (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        identifier VARCHAR(255) NOT NULL,
        value VARCHAR(255) NOT NULL,
        expires_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
      );
    `;
    console.log("Created/verification table");

    // Add custom columns to users table if they don't exist
    await sql`
      DO $$ BEGIN
        IF NOT EXISTS (
          SELECT 1 FROM information_schema.columns
          WHERE table_name = 'users' AND column_name = 'software_background'
        ) THEN
          ALTER TABLE users ADD COLUMN software_background software_background_enum NOT NULL DEFAULT 'beginner';
        END IF;
      END $$;
    `;
    console.log("Added/verified software_background column");

    await sql`
      DO $$ BEGIN
        IF NOT EXISTS (
          SELECT 1 FROM information_schema.columns
          WHERE table_name = 'users' AND column_name = 'hardware_background'
        ) THEN
          ALTER TABLE users ADD COLUMN hardware_background hardware_background_enum NOT NULL DEFAULT 'none';
        END IF;
      END $$;
    `;
    console.log("Added/verified hardware_background column");

    // Create indexes
    await sql`CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)`;
    await sql`CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)`;
    await sql`CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token)`;
    await sql`CREATE INDEX IF NOT EXISTS idx_accounts_user_id ON accounts(user_id)`;
    console.log("Created indexes");

    console.log("Migration completed successfully!");
  } catch (error) {
    console.error("Migration failed:", error);
    process.exit(1);
  }
}

migrate();
