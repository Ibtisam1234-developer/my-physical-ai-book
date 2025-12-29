import { betterAuth } from "better-auth";
import { Pool } from "pg";

// Background enum types
export type SoftwareBackground = "beginner" | "intermediate" | "advanced" | "expert";
export type HardwareBackground = "none" | "hobbyist" | "student" | "professional";

// Extend the user type to include custom fields
declare module "better-auth" {
  interface User {
    softwareBackground?: SoftwareBackground;
    hardwareBackground?: HardwareBackground;
  }
}

export interface AuthConfig {
  databaseUrl: string;
  jwtSecret: string;
  issuer: string;
}

export function createAuth(config: AuthConfig) {
  // Create a PostgreSQL connection pool
  const pool = new Pool({
    connectionString: config.databaseUrl,
    ssl: { rejectUnauthorized: false },
  });

  return betterAuth({
    database: {
      provider: "postgresql",
      connection: pool,
    },
    user: {
      additionalFields: {
        softwareBackground: {
          type: "string",
          required: true,
          enum: ["beginner", "intermediate", "advanced", "expert"],
          input: true,
        },
        hardwareBackground: {
          type: "string",
          required: true,
          enum: ["none", "hobbyist", "student", "professional"],
          input: true,
        },
      },
    },
    emailAndPassword: {
      enabled: true,
      minPasswordLength: 8,
      maxPasswordLength: 128,
    },
    advanced: {
      cookiePrefix: "physical-ai",
    },
  });
}

export type AuthClient = ReturnType<typeof createAuth>;
