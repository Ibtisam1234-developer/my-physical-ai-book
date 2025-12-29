import express from "express";
import cors from "cors";
import { Pool } from "pg";
import bcrypt from "bcrypt";
import * as jose from "jose";

const app = express();

// Environment variables
const DATABASE_URL = process.env.DATABASE_URL || "postgresql://neondb_owner:npg_bG3sNI7uROUA@ep-late-bird-agnqz2sx-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require";
const JWT_SECRET = process.env.JWT_SECRET || "your-jwt-secret-change-in-production";
const ISSUER = process.env.ISSUER || "physical-ai-auth-server";
const PORT = parseInt(process.env.PORT || "3001");
const FRONTEND_URL = process.env.FRONTEND_URL || "http://localhost:3000";
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

// Database connection
const pool = new Pool({
  connectionString: DATABASE_URL,
  ssl: { rejectUnauthorized: false },
});

// Middleware
app.use(cors({
  origin: [FRONTEND_URL, BACKEND_URL],
  credentials: true,
}));
app.use(express.json());

// Generate RSA key pair for RS256 (done once at startup)
let keyPair: jose.KeyPair | null = null;
let publicKeyJWK: any = null;

async function getKeyPair() {
  if (!keyPair) {
    const keys = await jose.generateKeyPair("RS256");
    keyPair = keys;
    publicKeyJWK = await jose.exportJWK(keys.publicKey);
  }
  return keyPair;
}

// Generate JWT token
async function generateToken(user: { id: string; email: string; software_background: string; hardware_background: string }) {
  const keys = await getKeyPair();

  const token = await new jose.SignJWT({
    sub: user.id,
    email: user.email,
    software_background: user.software_background,
    hardware_background: user.hardware_background,
  })
    .setProtectedHeader({ alg: "RS256", kid: "physical-ai-key-1" })
    .setIssuedAt()
    .setIssuer(ISSUER)
    .setExpirationTime("1h")
    .sign(keys.privateKey);

  return token;
}

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "healthy", timestamp: new Date().toISOString() });
});

// JWKS endpoint
app.get("/.well-known/jwks.json", async (req, res) => {
  try {
    await getKeyPair();

    const jwks = {
      keys: [
        {
          alg: "RS256",
          use: "sig",
          kty: "RSA",
          kid: "physical-ai-key-1",
          ...publicKeyJWK,
        },
      ],
    };

    res.json(jwks);
  } catch (error) {
    console.error("JWKS error:", error);
    res.status(500).json({ error: "Failed to generate JWKS" });
  }
});

// Signup endpoint
app.post("/api/auth/sign-up", async (req, res) => {
  try {
    const { email, password, software_background, hardware_background } = req.body;

    // Validate required fields
    if (!email || !password || !software_background || !hardware_background) {
      return res.status(400).json({ error: "All fields are required" });
    }

    // Validate enum values
    const validSoftwareBg = ["beginner", "intermediate", "advanced", "expert"];
    const validHardwareBg = ["none", "hobbyist", "student", "professional"];

    if (!validSoftwareBg.includes(software_background)) {
      return res.status(400).json({ error: "Invalid software_background" });
    }
    if (!validHardwareBg.includes(hardware_background)) {
      return res.status(400).json({ error: "Invalid hardware_background" });
    }

    // Check if user exists
    const existingUser = await pool.query(
      "SELECT id FROM users WHERE email = $1",
      [email]
    );

    if (existingUser.rows.length > 0) {
      return res.status(409).json({ error: "User already exists" });
    }

    // Hash password
    const passwordHash = await bcrypt.hash(password, 12);

    // Create user
    const result = await pool.query(
      `INSERT INTO users (email, password_hash, software_background, hardware_background)
       VALUES ($1, $2, $3, $4)
       RETURNING id, email, software_background, hardware_background, created_at`,
      [email, passwordHash, software_background, hardware_background]
    );

    const user = result.rows[0];

    // Generate token
    const token = await generateToken({
      id: user.id,
      email: user.email,
      software_background: user.software_background,
      hardware_background: user.hardware_background,
    });

    res.status(201).json({
      user: {
        id: user.id,
        email: user.email,
        softwareBackground: user.software_background,
        hardwareBackground: user.hardware_background,
      },
      token,
    });
  } catch (error) {
    console.error("Signup error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Signin endpoint
app.post("/api/auth/sign-in", async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    // Find user
    const result = await pool.query(
      "SELECT * FROM users WHERE email = $1",
      [email]
    );

    if (result.rows.length === 0) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const user = result.rows[0];

    // Verify password
    const validPassword = await bcrypt.compare(password, user.password_hash);
    if (!validPassword) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    // Generate token
    const token = await generateToken({
      id: user.id,
      email: user.email,
      software_background: user.software_background,
      hardware_background: user.hardware_background,
    });

    res.json({
      user: {
        id: user.id,
        email: user.email,
        softwareBackground: user.software_background,
        hardwareBackground: user.hardware_background,
      },
      token,
    });
  } catch (error) {
    console.error("Signin error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Get current user
app.get("/api/auth/me", async (req, res) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return res.status(401).json({ error: "No token provided" });
    }

    const token = authHeader.slice(7);

    // Verify token
    const keys = await getKeyPair();

    const payload = await jose.jwtVerify(token, keys.publicKey, {
      issuer: ISSUER,
    });

    // Get user from database
    const result = await pool.query(
      "SELECT id, email, software_background, hardware_background FROM users WHERE id = $1",
      [payload.payload.sub]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: "User not found" });
    }

    const user = result.rows[0];
    res.json({
      user: {
        id: user.id,
        email: user.email,
        softwareBackground: user.software_background,
        hardwareBackground: user.hardware_background,
      },
    });
  } catch (error) {
    console.error("Get user error:", error);
    res.status(401).json({ error: "Invalid token" });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Auth server running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`JWKS: http://localhost:${PORT}/.well-known/jwks.json`);
  console.log(`Signup: POST http://localhost:${PORT}/api/auth/sign-up`);
  console.log(`Signin: POST http://localhost:${PORT}/api/auth/sign-in`);
  console.log(`Me: GET http://localhost:${PORT}/api/auth/me`);
});
