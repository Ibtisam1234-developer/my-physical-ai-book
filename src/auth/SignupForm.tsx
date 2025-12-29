import React, { useState } from "react";
import { useAuth } from "./context";
import type { SoftwareBackground, HardwareBackground } from "./types";

const softwareBackgroundOptions: { value: SoftwareBackground; label: string }[] = [
  { value: "beginner", label: "Beginner - New to programming" },
  { value: "intermediate", label: "Intermediate - Some experience" },
  { value: "advanced", label: "Advanced - Professional experience" },
  { value: "expert", label: "Expert - Deep expertise" },
];

const hardwareBackgroundOptions: { value: HardwareBackground; label: string }[] = [
  { value: "none", label: "None - No hardware experience" },
  { value: "hobbyist", label: "Hobbyist - Personal projects" },
  { value: "student", label: "Student - Learning in school" },
  { value: "professional", label: "Professional - Work experience" },
];

interface SignupFormProps {
  onSuccess?: () => void;
}

export function SignupForm({ onSuccess }: SignupFormProps) {
  const { signup, isLoading } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [softwareBackground, setSoftwareBackground] = useState<SoftwareBackground>("beginner");
  const [hardwareBackground, setHardwareBackground] = useState<HardwareBackground>("none");
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }

    try {
      await signup({
        email,
        password,
        softwareBackground,
        hardwareBackground,
      });
      onSuccess?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Signup failed");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="auth-form">
      <h2>Create Account</h2>

      {error && <div className="auth-error">{error}</div>}

      <div className="form-group">
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          placeholder="your@email.com"
        />
      </div>

      <div className="form-group">
        <label htmlFor="password">Password</label>
        <div className="password-input">
          <input
            id="password"
            type={showPassword ? "text" : "password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            placeholder="At least 8 characters"
          />
          <button
            type="button"
            className="toggle-password"
            onClick={() => setShowPassword(!showPassword)}
          >
            {showPassword ? "Hide" : "Show"}
          </button>
        </div>
      </div>

      <div className="form-group">
        <label htmlFor="confirmPassword">Confirm Password</label>
        <input
          id="confirmPassword"
          type={showPassword ? "text" : "password"}
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required
          placeholder="Confirm your password"
        />
      </div>

      <div className="form-group">
        <label htmlFor="softwareBackground">Software Experience</label>
        <select
          id="softwareBackground"
          value={softwareBackground}
          onChange={(e) => setSoftwareBackground(e.target.value as SoftwareBackground)}
        >
          {softwareBackgroundOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="hardwareBackground">Hardware Experience</label>
        <select
          id="hardwareBackground"
          value={hardwareBackground}
          onChange={(e) => setHardwareBackground(e.target.value as HardwareBackground)}
        >
          {hardwareBackgroundOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      <button type="submit" disabled={isLoading} className="auth-submit">
        {isLoading ? "Creating account..." : "Create Account"}
      </button>
    </form>
  );
}
