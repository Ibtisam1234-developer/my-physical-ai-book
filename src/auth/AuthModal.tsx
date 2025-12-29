import React, { useState } from "react";
import { useAuth } from "./context";
import { SignupForm } from "./SignupForm";
import { SigninForm } from "./SigninForm";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialMode?: "signin" | "signup";
}

export function AuthModal({ isOpen, onClose, initialMode = "signin" }: AuthModalProps) {
  const [mode, setMode] = useState<"signin" | "signup">(initialMode);
  const { isAuthenticated, user, logout } = useAuth();

  if (!isOpen) return null;

  if (isAuthenticated) {
    return (
      <div className="auth-modal-overlay" onClick={onClose}>
        <div className="auth-modal" onClick={(e) => e.stopPropagation()}>
          <button className="auth-close" onClick={onClose}>×</button>
          <div className="auth-user-info">
            <h2>Welcome back!</h2>
            <p>{user?.email}</p>
            <div className="auth-badge">
              <span className="badge software">{user?.softwareBackground}</span>
              <span className="badge hardware">{user?.hardwareBackground}</span>
            </div>
          </div>
          <button onClick={logout} className="auth-logout">
            Sign Out
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-modal-overlay" onClick={onClose}>
      <div className="auth-modal" onClick={(e) => e.stopPropagation()}>
        <button className="auth-close" onClick={onClose}>×</button>

        <div className="auth-tabs">
          <button
            className={`auth-tab ${mode === "signin" ? "active" : ""}`}
            onClick={() => setMode("signin")}
          >
            Sign In
          </button>
          <button
            className={`auth-tab ${mode === "signup" ? "active" : ""}`}
            onClick={() => setMode("signup")}
          >
            Create Account
          </button>
        </div>

        {mode === "signin" ? (
          <SigninForm onSuccess={onClose} />
        ) : (
          <SignupForm onSuccess={onClose} />
        )}
      </div>
    </div>
  );
}
