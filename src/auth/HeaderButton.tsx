import React, { useState } from "react";
import { useAuth } from "./context";
import { AuthModal } from "./AuthModal";

interface AuthHeaderButtonProps {
  showLabels?: boolean;
}

export function AuthHeaderButton({ showLabels = true }: AuthHeaderButtonProps) {
  const { isAuthenticated, user } = useAuth();
  const [isModalOpen, setIsModalOpen] = useState(false);

  if (isAuthenticated && user) {
    return (
      <>
        <button className="auth-header-button" onClick={() => setIsModalOpen(true)}>
          <div className="auth-avatar">{user.email[0].toUpperCase()}</div>
          {showLabels && <span>{user.email}</span>}
        </button>
        <AuthModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          initialMode="signin"
        />
      </>
    );
  }

  return (
    <>
      <button className="auth-header-button" onClick={() => setIsModalOpen(true)}>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
          <polyline points="10 17 15 12 10 7" />
          <line x1="15" y1="12" x2="3" y2="12" />
        </svg>
        {showLabels && <span>Sign In</span>}
      </button>

      <AuthModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </>
  );
}
