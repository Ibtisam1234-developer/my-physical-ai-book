import React, { useState, useEffect } from "react";
import { useAuth } from "./context";

interface PersonalizeButtonProps {
  chapterId: string;
  chapterTitle: string;
  onPersonalize?: (content: string) => void;
  className?: string;
}

export function PersonalizeButton({
  chapterId,
  chapterTitle,
  onPersonalize,
  className = "",
}: PersonalizeButtonProps) {
  const { isAuthenticated, user, isLoading } = useAuth();
  const [isPersonalizing, setIsPersonalizing] = useState(false);
  const [error, setError] = useState("");
  const [showPersonalizedContent, setShowPersonalizedContent] = useState(false);

  // Load personalized content from localStorage based on chapterId
  const [personalizedContent, setPersonalizedContent] = useState<string | null>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem(`personalized_content_${chapterId}`) || null;
    }
    return null;
  });

  // Effect to clear content when chapterId changes (new page)
  useEffect(() => {
    const storedContent = localStorage.getItem(`personalized_content_${chapterId}`);
    setPersonalizedContent(storedContent || null);
  }, [chapterId]);

  const togglePersonalizedContent = () => {
    setShowPersonalizedContent(!showPersonalizedContent);
  };

  const clearPersonalizedContent = () => {
    setPersonalizedContent(null);
    if (typeof window !== 'undefined') {
      localStorage.removeItem(`personalized_content_${chapterId}`);
    }
    setShowPersonalizedContent(false);
  };

  const handlePersonalize = async () => {
    if (!isAuthenticated || !user) {
      setError("Please sign in to personalize content");
      return;
    }

    setIsPersonalizing(true);
    setError("");

    try {
      // Call the backend personalization API directly at port 8000
      const response = await fetch("http://localhost:8000/api/personalize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // Token is stored in localStorage by auth context
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
        body: JSON.stringify({
          chapterId,
          chapterTitle,
          softwareBackground: user.softwareBackground,
          hardwareBackground: user.hardwareBackground,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Failed to personalize");
      }

      const data = await response.json();
      setPersonalizedContent(data.content);
      // Save to localStorage to persist across page navigations in the session
      if (typeof window !== 'undefined') {
        localStorage.setItem(`personalized_content_${chapterId}`, data.content);
      }
      onPersonalize?.(data.content);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Personalization failed");
    } finally {
      setIsPersonalizing(false);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className={`personalize-button ${className}`}>
        <button
          className="personalize-btn"
          disabled={isLoading}
          onClick={() => setError("Please sign in to personalize this content")}
        >
          <span className="personalize-icon">✨</span>
          Personalize
        </button>
        {error && <p className="personalize-error">{error}</p>}
      </div>
    );
  }

  return (
    <div className={`personalize-button ${className}`}>
      <button
        className="personalize-btn"
        onClick={handlePersonalize}
        disabled={isPersonalizing || isLoading}
      >
        {isPersonalizing ? (
          <>
            <span className="spinner" />
            Personalizing...
          </>
        ) : (
          <>
            <span className="personalize-icon">✨</span>
            Personalize for {user?.softwareBackground}
          </>
        )}
      </button>

      {error && <p className="personalize-error">{error}</p>}

      {personalizedContent && (
        <div className="personalized-content-summary">
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '10px' }}>
            <button
              className="personalized-content-toggle"
              onClick={togglePersonalizedContent}
            >
              {showPersonalizedContent ? 'Hide Personalized Content' : 'Show Personalized Content'}
            </button>
            <button
              className="personalized-content-clear"
              onClick={clearPersonalizedContent}
              style={{
                padding: '0.25rem 0.5rem',
                backgroundColor: '#dc2626',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.75rem'
              }}
            >
              Regenerate
            </button>
          </div>
          {showPersonalizedContent && (
            <div className="personalized-content-full">
              <h4>Personalized for your level:</h4>
              <div
                className="personalized-text"
                dangerouslySetInnerHTML={{ __html: personalizedContent }}
              />
            </div>
          )}
        </div>
      )}

      <div className="personalize-badge">
        <span className="badge software">{user?.softwareBackground}</span>
        <span className="badge hardware">{user?.hardwareBackground}</span>
      </div>
    </div>
  );
}
