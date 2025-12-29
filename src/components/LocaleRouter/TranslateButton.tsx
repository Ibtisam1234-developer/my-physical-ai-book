import React from 'react';
import { useLocation, useHistory } from '@docusaurus/router';

export default function TranslateRouteButton() {
  const location = useLocation();
  const history = useHistory();

  // Check if user is signed in using the same keys as auth context
  const authToken = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
  const authUser = typeof window !== 'undefined' ? localStorage.getItem('auth_user') : null;
  const isSignedIn = !!(authToken && authUser);

  const handleRouting = () => {
    if (!isSignedIn) {
      alert("Please sign in to access the Urdu translation and earn your bonus points!");
      return;
    }

    const currentPath = location.pathname;
    const isCurrentlyUrdu = currentPath.startsWith('/ur/');

    let targetPath;
    if (isCurrentlyUrdu) {
      // Urdu to English: /ur/docs/intro/intro -> /docs/intro/
      // /ur/docs/module-1-ros2/intro -> /docs/module-1-ros2/intro
      const pathWithoutUr = currentPath.replace(/^\/ur/, '');
      // For intro pages specifically, convert /docs/intro/intro back to /docs/intro/
      if (pathWithoutUr === '/docs/intro/intro') {
        targetPath = '/docs/intro/';
      } else {
        targetPath = pathWithoutUr;
      }
    } else {
      // English to Urdu: /docs/intro/ -> /ur/docs/intro/intro
      // /docs/module-1-ros2/intro -> /ur/docs/module-1-ros2/intro
      // For intro page specifically, convert /docs/intro/ to /ur/docs/intro/intro
      if (currentPath === '/docs/intro/') {
        targetPath = '/ur/docs/intro/intro';
      } else {
        targetPath = `/ur${currentPath}`;
      }
    }

    // Force page refresh to ensure proper routing
    window.location.href = targetPath;
  };

  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <button
        onClick={handleRouting}
        className={`button button--primary ${isSignedIn ? '' : 'button--outline'}`}
        style={{ fontWeight: 'bold' }}
        aria-label={
          location.pathname.startsWith('/ur/')
            ? "Switch to English Version"
            : "Switch to Urdu Version"
        }
        title={
          location.pathname.startsWith('/ur/')
            ? "Switch to English Version"
            : "اردو میں پڑھیں (Read in Urdu)"
        }
      >
        {location.pathname.startsWith('/ur/')
          ? "Switch to English Version"
          : "اردو میں پڑھیں (Read in Urdu)"}
      </button>
    </div>
  );
}