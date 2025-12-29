import React from 'react';
import { useLocation } from '@docusaurus/router';
import Layout from '@theme-original/Layout';
import ChatBot from '@site/src/components/ChatBot/ChatBot';
import { AuthProvider } from '@site/src/auth';
import DocPersonalizeButton from '@site/src/components/PersonalizeButton/PersonalizeButton';

// Import auth styles
import '@site/src/auth/styles.css';

export default function LayoutWrapper(props) {
  const { pathname } = useLocation();
  const isHomePage = pathname === '/' || pathname === '/index.html';

  // Apply peacock green background only on home page
  const layoutStyle = isHomePage
    ? {
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #38b2ac 0%, #6ee7b7 100%)'
      }
    : {};

  return (
    <div style={isHomePage ? layoutStyle : {}}>
      <AuthProvider>
        <Layout {...props}>
          <div style={isHomePage ? {} : { background: 'white' }}>
            {/* Show personalize button only on documentation pages, not on homepage */}
            {!isHomePage && (
              <div style={{
                position: 'fixed',
                top: '120px',  // Moved down from 70px to make room for navbar dropdowns
                right: '20px',
                zIndex: 1000
              }}>
                <div style={{
                  backgroundColor: 'white',
                  borderRadius: '20px',
                  padding: '4px 12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}>
                  <DocPersonalizeButton />
                </div>
              </div>
            )}
            {props.children}
          </div>
        </Layout>
      </AuthProvider>
      <ChatBot />
    </div>
  );
}
