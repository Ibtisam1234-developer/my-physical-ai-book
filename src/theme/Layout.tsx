import React from 'react';
import { useLocation } from '@docusaurus/router';
import Layout from '@theme-original/Layout';
import ChatBot from '@site/src/components/ChatBot/ChatBot';

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
      <Layout {...props}>
        <div style={isHomePage ? {} : { background: 'white' }}>
          {props.children}
        </div>
      </Layout>
      <ChatBot />
    </div>
  );
}