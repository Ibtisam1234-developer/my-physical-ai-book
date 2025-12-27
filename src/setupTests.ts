/**
 * Jest setup file for configuring the test environment.
 *
 * This file is executed before each test file.
 */

import '@testing-library/jest-dom';

// Mock environment variables
process.env.VITE_API_URL = 'http://localhost:8000';
process.env.VITE_ENVIRONMENT = 'test';

// Mock window.matchMedia (for responsive design tests)
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock IntersectionObserver (for lazy loading tests)
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  takeRecords() {
    return [];
  }
  unobserve() {}
} as any;

// Mock EventSource (for SSE tests)
global.EventSource = class EventSource {
  constructor(public url: string) {}
  addEventListener() {}
  removeEventListener() {}
  close() {}
  CONNECTING = 0;
  OPEN = 1;
  CLOSED = 2;
  readyState = 0;
  onopen = null;
  onmessage = null;
  onerror = null;
} as any;
