// src/lib/auth-client.ts
import { useState, useEffect } from 'react';

// Define types based on the auth server
interface User {
  id: string;
  email: string;
  softwareBackground?: string;
  hardwareBackground?: string;
}

interface Session {
  user: User;
  token: string;
}

// Auth client implementation that communicates with the auth server
class AuthClient {
  private baseUrl: string;
  private session: Session | null = null;
  private listeners: Array<(session: Session | null) => void> = [];
  private initialized: boolean = false;

  constructor() {
    // Get the auth server URL from environment or default
    // In browser, we use the default since process.env is not available
    this.baseUrl = 'http://localhost:3001';

    // Initialize session from localStorage if available
    this.initSession();
  }

  // Initialize session from localStorage
  private initSession() {
    if (typeof window !== 'undefined' && !this.initialized) {
      const savedSession = localStorage.getItem('auth-session');
      if (savedSession) {
        try {
          this.session = JSON.parse(savedSession);
          this.initialized = true;
          // Notify listeners that session is loaded
          this.notifyListeners();
        } catch (e) {
          console.error('Failed to parse saved session', e);
        }
      }
      this.initialized = true;
    }
  }

  // Subscribe to session changes
  subscribe(listener: (session: Session | null) => void) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  // Get current session
  getSession(): Session | null {
    // Re-read from localStorage to ensure we have the latest
    if (typeof window !== 'undefined') {
      // Get token and user from auth context storage
      const token = localStorage.getItem('auth_token');
      const userStr = localStorage.getItem('auth_user');

      if (token && userStr) {
        try {
          const user = JSON.parse(userStr);
          this.session = {
            user: {
              id: user.id,
              email: user.email,
              softwareBackground: user.softwareBackground,
              hardwareBackground: user.hardwareBackground,
            },
            token: token
          };
        } catch (e) {
          console.error('Failed to parse saved session', e);
        }
      } else {
        // Fallback to old format for compatibility
        const savedSession = localStorage.getItem('auth-session');
        if (savedSession) {
          try {
            this.session = JSON.parse(savedSession);
          } catch (e) {
            console.error('Failed to parse saved session', e);
          }
        }
      }
    }
    return this.session;
  }

  // Use session hook for React components
  useSession() {
    const [session, setSession] = useState<Session | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
      // Initial load from localStorage
      const loadSession = () => {
        const savedSession = localStorage.getItem('auth-session');
        if (savedSession) {
          try {
            setSession(JSON.parse(savedSession));
          } catch (e) {
            console.error('Failed to parse saved session', e);
            setSession(null);
          }
        }
        setIsLoading(false);
      };

      loadSession();

      // Subscribe to session changes
      const unsubscribe = this.subscribe((newSession) => {
        setSession(newSession);
      });

      return unsubscribe;
    }, []);

    return {
      data: session,
      isPending: isLoading,
      isLoading: isLoading,
    };
  }

  // Sign in method
  async signIn(email: string, password: string): Promise<Session> {
    const response = await fetch(`${this.baseUrl}/api/auth/sign-in`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Sign in failed');
    }

    const data = await response.json();
    this.setSession(data);
    return data;
  }

  // Sign up method
  async signUp(
    email: string,
    password: string,
    softwareBackground: string,
    hardwareBackground: string
  ): Promise<Session> {
    const response = await fetch(`${this.baseUrl}/api/auth/sign-up`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email,
        password,
        software_background: softwareBackground,
        hardware_background: hardwareBackground,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Sign up failed');
    }

    const data = await response.json();
    this.setSession(data);
    return data;
  }

  // Get current user
  async getCurrentUser(): Promise<User | null> {
    if (!this.session?.token) {
      return null;
    }

    const response = await fetch(`${this.baseUrl}/api/auth/me`, {
      headers: {
        'Authorization': `Bearer ${this.session.token}`,
      },
    });

    if (!response.ok) {
      // If token is invalid, clear the session
      if (response.status === 401) {
        this.clearSession();
      }
      return null;
    }

    const data = await response.json();
    return data.user;
  }

  // Set session and notify listeners
  private setSession(session: Session) {
    this.session = session;
    if (typeof window !== 'undefined') {
      // Store in format compatible with auth context
      localStorage.setItem('auth_token', session.token);
      localStorage.setItem('auth_user', JSON.stringify(session.user));
      // Also store in old format for compatibility
      localStorage.setItem('auth-session', JSON.stringify(session));
    }
    this.notifyListeners();
  }

  // Clear session
  private clearSession() {
    this.session = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth-session');
      localStorage.removeItem('auth_token');
      localStorage.removeItem('auth_user');
    }
    this.notifyListeners();
  }

  // Notify all listeners of session change
  private notifyListeners() {
    this.listeners.forEach(listener => listener(this.session));
  }

  // Verify if session is still valid
  async verifySession(): Promise<boolean> {
    if (!this.session?.token) {
      return false;
    }

    try {
      // Check if token is expired by decoding it (if it's a JWT)
      const tokenParts = this.session.token.split('.');
      if (tokenParts.length === 3) {
        try {
          const payload = JSON.parse(atob(tokenParts[1]));
          const currentTime = Math.floor(Date.now() / 1000);
          if (payload.exp && payload.exp < currentTime) {
            // Token is expired
            this.clearSession();
            return false;
          }
        } catch (e) {
          console.error('Error decoding token:', e);
        }
      }

      const user = await this.getCurrentUser();
      if (user === null) {
        // Token is invalid or user doesn't exist
        this.clearSession();
        return false;
      }
      return true;
    } catch (error) {
      console.error('Session verification failed:', error);
      this.clearSession();
      return false;
    }
  }
}

// Create a singleton instance
export const authClient = new AuthClient();