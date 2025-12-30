// Auth API client for Docusaurus

import type { SignupParams, SigninParams, AuthResponse, User } from "./types";

class AuthAPI {
  private baseUrl: string;

  constructor(baseUrl?: string) {
    // Try to get from window (injected by Docusaurus), then environment, then default to deployed server
    if (typeof window !== 'undefined' && (window as any).docusaurus?.siteConfig?.customFields?.AUTH_SERVER_URL) {
      this.baseUrl = (window as any).docusaurus.siteConfig.customFields.AUTH_SERVER_URL;
    } else {
      this.baseUrl = baseUrl || "https://my-physical-ai-book-production-87e2.up.railway.app";
    }
  }

  async signup(params: SignupParams): Promise<AuthResponse> {
    const response = await fetch(`${this.baseUrl}/api/auth/sign-up`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: params.email,
        password: params.password,
        software_background: params.softwareBackground,
        hardware_background: params.hardwareBackground,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Signup failed");
    }

    return response.json();
  }

  async signin(params: SigninParams): Promise<AuthResponse> {
    const response = await fetch(`${this.baseUrl}/api/auth/sign-in`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: params.email,
        password: params.password,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Signin failed");
    }

    return response.json();
  }

  async getMe(token: string): Promise<{ user: User }> {
    const response = await fetch(`${this.baseUrl}/api/auth/me`, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to get user");
    }

    return response.json();
  }

  async getJWKS(): Promise<{ keys: any[] }> {
    const response = await fetch(`${this.baseUrl}/.well-known/jwks.json`);
    if (!response.ok) {
      throw new Error("Failed to fetch JWKS");
    }
    return response.json();
  }
}

export const authAPI = new AuthAPI();
