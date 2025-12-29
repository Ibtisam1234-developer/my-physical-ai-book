// Auth types for Physical AI platform

export interface User {
  id: string;
  email: string;
  softwareBackground: SoftwareBackground;
  hardwareBackground: HardwareBackground;
}

export type SoftwareBackground = "beginner" | "intermediate" | "advanced" | "expert";
export type HardwareBackground = "none" | "hobbyist" | "student" | "professional";

export interface AuthResponse {
  user: User;
  token: string;
}

export interface SignupParams {
  email: string;
  password: string;
  softwareBackground: SoftwareBackground;
  hardwareBackground: HardwareBackground;
}

export interface SigninParams {
  email: string;
  password: string;
}

export interface JWTPayload {
  sub: string;
  email: string;
  software_background: SoftwareBackground;
  hardware_background: HardwareBackground;
  iat?: number;
  exp?: number;
  iss?: string;
}
