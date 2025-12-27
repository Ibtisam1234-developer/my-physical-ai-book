/**
 * API client utilities for VLA chatbot integration.
 */

import axios from 'axios';

// Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Add request interceptor for authentication
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API service functions
export const chatAPI = {
  /**
   * Send a chat message and get response.
   */
  sendMessage: async (query: string, sessionId?: string) => {
    try {
      const response = await apiClient.post('/api/chat', {
        query,
        session_id: sessionId,
        temperature: 0.7,
        max_tokens: 1024,
        top_k: 7,
        score_threshold: 0.7
      });
      return response.data;
    } catch (error) {
      throw new Error(`Chat request failed: ${(error as Error).message}`);
    }
  },

  /**
   * Send a streaming chat message using fetch with streaming response.
   */
  sendStreamMessage: async (
    query: string,
    sessionId: string,
    onMessage: (data: any) => void,
    onError?: (error: any) => void
  ) => {
    // Create streaming request using fetch with streaming response
    const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token') || ''}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        temperature: 0.7,
        max_tokens: 1024,
        top_k: 7,
        score_threshold: 0.7
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    if (reader) {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = line.slice(6); // Remove 'data: ' prefix
              if (data.trim() === '[DONE]' || data.trim() === '') continue;

              const chunk = JSON.parse(data);
              onMessage(chunk);
            } catch (parseError) {
              if (onError) onError(parseError);
            }
          }
        }
      }
    }
  },

  /**
   * Get chat history for a session.
   */
  getChatHistory: async (sessionId: string) => {
    try {
      const response = await apiClient.get(`/api/chat/history/${sessionId}`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get chat history: ${(error as Error).message}`);
    }
  },

  /**
   * List user's chat sessions.
   */
  listChatSessions: async () => {
    try {
      const response = await apiClient.get('/api/chat/sessions');
      return response.data;
    } catch (error) {
      throw new Error(`Failed to list chat sessions: ${(error as Error).message}`);
    }
  },

  /**
   * Delete a chat session.
   */
  deleteChatSession: async (sessionId: string) => {
    try {
      const response = await apiClient.delete(`/api/chat/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to delete chat session: ${(error as Error).message}`);
    }
  },

  /**
   * Health check for chat service.
   */
  healthCheck: async () => {
    try {
      const response = await apiClient.get('/api/chat/health');
      return response.data;
    } catch (error) {
      throw new Error(`Chat service health check failed: ${(error as Error).message}`);
    }
  }
};

// Export default instance
export default apiClient;