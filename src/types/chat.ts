/**
 * TypeScript types for chat functionality.
 */

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: SourceCitation[];
}

export interface SourceCitation {
  source: string;
  section: string;
  filename: string;
  relevance_score: number;
}

export interface ChatRequest {
  query: string;
  session_id?: string;
  temperature?: number;
  max_tokens?: number;
  top_k?: number;
  score_threshold?: number;
}

export interface ChatResponse {
  response: string;
  sources: SourceCitation[];
  session_id: string;
  tokens_used: number;
  response_time_ms: number;
}

export interface StreamChunk {
  type: 'content' | 'sources' | 'done' | 'error';
  content?: string;
  sources?: SourceCitation[];
  session_id?: string;
  response_time?: number;
  tokens_used?: number;
  error?: string;
  timestamp: number;
}

export interface ChatHistoryResponse {
  session_id: string;
  messages: ChatMessage[];
  created_at: Date;
  updated_at: Date;
}