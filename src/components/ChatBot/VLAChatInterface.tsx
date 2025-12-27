/**
 * VLA Chat Interface component for Vision-Language-Action system.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { ChatMessage, StreamChunk } from '@site/src/types/chat';
import { apiClient } from '@site/src/utils/api';
import styles from './styles.module.css';

interface VLAChatInterfaceProps {
  onClose?: () => void;
  initialMessages?: ChatMessage[];
}

const VLAChatInterface: React.FC<VLAChatInterfaceProps> = ({ onClose, initialMessages = [] }) => {
  const { siteConfig } = useDocusaurusContext();
  const backendUrl = (siteConfig.customFields?.BACKEND_URL as string) || 'https://my-physical-ai-book-production.up.railway.app';

  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState('');
  const [sessionSources, setSessionSources] = useState<any[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages, currentStreamingMessage]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage: ChatMessage = {
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    const newInputValue = inputValue;
    setInputValue('');
    setIsLoading(true);
    setCurrentStreamingMessage('');

    try {
      // Use backend URL from config
      const fullUrl = `${backendUrl}/api/chat/stream`;

      // For streaming, we'll use Server-Sent Events
      const response = await fetch(fullUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: newInputValue,
          session_id: 'current_session', // In real app, this would be a proper session ID
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
                if (data.trim() === '') continue;

                const chunk: StreamChunk = JSON.parse(data);

                if (chunk.type === 'content' && chunk.content) {
                  setCurrentStreamingMessage(prev => prev + chunk.content);
                } else if (chunk.type === 'sources' && chunk.sources) {
                  setSessionSources(chunk.sources);
                } else if (chunk.type === 'done') {
                  // Add assistant message with streaming content
                  const assistantMessage: ChatMessage = {
                    role: 'assistant',
                    content: currentStreamingMessage,
                    timestamp: new Date(),
                    sources: sessionSources
                  };

                  setMessages(prev => [...prev, assistantMessage]);
                  setCurrentStreamingMessage('');
                  setIsLoading(false);
                  setSessionSources([]);
                } else if (chunk.type === 'error') {
                  const errorMessage: ChatMessage = {
                    role: 'assistant',
                    content: `Error: ${chunk.error}`,
                    timestamp: new Date(),
                  };
                  setMessages(prev => [...prev, errorMessage]);
                  setIsLoading(false);
                }
              } catch (parseError) {
                console.error('Error parsing SSE data:', parseError);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e as any);
    }
  };

  return (
    <div className={styles.vlaChatContainer}>
      <div className={styles.chatHeader}>
        <h3>AI Assistant</h3>
        <button className={styles.closeButton} onClick={onClose} aria-label="Close chat">
          ×
        </button>
      </div>

      <div className={styles.chatMessages}>
        {messages.map((message, index) => (
          <div
            key={index}
            className={`${styles.message} ${
              message.role === 'user' ? styles.userMessage : styles.assistantMessage
            }`}
          >
            <div className={styles.messageHeader}>
              <strong>{message.role === 'user' ? 'You' : 'AI Assistant'}</strong>
              <span className={styles.timestamp}>
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
            <div className={styles.messageContent}>
              {message.content}
            </div>
            {message.sources && message.sources.length > 0 && (
              <div className={styles.sources}>
                <details>
                  <summary>Sources ({message.sources.length})</summary>
                  <ul>
                    {message.sources.map((source: any, idx: number) => (
                      <li key={idx}>
                        <strong>{source.filename}</strong> - {source.section}
                        <br />
                        <small>Relevance: {(source.relevance_score * 100).toFixed(1)}%</small>
                      </li>
                    ))}
                  </ul>
                </details>
              </div>
            )}
          </div>
        ))}

        {currentStreamingMessage && (
          <div className={`${styles.message} ${styles.assistantMessage} ${styles.streaming}`}>
            <div className={styles.messageHeader}>
              <strong>AI Assistant</strong>
              <span className={styles.loadingDots}>●●●</span>
            </div>
            <div className={styles.messageContent}>
              {currentStreamingMessage}
              <span className={styles.streamingCursor}>▊</span>
            </div>
          </div>
        )}

        {isLoading && !currentStreamingMessage && (
          <div className={`${styles.message} ${styles.assistantMessage}`}>
            <div className={styles.messageHeader}>
              <strong>AI Assistant</strong>
            </div>
            <div className={styles.messageContent}>
              <span className={styles.loadingDots}>Thinking...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className={styles.chatInputForm} onSubmit={handleSendMessage}>
        <div className={styles.inputWrapper}>
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about Physical AI, ROS 2, humanoid robotics..."
            disabled={isLoading}
            rows={1}
            maxLength={10000}
          />
          <button
            type="submit"
            disabled={isLoading || !inputValue.trim()}
            className={styles.sendButton}
            aria-label="Send message"
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
        <div className={styles.inputFooter}>
          <small>Press Enter to send, Shift+Enter for new line</small>
          <span className={styles.charCount}>
            {inputValue.length} / 10000
          </span>
        </div>
      </form>
    </div>
  );
};

export default VLAChatInterface;
