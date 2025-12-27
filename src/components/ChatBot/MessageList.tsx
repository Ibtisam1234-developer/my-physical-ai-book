/**
 * MessageList component - displays chat messages with streaming support.
 */

import React, { useEffect, useRef } from 'react';
import { ChatMessage } from '@site/src/types/chat';
import styles from './styles.module.css';

interface MessageListProps {
  messages: ChatMessage[];
  currentStreamingMessage?: string;
  isLoading: boolean;
}

export default function MessageList({
  messages,
  currentStreamingMessage,
  isLoading,
}: MessageListProps): JSX.Element {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentStreamingMessage]);

  const formatTimestamp = (date: Date) => {
    return new Date(date).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className={styles.messageList}>
      {messages.map((message, index) => (
        <div
          key={index}
          className={`${styles.message} ${
            message.role === 'user' ? styles.userMessage : styles.assistantMessage
          }`}
        >
          <div className={styles.messageHeader}>
            <strong>{message.role === 'user' ? 'You' : 'AI Assistant'}</strong>
            <span className={styles.timestamp}>{formatTimestamp(message.timestamp)}</span>
          </div>
          <div className={styles.messageContent}>
            {message.content}
          </div>
          {message.sources && message.sources.length > 0 && (
            <div className={styles.sources}>
              <details>
                <summary>📚 Sources ({message.sources.length})</summary>
                <ul>
                  {message.sources.map((source, idx) => (
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

      {/* Streaming message */}
      {currentStreamingMessage && (
        <div className={`${styles.message} ${styles.assistantMessage} ${styles.streaming}`}>
          <div className={styles.messageHeader}>
            <strong>AI Assistant</strong>
            <span className={styles.loadingDots}>●●●</span>
          </div>
          <div className={styles.messageContent}>
            {currentStreamingMessage}
            <span className={styles.cursor}>▊</span>
          </div>
        </div>
      )}

      {/* Loading indicator */}
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
  );
}
