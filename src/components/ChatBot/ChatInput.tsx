/**
 * ChatInput component - text input with auto-resize and character limit.
 */

import React, { useState, useRef, useEffect } from 'react';
import styles from './styles.module.css';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

const MAX_CHARS = 10000;

export default function ChatInput({ onSend, disabled }: ChatInputProps): JSX.Element {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter to send (Shift+Enter for new line)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const charCount = input.length;
  const isOverLimit = charCount > MAX_CHARS;

  return (
    <form className={styles.chatInput} onSubmit={handleSubmit}>
      <div className={styles.inputWrapper}>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about Physical AI, ROS 2, humanoid robots..."
          disabled={disabled}
          rows={1}
          maxLength={MAX_CHARS}
          aria-label="Chat input"
          className={isOverLimit ? styles.overLimit : ''}
        />
        <div className={styles.inputFooter}>
          <span className={`${styles.charCount} ${isOverLimit ? styles.overLimit : ''}`}>
            {charCount} / {MAX_CHARS}
          </span>
          <button
            type="submit"
            disabled={disabled || !input.trim() || isOverLimit}
            aria-label="Send message"
            className={styles.sendButton}
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
      </div>
      <div className={styles.inputHint}>
        <small>Press Enter to send, Shift+Enter for new line</small>
      </div>
    </form>
  );
}
