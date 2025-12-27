"""
Structured logging utility with JSON formatting and correlation IDs.

Provides consistent logging across the application with:
- JSON-formatted log messages
- Correlation IDs for request tracing
- Contextual information (user_id, endpoint, etc.)
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from app.config import settings


# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format.
    
    Includes timestamp, level, message, and contextual fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }

        # Add extra fields from record
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "endpoint"):
            log_data["endpoint"] = record.endpoint
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "response_status"):
            log_data["response_status"] = record.response_status
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging() -> None:
    """Configure application logging with JSON formatter."""
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def set_correlation_id(correlation_id: str | None = None) -> str:
    """
    Set correlation ID for current context.
    
    Args:
        correlation_id: Optional correlation ID (generates UUID if not provided)
        
    Returns:
        The correlation ID that was set
    """
    cid = correlation_id or str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get correlation ID for current context."""
    return correlation_id_var.get()


def log_request(
    logger: logging.Logger,
    endpoint: str,
    user_id: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Log API request with structured data.
    
    Args:
        logger: Logger instance
        endpoint: API endpoint path
        user_id: Optional user ID
        **kwargs: Additional fields to log
    """
    extra = {"endpoint": endpoint}
    if user_id:
        extra["user_id"] = user_id
    extra.update(kwargs)
    
    logger.info("API request", extra=extra)


def log_response(
    logger: logging.Logger,
    endpoint: str,
    status: int,
    latency_ms: float,
    user_id: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Log API response with structured data.
    
    Args:
        logger: Logger instance
        endpoint: API endpoint path
        status: HTTP status code
        latency_ms: Response time in milliseconds
        user_id: Optional user ID
        **kwargs: Additional fields to log
    """
    extra = {
        "endpoint": endpoint,
        "response_status": status,
        "latency_ms": latency_ms,
    }
    if user_id:
        extra["user_id"] = user_id
    extra.update(kwargs)
    
    logger.info("API response", extra=extra)


# Convenience aliases for backward compatibility
log_request_info = log_request
log_response_info = log_response

# Initialize logging on module import
setup_logging()
