"""
Structured logging middleware for the Physical AI platform.
"""

import time
import uuid
import json
from typing import Callable, Awaitable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import structlog


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured logging middleware with correlation IDs.
    """

    def __init__(self, app):
        super().__init__(app)
        self.logger = structlog.get_logger(__name__)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """
        Process request with structured logging.
        """
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        # Log request start
        start_time = time.time()
        self.logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            correlation_id=correlation_id,
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent", "")
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate response time
            response_time = time.time() - start_time

            # Log request completion
            self.logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                response_time_ms=response_time * 1000,
                correlation_id=correlation_id,
                content_length=response.headers.get("content-length", "unknown")
            )

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        except Exception as e:
            # Calculate response time for error
            response_time = time.time() - start_time

            # Log error
            self.logger.error(
                "request_error",
                method=request.method,
                path=request.url.path,
                error=str(e),
                response_time_ms=response_time * 1000,
                correlation_id=correlation_id
            )

            # Re-raise the exception
            raise e

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.
        """
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        return request.client.host


# Setup structured logging
def setup_structured_logging():
    """
    Setup structured logging with JSON format.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def add_correlation_id():
    """
    Add correlation ID to logging context.
    """
    correlation_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    return correlation_id


def get_correlation_id():
    """
    Get current correlation ID from logging context.
    """
    ctx = structlog.contextvars.get_merged_contextvars()
    return ctx.get("correlation_id", "unknown")


# Example usage in other parts of the application
def log_request_info(
    logger,
    endpoint: str,
    user_id: str = None,
    session_id: str = None,
    **kwargs
):
    """
    Log request information with structured format.
    """
    extra_fields = {
        "endpoint": endpoint,
        "user_id": user_id,
        "session_id": session_id,
        **kwargs
    }

    correlation_id = get_correlation_id()
    if correlation_id != "unknown":
        extra_fields["correlation_id"] = correlation_id

    logger.info("request_info", **extra_fields)


def log_response_info(
    logger,
    endpoint: str,
    status_code: int,
    response_time: float,
    **kwargs
):
    """
    Log response information with structured format.
    """
    extra_fields = {
        "endpoint": endpoint,
        "status_code": status_code,
        "response_time_ms": response_time * 1000,
        **kwargs
    }

    correlation_id = get_correlation_id()
    if correlation_id != "unknown":
        extra_fields["correlation_id"] = correlation_id

    logger.info("response_info", **extra_fields)
