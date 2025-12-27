"""
Rate limiting middleware for API endpoints.
"""

import time
from typing import Dict, Optional, Callable, Awaitable
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from collections import defaultdict, deque
import asyncio


class RateLimitMiddleware:
    """
    Rate limiting middleware to prevent API abuse.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        exclude_paths: Optional[list] = None
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]
        self.request_counts = defaultdict(deque)  # Stores request timestamps

    async def __call__(self, request: Request, call_next: Callable[[Request], Awaitable]) -> any:
        """
        Process incoming request with rate limiting check.
        """
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            response = await call_next(request)
            return response

        # Get client IP for rate limiting
        client_ip = self._get_client_ip(request)

        # Check rate limits
        current_time = time.time()

        # Clean old requests (older than 1 hour)
        while (self.request_counts[client_ip] and
               current_time - self.request_counts[client_ip][0] > 3600):
            self.request_counts[client_ip].popleft()

        # Check hourly limit
        if len(self.request_counts[client_ip]) >= self.requests_per_hour:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Hourly rate limit exceeded"}
            )

        # Check minute limit
        recent_requests = [
            timestamp for timestamp in self.request_counts[client_ip]
            if current_time - timestamp <= 60
        ]
        if len(recent_requests) >= self.requests_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Minute rate limit exceeded"}
            )

        # Add current request to count
        self.request_counts[client_ip].append(current_time)

        # Continue with request
        response = await call_next(request)
        return response

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


class RateLimiter:
    """
    Rate limiter utility for specific endpoints.
    """

    def __init__(self):
        self.sliding_windows = defaultdict(deque)

    def is_allowed(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, int]:
        """
        Check if request is allowed based on rate limit.

        Args:
            identifier: Unique identifier for rate limiting (user_id, ip, etc.)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        current_time = time.time()
        window = self.sliding_windows[identifier]

        # Remove requests outside the time window
        while window and current_time - window[0] > window_seconds:
            window.popleft()

        # Check if we're within the limit
        if len(window) < max_requests:
            window.append(current_time)
            remaining = max_requests - len(window)
            return True, remaining
        else:
            # Calculate when next request is allowed
            next_allowed = window[0] + window_seconds
            wait_time = next_allowed - current_time
            return False, int(wait_time)
