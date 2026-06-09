from __future__ import annotations

import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("astra.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        request.state.correlation_id = correlation_id
        
        response = await call_next(request)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"[{correlation_id}] {request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed_ms:.1f}ms)"
        )
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response
