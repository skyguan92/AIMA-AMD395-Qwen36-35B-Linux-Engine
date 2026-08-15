"""Loopback media boundaries for native VL error/limit qualification."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import http.server
import socket
import threading
from typing import Any


LARGE_IMAGE_BYTES = 64 * 1024 * 1024 + 1
SLOW_IMAGE_SECONDS = 300.0
_LARGE_CHUNK = b"aima-invalid-png\0" * 4096


class ErrorLimitMediaServer:
    """Own generated HTTP failure endpoints and a reserved closed port."""

    def __init__(
        self,
        *,
        large_image_bytes: int = LARGE_IMAGE_BYTES,
        slow_image_seconds: float = SLOW_IMAGE_SECONDS,
    ) -> None:
        if large_image_bytes <= 64 * 1024 * 1024:
            raise ValueError("large image boundary must exceed 64 MiB")
        if slow_image_seconds <= 0:
            raise ValueError("slow image delay must be positive")
        self.large_image_bytes = large_image_bytes
        self.slow_image_seconds = slow_image_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._requests = {
            "empty_image": 0,
            "empty_video": 0,
            "large_image": 0,
            "slow_image": 0,
        }
        self._bytes_sent = dict.fromkeys(self._requests, 0)
        self._http: http.server.ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._reserved: socket.socket | None = None

    def _record_request(self, endpoint: str) -> None:
        with self._lock:
            self._requests[endpoint] += 1

    def _record_bytes(self, endpoint: str, count: int) -> None:
        with self._lock:
            self._bytes_sent[endpoint] += count

    def _handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _headers(self, content_type: str, length: int) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/empty-image":
                    endpoint = "empty_image"
                    owner._record_request(endpoint)
                    self._headers("image/png", 0)
                    return
                if self.path == "/empty-video":
                    endpoint = "empty_video"
                    owner._record_request(endpoint)
                    self._headers("video/mp4", 0)
                    return
                if self.path == "/large-image":
                    endpoint = "large_image"
                    owner._record_request(endpoint)
                    self._headers("image/png", owner.large_image_bytes)
                    remaining = owner.large_image_bytes
                    try:
                        while remaining > 0 and not owner._stop.is_set():
                            block = _LARGE_CHUNK[: min(remaining, len(_LARGE_CHUNK))]
                            self.wfile.write(block)
                            owner._record_bytes(endpoint, len(block))
                            remaining -= len(block)
                    except OSError:
                        pass
                    return
                if self.path == "/slow-image":
                    endpoint = "slow_image"
                    owner._record_request(endpoint)
                    self._headers("image/png", 1)
                    self.wfile.flush()
                    if not owner._stop.wait(owner.slow_image_seconds):
                        try:
                            self.wfile.write(b"x")
                            owner._record_bytes(endpoint, 1)
                        except OSError:
                            pass
                    return
                self.send_error(404)

            def log_message(self, _format: str, *args: Any) -> None:
                del args

        return Handler

    def __enter__(self) -> "ErrorLimitMediaServer":
        self._reserved = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._reserved.bind(("127.0.0.1", 0))
        self._http = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), self._handler()
        )
        self._http.daemon_threads = True
        self._http_thread = threading.Thread(
            target=self._http.serve_forever,
            name="native-vl-error-limit-fixture",
            daemon=True,
        )
        self._http_thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=5)
        if self._reserved is not None:
            self._reserved.close()

    @property
    def http_base(self) -> str:
        if self._http is None:
            raise RuntimeError("error/limit media server is not running")
        return f"http://127.0.0.1:{self._http.server_port}"

    @property
    def unreachable_base(self) -> str:
        if self._reserved is None:
            raise RuntimeError("unreachable media port is not reserved")
        return f"http://127.0.0.1:{self._reserved.getsockname()[1]}"

    @property
    def statistics(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                "requests": dict(self._requests),
                "bytes_sent": dict(self._bytes_sent),
            }
