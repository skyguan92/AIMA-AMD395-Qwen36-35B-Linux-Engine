"""Loopback HTTP/HTTPS media fixtures for native VL qualification."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import http.server
import mimetypes
from pathlib import Path
import ssl
import threading
from typing import Any


MODE_FIXTURES = {
    "image": "image-rgb-256.png",
    "video_a": "video-8f-4fps-128.mp4",
    "video_b": "video-12f-6fps-192x128.avi",
    "video_error": "corrupt-video.mp4",
}


class LocalMediaServers:
    """Own one mutable HTTP origin and one verified HTTPS origin."""

    def __init__(
        self,
        fixture_root: Path,
        certificate: Path,
        private_key: Path,
    ) -> None:
        self.fixture_root = fixture_root.resolve()
        self.certificate = certificate.resolve()
        self.private_key = private_key.resolve()
        for path in (self.fixture_root, self.certificate, self.private_key):
            if not path.exists():
                raise ValueError(f"local media server input is missing: {path}")
        self._lock = threading.Lock()
        self._mode = "video_a"
        self._request_counts = {"http": 0, "https": 0}
        self._http: http.server.ThreadingHTTPServer | None = None
        self._https: http.server.ThreadingHTTPServer | None = None
        self._threads: list[threading.Thread] = []

    def set_mode(self, mode: str) -> None:
        if mode not in MODE_FIXTURES:
            raise ValueError(f"unknown local media mode: {mode}")
        with self._lock:
            self._mode = mode

    def _handler(self, transport: str) -> type[http.server.BaseHTTPRequestHandler]:
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/image-rgb-256.png":
                    fixture = "image-rgb-256.png"
                elif self.path == "/mutable-video":
                    with owner._lock:
                        fixture = MODE_FIXTURES[owner._mode]
                else:
                    self.send_error(404)
                    return
                payload = (owner.fixture_root / fixture).read_bytes()
                content_type = (
                    mimetypes.guess_type(fixture)[0]
                    or "application/octet-stream"
                )
                with owner._lock:
                    owner._request_counts[transport] += 1
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format: str, *args: Any) -> None:
                del args

        return Handler

    def __enter__(self) -> "LocalMediaServers":
        self._http = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), self._handler("http")
        )
        self._https = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), self._handler("https")
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(self.certificate, self.private_key)
        self._https.socket = context.wrap_socket(
            self._https.socket, server_side=True
        )
        for name, server in (("http", self._http), ("https", self._https)):
            thread = threading.Thread(
                target=server.serve_forever,
                name=f"native-vl-{name}-fixture",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        return self

    def __exit__(self, *_args: object) -> None:
        for server in (self._http, self._https):
            if server is not None:
                server.shutdown()
                server.server_close()
        for thread in self._threads:
            thread.join(timeout=5)

    @property
    def http_base(self) -> str:
        if self._http is None:
            raise RuntimeError("local HTTP media server is not running")
        return f"http://127.0.0.1:{self._http.server_port}"

    @property
    def https_base(self) -> str:
        if self._https is None:
            raise RuntimeError("local HTTPS media server is not running")
        return f"https://127.0.0.1:{self._https.server_port}"

    @property
    def request_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._request_counts)
