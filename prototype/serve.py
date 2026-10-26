#!/usr/bin/env python3
"""
prototype/serve.py
Standalone HTTP server with port collision avoidance and proper MIME types
for Tamar "Tami" Sheaffer 70th Birthday 2.5D Spatial Prototype.
"""

import os
import io
import sys
import json
import gzip
import time
import socket
import argparse
import threading
import urllib.parse
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

DEFAULT_PORT = 7070
DEFAULT_DIR = Path(__file__).resolve().parent

GZIP_EXTENSIONS = {".js", ".mjs", ".glsl", ".json", ".css", ".html", ".svg"}

CUSTOM_MIME_TYPES = {
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".glsl": "text/plain; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
}

_diag_lock = threading.Lock()


class SpatialHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with CORS headers, gzip compression, and mobile diagnostics."""

    diag_log_path = None

    def __init__(self, *args, directory=None, **kwargs):
        if directory is None:
            directory = str(DEFAULT_DIR)
        self._cache_policy = "no-cache, must-revalidate"
        super().__init__(*args, directory=directory, **kwargs)

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in CUSTOM_MIME_TYPES:
            return CUSTOM_MIME_TYPES[ext]
        return super().guess_type(path)

    def end_headers(self):
        # Enable CORS and appropriate caching headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", getattr(self, "_cache_policy", "no-cache, must-revalidate"))
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _record_diagnostic(self, raw_payload):
        try:
            data = json.loads(raw_payload) if raw_payload else {}
        except Exception:
            data = {"raw": raw_payload[:2000] if raw_payload else ""}

        client_ip = self.headers.get("X-Forwarded-For") or self.client_address[0]
        record = {
            "server_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "client": client_ip,
            **data,
        }
        line = json.dumps(record, ensure_ascii=False)
        with _diag_lock:
            sys.stderr.write(f"[PHONE DIAG] {line}\n")
            sys.stderr.flush()
            if self.diag_log_path:
                try:
                    with open(self.diag_log_path, "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                except OSError:
                    pass

    def do_POST(self):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/__diag":
            length = min(int(self.headers.get("Content-Length", "0") or 0), 65536)
            body = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else ""
            self._record_diagnostic(body)
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/__diag":
            qs = urllib.parse.parse_qs(parsed.query)
            payload = qs.get("d", [""])[0]
            self._record_diagnostic(payload)
            self.send_response(204)
            self.end_headers()
            return

        # Cache static heavy vendor/scene assets while keeping code fresh
        if parsed.path.startswith("/js/vendor/") or parsed.path.startswith("/assets/"):
            self._cache_policy = "public, max-age=3600"
        else:
            self._cache_policy = "no-cache, must-revalidate"

        accept_enc = self.headers.get("Accept-Encoding", "")
        local_path = Path(self.translate_path(self.path))
        if local_path.is_dir():
            for idx in ("index.html", "index.htm"):
                candidate = local_path / idx
                if candidate.is_file():
                    local_path = candidate
                    break

        ext = local_path.suffix.lower()
        if "gzip" in accept_enc and local_path.is_file() and ext in GZIP_EXTENSIONS:
            try:
                raw_bytes = local_path.read_bytes()
                gz_bytes = gzip.compress(raw_bytes, compresslevel=6)
                ctype = self.guess_type(str(local_path))
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
                self.send_header("Content-Length", str(len(gz_bytes)))
                self.end_headers()
                self.copyfile(io.BytesIO(gz_bytes), self.wfile)
                return
            except OSError:
                pass

        super().do_GET()

    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass

    def log_message(self, format, *args):
        # Concise logging in terminal
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")
        sys.stderr.flush()

def is_port_free(port, host="127.0.0.1"):
    """Checks whether a port is currently open and bindable."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
        except OSError:
            return False

def find_free_port(start_port=DEFAULT_PORT, max_attempts=50, host="127.0.0.1"):
    """Finds first available port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        if is_port_free(p, host):
            return p
    # Fallback to OS assigned ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]

def create_server(port=None, directory=None, host="127.0.0.1", diag_log=None):
    """Creates a ThreadingHTTPServer instance with collision avoidance."""
    if directory is None:
        directory = DEFAULT_DIR
    directory = Path(directory).resolve()

    if diag_log is not None:
        SpatialHTTPRequestHandler.diag_log_path = str(Path(diag_log).resolve())

    if port is None or not is_port_free(port, host):
        port = find_free_port(port or DEFAULT_PORT, host=host)

    handler = lambda *args, **kwargs: SpatialHTTPRequestHandler(*args, directory=str(directory), **kwargs)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, port

def start_server_thread(port=None, directory=None, host="127.0.0.1"):
    """Starts server in a background daemon thread for testing."""
    server, actual_port = create_server(port, directory, host)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port, thread

def main():
    parser = argparse.ArgumentParser(description="Serve the Tami Sheaffer 70th Spatial Prototype")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help=f"Preferred port (default: {DEFAULT_PORT})")
    parser.add_argument("--dir", "-d", default=str(DEFAULT_DIR), help="Directory to serve")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--diag-log", default=None, help="Optional JSONL file path to persist mobile diagnostics")
    args = parser.parse_args()

    serve_dir = Path(args.dir).resolve()
    if not (serve_dir / "index.html").exists():
        print(f"Warning: {serve_dir}/index.html does not exist!")

    server, port = create_server(args.port, serve_dir, args.host, diag_log=args.diag_log)
    url = f"http://{args.host}:{port}"
    print("=" * 65)
    print(f"TAMAR SHEAFFER 70TH BIRTHDAY — 2.5D SPATIAL PROTOTYPE SERVER")
    print(f"Serving directory: {serve_dir}")
    print(f"Local URL:         {url}")
    print(f"Collision check:   Port {args.port} -> Bound port {port}")
    print("=" * 65)
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        server.shutdown()
        server.server_close()
        print("Server stopped.")

if __name__ == "__main__":
    main()
