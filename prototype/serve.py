#!/usr/bin/env python3
"""
prototype/serve.py
Standalone HTTP & Collaborative Family Beta Hub server with port collision avoidance,
gzip compression, mobile diagnostics, Universal UI + 3D spatial highlight feedback APIs,
read-only Biographical Wiki API (`/wiki`, `/api/wiki`), and Family Catalog + Drop-Folder
Photo Uploader (`/catalog`, `/api/catalog`, `/api/uploads`).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototype.beta_config import BetaFeedbackConfig, default_beta_feedback_config
from prototype.feedback_store import (
    BetaFeedbackStore,
    FeedbackNotFoundError,
    FeedbackPayloadTooLargeError,
    FeedbackValidationError,
)
from prototype.upload_service import (
    CatalogItemNotFoundError,
    FamilyPhotoUploadService,
    UploadPayloadTooLargeError,
    UploadValidationError,
)
from prototype.wiki_service import WikiNotFoundError, WikiPortalService, WikiValidationError

DEFAULT_PORT = 7070
DEFAULT_DIR = Path(__file__).resolve().parent
DIAG_MAX_BODY_BYTES = 65536
PORT_COLLISION_MAX_ATTEMPTS = 50
STATIC_CACHE_SECONDS = 3600
SNAPSHOT_CACHE_SECONDS = 86400
SERVER_REQUEST_QUEUE_SIZE = 128
REQUEST_SOCKET_TIMEOUT_SEC = 30.0

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
    ".gif": "image/gif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".svg": "image/svg+xml",
}

_diag_lock = threading.Lock()


def _safe_stderr_log(line: str) -> None:
    """Write diagnostic/access output to stderr without crashing when a daemon's PTY is revoked."""
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except OSError:
        pass


class SpatialThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying scoped diagnostics, beta feedback, wiki, and upload services."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = SERVER_REQUEST_QUEUE_SIZE

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[SimpleHTTPRequestHandler],
        *,
        beta_config: BetaFeedbackConfig,
        feedback_store: BetaFeedbackStore | None,
        wiki_service: WikiPortalService | None,
        upload_service: FamilyPhotoUploadService | None,
        media_root_dir: Path,
        diag_log_path: str | None = None,
        ephemeral_feedback_dir: str | None = None,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.beta_config = beta_config
        self.feedback_store = feedback_store
        self.wiki_service = wiki_service
        self.upload_service = upload_service
        self.media_root_dir = media_root_dir.resolve()
        self.diag_log_path = diag_log_path
        self._ephemeral_feedback_dir = ephemeral_feedback_dir

    def server_close(self) -> None:
        super().server_close()
        if self._ephemeral_feedback_dir and os.path.isdir(self._ephemeral_feedback_dir):
            shutil.rmtree(self._ephemeral_feedback_dir, ignore_errors=True)
            self._ephemeral_feedback_dir = None


class SpatialHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with CORS headers, gzip compression, Beta Feedback, Wiki, and Upload APIs."""

    timeout = REQUEST_SOCKET_TIMEOUT_SEC

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        if directory is None:
            directory = str(DEFAULT_DIR)
        self._cache_policy = "no-cache, must-revalidate"
        super().__init__(*args, directory=directory, **kwargs)

    @property
    def _spatial_server(self) -> SpatialThreadingHTTPServer:
        return self.server  # type: ignore[return-value]

    def guess_type(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in CUSTOM_MIME_TYPES:
            return CUSTOM_MIME_TYPES[ext]
        return super().guess_type(path)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, HEAD, POST, PATCH, DELETE, OPTIONS"
        )
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header(
            "Cache-Control", getattr(self, "_cache_policy", "no-cache, must-revalidate")
        )
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def _send_json(self, status_code: int, payload: dict[str, Any], *, head_only: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._cache_policy = "no-store"
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.copyfile(io.BytesIO(body), self.wfile)

    def _send_file_bytes(
        self, file_path: Path, *, cache_policy: str = "no-cache, must-revalidate", head_only: bool = False
    ) -> None:
        raw_bytes = file_path.read_bytes()
        ctype = self.guess_type(str(file_path))
        accept_enc = self.headers.get("Accept-Encoding", "")
        ext = file_path.suffix.lower()
        self._cache_policy = cache_policy

        if "gzip" in accept_enc and ext in GZIP_EXTENSIONS:
            gz_bytes = gzip.compress(raw_bytes, compresslevel=6)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(gz_bytes)))
            self.end_headers()
            if not head_only:
                self.copyfile(io.BytesIO(gz_bytes), self.wfile)
            return

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw_bytes)))
        self.end_headers()
        if not head_only:
            self.copyfile(io.BytesIO(raw_bytes), self.wfile)

    def _read_json_body(self, max_bytes: int) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0") or "0"
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise FeedbackValidationError("Invalid Content-Length header.") from exc

        if length <= 0:
            raise FeedbackValidationError("Request body cannot be empty.")
        if length > max_bytes:
            raise FeedbackPayloadTooLargeError(
                f"Request body ({length} bytes) exceeds maximum allowed ({max_bytes} bytes)."
            )

        raw_bytes = self.rfile.read(length)
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FeedbackValidationError(f"Malformed UTF-8 JSON body: {exc}") from exc

        if not isinstance(parsed, dict):
            raise FeedbackValidationError("JSON body must be an object.")
        return parsed

    def _record_diagnostic(self, raw_payload: str) -> None:
        try:
            data = json.loads(raw_payload) if raw_payload else {}
        except json.JSONDecodeError:
            data = {"raw": raw_payload[:2000] if raw_payload else ""}

        client_ip = self.headers.get("X-Forwarded-For") or self.client_address[0]
        record = {
            "server_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "client": client_ip,
            **data,
        }
        line = json.dumps(record, ensure_ascii=False)
        diag_log_path = getattr(self._spatial_server, "diag_log_path", None)
        with _diag_lock:
            _safe_stderr_log(f"[PHONE DIAG] {line}\n")
            if diag_log_path:
                with open(diag_log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    def _require_active_store(self) -> BetaFeedbackStore:
        cfg = self._spatial_server.beta_config
        store = self._spatial_server.feedback_store
        if not cfg.enabled or store is None:
            raise FeedbackNotFoundError("Beta feedback mode is disabled on this server.")
        return store

    def _require_upload_service(self) -> FamilyPhotoUploadService:
        cfg = self._spatial_server.beta_config
        svc = self._spatial_server.upload_service
        if not cfg.enabled or svc is None:
            raise FeedbackNotFoundError("Family upload portal is disabled on this server.")
        return svc

    def _require_wiki_service(self) -> WikiPortalService:
        cfg = self._spatial_server.beta_config
        svc = self._spatial_server.wiki_service
        if not cfg.enabled or svc is None:
            raise FeedbackNotFoundError("Wiki beta portal is disabled on this server.")
        return svc

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/__diag":
            length = min(int(self.headers.get("Content-Length", "0") or 0), DIAG_MAX_BODY_BYTES)
            body = (
                self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else ""
            )
            self._record_diagnostic(body)
            self.send_response(204)
            self.end_headers()
            return

        if parsed.path == "/api/feedback":
            try:
                store = self._require_active_store()
                payload = self._read_json_body(
                    self._spatial_server.beta_config.limits.max_request_bytes
                )
                ua = self.headers.get("User-Agent", "")
                entry = store.create_entry(payload, user_agent_hint=ua)
                _safe_stderr_log(
                    f"[BETA FEEDBACK] #{entry.seq_num} ({entry.scene_id}/{entry.category_id}) "
                    f"by {entry.author}: {entry.comment[:120]}\n"
                )
                self._send_json(201, {"entry": entry.to_dict()})
            except FeedbackNotFoundError as exc:
                self._send_json(404, {"error": str(exc)})
            except FeedbackPayloadTooLargeError as exc:
                self._send_json(413, {"error": str(exc)})
            except FeedbackValidationError as exc:
                self._send_json(400, {"error": str(exc)})
            return

        if parsed.path == "/api/uploads":
            try:
                upload_svc = self._require_upload_service()
                payload = self._read_json_body(
                    self._spatial_server.beta_config.uploads.max_upload_request_bytes
                )
                summary = upload_svc.ingest_batch(payload)
                _safe_stderr_log(
                    f"[FAMILY UPLOAD] ingested={summary['ingested_count']} "
                    f"duplicates={summary['duplicate_count']} total={summary['catalog_total_count']}\n"
                )
                self._send_json(201, summary)
            except FeedbackNotFoundError as exc:
                self._send_json(404, {"error": str(exc)})
            except (FeedbackPayloadTooLargeError, UploadPayloadTooLargeError) as exc:
                self._send_json(413, {"error": str(exc)})
            except (FeedbackValidationError, UploadValidationError) as exc:
                self._send_json(400, {"error": str(exc)})
            return

        self._send_json(404, {"error": f"Unknown endpoint: {parsed.path}"})

    def do_PATCH(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path.startswith("/api/feedback/"):
            entry_id = urllib.parse.unquote(parsed.path[len("/api/feedback/") :]).strip("/")
            try:
                store = self._require_active_store()
                payload = self._read_json_body(
                    self._spatial_server.beta_config.limits.max_request_bytes
                )
                updated = store.update_status(entry_id, str(payload.get("status", "")))
                self._send_json(200, {"entry": updated.to_dict()})
            except FeedbackNotFoundError as exc:
                self._send_json(404, {"error": str(exc)})
            except FeedbackPayloadTooLargeError as exc:
                self._send_json(413, {"error": str(exc)})
            except FeedbackValidationError as exc:
                self._send_json(400, {"error": str(exc)})
            return

        if parsed.path.startswith("/api/catalog/"):
            item_id = urllib.parse.unquote(parsed.path[len("/api/catalog/") :]).strip("/")
            try:
                upload_svc = self._require_upload_service()
                payload = self._read_json_body(
                    self._spatial_server.beta_config.limits.max_request_bytes
                )
                updated_res = upload_svc.update_item_metadata(item_id, payload)
                self._send_json(200, updated_res)
            except (FeedbackNotFoundError, CatalogItemNotFoundError) as exc:
                self._send_json(404, {"error": str(exc)})
            except FeedbackPayloadTooLargeError as exc:
                self._send_json(413, {"error": str(exc)})
            except (FeedbackValidationError, UploadValidationError) as exc:
                self._send_json(400, {"error": str(exc)})
            return

        self._send_json(404, {"error": f"Unknown endpoint: {parsed.path}"})

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path.startswith("/api/feedback/"):
            entry_id = urllib.parse.unquote(parsed.path[len("/api/feedback/") :]).strip("/")
            try:
                store = self._require_active_store()
                removed = store.delete_entry(entry_id)
                self._send_json(200, {"deleted": removed.entry_id})
            except FeedbackNotFoundError as exc:
                self._send_json(404, {"error": str(exc)})
            except FeedbackValidationError as exc:
                self._send_json(400, {"error": str(exc)})
            return

        self._send_json(404, {"error": f"Unknown endpoint: {parsed.path}"})

    def do_HEAD(self) -> None:
        self._handle_get_or_head(head_only=True)

    def do_GET(self) -> None:
        self._handle_get_or_head(head_only=False)

    def _handle_get_or_head(self, *, head_only: bool) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/__diag":
            qs = urllib.parse.parse_qs(parsed.query)
            payload = qs.get("d", [""])[0]
            if not head_only:
                self._record_diagnostic(payload)
            self.send_response(204)
            self.end_headers()
            return

        if parsed.path == "/api/beta-config":
            self._send_json(200, self._spatial_server.beta_config.to_dict(), head_only=head_only)
            return

        if parsed.path == "/api/feedback":
            try:
                store = self._require_active_store()
                qs = urllib.parse.parse_qs(parsed.query)
                scene_filter = qs.get("scene", [None])[0]
                include_resolved_raw = qs.get("include_resolved", ["1"])[0]
                include_resolved = include_resolved_raw not in ("0", "false", "False")
                entries = store.list_entries(
                    scene_id=scene_filter, include_resolved=include_resolved
                )
                self._send_json(
                    200,
                    {
                        "entries": [e.to_dict() for e in entries],
                        "count": len(entries),
                    },
                    head_only=head_only,
                )
            except FeedbackNotFoundError as exc:
                self._send_json(404, {"error": str(exc)}, head_only=head_only)
            except FeedbackValidationError as exc:
                self._send_json(400, {"error": str(exc)}, head_only=head_only)
            return

        if parsed.path.startswith("/api/feedback/snapshots/"):
            snap_name = urllib.parse.unquote(
                parsed.path[len("/api/feedback/snapshots/") :]
            ).strip("/")
            try:
                store = self._require_active_store()
                snap_path = store.resolve_snapshot_path(snap_name)
                self._send_file_bytes(
                    snap_path,
                    cache_policy=f"public, max-age={SNAPSHOT_CACHE_SECONDS}",
                    head_only=head_only,
                )
            except FeedbackNotFoundError as exc:
                self._send_json(404, {"error": str(exc)}, head_only=head_only)
            except FeedbackValidationError as exc:
                self._send_json(400, {"error": str(exc)}, head_only=head_only)
            return

        if parsed.path == "/api/wiki":
            try:
                wiki_svc = self._require_wiki_service()
                qs = urllib.parse.parse_qs(parsed.query)
                page_rel = qs.get("page", [None])[0]
                if page_rel is not None:
                    detail = wiki_svc.get_page(page_rel)
                    self._send_json(200, {"page": detail.to_dict()}, head_only=head_only)
                else:
                    nav = wiki_svc.list_navigation()
                    self._send_json(200, nav, head_only=head_only)
            except (FeedbackNotFoundError, WikiNotFoundError) as exc:
                self._send_json(404, {"error": str(exc)}, head_only=head_only)
            except WikiValidationError as exc:
                self._send_json(400, {"error": str(exc)}, head_only=head_only)
            return

        if parsed.path == "/api/catalog":
            try:
                upload_svc = self._require_upload_service()
                items = upload_svc.get_catalog_items()
                self._send_json(
                    200,
                    {"items": items, "count": len(items)},
                    head_only=head_only,
                )
            except FeedbackNotFoundError as exc:
                self._send_json(404, {"error": str(exc)}, head_only=head_only)
            return

        # Portal HTML routes (/wiki, /catalog) when beta mode is enabled
        if parsed.path in ("/wiki", "/wiki/"):
            if not self._spatial_server.beta_config.enabled:
                self._send_json(404, {"error": "Wiki portal disabled."}, head_only=head_only)
                return
            portal_html = Path(self.directory) / "portal" / "wiki.html"
            if portal_html.is_file():
                self._send_file_bytes(portal_html, head_only=head_only)
                return

        if parsed.path in ("/catalog", "/catalog/"):
            if not self._spatial_server.beta_config.enabled:
                self._send_json(404, {"error": "Catalog portal disabled."}, head_only=head_only)
                return
            portal_html = Path(self.directory) / "portal" / "catalog.html"
            if portal_html.is_file():
                self._send_file_bytes(portal_html, head_only=head_only)
                return

        # Read-only serving of /media/<source_group>/<filename> for curatorial story strips & catalog
        if parsed.path.startswith("/media/"):
            rel_part = urllib.parse.unquote(parsed.path[len("/media/") :]).lstrip("/")
            media_root = self._spatial_server.media_root_dir
            candidate = (media_root / rel_part).resolve()
            try:
                candidate.relative_to(media_root)
            except ValueError:
                self._send_json(400, {"error": "Invalid media path."}, head_only=head_only)
                return
            if candidate.is_file():
                self._send_file_bytes(
                    candidate,
                    cache_policy=f"public, max-age={STATIC_CACHE_SECONDS}",
                    head_only=head_only,
                )
                return
            self._send_json(404, {"error": "Media file not found."}, head_only=head_only)
            return

        if parsed.path.startswith("/js/vendor/") or parsed.path.startswith("/assets/"):
            self._cache_policy = f"public, max-age={STATIC_CACHE_SECONDS}"
        else:
            self._cache_policy = "no-cache, must-revalidate"

        local_path = Path(self.translate_path(self.path))
        if local_path.is_dir():
            for idx in ("index.html", "index.htm"):
                candidate = local_path / idx
                if candidate.is_file():
                    local_path = candidate
                    break

        if local_path.is_file():
            self._send_file_bytes(
                local_path, cache_policy=self._cache_policy, head_only=head_only
            )
            return

        if head_only:
            super().do_HEAD()
        else:
            super().do_GET()

    def copyfile(self, source: Any, outputfile: Any) -> None:
        try:
            super().copyfile(source, outputfile)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass

    def log_message(self, format: str, *args: Any) -> None:
        _safe_stderr_log(f"[{self.log_date_time_string()}] {format % args}\n")


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Checks whether a port is currently open and bindable."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(
    start_port: int = DEFAULT_PORT,
    max_attempts: int = PORT_COLLISION_MAX_ATTEMPTS,
    host: str = "127.0.0.1",
) -> int:
    """Finds first available port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        if is_port_free(p, host):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def create_server(
    port: int | None = None,
    directory: Path | str | None = None,
    host: str = "127.0.0.1",
    diag_log: Path | str | None = None,
    feedback_dir: Path | str | None = None,
    archive_root: Path | str | None = None,
    enable_beta: bool = True,
) -> tuple[SpatialThreadingHTTPServer, int]:
    """Creates a SpatialThreadingHTTPServer instance with collision avoidance and Family Beta services."""
    resolved_dir = Path(directory).resolve() if directory is not None else DEFAULT_DIR
    resolved_diag = str(Path(diag_log).resolve()) if diag_log is not None else None

    beta_cfg = default_beta_feedback_config(enabled=enable_beta)
    ephemeral_dir: str | None = None
    store: BetaFeedbackStore | None = None
    wiki_svc: WikiPortalService | None = None
    upload_svc: FamilyPhotoUploadService | None = None
    media_root_dir = PROJECT_ROOT / "media"

    if enable_beta:
        if feedback_dir is not None:
            resolved_feedback_dir = Path(feedback_dir).resolve()
        else:
            ephemeral_dir = tempfile.mkdtemp(prefix="tami_beta_ephemeral_")
            resolved_feedback_dir = Path(ephemeral_dir) / "feedback"
        store = BetaFeedbackStore(beta_cfg, resolved_feedback_dir)

        wiki_dir = PROJECT_ROOT / "wiki"
        if wiki_dir.is_dir():
            wiki_svc = WikiPortalService(wiki_dir)

        if archive_root is not None:
            root_for_uploads = Path(archive_root).resolve()
        else:
            if ephemeral_dir is None:
                ephemeral_dir = tempfile.mkdtemp(prefix="tami_beta_ephemeral_")
            root_for_uploads = Path(ephemeral_dir) / "archive_sandbox"
            (root_for_uploads / "drop_folder").mkdir(parents=True, exist_ok=True)
            (root_for_uploads / "media").mkdir(parents=True, exist_ok=True)
            (root_for_uploads / "catalog").mkdir(parents=True, exist_ok=True)

        media_root_dir = root_for_uploads / "media"
        upload_svc = FamilyPhotoUploadService(
            project_root=root_for_uploads,
            drop_dir=root_for_uploads / "drop_folder",
            media_dir=root_for_uploads / "media",
            catalog_dir=root_for_uploads / "catalog",
            limits=beta_cfg.uploads,
        )

    if port is None or not is_port_free(port, host):
        port = find_free_port(port or DEFAULT_PORT, host=host)

    handler = lambda *args, **kwargs: SpatialHTTPRequestHandler(
        *args, directory=str(resolved_dir), **kwargs
    )
    server = SpatialThreadingHTTPServer(
        (host, port),
        handler,
        beta_config=beta_cfg,
        feedback_store=store,
        wiki_service=wiki_svc,
        upload_service=upload_svc,
        media_root_dir=media_root_dir,
        diag_log_path=resolved_diag,
        ephemeral_feedback_dir=ephemeral_dir,
    )
    return server, port


def start_server_thread(
    port: int | None = None,
    directory: Path | str | None = None,
    host: str = "127.0.0.1",
    feedback_dir: Path | str | None = None,
    archive_root: Path | str | None = None,
    enable_beta: bool = True,
) -> tuple[SpatialThreadingHTTPServer, int, threading.Thread]:
    """Starts server in a background daemon thread for testing."""
    server, actual_port = create_server(
        port=port,
        directory=directory,
        host=host,
        feedback_dir=feedback_dir,
        archive_root=archive_root,
        enable_beta=enable_beta,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port, thread


def main() -> None:
    default_cfg = default_beta_feedback_config()
    default_feedback_dir = PROJECT_ROOT / default_cfg.storage_dir_relpath

    parser = argparse.ArgumentParser(
        description="Serve the Tami Sheaffer 70th Spatial Prototype & Family Beta Hub"
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Preferred port (default: {DEFAULT_PORT})",
    )
    parser.add_argument("--dir", "-d", default=str(DEFAULT_DIR), help="Directory to serve")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument(
        "--diag-log", default=None, help="Optional JSONL file path to persist mobile diagnostics"
    )
    parser.add_argument(
        "--feedback-dir",
        default=str(default_feedback_dir),
        help=f"Directory to persist beta feedback & snapshots (default: {default_feedback_dir})",
    )
    parser.add_argument(
        "--archive-root",
        default=str(PROJECT_ROOT),
        help=f"Project archive root containing drop_folder/, media/, catalog/ (default: {PROJECT_ROOT})",
    )
    parser.add_argument(
        "--disable-beta",
        action="store_true",
        help="Disable beta feedback UI and portals (clean gift presentation mode)",
    )
    args = parser.parse_args()

    serve_dir = Path(args.dir).resolve()
    if not (serve_dir / "index.html").exists():
        raise FileNotFoundError(f"Required prototype entry point missing: {serve_dir / 'index.html'}")

    enable_beta = not args.disable_beta
    server, port = create_server(
        port=args.port,
        directory=serve_dir,
        host=args.host,
        diag_log=args.diag_log,
        feedback_dir=args.feedback_dir if enable_beta else None,
        archive_root=args.archive_root if enable_beta else None,
        enable_beta=enable_beta,
    )
    url = f"http://{args.host}:{port}"
    print("=" * 68)
    print("TAMAR SHEAFFER 70TH BIRTHDAY — FAMILY BETA HUB & SPATIAL PROTOTYPE")
    print(f"Serving directory: {serve_dir}")
    print(f"Local URL:         {url}")
    print(f"Wiki Portal:       {url}/wiki")
    print(f"Catalog & Upload:  {url}/catalog")
    print(f"Beta Feedback:     {'ENABLED -> ' + str(Path(args.feedback_dir).resolve()) if enable_beta else 'DISABLED'}")
    print(f"Collision check:   Port {args.port} -> Bound port {port}")
    print("=" * 68)
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
