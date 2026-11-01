#!/usr/bin/env python3
"""
prototype/feedback_store.py
Domain validation and thread-safe persistence service for the Beta Feedback,
Suggestion Box, and 3D/2D Spatial Element Highlighting system.

Stores structured JSONL entries alongside verified JPEG/PNG viewport highlight snapshots
so both human reviewers (in the web UI) and coding agents (on local disk) can inspect
exact 3D world coordinates, layer IDs, texture UV/pixel coordinates, and annotated screenshots.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import re
import secrets
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from prototype.beta_config import BetaFeedbackConfig

VALID_HIGHLIGHT_MODES = frozenset({"point", "box", "ui"})
SAFE_LAYER_ID_RE = re.compile(r"^[A-Za-z0-9_:-]{1,96}$")
SAFE_SCENE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
VALID_ENTRY_STATUSES = frozenset({"open", "resolved"})
SAFE_SNAPSHOT_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.(?:jpg|jpeg|png)$")
DEFAULT_SCENE_MANIFEST_PATH = (
    _PROJECT_ROOT / "prototype" / "assets" / "scenes" / "scene_manifest.json"
)


class FeedbackValidationError(ValueError):
    """Raised when an incoming feedback or highlight payload violates schema invariants."""


class FeedbackPayloadTooLargeError(ValueError):
    """Raised when an incoming request or snapshot exceeds configured byte limits."""


class FeedbackNotFoundError(KeyError):
    """Raised when referencing a non-existent feedback entry or snapshot."""


def _require_finite_float(val: Any, field_name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise FeedbackValidationError(f"Field '{field_name}' must be a finite number.")
    fval = float(val)
    if fval != fval or fval in (float("inf"), float("-inf")):
        raise FeedbackValidationError(f"Field '{field_name}' must be finite (got NaN/Inf).")
    return fval


def _require_unit_interval(val: Any, field_name: str) -> float:
    fval = _require_finite_float(val, field_name)
    if fval < -0.05 or fval > 1.05:
        raise FeedbackValidationError(
            f"Field '{field_name}' must lie within normalized viewport range [0, 1] (got {fval:.4f})."
        )
    return max(0.0, min(1.0, round(fval, 5)))


@dataclass(frozen=True)
class CameraPoseRecord:
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    fov_deg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraPoseRecord:
        if not isinstance(data, dict):
            raise FeedbackValidationError("camera_pose must be an object.")
        return cls(
            x=round(_require_finite_float(data.get("x"), "camera_pose.x"), 3),
            y=round(_require_finite_float(data.get("y"), "camera_pose.y"), 3),
            z=round(_require_finite_float(data.get("z"), "camera_pose.z"), 3),
            yaw_deg=round(_require_finite_float(data.get("yaw_deg"), "camera_pose.yaw_deg"), 3),
            pitch_deg=round(
                _require_finite_float(data.get("pitch_deg"), "camera_pose.pitch_deg"), 3
            ),
            fov_deg=round(_require_finite_float(data.get("fov_deg"), "camera_pose.fov_deg"), 3),
        )


@dataclass(frozen=True)
class NormalizedRect:
    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedRect:
        if not isinstance(data, dict):
            raise FeedbackValidationError("screen_rect_norm must be an object.")
        raw_x0 = _require_unit_interval(data.get("x0"), "screen_rect_norm.x0")
        raw_y0 = _require_unit_interval(data.get("y0"), "screen_rect_norm.y0")
        raw_x1 = _require_unit_interval(data.get("x1"), "screen_rect_norm.x1")
        raw_y1 = _require_unit_interval(data.get("y1"), "screen_rect_norm.y1")
        return cls(
            x0=min(raw_x0, raw_x1),
            y0=min(raw_y0, raw_y1),
            x1=max(raw_x0, raw_x1),
            y1=max(raw_y0, raw_y1),
        )


@dataclass(frozen=True)
class SpatialHighlightRecord:
    mode: str
    layer_id: str
    layer_label_he: str
    screen_rect_norm: NormalizedRect
    pin_screen_norm: tuple[float, float]
    world_point: tuple[float, float, float] | None
    world_box_corners: tuple[tuple[float, float, float], ...] | None
    texture_uv: tuple[float, float] | None
    texture_pixel: tuple[int, int] | None
    ui_selector: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "layer_id": self.layer_id,
            "layer_label_he": self.layer_label_he,
            "screen_rect_norm": self.screen_rect_norm.to_dict(),
            "pin_screen_norm": list(self.pin_screen_norm),
            "world_point": list(self.world_point) if self.world_point is not None else None,
            "world_box_corners": (
                [list(pt) for pt in self.world_box_corners]
                if self.world_box_corners is not None
                else None
            ),
            "texture_uv": list(self.texture_uv) if self.texture_uv is not None else None,
            "texture_pixel": list(self.texture_pixel) if self.texture_pixel is not None else None,
            "ui_selector": self.ui_selector,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialHighlightRecord:
        if not isinstance(data, dict):
            raise FeedbackValidationError("highlight must be an object.")
        mode = str(data.get("mode", "")).strip()
        if mode not in VALID_HIGHLIGHT_MODES:
            raise FeedbackValidationError(
                f"Unsupported highlight.mode '{mode}'. Expected one of {sorted(VALID_HIGHLIGHT_MODES)}."
            )
        layer_id = str(data.get("layer_id", "")).strip()
        if not SAFE_LAYER_ID_RE.match(layer_id):
            raise FeedbackValidationError(
                f"Unsupported highlight.layer_id '{layer_id}'. Expected a safe identifier matching {SAFE_LAYER_ID_RE.pattern}."
            )
        layer_label_he = str(data.get("layer_label_he", "")).strip()
        if not layer_label_he:
            raise FeedbackValidationError("highlight.layer_label_he must be non-empty.")

        screen_rect = NormalizedRect.from_dict(data.get("screen_rect_norm"))
        raw_pin = data.get("pin_screen_norm")
        if not isinstance(raw_pin, (list, tuple)) or len(raw_pin) != 2:
            raise FeedbackValidationError("highlight.pin_screen_norm must be a [x, y] pair.")
        pin_screen = (
            _require_unit_interval(raw_pin[0], "pin_screen_norm[0]"),
            _require_unit_interval(raw_pin[1], "pin_screen_norm[1]"),
        )

        world_point: tuple[float, float, float] | None = None
        if data.get("world_point") is not None:
            wp = data["world_point"]
            if not isinstance(wp, (list, tuple)) or len(wp) != 3:
                raise FeedbackValidationError("highlight.world_point must be a [x, y, z] triple.")
            world_point = (
                round(_require_finite_float(wp[0], "world_point[0]"), 3),
                round(_require_finite_float(wp[1], "world_point[1]"), 3),
                round(_require_finite_float(wp[2], "world_point[2]"), 3),
            )

        world_box_corners: tuple[tuple[float, float, float], ...] | None = None
        if data.get("world_box_corners") is not None:
            raw_corners = data["world_box_corners"]
            if not isinstance(raw_corners, (list, tuple)) or len(raw_corners) != 4:
                raise FeedbackValidationError(
                    "highlight.world_box_corners must contain exactly 4 [x, y, z] corners."
                )
            parsed_corners: list[tuple[float, float, float]] = []
            for idx, c in enumerate(raw_corners):
                if not isinstance(c, (list, tuple)) or len(c) != 3:
                    raise FeedbackValidationError(
                        f"highlight.world_box_corners[{idx}] must be a [x, y, z] triple."
                    )
                parsed_corners.append(
                    (
                        round(_require_finite_float(c[0], f"world_box_corners[{idx}][0]"), 3),
                        round(_require_finite_float(c[1], f"world_box_corners[{idx}][1]"), 3),
                        round(_require_finite_float(c[2], f"world_box_corners[{idx}][2]"), 3),
                    )
                )
            world_box_corners = tuple(parsed_corners)

        texture_uv: tuple[float, float] | None = None
        if data.get("texture_uv") is not None:
            uv = data["texture_uv"]
            if not isinstance(uv, (list, tuple)) or len(uv) != 2:
                raise FeedbackValidationError("highlight.texture_uv must be a [u, v] pair.")
            texture_uv = (
                _require_unit_interval(uv[0], "texture_uv[0]"),
                _require_unit_interval(uv[1], "texture_uv[1]"),
            )

        texture_pixel: tuple[int, int] | None = None
        if data.get("texture_pixel") is not None:
            px = data["texture_pixel"]
            if not isinstance(px, (list, tuple)) or len(px) != 2:
                raise FeedbackValidationError("highlight.texture_pixel must be a [x, y] pair.")
            texture_pixel = (
                int(round(_require_finite_float(px[0], "texture_pixel[0]"))),
                int(round(_require_finite_float(px[1], "texture_pixel[1]"))),
            )

        ui_selector_raw = data.get("ui_selector")
        ui_selector = str(ui_selector_raw).strip()[:160] if ui_selector_raw else None

        return cls(
            mode=mode,
            layer_id=layer_id,
            layer_label_he=layer_label_he[:120],
            screen_rect_norm=screen_rect,
            pin_screen_norm=pin_screen,
            world_point=world_point,
            world_box_corners=world_box_corners,
            texture_uv=texture_uv,
            texture_pixel=texture_pixel,
            ui_selector=ui_selector,
        )


@dataclass(frozen=True)
class FeedbackEntry:
    entry_id: str
    seq_num: int
    created_at: str
    author: str
    category_id: str
    scene_id: str
    comment: str
    status: str
    camera_pose: CameraPoseRecord | None
    highlight: SpatialHighlightRecord | None
    snapshot_filename: str | None
    viewport: tuple[int, int] | None
    user_agent_hint: str | None
    page_path: str | None
    target_ref: str | None
    creation_mode: str | None = None
    selected_media_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "seq_num": self.seq_num,
            "created_at": self.created_at,
            "author": self.author,
            "category_id": self.category_id,
            "scene_id": self.scene_id,
            "comment": self.comment,
            "status": self.status,
            "camera_pose": self.camera_pose.to_dict() if self.camera_pose is not None else None,
            "highlight": self.highlight.to_dict() if self.highlight is not None else None,
            "snapshot_filename": self.snapshot_filename,
            "snapshot_url": (
                f"/api/feedback/snapshots/{self.snapshot_filename}"
                if self.snapshot_filename
                else None
            ),
            "viewport": list(self.viewport) if self.viewport is not None else None,
            "user_agent_hint": self.user_agent_hint,
            "page_path": self.page_path,
            "target_ref": self.target_ref,
            "creation_mode": self.creation_mode,
            "selected_media_ids": list(self.selected_media_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackEntry:
        vp = data.get("viewport")
        viewport = (int(vp[0]), int(vp[1])) if isinstance(vp, (list, tuple)) and len(vp) == 2 else None
        raw_media_ids = data.get("selected_media_ids")
        media_ids: tuple[str, ...] = ()
        if isinstance(raw_media_ids, (list, tuple)):
            media_ids = tuple(str(m).strip()[:64] for m in raw_media_ids if str(m).strip())
        return cls(
            entry_id=str(data["entry_id"]),
            seq_num=int(data["seq_num"]),
            created_at=str(data["created_at"]),
            author=str(data["author"]),
            category_id=str(data["category_id"]),
            scene_id=str(data["scene_id"]),
            comment=str(data["comment"]),
            status=str(data.get("status", "open")),
            camera_pose=(
                CameraPoseRecord.from_dict(data["camera_pose"])
                if data.get("camera_pose") is not None
                else None
            ),
            highlight=(
                SpatialHighlightRecord.from_dict(data["highlight"])
                if data.get("highlight") is not None
                else None
            ),
            snapshot_filename=(
                str(data["snapshot_filename"]) if data.get("snapshot_filename") else None
            ),
            viewport=viewport,
            user_agent_hint=(
                str(data["user_agent_hint"]) if data.get("user_agent_hint") else None
            ),
            page_path=str(data["page_path"]).strip()[:200] if data.get("page_path") else None,
            target_ref=str(data["target_ref"]).strip()[:200] if data.get("target_ref") else None,
            creation_mode=(
                str(data["creation_mode"]).strip()[:48] if data.get("creation_mode") else None
            ),
            selected_media_ids=media_ids,
        )


def _decode_and_verify_snapshot(
    data_url: str, max_bytes: int, jpeg_quality: int
) -> tuple[bytes, str]:
    """Decodes a base64 image data URL, verifies raster integrity via Pillow, and normalizes to JPEG."""
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise FeedbackValidationError(
            "snapshot_data_url must start with 'data:image/jpeg;base64,' or 'data:image/png;base64,'."
        )
    header_sep = data_url.find(",")
    if header_sep == -1:
        raise FeedbackValidationError("Malformed snapshot_data_url: missing base64 comma separator.")
    header = data_url[:header_sep].lower()
    if ";base64" not in header:
        raise FeedbackValidationError("Malformed snapshot_data_url: expected ';base64' encoding.")

    b64_payload = data_url[header_sep + 1 :].strip()
    try:
        raw_bytes = base64.b64decode(b64_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FeedbackValidationError(f"Invalid base64 image payload: {exc}") from exc

    if len(raw_bytes) == 0:
        raise FeedbackValidationError("Decoded snapshot image is empty (0 bytes).")
    if len(raw_bytes) > max_bytes:
        raise FeedbackPayloadTooLargeError(
            f"Decoded snapshot size ({len(raw_bytes)} bytes) exceeds max_snapshot_bytes ({max_bytes})."
        )

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw_bytes)) as img2:
            rgb = img2.convert("RGB")
            out = io.BytesIO()
            rgb.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
            return out.getvalue(), ".jpg"
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise FeedbackValidationError(f"Corrupt or unsupported snapshot image stream: {exc}") from exc


class BetaFeedbackStore:
    """Thread-safe file-backed store for collaborative beta feedback and spatial highlights."""

    def __init__(
        self,
        config: BetaFeedbackConfig,
        storage_dir: Path,
        *,
        scene_manifest_path: Path | None = None,
    ) -> None:
        self.config = config
        self.storage_dir = storage_dir.resolve()
        self.entries_path = self.storage_dir / config.entries_filename
        self.summary_path = self.storage_dir / config.summary_filename
        self.snapshots_dir = self.storage_dir / config.snapshots_subdir
        self.scene_manifest_path = (
            scene_manifest_path if scene_manifest_path is not None else DEFAULT_SCENE_MANIFEST_PATH
        ).resolve()
        self._lock = threading.Lock()
        self._valid_categories = frozenset(c.category_id for c in config.categories)
        self._valid_scenes = frozenset(config.valid_scene_ids)
        self._valid_creation_modes = frozenset(m.mode_id for m in config.creative_studio.modes)

    def _effective_valid_scenes(self) -> frozenset[str]:
        """Unions configured valid_scene_ids with any live scene keys in scene_manifest.json."""
        if not self.scene_manifest_path.is_file():
            return self._valid_scenes
        try:
            manifest_data = json.loads(self.scene_manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Failed to read scene manifest at {self.scene_manifest_path}: {exc}"
            ) from exc
        if not isinstance(manifest_data, dict):
            return self._valid_scenes
        raw_scenes = manifest_data.get("scenes")
        manifest_ids: set[str] = set()
        if isinstance(raw_scenes, dict):
            for key in raw_scenes.keys():
                clean_key = str(key).strip()
                if SAFE_SCENE_ID_RE.match(clean_key):
                    manifest_ids.add(clean_key)
        elif isinstance(raw_scenes, list):
            for item in raw_scenes:
                if isinstance(item, dict):
                    raw_id = item.get("scene_id") or item.get("id") or ""
                    clean_key = str(raw_id).strip()
                    if SAFE_SCENE_ID_RE.match(clean_key):
                        manifest_ids.add(clean_key)
        if not manifest_ids:
            return self._valid_scenes
        return self._valid_scenes | frozenset(manifest_ids)

    def _ensure_dirs(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _load_all_unlocked(self) -> list[FeedbackEntry]:
        if not self.entries_path.exists():
            return []
        entries: list[FeedbackEntry] = []
        raw_text = self.entries_path.read_text(encoding="utf-8")
        for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                entries.append(FeedbackEntry.from_dict(data))
            except Exception as exc:
                raise RuntimeError(
                    f"Corrupt feedback record in {self.entries_path}:{line_no}: {exc}"
                ) from exc
        return entries

    def _write_all_unlocked(self, entries: list[FeedbackEntry]) -> None:
        self._ensure_dirs()
        serialized_lines = "".join(
            json.dumps(entry.to_dict(), ensure_ascii=False) + "\n" for entry in entries
        )
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(self.storage_dir), prefix=".entries_tmp_", suffix=".jsonl"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(serialized_lines)
            os.replace(tmp_path_str, self.entries_path)
        except Exception:
            if os.path.exists(tmp_path_str):
                os.unlink(tmp_path_str)
            raise

        self._write_summary_unlocked(entries)

    def _write_summary_unlocked(self, entries: list[FeedbackEntry]) -> None:
        open_entries = [e for e in entries if e.status == "open"]
        by_scene: dict[str, int] = {s: 0 for s in self.config.valid_scene_ids}
        for s in sorted(self._effective_valid_scenes()):
            by_scene.setdefault(s, 0)
        for e in open_entries:
            by_scene[e.scene_id] = by_scene.get(e.scene_id, 0) + 1

        summary = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total_count": len(entries),
            "open_count": len(open_entries),
            "resolved_count": len(entries) - len(open_entries),
            "open_by_scene": by_scene,
            "entries": [e.to_dict() for e in entries],
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def list_entries(
        self,
        *,
        scene_id: str | None = None,
        include_resolved: bool = True,
    ) -> list[FeedbackEntry]:
        valid_scenes = self._effective_valid_scenes()
        with self._lock:
            entries = self._load_all_unlocked()
        if scene_id is not None:
            if scene_id not in valid_scenes:
                raise FeedbackValidationError(
                    f"Unknown scene_id filter '{scene_id}'. Expected one of {sorted(valid_scenes)}."
                )
            entries = [e for e in entries if e.scene_id == scene_id]
        if not include_resolved:
            entries = [e for e in entries if e.status == "open"]
        return entries

    def create_entry(
        self,
        payload: dict[str, Any],
        *,
        user_agent_hint: str | None = None,
    ) -> FeedbackEntry:
        if not isinstance(payload, dict):
            raise FeedbackValidationError("Feedback payload must be a JSON object.")

        author = str(payload.get("author", "")).strip()
        if not author:
            raise FeedbackValidationError("יש להזין שם שולח/ת (למשל טל או נעמה).")
        if len(author) > self.config.limits.max_author_chars:
            raise FeedbackValidationError(
                f"Author name exceeds {self.config.limits.max_author_chars} characters."
            )

        category_id = str(payload.get("category_id", "")).strip()
        if category_id not in self._valid_categories:
            raise FeedbackValidationError(
                f"Unsupported category_id '{category_id}'. Valid options: {sorted(self._valid_categories)}."
            )

        valid_scenes = self._effective_valid_scenes()
        scene_id = str(payload.get("scene_id", "")).strip()
        if scene_id not in valid_scenes:
            raise FeedbackValidationError(
                f"Unsupported scene_id '{scene_id}'. Valid options: {sorted(valid_scenes)}."
            )

        comment = str(payload.get("comment", "")).strip()
        if not comment:
            raise FeedbackValidationError("יש לכתוב הערה או הצעה לפני השליחה.")
        if len(comment) > self.config.limits.max_comment_chars:
            raise FeedbackValidationError(
                f"Comment exceeds {self.config.limits.max_comment_chars} characters."
            )

        camera_pose = (
            CameraPoseRecord.from_dict(payload["camera_pose"])
            if payload.get("camera_pose") is not None
            else None
        )
        highlight = (
            SpatialHighlightRecord.from_dict(payload["highlight"])
            if payload.get("highlight") is not None
            else None
        )

        viewport: tuple[int, int] | None = None
        if payload.get("viewport") is not None:
            vp = payload["viewport"]
            if not isinstance(vp, (list, tuple)) or len(vp) != 2:
                raise FeedbackValidationError("viewport must be a [width, height] pair.")
            vw = int(round(_require_finite_float(vp[0], "viewport[0]")))
            vh = int(round(_require_finite_float(vp[1], "viewport[1]")))
            if vw <= 0 or vh <= 0:
                raise FeedbackValidationError("viewport dimensions must be positive integers.")
            viewport = (vw, vh)

        page_path = (
            str(payload["page_path"]).strip()[:200] if payload.get("page_path") else None
        )
        target_ref = (
            str(payload["target_ref"]).strip()[:200] if payload.get("target_ref") else None
        )

        creation_mode: str | None = None
        if payload.get("creation_mode") is not None and str(payload.get("creation_mode")).strip():
            creation_mode = str(payload["creation_mode"]).strip()
            if creation_mode not in self._valid_creation_modes:
                raise FeedbackValidationError(
                    f"Unsupported creation_mode '{creation_mode}'. Valid options: {sorted(self._valid_creation_modes)}."
                )

        selected_media_ids: tuple[str, ...] = ()
        if payload.get("selected_media_ids") is not None:
            raw_ids = payload["selected_media_ids"]
            if not isinstance(raw_ids, (list, tuple)):
                raise FeedbackValidationError("selected_media_ids must be a list of media IDs.")
            cleaned_ids = [str(m).strip()[:64] for m in raw_ids if str(m).strip()]
            max_sel = self.config.creative_studio.max_selected_photos
            if len(cleaned_ids) > max_sel:
                raise FeedbackValidationError(
                    f"selected_media_ids length ({len(cleaned_ids)}) exceeds max_selected_photos ({max_sel})."
                )
            selected_media_ids = tuple(cleaned_ids)

        snapshot_bytes: bytes | None = None
        snapshot_ext = ".jpg"
        if payload.get("snapshot_data_url"):
            snapshot_bytes, snapshot_ext = _decode_and_verify_snapshot(
                str(payload["snapshot_data_url"]),
                max_bytes=self.config.limits.max_snapshot_bytes,
                jpeg_quality=self.config.picking.snapshot_pillow_jpeg_quality,
            )

        ts_slug = time.strftime("%Y%m%d-%H%M%S")
        rand_hex = secrets.token_hex(3)
        entry_id = f"fb-{ts_slug}-{rand_hex}"
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        with self._lock:
            self._ensure_dirs()
            existing = self._load_all_unlocked()
            next_seq = (max((e.seq_num for e in existing), default=0)) + 1

            snapshot_filename: str | None = None
            if snapshot_bytes is not None:
                snapshot_filename = f"{entry_id}{snapshot_ext}"
                snap_path = self.snapshots_dir / snapshot_filename
                snap_path.write_bytes(snapshot_bytes)

            entry = FeedbackEntry(
                entry_id=entry_id,
                seq_num=next_seq,
                created_at=created_at,
                author=author,
                category_id=category_id,
                scene_id=scene_id,
                comment=comment,
                status="open",
                camera_pose=camera_pose,
                highlight=highlight,
                snapshot_filename=snapshot_filename,
                viewport=viewport,
                user_agent_hint=user_agent_hint[:180] if user_agent_hint else None,
                page_path=page_path,
                target_ref=target_ref,
                creation_mode=creation_mode,
                selected_media_ids=selected_media_ids,
            )
            existing.append(entry)
            self._write_all_unlocked(existing)

        return entry

    def update_status(self, entry_id: str, status: str) -> FeedbackEntry:
        clean_status = str(status).strip()
        if clean_status not in VALID_ENTRY_STATUSES:
            raise FeedbackValidationError(
                f"Invalid status '{clean_status}'. Expected one of {sorted(VALID_ENTRY_STATUSES)}."
            )

        with self._lock:
            entries = self._load_all_unlocked()
            updated_entry: FeedbackEntry | None = None
            new_entries: list[FeedbackEntry] = []
            for e in entries:
                if e.entry_id == entry_id:
                    updated_entry = FeedbackEntry(
                        entry_id=e.entry_id,
                        seq_num=e.seq_num,
                        created_at=e.created_at,
                        author=e.author,
                        category_id=e.category_id,
                        scene_id=e.scene_id,
                        comment=e.comment,
                        status=clean_status,
                        camera_pose=e.camera_pose,
                        highlight=e.highlight,
                        snapshot_filename=e.snapshot_filename,
                        viewport=e.viewport,
                        user_agent_hint=e.user_agent_hint,
                        page_path=e.page_path,
                        target_ref=e.target_ref,
                        creation_mode=e.creation_mode,
                        selected_media_ids=e.selected_media_ids,
                    )
                    new_entries.append(updated_entry)
                else:
                    new_entries.append(e)

            if updated_entry is None:
                raise FeedbackNotFoundError(f"Feedback entry '{entry_id}' not found.")

            self._write_all_unlocked(new_entries)
            return updated_entry

    def delete_entry(self, entry_id: str) -> FeedbackEntry:
        with self._lock:
            entries = self._load_all_unlocked()
            removed: FeedbackEntry | None = None
            kept: list[FeedbackEntry] = []
            for e in entries:
                if e.entry_id == entry_id:
                    removed = e
                else:
                    kept.append(e)

            if removed is None:
                raise FeedbackNotFoundError(f"Feedback entry '{entry_id}' not found.")

            if removed.snapshot_filename:
                snap_path = self.snapshots_dir / removed.snapshot_filename
                if snap_path.exists():
                    snap_path.unlink()

            self._write_all_unlocked(kept)
            return removed

    def resolve_snapshot_path(self, filename: str) -> Path:
        clean_name = str(filename).strip()
        if not SAFE_SNAPSHOT_FILENAME_RE.match(clean_name):
            raise FeedbackValidationError(f"Invalid snapshot filename '{clean_name}'.")
        candidate = (self.snapshots_dir / clean_name).resolve()
        if candidate.parent != self.snapshots_dir.resolve() or not candidate.is_file():
            raise FeedbackNotFoundError(f"Snapshot '{clean_name}' not found.")
        return candidate
