#!/usr/bin/env python3
"""
prototype/upload_service.py
Thread-safe domain service for family photo uploads into `drop_folder/` + `media/drop_folder_imports/`
and live metadata/caption enrichment across the synchronized 4-format catalog
(`catalog.json`, `catalog.csv`, `catalog.sqlite`, `catalog/index.html`).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from prototype.beta_config import UploadLimitsConfig
from scripts.build_catalog import SUPPORTED_EXTENSIONS, build_all_catalogs

HEIC_EXTENSIONS = frozenset({".heic", ".heif"})
SIPS_BINARY = Path("/usr/bin/sips")
SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_\u0590-\u05FF-]+")


class UploadValidationError(ValueError):
    """Raised when an uploaded image or metadata payload fails validation."""


class UploadPayloadTooLargeError(ValueError):
    """Raised when an upload exceeds configured byte limits."""


class CatalogItemNotFoundError(KeyError):
    """Raised when updating metadata for a non-existent catalog item ID."""


@dataclass(frozen=True)
class UploadedPhotoResult:
    status: str  # "ingested" | "duplicate"
    filename: str
    rel_path: str
    sha256: str
    width: int
    height: int
    title: str
    message_he: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "filename": self.filename,
            "rel_path": self.rel_path,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "title": self.title,
            "message_he": self.message_he,
        }


def _sanitize_stem(raw_name: str) -> str:
    stem = Path(raw_name).stem.strip()
    cleaned = SAFE_STEM_RE.sub("_", stem).strip("._-")
    return cleaned[:72] if cleaned else "family_photo"


def _decode_data_url_bytes(data_url: str, max_bytes: int) -> tuple[bytes, str]:
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise UploadValidationError("קובץ תמונה חסר או שאינו בפורמט data URL תקין.")
    comma_idx = data_url.find(",")
    if comma_idx == -1:
        raise UploadValidationError("פורמט base64 לא תקין בקובץ שהועלה.")
    header = data_url[:comma_idx].lower()
    if ";base64" not in header:
        raise UploadValidationError("נדרש קידוד base64 להעלאת קובץ.")

    b64_part = data_url[comma_idx + 1 :].strip()
    try:
        raw_bytes = base64.b64decode(b64_part, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UploadValidationError(f"שגיאה בפענוח קובץ התמונה: {exc}") from exc

    if len(raw_bytes) == 0:
        raise UploadValidationError("הקובץ שהועלה ריק (0 בתים).")
    if len(raw_bytes) > max_bytes:
        raise UploadPayloadTooLargeError(
            f"גודל התמונה ({len(raw_bytes) // (1024 * 1024)}MB) חורג מהמקסימום המותר."
        )
    mime = header[5:].split(";", 1)[0].strip()
    return raw_bytes, mime


def _convert_heic_if_needed(
    raw_bytes: bytes, orig_ext: str, mime: str, timeout_sec: int
) -> tuple[bytes, str]:
    is_heic = orig_ext.lower() in HEIC_EXTENSIONS or "heic" in mime or "heif" in mime
    if not is_heic:
        return raw_bytes, orig_ext.lower()

    if not SIPS_BINARY.is_file():
        raise UploadValidationError(
            "המרת קובצי HEIC דורשת /usr/bin/sips ב-macOS; נא להעלות JPG או PNG."
        )

    with tempfile.TemporaryDirectory(prefix="tami_heic_") as tmp_dir:
        in_path = Path(tmp_dir) / f"input{orig_ext.lower() or '.heic'}"
        out_path = Path(tmp_dir) / "converted.jpg"
        in_path.write_bytes(raw_bytes)
        res = subprocess.run(
            [str(SIPS_BINARY), "-s", "format", "jpeg", str(in_path), "--out", str(out_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        if res.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
            err_msg = res.stderr.decode("utf-8", errors="replace").strip()
            raise UploadValidationError(f"המרת קובץ HEIC נכשלה: {err_msg[:180]}")
        return out_path.read_bytes(), ".jpg"


def _normalize_and_validate_image(
    raw_bytes: bytes, orig_filename: str, mime: str, limits: UploadLimitsConfig
) -> tuple[bytes, str, int, int, str]:
    orig_ext = Path(orig_filename).suffix.lower()
    converted_bytes, ext = _convert_heic_if_needed(
        raw_bytes, orig_ext, mime, limits.sips_convert_timeout_sec
    )

    try:
        with Image.open(io.BytesIO(converted_bytes)) as verify_img:
            verify_img.verify()
        with Image.open(io.BytesIO(converted_bytes)) as img:
            img.load()
            # Apply EXIF orientation if rotated on phone camera, otherwise preserve lossless bytes
            exif_orientation = img.getexif().get(0x0112, 1)
            fmt = (img.format or "JPEG").upper()
            if exif_orientation != 1:
                transposed = ImageOps.exif_transpose(img)
                out_buf = io.BytesIO()
                save_fmt = "PNG" if fmt == "PNG" else "JPEG"
                if save_fmt == "JPEG":
                    transposed.convert("RGB").save(out_buf, format="JPEG", quality=93)
                    ext = ".jpg"
                else:
                    transposed.save(out_buf, format="PNG")
                    ext = ".png"
                final_bytes = out_buf.getvalue()
                width, height = transposed.size
            else:
                final_bytes = converted_bytes
                width, height = img.size
                if ext not in SUPPORTED_EXTENSIONS:
                    ext_map = {
                        "JPEG": ".jpg",
                        "PNG": ".png",
                        "WEBP": ".webp",
                        "GIF": ".gif",
                        "TIFF": ".tiff",
                    }
                    ext = ext_map.get(fmt, ".jpg")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise UploadValidationError(f"קובץ התמונה '{orig_filename}' פגום או בפורמט לא נתמך: {exc}") from exc

    if width < limits.min_image_dim_px or height < limits.min_image_dim_px:
        raise UploadValidationError(
            f"התמונה '{orig_filename}' קטנה מדי ({width}x{height}px — מינימום {limits.min_image_dim_px}px)."
        )

    sha256_hex = hashlib.sha256(final_bytes).hexdigest().lower()
    return final_bytes, ext, width, height, sha256_hex


def _clean_string_list(val: Any) -> list[str]:
    if isinstance(val, str):
        return [p.strip() for p in val.split(",") if p.strip()]
    if isinstance(val, (list, tuple)):
        return [str(p).strip() for p in val if str(p).strip()]
    return []


class FamilyPhotoUploadService:
    """Manages family photo uploads to `drop_folder/` + live catalog synchronization."""

    def __init__(
        self,
        *,
        project_root: Path,
        drop_dir: Path,
        media_dir: Path,
        catalog_dir: Path,
        limits: UploadLimitsConfig,
    ) -> None:
        self.project_root = project_root.resolve()
        self.drop_dir = drop_dir.resolve()
        self.media_dir = media_dir.resolve()
        self.catalog_dir = catalog_dir.resolve()
        self.media_imports_dir = self.media_dir / "drop_folder_imports"
        self.limits = limits
        self._lock = threading.Lock()

    def _load_existing_catalog(self) -> list[dict[str, Any]]:
        json_path = self.catalog_dir / "catalog.json"
        if not json_path.is_file():
            return []
        return json.loads(json_path.read_text(encoding="utf-8"))

    def get_catalog_items(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._load_existing_catalog()

    def ingest_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise UploadValidationError("בקשת ההעלאה חייבת להיות אובייקט JSON.")

        uploader = str(payload.get("uploader", "")).strip()
        if not uploader:
            raise UploadValidationError("נא לבחור או להזין את שם המעלה (למשל גדי, גיא, טל או נעמה).")

        raw_files = payload.get("files")
        if not isinstance(raw_files, list) or len(raw_files) == 0:
            raise UploadValidationError("לא נבחרו תמונות להעלאה.")
        if len(raw_files) > self.limits.max_files_per_batch:
            raise UploadValidationError(
                f"ניתן להעלות עד {self.limits.max_files_per_batch} תמונות בכל שליחה."
            )

        shared_caption = str(payload.get("caption", "")).strip()[:1200]
        shared_era = str(payload.get("date_or_era", "")).strip()[:80]
        shared_places = _clean_string_list(payload.get("places"))
        shared_people = _clean_string_list(payload.get("people"))
        shared_themes = _clean_string_list(payload.get("art_themes"))

        prepared: list[tuple[str, bytes, str, int, int, str, dict[str, Any]]] = []
        for item in raw_files:
            if not isinstance(item, dict):
                raise UploadValidationError("רשומת קובץ לא תקינה.")
            orig_name = str(item.get("filename") or "photo.jpg").strip()
            data_url = str(item.get("data_url") or "")
            raw_bytes, mime = _decode_data_url_bytes(data_url, self.limits.max_single_file_bytes)
            final_bytes, ext, w, h, sha256_hex = _normalize_and_validate_image(
                raw_bytes, orig_name, mime, self.limits
            )

            item_title = (
                str(item.get("title") or "").strip()[:160]
                or _sanitize_stem(orig_name).replace("_", " ")
            )
            item_caption = str(item.get("caption") or shared_caption).strip()[:1200]
            item_era = str(item.get("date_or_era") or shared_era).strip()[:80]
            item_places = _clean_string_list(item.get("places")) or list(shared_places)
            item_people = _clean_string_list(item.get("people")) or list(shared_people)
            item_themes = _clean_string_list(item.get("art_themes")) or list(shared_themes)

            sidecar = {
                "title": item_title,
                "caption": item_caption,
                "date_or_era": item_era,
                "people": item_people,
                "places": item_places,
                "art_themes": item_themes,
                "uploaded_by": uploader,
                "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "confidence_notes": f"CONFIRMED: הועלה על ידי {uploader} בפורטל המשפחתי",
            }
            prepared.append((orig_name, final_bytes, ext, w, h, sha256_hex, sidecar))

        results: list[UploadedPhotoResult] = []
        ingested_any = False

        with self._lock:
            existing_records = self._load_existing_catalog()
            existing_by_sha = {str(r.get("sha256", "")).lower(): r for r in existing_records}

            self.drop_dir.mkdir(parents=True, exist_ok=True)
            self.media_imports_dir.mkdir(parents=True, exist_ok=True)

            for orig_name, final_bytes, ext, w, h, sha256_hex, sidecar in prepared:
                if sha256_hex in existing_by_sha:
                    dup = existing_by_sha[sha256_hex]
                    results.append(
                        UploadedPhotoResult(
                            status="duplicate",
                            filename=str(dup.get("filename", orig_name)),
                            rel_path=str(dup.get("rel_path", "")),
                            sha256=sha256_hex,
                            width=w,
                            height=h,
                            title=str(dup.get("title", sidecar["title"])),
                            message_he=f"התמונה כבר קיימת בקטלוג ({dup.get('id')}).",
                        )
                    )
                    continue

                stem = _sanitize_stem(orig_name)
                dest_filename = f"{stem}_{sha256_hex[:8]}{ext}"
                drop_img_path = self.drop_dir / dest_filename
                drop_json_path = self.drop_dir / f"{dest_filename}.json"
                media_img_path = self.media_imports_dir / dest_filename
                media_json_path = self.media_imports_dir / f"{dest_filename}.json"

                sidecar_with_source = {
                    **sidecar,
                    "source_url_or_path": f"drop_folder/{dest_filename}",
                }
                sidecar_text = json.dumps(sidecar_with_source, ensure_ascii=False, indent=2) + "\n"

                drop_img_path.write_bytes(final_bytes)
                drop_json_path.write_text(sidecar_text, encoding="utf-8")
                shutil.copy2(drop_img_path, media_img_path)
                media_json_path.write_text(sidecar_text, encoding="utf-8")

                rel_path = str(media_img_path.relative_to(self.project_root)).replace("\\", "/")
                existing_by_sha[sha256_hex] = {
                    "id": "NEW",
                    "filename": dest_filename,
                    "rel_path": rel_path,
                    "title": sidecar["title"],
                }
                ingested_any = True
                results.append(
                    UploadedPhotoResult(
                        status="ingested",
                        filename=dest_filename,
                        rel_path=rel_path,
                        sha256=sha256_hex,
                        width=w,
                        height=h,
                        title=sidecar["title"],
                        message_he="נשמר ב-drop_folder ונוסף לקטלוג בהצלחה!",
                    )
                )

            if ingested_any:
                stats = build_all_catalogs(
                    project_root=self.project_root,
                    media_dir=self.media_dir,
                    catalog_dir=self.catalog_dir,
                    min_size=self.limits.min_image_dim_px,
                    verbose=False,
                )
                total_count = int(stats["count"])
            else:
                total_count = len(existing_records)

        return {
            "results": [r.to_dict() for r in results],
            "ingested_count": sum(1 for r in results if r.status == "ingested"),
            "duplicate_count": sum(1 for r in results if r.status == "duplicate"),
            "catalog_total_count": total_count,
        }

    def update_item_metadata(self, item_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Updates caption/title/places/people/date_or_era sidecar for any existing catalog image."""
        if not isinstance(payload, dict):
            raise UploadValidationError("בקשת העדכון חייבת להיות אובייקט JSON.")

        editor = str(payload.get("editor", "")).strip() or "משפחה"

        with self._lock:
            records = self._load_existing_catalog()
            target = next((r for r in records if r.get("id") == item_id), None)
            if target is None:
                raise CatalogItemNotFoundError(f"פריט קטלוג '{item_id}' לא נמצא.")

            img_path = (self.project_root / str(target["rel_path"])).resolve()
            if not img_path.is_file():
                raise CatalogItemNotFoundError(f"קובץ המדיה עבור '{item_id}' לא נמצא בדיסק.")

            sidecar_path = img_path.with_name(img_path.name + ".json")
            existing_sidecar: dict[str, Any] = {}
            if sidecar_path.is_file():
                try:
                    existing_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing_sidecar = {}

            new_title = (
                str(payload["title"]).strip()[:160]
                if "title" in payload and str(payload["title"]).strip()
                else str(target.get("title") or "")
            )
            new_caption = (
                str(payload["caption"]).strip()[:1200]
                if "caption" in payload
                else str(target.get("caption") or "")
            )
            new_era = (
                str(payload["date_or_era"]).strip()[:80]
                if "date_or_era" in payload
                else str(target.get("date_or_era") or "")
            )
            new_places = (
                _clean_string_list(payload["places"])
                if "places" in payload
                else list(target.get("places") or [])
            )
            new_people = (
                _clean_string_list(payload["people"])
                if "people" in payload
                else list(target.get("people") or [])
            )
            new_themes = (
                _clean_string_list(payload["art_themes"])
                if "art_themes" in payload
                else list(target.get("art_themes") or [])
            )

            updated_sidecar = {
                **existing_sidecar,
                "title": new_title,
                "caption": new_caption,
                "date_or_era": new_era,
                "places": new_places,
                "places_locked": True,
                "people": new_people,
                "people_locked": True,
                "art_themes": new_themes,
                "source_url_or_path": str(target.get("source_url_or_path") or ""),
                "confidence_notes": f"עודכן על ידי {editor} ({time.strftime('%d/%m/%Y %H:%M')})",
            }
            sidecar_path.write_text(
                json.dumps(updated_sidecar, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            build_all_catalogs(
                project_root=self.project_root,
                media_dir=self.media_dir,
                catalog_dir=self.catalog_dir,
                min_size=self.limits.min_image_dim_px,
                verbose=False,
            )
            refreshed = self._load_existing_catalog()
            updated_record = next(
                (r for r in refreshed if r.get("sha256") == target.get("sha256")),
                target,
            )
            return {"item": updated_record, "catalog_total_count": len(refreshed)}
