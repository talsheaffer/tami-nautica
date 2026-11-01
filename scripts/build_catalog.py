#!/usr/bin/env python3
"""
scripts/build_catalog.py — Master Media Catalog Builder & Multi-Format Serializer

Produces 4 synchronized outputs:
1. catalog/catalog.json   (Normalized JSON array with 17 schema fields)
2. catalog/catalog.csv    (UTF-8 with BOM for full Hebrew Excel/Numbers support)
3. catalog/catalog.sqlite (SQLite relational database with indexes and media view)
4. catalog/index.html     (Responsive HTML contact sheet with zero broken paths)

Schema (17 fields):
  id, filename, rel_path, sha256, width, height, file_size_bytes,
  mime_type, source_group, source_url_or_path, title, caption,
  date_or_era, people, places, art_themes, confidence_notes
"""

from __future__ import annotations

import os
import sys
import json
import csv
import re
import hashlib
import sqlite3
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass(frozen=True)
class UprightDerivativeConfig:
    """SSOT for upright display derivatives of archival images whose stored pixels are rotated.

    A media sidecar may declare ``display_rotation_cw_deg`` (one of ``allowed_rotations_cw_deg``).
    The archival file is never modified; instead the catalog build writes a rotated JPEG copy under
    ``<media_dir>/<derivatives_subdir>/`` (excluded from scanning) and points the entry's
    ``display_rel_path`` at it. Entries without rotation get ``display_rel_path == rel_path``.
    """

    sidecar_field: str = "display_rotation_cw_deg"
    allowed_rotations_cw_deg: tuple[int, ...] = (0, 90, 180, 270)
    derivatives_subdir: str = "_upright_derivatives"
    jpeg_quality: int = 92

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["allowed_rotations_cw_deg"] = list(self.allowed_rotations_cw_deg)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UprightDerivativeConfig":
        return cls(
            sidecar_field=str(data["sidecar_field"]),
            allowed_rotations_cw_deg=tuple(int(v) for v in data["allowed_rotations_cw_deg"]),
            derivatives_subdir=str(data["derivatives_subdir"]),
            jpeg_quality=int(data["jpeg_quality"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "UprightDerivativeConfig":
        return cls.from_dict(json.loads(text))


UPRIGHT_DERIVATIVES = UprightDerivativeConfig()


def parse_display_rotation(sidecar_meta: Dict[str, Any], sidecar_path: Path,
                           cfg: UprightDerivativeConfig) -> int:
    """Returns the declared clockwise display rotation (0 when absent); fails loud on invalid values."""
    raw = sidecar_meta.get(cfg.sidecar_field, 0)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw not in cfg.allowed_rotations_cw_deg:
        raise ValueError(
            f"{sidecar_path}: '{cfg.sidecar_field}' must be one of {cfg.allowed_rotations_cw_deg}, got {raw!r}"
        )
    return raw


def write_upright_derivative(source_path: Path, media_dir: Path, rotation_cw_deg: int,
                             cfg: UprightDerivativeConfig) -> Path:
    """Writes (if absent) a clockwise-rotated JPEG copy of `source_path`; the source is never touched."""
    transpose_for_cw = {
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }
    rel_under_media = source_path.relative_to(media_dir)
    dest = (media_dir / cfg.derivatives_subdir / rel_under_media.parent
            / f"{source_path.stem}_cw{rotation_cw_deg}.jpg")
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as img:
        icc_profile = img.info.get("icc_profile")
        rotated = img.convert("RGB").transpose(transpose_for_cw[rotation_cw_deg])
    save_kwargs: Dict[str, Any] = {"quality": cfg.jpeg_quality}
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    rotated.save(dest, "JPEG", **save_kwargs)
    return dest


def prune_stale_upright_derivatives(media_dir: Path, referenced: set[Path],
                                    cfg: UprightDerivativeConfig) -> None:
    """Deletes derivative files no longer referenced by any catalog entry (no dead artifacts)."""
    root = media_dir / cfg.derivatives_subdir
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.resolve() not in referenced:
            path.unlink()


# Default tag vocabularies with canonical names and regex matchers
DEFAULT_VOCABULARIES = {
    "people": [
        {"canonical": "Tamar 'Tami' Sheaffer", "regex": r"\b(tami|tamar|sheaffer|תמי|תמר|זוגתי)\b"},
        {"canonical": "Gad Sheaffer", "regex": r"\b(gad|gadi|גדי|הש\"ג|ש\"ג)\b"},
        {"canonical": "Michael Fried", "regex": r"\b(michael|fried|מיכאל|פריד)\b"},
        {"canonical": "Ahuva Fried", "regex": r"\b(ahuva|אהובה)\b"},
        {"canonical": "Er'ela Sadan", "regex": r"\b(erela|er'ela|אראלה)\b"},
        {"canonical": "Ada Wertheimer", "regex": r"\b(ada|wertheimer|עדה|ורטהיימר)\b"},
        {"canonical": "Guy Sheaffer", "regex": r"\b(guy|גיא|בכורנו)\b"},
        {"canonical": "Tal Sheaffer", "regex": r"\b(tal|טל|טל שפר)\b"},
        {"canonical": "Ya'ara", "regex": r"\b(yaara|ya'ara|יערה)\b"},
        {"canonical": "Carmel", "regex": r"\b(carmel|כרמל)\b"},
        {"canonical": "Beni", "regex": r"\b(beni|benny|בני)\b"},
        {"canonical": "Peri", "regex": r"\b(peri|פרי)\b"}
    ],
    "places": [
        {"canonical": "Herzliya (parents' house & yard)", "regex": r"(?<!מוזיאון )(?<!רחוב )\b(?:herzliya|[במהל]?הרצליה(?! פיתוח)|גן הילדים של אהובה|בית פריד|בית ההורים בהרצליה)\b"},
        {"canonical": "Haifa: Arlozorov", "regex": r"\b(?:arlozorov|[במהל]?ארלוזורוב|הדר הכרמל|בר גיורא)\b"},
        {"canonical": "Haifa: Einstein", "regex": r"(?<!אלברט )(?<!שם )\b(?:einstein|[במהל]?איינשטיין|[במהל]?אינשטיין|אחוזה חיפה)\b"},
        {"canonical": "Haifa: Denya (80 Greenboim St & studio)", "regex": r"\b(?:denya|[במהל]?דניה|[במהל]?גרינבוים|greenboim|g80)\b"},
        {"canonical": "Paris (Cité des Arts)", "regex": r"\b(paris|פריז|cite des arts|cité des arts|סיטה)\b"},
        {"canonical": "Technion", "regex": r"\b(technion|טכניון)\b"},
        {"canonical": "Ramat Gan Museum", "regex": r"\b(ramat gan|מוזיאון רמת גן|בנה ביתך|שרי גולן)\b"},
        {"canonical": "Binyamin Gallery", "regex": r"\b(binyamin|בנימין|גלריה בנימין)\b"},
        {"canonical": "Bogliasco Italy", "regex": r"\b(bogliasco|בוגליאסקו|איטליה|liguria)\b"},
        {"canonical": "Montreal", "regex": r"\b(montreal|מונטריאול|קנדה)\b"},
        {"canonical": "New York", "regex": r"\b(new york|nyc|ניו יורק|מנהטן)\b"},
        {"canonical": "Southern France", "regex": r"\b(southern france|provence|arles|פרובנס|ארל|דרום צרפת)\b"}
    ],
    "art_motifs": [
        {"canonical": "paper", "regex": r"\b(paper|נייר|אריזה)\b"},
        {"canonical": "papier-mâché", "regex": r"\b(papier[- ]m[aâ]ch[eé]|עיסת נייר)\b"},
        {"canonical": "cement", "regex": r"\b(cement|concrete|מלט|בטון|יציקה|יציקות)\b"},
        {"canonical": "earth", "regex": r"\b(earth|hamra|חמרה|אדמה|לבנים כבושות|צופן)\b"},
        {"canonical": "capsules/enclosures", "regex": r"\b(capsule|capsules|enclosure|enclosures|קפסול|מארז)\b"},
        {"canonical": "pipes/conduits", "regex": r"\b(pipe|pipes|conduit|conduits|צינור|צינורות|מוביל)\b"},
        {"canonical": "misshapen furniture", "regex": r"\b(furniture|chair|table|רהיט|רהיטים|כיסא|שולחן|ארון)\b"},
        {"canonical": "broken/blurry reflections", "regex": r"\b(mirror|mirrors|reflection|reflections|מראה|מראות|השתקפ)\b"},
        {"canonical": "simulacra of home", "regex": r"\b(simulacr|home|house|בנה ביתך|סימולקר|עקורים|מחסה)\b"},
        {"canonical": "drawings", "regex": r"\b(drawing|drawings|pencil|pencils|ink|aquarelle|cruising|roaming|רישום|רישומים|עיפרון|עפרונות|דיו|שיטוט)\b"}
    ]
}

SOURCE_GROUP_COLORS = {
    "wordpress_blog": {"bg": "#2563eb", "fg": "#ffffff", "label": "WordPress Blog"},
    "darchei_etzim_pdfs": {"bg": "#dc2626", "fg": "#ffffff", "label": "Darchei Etzim PDF"},
    "portfolio_site": {"bg": "#7c3aed", "fg": "#ffffff", "label": "Portfolio Site"},
    "binyamin_gallery": {"bg": "#d97706", "fg": "#ffffff", "label": "Binyamin Gallery"},
    "museum_press": {"bg": "#059669", "fg": "#ffffff", "label": "Museum & Press"},
    "whatsapp_family": {"bg": "#0d9488", "fg": "#ffffff", "label": "WhatsApp Archive"},
    "drop_folder_imports": {"bg": "#0891b2", "fg": "#ffffff", "label": "Drop Ingestion"},
    "unknown": {"bg": "#64748b", "fg": "#ffffff", "label": "Media"}
}

# Canonical mapping from dedicated media subdirectories to controlled catalog source_group
FOLDER_TO_SOURCE_GROUP: Dict[str, str] = {
    "herzliya_fried_house": "drop_folder_imports",
    "haifa_homes_incoming": "drop_folder_imports",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tiff"}

SCHEMA_FIELDS = [
    "id",
    "filename",
    "rel_path",
    "sha256",
    "width",
    "height",
    "file_size_bytes",
    "mime_type",
    "source_group",
    "source_url_or_path",
    "title",
    "caption",
    "date_or_era",
    "people",
    "places",
    "art_themes",
    "confidence_notes"
]


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash in 64KB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def detect_mime_type(ext: str) -> str:
    ext = ext.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    elif ext == ".gif":
        return "image/gif"
    elif ext == ".tiff":
        return "image/tiff"
    return "application/octet-stream"


def match_tags(text: str, vocab_rules: List[Dict[str, str]]) -> List[str]:
    """Matches text against canonical vocabulary rules using compiled regexes."""
    matched = []
    text_lower = text.lower()
    for rule in vocab_rules:
        pattern = rule["regex"]
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append(rule["canonical"])
    return matched


def extract_year_or_era(filename: str, caption: str) -> str:
    """Heuristic extraction of 4-digit year or decade era."""
    text = f"{filename} {caption}"
    match_year = re.search(r"\b(19\d{2}|20[0-2]\d)\b", text)
    if match_year:
        return match_year.group(1)
    match_era = re.search(r"\b(19\d0s|20\d0s|childhood|technion|midrasha)\b", text, re.IGNORECASE)
    if match_era:
        return match_era.group(1)
    return "unknown"


def scan_media_directory(
    media_dir: Path,
    project_root: Path,
    min_size: int = 150,
    vocabularies: Dict[str, Any] = None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Scans media directory, validates images, deduplicates, and extracts metadata."""
    if vocabularies is None:
        vocabularies = DEFAULT_VOCABULARIES

    if not media_dir.exists():
        if verbose:
            print(f"[WARN] Media directory does not exist: {media_dir}")
        return []

    entries = []
    seen_hashes = {}

    existing_ids_by_sha: Dict[str, str] = {}
    existing_ids_by_rel: Dict[str, str] = {}
    used_ids: set[str] = set()
    max_existing_num = 0
    existing_catalog_path = project_root / "catalog" / "catalog.json"
    if existing_catalog_path.exists():
        try:
            with open(existing_catalog_path, "r", encoding="utf-8") as ef:
                existing_items = json.load(ef)
            if isinstance(existing_items, list):
                for item in existing_items:
                    if not isinstance(item, dict):
                        continue
                    eid = str(item.get("id") or "").strip()
                    if not eid:
                        continue
                    if item.get("sha256"):
                        existing_ids_by_sha[str(item["sha256"])] = eid
                    if item.get("rel_path"):
                        existing_ids_by_rel[str(item["rel_path"])] = eid
                    m_num = re.search(r"_(\d+)$", eid)
                    if m_num:
                        max_existing_num = max(max_existing_num, int(m_num.group(1)))
        except Exception as e:
            if verbose:
                print(f"[WARN] Could not load existing catalog IDs from {existing_catalog_path}: {e}")

    counter = max_existing_num + 1 if max_existing_num > 0 else 1

    # Walk all files in media directory deterministically (never re-ingest our own upright derivatives)
    for root, dirs, files in os.walk(media_dir):
        if Path(root) == media_dir:
            dirs[:] = [d for d in dirs if d != UPRIGHT_DERIVATIVES.derivatives_subdir]
        dirs.sort()
        rel_root = Path(root).relative_to(media_dir)
        raw_folder = rel_root.parts[0] if len(rel_root.parts) > 0 else "unknown"
        default_source_group = FOLDER_TO_SOURCE_GROUP.get(raw_folder, raw_folder)

        for file in sorted(files):
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            # Check if sidecar JSON exists
            sidecar_path = file_path.with_suffix(file_path.suffix + ".json")
            if not sidecar_path.exists():
                sidecar_path = file_path.with_suffix(".json")
            sidecar_meta: Dict[str, Any] = {}
            if sidecar_path.exists():
                with open(sidecar_path, "r", encoding="utf-8") as sf:
                    sidecar_meta = json.load(sf)

            source_group = str(sidecar_meta.get("source_group") or default_source_group)
            id_prefix = raw_folder[:4].upper() if raw_folder != "unknown" else source_group[:4].upper()

            # Validate image with Pillow
            if Image is None:
                raise RuntimeError("Pillow is required for image validation and dimension extraction.")

            try:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    img.load()
                    width, height = img.size
            except Exception as e:
                if verbose:
                    print(f"[ERROR] Corrupt image skipped: {file_path} ({e})")
                continue

            # Enforce minimum dimension filter (reject <150px on either dimension)
            if width < min_size or height < min_size:
                if verbose:
                    print(f"[SKIP] Below minimum size ({width}x{height} < {min_size}): {file_path}")
                continue

            # Calculate SHA-256 and deduplicate
            sha256 = compute_sha256(file_path)
            if sha256 in seen_hashes:
                if verbose:
                    print(f"[SKIP] Duplicate image hash {sha256[:12]} already indexed: {file_path} (original: {seen_hashes[sha256]})")
                continue
            seen_hashes[sha256] = str(file_path)

            file_size_bytes = file_path.stat().st_size
            mime_type = detect_mime_type(ext)
            rel_path = str(file_path.relative_to(project_root))

            # Metadata resolution (sidecar > fallback heuristics)
            title = sidecar_meta.get("title") or file.replace("_", " ").rsplit(".", 1)[0]
            caption = sidecar_meta.get("caption") or sidecar_meta.get("description") or ""
            source_url = sidecar_meta.get("source_url") or sidecar_meta.get("source_url_or_path") or f"local://{rel_path}"
            date_or_era = sidecar_meta.get("date_or_era") or extract_year_or_era(file, caption)

            # Tag matching
            combined_text = f"{title} {caption} {file} {source_group}"
            people_tags = list(sidecar_meta.get("people") or [])
            if not people_tags and not sidecar_meta.get("people_locked"):
                people_tags = match_tags(combined_text, vocabularies.get("people", []))

            places_tags = list(sidecar_meta.get("places") or [])
            if not places_tags and not sidecar_meta.get("places_locked"):
                places_tags = match_tags(combined_text, vocabularies.get("places", []))

            art_tags = list(sidecar_meta.get("art_themes") or [])
            if not art_tags and not sidecar_meta.get("art_themes_locked"):
                art_tags = match_tags(combined_text, vocabularies.get("art_motifs", []))

            # Default art theme or people tag if source hints
            if source_group in ("portfolio_site", "binyamin_gallery", "museum_press"):
                if "Tamar 'Tami' Sheaffer" not in people_tags:
                    people_tags.append("Tamar 'Tami' Sheaffer")
                if not art_tags:
                    art_tags.append("capsules/enclosures")

            confidence_notes = sidecar_meta.get("confidence_notes") or f"Verified: Non-corrupt {width}x{height} {mime_type} from {source_group}"

            preserved_id = (
                str(sidecar_meta.get("id") or "").strip()
                or existing_ids_by_sha.get(sha256, "")
                or existing_ids_by_rel.get(rel_path, "")
            )
            if preserved_id and preserved_id not in used_ids:
                item_id = preserved_id
            else:
                item_id = f"IMG_{id_prefix}_{counter:04d}"
                counter += 1
                while item_id in used_ids:
                    item_id = f"IMG_{id_prefix}_{counter:04d}"
                    counter += 1
            used_ids.add(item_id)

            display_rotation_cw_deg = parse_display_rotation(sidecar_meta, sidecar_path, UPRIGHT_DERIVATIVES)
            if display_rotation_cw_deg:
                derivative = write_upright_derivative(file_path, media_dir, display_rotation_cw_deg, UPRIGHT_DERIVATIVES)
                display_rel_path = str(derivative.relative_to(project_root))
            else:
                display_rel_path = rel_path

            entry = {
                "id": item_id,
                "filename": file,
                "rel_path": rel_path,
                "display_rel_path": display_rel_path,
                "display_rotation_cw_deg": display_rotation_cw_deg,
                "sha256": sha256,
                "width": width,
                "height": height,
                "file_size": file_size_bytes,
                "file_size_bytes": file_size_bytes,
                "mime_type": mime_type,
                "source_group": source_group,
                "source_url_or_path": source_url,
                "title": title,
                "caption": caption,
                "date_or_era": date_or_era,
                "people": sorted(list(set(people_tags))),
                "places": sorted(list(set(places_tags))),
                "art_themes": sorted(list(set(art_tags))),
                "confidence_notes": confidence_notes
            }
            entries.append(entry)

    return entries


def write_catalog_json(entries: List[Dict[str, Any]], output_path: Path):
    """Writes normalized JSON catalog."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def write_catalog_csv(entries: List[Dict[str, Any]], output_path: Path):
    """Writes UTF-8 with BOM CSV catalog for seamless Excel/Numbers Hebrew rendering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SCHEMA_FIELDS)
        for e in entries:
            row = [
                e["id"],
                e["filename"],
                e["rel_path"],
                e["sha256"],
                e["width"],
                e["height"],
                e["file_size_bytes"],
                e["mime_type"],
                e["source_group"],
                e["source_url_or_path"],
                e["title"],
                e["caption"],
                e["date_or_era"],
                " | ".join(e["people"]),
                " | ".join(e["places"]),
                " | ".join(e["art_themes"]),
                e["confidence_notes"]
            ]
            writer.writerow(row)


def write_catalog_sqlite(entries: List[Dict[str, Any]], output_path: Path):
    """Creates SQLite database with clean table definition, unique/query indexes, and media view."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(output_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE catalog (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            rel_path TEXT NOT NULL,
            sha256 TEXT NOT NULL UNIQUE,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            file_size INTEGER NOT NULL,
            mime_type TEXT NOT NULL,
            source_group TEXT NOT NULL,
            source_url_or_path TEXT,
            title TEXT,
            caption TEXT,
            date_or_era TEXT,
            people TEXT,
            places TEXT,
            art_themes TEXT,
            confidence_notes TEXT
        );
    """)

    cur.execute("CREATE UNIQUE INDEX idx_catalog_sha256 ON catalog(sha256);")
    cur.execute("CREATE INDEX idx_catalog_source_group ON catalog(source_group);")
    cur.execute("CREATE INDEX idx_catalog_date_or_era ON catalog(date_or_era);")
    cur.execute("CREATE VIEW media AS SELECT * FROM catalog;")

    insert_sql = """
        INSERT INTO catalog (
            id, filename, rel_path, sha256, width, height, file_size_bytes, file_size,
            mime_type, source_group, source_url_or_path, title, caption,
            date_or_era, people, places, art_themes, confidence_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    for e in entries:
        cur.execute(insert_sql, (
            e["id"],
            e["filename"],
            e["rel_path"],
            e["sha256"],
            e["width"],
            e["height"],
            e["file_size_bytes"],
            e.get("file_size", e["file_size_bytes"]),
            e["mime_type"],
            e["source_group"],
            e["source_url_or_path"],
            e["title"],
            e["caption"],
            e["date_or_era"],
            json.dumps(e["people"], ensure_ascii=False),
            json.dumps(e["places"], ensure_ascii=False),
            json.dumps(e["art_themes"], ensure_ascii=False),
            e["confidence_notes"]
        ))

    conn.commit()
    conn.close()


def generate_contact_sheet_html(entries: List[Dict[str, Any]], output_path: Path):
    """Generates a responsive, standalone HTML contact sheet with zero external dependencies."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    total_images = len(entries)
    source_counts = {}
    all_people = set()
    all_places = set()
    all_themes = set()

    for e in entries:
        grp = e["source_group"]
        source_counts[grp] = source_counts.get(grp, 0) + 1
        all_people.update(e["people"])
        all_places.update(e["places"])
        all_themes.update(e["art_themes"])

    catalog_json_str = json.dumps(entries, ensure_ascii=False).replace("<", "\\u003c")

    html_content = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ארכיון וקטלוג מדיה — תמר 'תמי' שפר (יום הולדת 70)</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.2);
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text-main);
      padding: 24px;
      line-height: 1.5;
    }}
    header {{
      max-width: 1400px;
      margin: 0 auto 32px auto;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 24px;
    }}
    .header-title-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    h1 {{
      font-size: 1.85rem;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: -0.5px;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 0.95rem;
      margin-top: 6px;
    }}
    .stats-bar {{
      display: flex;
      gap: 12px;
      margin-top: 18px;
      flex-wrap: wrap;
    }}
    .stat-pill {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.85rem;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .stat-pill strong {{
      color: var(--accent);
    }}
    .toolbar {{
      max-width: 1400px;
      margin: 0 auto 28px auto;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .search-row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .search-input {{
      flex: 1;
      min-width: 260px;
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 10px 16px;
      border-radius: 8px;
      font-size: 0.95rem;
      outline: none;
      direction: auto;
    }}
    .search-input:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-glow);
    }}
    .filter-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .filter-label {{
      font-size: 0.85rem;
      color: var(--text-muted);
      font-weight: 600;
      min-width: 80px;
    }}
    .filter-btn {{
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .filter-btn:hover {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    .filter-btn.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #0f172a;
      font-weight: 600;
    }}
    .gallery-grid {{
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
      cursor: pointer;
    }}
    .card:hover {{
      transform: translateY(-4px);
      border-color: var(--accent);
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 0 1px var(--accent-glow);
    }}
    .thumb-wrap {{
      width: 100%;
      height: 220px;
      background: #090d16;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      overflow: hidden;
    }}
    .thumb-wrap img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      transition: transform 0.3s ease;
    }}
    .card:hover .thumb-wrap img {{
      transform: scale(1.03);
    }}
    .badge-source {{
      position: absolute;
      top: 10px;
      right: 10px;
      font-size: 0.7rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }}
    .badge-dim {{
      position: absolute;
      bottom: 8px;
      left: 8px;
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.1);
      font-size: 0.7rem;
      padding: 2px 6px;
      border-radius: 4px;
      color: #e2e8f0;
      font-family: monospace;
    }}
    .card-body {{
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      flex: 1;
    }}
    .card-title {{
      font-size: 1rem;
      font-weight: 600;
      color: #fff;
      line-height: 1.35;
      direction: auto;
    }}
    .card-caption {{
      font-size: 0.8rem;
      color: var(--text-muted);
      line-height: 1.4;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      direction: auto;
    }}
    .card-meta {{
      margin-top: auto;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding-top: 10px;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .tag {{
      background: rgba(56, 189, 248, 0.1);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.2);
      font-size: 0.72rem;
      padding: 2px 8px;
      border-radius: 4px;
    }}
    .tag-place {{
      background: rgba(168, 85, 247, 0.1);
      color: #c084fc;
      border-color: rgba(168, 85, 247, 0.2);
    }}
    .tag-theme {{
      background: rgba(245, 158, 11, 0.1);
      color: #fbbf24;
      border-color: rgba(245, 158, 11, 0.2);
    }}
    .modal-backdrop {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.85);
      backdrop-filter: blur(8px);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }}
    .modal-backdrop.open {{
      display: flex;
    }}
    .modal-box {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      max-width: 1050px;
      width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
      position: relative;
    }}
    .modal-close {{
      position: absolute;
      top: 16px;
      left: 16px;
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: #fff;
      font-size: 1.2rem;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10;
    }}
    .modal-content-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      padding: 24px;
    }}
    @media (max-width: 768px) {{
      .modal-content-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    .modal-img-wrap {{
      background: #090d16;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 320px;
    }}
    .modal-img-wrap img {{
      max-width: 100%;
      max-height: 60vh;
      object-fit: contain;
    }}
    .modal-details {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .meta-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}
    .meta-table th, .meta-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      text-align: right;
    }}
    .meta-table th {{
      color: var(--text-muted);
      width: 30%;
      font-weight: 600;
    }}
    .meta-table td {{
      color: #fff;
      font-family: monospace;
      word-break: break-all;
    }}
    .modal-nav-btns {{
      display: flex;
      justify-content: space-between;
      padding: 16px 24px;
      border-top: 1px solid var(--card-border);
      background: #0f172a;
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-title-row">
      <div>
        <h1>תמר 'תמי' שפר — קטלוג יצירות וצילומי חיים</h1>
        <div class="subtitle">קטלוג מדיה היקפי לקראת יום הולדת 70 • כ"ה בתשרי תשפ"ז / 25 באוקטובר 2026</div>
      </div>
    </div>
    <div class="stats-bar">
      <div class="stat-pill">סה"כ פריטים: <strong id="stat-total">{total_images}</strong></div>
      <div class="stat-pill">מוצגים כעת: <strong id="stat-visible">{total_images}</strong></div>
      <div class="stat-pill">מקורות פעילים: <strong>{len(source_counts)}</strong></div>
      <div class="stat-pill">תגיות אנשים: <strong>{len(all_people)}</strong></div>
      <div class="stat-pill">מקומות: <strong>{len(all_places)}</strong></div>
      <div class="stat-pill">מוטיבים אמנותיים: <strong>{len(all_themes)}</strong></div>
    </div>
  </header>

  <div class="toolbar">
    <div class="search-row">
      <input type="text" id="searchInput" class="search-input" placeholder="חיפוש חופשי (כותרת, תיאור, שם קובץ, תגית, שנה)...">
    </div>

    <div class="filter-group">
      <span class="filter-label">מקור:</span>
      <button class="filter-btn active" data-filter-type="source" data-value="all">הכל</button>
      {"".join([f'<button class="filter-btn" data-filter-type="source" data-value="{grp}">{grp} ({cnt})</button>' for grp, cnt in sorted(source_counts.items())])}
    </div>

    <div class="filter-group">
      <span class="filter-label">מוטיב:</span>
      <button class="filter-btn active" data-filter-type="theme" data-value="all">הכל</button>
      {"".join([f'<button class="filter-btn" data-filter-type="theme" data-value="{t}">{t}</button>' for t in sorted(list(all_themes))[:8]])}
    </div>
  </div>

  <main class="gallery-grid" id="galleryGrid">
"""

    for idx, e in enumerate(entries):
        card_img_url = f"../{e['display_rel_path']}"
        grp = e["source_group"]
        grp_style = SOURCE_GROUP_COLORS.get(grp, SOURCE_GROUP_COLORS["unknown"])

        people_badges = "".join([f'<span class="tag">{p}</span>' for p in e["people"]])
        place_badges = "".join([f'<span class="tag tag-place">{pl}</span>' for pl in e["places"]])
        theme_badges = "".join([f'<span class="tag tag-theme">{th}</span>' for th in e["art_themes"]])

        html_content += f"""
    <div class="card" data-idx="{idx}" data-source="{grp}" data-themes="{','.join(e['art_themes'])}" data-people="{','.join(e['people'])}" data-places="{','.join(e['places'])}">
      <div class="thumb-wrap">
        <span class="badge-source" style="background-color: {grp_style['bg']}; color: {grp_style['fg']};">{grp}</span>
        <img src="{card_img_url}" alt="{e['title']}" loading="lazy">
        <span class="badge-dim">{e['width']}x{e['height']} ({e['width']}x{e['height']}px)</span>
      </div>
      <div class="card-body">
        <div class="card-title">{e['title']}</div>
        <div class="card-caption">{e['caption'] or e['filename']}</div>
        <div class="card-meta">
          {people_badges}
          {place_badges}
          {theme_badges}
        </div>
      </div>
    </div>
"""

    html_content += f"""
  </main>

  <!-- Modal Lightbox -->
  <div class="modal-backdrop" id="modalBackdrop">
    <div class="modal-box">
      <button class="modal-close" id="modalClose">&times;</button>
      <div class="modal-content-grid">
        <div class="modal-img-wrap">
          <img id="modalImg" src="" alt="">
        </div>
        <div class="modal-details">
          <h2 id="modalTitle" style="color: #fff; font-size: 1.4rem;"></h2>
          <p id="modalCaption" style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.5;"></p>
          <table class="meta-table">
            <tr><th>מזהה ID</th><td id="metaId"></td></tr>
            <tr><th>קבוצת מקור</th><td id="metaSource"></td></tr>
            <tr><th>מימדים</th><td id="metaDim"></td></tr>
            <tr><th>גודל קובץ</th><td id="metaSize"></td></tr>
            <tr><th>סוג קובץ</th><td id="metaMime"></td></tr>
            <tr><th>תאריך / עידן</th><td id="metaEra"></td></tr>
            <tr><th>נתיב יחסי</th><td id="metaRelPath"></td></tr>
            <tr><th>טביעת SHA-256</th><td id="metaSha256" style="font-size: 0.72rem;"></td></tr>
            <tr><th>אנשים מתויגים</th><td id="metaPeople"></td></tr>
            <tr><th>מקומות</th><td id="metaPlaces"></td></tr>
            <tr><th>מוטיבים אמנותיים</th><td id="metaThemes"></td></tr>
            <tr><th>הערות אימות</th><td id="metaNotes"></td></tr>
          </table>
          <div style="margin-top: 10px;">
            <a id="modalOrigLink" href="#" target="_blank" style="color: var(--accent); font-size: 0.85rem; text-decoration: underline;">צפה בקובץ המקורי בגודל מלא &larr;</a>
          </div>
        </div>
      </div>
      <div class="modal-nav-btns">
        <button class="filter-btn" id="modalPrev">&larr; הקודם</button>
        <button class="filter-btn" id="modalNext">הבא &rarr;</button>
      </div>
    </div>
  </div>

  <script>
    const CATALOG_DATA = {catalog_json_str};
    let currentIdx = 0;
    let activeFilters = {{
      source: 'all',
      theme: 'all',
      search: ''
    }};

    const galleryGrid = document.getElementById('galleryGrid');
    const cards = Array.from(document.querySelectorAll('.card'));
    const statVisible = document.getElementById('stat-visible');
    const searchInput = document.getElementById('searchInput');

    function applyFilters() {{
      let visibleCount = 0;
      cards.forEach(card => {{
        const idx = parseInt(card.getAttribute('data-idx'));
        const item = CATALOG_DATA[idx];
        const cardSource = card.getAttribute('data-source');
        const cardThemes = card.getAttribute('data-themes').split(',');

        let matchesSource = (activeFilters.source === 'all' || cardSource === activeFilters.source);
        let matchesTheme = (activeFilters.theme === 'all' || cardThemes.includes(activeFilters.theme));
        let matchesSearch = true;

        if (activeFilters.search) {{
          const q = activeFilters.search.toLowerCase();
          const searchable = (
            item.title + ' ' +
            item.caption + ' ' +
            item.filename + ' ' +
            item.date_or_era + ' ' +
            item.people.join(' ') + ' ' +
            item.places.join(' ') + ' ' +
            item.art_themes.join(' ')
          ).toLowerCase();
          matchesSearch = searchable.includes(q);
        }}

        if (matchesSource && matchesTheme && matchesSearch) {{
          card.style.display = 'flex';
          visibleCount++;
        }} else {{
          card.style.display = 'none';
        }}
      }});
      statVisible.textContent = visibleCount;
    }}

    searchInput.addEventListener('input', (e) => {{
      activeFilters.search = e.target.value.trim();
      applyFilters();
    }});

    document.querySelectorAll('.filter-btn[data-filter-type]').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const ftype = btn.getAttribute('data-filter-type');
        const val = btn.getAttribute('data-value');
        document.querySelectorAll(`.filter-btn[data-filter-type="${{ftype}}"]`).forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilters[ftype] = val;
        applyFilters();
      }});
    }});

    const modal = document.getElementById('modalBackdrop');
    const modalClose = document.getElementById('modalClose');
    const modalImg = document.getElementById('modalImg');
    const modalTitle = document.getElementById('modalTitle');
    const modalCaption = document.getElementById('modalCaption');
    const modalOrigLink = document.getElementById('modalOrigLink');
    const modalPrev = document.getElementById('modalPrev');
    const modalNext = document.getElementById('modalNext');

    function openModal(idx) {{
      currentIdx = idx;
      const item = CATALOG_DATA[idx];
      const imgPath = '../' + item.rel_path;

      modalImg.src = '../' + item.display_rel_path;
      modalTitle.textContent = item.title;
      modalCaption.textContent = item.caption || item.filename;
      modalOrigLink.href = imgPath;

      document.getElementById('metaId').textContent = item.id;
      document.getElementById('metaSource').textContent = item.source_group;
      document.getElementById('metaDim').textContent = `${{item.width}} x ${{item.height}} px`;
      document.getElementById('metaSize').textContent = `${{(item.file_size_bytes / 1024).toFixed(1)}} KB`;
      document.getElementById('metaMime').textContent = item.mime_type;
      document.getElementById('metaEra').textContent = item.date_or_era;
      document.getElementById('metaRelPath').textContent = item.rel_path;
      document.getElementById('metaSha256').textContent = item.sha256;
      document.getElementById('metaPeople').textContent = item.people.join(', ') || '—';
      document.getElementById('metaPlaces').textContent = item.places.join(', ') || '—';
      document.getElementById('metaThemes').textContent = item.art_themes.join(', ') || '—';
      document.getElementById('metaNotes').textContent = item.confidence_notes || '—';

      modal.classList.add('open');
    }}

    cards.forEach(card => {{
      card.addEventListener('click', () => {{
        openModal(parseInt(card.getAttribute('data-idx')));
      }});
    }});

    modalClose.addEventListener('click', () => modal.classList.remove('open'));
    modal.addEventListener('click', (e) => {{
      if (e.target === modal) modal.classList.remove('open');
    }});

    modalPrev.addEventListener('click', () => {{
      if (currentIdx > 0) openModal(currentIdx - 1);
    }});
    modalNext.addEventListener('click', () => {{
      if (currentIdx < CATALOG_DATA.length - 1) openModal(currentIdx + 1);
    }});

    window.addEventListener('keydown', (e) => {{
      if (!modal.classList.contains('open')) return;
      if (e.key === 'Escape') modal.classList.remove('open');
      if (e.key === 'ArrowRight' && currentIdx > 0) openModal(currentIdx - 1);
      if (e.key === 'ArrowLeft' && currentIdx < CATALOG_DATA.length - 1) openModal(currentIdx + 1);
    }});
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def build_all_catalogs(
    project_root: Path,
    media_dir: Path,
    catalog_dir: Path,
    min_size: int = 150,
    vocab_path: Optional[Path] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Runs end-to-end catalog builder across all 4 formats."""
    vocabularies = DEFAULT_VOCABULARIES
    if vocab_path and vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as vf:
            vocabularies = json.load(vf)

    if verbose:
        print(f"Scanning media files in: {media_dir} ...")

    entries = scan_media_directory(
        media_dir=media_dir,
        project_root=project_root,
        min_size=min_size,
        vocabularies=vocabularies,
        verbose=verbose
    )

    if verbose:
        print(f"Total valid media assets discovered: {len(entries)}")

    prune_stale_upright_derivatives(
        media_dir,
        {(project_root / e["display_rel_path"]).resolve() for e in entries},
        UPRIGHT_DERIVATIVES,
    )

    json_path = catalog_dir / "catalog.json"
    csv_path = catalog_dir / "catalog.csv"
    sqlite_path = catalog_dir / "catalog.sqlite"
    html_path = catalog_dir / "index.html"

    write_catalog_json(entries, json_path)
    write_catalog_csv(entries, csv_path)
    write_catalog_sqlite(entries, sqlite_path)
    generate_contact_sheet_html(entries, html_path)

    stats = {
        "count": len(entries),
        "json": str(json_path),
        "csv": str(csv_path),
        "sqlite": str(sqlite_path),
        "html": str(html_path)
    }

    if verbose:
        print(f"Successfully generated all 4 catalog formats in: {catalog_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Master Media Catalog Builder")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--media-dir", default="media", help="Media directory (relative or absolute)")
    parser.add_argument("--catalog-dir", default="catalog", help="Catalog output directory")
    parser.add_argument("--min-size", type=int, default=150, help="Minimum dimension threshold in pixels")
    parser.add_argument("--vocab", help="Path to custom tag_vocabularies.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose console output")

    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    media_dir = (project_root / args.media_dir).resolve() if not Path(args.media_dir).is_absolute() else Path(args.media_dir)
    catalog_dir = (project_root / args.catalog_dir).resolve() if not Path(args.catalog_dir).is_absolute() else Path(args.catalog_dir)
    vocab_path = Path(args.vocab).resolve() if args.vocab else None

    stats = build_all_catalogs(
        project_root=project_root,
        media_dir=media_dir,
        catalog_dir=catalog_dir,
        min_size=args.min_size,
        vocab_path=vocab_path,
        verbose=args.verbose
    )

    print(f"Catalog build complete: {stats['count']} items indexed across 4 formats.")


if __name__ == "__main__":
    main()
