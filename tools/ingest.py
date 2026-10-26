#!/usr/bin/env python3
"""
tools/ingest.py — Drop-Folder & CLI Ingestion Engine for Tamar Sheaffer Archive

Requirement R2 & Interface Contract § 4.2:
- Standalone executable CLI tool (chmod +x tools/ingest.py).
- Discovers candidate images in drop folder or specified CLI paths.
- Validates non-corrupt raster stream and decodes header using Pillow.
- Enforces dimension threshold (rejects <150px on either axis).
- Computes SHA-256 hash and rejects duplicates against catalog.
- Supports --dry-run without modifying catalog or moving files.
- Live mode: copies image to media/drop_folder_imports/<filename>,
  generates 17-field catalog record, appends to catalog.json, catalog.csv,
  catalog.sqlite, and updates catalog/index.html.
- Returns exit code 0 on success.
"""

import sys
import os
import re
import shutil
import hashlib
import json
import csv
import sqlite3
import argparse
import html
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    print("ERROR: Pillow (PIL) is required. Install via pip install pillow.", file=sys.stderr)
    sys.exit(1)


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.tif', '.tiff', '.bmp'}

SCHEMA_FIELDS = [
    'id', 'filename', 'rel_path', 'sha256', 'width', 'height',
    'file_size_bytes', 'mime_type', 'source_group', 'source_url_or_path',
    'title', 'caption', 'date_or_era', 'people', 'places',
    'art_themes', 'confidence_notes'
]

MIME_TYPES = {
    'JPEG': 'image/jpeg',
    'JPG': 'image/jpeg',
    'PNG': 'image/png',
    'WEBP': 'image/webp',
    'GIF': 'image/gif',
    'TIFF': 'image/tiff',
    'BMP': 'image/bmp'
}


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hexadecimal digest of file content in 64KB chunks."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest().lower()


def inspect_image(path: Path, min_size: int = 150) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates image header, checks non-corruption by decoding raster via Pillow,
    enforces minimum dimension threshold, and computes SHA-256.
    """
    if path.name.startswith('.') or path.name.startswith('._'):
        return False, "Hidden or system file ignored", None
    if not path.is_file():
        return False, "Not a regular file", None

    try:
        size_bytes = path.stat().st_size
    except OSError as e:
        return False, f"Cannot stat file: {e}", None

    if size_bytes == 0:
        return False, "Empty file (0 bytes)", None

    try:
        with Image.open(path) as img:
            fmt = img.format or "UNKNOWN"
            width, height = img.size
            # Calling load() forces decoding of raster pixels, catching truncated bodies
            img.load()
    except UnidentifiedImageError:
        return False, "Corrupted or unrecognized image format", None
    except Exception as e:
        return False, f"Corrupted image data: {e}", None

    if width < min_size or height < min_size:
        return False, f"Dimensions ({width}x{height}) below minimum threshold ({min_size}px)", None

    sha256_hash = compute_sha256(path)
    mime = MIME_TYPES.get(fmt.upper(), f"image/{fmt.lower()}")

    return True, "OK", {
        'width': width,
        'height': height,
        'format': fmt,
        'mime_type': mime,
        'file_size': size_bytes,
        'file_size_bytes': size_bytes,
        'sha256': sha256_hash
    }


def parse_sidecar_metadata(image_path: Path) -> Dict[str, Any]:
    """
    Inspects image directory for companion JSON sidecar metadata
    (e.g. photo.jpg.json or photo.json).
    """
    candidates = [
        image_path.with_name(image_path.name + '.json'),
        image_path.with_suffix('.json')
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            try:
                data = json.loads(c.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    title = data.get('title')
                    caption = data.get('description') or data.get('caption')
                    date_val = data.get('date_or_era') or data.get('date')
                    if not date_val and isinstance(data.get('photoTakenTime'), dict):
                        date_val = data['photoTakenTime'].get('formatted')
                    people = data.get('people')
                    if isinstance(people, str):
                        people = [p.strip() for p in people.split(',') if p.strip()]
                    places = data.get('places')
                    if isinstance(places, str):
                        places = [p.strip() for p in places.split(',') if p.strip()]
                    themes = data.get('art_themes') or data.get('themes')
                    if isinstance(themes, str):
                        themes = [t.strip() for t in themes.split(',') if t.strip()]

                    return {
                        'title': title,
                        'caption': caption,
                        'date_or_era': date_val,
                        'people': people or [],
                        'places': places or [],
                        'art_themes': themes or [],
                        'confidence_notes': data.get('confidence_notes')
                    }
            except Exception:
                pass
    return {}


def load_catalog(json_path: Path, sqlite_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads existing catalog records and builds SHA-256 lookup index.
    Prioritizes JSON, with fallback to SQLite database.
    """
    records: List[Dict[str, Any]] = []
    sha256_map: Dict[str, Dict[str, Any]] = {}

    if json_path.exists():
        try:
            content = json_path.read_text(encoding='utf-8')
            if content.strip():
                data = json.loads(content)
                if isinstance(data, list):
                    records = data
                    for r in records:
                        if 'sha256' in r:
                            sha256_map[r['sha256']] = r
                    return sha256_map, records
        except Exception as e:
            print(f"[WARN] Failed reading {json_path}: {e}", file=sys.stderr)

    if sqlite_path.exists():
        try:
            conn = sqlite3.connect(sqlite_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            table_name = 'catalog'
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('catalog', 'media')")
            found = [row[0] for row in cur.fetchall()]
            if 'catalog' in found:
                table_name = 'catalog'
            elif 'media' in found:
                table_name = 'media'

            cur.execute(f'SELECT * FROM {table_name}')
            for row in cur.fetchall():
                d = dict(row)
                for list_field in ['people', 'places', 'art_themes']:
                    if isinstance(d.get(list_field), str):
                        try:
                            d[list_field] = json.loads(d[list_field])
                        except Exception:
                            d[list_field] = [x.strip() for x in d[list_field].split('|') if x.strip()]
                if 'file_size' not in d and 'file_size_bytes' in d:
                    d['file_size'] = d['file_size_bytes']
                if 'file_size_bytes' not in d and 'file_size' in d:
                    d['file_size_bytes'] = d['file_size']
                records.append(d)
                if 'sha256' in d:
                    sha256_map[d['sha256']] = d
            conn.close()
        except Exception as e:
            print(f"[WARN] Failed reading {sqlite_path}: {e}", file=sys.stderr)

    return sha256_map, records


def generate_html_catalog(records: List[Dict[str, Any]], out_path: Path, project_root: Path) -> None:
    """Generates a self-contained, responsive HTML contact sheet with live search and filtering."""
    items_html = []
    for r in records:
        rel_img = r.get('rel_path', '')
        try:
            abs_img = (project_root / rel_img).resolve()
            rel_to_html = os.path.relpath(abs_img, out_path.parent)
        except Exception:
            rel_to_html = '../' + rel_img

        title = html.escape(r.get('title') or r.get('filename') or 'Untitled')
        caption = html.escape(r.get('caption') or '')
        dim = f"{r.get('width', 0)}x{r.get('height', 0)}"
        size_bytes = r.get('file_size_bytes') or r.get('file_size') or 0
        size_kb = f"{size_bytes // 1024} KB"
        src_grp = html.escape(r.get('source_group', 'unknown'))
        date_era = html.escape(r.get('date_or_era') or 'Undated')

        people_list = r.get('people') or []
        places_list = r.get('places') or []
        themes_list = r.get('art_themes') or []

        people_pills = ''.join(f'<span class="tag person">{html.escape(p)}</span>' for p in people_list)
        places_pills = ''.join(f'<span class="tag place">{html.escape(p)}</span>' for p in places_list)
        themes_pills = ''.join(f'<span class="tag theme">{html.escape(t)}</span>' for t in themes_list)

        search_corpus = f"{title} {caption} {date_era} {' '.join(people_list)} {' '.join(places_list)} {' '.join(themes_list)}"

        items_html.append(f"""
        <div class="media-card" data-group="{src_grp}" data-text="{html.escape(search_corpus.lower())}">
          <div class="img-wrap">
            <img src="{rel_to_html}" alt="{title}" loading="lazy" />
          </div>
          <div class="card-body">
            <div class="badges">
              <span class="badge group">{src_grp}</span>
              <span class="badge dim">{dim}</span>
              <span class="badge size">{size_kb}</span>
              <span class="badge date">{date_era}</span>
            </div>
            <h3 class="title">{title}</h3>
            {f'<p class="caption">{caption}</p>' if caption else ''}
            <div class="tags-wrap">
              {people_pills}
              {places_pills}
              {themes_pills}
            </div>
          </div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ארכיון מדיה וקטלוג יצירות — תמר (תמי) שפר | Tamar Sheaffer Media Archive</title>
  <style>
    :root {{
      --bg: #f8f7f4; --card-bg: #ffffff; --text: #1f2421;
      --text-muted: #5c6761; --primary: #8b3a2b; --border: #e3ded6;
      --badge-bg: #ece7e0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: var(--bg); color: var(--text); margin: 0; padding: 24px; direction: rtl;
    }}
    .header {{ max-width: 1200px; margin: 0 auto 24px; text-align: center; }}
    .header h1 {{ margin: 0 0 8px; font-size: 26px; color: var(--primary); }}
    .header p {{ margin: 0 0 16px; color: var(--text-muted); font-size: 14px; }}
    .stats-bar {{ display: flex; gap: 16px; justify-content: center; margin-bottom: 20px; font-weight: 600; font-size: 14px; }}
    .search-bar {{ max-width: 600px; margin: 0 auto 24px; }}
    .search-bar input {{
      width: 100%; box-sizing: border-box; padding: 12px 18px; border-radius: 24px;
      border: 1px solid var(--border); font-size: 15px; background: var(--card-bg);
    }}
    .gallery-grid {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 20px; max-width: 1200px; margin: 0 auto;
    }}
    .media-card {{
      background: var(--card-bg); border: 1px solid var(--border);
      border-radius: 8px; overflow: hidden; display: flex; flex-direction: column;
      box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }}
    .img-wrap {{
      width: 100%; height: 220px; background: #eee;
      display: flex; align-items: center; justify-content: center; overflow: hidden;
    }}
    .img-wrap img {{ width: 100%; height: 100%; object-fit: cover; }}
    .card-body {{ padding: 14px; display: flex; flex-direction: column; gap: 8px; flex: 1; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; font-size: 11px; }}
    .badge {{ background: var(--badge-bg); padding: 2px 8px; border-radius: 4px; }}
    .badge.group {{ background: #e8eaf6; color: #283593; font-weight: bold; }}
    .title {{ margin: 0; font-size: 15px; font-weight: 600; }}
    .caption {{ margin: 0; font-size: 12px; color: var(--text-muted); line-height: 1.4; }}
    .tags-wrap {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: auto; padding-top: 8px; }}
    .tag {{ font-size: 11px; padding: 2px 6px; border-radius: 12px; }}
    .tag.person {{ background: #e0f2f1; color: #004d40; }}
    .tag.place {{ background: #fff3e0; color: #e65100; }}
    .tag.theme {{ background: #fce4ec; color: #880e4f; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>תמר (תמי) שפר — קטלוג מדיה וארכיון צילומים</h1>
    <p>Tamar "Tami" Sheaffer 70th-Birthday Media Archive & Creative Catalog</p>
    <div class="stats-bar">
      <span>סה״כ פריטים בקטלוג: {len(records)}</span>
    </div>
    <div class="search-bar">
      <input type="text" id="filterInput" placeholder="חיפוש חופשי לפי שם יצירה, מקום, אדם, תמה..." oninput="filterCards()">
    </div>
  </div>
  <div class="gallery-grid" id="mediaGrid">
    {''.join(items_html)}
  </div>
  <script>
    function filterCards() {{
      const q = document.getElementById('filterInput').value.toLowerCase().trim();
      const cards = document.querySelectorAll('.media-card');
      cards.forEach(c => {{
        const text = (c.getAttribute('data-text') || '').toLowerCase();
        c.style.display = (!q || text.includes(q)) ? 'flex' : 'none';
      }});
    }}
  </script>
</body>
</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding='utf-8')


def save_catalogs(records: List[Dict[str, Any]],
                  json_path: Path,
                  csv_path: Path,
                  sqlite_path: Path,
                  html_path: Path,
                  project_root: Path) -> None:
    """Synchronizes records across JSON, CSV, SQLite database, and HTML contact sheet."""
    # Ensure both file_size and file_size_bytes are present in records
    for r in records:
        size = r.get('file_size_bytes') or r.get('file_size') or 0
        r['file_size_bytes'] = size
        r['file_size'] = size

    # 1. Atomic write JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = json_path.with_name(f"{json_path.name}.tmp.{os.getpid()}")
    with open(tmp_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp_json.replace(json_path)

    # 2. Atomic write CSV with UTF-8 BOM
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = csv_path.with_name(f"{csv_path.name}.tmp.{os.getpid()}")
    with open(tmp_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            row = dict(r)
            for list_field in ['people', 'places', 'art_themes']:
                v = row.get(list_field)
                if isinstance(v, list):
                    row[list_field] = ' | '.join(v)
            writer.writerow(row)
    tmp_csv.replace(csv_path)

    # 3. SQLite Database with table `catalog` and VIEW `media`
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS catalog (
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
        confidence_notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_catalog_sha256 ON catalog(sha256)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_catalog_source_group ON catalog(source_group)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_catalog_date_or_era ON catalog(date_or_era)')

    # Provide view `media` for compatibility with queries selecting from media
    cur.execute('DROP VIEW IF EXISTS media')
    cur.execute('CREATE VIEW media AS SELECT * FROM catalog')

    for r in records:
        cur.execute("""
        INSERT OR REPLACE INTO catalog (
            id, filename, rel_path, sha256, width, height, file_size_bytes, file_size, mime_type,
            source_group, source_url_or_path, title, caption, date_or_era,
            people, places, art_themes, confidence_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r.get('id'),
            r.get('filename'),
            r.get('rel_path'),
            r.get('sha256'),
            r.get('width'),
            r.get('height'),
            r.get('file_size_bytes', r.get('file_size', 0)),
            r.get('file_size', r.get('file_size_bytes', 0)),
            r.get('mime_type'),
            r.get('source_group'),
            r.get('source_url_or_path'),
            r.get('title'),
            r.get('caption'),
            r.get('date_or_era'),
            json.dumps(r.get('people') or [], ensure_ascii=False),
            json.dumps(r.get('places') or [], ensure_ascii=False),
            json.dumps(r.get('art_themes') or [], ensure_ascii=False),
            r.get('confidence_notes')
        ))
    conn.commit()
    conn.close()

    # 4. HTML Contact Sheet
    generate_html_catalog(records, html_path, project_root)


def discover_candidate_files(sources: List[Path]) -> List[Path]:
    """Recursively scans given sources for image files matching supported extensions."""
    results = []
    for s in sources:
        if s.is_file():
            if s.suffix.lower() in SUPPORTED_EXTENSIONS:
                results.append(s)
        elif s.is_dir():
            for root, _, files in os.walk(s):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        results.append(Path(root) / f)
    return sorted(list(set(results)))


def main():
    parser = argparse.ArgumentParser(
        description="Ingest dropped family photos and artworks into the Tamar Sheaffer catalog."
    )
    parser.add_argument('paths', nargs='*', help="Image files or directories to ingest")
    parser.add_argument('--drop-dir', default='drop_folder', help="Drop folder directory (default: drop_folder)")
    parser.add_argument('--dry-run', action='store_true', help="Validate and plan ingestion without writing files or catalog")
    parser.add_argument('--min-size', type=int, default=150, help="Minimum dimension threshold on both axes (default: 150)")
    parser.add_argument('--catalog-json', default='catalog/catalog.json', help="Path to catalog.json")
    parser.add_argument('--catalog-csv', default='catalog/catalog.csv', help="Path to catalog.csv")
    parser.add_argument('--catalog-sqlite', default='catalog/catalog.sqlite', help="Path to catalog.sqlite")
    parser.add_argument('--catalog-html', default='catalog/index.html', help="Path to catalog/index.html")
    parser.add_argument('--media-dir', default='media/drop_folder_imports', help="Destination media folder")
    parser.add_argument('--source-group', default='drop_folder_imports', help="Source group tag")
    parser.add_argument('--title', help="Title override")
    parser.add_argument('--caption', help="Caption override")
    parser.add_argument('--date-or-era', help="Date or era override")
    parser.add_argument('--people', help="Comma-separated people tags")
    parser.add_argument('--places', help="Comma-separated places tags")
    parser.add_argument('--themes', help="Comma-separated art themes")
    parser.add_argument('--confidence', default='CONFIRMED: Family drop folder import', help="Confidence notes")
    parser.add_argument('--rebuild-catalog', action='store_true', help="Rebuild CSV/SQLite/HTML from existing JSON")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose logging")

    args = parser.parse_args()

    project_root = Path.cwd()
    json_path = (project_root / args.catalog_json).resolve()
    csv_path = (project_root / args.catalog_csv).resolve()
    sqlite_path = (project_root / args.catalog_sqlite).resolve()
    html_path = (project_root / args.catalog_html).resolve()
    media_dir = (project_root / args.media_dir).resolve()

    existing_hashes, records = load_catalog(json_path, sqlite_path)

    if args.rebuild_catalog:
        print(f"[REBUILD] Rebuilding catalogs from {len(records)} existing JSON entries...")
        save_catalogs(records, json_path, csv_path, sqlite_path, html_path, project_root)
        print("[REBUILD] Catalogs updated successfully.")
        sys.exit(0)

    # Determine input sources
    candidate_sources = []
    if args.paths:
        for p in args.paths:
            candidate_sources.append(Path(p).resolve())
    else:
        drop_path = (project_root / args.drop_dir).resolve()
        if drop_path.exists():
            candidate_sources.append(drop_path)
        else:
            print(f"[WARN] Drop directory not found: {drop_path}", file=sys.stderr)

    candidates = discover_candidate_files(candidate_sources)
    print(f"[INGEST] Discovered {len(candidates)} candidate image file(s).")

    planned_entries = []
    ingested_count = 0
    duplicate_count = 0
    undersized_count = 0
    corrupt_count = 0

    for cand in candidates:
        ok, reason, meta = inspect_image(cand, min_size=args.min_size)
        if not ok:
            if 'below minimum threshold' in reason:
                undersized_count += 1
                print(f"  [REJECT SMALL] {cand.name} — {reason}")
            else:
                corrupt_count += 1
                print(f"  [REJECT CORRUPT] {cand.name} — {reason}")
            continue

        sha = meta['sha256']
        if sha in existing_hashes:
            duplicate_count += 1
            existing_rec = existing_hashes[sha]
            print(f"  [REJECT DUPLICATE] {cand.name} (SHA-256 {sha[:8]}...) matches existing {existing_rec.get('id')}: {existing_rec.get('rel_path')}")
            continue

        sidecar = parse_sidecar_metadata(cand)

        title = args.title or sidecar.get('title') or cand.stem.replace('_', ' ').title()
        caption = args.caption or sidecar.get('caption') or ''
        date_era = args.date_or_era or sidecar.get('date_or_era') or ''

        people = sidecar.get('people') or []
        if args.people:
            people = [p.strip() for p in args.people.split(',') if p.strip()]

        places = sidecar.get('places') or []
        if args.places:
            places = [p.strip() for p in args.places.split(',') if p.strip()]

        themes = sidecar.get('art_themes') or []
        if args.themes:
            themes = [t.strip() for t in args.themes.split(',') if t.strip()]

        new_id = f"media-drop-{len(records) + len(planned_entries) + 1:04d}"
        target_name = cand.name
        rel_path = f"{args.media_dir}/{target_name}"

        record = {
            'id': new_id,
            'filename': target_name,
            'rel_path': rel_path,
            'sha256': sha,
            'width': meta['width'],
            'height': meta['height'],
            'file_size': meta['file_size'],
            'file_size_bytes': meta['file_size_bytes'],
            'mime_type': meta['mime_type'],
            'source_group': args.source_group,
            'source_url_or_path': str(cand),
            'title': title,
            'caption': caption,
            'date_or_era': date_era,
            'people': people,
            'people_tagged': people,
            'places': places,
            'places_tagged': places,
            'art_themes': themes,
            'art_material_themes': themes,
            'confidence_notes': args.confidence,
            'confidence_curation_notes': args.confidence
        }

        if args.dry_run:
            planned_entries.append(record)
            print(f"  [DRY-RUN] Planned ingestion: {cand.name} ({meta['width']}x{meta['height']}, {meta['mime_type']}, {meta['file_size']} bytes, SHA-256: {sha[:12]}...) -> {rel_path}")
        else:
            media_dir.mkdir(parents=True, exist_ok=True)
            dest_file = media_dir / target_name
            if dest_file.exists() and compute_sha256(dest_file) != sha:
                disambig_name = f"{cand.stem}_{sha[:8]}{cand.suffix}"
                dest_file = media_dir / disambig_name
                record['filename'] = disambig_name
                record['rel_path'] = f"{args.media_dir}/{disambig_name}"

            shutil.copy2(cand, dest_file)
            records.append(record)
            existing_hashes[sha] = record
            ingested_count += 1
            print(f"  [INGESTED] {cand.name} -> {dest_file.name} (ID: {new_id})")

    if not args.dry_run and ingested_count > 0:
        save_catalogs(records, json_path, csv_path, sqlite_path, html_path, project_root)
        print(f"[CATALOG] Successfully updated catalog files (Total catalog items: {len(records)})")

    print(f"[SUMMARY] Processed: {len(candidates)} | Ingested: {ingested_count} | Planned: {len(planned_entries)} | Duplicates: {duplicate_count} | Undersized: {undersized_count} | Corrupt: {corrupt_count}")
    sys.exit(0)


if __name__ == '__main__':
    main()
