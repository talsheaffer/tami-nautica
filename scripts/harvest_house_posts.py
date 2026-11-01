#!/usr/bin/env python3
"""
scripts/harvest_house_posts.py
Targeted read-only harvester for Gad Sheaffer's ("דרכי עצים") biographical house & place posts
that predate 2025–2026, enriching `media/wordpress_blog/` and the 4-format catalog with actual
photographs of Arlozorov 87, Einstein 33A, Greenboim 80 (Denya), Herzliya, and Paris Cité des Arts.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_catalog import build_all_catalogs

WP_API_BASE = "https://public-api.wordpress.com/rest/v1.1/sites/theshingimel.wordpress.com"
HTTP_TIMEOUT_SEC = 10
MIN_IMAGE_DIM_PX = 150

TARGET_HOUSE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "query_url": f"{WP_API_BASE}/posts/slug:27-37",
        "places": ["Haifa: Arlozorov"],
        "people": ["Tamar 'Tami' Sheaffer", "Gad Sheaffer", "Guy Sheaffer"],
        "era": "1982–1992",
        "label_prefix": "ארלוזורוב 87 חיפה",
    },
    {
        "query_url": f"{WP_API_BASE}/posts/slug:38-50",
        "places": ["Haifa: Einstein"],
        "people": ["Tamar 'Tami' Sheaffer", "Gad Sheaffer", "Guy Sheaffer", "Tal Sheaffer"],
        "era": "1992–2004",
        "label_prefix": "רחוב איינשטיין 33א חיפה",
    },
    {
        "query_url": f"{WP_API_BASE}/posts/slug:g80",
        "places": ["Haifa: Denya (80 Greenboim St & studio)"],
        "people": ["Tamar 'Tami' Sheaffer", "Gad Sheaffer"],
        "era": "2004–Present",
        "label_prefix": "גרינבוים 80 דניה",
    },
    {
        "query_url": f"{WP_API_BASE}/posts/?search=" + urllib.parse.quote("אל המנוחה ואל הנחלה") + "&number=5",
        "places": ["Haifa: Denya (80 Greenboim St & studio)"],
        "people": ["Tamar 'Tami' Sheaffer", "Gad Sheaffer"],
        "era": "2004–Present",
        "label_prefix": "גרינבוים 80 דניה — בניית הבית",
    },
    {
        "query_url": f"{WP_API_BASE}/posts/slug:61",
        "places": ["Paris (Cité des Arts)"],
        "people": ["Tamar 'Tami' Sheaffer", "Gad Sheaffer"],
        "era": "2016",
        "label_prefix": "פריז — Cité des Arts",
    },
    {
        "query_url": f"{WP_API_BASE}/posts/slug:safe",
        "places": ["Herzliya (parents' house & yard)"],
        "people": ["Tamar 'Tami' Sheaffer", "Michael Fried", "Ahuva Fried"],
        "era": "Childhood / Family Memory",
        "label_prefix": "מקום בטוח — בית ההורים בהרצליה",
    },
    {
        "query_url": f"{WP_API_BASE}/posts/?search=" + urllib.parse.quote("אהובה פריד") + "&number=5",
        "places": ["Herzliya (parents' house & yard)"],
        "people": ["Ahuva Fried", "Michael Fried", "Tamar 'Tami' Sheaffer", "Er'ela Sadan"],
        "era": "Herzliya Family Archive",
        "label_prefix": "לזכרה של אהובה פריד — הרצליה",
    },
)

IMG_URL_RE = re.compile(
    r"https://theshingimel\.wordpress\.com/wp-content/uploads/[^\s\"'<>?]+"
)


def fetch_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "Tami70ReadOnlyArchiver/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_bytes(url: str) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    quoted_path = "/".join(urllib.parse.quote(urllib.parse.unquote(p)) for p in parsed.path.split("/"))
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, quoted_path, "", ""))
    req = urllib.request.Request(safe_url, headers={"User-Agent": "Tami70ReadOnlyArchiver/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
        return resp.read()


def main() -> None:
    catalog_json = PROJECT_ROOT / "catalog" / "catalog.json"
    existing = json.loads(catalog_json.read_text(encoding="utf-8")) if catalog_json.is_file() else []
    seen_sha = {str(r.get("sha256", "")).lower() for r in existing}

    dest_dir = PROJECT_ROOT / "media" / "wordpress_blog"
    dest_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    for spec in TARGET_HOUSE_SPECS:
        try:
            payload = fetch_json(spec["query_url"])
        except Exception as exc:
            print(f"[WARN] Query failed for {spec['label_prefix']}: {exc}")
            continue

        posts = payload.get("posts", [payload]) if "posts" in payload else [payload]
        for post in posts:
            post_title = re.sub(r"<[^>]+>", "", str(post.get("title") or "")).strip()
            post_url = str(post.get("URL") or "")
            content_html = str(post.get("content") or "")
            img_urls = sorted(set(IMG_URL_RE.findall(content_html)))

            for idx, img_url in enumerate(img_urls, start=1):
                # Skip resized WordPress `-150x150` thumbnails if original exists
                clean_url = re.sub(r"-\d+x\d+(\.[A-Za-z]+)$", r"\1", img_url)
                raw: bytes | None = None
                for candidate_url in (clean_url, img_url):
                    try:
                        raw = download_bytes(candidate_url)
                        break
                    except Exception:
                        continue
                if not raw:
                    continue

                sha = hashlib.sha256(raw).hexdigest().lower()
                if sha in seen_sha:
                    continue

                try:
                    with Image.open(io.BytesIO(raw)) as vimg:
                        vimg.verify()
                    with Image.open(io.BytesIO(raw)) as img:
                        img.load()
                        w, h = img.size
                        fmt = (img.format or "JPEG").upper()
                except (UnidentifiedImageError, OSError, SyntaxError):
                    continue

                if w < MIN_IMAGE_DIM_PX or h < MIN_IMAGE_DIM_PX:
                    continue

                ext = ".png" if fmt == "PNG" else ".webp" if fmt == "WEBP" else ".jpg"
                stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(clean_url).stem).strip("_")[:40] or "house"
                filename = f"wp_house_{stem}_{sha[:8]}{ext}"
                out_img = dest_dir / filename
                out_json = dest_dir / f"{filename}.json"

                sidecar = {
                    "title": f"{spec['label_prefix']} — {post_title} (#{idx})",
                    "caption": f"מתוך הפוסט ״{post_title}״ בבלוג דרכי עצים ({post_url})",
                    "date_or_era": spec["era"],
                    "places": spec["places"],
                    "people": spec["people"],
                    "art_themes": [],
                    "source_url_or_path": clean_url,
                    "confidence_notes": f"CONFIRMED: Extracted from house/place post '{post_title}' ({w}x{h})",
                }
                out_img.write_bytes(raw)
                out_json.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                seen_sha.add(sha)
                added += 1
                print(f"  + Added {filename} ({w}x{h}) [{spec['label_prefix']}]")

    if added > 0:
        stats = build_all_catalogs(
            project_root=PROJECT_ROOT,
            media_dir=PROJECT_ROOT / "media",
            catalog_dir=PROJECT_ROOT / "catalog",
            min_size=MIN_IMAGE_DIM_PX,
            verbose=False,
        )
        print(f"[DONE] Added {added} house/place photos. Total catalog count: {stats['count']}")
    else:
        print("[DONE] No new house photos added.")


if __name__ == "__main__":
    main()
