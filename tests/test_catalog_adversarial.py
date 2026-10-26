#!/usr/bin/env python3
"""
tests/test_catalog_adversarial.py — Rigorous Adversarial Catalog & Archive Verifier

Adversarially tests and cross-checks:
1. 100% of entries in catalog.json have valid, existing image files on disk at rel_path.
2. 100% of image files have matching SHA-256 hashes recomputed directly from disk.
3. 100% of image dimensions decoded via Pillow match catalog width and height.
4. All 17 schema fields exist and are populated with meaningful tags / values across 100% of entries.
5. 100% synchronization across catalog.json, catalog.sqlite, and catalog.csv.
6. catalog/index.html has 0 broken relative asset links, 0 404s, and embedded CATALOG_DATA matches catalog.json.
7. Unindexed media detector: 0 dangling or unindexed media files in media/.
8. Integrity guardrails: 0 duplicate SHA-256 hashes on disk, all images >= 150px on both dimensions.
"""

import os
import sys
import json
import csv
import re
import hashlib
import sqlite3
import unittest
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDIA_DIR = PROJECT_ROOT / "media"
CATALOG_DIR = PROJECT_ROOT / "catalog"
CATALOG_JSON = CATALOG_DIR / "catalog.json"
CATALOG_CSV = CATALOG_DIR / "catalog.csv"
CATALOG_SQLITE = CATALOG_DIR / "catalog.sqlite"
CATALOG_HTML = CATALOG_DIR / "index.html"

EXPECTED_17_FIELDS = [
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

KNOWN_SOURCE_GROUPS = {
    "wordpress_blog",
    "darchei_etzim_pdfs",
    "portfolio_site",
    "binyamin_gallery",
    "museum_press",
    "whatsapp_family",
    "drop_folder_imports"
}


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


class AdversarialCatalogIntegrityTest(unittest.TestCase):
    """Adversarial stress-testing suite for media archive and catalog files."""

    @classmethod
    def setUpClass(cls):
        if not CATALOG_JSON.exists():
            raise FileNotFoundError(f"Missing catalog.json at {CATALOG_JSON}")
        with open(CATALOG_JSON, "r", encoding="utf-8") as f:
            cls.catalog = json.load(f)

    def test_01_catalog_not_empty_and_sufficient_volume(self):
        """Asserts catalog contains >= 100 entries and spans >= 4 source groups."""
        self.assertIsInstance(self.catalog, list, "catalog.json root must be a list.")
        self.assertGreaterEqual(
            len(self.catalog), 100,
            f"Adversarial check failed: Expected >= 100 items, got {len(self.catalog)}"
        )
        sources = {e.get("source_group") for e in self.catalog}
        self.assertGreaterEqual(
            len(sources), 4,
            f"Adversarial check failed: Expected >= 4 source groups, got {len(sources)}: {sources}"
        )

    def test_02_all_rel_paths_exist_on_disk(self):
        """Asserts 100% of entries point to valid, existing image files on disk."""
        missing = []
        for e in self.catalog:
            rel = e.get("rel_path")
            if not rel:
                missing.append((e.get("id"), "EMPTY_PATH"))
                continue
            full_path = PROJECT_ROOT / rel
            if not full_path.exists():
                missing.append((e.get("id"), str(rel), "NOT_EXISTS"))
            elif not full_path.is_file():
                missing.append((e.get("id"), str(rel), "NOT_A_FILE"))

        self.assertEqual(
            len(missing), 0,
            f"Adversarial check failed: {len(missing)} entries have invalid or missing file paths: {missing}"
        )

    def test_03_actual_sha256_hash_matches_catalog_100_percent(self):
        """Asserts every single image file's recomputed SHA-256 matches catalog sha256."""
        mismatches = []
        for e in self.catalog:
            full_path = PROJECT_ROOT / e["rel_path"]
            actual_sha = compute_file_sha256(full_path)
            catalog_sha = e.get("sha256", "").strip().lower()
            if actual_sha != catalog_sha:
                mismatches.append({
                    "id": e.get("id"),
                    "file": e.get("rel_path"),
                    "catalog_sha": catalog_sha,
                    "actual_sha": actual_sha
                })

        self.assertEqual(
            len(mismatches), 0,
            f"Adversarial check failed: SHA-256 mismatches in {len(mismatches)} files: {mismatches[:5]}"
        )

    def test_04_pillow_dimensions_match_catalog_100_percent(self):
        """Decodes image dimensions using Pillow; compares against width and height in catalog."""
        dim_mismatches = []
        corrupted = []
        for e in self.catalog:
            full_path = PROJECT_ROOT / e["rel_path"]
            try:
                with Image.open(full_path) as img:
                    img.verify()
                with Image.open(full_path) as img:
                    actual_w, actual_h = img.size
            except Exception as exc:
                corrupted.append({"id": e.get("id"), "file": e.get("rel_path"), "error": str(exc)})
                continue

            catalog_w = e.get("width")
            catalog_h = e.get("height")

            if actual_w != catalog_w or actual_h != catalog_h:
                dim_mismatches.append({
                    "id": e.get("id"),
                    "file": e.get("rel_path"),
                    "catalog_dim": (catalog_w, catalog_h),
                    "actual_dim": (actual_w, actual_h)
                })

        self.assertEqual(
            len(corrupted), 0,
            f"Adversarial check failed: {len(corrupted)} images failed Pillow decode: {corrupted}"
        )
        self.assertEqual(
            len(dim_mismatches), 0,
            f"Adversarial check failed: {len(dim_mismatches)} images have dimension mismatches: {dim_mismatches[:5]}"
        )

    def test_05_all_17_schema_fields_present_and_meaningful(self):
        """Confirms all 17 schema fields exist and are populated with meaningful values/tags."""
        missing_fields = []
        unpopulated = []
        invalid_tags = []

        for e in self.catalog:
            eid = e.get("id", "UNKNOWN")
            for field in EXPECTED_17_FIELDS:
                if field not in e:
                    missing_fields.append((eid, field))
                elif e[field] is None:
                    unpopulated.append((eid, field, "NULL_VALUE"))

            # Specific validation for tags
            people = e.get("people")
            places = e.get("places")
            art_themes = e.get("art_themes")

            if not isinstance(people, list):
                invalid_tags.append((eid, "people", "NOT_A_LIST", type(people)))
            if not isinstance(places, list):
                invalid_tags.append((eid, "places", "NOT_A_LIST", type(places)))
            if not isinstance(art_themes, list):
                invalid_tags.append((eid, "art_themes", "NOT_A_LIST", type(art_themes)))

            # Check that tag items are non-empty strings
            for p in (people or []):
                if not isinstance(p, str) or not p.strip():
                    invalid_tags.append((eid, "people_item", p))
            for pl in (places or []):
                if not isinstance(pl, str) or not pl.strip():
                    invalid_tags.append((eid, "places_item", pl))
            for th in (art_themes or []):
                if not isinstance(th, str) or not th.strip():
                    invalid_tags.append((eid, "art_themes_item", th))

            # Validate source_group
            if e.get("source_group") not in KNOWN_SOURCE_GROUPS:
                invalid_tags.append((eid, "source_group", e.get("source_group")))

            # Validate title
            if not isinstance(e.get("title"), str) or not e.get("title").strip():
                unpopulated.append((eid, "title", "EMPTY_STRING"))

            # Validate dimensions >= 150
            if (e.get("width") or 0) < 150 or (e.get("height") or 0) < 150:
                invalid_tags.append((eid, "dimension_under_150", (e.get("width"), e.get("height"))))

            # Validate file size matches actual file on disk
            full_path = PROJECT_ROOT / e.get("rel_path", "")
            if full_path.exists():
                disk_size = full_path.stat().st_size
                if e.get("file_size_bytes") != disk_size:
                    invalid_tags.append((eid, "file_size_mismatch", (e.get("file_size_bytes"), disk_size)))

        self.assertEqual(len(missing_fields), 0, f"Missing fields: {missing_fields[:5]}")
        self.assertEqual(len(unpopulated), 0, f"Unpopulated required fields: {unpopulated[:5]}")
        self.assertEqual(len(invalid_tags), 0, f"Invalid tags or metadata: {invalid_tags[:5]}")

    def test_06_sqlite_synchronization_and_integrity(self):
        """Cross-checks with catalog/catalog.sqlite for 100% synchronization."""
        self.assertTrue(CATALOG_SQLITE.exists(), f"SQLite database missing at {CATALOG_SQLITE}")
        conn = sqlite3.connect(CATALOG_SQLITE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Check total count
        cur.execute("SELECT COUNT(*) FROM catalog;")
        sqlite_count = cur.fetchone()[0]
        self.assertEqual(
            sqlite_count, len(self.catalog),
            f"SQLite count {sqlite_count} != JSON count {len(self.catalog)}"
        )

        # Check indexes
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='catalog';")
        indexes = {r["name"]: (r["sql"] or "") for r in cur.fetchall()}
        self.assertTrue(any("sha256" in k.lower() or "sha256" in v.lower() for k, v in indexes.items()), "Missing sha256 index")
        self.assertTrue(any("source_group" in k.lower() or "source_group" in v.lower() for k, v in indexes.items()), "Missing source_group index")
        self.assertTrue(any("date_or_era" in k.lower() or "date_or_era" in v.lower() for k, v in indexes.items()), "Missing date_or_era index")

        # Check media view
        cur.execute("SELECT COUNT(*) FROM media;")
        view_count = cur.fetchone()[0]
        self.assertEqual(view_count, len(self.catalog), "media view count != catalog count")

        # Cross-check each entry by ID
        mismatches = []
        for entry in self.catalog:
            eid = entry["id"]
            cur.execute("SELECT * FROM catalog WHERE id = ?", (eid,))
            row = cur.fetchone()
            if not row:
                mismatches.append((eid, "ROW_MISSING_IN_SQLITE"))
                continue

            # Compare scalar fields
            for field in ["filename", "rel_path", "sha256", "width", "height", "file_size_bytes",
                          "mime_type", "source_group", "source_url_or_path", "title", "caption",
                          "date_or_era", "confidence_notes"]:
                db_val = row[field]
                json_val = entry[field]
                if db_val != json_val:
                    mismatches.append((eid, field, f"db={db_val} != json={json_val}"))

            # Compare array fields stored as JSON in SQLite
            for field in ["people", "places", "art_themes"]:
                db_raw = row[field]
                try:
                    db_list = json.loads(db_raw) if db_raw else []
                except Exception as ex:
                    mismatches.append((eid, field, f"corrupt JSON in sqlite: {ex}"))
                    continue
                if sorted(db_list) != sorted(entry[field]):
                    mismatches.append((eid, field, f"db={db_list} != json={entry[field]}"))

        conn.close()
        self.assertEqual(
            len(mismatches), 0,
            f"Adversarial check failed: {len(mismatches)} mismatches between SQLite and JSON: {mismatches[:5]}"
        )

    def test_07_csv_synchronization_and_integrity(self):
        """Cross-checks with catalog/catalog.csv for 100% synchronization."""
        self.assertTrue(CATALOG_CSV.exists(), f"CSV file missing at {CATALOG_CSV}")

        # Check BOM
        raw_bytes = CATALOG_CSV.read_bytes()
        self.assertTrue(raw_bytes.startswith(b"\xef\xbb\xbf"), "catalog.csv must begin with UTF-8 BOM")

        with open(CATALOG_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, EXPECTED_17_FIELDS, "CSV header columns do not match EXPECTED_17_FIELDS")
            rows = list(reader)

        self.assertEqual(
            len(rows), len(self.catalog),
            f"CSV row count ({len(rows)}) does not match JSON count ({len(self.catalog)})"
        )

        csv_by_id = {r[0]: r for r in rows}
        mismatches = []

        for entry in self.catalog:
            eid = entry["id"]
            if eid not in csv_by_id:
                mismatches.append((eid, "ROW_MISSING_IN_CSV"))
                continue
            r = csv_by_id[eid]
            row_dict = dict(zip(header, r))

            # Check scalar fields
            for field in ["filename", "rel_path", "sha256", "mime_type", "source_group",
                          "source_url_or_path", "title", "caption", "date_or_era", "confidence_notes"]:
                c_val = row_dict[field]
                j_val = entry[field] or ""
                if c_val != j_val:
                    mismatches.append((eid, field, f"csv={c_val!r} != json={j_val!r}"))

            # Check numeric fields
            for num_field in ["width", "height", "file_size_bytes"]:
                if int(row_dict[num_field]) != entry[num_field]:
                    mismatches.append((eid, num_field, f"csv={row_dict[num_field]} != json={entry[num_field]}"))

            # Check array fields (pipe-delimited in CSV)
            for arr_field in ["people", "places", "art_themes"]:
                raw_csv = row_dict[arr_field].strip()
                csv_items = [x.strip() for x in raw_csv.split(" | ") if x.strip()] if raw_csv else []
                json_items = entry[arr_field]
                if sorted(csv_items) != sorted(json_items):
                    mismatches.append((eid, arr_field, f"csv={csv_items} != json={json_items}"))

        self.assertEqual(
            len(mismatches), 0,
            f"Adversarial check failed: {len(mismatches)} mismatches between CSV and JSON: {mismatches[:5]}"
        )

    def test_08_index_html_assets_and_data_integrity(self):
        """Inspects catalog/index.html to confirm all thumbnails, links, and embedded data resolve."""
        self.assertTrue(CATALOG_HTML.exists(), f"index.html missing at {CATALOG_HTML}")
        html_text = CATALOG_HTML.read_text(encoding="utf-8")

        # 1. Check all img src paths
        img_srcs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html_text)
        self.assertGreater(len(img_srcs), 0, "No img tags found in index.html")

        broken_img_paths = []
        for src in img_srcs:
            if src.startswith("data:") or src.startswith("http://") or src.startswith("https://") or src == "":
                continue
            resolved = (CATALOG_DIR / src).resolve()
            if not resolved.exists():
                broken_img_paths.append((src, str(resolved)))

        self.assertEqual(
            len(broken_img_paths), 0,
            f"Adversarial check failed: {len(broken_img_paths)} broken img src paths in index.html: {broken_img_paths[:5]}"
        )

        # 2. Check all a href links
        a_hrefs = re.findall(r'<a[^>]+href=["\']([^"\']+)["\']', html_text)
        broken_hrefs = []
        for href in a_hrefs:
            if href.startswith("#") or href.startswith("data:") or href.startswith("http://") or href.startswith("https://"):
                continue
            resolved = (CATALOG_DIR / href).resolve()
            if not resolved.exists():
                broken_hrefs.append((href, str(resolved)))

        self.assertEqual(
            len(broken_hrefs), 0,
            f"Adversarial check failed: {len(broken_hrefs)} broken href links in index.html: {broken_hrefs[:5]}"
        )

        # 3. Check embedded CATALOG_DATA
        m = re.search(r'const\s+CATALOG_DATA\s*=\s*(\[[\s\S]*?\]);', html_text)
        self.assertIsNotNone(m, "Failed to find const CATALOG_DATA = [...] in index.html script tag")

        embedded_json = json.loads(m.group(1))
        self.assertEqual(
            len(embedded_json), len(self.catalog),
            f"Embedded CATALOG_DATA count ({len(embedded_json)}) != catalog.json count ({len(self.catalog)})"
        )

        # Check full-size link resolution for all items in CATALOG_DATA
        broken_fullsize = []
        for item in embedded_json:
            full_path = (CATALOG_DIR / ("../" + item["rel_path"])).resolve()
            if not full_path.exists():
                broken_fullsize.append((item["id"], item["rel_path"]))

        self.assertEqual(
            len(broken_fullsize), 0,
            f"Adversarial check failed: {len(broken_fullsize)} broken fullsize links in CATALOG_DATA: {broken_fullsize[:5]}"
        )

    def test_09_zero_unindexed_files_in_media_dir(self):
        """Asserts every image on disk in media/ is registered in catalog.json."""
        catalog_paths = {Path(e["rel_path"]).resolve() for e in self.catalog}
        media_files = []
        for root, _, files in os.walk(MEDIA_DIR):
            for f in files:
                p = (Path(root) / f).resolve()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tiff"}:
                    media_files.append(p)

        unindexed = [str(f.relative_to(PROJECT_ROOT)) for f in media_files if f not in catalog_paths]
        self.assertEqual(
            len(unindexed), 0,
            f"Adversarial check failed: {len(unindexed)} unindexed files in media/: {unindexed[:5]}"
        )

    def test_10_zero_sha256_duplicates_in_archive(self):
        """Asserts zero duplicate SHA-256 hashes across catalog entries."""
        sha_to_id = {}
        duplicates = []
        for e in self.catalog:
            sha = e["sha256"]
            eid = e["id"]
            if sha in sha_to_id:
                duplicates.append((eid, sha_to_id[sha], sha))
            else:
                sha_to_id[sha] = eid

        self.assertEqual(
            len(duplicates), 0,
            f"Adversarial check failed: duplicate SHA-256 hashes found: {duplicates}"
        )


    def test_11_live_http_server_all_assets_200(self):
        """Spins up a local HTTP server and verifies that 100% of thumbnails and fullsize images return HTTP 200 with zero 404s."""
        import http.server
        import threading
        import urllib.request
        import urllib.parse

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)
            def log_message(self, format, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        base_catalog_url = f"http://127.0.0.1:{port}/catalog/index.html"
        try:
            with urllib.request.urlopen(base_catalog_url) as req:
                self.assertEqual(req.status, 200, f"index.html failed with status {req.status}")
                html_text = req.read().decode("utf-8")

            # Extract thumbnails from HTML
            img_srcs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html_text)
            self.assertGreater(len(img_srcs), 0)

            http_errors = []
            for src in img_srcs:
                if not src:
                    continue
                quoted_src = "/".join(urllib.parse.quote(part) for part in src.split("/"))
                target_url = urllib.parse.urljoin(base_catalog_url, quoted_src)
                try:
                    with urllib.request.urlopen(target_url) as res:
                        self.assertEqual(res.status, 200)
                        _ = res.read()
                except Exception as exc:
                    http_errors.append((src, target_url, str(exc)))

            self.assertEqual(
                len(http_errors), 0,
                f"HTTP thumbnail fetch failed for {len(http_errors)} images: {http_errors[:5]}"
            )

            # Test fullsize links for every entry
            fullsize_errors = []
            for e in self.catalog:
                rel = e["rel_path"]
                quoted_rel = "/".join(urllib.parse.quote(part) for part in rel.split("/"))
                fullsize_url = f"http://127.0.0.1:{port}/{quoted_rel}"
                try:
                    with urllib.request.urlopen(fullsize_url) as res:
                        self.assertEqual(res.status, 200)
                        body = res.read()
                        self.assertEqual(
                            len(body), e["file_size_bytes"],
                            f"Byte size mismatch for {e['id']}: expected {e['file_size_bytes']}, got {len(body)}"
                        )
                except Exception as exc:
                    fullsize_errors.append((e["id"], fullsize_url, str(exc)))

            self.assertEqual(
                len(fullsize_errors), 0,
                f"HTTP fullsize image fetch failed for {len(fullsize_errors)} images: {fullsize_errors[:5]}"
            )
        finally:
            server.shutdown()

    def test_12_controlled_vocabulary_and_tag_semantic_health(self):
        """Verifies tag vocabulary alignment, source provenance, and non-empty metadata."""
        tag_vocab_path = CATALOG_DIR / "tag_vocabularies.json"
        self.assertTrue(tag_vocab_path.exists(), f"tag_vocabularies.json missing at {tag_vocab_path}")
        with open(tag_vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        known_people = {item["canonical"] for item in vocab.get("people", [])}
        known_places = {item["canonical"] for item in vocab.get("places", [])}
        known_themes = {item["canonical"] for item in vocab.get("art_motifs", [])}

        unexpected_people = set()
        unexpected_places = set()
        unexpected_themes = set()

        for e in self.catalog:
            for p in e.get("people", []):
                if p not in known_people:
                    unexpected_people.add(p)
            for pl in e.get("places", []):
                if pl not in known_places:
                    unexpected_places.add(pl)
            for th in e.get("art_themes", []):
                if th not in known_themes:
                    unexpected_themes.add(th)

        # All tags in harvested archive must belong to controlled vocabulary
        self.assertEqual(
            len(unexpected_people), 0,
            f"Found tags outside controlled people vocabulary: {unexpected_people}"
        )
        self.assertEqual(
            len(unexpected_places), 0,
            f"Found tags outside controlled places vocabulary: {unexpected_places}"
        )
        self.assertEqual(
            len(unexpected_themes), 0,
            f"Found tags outside controlled art_themes vocabulary: {unexpected_themes}"
        )


if __name__ == "__main__":
    unittest.main()

