#!/usr/bin/env python3
"""
tests/test_media_archive.py — Automated Validation Test Suite for Media Archive & Catalog

Verifies Requirement R2 & Milestone 2 Acceptance Criteria:
1. >= 100 unique images across >= 4 distinct source groups.
2. 0 duplicate SHA-256 hashes across all image files on disk.
3. 0 corrupt images (validated via PIL.Image.open and verify()).
4. All images >= 150px on both dimensions (width >= 150 and height >= 150).
5. 100% catalog entry coverage with 0 broken relative file paths across:
   - catalog/catalog.json
   - catalog/catalog.csv (with UTF-8 BOM)
   - catalog/catalog.sqlite (with PK id, unique index sha256, indexes on source_group, date_or_era)
   - catalog/index.html (with valid relative asset links)
6. All 17 schema fields present and normalized.

Standard library + Pillow only.
Usage:
    python3 -m unittest tests/test_media_archive.py
"""

from __future__ import annotations

import os
import sys
import json
import csv
import hashlib
import sqlite3
import re
import unittest
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_catalog import UPRIGHT_DERIVATIVES

MEDIA_DIR = PROJECT_ROOT / "media"
CATALOG_DIR = PROJECT_ROOT / "catalog"

EXPECTED_SCHEMA_FIELDS = [
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

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tiff"}


def get_all_media_files() -> list[Path]:
    """Finds all canonical archival image files located in MEDIA_DIR (excluding upright display derivatives)."""
    if not MEDIA_DIR.exists():
        return []
    media_files: list[Path] = []
    for root, dirs, files in os.walk(MEDIA_DIR):
        if Path(root) == MEDIA_DIR:
            dirs[:] = [d for d in dirs if d != UPRIGHT_DERIVATIVES.derivatives_subdir]
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                media_files.append(p)
    return sorted(media_files)


def get_all_upright_derivative_files() -> list[Path]:
    """Finds all generated upright display-derivative image files under MEDIA_DIR/<derivatives_subdir>."""
    derivatives_root = MEDIA_DIR / UPRIGHT_DERIVATIVES.derivatives_subdir
    if not derivatives_root.exists():
        return []
    derivative_files: list[Path] = []
    for root, _, files in os.walk(derivatives_root):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                derivative_files.append(p)
    return sorted(derivative_files)


class TestMediaArchiveCompleteness(unittest.TestCase):
    """Verifies that the harvested media archive meets volume and source diversity requirements."""

    def setUp(self):
        self.media_files = get_all_media_files()
        if not self.media_files:
            self.skipTest(f"Media archive not populated yet under {MEDIA_DIR}.")

    def test_minimum_harvested_images_count(self):
        """Acceptance Criteria: Archive must contain at least 100 unique image files."""
        self.assertGreaterEqual(
            len(self.media_files), 100,
            f"Expected at least 100 harvested image files, found only {len(self.media_files)} in {MEDIA_DIR}."
        )

    def test_minimum_source_diversity(self):
        """Acceptance Criteria: Media must span at least 4 distinct source groups."""
        source_groups = set()
        for f in self.media_files:
            rel = f.relative_to(MEDIA_DIR)
            if len(rel.parts) > 1:
                source_groups.add(rel.parts[0])

        self.assertGreaterEqual(
            len(source_groups), 4,
            f"Expected at least 4 distinct source groups, found {len(source_groups)}: {sorted(list(source_groups))}"
        )


class TestImageIntegrityAndDimensions(unittest.TestCase):
    """Verifies that zero corrupt images exist and all meet the >=150px threshold."""

    def setUp(self):
        self.media_files = get_all_media_files()
        if not self.media_files:
            self.skipTest(f"Media archive not populated yet under {MEDIA_DIR}.")
        if Image is None:
            self.fail("Pillow library is not installed.")

    def test_zero_corrupt_images(self):
        """Acceptance Criteria: Every image must decode successfully without corruption."""
        corrupt_files = []
        for f in self.media_files:
            try:
                with Image.open(f) as img:
                    img.verify()
                with Image.open(f) as img:
                    _ = img.size
                    _ = img.format
            except Exception as e:
                corrupt_files.append((str(f), str(e)))

        self.assertEqual(
            len(corrupt_files), 0,
            f"Found {len(corrupt_files)} corrupt images in archive: {corrupt_files[:5]}"
        )

    def test_minimum_dimensions_threshold(self):
        """Acceptance Criteria: Filter out tiny UI chrome/icons (<150px on both dimensions)."""
        under_sized = []
        for f in self.media_files:
            try:
                with Image.open(f) as img:
                    w, h = img.size
                    if w < 150 or h < 150:
                        under_sized.append((str(f), w, h))
            except Exception:
                continue

        self.assertEqual(
            len(under_sized), 0,
            f"Found {len(under_sized)} images smaller than 150px on one or both axes: {under_sized[:5]}"
        )

    def test_zero_duplicate_sha256_hashes(self):
        """Acceptance Criteria: Zero duplicate SHA-256 hashes stored on disk."""
        seen = {}
        duplicates = []
        for f in self.media_files:
            h = hashlib.sha256()
            with open(f, "rb") as fp:
                while chunk := fp.read(65536):
                    h.update(chunk)
            sha = h.hexdigest().lower()
            if sha in seen:
                duplicates.append((str(f), seen[sha], sha))
            else:
                seen[sha] = str(f)

        self.assertEqual(
            len(duplicates), 0,
            f"Found {len(duplicates)} duplicate files on disk with identical SHA-256: {duplicates[:5]}"
        )


class TestCatalogArtifacts(unittest.TestCase):
    """Verifies the 4 synchronized catalog outputs and 17-field schema contract."""

    def setUp(self):
        self.json_path = CATALOG_DIR / "catalog.json"
        self.csv_path = CATALOG_DIR / "catalog.csv"
        self.sqlite_path = CATALOG_DIR / "catalog.sqlite"
        self.html_path = CATALOG_DIR / "index.html"

        if not self.json_path.exists():
            self.skipTest(f"Catalog JSON not found at {self.json_path}.")

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.catalog_data = json.load(f)

    def test_catalog_json_schema_and_fields(self):
        """Catalog JSON must contain all 17 normalized fields per item."""
        self.assertIsInstance(self.catalog_data, list, "Catalog JSON root must be an array.")
        self.assertGreater(len(self.catalog_data), 0, "Catalog JSON must not be empty.")

        for idx, entry in enumerate(self.catalog_data):
            for field in EXPECTED_SCHEMA_FIELDS:
                self.assertIn(
                    field, entry,
                    f"Catalog entry #{idx} ({entry.get('id', 'unknown')}) missing required schema field: '{field}'"
                )

            self.assertIsInstance(entry["id"], str)
            self.assertIsInstance(entry["filename"], str)
            self.assertIsInstance(entry["rel_path"], str)
            self.assertIsInstance(entry["sha256"], str)
            self.assertEqual(len(entry["sha256"]), 64, f"Invalid SHA-256 length in entry {entry['id']}")
            self.assertIsInstance(entry["width"], int)
            self.assertIsInstance(entry["height"], int)
            self.assertGreaterEqual(entry["width"], 150)
            self.assertGreaterEqual(entry["height"], 150)
            self.assertIsInstance(entry["file_size_bytes"], int)
            self.assertIsInstance(entry["people"], list)
            self.assertIsInstance(entry["places"], list)
            self.assertIsInstance(entry["art_themes"], list)

    def test_zero_broken_relative_paths(self):
        """Every rel_path and display_rel_path in catalog.json must exist as a real file on disk."""
        broken_paths = []
        for entry in self.catalog_data:
            target = PROJECT_ROOT / entry["rel_path"]
            if not target.exists() or not target.is_file():
                broken_paths.append((entry["id"], entry["rel_path"]))
            display_rel = entry.get("display_rel_path")
            if display_rel is not None:
                display_target = PROJECT_ROOT / display_rel
                if not display_target.exists() or not display_target.is_file():
                    broken_paths.append((entry["id"], display_rel))

        self.assertEqual(
            len(broken_paths), 0,
            f"Found {len(broken_paths)} broken relative paths in catalog: {broken_paths[:5]}"
        )

    def test_catalog_100_percent_coverage(self):
        """Every archival image in media/ must have a catalog entry, and every upright derivative must be referenced."""
        media_files = get_all_media_files()
        if not media_files:
            self.skipTest("No media files found to test coverage.")

        catalog_paths = {entry["rel_path"] for entry in self.catalog_data}
        unindexed = []
        for f in media_files:
            rel = str(f.relative_to(PROJECT_ROOT))
            if rel not in catalog_paths:
                unindexed.append(rel)

        self.assertEqual(
            len(unindexed), 0,
            f"Found {len(unindexed)} unindexed media files on disk: {unindexed[:5]}"
        )

        referenced_derivatives = {
            entry["display_rel_path"]
            for entry in self.catalog_data
            if entry.get("display_rel_path") and entry["display_rel_path"] != entry["rel_path"]
        }
        orphan_derivatives = []
        for deriv in get_all_upright_derivative_files():
            rel = str(deriv.relative_to(PROJECT_ROOT))
            if rel not in referenced_derivatives:
                orphan_derivatives.append(rel)

        self.assertEqual(
            len(orphan_derivatives), 0,
            f"Found {len(orphan_derivatives)} unreferenced upright derivative files on disk: {orphan_derivatives[:5]}"
        )

    def test_catalog_csv_utf8_bom_and_columns(self):
        """Catalog CSV must exist, start with UTF-8 BOM, and contain all schema columns."""
        self.assertTrue(self.csv_path.exists(), f"catalog.csv missing at {self.csv_path}")

        raw_bytes = self.csv_path.read_bytes()
        self.assertTrue(
            raw_bytes.startswith(b"\xef\xbb\xbf"),
            "catalog.csv MUST start with UTF-8 BOM (\\xef\\xbb\\xbf) for proper Hebrew rendering in Excel/Numbers."
        )

        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(
                header, EXPECTED_SCHEMA_FIELDS,
                f"CSV header columns do not match expected schema fields.\nGot: {header}\nExpected: {EXPECTED_SCHEMA_FIELDS}"
            )
            rows = list(reader)
            self.assertEqual(
                len(rows), len(self.catalog_data),
                f"CSV row count ({len(rows)}) does not match JSON count ({len(self.catalog_data)})"
            )

    def test_catalog_sqlite_table_and_indexes(self):
        """Catalog SQLite database must have table, primary key, and required indexes."""
        self.assertTrue(self.sqlite_path.exists(), f"catalog.sqlite missing at {self.sqlite_path}")

        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()

        cur.execute("PRAGMA table_info(catalog);")
        columns = {row[1]: {"type": row[2], "pk": row[5]} for row in cur.fetchall()}
        self.assertIn("id", columns)
        self.assertEqual(columns["id"]["pk"], 1, "Field 'id' must be the primary key in catalog table.")

        for field in EXPECTED_SCHEMA_FIELDS:
            self.assertIn(field, columns, f"SQLite catalog table missing column: '{field}'")

        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='catalog';")
        indexes = {row[0]: row[1] or "" for row in cur.fetchall()}

        self.assertTrue(
            any("sha256" in name.lower() or "sha256" in sql.lower() for name, sql in indexes.items()),
            "SQLite catalog table must have an index on sha256."
        )
        self.assertTrue(
            any("source_group" in name.lower() or "source_group" in sql.lower() for name, sql in indexes.items()),
            "SQLite catalog table must have an index on source_group."
        )
        self.assertTrue(
            any("date_or_era" in name.lower() or "date_or_era" in sql.lower() for name, sql in indexes.items()),
            "SQLite catalog table must have an index on date_or_era."
        )

        cur.execute("SELECT COUNT(*) FROM catalog;")
        row_count = cur.fetchone()[0]
        self.assertEqual(row_count, len(self.catalog_data), "SQLite row count does not match catalog.json.")
        conn.close()

    def test_catalog_index_html_validity(self):
        """Catalog HTML contact sheet must exist, be valid HTML5, and contain zero broken relative links."""
        self.assertTrue(self.html_path.exists(), f"catalog/index.html missing at {self.html_path}")

        content = self.html_path.read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn('dir="rtl"', content)
        self.assertIn("gallery-grid", content)

        src_matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', content)
        self.assertGreater(len(src_matches), 0, "No img tags found in catalog/index.html")

        broken_html_links = []
        for src in src_matches:
            if src.startswith("data:") or src.startswith("http"):
                continue
            resolved = (CATALOG_DIR / src).resolve()
            if not resolved.exists():
                broken_html_links.append((src, str(resolved)))

        self.assertEqual(
            len(broken_html_links), 0,
            f"Found {len(broken_html_links)} broken relative image paths in catalog/index.html: {broken_html_links[:5]}"
        )


if __name__ == "__main__":
    unittest.main()
