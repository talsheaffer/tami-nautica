#!/usr/bin/env python3
"""
tests/test_ingestion_pipeline.py — Ingestion Pipeline Automated Test Suite

Verifies Milestone 2 (Requirement R2):
1. --dry-run with a synthetic test image leaves catalog and media untouched.
2. Live ingestion into a temporary test catalog copies the file, indexes the hash,
   and populates all 4 catalog formats (JSON, CSV, SQLite, HTML).
3. Duplicate rejection detects matching SHA-256 hashes and skips ingestion.
4. Small image rejection (<150px on any axis) eliminates tiny icons and badges.
5. Corrupted image rejection handles invalid headers and truncated bodies gracefully.
6. Sidecar metadata ingestion automatically extracts companion tags and titles.
"""

import unittest
import tempfile
import subprocess
import sys
import os
import shutil
import json
import sqlite3
import csv
from pathlib import Path
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestIngestionPipeline(unittest.TestCase):
    def setUp(self):
        """Creates an isolated temporary sandbox for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.drop_dir = self.root / 'drop_folder'
        self.media_dir = self.root / 'media' / 'drop_folder_imports'
        self.catalog_dir = self.root / 'catalog'
        self.drop_dir.mkdir(parents=True)
        self.catalog_dir.mkdir(parents=True)
        self.media_dir.mkdir(parents=True)

        self.catalog_json = self.catalog_dir / 'catalog.json'
        self.catalog_csv = self.catalog_dir / 'catalog.csv'
        self.catalog_sqlite = self.catalog_dir / 'catalog.sqlite'
        self.catalog_html = self.catalog_dir / 'index.html'

        self.ingest_script = PROJECT_ROOT / 'tools' / 'ingest.py'

    def tearDown(self):
        """Cleans up temporary sandbox."""
        self.temp_dir.cleanup()

    def run_ingest(self, extra_args=None):
        """Helper to invoke tools/ingest.py targeting the sandbox."""
        cmd = [
            sys.executable, str(self.ingest_script),
            '--drop-dir', str(self.drop_dir),
            '--media-dir', str(os.path.relpath(self.media_dir, self.root)),
            '--catalog-json', str(os.path.relpath(self.catalog_json, self.root)),
            '--catalog-csv', str(os.path.relpath(self.catalog_csv, self.root)),
            '--catalog-sqlite', str(os.path.relpath(self.catalog_sqlite, self.root)),
            '--catalog-html', str(os.path.relpath(self.catalog_html, self.root)),
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, cwd=str(self.root), capture_output=True, text=True)

    def test_dry_run_synthetic_image(self):
        """--dry-run should validate image, print plan, but make no filesystem or catalog changes."""
        test_img = self.drop_dir / 'test_sample.jpg'
        im = Image.new('RGB', (300, 200), color=(120, 160, 200))
        im.save(test_img, format='JPEG')

        res = self.run_ingest(['--dry-run'])
        self.assertEqual(res.returncode, 0, f"Dry run failed with code {res.returncode}:\n{res.stderr}")
        self.assertIn('[DRY-RUN] Planned ingestion', res.stdout)
        self.assertIn('test_sample.jpg', res.stdout)
        self.assertIn('300x200', res.stdout)

        # Assert no file was copied to media dir
        self.assertEqual(len(list(self.media_dir.glob('*'))), 0)
        # Assert no catalog files created
        self.assertFalse(self.catalog_json.exists())
        self.assertFalse(self.catalog_sqlite.exists())
        self.assertFalse(self.catalog_csv.exists())
        self.assertFalse(self.catalog_html.exists())

    def test_live_ingestion_synthetic_image(self):
        """Live ingestion should copy file and update JSON, CSV, SQLite, and HTML catalogs."""
        test_img = self.drop_dir / 'photo_live.jpg'
        im = Image.new('RGB', (400, 300), color=(200, 100, 50))
        im.save(test_img, format='JPEG')

        res = self.run_ingest()
        self.assertEqual(res.returncode, 0, f"Live ingestion failed with code {res.returncode}:\n{res.stderr}")
        self.assertIn('[INGESTED] photo_live.jpg', res.stdout)

        # 1. Verify copied file exists and binary matches
        copied = list(self.media_dir.glob('*'))
        self.assertEqual(len(copied), 1)
        self.assertEqual(copied[0].name, 'photo_live.jpg')
        self.assertEqual(copied[0].read_bytes(), test_img.read_bytes())

        # 2. Verify catalog.json schema and fields
        self.assertTrue(self.catalog_json.exists())
        data = json.loads(self.catalog_json.read_text(encoding='utf-8'))
        self.assertEqual(len(data), 1)
        rec = data[0]
        self.assertEqual(rec['filename'], 'photo_live.jpg')
        self.assertEqual(rec['width'], 400)
        self.assertEqual(rec['height'], 300)
        self.assertEqual(rec['mime_type'], 'image/jpeg')

        expected_fields = [
            'id', 'filename', 'rel_path', 'sha256', 'width', 'height',
            'file_size', 'mime_type', 'source_group', 'source_url_or_path',
            'title', 'caption', 'date_or_era', 'people', 'places',
            'art_themes', 'confidence_notes'
        ]
        for field in expected_fields:
            self.assertIn(field, rec, f"Required field missing from catalog.json: '{field}'")

        # 3. Verify catalog.csv format
        self.assertTrue(self.catalog_csv.exists())
        with open(self.catalog_csv, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['filename'], 'photo_live.jpg')
            self.assertEqual(int(rows[0]['width']), 400)
            self.assertEqual(int(rows[0]['height']), 300)

        # 4. Verify catalog.sqlite schema and query
        self.assertTrue(self.catalog_sqlite.exists())
        conn = sqlite3.connect(self.catalog_sqlite)
        cur = conn.cursor()
        cur.execute('SELECT filename, width, height, mime_type FROM media WHERE sha256 = ?', (rec['sha256'],))
        row = cur.fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 'photo_live.jpg')
        self.assertEqual(row[1], 400)
        self.assertEqual(row[2], 300)
        self.assertEqual(row[3], 'image/jpeg')

        # 5. Verify catalog/index.html contact sheet
        self.assertTrue(self.catalog_html.exists())
        html_doc = self.catalog_html.read_text(encoding='utf-8')
        self.assertIn('photo_live.jpg', html_doc)
        self.assertIn('400x300', html_doc)

    def test_duplicate_rejection(self):
        """Duplicate image should be detected via SHA-256 and skipped without creating duplicates."""
        test_img = self.drop_dir / 'original.jpg'
        im = Image.new('RGB', (250, 250), color=(50, 150, 50))
        im.save(test_img, format='JPEG')

        # First ingestion
        res1 = self.run_ingest()
        self.assertEqual(res1.returncode, 0)
        self.assertIn('[INGESTED] original.jpg', res1.stdout)

        # Second ingestion with duplicate file under different filename
        dup_img = self.drop_dir / 'duplicate_copy.jpg'
        shutil.copy2(test_img, dup_img)
        res2 = self.run_ingest()
        self.assertEqual(res2.returncode, 0)
        self.assertIn('[REJECT DUPLICATE]', res2.stdout)

        # Verify catalog still contains exactly 1 item
        data = json.loads(self.catalog_json.read_text(encoding='utf-8'))
        self.assertEqual(len(data), 1)

    def test_small_image_rejection(self):
        """Images smaller than 150px on any axis should be rejected as UI icons / thumbnails."""
        small_img = self.drop_dir / 'tiny_icon.png'
        im = Image.new('RGB', (100, 120), color=(0, 0, 0))
        im.save(small_img, format='PNG')

        res = self.run_ingest()
        self.assertEqual(res.returncode, 0)
        self.assertIn('[REJECT SMALL]', res.stdout)
        self.assertIn('below minimum threshold', res.stdout)

        # Verify no file copied and catalog was not created
        self.assertEqual(len(list(self.media_dir.glob('*'))), 0)
        self.assertFalse(self.catalog_json.exists())

    def test_corrupt_file_rejection(self):
        """Corrupted or non-image files should be rejected gracefully without crashing."""
        corrupt_file = self.drop_dir / 'corrupt.jpg'
        corrupt_file.write_bytes(b'NotAnImageDataAtAll' * 20)

        res = self.run_ingest()
        self.assertEqual(res.returncode, 0)
        self.assertIn('[REJECT CORRUPT]', res.stdout)
        self.assertEqual(len(list(self.media_dir.glob('*'))), 0)

    def test_sidecar_metadata_parsing(self):
        """Companion sidecar JSON files should populate custom title, date, and tags."""
        img_file = self.drop_dir / 'family_gathering.jpg'
        im = Image.new('RGB', (500, 400), color=(70, 80, 90))
        im.save(img_file, format='JPEG')

        sidecar_file = self.drop_dir / 'family_gathering.jpg.json'
        sidecar_file.write_text(json.dumps({
            'title': 'חצר הבית בהרצליה',
            'caption': 'תמי ומיכאל עובדים בגינה',
            'date_or_era': '1965',
            'people': ['תמר שפר', 'מיכאל שפר'],
            'places': ['הרצליה - בית הילדות והחצר'],
            'art_themes': ['חצר ועצים']
        }, ensure_ascii=False), encoding='utf-8')

        res = self.run_ingest()
        self.assertEqual(res.returncode, 0)
        data = json.loads(self.catalog_json.read_text(encoding='utf-8'))
        rec = data[0]
        self.assertEqual(rec['title'], 'חצר הבית בהרצליה')
        self.assertEqual(rec['caption'], 'תמי ומיכאל עובדים בגינה')
        self.assertEqual(rec['date_or_era'], '1965')
        self.assertIn('תמר שפר', rec['people'])
        self.assertIn('הרצליה - בית הילדות והחצר', rec['places'])


if __name__ == '__main__':
    unittest.main()
