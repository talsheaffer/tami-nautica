#!/usr/bin/env python3
"""
tests/test_ingest.py — Ingestion Pipeline Automated Test Suite Runner

Runs TestIngestionPipeline from tests/test_ingestion_pipeline.py.
"""

import sys
import unittest
from pathlib import Path

tests_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(tests_dir))

from test_ingestion_pipeline import TestIngestionPipeline

if __name__ == '__main__':
    unittest.main()
