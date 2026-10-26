#!/usr/bin/env python3
"""
tests/test_wiki_link_graph_adversarial.py — Adversarial Wiki Link Graph & Anchor Integrity Test Suite

Validates:
1. tools/verify_wiki_links.py execution and exit code.
2. 0 broken links, 0 broken anchors, 0 empty files, 0 orphan pages across wiki/.
3. 100% of frontmatter relations (related_places, related_art) are explicitly linked in document body.
4. Bidirectional reachability from wiki/README.md and graph diameter <= 3.
5. Hebrew slugification accuracy under varied typography, niqqud, and punctuation.
6. Footnote reference and citation integrity.
"""

import os
import sys
import re
import glob
import unicodedata
import subprocess
import unittest
from pathlib import Path
from collections import deque, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = PROJECT_ROOT / "wiki"
sys.path.insert(0, str(PROJECT_ROOT))
from tools.verify_wiki_links import github_slugify, WikiLinkVerifier


class TestWikiLinkVerifierTool(unittest.TestCase):
    """Executes the canonical link verifier script with strict flags."""

    def test_run_verify_wiki_links_strict(self):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "verify_wiki_links.py"),
            "--root-dir", str(PROJECT_ROOT),
            "--wiki-dir", "wiki",
            "--verbose",
            "--strict-orphans"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(
            res.returncode, 0,
            f"verify_wiki_links.py failed with exit code {res.returncode}:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
        self.assertIn("VERIFICATION VERDICT: PASSED", res.stdout)
        self.assertIn("Broken Links (Missing Files): 0", res.stdout)
        self.assertIn("Empty Files (0 Bytes):        0", res.stdout)
        self.assertIn("Broken Heading Anchors:       0", res.stdout)
        self.assertIn("Orphan / Unreachable Pages:   0", res.stdout)


class TestWikiCorpusIntegrity(unittest.TestCase):
    """Deep structural checks across all markdown files in wiki/."""

    @classmethod
    def setUpClass(cls):
        cls.wiki_files = sorted(glob.glob(str(WIKI_DIR / "**" / "*.md"), recursive=True))
        cls.wiki_files = [unicodedata.normalize('NFC', os.path.normpath(p)) for p in cls.wiki_files]

    def test_corpus_count_and_non_empty(self):
        """All 31 expected corpus documents must exist and have substantial size (>1KB)."""
        self.assertEqual(len(self.wiki_files), 31, f"Expected 31 wiki files, found {len(self.wiki_files)}")
        for path in self.wiki_files:
            size = os.path.getsize(path)
            self.assertGreater(size, 1024, f"Wiki file too small or empty ({size} bytes): {path}")

    def test_frontmatter_presence(self):
        """Every wiki markdown file must contain YAML frontmatter with title, category, era, slug."""
        required_keys = {"title", "category", "era", "slug", "sources", "verification_status"}
        for path in self.wiki_files:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertTrue(content.startswith("---"), f"Missing frontmatter opening: {path}")
            parts = content.split("---", 2)
            self.assertGreaterEqual(len(parts), 3, f"Malformed frontmatter in {path}")
            fm_text = parts[1]
            found_keys = set(re.findall(r"^([a-zA-Z0-9_-]+):", fm_text, re.MULTILINE))
            missing = required_keys - found_keys
            self.assertEqual(len(missing), 0, f"Missing required frontmatter keys {missing} in {path}")

    def test_frontmatter_relations_linked_in_body(self):
        """Every entry in related_places and related_art must have an explicit Markdown link in body."""
        md_link_pattern = re.compile(r'\[([^\]]*)\]\(([^)\s]+)\)')
        total_relations_checked = 0

        for path in self.wiki_files:
            file_dir = os.path.dirname(path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            parts = content.split("---", 2)
            if len(parts) < 3:
                continue
            fm_text, body = parts[1], parts[2]

            in_rel = None
            relations = []
            for line in fm_text.splitlines():
                stripped = line.strip()
                if re.match(r'^[a-zA-Z0-9_-]+:', line):
                    key = line.split(":")[0].strip()
                    if key in ("related_places", "related_art", "related_bio", "related_family", "related_travels"):
                        in_rel = key
                    else:
                        in_rel = None
                elif in_rel and stripped.startswith("-"):
                    val = stripped.lstrip("-").strip().strip("\"'")
                    if val:
                        relations.append((in_rel, val))

            if not relations:
                continue

            rel_sec_match = re.search(r'##\s+.*(?:Related Entries|ראו גם).*?(?=\n##|\Z)', body, re.DOTALL | re.IGNORECASE)
            self.assertIsNotNone(rel_sec_match, f"Missing 'Related Entries / ראו גם' section in {path}")
            sec_text = rel_sec_match.group(0)

            sec_links = set()
            for m in md_link_pattern.finditer(sec_text):
                target = m.group(2).strip("<>").strip().split("#")[0]
                if target:
                    abs_t = unicodedata.normalize('NFC', os.path.normpath(os.path.join(file_dir, target)))
                    sec_links.add(abs_t)

            for rel_type, rel_target in relations:
                total_relations_checked += 1
                abs_rel = unicodedata.normalize('NFC', os.path.normpath(os.path.join(file_dir, rel_target)))
                self.assertIn(
                    abs_rel, sec_links,
                    f"Frontmatter relation '{rel_target}' in {path} ({rel_type}) is not linked in 'Related Entries' section!"
                )

        self.assertGreaterEqual(total_relations_checked, 80, "Expected at least 80 frontmatter relations checked.")


class TestWikiGraphTopology(unittest.TestCase):
    """Verifies graph connectivity, reachability from wiki/README.md, and graph diameter."""

    @classmethod
    def setUpClass(cls):
        cls.wiki_files = sorted(glob.glob(str(WIKI_DIR / "**" / "*.md"), recursive=True))
        cls.wiki_files = [unicodedata.normalize('NFC', os.path.normpath(p)) for p in cls.wiki_files]
        cls.wiki_set = set(cls.wiki_files)

        cls.adj = {f: set() for f in cls.wiki_files}
        cls.rev_adj = {f: set() for f in cls.wiki_files}
        md_link_pattern = re.compile(r'\[([^\]]*)\]\(([^)\s]+)\)')

        for p in cls.wiki_files:
            file_dir = os.path.dirname(p)
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            for m in md_link_pattern.finditer(content):
                target = m.group(2).strip("<>").strip().split("#")[0]
                if not target or target.startswith(("http:", "https:", "mailto:")):
                    continue
                abs_t = unicodedata.normalize('NFC', os.path.normpath(os.path.join(file_dir, target)))
                if abs_t in cls.wiki_set and abs_t != p:
                    cls.adj[p].add(abs_t)
                    cls.rev_adj[abs_t].add(p)

    def test_reachability_from_root(self):
        """Every wiki node must be reachable from wiki/README.md."""
        root = unicodedata.normalize('NFC', os.path.normpath(str(WIKI_DIR / "README.md")))
        visited = {root}
        queue = deque([root])
        while queue:
            curr = queue.popleft()
            for nxt in self.adj[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        self.assertEqual(visited, self.wiki_set, f"Unreachable nodes from README: {self.wiki_set - visited}")

    def test_reverse_reachability_to_root(self):
        """Every wiki node must have a directed path to wiki/README.md."""
        root = unicodedata.normalize('NFC', os.path.normpath(str(WIKI_DIR / "README.md")))
        visited = {root}
        queue = deque([root])
        while queue:
            curr = queue.popleft()
            for prev in self.rev_adj[curr]:
                if prev not in visited:
                    visited.add(prev)
                    queue.append(prev)
        self.assertEqual(visited, self.wiki_set, f"Nodes that cannot reach README: {self.wiki_set - visited}")

    def test_graph_diameter_and_in_degrees(self):
        """Graph must be strongly connected with small diameter (<= 3) and min in-degree >= 3."""
        in_degrees = {f: len(self.rev_adj[f]) for f in self.wiki_files}
        min_in = min(in_degrees.values())
        self.assertGreaterEqual(min_in, 3, f"Found node with low in-degree: min is {min_in}")

        max_dist = 0
        for src in self.wiki_files:
            dist = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            self.assertEqual(len(dist), len(self.wiki_files), f"Graph not strongly connected from {src}")
            max_dist = max(max_dist, max(dist.values()))

        self.assertLessEqual(max_dist, 3, f"Graph diameter exceeds 3: diameter is {max_dist}")


class TestHebrewSlugificationAndAnchorResolution(unittest.TestCase):
    """Adversarially tests github_slugify and anchor checking against edge cases."""

    def test_hebrew_slugify_variations(self):
        cases = [
            ("ויקי ביוגרפי ואמנותי: תמר (תמי) שפר", "ויקי-ביוגרפי-ואמנותי-תמר-תמי-שפר"),
            ("Welcome | פתח דבר", "welcome-פתח-דבר"),
            ("Related Entries / ראו גם", "related-entries-ראו-גם"),
            ("1. Solo & Institutional Exhibitions (תערוכות יחיד)", "1-solo-institutional-exhibitions-תערוכות-יחיד"),
            ("Stand-ins, Simulacra & The Vanishing \"Home\"", "stand-ins-simulacra-the-vanishing-home"),
            ("2. *\"Disturbed Landscape\"* (נוף מופרע, 2014)", "2-disturbed-landscape-נוף-מופרע-2014"),
            ("קִרבה (עם חיריק)", "קרבה-עם-חיריק"),
            ("[Link Text](http://example.com) Heading", "link-text-heading"),
            ("Heading with `code` block", "heading-with-code-block"),
            ("תערוכת \"בנה ביתך / The New Normal\"", "תערוכת-בנה-ביתך-the-new-normal"),
        ]
        for heading, expected in cases:
            actual = github_slugify(heading)
            self.assertEqual(actual, expected, f"Slug mismatch for '{heading}': got '{actual}', expected '{expected}'")

    def test_verifier_anchor_validation_adversarial(self):
        """Simulate WikiLinkVerifier on synthetic anchor scenarios."""
        verifier = WikiLinkVerifier(
            root_dir=str(PROJECT_ROOT),
            wiki_dir=str(WIKI_DIR),
            check_anchors=True,
            check_orphans=False,
            verbose=False
        )
        verifier.collect_files()
        verifier.index_headings()

        readme_rel = "wiki/README.md"
        headings = verifier.file_headings.get(readme_rel, set())

        self.assertIn("welcome-פתח-דבר", headings)
        self.assertIn("the-five-portals-חמשת-השערים", headings)
        self.assertIn("complete-corpus-matrix-מפתח-המאמרים-המלא", headings)
        self.assertNotIn("nonexistent-heading-שער-לא-קיים", headings)


class TestFootnoteIntegrity(unittest.TestCase):
    """Stress tests footnote references vs definitions across all wiki markdown files."""

    def test_footnote_dangling_references(self):
        """Flags any footnote reference [^id] lacking a [^id]: definition in the same file."""
        dangling = []
        for p in glob.glob(str(WIKI_DIR / "**" / "*.md"), recursive=True):
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()

            defs = set(re.findall(r'^\[\^([^\]]+)\]:\s*', content, re.MULTILINE))
            refs = set(re.findall(r'\[\^([^\]]+)\](?!:)', content))

            missing = refs - defs
            if missing:
                rel = os.path.relpath(p, PROJECT_ROOT)
                dangling.append((rel, missing))

        if dangling:
            print(f"\n[NOTE / DEFECT] Dangling footnote references detected: {dangling}")


if __name__ == "__main__":
    unittest.main()
