#!/usr/bin/env python3
"""
tests/test_governance.py — Project Governance & Read-Only Safety Quality Gate

Verifies Requirement R1:
1. Root governance artifacts (MISSION.md, TODO.md, PROGRESS.md, PROJECT.md) exist and are non-empty.
2. All acceptance criteria headers and thematic anchors exist across governance documents.
3. Zero outbound mutation strings or write queries exist in repository scripts.
4. Link checker tool exists and is executable.

Standard library only. Runnable via:
    python3 -m unittest discover -s tests
    python3 tests/test_governance.py
"""

import os
import sys
import re
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestGovernanceArtifacts(unittest.TestCase):
    """Verifies existence, non-emptiness, and UTF-8 validity of root governance files."""

    REQUIRED_FILES = [
        "MISSION.md",
        "TODO.md",
        "PROGRESS.md",
        "PROJECT.md",
    ]

    def test_required_governance_files_exist(self):
        """Root MISSION.md, TODO.md, PROGRESS.md, and PROJECT.md must exist."""
        for filename in self.REQUIRED_FILES:
            filepath = PROJECT_ROOT / filename
            self.assertTrue(
                filepath.exists(),
                f"Required governance file does not exist: {filepath}"
            )
            self.assertTrue(
                filepath.is_file(),
                f"Governance path is not a regular file: {filepath}"
            )

    def test_governance_files_non_empty(self):
        """Governance files must be non-empty and substantive (>200 bytes)."""
        for filename in self.REQUIRED_FILES:
            filepath = PROJECT_ROOT / filename
            if not filepath.exists():
                self.skipTest(f"Skipping size check; {filename} does not exist yet.")
            size = filepath.stat().st_size
            self.assertGreater(
                size, 0,
                f"Governance file is empty (0 bytes): {filepath}"
            )
            self.assertGreaterEqual(
                size, 200,
                f"Governance file is too short (<200 bytes) to contain required content: {filepath} ({size} bytes)"
            )

    def test_governance_files_valid_utf8(self):
        """All governance files must decode cleanly as valid UTF-8."""
        for filename in self.REQUIRED_FILES:
            filepath = PROJECT_ROOT / filename
            if not filepath.exists():
                self.skipTest(f"Skipping UTF-8 check; {filename} does not exist yet.")
            try:
                content = filepath.read_text(encoding="utf-8")
                self.assertGreater(len(content.strip()), 0)
            except UnicodeDecodeError as e:
                self.fail(f"File {filename} contains invalid UTF-8: {e}")


class TestAcceptanceCriteriaCoverage(unittest.TestCase):
    """Verifies that governance documents cover all acceptance criteria and core themes."""

    EXPECTED_ACCEPTANCE_CRITERIA_DOMAINS = [
        "Project Documentation & Read-Only Safety",
        "High-Recall Media Archive & Drop-Folder Ingestion",
        "Biographical & Artistic Wiki",
        "Delivery Plan, Precedents & Spatial Concept Report",
        "Playable 2.5D Spatial Prototype",
    ]

    def test_original_request_acceptance_criteria_present(self):
        """ORIGINAL_REQUEST.md must contain all 5 authoritative acceptance criteria headers."""
        req_file = PROJECT_ROOT / "ORIGINAL_REQUEST.md"
        self.assertTrue(req_file.exists(), "ORIGINAL_REQUEST.md not found at project root.")
        content = req_file.read_text(encoding="utf-8")

        for domain in self.EXPECTED_ACCEPTANCE_CRITERIA_DOMAINS:
            self.assertIn(
                domain, content,
                f"Authoritative acceptance criteria domain missing from ORIGINAL_REQUEST.md: '{domain}'"
            )

    def test_todo_covers_all_milestones_and_criteria(self):
        """TODO.md must cover all acceptance criteria domains and contain checklist items."""
        todo_file = PROJECT_ROOT / "TODO.md"
        if not todo_file.exists():
            self.skipTest("TODO.md does not exist yet.")

        content = todo_file.read_text(encoding="utf-8")

        # Verify that all 5 milestone requirement groups are represented
        required_milestones = ["Milestone 1", "Milestone 2", "Milestone 3", "Milestone 4", "Milestone 5"]
        for m in required_milestones:
            self.assertIn(m, content, f"TODO.md missing section for {m}")

        # Verify presence of actionable markdown checkboxes
        checkboxes = re.findall(r'- \[[ xX]\]', content)
        self.assertGreaterEqual(
            len(checkboxes), 15,
            f"TODO.md must contain granular checkboxes covering all criteria (found only {len(checkboxes)})"
        )

    def test_mission_covers_core_themes(self):
        """MISSION.md must capture the emotional vision, dual identity, materiality, and safety."""
        mission_file = PROJECT_ROOT / "MISSION.md"
        if not mission_file.exists():
            self.skipTest("MISSION.md does not exist yet.")

        content = mission_file.read_text(encoding="utf-8")

        # 1. Subject & Milestone
        self.assertTrue(
            "Tamar" in content or "Tami" in content or "תמר" in content or "תמי" in content,
            "MISSION.md must explicitly name Tamar / Tami Sheaffer."
        )
        self.assertTrue(
            "70" in content,
            "MISSION.md must explicitly reference the 70th birthday milestone."
        )

        # 2. Dual identity: systems engineer & tactile artist
        self.assertTrue(
            "Elbit" in content or "Rafael" in content or "engineer" in content.lower(),
            "MISSION.md must address Tami's defense systems engineering career."
        )
        self.assertTrue(
            "HaMidrasha" in content or "המדרשה" in content or "art" in content.lower(),
            "MISSION.md must address Tami's art education and artistic career."
        )

        # 3. Materiality & non-hollow philosophy
        self.assertTrue(
            "paper" in content.lower() or "papier-mâché" in content.lower() or "cement" in content.lower(),
            "MISSION.md must address Tami's tactile materials (paper, cement, earth)."
        )

        # 4. Strict read-only safety
        self.assertTrue(
            "read-only" in content.lower(),
            "MISSION.md must emphasize strict read-only safety."
        )

    def test_progress_contains_mandatory_tables(self):
        """PROGRESS.md must contain the 4 required live tracking tables/sections."""
        progress_file = PROJECT_ROOT / "PROGRESS.md"
        if not progress_file.exists():
            self.skipTest("PROGRESS.md does not exist yet.")

        content = progress_file.read_text(encoding="utf-8")

        mandatory_sections = [
            "Source Harvesting Statistics",
            "Access & Integrity Log",
            "Technical & Prototype Benchmarks",
            "Family Verification & Curation Register",
        ]

        for sec in mandatory_sections:
            self.assertIn(
                sec, content,
                f"PROGRESS.md missing required live tracking table/section: '{sec}'"
            )

    def test_project_spec_integrity(self):
        """PROJECT.md must document architecture, milestones, interface contracts, and layout."""
        project_file = PROJECT_ROOT / "PROJECT.md"
        if not project_file.exists():
            self.skipTest("PROJECT.md does not exist yet.")

        content = project_file.read_text(encoding="utf-8")

        self.assertIn("Architecture", content)
        self.assertIn("Milestones", content)
        self.assertIn("Interface Contracts", content)
        self.assertIn("Code Layout", content)


class TestReadOnlySafetyGuardrails(unittest.TestCase):
    """Ensures zero outbound mutation strings or write queries exist in repository scripts."""

    FORBIDDEN_MUTATION_PATTERNS = [
        # WhatsApp Baileys socket mutations
        (r'\b(?:\w+\s*\.)?sendMessage\b', "Baileys sendMessage call"),
        (r'\b(?:\w+\s*\.)?sendPresenceUpdate\b', "Baileys sendPresenceUpdate call"),
        (r'\b(?:\w+\s*\.)?groupCreate\b', "Baileys groupCreate call"),
        (r'\b(?:\w+\s*\.)?groupParticipantsUpdate\b', "Baileys groupParticipantsUpdate call"),
        (r'\b(?:\w+\s*\.)?chatModify\b', "Baileys chatModify call"),
        (r'\b(?:\w+\s*\.)?updateBlockStatus\b', "Baileys updateBlockStatus call"),
        (r'\b(?:\w+\s*\.)?relayMessage\b', "Baileys relayMessage call"),
        (r'\bcreate_poll\b', "WhatsApp create_poll call"),
        (r'\bsend_poll\b', "WhatsApp send_poll call"),

        # SQL write operations targeting personal message/chat tables
        (r'\b(?:INSERT(?:\s+OR\s+\w+)?|REPLACE)\s+INTO\s+[`"\[]?(?:zwa\w*|z_\w*|message|chat|contact|interaction)', "SQL INSERT query on personal database"),
        (r'\bUPDATE(?:\s+OR\s+\w+)?\s+[`"\[]?(?:zwa\w*|z_\w*|message|chat|contact|interaction)', "SQL UPDATE query on personal database"),
        (r'\bDELETE\s+FROM\s+[`"\[]?(?:zwa\w*|z_\w*|message|chat|contact|interaction)', "SQL DELETE query on personal database"),
        (r'\bDROP\s+TABLE\b', "SQL DROP TABLE query"),
        (r'\bALTER\s+TABLE\b', "SQL ALTER TABLE query"),

        # Outbound network transmissions
        (r'\bsmtplib\.SMTP\b', "SMTP outbound email transmission"),
        (r'\bpost_to_facebook\b', "Facebook outbound posting call"),
        (r'\bpublish_post\b', "Social media publish call"),
    ]

    SCRIPT_EXTENSIONS = {'.py', '.sh', '.bash', '.zsh', '.js', '.mjs', '.cjs', '.ts', '.mts', '.cts', '.jsx', '.tsx'}
    EXCLUDED_DIRS = {'.git', '.agents', 'node_modules', 'venv', '__pycache__', 'tests'}

    def test_scripts_zero_outbound_mutation_strings(self):
        """All executable scripts must be free of outbound mutation strings."""
        violations = []

        for root, dirs, files in os.walk(PROJECT_ROOT):
            dirs[:] = [d for d in dirs if d not in self.EXCLUDED_DIRS and not d.startswith('.')]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.SCRIPT_EXTENSIONS:
                    script_path = Path(root) / file
                    try:
                        content = script_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception as e:
                        continue

                    # 1. Line-by-line check (records exact line numbers for single-line statements)
                    for idx, line in enumerate(content.splitlines(), 1):
                        stripped = line.strip()
                        if stripped.startswith(('#', '//', '/*', '*', '"""', "'''")):
                            continue

                        for pattern, desc in self.FORBIDDEN_MUTATION_PATTERNS:
                            if re.search(pattern, line, re.IGNORECASE):
                                violations.append(
                                    f"{script_path.relative_to(PROJECT_ROOT)}:{idx} — {desc}: `{stripped}`"
                                )

                    # 2. Multi-line check (catches cross-line statement formatting and multiline SQL)
                    cleaned = re.sub(r'/\*.*?\*/', ' ', content, flags=re.DOTALL)
                    uncommented = "\n".join(l for l in cleaned.splitlines() if not l.strip().startswith(('#', '//', '/*', '*', '"""', "'''")))
                    normalized = " ".join(uncommented.split())
                    for pattern, desc in self.FORBIDDEN_MUTATION_PATTERNS:
                        if re.search(pattern, normalized, re.IGNORECASE):
                            if not any(desc in v and str(script_path.relative_to(PROJECT_ROOT)) in v for v in violations):
                                violations.append(
                                    f"{script_path.relative_to(PROJECT_ROOT)}:multiline — {desc} (cross-line pattern detected)"
                                )

        self.assertEqual(
            len(violations), 0,
            f"Detected forbidden outbound mutation strings in repository scripts:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )


class TestWikiLinkCheckerTool(unittest.TestCase):
    """Verifies that tools/verify_wiki_links.py is present and executable."""

    def test_verify_wiki_links_tool_exists(self):
        """tools/verify_wiki_links.py must exist and be non-empty."""
        tool_path = PROJECT_ROOT / "tools" / "verify_wiki_links.py"
        self.assertTrue(
            tool_path.exists(),
            f"Wiki link verification tool does not exist: {tool_path}"
        )
        self.assertGreater(
            tool_path.stat().st_size, 0,
            f"Wiki link verification tool is empty (0 bytes): {tool_path}"
        )

    def test_verify_wiki_links_executable(self):
        """tools/verify_wiki_links.py must exit 0 when invoked with --help."""
        tool_path = PROJECT_ROOT / "tools" / "verify_wiki_links.py"
        if not tool_path.exists():
            self.skipTest("tools/verify_wiki_links.py does not exist yet.")

        res = subprocess.run(
            [sys.executable, str(tool_path), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.assertEqual(
            res.returncode, 0,
            f"tools/verify_wiki_links.py --help failed with code {res.returncode}:\n{res.stderr}"
        )
        self.assertIn("verify", res.stdout.lower())


if __name__ == '__main__':
    unittest.main()
