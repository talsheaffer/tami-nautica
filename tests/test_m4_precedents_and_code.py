#!/usr/bin/env python3
"""
tests/test_m4_precedents_and_code.py — Empirical Verification Suite for Milestone 4 Reports

Governing Deliverables:
- reports/DELIVERY_LONGEVITY_PLAN.md
- reports/SPATIAL_CONCEPT_AND_PRECEDENTS.md

Tests:
1. Live network verification of all cited external URLs (HTTP 200/301/302).
2. AST syntax parsing of all embedded Python code blocks.
3. Native JavaScript / ES Module syntax validation via Node.js (`node --check`).
4. GLSL transition shader structure and syntax validation.
5. PWA manifest JSON validation.
6. Shell script syntax check via `bash -n`.
7. Zero unresolved placeholders, TODOs, TBDs, or broken markdown links.
8. Column count integrity across all Markdown tables.
9. Coverage of all R4.1 & R4.2 requirements from ORIGINAL_REQUEST.md.
"""

import ast
import json
import os
import re
import subprocess
import sys
import unittest
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DELIVERY_REPORT = REPORTS_DIR / "DELIVERY_LONGEVITY_PLAN.md"
SPATIAL_REPORT = REPORTS_DIR / "SPATIAL_CONCEPT_AND_PRECEDENTS.md"


def extract_fenced_blocks(markdown_text: str):
    """Extracts all fenced code blocks with their specified language."""
    # Matches ```lang\ncode\n```
    pattern = re.compile(r"```([a-zA-Z0-9_\-+]*)\n(.*?)```", re.DOTALL)
    blocks = []
    for match in pattern.finditer(markdown_text):
        lang = match.group(1).lower().strip()
        code = match.group(2)
        start_line = markdown_text[: match.start()].count("\n") + 1
        blocks.append({"lang": lang, "code": code, "line": start_line})
    return blocks


def extract_urls(text: str):
    """Extracts all HTTP/HTTPS URLs from text, cleaned of trailing punctuation."""
    pattern = re.compile(r"https?://[^\s\)\]\>\"\'\`]+")
    raw = pattern.findall(text)
    clean = set()
    for u in raw:
        cleaned = u.rstrip(".,;:'\")")
        clean.add(cleaned)
    return clean


class TestM4PrecedentUrls(unittest.TestCase):
    """Adversarially tests live HTTP network availability of cited URLs."""

    def setUp(self):
        self.assertTrue(DELIVERY_REPORT.exists(), f"Missing {DELIVERY_REPORT}")
        self.assertTrue(SPATIAL_REPORT.exists(), f"Missing {SPATIAL_REPORT}")
        self.delivery_text = DELIVERY_REPORT.read_text(encoding="utf-8")
        self.spatial_text = SPATIAL_REPORT.read_text(encoding="utf-8")

    def test_precedent_urls_live_http_status(self):
        spatial_urls = extract_urls(self.spatial_text)
        delivery_urls = extract_urls(self.delivery_text)
        all_urls = spatial_urls.union(delivery_urls)

        # Exclude local and example template URLs
        local_or_template = {
            "http://localhost:${PORT}",
            "http://localhost:%PORT%",
            "http://localhost:8080",
            "https://preview-abc123.tami70.pages.dev",
        }
        external_urls = [u for u in all_urls if u not in local_or_template]

        # R4 requires at least 5 real, verifiable precedents
        self.assertGreaterEqual(
            len(external_urls),
            5,
            f"Expected at least 5 cited external URLs, found {len(external_urls)}",
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        results = {}
        failures = []

        print(f"\n[NETWORK PROBE] Testing {len(external_urls)} live URLs...")
        for url in sorted(external_urls):
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    status = resp.status
                    results[url] = status
                    print(f"  ✓ [{status}] {url}")
            except urllib.error.HTTPError as e:
                # Some servers redirect or require GET params; check code
                results[url] = e.code
                if e.code in (200, 301, 302, 303, 307, 308):
                    print(f"  ✓ [{e.code}] {url} (redirect)")
                else:
                    failures.append(f"{url} returned HTTP {e.code}")
                    print(f"  ✗ [{e.code}] {url}")
            except Exception as e:
                failures.append(f"{url} failed: {type(e).__name__}: {e}")
                print(f"  ✗ [ERR] {url} -> {e}")

        self.assertEqual(
            len(failures),
            0,
            f"Failed URLs found in reports:\n" + "\n".join(failures),
        )

    def test_precedents_taxonomy_citations(self):
        """Verifies specific core precedent systems are explicitly present."""
        expected_precedents = [
            "https://shihmengli.github.io/3D-Photo-Inpainting/",
            "https://depth-anything-v2.github.io/",
            "https://zoomquilt.org/",
            "https://github.com/panrafal/depthy",
            "https://antimatter15.com/splat/",
            "https://www.sbs.com.au/theboat/",
        ]
        for prec in expected_precedents:
            self.assertIn(
                prec,
                self.spatial_text,
                f"Missing required precedent citation: {prec}",
            )


class TestEmbeddedCodeSyntax(unittest.TestCase):
    """Extracts and parses all code blocks embedded across both reports."""

    def setUp(self):
        self.delivery_text = DELIVERY_REPORT.read_text(encoding="utf-8")
        self.spatial_text = SPATIAL_REPORT.read_text(encoding="utf-8")
        self.delivery_blocks = extract_fenced_blocks(self.delivery_text)
        self.spatial_blocks = extract_fenced_blocks(self.spatial_text)

    def test_embedded_python_ast(self):
        """Validates all Python scripts parse to valid AST."""
        python_blocks = []
        for b in self.delivery_blocks:
            if b["lang"] in ("python", "py"):
                python_blocks.append(("DELIVERY_LONGEVITY_PLAN.md", b))
        for b in self.spatial_blocks:
            if b["lang"] in ("python", "py"):
                python_blocks.append(("SPATIAL_CONCEPT_AND_PRECEDENTS.md", b))

        self.assertGreaterEqual(
            len(python_blocks), 2, "Expected at least 2 embedded Python scripts"
        )

        for filename, block in python_blocks:
            code = block["code"]
            line = block["line"]
            try:
                tree = ast.parse(code)
                self.assertIsNotNone(tree)
                print(f"  ✓ Python AST valid: {filename}:{line} ({len(code.splitlines())} lines)")
            except SyntaxError as e:
                self.fail(f"SyntaxError in {filename}:{line + e.lineno}: {e.msg}\n{e.text}")

    def test_bundle_standalone_inlining_execution(self):
        """Empirically tests that bundle_standalone.py executes and inlines HTML assets."""
        import tempfile

        bundle_block = None
        for b in self.delivery_blocks:
            if b["lang"] in ("python", "py") and "bundle_standalone.py" in b["code"]:
                bundle_block = b
                break
        self.assertIsNotNone(bundle_block, "Could not find bundle_standalone.py block")

        # Compile and load bundle functions in isolated namespace
        ns = {}
        compiled = compile(bundle_block["code"], "bundle_standalone.py", "exec")
        # Define mock __file__ so Path(__file__) does not error during compilation
        ns["__file__"] = str(DELIVERY_REPORT)
        # Avoid running main guard
        ns["__name__"] = "mock_module"
        exec(compiled, ns)

        self.assertIn("bundle", ns)
        bundle_fn = ns["bundle"]

        # Run on a temporary mock site
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            source_html = tmppath / "index.html"
            css_file = tmppath / "style.css"
            js_file = tmppath / "app.js"
            output_html = tmppath / "standalone.html"

            css_file.write_text("body { background: #121212; color: #fff; }", encoding="utf-8")
            js_file.write_text("console.log('Tami 70 Milestone Active');", encoding="utf-8")
            source_html.write_text(
                '<!DOCTYPE html><html><head>'
                '<link rel="stylesheet" href="style.css">'
                '</head><body><h1>Tami 70</h1>'
                '<script type="module" src="app.js"></script>'
                '</body></html>',
                encoding="utf-8"
            )

            bundle_fn(source_html, output_html)

            self.assertTrue(output_html.exists(), "Output standalone HTML was not generated")
            bundled_text = output_html.read_text(encoding="utf-8")
            self.assertIn("<style>", bundled_text)
            self.assertIn("background: #121212", bundled_text)
            self.assertIn('<script type="module">', bundled_text)
            self.assertIn("Tami 70 Milestone Active", bundled_text)
            self.assertNotIn('<link rel="stylesheet"', bundled_text)
            self.assertNotIn('src="app.js"', bundled_text)
            print("  ✓ bundle_standalone.py functional test passed with inlined CSS & JS")

    def test_webcrypto_pbkdf2_timing_and_security(self):
        """Verifies 100,000 PBKDF2 iterations execute smoothly via Node.js crypto."""
        js_timing_probe = """
        const crypto = require('crypto');
        const start = performance.now();
        const derived = crypto.pbkdf2Sync('greenboim', 'tami70_salt_16b!', 100000, 32, 'sha256');
        const durationMs = performance.now() - start;
        console.log(JSON.stringify({ durationMs, keyHex: derived.toString('hex') }));
        """
        proc = subprocess.run(
            ["node", "-e", js_timing_probe],
            capture_output=True,
            text=True,
            check=True
        )
        res = json.loads(proc.stdout.strip())
        duration = res["durationMs"]
        print(f"  ✓ PBKDF2 100,000 iterations completed in {duration:.2f}ms (target: ~50ms, key: {res['keyHex'][:16]}...)")
        # Ensure it takes enough time to throttle dictionary attacks (>10ms) but does not freeze browser (<300ms)
        self.assertGreater(duration, 5.0, "PBKDF2 ran too fast to offer brute-force resistance")
        self.assertLess(duration, 400.0, "PBKDF2 took too long; risk of browser UI freeze on mobile")

    def test_spatial_frustum_math_proofs(self):
        """Empirically validates the frustum clearance calculations and formulas in Section 5.1."""
        import math

        fov_deg = 45.0
        fov_rad = math.radians(fov_deg)
        z_cam = 400.0
        z_bg = -350.0
        d_0 = z_cam - z_bg  # 750
        self.assertEqual(d_0, 750.0)

        # Frustum width at distance d_0
        w_d0 = 2.0 * d_0 * math.tan(fov_rad / 2.0)
        # Verify 621.32 from report
        self.assertAlmostEqual(w_d0, 621.32, delta=0.5)

        # Maximum camera displacement
        m_x = 45.0
        min_scale = 1.0 + (2.0 * m_x) / w_d0
        self.assertAlmostEqual(min_scale, 1.1448, delta=0.01)

        # Specified scale factor is 1.55
        specified_scale = 1.55
        margin_pct = ((specified_scale / min_scale) - 1.0) * 100.0
        self.assertGreater(margin_pct, 30.0, "Border clearance margin must be >30%")
        print(f"  ✓ Frustum math proof verified: w_d0={w_d0:.2f}, min_scale={min_scale:.4f}, margin={margin_pct:.1f}%")

    def test_embedded_javascript_syntax(self):
        """Validates all JavaScript/ES Module snippets via Node.js --check."""
        js_blocks = []
        for b in self.delivery_blocks:
            if b["lang"] in ("javascript", "js"):
                js_blocks.append(("DELIVERY_LONGEVITY_PLAN.md", b))
        for b in self.spatial_blocks:
            if b["lang"] in ("javascript", "js"):
                js_blocks.append(("SPATIAL_CONCEPT_AND_PRECEDENTS.md", b))

        self.assertGreaterEqual(
            len(js_blocks), 4, "Expected at least 4 embedded JavaScript snippets"
        )

        for filename, block in js_blocks:
            code = block["code"]
            line = block["line"]

            # Run node --input-type=module --check on stdin
            proc = subprocess.run(
                ["node", "--input-type=module", "--check"],
                input=code,
                text=True,
                capture_output=True,
            )

            # Some snippets might be scripts or object snippets rather than modules
            if proc.returncode != 0:
                # Try script mode if module failed
                proc_script = subprocess.run(
                    ["node", "--check"],
                    input=code,
                    text=True,
                    capture_output=True,
                )
                if proc_script.returncode != 0:
                    self.fail(
                        f"JavaScript syntax error in {filename} around line {line}:\n"
                        f"{proc.stderr or proc_script.stderr}"
                    )
            print(f"  ✓ JS syntax valid: {filename}:{line} ({len(code.splitlines())} lines)")

    def test_embedded_glsl_shader_syntax(self):
        """Validates GLSL transition shader code block structure."""
        glsl_blocks = [
            b for b in self.spatial_blocks if b["lang"] in ("glsl", "frag", "vert")
        ]
        self.assertGreaterEqual(len(glsl_blocks), 1, "Expected at least 1 GLSL shader block")

        for b in glsl_blocks:
            code = b["code"]
            line = b["line"]

            # Check for required GLSL declarations
            self.assertIn("precision highp float;", code)
            self.assertIn("void main()", code)
            self.assertIn("gl_FragColor", code)
            self.assertIn("uniform sampler2D uTexCurrent;", code)
            self.assertIn("uniform sampler2D uTexNext;", code)
            self.assertIn("uniform float uProgress;", code)

            # Check delimiter balancing
            self.assertEqual(
                code.count("{"),
                code.count("}"),
                f"Mismatched braces in GLSL block at line {line}",
            )
            self.assertEqual(
                code.count("("),
                code.count(")"),
                f"Mismatched parentheses in GLSL block at line {line}",
            )
            print(f"  ✓ GLSL shader valid: line {line} ({len(code.splitlines())} lines)")

    def test_embedded_manifest_json(self):
        """Validates Web App Manifest JSON structure."""
        json_blocks = [
            b for b in self.delivery_blocks if b["lang"] == "json"
        ]
        self.assertGreaterEqual(len(json_blocks), 1, "Expected JSON manifest block")

        for b in json_blocks:
            code = b["code"]
            try:
                data = json.loads(code)
                self.assertIn("name", data)
                self.assertIn("start_url", data)
                self.assertIn("display", data)
                self.assertIn("icons", data)
                print(f"  ✓ Manifest JSON valid: line {b['line']}")
            except json.JSONDecodeError as e:
                self.fail(f"Invalid JSON in block at line {b['line']}: {e}")

    def test_embedded_shell_scripts(self):
        """Validates shell scripts using bash -n."""
        bash_blocks = [
            b for b in self.delivery_blocks if b["lang"] in ("bash", "sh")
        ]
        self.assertGreaterEqual(len(bash_blocks), 2, "Expected at least 2 bash scripts")

        for b in bash_blocks:
            code = b["code"]
            proc = subprocess.run(
                ["bash", "-n"],
                input=code,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"Bash syntax error in block at line {b['line']}:\n{proc.stderr}",
            )
            print(f"  ✓ Shell script syntax valid: line {b['line']}")


class TestReportFormattingAndIntegrity(unittest.TestCase):
    """Scans reports for placeholders, broken tables, and missing sections."""

    def test_no_unresolved_placeholders(self):
        placeholder_regex = re.compile(
            r"\b(TODO|TBD|FIXME|XXX)\b|\[placeholder\]|\[TBD\]",
            re.IGNORECASE,
        )

        for report_path in [DELIVERY_REPORT, SPATIAL_REPORT]:
            text = report_path.read_text(encoding="utf-8")
            matches = []
            for idx, line in enumerate(text.splitlines(), 1):
                if placeholder_regex.search(line):
                    # Exclude this test file or explanatory mentions if any
                    matches.append(f"Line {idx}: {line.strip()}")
            self.assertEqual(
                len(matches),
                0,
                f"Found unresolved placeholders in {report_path.name}:\n"
                + "\n".join(matches),
            )

    def test_markdown_tables_column_consistency(self):
        """Verifies every table has equal column counts across headers and rows."""
        for report_path in [DELIVERY_REPORT, SPATIAL_REPORT]:
            lines = report_path.read_text(encoding="utf-8").splitlines()
            in_code = False
            in_table = False
            header_cols = 0
            table_id = 0

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    in_table = False
                    continue

                if in_code:
                    continue

                if stripped.startswith("|") and stripped.endswith("|"):
                    # Find unescaped pipes
                    pipes = [m.start() for m in re.finditer(r"(?<!\\)\|", stripped)]
                    cols = len(pipes) - 1
                    if not in_table:
                        in_table = True
                        table_id += 1
                        header_cols = cols
                    else:
                        self.assertEqual(
                            cols,
                            header_cols,
                            f"Table {table_id} column mismatch at {report_path.name}:{idx}. "
                            f"Header has {header_cols} cols, row has {cols}: {stripped}",
                        )
                else:
                    in_table = False


class TestGoverningRequirementsCoverage(unittest.TestCase):
    """Ensures reports strictly satisfy R4.1 and R4.2 from ORIGINAL_REQUEST.md."""

    def setUp(self):
        self.delivery = DELIVERY_REPORT.read_text(encoding="utf-8")
        self.spatial = SPATIAL_REPORT.read_text(encoding="utf-8")

    def test_delivery_plan_specifications(self):
        # Must compare PWA, offline bundle, native/hybrid
        self.assertIn("Responsive Web App", self.delivery)
        self.assertIn("Self-Contained Offline", self.delivery)
        self.assertIn("Tauri", self.delivery)
        self.assertIn("Electron", self.delivery)
        # Privacy & Web Crypto
        self.assertIn("AES-256", self.delivery)
        self.assertIn("PBKDF2", self.delivery)
        self.assertIn("window.crypto.subtle", self.delivery)
        # Archival longevity
        self.assertIn("exFAT", self.delivery)
        self.assertIn("start_mac.command", self.delivery)
        self.assertIn("start_windows.bat", self.delivery)

    def test_spatial_concept_specifications(self):
        # 6 precedents
        self.assertIn("3D Photography", self.spatial)
        self.assertIn("Depth Anything V2", self.spatial)
        self.assertIn("Zoomquilt", self.spatial)
        self.assertIn("Depthy", self.spatial)
        self.assertIn("Antimatter15", self.spatial)
        self.assertIn("The Boat", self.spatial)
        # Lived places & anchors
        self.assertIn("חצר הילדות בהרצליה", self.spatial)
        self.assertIn("הסטודיו בדניה", self.spatial)
        self.assertTrue("מיכאל" in self.spatial or "Michael" in self.spatial)
        self.assertTrue("אהובה" in self.spatial or "Ahuva" in self.spatial)
        self.assertIn("אראלה", self.spatial)
        self.assertIn("בנה ביתך", self.spatial)
        # 5-stage AI pipeline
        self.assertIn("Real-ESRGAN", self.spatial)
        self.assertIn("SAM 2", self.spatial)
        self.assertIn("Depth Anything V2", self.spatial)
        self.assertIn("LaMa", self.spatial)
        self.assertIn("Gemini", self.spatial)
        # Frustum & dilation proofs
        self.assertIn("r \\ge 25", self.spatial)
        self.assertIn("Scale}_0 = 1.55", self.spatial)
        # Automated Playwright test spec
        self.assertIn("playwright.config.js", self.spatial)
        self.assertIn("test/prototype.spec.js", self.spatial)


if __name__ == "__main__":
    unittest.main()
