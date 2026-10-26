#!/usr/bin/env python3
"""
tools/verify_wiki_links.py — Wiki Link & Integrity Verification Engine

Verifies internal links, anchor fragments, image targets, Hebrew UTF-8 paths,
and detects orphan pages across the Tamar Sheaffer 70th-Birthday project wiki.

Standard library only. No external dependencies required.
"""

import sys
import os
import re
import argparse
import json
import urllib.parse
import unicodedata
from pathlib import Path


def github_slugify(heading_text: str) -> str:
    """
    Generate GitHub-flavored Markdown slug from heading text,
    supporting Hebrew characters, numbers, and stripping punctuation.
    """
    # Strip markdown links e.g. [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', heading_text)
    # Strip formatting: bold, italic, code, strikethrough
    text = re.sub(r'[*_`~]', '', text)
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Downcase
    text = text.lower().strip()
    
    # Keep unicode letters, digits, spaces, hyphens, and underscores
    # In Python regex \w includes unicode word characters (including Hebrew)
    text = re.sub(r'[^\w\s\-]', '', text)
    # Convert spaces to hyphens
    text = re.sub(r'\s+', '-', text)
    # Return NFC normalized string
    return unicodedata.normalize('NFC', text)


class WikiLinkVerifier:
    def __init__(self, root_dir: str, wiki_dir: str, check_anchors: bool = True,
                 check_orphans: bool = True, strict_orphans: bool = True,
                 check_images: bool = True, check_root_docs: bool = True,
                 verbose: bool = False):
        self.root_dir = os.path.abspath(root_dir)
        self.wiki_dir = os.path.abspath(wiki_dir)
        self.check_anchors = check_anchors
        self.check_orphans = check_orphans
        self.strict_orphans = strict_orphans
        self.check_images = check_images
        self.check_root_docs = check_root_docs
        self.verbose = verbose

        self.md_files = []          # Relative paths to root_dir
        self.wiki_files = set()     # Subset of md_files located under wiki_dir
        self.file_headings = {}     # rel_path -> set of valid anchor slugs
        self.graph = {}             # rel_path -> set of linked rel_paths
        self.errors = []
        self.warnings = []

        self.stats = {
            'total_files_scanned': 0,
            'wiki_files_count': 0,
            'root_files_count': 0,
            'total_links_found': 0,
            'internal_links_checked': 0,
            'external_links_skipped': 0,
            'image_links_checked': 0,
            'broken_links_count': 0,
            'empty_files_count': 0,
            'broken_anchors_count': 0,
            'orphan_pages_count': 0,
        }

    def collect_files(self):
        """Discover all markdown files in wiki_dir and optionally root_dir."""
        # 1. Collect wiki files
        if os.path.exists(self.wiki_dir):
            for root, dirs, files in os.walk(self.wiki_dir):
                # Filter out hidden directories and common vendor paths
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'venv', '__pycache__')]
                for file in files:
                    if file.endswith('.md'):
                        abs_p = os.path.join(root, file)
                        rel_p = unicodedata.normalize('NFC', os.path.relpath(abs_p, self.root_dir))
                        self.md_files.append(rel_p)
                        self.wiki_files.add(rel_p)

        # 2. Collect root docs if requested
        if self.check_root_docs and os.path.exists(self.root_dir):
            for item in os.listdir(self.root_dir):
                if item.endswith('.md') and not item.startswith('.'):
                    abs_p = os.path.join(self.root_dir, item)
                    if os.path.isfile(abs_p):
                        rel_p = unicodedata.normalize('NFC', item)
                        if rel_p not in self.md_files:
                            self.md_files.append(rel_p)

        self.stats['total_files_scanned'] = len(self.md_files)
        self.stats['wiki_files_count'] = len(self.wiki_files)
        self.stats['root_files_count'] = len(self.md_files) - len(self.wiki_files)
        self.graph = {f: set() for f in self.md_files}

    def index_headings(self):
        """Pass 1: Parse headings and explicit anchors in all markdown files."""
        for rel_file in self.md_files:
            abs_file = os.path.join(self.root_dir, rel_file)
            headings = set()
            slug_counts = {}

            try:
                with open(abs_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except Exception as e:
                self.errors.append({
                    'type': 'FILE_READ_ERROR',
                    'source_file': rel_file,
                    'line': 0,
                    'target': '',
                    'message': f'Cannot read file {rel_file}: {e}'
                })
                continue

            # 1. Parse markdown headings outside fenced code blocks
            non_code_lines = []
            fence_char = None
            for line in content.splitlines():
                stripped = line.strip()
                if fence_char is None:
                    if stripped.startswith('```'):
                        fence_char = '`'
                        continue
                    elif stripped.startswith('~~~'):
                        fence_char = '~'
                        continue
                    non_code_lines.append(line)
                    if stripped.startswith('#'):
                        # Strip leading # and whitespace
                        heading_text = re.sub(r'^#+\s*', '', stripped).strip()
                        if heading_text:
                            base_slug = github_slugify(heading_text)
                            if base_slug:
                                if base_slug in slug_counts:
                                    slug_counts[base_slug] += 1
                                    slug = f'{base_slug}-{slug_counts[base_slug]}'
                                else:
                                    slug_counts[base_slug] = 0
                                    slug = base_slug
                                headings.add(slug)
                                # Allow collapsed hyphens
                                collapsed = re.sub(r'-+', '-', slug)
                                headings.add(collapsed)
                else:
                    if (fence_char == '`' and stripped.startswith('```')) or (fence_char == '~' and stripped.startswith('~~~')):
                        fence_char = None
                    continue

            # 2. Parse HTML anchors <a id="..." name="..."> and <span/div id="..."> outside code fences
            clean_content = '\n'.join(non_code_lines)
            html_anchors = re.findall(
                r'<(?:a|span|div)[^>]*(?:id|name)=[\'"]([^\'"]+)[\'"]', clean_content, re.IGNORECASE
            )
            for a in html_anchors:
                clean_anchor = unicodedata.normalize('NFC', a.strip().lower())
                headings.add(clean_anchor)

            self.file_headings[rel_file] = headings

    def check_links(self):
        """Pass 2: Extract and validate all links in markdown files."""
        md_link_pattern = re.compile(
            r'(!?)\[([^\]]*)\]\((<[^>]+>|[^)\s]+)(?:\s+[\'"][^\'"]*[\'"])?\)'
        )
        html_link_pattern = re.compile(
            r'<(?:a|img)[^>]*(?:href|src)=[\'"]([^\'"]+)[\'"]', re.IGNORECASE
        )

        for rel_file in self.md_files:
            abs_file = os.path.join(self.root_dir, rel_file)
            file_dir = os.path.dirname(abs_file)

            try:
                with open(abs_file, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
            except Exception:
                continue

            fence_char = None
            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                # Toggle fenced code blocks (backticks or tildes)
                if fence_char is None:
                    if stripped.startswith('```'):
                        fence_char = '`'
                        continue
                    elif stripped.startswith('~~~'):
                        fence_char = '~'
                        continue
                else:
                    if (fence_char == '`' and stripped.startswith('```')) or (fence_char == '~' and stripped.startswith('~~~')):
                        fence_char = None
                    continue

                # Strip inline code spans to prevent false links in doc examples
                clean_line = re.sub(r'`[^`]*?`', '', line)

                # Extract markdown links: returns list of (is_img, text, target)
                md_matches = md_link_pattern.findall(clean_line)
                # Extract html href/src
                html_matches = html_link_pattern.findall(clean_line)

                # Combine all targets to verify
                items_to_check = []
                for is_img, text, target in md_matches:
                    items_to_check.append((target, bool(is_img), text))
                for target in html_matches:
                    items_to_check.append((target, False, 'HTML link'))

                for raw_target, is_img, link_text in items_to_check:
                    target = raw_target.strip('<>').strip()
                    if not target:
                        self.warnings.append({
                            'type': 'EMPTY_LINK_TARGET',
                            'source_file': rel_file,
                            'line': line_num,
                            'target': raw_target,
                            'message': f'Empty link target found in {rel_file}:{line_num}'
                        })
                        continue

                    self.stats['total_links_found'] += 1

                    # Check for external protocols
                    if re.match(r'^(?:https?|mailto|ftp|tel|data):', target, re.IGNORECASE):
                        self.stats['external_links_skipped'] += 1
                        if self.verbose:
                            print(f"[EXT] {rel_file}:{line_num} -> {target}")
                        continue

                    # Handle image links
                    if is_img:
                        self.stats['image_links_checked'] += 1
                        if not self.check_images:
                            continue

                    self.stats['internal_links_checked'] += 1

                    # Parse URL path and fragment
                    parsed = urllib.parse.urlparse(target)
                    unquoted_path = urllib.parse.unquote(parsed.path)
                    fragment = urllib.parse.unquote(parsed.fragment)

                    # Resolve target path
                    if not unquoted_path:
                        # Same-file anchor link (#heading)
                        target_rel = rel_file
                    elif unquoted_path.startswith('/'):
                        # Absolute from project root
                        target_abs = os.path.join(self.root_dir, unquoted_path.lstrip('/'))
                        target_rel = unicodedata.normalize('NFC', os.path.relpath(target_abs, self.root_dir))
                    else:
                        # Relative to directory of current file
                        target_abs = os.path.join(file_dir, unquoted_path)
                        target_rel = unicodedata.normalize('NFC', os.path.relpath(target_abs, self.root_dir))

                    # Target file existence check
                    target_abs = os.path.join(self.root_dir, target_rel)
                    if not os.path.exists(target_abs):
                        # Try adding .md if omitted (e.g. 'bio' -> 'bio.md')
                        candidate_md = target_abs.rstrip('/') + '.md'
                        if os.path.exists(candidate_md):
                            target_abs = candidate_md
                            target_rel = unicodedata.normalize('NFC', os.path.relpath(target_abs, self.root_dir))
                        else:
                            self.stats['broken_links_count'] += 1
                            self.errors.append({
                                'type': 'TARGET_NOT_FOUND',
                                'source_file': rel_file,
                                'line': line_num,
                                'target': raw_target,
                                'resolved_path': target_rel,
                                'message': f'Target file does not exist: {target_rel}'
                            })
                            continue
                    elif os.path.isdir(target_abs):
                        # Try index.md or README.md inside directory
                        if os.path.exists(os.path.join(target_abs, 'index.md')):
                            target_abs = os.path.join(target_abs, 'index.md')
                            target_rel = unicodedata.normalize('NFC', os.path.relpath(target_abs, self.root_dir))
                        elif os.path.exists(os.path.join(target_abs, 'README.md')):
                            target_abs = os.path.join(target_abs, 'README.md')
                            target_rel = unicodedata.normalize('NFC', os.path.relpath(target_abs, self.root_dir))
                        else:
                            self.stats['broken_links_count'] += 1
                            self.errors.append({
                                'type': 'TARGET_NOT_FOUND',
                                'source_file': rel_file,
                                'line': line_num,
                                'target': raw_target,
                                'resolved_path': target_rel,
                                'message': f'Directory missing index.md: {target_rel}'
                            })
                            continue

                    # Check for empty file (0 bytes)
                    if os.path.isfile(target_abs) and os.path.getsize(target_abs) == 0:
                        self.stats['empty_files_count'] += 1
                        self.errors.append({
                            'type': 'TARGET_EMPTY',
                            'source_file': rel_file,
                            'line': line_num,
                            'target': raw_target,
                            'resolved_path': target_rel,
                            'message': f'Target file exists but is empty (0 bytes): {target_rel}'
                        })

                    # Add edge to link graph if target is a markdown document
                    if target_rel in self.md_files:
                        self.graph[rel_file].add(target_rel)

                    # Validate anchor fragment
                    if self.check_anchors and fragment:
                        target_headings = self.file_headings.get(target_rel, set())
                        norm_frag = unicodedata.normalize('NFC', fragment.lower().strip())
                        collapsed_frag = re.sub(r'-+', '-', norm_frag)

                        if norm_frag not in target_headings and collapsed_frag not in target_headings:
                            self.stats['broken_anchors_count'] += 1
                            self.errors.append({
                                'type': 'ANCHOR_NOT_FOUND',
                                'source_file': rel_file,
                                'line': line_num,
                                'target': raw_target,
                                'resolved_path': target_rel,
                                'fragment': fragment,
                                'message': f'Anchor "#{fragment}" not found in {target_rel}'
                            })
                        elif self.verbose:
                            print(f"[ANCHOR OK] {rel_file}:{line_num} -> {target_rel}#{fragment}")
                    elif self.verbose:
                        print(f"[FILE OK] {rel_file}:{line_num} -> {target_rel}")

    def check_for_orphans(self):
        """Pass 3: Detect orphan wiki pages not linked from anywhere."""
        if not self.check_orphans or not self.wiki_files:
            return

        in_degrees = {f: 0 for f in self.wiki_files}
        for src, targets in self.graph.items():
            for tgt in targets:
                if tgt in in_degrees and tgt != src:
                    in_degrees[tgt] += 1

        # Designated valid entry points: only main wiki root documents
        wiki_rel = unicodedata.normalize('NFC', os.path.relpath(self.wiki_dir, self.root_dir))
        entry_points = {
            unicodedata.normalize('NFC', os.path.normpath(os.path.join(wiki_rel, 'README.md'))),
            unicodedata.normalize('NFC', os.path.normpath(os.path.join(wiki_rel, 'index.md'))),
        }

        # Check in-degree
        for f, count in in_degrees.items():
            if count == 0 and f not in entry_points:
                self.stats['orphan_pages_count'] += 1
                error_item = {
                    'type': 'ORPHAN_PAGE',
                    'source_file': f,
                    'line': 0,
                    'target': '',
                    'resolved_path': f,
                    'message': f'Orphan wiki page detected (0 inbound links): {f}'
                }
                if self.strict_orphans:
                    self.errors.append(error_item)
                else:
                    self.warnings.append(error_item)

        # Reachability BFS from entry points
        visited = set()
        queue = [ep for ep in entry_points if ep in self.wiki_files]
        visited.update(queue)

        # Also add any wiki page linked from root documents
        for src in self.md_files:
            if src not in self.wiki_files:
                for tgt in self.graph.get(src, []):
                    if tgt in self.wiki_files and tgt not in visited:
                        visited.add(tgt)
                        queue.append(tgt)

        while queue:
            curr = queue.pop(0)
            for nxt in self.graph.get(curr, []):
                if nxt in self.wiki_files and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        unreachable = self.wiki_files - visited
        for unreach in unreachable:
            # Avoid duplicate error if already reported under zero in-degree
            if in_degrees.get(unreach, 0) > 0:
                self.stats['orphan_pages_count'] += 1
                error_item = {
                    'type': 'UNREACHABLE_PAGE',
                    'source_file': unreach,
                    'line': 0,
                    'target': '',
                    'resolved_path': unreach,
                    'message': f'Wiki page is in a disconnected island/cycle (unreachable from README.md): {unreach}'
                }
                if self.strict_orphans:
                    self.errors.append(error_item)
                else:
                    self.warnings.append(error_item)

    def run(self) -> int:
        """Execute full verification pipeline and return exit code."""
        self.collect_files()
        self.index_headings()
        self.check_links()
        self.check_for_orphans()
        return 1 if self.errors else 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify Markdown internal links, anchors, Hebrew UTF-8 paths, and orphan pages."
    )
    parser.add_argument(
        '--root-dir', default=os.getcwd(),
        help="Repository root directory (defaults to current working directory)."
    )
    parser.add_argument(
        '--wiki-dir', default='wiki',
        help="Wiki directory relative to root (defaults to 'wiki')."
    )
    parser.add_argument(
        '--check-anchors', action=argparse.BooleanOptionalAction, default=True,
        help="Verify #heading anchor fragments match target headings."
    )
    parser.add_argument(
        '--check-orphans', action=argparse.BooleanOptionalAction, default=True,
        help="Verify all wiki pages have inbound links and are reachable."
    )
    parser.add_argument(
        '--strict-orphans', action=argparse.BooleanOptionalAction, default=True,
        help="Treat orphan pages as fatal errors (exit code 1)."
    )
    parser.add_argument(
        '--check-images', action=argparse.BooleanOptionalAction, default=True,
        help="Verify linked image files exist and are non-empty."
    )
    parser.add_argument(
        '--check-root-docs', action=argparse.BooleanOptionalAction, default=True,
        help="Include root .md files in link checks."
    )
    parser.add_argument(
        '--json', action='store_true', dest='json_output',
        help="Output results as structured JSON."
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Show verbose output of every inspected link."
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    wiki_dir = os.path.join(root_dir, args.wiki_dir) if not os.path.isabs(args.wiki_dir) else args.wiki_dir

    verifier = WikiLinkVerifier(
        root_dir=root_dir,
        wiki_dir=wiki_dir,
        check_anchors=args.check_anchors,
        check_orphans=args.check_orphans,
        strict_orphans=args.strict_orphans,
        check_images=args.check_images,
        check_root_docs=args.check_root_docs,
        verbose=args.verbose
    )

    exit_code = verifier.run()

    if args.json_output:
        report = {
            'status': 'PASS' if exit_code == 0 else 'FAIL',
            'exit_code': exit_code,
            'statistics': verifier.stats,
            'errors': verifier.errors,
            'warnings': verifier.warnings
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("\n==================================================")
        print(" TAMAR SHEAFFER WIKI LINK VERIFIER REPORT")
        print("==================================================")
        print(f"Total Markdown Files Scanned: {verifier.stats['total_files_scanned']}")
        print(f"  - Wiki Pages:                {verifier.stats['wiki_files_count']}")
        print(f"  - Root Docs:                 {verifier.stats['root_files_count']}")
        print(f"Total Links Found:            {verifier.stats['total_links_found']}")
        print(f"  - Internal Checked:          {verifier.stats['internal_links_checked']}")
        print(f"  - External Skipped:          {verifier.stats['external_links_skipped']}")
        print(f"  - Images Checked:            {verifier.stats['image_links_checked']}")
        print("--------------------------------------------------")
        print(f"Broken Links (Missing Files): {verifier.stats['broken_links_count']}")
        print(f"Empty Files (0 Bytes):        {verifier.stats['empty_files_count']}")
        print(f"Broken Heading Anchors:       {verifier.stats['broken_anchors_count']}")
        print(f"Orphan / Unreachable Pages:   {verifier.stats['orphan_pages_count']}")
        print("==================================================")

        if verifier.warnings:
            print(f"\n[!] WARNINGS ({len(verifier.warnings)}):")
            for w in verifier.warnings:
                print(f"  - [{w['type']}] {w['source_file']}:{w['line']} -> {w['message']}")

        if verifier.errors:
            print(f"\n[x] ERRORS ({len(verifier.errors)}):")
            for e in verifier.errors:
                loc = f"{e['source_file']}:{e['line']}" if e['line'] > 0 else e['source_file']
                print(f"  - [{e['type']}] {loc}")
                if e.get('target'):
                    print(f"      Target:   {e['target']}")
                print(f"      Reason:   {e['message']}")
            print("\nVERIFICATION VERDICT: FAILED (Exit Code 1)")
        else:
            print("\nVERIFICATION VERDICT: PASSED (All internal links and anchors valid)")

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
