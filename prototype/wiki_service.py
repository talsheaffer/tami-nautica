#!/usr/bin/env python3
"""
prototype/wiki_service.py
Read-only domain service for serving the 31-page Biographical & Artistic Wiki
over the collaborative Family Beta Hub. Parses YAML frontmatter, builds the
categorized navigation tree, and enforces strict path-traversal containment.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class WikiNotFoundError(KeyError):
    """Raised when a requested wiki markdown page does not exist."""


class WikiValidationError(ValueError):
    """Raised when a requested wiki path is malformed or attempts traversal."""


CATEGORY_ORDER: tuple[tuple[str, str, str], ...] = (
    ("overview", "סקירה ראשית וציר זמן", "🌟"),
    ("places", "בתים ומרחבי חיים (הרצליה, חיפה, פריז)", "🏡"),
    ("family", "משפחה וחברים קרובים", "👨‍👩‍👦‍👦"),
    ("bio", "ביוגרפיה, הנדסה ופמיניזם", "🧭"),
    ("art", "אמנות, חומרים ותערוכות", "🎨"),
    ("travels", "מסעות ושהויות אמן בעולם", "✈️"),
)

SAFE_WIKI_RELPATH_RE = re.compile(r"^[A-Za-z0-9_./-]+\.md$")


@dataclass(frozen=True)
class WikiPageSummary:
    rel_path: str
    slug: str
    title: str
    category: str
    era: str
    verification_status: str
    people: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["people"] = list(self.people)
        return data


@dataclass(frozen=True)
class WikiPageDetail:
    summary: WikiPageSummary
    markdown_body: str
    sources: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary.to_dict(),
            "markdown_body": self.markdown_body,
            "sources": list(self.sources),
        }


def _strip_quotes(raw: str) -> str:
    val = raw.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        return val[1:-1]
    return val


def parse_wiki_markdown(raw_text: str, rel_path: str) -> tuple[dict[str, Any], str]:
    """Parses lightweight YAML frontmatter + Markdown body without external YAML deps."""
    meta: dict[str, Any] = {
        "title": rel_path,
        "slug": Path(rel_path).stem,
        "category": Path(rel_path).parts[0] if len(Path(rel_path).parts) > 1 else "overview",
        "era": "",
        "verification_status": "verified",
        "people": [],
        "sources": [],
    }

    if not raw_text.startswith("---\n"):
        return meta, raw_text

    end_idx = raw_text.find("\n---\n", 4)
    if end_idx == -1:
        return meta, raw_text

    fm_block = raw_text[4:end_idx]
    body = raw_text[end_idx + 5 :].lstrip("\n")

    current_list_key: str | None = None
    current_source: dict[str, str] | None = None

    for line in fm_block.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0 and ":" in stripped:
            key, val_part = stripped.split(":", 1)
            key = key.strip()
            val_clean = _strip_quotes(val_part)
            current_source = None
            if not val_clean:
                current_list_key = key
                if key not in meta or not isinstance(meta[key], list):
                    meta[key] = []
            else:
                current_list_key = None
                meta[key] = val_clean
        elif current_list_key == "sources":
            if stripped.startswith("- "):
                current_source = {}
                meta.setdefault("sources", []).append(current_source)
                rest = stripped[2:].strip()
                if ":" in rest:
                    sk, sv = rest.split(":", 1)
                    current_source[sk.strip()] = _strip_quotes(sv)
            elif current_source is not None and ":" in stripped:
                sk, sv = stripped.split(":", 1)
                current_source[sk.strip()] = _strip_quotes(sv)
        elif current_list_key is not None and stripped.startswith("- "):
            item_val = _strip_quotes(stripped[2:])
            if item_val:
                meta.setdefault(current_list_key, []).append(item_val)

    if rel_path == "README.md":
        meta["category"] = "overview"

    return meta, body


class WikiPortalService:
    """Strictly read-only service over `/Users/talsheaffer/code/tami/wiki`."""

    def __init__(self, wiki_dir: Path) -> None:
        self.wiki_dir = wiki_dir.resolve()
        if not self.wiki_dir.is_dir():
            raise RuntimeError(f"Wiki directory not found: {self.wiki_dir}")

    def resolve_safe_md_path(self, rel_path: str) -> tuple[str, Path]:
        clean = rel_path.strip().lstrip("/")
        if not clean:
            clean = "README.md"
        if ".." in Path(clean).parts or not SAFE_WIKI_RELPATH_RE.match(clean):
            raise WikiValidationError(f"Invalid wiki page path: '{rel_path}'.")
        resolved = (self.wiki_dir / clean).resolve()
        try:
            norm_rel = str(resolved.relative_to(self.wiki_dir)).replace("\\", "/")
        except ValueError as exc:
            raise WikiValidationError(f"Wiki path escapes wiki root: '{rel_path}'.") from exc
        if not resolved.is_file():
            raise WikiNotFoundError(f"Wiki page not found: '{norm_rel}'.")
        return norm_rel, resolved

    def list_navigation(self) -> dict[str, Any]:
        pages_by_cat: dict[str, list[dict[str, Any]]] = {cid: [] for cid, _, _ in CATEGORY_ORDER}
        all_md_paths = sorted(self.wiki_dir.rglob("*.md"))

        for md_path in all_md_paths:
            rel_path = str(md_path.relative_to(self.wiki_dir)).replace("\\", "/")
            raw_text = md_path.read_text(encoding="utf-8")
            meta, _ = parse_wiki_markdown(raw_text, rel_path)
            cat = str(meta.get("category") or "overview")
            if cat not in pages_by_cat:
                cat = "overview"
            summary = WikiPageSummary(
                rel_path=rel_path,
                slug=str(meta.get("slug") or md_path.stem),
                title=str(meta.get("title") or rel_path),
                category=cat,
                era=str(meta.get("era") or ""),
                verification_status=str(meta.get("verification_status") or "verified"),
                people=tuple(str(p) for p in (meta.get("people") or [])),
            )
            pages_by_cat[cat].append(summary.to_dict())

        # Sort index.md / README.md first inside each category
        for cat_list in pages_by_cat.values():
            cat_list.sort(
                key=lambda item: (
                    0 if item["rel_path"].endswith(("README.md", "index.md")) else 1,
                    item["rel_path"],
                )
            )

        sections = [
            {
                "category_id": cid,
                "label_he": label_he,
                "icon": icon,
                "pages": pages_by_cat.get(cid, []),
            }
            for cid, label_he, icon in CATEGORY_ORDER
            if pages_by_cat.get(cid)
        ]
        total_pages = sum(len(sec["pages"]) for sec in sections)
        return {"total_pages": total_pages, "sections": sections}

    def get_page(self, rel_path: str) -> WikiPageDetail:
        norm_rel, md_path = self.resolve_safe_md_path(rel_path)
        raw_text = md_path.read_text(encoding="utf-8")
        meta, body = parse_wiki_markdown(raw_text, norm_rel)
        cat = str(meta.get("category") or "overview")
        summary = WikiPageSummary(
            rel_path=norm_rel,
            slug=str(meta.get("slug") or md_path.stem),
            title=str(meta.get("title") or norm_rel),
            category=cat,
            era=str(meta.get("era") or ""),
            verification_status=str(meta.get("verification_status") or "verified"),
            people=tuple(str(p) for p in (meta.get("people") or [])),
        )
        raw_sources = meta.get("sources") or []
        sources = tuple(
            {
                "type": str(s.get("type", "source")),
                "title": str(s.get("title", s.get("ref", ""))),
                "ref": str(s.get("ref", "")),
            }
            for s in raw_sources
            if isinstance(s, dict)
        )
        return WikiPageDetail(summary=summary, markdown_body=body, sources=sources)
