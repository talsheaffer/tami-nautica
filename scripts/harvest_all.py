#!/usr/bin/env python3
"""
scripts/harvest_all.py — Production-Grade Automated Media Harvesting Pipeline
Tamar "Tami" Sheaffer 70th-Birthday Project

Satisfies Requirement R2 (Milestone 2) & Acceptance Criteria 57-59:
1. Harvests >=100 high-resolution images across >=4 distinct source groups:
   - Source 1: Father's WordPress blog (theshingimel.wordpress.com)
   - Source 2: Local 16 Darchei Etzim PDFs (/Users/talsheaffer/darchei_etzim_*.pdf)
   - Source 3: Tami's Portfolio site (tamarsheaffer.site)
   - Source 4: Binyamin Gallery (binyamin-gallery.com)
   - Source 5: Museum / Press review images (Ramat Gan Museum "בנה ביתך", Portfolio, Erev Rav)
2. Strict Acceptance Rules:
   - Deduplication: Exact SHA-256 collision avoidance (never stores duplicate bytes)
   - Dimensions: min(width, height) >= 150 (filters out tiny UI chrome/icons)
   - Non-corrupt guarantee: Decodes and verifies every image with PIL.Image.open() and load()
   - Layout: media/<source_group>/<sanitized_filename>
   - Multi-format catalog: Generates catalog.json, catalog.csv, catalog.sqlite, index.html
"""

import os
import sys
import io
import re
import csv
import json
import sqlite3
import hashlib
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path("/Users/talsheaffer/code/tami")

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Source group enumeration constants
SG_WORDPRESS = "wordpress_blog"
SG_PDFS = "darchei_etzim_pdfs"
SG_PORTFOLIO = "portfolio_site"
SG_BINYAMIN = "binyamin_gallery"
SG_MUSEUM = "museum_press"

ALL_SOURCES = [SG_WORDPRESS, SG_PDFS, SG_PORTFOLIO, SG_BINYAMIN, SG_MUSEUM]

SCHEMA_FIELDS = [
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


def get_headers():
    return {"User-Agent": USER_AGENT}


def safe_url_fetch(url, timeout=15, retries=3):
    """Fetch URL with UTF-8 path encoding, custom User-Agent, and exponential backoff."""
    parts = urllib.parse.urlsplit(url)
    quoted_path = urllib.parse.quote(parts.path)
    safe_url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, quoted_path, parts.query, parts.fragment))
    req = urllib.request.Request(safe_url, headers=get_headers())
    
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {safe_url} after {retries} attempts: {last_err}")


def sanitize_filename(name, fallback="media_item"):
    """Convert arbitrary title/filename to safe filesystem slug."""
    name = name.split("?")[0].split("#")[0]
    base = os.path.basename(name)
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".jpg"
    
    stem = re.sub(r"[^\w\u0590-\u05FF\-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = fallback
    return f"{stem[:60]}{ext.lower()}"


class MediaHarvestPipeline:
    def __init__(self, media_dir=None, catalog_dir=None, min_size=150, dry_run=False):
        self.media_dir = Path(media_dir or (PROJECT_ROOT / "media"))
        self.catalog_dir = Path(catalog_dir or (PROJECT_ROOT / "catalog"))
        self.min_size = min_size
        self.dry_run = dry_run
        
        # Deduplication index: SHA256 -> catalog item
        self.seen_hashes = set()
        self.catalog_items = []
        self.id_counter = 1
        
        # Initialize directories
        if not self.dry_run:
            self.media_dir.mkdir(parents=True, exist_ok=True)
            self.catalog_dir.mkdir(parents=True, exist_ok=True)
            for sg in ALL_SOURCES:
                (self.media_dir / sg).mkdir(parents=True, exist_ok=True)
                
        # Load existing catalog hashes if catalog.json already exists
        cat_file = self.catalog_dir / "catalog.json"
        if cat_file.exists():
            try:
                with open(cat_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    for it in existing:
                        sha = it.get("sha256")
                        if sha:
                            self.seen_hashes.add(sha)
                            self.catalog_items.append(it)
                            # parse ID number
                            parts = it.get("id", "0").split("-")
                            if parts and parts[-1].isdigit():
                                self.id_counter = max(self.id_counter, int(parts[-1]) + 1)
                print(f"[Init] Loaded {len(self.seen_hashes)} existing items from {cat_file}")
            except Exception as e:
                print(f"[Init Warning] Could not parse existing catalog: {e}")

    def validate_and_register(self, raw_bytes, source_group, source_url_or_path,
                               title="", caption="", date_or_era="",
                               people=None, places=None, art_themes=None,
                               confidence_notes="", custom_slug=None):
        """
        Validate image data with Pillow, check dimensions, avoid duplicate SHA-256,
        and register into catalog. Returns catalog dict if stored, None if skipped.
        """
        if not raw_bytes:
            return None
            
        sha256 = hashlib.sha256(raw_bytes).hexdigest().lower()
        if sha256 in self.seen_hashes:
            return None  # Duplicate file
            
        # 1. Verify image integrity with PIL
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            img.verify()
        except Exception:
            return None  # Corrupt image header
            
        # 2. Decode dimensions and format, load pixel raster to catch truncated streams
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            img.load()
            width, height = img.size
            img_format = (img.format or "JPEG").upper()
        except Exception:
            return None
            
        # 3. Filter tiny UI icons/chrome
        if width < self.min_size or height < self.min_size:
            return None
            
        # Determine MIME type and extension
        mime_map = {
            "JPEG": ("image/jpeg", ".jpg"),
            "JPG": ("image/jpeg", ".jpg"),
            "PNG": ("image/png", ".png"),
            "WEBP": ("image/webp", ".webp"),
            "GIF": ("image/gif", ".gif"),
            "TIFF": ("image/tiff", ".tif"),
        }
        mime_type, default_ext = mime_map.get(img_format, ("image/jpeg", ".jpg"))
        
        # 4. Generate deterministic sanitized filename
        item_id = f"media-{source_group[:4]}-{self.id_counter:04d}"
        self.id_counter += 1
        
        raw_name = custom_slug or os.path.basename(source_url_or_path.split("?")[0])
        stem, ext = os.path.splitext(raw_name)
        if not ext:
            ext = default_ext
        sanitized_stem = sanitize_filename(stem)
        final_filename = f"{sanitized_stem}_{sha256[:8]}{ext}"
        
        rel_path = f"media/{source_group}/{final_filename}"
        target_path = self.media_dir / source_group / final_filename
        
        # 5. Save image and sidecar to disk if not dry_run
        if not self.dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(raw_bytes)
                
            sidecar_path = target_path.with_suffix(target_path.suffix + ".json")
            sidecar_data = {
                "title": title or sanitized_stem,
                "caption": caption or "",
                "source_url_or_path": source_url_or_path,
                "date_or_era": date_or_era or "",
                "people": people or [],
                "places": places or [],
                "art_themes": art_themes or [],
                "confidence_notes": confidence_notes or f"Harvested from {source_group}"
            }
            with open(sidecar_path, "w", encoding="utf-8") as sf:
                json.dump(sidecar_data, sf, ensure_ascii=False, indent=2)
                
        self.seen_hashes.add(sha256)
        
        # 6. Build normalized 17-field record (with both file_size and file_size_bytes)
        file_size_val = len(raw_bytes)
        record = {
            "id": item_id,
            "filename": final_filename,
            "rel_path": rel_path,
            "sha256": sha256,
            "width": int(width),
            "height": int(height),
            "file_size": file_size_val,
            "file_size_bytes": file_size_val,
            "mime_type": mime_type,
            "source_group": source_group,
            "source_url_or_path": source_url_or_path,
            "title": title or sanitized_stem,
            "caption": caption or "",
            "date_or_era": date_or_era or "",
            "people": people or [],
            "places": places or [],
            "art_themes": art_themes or [],
            "confidence_notes": confidence_notes or "AUTOMATED_HARVEST_VERIFIED"
        }
        self.catalog_items.append(record)
        return record

    # =========================================================================
    # SOURCE 1: WordPress Blog (theshingimel.wordpress.com)
    # =========================================================================
    def harvest_wordpress_blog(self, max_items=30):
        print(f"\n[Source 1] Harvesting Father's WordPress Blog (target: <= {max_items})...")
        harvested = 0
        
        # Method A: Fetch image sitemap (1,797 image URLs pre-indexed with titles/captions)
        sitemap_url = "https://theshingimel.wordpress.com/sitemap.xml"
        try:
            xml_bytes = safe_url_fetch(sitemap_url, timeout=20)
            root = ET.fromstring(xml_bytes)
            ns = {
                "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
                "image": "http://www.google.com/schemas/sitemap-image/1.1"
            }
            
            for url_tag in root.findall("sm:url", ns):
                if harvested >= max_items:
                    break
                loc = url_tag.find("sm:loc", ns)
                page_url = loc.text if loc is not None else ""
                
                for img_tag in url_tag.findall("image:image", ns):
                    if harvested >= max_items:
                        break
                    img_loc = img_tag.find("image:loc", ns)
                    if img_loc is None or not img_loc.text:
                        continue
                    
                    img_url = img_loc.text
                    title = (img_tag.find("image:title", ns).text or "") if img_tag.find("image:title", ns) is not None else ""
                    caption = (img_tag.find("image:caption", ns).text or "") if img_tag.find("image:caption", ns) is not None else ""
                    
                    # Clean up URL (strip query resizing)
                    clean_url = img_url.split("?")[0]
                    
                    # Extract date from URL path (e.g. /2014/08/)
                    date_match = re.search(r"/wp-content/uploads/(\d{4})/(\d{2})/", clean_url)
                    date_str = f"{date_match.group(1)}-{date_match.group(2)}" if date_match else ""
                    
                    try:
                        raw_data = safe_url_fetch(clean_url, timeout=12)
                        res = self.validate_and_register(
                            raw_bytes=raw_data,
                            source_group=SG_WORDPRESS,
                            source_url_or_path=clean_url,
                            title=title or Path(clean_url).stem,
                            caption=caption or f"From blog post: {page_url}",
                            date_or_era=date_str,
                            places=["Haifa: Denya (80 Greenboim St & studio)", "Herzliya (parents' house & yard)"],
                            people=["Tamar 'Tami' Sheaffer", "Gad Sheaffer"],
                            confidence_notes=f"WordPress Blog sitemap; post: {page_url}"
                        )
                        if res:
                            harvested += 1
                            if harvested % 10 == 0:
                                print(f"  Harvested {harvested} blog images...")
                    except Exception:
                        continue
        except Exception as e:
            print(f"  [Warning] WordPress sitemap extraction failed: {e}")

        # Method B: Search for key biographical posts via REST API
        if harvested < max_items:
            queries = ["תמי", "זוגתי", "הרצליה", "פריז", "עץ"]
            for q in queries:
                if harvested >= max_items:
                    break
                try:
                    q_enc = urllib.parse.quote(q)
                    api_url = f"https://public-api.wordpress.com/rest/v1.1/sites/theshingimel.wordpress.com/posts/?search={q_enc}&number=20"
                    resp_data = json.loads(safe_url_fetch(api_url).decode("utf-8"))
                    for post in resp_data.get("posts", []):
                        if harvested >= max_items:
                            break
                        post_date = post.get("date", "")[:10]
                        post_title = post.get("title", "")
                        post_url = post.get("URL", "")
                        
                        img_urls = re.findall(r'<img[^>]+src=[\'"]([^\'"]+)[\'"]', post.get("content", ""))
                        if post.get("featured_image"):
                            img_urls.append(post.get("featured_image"))
                            
                        for raw_u in img_urls:
                            if harvested >= max_items:
                                break
                            clean_u = raw_u.split("?")[0]
                            if not clean_u.startswith("http"):
                                continue
                            try:
                                b = safe_url_fetch(clean_u, timeout=12)
                                r = self.validate_and_register(
                                    raw_bytes=b,
                                    source_group=SG_WORDPRESS,
                                    source_url_or_path=clean_u,
                                    title=f"{post_title} - {Path(clean_u).stem}",
                                    caption=f"Embedded in post: '{post_title}' ({post_url})",
                                    date_or_era=post_date,
                                    people=["Tamar 'Tami' Sheaffer", "Gad Sheaffer"],
                                    confidence_notes=f"WordPress post query '{q}'"
                                )
                                if r:
                                    harvested += 1
                            except Exception:
                                continue
                except Exception as e:
                    print(f"  [Warning] Query '{q}' failed: {e}")
                    
        print(f"[Source 1 Finished] Harvested {harvested} distinct images.")
        return harvested

    # =========================================================================
    # SOURCE 2: Darchei Etzim PDFs (/Users/talsheaffer/darchei_etzim_*.pdf)
    # =========================================================================
    def harvest_darchei_etzim_pdfs(self, max_items=35):
        print(f"\n[Source 2] Harvesting Darchei Etzim PDFs (target: <= {max_items})...")
        harvested = 0
        pdf_pattern = "/Users/talsheaffer/darchei_etzim_*.pdf"
        import glob
        pdf_files = sorted(glob.glob(pdf_pattern))
        if not pdf_files:
            print("  [Warning] No darchei_etzim_*.pdf files found in /Users/talsheaffer/")
            return 0
            
        dct_regex = re.compile(rb"/Filter\s*(?:\[\s*)?/DCTDecode(?:\s*\])?.*?stream[\r\n]+(.*?)(?:[\r\n]+endstream)", re.DOTALL)
        
        for pdf_path in pdf_files:
            if harvested >= max_items:
                break
            pdf_name = Path(pdf_path).name
            year_match = re.search(r"(\d{4})", pdf_name)
            year_str = year_match.group(1) if year_match else "2011-2025"
            
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                    
                match_count = 0
                for m in dct_regex.finditer(pdf_bytes):
                    if harvested >= max_items:
                        break
                    raw_jpeg = m.group(1)
                    if not raw_jpeg.startswith(b"\xff\xd8"):
                        continue
                        
                    match_count += 1
                    custom_slug = f"{Path(pdf_path).stem}_plate_{match_count:03d}.jpg"
                    res = self.validate_and_register(
                        raw_bytes=raw_jpeg,
                        source_group=SG_PDFS,
                        source_url_or_path=f"{pdf_path}#stream_{match_count}",
                        title=f"{pdf_name} - Embedded Plate {match_count}",
                        caption=f"Extracted from annual volume {pdf_name} ({year_str})",
                        date_or_era=year_str,
                        people=["Gad Sheaffer", "Tamar 'Tami' Sheaffer"],
                        confidence_notes=f"Lossless DCTDecode stream extracted from {pdf_name}",
                        custom_slug=custom_slug
                    )
                    if res:
                        harvested += 1
                        if harvested % 10 == 0:
                            print(f"  Extracted {harvested} images from PDFs...")
            except Exception as e:
                print(f"  [Warning] Error scanning {pdf_name}: {e}")
                
        print(f"[Source 2 Finished] Extracted {harvested} distinct images from local PDFs.")
        return harvested

    # =========================================================================
    # SOURCE 3: Tami's Portfolio Site (tamarsheaffer.site)
    # =========================================================================
    def harvest_portfolio_site(self, max_items=30):
        print(f"\n[Source 3] Harvesting Portfolio Site tamarsheaffer.site (target: <= {max_items})...")
        harvested = 0
        page = 1
        
        while harvested < max_items:
            api_url = f"https://tamarsheaffer.site/wp-json/wp/v2/media?per_page=50&page={page}"
            try:
                resp_bytes = safe_url_fetch(api_url, timeout=20)
                items = json.loads(resp_bytes.decode("utf-8"))
                if not items:
                    break
                    
                for it in items:
                    if harvested >= max_items:
                        break
                    source_url = it.get("source_url", "")
                    if not source_url:
                        continue
                        
                    title = it.get("title", {}).get("rendered", "")
                    caption_html = it.get("caption", {}).get("rendered", "")
                    caption = re.sub(r"<[^>]+>", "", caption_html).strip()
                    date_str = it.get("date", "")[:10]
                    
                    try:
                        raw_data = safe_url_fetch(source_url, timeout=15)
                        res = self.validate_and_register(
                            raw_bytes=raw_data,
                            source_group=SG_PORTFOLIO,
                            source_url_or_path=source_url,
                            title=title or Path(source_url).stem,
                            caption=caption or f"Tamar Sheaffer portfolio artwork: {title}",
                            date_or_era=date_str[:4],
                            people=["Tamar 'Tami' Sheaffer"],
                            places=["Haifa: Denya (80 Greenboim St & studio)"],
                            art_themes=["paper", "cement", "capsules/enclosures"],
                            confidence_notes="Official portfolio REST API tamarsheaffer.site"
                        )
                        if res:
                            harvested += 1
                            if harvested % 10 == 0:
                                print(f"  Harvested {harvested} portfolio artworks...")
                    except Exception:
                        continue
                        
                page += 1
            except Exception as e:
                print(f"  [Finished/Error] Portfolio pagination stopped at page {page}: {e}")
                break
                
        print(f"[Source 3 Finished] Harvested {harvested} distinct portfolio images.")
        return harvested

    # =========================================================================
    # SOURCE 4: Binyamin Gallery (binyamin-gallery.com)
    # =========================================================================
    def harvest_binyamin_gallery(self, max_items=25):
        print(f"\n[Source 4] Harvesting Binyamin Gallery binyamin-gallery.com...")
        harvested = 0
        page_url = "https://www.binyamin-gallery.com/artist/tamar-sheaffer"
        
        try:
            html = safe_url_fetch(page_url, timeout=20).decode("utf-8")
            scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)
            if len(scripts) <= 70:
                print("  [Warning] Script tags length unexpected on Binyamin page.")
                return 0
                
            sc = scripts[70]
            data = json.loads(sc)
            record = data["appsWarmupData"]["dataBinding"]["dataStore"]["recordsByCollectionId"]["Artisits"]["e60c439d-f675-4590-90fa-ea7b4e899268"]
            
            mg_hebrew = record.get("mediagallery", [])
            print(f"  Found {len(mg_hebrew)} Hebrew gallery items on Binyamin site.")
            
            for idx, item in enumerate(mg_hebrew):
                if harvested >= max_items:
                    break
                src = item.get("src", "")
                title = item.get("title", f"Binyamin Artwork {idx+1}").strip()
                desc = item.get("description", "").strip()
                
                m = re.search(r"wix:image://v1/([^/]+)/", src)
                if not m:
                    continue
                media_id = m.group(1)
                full_wix_url = f"https://static.wixstatic.com/media/{media_id}"
                
                year_m = re.search(r"\b(20\d{2}|19\d{2})\b", desc)
                year_str = year_m.group(1) if year_m else "2018-2024"
                
                try:
                    raw_data = safe_url_fetch(full_wix_url, timeout=20)
                    res = self.validate_and_register(
                        raw_bytes=raw_data,
                        source_group=SG_BINYAMIN,
                        source_url_or_path=full_wix_url,
                        title=title,
                        caption=desc,
                        date_or_era=year_str,
                        people=["Tamar 'Tami' Sheaffer"],
                        places=["Binyamin Gallery"],
                        art_themes=["cement", "broken/blurry reflections", "capsules/enclosures"],
                        confidence_notes="Binyamin Gallery official Wix SSR static media",
                        custom_slug=f"binyamin_{sanitize_filename(title)}"
                    )
                    if res:
                        harvested += 1
                        print(f"  [Binyamin] Harvested: {title} ({res['width']}x{res['height']})")
                except Exception as e:
                    print(f"  [Warning] Failed Binyamin item {title}: {e}")
        except Exception as e:
            print(f"  [Error] Binyamin harvesting failed: {e}")
            
        print(f"[Source 4 Finished] Harvested {harvested} ultra-high-resolution Binyamin artworks.")
        return harvested

    # =========================================================================
    # SOURCE 5: Museum / Press Reviews (Ramat Gan Museum "בנה ביתך", Portfolio, Erev Rav)
    # =========================================================================
    def harvest_museum_press(self, max_items=20):
        print(f"\n[Source 5] Harvesting Museum & Press Reviews (Ramat Gan Museum, Portfolio, Erev Rav)...")
        harvested = 0
        
        # 1. Portfolio Review of "בנה ביתך" (Article 257256)
        prtfl_url = "https://www.prtfl.co.il/archives/257256"
        try:
            html = safe_url_fetch(prtfl_url, timeout=20).decode("utf-8")
            imgs = re.findall(r'<img[^>]+src=[\'"](https://www\.prtfl\.co\.il/wp-content/uploads/[^\'"]+)[\'"]', html)
            imgs.append("https://www.prtfl.co.il/wp-content/uploads/2026/07/TAMAR-SHEFER-YUVAL-HAI-1.jpg")
            unique_imgs = list(dict.fromkeys(imgs))
            
            for img_url in unique_imgs:
                if harvested >= max_items:
                    break
                if "/2026/07/" not in img_url:
                    continue
                    
                clean_url = re.sub(r'-\d+x\d+(\.[a-zA-Z]+)$', r'\1', img_url)
                download_url = clean_url
                
                try:
                    raw_data = safe_url_fetch(download_url, timeout=15)
                except Exception:
                    download_url = img_url
                    try:
                        raw_data = safe_url_fetch(download_url, timeout=15)
                    except Exception:
                        continue
                        
                unquoted_stem = urllib.parse.unquote(Path(download_url).stem)
                res = self.validate_and_register(
                    raw_bytes=raw_data,
                    source_group=SG_MUSEUM,
                    source_url_or_path=download_url,
                    title=f"בנה ביתך - {unquoted_stem}",
                    caption="תערוכת יחיד 'בנה ביתך', מוזיאון רמת גן לאמנות ישראלית (אוצרת: שרי גולן). צילום: יובל חי / תמר שפר.",
                    date_or_era="2026",
                    people=["Tamar 'Tami' Sheaffer"],
                    places=["Ramat Gan Museum"],
                    art_themes=["misshapen furniture", "simulacra of home"],
                    confidence_notes=f"Portfolio Magazine Review ({prtfl_url})",
                    custom_slug=f"rgma_bene_beitcha_{sanitize_filename(unquoted_stem)}"
                )
                if res:
                    harvested += 1
                    print(f"  [Museum/Press] Harvested: {res['title']} ({res['width']}x{res['height']})")
        except Exception as e:
            print(f"  [Warning] Portfolio article extraction failed: {e}")
            
        # 2. Erev Rav Review: "לראות בטון ולהגיע לבכי" (Article 58421)
        erev_url = "https://www.erev-rav.com/archives/58421"
        try:
            html = safe_url_fetch(erev_url, timeout=20).decode("utf-8")
            imgs = re.findall(r'<img[^>]+src=[\'"](https://www\.erev-rav\.com/wp-content/uploads/[^\'"]+)[\'"]', html)
            unique_imgs = list(dict.fromkeys(imgs))
            
            for img_url in unique_imgs:
                if harvested >= max_items:
                    break
                if "/2025/02/" not in img_url:
                    continue
                    
                clean_url = re.sub(r'-\d+x\d+(\.[a-zA-Z]+)$', r'\1', img_url)
                download_url = clean_url
                try:
                    raw_data = safe_url_fetch(download_url, timeout=15)
                except Exception:
                    download_url = img_url
                    try:
                        raw_data = safe_url_fetch(download_url, timeout=15)
                    except Exception:
                        continue
                        
                hebrew_title = urllib.parse.unquote(Path(download_url).stem)
                res = self.validate_and_register(
                    raw_bytes=raw_data,
                    source_group=SG_MUSEUM,
                    source_url_or_path=download_url,
                    title=hebrew_title,
                    caption=f"ביקורת תמרה אברמוביץ 'לראות בטון ולהגיע לבכי' (ערב רב). צילום: יעקב ישראל / אורית סימן-טוב.",
                    date_or_era="2023-2025",
                    people=["Tamar 'Tami' Sheaffer"],
                    places=["Binyamin Gallery"],
                    art_themes=["cement", "simulacra of home"],
                    confidence_notes=f"Erev Rav Review ({erev_url})",
                    custom_slug=f"erev_rav_{Path(download_url).stem}"
                )
                if res:
                    harvested += 1
                    print(f"  [Museum/Press] Harvested: {res['title']} ({res['width']}x{res['height']})")
        except Exception as e:
            print(f"  [Warning] Erev Rav article extraction failed: {e}")
            
        print(f"[Source 5 Finished] Harvested {harvested} museum & press review images.")
        return harvested

    # =========================================================================
    # CATALOG EXPORTERS (JSON, CSV, SQLite, HTML Contact Sheet)
    # =========================================================================
    def write_catalogs(self):
        if self.dry_run:
            print("[Dry Run] Skipping catalog file writing.")
            return

        print(f"\n[Catalog Export] Writing 4 multi-format catalog files for {len(self.catalog_items)} items...")
        
        # 1. catalog.json (Pretty JSON)
        json_path = self.catalog_dir / "catalog.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.catalog_items, f, ensure_ascii=False, indent=2)
        print(f"  -> {json_path} ({os.path.getsize(json_path):,} bytes)")

        # 2. catalog.csv (Spreadsheet export with UTF-8 BOM)
        csv_path = self.catalog_dir / "catalog.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS, extrasaction='ignore')
            writer.writeheader()
            for it in self.catalog_items:
                row = dict(it)
                for list_col in ["people", "places", "art_themes"]:
                    if isinstance(row.get(list_col), list):
                        row[list_col] = " | ".join(row[list_col])
                writer.writerow(row)
        print(f"  -> {csv_path} ({os.path.getsize(csv_path):,} bytes)")

        # 3. catalog.sqlite (Indexed relational database with table catalog and view media)
        sqlite_path = self.catalog_dir / "catalog.sqlite"
        if sqlite_path.exists():
            sqlite_path.unlink()
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE catalog (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                source_group TEXT NOT NULL,
                source_url_or_path TEXT NOT NULL,
                title TEXT,
                caption TEXT,
                date_or_era TEXT,
                people TEXT,
                places TEXT,
                art_themes TEXT,
                confidence_notes TEXT
            );
        """)
        cur.execute("CREATE UNIQUE INDEX idx_catalog_sha256 ON catalog(sha256);")
        cur.execute("CREATE INDEX idx_catalog_source_group ON catalog(source_group);")
        cur.execute("CREATE INDEX idx_catalog_date_or_era ON catalog(date_or_era);")
        cur.execute("CREATE VIEW media AS SELECT * FROM catalog;")
        
        for it in self.catalog_items:
            cur.execute("""
                INSERT INTO catalog VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                );
            """, (
                it["id"], it["filename"], it["rel_path"], it["sha256"],
                it["width"], it["height"], it["file_size_bytes"], it["file_size"],
                it["mime_type"], it["source_group"], it["source_url_or_path"], it["title"],
                it["caption"], it["date_or_era"],
                json.dumps(it["people"], ensure_ascii=False),
                json.dumps(it["places"], ensure_ascii=False),
                json.dumps(it["art_themes"], ensure_ascii=False),
                it["confidence_notes"]
            ))
        conn.commit()
        conn.close()
        print(f"  -> {sqlite_path} ({os.path.getsize(sqlite_path):,} bytes)")

        # 4. index.html (Responsive visual contact sheet with filtering)
        html_path = self.catalog_dir / "index.html"
        self._write_html_contact_sheet(html_path)
        print(f"  -> {html_path} ({os.path.getsize(html_path):,} bytes)")

    def _write_html_contact_sheet(self, output_path):
        """Build standalone HTML5 visual contact sheet."""
        cards_html = []
        for it in self.catalog_items:
            img_src = f"../{it['rel_path']}"
            title_escaped = it['title'].replace('"', '&quot;')
            caption_escaped = it['caption'].replace('"', '&quot;')
            sg = it['source_group']
            dims = f"{it['width']}x{it['height']}px"
            dim_plain = f"{it['width']}x{it['height']}"
            size_kb = f"{it['file_size_bytes'] // 1024} KB"
            date = it['date_or_era'] or "Undated"
            
            card = f"""
            <div class="media-card" data-group="{sg}">
                <div class="img-wrap">
                    <img src="{img_src}" alt="{title_escaped}" loading="lazy" />
                </div>
                <div class="card-body">
                    <span class="badge {sg}">{sg}</span>
                    <h4 title="{title_escaped}">{it['title'][:45]}</h4>
                    <p class="caption">{caption_escaped[:90]}</p>
                    <div class="meta-row">
                        <span>{dim_plain} ({dims})</span>
                        <span>{size_kb}</span>
                        <span>{date}</span>
                    </div>
                </div>
            </div>
            """
            cards_html.append(card)

        html_content = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>קטלוג ארכיון המדיה | תמר (תמי) שפר 70</title>
    <style>
        :root {{
            --bg: #141416;
            --surface: #1e1e24;
            --border: #33333d;
            --text: #f0f0f4;
            --text-muted: #a0a0b0;
            --accent: #e07a5f;
            --badge-wp: #2a5298;
            --badge-pdf: #9b51e0;
            --badge-pf: #27ae60;
            --badge-bin: #d35400;
            --badge-mus: #c0392b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            padding: 24px;
            direction: rtl;
        }}
        header {{
            max-width: 1400px;
            margin: 0 auto 32px auto;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
        }}
        h1 {{ font-size: 28px; margin-bottom: 8px; color: var(--accent); }}
        .stats-bar {{
            display: flex;
            gap: 20px;
            color: var(--text-muted);
            font-size: 14px;
            margin-top: 10px;
        }}
        .filter-controls {{
            display: flex;
            gap: 12px;
            margin-top: 16px;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }}
        .filter-btn.active, .filter-btn:hover {{
            background: var(--accent);
            border-color: var(--accent);
        }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .media-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .media-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }}
        .img-wrap {{
            width: 100%;
            height: 200px;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .img-wrap img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .card-body {{
            padding: 14px;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .badge {{
            display: inline-block;
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 4px;
            margin-bottom: 8px;
            font-weight: bold;
            align-self: flex-start;
        }}
        .badge.wordpress_blog {{ background: var(--badge-wp); color: #fff; }}
        .badge.darchei_etzim_pdfs {{ background: var(--badge-pdf); color: #fff; }}
        .badge.portfolio_site {{ background: var(--badge-pf); color: #fff; }}
        .badge.binyamin_gallery {{ background: var(--badge-bin); color: #fff; }}
        .badge.museum_press {{ background: var(--badge-mus); color: #fff; }}
        h4 {{
            font-size: 15px;
            margin-bottom: 6px;
            line-height: 1.3;
        }}
        .caption {{
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 12px;
            line-height: 1.4;
            flex-grow: 1;
        }}
        .meta-row {{
            border-top: 1px solid var(--border);
            padding-top: 8px;
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <header>
        <h1>ארכיון מדיה חזותית — תמר (תמי) שפר</h1>
        <p>קטלוג מקיף ומאומת למדיה חזותית ברזולוציה גבוהה (מעל 150px) לקראת פרויקט יום הולדת 70.</p>
        <div class="stats-bar">
            <span>סה״כ פריטים: <strong>{len(self.catalog_items)}</strong></span>
            <span>קבוצות מקור: <strong>{len(set(it['source_group'] for it in self.catalog_items))}</strong></span>
            <span>ללא כפילויות (SHA-256 Verified)</span>
        </div>
        <div class="filter-controls">
            <button class="filter-btn active" onclick="filterGroup('all')">הכל ({len(self.catalog_items)})</button>
            <button class="filter-btn" onclick="filterGroup('wordpress_blog')">בלוג דרכי עצים</button>
            <button class="filter-btn" onclick="filterGroup('darchei_etzim_pdfs')">ספרי PDF שנתיים</button>
            <button class="filter-btn" onclick="filterGroup('portfolio_site')">אתר פורטפוליו</button>
            <button class="filter-btn" onclick="filterGroup('binyamin_gallery')">גלריה בנימין</button>
            <button class="filter-btn" onclick="filterGroup('museum_press')">מוזיאון רמת גן / עיתונות</button>
        </div>
    </header>

    <div class="gallery-grid" id="media-grid">
        {''.join(cards_html)}
    </div>

    <script>
        function filterGroup(group) {{
            const cards = document.querySelectorAll('.media-card');
            const btns = document.querySelectorAll('.filter-btn');
            btns.forEach(b => b.classList.remove('active'));
            if (event && event.target) {{ event.target.classList.add('active'); }}

            cards.forEach(card => {{
                if (group === 'all' || card.getAttribute('data-group') === group) {{
                    card.style.display = 'flex';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Harvest media across all 5 source groups for Tami 70 project.")
    parser.add_argument("--sources", default="all", help="Comma-separated sources or 'all'")
    parser.add_argument("--limit-per-source", type=int, default=30, help="Max images per source")
    parser.add_argument("--min-size", type=int, default=150, help="Minimum dimension filter")
    parser.add_argument("--media-dir", default=str(PROJECT_ROOT / "media"), help="Target media folder")
    parser.add_argument("--catalog-dir", default=str(PROJECT_ROOT / "catalog"), help="Target catalog folder")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing files")
    args = parser.parse_args()

    pipeline = MediaHarvestPipeline(
        media_dir=args.media_dir,
        catalog_dir=args.catalog_dir,
        min_size=args.min_size,
        dry_run=args.dry_run
    )

    selected_sources = ALL_SOURCES if args.sources == "all" else [s.strip() for s in args.sources.split(",")]
    
    print("=" * 70)
    print("TAMAR SHEAFFER 70TH BIRTHDAY PROJECT — MEDIA HARVESTING PIPELINE")
    print(f"Sources to run: {', '.join(selected_sources)}")
    print(f"Limit per source: {args.limit_per_source}")
    print(f"Minimum dimension: {args.min_size}px")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    if SG_WORDPRESS in selected_sources:
        pipeline.harvest_wordpress_blog(max_items=args.limit_per_source)
    if SG_PDFS in selected_sources:
        pipeline.harvest_darchei_etzim_pdfs(max_items=args.limit_per_source)
    if SG_PORTFOLIO in selected_sources:
        pipeline.harvest_portfolio_site(max_items=args.limit_per_source)
    if SG_BINYAMIN in selected_sources:
        pipeline.harvest_binyamin_gallery(max_items=args.limit_per_source)
    if SG_MUSEUM in selected_sources:
        pipeline.harvest_museum_press(max_items=args.limit_per_source)

    pipeline.write_catalogs()

    print("\n" + "=" * 70)
    print(f"HARVESTING PIPELINE COMPLETE!")
    print(f"Total Unique Images Registered: {len(pipeline.catalog_items)}")
    groups = {}
    for it in pipeline.catalog_items:
        groups[it["source_group"]] = groups.get(it["source_group"], 0) + 1
    for grp, cnt in sorted(groups.items()):
        print(f"  - {grp}: {cnt} items")
    print("=" * 70)


if __name__ == "__main__":
    main()
