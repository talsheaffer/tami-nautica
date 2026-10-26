# PROJECT SPECIFICATION: Tamar "Tami" Sheaffer 70th-Birthday Project

**Project Name:** `tami`  
**Milestone Target:** 25 October 2026 (כ"ה בתשרי תשפ"ז)  
**Root Working Directory:** `/Users/talsheaffer/code/tami`  
**Execution Architecture:** Dual-Track (Sequential Milestones M1–M5 + Independent Opaque-Box E2E Testing)  

---

## 1. System Architecture & Boundaries

The project is structured into five distinct, decoupled architectural modules governed by rigid contracts:

```
+---------------------------------------------------------------------------------------------------------+
|                                    PROJECT MODULE ARCHITECTURE                                          |
+---------------------------------------------------------------------------------------------------------+
|  1. GOVERNANCE & METRICS (`governance`)                                                                 |
|     Root artifacts: `MISSION.md`, `TODO.md`, `PROGRESS.md`, `PROJECT.md`                                |
|     Verification: `tests/test_governance.py`, `tools/verify_wiki_links.py`                              |
+---------------------------------------------------------------------------------------------------------+
|  2. MEDIA ARCHIVE & INGESTION (`media_archive`)                                                         |
|     Storage: `media/<source_group>/`, `catalog/catalog.{json,csv,sqlite,html}`                          |
|     Pipeline: `drop_folder/`, `tools/ingest.py`, `tests/test_ingest.py`                                 |
+---------------------------------------------------------------------------------------------------------+
|  3. BIOGRAPHICAL & ARTISTIC WIKI (`wiki`)                                                               |
|     Corpus: 24 hyperlinked pages across `bio/`, `family/`, `places/`, `art/`, `travels/`                |
|     Integrity: Footnote citations, `[Verification needed for family]` flags, zero broken links          |
+---------------------------------------------------------------------------------------------------------+
|  4. RESEARCH REPORTS & STRATEGY (`reports`)                                                             |
|     `reports/DELIVERY_LONGEVITY_PLAN.md`: Dual-Release (PWA + 20-Yr Offline Archival Time-Capsule)      |
|     `reports/SPATIAL_CONCEPT_AND_PRECEDENTS.md`: 6 real web precedents, 5-stage AI depth pipeline       |
+---------------------------------------------------------------------------------------------------------+
|  5. PLAYABLE 2.5D SPATIAL PROTOTYPE (`prototype`)                                                       |
|     Engine: Three.js (WebGL 2.0) multi-plane parallax viewer                                            |
|     Scenes: 2 real scenes (Herzliya Yard & Denya Studio/Chamber) with LaMa-inpainted background plates   |
|     Verification: Headless Playwright test at 1440x900 (desktop) & 390x844 (mobile) with zero errors   |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Feature Inventory

| # | Feature Key | Title & Description | Milestone | Source Requirement |
|---|---|---|---|---|
| 1 | `FEAT-GOV-MISSION` | **Project Mission Document**: Root `MISSION.md` capturing emotional, artistic, engineering, and non-hollow philosophy. | M1 | R1, Criteria 53 |
| 2 | `FEAT-GOV-TODO` | **Live Todo Checklist**: Root `TODO.md` with granular checkboxes tracking all R1–R5 tasks and verification steps. | M1 | R1, Criteria 53 |
| 3 | `FEAT-GOV-PROGRESS` | **Verbose Live Progress Log**: Root `PROGRESS.md` recording harvest counts, access logs, benchmarks, and family curation items. | M1 | R1, Criteria 53 |
| 4 | `FEAT-TOOL-LINKCHECK` | **Wiki & Doc Link Checker**: `tools/verify_wiki_links.py` traversing markdown links, validating relative targets and UTF-8 paths. | M1 | R1, R3, Criteria 62 |
| 5 | `FEAT-MEDIA-HARVEST` | **High-Recall Media Collection**: ≥100 deduplicated images (≥150px) across ≥4 sources (WordPress, PDFs, portfolio, gallery, museum). | M2 | R2, Criteria 57 |
| 6 | `FEAT-MEDIA-CATALOG` | **Multi-Format Metadata Catalog**: 17-field normalized catalog in `catalog.json`, `catalog.csv`, `catalog.sqlite`, and `index.html`. | M2 | R2, Criteria 58 |
| 7 | `FEAT-MEDIA-INGEST` | **Drop-Folder & CLI Ingest Tool**: `drop_folder/` and `tools/ingest.py` supporting `--dry-run`, deduplication, and metadata extraction. | M2 | R2, Criteria 59 |
| 8 | `FEAT-WIKI-PAGES` | **24-Page Biographical & Art Wiki**: Structured markdown articles across `bio/`, `family/`, `places/`, `art/`, and `travels/`. | M3 | R3, Criteria 63 |
| 9 | `FEAT-WIKI-PROVENANCE` | **Source Attribution & Citations**: Every claim backed by structured footnotes; uncertain facts flagged for family curation. | M3 | R3, Criteria 63 |
| 10 | `FEAT-WIKI-INTEGRITY` | **Zero Broken Links Guarantee**: Verified 100% link resolution across the entire wiki corpus via automated checker. | M3 | R3, Criteria 62 |
| 11 | `FEAT-REP-DELIVERY` | **Delivery & Longevity Plan**: `reports/DELIVERY_LONGEVITY_PLAN.md` detailing Dual-Release architecture, PWA, privacy gate, and offline bundle. | M4 | R4, Criteria 67 |
| 12 | `FEAT-REP-SPATIAL` | **Spatial Concept & Precedents**: `reports/SPATIAL_CONCEPT_AND_PRECEDENTS.md` citing ≥5 real projects with URLs and 5-stage AI pipeline. | M4 | R4, Criteria 66, 67 |
| 13 | `FEAT-PROTO-VIEWER` | **2.5D Multi-Plane Spatial Prototype**: Responsive Three.js viewer with 2 real scenes (Herzliya yard and Denya studio/chamber). | M5 | R5, Criteria 71, 72 |
| 14 | `FEAT-PROTO-E2ETEST` | **Headless Playwright Test Suite**: Automated verification across desktop (1440x900) and mobile (390x844) viewports. | M5 | R5, Criteria 70 |
| 15 | `FEAT-SEC-READONLY` | **Strict Read-Only Guardrails**: 100% non-mutating inspection of WhatsApp archives, browser profiles, and local databases. | Global | Acceptance Criteria 54 |

---

## 3. Milestones & Dependency Graph

```
[M1: Governance & Verification Baseline]
         |
         +-----------------------+
         |                       |
         v                       v
[M2: Media Archive & Ingestion] [M4: Delivery & Spatial Reports]
         |                       |
         v                       |
[M3: Biographical & Art Wiki]    |
         |                       v
         +------------------> [M5: Playable 2.5D Spatial Prototype]
                                 |
                                 v
                     [E2E: Full Suite Validation]
```

| Milestone | Scope | Deliverables | Verification Command | Dependencies |
|---|---|---|---|---|
| **M1** | Project Governance & Tooling (R1) | Root `MISSION.md`, `TODO.md`, `PROGRESS.md`, `PROJECT.md`, `tools/verify_wiki_links.py`, `tests/test_governance.py` | `python3 -m unittest tests/test_governance.py` | None |
| **M2** | Media Archive & Ingestion Pipeline (R2) | ≥100 harvested images, `catalog.{json,csv,sqlite,html}`, `drop_folder/`, `tools/ingest.py`, `tests/test_ingest.py` | `python3 tests/test_ingest.py` | M1 |
| **M3** | Biographical & Artistic Wiki (R3) | 24+ wiki pages across 5 directories, citations, uncertainty flags, link checker validation | `python3 tools/verify_wiki_links.py` | M1, M2 |
| **M4** | Delivery & Spatial Concept Reports (R4) | `reports/DELIVERY_LONGEVITY_PLAN.md` and `reports/SPATIAL_CONCEPT_AND_PRECEDENTS.md` | Auditor rubric verification | M1 |
| **M5** | Playable 2.5D Spatial Prototype (R5) | WebGL 2.5D interactive prototype, 2 real scenes, depth maps, inpainted plates, Playwright test | `npx playwright test prototype/tests/prototype.spec.js` | M2, M4 |
| **E2E** | Independent Opaque-Box E2E Suite | `tools/run_e2e_tests.py`, test runners, and `TEST_READY.md` verification report | `python3 tools/run_e2e_tests.py` | M1–M5 |

---

## 4. Module Interface Contracts

### 4.1 Media Metadata Catalog Contract
- **Locations:** `catalog/catalog.json`, `catalog/catalog.csv`, `catalog/catalog.sqlite`, `catalog/index.html`
- **Schema (17 normalized fields):**
  1. `id` (string, UUIDv4 or formatted slug, e.g. `media-wp-001`)
  2. `filename` (string, base name, e.g. `herzliya_yard_1965.jpg`)
  3. `rel_path` (string, relative path from root, e.g. `media/wordpress_blog/herzliya_yard_1965.jpg`)
  4. `sha256` (string, 64-char hex hash)
  5. `width` (integer, pixels >= 150)
  6. `height` (integer, pixels >= 150)
  7. `file_size` (integer, bytes)
  8. `mime_type` (string, e.g. `image/jpeg`, `image/png`, `image/webp`)
  9. `source_group` (enum string: `wordpress_blog`, `darchei_etzim_pdfs`, `portfolio_site`, `binyamin_gallery`, `museum_press`, `whatsapp_family`)
  10. `source_url_or_path` (string, originating URL or local PDF page identifier)
  11. `title` (string, descriptive title in English / Hebrew)
  12. `caption` (string, primary excerpt or image caption)
  13. `date_or_era` (string, e.g. `1965`, `2016-03`, `circa 1980s`)
  14. `people` (JSON array of strings / pipe-separated string)
  15. `places` (JSON array of strings / pipe-separated string)
  16. `art_themes` (JSON array of strings / pipe-separated string)
  17. `confidence_notes` (string, curation notes or verification status)

### 4.2 Drop-Folder Ingestion Script Contract (`tools/ingest.py`)
- **CLI Synopsis:**
  ```bash
  python3 tools/ingest.py [OPTIONS] [FILE_OR_DIR ...]
  ```
- **Supported Flags:**
  - `--drop-dir <path>`: Batch ingests all image files found in the specified directory (defaults to `drop_folder/`).
  - `--dry-run`: Runs full validation, dimension decoding, and SHA-256 calculation without modifying catalog files or moving/copying assets.
  - `--min-size <int>`: Minimum dimension threshold on both axes (default: `150`).
  - `--catalog-json <path>`: Target JSON catalog path (default: `catalog/catalog.json`).
  - `--catalog-csv <path>`: Target CSV catalog path (default: `catalog/catalog.csv`).
  - `--catalog-sqlite <path>`: Target SQLite database path (default: `catalog/catalog.sqlite`).
- **Exit Codes:**
  - `0`: Success (all images ingested or dry-run passed).
  - `1`: Validation error (corrupt image, duplicate hash, or invalid CLI arguments).

### 4.3 Wiki Pages Manifest & Frontmatter Contract
All wiki pages must reside under `wiki/` and adhere to GitHub-flavored Markdown with standard YAML frontmatter:

#### The 24 Wiki Pages Manifest:
1. `wiki/README.md` (Master index and interactive navigation graph)
2. `wiki/bio/index.md` (Biography portal)
3. `wiki/bio/tamar-sheaffer.md` (Master biographical profile & 70th milestone)
4. `wiki/bio/early-life-herzliya.md` (Herzliya childhood, parents' values, Technion)
5. `wiki/bio/engineering-and-feminism.md` (Elbit & Rafael career, workplace feminism)
6. `wiki/bio/inner-life-and-worldview.md` (Existential anxiety, politics, family care)
7. `wiki/family/index.md` (Family and personal relationships portal)
8. `wiki/family/ahuva-and-michael-fried.md` (Parents: Ahuva Lubetkin, Michael Parita, kindergarten)
9. `wiki/family/erela-fried.md` (Sister Er'ela: laughter, creativity, memory)
10. `wiki/family/gad-sheaffer.md` (Husband Gadi: Technion, "דרכי עצים", woodworking)
11. `wiki/family/sons-guy-and-tal.md` (Sons Guy & Tal: births, life paths, devotion)
12. `wiki/family/granddaughters-yaara-carmel.md` (Granddaughters Ya'ara & Carmel: joy, games)
13. `wiki/family/close-friends-beni-peri.md` (Lifelong friends Beni & Peri)
14. `wiki/places/index.md` (Lived places portal)
15. `wiki/places/herzliya-home-and-yard.md` (Childhood home & kindergarten yard)
16. `wiki/places/haifa-arlozorov-87.md` (Arlozorov 87, Hadar: young family, Guy's birth)
17. `wiki/places/haifa-einstein.md` (Einstein St., Ahuza: second home, Tal's birth)
18. `wiki/places/haifa-greenboim-80-denya.md` (80 Greenboim St., Denya: home & art studio)
19. `wiki/places/paris-cite-studio.md` (Paris Cité residency studio on the Seine)
20. `wiki/art/index.md` (Artistic oeuvre portal)
21. `wiki/art/education-midrasha-haifa.md` (Art training: HaMidrasha & Haifa MFA)
22. `wiki/art/artist-statement-philosophy.md` (Viewpoint-dependence, non-hollow, Sisyphean)
23. `wiki/art/materials-paper-and-cement.md` (Paper, papier-mâché, cement, earth, conduits)
24. `wiki/art/motifs-capsules-and-mirrors.md` (Capsules, enclosures, distorted reflections)
25. `wiki/art/binyamin-gallery-cooperative.md` (Founding member, cooperative gallery)
26. `wiki/art/solo-exhibition-bene-beitcha.md` (Ramat Gan Museum exhibition: "בנה ביתך")
27. `wiki/art/exhibitions-chronology.md` (Annotated chronological exhibitions catalog)
28. `wiki/travels/index.md` (Travels portal)
29. `wiki/travels/montreal-and-new-york.md` (Visits to Tal, North American travels)
30. `wiki/travels/southern-france-and-paris.md` (Paris residency & Provence/Arles with Guy)
31. `wiki/travels/bogliasco-italy-residency.md` (Ligurian fellowship & Mediterranean light)

#### Frontmatter Schema:
```yaml
---
title: "Article Title | כותרת בעברית"
slug: "slug-identifier"
category: "bio|family|places|art|travels"
era: "Time Period"
people: ["Name 1", "Name 2"]
related_places: ["../places/target.md"]
related_art: ["../art/target.md"]
sources:
  - type: "blog|pdf|site|museum|user_seed"
    title: "Source Title"
    ref: "URL or Page"
verification_status: "verified|pending_family_review"
---
```

### 4.4 Spatial Prototype Contract (`prototype/`)
- **Technology Stack:** HTML5, CSS3, Three.js (WebGL 2.0 via local ES Modules).
- **Entry Point:** `prototype/index.html` (runnable via static HTTP server, e.g. `python3 -m http.server 8080`).
- **Scene Requirements:**
  - **Scene 1 (The Herzliya Yard & Workshop):** Multi-plane reconstruction of the sunlit childhood yard, Michael's woodworking bench, and paper armatures.
  - **Scene 2 (The Denya Studio & Distorted Chamber):** Studio overlooking Mount Carmel, heavy cement blocks, tubular conduit pipes, and a distorted mirror reflecting family moments.
- **Occlusion Handling:** All foreground cutouts must have corresponding LaMa-inpainted background plates to eliminate tearing or black holes during camera translation.
- **Automated Verification:**
  - Tested via Playwright: `prototype/tests/prototype.spec.js`.
  - Viewports: Desktop `1440x900` and Mobile `390x844`.
  - Assertions: 0 uncaught JavaScript errors, 0 failed asset requests (HTTP 404/500), and camera movement triggers measurable WebGL transform matrix changes.

### 4.5 Strict Read-Only Security Contract
- **WhatsApp Isolation:**
  - Sandboxed macOS store container `~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared` must not be accessed with write connections.
  - Headless Baileys directory `/Users/talsheaffer/compost-bot/auth` must only be inspected offline. No active WebSocket connections to WhatsApp servers may be established.
  - Zero messages, zero polls, zero mutations.
- **Browser State Safety:**
  - All Playwright or browser harvesting profiles must run with ephemeral storage or read-only clones, never modifying user sessions or causing auth alerts.

---

## 5. Complete Project File Layout & Code Layout

```
/Users/talsheaffer/code/tami/
├── ORIGINAL_REQUEST.md
├── MISSION.md
├── TODO.md
├── PROGRESS.md
├── PROJECT.md
├── catalog/
│   ├── catalog.json
│   ├── catalog.csv
│   ├── catalog.sqlite
│   └── index.html
├── media/
│   ├── wordpress_blog/
│   ├── darchei_etzim_pdfs/
│   ├── portfolio_site/
│   ├── binyamin_gallery/
│   ├── museum_press/
│   └── whatsapp_family/
├── drop_folder/
│   └── README.md
├── tools/
│   ├── ingest.py
│   ├── verify_wiki_links.py
│   └── run_e2e_tests.py
├── wiki/
│   ├── README.md
│   ├── bio/
│   │   ├── index.md
│   │   ├── tamar-sheaffer.md
│   │   ├── early-life-herzliya.md
│   │   ├── engineering-and-feminism.md
│   │   └── inner-life-and-worldview.md
│   ├── family/
│   │   ├── index.md
│   │   ├── ahuva-and-michael-fried.md
│   │   ├── erela-fried.md
│   │   ├── gad-sheaffer.md
│   │   ├── sons-guy-and-tal.md
│   │   ├── granddaughters-yaara-carmel.md
│   │   └── close-friends-beni-peri.md
│   ├── places/
│   │   ├── index.md
│   │   ├── herzliya-home-and-yard.md
│   │   ├── haifa-arlozorov-87.md
│   │   ├── haifa-einstein.md
│   │   ├── haifa-greenboim-80-denya.md
│   │   └── paris-cite-studio.md
│   ├── art/
│   │   ├── index.md
│   │   ├── education-midrasha-haifa.md
│   │   ├── artist-statement-philosophy.md
│   │   ├── materials-paper-and-cement.md
│   │   ├── motifs-capsules-and-mirrors.md
│   │   ├── binyamin-gallery-cooperative.md
│   │   ├── solo-exhibition-bene-beitcha.md
│   │   └── exhibitions-chronology.md
│   └── travels/
│       ├── index.md
│       ├── montreal-and-new-york.md
│       ├── southern-france-and-paris.md
│       └── bogliasco-italy-residency.md
├── reports/
│   ├── DELIVERY_LONGEVITY_PLAN.md
│   └── SPATIAL_CONCEPT_AND_PRECEDENTS.md
├── prototype/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── main.js
│   │   ├── scene_manager.js
│   │   └── vendor/
│   │       └── three.module.js
│   ├── assets/
│   │   ├── scenes/
│   │   │   ├── scene1_herzliya_fg.png
│   │   │   ├── scene1_herzliya_bg_inpainted.jpg
│   │   │   ├── scene2_denya_fg.png
│   │   │   └── scene2_denya_bg_inpainted.jpg
│   │   └── depth/
│   │       ├── scene1_herzliya_depth.png
│   │       └── scene2_denya_depth.png
│   └── tests/
│       └── prototype.spec.js
└── tests/
    ├── test_governance.py
    ├── test_ingest.py
    └── e2e/
        └── test_full_pipeline.py
```

---

## 6. Verification & Quality Assurance Commands

| Target | Test Command | Expected Result |
|---|---|---|
| **Governance Suite** | `python3 -m unittest tests/test_governance.py` | 100% pass: All root files exist, non-empty, and read-only strings enforced. |
| **Ingestion Pipeline** | `python3 tests/test_ingest.py` | 100% pass: CLI accepts sample image, extracts dimensions, updates catalogs, respects `--dry-run`. |
| **Wiki Link Checker** | `python3 tools/verify_wiki_links.py` | Exit code 0: 0 broken internal links, 0 orphaned pages across all 24+ wiki entries. |
| **Spatial Prototype** | `npx playwright test prototype/tests/prototype.spec.js` | 100% pass: Loads at 1440x900 & 390x844 with 0 uncaught errors and verified camera response. |
| **Opaque-Box E2E Suite** | `python3 tools/run_e2e_tests.py` | Generates `TEST_READY.md` confirming full R1–R5 criteria satisfaction. |
