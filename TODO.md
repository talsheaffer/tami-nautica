# Tamar "Tami" Sheaffer 70th-Birthday Project — Master TODO Tracker

> **Milestone Date:** 25 October 2026 | **Subject:** Tamar "Tami" Sheaffer (תמר שפר)  
> **Status:** Active Execution | **Mode:** Development (Strict Read-Only External Access)  
> **Repository:** `/Users/talsheaffer/code/tami`

---

## Progress Overview
| Milestone | Focus | Deliverables | Status |
|---|---|---|---|
| **Phase 0** | Discovery & Architectural Census | Survey reports, source mining, biographical & spatial blueprints | **COMPLETED** |
| **Milestone 1 (M1)** | Governance & Live Tracking Baseline (R1) | `MISSION.md`, `TODO.md`, `PROGRESS.md`, `PROJECT.md`, link checker | **COMPLETED** |
| **Milestone 2 (M2)** | High-Recall Media Archive & Pipeline (R2) | >=100 images, catalog (JSON/CSV/SQLite/HTML), drop-folder & `tools/ingest.py` | **COMPLETED** |
| **Milestone 3 (M3)** | Interconnected Biographical & Art Wiki (R3) | 24+ wiki pages, 0 broken links, footnote citations, uncertainty tags | **COMPLETED** |
| **Milestone 4 (M4)** | Delivery Architecture & Spatial/AI Report (R4) | Dual-Release plan, 5+ precedent analyses, concrete AI workflow report | **COMPLETED** |
| **Milestone 5 (M5)** | Playable 2.5D Spatial Prototype (R5) | WebGL 2.5D prototype, 2+ scenes, depth/inpainted layers, Playwright tests | **COMPLETED** |
| **E2E Testing** | Opaque-Box E2E Requirements Suite | Automated acceptance validation across R1–R5, `TEST_READY.md` | **COMPLETED** |

---

## Phase 0: Discovery & Architecture Baseline (COMPLETED)
- [x] Comprehensive review of `ORIGINAL_REQUEST.md` (biographical anchors, motifs, requirements R1–R5).
- [x] Quantitative harvesting census across all 6 primary candidate source groups:
  - [x] Father's WordPress Blog: 1,254 posts via REST API, 1,797 image URLs via XML sitemap.
  - [x] 16 Local Darchei Etzim PDFs: 4,632 total pages (~265 MB) spanning 2011–2025.
  - [x] Tami's Portfolio: 27 structured pages, 302 media items via REST API.
  - [x] Binyamin Gallery: 20 ultra-high-resolution artwork records via SSR extraction.
  - [x] Ramat Gan Museum of Israeli Art: solo exhibition "בנה ביתך" (April–Dec 2026) & press reviews.
  - [x] WhatsApp history & Baileys session: local file state verified; read-only sandbox boundaries mapped.
- [x] Local runtime benchmark verification: Node.js `v25.9.0`, npm `11.12.1`, Python `3.14.6`, Pillow `12.2.0`, Swift `PDFKit`.
- [x] Discovery handoff reports published in `.agents/teamwork/` (`survey_miner_sources`, `survey_explorer_bio`, `survey_explorer_spatial`).

---

## Milestone 1: Project Governance & Live Tracking Baseline (R1)
*Objective: Establish project mission, living tracking checklist, verbose progress log, and core verification tools.*

### 1.1 Governance Artifacts
- [x] Root `MISSION.md` authored in `/Users/talsheaffer/code/tami/`:
  - [x] Captures artistic and emotional core (70th birthday gift, tribute to Tami's dual life).
  - [x] Explains non-hollow spatial design philosophy (materiality, paper, cement, mirrors).
  - [x] Documents strict 100% read-only safety guardrails for family data.
  - [x] Outlines multi-year archival durability and longevity vision.
- [x] Root `TODO.md` initialized with structured live checklist covering all R1–R5 acceptance criteria.
- [x] Root `PROGRESS.md` initialized with 4 live tracking tables:
  - [x] Table 1: Source Harvesting Statistics (WordPress, PDFs, Portfolio, Binyamin, Ramat Gan, WhatsApp).
  - [x] Table 2: Access & Integrity Log (documenting 100% read-only verification across all systems).
  - [x] Table 3: Technical & Prototype Benchmarks (Node v25, Python 3.14 + Pillow, Playwright setup).
  - [x] Table 4: Family Verification & Curation Register (tracking all open biographical/artistic questions).
- [x] Root `PROJECT.md` maintained as architectural contract (module boundaries, schemas, file layout).

### 1.2 Verification Tooling & Quality Gate
- [x] Tool script `tools/verify_wiki_links.py` implemented:
  - [x] Validates all relative links in `wiki/`.
  - [x] Enforces frontmatter metadata presence.
  - [x] Returns exit code 0 on zero broken links; non-zero on broken link or empty target.
- [x] Tool script `tools/run_e2e_tests.py` scaffolding created for test track coordination.
- [x] M1 Acceptance Criteria Verification:
  - [x] `MISSION.md`, `TODO.md`, `PROGRESS.md` exist and are non-empty in repository root.
  - [x] Zero write queries, outbound messages, or account mutations occurred.
  - [x] Automated validation confirms governance files adhere to project specifications.

---

## Milestone 2: High-Recall Media Archive & Ingestion Pipeline (R2)
*Objective: Collect, validate, deduplicate, and catalog >=100 high-resolution images across >=4 sources; provide drop-folder ingestion tool.*

### 2.1 Media Harvesting & Storage
- [x] Harvest media from **WordPress Blog** (`theshingimel.wordpress.com`):
  - [x] Download candidate photographs (family, travels, portraits, Tami).
  - [x] Filter out UI chrome, social icons, and images `<150px` on both axes.
- [x] Harvest media from **Local Darchei Etzim PDFs** (`darchei_etzim_*.pdf`):
  - [x] Extract high-resolution photo plates and page spreads via Swift `PDFKit` & DCTDecode streams.
  - [x] Isolate images unique to the print annuals not found on the web.
- [x] Harvest media from **Tami's Portfolio Site** (`tamarsheaffer.site`):
  - [x] Extract original artwork photography across all 27 project pages.
- [x] Harvest media from **Binyamin Gallery** (`binyamin-gallery.com`):
  - [x] Download 20 ultra-high-resolution installation photographs (up to 4928x3264).
- [x] Harvest media from **Ramat Gan Museum & Press** (*Portfolio*, *Erev Rav*):
  - [x] Collect installation views of "בנה ביתך / The New Normal" (2026).
- [x] Deduplication & Storage Structure:
  - [x] Compute SHA-256 hash for every harvested image.
  - [x] Store deduplicated images in `media/<source_group>/<normalized_filename>`.
  - [x] Verify count: $\ge 100$ distinct, non-corrupt images across $\ge 4$ source groups (127 verified assets).

### 2.2 Metadata Catalog & Browsable Contact Sheet
- [x] Build canonical JSON catalog (`catalog/catalog.json`) with 17 normalized fields:
  - `id`, `filename`, `rel_path`, `sha256`, `width`, `height`, `file_size_bytes`, `mime_type`, `source_group`, `source_url_or_path`, `title`, `caption`, `date_or_era`, `people`, `places`, `art_themes`, `confidence_notes`.
- [x] Export spreadsheet catalog (`catalog/catalog.csv` with UTF-8 BOM).
- [x] Generate SQLite database catalog (`catalog/catalog.sqlite`) with indexed query support and view `media`.
- [x] Generate interactive visual contact sheet (`catalog/index.html`):
  - [x] Displays thumbnails, dimensions, source provenance, and tagged metadata.
  - [x] Zero broken relative image links.

### 2.3 Drop-Folder & CLI Ingestion Pipeline
- [x] Create drop-folder structure: `drop_folder/` and `drop_folder/README.md`.
- [x] Implement CLI ingestion tool: `tools/ingest.py`:
  - [x] Flags: `--drop-dir`, `--media-dir`, `--dry-run`, `--min-size 150`, tag flags.
  - [x] Zero external dependencies (Python 3.14 stdlib + Pillow).
  - [x] SHA-256 deduplication and dimension decoding.
  - [x] Atomic catalog updating (`catalog.json`, `catalog.csv`, `catalog.sqlite`, `index.html`).
- [x] Automated Testing (`tests/test_ingest.py` & `tests/test_media_archive.py`):
  - [x] Dry-run test on sample image (extracts dimensions, computes hash, simulates entry without disk mutation).
  - [x] Ingest test copying image to `media/drop_folder_imports/` and verifying catalog record.
  - [x] Duplicate rejection test asserting identical SHA-256 is not re-ingested.
  - [x] Dimension filtering and corrupted image rejection tests.

---

## Milestone 3: Interconnected Biographical & Artistic Wiki (R3)
*Objective: Build an interconnected, 24+ page Markdown wiki covering all biographical and artistic anchors, with zero broken links, footnote citations, and explicit uncertainty tags.*

### 3.1 Wiki Infrastructure & Portals
- [x] Initialize `wiki/README.md` with complete thematic graph and navigation index.
- [x] Portal pages:
  - [x] `wiki/bio/index.md` (Biography Portal)
  - [x] `wiki/family/index.md` (Family & Friends Portal)
  - [x] `wiki/places/index.md` (Lived Spaces Portal)
  - [x] `wiki/art/index.md` (Artistic Oeuvre Portal)
  - [x] `wiki/travels/index.md` (Travels & Residencies Portal)

### 3.2 Biography & Career Pages (`wiki/bio/`)
- [x] `tamar-sheaffer.md`: Master biography, 70th-birthday milestone anchor (25/10/1956–25/10/2026).
- [x] `early-life-herzliya.md`: Childhood in Herzliya, parents' values, early art passion, Technion choice.
- [x] `engineering-and-feminism.md`: Systems engineering at Elbit Systems and Rafael; outspoken workplace feminism.
- [x] `inner-life-and-worldview.md`: Existential anxieties, Israeli political turmoil, deep concern for family health.

### 3.3 Family & Friends Pages (`wiki/family/`)
- [x] `ahuva-and-michael-fried.md`: Parents Michael (Parita 1939, hands of gold, safety worrier, *Bold and the Beautiful*) & Ahuva (Lubetkin sister, Aliyat Hanoar 1938, kindergarten, yard).
- [x] `erela-sadan.md`: Sister Er'ela; unbreakable bond, telephone laughter, artistic talent, memory (`[Verification needed for family]` dates).
- [x] `gad-sheaffer.md`: Husband Gadi; Technion meeting, author of "דרכי עצים", woodworking, travels, intellectual partner.
- [x] `sons-guy-and-tal.md`: Sons Guy (born Arlozorov, father of granddaughters) and Tal (born Einstein, Montreal/NYC).
- [x] `granddaughters-yaara-carmel.md`: Cherished granddaughters Ya'ara and Carmel; play, K-pop dances, Haifa boars, swings.
- [x] `close-friends-beni-peri.md`: Lifelong close friends Beni and Peri (`[Verification needed for family]` anecdotes).

### 3.4 Lived Places Pages (`wiki/places/`)
- [x] `herzliya-home-and-yard.md`: Childhood home, carpentry bench, kindergarten yard, emotional resonance.
- [x] `haifa-arlozorov-87.md`: First married home (corner Bar-Giora, Hadar), young family, Guy's birth.
- [x] `haifa-einstein.md`: Second family home (Ahuza), Carmel ridge, Tal's birth.
- [x] `haifa-greenboim-80-denya.md`: Current sanctuary (Denya), custom home, lush garden, art studio.
- [x] `paris-cite-studio.md`: 6-month artist residency at Cité Internationale des Arts (2016); studio overlooking the Seine.

### 3.5 Artistic Oeuvre Pages (`wiki/art/`)
- [x] `education-midrasha-haifa.md`: Early retirement transition, HaMidrasha (2005–2009), Haifa MFA (2009–2011).
- [x] `artist-statement-philosophy.md`: Viewpoint-dependent reality, low-tech Sisyphean labor, mock-ups and simulacra.
- [x] `materials-paper-and-cement.md`: Tactile paper pulp, papier-mâché, cast concrete, earth, drafting blueprints.
- [x] `motifs-capsules-and-mirrors.md`: Enclosed capsules, impossible scaffoldings, pipes, broken/blurry reflections.
- [x] `binyamin-gallery-cooperative.md`: Founding cooperative member, artist-run gallery in Tel Aviv.
- [x] `solo-exhibition-bene-beitcha.md`: Crowning solo exhibition "בנה ביתך / The New Normal" (Ramat Gan Museum, 2026).
- [x] `exhibitions-chronology.md`: Complete chronological catalog of all solo and group exhibitions.

### 3.6 Travels & Residencies Pages (`wiki/travels/`)
- [x] `montreal-and-new-york.md`: Visits to Tal in Montreal and NYC; North American architecture and museums.
- [x] `southern-france-and-paris.md`: Provence, Arles, Le Corbusier with Guy; Paris architectural explorations.
- [x] `bogliasco-italy-residency.md`: 2020 Bogliasco Foundation Fellow (Genoa, Italy); Mediterranean reflections.

### 3.7 Wiki Verification & Quality Gate
- [x] Execute `python3 tools/verify_wiki_links.py`:
  - [x] Confirms 100% zero broken internal relative links across all 24+ pages (360 links, 31 wiki pages + 5 root docs).
- [x] Independent audit confirms:
  - [x] Every factual claim cites a concrete source (blog URL, PDF page, site, catalog, or user seed).
  - [x] All uncertain details (Er'ela's dates, Beni/Peri anecdotes) flagged with `[Verification needed for family]`.
  - [x] Hebrew typography formatted cleanly with proper RTL directionality.

---

## Milestone 4: Delivery Architecture & Spatial/AI Concept Report (R4)
*Objective: Deliver two deeply researched planning documents evaluating delivery formats and formulating the non-hollow 2.5D spatial/AI concept.*

### 4.1 Delivery & Longevity Plan (`reports/DELIVERY_LONGEVITY_PLAN.md`)
- [x] Format Evaluation Matrix comparing:
  - [x] Responsive Web App / PWA
  - [x] Self-Contained Offline Portable Bundle
  - [x] Native / Hybrid Desktop & Mobile (Tauri / Electron / Capacitor)
- [x] Detailed evaluation across:
  - [x] Desktop vs Phone UX (mouse pointer tilt vs touch swipe / `DeviceOrientation` gyroscope).
  - [x] Family privacy & surprise sharing (client-side passphrase gate vs offline USB delivery).
  - [x] Archival durability (10–20+ year zero-rot preservation without fragile cloud dependencies).
- [x] Concrete Recommendation: The Dual-Release Architecture:
  - [x] Passphrase-protected PWA for immediate birthday sharing.
  - [x] Immortal, self-contained offline bundle on physical commemorative USB-C drive.

### 4.2 Spatial Concept, Precedents & AI Workflow Report (`reports/SPATIAL_CONCEPT_AND_PRECEDENTS.md`)
- [x] Online Precedents Analysis (analyzing >=5 real projects with active URLs):
  - [x] *3D Photo Inpainting* (Shih et al., CVPR 2020) — Layered Depth Images & occlusion synthesis.
  - [x] *Depth Anything V2* (Yang et al., 2024) — Monocular boundary depth estimation.
  - [x] *Zoomquilt & Arkadia* (Baumgarten et al.) — Infinite multi-plane canvas traversal.
  - [x] *Depthy & Spite Player* (Lindemann / Cabello) — Real-time WebGL depth raymarching.
  - [x] *Antimatter15 Splats* (Kwok) — In-browser 3D Gaussian Splatting.
  - [x] *The Boat* (SBS Australia / Huynh) — Tactile multi-plane graphic narrative.
- [x] Concrete Spatial Concept: "ארכיטקטורה של דאגה ושבריריות":
  - [x] Scene 1: Herzliya Yard & Workshop (carpentry bench, citrus trees, wire mesh & paper).
  - [x] Scene 2: Denya Studio & Distorted Chamber (cement blocks, tubular pipes, warped mirror revealing family).
- [x] Multi-Stage AI Production Pipeline:
  - [x] Super-resolution restoration (Real-ESRGAN / Magnific AI).
  - [x] Foreground alpha segmentation (SAM 2 / rembg).
  - [x] 16-bit metric depth estimation (Depth Anything V2 Large).
  - [x] Dilated-mask background inpainting (LaMa Fast Fourier Convolutions) to guarantee non-hollow exploration.
  - [x] Evaluation of candidate subscription tools (Replicate GPU, Adobe Firefly, Google Gemini 1.5 Pro/Ultra).
- [x] Rubric Audit: confirms actionable comparison tables, concrete code snippets, zero generic prose.

---

## Milestone 5: Playable 2.5D Spatial Prototype & Headless Verification (R5)
*Objective: Build a playable, minimalist 2.5D spatial journey prototype with 2+ real scenes, multi-plane depth/inpainting, camera interaction, and headless Playwright verification.*

### 5.1 Prototype Core Engine (`prototype/`)
- [x] Set up lightweight Three.js WebGL 2.0 rendering engine (`prototype/index.html`, `prototype/js/app.js`, `prototype/js/scene_manager.js`, `prototype/js/vendor/three.module.js`).
- [x] Implement 4-plane scene architecture:
  - [x] Layer 0: Inpainted background plate ($Z = -350$, dynamic widescreen scale $S_0 \ge 1.875$, zero boundary holes).
  - [x] Layer 1: Midground architectural structures ($Z = -150$, scale compensation $1.375\times$).
  - [x] Layer 2: Segmented foreground subjects/sculptures ($Z = 0$, alpha cutout).
  - [x] Layer 3: Atmospheric particles & floating dust vignette ($Z = +100$, additive blending).
- [x] Implement Scene 1: חצר הילדות בהרצליה (Herzliya Childhood Yard & Michael's carpentry workshop).
- [x] Implement Scene 2: הסטודיו בדניה והמבנה המעוות (Denya Studio & "בנה ביתך" Ramat Gan Museum curved installation).
- [x] Pre-process real archival photos with segmented alpha cutouts and dilated-mask inpainting (`prototype/build_assets.py`).

### 5.2 Interaction & Transitions
- [x] Desktop pointer parallax interaction: smooth camera tilt based on normalized mouse coordinates with damping ($\lambda = 0.05$).
- [x] Mobile touch & gyroscope interaction: touch drag and device orientation tilt.
- [x] Clamped motion bounds ($[-45, +45]$ X, $[-25, +25]$ Y) preventing edge tearing.
- [x] Dynamic Widescreen Frustum Scaling (M4 Reviewer 2 Finding 1) preventing black borders on 16:9 and 21:9 ultrawide screens.
- [x] Thematic Spatial Transition:
  - [x] GLSL ripple/reflection distortion shader simulating Tami's curved mirror installations (`assets/shaders/warped_reflection_transition.glsl`).
  - [x] Implemented M4 Reviewer 2 Finding 2 fix (`max(0.0, 1.0 - dist)`) preventing corner negative color falloff.
  - [x] Smooth cross-dissolve between Scene 1 and Scene 2 triggered by UI selector or keyboard shortcuts (1, 2).
- [x] Minimalist UI Chrome:
  - [x] Clean RTL Hebrew typography (`dir="rtl"`, `lang="he"`, Heebo & Assistant fonts).
  - [x] Scene navigation pills, auxiliary audio/gyro controls, and interpretive curatorial story drawer.
  - [x] Standalone HTTP server with port collision avoidance (`prototype/serve.py`).

### 5.3 Headless Browser & Automated Verification (`tests/test_prototype.py`, `prototype/tests/prototype.spec.js`)
- [x] Desktop Viewport Test ($1440 \times 900$):
  - [x] Loads prototype server with zero uncaught JavaScript / WebGL console errors.
  - [x] Zero failed asset requests (HTTP 404 / 500).
  - [x] Verifies active WebGL context (`webgl2` or `webgl`).
  - [x] Simulates mouse movement (center, top-left, bottom-right) and asserts camera matrix translation ($\Delta X > 20$, $\Delta Y > 10$).
- [x] Scene Transition Test:
  - [x] Triggers transition to Denya scene and asserts prototype state updates cleanly to `denya`.
- [x] Mobile Viewport Test ($390 \times 844$, touch emulation):
  - [x] Loads without errors under mobile user-agent.
  - [x] Asserts `dir="rtl"` and Hebrew computed styles on UI elements.
  - [x] Verifies mobile state responsiveness.
- [x] Full automated test suite passes: 87 unit tests in `python3 -m unittest discover -s tests` (87/87 passing).
- [x] Milestone 5 Remediation & Reviewer Hardening:
  - [x] Removed external Google Fonts `@import` from `prototype/css/style.css` in favor of Hebrew-optimized native system font stack (100% offline self-containment for 20-year archival longevity).
  - [x] Implemented WebGL resource cleanup `dispose()` in `SpatialSceneManager` (cancels RAF, unbinds window event listeners, disposes geometries, materials, textures, render targets, forces context loss).
  - [x] Handled iOS Safari `DeviceOrientationEvent.requestPermission()` asynchronous Promise with synchronized UI toggle feedback.
  - [x] Guarded `switchScene()` / `transitionTo()` against rapid clicks and lockups.
  - [x] Replaced fixed `time.sleep()` calls in `tests/test_prototype.py` with active polling on `window.__tamiSceneManager.layersReady`, adding explicit assertions for $\Delta X > 20$, $\Delta Y > 10$, camera matrix shifts, and zero WebGL errors (`gl.getError() == 0`).
  - [x] Hardened `tests/test_adversarial_m5_longevity.py` with active transition polling, XML namespace exemption, and dynamic frustum scaling $\ge 1.875$ accounting for macOS window manager height clamping.

---

## E2E Testing Track: Opaque-Box Acceptance Suite
*Objective: Independent requirements-driven validation asserting all acceptance criteria before sign-off.*

- [x] Execute master test runner suites (`python3 -m unittest discover -s tests` & `python3 tools/verify_wiki_links.py --wiki-dir wiki/ --strict-orphans`):
  - [x] Tier 1: Governance & Read-Only Safety (R1).
  - [x] Tier 2: Media Archive, Catalog & Drop-Folder Ingestion (R2).
  - [x] Tier 3: Wiki Structure, 0 Broken Links & Biographical Coverage (R3).
  - [x] Tier 4: Delivery Plan & Spatial Precedent Verification (R4).
  - [x] Tier 5: Headless Playwright Prototype Execution & Viewport Parity (R5).
- [x] Verify 100% pass across all test assertions (87/87 tests pass, 360/360 links pass).
- [x] Publish `TEST_READY.md` confirming project completion.
