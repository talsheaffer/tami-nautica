# Original User Request

## 2026-10-26T01:00:15Z

Build the research foundation, high-recall tagged media archive, biographical and artistic wiki, technical/conceptual plans, and a small playable 2.5D spatial journey prototype for a surprise 70th-birthday virtual photo album gift honoring Israeli installation artist and retired systems engineer Tamar "Tami" Sheaffer (תמר שפר). The goal is to merge her life places, family biography, and relationships with her tactile artistic world in an atmospheric, minimalist, non-hollow interactive exploration.

Working directory: /Users/talsheaffer/code/tami
Integrity mode: development

## Reference Material & Biographical Seeds

### Sources to Harvest
- **Father's WordPress Blog ("דרכי עצים"):** `https://theshingimel.wordpress.com/` — rich family history and photos, including Tami's parents (אהובה and מיכאל), her sister (אראלה), travels, homes, Paris residency, and frequent mentions of Tami as "זוגתי שתחיה".
- **Local Bookified Blog PDFs on Mac:** `/Users/talsheaffer/darchei_etzim_*.pdf` (covering 2011–2025) — cross-reference with the live WordPress site because some images or posts may only appear in one or the other.
- **Tami's Portfolio & Social Media:** `https://tamarsheaffer.site/` and her Instagram account.
- **Binyamin Gallery:** `https://www.binyamin-gallery.com/` (Tami is a founding member).
- **Exhibitions, Press & Interviews:** Online reviews, catalogs, interviews, curator texts, and coverage of her exhibitions—especially her current/recent exhibition at the **Ramat Gan Museum of Israeli Art**.
- **Local WhatsApp (`שפר` family chat):** Local macOS WhatsApp storage (`~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared`) and headless Baileys session (`/Users/talsheaffer/compost-bot/auth`, documented in `/Users/talsheaffer/.gemini/config/skills/whatsapp/SKILL.md`).
- **Google Photos & Instagram (Login-Gated Sources):** Attempt read-only harvesting via existing logged-in browser profiles / Playwright on this Mac where accessible without intrusive auth prompts; additionally provide a clean drop-folder and ingestion script for importing Google Photos exports/shared albums later.

### Biographical & Thematic Seeds (Verify, Expand & Cross-Link)
- **Milestone:** 70th birthday on 25/10/2026 — Tamar Sheaffer (תמר שפר), known as Tami (תמי).
- **Parents & Childhood in Herzliya:** Parents Michael (מיכאל) and Ahuva (אהובה) were "מלח הארץ", warm, wonderful people. Michael had "hands of gold", was the caring/concerned safety-minded worrier, and loved *The Bold and the Beautiful* (היפים והאמיצים). Ahuva was wonderful with children and ran a kindergarten. Their house in Herzliya had a lovely yard and strong emotional resonance.
- **Early Life, Technion & Haifa Homes:** Grew up in Herzliya; loved art from an early age, chose the "practical" path at the Technion where she met her husband (the user's father). Lived in three Haifa homes: **Arlozorov** (where older son **Guy / גיא** was born), **Einstein** (where younger son **Tal** was born), and **80 Greenboim St., Denya (דניה - גרינבוים 80)** (current home, which also houses her art studio).
- **Engineering Career:** Worked for decades as a systems engineer in defense tech—first at **Elbit**, then **Rafael**. Worked hard to prove herself and succeeded; was an outspoken workplace feminist who cared deeply about how women were treated professionally.
- **Art Career & HaMidrasha:** Took early retirement to dedicate herself to art, studying at **HaMidrasha (המדרשה)** and building an active artistic career. Stayed in Paris for 6 months with her husband for an artist residency at the **Cité Internationale des Arts**. Works intensely and sometimes gets consumed by preparation leading up to exhibitions.
- **Sister, Family, Friends & Travels:** Deeply close to her sister **Er'ela (אראלה)**—they spoke on the phone constantly and laughed endlessly; Er'ela also had an artistic side and passed away a few years ago. Has two cherished granddaughters from Guy, **Ya'ara (יערה)** and **Carmel (כרמל)**, whom she loves playing with. Close friends include **Beni (בני)** and **Peri (פרי)**. Traveled widely abroad with her husband, including visiting Tal in Montreal, meeting him in New York during his two years in Montreal, and a recent trip to Southern France where Guy joined them.
- **Inner Life & Worries:** Spends a lot of energy worrying—about Israel, its future, Bibi/politics, and the health of her husband, herself, Tal, and Tal's wife.
- **Artistic Sensibilities & Motifs:** Highly visual and tactile. Loves **paper and papier-mâché**, as well as heavy "substance" (**cement, earth**). Creates **installation art** often resembling misshapen furniture or architectural structures, **pipes, chambers, enclosures**, and recurring **mirrors, broken reflections, and blurry/distorted reflections**. Conceptually tied to anxiety about the future—the feeling that "the home" (Israel) is falling apart and essence is being replaced by stand-ins, mock-ups, and simulacra.

## Requirements

### R1. Project Mission, Todo Tracker & Verbose Live Progress Report
Create and actively maintain three project management documents inside the working directory throughout execution: (1) a clear project mission document capturing the artistic, emotional, and technical vision; (2) a concise live todo checklist; and (3) a verbose live progress report logging source-by-source findings, harvesting statistics, technical experiments, and open questions for family curation.

### R2. High-Recall Tagged Media Archive & Reusable Ingestion Pipeline
Collect candidate photographs, artwork images, and primary text excerpts with high recall across the father's WordPress blog, the local `darchei_etzim_*.pdf` books, Tami's portfolio site, Binyamin Gallery, Ramat Gan Museum and press/interview pages, local WhatsApp `שפר` history, and—where reachable via existing local browser state—Instagram and Google Photos. Store downloaded images locally (deduplicated at highest available resolution, excluding tiny UI chrome/icons) with a structured, searchable metadata catalog tagging every item by source/URL, content description, people, places/homes, approximate date/era, artistic materials/motifs, and confidence/curation notes; also provide a drop-folder and working CLI ingestion script so future Google Photos or manual exports are automatically indexed into the same catalog.

### R3. Interconnected Biographical & Artistic Wiki
Synthesize all harvested material and the user's biographical seeds into a structured, hyperlinked biographical and artistic wiki (supporting Hebrew names/quotes cleanly) covering Tami's timeline, places/homes, family members, close friends, engineering career, artistic education and exhibitions, residencies and travels, personality, and material/conceptual themes. Every claim must link to its concrete source (blog post URL, PDF year/page, article/interview, WhatsApp context, or user seed note), explicitly flagging gaps or uncertain dates for family verification rather than guessing.

### R4. Delivery Architecture Plan & Non-Hollow 2.5D / AI Spatial Concept Report
Deliver two deeply researched planning documents:
1. **Delivery & Longevity Plan:** Compare candidate delivery formats (responsive web app / PWA, self-contained offline/portable bundle, native/hybrid options) across desktop and phone UX, touch vs pointer controls, privacy for family photos, ease of sharing as a surprise gift, and multi-year archival durability (zero fragile server dependencies that could break later), ending with a concrete recommendation.
2. **Spatial Concept, Precedent & AI Workflow Report:** Research and cite real online precedents (interactive documentaries, 2.5D parallax/multi-plane photo narratives, Zoomquilt/infinite-canvas journeys, single-image Gaussian splat / depth-inpainted worlds, museum/artist digital experiences) and design a concrete visual/technical concept for moving through Tami's key spaces (especially the Herzliya house/yard and the Denya house/studio) blended with her installation motifs (paper, cement, enclosures, distorted reflections). Evaluate open-source libraries alongside practical AI workflows (segmentation, monocular depth, layered inpainting/outpainting, Gemini Ultra image generation, and optional paid tools/subscriptions worth getting).

### R5. Small Playable 2.5D Spatial Prototype & Strict Read-Only Guardrails
Build a minimalist, elegant playable prototype (runnable locally and usable on both desktop and phone viewports) featuring 1–2 real scenes constructed from collected photos/artworks of Tami's spaces and installations—using local open-weight vision/depth/segmentation/inpainting tools to achieve genuine multi-plane perspective and at least one thematic transition without feeling like a hollow 3D box. **Strict Infrastructure & Privacy Constraint:** All access to local WhatsApp data (`~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared` and `/Users/talsheaffer/compost-bot/auth`), browser profiles, and personal files must be **100% read-only**—never send messages, create polls, modify databases, or trigger any external notification that could reveal the surprise.

## Acceptance Criteria

### Project Documentation & Read-Only Safety
- [ ] Mission document, concise todo document, and verbose progress report exist in `/Users/talsheaffer/code/tami`, are non-empty, and the progress report records concrete per-source harvesting counts, access outcomes (including what succeeded vs required fallback on login-gated sources), and prototype iteration notes.
- [ ] Zero outbound messages, write queries, or account mutations occur on WhatsApp or any personal account; any local database inspection uses read-only connections or copies.

### High-Recall Media Archive & Drop-Folder Ingestion
- [ ] Automated validation confirms at least 100 distinct, non-corrupt candidate image files (verified via image header/dimension decoding, filtering out tiny icons `<150px` on both axes) stored locally under `/Users/talsheaffer/code/tami` across at least 4 distinct source groups (WordPress blog, local `darchei_etzim` PDFs, `tamarsheaffer.site` / `binyamin-gallery.com`, museum/press articles, and/or local WhatsApp `שפר` media).
- [ ] 100% of archived images appear in a structured machine-readable catalog (JSON/SQLite/CSV) + human-browsable index (Markdown or HTML contact sheet) populated with source provenance, content summary, and structured tags (people, places, era, art/material themes), with zero broken file paths or exact SHA-256 duplicates.
- [ ] A dedicated drop-folder and ingestion script exist, documented with usage instructions, and pass an automated dry-run test that ingests a sample image, extracts dimensions/hash, and appends a valid catalog entry.

### Biographical & Artistic Wiki
- [ ] Automated link checker confirms zero broken internal links across all wiki pages.
- [ ] Independent auditor rubric confirms dedicated, source-cited entries/sections covering all user-provided anchors: parents (Michael & Ahuva), sister (Er'ela), immediate family & granddaughters (husband, Guy, Tal, Ya'ara, Carmel), friends (Beni, Peri), Herzliya childhood home/yard, Haifa homes (Arlozorov, Einstein, 80 Greenboim St. Denya + studio), Elbit & Rafael career + feminism, HaMidrasha, Paris Cité residency, Ramat Gan Museum & Binyamin Gallery work, travels (Montreal, NYC, Southern France, etc.), and material/conceptual art motifs (paper/papier-mâché, cement/earth, misshapen structures/pipes/chambers, broken/blurry reflections, anxiety over "the home"/Israel & simulacra), with unverified details explicitly flagged rather than fabricated.

### Delivery Plan, Precedents & Spatial Concept Report
- [ ] Concept report cites at least 5 real, verifiable existing projects/techniques/demos with valid URLs demonstrating non-hollow 2.5D parallax, depth-inpainted photo exploration, spatial scrollytelling, or single-image 3D/splat storytelling.
- [ ] Independent auditor rubric confirms both planning documents provide actionable comparisons (libraries, asset pipeline steps, Gemini Ultra + candidate subscription workflows, mobile vs desktop UX, privacy, and zero-rot archival delivery) rather than generic high-level prose.

### Playable 2.5D Spatial Prototype
- [ ] Automated headless browser test (Playwright) launches the local prototype server and loads the experience at both desktop (`1440x900`) and mobile (`390x844`) viewports with zero uncaught JS/WebGL errors and zero failed (`404`/`500`) asset requests.
- [ ] Prototype uses real collected images from the archive (minimum 2 distinct scenes/layers combining lived place + artwork/installation), pre-processed with real depth maps and/or foreground-background segmentation + background fill/inpainting stored in the project.
- [ ] Automated interaction capture across distinct scroll/pointer/touch camera states (verified by independent visual auditor) confirms visible multi-plane parallax perspective shift without raw un-inpainted background holes during normal navigation, at least one working spatial/thematic transition between scenes, minimalist UI chrome, and proper RTL rendering for any Hebrew text.
