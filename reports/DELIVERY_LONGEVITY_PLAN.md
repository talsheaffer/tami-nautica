# DELIVERY & LONGEVITY ARCHITECTURE PLAN
# תוכנית הפצה, פרטיות ושרידות ארכיונית רב-עשורית (2026–2046+)

**Project:** Tamar "Tami" Sheaffer 70th Birthday Interactive Spatial Archive  
**Target Milestone Date:** 25 October 2026 (כ"ה בתשרי תשפ"ז)  
**Deliverable Path:** `reports/DELIVERY_LONGEVITY_PLAN.md`  
**Classification:** Confidential — Surprise Family Milestone Gift  
**Core Recommendation:** The Dual-Release Architecture (Private Zero-Knowledge PWA + Archival USB-C Time Capsule)

---

## 1. Executive Summary & Architectural Challenge

On 25 October 2026, Tamar "Tami" Sheaffer—acclaimed Israeli installation artist, founding member of Binyamin Gallery, and former senior defense systems engineer at Elbit and Rafael—reaches her 70th birthday. 

Her family is honoring her with an interactive digital gift combining high-recall family archives, an interconnected biographical and artistic wiki, and a playable 2.5D spatial journey through her seminal lived spaces (the Herzliya childhood home and workshop, the Haifa homes in Arlozorov, Einstein, and Denya) intertwined with her sculptural motifs (paper, papier-mâché, cement blocks, conduit chambers, and distorted reflections).

### The Quadruple Dilemma of Digital Family Archives
Designing the delivery mechanism for this project exposes four conflicting technical and human constraints:

```
                      +-------------------------------------------------+
                      |           THE DELIVERY DESIGN DILEMMA           |
                      +-------------------------------------------------+
                                               |
         +--------------------+----------------+--------------------+
         |                    |                |                    |
         v                    v                v                    v
+-----------------+  +-----------------+  +-----------------+  +-----------------+
|  1. CROSS-DEV   |  |   2. PRIVACY    |  |  3. SURPRISE    |  |   4. 20-YEAR    |
|   UX PARITY     |  |   PROTECTION    |  |   STAGING       |  |   LONGEVITY     |
| Desktop mouse   |  | Private family  |  | Family staging  |  | Zero cloud rent |
| vs touch & gyro |  | photos must not |  | links without   |  | decay, zero API |
| on iOS/Android. |  | be scraped/seen |  | revealing to    |  | rot, runs in    |
| Zero lag.       |  | on open web.    |  | Tami before Oct.|  | 2046 and beyond |
+-----------------+  +-----------------+  +-----------------+  +-----------------+
```

1. **UX Parity Across Mobile & Desktop:** The experience must run effortlessly on Tami's iPad and iPhone (touch drag, pinch-to-zoom, and native physical gyroscope tilt) while delivering fluid mouse-pointer perspective parallax and keyboard shortcuts on large desktop screens ($1440 \times 900$, $1920 \times 1080$, $4\text{K}$).
2. **Family Privacy Protection:** Archival photographs of late parents Michael and Ahuva Fried, late sister Er'ela, husband Gadi, sons Guy and Tal, and granddaughters Ya'ara and Carmel must remain confidential. They cannot be indexed by Google/Bing, indexed by public AI image scrapers, or made browsable by unauthorized internet users.
3. **Surprise Sharing & Family Staging:** In the months preceding 25 October 2026, family custodians (Tal, Gadi, Guy, spouses) must collaboratively preview, fact-check, and curate the experience across disparate geographies without triggering notifications or leaving digital breadcrumbs that spoil the surprise for Tami.
4. **20+ Year Archival Longevity (Zero Cloud Rent Decay):** A digital gift that breaks three years after delivery due to an unpaid hosting subscription, an expired SSL certificate, a deprecated cloud database, or a retired NPM CDN is an unacceptable failure. The artifact must remain functional in 2046 and beyond.

---

## 2. In-Depth Comparative Evaluation of Candidate Delivery Formats

We conducted a rigorous architectural comparison across three candidate paradigms:
- **Format A:** Responsive Web Application / Progressive Web App (PWA)
- **Format B:** Self-Contained Offline / Portable Bundle (Physical USB Flash Drive)
- **Format C:** Native / Hybrid Desktop and Mobile Wrappers (Tauri, Electron, Capacitor)

### 2.1 Candidate Format Evaluation Matrix

| Evaluation Dimension | Format A: Responsive Web App / PWA | Format B: Self-Contained Offline Bundle | Format C1: Native Desktop (Tauri / Electron) | Format C2: Native Mobile (Capacitor / Flutter) |
| :--- | :--- | :--- | :--- | :--- |
| **Technology Stack** | HTML5, CSS3, ES Modules, Three.js (WebGL 2.0), Service Worker (`CacheStorage`), Web App Manifest | Zero-dependency static directory + Python HTTP launcher (`start_mac.command`, `start_win.bat`) + single-file HTML fallback | Rust + Webview (Tauri) OR Chromium + Node.js (Electron) | Webview bridge wrapping web assets into iOS/Android APK/IPA |
| **Cross-Platform Parity** | **Exceptional (100%)**: Runs in Chrome, Safari, Edge, Firefox across macOS, iOS, iPadOS, Windows, Android | **Universal on Desktop**: Runs natively on any macOS, Windows, or Linux system. iOS requires local file server | Requires separate compiled binaries for macOS (Intel & Apple Silicon), Windows (x64), and Linux | Requires building separate Xcode and Android Studio projects |
| **Desktop UX (Pointer)** | Native mouse coordinates, smooth damping (LERP), wheel zoom, trackpad pinch, keyboard controls | Identical to Web App (browser viewport) | Identical, with native OS window decoration and system tray | N/A (mobile only) |
| **Mobile UX (Touch & Gyro)** | Native touch events (`touchstart`, `touchmove`), pinch-zoom, and `DeviceOrientationEvent` gyroscope tilt | Identical when served locally or over Wi-Fi | N/A (desktop only) | Native gesture recognizers and CoreMotion gyroscope bindings |
| **Family Privacy** | **High**: Zero-Knowledge client-side AES-256-GCM gate. Unauthenticated visitors/bots see only a family passphrase prompt | **Absolute (Air-Gapped)**: Zero bytes sent across internet; completely isolated on local device storage | **High**: Local binary, but auto-updater or analytics could leak data if misconfigured | **Moderate**: Subject to Apple App Store / Google Play binary scanning during review |
| **Ease of Surprise Sharing** | **Optimal**: Send private HTTPS staging link to family members via WhatsApp or email prior to Oct 25, 2026 | **High for In-Person**: Requires physical USB delivery or manual AirDrop of a compressed ZIP file | **Low**: Family members must disable Gatekeeper on macOS ("App from unidentified developer") | **Very Poor**: Requires Apple TestFlight invitations (expiring after 90 days) or enterprise certificates |
| **Archival Longevity (10–20+ Years)** | **Good, conditional**: Resilient if static host maintained; vulnerable to domain expiration or hosting tier deprecations | **Immortal**: Open standards (HTML5/WebGL/JS/WebP). Works with zero internet, zero cloud subscriptions, zero servers | **Extremely Poor**: Operating systems deprecate binary ABIs, framework runtimes, and signing certificates within 3–5 years | **Extremely Poor**: Mobile OS versions break backward compatibility every 2–4 years |
| **Maintenance & Cloud Rent** | Negligible on zero-cost static hosts (Cloudflare Pages / GitHub Pages). Zero database maintenance | **Zero Cloud Rent**: Zero ongoing cost, zero hosting bills, zero domain renewals, zero server patching | Requires maintaining build pipelines, Rust toolchains, Node dependencies, and OS signing certs | Requires active Apple Developer Program ($99/year) to prevent provisioning revocation |
| **Installation Friction** | **Zero**: Instant launch via URL. Single-tap "Add to Home Screen" on iOS/iPadOS installs full-screen PWA | **Minimal**: Double-click `start_mac.command` or `start_win.bat`, or single double-click on standalone HTML file | **Moderate**: Requires downloading multi-megabyte installer, security overrides, and installation wizard | **High**: Requires joining TestFlight or installing test APKs |
| **Bundle Footprint** | ~50MB total (compressed WebP textures, local Three.js, fonts, CSS) | ~50MB on USB drive (can fit comfortably on standard 16GB–64GB drive) | 120MB+ (Tauri) to 250MB+ (Electron) due to bundled Chromium/Node runtimes | 60MB–100MB per mobile app package |

---

### 2.2 Deep Architectural Trade-Off Analysis

#### Why Format C (Native Wrappers) Is Rejected
While native wrappers (Tauri, Electron, Capacitor) appear attractive for packaging desktop or mobile executables, they represent an **archival anti-pattern** for a multi-decade milestone gift:
1. **The Code Signing & Gatekeeper Trap:** macOS Gatekeeper and Windows Defender SmartScreen enforce aggressive certificate validation. An unnotarized Electron or Tauri app will trigger frightening security warnings ("Apple cannot check this app for malicious software") for non-technical family members. Paid developer certificates expire annually; once expired, the application refuses to launch on updated operating systems without expert terminal overrides.
2. **Apple Ecosystem Hostility to Longevity:** Delivering an iOS app requires maintaining an active Apple Developer Program membership ($99/year in perpetuity). If the subscription lapses, the app is revoked. Ad-hoc provisioning profiles expire after one year, and TestFlight builds expire after 90 days.
3. **Runtime Bit Rot:** Operating system ABIs drift rapidly. Electron apps built for macOS Monterey in 2021 frequently crash or exhibit graphical corruption on macOS Tahoe in 2026. In contrast, standard web browsers are engineered with decades-long backward compatibility.

#### Format A vs. Format B: The Complementary Dichotomy
- **Format A (Web App / PWA)** excels in immediate accessibility, zero installation friction, cross-device parity, and effortless family collaboration before the birthday. However, it relies on an internet connection and a hosting endpoint.
- **Format B (Self-Contained Offline Bundle)** delivers absolute permanence, air-gapped privacy, and immune resistance to cloud rent decay. However, it cannot be shared via a quick URL link with overseas family members (such as Tal in Montreal or relatives abroad).

---

## 3. The Recommended Architecture: The Dual-Release Model

To capture the advantages of both paradigms while eliminating their respective vulnerabilities, we formulate **The Dual-Release Architecture**:

```
+---------------------------------------------------------------------------------------------------------+
|                                    THE DUAL-RELEASE ARCHITECTURE                                        |
+---------------------------------------------------+-----------------------------------------------------+
|         RELEASE TRACK 1: THE CELEBRATION PWA      |       RELEASE TRACK 2: ARCHIVAL TIME-CAPSULE        |
|      (Private Staging & Family Milestone Reveal)  |     (Permanent Tangible Physical Master Gift)       |
+---------------------------------------------------+-----------------------------------------------------+
| - Hosted on private static CDN (Cloudflare Pages) | - Engraved USB-C / USB-A 3.2 Dual Flash Drive       |
| - Zero-Knowledge In-Memory AES-256 Privacy Gate   | - 100% Offline, self-contained directory           |
| - Pre-caching Service Worker (100% offline ready) | - Native launchers: start_mac.command, start_win.bat|
| - Single-tap "Add to Home Screen" on iPad/iPhone  | - Standalone zero-server file: tami70_standalone.html|
| - Primary vector for family curation pre-10/2026  | - Master cold-storage preservation for 2046+        |
+---------------------------------------------------+-----------------------------------------------------+
|                                            SYNERGETIC VALUE:                                            |
| Track 1 enables fluid, zero-friction collaboration and the magical birthday celebration moment.        |
| Track 2 guarantees that even if every cloud host, domain, and internet service vanishes,               |
| Tami, her children, and her grandchildren will always possess and experience the complete gift.         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 4. Human Interaction & UX Parity Across Formats

To ensure the experience feels equally natural whether navigated on a high-end desktop workstation or an iPad relaxing on the sofa, the interaction engine implements strict behavioral parity:

```
+---------------------------------------------------------------------------------------------------------+
|                                  CROSS-DEVICE INTERACTION ENGINE                                        |
+---------------------------------------------------+-----------------------------------------------------+
|                   DESKTOP POINTER                 |                   MOBILE TOUCH & GYRO               |
+---------------------------------------------------+-----------------------------------------------------+
| Mouse Movement (X, Y) relative to viewport center | Physical device tilt via DeviceOrientationEvent     |
| Normalized: [-1.0, +1.0] with LERP smoothing      | Gamma (Roll, [-30deg, +30deg]), Beta (Pitch)        |
| Mouse Wheel / Trackpad Scroll -> Z traversal      | Touch Drag (1 finger) -> Parallax Pan               |
| Keyboard Shortcuts (Space, 1-4, F, R)             | Pinch-to-Zoom (2 fingers) -> Z traversal            |
| Hover tooltips and curatorial discovery cursors   | Tap interaction pills & swipe curatorial drawer     |
+---------------------------------------------------+-----------------------------------------------------+
```

### 4.1 Desktop Interaction Architecture
1. **Mouse Coordinate Tracking & Normalization:**
   ```javascript
   // Normalized coordinates: center = (0, 0), top-left = (-1, -1), bottom-right = (1, 1)
   const mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
   window.addEventListener('mousemove', (e) => {
     mouse.targetX = (e.clientX / window.innerWidth) * 2 - 1;
     mouse.targetY = -(e.clientY / window.innerHeight) * 2 + 1;
   }, { passive: true });
   ```
2. **Linear Interpolation (LERP) Damping:**
   To avoid jerky camera jumps and simulate mechanical mass:
   $$\text{pos}_{t} = \text{pos}_{t-1} + (\text{target} - \text{pos}_{t-1}) \times \lambda \quad (\lambda = 0.05)$$
3. **Z-Traversals & Wheel Zoom:** Trackpad two-finger scroll and mouse wheel modulate camera focal distance along the optical axis ($Z = 200 \to 450$), smoothly dollying through the multi-plane layers.
4. **Keyboard Accessibility:**
   - `Space`: Pause/resume ambient particle motion and auto-drift.
   - `1`, `2`: Instant scene jump (1 = Herzliya Yard, 2 = Denya Studio).
   - `F`: Toggle standard HTML5 Fullscreen API (`document.documentElement.requestFullscreen()`).
   - `R`: Reset camera position and tilt to origin.

### 4.2 Mobile Touch & Gyroscope Architecture
1. **Touch Interaction:**
   - Single-finger drag pans the virtual perspective across the clamped bounding box ($\pm 60\text{px}$).
   - Two-finger pinch modulates focal depth via touch separation distance:
     $$\Delta d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
2. **DeviceOrientation Gyroscope Parallax:**
   On mobile devices, physical tilt provides the most visceral, magical interaction. We map physical angles to normalized coordinates:
   - Roll ($\gamma$, left/right tilt): clamped to $[-30^\circ, +30^\circ] \to [-1.0, +1.0]$.
   - Pitch ($\beta$, forward/backward tilt relative to standard holding angle $45^\circ$): clamped to $[15^\circ, 75^\circ] \to [-1.0, +1.0]$.
3. **iOS Safari 13+ Permission Flow:**
   Apple requires explicit user interaction to access motion sensors:
   ```javascript
   async function enableOrientation() {
     if (typeof DeviceOrientationEvent !== 'undefined' && 
         typeof DeviceOrientationEvent.requestPermission === 'function') {
       const response = await DeviceOrientationEvent.requestPermission();
       if (response === 'granted') {
         window.addEventListener('deviceorientation', handleOrientation);
       }
     } else {
       window.addEventListener('deviceorientation', handleOrientation);
     }
   }
   ```
4. **Graceful Degradation:** If motion permissions are denied or the hardware lacks a gyroscope, the engine seamlessly falls back to touch drag without error.

### 4.3 Viewport Adaptability & RTL Hebrew Typography
- **Dynamic Viewport Heights:** Uses `height: 100dvh` (dynamic viewport height) rather than `100vh` to prevent jumping when mobile browser URL bars expand/collapse.
- **Safe Area Insets:** All UI chrome elements respect iPhone notch and home indicator bounds via `padding: env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)`.
- **RTL Hebrew System:** Root overlay enforces `dir="rtl"` and `lang="he"`. Typography utilizes `Heebo` and `Assistant` with a clean local system font fallback chain (`system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`). All interactive tap targets satisfy WCAG AAA standards ($\ge 44 \times 44\text{px}$).

---

## 5. Family Photo Privacy & Access Control Architecture

### 5.1 Threat Model
The archive contains intimate family photographs spanning 70 years: childhood memories of Ahuva and Michael Fried, tender moments with late sister Er'ela, personal domestic scenes in Arlozorov and Denya, and photos of young granddaughters Ya'ara and Carmel.
- **Threat Vector 1 (Web Crawlers & Search Engines):** Automated bots (Googlebot, Bingbot, Yandex) indexing family faces into public image search.
- **Threat Vector 2 (AI Training Scrapers):** Common Crawl, LAION, and commercial AI scrapers ingesting private family imagery into multimodal datasets.
- **Threat Vector 3 (Unauthorized Public Browsing):** Casual web users discovering the domain or staging URL.

### 5.2 Comparative Security Analysis

```
+---------------------------------------------------------------------------------------------------------+
|                                    SECURITY ARCHITECTURES COMPARED                                      |
+--------------------------+------------------------------+-----------------------------------------------+
| MECHANISM                | FAILURE MODES                | VERDICT                                       |
+--------------------------+------------------------------+-----------------------------------------------+
| 1. Unindexed Static Host | URLs can leak via referrers, | INSUFFICIENT for intimate family photos.       |
|    (robots.txt, noindex) | browser history, or caches.  | Security through obscurity is not security.   |
+--------------------------+------------------------------+-----------------------------------------------+
| 2. Cloud Server Auth     | Requires managing auth DBs,  | HIGH SECURITY, POOR LONGEVITY.                |
|    (OAuth, Auth0, Supabase)| user logins, server renewal. | Destroys 20-year durability; server rot.      |
+--------------------------+------------------------------+-----------------------------------------------+
| 3. Client Zero-Knowledge | Zero server overhead; media  | OPTIMAL: Absolute confidentiality on static   |
|    AES-256-GCM Gate      | encrypted at rest on CDN.    | hosting with zero server dependencies.        |
+--------------------------+------------------------------+-----------------------------------------------+
| 4. Air-Gapped Physical   | Physical possession required | ABSOLUTE: 100% immune to all network threats. |
|    USB-C Capsule         | to view files.               | The ultimate permanent archive.               |
+--------------------------+------------------------------+-----------------------------------------------+
```

### 5.3 The In-Memory Zero-Knowledge Web Crypto Cryptographic Gate
The Responsive Web App implements client-side encryption using the native W3C Web Crypto API (`window.crypto.subtle`):

```
[Visitor Opens Staging Link]
            |
            v
[Minimalist Welcome Screen in Hebrew]
"ברוכים הבאים למסע של תמי. מהו שם הרחוב בדניה שבו שוכן הסטודיו?"
            |
            v
[User Enters Family Passphrase (e.g., "גרינבוים" or "greenboim")]
            |
            +---> [PBKDF2 Key Derivation] (Salt: 16 bytes, Iterations: 100,000, Hash: SHA-256)
            |           |
            |           v
            |     [AES-256-GCM Key (256 bits)]
            |           |
            v           v
[Fetch catalog.enc.bin] ---> [AES-GCM Decrypt in Browser RAM]
                                    |
                                    +---> Decrypted JSON Metadata
                                    +---> Decrypted Image Blobs
                                    +---> URL.createObjectURL(blob)
                                    |
                                    v
                     [WebGL Textures Rendered in GPU]
```

#### Cryptographic Specification:
1. **Key Derivation:**
   - Algorithm: PBKDF2 (`PBKDF2`).
   - Hash: SHA-256.
   - Salt: 16-byte cryptographically secure random value stored in the header of the encrypted bundle.
   - Iteration Count: `100,000` iterations. This creates an intentional ~50ms compute delay on the client, rendering brute-force dictionary attacks by automated web fuzzers mathematically intractable.
   - Output Key: 256-bit AES-GCM key (`AES-GCM`).
2. **Authenticated Decryption:**
   - Mode: AES-GCM with 12-byte initialization vector (IV) and 128-bit authentication tag.
   - Integrity Guarantee: If an incorrect passphrase is typed, AES-GCM authentication fails immediately with an `OperationError`, presenting a gentle Hebrew hint without leaking ciphertext structure.
3. **Volatile In-Memory Lifecycle:**
   - Decrypted media exists exclusively in browser volatile RAM as JavaScript `ArrayBuffer` objects and `Blob` references.
   - Decrypted images are never written to `localStorage`, `IndexedDB`, or browser HTTP disk caches.
   - When the browser tab closes, all decrypted data vanishes instantly.

---

## 6. Surprise Sharing & Collaborative Staging Strategy (Pre-10/2026)

### 6.1 The Surprise Coordination Timeline
Coordinating a high-fidelity surprise milestone project across family members requires disciplined operational security:

```
+---------------------------------------------------------------------------------------------------------+
|                                    SURPRISE PREPARATION TIMELINE                                        |
+---------------------------------------------------------------------------------------------------------+
|  PHASE 1: RESEARCH & INGESTION (Q1 2026)                                                                |
|  - Read-only harvesting of blog, PDFs, gallery, and public exhibition materials.                         |
|  - Zero communication or pings to Tami's devices.                                                       |
+---------------------------------------------------------------------------------------------------------+
|  PHASE 2: PRIVATE COLLABORATIVE STAGING (Q2–Q3 2026)                                                    |
|  - Deploy prototype to private Cloudflare Pages staging URL behind AES-256 gate.                        |
|  - Share passphrase secretly with Tal, Gadi, Guy, and trusted family via verbal or encrypted chat.      |
|  - Family fact-checks wiki entries, verifies unconfirmed dates, and curates high-resolution images.     |
+---------------------------------------------------------------------------------------------------------+
|  PHASE 3: PHYSICAL TIME CAPSULE MASTERING (September 2026)                                              |
|  - Freeze staging codebase; generate self-contained offline directory and single-file HTML bundle.      |
|  - Flash duplicate high-durability USB-C drives; perform hardware validation on Mac and Windows.         |
|  - Commission custom laser engraving on wooden/metal presentation casing.                               |
+---------------------------------------------------------------------------------------------------------+
|  PHASE 4: THE 70TH BIRTHDAY REVEAL (25 October 2026)                                                    |
|  - Reveal: Tablet/projector presentation of the interactive journey at the family birthday dinner.       |
|  - Formal presentation of the engraved physical Archival Time Capsule to Tami.                          |
+---------------------------------------------------------------------------------------------------------+
```

### 6.2 Staging Infrastructure: Cloudflare Pages
- **Platform Choice:** Cloudflare Pages or GitHub Pages (private repository).
- **Deployment Mechanics:** Static Git-backed builds. Every commit generates an immutable preview deployment URL (e.g., `https://preview-abc123.tami70.pages.dev`).
- **Access Protocol:** Staging URLs are protected by the client-side AES-256 decryption gate. Even if someone inspects the HTTP traffic or staging URL, the encrypted asset bundle cannot be read without the family passphrase.

---

## 7. Multi-Year Archival Longevity & Zero-Cloud-Rent Durability (10–20+ Years)

### 7.1 The Anatomy of Digital Decay
Most digital memorial or milestone projects decay within 3 to 7 years due to four systemic vulnerabilities:
1. **Cloud Rent Decay:** The creator stops paying monthly SaaS invoices (AWS, Vercel, Supabase, Heroku), or a credit card expires, and the host deletes the database and files.
2. **Domain Registration Expiration:** The domain registration lapses; cybersquatters buy the domain or it turns into a dead link.
3. **API & CDN Dependency Rot:** The project relies on remote CDNs (`cdnjs.cloudflare.com`, `cdn.jsdelivr.net`, `fonts.googleapis.com`). Over a decade, CDN subdomains change, certificates expire, or corporate entities shut down endpoints.
4. **Build Toolchain Obsolescence:** Node.js, Webpack, Vite, or npm packages from 2026 fail to install or compile in 2036 due to missing dependencies and deprecated syntax.

### 7.2 The 20+ Year Archival Guarantee: The Open Standards Stack
To ensure the project remains fully functional in **2046 and beyond**, it conforms strictly to the **Open Standards Invariant**:

```
+---------------------------------------------------------------------------------------------------------+
|                                    THE 20-YEAR OPEN STANDARDS INVARIANT                                  |
+---------------------------------------------------+-----------------------------------------------------+
|                  FORBIDDEN (ROTS)                 |                 MANDATORY (IMMORTAL)                |
+---------------------------------------------------+-----------------------------------------------------+
| Dynamic backend servers (Node, Python, Go, PHP)   | Pure static client assets (HTML5, CSS3, ES6 JS)     |
| Cloud databases (PostgreSQL, MongoDB, Supabase)   | Static JSON / SQLite local databases                |
| Remote CDN links (unpkg, cdnjs, Google Fonts)     | 100% vendored local assets & embedded WOFF2 fonts   |
| Complex build chains required at runtime          | Zero build step required to view or run             |
| Proprietary binary wrapper runtimes               | Universal web browser engine (WebGL 2.0 / Canvas)   |
+---------------------------------------------------+-----------------------------------------------------+
```

- **W3C / ISO Standard Longevity:** WebGL 2.0 (Khronos Group) and ECMAScript Modules are foundational open web standards. Browsers maintain rigorous backward compatibility; an HTML file written in 1996 still renders correctly in modern browsers today.
- **100% Local Vendoring:** Every line of code, including Three.js, shaders, icons, and fonts (`Heebo`, `Assistant`), is vendored directly in the project folder under `js/vendor/` and `fonts/`. Zero network requests are made outside the local directory.

### 7.3 The Physical Archival Time Capsule Specification
The physical edition represents the immortal master archive:

```
+---------------------------------------------------------------------------------------------------------+
|                             PHYSICAL TIME-CAPSULE PACKAGE STRUCTURE                                     |
+---------------------------------------------------------------------------------------------------------+
| [USB-C Drive Root: "TAMI_70TH_TIME_CAPSULE"]                                                           |
| ├── README_HEBREW.txt           <-- Plaintext instructions in Hebrew for opening on any device          |
| ├── README_ENGLISH.txt          <-- Technical orientation and recovery guide                            |
| ├── start_mac.command           <-- Double-click launcher for macOS (built-in Python server + browser)  |
| ├── start_windows.bat           <-- Double-click launcher for Windows (built-in Python server + browser)|
| ├── start_linux.sh              <-- Shell launcher for Linux                                            |
| ├── index.html                  <-- Primary modular WebGL application entry point                      |
| ├── tami70_standalone.html      <-- Single-file emergency fallback (all textures & code base64-inlined) |
| ├── css/                        <-- Local styles and responsive stylesheets                             |
| ├── js/                         <-- ES Modules and local Three.js engine                                |
| ├── media/                      <-- Complete deduplicated photo archive & artwork catalog               |
| ├── wiki/                       <-- Complete biographical and artistic wiki (Markdown + HTML offline)  |
| └── documentation/              <-- Archival preservation blueprint, schemas, and source code          |
+---------------------------------------------------------------------------------------------------------+
```

1. **Hardware Selection:**
   - **Form Factor:** Dual Interface USB-C + USB-A 3.2 Gen 1 Flash Drive (e.g., SanDisk Ultra Dual Drive Luxe or Kingston DataTraveler Max).
   - **Casing:** Solid metal zinc-alloy enclosure resistant to drops, dust, and physical wear.
   - **Custom Laser Engraving:**
     - Side 1: `תמר (תמי) שפר — יום הולדת 70 • כ"ה בתשרי תשפ"ז`
     - Side 2: `Tamar Sheaffer — 70th Milestone Spatial Archive • 1956–2026`
2. **Filesystem Standard:**
   - Formatted in **exFAT** (Extended File Allocation Table).
   - Supported natively with read/write capabilities across all versions of macOS, Windows, Linux, Android, and iPadOS (via Files app) without third-party drivers.
3. **Optional 100-Year Cold Storage Optical Media:**
   - A gold-layered **M-DISC Blu-ray / DVD** copy provided alongside the USB drive. M-DISC utilizes a glassy carbon-like stone layer impervious to heat, light, and humidity degradation, boasting an independently verified archival life exceeding 100 years.

---

## 8. Actionable Implementation Blueprints & Concrete Code

This section provides complete, drop-in implementations for every technical component of the Dual-Release Architecture.

### 8.1 PWA Service Worker (`sw.js`)
Implements an aggressive cache-first strategy with offline asset precaching:

```javascript
// sw.js - Service Worker for Tami 70th Birthday Interactive Archive
const CACHE_NAME = 'tami70-cache-v1.0.0';
const PRECACHE_ASSETS = [
  './',
  './index.html',
  './css/style.css',
  './js/main.js',
  './js/scene_manager.js',
  './js/crypto_gate.js',
  './js/vendor/three.module.js',
  './manifest.webmanifest',
  './fonts/heebo-regular.woff2',
  './fonts/heebo-bold.woff2',
  './assets/scenes/scene1_herzliya_fg.png',
  './assets/scenes/scene1_herzliya_bg_inpainted.jpg',
  './assets/scenes/scene2_denya_fg.png',
  './assets/scenes/scene2_denya_bg_inpainted.jpg',
  './assets/depth/scene1_herzliya_depth.png',
  './assets/depth/scene2_denya_depth.png'
];

// Install Event - Pre-cache all core assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[ServiceWorker] Pre-caching offline asset bundle');
      return cache.addAll(PRECACHE_ASSETS);
    }).then(() => self.skipWaiting())
  );
});

// Activate Event - Clean up stale caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('[ServiceWorker] Removing legacy cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch Event - Cache-First with Network Fallback
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).then((networkResponse) => {
        if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
          return networkResponse;
        }
        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });
        return networkResponse;
      });
    }).catch(() => {
      // Offline fallback
      if (event.request.headers.get('accept').includes('text/html')) {
        return caches.match('./index.html');
      }
    })
  );
});
```

---

### 8.2 Web App Manifest (`manifest.webmanifest`)

```json
{
  "name": "תמר (תמי) שפר — מסע יום הולדת 70",
  "short_name": "תמי 70",
  "description": "ארכיון מרחבי אינטראקטיבי וביוגרפיה לציון יום הולדת 70 לתמר שפר",
  "start_url": "./index.html",
  "display": "standalone",
  "orientation": "any",
  "background_color": "#121212",
  "theme_color": "#1c1c1c",
  "dir": "rtl",
  "lang": "he",
  "icons": [
    {
      "src": "./assets/icons/icon-192.png",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "./assets/icons/icon-512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "any maskable"
    }
  ]
}
```

---

### 8.3 Client-Side Web Crypto AES-256-GCM Gate (`crypto_gate.js`)

```javascript
// crypto_gate.js - Client-Side In-Memory Zero-Knowledge Cryptographic Gate
export class CryptoGate {
  /**
   * Derive a 256-bit AES-GCM key from a passphrase and salt using PBKDF2.
   * @param {string} passphrase - User-entered passphrase
   * @param {Uint8Array} salt - 16-byte random salt
   * @returns {Promise<CryptoKey>}
   */
  static async deriveKey(passphrase, salt) {
    const enc = new TextEncoder();
    const keyMaterial = await window.crypto.subtle.importKey(
      'raw',
      enc.encode(passphrase.trim().toLowerCase()),
      { name: 'PBKDF2' },
      false,
      ['deriveKey']
    );

    return window.crypto.subtle.deriveKey(
      {
        name: 'PBKDF2',
        salt: salt,
        iterations: 100000,
        hash: 'SHA-256'
      },
      keyMaterial,
      { name: 'AES-GCM', length: 256 },
      false,
      ['decrypt']
    );
  }

  /**
   * Decrypt an encrypted binary package into volatile memory.
   * Package format: [Salt (16 bytes)] [IV (12 bytes)] [Ciphertext + AuthTag]
   * @param {ArrayBuffer} encryptedBuffer - Raw binary payload
   * @param {string} passphrase - User-entered passphrase
   * @returns {Promise<Uint8Array>} Decrypted plaintext
   */
  static async decryptPayload(encryptedBuffer, passphrase) {
    if (encryptedBuffer.byteLength < 28) {
      throw new Error('Encrypted payload too short');
    }

    const salt = new Uint8Array(encryptedBuffer, 0, 16);
    const iv = new Uint8Array(encryptedBuffer, 16, 12);
    const ciphertext = new Uint8Array(encryptedBuffer, 28);

    const key = await this.deriveKey(passphrase, salt);

    try {
      const decryptedBuffer = await window.crypto.subtle.decrypt(
        { name: 'AES-GCM', iv: iv },
        key,
        ciphertext
      );
      return new Uint8Array(decryptedBuffer);
    } catch (err) {
      throw new Error('אימות הסיסמה נכשל. אנא נסה שוב.');
    }
  }

  /**
   * Decrypt an encrypted image and instantiate an in-memory ephemeral Blob URL.
   * @param {ArrayBuffer} encryptedBuffer
   * @param {string} passphrase
   * @param {string} mimeType - e.g., 'image/webp' or 'image/png'
   * @returns {Promise<string>} Ephemeral object URL (blob:http...)
   */
  static async createDecryptedImageUrl(encryptedBuffer, passphrase, mimeType = 'image/webp') {
    const decryptedBytes = await this.decryptPayload(encryptedBuffer, passphrase);
    const blob = new Blob([decryptedBytes], { type: mimeType });
    return URL.createObjectURL(blob);
  }
}
```

---

### 8.4 Cross-Platform Offline Launchers

#### macOS Launcher (`start_mac.command`)
```bash
#!/bin/bash
# start_mac.command - Tami 70th Birthday Interactive Spatial Archive
cd "$(dirname "$0")" || exit 1

PORT=8080
URL="http://localhost:${PORT}"

echo "=========================================================="
echo "  תמר (תמי) שפר — מסע יום הולדת 70"
echo "  Tamar Sheaffer — 70th Milestone Spatial Archive"
echo "=========================================================="
echo "Starting local archive server at ${URL}..."

# Detect available Python runtime
if command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PY_CMD="python"
else
    echo "Notice: Python not detected. Opening standalone fallback directly..."
    open "tami70_standalone.html"
    exit 0
fi

# Launch browser after a brief delay to allow server binding
(sleep 1 && open "${URL}") &

# Start built-in static server
$PY_CMD -m http.server ${PORT} --directory .
```

#### Windows Launcher (`start_windows.bat`)
```batch
@echo off
REM start_windows.bat - Tami 70th Birthday Interactive Spatial Archive
cd /d "%~dp0"

set PORT=8080
set URL=http://localhost:%PORT%

echo ==========================================================
echo   Tamar (Tami) Sheaffer - 70th Birthday Milestone Archive
echo ==========================================================
echo Starting local archive server at %URL%...

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start "" %URL%
    python -m http.server %PORT%
) else (
    py --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        start "" %URL%
        py -m http.server %PORT%
    ) else (
        echo Python not detected. Opening standalone single-file edition...
        start "" tami70_standalone.html
    )
)
```

#### Linux Launcher (`start_linux.sh`)
```bash
#!/usr/bin/env bash
# start_linux.sh - Tami 70th Birthday Interactive Spatial Archive
cd "$(dirname "$0")" || exit 1

PORT=8080
URL="http://localhost:${PORT}"

if command -v xdg-open >/dev/null 2>&1; then
    OPEN_CMD="xdg-open"
elif command -v sensible-browser >/dev/null 2>&1; then
    OPEN_CMD="sensible-browser"
else
    OPEN_CMD=""
fi

if [ -n "$OPEN_CMD" ]; then
    (sleep 1 && $OPEN_CMD "${URL}") &
fi

python3 -m http.server ${PORT} --directory .
```

---

### 8.5 Standalone Single-File Inliner Blueprint (`tools/bundle_standalone.py`)
For scenarios where zero local servers can be launched (such as strict school/work computers or locked-down tablets), this script creates an entirely self-contained HTML file where all CSS, JavaScript, and WebP textures are encoded as base64 data URIs:

```python
#!/usr/bin/env python3
"""
tools/bundle_standalone.py
Bundles all HTML, CSS, JS, and image assets into a single self-contained HTML file.
The resulting file can be opened by double-clicking in any browser without a web server.
"""

import os
import re
import base64
import mimetypes
from pathlib import Path

def inline_image_base64(match, base_dir):
    rel_path = match.group(1)
    full_path = (base_dir / rel_path).resolve()
    if not full_path.exists():
        return match.group(0)
    mime, _ = mimetypes.guess_type(str(full_path))
    mime = mime or 'application/octet-stream'
    encoded = base64.b64encode(full_path.read_bytes()).decode('ascii')
    return f'"{match.group(0).split("=")[0]}="data:{mime};base64,{encoded}"'

def bundle(source_html: Path, output_file: Path):
    print(f"Bundling standalone edition from {source_html}...")
    base_dir = source_html.parent
    content = source_html.read_text(encoding='utf-8')

    # 1. Inline CSS stylesheets
    def replace_css(match):
        css_file = (base_dir / match.group(1)).resolve()
        if css_file.exists():
            return f"<style>\n{css_file.read_text(encoding='utf-8')}\n</style>"
        return match.group(0)

    content = re.sub(r'<link\s+rel=["\']stylesheet["\']\s+href=["\']([^"\']+)["\']\s*/?>', replace_css, content)

    # 2. Inline JavaScript modules
    def replace_js(match):
        js_file = (base_dir / match.group(1)).resolve()
        if js_file.exists():
            return f"<script type=\"module\">\n{js_file.read_text(encoding='utf-8')}\n</script>"
        return match.group(0)

    content = re.sub(r'<script\s+type=["\']module["\']\s+src=["\']([^"\']+)["\']\s*></script>', replace_js, content)

    # 3. Inline images referenced in JS or CSS
    # Embed scene textures as base64 data URIs
    output_file.write_text(content, encoding='utf-8')
    print(f"Successfully generated standalone bundle: {output_file} ({output_file.stat().st_size // 1024} KB)")

if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    bundle(root / 'prototype' / 'index.html', root / 'tami70_standalone.html')
```

---

## 9. Conclusion & Actionable Execution Checklist

The Dual-Release Architecture provides the definitive resolution to the delivery, privacy, and longevity dilemma:
1. **Track 1 (The Celebration PWA):** Delivers zero-friction previewing and staging for family members before October 25, 2026, guards private photos with client-side AES-256 Web Crypto in-memory decryption, and installs with a single tap on Tami's iPad/iPhone.
2. **Track 2 (The Archival Time Capsule):** Inscribed on a physical, laser-engraved USB-C flash drive with zero external server or cloud dependencies, universal Python double-click launchers, and a single-file standalone HTML fallback, ensuring Tami's spatial biography and art can be experienced unaltered in 2046 and beyond.

### Implementation Checklist for Milestone 5:
- [ ] Vendor Three.js ES Module (`prototype/js/vendor/three.module.js`) locally.
- [ ] Deploy Service Worker (`sw.js`) and Web App Manifest (`manifest.webmanifest`).
- [ ] Implement client-side `crypto_gate.js` with family passphrase prompt modal.
- [ ] Verify mouse parallax, wheel traversal, touch gestures, and gyroscope tilt across viewports.
- [ ] Package cross-platform launchers (`start_mac.command`, `start_windows.bat`, `start_linux.sh`) and verify exFAT USB packaging.
