---
trigger: always_on
description: Project coding & architecture rules for Tamar "Tami" Sheaffer 70th-Birthday Spatial Archive (No Silent Fallbacks, Real ML Vision Pipeline in .venv, Read-Only Privacy Guardrails, SoC, SSOT Configs, Strict Typing, Clean Root)
---

# Project Coding Guidelines (`tami`)

1. **No Silent Fallbacks, Toy Approximations, or Hidden Defaults**:
   - Asset generation (`prototype/build_assets.py`) MUST execute real neural computer vision models inside `/Users/talsheaffer/code/tami/.venv` (Depth Anything V2 monocular depth estimation, neural foreground matting/segmentation via `rembg`, and neural Fast Fourier Convolution LaMa inpainting via `simple-lama-inpainting`).
   - Never fall back to hardcoded geometric masks (`ImageDraw.polygon` / `ImageDraw.ellipse`) or `ImageFilter.GaussianBlur` stand-ins for segmentation or inpainting. If `.venv` packages or weights are unavailable, fail immediately with a clear `RuntimeError` explaining which `.venv` interpreter/model is required.
   - Never silently clamp or override scene/pipeline configuration values downstream, and never swallow WebGL/asset errors silently.

2. **Strict Read-Only Safety & Surprise-Gift Privacy**:
   - All access to family sources, local WhatsApp databases, and personal accounts must remain 100% read-only (enforced by `tests/test_governance.py`). Never send outbound messages, mutate external accounts, or commit transient diagnostics into tracked archive data.

3. **Separation of Concerns (SoC) & Modularity**:
   - Keep scene/asset configuration (`prototype/scene_config.py` / `prototype/assets/scenes/manifest.json`), offline ML asset synthesis (`prototype/build_assets.py`), HTTP serving & diagnostics (`prototype/serve.py`), Three.js WebGL rendering (`prototype/js/scene_manager.js`), and UI/audio chrome (`prototype/js/app.js`) strictly separated.
   - Extract reusable vision/geometry primitives into focused functions with explicit type annotations (`from __future__ import annotations`).

4. **Single Source of Truth (SSOT) & Zero Magic Numbers**:
   - All asset dimensions, segmentation thresholds, morphological dilation radii, depth-smoothing parameters, and scene layer specs must live in typed `@dataclass` configs serialized cleanly to JSON (`prototype/assets/scenes/scene_manifest.json`) so Python offline builders and JavaScript WebGL runtime share an explicit contract.

5. **No Legacy Shims, DRY, KISS, & Repository Tidiness**:
   - Remove obsolete code paths and dead parameters immediately rather than keeping legacy shims; update `tests/` alongside API changes.
   - Keep `/Users/talsheaffer/code/tami` root clean (preserve the 5 root governance markdown documents verified by `tools/verify_wiki_links.py`; place scratch renders and phone telemetry logs in the Antigravity `scratch/` directory).

6. **2.5D Depth Topology — Contact Continuity ($C^0$) & Occlusion Discontinuity ($C^{-1}$)**:
   - Follow `.agents/skills/coding-guidelines/references/depth_topology_principles.md` on every existing and new scene:
     - **Contact Continuity ($C^0$)**: Wherever an object touches a supporting surface (floor footprints, steps, tabletops, wall bolts, framed wall art, seats), its world depth $Z_{\text{world}}^{\text{obj}} = Z_k + A_k \cdot d_k$ MUST match the supporting surface's $Z_{\text{world}}^{\text{host}}$ ($\Delta Z_{\text{world}} = 0$). Never place floor-standing sculptures/people or multi-depth wall-bolted elements (e.g., stair handrails) on detached constant-$Z$ billboards that float or shear off their anchors under off-axis parallax; use harmonic contact-anchored depth displacement (`pinMarginUv = 0.0`).
     - **Free-Silhouette Discontinuity ($C^{-1}$)**: Across non-contacting occlusion edges where an object stands in front of a distant background, split the occluder onto a separate alpha-matted/cutout mesh, extrapolate its interior depth outward beyond its alpha silhouette via Euclidean Distance Transform (`EDT`) so straddling triangles never rubber-sheet toward `0`/wall depth, and ensure `layer0`'s LaMa-inpainted background depth contains zero ghost foreground depth bulges.
     - **Bore-Sight & Automated Gate**: Enable `boreSightAtRest: true` on all multi-layer scenes and enforce programmatic contact/discontinuity checks in `tests/test_depth_principles.py`.

7. **3D Portal Flight & Scene Transition Laws**:
   - Follow `.agents/skills/coding-guidelines/references/flight_transition_principles.md` across all scene transitions and portals:
     - **Pure Flight (Zero Effects)**: Never apply `ripple`, `directionalWarp`, or `specularFlash` shader effects; route transitions between portal-connected scenes through real 3D forward/reverse portal flight, using pure radiometric cross-dissolve only when an optical blend is required.
     - **Continuous Camera, Clipping & Hue**: Land at exact rotated-frustum `effectiveFitZ` with portal clip planes active across all $t \in [0, 1]$ (no mid-flight unclip snaps), keep opaque `layer0` walls at `uFadeToClear = 0.0` (never fade walls to black), and interpolate any host-to-scene hue adaptation (`uTint`) smoothly across flight progress.
     - **Advance Prewarming & Bijective State Isomorphism**: Preload and GPU-prewarm all target bundles, occluders, veils, and child portals in advance, and synchronize every mutable scene state (`companionFall` position **and** 3D rotation, depth-displaced child occluders, live dissolve animations, carpet unroll state, and active-ancestor occluder lock suppression) 1:1 between standalone scenes and embedded `dioramaGroup`s, verified by `tests/test_flight_principles.py`.
