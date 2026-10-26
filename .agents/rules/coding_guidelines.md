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
