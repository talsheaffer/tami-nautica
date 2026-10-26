---
name: coding-guidelines
description: >-
  Project-specific software engineering and spatial-pipeline standards for the
  Tamar "Tami" Sheaffer 70th-Birthday project (/Users/talsheaffer/code/tami).
  Enforces zero silent fallbacks or fake vision approximations, real neural
  depth/segmentation/inpainting via .venv, SSOT typed dataclass configs shared
  with the WebGL runtime, strict read-only privacy guardrails, Separation of
  Concerns, strict typing, and empirical desktop/mobile visual verification.
  Activate whenever modifying prototype/, tools/, scripts/, or tests/.
---

# `tami` Project Coding Guidelines & Spatial Pipeline Skill

Apply these rules whenever designing, writing, refactoring, or auditing code in `/Users/talsheaffer/code/tami`.

## 1. Zero Silent Fallbacks in Vision & WebGL Pipelines (Mandatory)

- **Always Use `/Users/talsheaffer/code/tami/.venv/bin/python` for ML Asset Builds**:
  - Monocular depth estimation: **Depth Anything V2** (`transformers` `depth-anything/Depth-Anything-V2-Small-hf` or `Base-hf` on MPS/CPU).
  - Foreground sculpture/subject segmentation: **Neural salient/object matting** (`rembg` `isnet-general-use` / `u2net` combined with metric depth discontinuities—never hardcoded polygons or ellipses).
  - Occlusion synthesis: **Neural LaMa Fast Fourier Convolution Inpainting** (`simple-lama-inpainting` on morphologically dilated masks $r \ge 25\text{px}$—never Gaussian blur).
- **Fail Fast**: If an import, model checkpoint, or WebGL shader compilation fails, raise an explicit error immediately (and emit `/__diag` telemetry in browser runtime). Never catch-and-substitute degraded placeholder outputs.

## 2. Separation of Concerns & Shared SSOT Configuration

1. **Configuration (`prototype/scene_config.py`)**:
   - Hierarchical `@dataclass` definitions (`PipelineConfig`, `SceneAssetConfig`, `CameraRigConfig`) with `.to_dict()`, `.from_dict()`, `.to_json()`, `.from_json()`.
   - Zero magic numbers inline inside `build_assets.py` or `scene_manager.js`.
2. **Offline AI Asset Builder (`prototype/build_assets.py`)**:
   - Pure offline pipeline consuming `PipelineConfig` and authentic `media/` files, emitting verified RGBA layers, LaMa-inpainted background plates, 8-bit/16-bit normalized depth maps (`*_depth.png`), and `scene_manifest.json`.
3. **Runtime Engine (`prototype/js/scene_manager.js`) & UI (`prototype/js/app.js`)**:
   - `scene_manager.js` handles WebGL2 geometry, depth-displaced multi-plane meshes, soft procedural radial dust sprites, and touch/gyro camera kinematics.
   - `app.js` handles RTL Hebrew UI chrome, curatorial drawer, and Web Audio soundscape.
4. **Server (`prototype/serve.py`)**:
   - Threaded HTTP serving, gzip compression, port-collision avoidance, and `/__diag` telemetry collection.

## 3. Read-Only Safety & Governance Integrity

- Preserve 100% read-only guarantees tested in `tests/test_governance.py` (`TestReadOnlySafetyGuardrails`).
- Do not add stray `.md` files to the repository root (`/Users/talsheaffer/code/tami`), as `tools/verify_wiki_links.py --strict-orphans` validates the exact root governance markdown graph; store transient files in the Antigravity `scratch/` directory.

## 4. Empirical Verification Protocol

Before declaring any prototype or tool change complete:
1. Run syntax & type checks: `.venv/bin/python` imports, `node --check` on all `.js` files.
2. Render **both** desktop (`1440x900`) and mobile portrait (`411x705` / `390x844` via CDP mobile emulation) screenshots into `scratch/` and **visually inspect the rendered PNGs** with `read_file` to verify zero cutoff polygons, zero blur holes, zero untextured point squares, and clean parallax framing.
3. Run the automated verification suites:
   - `python3 -m unittest discover -s tests`
   - `python3 tools/verify_wiki_links.py --wiki-dir wiki/ --strict-orphans`
