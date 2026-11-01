#!/usr/bin/env python3
r"""
tests/test_depth_principles.py — Automated 2.5D Depth Topology Invariant Gate

Enforces the 5 mandatory laws recorded in
`.agents/skills/coding-guidelines/references/depth_topology_principles.md`:
  1. Law 1 ($C^0$ Contact Continuity): $|\Delta Z_{\text{world}}| \le \text{tol}$ where objects touch
     supporting floor, step, tabletop, sofa, or wall surfaces.
  2. Law 2 ($C^{-1}$ Free-Silhouette Discontinuity): `pinMarginUv == 0.0` and outward `EDT`
     depth extrapolation across every displaced alpha-cutout silhouette so straddling mesh
     triangles never rubber-sheet toward `0` / wall depth.
  3. Law 3 (Ghost-Free Occluded `layer0` Depth): harmonic healing removes foreground depth mounds
     from inpainted room floors/walls.
  4. Law 4 (Depth Monotonicity & Dynamic Range): zero silent clamping (`ValueError` on overflow)
     and strict back-wall clearance behind architectural tunnels (`plan_b_pipe`, `plan_b_pipe_v3`).
  5. Law 5 (`boreSightAtRest` & Automated Regression Gate): enforced across active scenes in
     `prototype/js/scene_manager.js` and materialized textures in `prototype/assets/scenes/`.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from prototype.depth_topology import (
    ContactAnchorBox,
    ContactAnchoredLayerDepthConfig,
    LayerTopologyVerificationConfig,
    LayerTopologyVerificationReport,
    RoomDepthCleanConfig,
    clean_room_depth_under_cutouts,
    compute_room_relative_neural_protrusion,
    compute_world_z_map,
    extract_contact_anchor_mask,
    extract_layer_window_from_expanded,
    solve_contact_anchored_layer_depth,
    verify_layer_contact_and_discontinuity,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
PROTOTYPE_DIR: Path = REPO_ROOT / "prototype"
SCENES_DIR: Path = PROTOTYPE_DIR / "assets" / "scenes"
SCENE_MANAGER_JS: Path = PROTOTYPE_DIR / "js" / "scene_manager.js"
PRINCIPLES_DOC: Path = (
    REPO_ROOT / ".agents" / "skills" / "coding-guidelines" / "references" / "depth_topology_principles.md"
)


@dataclass(frozen=True)
class DepthTestThresholds:
    """Canonical numerical tolerances for synthetic and live-asset topology tests."""

    synthetic_height: int = 180
    synthetic_width: int = 270
    synthetic_inner_height: int = 100
    synthetic_inner_width: int = 150
    room_pin_margin_uv: float = 0.045
    max_synthetic_heal_err_u8: float = 1.5
    live_rim_erosion_px: int = 2
    live_rim_probe_px: int = 6
    max_live_rim_jump_world: float = 3.0
    max_live_contact_p95_dz_world: float = 2.5
    max_live_contact_max_dz_world: float = 4.0
    shelf_plinth_y_expanded: int = 1243
    shelf_plinth_x0_expanded: int = 1062
    shelf_plinth_x1_expanded: int = 1638
    max_shelf_plinth_dz_world: float = 3.0


THRESHOLDS = DepthTestThresholds()

BORE_SIGHT_REQUIRED_SCENES: tuple[str, ...] = (
    "shelf_portal",
    "paper_capsules",
    "tzofen_brick_table",
    "bene_beitcha",
    "humpty_dumpty_bundle",
    "denya_model",
    "denya_living_room",
    "denya_couch_guitar",
    "wandering_drawing_1",
    "scroll_palm_world",
    "wandering_drawing",
    "fried_yard_installation",
    "fried_house_railing",
    "fried_table_hybrid",
    "herzliya_tree_canopy",
    "marseille_mirror",
)


def _extract_js_scene_block(js_source: str, scene_key: str) -> str:
    """Extracts a top-level `sceneConfigs` object literal block for `scene_key` from `scene_manager.js`."""
    pattern = re.compile(rf"\b{re.escape(scene_key)}\s*:\s*\{{")
    match = pattern.search(js_source)
    if not match:
        raise AssertionError(f"Scene {scene_key!r} not found in {SCENE_MANAGER_JS}.")
    start = match.end() - 1
    depth = 0
    for idx in range(start, len(js_source)):
        ch = js_source[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return js_source[start : idx + 1]
    raise AssertionError(f"Unbalanced braces in scene block {scene_key!r}.")


def _extract_extra_layer_blocks(scene_block: str) -> dict[str, str]:
    """Extracts brace-balanced `{ id: '...', ... }` object blocks from `extraLayers: [...]`."""
    extra_match = re.search(r"\bextraLayers\s*:\s*\[", scene_block)
    if not extra_match:
        return {}
    arr_start = extra_match.end() - 1
    bracket_depth = 0
    arr_end = len(scene_block)
    for idx in range(arr_start, len(scene_block)):
        ch = scene_block[idx]
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth == 0:
                arr_end = idx + 1
                break
    arr_text = scene_block[arr_start + 1 : arr_end - 1]
    result: dict[str, str] = {}
    pos = 0
    while pos < len(arr_text):
        if arr_text[pos] == "{":
            brace_depth = 0
            in_template_expr = 0
            end_idx = pos
            i = pos
            while i < len(arr_text):
                if arr_text[i : i + 2] == "${":
                    in_template_expr += 1
                    i += 2
                    continue
                ch = arr_text[i]
                if ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    if in_template_expr > 0:
                        in_template_expr -= 1
                    else:
                        brace_depth -= 1
                        if brace_depth == 0:
                            end_idx = i + 1
                            break
                i += 1
            obj_text = arr_text[pos:end_idx]
            id_m = re.search(r"\bid\s*:\s*['\"]([^'\"]+)['\"]", obj_text)
            if id_m:
                result[id_m.group(1)] = obj_text
            pos = max(pos + 1, end_idx)
        else:
            pos += 1
    return result


def _parse_float_prop(block: str, prop_name: str, default: float) -> float:
    m = re.search(rf"\b{re.escape(prop_name)}\s*:\s*(-?\d+(?:\.\d+)?)", block)
    return float(m.group(1)) if m else default


def _parse_asset_prop(block: str, prop_name: str) -> Path | None:
    m = re.search(
        rf"\b{re.escape(prop_name)}\s*:\s*[`'\"]([^`'\"]+?)(?:\?\$\{{assetVer\}})?[`'\"]",
        block,
    )
    if not m:
        return None
    return PROTOTYPE_DIR / m.group(1)


class TestDepthTopologyUnitInvariants(unittest.TestCase):
    """Tests mathematical solvers, fail-fast validation, and lossless config serialization."""

    def test_principles_reference_documented(self) -> None:
        self.assertTrue(PRINCIPLES_DOC.is_file(), f"Missing {PRINCIPLES_DOC}")
        text = PRINCIPLES_DOC.read_text(encoding="utf-8")
        self.assertIn("Contact Continuity", text)
        self.assertIn("Discontinuity", text)
        self.assertIn("pinMarginUv", text)

    def test_config_roundtrip_serialization(self) -> None:
        clean_cfg = RoomDepthCleanConfig(
            alpha_threshold_u8=18,
            mask_dilation_px=14,
            extra_clean_boxes_expanded=((10, 20, 40, 50),),
            preserve_boxes_expanded=((0, 0, 8, 8),),
            post_smooth_sigma_px=2.0,
            interior_blend_band_px=8.0,
        )
        self.assertEqual(clean_cfg, RoomDepthCleanConfig.from_json(clean_cfg.to_json()))

        solve_cfg = ContactAnchoredLayerDepthConfig(
            bg_z=-350.0,
            room_depth_amplitude=120.0,
            room_pin_margin_uv=THRESHOLDS.room_pin_margin_uv,
            layer_z=-290.0,
            layer_depth_amplitude=140.0,
            layer_pin_margin_uv=0.0,
            contact_boxes=(
                ContactAnchorBox(x0=10, y0=12, x1=35, y1=60, mode="left_band", band_px=4),
                ContactAnchorBox(x0=30, y0=50, x1=90, y1=85, mode="bottom_band", band_px=5),
            ),
            neural_relief_amplitude=22.0,
        )
        self.assertEqual(solve_cfg, ContactAnchoredLayerDepthConfig.from_json(solve_cfg.to_json()))

        verify_cfg = LayerTopologyVerificationConfig()
        self.assertEqual(verify_cfg, LayerTopologyVerificationConfig.from_json(verify_cfg.to_json()))

        report = LayerTopologyVerificationReport(
            contact_pixel_count=42,
            opaque_pixel_count=400,
            rim_probe_pixel_count=180,
            max_abs_contact_dz_world=0.45,
            p95_abs_contact_dz_world=0.31,
            mean_abs_contact_dz_world=0.18,
            max_abs_rim_jump_world=0.0,
            p95_abs_rim_jump_world=0.0,
            min_signed_dz_vs_host_world=0.0,
            contact_passed=True,
            discontinuity_passed=True,
            monotonicity_passed=True,
            all_passed=True,
        )
        self.assertEqual(report, LayerTopologyVerificationReport.from_json(report.to_json()))

    def test_fail_fast_validation_rejects_bad_pin_or_range_overflow(self) -> None:
        with self.assertRaises(ValueError):
            ContactAnchoredLayerDepthConfig(
                bg_z=-350.0,
                room_depth_amplitude=100.0,
                room_pin_margin_uv=THRESHOLDS.room_pin_margin_uv,
                layer_z=-200.0,
                layer_depth_amplitude=80.0,
                layer_pin_margin_uv=0.015,
            )

        h, w = THRESHOLDS.synthetic_height, THRESHOLDS.synthetic_width
        ih, iw = THRESHOLDS.synthetic_inner_height, THRESHOLDS.synthetic_inner_width
        room_depth = np.full((h, w), 200, dtype=np.uint8)
        cutout = np.zeros((ih, iw, 4), dtype=np.uint8)
        cutout[30:70, 40:90, 3] = 255
        # Room Z is -350 + 100*(200/255) = -271.57, which overflows [-250, -210]!
        overflow_cfg = ContactAnchoredLayerDepthConfig(
            bg_z=-350.0,
            room_depth_amplitude=100.0,
            room_pin_margin_uv=0.0,
            layer_z=-250.0,
            layer_depth_amplitude=40.0,
            neural_relief_amplitude=0.0,
        )
        with self.assertRaises(ValueError):
            solve_contact_anchored_layer_depth(room_depth, cutout, overflow_cfg)

    def test_clean_room_depth_removes_foreground_ghost_mound(self) -> None:
        h, w = THRESHOLDS.synthetic_height, THRESHOLDS.synthetic_width
        ih, iw = THRESHOLDS.synthetic_inner_height, THRESHOLDS.synthetic_inner_width
        yy = np.linspace(20.0, 220.0, h, dtype=np.float32)[:, None]
        true_floor = np.rint(np.broadcast_to(yy, (h, w))).astype(np.uint8)

        contaminated = true_floor.copy()
        pad_y = (h - ih) // 2
        pad_x = (w - iw) // 2
        cutout = np.zeros((ih, iw, 4), dtype=np.uint8)
        cutout[25:75, 50:100, 3] = 255
        contaminated[pad_y + 25 : pad_y + 75, pad_x + 50 : pad_x + 100] = 250

        cleaned = clean_room_depth_under_cutouts(
            room_depth=contaminated,
            cutout_layers=[cutout],
            cfg=RoomDepthCleanConfig(
                alpha_threshold_u8=16,
                mask_dilation_px=6,
                post_smooth_sigma_px=1.0,
                interior_blend_band_px=5.0,
            ),
        )
        err = np.abs(cleaned.astype(np.float32) - true_floor.astype(np.float32))
        self.assertLessEqual(float(err.max()), THRESHOLDS.max_synthetic_heal_err_u8)

    def test_multi_anchor_handrail_and_floor_sculpture_solver(self) -> None:
        h, w = THRESHOLDS.synthetic_height, THRESHOLDS.synthetic_width
        ih, iw = THRESHOLDS.synthetic_inner_height, THRESHOLDS.synthetic_inner_width
        yy = np.linspace(0.1, 0.85, h, dtype=np.float32)[:, None]
        xx = np.linspace(-0.05, 0.05, w, dtype=np.float32)[None, :]
        room_u8 = np.rint(np.clip(yy + xx, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Synthetic wall-bolted diagonal handrail with 2 vertical step posts
        rail_rgba = np.zeros((ih, iw, 4), dtype=np.uint8)
        for x in range(15, 125):
            y_bar = int(round(22 + 0.36 * (x - 15)))
            rail_rgba[max(0, y_bar - 2) : min(ih, y_bar + 3), x, :3] = (70, 120, 80)
            rail_rgba[max(0, y_bar - 2) : min(ih, y_bar + 3), x, 3] = 255
        for post_x, post_y1 in ((55, 76), (112, 92)):
            y_top = int(round(22 + 0.36 * (post_x - 15)))
            rail_rgba[y_top:post_y1, post_x - 2 : post_x + 3, 3] = 255

        solve_cfg = ContactAnchoredLayerDepthConfig(
            bg_z=-310.0,
            room_depth_amplitude=210.0,
            room_pin_margin_uv=THRESHOLDS.room_pin_margin_uv,
            layer_z=-295.0,
            layer_depth_amplitude=220.0,
            layer_pin_margin_uv=0.0,
            contact_boxes=(
                ContactAnchorBox(x0=14, y0=18, x1=24, y1=30, mode="left_band", band_px=4),
                ContactAnchorBox(x0=50, y0=68, x1=62, y1=78, mode="bottom_band", band_px=4),
                ContactAnchorBox(x0=107, y0=84, x1=118, y1=94, mode="bottom_band", band_px=4),
            ),
            neural_relief_amplitude=14.0,
            enforce_front_of_host=True,
        )
        raw_neural = room_u8.copy()
        protrusion = compute_room_relative_neural_protrusion(
            raw_neural_depth_expanded=raw_neural,
            cutout_rgba=rail_rgba,
            clean_cfg=RoomDepthCleanConfig(mask_dilation_px=5),
        )
        solved = solve_contact_anchored_layer_depth(
            room_depth_expanded=room_u8,
            cutout_rgba=rail_rgba,
            cfg=solve_cfg,
            neural_protrusion_norm=protrusion,
        )
        report = verify_layer_contact_and_discontinuity(
            room_depth_expanded=room_u8,
            layer_rgba=rail_rgba,
            layer_depth=solved.depth_u8,
            contact_mask=solved.contact_mask,
            bg_z=solve_cfg.bg_z,
            room_depth_amplitude=solve_cfg.room_depth_amplitude,
            room_pin_margin_uv=solve_cfg.room_pin_margin_uv,
            layer_z=solve_cfg.layer_z,
            layer_depth_amplitude=solve_cfg.layer_depth_amplitude,
            layer_pin_margin_uv=solve_cfg.layer_pin_margin_uv,
            verify_cfg=LayerTopologyVerificationConfig(),
        )
        self.assertTrue(report.all_passed, report.to_json())


class TestLiveSceneDepthTopologyGate(unittest.TestCase):
    """Audits active runtime configs in `scene_manager.js` and materialized textures on disk."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.js_source = SCENE_MANAGER_JS.read_text(encoding="utf-8")

    def test_all_multi_layer_scenes_enable_bore_sight_at_rest(self) -> None:
        for scene_key in BORE_SIGHT_REQUIRED_SCENES:
            block = _extract_js_scene_block(self.js_source, scene_key)
            self.assertRegex(
                block,
                r"\bboreSightAtRest\s*:\s*true\b",
                f"Scene {scene_key!r} must set `boreSightAtRest: true` (Law 5).",
            )

    def test_displaced_cutout_layers_enforce_zero_pin_margin_and_edt_rim(self) -> None:
        for scene_key in BORE_SIGHT_REQUIRED_SCENES + ("plan_b_pipe", "plan_b_pipe_v3"):
            block = _extract_js_scene_block(self.js_source, scene_key)

            # Check midDepthTexture / fgDepthTexture if configured on the scene
            for prefix, tex_prop, amp_prop, pin_prop in (
                ("mid", "midDepthTexture", "midDepthAmplitude", "midPinMarginUv"),
                ("fg", "fgDepthTexture", "fgDepthAmplitude", "fgPinMarginUv"),
            ):
                if re.search(rf"\b{amp_prop}\s*:\s*\d", block):
                    color_path = _parse_asset_prop(block, f"{prefix}Texture")
                    depth_path = _parse_asset_prop(block, tex_prop)
                    amp = _parse_float_prop(block, amp_prop, 0.0)
                    pin = _parse_float_prop(block, pin_prop, 0.0)
                    self.assertEqual(
                        pin,
                        0.0,
                        f"{scene_key}.{pin_prop} must be 0.0 on cutout layers (Law 2).",
                    )
                    self.assertIsNotNone(color_path, f"Missing {prefix}Texture in {scene_key}")
                    self.assertIsNotNone(depth_path, f"Missing {tex_prop} in {scene_key}")
                    assert color_path is not None and depth_path is not None
                    self._assert_edt_rim_continuity(
                        label=f"{scene_key}.{prefix}",
                        rgba_path=color_path,
                        depth_path=depth_path,
                        amplitude=amp,
                    )

            # Check extraLayers with depthTexture + depthAmplitude
            for extra_id, extra_body in _extract_extra_layer_blocks(block).items():
                if "depthTexture" not in extra_body or "depthAmplitude" not in extra_body:
                    continue
                pin_match = re.search(r"\bpinMarginUv\s*:\s*(-?\d+(?:\.\d+)?)", extra_body)
                self.assertIsNotNone(
                    pin_match,
                    f"{scene_key}.extraLayers[{extra_id!r}] with depthTexture must explicitly set `pinMarginUv: 0.0`.",
                )
                assert pin_match is not None
                self.assertEqual(
                    float(pin_match.group(1)),
                    0.0,
                    f"{scene_key}.extraLayers[{extra_id!r}].pinMarginUv must be 0.0.",
                )
                tex_path = _parse_asset_prop(extra_body, "texture")
                d_path = _parse_asset_prop(extra_body, "depthTexture")
                amp = _parse_float_prop(extra_body, "depthAmplitude", 0.0)
                if tex_path and d_path and amp > 0.0:
                    self._assert_edt_rim_continuity(
                        label=f"{scene_key}.extra[{extra_id}]",
                        rgba_path=tex_path,
                        depth_path=d_path,
                        amplitude=amp,
                    )

    def _assert_edt_rim_continuity(
        self,
        label: str,
        rgba_path: Path,
        depth_path: Path,
        amplitude: float,
    ) -> None:
        self.assertTrue(rgba_path.is_file(), f"{label}: missing RGBA {rgba_path}")
        self.assertTrue(depth_path.is_file(), f"{label}: missing depth {depth_path}")
        rgba = np.asarray(Image.open(rgba_path).convert("RGBA"))
        depth = np.asarray(Image.open(depth_path).convert("L"))
        self.assertEqual(
            rgba.shape[:2],
            depth.shape,
            f"{label}: RGBA shape {rgba.shape[:2]} != depth shape {depth.shape}",
        )
        opaque = rgba[:, :, 3] >= 20
        self.assertTrue(bool(np.any(opaque)), f"{label}: cutout has zero opaque pixels")
        if bool(np.all(opaque)):
            return

        dist_out, nearest_in = ndimage.distance_transform_edt(~opaque, return_indices=True)
        rim = (~opaque) & (dist_out <= float(THRESHOLDS.live_rim_probe_px))
        if not np.any(rim):
            return
        ry, rx = np.where(rim)
        ny, nx = nearest_in[0, ry, rx], nearest_in[1, ry, rx]
        dz_world = (
            np.abs(depth[ry, rx].astype(np.float32) - depth[ny, nx].astype(np.float32))
            / 255.0
            * float(amplitude)
        )
        p99_jump = float(np.percentile(dz_world, 99.0))
        self.assertLessEqual(
            p99_jump,
            THRESHOLDS.max_live_rim_jump_world,
            f"{label}: exterior rim depth jumps by {p99_jump:.2f} world units (missing outward EDT!).",
        )

    def test_plan_b_pipe_wall_strictly_behind_pipe_shell(self) -> None:
        for scene_key in ("plan_b_pipe", "plan_b_pipe_v3"):
            block = _extract_js_scene_block(self.js_source, scene_key)
            bg_z = _parse_float_prop(block, "bgZ", -350.0)
            room_amp = _parse_float_prop(block, "roomDepthAmplitude", 92.0)
            bg_depth_path = _parse_asset_prop(block, "depthTexture")
            assert bg_depth_path is not None

            extras = _extract_extra_layer_blocks(block)
            self.assertIn("pipe_shell", extras, f"Missing pipe_shell in {scene_key}")
            pipe_body = extras["pipe_shell"]
            pipe_z = _parse_float_prop(pipe_body, "zPos", -430.0)
            pipe_amp = _parse_float_prop(pipe_body, "depthAmplitude", 470.0)
            pipe_rgba_path = _parse_asset_prop(pipe_body, "texture")
            pipe_depth_path = _parse_asset_prop(pipe_body, "depthTexture")
            assert pipe_rgba_path is not None and pipe_depth_path is not None

            bg_d = np.asarray(Image.open(bg_depth_path).convert("L"))
            pipe_rgba = np.asarray(Image.open(pipe_rgba_path).convert("RGBA"))
            pipe_d = np.asarray(Image.open(pipe_depth_path).convert("L"))

            z_bg = compute_world_z_map(bg_d, bg_z, room_amp, THRESHOLDS.room_pin_margin_uv)
            z_pipe = compute_world_z_map(pipe_d, pipe_z, pipe_amp, 0.0)
            pipe_mask = pipe_rgba[:, :, 3] >= 128
            min_clearance = float((z_pipe[pipe_mask] - z_bg[pipe_mask]).min())
            self.assertGreater(
                min_clearance,
                5.0,
                f"{scene_key}: background wall pokes within {min_clearance:.2f} units of pipe_shell!",
            )

    def test_shelf_portal_cabinet_plinth_matches_gallery_floor_depth(self) -> None:
        block = _extract_js_scene_block(self.js_source, "shelf_portal")
        bg_z = _parse_float_prop(block, "bgZ", -350.0)
        fg_z = _parse_float_prop(block, "fgZ", 0.0)
        room_amp = _parse_float_prop(block, "roomDepthAmplitude", 96.0)
        bg_depth_path = _parse_asset_prop(block, "depthTexture")
        assert bg_depth_path is not None and bg_depth_path.is_file()

        bg_d = np.asarray(Image.open(bg_depth_path).convert("L"))
        z_bg = compute_world_z_map(bg_d, bg_z, room_amp, THRESHOLDS.room_pin_margin_uv)
        plinth_z = z_bg[
            THRESHOLDS.shelf_plinth_y_expanded,
            THRESHOLDS.shelf_plinth_x0_expanded : THRESHOLDS.shelf_plinth_x1_expanded,
        ]
        max_plinth_dz = float(np.abs(plinth_z - fg_z).max())
        self.assertLessEqual(
            max_plinth_dz,
            THRESHOLDS.max_shelf_plinth_dz_world,
            f"shelf_portal cabinet plinth (fgZ={fg_z}) floats by {max_plinth_dz:.2f} units vs gallery floor!",
        )

    def test_bene_beitcha_sculptures_anchored_to_museum_floor(self) -> None:
        block = _extract_js_scene_block(self.js_source, "bene_beitcha")
        bg_z = _parse_float_prop(block, "bgZ", -350.0)
        room_amp = _parse_float_prop(block, "roomDepthAmplitude", 92.0)
        bg_depth_path = _parse_asset_prop(block, "depthTexture")
        assert bg_depth_path is not None
        bg_d = np.asarray(Image.open(bg_depth_path).convert("L"))

        for prefix, z_prop, amp_prop, d_prop in (
            ("mid", "midZ", "midDepthAmplitude", "midDepthTexture"),
            ("fg", "fgZ", "fgDepthAmplitude", "fgDepthTexture"),
        ):
            d_path = _parse_asset_prop(block, d_prop)
            self.assertIsNotNone(
                d_path,
                f"bene_beitcha must configure `{d_prop}` so `{prefix}` sculptures do not float on a flat plane!",
            )
            c_path = _parse_asset_prop(block, f"{prefix}Texture")
            assert d_path is not None and c_path is not None
            layer_z = _parse_float_prop(block, z_prop, -150.0 if prefix == "mid" else 0.0)
            layer_amp = _parse_float_prop(block, amp_prop, 0.0)

            rgba = np.asarray(Image.open(c_path).convert("RGBA"))
            ld = np.asarray(Image.open(d_path).convert("L"))
            dummy_cfg = ContactAnchoredLayerDepthConfig(
                bg_z=bg_z,
                room_depth_amplitude=room_amp,
                room_pin_margin_uv=THRESHOLDS.room_pin_margin_uv,
                layer_z=layer_z,
                layer_depth_amplitude=max(1.0, layer_amp),
            )
            contact = extract_contact_anchor_mask(rgba[:, :, 3] >= 20, dummy_cfg)
            room_z_exp = compute_world_z_map(bg_d, bg_z, room_amp, THRESHOLDS.room_pin_margin_uv)
            host_z = extract_layer_window_from_expanded(room_z_exp, rgba.shape[:2])
            obj_z = compute_world_z_map(ld, layer_z, layer_amp, 0.0)
            dz = np.abs(obj_z[contact] - host_z[contact])
            p95_dz = float(np.percentile(dz, 95.0))
            self.assertLessEqual(
                p95_dz,
                THRESHOLDS.max_live_contact_p95_dz_world,
                f"bene_beitcha.{prefix} floats above museum floor (p95 |ΔZ| = {p95_dz:.2f})!",
            )

    def test_fried_house_handrail_anchored_to_wall_and_steps(self) -> None:
        block = _extract_js_scene_block(self.js_source, "fried_house_railing")
        bg_z = _parse_float_prop(block, "bgZ", -310.0)
        room_amp = _parse_float_prop(block, "roomDepthAmplitude", 210.0)
        bg_depth_path = _parse_asset_prop(block, "depthTexture")
        assert bg_depth_path is not None

        extras = _extract_extra_layer_blocks(block)
        self.assertIn("green_handrail", extras, "Missing green_handrail in fried_house_railing")
        rail_body = extras["green_handrail"]
        rail_d_path = _parse_asset_prop(rail_body, "depthTexture")
        self.assertIsNotNone(
            rail_d_path,
            "fried_house_railing.green_handrail must configure `depthTexture` so it stays bolted to the wall!",
        )
        rail_c_path = _parse_asset_prop(rail_body, "texture")
        rail_z = _parse_float_prop(rail_body, "zPos", -156.0)
        rail_amp = _parse_float_prop(rail_body, "depthAmplitude", 0.0)
        assert rail_d_path is not None and rail_c_path is not None

        bg_d = np.asarray(Image.open(bg_depth_path).convert("L"))
        rail_rgba = np.asarray(Image.open(rail_c_path).convert("RGBA"))
        rail_d = np.asarray(Image.open(rail_d_path).convert("L"))

        room_z_exp = compute_world_z_map(bg_d, bg_z, room_amp, THRESHOLDS.room_pin_margin_uv)
        host_z = extract_layer_window_from_expanded(room_z_exp, rail_rgba.shape[:2])
        obj_z = compute_world_z_map(rail_d, rail_z, rail_amp, 0.0)

        opaque = rail_rgba[:, :, 3] >= 20
        xs_all = np.where(opaque)[1]
        x_left = int(xs_all.min())
        wall_bolt = opaque.copy()
        wall_bolt[:, x_left + 8 :] = False
        self.assertTrue(bool(np.any(wall_bolt)), "Empty wall bolt mask on green_handrail")
        wall_bolt_dz = float(np.percentile(np.abs(obj_z[wall_bolt] - host_z[wall_bolt]), 95.0))
        self.assertLessEqual(
            wall_bolt_dz,
            THRESHOLDS.max_live_contact_p95_dz_world,
            f"fried_house_railing.green_handrail detaches from house wall (p95 |ΔZ| = {wall_bolt_dz:.2f})!",
        )

    def test_marseille_mirror_reflected_people_anchored_to_stone_plaza(self) -> None:
        block = _extract_js_scene_block(self.js_source, "marseille_mirror")
        bg_z = _parse_float_prop(block, "bgZ", -700.0)
        room_amp = _parse_float_prop(block, "roomDepthAmplitude", 105.0)
        bg_depth_path = _parse_asset_prop(block, "depthTexture")
        assert bg_depth_path is not None

        extras = _extract_extra_layer_blocks(block)
        self.assertIn("refl_people", extras, "Missing refl_people in marseille_mirror")
        refl_body = extras["refl_people"]
        refl_d_path = _parse_asset_prop(refl_body, "depthTexture")
        self.assertIsNotNone(
            refl_d_path,
            "marseille_mirror.refl_people must configure `depthTexture` so reflected pedestrians anchor to virtual plaza!",
        )
        refl_c_path = _parse_asset_prop(refl_body, "texture")
        refl_z = _parse_float_prop(refl_body, "zPos", -660.0)
        refl_amp = _parse_float_prop(refl_body, "depthAmplitude", 0.0)
        assert refl_d_path is not None and refl_c_path is not None

        bg_d = np.asarray(Image.open(bg_depth_path).convert("L"))
        refl_rgba = np.asarray(Image.open(refl_c_path).convert("RGBA"))
        refl_d = np.asarray(Image.open(refl_d_path).convert("L"))

        dummy_cfg = ContactAnchoredLayerDepthConfig(
            bg_z=bg_z,
            room_depth_amplitude=room_amp,
            room_pin_margin_uv=THRESHOLDS.room_pin_margin_uv,
            layer_z=refl_z,
            layer_depth_amplitude=max(1.0, refl_amp),
            contact_boxes=(
                ContactAnchorBox(x0=945, y0=482, x1=1029, y1=549, mode="top_band", band_px=8),
                ContactAnchorBox(x0=1389, y0=734, x1=1495, y1=851, mode="left_band", band_px=8),
            ),
        )
        contact = extract_contact_anchor_mask(refl_rgba[:, :, 3] >= 20, dummy_cfg)
        room_z_exp = compute_world_z_map(bg_d, bg_z, room_amp, THRESHOLDS.room_pin_margin_uv)
        host_z = extract_layer_window_from_expanded(room_z_exp, refl_rgba.shape[:2])
        obj_z = compute_world_z_map(refl_d, refl_z, refl_amp, 0.0)
        p95_dz = float(np.percentile(np.abs(obj_z[contact] - host_z[contact]), 95.0))
        self.assertLessEqual(
            p95_dz,
            THRESHOLDS.max_live_contact_p95_dz_world,
            f"marseille_mirror.refl_people floats off reflected stone plaza (p95 |ΔZ| = {p95_dz:.2f})!",
        )

    def test_denya_living_and_couch_wall_frames_and_furniture_anchored(self) -> None:
        for scene_key in ("denya_living_room", "denya_couch_guitar"):
            block = _extract_js_scene_block(self.js_source, scene_key)
            bg_z = _parse_float_prop(block, "bgZ", -350.0)
            room_amp = _parse_float_prop(block, "roomDepthAmplitude", 102.0)
            bg_depth_path = _parse_asset_prop(block, "depthTexture")
            assert bg_depth_path is not None

            for d_prop in ("midDepthTexture", "fgDepthTexture"):
                self.assertIsNotNone(
                    _parse_asset_prop(block, d_prop),
                    f"{scene_key} must configure `{d_prop}` for contact-anchored 3D furniture/people!",
                )

            extras = _extract_extra_layer_blocks(block)
            self.assertIn("wandering2_wall_frame", extras, f"Missing wandering2_wall_frame in {scene_key}")
            frame_body = extras["wandering2_wall_frame"]
            frame_c_path = _parse_asset_prop(frame_body, "texture")
            frame_d_path = _parse_asset_prop(frame_body, "depthTexture")
            frame_z = _parse_float_prop(frame_body, "zPos", bg_z)
            frame_amp = _parse_float_prop(frame_body, "depthAmplitude", 0.0)
            self.assertIsNotNone(
                frame_d_path,
                f"{scene_key}.wandering2_wall_frame must configure `depthTexture` to stay flush with wall!",
            )
            assert frame_c_path is not None and frame_d_path is not None

            bg_d = np.asarray(Image.open(bg_depth_path).convert("L"))
            frame_rgba = np.asarray(Image.open(frame_c_path).convert("RGBA"))
            frame_d = np.asarray(Image.open(frame_d_path).convert("L"))

            room_z_exp = compute_world_z_map(bg_d, bg_z, room_amp, THRESHOLDS.room_pin_margin_uv)
            host_z = extract_layer_window_from_expanded(room_z_exp, frame_rgba.shape[:2])
            obj_z = compute_world_z_map(frame_d, frame_z, frame_amp, 0.0)
            opaque = frame_rgba[:, :, 3] >= 20
            p95_dz = float(np.percentile(np.abs(obj_z[opaque] - host_z[opaque]), 95.0))
            self.assertLessEqual(
                p95_dz,
                THRESHOLDS.max_live_contact_p95_dz_world,
                f"{scene_key}.wandering2_wall_frame detaches from living-room wall (p95 |ΔZ| = {p95_dz:.2f})!",
            )

    def test_new_scenes_tzofen_tree_scroll_and_humpty_contact_continuity(self) -> None:
        from prototype.scroll_humpty_scenes_builder import default_scroll_humpty_bundle_config
        from prototype.tzofen_tree_scenes_builder import default_tzofen_tree_bundle_config

        tz_bundle = default_tzofen_tree_bundle_config()
        sh_bundle = default_scroll_humpty_bundle_config()

        checks = [
            ("tzofen_brick_table", "fg", "fgZ", "fgDepthAmplitude", "fgDepthTexture", "fgTexture", tz_bundle.tzofen.fg_contact_solve),
            ("herzliya_tree_canopy", "mid", "midZ", "midDepthAmplitude", "midDepthTexture", "midTexture", tz_bundle.tree_canopy.mid_contact_solve),
            ("herzliya_tree_canopy", "fg", "fgZ", "fgDepthAmplitude", "fgDepthTexture", "fgTexture", tz_bundle.tree_canopy.fg_contact_solve),
            ("scroll_palm_world", "mid", "midZ", "midDepthAmplitude", "midDepthTexture", "midTexture", sh_bundle.scroll_palm_world.mid_depth_solve),
            ("scroll_palm_world", "fg", "fgZ", "fgDepthAmplitude", "fgDepthTexture", "fgTexture", sh_bundle.scroll_palm_world.fg_depth_solve),
            ("humpty_dumpty_bundle", "fg", "fgZ", "fgDepthAmplitude", "fgDepthTexture", "fgTexture", sh_bundle.humpty_dumpty_bundle.fg_depth_solve),
        ]
        for scene_key, layer_name, z_prop, amp_prop, d_prop, c_prop, solve_cfg in checks:
            block = _extract_js_scene_block(self.js_source, scene_key)
            bg_z = _parse_float_prop(block, "bgZ", -350.0)
            room_amp = _parse_float_prop(block, "roomDepthAmplitude", 96.0)
            bg_depth_path = _parse_asset_prop(block, "depthTexture")
            d_path = _parse_asset_prop(block, d_prop)
            c_path = _parse_asset_prop(block, c_prop)
            self.assertIsNotNone(d_path, f"{scene_key}.{layer_name} missing {d_prop}")
            assert bg_depth_path is not None and d_path is not None and c_path is not None

            layer_z = _parse_float_prop(block, z_prop, bg_z)
            layer_amp = _parse_float_prop(block, amp_prop, 0.0)
            bg_d = np.asarray(Image.open(bg_depth_path).convert("L"))
            rgba = np.asarray(Image.open(c_path).convert("RGBA"))
            ld = np.asarray(Image.open(d_path).convert("L"))

            contact = extract_contact_anchor_mask(rgba[:, :, 3] >= solve_cfg.alpha_threshold_u8, solve_cfg)
            self.assertTrue(bool(np.any(contact)), f"{scene_key}.{layer_name} produced empty contact mask")
            room_z_exp = compute_world_z_map(bg_d, bg_z, room_amp, THRESHOLDS.room_pin_margin_uv)
            host_z = extract_layer_window_from_expanded(room_z_exp, rgba.shape[:2])
            obj_z = compute_world_z_map(ld, layer_z, layer_amp, 0.0)
            p95_dz = float(np.percentile(np.abs(obj_z[contact] - host_z[contact]), 95.0))
            self.assertLessEqual(
                p95_dz,
                THRESHOLDS.max_live_contact_p95_dz_world,
                f"{scene_key}.{layer_name} floats off host surface (p95 |ΔZ| = {p95_dz:.2f})!",
            )


if __name__ == "__main__":
    unittest.main()
