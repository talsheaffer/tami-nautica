#!/usr/bin/env python3
"""
prototype/build_assets.py
Offline neural 2.5D spatial asset builder for Tamar "Tami" Sheaffer's 70th Birthday Prototype.
Executes real Depth Anything V2 monocular depth estimation (plus sculpted dome relief for archival
drawings), IS-Net & Meta SAM neural segmentation, graphite-paper stroke segmentation, cross-scene
object/drawing compositing, LaMa Fast Fourier Convolution occlusion inpainting, and wide-angle
outpaint registration & center-locked blending.

Usage:
    /Users/talsheaffer/code/tami/.venv/bin/python prototype/build_assets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import replace

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from prototype.depth_topology import (
    ContactAnchorBox,
    ContactAnchoredLayerDepthConfig,
    LayerTopologyVerificationConfig,
    RoomDepthCleanConfig,
    clean_room_depth_under_cutouts,
    compute_room_relative_neural_protrusion,
    compute_world_z_map,
    extrapolate_depth_outward_edt,
    glsl_smoothstep,
    solve_contact_anchored_layer_depth,
    verify_layer_contact_and_discontinuity,
)
from prototype.scene_config import (
    ApertureBox,
    CanvasDimensionsConfig,
    ExtraDepthLayerSpec,
    FloatingElementExtractionConfig,
    InpaintingConfig,
    InpaintRegionDilationOverride,
    NeuralPortraitReliefConfig,
    OutpaintingConfig,
    SceneBuildSpec,
    SpatialPipelineConfig,
    default_spatial_pipeline_config,
)
from prototype.vision_primitives import (
    DepthAnythingEstimator,
    LamaOcclusionInpainter,
    SalientMatteExtractor,
    SamBoxMaskExtractor,
    apply_layer_overlays,
    build_aperture_reflection_layer,
    build_ceiling_mirror_reflection_layer,
    build_diffuse_aperture_fusion_layers,
    compose_outpainted_room,
    decontaminate_matte_edge_rgb,
    extract_graphite_drawing_layers,
    harmonize_v3_rear_aperture_luminance,
    partition_sculpture_layers,
    register_expanded_foreground_plate,
    synthesize_pipe_discontinuity_assets,
)


def _scale_contact_anchor_box(box: ContactAnchorBox, scale: float) -> ContactAnchorBox:
    if abs(scale - 1.0) <= 1e-6:
        return box
    x0 = int(round(box.x0 * scale))
    y0 = int(round(box.y0 * scale))
    x1 = max(x0 + 1, int(round(box.x1 * scale)))
    y1 = max(y0 + 1, int(round(box.y1 * scale)))
    return ContactAnchorBox(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        mode=box.mode,
        band_px=max(1, int(round(box.band_px * scale))),
    )


def _scale_contact_depth_config(
    cfg: ContactAnchoredLayerDepthConfig,
    scale: float,
) -> ContactAnchoredLayerDepthConfig:
    if abs(scale - 1.0) <= 1e-6:
        return cfg
    return ContactAnchoredLayerDepthConfig(
        bg_z=cfg.bg_z,
        room_depth_amplitude=cfg.room_depth_amplitude,
        room_pin_margin_uv=cfg.room_pin_margin_uv,
        layer_z=cfg.layer_z,
        layer_depth_amplitude=cfg.layer_depth_amplitude,
        layer_pin_margin_uv=cfg.layer_pin_margin_uv,
        alpha_threshold_u8=cfg.alpha_threshold_u8,
        component_bridge_px=max(0, int(round(cfg.component_bridge_px * scale))),
        min_component_area_px=max(1, int(round(cfg.min_component_area_px * scale * scale))),
        auto_floor_band_px=max(1, int(round(cfg.auto_floor_band_px * scale))),
        auto_floor_max_rise_px=max(0, int(round(cfg.auto_floor_max_rise_px * scale))),
        contact_boxes=tuple(_scale_contact_anchor_box(b, scale) for b in cfg.contact_boxes),
        neural_relief_amplitude=cfg.neural_relief_amplitude,
        contact_relief_taper_px=cfg.contact_relief_taper_px * scale,
        enforce_front_of_host=cfg.enforce_front_of_host,
        host_clearance_tolerance_world=cfg.host_clearance_tolerance_world,
        interior_smooth_sigma_px=cfg.interior_smooth_sigma_px * scale,
        overflow_tolerance_world=cfg.overflow_tolerance_world,
    )


def _scale_room_depth_clean_config(
    cfg: RoomDepthCleanConfig,
    scale: float,
) -> RoomDepthCleanConfig:
    if abs(scale - 1.0) <= 1e-6:
        return cfg

    def _sb(b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x0 = int(round(b[0] * scale))
        y0 = int(round(b[1] * scale))
        x1 = max(x0 + 1, int(round(b[2] * scale)))
        y1 = max(y0 + 1, int(round(b[3] * scale)))
        return (x0, y0, x1, y1)

    return RoomDepthCleanConfig(
        alpha_threshold_u8=cfg.alpha_threshold_u8,
        mask_dilation_px=max(0, int(round(cfg.mask_dilation_px * scale))),
        extra_clean_boxes_expanded=tuple(_sb(b) for b in cfg.extra_clean_boxes_expanded),
        preserve_boxes_expanded=tuple(_sb(b) for b in cfg.preserve_boxes_expanded),
        post_smooth_sigma_px=cfg.post_smooth_sigma_px * scale,
        interior_blend_band_px=cfg.interior_blend_band_px * scale,
    )


def _scale_outpainting_config(cfg: OutpaintingConfig, scale: float) -> OutpaintingConfig:
    if abs(scale - 1.0) <= 1e-6:
        return cfg
    return replace(
        cfg,
        feather_collar_px=max(1, int(round(cfg.feather_collar_px * scale))),
        color_match_band_px=max(4, int(round(cfg.color_match_band_px * scale))),
        homography_patch_width=max(24, int(round(cfg.homography_patch_width * scale))),
        homography_patch_height=max(24, int(round(cfg.homography_patch_height * scale))),
        homography_search_margin_px=max(16, int(round(cfg.homography_search_margin_px * scale))),
        max_warp_border_pull_px=cfg.max_warp_border_pull_px * scale,
        similarity_coarse_scale_min=min(
            cfg.similarity_coarse_scale_min, cfg.similarity_coarse_scale_min / scale
        ),
        similarity_coarse_scale_max=max(
            cfg.similarity_coarse_scale_max, cfg.similarity_coarse_scale_max / scale
        ),
    )


def _scale_inpainting_config(
    cfg: InpaintingConfig,
    scale: float,
    jpeg_quality_override: int | None,
) -> InpaintingConfig:
    jq = jpeg_quality_override if jpeg_quality_override is not None else cfg.jpeg_quality
    if abs(scale - 1.0) <= 1e-6 and jq == cfg.jpeg_quality:
        return cfg
    return InpaintingConfig(
        dilation_radius_px=max(0, int(round(cfg.dilation_radius_px * scale))),
        mask_blur_sigma=cfg.mask_blur_sigma * scale,
        jpeg_quality=jq,
    )


def _override_floating_extraction_blur(
    fe_cfg: FloatingElementExtractionConfig,
    mirror_blur_override: tuple[float, float] | None,
) -> FloatingElementExtractionConfig:
    if mirror_blur_override is None:
        return fe_cfg
    return replace(
        fe_cfg,
        mirror_blur_base_sigma=float(mirror_blur_override[0]),
        mirror_blur_dist_gain=float(mirror_blur_override[1]),
    )


def _scale_extra_layer_spec(el: ExtraDepthLayerSpec, scale: float) -> ExtraDepthLayerSpec:
    if abs(scale - 1.0) <= 1e-6:
        return el
    return replace(
        el,
        overlays=tuple(ov.scaled(scale) for ov in el.overlays),
        sam_boxes=tuple(b.scaled(scale) for b in el.sam_boxes),
        sam_point_groups=tuple(pg.scaled(scale) for pg in el.sam_point_groups),
        oriented_matte_boxes=tuple(omb.scaled(scale) for omb in el.oriented_matte_boxes),
        exclude_boxes=tuple(eb.scaled(scale) for eb in el.exclude_boxes),
        max_pinhole_fill_px=max(0, int(round(el.max_pinhole_fill_px * scale * scale))),
        min_major_component_px=max(1, int(round(el.min_major_component_px * scale * scale))),
        occluder_inpaint_boxes=tuple(ob.scaled(scale) for ob in el.occluder_inpaint_boxes),
        right_shoulder_extend_box=(
            el.right_shoulder_extend_box.scaled(scale)
            if el.right_shoulder_extend_box is not None
            else None
        ),
        right_shoulder_extend_px=int(round(el.right_shoulder_extend_px * scale)),
        pre_lama_dilation_px=(
            max(0, int(round(el.pre_lama_dilation_px * scale)))
            if el.pre_lama_dilation_px is not None
            else None
        ),
        pre_lama_blur_sigma=(
            el.pre_lama_blur_sigma * scale if el.pre_lama_blur_sigma is not None else None
        ),
        ceiling_mirror_reflection=(
            el.ceiling_mirror_reflection.scaled(scale)
            if el.ceiling_mirror_reflection is not None
            else None
        ),
        contact_depth_solve=(
            _scale_contact_depth_config(el.contact_depth_solve, scale)
            if el.contact_depth_solve is not None
            else None
        ),
        portrait_relief=(
            el.portrait_relief.scaled(scale) if el.portrait_relief is not None else None
        ),
    )


def _scale_scene_spec_for_canvas(spec: SceneBuildSpec, scale: float) -> SceneBuildSpec:
    """Scales pixel-space coordinates on `SceneBuildSpec` from 1.0x baseline to `canvas_scale_mult`."""
    if abs(scale - 1.0) <= 1e-6:
        return spec
    return replace(
        spec,
        sam_fg_boxes=tuple(b.scaled(scale) for b in spec.sam_fg_boxes),
        sam_mid_boxes=tuple(b.scaled(scale) for b in spec.sam_mid_boxes),
        sam_mid_point_groups=tuple(pg.scaled(scale) for pg in spec.sam_mid_point_groups),
        sam_mid_exclude_boxes=tuple(eb.scaled(scale) for eb in spec.sam_mid_exclude_boxes),
        sam_mid_max_pinhole_fill_px=max(
            0, int(round(spec.sam_mid_max_pinhole_fill_px * scale * scale))
        ),
        sam_mid_min_major_component_px=max(
            1, int(round(spec.sam_mid_min_major_component_px * scale * scale))
        ),
        inpaint_region_dilation_overrides=tuple(
            ov.scaled(scale) for ov in spec.inpaint_region_dilation_overrides
        ),
        fg_overlays=tuple(ov.scaled(scale) for ov in spec.fg_overlays),
        mid_overlays=tuple(ov.scaled(scale) for ov in spec.mid_overlays),
        reflection_apertures=tuple(
            ApertureBox(
                x0=int(round(ap.x0 * scale)),
                y0=int(round(ap.y0 * scale)),
                x1=max(int(round(ap.x0 * scale)) + 1, int(round(ap.x1 * scale))),
                y1=max(int(round(ap.y0 * scale)) + 1, int(round(ap.y1 * scale))),
                max_alpha=ap.max_alpha,
                feather_sigma=ap.feather_sigma * scale,
            )
            for ap in spec.reflection_apertures
        ),
        feather_collar_px=(
            int(round(spec.feather_collar_px * scale))
            if spec.feather_collar_px is not None
            else None
        ),
        registration_core_margin_xy=(
            int(round(spec.registration_core_margin_xy[0] * scale)),
            int(round(spec.registration_core_margin_xy[1] * scale)),
        ),
        rear_chamber_soften_box=(
            spec.rear_chamber_soften_box.scaled(scale)
            if spec.rear_chamber_soften_box is not None
            else None
        ),
        sam_fg_point_groups=tuple(pg.scaled(scale) for pg in spec.sam_fg_point_groups),
        fg_oriented_matte_boxes=tuple(omb.scaled(scale) for omb in spec.fg_oriented_matte_boxes),
        sam_core_support_dilate_px=max(1, int(round(spec.sam_core_support_dilate_px * scale))),
        sam_fg_exclude_boxes=tuple(eb.scaled(scale) for eb in spec.sam_fg_exclude_boxes),
        sam_fg_right_edge_fade_x_start=(
            int(round(spec.sam_fg_right_edge_fade_x_start * scale))
            if spec.sam_fg_right_edge_fade_x_start is not None
            else None
        ),
        sam_max_pinhole_fill_px=max(0, int(round(spec.sam_max_pinhole_fill_px * scale * scale))),
        sam_min_major_component_px=max(
            1, int(round(spec.sam_min_major_component_px * scale * scale))
        ),
        extra_layers=tuple(_scale_extra_layer_spec(el, scale) for el in spec.extra_layers),
        ceiling_mirror_reflection=(
            spec.ceiling_mirror_reflection.scaled(scale)
            if spec.ceiling_mirror_reflection is not None
            else None
        ),
        room_depth_clean=(
            _scale_room_depth_clean_config(spec.room_depth_clean, scale)
            if spec.room_depth_clean is not None
            else None
        ),
        mid_contact_depth_solve=(
            _scale_contact_depth_config(spec.mid_contact_depth_solve, scale)
            if spec.mid_contact_depth_solve is not None
            else None
        ),
        fg_contact_depth_solve=(
            _scale_contact_depth_config(spec.fg_contact_depth_solve, scale)
            if spec.fg_contact_depth_solve is not None
            else None
        ),
        fg_portrait_relief=(
            spec.fg_portrait_relief.scaled(scale) if spec.fg_portrait_relief is not None else None
        ),
    )


def _build_region_aware_inpaint_mask(
    binary_mask_u8: np.ndarray,
    base_dilation_px: int,
    blur_sigma: float,
    overrides: tuple[InpaintRegionDilationOverride, ...],
) -> np.ndarray:
    """Builds a spatially varying morphological LaMa dilation mask (preserving nearby wall relief)."""
    h, w = binary_mask_u8.shape
    if base_dilation_px > 0:
        k_base = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * base_dilation_px + 1, 2 * base_dilation_px + 1)
        )
        dilated = cv2.dilate(binary_mask_u8, k_base, iterations=1)
    else:
        dilated = binary_mask_u8.copy()

    for ov in overrides:
        r = ov.dilation_radius_px
        if r <= 0:
            ov_dil = binary_mask_u8
        else:
            k_ov = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            ov_dil = cv2.dilate(binary_mask_u8, k_ov, iterations=1)
        x0 = max(0, min(w, ov.box.x0))
        x1 = max(0, min(w, ov.box.x1))
        y0 = max(0, min(h, ov.box.y0))
        y1 = max(0, min(h, ov.box.y1))
        dilated[y0:y1, x0:x1] = ov_dil[y0:y1, x0:x1]

    if blur_sigma > 0.0:
        dilated_f = ndimage.gaussian_filter(dilated.astype(np.float32), sigma=blur_sigma)
        dilated = np.where(dilated_f > 64.0, 255, 0).astype(np.uint8)
    return dilated


def _all_scene_assets_exist(output_dir: Path, spec: SceneBuildSpec) -> bool:
    """Returns True iff every texture required by `spec` already exists on disk."""
    base_files = (
        output_dir / spec.bg_output_filename,
        output_dir / spec.depth_output_filename,
        output_dir / spec.fg_output_filename,
        output_dir / spec.mid_output_filename,
    )
    if not all(p.is_file() for p in base_files):
        return False
    if spec.mid_depth_output_filename is not None and not (
        output_dir / spec.mid_depth_output_filename
    ).is_file():
        return False
    if spec.fg_depth_output_filename is not None and not (
        output_dir / spec.fg_depth_output_filename
    ).is_file():
        return False
    if spec.reflection_output_filename is not None and not (
        output_dir / spec.reflection_output_filename
    ).is_file():
        return False
    for el in spec.extra_layers:
        if not (output_dir / el.output_filename).is_file():
            return False
        if el.depth_output_filename is not None and not (
            output_dir / el.depth_output_filename
        ).is_file():
            return False
    return True


def _build_extra_layer_manifest_record(
    el_spec: ExtraDepthLayerSpec,
    canvas_wh: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Serializes an `ExtraDepthLayerSpec` into its runtime `scene_manifest.json` entry."""
    rec: dict[str, Any] = {
        "id": el_spec.layer_id,
        "labelHe": el_spec.label_he,
        "texture": f"assets/scenes/{el_spec.output_filename}",
        "zPos": el_spec.z_pos,
        "renderOrder": el_spec.render_order,
        "expanded": el_spec.is_expanded,
        "baseOpacity": el_spec.base_opacity,
        "excursionBoost": el_spec.excursion_boost,
        "depthTest": el_spec.depth_test,
        "depthWrite": el_spec.depth_write,
    }
    if canvas_wh is not None:
        rec["canvasWidth"] = int(canvas_wh[0])
        rec["canvasHeight"] = int(canvas_wh[1])
    if el_spec.depth_output_filename is not None:
        rec["depthTexture"] = f"assets/scenes/{el_spec.depth_output_filename}"
    if el_spec.depth_amplitude is not None:
        rec["depthAmplitude"] = el_spec.depth_amplitude
    if el_spec.alpha_cutout is not None:
        rec["alphaCutout"] = el_spec.alpha_cutout
    if el_spec.pin_margin_uv is not None:
        rec["pinMarginUv"] = el_spec.pin_margin_uv
    if el_spec.segments_x is not None:
        rec["segmentsX"] = el_spec.segments_x
    if el_spec.segments_y is not None:
        rec["segmentsY"] = el_spec.segments_y
    return rec


def _apply_optional_mid_fg_depth_manifest_fields(
    rec: dict[str, Any],
    spec: SceneBuildSpec,
) -> dict[str, Any]:
    """Attaches optional SSOT `mid`/`fg` depth relief & `boreSightAtRest` fields to a manifest record."""
    rec["boreSightAtRest"] = spec.bore_sight_at_rest
    if spec.mid_depth_output_filename is not None:
        rec["midDepthTexture"] = f"assets/scenes/{spec.mid_depth_output_filename}"
    if spec.mid_depth_amplitude is not None:
        rec["midDepthAmplitude"] = spec.mid_depth_amplitude
    if spec.mid_pin_margin_uv is not None:
        rec["midPinMarginUv"] = spec.mid_pin_margin_uv
    if spec.mid_segments_x is not None:
        rec["midSegmentsX"] = spec.mid_segments_x
    if spec.mid_segments_y is not None:
        rec["midSegmentsY"] = spec.mid_segments_y
    if spec.fg_depth_output_filename is not None:
        rec["fgDepthTexture"] = f"assets/scenes/{spec.fg_depth_output_filename}"
    if spec.fg_depth_amplitude is not None:
        rec["fgDepthAmplitude"] = spec.fg_depth_amplitude
    if spec.fg_pin_margin_uv is not None:
        rec["fgPinMarginUv"] = spec.fg_pin_margin_uv
    if spec.fg_segments_x is not None:
        rec["fgSegmentsX"] = spec.fg_segments_x
    if spec.fg_segments_y is not None:
        rec["fgSegmentsY"] = spec.fg_segments_y
    return rec


def solve_neural_portrait_layer_depth(
    cutout_rgba: np.ndarray,
    raw_neural_depth: np.ndarray,
    pr_cfg: NeuralPortraitReliefConfig,
    verify_cfg: LayerTopologyVerificationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Synthesizes volumetric Depth Anything V2 portrait relief with frustum-border pinning (`d = 0`
    along bleeding canvas edges) and Law 2 outward EDT silhouette extrapolation (`pinMarginUv = 0`).
    """
    if cutout_rgba.ndim != 3 or cutout_rgba.shape[2] != 4:
        raise ValueError(f"Expected HxWx4 RGBA cutout, got shape {cutout_rgba.shape}.")
    lh, lw = cutout_rgba.shape[:2]
    if raw_neural_depth.shape != (lh, lw):
        raise ValueError(
            f"raw_neural_depth shape {raw_neural_depth.shape} != cutout shape {(lh, lw)}."
        )

    opaque = cutout_rgba[:, :, 3] >= pr_cfg.alpha_threshold_u8
    if not np.any(opaque):
        raise ValueError("Portrait RGBA cutout contains zero opaque pixels.")

    labeled, num_comp = ndimage.label(opaque)
    active_opaque = np.zeros_like(opaque)
    d_interior = np.zeros((lh, lw), dtype=np.float32)
    for cid in range(1, num_comp + 1):
        comp = labeled == cid
        if int(comp.sum()) < pr_cfg.min_component_area_px:
            continue
        active_opaque |= comp
        vals = raw_neural_depth[comp]
        lo = float(np.percentile(vals, pr_cfg.lower_percentile))
        hi = float(np.percentile(vals, pr_cfg.upper_percentile))
        span = max(1e-4, hi - lo)
        d_interior[comp] = np.clip((raw_neural_depth[comp] - lo) / span, 0.0, 1.0)

    if not np.any(active_opaque):
        active_opaque = opaque.copy()
        vals = raw_neural_depth[active_opaque]
        lo = float(np.percentile(vals, pr_cfg.lower_percentile))
        hi = float(np.percentile(vals, pr_cfg.upper_percentile))
        span = max(1e-4, hi - lo)
        d_interior[active_opaque] = np.clip(
            (raw_neural_depth[active_opaque] - lo) / span, 0.0, 1.0
        )

    # Ensure any tiny opaque island (< min_component_area_px) inherits nearest active component depth
    # before smoothing/EDT so verification over all `alpha >= threshold` pixels sees zero rim jump.
    tiny_opaque = opaque & (~active_opaque)
    if np.any(tiny_opaque):
        _, near_act = ndimage.distance_transform_edt(~active_opaque, return_indices=True)
        d_interior[tiny_opaque] = d_interior[
            near_act[0, tiny_opaque], near_act[1, tiny_opaque]
        ]

    if pr_cfg.interior_smooth_sigma_px > 0.0:
        op_f = opaque.astype(np.float32)
        num = ndimage.gaussian_filter(
            d_interior * op_f, sigma=pr_cfg.interior_smooth_sigma_px, mode="nearest"
        )
        den = np.maximum(
            1e-5,
            ndimage.gaussian_filter(op_f, sigma=pr_cfg.interior_smooth_sigma_px, mode="nearest"),
        )
        d_interior = np.where(opaque, np.clip(num / den, 0.0, 1.0), 0.0).astype(np.float32)

    pin_left, pin_right, pin_top, pin_bottom = pr_cfg.pin_edges
    band = pr_cfg.border_band_px
    border_mask = np.zeros_like(opaque)
    if pin_left:
        border_mask[:, :band] |= opaque[:, :band]
    if pin_right:
        border_mask[:, lw - band :] |= opaque[:, lw - band :]
    if pin_top:
        border_mask[:band, :] |= opaque[:band, :]
    if pin_bottom:
        border_mask[lh - band :, :] |= opaque[lh - band :, :]

    if np.any(border_mask):
        dist_border = ndimage.distance_transform_edt(~border_mask).astype(np.float32)
        taper = glsl_smoothstep(0.0, pr_cfg.border_taper_px, dist_border)
        d_interior = (d_interior * taper).astype(np.float32)
        d_interior[border_mask] = 0.0

    d_extrapolated = extrapolate_depth_outward_edt(
        interior_depth=d_interior,
        opaque_mask=opaque,
    )
    depth_u8 = np.rint(np.clip(d_extrapolated, 0.0, 1.0) * 255.0).astype(np.uint8)

    world_z = compute_world_z_map(
        depth_map=depth_u8,
        plane_z=pr_cfg.layer_z,
        depth_amplitude=pr_cfg.layer_depth_amplitude,
        pin_margin_uv=pr_cfg.layer_pin_margin_uv,
    )
    dist_outside, nearest_in = ndimage.distance_transform_edt(~opaque, return_indices=True)
    exterior_rim = (~opaque) & (dist_outside <= float(verify_cfg.rim_exterior_probe_px))
    if np.any(exterior_rim):
        ry, rx = np.where(exterior_rim)
        rim_jumps = np.abs(
            world_z[ry, rx] - world_z[nearest_in[0, ry, rx], nearest_in[1, ry, rx]]
        )
        max_rim_jump = float(rim_jumps.max())
        p95_rim_jump = float(np.percentile(rim_jumps, 95.0))
        rim_count = int(len(ry))
    else:
        max_rim_jump = 0.0
        p95_rim_jump = 0.0
        rim_count = 0

    if max_rim_jump > verify_cfg.max_allowed_rim_jump_world:
        raise RuntimeError(
            f"Portrait layer EDT rim jump {max_rim_jump:.3f} exceeds max allowed "
            f"{verify_cfg.max_allowed_rim_jump_world:.3f}."
        )

    metrics = {
        "opaque_pixel_count": int(opaque.sum()),
        "border_pinned_pixel_count": int(border_mask.sum()),
        "rim_probe_pixel_count": rim_count,
        "world_z_min": float(world_z[opaque].min()),
        "world_z_max": float(world_z[opaque].max()),
        "max_abs_rim_jump_world": max_rim_jump,
        "p95_abs_rim_jump_world": p95_rim_jump,
        "discontinuity_passed": True,
    }
    return depth_u8, metrics


def _solve_and_save_contact_layer_depth(
    scene_id: str,
    layer_name: str,
    cutout_rgba: np.ndarray,
    room_depth_expanded_u8: np.ndarray,
    raw_uninpainted_exp_depth: np.ndarray,
    clean_cfg: RoomDepthCleanConfig,
    solve_cfg: ContactAnchoredLayerDepthConfig,
    verify_cfg: LayerTopologyVerificationConfig,
    depth_out_path: Path,
) -> dict[str, Any]:
    """Solves, verifies (fail-fast), and saves a contact-anchored + neural-relieved + EDT cutout depth PNG."""
    protrusion = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_uninpainted_exp_depth,
        cutout_rgba=cutout_rgba,
        clean_cfg=clean_cfg,
        alpha_threshold_u8=solve_cfg.alpha_threshold_u8,
    )
    solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=room_depth_expanded_u8,
        cutout_rgba=cutout_rgba,
        cfg=solve_cfg,
        neural_protrusion_norm=protrusion,
    )
    report = verify_layer_contact_and_discontinuity(
        room_depth_expanded=room_depth_expanded_u8,
        layer_rgba=cutout_rgba,
        layer_depth=solved.depth_u8,
        contact_mask=solved.contact_mask,
        bg_z=solve_cfg.bg_z,
        room_depth_amplitude=solve_cfg.room_depth_amplitude,
        room_pin_margin_uv=solve_cfg.room_pin_margin_uv,
        layer_z=solve_cfg.layer_z,
        layer_depth_amplitude=solve_cfg.layer_depth_amplitude,
        layer_pin_margin_uv=solve_cfg.layer_pin_margin_uv,
        verify_cfg=verify_cfg,
    )
    if not report.all_passed:
        raise RuntimeError(
            f"[{scene_id}:{layer_name}] Contact/discontinuity verification failed: "
            f"{report.to_json()}"
        )
    Image.fromarray(solved.depth_u8, mode="L").save(depth_out_path, format="PNG", optimize=True)
    print(
        f"  -> Saved {depth_out_path.name} ({solved.depth_u8.shape[1]}x{solved.depth_u8.shape[0]}, "
        f"max_contact_dz={report.max_abs_contact_dz_world:.3f}, "
        f"max_rim_jump={report.max_abs_rim_jump_world:.3f})"
    )
    return report.to_dict()


def _solve_and_save_portrait_layer_depth(
    cutout_rgba: np.ndarray,
    raw_neural_depth: np.ndarray,
    pr_cfg: NeuralPortraitReliefConfig,
    verify_cfg: LayerTopologyVerificationConfig,
    depth_out_path: Path,
) -> dict[str, Any]:
    """Solves, verifies (fail-fast), and saves a volumetric portrait relief + outward EDT depth PNG."""
    depth_u8, metrics = solve_neural_portrait_layer_depth(
        cutout_rgba=cutout_rgba,
        raw_neural_depth=raw_neural_depth,
        pr_cfg=pr_cfg,
        verify_cfg=verify_cfg,
    )
    Image.fromarray(depth_u8, mode="L").save(depth_out_path, format="PNG", optimize=True)
    print(
        f"  -> Saved {depth_out_path.name} ({depth_u8.shape[1]}x{depth_u8.shape[0]}, "
        f"z=[{metrics['world_z_min']:.2f}, {metrics['world_z_max']:.2f}], "
        f"max_rim_jump={metrics['max_abs_rim_jump_world']:.3f})"
    )
    return metrics


def _synthesize_scene_topology_depths(
    project_root: Path,
    output_dir: Path,
    scene_dims: CanvasDimensionsConfig,
    scene_outpaint_cfg: OutpaintingConfig,
    spec: SceneBuildSpec,
    depth_estimator: DepthAnythingEstimator,
    src_path: Path,
    expanded_bg_rgb: Image.Image,
    room_depth_img: Image.Image,
    depth_path: Path,
    fg_img: Image.Image,
    mid_img: Image.Image,
    fg_expanded_plate_rgb: Image.Image | None = None,
    raw_outpaint_rgb: Image.Image | None = None,
) -> dict[str, Any]:
    """
    Executes Law 3 ghost-free `layer0` room depth healing, Law 1/2/4 contact-anchored sculpture/person
    depth solving, and volumetric portrait relief + outward EDT synthesis for configured layers.
    """
    has_room_clean = spec.room_depth_clean is not None
    has_contact_solves = (
        spec.mid_contact_depth_solve is not None
        or spec.fg_contact_depth_solve is not None
        or any(el.contact_depth_solve is not None for el in spec.extra_layers)
    )
    has_portrait_solves = spec.fg_portrait_relief is not None or any(
        el.portrait_relief is not None for el in spec.extra_layers
    )
    if not (has_room_clean or has_contact_solves or has_portrait_solves):
        return {}

    extra_by_id = {el.layer_id: el for el in spec.extra_layers}
    layer_rgba_cache: dict[str, np.ndarray] = {}

    def _get_layer_rgba(layer_key: str) -> np.ndarray:
        if layer_key in layer_rgba_cache:
            return layer_rgba_cache[layer_key]
        if layer_key == "fg":
            arr = np.asarray(fg_img.convert("RGBA"), dtype=np.uint8)
        elif layer_key == "mid":
            arr = np.asarray(mid_img.convert("RGBA"), dtype=np.uint8)
        elif layer_key in extra_by_id:
            el_p = output_dir / extra_by_id[layer_key].output_filename
            if not el_p.is_file():
                raise FileNotFoundError(
                    f"Scene '{spec.scene_id}' missing extra layer RGBA for topology solve: {el_p}"
                )
            with Image.open(el_p) as el_im:
                arr = np.asarray(el_im.convert("RGBA"), dtype=np.uint8)
        else:
            raise ValueError(
                f"Scene '{spec.scene_id}' references unknown layer '{layer_key}' in topology config."
            )
        layer_rgba_cache[layer_key] = arr
        return arr

    room_depth_u8 = np.asarray(room_depth_img.convert("L"), dtype=np.uint8)
    if spec.room_depth_clean is not None:
        clean_cutouts = [_get_layer_rgba(lk) for lk in spec.room_depth_clean_layers]
        if not clean_cutouts and not spec.room_depth_clean.extra_clean_boxes_expanded:
            raise ValueError(
                f"Scene '{spec.scene_id}' sets room_depth_clean without room_depth_clean_layers "
                f"or extra_clean_boxes_expanded."
            )
        print(
            f"[{spec.scene_id}] Cleaning layer0 room depth under cutouts "
            f"{list(spec.room_depth_clean_layers)}..."
        )
        room_depth_u8 = clean_room_depth_under_cutouts(
            room_depth=room_depth_u8,
            cutout_layers=clean_cutouts,
            cfg=spec.room_depth_clean,
        )
        Image.fromarray(room_depth_u8, mode="L").save(depth_path, format="PNG", optimize=True)
        print(
            f"  -> Saved ghost-free {depth_path.name} "
            f"({room_depth_u8.shape[1]}x{room_depth_u8.shape[0]})"
        )

    if not (has_contact_solves or has_portrait_solves):
        return {}

    if spec.topology_verification is None:
        raise ValueError(
            f"Scene '{spec.scene_id}' configures cutout depth relief without topology_verification."
        )
    verify_cfg = spec.topology_verification

    orig_inner_rgb: Image.Image | None = None
    raw_uninpainted_exp_depth: np.ndarray | None = None
    raw_inner_portrait_depth: np.ndarray | None = None
    raw_expanded_portrait_depth: np.ndarray | None = None

    def _ensure_orig_inner_rgb() -> Image.Image:
        nonlocal orig_inner_rgb
        if orig_inner_rgb is None:
            orig_inner_rgb = (
                Image.open(src_path)
                .convert("RGB")
                .resize((scene_dims.width, scene_dims.height), Image.Resampling.LANCZOS)
            )
        return orig_inner_rgb

    def _ensure_uninpainted_exp_depth() -> np.ndarray:
        nonlocal raw_uninpainted_exp_depth
        if raw_uninpainted_exp_depth is None:
            ew, eh = expanded_bg_rgb.size
            if spec.inpaint_from_outpaint_inner and raw_outpaint_rgb is not None:
                comp_rgb = raw_outpaint_rgb.resize((ew, eh), Image.Resampling.LANCZOS)
            else:
                inner = _ensure_orig_inner_rgb()
                px = (ew - scene_dims.width) // 2
                py = (eh - scene_dims.height) // 2
                comp_rgb = expanded_bg_rgb.copy()
                comp_rgb.paste(inner, (px, py))
            print(
                f"[{spec.scene_id}] Computing Depth Anything V2 sculptural relief on "
                f"un-inpainted {ew}x{eh} composite..."
            )
            raw_uninpainted_exp_depth = depth_estimator.predict_depth(comp_rgb)
        return raw_uninpainted_exp_depth

    def _ensure_portrait_depth(is_expanded: bool) -> np.ndarray:
        nonlocal fg_expanded_plate_rgb, raw_expanded_portrait_depth, raw_inner_portrait_depth
        if is_expanded:
            if raw_expanded_portrait_depth is None:
                if fg_expanded_plate_rgb is None:
                    if not spec.fg_expanded_raw_relpath:
                        raise ValueError(
                            f"Scene '{spec.scene_id}' requires fg_expanded_raw_relpath for expanded portrait relief."
                        )
                    fg_exp_path = project_root / spec.fg_expanded_raw_relpath
                    if not fg_exp_path.is_file():
                        raise FileNotFoundError(
                            f"Expanded foreground source missing for portrait relief: {fg_exp_path}"
                        )
                    fg_exp_raw = Image.open(fg_exp_path).convert("RGB")
                    fg_expanded_plate_rgb, _ = register_expanded_foreground_plate(
                        inner_rgb=_ensure_orig_inner_rgb(),
                        raw_fg_expanded_rgb=fg_exp_raw,
                        dims=scene_dims,
                        outpaint_cfg=scene_outpaint_cfg,
                        min_registration_corr=spec.min_registration_corr,
                    )
                ew, eh = fg_expanded_plate_rgb.size
                print(
                    f"[{spec.scene_id}] Computing Depth Anything V2 volumetric portrait relief on "
                    f"registered {ew}x{eh} foreground plate..."
                )
                raw_expanded_portrait_depth = depth_estimator.predict_depth(fg_expanded_plate_rgb)
            return raw_expanded_portrait_depth
        if raw_inner_portrait_depth is None:
            inner = _ensure_orig_inner_rgb()
            print(
                f"[{spec.scene_id}] Computing Depth Anything V2 volumetric portrait relief on "
                f"{inner.size[0]}x{inner.size[1]} source plate..."
            )
            raw_inner_portrait_depth = depth_estimator.predict_depth(inner)
        return raw_inner_portrait_depth

    reports: dict[str, Any] = {}

    if spec.mid_contact_depth_solve is not None:
        if spec.room_depth_clean is None or spec.mid_depth_output_filename is None:
            raise ValueError(
                f"Scene '{spec.scene_id}' mid_contact_depth_solve requires room_depth_clean "
                f"and mid_depth_output_filename."
            )
        print(f"[{spec.scene_id}] Solving Law 1/2/4 contact-anchored depth for 'mid'...")
        reports["mid"] = _solve_and_save_contact_layer_depth(
            scene_id=spec.scene_id,
            layer_name="mid",
            cutout_rgba=_get_layer_rgba("mid"),
            room_depth_expanded_u8=room_depth_u8,
            raw_uninpainted_exp_depth=_ensure_uninpainted_exp_depth(),
            clean_cfg=spec.room_depth_clean,
            solve_cfg=spec.mid_contact_depth_solve,
            verify_cfg=verify_cfg,
            depth_out_path=output_dir / spec.mid_depth_output_filename,
        )

    if spec.fg_contact_depth_solve is not None:
        if spec.room_depth_clean is None or spec.fg_depth_output_filename is None:
            raise ValueError(
                f"Scene '{spec.scene_id}' fg_contact_depth_solve requires room_depth_clean "
                f"and fg_depth_output_filename."
            )
        print(f"[{spec.scene_id}] Solving Law 1/2/4 contact-anchored depth for 'fg'...")
        reports["fg"] = _solve_and_save_contact_layer_depth(
            scene_id=spec.scene_id,
            layer_name="fg",
            cutout_rgba=_get_layer_rgba("fg"),
            room_depth_expanded_u8=room_depth_u8,
            raw_uninpainted_exp_depth=_ensure_uninpainted_exp_depth(),
            clean_cfg=spec.room_depth_clean,
            solve_cfg=spec.fg_contact_depth_solve,
            verify_cfg=verify_cfg,
            depth_out_path=output_dir / spec.fg_depth_output_filename,
        )
    elif spec.fg_portrait_relief is not None:
        if spec.fg_depth_output_filename is None:
            raise ValueError(
                f"Scene '{spec.scene_id}' fg_portrait_relief requires fg_depth_output_filename."
            )
        print(f"[{spec.scene_id}] Solving volumetric portrait relief + outward EDT for 'fg'...")
        reports["fg"] = _solve_and_save_portrait_layer_depth(
            cutout_rgba=_get_layer_rgba("fg"),
            raw_neural_depth=_ensure_portrait_depth(spec.fg_is_expanded),
            pr_cfg=spec.fg_portrait_relief,
            verify_cfg=verify_cfg,
            depth_out_path=output_dir / spec.fg_depth_output_filename,
        )

    for el_spec in spec.extra_layers:
        if el_spec.contact_depth_solve is not None:
            if spec.room_depth_clean is None or el_spec.depth_output_filename is None:
                raise ValueError(
                    f"Scene '{spec.scene_id}' extra layer '{el_spec.layer_id}' contact_depth_solve "
                    f"requires room_depth_clean and depth_output_filename."
                )
            print(
                f"[{spec.scene_id}] Solving Law 1/2/4 contact-anchored depth for "
                f"extra layer '{el_spec.layer_id}'..."
            )
            reports[el_spec.layer_id] = _solve_and_save_contact_layer_depth(
                scene_id=spec.scene_id,
                layer_name=el_spec.layer_id,
                cutout_rgba=_get_layer_rgba(el_spec.layer_id),
                room_depth_expanded_u8=room_depth_u8,
                raw_uninpainted_exp_depth=_ensure_uninpainted_exp_depth(),
                clean_cfg=spec.room_depth_clean,
                solve_cfg=el_spec.contact_depth_solve,
                verify_cfg=verify_cfg,
                depth_out_path=output_dir / el_spec.depth_output_filename,
            )
        elif el_spec.portrait_relief is not None:
            if el_spec.depth_output_filename is None:
                raise ValueError(
                    f"Scene '{spec.scene_id}' extra layer '{el_spec.layer_id}' portrait_relief "
                    f"requires depth_output_filename."
                )
            print(
                f"[{spec.scene_id}] Solving volumetric portrait relief + outward EDT for "
                f"extra layer '{el_spec.layer_id}'..."
            )
            reports[el_spec.layer_id] = _solve_and_save_portrait_layer_depth(
                cutout_rgba=_get_layer_rgba(el_spec.layer_id),
                raw_neural_depth=_ensure_portrait_depth(el_spec.is_expanded),
                pr_cfg=el_spec.portrait_relief,
                verify_cfg=verify_cfg,
                depth_out_path=output_dir / el_spec.depth_output_filename,
            )

    return reports


def _compute_center_locked_corr(
    inner_rgb: Image.Image,
    expanded_bg_rgb: Image.Image,
    dims_wh: tuple[int, int],
) -> float:
    """Computes Pearson correlation between inner plate and center of expanded outpainted plate."""
    iw, ih = dims_wh
    ew, eh = expanded_bg_rgb.size
    px = (ew - iw) // 2
    py = (eh - ih) // 2
    a = np.asarray(inner_rgb.convert("L"), dtype=np.float32).ravel()
    b = np.asarray(
        expanded_bg_rgb.crop((px, py, px + iw, py + ih)).convert("L"), dtype=np.float32
    ).ravel()
    a_c = a - float(a.mean())
    b_c = b - float(b.mean())
    denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
    return float(np.dot(a_c, b_c) / denom) if denom > 1e-6 else 1.0


def _build_reused_scene_record(
    output_dir: Path,
    cfg: SpatialPipelineConfig,
    spec: SceneBuildSpec,
    existing_rec: dict[str, Any] | None,
) -> dict[str, Any]:
    """Builds an up-to-date SSOT manifest record for a scene whose verified assets already exist."""
    sdims = spec.scene_dimensions(cfg.dimensions)
    with Image.open(output_dir / spec.bg_output_filename) as bg_im:
        exp_w, exp_h = bg_im.size
        if existing_rec is not None and "outpaintRegistrationCorr" in existing_rec:
            reg_corr = float(existing_rec["outpaintRegistrationCorr"])
        else:
            lama_p = output_dir / spec.lama_inner_output_filename
            if lama_p.is_file():
                with Image.open(lama_p) as lama_im:
                    reg_corr = _compute_center_locked_corr(
                        lama_im, bg_im, (sdims.width, sdims.height)
                    )
            else:
                reg_corr = spec.min_registration_corr

    with Image.open(output_dir / spec.mid_output_filename) as mid_im:
        mid_w, mid_h = mid_im.size
        computed_mid_cov = float((np.asarray(mid_im.convert("RGBA"))[:, :, 3] > 16).mean())

    if (
        existing_rec is not None
        and "fgCoverageRatio" in existing_rec
        and "midCoverageRatio" in existing_rec
    ):
        fg_cov = float(existing_rec["fgCoverageRatio"])
        mid_cov = float(existing_rec["midCoverageRatio"])
    else:
        with Image.open(output_dir / spec.fg_output_filename) as fg_im:
            fg_cov = float((np.asarray(fg_im.convert("RGBA"))[:, :, 3] > 16).mean())
        mid_cov = computed_mid_cov

    extra_layer_records = [
        _build_extra_layer_manifest_record(
            el,
            canvas_wh=spec.expected_extra_layer_dimensions(el, cfg.dimensions, cfg.outpainting),
        )
        for el in spec.extra_layers
    ]
    refl_file = spec.reflection_output_filename
    cmr = spec.ceiling_mirror_reflection
    rec = _apply_optional_mid_fg_depth_manifest_fields(
        {
            "id": spec.scene_id,
            "title": spec.title_he,
            "shortLabelHe": spec.short_label_he,
            "icon": spec.icon,
            "segmentationBackend": spec.segmentation_backend,
            "depthBackend": spec.depth_backend,
            "innerWidth": sdims.width,
            "innerHeight": sdims.height,
            "midWidth": mid_w,
            "midHeight": mid_h,
            "bgExpandedWidth": exp_w,
            "bgExpandedHeight": exp_h,
            "bgExpansionFactor": cfg.outpainting.expansion_factor,
            "fgExpanded": spec.fg_is_expanded,
            "outpaintRegistrationCorr": round(reg_corr, 4),
            "roomDepthAmplitude": spec.room_depth_amplitude,
            "sceneBgZ": spec.scene_bg_z,
            "minInnerScale": spec.min_inner_scale,
            "fgZ": spec.fg_z,
            "midZ": spec.mid_z,
            "bgSegmentsX": spec.bg_segments_x,
            "bgSegmentsY": spec.bg_segments_y,
            "tunnelBoreTarget": (
                spec.tunnel_bore_target.to_dict() if spec.tunnel_bore_target is not None else None
            ),
            "initialCameraY": spec.initial_camera_y,
            "initialCameraZ": spec.initial_camera_z,
            "minCameraZ": spec.min_camera_z,
            "maxCameraZ": spec.max_camera_z,
            "maxYawDeg": spec.max_yaw_deg,
            "maxPitchDeg": spec.max_pitch_deg,
            "maxStrafeX": spec.max_strafe_x,
            "maxStrafeY": spec.max_strafe_y,
            "dollySpeedMult": spec.dolly_speed_mult,
            "fgRenderOrder": spec.fg_render_order,
            "midRenderOrder": spec.mid_render_order,
            "particleColorHex": spec.particle_color_hex,
            "particleCount": spec.particle_count,
            "bgTexture": f"assets/scenes/{spec.bg_output_filename}",
            "midTexture": f"assets/scenes/{spec.mid_output_filename}",
            "fgTexture": f"assets/scenes/{spec.fg_output_filename}",
            "depthTexture": f"assets/scenes/{spec.depth_output_filename}",
            "extraLayers": extra_layer_records,
            "reflectionTexture": f"assets/scenes/{refl_file}" if refl_file else None,
            "hasReflection": refl_file is not None,
            "reflectionBaseOpacity": cmr.base_opacity if cmr is not None else None,
            "reflectionExcursionBoost": cmr.excursion_boost if cmr is not None else None,
            "fgCoverageRatio": round(fg_cov, 4),
            "midCoverageRatio": round(mid_cov, 4),
        },
        spec,
    )
    if existing_rec is not None and "topologyVerification" in existing_rec:
        rec["topologyVerification"] = existing_rec["topologyVerification"]
    return rec


def build_single_scene(
    project_root: Path,
    output_dir: Path,
    cfg: SpatialPipelineConfig,
    spec: SceneBuildSpec,
    depth_estimator: DepthAnythingEstimator,
    matte_extractor: SalientMatteExtractor | None,
    sam_extractor: SamBoxMaskExtractor | None,
    lama_inpainter: LamaOcclusionInpainter | None,
    reuse_inner_layers: bool = False,
) -> dict[str, Any]:
    """Synthesizes all verified multi-plane, outpainted room, and depth textures for a single scene."""
    raw_spec = spec
    canvas_scale = raw_spec.canvas_scale_mult
    float_scale = raw_spec.floating_overlay_scale_mult
    scene_dims = raw_spec.scene_dimensions(cfg.dimensions)
    float_dims = raw_spec.floating_overlay_dimensions(cfg.dimensions)
    exp_dims = raw_spec.expanded_dimensions(cfg.dimensions, cfg.outpainting)
    scene_outpaint_cfg = _scale_outpainting_config(cfg.outpainting, canvas_scale)
    scene_inpaint_cfg = _scale_inpainting_config(
        cfg.inpainting, canvas_scale, raw_spec.inpainting_jpeg_quality_override
    )
    scene_fe_cfg = _override_floating_extraction_blur(
        cfg.floating_extraction, raw_spec.mirror_blur_override
    )
    spec = _scale_scene_spec_for_canvas(raw_spec, canvas_scale)

    src_path = project_root / spec.source_image_relpath
    if not src_path.is_file():
        raise FileNotFoundError(f"Required source photograph does not exist: {src_path}")

    outpaint_raw_path = project_root / spec.outpaint_raw_relpath
    if not outpaint_raw_path.is_file():
        raise FileNotFoundError(f"Required wide-angle outpaint source missing: {outpaint_raw_path}")
    raw_outpaint_rgb = Image.open(outpaint_raw_path).convert("RGB")

    fg_path = output_dir / spec.fg_output_filename
    mid_path = output_dir / spec.mid_output_filename
    lama_inner_path = output_dir / spec.lama_inner_output_filename
    built_extra_paths: set[Path] = set()
    exp_plate_rgb: Image.Image | None = None

    if reuse_inner_layers and fg_path.is_file() and mid_path.is_file() and lama_inner_path.is_file():
        print(f"[{spec.scene_id}] Reusing verified inner cutouts & LaMa plate ({lama_inner_path.name})")
        fg_img = Image.open(fg_path).convert("RGBA")
        mid_img = Image.open(mid_path).convert("RGBA")
        lama_inner_rgb = Image.open(lama_inner_path).convert("RGB")
    else:
        if lama_inpainter is None:
            raise RuntimeError("LaMa inpainter required when reuse_inner_layers is not satisfied.")

        target_size = (scene_dims.width, scene_dims.height)
        print(
            f"[{spec.scene_id}] Loading source image: {src_path.name} "
            f"(inner={scene_dims.width}x{scene_dims.height}, expanded={exp_dims.width}x{exp_dims.height})"
        )
        base_rgb = Image.open(src_path).convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
        rgb_np = np.asarray(base_rgb, dtype=np.uint8)
        inpaint_source_rgb = base_rgb
        dilation_override: int | None = None
        blur_override: float | None = None

        if spec.segmentation_backend == "isnet":
            if matte_extractor is None:
                raise RuntimeError("SalientMatteExtractor required for 'isnet' scene.")
            print(f"[{spec.scene_id}] Running Depth Anything V2 ({cfg.depth_model.model_id})...")
            scene_depth = depth_estimator.predict_depth(base_rgb)
            print(f"[{spec.scene_id}] Running IS-Net neural matting ({cfg.segmentation.rembg_model_name})...")
            alpha_u8 = matte_extractor.extract_alpha(base_rgb)
            print(f"[{spec.scene_id}] Partitioning Foreground (Z=0) & Midground (Z=-150) planes...")
            fg_img, mid_img, union_mask_u8 = partition_sculpture_layers(
                rgb_np=rgb_np,
                alpha_u8=alpha_u8,
                depth_norm=scene_depth,
                seg_cfg=cfg.segmentation,
                spec=spec,
            )
            if spec.sam_fg_boxes or spec.sam_mid_boxes:
                if sam_extractor is None:
                    raise RuntimeError("SamBoxMaskExtractor required for SAM supplementation.")
                print(
                    f"[{spec.scene_id}] Supplementing details via Meta SAM ({cfg.segmentation.sam_model_id})..."
                )
                fg_img, mid_img, union_mask_u8 = sam_extractor.supplement_rgba_layers(
                    base_rgb, fg_img, mid_img, union_mask_u8, spec
                )
        elif spec.segmentation_backend == "sam_boxes":
            if sam_extractor is None:
                raise RuntimeError("SamBoxMaskExtractor required for 'sam_boxes' scene.")
            print(
                f"[{spec.scene_id}] Running Meta SAM promptable neural segmentation "
                f"({cfg.segmentation.sam_model_id})..."
            )
            fg_img, mid_img, union_mask_u8 = sam_extractor.extract_fg_mid_layers(base_rgb, spec)
        elif spec.segmentation_backend == "sam_point_groups":
            if sam_extractor is None:
                raise RuntimeError("SamBoxMaskExtractor required for 'sam_point_groups' scene.")
            raw_fg_exp: Image.Image | None = None
            if spec.fg_is_expanded:
                if not spec.fg_expanded_raw_relpath:
                    raise ValueError(
                        f"Scene '{spec.scene_id}' sets fg_is_expanded=True without fg_expanded_raw_relpath."
                    )
                fg_exp_path = project_root / spec.fg_expanded_raw_relpath
                if not fg_exp_path.is_file():
                    raise FileNotFoundError(
                        f"Required expanded foreground source missing: {fg_exp_path}"
                    )
                raw_fg_exp = Image.open(fg_exp_path).convert("RGB")
            print(
                f"[{spec.scene_id}] Running Meta SAM + oriented IS-Net foreground segmentation "
                f"({cfg.segmentation.sam_model_id}, expanded={spec.fg_is_expanded})..."
            )
            fg_img, mid_img, union_mask_u8, exp_plate_rgb = (
                sam_extractor.extract_point_group_fg_and_inner_mask(
                    inner_rgb=base_rgb,
                    raw_fg_expanded_rgb=raw_fg_exp,
                    spec=spec,
                    dims=scene_dims,
                    outpaint_cfg=scene_outpaint_cfg,
                    matte_extractor=matte_extractor,
                )
            )
        elif spec.segmentation_backend == "graphite_paper":
            print(f"[{spec.scene_id}] Segmenting multi-plane graphite islands off archival paper...")
            fg_img, mid_img, union_mask_u8, inpaint_source_rgb = extract_graphite_drawing_layers(
                rgb_image=base_rgb,
                spec=spec,
                draw_cfg=cfg.graphite_drawing,
            )
            dilation_override = max(
                0, int(round(cfg.graphite_drawing.lama_dilation_radius_px * canvas_scale))
            )
            blur_override = cfg.graphite_drawing.lama_mask_blur_sigma * canvas_scale
        else:
            raise ValueError(f"Unsupported segmentation_backend: {spec.segmentation_backend}")

        if spec.sam_mid_point_groups or spec.sam_mid_exclude_boxes:
            merged_mid_alpha = np.asarray(mid_img.convert("RGBA"), dtype=np.uint8)[:, :, 3].copy()
            if spec.sam_mid_point_groups:
                if sam_extractor is None:
                    raise RuntimeError("SamBoxMaskExtractor required for sam_mid_point_groups.")
                print(
                    f"[{spec.scene_id}] Supplementing midground via "
                    f"{len(spec.sam_mid_point_groups)} SAM point group(s)..."
                )
                pg_mid_alpha = sam_extractor.segment_point_groups(
                    base_rgb,
                    spec.sam_mid_point_groups,
                    exclude_boxes=(),
                    max_pinhole_fill_px=spec.sam_mid_max_pinhole_fill_px,
                    min_major_component_px=spec.sam_mid_min_major_component_px,
                )
                merged_mid_alpha = np.maximum(merged_mid_alpha, pg_mid_alpha)
            if not spec.fg_is_expanded:
                fg_alpha_inner = np.asarray(fg_img.convert("RGBA"), dtype=np.uint8)[:, :, 3]
                merged_mid_alpha = np.where(fg_alpha_inner > 128, 0, merged_mid_alpha).astype(
                    np.uint8
                )
            for eb in spec.sam_mid_exclude_boxes:
                merged_mid_alpha[eb.y0 : eb.y1, eb.x0 : eb.x1] = 0
            mid_rgb_dec = decontaminate_matte_edge_rgb(rgb_np, merged_mid_alpha)
            mid_img = Image.fromarray(np.dstack([mid_rgb_dec, merged_mid_alpha]), mode="RGBA")
            if not spec.fg_is_expanded:
                fg_alpha_inner = np.asarray(fg_img.convert("RGBA"), dtype=np.uint8)[:, :, 3]
                union_mask_u8 = np.where(
                    (fg_alpha_inner > 8) | (merged_mid_alpha > 8), 255, 0
                ).astype(np.uint8)

        if spec.inpaint_from_outpaint_inner:
            pad_x0 = (exp_dims.width - scene_dims.width) // 2
            pad_y0 = (exp_dims.height - scene_dims.height) // 2
            inpaint_source_rgb = raw_outpaint_rgb.resize(
                (exp_dims.width, exp_dims.height), Image.Resampling.LANCZOS
            ).crop((pad_x0, pad_y0, pad_x0 + scene_dims.width, pad_y0 + scene_dims.height))
            union_mask_u8 = np.zeros((scene_dims.height, scene_dims.width), dtype=np.uint8)

        for el_spec in spec.extra_layers:
            el_out_path = output_dir / el_spec.output_filename
            if el_spec.sam_boxes:
                if sam_extractor is None:
                    raise RuntimeError(
                        f"Extra layer '{el_spec.layer_id}' requires SamBoxMaskExtractor for sam_boxes."
                    )
                print(
                    f"[{spec.scene_id}] Segmenting extra SAM box layer '{el_spec.layer_id}' "
                    f"(Z={el_spec.z_pos}, {len(el_spec.sam_boxes)} box(es))..."
                )
                if el_spec.is_expanded and exp_plate_rgb is not None:
                    plate_for_boxes = exp_plate_rgb
                elif spec.inpaint_from_outpaint_inner:
                    plate_for_boxes = inpaint_source_rgb
                else:
                    plate_for_boxes = base_rgb
                el_alpha_u8 = sam_extractor.segment_box_group(plate_for_boxes, el_spec.sam_boxes)
                el_rgb_u8 = decontaminate_matte_edge_rgb(
                    np.asarray(plate_for_boxes, dtype=np.uint8), el_alpha_u8
                )
                el_img = Image.fromarray(np.dstack([el_rgb_u8, el_alpha_u8]), mode="RGBA")
                if el_spec.overlays:
                    el_img = apply_layer_overlays(
                        base_layer_rgba=el_img,
                        overlays=el_spec.overlays,
                        project_root=project_root,
                        dims=scene_dims,
                        draw_cfg=cfg.graphite_drawing,
                        sam_extractor=sam_extractor,
                        matte_extractor=matte_extractor,
                        fe_cfg=scene_fe_cfg,
                    )
                el_img.save(el_out_path, format="PNG", optimize=True)
                built_extra_paths.add(el_out_path)
                print(f"  -> Saved {el_out_path.name} ({el_img.size[0]}x{el_img.size[1]})")
                if el_spec.pre_lama_dilation_px is not None:
                    pre_mask_u8 = np.where(el_alpha_u8 > 8, 255, 0).astype(np.uint8)
                    if el_spec.is_expanded and exp_plate_rgb is not None:
                        pw, ph = plate_for_boxes.size
                        px = (pw - scene_dims.width) // 2
                        py = (ph - scene_dims.height) // 2
                        pre_mask_u8 = pre_mask_u8[
                            py : py + scene_dims.height, px : px + scene_dims.width
                        ]
                    print(
                        f"[{spec.scene_id}] Pre-inpainting '{el_spec.layer_id}' bodies "
                        f"(r={el_spec.pre_lama_dilation_px}px) while preserving cast ground shadows..."
                    )
                    inpaint_source_rgb = lama_inpainter.inpaint_background(
                        inpaint_source_rgb,
                        pre_mask_u8,
                        dilation_radius_px=el_spec.pre_lama_dilation_px,
                        mask_blur_sigma=el_spec.pre_lama_blur_sigma,
                    )
                else:
                    el_inner_mask = np.where(el_alpha_u8 > 8, 255, 0).astype(np.uint8)
                    if el_spec.is_expanded and exp_plate_rgb is not None:
                        pw, ph = plate_for_boxes.size
                        px = (pw - scene_dims.width) // 2
                        py = (ph - scene_dims.height) // 2
                        el_inner_mask = el_inner_mask[
                            py : py + scene_dims.height, px : px + scene_dims.width
                        ]
                    union_mask_u8 = np.maximum(union_mask_u8, el_inner_mask)
            elif el_spec.sam_point_groups:
                if sam_extractor is None:
                    raise RuntimeError(
                        f"Extra layer '{el_spec.layer_id}' requires SamBoxMaskExtractor for sam_point_groups."
                    )
                if el_spec.is_expanded and exp_plate_rgb is None:
                    if not spec.fg_expanded_raw_relpath:
                        raise ValueError(
                            f"Extra layer '{el_spec.layer_id}' sets is_expanded=True without fg_expanded_raw_relpath."
                        )
                    fg_exp_raw = Image.open(project_root / spec.fg_expanded_raw_relpath).convert("RGB")
                    exp_plate_rgb, _ = register_expanded_foreground_plate(
                        inner_rgb=base_rgb,
                        raw_fg_expanded_rgb=fg_exp_raw,
                        dims=scene_dims,
                        outpaint_cfg=scene_outpaint_cfg,
                        min_registration_corr=spec.min_registration_corr,
                    )
                print(
                    f"[{spec.scene_id}] Segmenting extra 3D person layer '{el_spec.layer_id}' "
                    f"(Z={el_spec.z_pos}, expanded={el_spec.is_expanded})..."
                )
                person_plate = exp_plate_rgb if el_spec.is_expanded else base_rgb
                assert person_plate is not None
                el_img, el_inner_mask_u8 = sam_extractor.extract_extra_person_depth_layer(
                    plate_rgb=person_plate,
                    el_spec=el_spec,
                    dims=scene_dims,
                    support_dilate_px=spec.sam_core_support_dilate_px,
                    matte_extractor=matte_extractor,
                    lama_inpainter=lama_inpainter,
                )
                if el_spec.overlays:
                    el_img = apply_layer_overlays(
                        base_layer_rgba=el_img,
                        overlays=el_spec.overlays,
                        project_root=project_root,
                        dims=scene_dims,
                        draw_cfg=cfg.graphite_drawing,
                        sam_extractor=sam_extractor,
                        matte_extractor=matte_extractor,
                        fe_cfg=scene_fe_cfg,
                    )
                el_img.save(el_out_path, format="PNG", optimize=True)
                built_extra_paths.add(el_out_path)
                if not spec.inpaint_from_outpaint_inner:
                    union_mask_u8 = np.maximum(union_mask_u8, el_inner_mask_u8)
                print(f"  -> Saved {el_out_path.name} ({el_img.size[0]}x{el_img.size[1]})")

        if spec.rear_chamber_soften_box is not None:
            rc = spec.rear_chamber_soften_box
            cv2.ellipse(
                union_mask_u8,
                ((rc.x0 + rc.x1) // 2, (rc.y0 + rc.y1) // 2),
                (max(4, (rc.x1 - rc.x0) // 2), max(4, (rc.y1 - rc.y0) // 2)),
                0.0,
                0.0,
                360.0,
                255,
                -1,
            )

        if np.any(union_mask_u8 > 0):
            print(f"[{spec.scene_id}] Running neural LaMa FFC occlusion inpainting...")
            eff_radius = (
                dilation_override
                if dilation_override is not None
                else scene_inpaint_cfg.dilation_radius_px
            )
            eff_blur = (
                blur_override
                if blur_override is not None
                else scene_inpaint_cfg.mask_blur_sigma
            )
            if spec.inpaint_region_dilation_overrides:
                custom_mask_u8 = _build_region_aware_inpaint_mask(
                    binary_mask_u8=union_mask_u8,
                    base_dilation_px=eff_radius,
                    blur_sigma=eff_blur,
                    overrides=spec.inpaint_region_dilation_overrides,
                )
                lama_inner_rgb = lama_inpainter.inpaint_background(
                    inpaint_source_rgb,
                    custom_mask_u8,
                    dilation_radius_px=0,
                    mask_blur_sigma=0.0,
                )
            else:
                lama_inner_rgb = lama_inpainter.inpaint_background(
                    inpaint_source_rgb,
                    union_mask_u8,
                    dilation_radius_px=eff_radius,
                    mask_blur_sigma=eff_blur,
                )
        else:
            lama_inner_rgb = inpaint_source_rgb

        if spec.rear_chamber_soften_box is not None and spec.diffuse_aperture_fusion is not None:
            lama_inner_rgb = harmonize_v3_rear_aperture_luminance(
                lama_inner_rgb, spec.diffuse_aperture_fusion
            )
        lama_inner_rgb.save(
            lama_inner_path, format="JPEG", quality=scene_inpaint_cfg.jpeg_quality, optimize=True
        )
        print(f"  -> Saved {lama_inner_path.name} ({lama_inner_rgb.size[0]}x{lama_inner_rgb.size[1]})")

        if spec.fg_overlays:
            print(f"[{spec.scene_id}] Compositing {len(spec.fg_overlays)} FG overlay element(s)...")
            fg_img = apply_layer_overlays(
                base_layer_rgba=fg_img,
                overlays=spec.fg_overlays,
                project_root=project_root,
                dims=scene_dims,
                draw_cfg=cfg.graphite_drawing,
                sam_extractor=sam_extractor,
                matte_extractor=matte_extractor,
                fe_cfg=scene_fe_cfg,
            )
        if raw_spec.mid_overlays:
            if raw_spec.uses_floating_overlay_canvas_for_mid():
                print(
                    f"[{spec.scene_id}] Compositing {len(raw_spec.mid_overlays)} high-res MID overlay "
                    f"element(s) on {float_dims.width}x{float_dims.height}..."
                )
                mid_img = Image.new("RGBA", (float_dims.width, float_dims.height), (0, 0, 0, 0))
                mid_img = apply_layer_overlays(
                    base_layer_rgba=mid_img,
                    overlays=tuple(ov.scaled(float_scale) for ov in raw_spec.mid_overlays),
                    project_root=project_root,
                    dims=float_dims,
                    draw_cfg=cfg.graphite_drawing,
                    sam_extractor=sam_extractor,
                    matte_extractor=matte_extractor,
                    fe_cfg=scene_fe_cfg,
                )
            else:
                print(f"[{spec.scene_id}] Compositing {len(spec.mid_overlays)} MID overlay element(s)...")
                mid_img = apply_layer_overlays(
                    base_layer_rgba=mid_img,
                    overlays=spec.mid_overlays,
                    project_root=project_root,
                    dims=scene_dims,
                    draw_cfg=cfg.graphite_drawing,
                    sam_extractor=sam_extractor,
                    matte_extractor=matte_extractor,
                    fe_cfg=scene_fe_cfg,
                )

        fg_img.save(fg_path, format="PNG", optimize=True)
        mid_img.save(mid_path, format="PNG", optimize=True)

    fg_coverage = float((np.asarray(fg_img)[:, :, 3] > 16).mean())
    mid_coverage = float((np.asarray(mid_img)[:, :, 3] > 16).mean())

    print(
        f"[{spec.scene_id}] Registering & center-locking {scene_outpaint_cfg.expansion_factor:.2f}x "
        f"outpainted plate ({outpaint_raw_path.name})..."
    )
    pipe_or_expanded_bg_rgb, reg_corr = compose_outpainted_room(
        inner_rgb=lama_inner_rgb,
        raw_outpaint_rgb=raw_outpaint_rgb,
        dims=scene_dims,
        outpaint_cfg=scene_outpaint_cfg,
        min_registration_corr=spec.min_registration_corr,
        feather_collar_px=spec.feather_collar_px,
        registration_core_margin_xy=spec.registration_core_margin_xy,
        harmonize_seam_field=spec.harmonize_seam_field,
        seam_harmonic_downsample=spec.seam_harmonic_downsample,
        seam_harmonic_iters=spec.seam_harmonic_iters,
    )
    bg_path = output_dir / spec.bg_output_filename
    depth_path = output_dir / spec.depth_output_filename
    exp_w, exp_h = pipe_or_expanded_bg_rgb.size

    if spec.depth_backend == "pipe_tunnel_relief":
        if spec.pipe_tunnel_relief is None:
            raise ValueError(f"Scene '{spec.scene_id}' requires pipe_tunnel_relief configuration.")
        if sam_extractor is None or lama_inpainter is None:
            raise RuntimeError(
                f"Scene '{spec.scene_id}' requires Meta SAM and LaMa inpainter for pipe_tunnel_relief."
            )
        pipe_shell_specs = [el for el in spec.extra_layers if el.pipe_shell_mode]
        if len(pipe_shell_specs) != 1 or pipe_shell_specs[0].depth_output_filename is None:
            raise ValueError(
                f"Scene '{spec.scene_id}' with depth_backend='pipe_tunnel_relief' requires "
                f"exactly one pipe_shell_mode extra layer with depth_output_filename."
            )
        pipe_shell_spec = pipe_shell_specs[0]
        print(
            f"[{spec.scene_id}] Synthesizing true two-surface 3D concrete pipe shell + "
            f"clean LaMa gallery wall on {exp_w}x{exp_h}..."
        )
        expanded_bg_rgb, room_depth_img, pipe_shell_rgba, pipe_shell_depth_img = (
            synthesize_pipe_discontinuity_assets(
                pipe_plate_expanded_rgb=pipe_or_expanded_bg_rgb,
                dims=scene_dims,
                ptr_cfg=spec.pipe_tunnel_relief,
                depth_estimator=depth_estimator,
                sam_extractor=sam_extractor,
                lama_inpainter=lama_inpainter,
            )
        )
        expanded_bg_rgb.save(
            bg_path, format="JPEG", quality=scene_inpaint_cfg.jpeg_quality, optimize=True
        )
        print(f"  -> Saved {bg_path.name} ({exp_w}x{exp_h}, corr={reg_corr:.4f})")
        room_depth_img.save(depth_path, format="PNG", optimize=True)
        print(f"  -> Saved {depth_path.name} ({room_depth_img.size[0]}x{room_depth_img.size[1]})")

        pipe_shell_out_path = output_dir / pipe_shell_spec.output_filename
        pipe_shell_depth_path = output_dir / pipe_shell_spec.depth_output_filename
        pipe_shell_rgba.save(pipe_shell_out_path, format="PNG", optimize=True)
        pipe_shell_depth_img.save(pipe_shell_depth_path, format="PNG", optimize=True)
        built_extra_paths.add(pipe_shell_out_path)
        print(
            f"  -> Saved {pipe_shell_out_path.name} & {pipe_shell_depth_path.name} "
            f"({pipe_shell_rgba.size[0]}x{pipe_shell_rgba.size[1]})"
        )
    else:
        expanded_bg_rgb = pipe_or_expanded_bg_rgb
        expanded_bg_rgb.save(
            bg_path, format="JPEG", quality=scene_inpaint_cfg.jpeg_quality, optimize=True
        )
        print(f"  -> Saved {bg_path.name} ({exp_w}x{exp_h}, corr={reg_corr:.4f})")

        if spec.depth_backend == "depth_anything":
            print(f"[{spec.scene_id}] Computing Depth Anything V2 3D geometry depth on {exp_w}x{exp_h}...")
            room_depth_img = depth_estimator.build_smoothed_room_depth_uint8(expanded_bg_rgb)
        elif spec.depth_backend == "drawing_dome_relief":
            print(
                f"[{spec.scene_id}] Computing sculpted 3D dome relief + neural depth on {exp_w}x{exp_h}..."
            )
            room_depth_img = depth_estimator.build_drawing_dome_relief_depth_uint8(
                expanded_bg_rgb, cfg.graphite_drawing
            )
        else:
            raise ValueError(f"Unsupported depth_backend: {spec.depth_backend}")

        room_depth_img.save(depth_path, format="PNG", optimize=True)
        print(f"  -> Saved {depth_path.name} ({room_depth_img.size[0]}x{room_depth_img.size[1]})")

    pad_x = (exp_w - scene_dims.width) // 2
    pad_y = (exp_h - scene_dims.height) // 2
    inner_mirror_bg = expanded_bg_rgb.crop(
        (pad_x, pad_y, pad_x + scene_dims.width, pad_y + scene_dims.height)
    )

    diffuse_fusion_cache: dict[str, Image.Image] | None = None
    extra_layer_records: list[dict[str, Any]] = []
    for raw_el, el_spec in zip(raw_spec.extra_layers, spec.extra_layers):
        el_out_path = output_dir / el_spec.output_filename
        use_float_canvas = raw_spec.uses_floating_overlay_canvas_for_extra_layer(raw_el)
        if el_out_path in built_extra_paths or (reuse_inner_layers and el_out_path.is_file()):
            pass
        elif el_spec.pipe_shell_mode:
            raise RuntimeError(
                f"pipe_shell_mode extra layer '{el_spec.layer_id}' was not synthesized by "
                f"pipe_tunnel_relief backend in scene '{spec.scene_id}'."
            )
        elif el_spec.diffuse_aperture_role is not None:
            if spec.diffuse_aperture_fusion is None:
                raise ValueError(
                    f"Extra layer '{el_spec.layer_id}' sets diffuse_aperture_role without "
                    f"spec.diffuse_aperture_fusion."
                )
            if diffuse_fusion_cache is None:
                if sam_extractor is None or matte_extractor is None or lama_inpainter is None:
                    raise RuntimeError(
                        f"Scene '{spec.scene_id}' requires SAM, IS-Net, and LaMa for diffuse_aperture_fusion."
                    )
                print(
                    f"[{spec.scene_id}] Synthesizing 3D diffuse bluish-white aperture fusion layers "
                    f"from {spec.diffuse_aperture_fusion.source_image_relpath}..."
                )
                diffuse_fusion_cache = build_diffuse_aperture_fusion_layers(
                    fusion_cfg=spec.diffuse_aperture_fusion,
                    project_root=project_root,
                    dims=scene_dims,
                    sam_extractor=sam_extractor,
                    matte_extractor=matte_extractor,
                    lama_inpainter=lama_inpainter,
                    depth_estimator=depth_estimator,
                )
            fused_layer_img = diffuse_fusion_cache[el_spec.diffuse_aperture_role]
            fused_layer_img.save(el_out_path, format="PNG", optimize=True)
            print(f"  -> Saved {el_out_path.name} ({fused_layer_img.size[0]}x{fused_layer_img.size[1]})")
        elif el_spec.ceiling_mirror_reflection is not None:
            if use_float_canvas:
                assert raw_el.ceiling_mirror_reflection is not None
                cmr_el = raw_el.ceiling_mirror_reflection.scaled(float_scale)
                el_w, el_h = float_dims.width, float_dims.height
                cmr_dims = float_dims
            else:
                cmr_el = el_spec.ceiling_mirror_reflection
                el_w, el_h = (
                    (exp_w, exp_h) if el_spec.is_expanded else (scene_dims.width, scene_dims.height)
                )
                cmr_dims = scene_dims
            print(
                f"[{spec.scene_id}] Building 3D mirror reflection layer '{el_spec.layer_id}' "
                f"(Z={el_spec.z_pos}, size={el_w}x{el_h}, {len(cmr_el.items)} reflected item(s))..."
            )
            el_refl_img = build_ceiling_mirror_reflection_layer(
                canvas_width=el_w,
                canvas_height=el_h,
                mirror_bg_inner_rgb=inner_mirror_bg,
                cmr_cfg=cmr_el,
                project_root=project_root,
                dims=cmr_dims,
                draw_cfg=cfg.graphite_drawing,
                fe_cfg=scene_fe_cfg,
                sam_extractor=sam_extractor,
                matte_extractor=matte_extractor,
                outpaint_expansion_factor=scene_outpaint_cfg.expansion_factor,
            )
            el_refl_img.save(el_out_path, format="PNG", optimize=True)
            print(f"  -> Saved {el_out_path.name} ({el_w}x{el_h})")
        else:
            if use_float_canvas:
                el_canvas_wh = (float_dims.width, float_dims.height)
                ov_dims = float_dims
                ov_specs = tuple(ov.scaled(float_scale) for ov in raw_el.overlays)
            else:
                el_canvas_wh = (
                    (exp_w, exp_h) if el_spec.is_expanded else (scene_dims.width, scene_dims.height)
                )
                ov_dims = scene_dims
                ov_specs = el_spec.overlays
            print(
                f"[{spec.scene_id}] Building extra depth layer '{el_spec.layer_id}' "
                f"(Z={el_spec.z_pos}, size={el_canvas_wh[0]}x{el_canvas_wh[1]}, {len(ov_specs)} element(s))..."
            )
            el_base = Image.new("RGBA", el_canvas_wh, (0, 0, 0, 0))
            el_img = apply_layer_overlays(
                base_layer_rgba=el_base,
                overlays=ov_specs,
                project_root=project_root,
                dims=ov_dims,
                draw_cfg=cfg.graphite_drawing,
                sam_extractor=sam_extractor,
                matte_extractor=matte_extractor,
                fe_cfg=scene_fe_cfg,
            )
            el_img.save(el_out_path, format="PNG", optimize=True)
            print(f"  -> Saved {el_out_path.name} ({el_canvas_wh[0]}x{el_canvas_wh[1]})")

        expected_el_wh = raw_spec.expected_extra_layer_dimensions(
            raw_el, cfg.dimensions, cfg.outpainting
        )
        extra_layer_records.append(
            _build_extra_layer_manifest_record(raw_el, canvas_wh=expected_el_wh)
        )

    topology_reports = _synthesize_scene_topology_depths(
        project_root=project_root,
        output_dir=output_dir,
        scene_dims=scene_dims,
        scene_outpaint_cfg=scene_outpaint_cfg,
        spec=spec,
        depth_estimator=depth_estimator,
        src_path=src_path,
        expanded_bg_rgb=expanded_bg_rgb,
        room_depth_img=room_depth_img,
        depth_path=depth_path,
        fg_img=fg_img,
        mid_img=mid_img,
        fg_expanded_plate_rgb=exp_plate_rgb,
        raw_outpaint_rgb=raw_outpaint_rgb,
    )

    reflection_filename: str | None = None
    refl_base_opacity: float | None = None
    refl_excursion_boost: float | None = None
    if spec.reflection_output_filename and spec.ceiling_mirror_reflection is not None:
        cmr = spec.ceiling_mirror_reflection
        ref_layer_img = build_ceiling_mirror_reflection_layer(
            canvas_width=scene_dims.width,
            canvas_height=scene_dims.height,
            mirror_bg_inner_rgb=inner_mirror_bg,
            cmr_cfg=cmr,
            project_root=project_root,
            dims=scene_dims,
            draw_cfg=cfg.graphite_drawing,
            fe_cfg=scene_fe_cfg,
            sam_extractor=sam_extractor,
            matte_extractor=matte_extractor,
            outpaint_expansion_factor=scene_outpaint_cfg.expansion_factor,
        )
        ref_out_path = output_dir / spec.reflection_output_filename
        ref_layer_img.save(ref_out_path, format="PNG", optimize=True)
        reflection_filename = spec.reflection_output_filename
        refl_base_opacity = cmr.base_opacity
        refl_excursion_boost = cmr.excursion_boost
        print(
            f"  -> Saved {ref_out_path.name} "
            f"({len(cmr.items)} perspective-convergent mirror reflections)"
        )
    elif spec.reflection_output_filename and spec.reflection_source_relpath:
        ref_src_path = project_root / spec.reflection_source_relpath
        if not ref_src_path.is_file():
            raise FileNotFoundError(f"Required reflection source image missing: {ref_src_path}")
        ref_src_img = Image.open(ref_src_path)
        ref_layer_img = build_aperture_reflection_layer(
            canvas_width=scene_dims.width,
            canvas_height=scene_dims.height,
            reflection_src=ref_src_img,
            apertures=spec.reflection_apertures,
        )
        ref_out_path = output_dir / spec.reflection_output_filename
        ref_layer_img.save(ref_out_path, format="PNG", optimize=True)
        reflection_filename = spec.reflection_output_filename
        print(f"  -> Saved {ref_out_path.name} ({len(spec.reflection_apertures)} bounded apertures)")

    rec = _apply_optional_mid_fg_depth_manifest_fields(
        {
            "id": raw_spec.scene_id,
            "title": raw_spec.title_he,
            "shortLabelHe": raw_spec.short_label_he,
            "icon": raw_spec.icon,
            "segmentationBackend": raw_spec.segmentation_backend,
            "depthBackend": raw_spec.depth_backend,
            "innerWidth": scene_dims.width,
            "innerHeight": scene_dims.height,
            "midWidth": mid_img.width,
            "midHeight": mid_img.height,
            "bgExpandedWidth": exp_w,
            "bgExpandedHeight": exp_h,
            "bgExpansionFactor": scene_outpaint_cfg.expansion_factor,
            "fgExpanded": raw_spec.fg_is_expanded,
            "outpaintRegistrationCorr": round(reg_corr, 4),
            "roomDepthAmplitude": raw_spec.room_depth_amplitude,
            "sceneBgZ": raw_spec.scene_bg_z,
            "minInnerScale": raw_spec.min_inner_scale,
            "fgZ": raw_spec.fg_z,
            "midZ": raw_spec.mid_z,
            "bgSegmentsX": raw_spec.bg_segments_x,
            "bgSegmentsY": raw_spec.bg_segments_y,
            "tunnelBoreTarget": (
                raw_spec.tunnel_bore_target.to_dict()
                if raw_spec.tunnel_bore_target is not None
                else None
            ),
            "initialCameraY": raw_spec.initial_camera_y,
            "initialCameraZ": raw_spec.initial_camera_z,
            "minCameraZ": raw_spec.min_camera_z,
            "maxCameraZ": raw_spec.max_camera_z,
            "maxYawDeg": raw_spec.max_yaw_deg,
            "maxPitchDeg": raw_spec.max_pitch_deg,
            "maxStrafeX": raw_spec.max_strafe_x,
            "maxStrafeY": raw_spec.max_strafe_y,
            "dollySpeedMult": raw_spec.dolly_speed_mult,
            "fgRenderOrder": raw_spec.fg_render_order,
            "midRenderOrder": raw_spec.mid_render_order,
            "particleColorHex": raw_spec.particle_color_hex,
            "particleCount": raw_spec.particle_count,
            "bgTexture": f"assets/scenes/{raw_spec.bg_output_filename}",
            "midTexture": f"assets/scenes/{raw_spec.mid_output_filename}",
            "fgTexture": f"assets/scenes/{raw_spec.fg_output_filename}",
            "depthTexture": f"assets/scenes/{raw_spec.depth_output_filename}",
            "extraLayers": extra_layer_records,
            "reflectionTexture": f"assets/scenes/{reflection_filename}" if reflection_filename else None,
            "hasReflection": reflection_filename is not None,
            "reflectionBaseOpacity": refl_base_opacity,
            "reflectionExcursionBoost": refl_excursion_boost,
            "fgCoverageRatio": round(fg_coverage, 4),
            "midCoverageRatio": round(mid_coverage, 4),
        },
        raw_spec,
    )
    if topology_reports:
        rec["topologyVerification"] = topology_reports
    return rec


def run_pipeline(
    cfg: SpatialPipelineConfig,
    project_root: Path = PROJECT_ROOT,
    reuse_inner_layers: bool = False,
    rebuild_inner_scenes: frozenset[str] = frozenset(),
    reuse_full_scenes: bool = False,
    only_scenes: frozenset[str] | None = None,
    skip_manifest: bool = False,
    manifest_out_path: Path | None = None,
) -> Path:
    output_dir = project_root / cfg.output_dir_relpath
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / cfg.manifest_filename

    known_scene_ids = {spec.scene_id for spec in cfg.scenes}
    if only_scenes is not None:
        unknown_ids = only_scenes - known_scene_ids
        if unknown_ids:
            raise ValueError(f"Unknown scene IDs passed to --scenes: {sorted(unknown_ids)}")

    existing_records: dict[str, Any] = {}
    if not skip_manifest and manifest_path.is_file():
        prev_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_records = dict(prev_manifest.get("scenes", {}))

    scenes_to_build: list[SceneBuildSpec] = []
    for spec in cfg.scenes:
        if only_scenes is not None and spec.scene_id not in only_scenes:
            if not skip_manifest and not _all_scene_assets_exist(output_dir, spec):
                raise FileNotFoundError(
                    f"Cannot reuse non-selected scene '{spec.scene_id}' because required "
                    f"textures are missing in {output_dir}."
                )
            continue
        if (
            reuse_full_scenes
            and spec.scene_id not in rebuild_inner_scenes
            and _all_scene_assets_exist(output_dir, spec)
        ):
            continue
        scenes_to_build.append(spec)

    need_inner_models = False
    need_isnet = False
    need_sam = False
    for spec in scenes_to_build:
        scene_reuse = reuse_inner_layers and (spec.scene_id not in rebuild_inner_scenes)
        has_inner = (
            (output_dir / spec.fg_output_filename).is_file()
            and (output_dir / spec.mid_output_filename).is_file()
            and (output_dir / spec.lama_inner_output_filename).is_file()
        )
        if spec.depth_backend == "pipe_tunnel_relief" or any(
            el.pipe_shell_mode for el in spec.extra_layers
        ):
            need_inner_models = True
            need_sam = True
        cmr_configs = [
            *(
                (spec.ceiling_mirror_reflection,)
                if spec.ceiling_mirror_reflection is not None
                else ()
            ),
            *(
                el.ceiling_mirror_reflection
                for el in spec.extra_layers
                if el.ceiling_mirror_reflection is not None
            ),
        ]
        for cmr in cmr_configs:
            for it in cmr.items:
                if it.overlay.extraction_mode == "sam_crop":
                    need_sam = True
                elif it.overlay.extraction_mode == "isnet_dehalo_capsule":
                    need_isnet = True
        for el in spec.extra_layers:
            if el.diffuse_aperture_role is not None and not (
                scene_reuse and (output_dir / el.output_filename).is_file()
            ):
                need_inner_models = True
                need_sam = True
                need_isnet = True
        if not (scene_reuse and has_inner):
            need_inner_models = True
            if spec.segmentation_backend == "isnet":
                need_isnet = True
                if spec.sam_fg_boxes or spec.sam_mid_boxes or spec.sam_mid_point_groups:
                    need_sam = True
            elif spec.segmentation_backend in ("sam_boxes", "sam_point_groups"):
                need_sam = True
            elif spec.sam_mid_point_groups:
                need_sam = True
            if spec.fg_oriented_matte_boxes:
                need_isnet = True
            for el in spec.extra_layers:
                if el.sam_boxes or el.sam_point_groups:
                    need_sam = True
                if el.oriented_matte_boxes:
                    need_isnet = True
            all_overlays = [
                *spec.fg_overlays,
                *spec.mid_overlays,
                *(ov for el in spec.extra_layers for ov in el.overlays),
            ]
            if any(o.extraction_mode == "sam_crop" for o in all_overlays):
                need_sam = True
            if any(o.extraction_mode == "isnet_dehalo_capsule" for o in all_overlays):
                need_isnet = True

    depth_estimator = DepthAnythingEstimator(cfg.depth_model) if scenes_to_build else None
    matte_extractor = SalientMatteExtractor(cfg.segmentation) if need_isnet else None
    sam_extractor = SamBoxMaskExtractor(cfg.segmentation) if need_sam else None
    lama_inpainter = LamaOcclusionInpainter(cfg.inpainting) if need_inner_models else None

    scene_records: dict[str, Any] = {}
    for spec in cfg.scenes:
        if spec not in scenes_to_build:
            if skip_manifest and only_scenes is not None and spec.scene_id not in only_scenes:
                continue
            scene_records[spec.scene_id] = _build_reused_scene_record(
                output_dir=output_dir,
                cfg=cfg,
                spec=spec,
                existing_rec=existing_records.get(spec.scene_id),
            )
            print(f"[{spec.scene_id}] Reusing complete verified scene assets.")
            continue

        assert depth_estimator is not None
        scene_reuse = reuse_inner_layers and (spec.scene_id not in rebuild_inner_scenes)
        scene_records[spec.scene_id] = build_single_scene(
            project_root=project_root,
            output_dir=output_dir,
            cfg=cfg,
            spec=spec,
            depth_estimator=depth_estimator,
            matte_extractor=matte_extractor,
            sam_extractor=sam_extractor,
            lama_inpainter=lama_inpainter,
            reuse_inner_layers=scene_reuse,
        )

    if skip_manifest:
        if manifest_out_path is not None:
            manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
            isolated_manifest = {
                "pipeline_config": cfg.to_dict(),
                "scenes": scene_records,
            }
            manifest_out_path.write_text(
                json.dumps(isolated_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Saved isolated scene manifest: {manifest_out_path}")
            return manifest_out_path
        print("Skipped writing runtime scene_manifest.json (--skip-manifest).")
        return manifest_path

    # Merge companion multi-scene builders (`new_scenes_builder.py` & `fried_house_scene_builder.py`)
    # into the unified SSOT runtime manifest when their verified textures are present on disk.
    from prototype.fried_house_scene_builder import default_fried_house_scene_config
    from prototype.new_scenes_builder import default_new_scenes_bundle_config

    new_bundle = default_new_scenes_bundle_config()
    if Path(new_bundle.spec_output_abspath).is_file():
        nb_payload = json.loads(Path(new_bundle.spec_output_abspath).read_text(encoding="utf-8"))
        for sid, srec in nb_payload.get("scenes", {}).items():
            rec_copy = dict(srec)
            rec_copy.setdefault("bgExpandedWidth", rec_copy.get("expandedWidth", 2700))
            rec_copy.setdefault("bgExpandedHeight", rec_copy.get("expandedHeight", 1800))
            scene_records[sid] = rec_copy

    fried_cfg = default_fried_house_scene_config()
    fried_bg = output_dir / fried_cfg.bg_output_filename
    if fried_bg.is_file():
        exp_w, exp_h = fried_cfg.outpainting.expanded_dimensions(fried_cfg.dimensions)
        scene_records[fried_cfg.scene_id] = {
            "id": fried_cfg.scene_id,
            "title": fried_cfg.title_he,
            "shortLabelHe": fried_cfg.short_label_he,
            "icon": fried_cfg.icon,
            "exhibitionGroup": fried_cfg.exhibition_group,
            "sourceCatalogIds": [fried_cfg.herz_catalog_id, fried_cfg.cushion_stone_fg.catalog_id],
            "segmentationBackend": "fried_house_hybrid_sam_isnet",
            "depthBackend": "fried_house_calibrated_da2",
            "innerWidth": fried_cfg.dimensions.width,
            "innerHeight": fried_cfg.dimensions.height,
            "bgExpandedWidth": exp_w,
            "bgExpandedHeight": exp_h,
            "bgExpansionFactor": fried_cfg.outpainting.expansion_factor,
            "fgExpanded": False,
            "boreSightAtRest": fried_cfg.bore_sight_at_rest,
            "roomDepthAmplitude": fried_cfg.depth_calibration.room_depth_amplitude,
            "sceneBgZ": fried_cfg.scene_bg_z,
            "midZ": fried_cfg.mid_z,
            "fgZ": fried_cfg.fg_z,
            "bgSegmentsX": fried_cfg.bg_segments_x,
            "bgSegmentsY": fried_cfg.bg_segments_y,
            "initialCameraY": fried_cfg.initial_camera_y,
            "initialCameraZ": fried_cfg.initial_camera_z,
            "minCameraZ": fried_cfg.min_camera_z,
            "maxCameraZ": fried_cfg.max_camera_z,
            "maxYawDeg": fried_cfg.max_yaw_deg,
            "maxPitchDeg": fried_cfg.max_pitch_deg,
            "maxStrafeX": fried_cfg.max_strafe_x,
            "maxStrafeY": fried_cfg.max_strafe_y,
            "dollySpeedMult": fried_cfg.dolly_speed_mult,
            "midRenderOrder": fried_cfg.mid_render_order,
            "fgRenderOrder": fried_cfg.fg_render_order,
            "particleColorHex": fried_cfg.particle_color_hex,
            "particleCount": fried_cfg.particle_count,
            "bgTexture": f"assets/scenes/{fried_cfg.bg_output_filename}",
            "midTexture": f"assets/scenes/{fried_cfg.mid_output_filename}",
            "fgTexture": f"assets/scenes/{fried_cfg.fg_output_filename}",
            "depthTexture": f"assets/scenes/{fried_cfg.depth_output_filename}",
            "extraLayers": [
                {
                    "id": fried_cfg.handrail.layer_id,
                    "labelHe": fried_cfg.handrail.label_he,
                    "texture": f"assets/scenes/{fried_cfg.railing_output_filename}",
                    "zPos": fried_cfg.railing_z,
                    "renderOrder": fried_cfg.railing_render_order,
                    "expanded": False,
                    "baseOpacity": 1.0,
                    "excursionBoost": 0.0,
                    "depthTest": False,
                }
            ],
            "reflectionTexture": None,
            "hasReflection": False,
        }

    from prototype.railing_occluder_scene_builder import (
        build_railing_occluder_manifest_entry,
        default_railing_occluder_scene_config,
    )

    rail_occ_cfg = default_railing_occluder_scene_config()
    if (output_dir / rail_occ_cfg.bg_output_filename).is_file():
        scene_records[rail_occ_cfg.scene_id] = build_railing_occluder_manifest_entry(rail_occ_cfg)

    from prototype.denya_living_room_scene_builder import default_denya_living_room_bundle_config

    denya_living_cfg = default_denya_living_room_bundle_config()
    denya_living_spec_path = project_root / denya_living_cfg.spec_output_relpath
    if denya_living_spec_path.is_file():
        dl_payload = json.loads(denya_living_spec_path.read_text(encoding="utf-8"))
        for sid, srec in dl_payload.get("scenes", {}).items():
            scene_records[sid] = dict(srec)

    from prototype.table_installation_scene_builder import (
        build_fried_table_hybrid_manifest_entry,
        default_fried_table_hybrid_scene_config,
    )

    table_hybrid_cfg = default_fried_table_hybrid_scene_config()
    if (output_dir / table_hybrid_cfg.bg_output_filename).is_file():
        scene_records[table_hybrid_cfg.scene_id] = build_fried_table_hybrid_manifest_entry(
            table_hybrid_cfg
        )

    merged_scene_records = {**existing_records, **scene_records}
    manifest = {
        "pipeline_config": cfg.to_dict(),
        "scenes": merged_scene_records,
    }
    target_manifest_path = manifest_out_path if manifest_out_path is not None else manifest_path
    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    target_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved SSOT scene manifest: {target_manifest_path}")
    return target_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize neural 2.5D spatial scene layers")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a JSON SpatialPipelineConfig file",
    )
    parser.add_argument(
        "--reuse-inner-layers",
        action="store_true",
        help="Reuse existing verified inner sculpture PNGs and LaMa plates to rebuild outpaint + depth",
    )
    parser.add_argument(
        "--reuse-full-scenes",
        action="store_true",
        help="Skip scenes whose full textures already exist unless listed in --rebuild-inner-scene",
    )
    parser.add_argument(
        "--rebuild-inner-scene",
        action="append",
        default=[],
        help="Scene ID whose inner segmentation and LaMa plate must be rebuilt",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional subset of scene IDs to rebuild while reusing verified assets for others",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Do not read or overwrite prototype/assets/scenes/scene_manifest.json",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional output path to write the generated manifest JSON instead of scene_manifest.json",
    )
    args = parser.parse_args()

    cfg = (
        SpatialPipelineConfig.load_json(args.config)
        if args.config is not None
        else default_spatial_pipeline_config()
    )
    run_pipeline(
        cfg,
        reuse_inner_layers=args.reuse_inner_layers,
        rebuild_inner_scenes=frozenset(args.rebuild_inner_scene),
        reuse_full_scenes=args.reuse_full_scenes,
        only_scenes=frozenset(args.scenes) if args.scenes is not None else None,
        skip_manifest=args.skip_manifest,
        manifest_out_path=args.manifest_out,
    )


if __name__ == "__main__":
    main()
