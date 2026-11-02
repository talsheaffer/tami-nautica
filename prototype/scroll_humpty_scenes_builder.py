#!/usr/bin/env python3
"""
prototype/scroll_humpty_scenes_builder.py
Offline neural 2.5D spatial asset builder and typed SSOT configuration models for Workstream E:
  1. Surgical neural LaMa bottom-right Gemini sparkle watermark removal into `prototype/assets/sources/`:
     - `scroll_palm_world_clean.png`
     - `scroll_palm_world_wide_3x2_clean.jpg`
     - `humpty_dumpty_bundle_clean.png`
  2. Scene 14 (`scroll_palm_world` / `scene14_scroll_world_*`):
     Coastal clearing world inside Tami's scroll (*מגילה וכף רגל*, `IMG_MUSE_0256`) with
     $C^0$ sand-contact-anchored 3D palm/agave/rod/philodendron depth and $C^{-1}$ EDT silhouettes.
  3. Scene 15 (`humpty_dumpty_bundle` / `scene15_humpty_bundle_*`):
     Tami's white cloth-and-cord wrapped Humpty Dumpty bundle sculpture on a seamlessly extended
     warm taupe gallery surface with $C^0$ tabletop contact continuity and sculptural 3D relief.
  4. Scene 7 (`wandering_drawing_1` / `scene7_wandering1_*`) Seamless Portal Upgrade:
     Forward 3D sculpted ceramic foot & wrapped pole `extraLayer` (`scroll_foot`) + dissolvable
     graphite aperture `scroll_veil` opening directly into `scroll_palm_world`.

Strict No-Fallback Guarantee: executes real neural models (`Depth Anything V2`, `Meta SAM`,
`IS-Net` via `rembg`, and `LaMa` via `simple-lama-inpainting`) inside `.venv`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import cv2
    import numpy as np
    import scipy.ndimage as ndi
    from PIL import Image
    from rembg import remove
except ImportError as exc:
    raise RuntimeError(
        "Missing neural vision dependencies inside virtual environment. Run with: "
        "`/Users/talsheaffer/code/tami/.venv/bin/python -u prototype/scroll_humpty_scenes_builder.py`."
    ) from exc

from prototype.depth_topology import (
    ContactAnchorBox,
    ContactAnchoredLayerDepthConfig,
    LayerTopologyVerificationConfig,
    LayerTopologyVerificationReport,
    RoomDepthCleanConfig,
    _solve_harmonic_field_on_domain,
    clean_room_depth_under_cutouts,
    compute_room_relative_neural_protrusion,
    compute_world_z_map,
    extract_contact_anchor_mask,
    extract_layer_window_from_expanded,
    solve_contact_anchored_layer_depth,
    verify_layer_contact_and_discontinuity,
)
from prototype.new_scenes_builder import (
    _assert_zero_border_alpha,
    _cosine_feather_2d,
    _inpaint_masked_plate_lama,
    _prune_tiny_alpha_specks_rgba,
    build_wandering_drawing_1_scroll_portal_layers,
    default_new_scenes_bundle_config,
)
from prototype.scene_config import (
    BoundingBoxPrompt,
    CanvasDimensionsConfig,
    DepthModelConfig,
    InpaintingConfig,
    SamPointGroupPrompt,
    SegmentationConfig,
)
from prototype.vision_primitives import (
    DepthAnythingEstimator,
    LamaOcclusionInpainter,
    SalientMatteExtractor,
    SamBoxMaskExtractor,
    decontaminate_matte_edge_rgb,
    select_torch_device,
)


# ============================================================================
# Hierarchical, Serializable Configuration Data-Models (SSOT)
# ============================================================================


@dataclass(frozen=True)
class WatermarkCleanTargetSpec:
    name: str
    source_abspath: str
    output_relpath: str
    output_format: str
    jpeg_quality: int
    corner_crop_px: int
    star_box_from_br: tuple[int, int, int, int]
    astroid_exponent: float
    inpaint_dilation_px: int
    inpaint_blur_sigma: float
    blend_dilation_px: int
    alpha_zero_floor: float
    min_identical_pixel_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_abspath": self.source_abspath,
            "output_relpath": self.output_relpath,
            "output_format": self.output_format,
            "jpeg_quality": self.jpeg_quality,
            "corner_crop_px": self.corner_crop_px,
            "star_box_from_br": list(self.star_box_from_br),
            "astroid_exponent": self.astroid_exponent,
            "inpaint_dilation_px": self.inpaint_dilation_px,
            "inpaint_blur_sigma": self.inpaint_blur_sigma,
            "blend_dilation_px": self.blend_dilation_px,
            "alpha_zero_floor": self.alpha_zero_floor,
            "min_identical_pixel_ratio": self.min_identical_pixel_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WatermarkCleanTargetSpec:
        sb = data["star_box_from_br"]
        return cls(
            name=str(data["name"]),
            source_abspath=str(data["source_abspath"]),
            output_relpath=str(data["output_relpath"]),
            output_format=str(data["output_format"]),
            jpeg_quality=int(data["jpeg_quality"]),
            corner_crop_px=int(data["corner_crop_px"]),
            star_box_from_br=(int(sb[0]), int(sb[1]), int(sb[2]), int(sb[3])),
            astroid_exponent=float(data["astroid_exponent"]),
            inpaint_dilation_px=int(data["inpaint_dilation_px"]),
            inpaint_blur_sigma=float(data["inpaint_blur_sigma"]),
            blend_dilation_px=int(data["blend_dilation_px"]),
            alpha_zero_floor=float(data["alpha_zero_floor"]),
            min_identical_pixel_ratio=float(data["min_identical_pixel_ratio"]),
        )


@dataclass(frozen=True)
class SceneCameraRigSpec:
    min_inner_scale: float
    initial_camera_y: float
    initial_camera_z: float
    min_camera_z: float
    max_camera_z: float
    max_yaw_deg: float
    max_pitch_deg: float
    max_strafe_x: float
    max_strafe_y: float
    dolly_speed_mult: float
    particle_color_hex: str
    particle_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_inner_scale": self.min_inner_scale,
            "initial_camera_y": self.initial_camera_y,
            "initial_camera_z": self.initial_camera_z,
            "min_camera_z": self.min_camera_z,
            "max_camera_z": self.max_camera_z,
            "max_yaw_deg": self.max_yaw_deg,
            "max_pitch_deg": self.max_pitch_deg,
            "max_strafe_x": self.max_strafe_x,
            "max_strafe_y": self.max_strafe_y,
            "dolly_speed_mult": self.dolly_speed_mult,
            "particle_color_hex": self.particle_color_hex,
            "particle_count": self.particle_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneCameraRigSpec:
        return cls(
            min_inner_scale=float(data["min_inner_scale"]),
            initial_camera_y=float(data["initial_camera_y"]),
            initial_camera_z=float(data["initial_camera_z"]),
            min_camera_z=float(data["min_camera_z"]),
            max_camera_z=float(data["max_camera_z"]),
            max_yaw_deg=float(data["max_yaw_deg"]),
            max_pitch_deg=float(data["max_pitch_deg"]),
            max_strafe_x=float(data["max_strafe_x"]),
            max_strafe_y=float(data["max_strafe_y"]),
            dolly_speed_mult=float(data["dolly_speed_mult"]),
            particle_color_hex=str(data["particle_color_hex"]),
            particle_count=int(data["particle_count"]),
        )


@dataclass(frozen=True)
class ScrollPalmWorldSceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    asset_prefix: str
    wide_source_relpath: str
    portrait_source_relpath: str
    unsharp_sigma_px: float
    unsharp_amount: float
    perimeter_taper_px: int
    interior_inpaint_inset_px: int
    bg_inpaint_dilation_px: int
    bg_inpaint_blur_sigma: float
    bg_jpeg_quality: int
    min_component_area_px: int
    palm_group_prompt_count: int
    sand_suppress_min_red: float
    sand_suppress_min_red_minus_green: float
    sand_suppress_max_green_minus_blue: float
    fg_plant_sand_suppress_boxes: tuple[BoundingBoxPrompt, ...]
    mid_plant_sand_suppress_boxes: tuple[BoundingBoxPrompt, ...]
    fg_sam_prompts: tuple[SamPointGroupPrompt, ...]
    mid_sam_prompts: tuple[SamPointGroupPrompt, ...]
    room_depth_clean: RoomDepthCleanConfig
    mid_depth_solve: ContactAnchoredLayerDepthConfig
    fg_depth_solve: ContactAnchoredLayerDepthConfig
    camera_rig: SceneCameraRigSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "asset_prefix": self.asset_prefix,
            "wide_source_relpath": self.wide_source_relpath,
            "portrait_source_relpath": self.portrait_source_relpath,
            "unsharp_sigma_px": self.unsharp_sigma_px,
            "unsharp_amount": self.unsharp_amount,
            "perimeter_taper_px": self.perimeter_taper_px,
            "interior_inpaint_inset_px": self.interior_inpaint_inset_px,
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "bg_inpaint_blur_sigma": self.bg_inpaint_blur_sigma,
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "min_component_area_px": self.min_component_area_px,
            "palm_group_prompt_count": self.palm_group_prompt_count,
            "sand_suppress_min_red": self.sand_suppress_min_red,
            "sand_suppress_min_red_minus_green": self.sand_suppress_min_red_minus_green,
            "sand_suppress_max_green_minus_blue": self.sand_suppress_max_green_minus_blue,
            "fg_plant_sand_suppress_boxes": [b.to_dict() for b in self.fg_plant_sand_suppress_boxes],
            "mid_plant_sand_suppress_boxes": [b.to_dict() for b in self.mid_plant_sand_suppress_boxes],
            "fg_sam_prompts": [p.to_dict() for p in self.fg_sam_prompts],
            "mid_sam_prompts": [p.to_dict() for p in self.mid_sam_prompts],
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "fg_depth_solve": self.fg_depth_solve.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScrollPalmWorldSceneSpec:
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            asset_prefix=str(data["asset_prefix"]),
            wide_source_relpath=str(data["wide_source_relpath"]),
            portrait_source_relpath=str(data["portrait_source_relpath"]),
            unsharp_sigma_px=float(data["unsharp_sigma_px"]),
            unsharp_amount=float(data["unsharp_amount"]),
            perimeter_taper_px=int(data["perimeter_taper_px"]),
            interior_inpaint_inset_px=int(data["interior_inpaint_inset_px"]),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            bg_inpaint_blur_sigma=float(data["bg_inpaint_blur_sigma"]),
            bg_jpeg_quality=int(data["bg_jpeg_quality"]),
            min_component_area_px=int(data["min_component_area_px"]),
            palm_group_prompt_count=int(data["palm_group_prompt_count"]),
            sand_suppress_min_red=float(data["sand_suppress_min_red"]),
            sand_suppress_min_red_minus_green=float(data["sand_suppress_min_red_minus_green"]),
            sand_suppress_max_green_minus_blue=float(data["sand_suppress_max_green_minus_blue"]),
            fg_plant_sand_suppress_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["fg_plant_sand_suppress_boxes"]
            ),
            mid_plant_sand_suppress_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["mid_plant_sand_suppress_boxes"]
            ),
            fg_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["fg_sam_prompts"]),
            mid_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["mid_sam_prompts"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            mid_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_depth_solve"]),
            fg_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["fg_depth_solve"]),
            camera_rig=SceneCameraRigSpec.from_dict(data["camera_rig"]),
        )


@dataclass(frozen=True)
class HumptyDumptyBundleSceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    asset_prefix: str
    source_relpath: str
    source_scale: float
    dest_xy_expanded: tuple[int, int]
    plate_feather_collar_px: int
    studio_clean_grain_box_src: tuple[int, int, int, int]
    studio_fit_exclude_box_src: tuple[int, int, int, int]
    studio_fit_tanh_gain: float
    studio_fit_low_percentile: float
    studio_fit_low_scale: float
    studio_fit_high_percentile: float
    studio_fit_high_scale: float
    harmonic_downsample_factor: int
    harmonic_lowpass_sigma_px: float
    harmonic_plate_inset_px: int
    harmonic_outer_border_px: int
    auth_grain_erode_px: float
    auth_grain_feather_sigma_px: float
    lama_midfreq_sigma_px: float
    lama_midfreq_clip_u8: float
    quilt_tile_px: int
    quilt_overlap_px: int
    grain_highpass_sigma_px: float
    grain_restore_strength: float
    bg_shadow_crevice_box_src: tuple[int, int, int, int]
    grounding_shadow_box_inner: tuple[int, int, int, int]
    grounding_shadow_sigma_xy: tuple[float, float]
    grounding_shadow_strength: float
    bg_inpaint_dilation_px: int
    bg_inpaint_blur_sigma: float
    bg_jpeg_quality: int
    sam_support_dilate_px: int
    sam_core_erode_px: int
    min_component_area_px: int
    fg_sam_prompts: tuple[SamPointGroupPrompt, ...]
    room_depth_clean: RoomDepthCleanConfig
    fg_depth_solve: ContactAnchoredLayerDepthConfig
    camera_rig: SceneCameraRigSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "asset_prefix": self.asset_prefix,
            "source_relpath": self.source_relpath,
            "source_scale": self.source_scale,
            "dest_xy_expanded": list(self.dest_xy_expanded),
            "plate_feather_collar_px": self.plate_feather_collar_px,
            "studio_clean_grain_box_src": list(self.studio_clean_grain_box_src),
            "studio_fit_exclude_box_src": list(self.studio_fit_exclude_box_src),
            "studio_fit_tanh_gain": self.studio_fit_tanh_gain,
            "studio_fit_low_percentile": self.studio_fit_low_percentile,
            "studio_fit_low_scale": self.studio_fit_low_scale,
            "studio_fit_high_percentile": self.studio_fit_high_percentile,
            "studio_fit_high_scale": self.studio_fit_high_scale,
            "harmonic_downsample_factor": self.harmonic_downsample_factor,
            "harmonic_lowpass_sigma_px": self.harmonic_lowpass_sigma_px,
            "harmonic_plate_inset_px": self.harmonic_plate_inset_px,
            "harmonic_outer_border_px": self.harmonic_outer_border_px,
            "auth_grain_erode_px": self.auth_grain_erode_px,
            "auth_grain_feather_sigma_px": self.auth_grain_feather_sigma_px,
            "lama_midfreq_sigma_px": self.lama_midfreq_sigma_px,
            "lama_midfreq_clip_u8": self.lama_midfreq_clip_u8,
            "quilt_tile_px": self.quilt_tile_px,
            "quilt_overlap_px": self.quilt_overlap_px,
            "grain_highpass_sigma_px": self.grain_highpass_sigma_px,
            "grain_restore_strength": self.grain_restore_strength,
            "bg_shadow_crevice_box_src": list(self.bg_shadow_crevice_box_src),
            "grounding_shadow_box_inner": list(self.grounding_shadow_box_inner),
            "grounding_shadow_sigma_xy": list(self.grounding_shadow_sigma_xy),
            "grounding_shadow_strength": self.grounding_shadow_strength,
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "bg_inpaint_blur_sigma": self.bg_inpaint_blur_sigma,
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "sam_support_dilate_px": self.sam_support_dilate_px,
            "sam_core_erode_px": self.sam_core_erode_px,
            "min_component_area_px": self.min_component_area_px,
            "fg_sam_prompts": [p.to_dict() for p in self.fg_sam_prompts],
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "fg_depth_solve": self.fg_depth_solve.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumptyDumptyBundleSceneSpec:
        dxy = data["dest_xy_expanded"]
        gb = data["studio_clean_grain_box_src"]
        eb = data["studio_fit_exclude_box_src"]
        scb = data["bg_shadow_crevice_box_src"]
        gsb = data["grounding_shadow_box_inner"]
        gss = data["grounding_shadow_sigma_xy"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            asset_prefix=str(data["asset_prefix"]),
            source_relpath=str(data["source_relpath"]),
            source_scale=float(data["source_scale"]),
            dest_xy_expanded=(int(dxy[0]), int(dxy[1])),
            plate_feather_collar_px=int(data["plate_feather_collar_px"]),
            studio_clean_grain_box_src=(int(gb[0]), int(gb[1]), int(gb[2]), int(gb[3])),
            studio_fit_exclude_box_src=(int(eb[0]), int(eb[1]), int(eb[2]), int(eb[3])),
            studio_fit_tanh_gain=float(data["studio_fit_tanh_gain"]),
            studio_fit_low_percentile=float(data["studio_fit_low_percentile"]),
            studio_fit_low_scale=float(data["studio_fit_low_scale"]),
            studio_fit_high_percentile=float(data["studio_fit_high_percentile"]),
            studio_fit_high_scale=float(data["studio_fit_high_scale"]),
            harmonic_downsample_factor=int(data["harmonic_downsample_factor"]),
            harmonic_lowpass_sigma_px=float(data["harmonic_lowpass_sigma_px"]),
            harmonic_plate_inset_px=int(data["harmonic_plate_inset_px"]),
            harmonic_outer_border_px=int(data["harmonic_outer_border_px"]),
            auth_grain_erode_px=float(data["auth_grain_erode_px"]),
            auth_grain_feather_sigma_px=float(data["auth_grain_feather_sigma_px"]),
            lama_midfreq_sigma_px=float(data["lama_midfreq_sigma_px"]),
            lama_midfreq_clip_u8=float(data["lama_midfreq_clip_u8"]),
            quilt_tile_px=int(data["quilt_tile_px"]),
            quilt_overlap_px=int(data["quilt_overlap_px"]),
            grain_highpass_sigma_px=float(data["grain_highpass_sigma_px"]),
            grain_restore_strength=float(data["grain_restore_strength"]),
            bg_shadow_crevice_box_src=(int(scb[0]), int(scb[1]), int(scb[2]), int(scb[3])),
            grounding_shadow_box_inner=(int(gsb[0]), int(gsb[1]), int(gsb[2]), int(gsb[3])),
            grounding_shadow_sigma_xy=(float(gss[0]), float(gss[1])),
            grounding_shadow_strength=float(data["grounding_shadow_strength"]),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            bg_inpaint_blur_sigma=float(data["bg_inpaint_blur_sigma"]),
            bg_jpeg_quality=int(data["bg_jpeg_quality"]),
            sam_support_dilate_px=int(data["sam_support_dilate_px"]),
            sam_core_erode_px=int(data["sam_core_erode_px"]),
            min_component_area_px=int(data["min_component_area_px"]),
            fg_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["fg_sam_prompts"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            fg_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["fg_depth_solve"]),
            camera_rig=SceneCameraRigSpec.from_dict(data["camera_rig"]),
        )


@dataclass(frozen=True)
class ScrollHumptyBundleConfig:
    scenes_output_relpath: str
    spec_output_abspath: str
    inner_dimensions: CanvasDimensionsConfig
    expanded_bg_scale: float
    lama_max_working_side_px: int
    lama_roi_context_margin_px: int
    edt_rim_probe_px: int
    max_edt_rim_jump_world: float
    depth_model: DepthModelConfig
    segmentation: SegmentationConfig
    watermark_inpainting: InpaintingConfig
    scene_inpainting: InpaintingConfig
    topology_verification: LayerTopologyVerificationConfig
    watermark_targets: tuple[WatermarkCleanTargetSpec, ...]
    scroll_palm_world: ScrollPalmWorldSceneSpec
    humpty_dumpty_bundle: HumptyDumptyBundleSceneSpec

    @property
    def expanded_dimensions(self) -> CanvasDimensionsConfig:
        return CanvasDimensionsConfig(
            width=int(round(self.inner_dimensions.width * self.expanded_bg_scale)),
            height=int(round(self.inner_dimensions.height * self.expanded_bg_scale)),
        )

    @property
    def inner_offset_xy(self) -> tuple[int, int]:
        exp = self.expanded_dimensions
        return (
            (exp.width - self.inner_dimensions.width) // 2,
            (exp.height - self.inner_dimensions.height) // 2,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenes_output_relpath": self.scenes_output_relpath,
            "spec_output_abspath": self.spec_output_abspath,
            "inner_dimensions": self.inner_dimensions.to_dict(),
            "expanded_bg_scale": self.expanded_bg_scale,
            "lama_max_working_side_px": self.lama_max_working_side_px,
            "lama_roi_context_margin_px": self.lama_roi_context_margin_px,
            "edt_rim_probe_px": self.edt_rim_probe_px,
            "max_edt_rim_jump_world": self.max_edt_rim_jump_world,
            "depth_model": self.depth_model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "watermark_inpainting": self.watermark_inpainting.to_dict(),
            "scene_inpainting": self.scene_inpainting.to_dict(),
            "topology_verification": self.topology_verification.to_dict(),
            "watermark_targets": [w.to_dict() for w in self.watermark_targets],
            "scroll_palm_world": self.scroll_palm_world.to_dict(),
            "humpty_dumpty_bundle": self.humpty_dumpty_bundle.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScrollHumptyBundleConfig:
        return cls(
            scenes_output_relpath=str(data["scenes_output_relpath"]),
            spec_output_abspath=str(data["spec_output_abspath"]),
            inner_dimensions=CanvasDimensionsConfig.from_dict(data["inner_dimensions"]),
            expanded_bg_scale=float(data["expanded_bg_scale"]),
            lama_max_working_side_px=int(data["lama_max_working_side_px"]),
            lama_roi_context_margin_px=int(data["lama_roi_context_margin_px"]),
            edt_rim_probe_px=int(data["edt_rim_probe_px"]),
            max_edt_rim_jump_world=float(data["max_edt_rim_jump_world"]),
            depth_model=DepthModelConfig.from_dict(data["depth_model"]),
            segmentation=SegmentationConfig.from_dict(data["segmentation"]),
            watermark_inpainting=InpaintingConfig.from_dict(data["watermark_inpainting"]),
            scene_inpainting=InpaintingConfig.from_dict(data["scene_inpainting"]),
            topology_verification=LayerTopologyVerificationConfig.from_dict(
                data["topology_verification"]
            ),
            watermark_targets=tuple(
                WatermarkCleanTargetSpec.from_dict(w) for w in data["watermark_targets"]
            ),
            scroll_palm_world=ScrollPalmWorldSceneSpec.from_dict(data["scroll_palm_world"]),
            humpty_dumpty_bundle=HumptyDumptyBundleSceneSpec.from_dict(
                data["humpty_dumpty_bundle"]
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw_json: str) -> ScrollHumptyBundleConfig:
        return cls.from_dict(json.loads(raw_json))


# ============================================================================
# Canonical Configuration Factory (Zero Magic Numbers Inline)
# ============================================================================


def default_scroll_humpty_bundle_config() -> ScrollHumptyBundleConfig:
    depth_cfg = DepthModelConfig(
        model_id="depth-anything/Depth-Anything-V2-Small-hf",
        background_gaussian_sigma=10.0,
        depth_discontinuity_percentile_fg=62.0,
        depth_discontinuity_percentile_mid=30.0,
    )
    seg_cfg = SegmentationConfig(
        rembg_model_name="isnet-general-use",
        sam_model_id="facebook/sam-vit-base",
        alpha_foreground_threshold=42,
        alpha_full_opaque_threshold=185,
        min_component_area_px=120,
        sam_morph_close_px=3,
        edge_feather_sigma=0.80,
    )
    wm_inpaint_cfg = InpaintingConfig(
        dilation_radius_px=8,
        mask_blur_sigma=2.0,
        jpeg_quality=95,
    )
    scene_inpaint_cfg = InpaintingConfig(
        dilation_radius_px=26,
        mask_blur_sigma=4.5,
        jpeg_quality=95,
    )
    verify_cfg = LayerTopologyVerificationConfig(
        alpha_threshold_u8=20,
        rim_erosion_px=2,
        rim_exterior_probe_px=6,
        max_allowed_contact_dz_world=2.5,
        max_allowed_p95_contact_dz_world=1.8,
        max_allowed_rim_jump_world=2.5,
        host_monotonicity_tolerance_world=2.0,
    )

    watermark_targets = (
        WatermarkCleanTargetSpec(
            name="scroll_palm_world_clean",
            source_abspath="/Users/talsheaffer/Downloads/Gemini_Generated_Image_.png",
            output_relpath="prototype/assets/sources/scroll_palm_world_clean.png",
            output_format="PNG",
            jpeg_quality=98,
            corner_crop_px=256,
            star_box_from_br=(82, 82, 30, 30),
            astroid_exponent=0.55,
            inpaint_dilation_px=8,
            inpaint_blur_sigma=2.0,
            blend_dilation_px=3,
            alpha_zero_floor=0.035,
            min_identical_pixel_ratio=0.998,
        ),
        WatermarkCleanTargetSpec(
            name="scroll_palm_world_wide_3x2_clean",
            source_abspath="/Users/talsheaffer/.gemini/antigravity/brain/235713ae-3cfb-48f0-94e4-2012c0824fd5/coastal_clearing_wide_3x2_1793481185928.jpg",
            output_relpath="prototype/assets/sources/scroll_palm_world_wide_3x2_clean.jpg",
            output_format="JPEG",
            jpeg_quality=98,
            corner_crop_px=256,
            star_box_from_br=(62, 62, 18, 18),
            astroid_exponent=0.55,
            inpaint_dilation_px=8,
            inpaint_blur_sigma=2.0,
            blend_dilation_px=3,
            alpha_zero_floor=0.035,
            min_identical_pixel_ratio=0.998,
        ),
        WatermarkCleanTargetSpec(
            name="humpty_dumpty_bundle_clean",
            source_abspath="/Users/talsheaffer/Downloads/Gemini_Generated_Image_ (1).png",
            output_relpath="prototype/assets/sources/humpty_dumpty_bundle_clean.png",
            output_format="PNG",
            jpeg_quality=98,
            corner_crop_px=256,
            star_box_from_br=(82, 82, 30, 30),
            astroid_exponent=0.55,
            inpaint_dilation_px=8,
            inpaint_blur_sigma=2.0,
            blend_dilation_px=3,
            alpha_zero_floor=0.035,
            min_identical_pixel_ratio=0.998,
        ),
    )

    scroll_palm_spec = ScrollPalmWorldSceneSpec(
        scene_id="scroll_palm_world",
        title_he="עולם הדקל והמוטות בתוך המגילה — ׳מגילה וכף רגל׳",
        short_label_he="עולם המגילה",
        icon="🌴",
        asset_prefix="scene14_scroll_world",
        wide_source_relpath="prototype/assets/sources/scroll_palm_world_wide_3x2_clean.jpg",
        portrait_source_relpath="prototype/assets/sources/scroll_palm_world_clean.png",
        unsharp_sigma_px=1.35,
        unsharp_amount=0.22,
        perimeter_taper_px=28,
        interior_inpaint_inset_px=30,
        bg_inpaint_dilation_px=26,
        bg_inpaint_blur_sigma=4.5,
        bg_jpeg_quality=95,
        min_component_area_px=140,
        palm_group_prompt_count=2,
        sand_suppress_min_red=138.0,
        sand_suppress_min_red_minus_green=9.0,
        sand_suppress_max_green_minus_blue=29.0,
        fg_plant_sand_suppress_boxes=(
            BoundingBoxPrompt(x0=720, y0=888, x1=935, y1=1000),
            BoundingBoxPrompt(x0=885, y0=742, x1=1212, y1=992),
            BoundingBoxPrompt(x0=2, y0=745, x1=312, y1=1000),
        ),
        mid_plant_sand_suppress_boxes=(
            BoundingBoxPrompt(x0=738, y0=472, x1=916, y1=648),
        ),
        fg_sam_prompts=(
            # FG-1: Bottom-center foreground philodendron rosette entering clearing
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=720, y0=888, x1=935, y1=1000),
                points_xy=(
                    (780.0, 965.0),
                    (815.0, 935.0),
                    (825.0, 975.0),
                    (855.0, 945.0),
                    (860.0, 982.0),
                    (898.0, 975.0),
                    (755.0, 910.0),
                    (795.0, 905.0),
                    (840.0, 905.0),
                    (895.0, 915.0),
                    (930.0, 955.0),
                    (725.0, 975.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
            ),
            # FG-2: Lush dark-green foreground plant right-of-center
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=885, y0=742, x1=1212, y1=992),
                points_xy=(
                    (1035.0, 835.0),
                    (1015.0, 875.0),
                    (1060.0, 880.0),
                    (1035.0, 925.0),
                    (965.0, 865.0),
                    (1115.0, 875.0),
                    (1145.0, 925.0),
                    (955.0, 920.0),
                    (925.0, 790.0),
                    (1095.0, 775.0),
                    (1185.0, 830.0),
                    (1195.0, 945.0),
                    (920.0, 965.0),
                    (1075.0, 982.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
            ),
            # FG-3: Bottom-left foreground spiky agave/yucca
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=2, y0=745, x1=312, y1=1000),
                points_xy=(
                    (135.0, 945.0),
                    (145.0, 905.0),
                    (112.0, 875.0),
                    (165.0, 865.0),
                    (82.0, 925.0),
                    (198.0, 935.0),
                    (55.0, 965.0),
                    (45.0, 810.0),
                    (225.0, 815.0),
                    (265.0, 895.0),
                    (255.0, 975.0),
                    (18.0, 885.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
            ),
            # FG-4A: Left diagonal black cylindrical rod
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=0, y0=735, x1=332, y1=915),
                points_xy=(
                    (42.0, 868.0),
                    (185.0, 795.0),
                    (295.0, 750.0),
                    (150.0, 760.0),
                    (220.0, 825.0),
                    (70.0, 825.0),
                ),
                point_labels=(1, 1, 1, 0, 0, 0),
            ),
            # FG-4B: Upper-left angled black cylindrical rod
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=295, y0=745, x1=555, y1=825),
                points_xy=(
                    (345.0, 770.0),
                    (435.0, 792.0),
                    (515.0, 808.0),
                    (420.0, 758.0),
                    (410.0, 820.0),
                ),
                point_labels=(1, 1, 1, 0, 0),
            ),
            # FG-4C: Long foreground left-center black cylindrical rod
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=235, y0=815, x1=695, y1=915),
                points_xy=(
                    (290.0, 888.0),
                    (435.0, 865.0),
                    (575.0, 845.0),
                    (665.0, 830.0),
                    (440.0, 835.0),
                    (460.0, 892.0),
                ),
                point_labels=(1, 1, 1, 1, 0, 0),
            ),
            # FG-4D: Central horizontal black cylindrical rod
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=612, y0=832, x1=1005, y1=885),
                points_xy=(
                    (655.0, 862.0),
                    (775.0, 858.0),
                    (895.0, 852.0),
                    (975.0, 848.0),
                    (775.0, 830.0),
                    (775.0, 882.0),
                ),
                point_labels=(1, 1, 1, 1, 0, 0),
            ),
            # FG-4E: Bottom-center diagonal black cylindrical rod
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=595, y0=885, x1=835, y1=998),
                points_xy=(
                    (625.0, 908.0),
                    (695.0, 938.0),
                    (758.0, 966.0),
                    (675.0, 895.0),
                    (650.0, 955.0),
                ),
                point_labels=(1, 1, 1, 0, 0),
            ),
            # FG-4F: Right long diagonal black cylindrical rod
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=995, y0=768, x1=1496, y1=902),
                points_xy=(
                    (1135.0, 805.0),
                    (1265.0, 835.0),
                    (1395.0, 865.0),
                    (1475.0, 882.0),
                    (1265.0, 800.0),
                    (1265.0, 868.0),
                    (1415.0, 830.0),
                ),
                point_labels=(1, 1, 1, 1, 0, 0, 0),
            ),
            # FG-4G: Bottom-right black cylindrical rod
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=1312, y0=898, x1=1498, y1=978),
                points_xy=(
                    (1355.0, 922.0),
                    (1435.0, 945.0),
                    (1482.0, 958.0),
                    (1415.0, 905.0),
                    (1395.0, 968.0),
                ),
                point_labels=(1, 1, 1, 0, 0),
            ),
            # FG-5: Foreground grey coastal stones
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=408, y0=882, x1=585, y1=1000),
                points_xy=(
                    (445.0, 912.0),
                    (515.0, 968.0),
                    (395.0, 945.0),
                    (585.0, 940.0),
                    (485.0, 875.0),
                ),
                point_labels=(1, 1, 0, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=568, y0=768, x1=688, y1=842),
                points_xy=(
                    (615.0, 802.0),
                    (645.0, 808.0),
                    (555.0, 795.0),
                    (688.0, 795.0),
                ),
                point_labels=(1, 1, 0, 0),
            ),
        ),
        mid_sam_prompts=(
            # MID-1A: Central coconut palm trunk
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=905, y0=240, x1=1025, y1=575),
                points_xy=(
                    (945.0, 340.0),
                    (955.0, 425.0),
                    (965.0, 505.0),
                    (975.0, 548.0),
                    (895.0, 440.0),
                    (1040.0, 460.0),
                    (965.0, 585.0),
                ),
                point_labels=(1, 1, 1, 1, 0, 0, 0),
            ),
            # MID-1B: Central coconut palm arching fronds
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=395, y0=0, x1=1295, y1=575),
                points_xy=(
                    (555.0, 95.0),
                    (465.0, 195.0),
                    (565.0, 235.0),
                    (725.0, 145.0),
                    (695.0, 285.0),
                    (835.0, 125.0),
                    (835.0, 275.0),
                    (965.0, 85.0),
                    (995.0, 215.0),
                    (1125.0, 85.0),
                    (1095.0, 345.0),
                    (1035.0, 455.0),
                    (370.0, 260.0),
                    (470.0, 395.0),
                    (635.0, 435.0),
                    (795.0, 435.0),
                    (895.0, 485.0),
                    (1245.0, 435.0),
                    (1145.0, 320.0),
                    (1165.0, 365.0),
                    (1185.0, 405.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            ),
            # MID-2: Mid-left spiky yucca/agave
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=108, y0=535, x1=298, y1=742),
                points_xy=(
                    (202.0, 625.0),
                    (198.0, 668.0),
                    (168.0, 608.0),
                    (232.0, 612.0),
                    (202.0, 712.0),
                    (125.0, 575.0),
                    (278.0, 575.0),
                    (135.0, 705.0),
                    (275.0, 705.0),
                    (202.0, 525.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
            ),
            # MID-3A: Mid-clearing boulder
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=472, y0=494, x1=648, y1=604),
                points_xy=(
                    (548.0, 545.0),
                    (515.0, 558.0),
                    (585.0, 565.0),
                    (612.0, 575.0),
                    (476.0, 535.0),
                    (642.0, 545.0),
                    (634.0, 515.0),
                    (555.0, 498.0),
                    (550.0, 602.0),
                ),
                point_labels=(1, 1, 1, 1, 0, 0, 0, 0, 0),
            ),
            # MID-3B: Mid-clearing spiky agaves & ferns
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=558, y0=588, x1=675, y1=686),
                points_xy=(
                    (615.0, 635.0),
                    (615.0, 658.0),
                    (552.0, 635.0),
                    (680.0, 635.0),
                ),
                point_labels=(1, 1, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=600, y0=476, x1=740, y1=596),
                points_xy=(
                    (672.0, 542.0),
                    (668.0, 568.0),
                    (606.0, 535.0),
                    (734.0, 535.0),
                    (670.0, 482.0),
                ),
                point_labels=(1, 1, 0, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=738, y0=472, x1=916, y1=648),
                points_xy=(
                    (845.0, 512.0),
                    (792.0, 575.0),
                    (856.0, 598.0),
                    (802.0, 528.0),
                    (745.0, 520.0),
                    (908.0, 555.0),
                ),
                point_labels=(1, 1, 1, 0, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=775, y0=712, x1=882, y1=808),
                points_xy=(
                    (832.0, 755.0),
                    (832.0, 782.0),
                    (768.0, 760.0),
                    (888.0, 760.0),
                ),
                point_labels=(1, 1, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=1055, y0=568, x1=1148, y1=655),
                points_xy=(
                    (1098.0, 608.0),
                    (1048.0, 610.0),
                    (1152.0, 610.0),
                ),
                point_labels=(1, 0, 0),
            ),
            # MID-4: Mid-clearing black cylindrical rods
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=258, y0=620, x1=505, y1=652),
                points_xy=(
                    (315.0, 638.0),
                    (425.0, 634.0),
                    (365.0, 612.0),
                    (365.0, 658.0),
                ),
                point_labels=(1, 1, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=492, y0=582, x1=722, y1=618),
                points_xy=(
                    (555.0, 598.0),
                    (695.0, 606.0),
                    (645.0, 580.0),
                    (645.0, 624.0),
                ),
                point_labels=(1, 1, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=352, y0=692, x1=712, y1=736),
                points_xy=(
                    (415.0, 708.0),
                    (555.0, 714.0),
                    (665.0, 720.0),
                    (535.0, 688.0),
                    (535.0, 738.0),
                ),
                point_labels=(1, 1, 1, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=685, y0=612, x1=756, y1=712),
                points_xy=(
                    (708.0, 685.0),
                    (735.0, 638.0),
                    (682.0, 655.0),
                    (758.0, 665.0),
                ),
                point_labels=(1, 1, 0, 0),
            ),
        ),
        room_depth_clean=RoomDepthCleanConfig(
            alpha_threshold_u8=20,
            mask_dilation_px=18,
            post_smooth_sigma_px=3.0,
            interior_blend_band_px=12.0,
        ),
        mid_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=-360.0,
            room_depth_amplitude=118.0,
            room_pin_margin_uv=0.045,
            layer_z=-358.0,
            layer_depth_amplitude=158.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=16,
            contact_boxes=(
                # Top crown join where arching palm fronds meet outer canopy (`ΔZ_world == 0`)
                ContactAnchorBox(x0=380, y0=0, x1=1310, y1=40, mode="top_band", band_px=36),
                # Palm trunk sand footprint
                ContactAnchorBox(x0=895, y0=490, x1=1040, y1=585, mode="bottom_band", band_px=10),
                # Mid-left spiky yucca sand footprint
                ContactAnchorBox(x0=100, y0=530, x1=305, y1=750, mode="bottom_band", band_px=10),
                # Mid boulder + upper-mid plants & rods sand footprints
                ContactAnchorBox(x0=468, y0=475, x1=744, y1=608, mode="bottom_band", band_px=8),
                ContactAnchorBox(x0=735, y0=470, x1=920, y1=652, mode="bottom_band", band_px=8),
                ContactAnchorBox(x0=250, y0=580, x1=765, y1=745, mode="bottom_band", band_px=8),
                ContactAnchorBox(x0=770, y0=708, x1=890, y1=815, mode="bottom_band", band_px=8),
                ContactAnchorBox(x0=1050, y0=565, x1=1155, y1=660, mode="bottom_band", band_px=8),
            ),
            neural_relief_amplitude=34.0,
            contact_relief_taper_px=32.0,
            interior_smooth_sigma_px=1.2,
        ),
        fg_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=-360.0,
            room_depth_amplitude=118.0,
            room_pin_margin_uv=0.045,
            layer_z=-354.0,
            layer_depth_amplitude=196.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=16,
            contact_boxes=(
                # Perimeter join bands where fg rods/leaves meet outer collar (`ΔZ_world == 0`)
                ContactAnchorBox(x0=0, y0=960, x1=1500, y1=1000, mode="bottom_band", band_px=38),
                ContactAnchorBox(x0=0, y0=730, x1=42, y1=1000, mode="left_band", band_px=38),
                ContactAnchorBox(x0=1458, y0=760, x1=1500, y1=1000, mode="right_band", band_px=38),
                # Grounded sand footprints of foreground plants, stones & rods
                ContactAnchorBox(x0=2, y0=740, x1=335, y1=1000, mode="bottom_band", band_px=10),
                ContactAnchorBox(x0=235, y0=745, x1=700, y1=1000, mode="bottom_band", band_px=10),
                ContactAnchorBox(x0=595, y0=830, x1=1010, y1=1000, mode="bottom_band", band_px=10),
                ContactAnchorBox(x0=885, y0=742, x1=1215, y1=995, mode="bottom_band", band_px=12),
                ContactAnchorBox(x0=995, y0=768, x1=1500, y1=985, mode="bottom_band", band_px=10),
            ),
            neural_relief_amplitude=42.0,
            contact_relief_taper_px=28.0,
            interior_smooth_sigma_px=1.2,
        ),
        camera_rig=SceneCameraRigSpec(
            min_inner_scale=1.96,
            initial_camera_y=-8.0,
            initial_camera_z=535.0,
            min_camera_z=-95.0,
            max_camera_z=680.0,
            max_yaw_deg=22.0,
            max_pitch_deg=15.0,
            max_strafe_x=115.0,
            max_strafe_y=72.0,
            dolly_speed_mult=1.35,
            particle_color_hex="0xE8DEC2",
            particle_count=65,
        ),
    )

    humpty_spec = HumptyDumptyBundleSceneSpec(
        scene_id="humpty_dumpty_bundle",
        title_he="צרור עטוף בבד וחבל — פסל על משטח הגלריה",
        short_label_he="צרור עטוף בבד וחבל",
        icon="",
        asset_prefix="scene15_humpty_bundle",
        source_relpath="prototype/assets/sources/humpty_dumpty_bundle_clean.png",
        source_scale=1.12,
        dest_xy_expanded=(515, 248),
        plate_feather_collar_px=96,
        studio_clean_grain_box_src=(28, 28, 515, 845),
        studio_fit_exclude_box_src=(485, 255, 1110, 880),
        studio_fit_tanh_gain=0.90,
        studio_fit_low_percentile=3.0,
        studio_fit_low_scale=0.94,
        studio_fit_high_percentile=98.0,
        studio_fit_high_scale=1.02,
        harmonic_downsample_factor=6,
        harmonic_lowpass_sigma_px=24.0,
        harmonic_plate_inset_px=28,
        harmonic_outer_border_px=12,
        auth_grain_erode_px=42.0,
        auth_grain_feather_sigma_px=20.0,
        lama_midfreq_sigma_px=24.0,
        lama_midfreq_clip_u8=3.0,
        quilt_tile_px=280,
        quilt_overlap_px=96,
        grain_highpass_sigma_px=14.0,
        grain_restore_strength=0.92,
        bg_shadow_crevice_box_src=(555, 420, 1015, 880),
        grounding_shadow_box_inner=(555, 730, 915, 804),
        grounding_shadow_sigma_xy=(68.0, 22.0),
        grounding_shadow_strength=0.11,
        bg_inpaint_dilation_px=28,
        bg_inpaint_blur_sigma=5.0,
        bg_jpeg_quality=95,
        sam_support_dilate_px=7,
        sam_core_erode_px=5,
        min_component_area_px=2500,
        fg_sam_prompts=(
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=546, y0=328, x1=918, y1=856),
                points_xy=(
                    (640.0, 410.0),
                    (740.0, 440.0),
                    (670.0, 560.0),
                    (780.0, 560.0),
                    (865.0, 580.0),
                    (640.0, 720.0),
                    (760.0, 740.0),
                    (660.0, 825.0),
                    (835.0, 825.0),
                    (480.0, 420.0),
                    (490.0, 700.0),
                    (535.0, 845.0),
                    (720.0, 868.0),
                    (945.0, 550.0),
                    (945.0, 760.0),
                    (935.0, 850.0),
                    (700.0, 290.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            ),
        ),
        room_depth_clean=RoomDepthCleanConfig(
            alpha_threshold_u8=20,
            mask_dilation_px=22,
            post_smooth_sigma_px=3.5,
            interior_blend_band_px=14.0,
        ),
        fg_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=-350.0,
            room_depth_amplitude=96.0,
            room_pin_margin_uv=0.045,
            layer_z=-346.0,
            layer_depth_amplitude=168.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=12,
            contact_boxes=(
                ContactAnchorBox(x0=510, y0=720, x1=960, y1=820, mode="bottom_band", band_px=12),
            ),
            neural_relief_amplitude=54.0,
            contact_relief_taper_px=34.0,
            interior_smooth_sigma_px=1.4,
        ),
        camera_rig=SceneCameraRigSpec(
            min_inner_scale=1.94,
            initial_camera_y=0.0,
            initial_camera_z=530.0,
            min_camera_z=-85.0,
            max_camera_z=690.0,
            max_yaw_deg=24.0,
            max_pitch_deg=16.5,
            max_strafe_x=135.0,
            max_strafe_y=85.0,
            dolly_speed_mult=1.30,
            particle_color_hex="0xDFD4C4",
            particle_count=55,
        ),
    )

    return ScrollHumptyBundleConfig(
        scenes_output_relpath="prototype/assets/scenes",
        spec_output_abspath="/Users/talsheaffer/.gemini/antigravity/brain/235713ae-3cfb-48f0-94e4-2012c0824fd5/scratch/workstream_e_specs.json",
        inner_dimensions=CanvasDimensionsConfig(width=1500, height=1000),
        expanded_bg_scale=1.80,
        lama_max_working_side_px=1350,
        lama_roi_context_margin_px=180,
        edt_rim_probe_px=6,
        max_edt_rim_jump_world=2.5,
        depth_model=depth_cfg,
        segmentation=seg_cfg,
        watermark_inpainting=wm_inpaint_cfg,
        scene_inpainting=scene_inpaint_cfg,
        topology_verification=verify_cfg,
        watermark_targets=watermark_targets,
        scroll_palm_world=scroll_palm_spec,
        humpty_dumpty_bundle=humpty_spec,
    )


# ============================================================================
# Task 1: Surgical Bottom-Right Gemini Watermark Healing
# ============================================================================


def clean_gemini_watermarks(
    project_root: Path,
    cfg: ScrollHumptyBundleConfig,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for target in cfg.watermark_targets:
        src_path = Path(target.source_abspath)
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing watermark source image: {src_path}")

        rgb = Image.open(src_path).convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        h, w, _ = arr.shape
        cw = min(target.corner_crop_px, w)
        ch = min(target.corner_crop_px, h)
        x0, y0 = w - cw, h - ch
        crop_rgb = rgb.crop((x0, y0, w, h))

        rmax, bmax, rmin, bmin = target.star_box_from_br
        sx0 = max(0, cw - rmax)
        sy0 = max(0, ch - bmax)
        sx1 = min(cw, cw - rmin)
        sy1 = min(ch, ch - bmin)

        mask_crop = np.zeros((ch, cw), dtype=np.uint8)
        cx = 0.5 * (sx0 + sx1)
        cy = 0.5 * (sy0 + sy1)
        rx = max(1.0, 0.5 * (sx1 - sx0))
        ry = max(1.0, 0.5 * (sy1 - sy0))
        yy, xx = np.ogrid[:ch, :cw]
        nx = np.abs(xx - cx) / rx
        ny = np.abs(yy - cy) / ry
        astroid = (
            np.power(nx, target.astroid_exponent) + np.power(ny, target.astroid_exponent)
        ) <= 1.0
        mask_crop[astroid] = 255

        healed_crop = np.asarray(
            lama_inpainter.inpaint_background(
                crop_rgb,
                mask_crop,
                dilation_radius_px=target.inpaint_dilation_px,
                mask_blur_sigma=target.inpaint_blur_sigma,
            ),
            dtype=np.float32,
        )

        k_blend = 2 * target.blend_dilation_px + 1
        dil = cv2.dilate(
            mask_crop,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_blend, k_blend)),
        )
        alpha = np.clip(
            ndi.gaussian_filter(dil.astype(np.float32) / 255.0, sigma=target.inpaint_blur_sigma)
            * 1.12,
            0.0,
            1.0,
        )[:, :, None]
        alpha = np.where(alpha < target.alpha_zero_floor, 0.0, alpha)

        orig_crop_f = arr[y0:h, x0:w].astype(np.float32)
        comp_crop = np.clip(
            np.round(alpha * healed_crop + (1.0 - alpha) * orig_crop_f), 0, 255
        ).astype(np.uint8)

        out = arr.copy()
        out[y0:h, x0:w] = comp_crop
        identical_ratio = float(np.all(out == arr, axis=2).mean())
        if identical_ratio < target.min_identical_pixel_ratio:
            raise RuntimeError(
                f"Watermark heal on {target.name} modified too many pixels: "
                f"identical_ratio={identical_ratio:.5f} < {target.min_identical_pixel_ratio:.5f}"
            )

        out_path = project_root / target.output_relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img = Image.fromarray(out, mode="RGB")
        if target.output_format.upper() == "JPEG":
            out_img.save(
                out_path,
                format="JPEG",
                quality=target.jpeg_quality,
                subsampling=0,
                optimize=True,
            )
        else:
            out_img.save(out_path, format="PNG")

        print(
            f"[Watermark Clean] {target.name} ({w}x{h}) -> {target.output_relpath} "
            f"(bit_identical={identical_ratio * 100.0:.3f}%)"
        )
        results[target.name] = {
            "outputRelpath": target.output_relpath,
            "width": w,
            "height": h,
            "bitIdenticalPixelRatio": round(identical_ratio, 6),
        }
    return results


# ============================================================================
# Shared Verification Helpers ($C^0$ Contact + $C^{-1}$ EDT Rim Gate)
# ============================================================================


def _compute_edt_rim_p99_jump_world(
    rgba_u8: np.ndarray,
    depth_u8: np.ndarray,
    layer_amplitude: float,
    alpha_threshold_u8: int,
    rim_probe_px: int,
) -> float:
    opaque = rgba_u8[:, :, 3] >= alpha_threshold_u8
    if not np.any(opaque) or np.all(opaque):
        return 0.0
    dist_out, nearest_in = ndi.distance_transform_edt(~opaque, return_indices=True)
    rim = (~opaque) & (dist_out <= float(rim_probe_px))
    if not np.any(rim):
        return 0.0
    ry, rx = np.where(rim)
    ny, nx = nearest_in[0, ry, rx], nearest_in[1, ry, rx]
    dz_world = (
        np.abs(depth_u8[ry, rx].astype(np.float32) - depth_u8[ny, nx].astype(np.float32))
        / 255.0
        * float(layer_amplitude)
    )
    return float(np.percentile(dz_world, 99.0))


def _compute_auto_floor_p95_dz_world(
    bg_depth_u8: np.ndarray,
    layer_rgba_u8: np.ndarray,
    layer_depth_u8: np.ndarray,
    solve_cfg: ContactAnchoredLayerDepthConfig,
) -> float:
    auto_cfg = ContactAnchoredLayerDepthConfig(
        bg_z=solve_cfg.bg_z,
        room_depth_amplitude=solve_cfg.room_depth_amplitude,
        room_pin_margin_uv=solve_cfg.room_pin_margin_uv,
        layer_z=solve_cfg.layer_z,
        layer_depth_amplitude=solve_cfg.layer_depth_amplitude,
        layer_pin_margin_uv=0.0,
        alpha_threshold_u8=solve_cfg.alpha_threshold_u8,
    )
    contact = extract_contact_anchor_mask(
        layer_rgba_u8[:, :, 3] >= solve_cfg.alpha_threshold_u8, auto_cfg
    )
    if not np.any(contact):
        return 0.0
    room_z_exp = compute_world_z_map(
        bg_depth_u8, solve_cfg.bg_z, solve_cfg.room_depth_amplitude, solve_cfg.room_pin_margin_uv
    )
    host_z = extract_layer_window_from_expanded(room_z_exp, layer_rgba_u8.shape[:2])
    obj_z = compute_world_z_map(
        layer_depth_u8, solve_cfg.layer_z, solve_cfg.layer_depth_amplitude, 0.0
    )
    return float(np.percentile(np.abs(obj_z[contact] - host_z[contact]), 95.0))


def _verify_and_gate_layer_topology(
    label: str,
    bg_depth_u8: np.ndarray,
    layer_rgba_u8: np.ndarray,
    layer_depth_u8: np.ndarray,
    contact_mask: np.ndarray,
    solve_cfg: ContactAnchoredLayerDepthConfig,
    verify_cfg: LayerTopologyVerificationConfig,
    edt_rim_probe_px: int,
    max_edt_rim_jump_world: float,
) -> dict[str, Any]:
    report: LayerTopologyVerificationReport = verify_layer_contact_and_discontinuity(
        room_depth_expanded=bg_depth_u8,
        layer_rgba=layer_rgba_u8,
        layer_depth=layer_depth_u8,
        contact_mask=contact_mask,
        bg_z=solve_cfg.bg_z,
        room_depth_amplitude=solve_cfg.room_depth_amplitude,
        room_pin_margin_uv=solve_cfg.room_pin_margin_uv,
        layer_z=solve_cfg.layer_z,
        layer_depth_amplitude=solve_cfg.layer_depth_amplitude,
        layer_pin_margin_uv=solve_cfg.layer_pin_margin_uv,
        verify_cfg=verify_cfg,
    )
    p99_rim_jump = _compute_edt_rim_p99_jump_world(
        rgba_u8=layer_rgba_u8,
        depth_u8=layer_depth_u8,
        layer_amplitude=solve_cfg.layer_depth_amplitude,
        alpha_threshold_u8=solve_cfg.alpha_threshold_u8,
        rim_probe_px=edt_rim_probe_px,
    )
    auto_floor_p95_dz = _compute_auto_floor_p95_dz_world(
        bg_depth_u8=bg_depth_u8,
        layer_rgba_u8=layer_rgba_u8,
        layer_depth_u8=layer_depth_u8,
        solve_cfg=solve_cfg,
    )
    if not report.contact_passed or not report.discontinuity_passed:
        raise RuntimeError(
            f"{label} failed depth topology verification: {report.to_json()}"
        )
    if p99_rim_jump > max_edt_rim_jump_world:
        raise RuntimeError(
            f"{label} failed outward EDT rim gate: p99_rim_jump={p99_rim_jump:.3f} > {max_edt_rim_jump_world:.3f}"
        )
    if auto_floor_p95_dz > verify_cfg.max_allowed_contact_dz_world:
        raise RuntimeError(
            f"{label} failed auto-floor contact gate: p95_dz={auto_floor_p95_dz:.3f} > {verify_cfg.max_allowed_contact_dz_world:.3f}"
        )
    out = report.to_dict()
    out["liveGateP99RimJumpWorld"] = round(p99_rim_jump, 4)
    out["autoFloorP95AbsDzWorld"] = round(auto_floor_p95_dz, 4)
    return out


# ============================================================================
# Task 2: Scene 14 Builder — `scroll_palm_world` (`scene14_scroll_world_*`)
# ============================================================================


def _prune_unanchored_cutout_islands(
    layer_rgba: Image.Image,
    solve_cfg: ContactAnchoredLayerDepthConfig,
) -> Image.Image:
    """
    Ensures 100% of 4-connected opaque components (`alpha >= solve_cfg.alpha_threshold_u8`)
    satisfy `size >= solve_cfg.min_component_area_px` AND belong to a contact-anchored PDE domain
    (`anchored_opaque == opaque_mask`), guaranteeing `max_abs_rim_jump_world == 0.0` everywhere.
    """
    arr = np.asarray(layer_rgba.convert("RGBA"), dtype=np.uint8).copy()
    thresh = solve_cfg.alpha_threshold_u8
    opaque = arr[:, :, 3] >= thresh
    if not np.any(opaque):
        return layer_rgba

    # Mirror `solve_contact_anchored_layer_depth`'s exact 4-connected size + PDE anchor filter
    for _ in range(2):
        opaque = arr[:, :, 3] >= thresh
        if not np.any(opaque):
            break
        raw_lbl, raw_n = ndi.label(opaque)
        if raw_n > 0:
            sizes = np.bincount(raw_lbl.ravel())
            keep = sizes >= solve_cfg.min_component_area_px
            keep[0] = False
            clean_opaque = keep[raw_lbl]
        else:
            clean_opaque = opaque

        contact = extract_contact_anchor_mask(clean_opaque, solve_cfg)
        if solve_cfg.component_bridge_px > 0:
            pde_domain = ndi.distance_transform_edt(~clean_opaque) <= float(
                solve_cfg.component_bridge_px
            )
        else:
            pde_domain = clean_opaque.copy()

        lbl_pde, n_pde = ndi.label(pde_domain)
        anchored_pde = np.zeros_like(pde_domain)
        for pid in range(1, n_pde + 1):
            comp = lbl_pde == pid
            if np.any(comp & contact):
                anchored_pde |= comp

        anchored_opaque = clean_opaque & anchored_pde
        # Drop any >= thresh pixel not in anchored_opaque, plus any sub-threshold halo > 3px away
        near_anchored = ndi.distance_transform_edt(~anchored_opaque) <= 3.0
        drop_mask = (opaque & (~anchored_opaque)) | ((~opaque) & (~near_anchored))
        arr[drop_mask] = 0

    return Image.fromarray(arr, mode="RGBA")


def _segment_prompt_union_u8(
    rgb_image: Image.Image,
    prompts: Sequence[SamPointGroupPrompt],
    sam_extractor: SamBoxMaskExtractor,
    min_component_area_px: int,
) -> np.ndarray:
    w, h = rgb_image.size
    union_u8 = np.zeros((h, w), dtype=np.uint8)
    for prompt in prompts:
        mask_u8 = sam_extractor.segment_point_groups(
            rgb_image=rgb_image,
            point_groups=(prompt,),
            exclude_boxes=(),
            max_pinhole_fill_px=2400,
            min_major_component_px=min_component_area_px,
            right_edge_fade_x_start=None,
        )
        if prompt.box is not None:
            b = prompt.box
            box_clip = np.zeros_like(mask_u8)
            box_clip[max(0, b.y0) : min(h, b.y1), max(0, b.x0) : min(w, b.x1)] = 255
            mask_u8 = np.minimum(mask_u8, box_clip)
        union_u8 = np.maximum(union_u8, mask_u8)
    return union_u8


def build_scroll_palm_world_scene(
    project_root: Path,
    cfg: ScrollHumptyBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    spec = cfg.scroll_palm_world
    inner_dims = cfg.inner_dimensions
    exp_dims = cfg.expanded_dimensions
    ox, oy = cfg.inner_offset_xy
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Scene 14: {spec.scene_id}] Building coastal palm & rods world "
        f"(inner={inner_dims.width}x{inner_dims.height}, expanded={exp_dims.width}x{exp_dims.height})..."
    )

    wide_src = Image.open(project_root / spec.wide_source_relpath).convert("RGB")
    resized_2700 = wide_src.resize((exp_dims.width, exp_dims.height), Image.Resampling.LANCZOS)
    arr_f = np.asarray(resized_2700, dtype=np.float32)
    blur_f = ndi.gaussian_filter(arr_f, sigma=(spec.unsharp_sigma_px, spec.unsharp_sigma_px, 0.0))
    sharp_f = np.clip(arr_f + spec.unsharp_amount * (arr_f - blur_f), 0.0, 255.0)
    master_2700_rgb = Image.fromarray(np.rint(sharp_f).astype(np.uint8), mode="RGB")

    inner_rgb = master_2700_rgb.crop((ox, oy, ox + inner_dims.width, oy + inner_dims.height))
    inner_np = np.asarray(inner_rgb, dtype=np.uint8)

    # 1. Segment FG (`philodendron + lush foreground plants + closest black cylindrical rods & stones`)
    fg_sam_u8 = _segment_prompt_union_u8(
        rgb_image=inner_rgb,
        prompts=spec.fg_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=spec.min_component_area_px,
    )
    # Refine crisp leaf tips on the bottom-center & right-center plants with IS-Net inside dilated SAM support
    isnet_rgba = remove(inner_rgb, session=matte_extractor.session)
    isnet_a = np.asarray(isnet_rgba, dtype=np.uint8)[:, :, 3]
    plant_roi = np.zeros_like(fg_sam_u8)
    plant_roi[740:1000, 720:1215] = 255
    fg_plant_support = cv2.dilate(
        np.minimum(fg_sam_u8, plant_roi),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    fg_refined_plants = np.where(fg_plant_support > 0, np.maximum(fg_sam_u8, isnet_a), 0).astype(
        np.uint8
    )
    fg_alpha_raw = np.maximum(fg_sam_u8, fg_refined_plants)

    # Suppress bright neutral-beige dry sand webbing trapped between spiky foreground green leaf blades
    plant_sand_roi = np.zeros((inner_dims.height, inner_dims.width), dtype=bool)
    for b in spec.fg_plant_sand_suppress_boxes:
        plant_sand_roi[
            max(0, b.y0) : min(inner_dims.height, b.y1),
            max(0, b.x0) : min(inner_dims.width, b.x1),
        ] = True
    r_f = inner_np[:, :, 0].astype(np.float32)
    g_f = inner_np[:, :, 1].astype(np.float32)
    b_f = inner_np[:, :, 2].astype(np.float32)
    sand_webbing = (
        plant_sand_roi
        & (r_f >= spec.sand_suppress_min_red)
        & ((r_f - g_f) >= spec.sand_suppress_min_red_minus_green)
        & ((g_f - b_f) <= spec.sand_suppress_max_green_minus_blue)
    )
    fg_alpha_clean_f = np.where(sand_webbing, 0.0, fg_alpha_raw.astype(np.float32))
    fg_alpha_raw = np.clip(
        np.rint(
            np.where(
                plant_sand_roi,
                ndi.gaussian_filter(fg_alpha_clean_f, sigma=0.55),
                fg_alpha_raw.astype(np.float32),
            )
        ),
        0,
        255,
    ).astype(np.uint8)

    # 2. Segment MID (`central arching coconut palm trunk & fronds + mid spiky agaves, boulder & rods`)
    n_palm = spec.palm_group_prompt_count
    palm_u8 = _segment_prompt_union_u8(
        rgb_image=inner_rgb,
        prompts=spec.mid_sam_prompts[:n_palm],
        sam_extractor=sam_extractor,
        min_component_area_px=spec.min_component_area_px,
    )
    nl_p, lbl_p, stats_p, _ = cv2.connectedComponentsWithStats(
        (palm_u8 >= 20).astype(np.uint8), connectivity=8
    )
    if nl_p > 1:
        largest_palm = 1 + int(np.argmax(stats_p[1:, cv2.CC_STAT_AREA]))
        palm_support = cv2.dilate(
            (lbl_p == largest_palm).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        palm_u8 = np.where(palm_support > 0, palm_u8, 0).astype(np.uint8)

    mid_other_u8 = _segment_prompt_union_u8(
        rgb_image=inner_rgb,
        prompts=spec.mid_sam_prompts[n_palm:],
        sam_extractor=sam_extractor,
        min_component_area_px=spec.min_component_area_px,
    )
    mid_sam_u8 = np.maximum(palm_u8, mid_other_u8)
    # Enforce strict foreground priority (`mid` never duplicates `fg` pixels)
    fg_excl = cv2.dilate(
        (fg_alpha_raw >= 64).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    mid_alpha_raw = np.where(fg_excl > 0, 0, mid_sam_u8).astype(np.uint8)

    # Suppress flat dry sand webbing trapped between adjacent mid-ground cycad rosettes
    mid_sand_roi = np.zeros((inner_dims.height, inner_dims.width), dtype=bool)
    for b in spec.mid_plant_sand_suppress_boxes:
        mid_sand_roi[
            max(0, b.y0) : min(inner_dims.height, b.y1),
            max(0, b.x0) : min(inner_dims.width, b.x1),
        ] = True
    mid_sand_webbing = (
        mid_sand_roi
        & (r_f >= spec.sand_suppress_min_red)
        & ((r_f - g_f) >= spec.sand_suppress_min_red_minus_green)
        & ((g_f - b_f) <= spec.sand_suppress_max_green_minus_blue)
    )
    mid_alpha_clean_f = np.where(mid_sand_webbing, 0.0, mid_alpha_raw.astype(np.float32))
    mid_alpha_raw = np.clip(
        np.rint(
            np.where(
                mid_sand_roi,
                ndi.gaussian_filter(mid_alpha_clean_f, sigma=0.55),
                mid_alpha_raw.astype(np.float32),
            )
        ),
        0,
        255,
    ).astype(np.uint8)

    # Apply smooth perimeter cosine taper so both `fg` and `mid` strictly obey zero-border-alpha
    taper_2d = _cosine_feather_2d(
        height=inner_dims.height,
        width=inner_dims.width,
        left_px=spec.perimeter_taper_px,
        top_px=spec.perimeter_taper_px,
        right_px=spec.perimeter_taper_px,
        bottom_px=spec.perimeter_taper_px,
        elliptical=False,
    )
    fg_alpha_tapered = np.clip(np.rint(fg_alpha_raw.astype(np.float32) * taper_2d), 0, 255).astype(
        np.uint8
    )
    mid_alpha_tapered = np.clip(
        np.rint(mid_alpha_raw.astype(np.float32) * taper_2d), 0, 255
    ).astype(np.uint8)

    fg_rgb_decontam = decontaminate_matte_edge_rgb(inner_np, fg_alpha_tapered)
    mid_rgb_decontam = decontaminate_matte_edge_rgb(inner_np, mid_alpha_tapered)

    fg_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            Image.fromarray(np.dstack([fg_rgb_decontam, fg_alpha_tapered]), mode="RGBA"),
            min_area_px=spec.min_component_area_px,
            alpha_thresh_u8=spec.fg_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.fg_depth_solve,
    )
    mid_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            Image.fromarray(np.dstack([mid_rgb_decontam, mid_alpha_tapered]), mode="RGBA"),
            min_area_px=spec.min_component_area_px,
            alpha_thresh_u8=spec.mid_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.mid_depth_solve,
    )

    _assert_zero_border_alpha(fg_canvas, f"{spec.scene_id}.fg")
    _assert_zero_border_alpha(mid_canvas, f"{spec.scene_id}.mid")

    fg_np = np.asarray(fg_canvas, dtype=np.uint8)
    mid_np = np.asarray(mid_canvas, dtype=np.uint8)

    # 3. Neural LaMa background inpainting on 2700x1800 (`dilation >= 25px`)
    # Inset the binary seed by `interior_inpaint_inset_px` so stems crossing into the outer 1.80x
    # collar remain continuous on `layer0` where `fg`/`mid` taper to alpha=0 at ΔZ_world == 0.
    inset = spec.interior_inpaint_inset_px
    inner_union = ((fg_np[:, :, 3] >= 24) | (mid_np[:, :, 3] >= 24)).astype(np.uint8) * 255
    inner_union[:inset, :] = 0
    inner_union[-inset:, :] = 0
    inner_union[:, :inset] = 0
    inner_union[:, -inset:] = 0

    bg_mask_2700 = np.zeros((exp_dims.height, exp_dims.width), dtype=np.uint8)
    bg_mask_2700[oy : oy + inner_dims.height, ox : ox + inner_dims.width] = inner_union

    print(
        f"  -> Inpainting `fg | mid` silhouettes out of {exp_dims.width}x{exp_dims.height} "
        f"coastal clearing via neural LaMa (dilation={spec.bg_inpaint_dilation_px}px)..."
    )
    bg_inpainted_rgb = _inpaint_masked_plate_lama(
        src_rgb=master_2700_rgb,
        binary_mask_u8=bg_mask_2700,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.bg_inpaint_dilation_px,
        blur_sigma=spec.bg_inpaint_blur_sigma,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )
    # Clamp any dilation spill outside the interior inset so outer palm crown & philodendron stay 100% crisp
    keep_interior = np.zeros((exp_dims.height, exp_dims.width), dtype=np.float32)
    pad_c = max(6, inset - 6)
    keep_interior[
        oy + pad_c : oy + inner_dims.height - pad_c,
        ox + pad_c : ox + inner_dims.width - pad_c,
    ] = 1.0
    keep_interior = ndi.gaussian_filter(keep_interior, sigma=6.0)[:, :, None]
    bg_final_f = keep_interior * np.asarray(bg_inpainted_rgb, dtype=np.float32) + (
        1.0 - keep_interior
    ) * np.asarray(master_2700_rgb, dtype=np.float32)
    bg_inpainted_rgb = Image.fromarray(np.clip(np.rint(bg_final_f), 0, 255).astype(np.uint8), mode="RGB")

    # 4. Neural Depth Anything V2 on intact composite & LaMa-inpainted background + Law 3 harmonic clean
    raw_intact_depth_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(master_2700_rgb) * 255.0), 0, 255
    ).astype(np.uint8)
    inpainted_room_depth_img = depth_estimator.build_smoothed_room_depth_uint8(bg_inpainted_rgb)
    inpainted_room_depth_u8 = np.asarray(inpainted_room_depth_img, dtype=np.uint8)

    # Mask cutout alpha for room-depth cleaning to interior region matching `bg_mask_2700`
    fg_clean_cutout = fg_np.copy()
    mid_clean_cutout = mid_np.copy()
    for cut in (fg_clean_cutout, mid_clean_cutout):
        cut[:inset, :, 3] = 0
        cut[-inset:, :, 3] = 0
        cut[:, :inset, 3] = 0
        cut[:, -inset:, 3] = 0

    bg_depth_clean_u8 = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(mid_clean_cutout, fg_clean_cutout),
        cfg=spec.room_depth_clean,
    )

    # 5. Solve Law 1 ($C^0$ contact-anchored) + Law 2 ($C^{-1}$ outward EDT) layer depths
    mid_protrusion = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_depth_u8,
        cutout_rgba=mid_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.mid_depth_solve.alpha_threshold_u8,
    )
    fg_protrusion = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_depth_u8,
        cutout_rgba=fg_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.fg_depth_solve.alpha_threshold_u8,
    )

    mid_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=mid_np,
        cfg=spec.mid_depth_solve,
        neural_protrusion_norm=mid_protrusion,
    )
    fg_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=fg_np,
        cfg=spec.fg_depth_solve,
        neural_protrusion_norm=fg_protrusion,
    )

    mid_verify = _verify_and_gate_layer_topology(
        label=f"{spec.scene_id}.mid",
        bg_depth_u8=bg_depth_clean_u8,
        layer_rgba_u8=mid_np,
        layer_depth_u8=mid_solved.depth_u8,
        contact_mask=mid_solved.contact_mask,
        solve_cfg=spec.mid_depth_solve,
        verify_cfg=cfg.topology_verification,
        edt_rim_probe_px=cfg.edt_rim_probe_px,
        max_edt_rim_jump_world=cfg.max_edt_rim_jump_world,
    )
    fg_verify = _verify_and_gate_layer_topology(
        label=f"{spec.scene_id}.fg",
        bg_depth_u8=bg_depth_clean_u8,
        layer_rgba_u8=fg_np,
        layer_depth_u8=fg_solved.depth_u8,
        contact_mask=fg_solved.contact_mask,
        solve_cfg=spec.fg_depth_solve,
        verify_cfg=cfg.topology_verification,
        edt_rim_probe_px=cfg.edt_rim_probe_px,
        max_edt_rim_jump_world=cfg.max_edt_rim_jump_world,
    )

    pfx = spec.asset_prefix
    bg_inpainted_rgb.save(
        scenes_dir / f"{pfx}_bg_inpainted.jpg",
        format="JPEG",
        quality=spec.bg_jpeg_quality,
        subsampling=0,
        optimize=True,
    )
    Image.fromarray(bg_depth_clean_u8, mode="L").save(
        scenes_dir / f"{pfx}_bg_depth.png", format="PNG"
    )
    mid_canvas.save(scenes_dir / f"{pfx}_mid.png", format="PNG")
    Image.fromarray(mid_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_mid_depth.png", format="PNG"
    )
    fg_canvas.save(scenes_dir / f"{pfx}_fg.png", format="PNG")
    Image.fromarray(fg_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_fg_depth.png", format="PNG"
    )

    mid_cov = float((mid_np[:, :, 3] >= 20).mean())
    fg_cov = float((fg_np[:, :, 3] >= 20).mean())
    print(
        f"  -> Saved {pfx}_* (mid_cov={mid_cov:.4f}, fg_cov={fg_cov:.4f}, "
        f"mid_p95_dz={mid_verify['p95_abs_contact_dz_world']:.3f}, "
        f"fg_p95_dz={fg_verify['p95_abs_contact_dz_world']:.3f})"
    )

    rig = spec.camera_rig
    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "icon": spec.icon,
        "bgTexture": f"assets/scenes/{pfx}_bg_inpainted.jpg",
        "depthTexture": f"assets/scenes/{pfx}_bg_depth.png",
        "midTexture": f"assets/scenes/{pfx}_mid.png",
        "midDepthTexture": f"assets/scenes/{pfx}_mid_depth.png",
        "midDepthAmplitude": spec.mid_depth_solve.layer_depth_amplitude,
        "midPinMarginUv": 0.0,
        "midSegmentsX": 224,
        "midSegmentsY": 150,
        "midDepthTest": False,
        "midDepthWrite": False,
        "fgTexture": f"assets/scenes/{pfx}_fg.png",
        "fgDepthTexture": f"assets/scenes/{pfx}_fg_depth.png",
        "fgDepthAmplitude": spec.fg_depth_solve.layer_depth_amplitude,
        "fgPinMarginUv": 0.0,
        "fgSegmentsX": 224,
        "fgSegmentsY": 150,
        "fgDepthTest": False,
        "fgDepthWrite": False,
        "fgExpanded": False,
        "boreSightAtRest": True,
        "bgZ": spec.mid_depth_solve.bg_z,
        "midZ": spec.mid_depth_solve.layer_z,
        "fgZ": spec.fg_depth_solve.layer_z,
        "minInnerScale": rig.min_inner_scale,
        "bgSegmentsX": 224,
        "bgSegmentsY": 150,
        "roomDepthAmplitude": spec.mid_depth_solve.room_depth_amplitude,
        "initialCameraY": rig.initial_camera_y,
        "initialCameraZ": rig.initial_camera_z,
        "minCameraZ": rig.min_camera_z,
        "maxCameraZ": rig.max_camera_z,
        "maxYawDeg": rig.max_yaw_deg,
        "maxPitchDeg": rig.max_pitch_deg,
        "maxStrafeX": rig.max_strafe_x,
        "maxStrafeY": rig.max_strafe_y,
        "dollySpeedMult": rig.dolly_speed_mult,
        "midRenderOrder": 6,
        "fgRenderOrder": 8,
        "particleColor": rig.particle_color_hex,
        "particleCount": rig.particle_count,
        "hasReflection": False,
        "midCoverageRatio": round(mid_cov, 4),
        "fgCoverageRatio": round(fg_cov, 4),
        "depthTopologyVerification": {
            "mid": mid_verify,
            "fg": fg_verify,
        },
    }


# ============================================================================
# Task 3: Scene 15 Builder — `humpty_dumpty_bundle` (`scene15_humpty_bundle_*`)
# ============================================================================


def _extract_highpass_grain_tile(
    src_f32: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
    sigma_px: float,
) -> np.ndarray:
    gx0, gy0, gx1, gy1 = box_xyxy
    patch = src_f32[gy0:gy1, gx0:gx1]
    low = ndi.gaussian_filter(patch, sigma=(sigma_px, sigma_px, 0.0))
    return (patch - low).astype(np.float32)


def _quilt_nonperiodic_grain_2700(
    grain_patch_f32: np.ndarray,
    target_wh: tuple[int, int],
    tile_px: int,
    overlap_px: int,
) -> np.ndarray:
    """
    Synthesizes a seamless, non-periodic 2700x1800 studio grain field from authentic source
    taupe surface grain via deterministic staggered 2D cosine-feathered patch quilting.
    """
    tw, th = target_wh
    ph, pw, _ = grain_patch_f32.shape
    t_sz = min(tile_px, pw - 4, ph - 4)
    step = max(32, t_sz - overlap_px)

    acc = np.zeros((th, tw, 3), dtype=np.float32)
    w_acc = np.zeros((th, tw, 1), dtype=np.float32)
    win = _cosine_feather_2d(
        height=t_sz,
        width=t_sz,
        left_px=overlap_px,
        top_px=overlap_px,
        right_px=overlap_px,
        bottom_px=overlap_px,
        elliptical=False,
    )[:, :, None]
    win = np.maximum(win, 1e-3)

    variants = (
        grain_patch_f32,
        grain_patch_f32[:, ::-1, :],
        grain_patch_f32[::-1, :, :],
        grain_patch_f32[::-1, ::-1, :],
    )
    rng = np.random.default_rng(7015)

    for gy in range(0, th, step):
        for gx in range(0, tw, step):
            v = variants[int(rng.integers(0, len(variants)))]
            ox = int(rng.integers(0, max(1, pw - t_sz)))
            oy = int(rng.integers(0, max(1, ph - t_sz)))
            sub = v[oy : oy + t_sz, ox : ox + t_sz]

            y1 = min(th, gy + t_sz)
            x1 = min(tw, gx + t_sz)
            sh, sw = y1 - gy, x1 - gx
            acc[gy:y1, gx:x1] += sub[:sh, :sw] * win[:sh, :sw]
            w_acc[gy:y1, gx:x1] += win[:sh, :sw]

    return acc / np.maximum(1e-4, w_acc)


def _synthesize_humpty_studio_fields(
    src_rgb: Image.Image,
    exp_dims: CanvasDimensionsConfig,
    spec: HumptyDumptyBundleSceneSpec,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    Solves a $C^1$-seamless 2700x1800 warm-taupe studio illumination field (`harm_2700`) via
    masked normalized low-pass + 2D harmonic Laplace interpolation anchored to clean studio
    pixels and damped 3-term planar outer bounds, paired with quilted + authentic high-frequency
    studio surface grain (`blended_grain_2700`) and the intact `master_2700_rgb` composite.
    """
    src_f = np.asarray(src_rgb, dtype=np.float32)
    sh, sw, _ = src_f.shape
    scaled_w = int(round(sw * spec.source_scale))
    scaled_h = int(round(sh * spec.source_scale))
    scaled_rgb = src_rgb.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    scaled_f = np.asarray(scaled_rgb, dtype=np.float32)

    dx, dy = spec.dest_xy_expanded
    ew, eh = exp_dims.width, exp_dims.height

    ex0, ey0, ex1, ey1 = [int(round(v * spec.source_scale)) for v in spec.studio_fit_exclude_box_src]
    clean_mask = np.ones((scaled_h, scaled_w), dtype=bool)
    clean_mask[max(0, ey0) : min(scaled_h, ey1), max(0, ex0) : min(scaled_w, ex1)] = False

    # Masked normalized Gaussian low-pass strictly from clean studio pixels (zero white-cloth or shadow bleed)
    lp_sig = spec.harmonic_lowpass_sigma_px
    w_clean = np.maximum(1e-4, ndi.gaussian_filter(clean_mask.astype(np.float32), sigma=lp_sig))
    scaled_low = np.stack(
        [
            ndi.gaussian_filter(scaled_f[:, :, c] * clean_mask, sigma=lp_sig) / w_clean
            for c in range(3)
        ],
        axis=2,
    )

    plate_clean = np.zeros((eh, ew), dtype=bool)
    inset = spec.harmonic_plate_inset_px
    plate_clean[dy + inset : dy + scaled_h - inset, dx + inset : dx + scaled_w - inset] = (
        clean_mask[inset : scaled_h - inset, inset : scaled_w - inset]
    )

    yy_s, xx_s = np.where(clean_mask)[0][::6], np.where(clean_mask)[1][::6]
    gx_pts = (xx_s.astype(np.float32) + float(dx) - 0.5 * ew) / (0.5 * ew)
    gy_pts = (yy_s.astype(np.float32) + float(dy) - 0.5 * eh) / (0.5 * eh)
    gain = spec.studio_fit_tanh_gain
    tx_pts = np.tanh(gx_pts * gain)
    ty_pts = np.tanh(gy_pts * gain)
    design = np.column_stack([np.ones_like(tx_pts), tx_pts, ty_pts])

    yy_full = np.tanh(((np.arange(eh, dtype=np.float32) - 0.5 * eh) / (0.5 * eh)) * gain)[:, None]
    xx_full = np.tanh(((np.arange(ew, dtype=np.float32) - 0.5 * ew) / (0.5 * ew)) * gain)[None, :]
    dirichlet_full = np.zeros((eh, ew, 3), dtype=np.float32)
    for ch_idx in range(3):
        vals = scaled_f[yy_s, xx_s, ch_idx]
        lo_bound = float(np.percentile(vals, spec.studio_fit_low_percentile)) * spec.studio_fit_low_scale
        hi_bound = float(np.percentile(vals, spec.studio_fit_high_percentile)) * spec.studio_fit_high_scale
        coeffs, _, _, _ = np.linalg.lstsq(design, vals, rcond=None)
        pred = coeffs[0] + coeffs[1] * xx_full + coeffs[2] * yy_full
        dirichlet_full[:, :, ch_idx] = np.clip(pred, lo_bound, hi_bound)
    dirichlet_full[dy : dy + scaled_h, dx : dx + scaled_w] = scaled_low

    ds = spec.harmonic_downsample_factor
    gh, gw = eh // ds, ew // ds
    dom_ds = np.ones((gh, gw), dtype=bool)
    dir_mask_full = plate_clean.copy()
    ob = spec.harmonic_outer_border_px
    dir_mask_full[:ob, :] = True
    dir_mask_full[-ob:, :] = True
    dir_mask_full[:, :ob] = True
    dir_mask_full[:, -ob:] = True
    dir_mask_ds = (
        cv2.resize(dir_mask_full.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) > 0
    )

    harm_2700 = np.zeros((eh, ew, 3), dtype=np.float32)
    for ch_idx in range(3):
        d_vals_ds = cv2.resize(
            dirichlet_full[:, :, ch_idx], (gw, gh), interpolation=cv2.INTER_AREA
        ).astype(np.float64)
        sol_ds = _solve_harmonic_field_on_domain(dom_ds, dir_mask_ds, d_vals_ds).astype(np.float32)
        harm_2700[:, :, ch_idx] = cv2.resize(sol_ds, (ew, eh), interpolation=cv2.INTER_CUBIC)

    grain_patch = _extract_highpass_grain_tile(
        src_f32=scaled_f,
        box_xyxy=tuple(int(round(v * spec.source_scale)) for v in spec.studio_clean_grain_box_src),  # type: ignore[arg-type]
        sigma_px=spec.grain_highpass_sigma_px,
    )
    quilted_grain = _quilt_nonperiodic_grain_2700(
        grain_patch_f32=grain_patch,
        target_wh=(ew, eh),
        tile_px=spec.quilt_tile_px,
        overlap_px=spec.quilt_overlap_px,
    )

    eroded_clean = ndi.distance_transform_edt(plate_clean) >= spec.auth_grain_erode_px
    auth_w = np.where(
        plate_clean,
        ndi.gaussian_filter(eroded_clean.astype(np.float32), sigma=spec.auth_grain_feather_sigma_px),
        0.0,
    )[:, :, None]
    auth_grain_full = np.zeros((eh, ew, 3), dtype=np.float32)
    auth_grain_full[dy : dy + scaled_h, dx : dx + scaled_w] = np.where(
        clean_mask[:, :, None], scaled_f - scaled_low, 0.0
    )
    blended_grain_2700 = auth_w * auth_grain_full + (1.0 - auth_w) * (
        quilted_grain * spec.grain_restore_strength
    )

    clean_studio_f = np.clip(harm_2700 + blended_grain_2700, 0.0, 255.0)
    plate_weight = _cosine_feather_2d(
        height=scaled_h,
        width=scaled_w,
        left_px=spec.plate_feather_collar_px,
        top_px=spec.plate_feather_collar_px,
        right_px=spec.plate_feather_collar_px,
        bottom_px=spec.plate_feather_collar_px,
        elliptical=False,
    )[:, :, None]
    comp_f = clean_studio_f.copy()
    roi = comp_f[dy : dy + scaled_h, dx : dx + scaled_w]
    comp_f[dy : dy + scaled_h, dx : dx + scaled_w] = (
        plate_weight * scaled_f + (1.0 - plate_weight) * roi
    )
    master_2700_rgb = Image.fromarray(np.clip(np.rint(comp_f), 0, 255).astype(np.uint8), mode="RGB")
    return master_2700_rgb, harm_2700, blended_grain_2700


def build_humpty_dumpty_bundle_scene(
    project_root: Path,
    cfg: ScrollHumptyBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    spec = cfg.humpty_dumpty_bundle
    inner_dims = cfg.inner_dimensions
    exp_dims = cfg.expanded_dimensions
    ox, oy = cfg.inner_offset_xy
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Scene 15: {spec.scene_id}] Building wrapped Humpty Dumpty sculpture scene "
        f"(inner={inner_dims.width}x{inner_dims.height}, expanded={exp_dims.width}x{exp_dims.height})..."
    )

    src_rgb = Image.open(project_root / spec.source_relpath).convert("RGB")
    sw, sh = src_rgb.size
    scaled_w = int(round(sw * spec.source_scale))
    scaled_h = int(round(sh * spec.source_scale))

    # 1. Segment wrapped white cloth, twine cords & Humpty Dumpty via SAM + IS-Net on native 1216x880
    sam_u8 = _segment_prompt_union_u8(
        rgb_image=src_rgb,
        prompts=spec.fg_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=spec.min_component_area_px,
    )
    isnet_rgba = remove(src_rgb, session=matte_extractor.session)
    isnet_a = np.asarray(isnet_rgba, dtype=np.uint8)[:, :, 3]

    k_sup = 2 * spec.sam_support_dilate_px + 1
    k_ero = 2 * spec.sam_core_erode_px + 1
    sam_support = cv2.dilate(
        (sam_u8 >= 96).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sup, k_sup)),
    )
    sam_core = (
        cv2.erode(
            (sam_u8 >= 160).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_ero, k_ero)),
        )
        * 255
    )
    fused_src_alpha = np.where(sam_support > 0, np.maximum(isnet_a, sam_core), 0).astype(np.uint8)
    decontam_src_rgb = decontaminate_matte_edge_rgb(np.asarray(src_rgb, dtype=np.uint8), fused_src_alpha)
    scaled_fg_patch = Image.fromarray(
        np.dstack([decontam_src_rgb, fused_src_alpha]), mode="RGBA"
    ).resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    # 2. Synthesize seamless 2700x1800 harmonic + quilted studio fields & place FG onto 1500x1000
    master_2700_rgb, harm_2700, blended_grain_2700 = _synthesize_humpty_studio_fields(
        src_rgb=src_rgb,
        exp_dims=exp_dims,
        spec=spec,
    )

    fg_exp_canvas = Image.fromarray(
        np.zeros((exp_dims.height, exp_dims.width, 4), dtype=np.uint8), mode="RGBA"
    )
    fg_exp_canvas.alpha_composite(scaled_fg_patch, dest=spec.dest_xy_expanded)
    fg_inner_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            fg_exp_canvas.crop((ox, oy, ox + inner_dims.width, oy + inner_dims.height)),
            min_area_px=spec.min_component_area_px,
            alpha_thresh_u8=spec.fg_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.fg_depth_solve,
    )
    _assert_zero_border_alpha(fg_inner_canvas, f"{spec.scene_id}.fg")

    mid_empty_canvas = Image.fromarray(
        np.zeros((inner_dims.height, inner_dims.width, 4), dtype=np.uint8), mode="RGBA"
    )
    _assert_zero_border_alpha(mid_empty_canvas, f"{spec.scene_id}.mid")

    fg_np = np.asarray(fg_inner_canvas, dtype=np.uint8)

    # 3. Neural LaMa inpainting (`dilation >= 25px`) + frequency-separated harmonic studio envelope
    # so zero low-frequency grey FFC blob or rectangle seam can appear on the warm taupe gallery sweep.
    bg_mask_2700 = np.zeros((exp_dims.height, exp_dims.width), dtype=np.uint8)
    bg_mask_2700[oy : oy + inner_dims.height, ox : ox + inner_dims.width] = np.where(
        fg_np[:, :, 3] >= 20, 255, 0
    ).astype(np.uint8)
    dx_exp, dy_exp = spec.dest_xy_expanded
    cx0, cy0, cx1, cy1 = [int(round(v * spec.source_scale)) for v in spec.bg_shadow_crevice_box_src]
    bg_mask_2700[
        max(0, dy_exp + cy0) : min(exp_dims.height, dy_exp + cy1),
        max(0, dx_exp + cx0) : min(exp_dims.width, dx_exp + cx1),
    ] = 255

    print(
        f"  -> Inpainting sculpture & dark contact crevice out of {exp_dims.width}x{exp_dims.height} "
        f"warm taupe backdrop via neural LaMa (dilation={spec.bg_inpaint_dilation_px}px)..."
    )
    bg_inpainted_raw = _inpaint_masked_plate_lama(
        src_rgb=master_2700_rgb,
        binary_mask_u8=bg_mask_2700,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.bg_inpaint_dilation_px,
        blur_sigma=spec.bg_inpaint_blur_sigma,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )
    lama_f = np.asarray(bg_inpainted_raw, dtype=np.float32)
    lama_low = ndi.gaussian_filter(
        lama_f, sigma=(spec.lama_midfreq_sigma_px, spec.lama_midfreq_sigma_px, 0.0)
    )
    lama_midfreq = np.clip(
        lama_f - lama_low, -spec.lama_midfreq_clip_u8, spec.lama_midfreq_clip_u8
    )

    hole_dil = cv2.dilate(
        bg_mask_2700,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * spec.bg_inpaint_dilation_px + 1, 2 * spec.bg_inpaint_dilation_px + 1),
        ),
    )
    hole_w = ndi.gaussian_filter((hole_dil > 0).astype(np.float32), sigma=8.0)[:, :, None]

    # Subtle natural ambient grounding shadow right under the resting cloth bundle base
    gx0, gy0, gx1, gy1 = spec.grounding_shadow_box_inner
    sx_exp0, sy_exp0 = ox + gx0, oy + gy0
    sx_exp1, sy_exp1 = ox + gx1, oy + gy1
    scx = 0.5 * (sx_exp0 + sx_exp1)
    scy = 0.5 * (sy_exp0 + sy_exp1)
    srx = max(1.0, 0.5 * (sx_exp1 - sx_exp0))
    sry = max(1.0, 0.5 * (sy_exp1 - sy_exp0))
    yy_g, xx_g = np.ogrid[: exp_dims.height, : exp_dims.width]
    ellip_seed = (((xx_g - scx) / srx) ** 2 + ((yy_g - scy) / sry) ** 2 <= 1.0).astype(np.float32)
    sig_x, sig_y = spec.grounding_shadow_sigma_xy
    ambient_shadow = ndi.gaussian_filter(ellip_seed, sigma=(sig_y, sig_x))
    if float(ambient_shadow.max()) > 1e-6:
        ambient_shadow = ambient_shadow / float(ambient_shadow.max())
    shadow_mult = (1.0 - spec.grounding_shadow_strength * ambient_shadow)[:, :, None]

    bg_inpainted_f = np.clip(
        harm_2700 * shadow_mult + hole_w * lama_midfreq + blended_grain_2700,
        0.0,
        255.0,
    )
    bg_inpainted_rgb = Image.fromarray(np.rint(bg_inpainted_f).astype(np.uint8), mode="RGB")

    # 4. Depth Anything V2 on intact composite & LaMa-inpainted backdrop + Law 3 harmonic clean
    raw_intact_depth_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(master_2700_rgb) * 255.0), 0, 255
    ).astype(np.uint8)
    inpainted_room_depth_img = depth_estimator.build_smoothed_room_depth_uint8(bg_inpainted_rgb)
    inpainted_room_depth_u8 = np.asarray(inpainted_room_depth_img, dtype=np.uint8)

    bg_depth_clean_u8 = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(fg_np,),
        cfg=spec.room_depth_clean,
    )

    # 5. Solve Law 1 ($C^0$ tabletop base anchor) + Law 2 ($C^{-1}$ outward EDT) 3D sculptural relief
    fg_protrusion = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_depth_u8,
        cutout_rgba=fg_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.fg_depth_solve.alpha_threshold_u8,
    )
    fg_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=fg_np,
        cfg=spec.fg_depth_solve,
        neural_protrusion_norm=fg_protrusion,
    )

    fg_verify = _verify_and_gate_layer_topology(
        label=f"{spec.scene_id}.fg",
        bg_depth_u8=bg_depth_clean_u8,
        layer_rgba_u8=fg_np,
        layer_depth_u8=fg_solved.depth_u8,
        contact_mask=fg_solved.contact_mask,
        solve_cfg=spec.fg_depth_solve,
        verify_cfg=cfg.topology_verification,
        edt_rim_probe_px=cfg.edt_rim_probe_px,
        max_edt_rim_jump_world=cfg.max_edt_rim_jump_world,
    )

    pfx = spec.asset_prefix
    bg_inpainted_rgb.save(
        scenes_dir / f"{pfx}_bg_inpainted.jpg",
        format="JPEG",
        quality=spec.bg_jpeg_quality,
        subsampling=0,
        optimize=True,
    )
    Image.fromarray(bg_depth_clean_u8, mode="L").save(
        scenes_dir / f"{pfx}_bg_depth.png", format="PNG"
    )
    mid_empty_canvas.save(scenes_dir / f"{pfx}_mid.png", format="PNG")
    fg_inner_canvas.save(scenes_dir / f"{pfx}_fg.png", format="PNG")
    Image.fromarray(fg_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_fg_depth.png", format="PNG"
    )

    fg_cov = float((fg_np[:, :, 3] >= 20).mean())
    print(
        f"  -> Saved {pfx}_* (fg_cov={fg_cov:.4f}, "
        f"fg_p95_dz={fg_verify['p95_abs_contact_dz_world']:.3f}, "
        f"auto_floor_p95_dz={fg_verify['autoFloorP95AbsDzWorld']:.3f})"
    )

    rig = spec.camera_rig
    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "icon": spec.icon,
        "bgTexture": f"assets/scenes/{pfx}_bg_inpainted.jpg",
        "depthTexture": f"assets/scenes/{pfx}_bg_depth.png",
        "midTexture": f"assets/scenes/{pfx}_mid.png",
        "midDepthTexture": None,
        "fgTexture": f"assets/scenes/{pfx}_fg.png",
        "fgDepthTexture": f"assets/scenes/{pfx}_fg_depth.png",
        "fgDepthAmplitude": spec.fg_depth_solve.layer_depth_amplitude,
        "fgPinMarginUv": 0.0,
        "fgSegmentsX": 224,
        "fgSegmentsY": 150,
        "fgDepthTest": False,
        "fgDepthWrite": False,
        "fgExpanded": False,
        "boreSightAtRest": True,
        "bgZ": spec.fg_depth_solve.bg_z,
        "midZ": -240.0,
        "fgZ": spec.fg_depth_solve.layer_z,
        "minInnerScale": rig.min_inner_scale,
        "bgSegmentsX": 224,
        "bgSegmentsY": 150,
        "roomDepthAmplitude": spec.fg_depth_solve.room_depth_amplitude,
        "initialCameraY": rig.initial_camera_y,
        "initialCameraZ": rig.initial_camera_z,
        "minCameraZ": rig.min_camera_z,
        "maxCameraZ": rig.max_camera_z,
        "maxYawDeg": rig.max_yaw_deg,
        "maxPitchDeg": rig.max_pitch_deg,
        "maxStrafeX": rig.max_strafe_x,
        "maxStrafeY": rig.max_strafe_y,
        "dollySpeedMult": rig.dolly_speed_mult,
        "midRenderOrder": 6,
        "fgRenderOrder": 8,
        "particleColor": rig.particle_color_hex,
        "particleCount": rig.particle_count,
        "hasReflection": False,
        "fgCoverageRatio": round(fg_cov, 4),
        "depthTopologyVerification": {
            "fg": fg_verify,
        },
    }


# ============================================================================
# Task 4: Upgrade `wandering_drawing_1` Scroll Foot & Dissolvable Veil
# ============================================================================


def build_wandering_1_portal_upgrade(
    project_root: Path,
    cfg: ScrollHumptyBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    new_scenes_cfg = default_new_scenes_bundle_config()
    portal_spec = new_scenes_cfg.wandering_drawing_1.scroll_portal
    print(
        f"\n[Scene 7 Upgrade: wandering_drawing_1] Extracting SAM `scroll_foot` extraLayer "
        f"& dissolvable `scroll_veil` over innerBoxPx={list(portal_spec.inner_box_px)}..."
    )
    summary = build_wandering_drawing_1_scroll_portal_layers(
        project_root=project_root,
        cfg=new_scenes_cfg,
        depth_estimator=depth_estimator,
        sam_extractor=sam_extractor,
        lama_inpainter=lama_inpainter,
    )

    scenes_dir = project_root / cfg.scenes_output_relpath
    pfx = new_scenes_cfg.wandering_drawing_1.asset_prefix
    foot_rgba = np.asarray(Image.open(scenes_dir / f"{pfx}_scroll_foot.png").convert("RGBA"))
    foot_depth = np.asarray(Image.open(scenes_dir / f"{pfx}_scroll_foot_depth.png").convert("L"))
    mid_rgba = np.asarray(Image.open(scenes_dir / f"{pfx}_mid.png").convert("RGBA"))
    mid_depth = np.asarray(Image.open(scenes_dir / f"{pfx}_mid_depth.png").convert("L"))

    foot_rim_p99 = _compute_edt_rim_p99_jump_world(
        rgba_u8=foot_rgba,
        depth_u8=foot_depth,
        layer_amplitude=portal_spec.foot_depth_amplitude,
        alpha_threshold_u8=20,
        rim_probe_px=cfg.edt_rim_probe_px,
    )
    mid_rim_p99 = _compute_edt_rim_p99_jump_world(
        rgba_u8=mid_rgba,
        depth_u8=mid_depth,
        layer_amplitude=24.0,
        alpha_threshold_u8=20,
        rim_probe_px=cfg.edt_rim_probe_px,
    )
    if foot_rim_p99 > cfg.max_edt_rim_jump_world or mid_rim_p99 > cfg.max_edt_rim_jump_world:
        raise RuntimeError(
            f"wandering_drawing_1 EDT rim gate failed: foot={foot_rim_p99:.3f}, mid={mid_rim_p99:.3f}"
        )

    print(
        f"  -> Saved {pfx}_{{mid,mid_depth,scroll_foot,scroll_foot_depth,scroll_veil}}.png "
        f"(foot_rim_p99={foot_rim_p99:.3f}, veil_cov={summary['veilCoverageRatio']:.4f}, "
        f"recomp_max_err={summary['maxRestPoseRecompErrorU8']} u8)"
    )

    return {
        "sceneId": "wandering_drawing_1",
        "midZ": portal_spec.mid_z,
        "midRenderOrder": 6,
        "fgRenderOrder": 9,
        "midTexture": summary["midTexture"],
        "midDepthTexture": summary["midDepthTexture"],
        "extraLayers": [
            {
                "id": portal_spec.foot_layer_id,
                "labelHe": portal_spec.foot_label_he,
                "texture": summary["scrollFootTexture"],
                "depthTexture": summary["scrollFootDepthTexture"],
                "zPos": portal_spec.foot_z_pos,
                "depthAmplitude": portal_spec.foot_depth_amplitude,
                "pinMarginUv": 0.0,
                "segmentsX": 192,
                "segmentsY": 128,
                "renderOrder": 7,
                "expanded": False,
                "baseOpacity": 1.0,
                "excursionBoost": 0.0,
                "depthTest": False,
                "depthWrite": False,
            }
        ],
        "portalShelves": [
            {
                "id": portal_spec.portal_id,
                "targetSceneId": portal_spec.target_scene_id,
                "portalMode": "aperture",
                "labelHe": portal_spec.label_he,
                "shortBadgeHe": portal_spec.short_badge_he,
                "innerBoxPx": list(portal_spec.inner_box_px),
                "worldZ": portal_spec.mid_z,
                "recessDepth": 135.0,
                "embedDiorama": True,
                "showFrame": False,
                "restFrameOpacity": 0.0,
                "frameRenderOrder": 6.2,
                "veilTexture": summary["scrollVeilTexture"],
                "occluder": {
                    "texture": summary["scrollVeilTexture"],
                    "worldZ": portal_spec.mid_z,
                    "renderOrder": 6.5,
                    "fullInnerCanvas": True,
                    "alphaHitThreshold": 24,
                    "innerBoxPx": list(portal_spec.inner_box_px),
                    "labelHe": "רישום הדקל והמוטות במגילה",
                    "coverageRatio": summary["veilCoverageRatio"],
                    "locksPortal": True,
                },
            }
        ],
        "verification": {
            "scrollFootEdtRimP99JumpWorld": round(foot_rim_p99, 4),
            "midEdtRimP99JumpWorld": round(mid_rim_p99, 4),
            "maxRestPoseRecompErrorU8": summary["maxRestPoseRecompErrorU8"],
            "veilCoverageRatio": summary["veilCoverageRatio"],
            "scrollFootCoverageRatio": summary["scrollFootCoverageRatio"],
        },
    }


# ============================================================================
# Top-Level Orchestrator & CLI
# ============================================================================


def build_all_workstream_e_assets(
    project_root: Path | None = None,
    config: ScrollHumptyBundleConfig | None = None,
    only_humpty_and_wandering1: bool = False,
    only_scroll_and_humpty: bool = False,
) -> dict[str, Any]:
    root = project_root or _REPO_ROOT
    cfg = config or default_scroll_humpty_bundle_config()

    roundtrip = ScrollHumptyBundleConfig.from_json(cfg.to_json())
    if roundtrip != cfg:
        raise RuntimeError("ScrollHumptyBundleConfig failed lossless JSON round-trip check.")

    device = select_torch_device()
    print(f"Initializing neural vision models on device='{device}'...")
    depth_estimator = DepthAnythingEstimator(cfg.depth_model, device=device)
    sam_extractor = SamBoxMaskExtractor(cfg.segmentation)
    matte_extractor = SalientMatteExtractor(cfg.segmentation)
    scene_lama = LamaOcclusionInpainter(cfg.scene_inpainting)

    spec_out = Path(cfg.spec_output_abspath)
    existing = (
        json.loads(spec_out.read_text(encoding="utf-8"))
        if spec_out.is_file()
        else {}
    )
    if only_humpty_and_wandering1 and existing:
        watermark_report = existing["watermarkCleaning"]
        scroll_world_spec = existing["scenes"]["scroll_palm_world"]
    elif only_scroll_and_humpty and existing:
        watermark_report = existing["watermarkCleaning"]
        scroll_world_spec = build_scroll_palm_world_scene(
            project_root=root,
            cfg=cfg,
            depth_estimator=depth_estimator,
            sam_extractor=sam_extractor,
            matte_extractor=matte_extractor,
            lama_inpainter=scene_lama,
        )
    else:
        wm_lama = LamaOcclusionInpainter(cfg.watermark_inpainting)
        watermark_report = clean_gemini_watermarks(
            project_root=root,
            cfg=cfg,
            lama_inpainter=wm_lama,
        )
        scroll_world_spec = build_scroll_palm_world_scene(
            project_root=root,
            cfg=cfg,
            depth_estimator=depth_estimator,
            sam_extractor=sam_extractor,
            matte_extractor=matte_extractor,
            lama_inpainter=scene_lama,
        )

    humpty_bundle_spec = build_humpty_dumpty_bundle_scene(
        project_root=root,
        cfg=cfg,
        depth_estimator=depth_estimator,
        sam_extractor=sam_extractor,
        matte_extractor=matte_extractor,
        lama_inpainter=scene_lama,
    )
    if only_scroll_and_humpty and "wanderingDrawing1PortalUpgrade" in existing:
        wandering1_upgrade_spec = existing["wanderingDrawing1PortalUpgrade"]
    else:
        wandering1_upgrade_spec = build_wandering_1_portal_upgrade(
            project_root=root,
            cfg=cfg,
            depth_estimator=depth_estimator,
            sam_extractor=sam_extractor,
            lama_inpainter=scene_lama,
        )

    payload = {
        "schemaVersion": "1.0",
        "workstream": "Workstream E — Scroll Palm World, Humpty Dumpty & Wandering 1 Seamless Portal",
        "generatedBy": "prototype/scroll_humpty_scenes_builder.py",
        "watermarkCleaning": watermark_report,
        "scenes": {
            scroll_world_spec["id"]: scroll_world_spec,
            humpty_bundle_spec["id"]: humpty_bundle_spec,
        },
        "wanderingDrawing1PortalUpgrade": wandering1_upgrade_spec,
    }

    spec_out.parent.mkdir(parents=True, exist_ok=True)
    spec_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] Exported Workstream E runtime specs & topology metrics -> {spec_out}")
    return payload


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Workstream E neural 2.5D spatial scenes and Wandering 1 scroll portal assets."
    )
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Validate lossless SSOT config serialization without running heavy neural inference.",
    )
    parser.add_argument(
        "--only-humpty-and-wandering1",
        action="store_true",
        help="Rebuild Scene 15 and Wandering 1 portal layers while preserving verified Scene 14 specs.",
    )
    parser.add_argument(
        "--only-scroll-and-humpty",
        action="store_true",
        help="Rebuild Scene 14 and Scene 15 while preserving verified watermark & Wandering 1 portal assets.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli_args()
    cfg_main = default_scroll_humpty_bundle_config()
    if ScrollHumptyBundleConfig.from_json(cfg_main.to_json()) != cfg_main:
        raise RuntimeError("ScrollHumptyBundleConfig failed lossless JSON round-trip check.")
    if args.dry_run_config:
        print("ScrollHumptyBundleConfig lossless JSON round-trip OK.")
    else:
        build_all_workstream_e_assets(
            config=cfg_main,
            only_humpty_and_wandering1=bool(args.only_humpty_and_wandering1),
            only_scroll_and_humpty=bool(args.only_scroll_and_humpty),
        )
