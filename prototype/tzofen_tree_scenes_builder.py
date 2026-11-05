#!/usr/bin/env python3
"""
prototype/tzofen_tree_scenes_builder.py
Offline neural 2.5D spatial asset builder and typed SSOT configuration models for Feedback #36:
  1. Scene 12 (`tzofen_brick_table` / `scene12_tzofen_*`):
     Restored unified installation of Tamar "Tami" Sheaffer's *"צופן"* (2011, `IMG_BINY_0019`) —
     white steel multi-tier table bearing 380 terracotta mud-bricks as a cohesive sculptural mesh
     (`fg`) with $C^0$ floor-contact continuity across all 7 supporting steel legs/plinths,
     $C^{-1}$ Voronoi EDT silhouette discontinuity, split wall/floor LaMa-inpainted Binyamin
     Gallery stone-tile backdrop, and an open portal window looking into `herzliya_tree_canopy`.
  2. Archived Scene 12 v2 (`tzofen_brick_table_v2` / `scene12_tzofen_v2_*`):
     Experimental object-segmented variant separating the 11 completely unoccluded perimeter/top
     terracotta mud-bricks onto `fg` via per-brick Meta SAM prompts while keeping the steel table
     and depth-interleaved bricks intact on `mid` with local neural `LaMa` + Lab-neutralized
     tabletop/shelf healing.
  3. Scene 13 (`herzliya_tree_canopy` / `scene13_herzliya_canopy_*`):
     Upward view into the sunlit green tree canopy in the Herzliya yard (`IMG_HERZ_0372`),
     decomposed into 3D foreground forked trunk & wire, mid-ground ascending boughs, and
     LaMa-healed sunlit foliage canopy.

Strict No-Fallback Guarantee: executes real neural models (`Depth Anything V2`, `Meta SAM`,
`IS-Net` via `rembg`, and `LaMa` via `simple-lama-inpainting`) inside `.venv`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import cv2
    import numpy as np
    import scipy.ndimage as ndi
    from PIL import Image
except ImportError as exc:
    raise RuntimeError(
        "Missing neural vision dependencies inside virtual environment. Run with: "
        "`/Users/talsheaffer/code/tami/.venv/bin/python -u prototype/tzofen_tree_scenes_builder.py`."
    ) from exc

from prototype.depth_topology import (
    ContactAnchorBox,
    ContactAnchoredLayerDepthConfig,
    LayerTopologyVerificationConfig,
    RoomDepthCleanConfig,
    _solve_harmonic_field_on_domain,
    clean_room_depth_under_cutouts,
    extract_contact_anchor_mask,
    solve_contact_anchored_layer_depth,
    verify_layer_contact_and_discontinuity,
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


def _pg(
    box: tuple[int, int, int, int],
    pos: list[tuple[float, float]],
    neg: list[tuple[float, float]],
) -> SamPointGroupPrompt:
    return SamPointGroupPrompt(
        box=BoundingBoxPrompt(*box),
        points_xy=tuple(pos) + tuple(neg),
        point_labels=tuple([1] * len(pos) + [0] * len(neg)),
    )


@dataclass(frozen=True)
class TzofenUnifiedCutoutConfig:
    """SSOT parameters for extracting the clean unified Tzofen installation silhouette & room plate."""

    upper_y_min_px: int
    upper_y_max_px: int
    da2_upper_offset: float
    da2_upper_span: float
    sal_upper_da2_min: float
    brick_upper_rb_offset: float
    brick_upper_rb_span: float
    brick_upper_da2_min: float
    sam_lower_dilate_px: int
    da2_lower_offset: float
    da2_lower_span: float
    wall_like_y_max_px: int
    wall_like_luma_min: float
    wall_like_rb_min: float
    wall_like_da2_max: float
    right_wall_shadow_x_min_px: int
    right_wall_shadow_rb_max: float
    floor_glare_box: BoundingBoxPrompt
    floor_glare_luma_min: float
    floor_glare_rb_max: float
    alpha_ramp_offset: float
    alpha_ramp_span: float
    alpha_blur_sigma: float
    side_border_clear_px: int
    left_vent_trim_box: BoundingBoxPrompt
    left_vent_trim_rb_max: float
    right_baseboard_trim_box: BoundingBoxPrompt
    right_baseboard_trim_luma_max: float
    min_sculp_component_px: int
    max_sculp_pinhole_px: int
    bottom_leg_reflect_dilate_px: int
    max_border_leg_x_px: int
    wall_vent_mask_box: BoundingBoxPrompt
    wall_right_corner_mask_box: BoundingBoxPrompt
    wall_pad_width_px: int
    wall_pad_height_px: int
    wall_donor_min_luma: float
    wall_donor_max_sat: float
    wall_outlet_preserve_x_min_px: int
    wall_dc_sigma_px: float
    wall_grain_sigma_px: float
    wall_dc_blend_band_px: float
    floor_vent_rim_box: BoundingBoxPrompt
    floor_leg_shadow_box: BoundingBoxPrompt
    floor_leg_shadow_luma_max: float
    floor_bottom_leg_dilate_px: int
    floor_bottom_leg_rows_px: int
    bg_occ_alpha_min_u8: int
    bg_lama_dilation_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "upper_y_min_px": self.upper_y_min_px,
            "upper_y_max_px": self.upper_y_max_px,
            "da2_upper_offset": self.da2_upper_offset,
            "da2_upper_span": self.da2_upper_span,
            "sal_upper_da2_min": self.sal_upper_da2_min,
            "brick_upper_rb_offset": self.brick_upper_rb_offset,
            "brick_upper_rb_span": self.brick_upper_rb_span,
            "brick_upper_da2_min": self.brick_upper_da2_min,
            "sam_lower_dilate_px": self.sam_lower_dilate_px,
            "da2_lower_offset": self.da2_lower_offset,
            "da2_lower_span": self.da2_lower_span,
            "wall_like_y_max_px": self.wall_like_y_max_px,
            "wall_like_luma_min": self.wall_like_luma_min,
            "wall_like_rb_min": self.wall_like_rb_min,
            "wall_like_da2_max": self.wall_like_da2_max,
            "right_wall_shadow_x_min_px": self.right_wall_shadow_x_min_px,
            "right_wall_shadow_rb_max": self.right_wall_shadow_rb_max,
            "floor_glare_box": self.floor_glare_box.to_dict(),
            "floor_glare_luma_min": self.floor_glare_luma_min,
            "floor_glare_rb_max": self.floor_glare_rb_max,
            "alpha_ramp_offset": self.alpha_ramp_offset,
            "alpha_ramp_span": self.alpha_ramp_span,
            "alpha_blur_sigma": self.alpha_blur_sigma,
            "side_border_clear_px": self.side_border_clear_px,
            "left_vent_trim_box": self.left_vent_trim_box.to_dict(),
            "left_vent_trim_rb_max": self.left_vent_trim_rb_max,
            "right_baseboard_trim_box": self.right_baseboard_trim_box.to_dict(),
            "right_baseboard_trim_luma_max": self.right_baseboard_trim_luma_max,
            "min_sculp_component_px": self.min_sculp_component_px,
            "max_sculp_pinhole_px": self.max_sculp_pinhole_px,
            "bottom_leg_reflect_dilate_px": self.bottom_leg_reflect_dilate_px,
            "max_border_leg_x_px": self.max_border_leg_x_px,
            "wall_vent_mask_box": self.wall_vent_mask_box.to_dict(),
            "wall_right_corner_mask_box": self.wall_right_corner_mask_box.to_dict(),
            "wall_pad_width_px": self.wall_pad_width_px,
            "wall_pad_height_px": self.wall_pad_height_px,
            "wall_donor_min_luma": self.wall_donor_min_luma,
            "wall_donor_max_sat": self.wall_donor_max_sat,
            "wall_outlet_preserve_x_min_px": self.wall_outlet_preserve_x_min_px,
            "wall_dc_sigma_px": self.wall_dc_sigma_px,
            "wall_grain_sigma_px": self.wall_grain_sigma_px,
            "wall_dc_blend_band_px": self.wall_dc_blend_band_px,
            "floor_vent_rim_box": self.floor_vent_rim_box.to_dict(),
            "floor_leg_shadow_box": self.floor_leg_shadow_box.to_dict(),
            "floor_leg_shadow_luma_max": self.floor_leg_shadow_luma_max,
            "floor_bottom_leg_dilate_px": self.floor_bottom_leg_dilate_px,
            "floor_bottom_leg_rows_px": self.floor_bottom_leg_rows_px,
            "bg_occ_alpha_min_u8": self.bg_occ_alpha_min_u8,
            "bg_lama_dilation_px": self.bg_lama_dilation_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TzofenUnifiedCutoutConfig:
        return cls(
            upper_y_min_px=int(data["upper_y_min_px"]),
            upper_y_max_px=int(data["upper_y_max_px"]),
            da2_upper_offset=float(data["da2_upper_offset"]),
            da2_upper_span=float(data["da2_upper_span"]),
            sal_upper_da2_min=float(data["sal_upper_da2_min"]),
            brick_upper_rb_offset=float(data["brick_upper_rb_offset"]),
            brick_upper_rb_span=float(data["brick_upper_rb_span"]),
            brick_upper_da2_min=float(data["brick_upper_da2_min"]),
            sam_lower_dilate_px=int(data["sam_lower_dilate_px"]),
            da2_lower_offset=float(data["da2_lower_offset"]),
            da2_lower_span=float(data["da2_lower_span"]),
            wall_like_y_max_px=int(data["wall_like_y_max_px"]),
            wall_like_luma_min=float(data["wall_like_luma_min"]),
            wall_like_rb_min=float(data["wall_like_rb_min"]),
            wall_like_da2_max=float(data["wall_like_da2_max"]),
            right_wall_shadow_x_min_px=int(data["right_wall_shadow_x_min_px"]),
            right_wall_shadow_rb_max=float(data["right_wall_shadow_rb_max"]),
            floor_glare_box=BoundingBoxPrompt.from_dict(data["floor_glare_box"]),
            floor_glare_luma_min=float(data["floor_glare_luma_min"]),
            floor_glare_rb_max=float(data["floor_glare_rb_max"]),
            alpha_ramp_offset=float(data["alpha_ramp_offset"]),
            alpha_ramp_span=float(data["alpha_ramp_span"]),
            alpha_blur_sigma=float(data["alpha_blur_sigma"]),
            side_border_clear_px=int(data["side_border_clear_px"]),
            left_vent_trim_box=BoundingBoxPrompt.from_dict(data["left_vent_trim_box"]),
            left_vent_trim_rb_max=float(data["left_vent_trim_rb_max"]),
            right_baseboard_trim_box=BoundingBoxPrompt.from_dict(data["right_baseboard_trim_box"]),
            right_baseboard_trim_luma_max=float(data["right_baseboard_trim_luma_max"]),
            min_sculp_component_px=int(data["min_sculp_component_px"]),
            max_sculp_pinhole_px=int(data["max_sculp_pinhole_px"]),
            bottom_leg_reflect_dilate_px=int(data["bottom_leg_reflect_dilate_px"]),
            max_border_leg_x_px=int(data["max_border_leg_x_px"]),
            wall_vent_mask_box=BoundingBoxPrompt.from_dict(data["wall_vent_mask_box"]),
            wall_right_corner_mask_box=BoundingBoxPrompt.from_dict(
                data["wall_right_corner_mask_box"]
            ),
            wall_pad_width_px=int(data["wall_pad_width_px"]),
            wall_pad_height_px=int(data["wall_pad_height_px"]),
            wall_donor_min_luma=float(data["wall_donor_min_luma"]),
            wall_donor_max_sat=float(data["wall_donor_max_sat"]),
            wall_outlet_preserve_x_min_px=int(data["wall_outlet_preserve_x_min_px"]),
            wall_dc_sigma_px=float(data["wall_dc_sigma_px"]),
            wall_grain_sigma_px=float(data["wall_grain_sigma_px"]),
            wall_dc_blend_band_px=float(data["wall_dc_blend_band_px"]),
            floor_vent_rim_box=BoundingBoxPrompt.from_dict(data["floor_vent_rim_box"]),
            floor_leg_shadow_box=BoundingBoxPrompt.from_dict(data["floor_leg_shadow_box"]),
            floor_leg_shadow_luma_max=float(data["floor_leg_shadow_luma_max"]),
            floor_bottom_leg_dilate_px=int(data["floor_bottom_leg_dilate_px"]),
            floor_bottom_leg_rows_px=int(data["floor_bottom_leg_rows_px"]),
            bg_occ_alpha_min_u8=int(data["bg_occ_alpha_min_u8"]),
            bg_lama_dilation_px=int(data["bg_lama_dilation_px"]),
        )


@dataclass(frozen=True)
class TzofenSceneBuildConfig:
    """SSOT build spec for the restored unified Scene 12 (`tzofen_brick_table`)."""

    scene_id: str
    asset_prefix: str
    source_image_relpath: str
    inner_dims: CanvasDimensionsConfig
    expanded_dims: CanvasDimensionsConfig
    scaled_height_px: int
    inner_y_offset_px: int
    baseboard_inner_y: int
    jpeg_quality: int
    bg_z: float
    mid_z: float
    fg_z: float
    room_depth_amplitude: float
    fg_depth_amplitude: float
    room_pin_margin_uv: float
    window_portal_inner_box_px: tuple[float, float, float, float]
    window_frame_sample_box_px: tuple[int, int, int, int]
    portal_z_clearance_behind_frame: float
    portal_recess_depth: float
    sam_prompts: tuple[SamPointGroupPrompt, ...]
    bg_seed_boxes: tuple[BoundingBoxPrompt, ...]
    unified_cutout: TzofenUnifiedCutoutConfig
    room_depth_clean: RoomDepthCleanConfig
    fg_contact_solve: ContactAnchoredLayerDepthConfig
    topology_verification: LayerTopologyVerificationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "asset_prefix": self.asset_prefix,
            "source_image_relpath": self.source_image_relpath,
            "inner_dims": self.inner_dims.to_dict(),
            "expanded_dims": self.expanded_dims.to_dict(),
            "scaled_height_px": self.scaled_height_px,
            "inner_y_offset_px": self.inner_y_offset_px,
            "baseboard_inner_y": self.baseboard_inner_y,
            "jpeg_quality": self.jpeg_quality,
            "bg_z": self.bg_z,
            "mid_z": self.mid_z,
            "fg_z": self.fg_z,
            "room_depth_amplitude": self.room_depth_amplitude,
            "fg_depth_amplitude": self.fg_depth_amplitude,
            "room_pin_margin_uv": self.room_pin_margin_uv,
            "window_portal_inner_box_px": list(self.window_portal_inner_box_px),
            "window_frame_sample_box_px": list(self.window_frame_sample_box_px),
            "portal_z_clearance_behind_frame": self.portal_z_clearance_behind_frame,
            "portal_recess_depth": self.portal_recess_depth,
            "sam_prompts": [p.to_dict() for p in self.sam_prompts],
            "bg_seed_boxes": [b.to_dict() for b in self.bg_seed_boxes],
            "unified_cutout": self.unified_cutout.to_dict(),
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "fg_contact_solve": self.fg_contact_solve.to_dict(),
            "topology_verification": self.topology_verification.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TzofenSceneBuildConfig:
        wp = data["window_portal_inner_box_px"]
        wf = data["window_frame_sample_box_px"]
        return cls(
            scene_id=str(data["scene_id"]),
            asset_prefix=str(data["asset_prefix"]),
            source_image_relpath=str(data["source_image_relpath"]),
            inner_dims=CanvasDimensionsConfig.from_dict(data["inner_dims"]),
            expanded_dims=CanvasDimensionsConfig.from_dict(data["expanded_dims"]),
            scaled_height_px=int(data["scaled_height_px"]),
            inner_y_offset_px=int(data["inner_y_offset_px"]),
            baseboard_inner_y=int(data["baseboard_inner_y"]),
            jpeg_quality=int(data["jpeg_quality"]),
            bg_z=float(data["bg_z"]),
            mid_z=float(data["mid_z"]),
            fg_z=float(data["fg_z"]),
            room_depth_amplitude=float(data["room_depth_amplitude"]),
            fg_depth_amplitude=float(data["fg_depth_amplitude"]),
            room_pin_margin_uv=float(data["room_pin_margin_uv"]),
            window_portal_inner_box_px=(float(wp[0]), float(wp[1]), float(wp[2]), float(wp[3])),
            window_frame_sample_box_px=(int(wf[0]), int(wf[1]), int(wf[2]), int(wf[3])),
            portal_z_clearance_behind_frame=float(data["portal_z_clearance_behind_frame"]),
            portal_recess_depth=float(data["portal_recess_depth"]),
            sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["sam_prompts"]),
            bg_seed_boxes=tuple(BoundingBoxPrompt.from_dict(b) for b in data["bg_seed_boxes"]),
            unified_cutout=TzofenUnifiedCutoutConfig.from_dict(data["unified_cutout"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            fg_contact_solve=ContactAnchoredLayerDepthConfig.from_dict(data["fg_contact_solve"]),
            topology_verification=LayerTopologyVerificationConfig.from_dict(
                data["topology_verification"]
            ),
        )


@dataclass(frozen=True)
class TzofenSegmentedV2BuildConfig:
    """SSOT build spec for the archived object-segmented variant (`tzofen_brick_table_v2`)."""

    scene_id: str
    asset_prefix: str
    bg_z: float
    mid_z: float
    fg_z: float
    room_depth_amplitude: float
    mid_depth_amplitude: float
    fg_depth_amplitude: float
    room_pin_margin_uv: float
    fg_brick_prompts: tuple[SamPointGroupPrompt, ...]
    prompt_box_pad_px: int
    sam_pinhole_fill_px: int
    sam_min_component_px: int
    fg_sdf_erode_bias_px: float
    fg_sdf_band_px: float
    mid_open_air_boxes: tuple[BoundingBoxPrompt, ...]
    mid_air_zone_pad_px: int
    mid_sdf_band_px: float
    brick_inpaint_dilate_px: int
    brick_lama_dilation_px: int
    brick_inpaint_feather_px: float
    neutral_steel_boxes: tuple[BoundingBoxPrompt, ...]
    neutral_steel_min_luma: float
    neutral_steel_max_sat: float
    brick_height_relief_weight: float
    brick_height_saturation_px: float
    mid_contact_solve: ContactAnchoredLayerDepthConfig
    fg_contact_solve: ContactAnchoredLayerDepthConfig
    topology_verification: LayerTopologyVerificationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "asset_prefix": self.asset_prefix,
            "bg_z": self.bg_z,
            "mid_z": self.mid_z,
            "fg_z": self.fg_z,
            "room_depth_amplitude": self.room_depth_amplitude,
            "mid_depth_amplitude": self.mid_depth_amplitude,
            "fg_depth_amplitude": self.fg_depth_amplitude,
            "room_pin_margin_uv": self.room_pin_margin_uv,
            "fg_brick_prompts": [p.to_dict() for p in self.fg_brick_prompts],
            "prompt_box_pad_px": self.prompt_box_pad_px,
            "sam_pinhole_fill_px": self.sam_pinhole_fill_px,
            "sam_min_component_px": self.sam_min_component_px,
            "fg_sdf_erode_bias_px": self.fg_sdf_erode_bias_px,
            "fg_sdf_band_px": self.fg_sdf_band_px,
            "mid_open_air_boxes": [b.to_dict() for b in self.mid_open_air_boxes],
            "mid_air_zone_pad_px": self.mid_air_zone_pad_px,
            "mid_sdf_band_px": self.mid_sdf_band_px,
            "brick_inpaint_dilate_px": self.brick_inpaint_dilate_px,
            "brick_lama_dilation_px": self.brick_lama_dilation_px,
            "brick_inpaint_feather_px": self.brick_inpaint_feather_px,
            "neutral_steel_boxes": [b.to_dict() for b in self.neutral_steel_boxes],
            "neutral_steel_min_luma": self.neutral_steel_min_luma,
            "neutral_steel_max_sat": self.neutral_steel_max_sat,
            "brick_height_relief_weight": self.brick_height_relief_weight,
            "brick_height_saturation_px": self.brick_height_saturation_px,
            "mid_contact_solve": self.mid_contact_solve.to_dict(),
            "fg_contact_solve": self.fg_contact_solve.to_dict(),
            "topology_verification": self.topology_verification.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TzofenSegmentedV2BuildConfig:
        return cls(
            scene_id=str(data["scene_id"]),
            asset_prefix=str(data["asset_prefix"]),
            bg_z=float(data["bg_z"]),
            mid_z=float(data["mid_z"]),
            fg_z=float(data["fg_z"]),
            room_depth_amplitude=float(data["room_depth_amplitude"]),
            mid_depth_amplitude=float(data["mid_depth_amplitude"]),
            fg_depth_amplitude=float(data["fg_depth_amplitude"]),
            room_pin_margin_uv=float(data["room_pin_margin_uv"]),
            fg_brick_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["fg_brick_prompts"]
            ),
            prompt_box_pad_px=int(data["prompt_box_pad_px"]),
            sam_pinhole_fill_px=int(data["sam_pinhole_fill_px"]),
            sam_min_component_px=int(data["sam_min_component_px"]),
            fg_sdf_erode_bias_px=float(data["fg_sdf_erode_bias_px"]),
            fg_sdf_band_px=float(data["fg_sdf_band_px"]),
            mid_open_air_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["mid_open_air_boxes"]
            ),
            mid_air_zone_pad_px=int(data["mid_air_zone_pad_px"]),
            mid_sdf_band_px=float(data["mid_sdf_band_px"]),
            brick_inpaint_dilate_px=int(data["brick_inpaint_dilate_px"]),
            brick_lama_dilation_px=int(data["brick_lama_dilation_px"]),
            brick_inpaint_feather_px=float(data["brick_inpaint_feather_px"]),
            neutral_steel_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["neutral_steel_boxes"]
            ),
            neutral_steel_min_luma=float(data["neutral_steel_min_luma"]),
            neutral_steel_max_sat=float(data["neutral_steel_max_sat"]),
            brick_height_relief_weight=float(data["brick_height_relief_weight"]),
            brick_height_saturation_px=float(data["brick_height_saturation_px"]),
            mid_contact_solve=ContactAnchoredLayerDepthConfig.from_dict(
                data["mid_contact_solve"]
            ),
            fg_contact_solve=ContactAnchoredLayerDepthConfig.from_dict(data["fg_contact_solve"]),
            topology_verification=LayerTopologyVerificationConfig.from_dict(
                data["topology_verification"]
            ),
        )


@dataclass(frozen=True)
class TreeCanopySceneBuildConfig:
    scene_id: str
    source_image_relpath: str
    inner_dims: CanvasDimensionsConfig
    expanded_dims: CanvasDimensionsConfig
    scaled_source_width_px: int
    crop_left_px: int
    jpeg_quality: int
    bg_z: float
    mid_z: float
    fg_z: float
    room_depth_amplitude: float
    mid_depth_amplitude: float
    fg_depth_amplitude: float
    room_pin_margin_uv: float
    fg_sam_prompts: tuple[SamPointGroupPrompt, ...]
    mid_sam_prompts: tuple[SamPointGroupPrompt, ...]
    room_depth_clean: RoomDepthCleanConfig
    mid_contact_solve: ContactAnchoredLayerDepthConfig
    fg_contact_solve: ContactAnchoredLayerDepthConfig
    topology_verification: LayerTopologyVerificationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "source_image_relpath": self.source_image_relpath,
            "inner_dims": self.inner_dims.to_dict(),
            "expanded_dims": self.expanded_dims.to_dict(),
            "scaled_source_width_px": self.scaled_source_width_px,
            "crop_left_px": self.crop_left_px,
            "jpeg_quality": self.jpeg_quality,
            "bg_z": self.bg_z,
            "mid_z": self.mid_z,
            "fg_z": self.fg_z,
            "room_depth_amplitude": self.room_depth_amplitude,
            "mid_depth_amplitude": self.mid_depth_amplitude,
            "fg_depth_amplitude": self.fg_depth_amplitude,
            "room_pin_margin_uv": self.room_pin_margin_uv,
            "fg_sam_prompts": [p.to_dict() for p in self.fg_sam_prompts],
            "mid_sam_prompts": [p.to_dict() for p in self.mid_sam_prompts],
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_contact_solve": self.mid_contact_solve.to_dict(),
            "fg_contact_solve": self.fg_contact_solve.to_dict(),
            "topology_verification": self.topology_verification.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeCanopySceneBuildConfig:
        return cls(
            scene_id=str(data["scene_id"]),
            source_image_relpath=str(data["source_image_relpath"]),
            inner_dims=CanvasDimensionsConfig.from_dict(data["inner_dims"]),
            expanded_dims=CanvasDimensionsConfig.from_dict(data["expanded_dims"]),
            scaled_source_width_px=int(data["scaled_source_width_px"]),
            crop_left_px=int(data["crop_left_px"]),
            jpeg_quality=int(data["jpeg_quality"]),
            bg_z=float(data["bg_z"]),
            mid_z=float(data["mid_z"]),
            fg_z=float(data["fg_z"]),
            room_depth_amplitude=float(data["room_depth_amplitude"]),
            mid_depth_amplitude=float(data["mid_depth_amplitude"]),
            fg_depth_amplitude=float(data["fg_depth_amplitude"]),
            room_pin_margin_uv=float(data["room_pin_margin_uv"]),
            fg_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["fg_sam_prompts"]),
            mid_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["mid_sam_prompts"]
            ),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            mid_contact_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_contact_solve"]),
            fg_contact_solve=ContactAnchoredLayerDepthConfig.from_dict(data["fg_contact_solve"]),
            topology_verification=LayerTopologyVerificationConfig.from_dict(
                data["topology_verification"]
            ),
        )


@dataclass(frozen=True)
class TzofenTreeBundleConfig:
    depth_model: DepthModelConfig
    segmentation: SegmentationConfig
    inpainting: InpaintingConfig
    tzofen: TzofenSceneBuildConfig
    tzofen_v2: TzofenSegmentedV2BuildConfig
    tree_canopy: TreeCanopySceneBuildConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_model": self.depth_model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "inpainting": self.inpainting.to_dict(),
            "tzofen": self.tzofen.to_dict(),
            "tzofen_v2": self.tzofen_v2.to_dict(),
            "tree_canopy": self.tree_canopy.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TzofenTreeBundleConfig:
        return cls(
            depth_model=DepthModelConfig.from_dict(data["depth_model"]),
            segmentation=SegmentationConfig.from_dict(data["segmentation"]),
            inpainting=InpaintingConfig.from_dict(data["inpainting"]),
            tzofen=TzofenSceneBuildConfig.from_dict(data["tzofen"]),
            tzofen_v2=TzofenSegmentedV2BuildConfig.from_dict(data["tzofen_v2"]),
            tree_canopy=TreeCanopySceneBuildConfig.from_dict(data["tree_canopy"]),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, raw_json: str) -> TzofenTreeBundleConfig:
        return cls.from_dict(json.loads(raw_json))


def default_tzofen_tree_bundle_config() -> TzofenTreeBundleConfig:
    inner = CanvasDimensionsConfig(width=1500, height=1000)
    expanded = CanvasDimensionsConfig(width=2700, height=1800)
    verify_cfg = LayerTopologyVerificationConfig(
        alpha_threshold_u8=20,
        rim_erosion_px=2,
        rim_exterior_probe_px=6,
        max_allowed_contact_dz_world=2.5,
        max_allowed_p95_contact_dz_world=1.5,
        max_allowed_rim_jump_world=2.0,
        host_monotonicity_tolerance_world=2.5,
    )

    tzofen_floor_contacts = (
        ContactAnchorBox(x0=360, y0=616, x1=380, y1=636, mode="bottom_band", band_px=6),
        ContactAnchorBox(x0=1032, y0=624, x1=1054, y1=644, mode="bottom_band", band_px=6),
        ContactAnchorBox(x0=200, y0=955, x1=236, y1=999, mode="all_opaque", band_px=6),
        ContactAnchorBox(x0=430, y0=955, x1=512, y1=999, mode="all_opaque", band_px=6),
        ContactAnchorBox(x0=624, y0=938, x1=650, y1=961, mode="bottom_band", band_px=6),
        ContactAnchorBox(x0=850, y0=934, x1=876, y1=956, mode="bottom_band", band_px=6),
        ContactAnchorBox(x0=1242, y0=928, x1=1270, y1=950, mode="bottom_band", band_px=6),
    )

    tzofen_sam_prompts = (
        _pg(
            (240, 130, 355, 305),
            [(265, 165), (320, 165), (295, 225), (344, 210)],
            [(220, 180), (362, 180)],
        ),
        _pg(
            (670, 150, 800, 445),
            [(735, 175), (710, 260), (710, 350), (710, 415)],
            [(655, 240), (785, 240)],
        ),
        _pg(
            (960, 136, 1142, 255),
            [
                (974, 155),
                (974, 195),
                (972, 232),
                (1015, 147),
                (1065, 148),
                (1115, 149),
                (1126, 168),
                (1125, 186),
            ],
            [
                (1010, 185),
                (1055, 185),
                (1095, 185),
                (1000, 225),
                (1095, 230),
                (945, 185),
                (1155, 165),
                (1050, 125),
            ],
        ),
        _pg(
            (10, 405, 1490, 502),
            [
                (25, 485),
                (260, 465),
                (475, 455),
                (725, 488),
                (1100, 488),
                (1320, 470),
                (1465, 486),
            ],
            [(8, 455), (1484, 488), (1492, 465), (100, 515), (1320, 520)],
        ),
        _pg(
            (170, 460, 650, 962),
            [
                (188, 595),
                (193, 665),
                (265, 580),
                (205, 760),
                (210, 935),
                (330, 840),
                (468, 860),
                (468, 945),
                (565, 840),
                (590, 725),
                (636, 720),
                (636, 935),
                (490, 680),
            ],
            [
                (150, 750),
                (310, 770),
                (310, 910),
                (565, 910),
                (565, 795),
                (410, 660),
                (465, 530),
                (575, 528),
            ],
        ),
        _pg(
            (562, 495, 898, 955),
            [
                (636, 560),
                (668, 565),
                (680, 580),
                (740, 575),
                (795, 565),
                (815, 580),
                (840, 525),
                (863, 560),
                (863, 750),
                (863, 930),
                (600, 615),
                (600, 735),
                (646, 725),
                (735, 530),
                (795, 625),
                (838, 625),
                (815, 670),
            ],
            [
                (575, 528),
                (672, 540),
                (813, 545),
                (735, 755),
                (745, 850),
                (595, 805),
                (595, 900),
            ],
        ),
        _pg(
            (852, 495, 1290, 950),
            [
                (896, 530),
                (863, 845),
                (900, 845),
                (945, 645),
                (960, 730),
                (1050, 780),
                (1060, 880),
                (1215, 875),
                (1254, 925),
                (1262, 740),
                (1272, 560),
                (1126, 755),
                (1195, 775),
            ],
            [(950, 532), (1140, 610), (920, 920), (1310, 750), (1080, 950)],
        ),
        _pg(
            (358, 495, 550, 636),
            [(369, 555), (369, 600), (369, 626), (420, 528), (520, 535)],
            [(348, 580), (430, 585), (395, 625), (465, 530)],
        ),
        _pg(
            (885, 495, 1055, 644),
            [(1042, 540), (1042, 605), (1042, 634), (975, 565), (920, 530)],
            [(950, 532), (985, 525), (985, 610), (1065, 580)],
        ),
        _pg(
            (1350, 492, 1472, 602),
            [(1365, 530), (1380, 545), (1440, 545), (1452, 545)],
            [(1330, 550), (1485, 550), (1410, 620)],
        ),
    )

    tzofen_cfg = TzofenSceneBuildConfig(
        scene_id="tzofen_brick_table",
        asset_prefix="scene12_tzofen",
        source_image_relpath="media/binyamin_gallery/binyamin_צופן.jpg_afbc9c55.jpg",
        inner_dims=inner,
        expanded_dims=expanded,
        scaled_height_px=919,
        inner_y_offset_px=42,
        baseboard_inner_y=484,
        jpeg_quality=93,
        bg_z=-350.0,
        mid_z=-260.0,
        fg_z=-350.0,
        room_depth_amplitude=128.0,
        fg_depth_amplitude=175.0,
        room_pin_margin_uv=0.045,
        # Aperture spans the plaster opening plus a 2-4 px tuck under the frame posts, top bar and
        # plank, which stay on `fg` and hide the portal's edges.
        window_portal_inner_box_px=(978.0, 148.0, 1126.0, 254.0),
        window_frame_sample_box_px=(967, 139, 1138, 248),
        portal_z_clearance_behind_frame=2.5,
        portal_recess_depth=115.0,
        sam_prompts=tzofen_sam_prompts,
        bg_seed_boxes=(
            BoundingBoxPrompt(0, 0, 1500, 132),
            BoundingBoxPrompt(0, 42, 230, 340),
            BoundingBoxPrompt(1200, 42, 1500, 340),
            BoundingBoxPrompt(0, 520, 170, 960),
            BoundingBoxPrompt(1290, 610, 1500, 960),
            BoundingBoxPrompt(680, 680, 840, 950),
            BoundingBoxPrompt(1060, 520, 1230, 760),
            BoundingBoxPrompt(230, 860, 425, 955),
            BoundingBoxPrompt(510, 860, 620, 950),
            BoundingBoxPrompt(0, 962, 1500, 1000),
        ),
        unified_cutout=TzofenUnifiedCutoutConfig(
            upper_y_min_px=130,
            upper_y_max_px=500,
            da2_upper_offset=0.055,
            da2_upper_span=0.035,
            sal_upper_da2_min=0.035,
            brick_upper_rb_offset=42.0,
            brick_upper_rb_span=18.0,
            brick_upper_da2_min=0.025,
            sam_lower_dilate_px=7,
            da2_lower_offset=0.075,
            da2_lower_span=0.035,
            wall_like_y_max_px=476,
            wall_like_luma_min=233.0,
            wall_like_rb_min=3.5,
            wall_like_da2_max=0.065,
            right_wall_shadow_x_min_px=1422,
            right_wall_shadow_rb_max=28.0,
            floor_glare_box=BoundingBoxPrompt(1160, 502, 1236, 566),
            floor_glare_luma_min=205.0,
            floor_glare_rb_max=32.0,
            alpha_ramp_offset=0.42,
            alpha_ramp_span=0.22,
            alpha_blur_sigma=0.55,
            side_border_clear_px=8,
            left_vent_trim_box=BoundingBoxPrompt(0, 0, 24, 465),
            left_vent_trim_rb_max=28.0,
            right_baseboard_trim_box=BoundingBoxPrompt(1474, 472, 1500, 502),
            right_baseboard_trim_luma_max=205.0,
            min_sculp_component_px=48,
            max_sculp_pinhole_px=260,
            bottom_leg_reflect_dilate_px=13,
            max_border_leg_x_px=520,
            wall_vent_mask_box=BoundingBoxPrompt(0, 290, 230, 484),
            wall_right_corner_mask_box=BoundingBoxPrompt(1465, 468, 1500, 484),
            wall_pad_width_px=1504,
            wall_pad_height_px=480,
            wall_donor_min_luma=225.0,
            wall_donor_max_sat=45.0,
            wall_outlet_preserve_x_min_px=1420,
            wall_dc_sigma_px=18.0,
            wall_grain_sigma_px=2.5,
            wall_dc_blend_band_px=12.0,
            floor_vent_rim_box=BoundingBoxPrompt(0, 484, 45, 508),
            floor_leg_shadow_box=BoundingBoxPrompt(755, 590, 855, 845),
            floor_leg_shadow_luma_max=108.0,
            floor_bottom_leg_dilate_px=35,
            floor_bottom_leg_rows_px=10,
            bg_occ_alpha_min_u8=6,
            bg_lama_dilation_px=26,
        ),
        room_depth_clean=RoomDepthCleanConfig(
            alpha_threshold_u8=16,
            mask_dilation_px=28,
            post_smooth_sigma_px=4.0,
            interior_blend_band_px=14.0,
        ),
        fg_contact_solve=ContactAnchoredLayerDepthConfig(
            bg_z=-350.0,
            room_depth_amplitude=128.0,
            room_pin_margin_uv=0.045,
            layer_z=-350.0,
            layer_depth_amplitude=175.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=22,
            min_component_area_px=25,
            auto_floor_band_px=6,
            auto_floor_max_rise_px=48,
            contact_boxes=tzofen_floor_contacts,
            neural_relief_amplitude=34.0,
            contact_relief_taper_px=12.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.0,
            interior_smooth_sigma_px=2.2,
            overflow_tolerance_world=0.5,
        ),
        topology_verification=verify_cfg,
    )

    tzofen_v2_cfg = TzofenSegmentedV2BuildConfig(
        scene_id="tzofen_brick_table_v2",
        asset_prefix="scene12_tzofen_v2",
        bg_z=-340.0,
        mid_z=-340.0,
        fg_z=-340.0,
        room_depth_amplitude=155.0,
        mid_depth_amplitude=200.0,
        fg_depth_amplitude=230.0,
        room_pin_margin_uv=0.045,
        fg_brick_prompts=(
            _pg(
                (16, 378, 164, 480),
                [(55, 425), (125, 425), (85, 392), (85, 468)],
                [(10, 425), (174, 425), (85, 370), (85, 488)],
            ),
            _pg(
                (246, 386, 380, 468),
                [(285, 425), (345, 425), (315, 396), (315, 456)],
                [(236, 425), (390, 425), (315, 378), (315, 476)],
            ),
            _pg(
                (448, 410, 569, 486),
                [(485, 448), (540, 448), (510, 420), (510, 476)],
                [(438, 448), (510, 402), (510, 492)],
            ),
            _pg(
                (566, 395, 686, 486),
                [(600, 440), (655, 440), (625, 406), (625, 476)],
                [(696, 448), (625, 386), (625, 492)],
            ),
            _pg(
                (732, 412, 855, 488),
                [(765, 450), (825, 450), (795, 422), (795, 478)],
                [(722, 455), (795, 404), (795, 493)],
            ),
            _pg(
                (852, 412, 970, 488),
                [(885, 450), (945, 450), (912, 422), (912, 478)],
                [(912, 404), (912, 493)],
            ),
            _pg(
                (960, 394, 1084, 488),
                [(995, 440), (1055, 440), (1025, 406), (1025, 478)],
                [(1094, 445), (1025, 386), (1025, 493)],
            ),
            _pg(
                (1284, 384, 1425, 479),
                [(1325, 432), (1395, 432), (1360, 396), (1360, 468)],
                [(1274, 435), (1435, 435), (1360, 376), (1360, 486)],
            ),
            _pg(
                (700, 148, 800, 203),
                [(730, 175), (775, 175)],
                [(690, 175), (810, 175), (750, 140), (750, 212)],
            ),
            _pg(
                (910, 586, 1007, 715),
                [(945, 625), (975, 680), (955, 600), (955, 702)],
                [(898, 650), (1018, 650), (955, 576), (955, 724)],
            ),
            _pg(
                (1168, 796, 1269, 923),
                [(1205, 840), (1235, 895), (1218, 810), (1218, 910)],
                [(1156, 860), (1280, 860), (1218, 786), (1218, 932)],
            ),
        ),
        prompt_box_pad_px=2,
        sam_pinhole_fill_px=250,
        sam_min_component_px=150,
        fg_sdf_erode_bias_px=0.25,
        fg_sdf_band_px=1.1,
        mid_open_air_boxes=(
            BoundingBoxPrompt(12, 375, 156, 438),
            BoundingBoxPrompt(698, 145, 802, 199),
            BoundingBoxPrompt(1358, 380, 1475, 438),
            BoundingBoxPrompt(908, 584, 1012, 658),
            BoundingBoxPrompt(1166, 792, 1244, 822),
            BoundingBoxPrompt(1262, 822, 1274, 924),
        ),
        mid_air_zone_pad_px=4,
        mid_sdf_band_px=1.2,
        brick_inpaint_dilate_px=8,
        brick_lama_dilation_px=8,
        brick_inpaint_feather_px=4.0,
        neutral_steel_boxes=(
            BoundingBoxPrompt(10, 428, 220, 498),
            BoundingBoxPrompt(435, 425, 1100, 498),
            BoundingBoxPrompt(1275, 428, 1475, 498),
            BoundingBoxPrompt(900, 652, 1035, 748),
            BoundingBoxPrompt(1150, 818, 1272, 945),
        ),
        neutral_steel_min_luma=135.0,
        neutral_steel_max_sat=48.0,
        brick_height_relief_weight=0.58,
        brick_height_saturation_px=95.0,
        mid_contact_solve=ContactAnchoredLayerDepthConfig(
            bg_z=-340.0,
            room_depth_amplitude=155.0,
            room_pin_margin_uv=0.045,
            layer_z=-340.0,
            layer_depth_amplitude=200.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=24,
            min_component_area_px=25,
            auto_floor_band_px=6,
            auto_floor_max_rise_px=48,
            contact_boxes=tzofen_floor_contacts,
            neural_relief_amplitude=24.0,
            contact_relief_taper_px=12.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.0,
            interior_smooth_sigma_px=3.0,
            overflow_tolerance_world=0.5,
        ),
        fg_contact_solve=ContactAnchoredLayerDepthConfig(
            bg_z=-340.0,
            room_depth_amplitude=200.0,
            room_pin_margin_uv=0.0,
            layer_z=-340.0,
            layer_depth_amplitude=230.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=3,
            min_component_area_px=30,
            auto_floor_band_px=5,
            auto_floor_max_rise_px=150,
            contact_boxes=(),
            neural_relief_amplitude=32.0,
            contact_relief_taper_px=2.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.0,
            interior_smooth_sigma_px=1.6,
            overflow_tolerance_world=0.5,
        ),
        topology_verification=verify_cfg,
    )

    tree_cfg = TreeCanopySceneBuildConfig(
        scene_id="herzliya_tree_canopy",
        source_image_relpath="media/herzliya_fried_house/fried2_pict0038_51ebb2cc.jpg",
        inner_dims=inner,
        expanded_dims=expanded,
        scaled_source_width_px=1508,
        crop_left_px=4,
        jpeg_quality=93,
        bg_z=-360.0,
        mid_z=-360.0,
        fg_z=-360.0,
        room_depth_amplitude=112.0,
        mid_depth_amplitude=145.0,
        fg_depth_amplitude=165.0,
        room_pin_margin_uv=0.045,
        fg_sam_prompts=(
            _pg(
                (195, 280, 910, 1000),
                [
                    (255, 360),
                    (295, 520),
                    (335, 680),
                    (410, 860),
                    (465, 965),
                    (565, 965),
                    (665, 945),
                    (755, 830),
                    (835, 735),
                ],
                [
                    (150, 450),
                    (185, 660),
                    (275, 880),
                    (455, 680),
                    (635, 835),
                    (875, 885),
                    (710, 680),
                ],
            ),
            _pg(
                (485, 405, 720, 855),
                [(685, 430), (645, 515), (605, 600), (560, 710), (515, 810)],
                [(590, 450), (705, 540), (505, 650), (645, 735)],
            ),
            _pg(
                (265, 0, 690, 395),
                [
                    (335, 330),
                    (415, 240),
                    (505, 145),
                    (595, 55),
                    (645, 12),
                    (445, 250),
                    (535, 165),
                    (625, 65),
                ],
                [(355, 160), (495, 60), (545, 255), (640, 180), (295, 210)],
            ),
            _pg(
                (0, 25, 255, 365),
                [(20, 75), (85, 175), (145, 265), (195, 345)],
                [(135, 120), (40, 230), (195, 210)],
            ),
        ),
        mid_sam_prompts=(
            _pg(
                (805, 0, 925, 715),
                [(848, 660), (855, 540), (860, 420), (868, 280), (862, 150), (845, 55)],
                [(812, 420), (905, 420), (812, 230), (905, 230)],
            ),
            _pg(
                (870, 0, 1215, 715),
                [(895, 680), (965, 570), (1035, 430), (1085, 310), (1128, 185), (1175, 55)],
                [(945, 430), (1120, 430), (1020, 210), (1175, 220)],
            ),
            _pg(
                (885, 455, 1500, 780),
                [
                    (995, 615),
                    (1110, 535),
                    (1210, 480),
                    (935, 735),
                    (1095, 705),
                    (1265, 708),
                    (1430, 720),
                ],
                [(1045, 650), (1220, 600), (1080, 760), (1320, 645)],
            ),
        ),
        room_depth_clean=RoomDepthCleanConfig(
            alpha_threshold_u8=16,
            mask_dilation_px=22,
            post_smooth_sigma_px=4.5,
            interior_blend_band_px=14.0,
        ),
        mid_contact_solve=ContactAnchoredLayerDepthConfig(
            bg_z=-360.0,
            room_depth_amplitude=112.0,
            room_pin_margin_uv=0.045,
            layer_z=-360.0,
            layer_depth_amplitude=145.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=34,
            min_component_area_px=25,
            auto_floor_band_px=8,
            auto_floor_max_rise_px=64,
            contact_boxes=(
                ContactAnchorBox(x0=1120, y0=0, x1=1208, y1=85, mode="top_band", band_px=10),
                ContactAnchorBox(x0=805, y0=350, x1=885, y1=405, mode="top_band", band_px=10),
                ContactAnchorBox(x0=850, y0=690, x1=942, y1=742, mode="bottom_band", band_px=10),
            ),
            neural_relief_amplitude=24.0,
            contact_relief_taper_px=16.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.0,
            interior_smooth_sigma_px=2.5,
            overflow_tolerance_world=0.5,
        ),
        fg_contact_solve=ContactAnchoredLayerDepthConfig(
            bg_z=-360.0,
            room_depth_amplitude=112.0,
            room_pin_margin_uv=0.045,
            layer_z=-360.0,
            layer_depth_amplitude=165.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=28,
            min_component_area_px=25,
            auto_floor_band_px=8,
            auto_floor_max_rise_px=64,
            contact_boxes=(
                ContactAnchorBox(x0=560, y0=0, x1=685, y1=80, mode="top_band", band_px=10),
                ContactAnchorBox(x0=0, y0=30, x1=90, y1=155, mode="left_band", band_px=10),
                ContactAnchorBox(x0=780, y0=690, x1=895, y1=780, mode="top_band", band_px=10),
            ),
            neural_relief_amplitude=42.0,
            contact_relief_taper_px=18.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.0,
            interior_smooth_sigma_px=2.5,
            overflow_tolerance_world=0.5,
        ),
        topology_verification=verify_cfg,
    )

    return TzofenTreeBundleConfig(
        depth_model=DepthModelConfig(
            model_id="depth-anything/Depth-Anything-V2-Small-hf",
            background_gaussian_sigma=16.0,
            depth_discontinuity_percentile_fg=62.0,
            depth_discontinuity_percentile_mid=30.0,
        ),
        segmentation=SegmentationConfig(
            rembg_model_name="isnet-general-use",
            sam_model_id="facebook/sam-vit-base",
            alpha_foreground_threshold=92,
            alpha_full_opaque_threshold=195,
            min_component_area_px=90,
            sam_morph_close_px=3,
            edge_feather_sigma=0.65,
        ),
        inpainting=InpaintingConfig(
            dilation_radius_px=26,
            mask_blur_sigma=3.0,
            jpeg_quality=93,
        ),
        tzofen=tzofen_cfg,
        tzofen_v2=tzofen_v2_cfg,
        tree_canopy=tree_cfg,
    )


def _prune_small_alpha_components(
    alpha_u8: np.ndarray,
    min_area_px: int,
    alpha_threshold_u8: int,
) -> np.ndarray:
    fg = alpha_u8 >= alpha_threshold_u8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), connectivity=4
    )
    keep = np.zeros_like(fg)
    for lbl in range(1, num_labels):
        if int(stats[lbl, cv2.CC_STAT_AREA]) >= min_area_px:
            keep |= labels == lbl
    fringe_zone = (
        cv2.dilate(
            keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        > 0
    ) & (alpha_u8 < alpha_threshold_u8)
    return np.where(keep | fringe_zone, alpha_u8, np.uint8(0))


def _heal_expanded_collar_lama(
    inner_rgb: np.ndarray,
    expanded_seed_rgb: np.ndarray,
    extra_heal_mask_exp: np.ndarray | None,
    lama: LamaOcclusionInpainter,
) -> np.ndarray:
    ih, iw = inner_rgb.shape[:2]
    eh, ew = expanded_seed_rgb.shape[:2]
    px = (ew - iw) // 2
    py = (eh - ih) // 2

    canvas = expanded_seed_rgb.copy()
    canvas[py : py + ih, px : px + iw] = inner_rgb

    seam_mask = np.zeros((eh, ew), dtype=np.uint8)
    band = 56
    seam_mask[py - band : py + band, px - band : px + iw + band] = 255
    seam_mask[py + ih - band : py + ih + band, px - band : px + iw + band] = 255
    seam_mask[py - band : py + ih + band, px - band : px + band] = 255
    seam_mask[py - band : py + ih + band, px + iw - band : px + iw + band] = 255
    if extra_heal_mask_exp is not None:
        seam_mask = np.maximum(seam_mask, extra_heal_mask_exp)

    half_w, half_h = ew // 2, eh // 2
    small_img = Image.fromarray(canvas, mode="RGB").resize(
        (half_w, half_h), Image.Resampling.LANCZOS
    )
    small_mask = np.asarray(
        Image.fromarray(seam_mask, mode="L").resize((half_w, half_h), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    healed_small = lama.inpaint_background(small_img, small_mask)
    healed_exp = np.asarray(
        healed_small.resize((ew, eh), Image.Resampling.LANCZOS), dtype=np.float32
    )

    core_keep = np.zeros((eh, ew), dtype=np.float32)
    margin = 38
    core_keep[py + margin : py + ih - margin, px + margin : px + iw - margin] = 1.0
    if extra_heal_mask_exp is not None:
        core_keep = np.where(extra_heal_mask_exp > 0, 0.0, core_keep)
    core_keep = ndi.gaussian_filter(core_keep, sigma=18.0)
    blended = core_keep[:, :, None] * canvas.astype(np.float32) + (
        1.0 - core_keep[:, :, None]
    ) * healed_exp
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def _fill_interior_alpha_pinholes(
    alpha_u8: np.ndarray,
    max_hole_area_px: int,
    preserve_open_box_px: tuple[float, float, float, float],
    alpha_threshold_u8: int,
) -> np.ndarray:
    """Fill tiny interior dark pinholes inside solid brick/steel cutouts without touching open bays or portals."""
    h, w = alpha_u8.shape
    holes = (alpha_u8 < alpha_threshold_u8).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    wx0, wy0, wx1, wy1 = preserve_open_box_px
    px0 = max(0, int(np.floor(wx0)) - 10)
    py0 = max(0, int(np.floor(wy0)) - 10)
    px1 = min(w, int(np.ceil(wx1)) + 10)
    py1 = min(h, int(np.ceil(wy1)) + 10)
    out = alpha_u8.copy()
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area > max_hole_area_px:
            continue
        x0 = int(stats[lbl, cv2.CC_STAT_LEFT])
        y0 = int(stats[lbl, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[lbl, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if x0 == 0 or y0 == 0 or x1 >= w or y1 >= h:
            continue
        if x1 > px0 and x0 < px1 and y1 > py0 and y0 < py1:
            continue
        out[labels == lbl] = 255
    return out


def _prune_unanchored_cutout_rgba(
    rgba_u8: np.ndarray,
    solve_cfg: ContactAnchoredLayerDepthConfig,
) -> np.ndarray:
    """Guarantees every 4-connected opaque island belongs to a contact-anchored PDE domain."""
    arr = rgba_u8.copy()
    thresh = solve_cfg.alpha_threshold_u8
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
        near_anchored = ndi.distance_transform_edt(~anchored_opaque) <= 3.0
        drop_mask = (opaque & (~anchored_opaque)) | ((~opaque) & (~near_anchored))
        arr[drop_mask] = 0
    return arr


def _neutralize_warm_terracotta_bleed_rgb(
    img_rgb_u8: np.ndarray,
    inpaint_mask_bool: np.ndarray,
    neutral_donor_bool: np.ndarray,
) -> np.ndarray:
    """Strips residual warm orange `a*, b*` chroma bleed inside healed steel shelves/tabletop."""
    if not np.any(inpaint_mask_bool) or not np.any(neutral_donor_bool):
        return img_rgb_u8.copy()

    lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    _, near_idx = ndi.distance_transform_edt(~neutral_donor_bool, return_indices=True)
    ref_a = ndi.gaussian_filter(lab[:, :, 1][near_idx[0], near_idx[1]], sigma=6.0)
    ref_b = ndi.gaussian_filter(lab[:, :, 2][near_idx[0], near_idx[1]], sigma=6.0)

    dist_in = ndi.distance_transform_edt(inpaint_mask_bool).astype(np.float32)
    blend = np.clip(dist_in / 4.0, 0.0, 1.0)

    clamped_a = np.minimum(lab[:, :, 1], ref_a + 1.5)
    clamped_b = np.minimum(lab[:, :, 2], ref_b + 2.5)
    lab[:, :, 1] = np.where(
        inpaint_mask_bool, (1.0 - blend) * lab[:, :, 1] + blend * clamped_a, lab[:, :, 1]
    )
    lab[:, :, 2] = np.where(
        inpaint_mask_bool, (1.0 - blend) * lab[:, :, 2] + blend * clamped_b, lab[:, :, 2]
    )
    return cv2.cvtColor(np.clip(np.round(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _reflect_bottom_legs_into_collar(
    rgb_u8: np.ndarray,
    alpha_u8: np.ndarray,
    y0: int,
    y1: int,
    alpha_thr: int,
    dilate_px: int,
    max_border_leg_x_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirrors frame-exiting front legs (`x < max_border_leg_x_px`) across `y1 - 1` via `BORDER_REFLECT_101`."""
    ih, _ = alpha_u8.shape
    out_rgb = rgb_u8.copy()
    out_a = alpha_u8.copy()
    if ih > y1 + 1:
        touch_cols = np.any(out_a[y1 - 3 : y1, :] >= alpha_thr, axis=0)
        touch_cols[max_border_leg_x_px:] = False
        kw = dilate_px * 2 + 1
        touch_dil = (
            cv2.dilate(touch_cols[None, :].astype(np.uint8), np.ones((1, kw), dtype=np.uint8))[0]
            > 0
        )
        n_bot = ih - y1
        refl_rgb = cv2.copyMakeBorder(
            out_rgb[y0:y1, :], 0, n_bot, 0, 0, borderType=cv2.BORDER_REFLECT_101
        )[y1 - y0 :]
        refl_a = cv2.copyMakeBorder(
            out_a[y0:y1, :], 0, n_bot, 0, 0, borderType=cv2.BORDER_REFLECT_101
        )[y1 - y0 :]
        out_rgb[y1:, touch_dil] = refl_rgb[:, touch_dil]
        out_a[y1:, touch_dil] = refl_a[:, touch_dil]

    out_a[0, :] = 0
    out_a[-1, :] = 0
    out_a[:, 0] = 0
    out_a[:, -1] = 0
    return out_rgb, out_a


def _extract_unified_tzofen_cutout_and_room(
    cfg: TzofenSceneBuildConfig,
    repo_root: Path,
    depth_est: DepthAnythingEstimator,
    matte_ext: SalientMatteExtractor,
    sam_ext: SamBoxMaskExtractor,
    lama: LamaOcclusionInpainter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the clean unified installation RGBA (`tz_pad`, `unified_fg_rgba`), room-detrended
    `da2_excess`, split-LaMa healed room plates (`lama_inner_rgb`, `bg_exp_rgb`), and ghost-free
    `clean_room_u8` (`2700x1800`) shared by `tzofen_brick_table` and `tzofen_brick_table_v2`.
    """
    iw, ih = cfg.inner_dims.width, cfg.inner_dims.height
    ew, eh = cfg.expanded_dims.width, cfg.expanded_dims.height
    px, py = (ew - iw) // 2, (eh - ih) // 2
    ucfg = cfg.unified_cutout
    alpha_thr = cfg.topology_verification.alpha_threshold_u8

    src_path = repo_root / cfg.source_image_relpath
    if not src_path.is_file():
        raise FileNotFoundError(f"Missing Tzofen source image: {src_path}")
    tz_orig = Image.open(src_path).convert("RGB")
    tz_scaled = np.asarray(
        tz_orig.resize((iw, cfg.scaled_height_px), Image.Resampling.LANCZOS), dtype=np.uint8
    )

    y0 = cfg.inner_y_offset_px
    y1 = y0 + cfg.scaled_height_px
    tz_pad = cv2.copyMakeBorder(
        tz_scaled, y0, ih - y1, 0, 0, borderType=cv2.BORDER_REFLECT_101
    )
    tz_pad_img = Image.fromarray(tz_pad, mode="RGB")

    da2_scaled = depth_est.predict_depth(Image.fromarray(tz_scaled, mode="RGB"))
    sal_scaled = (
        matte_ext.extract_alpha(Image.fromarray(tz_scaled, mode="RGB")).astype(np.float32) / 255.0
    )
    da2 = cv2.copyMakeBorder(
        da2_scaled, y0, ih - y1, 0, 0, borderType=cv2.BORDER_REFLECT_101
    )
    sal = np.zeros((ih, iw), dtype=np.float32)
    sal[y0:y1] = sal_scaled

    sam_u8 = sam_ext.segment_point_groups(
        tz_pad_img,
        cfg.sam_prompts,
        (),
        max_pinhole_fill_px=0,
        min_major_component_px=80,
    )
    sam_f = sam_u8.astype(np.float32) / 255.0

    bg_seed = np.zeros((ih, iw), dtype=bool)
    for b in cfg.bg_seed_boxes:
        bg_seed[b.y0 : b.y1, b.x0 : b.x1] = True
    d_bg_row = np.zeros(ih, dtype=np.float32)
    for y in range(ih):
        m = bg_seed[y]
        d_bg_row[y] = float(np.median(da2[y, m])) if np.any(m) else d_bg_row[max(0, y - 1)]
    d_bg_row = ndi.gaussian_filter1d(d_bg_row, sigma=8.0)
    da2_excess = da2 - d_bg_row[:, None]

    rgb_f = tz_pad.astype(np.float32)
    r_minus_b = rgb_f[:, :, 0] - rgb_f[:, :, 2]
    luma = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]

    yy = np.arange(ih)[:, None]
    xx = np.arange(iw)[None, :]
    upper_zone = (yy >= ucfg.upper_y_min_px) & (yy <= ucfg.upper_y_max_px)
    da2_upper_fg = (
        np.clip((da2_excess - ucfg.da2_upper_offset) / ucfg.da2_upper_span, 0.0, 1.0) * upper_zone
    )
    sal_upper = sal * upper_zone * (da2_excess > ucfg.sal_upper_da2_min)
    brick_upper = (
        np.clip((r_minus_b - ucfg.brick_upper_rb_offset) / ucfg.brick_upper_rb_span, 0.0, 1.0)
        * upper_zone
        * (da2_excess > ucfg.brick_upper_da2_min)
    )

    kd = ucfg.sam_lower_dilate_px
    if kd % 2 == 0:
        kd += 1
    sam_dil = (
        cv2.dilate(
            (sam_f > 0.35).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)),
        )
        > 0
    )
    lower_zone = yy > ucfg.upper_y_max_px
    sal_lower = sal * lower_zone * sam_dil
    da2_lower_fg = (
        np.clip((da2_excess - ucfg.da2_lower_offset) / ucfg.da2_lower_span, 0.0, 1.0)
        * lower_zone
        * sam_dil
    )

    alpha_f = np.maximum.reduce(
        [sam_f, da2_upper_fg, sal_upper, brick_upper, sal_lower, da2_lower_fg]
    )
    wall_like = (
        (yy < ucfg.wall_like_y_max_px)
        & (luma > ucfg.wall_like_luma_min)
        & (r_minus_b >= ucfg.wall_like_rb_min)
        & (da2_excess < ucfg.wall_like_da2_max)
    )
    right_wall_shadow = (
        (yy < ucfg.wall_like_y_max_px)
        & (xx >= ucfg.right_wall_shadow_x_min_px)
        & (r_minus_b < ucfg.right_wall_shadow_rb_max)
    )
    fgb = ucfg.floor_glare_box
    floor_glare_sliver = (
        (yy >= fgb.y0)
        & (yy < fgb.y1)
        & (xx >= fgb.x0)
        & (xx < fgb.x1)
        & (luma > ucfg.floor_glare_luma_min)
        & (r_minus_b < ucfg.floor_glare_rb_max)
    )
    alpha_f = np.where(wall_like | right_wall_shadow | floor_glare_sliver, 0.0, alpha_f)
    alpha_f = ndi.gaussian_filter(
        np.clip((alpha_f - ucfg.alpha_ramp_offset) / ucfg.alpha_ramp_span, 0.0, 1.0).astype(
            np.float32
        ),
        sigma=ucfg.alpha_blur_sigma,
    )
    alpha_f[: ucfg.upper_y_min_px, :] = 0.0
    alpha_f[y1:, :] = 0.0
    alpha_f[:, : ucfg.side_border_clear_px] = 0.0
    alpha_f[:, iw - ucfg.side_border_clear_px :] = 0.0

    # Clean left wall-grille teeth and right baseboard corner nub before pinhole fill:
    alpha_u8 = np.clip(np.round(alpha_f * 255.0), 0, 255).astype(np.uint8)
    lvt = ucfg.left_vent_trim_box
    alpha_u8[lvt.y0 : lvt.y1, lvt.x0 : lvt.x1] = np.where(
        r_minus_b[lvt.y0 : lvt.y1, lvt.x0 : lvt.x1] < ucfg.left_vent_trim_rb_max,
        np.uint8(0),
        alpha_u8[lvt.y0 : lvt.y1, lvt.x0 : lvt.x1],
    )
    rbt = ucfg.right_baseboard_trim_box
    alpha_u8[rbt.y0 : rbt.y1, rbt.x0 : rbt.x1] = np.where(
        luma[rbt.y0 : rbt.y1, rbt.x0 : rbt.x1] < ucfg.right_baseboard_trim_luma_max,
        np.uint8(0),
        alpha_u8[rbt.y0 : rbt.y1, rbt.x0 : rbt.x1],
    )

    alpha_u8 = _prune_small_alpha_components(
        alpha_u8,
        min_area_px=ucfg.min_sculp_component_px,
        alpha_threshold_u8=alpha_thr,
    )
    alpha_u8 = _fill_interior_alpha_pinholes(
        alpha_u8,
        max_hole_area_px=ucfg.max_sculp_pinhole_px,
        preserve_open_box_px=cfg.window_portal_inner_box_px,
        alpha_threshold_u8=alpha_thr,
    )
    # Open only the plaster wall seen through the window frame. The frame posts, the white plank and
    # the terracotta bricks standing in front of the opening stay on `fg`, so they occlude the
    # treetop portal exactly as in the photograph (a hard rectangle cut used to erase them).
    wx0, wy0, wx1, wy1 = cfg.window_portal_inner_box_px
    bx0, by0 = int(np.ceil(wx0)), int(np.ceil(wy0))
    bx1, by1 = int(np.floor(wx1)), int(np.floor(wy1))
    win_wall = (luma[by0:by1, bx0:bx1] > ucfg.wall_like_luma_min) & (
        r_minus_b[by0:by1, bx0:bx1] >= ucfg.wall_like_rb_min
    )
    win_wall = ndi.binary_opening(win_wall, iterations=1)
    win_labels, n_win = ndi.label(win_wall)
    if n_win == 0:
        raise RuntimeError("Tzofen window opening contains no plaster-wall pixels to open.")
    win_sizes = ndi.sum(win_wall, win_labels, index=np.arange(1, n_win + 1))
    win_wall = win_labels == (int(np.argmax(win_sizes)) + 1)
    win_open = ndi.gaussian_filter(win_wall.astype(np.float32), sigma=0.6)
    alpha_u8[by0:by1, bx0:bx1] = np.clip(
        np.round(alpha_u8[by0:by1, bx0:bx1].astype(np.float32) * (1.0 - win_open)), 0, 255
    ).astype(np.uint8)

    refl_rgb, refl_a_u8 = _reflect_bottom_legs_into_collar(
        rgb_u8=tz_pad,
        alpha_u8=alpha_u8,
        y0=y0,
        y1=y1,
        alpha_thr=alpha_thr,
        dilate_px=ucfg.bottom_leg_reflect_dilate_px,
        max_border_leg_x_px=ucfg.max_border_leg_x_px,
    )
    unified_fg_clean_rgb = decontaminate_matte_edge_rgb(refl_rgb, refl_a_u8)
    unified_fg_rgba = _prune_unanchored_cutout_rgba(
        np.dstack([unified_fg_clean_rgb, refl_a_u8]),
        cfg.fg_contact_solve,
    )

    # Split authentic `1500x919` room at baseboard (`by = 442`) so upper wall LaMa never pulls dark
    # floor colors and lower floor LaMa preserves 100% of authentic Binyamin Gallery stone tiles:
    by = cfg.baseboard_inner_y - y0
    alpha_auth = unified_fg_rgba[y0:y1, :, 3]

    wall_rgb = tz_scaled[:by, :].copy()
    wall_occ = (alpha_auth[:by, :] >= ucfg.bg_occ_alpha_min_u8).copy()
    wall_occ[
        max(0, int(np.floor(wy0)) - y0 - 8) : min(by, int(np.ceil(wy1)) - y0 + 8),
        max(0, int(np.floor(wx0)) - 8) : min(iw, int(np.ceil(wx1)) + 8),
    ] = True
    wvb = ucfg.wall_vent_mask_box
    wall_occ[max(0, wvb.y0 - y0) : min(by, wvb.y1 - y0), wvb.x0 : wvb.x1] = True
    wrc = ucfg.wall_right_corner_mask_box
    wall_occ[max(0, wrc.y0 - y0) : min(by, wrc.y1 - y0), wrc.x0 : wrc.x1] = True

    wall_seed_rgb = np.median(wall_rgb[:24, :], axis=(0, 1)).round().astype(np.uint8)
    wall_pad_rgb = np.broadcast_to(
        wall_seed_rgb, (ucfg.wall_pad_height_px, ucfg.wall_pad_width_px, 3)
    ).copy()
    wall_pad_rgb[:by, :iw] = wall_rgb
    wall_pad_occ = np.zeros((ucfg.wall_pad_height_px, ucfg.wall_pad_width_px), dtype=np.uint8)
    wall_pad_occ[:by, :iw] = (wall_occ.astype(np.uint8)) * 255
    wall_inp_raw = np.asarray(
        lama.inpaint_background(
            Image.fromarray(wall_pad_rgb, mode="RGB"),
            wall_pad_occ,
            dilation_radius_px=ucfg.bg_lama_dilation_px,
        ),
        dtype=np.float32,
    )[:by, :iw]

    # Harmonize wall low-frequency DC to eliminate wide-mask LaMa brightness sag while preserving
    # fine plaster grain and the authentic wall electrical outlet on the far right:
    wall_src_f = wall_rgb.astype(np.float32)
    wall_sat = cv2.cvtColor(wall_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)[:, :, 1]
    wall_luma = (
        0.299 * wall_src_f[:, :, 0]
        + 0.587 * wall_src_f[:, :, 1]
        + 0.114 * wall_src_f[:, :, 2]
    )
    wall_donor = (
        (~wall_occ)
        & (wall_luma >= ucfg.wall_donor_min_luma)
        & (wall_sat <= ucfg.wall_donor_max_sat)
    )
    wall_donor[:, ucfg.wall_outlet_preserve_x_min_px :] = False
    if not np.any(wall_donor):
        raise RuntimeError("Empty wall donor mask in Tzofen room healing")

    _, near_w_idx = ndi.distance_transform_edt(~wall_donor, return_indices=True)
    wall_field = wall_src_f[near_w_idx[0], near_w_idx[1]].copy()
    ds = 8
    fh, fw = max(4, by // ds), max(4, iw // ds)
    f_small = cv2.resize(wall_field, (fw, fh), interpolation=cv2.INTER_AREA)
    d_small = cv2.resize(wall_donor.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST) > 0
    fixed_small = f_small.copy()
    for _ in range(300):
        nb = 0.25 * (
            np.roll(f_small, 1, axis=0)
            + np.roll(f_small, -1, axis=0)
            + np.roll(f_small, 1, axis=1)
            + np.roll(f_small, -1, axis=1)
        )
        nb[0, :] = f_small[1, :]
        nb[-1, :] = f_small[-2, :]
        nb[:, 0] = f_small[:, 1]
        nb[:, -1] = f_small[:, -2]
        f_small = np.where(d_small[:, :, None], fixed_small, nb)
    wall_ref_low = ndi.gaussian_filter(
        cv2.resize(f_small, (iw, by), interpolation=cv2.INTER_CUBIC),
        sigma=(ucfg.wall_dc_sigma_px, ucfg.wall_dc_sigma_px, 0.0),
    )
    wall_grain = wall_inp_raw - ndi.gaussian_filter(
        wall_inp_raw, sigma=(ucfg.wall_grain_sigma_px, ucfg.wall_grain_sigma_px, 0.0)
    )
    wall_target = wall_ref_low + wall_grain
    dist_out_wall = ndi.distance_transform_edt(~wall_occ).astype(np.float32)
    w_wall = np.clip(1.0 - dist_out_wall / max(1.0, ucfg.wall_dc_blend_band_px), 0.0, 1.0)
    w_wall[:, ucfg.wall_outlet_preserve_x_min_px :] = np.where(
        wall_occ[:, ucfg.wall_outlet_preserve_x_min_px :],
        1.0,
        0.0,
    )
    healed_wall = np.clip(
        np.round((1.0 - w_wall[:, :, None]) * wall_src_f + w_wall[:, :, None] * wall_target),
        0,
        255,
    ).astype(np.uint8)

    # Single-pass floor LaMa preserving crisp Binyamin Gallery stone tiles (padded via
    # BORDER_REFLECT_101 to multiples of 8 so LaMa never pulls zero-padded black corner pixels):
    floor_rgb = tz_scaled[by:, :].copy()
    fvr = ucfg.floor_vent_rim_box
    fvr_h = max(0, fvr.y1 - cfg.baseboard_inner_y)
    fvr_w = fvr.x1 - fvr.x0
    if fvr_h > 0 and fvr_w > 0:
        floor_rgb[:fvr_h, fvr.x0 : fvr.x1] = floor_rgb[:fvr_h, fvr.x1 : fvr.x1 + fvr_w]
    floor_occ = (alpha_auth[by:, :] >= ucfg.bg_occ_alpha_min_u8).copy()
    fls = ucfg.floor_leg_shadow_box
    sy0 = max(0, fls.y0 - cfg.baseboard_inner_y)
    sy1 = min(floor_rgb.shape[0], fls.y1 - cfg.baseboard_inner_y)
    floor_luma = (
        0.299 * floor_rgb[:, :, 0].astype(np.float32)
        + 0.587 * floor_rgb[:, :, 1].astype(np.float32)
        + 0.114 * floor_rgb[:, :, 2].astype(np.float32)
    )
    floor_occ[sy0:sy1, fls.x0 : fls.x1] |= (
        floor_luma[sy0:sy1, fls.x0 : fls.x1] < ucfg.floor_leg_shadow_luma_max
    )
    touch_cols = np.any(floor_occ[-24:, :], axis=0)
    kw_fl = ucfg.floor_bottom_leg_dilate_px * 2 + 1
    touch_dil = (
        cv2.dilate(touch_cols[None, :].astype(np.uint8), np.ones((1, kw_fl), dtype=np.uint8))[0] > 0
    )
    floor_occ[-ucfg.floor_bottom_leg_rows_px :, touch_dil] = True

    fh_raw, fw_raw = floor_rgb.shape[:2]
    fh_pad = ((fh_raw + 7) // 8) * 8
    fw_pad = ((fw_raw + 7) // 8) * 8
    floor_pad_rgb = cv2.copyMakeBorder(
        floor_rgb, 0, fh_pad - fh_raw, 0, fw_pad - fw_raw, borderType=cv2.BORDER_REFLECT_101
    )
    floor_pad_occ = cv2.copyMakeBorder(
        (floor_occ.astype(np.uint8)) * 255,
        0,
        fh_pad - fh_raw,
        0,
        fw_pad - fw_raw,
        borderType=cv2.BORDER_REFLECT_101,
    )
    healed_floor = np.asarray(
        lama.inpaint_background(
            Image.fromarray(floor_pad_rgb, mode="RGB"),
            floor_pad_occ,
            dilation_radius_px=ucfg.bg_lama_dilation_px,
        ),
        dtype=np.uint8,
    )[:fh_raw, :fw_raw]

    auth_clean_rgb = np.vstack([healed_wall, healed_floor])
    bg_exp_rgb = cv2.copyMakeBorder(
        auth_clean_rgb,
        py + y0,
        eh - (py + y1),
        px,
        ew - (px + iw),
        borderType=cv2.BORDER_REFLECT_101,
    )
    lama_inner_rgb = bg_exp_rgb[py : py + ih, px : px + iw].copy()

    raw_auth_da2 = depth_est.predict_depth(Image.fromarray(auth_clean_rgb, mode="RGB"))
    raw_auth_u8 = np.clip(np.round(raw_auth_da2 * 255.0), 0, 255).astype(np.uint8)
    raw_room_u8 = cv2.copyMakeBorder(
        raw_auth_u8,
        py + y0,
        eh - (py + y1),
        px,
        ew - (px + iw),
        borderType=cv2.BORDER_REFLECT_101,
    )
    clean_room_pre_u8 = clean_room_depth_under_cutouts(
        raw_room_u8, [unified_fg_rgba], cfg.room_depth_clean
    )
    clean_room_u8 = cv2.copyMakeBorder(
        clean_room_pre_u8[py + y0 : py + y1, px : px + iw],
        py + y0,
        eh - (py + y1),
        px,
        ew - (px + iw),
        borderType=cv2.BORDER_REFLECT_101,
    )
    return tz_pad, unified_fg_rgba, da2_excess, lama_inner_rgb, bg_exp_rgb, clean_room_u8


def build_tzofen_unified_scene(
    cfg: TzofenSceneBuildConfig,
    scenes_dir: Path,
    tz_pad: np.ndarray,
    unified_fg_rgba: np.ndarray,
    da2_excess: np.ndarray,
    lama_inner_rgb: np.ndarray,
    bg_exp_rgb: np.ndarray,
    clean_room_u8: np.ndarray,
) -> dict[str, Any]:
    """Writes restored unified Scene 12 (`tzofen_brick_table` / `scene12_tzofen_*`)."""
    ih, iw = unified_fg_rgba.shape[:2]
    fg_alpha_u8 = unified_fg_rgba[:, :, 3]
    bmax = int(
        max(
            fg_alpha_u8[0, :].max(),
            fg_alpha_u8[-1, :].max(),
            fg_alpha_u8[:, 0].max(),
            fg_alpha_u8[:, -1].max(),
        )
    )
    if bmax != 0:
        raise RuntimeError(f"Tzofen unified fg border alpha must be 0, got {bmax}")

    prefix = cfg.asset_prefix
    Image.fromarray(unified_fg_rgba, mode="RGBA").save(scenes_dir / f"{prefix}_fg.png")
    Image.fromarray(np.zeros((ih, iw, 4), dtype=np.uint8), mode="RGBA").save(
        scenes_dir / f"{prefix}_mid.png"
    )
    stale_mid_depth = scenes_dir / f"{prefix}_mid_depth.png"
    if stale_mid_depth.exists():
        stale_mid_depth.unlink()

    Image.fromarray(lama_inner_rgb, mode="RGB").save(
        scenes_dir / f"{prefix}_lama_inner.jpg", quality=cfg.jpeg_quality
    )
    Image.fromarray(bg_exp_rgb, mode="RGB").save(
        scenes_dir / f"{prefix}_bg_inpainted.jpg", quality=cfg.jpeg_quality
    )
    Image.fromarray(clean_room_u8, mode="L").save(scenes_dir / f"{prefix}_depth.png")

    fg_excess_pos = np.maximum(0.0, da2_excess)
    fg_opaque = fg_alpha_u8 >= cfg.fg_contact_solve.alpha_threshold_u8
    prot_scale = max(1e-5, float(np.percentile(fg_excess_pos[fg_opaque], 98.0)))
    fg_prot_norm = np.clip(fg_excess_pos / prot_scale, 0.0, 1.0).astype(np.float32)

    fg_solve = solve_contact_anchored_layer_depth(
        room_depth_expanded=clean_room_u8,
        cutout_rgba=unified_fg_rgba,
        cfg=cfg.fg_contact_solve,
        neural_protrusion_norm=fg_prot_norm,
    )
    fg_depth_u8 = fg_solve.depth_u8
    Image.fromarray(fg_depth_u8, mode="L").save(scenes_dir / f"{prefix}_fg_depth.png")

    fg_report = verify_layer_contact_and_discontinuity(
        room_depth_expanded=clean_room_u8,
        layer_rgba=unified_fg_rgba,
        layer_depth=fg_depth_u8,
        contact_mask=fg_solve.contact_mask,
        bg_z=cfg.fg_contact_solve.bg_z,
        room_depth_amplitude=cfg.fg_contact_solve.room_depth_amplitude,
        room_pin_margin_uv=cfg.fg_contact_solve.room_pin_margin_uv,
        layer_z=cfg.fg_contact_solve.layer_z,
        layer_depth_amplitude=cfg.fg_contact_solve.layer_depth_amplitude,
        layer_pin_margin_uv=cfg.fg_contact_solve.layer_pin_margin_uv,
        verify_cfg=cfg.topology_verification,
    )
    if not fg_report.all_passed:
        raise RuntimeError(f"Tzofen unified fg failed depth topology gate: {fg_report.to_dict()}")

    rgb_f = tz_pad.astype(np.float32)
    luma = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    wx0, wy0, wx1, wy1 = cfg.window_frame_sample_box_px
    frame_mask = (fg_alpha_u8[wy0:wy1, wx0:wx1] >= 200) & (luma[wy0:wy1, wx0:wx1] > 185.0)
    if not np.any(frame_mask):
        raise RuntimeError("Empty white window frame mask on unified Tzofen fg!")
    frame_z_med = float(np.median(fg_solve.world_z_map[wy0:wy1, wx0:wx1][frame_mask]))
    portal_world_z = round(frame_z_med - cfg.portal_z_clearance_behind_frame, 2)

    return {
        "id": cfg.scene_id,
        "midCoverageRatio": 0.0,
        "fgCoverageRatio": round(float((fg_alpha_u8 >= 20).mean()), 4),
        "windowFrameMedianWorldZ": round(frame_z_med, 3),
        "portalWorldZ": portal_world_z,
        "portalInnerBoxPx": list(cfg.window_portal_inner_box_px),
        "portalRecessDepth": cfg.portal_recess_depth,
        "topologyVerification": {
            "fg": fg_report.to_dict(),
        },
    }


def _compute_v2_brick_protrusion(
    da2_excess: np.ndarray,
    fg_brick_alpha_u8: np.ndarray,
    v2_cfg: TzofenSegmentedV2BuildConfig,
) -> np.ndarray:
    """Volumetric relief for the 11 unoccluded foreground bricks anchored C^0 to `mid`."""
    ih, iw = fg_brick_alpha_u8.shape
    fg_solve_cfg = v2_cfg.fg_contact_solve
    opaque = fg_brick_alpha_u8 >= fg_solve_cfg.alpha_threshold_u8
    da2_pos = np.maximum(0.0, da2_excess)
    da2_scale = max(1e-5, float(np.percentile(da2_pos[opaque], 96.0)))
    da2_norm = np.clip(da2_pos / da2_scale, 0.0, 1.0).astype(np.float32)

    bridged = ndi.distance_transform_edt(~opaque) <= float(fg_solve_cfg.component_bridge_px)
    labeled, num_groups = ndi.label(bridged)
    rise_norm = np.zeros((ih, iw), dtype=np.float32)
    for gid in range(1, num_groups + 1):
        grp = opaque & (labeled == gid)
        if not np.any(grp):
            continue
        _, xs = np.where(grp)
        x_min, x_max = int(xs.min()), int(xs.max())
        for col in range(x_min, x_max + 1):
            c_ys = np.where(grp[:, col])[0]
            if len(c_ys) == 0:
                continue
            y_bot = float(c_ys.max())
            rise_px = np.maximum(0.0, y_bot - np.arange(ih, dtype=np.float32))
            rise_norm[grp[:, col], col] = np.clip(
                rise_px[grp[:, col]] / max(1.0, v2_cfg.brick_height_saturation_px), 0.0, 1.0
            )

    edt_in = ndi.distance_transform_edt(opaque).astype(np.float32)
    block_dome = np.clip(edt_in / 14.0, 0.0, 1.0)
    w_h = v2_cfg.brick_height_relief_weight
    combined = (1.0 - w_h) * da2_norm + w_h * np.maximum(rise_norm, 0.55 * block_dome)
    return np.clip(combined, 0.0, 1.0).astype(np.float32)


def build_tzofen_segmented_v2_scene(
    base_cfg: TzofenSceneBuildConfig,
    v2_cfg: TzofenSegmentedV2BuildConfig,
    scenes_dir: Path,
    tz_pad: np.ndarray,
    unified_fg_rgba: np.ndarray,
    da2_excess: np.ndarray,
    lama_inner_rgb: np.ndarray,
    bg_exp_rgb: np.ndarray,
    clean_room_u8: np.ndarray,
    sam_ext: SamBoxMaskExtractor,
    lama: LamaOcclusionInpainter,
) -> dict[str, Any]:
    """
    Writes archived segmented Scene 12 v2 (`tzofen_brick_table_v2` / `scene12_tzofen_v2_*`):
    lifts the 11 completely unoccluded perimeter/top terracotta bricks onto `fg` via per-object
    SAM prompts while keeping the steel table + interleaved bricks intact on `mid` with local
    neural `LaMa` + Lab-neutralized tabletop/shelf healing.
    """
    ih, iw = unified_fg_rgba.shape[:2]
    ew, eh = base_cfg.expanded_dims.width, base_cfg.expanded_dims.height
    px, py = (ew - iw) // 2, (eh - ih) // 2
    alpha_thr = v2_cfg.topology_verification.alpha_threshold_u8
    sculp_a = unified_fg_rgba[:, :, 3]
    tz_pad_img = Image.fromarray(tz_pad, mode="RGB")

    fg_bool = np.zeros((ih, iw), dtype=bool)
    pad_b = v2_cfg.prompt_box_pad_px
    for pr in v2_cfg.fg_brick_prompts:
        m_u8 = sam_ext.segment_point_groups(
            tz_pad_img,
            (pr,),
            (),
            max_pinhole_fill_px=v2_cfg.sam_pinhole_fill_px,
            min_major_component_px=v2_cfg.sam_min_component_px,
        )
        bx = pr.box
        by0 = max(0, bx.y0 - pad_b)
        by1 = min(ih, bx.y1 + pad_b)
        bx0 = max(0, bx.x0 - pad_b)
        bx1 = min(iw, bx.x1 + pad_b)
        sub = np.zeros((ih, iw), dtype=bool)
        sub[by0:by1, bx0:bx1] = m_u8[by0:by1, bx0:bx1] >= 128
        fg_bool |= sub

    fg_bool &= sculp_a >= alpha_thr
    dist_in_fg = ndi.distance_transform_edt(fg_bool).astype(np.float32)
    dist_out_fg = ndi.distance_transform_edt(~fg_bool).astype(np.float32)
    sdf_fg = dist_in_fg - dist_out_fg - v2_cfg.fg_sdf_erode_bias_px
    fg_a_u8 = np.minimum(
        sculp_a,
        np.clip(
            np.round(np.clip(0.5 + sdf_fg / max(0.25, v2_cfg.fg_sdf_band_px), 0.0, 1.0) * 255.0),
            0,
            255,
        ).astype(np.uint8),
    )
    fg_a_u8[0, :] = 0
    fg_a_u8[-1, :] = 0
    fg_a_u8[:, 0] = 0
    fg_a_u8[:, -1] = 0

    # Construct `mid_a_u8`: keep intact unified sculpture silhouette everywhere except open-air
    # protrusions above shelves/tabletop behind lifted foreground bricks:
    mid_bool = (sculp_a >= 128).copy()
    air_cut_zone = np.zeros((ih, iw), dtype=bool)
    zp = v2_cfg.mid_air_zone_pad_px
    for ab in v2_cfg.mid_open_air_boxes:
        mid_bool[ab.y0 : ab.y1, ab.x0 : ab.x1] = False
        air_cut_zone[
            max(0, ab.y0 - zp) : min(ih, ab.y1 + zp),
            max(0, ab.x0 - zp) : min(iw, ab.x1 + zp),
        ] = True

    dist_in_m = ndi.distance_transform_edt(mid_bool).astype(np.float32)
    dist_out_m = ndi.distance_transform_edt(~mid_bool).astype(np.float32)
    mid_sdf_u8 = np.clip(
        np.round(
            np.clip(0.5 + (dist_in_m - dist_out_m) / max(0.25, v2_cfg.mid_sdf_band_px), 0.0, 1.0)
            * 255.0
        ),
        0,
        255,
    ).astype(np.uint8)
    mid_a_u8 = np.where(air_cut_zone, np.minimum(sculp_a, mid_sdf_u8), sculp_a).astype(np.uint8)
    wx0, wy0, wx1, wy1 = base_cfg.window_portal_inner_box_px
    mid_a_u8[
        int(np.ceil(wy0)) + 1 : int(np.floor(wy1)),
        int(np.ceil(wx0)) + 1 : int(np.floor(wx1)),
    ] = 0
    mid_a_u8[0, :] = 0
    mid_a_u8[-1, :] = 0
    mid_a_u8[:, 0] = 0
    mid_a_u8[:, -1] = 0

    # Heal lifted foreground bricks on `mid` RGB via neural `LaMa` seeded from interior sculpture RGB:
    kd_b = v2_cfg.brick_inpaint_dilate_px * 2 + 1
    inpaint_brick_mask = (
        cv2.dilate(
            fg_bool.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd_b, kd_b)),
        )
        > 0
    )
    valid_mid_donor = (mid_a_u8 >= 180) & (~inpaint_brick_mask)
    if not np.any(valid_mid_donor):
        raise RuntimeError("Empty valid mid donor mask in Tzofen v2")
    _, near_mid_idx = ndi.distance_transform_edt(~valid_mid_donor, return_indices=True)
    mid_src_rgb = unified_fg_rgba[:, :, :3]
    mid_seed_rgb = np.where(
        (mid_a_u8 >= 128)[:, :, None],
        mid_src_rgb,
        mid_src_rgb[near_mid_idx[0], near_mid_idx[1]],
    )
    healed_mid_rgb = np.asarray(
        lama.inpaint_background(
            Image.fromarray(mid_seed_rgb, mode="RGB"),
            (inpaint_brick_mask.astype(np.uint8)) * 255,
            dilation_radius_px=v2_cfg.brick_lama_dilation_px,
        ),
        dtype=np.float32,
    )
    d_in_mask = ndi.distance_transform_edt(inpaint_brick_mask).astype(np.float32)
    w_m = np.clip(d_in_mask / max(1.0, v2_cfg.brick_inpaint_feather_px), 0.0, 1.0)[:, :, None]
    mid_comp_rgb = np.clip(
        np.round((1.0 - w_m) * mid_seed_rgb.astype(np.float32) + w_m * healed_mid_rgb),
        0,
        255,
    ).astype(np.uint8)

    # Strip warm terracotta chroma bleed strictly inside white/grey steel shelf zones:
    steel_zone = np.zeros((ih, iw), dtype=bool)
    for sb in v2_cfg.neutral_steel_boxes:
        steel_zone[sb.y0 : sb.y1, sb.x0 : sb.x1] = True
    mid_hsv_sat = cv2.cvtColor(mid_src_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)[:, :, 1]
    mid_luma = (
        0.299 * mid_src_rgb[:, :, 0].astype(np.float32)
        + 0.587 * mid_src_rgb[:, :, 1].astype(np.float32)
        + 0.114 * mid_src_rgb[:, :, 2].astype(np.float32)
    )
    neutral_steel_donor = (
        valid_mid_donor
        & (mid_luma >= v2_cfg.neutral_steel_min_luma)
        & (mid_hsv_sat <= v2_cfg.neutral_steel_max_sat)
    )
    mid_comp_rgb = _neutralize_warm_terracotta_bleed_rgb(
        img_rgb_u8=mid_comp_rgb,
        inpaint_mask_bool=inpaint_brick_mask & steel_zone,
        neutral_donor_bool=neutral_steel_donor,
    )

    fg_rgba = _prune_unanchored_cutout_rgba(
        np.dstack([decontaminate_matte_edge_rgb(tz_pad, fg_a_u8), fg_a_u8]),
        v2_cfg.fg_contact_solve,
    )
    mid_rgba = _prune_unanchored_cutout_rgba(
        np.dstack([decontaminate_matte_edge_rgb(mid_comp_rgb, mid_a_u8), mid_a_u8]),
        v2_cfg.mid_contact_solve,
    )
    fg_a_u8 = fg_rgba[:, :, 3]
    mid_a_u8 = mid_rgba[:, :, 3]

    for lname, a_chk in (("v2_mid", mid_a_u8), ("v2_fg", fg_a_u8)):
        bmax = int(
            max(a_chk[0, :].max(), a_chk[-1, :].max(), a_chk[:, 0].max(), a_chk[:, -1].max())
        )
        if bmax != 0:
            raise RuntimeError(f"Tzofen {lname} border alpha must be 0, got {bmax}")

    prefix = v2_cfg.asset_prefix
    Image.fromarray(mid_rgba, mode="RGBA").save(scenes_dir / f"{prefix}_mid.png")
    Image.fromarray(fg_rgba, mode="RGBA").save(scenes_dir / f"{prefix}_fg.png")
    Image.fromarray(lama_inner_rgb, mode="RGB").save(
        scenes_dir / f"{prefix}_lama_inner.jpg", quality=base_cfg.jpeg_quality
    )
    Image.fromarray(bg_exp_rgb, mode="RGB").save(
        scenes_dir / f"{prefix}_bg_inpainted.jpg", quality=base_cfg.jpeg_quality
    )
    Image.fromarray(clean_room_u8, mode="L").save(scenes_dir / f"{prefix}_depth.png")

    # Harmonically remove lifted foreground brick depth mounds from `da2_excess` before solving `mid`:
    full_dom = np.ones((ih, iw), dtype=bool)
    brick_heal = inpaint_brick_mask.copy()
    brick_heal[0, :] = False
    brick_heal[-1, :] = False
    brick_heal[:, 0] = False
    brick_heal[:, -1] = False
    da2_excess_mid = _solve_harmonic_field_on_domain(
        domain_mask=full_dom,
        dirichlet_mask=~brick_heal,
        dirichlet_values=da2_excess.astype(np.float64),
    ).astype(np.float32)

    mid_excess_pos = np.maximum(0.0, da2_excess_mid)
    mid_opaque = mid_a_u8 >= v2_cfg.mid_contact_solve.alpha_threshold_u8
    mid_scale = max(1e-5, float(np.percentile(mid_excess_pos[mid_opaque], 98.0)))
    mid_prot_norm = np.clip(mid_excess_pos / mid_scale, 0.0, 1.0).astype(np.float32)

    mid_solve = solve_contact_anchored_layer_depth(
        room_depth_expanded=clean_room_u8,
        cutout_rgba=mid_rgba,
        cfg=v2_cfg.mid_contact_solve,
        neural_protrusion_norm=mid_prot_norm,
    )
    mid_depth_u8 = mid_solve.depth_u8
    Image.fromarray(mid_depth_u8, mode="L").save(scenes_dir / f"{prefix}_mid_depth.png")

    mid_report = verify_layer_contact_and_discontinuity(
        room_depth_expanded=clean_room_u8,
        layer_rgba=mid_rgba,
        layer_depth=mid_depth_u8,
        contact_mask=mid_solve.contact_mask,
        bg_z=v2_cfg.mid_contact_solve.bg_z,
        room_depth_amplitude=v2_cfg.mid_contact_solve.room_depth_amplitude,
        room_pin_margin_uv=v2_cfg.mid_contact_solve.room_pin_margin_uv,
        layer_z=v2_cfg.mid_contact_solve.layer_z,
        layer_depth_amplitude=v2_cfg.mid_contact_solve.layer_depth_amplitude,
        layer_pin_margin_uv=v2_cfg.mid_contact_solve.layer_pin_margin_uv,
        verify_cfg=v2_cfg.topology_verification,
    )
    if not mid_report.all_passed:
        raise RuntimeError(f"Tzofen v2 mid failed depth topology gate: {mid_report.to_dict()}")

    mid_host_depth_exp_u8 = cv2.copyMakeBorder(
        mid_depth_u8, py, py, px, px, borderType=cv2.BORDER_REPLICATE
    )
    fg_prot_norm = _compute_v2_brick_protrusion(
        da2_excess=da2_excess,
        fg_brick_alpha_u8=fg_a_u8,
        v2_cfg=v2_cfg,
    )
    fg_solve = solve_contact_anchored_layer_depth(
        room_depth_expanded=mid_host_depth_exp_u8,
        cutout_rgba=fg_rgba,
        cfg=v2_cfg.fg_contact_solve,
        neural_protrusion_norm=fg_prot_norm,
    )
    fg_depth_u8 = fg_solve.depth_u8
    Image.fromarray(fg_depth_u8, mode="L").save(scenes_dir / f"{prefix}_fg_depth.png")

    fg_report = verify_layer_contact_and_discontinuity(
        room_depth_expanded=mid_host_depth_exp_u8,
        layer_rgba=fg_rgba,
        layer_depth=fg_depth_u8,
        contact_mask=fg_solve.contact_mask,
        bg_z=v2_cfg.fg_contact_solve.bg_z,
        room_depth_amplitude=v2_cfg.fg_contact_solve.room_depth_amplitude,
        room_pin_margin_uv=v2_cfg.fg_contact_solve.room_pin_margin_uv,
        layer_z=v2_cfg.fg_contact_solve.layer_z,
        layer_depth_amplitude=v2_cfg.fg_contact_solve.layer_depth_amplitude,
        layer_pin_margin_uv=v2_cfg.fg_contact_solve.layer_pin_margin_uv,
        verify_cfg=v2_cfg.topology_verification,
    )
    if not fg_report.all_passed:
        raise RuntimeError(f"Tzofen v2 fg failed depth topology gate: {fg_report.to_dict()}")

    fx0, fy0, fx1, fy1 = base_cfg.window_frame_sample_box_px
    frame_mask = (mid_a_u8[fy0:fy1, fx0:fx1] >= 200) & (mid_luma[fy0:fy1, fx0:fx1] > 185.0)
    if not np.any(frame_mask):
        raise RuntimeError("Empty white window frame mask on Tzofen v2 mid!")
    frame_z_med = float(np.median(mid_solve.world_z_map[fy0:fy1, fx0:fx1][frame_mask]))
    portal_world_z = round(frame_z_med - base_cfg.portal_z_clearance_behind_frame, 2)

    fg_opaque = fg_a_u8 >= v2_cfg.fg_contact_solve.alpha_threshold_u8
    brick_relief_over_mid = fg_solve.world_z_map[fg_opaque] - mid_solve.world_z_map[fg_opaque]

    return {
        "id": v2_cfg.scene_id,
        "midCoverageRatio": round(float((mid_a_u8 >= 20).mean()), 4),
        "fgCoverageRatio": round(float((fg_a_u8 >= 20).mean()), 4),
        "brickReliefOverTableP50World": round(float(np.percentile(brick_relief_over_mid, 50.0)), 2),
        "brickReliefOverTableP95World": round(float(np.percentile(brick_relief_over_mid, 95.0)), 2),
        "brickReliefOverTableMaxWorld": round(float(brick_relief_over_mid.max()), 2),
        "windowFrameMedianWorldZ": round(frame_z_med, 3),
        "portalWorldZ": portal_world_z,
        "portalInnerBoxPx": list(base_cfg.window_portal_inner_box_px),
        "portalRecessDepth": base_cfg.portal_recess_depth,
        "topologyVerification": {
            "mid": mid_report.to_dict(),
            "fg": fg_report.to_dict(),
        },
    }


def build_tree_canopy_scene(
    cfg: TreeCanopySceneBuildConfig,
    repo_root: Path,
    scenes_dir: Path,
    depth_est: DepthAnythingEstimator,
    sam_ext: SamBoxMaskExtractor,
    lama: LamaOcclusionInpainter,
) -> dict[str, Any]:
    iw, ih = cfg.inner_dims.width, cfg.inner_dims.height
    ew, eh = cfg.expanded_dims.width, cfg.expanded_dims.height
    px, py = (ew - iw) // 2, (eh - ih) // 2

    src_path = repo_root / cfg.source_image_relpath
    if not src_path.is_file():
        raise FileNotFoundError(f"Missing Tree Canopy source image: {src_path}")
    tr_orig = Image.open(src_path).convert("RGB")
    tr_inner_img = tr_orig.resize(
        (cfg.scaled_source_width_px, ih), Image.Resampling.LANCZOS
    ).crop((cfg.crop_left_px, 0, cfg.crop_left_px + iw, ih))
    tr_rgb = np.asarray(tr_inner_img, dtype=np.uint8)
    rgb_f = tr_rgb.astype(np.float32)

    da2_tr = depth_est.predict_depth(tr_inner_img)
    sam_fg_tr = (
        sam_ext.segment_point_groups(
            tr_inner_img,
            cfg.fg_sam_prompts,
            (),
            max_pinhole_fill_px=300,
            min_major_component_px=250,
        ).astype(np.float32)
        / 255.0
    )
    sam_mid_tr = (
        sam_ext.segment_point_groups(
            tr_inner_img,
            cfg.mid_sam_prompts,
            (),
            max_pinhole_fill_px=250,
            min_major_component_px=200,
        ).astype(np.float32)
        / 255.0
    )

    g_minus_r = rgb_f[:, :, 1] - rgb_f[:, :, 0]
    luma_tr = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    sky_tr = (luma_tr > 205.0) & ((rgb_f[:, :, 2] - rgb_f[:, :, 0]) > -8.0)

    da2_local = da2_tr - ndi.gaussian_filter(da2_tr, sigma=18.0)
    yy_t = np.arange(ih)[:, None]
    xx_t = np.arange(iw)[None, :]
    wire_y_center = 768.5 - (768.5 - 590.5) * (xx_t / float(iw))
    wire_band = np.abs(yy_t - wire_y_center) <= 9.0
    wire_alpha = np.clip((da2_local - 0.08) / 0.08, 0.0, 1.0) * wire_band * (~sky_tr)

    fg_bark_gate = np.clip((4.5 - g_minus_r) / 7.5, 0.0, 1.0) * (~sky_tr)
    fg_da2_gate = np.clip((da2_tr - 0.42) / 0.08, 0.0, 1.0)
    trunk_solid = (sam_fg_tr > 0.5) & (da2_tr > 0.62) & (yy_t > 320) & (~sky_tr) & (g_minus_r < 8.0)
    fg_tr_alpha = np.maximum.reduce(
        [sam_fg_tr * fg_bark_gate * fg_da2_gate, trunk_solid.astype(np.float32), wire_alpha]
    )
    sam_fg_dil = (
        cv2.dilate(
            (sam_fg_tr > 0.3).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        )
        > 0
    )
    da2_fg_ridge = np.clip((da2_tr - 0.56) / 0.06, 0.0, 1.0) * fg_bark_gate * sam_fg_dil
    fg_tr_alpha = np.maximum(fg_tr_alpha, da2_fg_ridge)
    fg_tr_alpha = ndi.gaussian_filter(
        np.where(fg_tr_alpha > 0.35, 1.0, 0.0).astype(np.float32), sigma=0.7
    )

    mid_bark_gate = np.clip((3.5 - g_minus_r) / 6.5, 0.0, 1.0) * (~sky_tr)
    mid_da2_gate = np.clip((da2_tr - 0.35) / 0.07, 0.0, 1.0)
    mid_tr_alpha = (
        sam_mid_tr
        * mid_bark_gate
        * mid_da2_gate
        * (1.0 - np.clip(fg_tr_alpha * 1.5, 0.0, 1.0))
    )
    mid_tr_alpha = ndi.gaussian_filter(
        np.where(mid_tr_alpha > 0.35, 1.0, 0.0).astype(np.float32), sigma=0.7
    )

    fg_u8 = _prune_small_alpha_components(
        np.clip(np.round(fg_tr_alpha * 255.0), 0, 255).astype(np.uint8),
        min_area_px=350,
        alpha_threshold_u8=cfg.topology_verification.alpha_threshold_u8,
    )
    mid_u8 = _prune_small_alpha_components(
        np.clip(np.round(mid_tr_alpha * 255.0), 0, 255).astype(np.uint8),
        min_area_px=1250,
        alpha_threshold_u8=cfg.topology_verification.alpha_threshold_u8,
    )

    fg_rgba = np.dstack([decontaminate_matte_edge_rgb(tr_rgb, fg_u8), fg_u8])
    mid_rgba = np.dstack([decontaminate_matte_edge_rgb(tr_rgb, mid_u8), mid_u8])
    Image.fromarray(fg_rgba, mode="RGBA").save(scenes_dir / "scene13_herzliya_canopy_fg.png")
    Image.fromarray(mid_rgba, mode="RGBA").save(scenes_dir / "scene13_herzliya_canopy_mid.png")

    lama_mask = np.where((fg_u8 >= 10) | (mid_u8 >= 10), np.uint8(255), np.uint8(0))
    lama_inner_img = lama.inpaint_background(tr_inner_img, lama_mask)
    lama_inner_img.save(
        scenes_dir / "scene13_herzliya_canopy_lama_inner.jpg", quality=cfg.jpeg_quality
    )
    lama_inner_rgb = np.asarray(lama_inner_img, dtype=np.uint8)

    seed_exp = cv2.copyMakeBorder(
        lama_inner_rgb, py, py, px, px, borderType=cv2.BORDER_REFLECT_101
    )
    bg_exp_rgb = _heal_expanded_collar_lama(lama_inner_rgb, seed_exp, None, lama)
    bg_exp_img = Image.fromarray(bg_exp_rgb, mode="RGB")
    bg_exp_img.save(
        scenes_dir / "scene13_herzliya_canopy_bg_inpainted.jpg", quality=cfg.jpeg_quality
    )

    raw_room_da2 = depth_est.predict_depth(bg_exp_img)
    raw_room_u8 = np.clip(np.round(raw_room_da2 * 255.0), 0, 255).astype(np.uint8)
    clean_room_u8 = clean_room_depth_under_cutouts(
        raw_room_u8, [fg_rgba, mid_rgba], cfg.room_depth_clean
    )
    Image.fromarray(clean_room_u8, mode="L").save(scenes_dir / "scene13_herzliya_canopy_depth.png")

    healed_win_norm = clean_room_u8[py : py + ih, px : px + iw].astype(np.float32) / 255.0
    tr_excess_pos = np.maximum(0.0, da2_tr.astype(np.float32) - healed_win_norm)

    mid_opaque = mid_u8 >= cfg.mid_contact_solve.alpha_threshold_u8
    mid_scale = max(1e-5, float(np.percentile(tr_excess_pos[mid_opaque], 98.0)))
    mid_prot_norm = np.clip(tr_excess_pos / mid_scale, 0.0, 1.0).astype(np.float32)

    fg_opaque = fg_u8 >= cfg.fg_contact_solve.alpha_threshold_u8
    fg_scale = max(1e-5, float(np.percentile(tr_excess_pos[fg_opaque], 98.0)))
    fg_prot_norm = np.clip(tr_excess_pos / fg_scale, 0.0, 1.0).astype(np.float32)

    mid_solve = solve_contact_anchored_layer_depth(
        room_depth_expanded=clean_room_u8,
        cutout_rgba=mid_rgba,
        cfg=cfg.mid_contact_solve,
        neural_protrusion_norm=mid_prot_norm,
    )
    fg_solve = solve_contact_anchored_layer_depth(
        room_depth_expanded=clean_room_u8,
        cutout_rgba=fg_rgba,
        cfg=cfg.fg_contact_solve,
        neural_protrusion_norm=fg_prot_norm,
    )
    mid_depth_u8 = mid_solve.depth_u8
    fg_depth_u8 = fg_solve.depth_u8
    Image.fromarray(mid_depth_u8, mode="L").save(
        scenes_dir / "scene13_herzliya_canopy_mid_depth.png"
    )
    Image.fromarray(fg_depth_u8, mode="L").save(
        scenes_dir / "scene13_herzliya_canopy_fg_depth.png"
    )

    mid_report = verify_layer_contact_and_discontinuity(
        room_depth_expanded=clean_room_u8,
        layer_rgba=mid_rgba,
        layer_depth=mid_depth_u8,
        contact_mask=mid_solve.contact_mask,
        bg_z=cfg.mid_contact_solve.bg_z,
        room_depth_amplitude=cfg.mid_contact_solve.room_depth_amplitude,
        room_pin_margin_uv=cfg.mid_contact_solve.room_pin_margin_uv,
        layer_z=cfg.mid_contact_solve.layer_z,
        layer_depth_amplitude=cfg.mid_contact_solve.layer_depth_amplitude,
        layer_pin_margin_uv=cfg.mid_contact_solve.layer_pin_margin_uv,
        verify_cfg=cfg.topology_verification,
    )
    fg_report = verify_layer_contact_and_discontinuity(
        room_depth_expanded=clean_room_u8,
        layer_rgba=fg_rgba,
        layer_depth=fg_depth_u8,
        contact_mask=fg_solve.contact_mask,
        bg_z=cfg.fg_contact_solve.bg_z,
        room_depth_amplitude=cfg.fg_contact_solve.room_depth_amplitude,
        room_pin_margin_uv=cfg.fg_contact_solve.room_pin_margin_uv,
        layer_z=cfg.fg_contact_solve.layer_z,
        layer_depth_amplitude=cfg.fg_contact_solve.layer_depth_amplitude,
        layer_pin_margin_uv=cfg.fg_contact_solve.layer_pin_margin_uv,
        verify_cfg=cfg.topology_verification,
    )
    if not mid_report.all_passed or not fg_report.all_passed:
        raise RuntimeError(
            f"Tree canopy failed depth topology gate: mid={mid_report.to_dict()}, fg={fg_report.to_dict()}"
        )

    return {
        "id": cfg.scene_id,
        "midCoverageRatio": round(float((mid_u8 >= 20).mean()), 4),
        "fgCoverageRatio": round(float((fg_u8 >= 20).mean()), 4),
        "topologyVerification": {
            "mid": mid_report.to_dict(),
            "fg": fg_report.to_dict(),
        },
    }


def run_tzofen_tree_pipeline(
    bundle_cfg: TzofenTreeBundleConfig | None = None,
    specs_out_path: Path | None = None,
    only_tzofen: bool = False,
) -> dict[str, Any]:
    cfg = bundle_cfg or default_tzofen_tree_bundle_config()
    roundtrip = TzofenTreeBundleConfig.from_json(cfg.to_json())
    if roundtrip != cfg:
        raise RuntimeError("TzofenTreeBundleConfig failed lossless JSON round-trip!")

    repo_root = _REPO_ROOT
    scenes_dir = repo_root / "prototype" / "assets" / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    device = select_torch_device()
    print(f"[tzofen_tree_scenes_builder] Using torch device: {device}", flush=True)
    depth_est = DepthAnythingEstimator(cfg.depth_model, device=device)
    matte_ext = SalientMatteExtractor(cfg.segmentation)
    sam_ext = SamBoxMaskExtractor(cfg.segmentation)
    lama = LamaOcclusionInpainter(cfg.inpainting)

    print("[tzofen_tree_scenes_builder] Extracting Tzofen unified cutout & room plate...", flush=True)
    tz_pad, unified_fg_rgba, da2_excess, lama_inner_rgb, bg_exp_rgb, clean_room_u8 = (
        _extract_unified_tzofen_cutout_and_room(
            cfg.tzofen, repo_root, depth_est, matte_ext, sam_ext, lama
        )
    )

    print("[tzofen_tree_scenes_builder] Building Scene 12 (unified): tzofen_brick_table...", flush=True)
    tzofen_meta = build_tzofen_unified_scene(
        cfg=cfg.tzofen,
        scenes_dir=scenes_dir,
        tz_pad=tz_pad,
        unified_fg_rgba=unified_fg_rgba,
        da2_excess=da2_excess,
        lama_inner_rgb=lama_inner_rgb,
        bg_exp_rgb=bg_exp_rgb,
        clean_room_u8=clean_room_u8,
    )
    print(f"[tzofen_tree_scenes_builder] Tzofen unified done: {tzofen_meta}", flush=True)

    print("[tzofen_tree_scenes_builder] Building Scene 12 v2 (archived): tzofen_brick_table_v2...", flush=True)
    tzofen_v2_meta = build_tzofen_segmented_v2_scene(
        base_cfg=cfg.tzofen,
        v2_cfg=cfg.tzofen_v2,
        scenes_dir=scenes_dir,
        tz_pad=tz_pad,
        unified_fg_rgba=unified_fg_rgba,
        da2_excess=da2_excess,
        lama_inner_rgb=lama_inner_rgb,
        bg_exp_rgb=bg_exp_rgb,
        clean_room_u8=clean_room_u8,
        sam_ext=sam_ext,
        lama=lama,
    )
    print(f"[tzofen_tree_scenes_builder] Tzofen v2 done: {tzofen_v2_meta}", flush=True)

    scenes_payload: dict[str, Any] = {
        "tzofen_brick_table": tzofen_meta,
        "tzofen_brick_table_v2": tzofen_v2_meta,
    }
    if not only_tzofen:
        print("[tzofen_tree_scenes_builder] Building Scene 13: herzliya_tree_canopy...", flush=True)
        tree_meta = build_tree_canopy_scene(
            cfg.tree_canopy, repo_root, scenes_dir, depth_est, sam_ext, lama
        )
        print(f"[tzofen_tree_scenes_builder] Tree canopy done: {tree_meta}", flush=True)
        scenes_payload["herzliya_tree_canopy"] = tree_meta

    summary = {
        "schemaVersion": "1.0",
        "generatedBy": "prototype/tzofen_tree_scenes_builder.py",
        "scenes": scenes_payload,
    }
    if specs_out_path is not None:
        specs_out_path.parent.mkdir(parents=True, exist_ok=True)
        specs_out_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Tzofen & Herzliya Tree Canopy scenes.")
    parser.add_argument(
        "--specs-out",
        type=Path,
        default=None,
        help="Optional JSON path to write build metrics & portal coordinates.",
    )
    parser.add_argument(
        "--only-tzofen",
        action="store_true",
        help="Rebuild only Scene 12 + Scene 12 v2 while keeping Scene 13 intact.",
    )
    args = parser.parse_args()
    run_tzofen_tree_pipeline(specs_out_path=args.specs_out, only_tzofen=bool(args.only_tzofen))


if __name__ == "__main__":
    main()
