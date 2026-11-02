#!/usr/bin/env python3
"""
prototype/tzofen_tree_scenes_builder.py
Offline neural 2.5D spatial asset builder and typed SSOT configuration models for Feedback #36:
  1. Scene 12 (`tzofen_brick_table` / `scene12_tzofen_*`):
     Tamar "Tami" Sheaffer's architectural installation *"צופן"* (2011, `IMG_BINY_0019`) —
     white steel multi-tier table bearing 380 terracotta mud-bricks, with $C^0$ floor-contact
     continuity across all 7 supporting steel legs/plinths, $C^{-1}$ Voronoi EDT silhouette
     discontinuity, LaMa-inpainted Binyamin Gallery backdrop, and an open portal window in
     the upper-right white steel frame looking into `herzliya_tree_canopy`.
  2. Scene 13 (`herzliya_tree_canopy` / `scene13_herzliya_canopy_*`):
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
from dataclasses import asdict, dataclass
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
    compute_world_z_map,
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
class TzofenBrickSegmentationConfig:
    """SSOT parameters separating warm terracotta mud-bricks (`fg`) from white steel (`mid`)."""

    brick_sam_prompts: tuple[SamPointGroupPrompt, ...]
    tabletop_split_y_px: int
    upper_sat_min: float
    upper_b_star_min: float
    upper_hue_max: float
    upper_r_minus_b_min: float
    upper_shadow_val_max: float
    upper_shadow_r_minus_b_min: float
    upper_shadow_sat_min: float
    upper_shadow_hue_max: float
    lower_core_sat_min: float
    lower_core_a_star_min: float
    lower_core_b_star_min: float
    lower_core_hue_max: float
    lower_core_r_minus_b_min: float
    lower_warm_sat_min: float
    lower_warm_a_star_min: float
    lower_warm_b_star_min: float
    lower_warm_hue_max: float
    lower_shadow_dilate_px: int
    lower_shadow_val_max: float
    lower_shadow_sat_min: float
    lower_shadow_r_minus_b_min: float
    lower_shadow_hue_max: float
    dark_recessed_y_max_px: int
    dark_recessed_val_max: float
    dark_recessed_sat_min: float
    dark_recessed_hue_max: float
    dark_recessed_r_over_g_min: float
    dark_recessed_r_minus_b_min: float
    sam_brick_gate_sat_min: float
    sam_brick_gate_r_minus_b_min: float
    sam_brick_gate_luma_max: float
    rim_pale_luma_min: float
    rim_pale_sat_max: float
    rim_neutral_r_minus_b_max: float
    rim_neutral_luma_min: float
    rim_r_over_g_min: float
    aa_erode_bias_px: float
    aa_half_width_px: float
    decontam_band_px: float
    decontam_interior_dist_px: float
    decontam_core_sat_min: float
    decontam_core_r_over_g_min: float
    decontam_min_luma_ratio: float
    decontam_outward_pad_px: float
    warm_stain_r_minus_b_min: float
    warm_stain_sat_min: float
    warm_stain_hue_max: float
    baseboard_exclude_box: BoundingBoxPrompt
    left_vent_exclude_box: BoundingBoxPrompt
    wall_vent_boxes: tuple[BoundingBoxPrompt, ...]
    upper_steel_boxes: tuple[BoundingBoxPrompt, ...]
    min_brick_component_px: int
    max_brick_pinhole_px: int
    brick_edge_feather_sigma: float
    brick_inpaint_dilate_px: int
    brick_inpaint_feather_px: float
    tabletop_deck_x_range_px: tuple[int, int]
    tabletop_deck_y_range_px: tuple[int, int]
    mid_brick_exclusion_dilate_px: int
    mid_steel_max_sat: float
    mid_steel_max_r_minus_b: float
    mid_steel_max_r_over_g: float
    mid_steel_min_luma: float
    mid_recessed_shelf_y_max_px: int
    mid_recessed_steel_min_luma: float
    mid_recessed_steel_max_sat: float
    mid_recessed_steel_max_r_over_g: float
    mid_horiz_shelf_bridge_px: int
    mid_vert_divider_bridge_px: int
    mid_vert_bridge_min_y_px: int
    mid_max_pinhole_px: int
    bg_occ_dilate_px: int
    bg_wall_donor_min_luma: float
    bg_wall_donor_max_sat: float
    bg_floor_shadow_luma_min: float
    bg_floor_shadow_dilate_px: int
    bg_floor_donor_max_sat: float
    bg_floor_donor_max_abs_rb: float
    bg_dc_harmonize_sigma_px: float
    bg_dc_blend_band_px: float
    mirror_authentic_collar: bool
    healed_steel_max_r_minus_b: float
    healed_steel_min_luma: float
    healed_steel_max_wall_luma: float
    brick_height_relief_weight: float
    brick_height_saturation_px: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "brick_sam_prompts": [p.to_dict() for p in self.brick_sam_prompts],
            "tabletop_split_y_px": self.tabletop_split_y_px,
            "upper_sat_min": self.upper_sat_min,
            "upper_b_star_min": self.upper_b_star_min,
            "upper_hue_max": self.upper_hue_max,
            "upper_r_minus_b_min": self.upper_r_minus_b_min,
            "upper_shadow_val_max": self.upper_shadow_val_max,
            "upper_shadow_r_minus_b_min": self.upper_shadow_r_minus_b_min,
            "upper_shadow_sat_min": self.upper_shadow_sat_min,
            "upper_shadow_hue_max": self.upper_shadow_hue_max,
            "lower_core_sat_min": self.lower_core_sat_min,
            "lower_core_a_star_min": self.lower_core_a_star_min,
            "lower_core_b_star_min": self.lower_core_b_star_min,
            "lower_core_hue_max": self.lower_core_hue_max,
            "lower_core_r_minus_b_min": self.lower_core_r_minus_b_min,
            "lower_warm_sat_min": self.lower_warm_sat_min,
            "lower_warm_a_star_min": self.lower_warm_a_star_min,
            "lower_warm_b_star_min": self.lower_warm_b_star_min,
            "lower_warm_hue_max": self.lower_warm_hue_max,
            "lower_shadow_dilate_px": self.lower_shadow_dilate_px,
            "lower_shadow_val_max": self.lower_shadow_val_max,
            "lower_shadow_sat_min": self.lower_shadow_sat_min,
            "lower_shadow_r_minus_b_min": self.lower_shadow_r_minus_b_min,
            "lower_shadow_hue_max": self.lower_shadow_hue_max,
            "dark_recessed_y_max_px": self.dark_recessed_y_max_px,
            "dark_recessed_val_max": self.dark_recessed_val_max,
            "dark_recessed_sat_min": self.dark_recessed_sat_min,
            "dark_recessed_hue_max": self.dark_recessed_hue_max,
            "dark_recessed_r_over_g_min": self.dark_recessed_r_over_g_min,
            "dark_recessed_r_minus_b_min": self.dark_recessed_r_minus_b_min,
            "sam_brick_gate_sat_min": self.sam_brick_gate_sat_min,
            "sam_brick_gate_r_minus_b_min": self.sam_brick_gate_r_minus_b_min,
            "sam_brick_gate_luma_max": self.sam_brick_gate_luma_max,
            "rim_pale_luma_min": self.rim_pale_luma_min,
            "rim_pale_sat_max": self.rim_pale_sat_max,
            "rim_neutral_r_minus_b_max": self.rim_neutral_r_minus_b_max,
            "rim_neutral_luma_min": self.rim_neutral_luma_min,
            "rim_r_over_g_min": self.rim_r_over_g_min,
            "aa_erode_bias_px": self.aa_erode_bias_px,
            "aa_half_width_px": self.aa_half_width_px,
            "decontam_band_px": self.decontam_band_px,
            "decontam_interior_dist_px": self.decontam_interior_dist_px,
            "decontam_core_sat_min": self.decontam_core_sat_min,
            "decontam_core_r_over_g_min": self.decontam_core_r_over_g_min,
            "decontam_min_luma_ratio": self.decontam_min_luma_ratio,
            "decontam_outward_pad_px": self.decontam_outward_pad_px,
            "warm_stain_r_minus_b_min": self.warm_stain_r_minus_b_min,
            "warm_stain_sat_min": self.warm_stain_sat_min,
            "warm_stain_hue_max": self.warm_stain_hue_max,
            "baseboard_exclude_box": self.baseboard_exclude_box.to_dict(),
            "left_vent_exclude_box": self.left_vent_exclude_box.to_dict(),
            "wall_vent_boxes": [b.to_dict() for b in self.wall_vent_boxes],
            "upper_steel_boxes": [b.to_dict() for b in self.upper_steel_boxes],
            "min_brick_component_px": self.min_brick_component_px,
            "max_brick_pinhole_px": self.max_brick_pinhole_px,
            "brick_edge_feather_sigma": self.brick_edge_feather_sigma,
            "brick_inpaint_dilate_px": self.brick_inpaint_dilate_px,
            "brick_inpaint_feather_px": self.brick_inpaint_feather_px,
            "tabletop_deck_x_range_px": list(self.tabletop_deck_x_range_px),
            "tabletop_deck_y_range_px": list(self.tabletop_deck_y_range_px),
            "mid_brick_exclusion_dilate_px": self.mid_brick_exclusion_dilate_px,
            "mid_steel_max_sat": self.mid_steel_max_sat,
            "mid_steel_max_r_minus_b": self.mid_steel_max_r_minus_b,
            "mid_steel_max_r_over_g": self.mid_steel_max_r_over_g,
            "mid_steel_min_luma": self.mid_steel_min_luma,
            "mid_recessed_shelf_y_max_px": self.mid_recessed_shelf_y_max_px,
            "mid_recessed_steel_min_luma": self.mid_recessed_steel_min_luma,
            "mid_recessed_steel_max_sat": self.mid_recessed_steel_max_sat,
            "mid_recessed_steel_max_r_over_g": self.mid_recessed_steel_max_r_over_g,
            "mid_horiz_shelf_bridge_px": self.mid_horiz_shelf_bridge_px,
            "mid_vert_divider_bridge_px": self.mid_vert_divider_bridge_px,
            "mid_vert_bridge_min_y_px": self.mid_vert_bridge_min_y_px,
            "mid_max_pinhole_px": self.mid_max_pinhole_px,
            "bg_occ_dilate_px": self.bg_occ_dilate_px,
            "bg_wall_donor_min_luma": self.bg_wall_donor_min_luma,
            "bg_wall_donor_max_sat": self.bg_wall_donor_max_sat,
            "bg_floor_shadow_luma_min": self.bg_floor_shadow_luma_min,
            "bg_floor_shadow_dilate_px": self.bg_floor_shadow_dilate_px,
            "bg_floor_donor_max_sat": self.bg_floor_donor_max_sat,
            "bg_floor_donor_max_abs_rb": self.bg_floor_donor_max_abs_rb,
            "bg_dc_harmonize_sigma_px": self.bg_dc_harmonize_sigma_px,
            "bg_dc_blend_band_px": self.bg_dc_blend_band_px,
            "mirror_authentic_collar": self.mirror_authentic_collar,
            "healed_steel_max_r_minus_b": self.healed_steel_max_r_minus_b,
            "healed_steel_min_luma": self.healed_steel_min_luma,
            "healed_steel_max_wall_luma": self.healed_steel_max_wall_luma,
            "brick_height_relief_weight": self.brick_height_relief_weight,
            "brick_height_saturation_px": self.brick_height_saturation_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TzofenBrickSegmentationConfig:
        tdx = data["tabletop_deck_x_range_px"]
        tdy = data["tabletop_deck_y_range_px"]
        return cls(
            brick_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["brick_sam_prompts"]
            ),
            tabletop_split_y_px=int(data["tabletop_split_y_px"]),
            upper_sat_min=float(data["upper_sat_min"]),
            upper_b_star_min=float(data["upper_b_star_min"]),
            upper_hue_max=float(data["upper_hue_max"]),
            upper_r_minus_b_min=float(data["upper_r_minus_b_min"]),
            upper_shadow_val_max=float(data["upper_shadow_val_max"]),
            upper_shadow_r_minus_b_min=float(data["upper_shadow_r_minus_b_min"]),
            upper_shadow_sat_min=float(data["upper_shadow_sat_min"]),
            upper_shadow_hue_max=float(data["upper_shadow_hue_max"]),
            lower_core_sat_min=float(data["lower_core_sat_min"]),
            lower_core_a_star_min=float(data["lower_core_a_star_min"]),
            lower_core_b_star_min=float(data["lower_core_b_star_min"]),
            lower_core_hue_max=float(data["lower_core_hue_max"]),
            lower_core_r_minus_b_min=float(data["lower_core_r_minus_b_min"]),
            lower_warm_sat_min=float(data["lower_warm_sat_min"]),
            lower_warm_a_star_min=float(data["lower_warm_a_star_min"]),
            lower_warm_b_star_min=float(data["lower_warm_b_star_min"]),
            lower_warm_hue_max=float(data["lower_warm_hue_max"]),
            lower_shadow_dilate_px=int(data["lower_shadow_dilate_px"]),
            lower_shadow_val_max=float(data["lower_shadow_val_max"]),
            lower_shadow_sat_min=float(data["lower_shadow_sat_min"]),
            lower_shadow_r_minus_b_min=float(data["lower_shadow_r_minus_b_min"]),
            lower_shadow_hue_max=float(data["lower_shadow_hue_max"]),
            dark_recessed_y_max_px=int(data["dark_recessed_y_max_px"]),
            dark_recessed_val_max=float(data["dark_recessed_val_max"]),
            dark_recessed_sat_min=float(data["dark_recessed_sat_min"]),
            dark_recessed_hue_max=float(data["dark_recessed_hue_max"]),
            dark_recessed_r_over_g_min=float(data["dark_recessed_r_over_g_min"]),
            dark_recessed_r_minus_b_min=float(data["dark_recessed_r_minus_b_min"]),
            sam_brick_gate_sat_min=float(data["sam_brick_gate_sat_min"]),
            sam_brick_gate_r_minus_b_min=float(data["sam_brick_gate_r_minus_b_min"]),
            sam_brick_gate_luma_max=float(data["sam_brick_gate_luma_max"]),
            rim_pale_luma_min=float(data["rim_pale_luma_min"]),
            rim_pale_sat_max=float(data["rim_pale_sat_max"]),
            rim_neutral_r_minus_b_max=float(data["rim_neutral_r_minus_b_max"]),
            rim_neutral_luma_min=float(data["rim_neutral_luma_min"]),
            rim_r_over_g_min=float(data["rim_r_over_g_min"]),
            aa_erode_bias_px=float(data["aa_erode_bias_px"]),
            aa_half_width_px=float(data["aa_half_width_px"]),
            decontam_band_px=float(data["decontam_band_px"]),
            decontam_interior_dist_px=float(data["decontam_interior_dist_px"]),
            decontam_core_sat_min=float(data["decontam_core_sat_min"]),
            decontam_core_r_over_g_min=float(data["decontam_core_r_over_g_min"]),
            decontam_min_luma_ratio=float(data["decontam_min_luma_ratio"]),
            decontam_outward_pad_px=float(data["decontam_outward_pad_px"]),
            warm_stain_r_minus_b_min=float(data["warm_stain_r_minus_b_min"]),
            warm_stain_sat_min=float(data["warm_stain_sat_min"]),
            warm_stain_hue_max=float(data["warm_stain_hue_max"]),
            baseboard_exclude_box=BoundingBoxPrompt.from_dict(data["baseboard_exclude_box"]),
            left_vent_exclude_box=BoundingBoxPrompt.from_dict(data["left_vent_exclude_box"]),
            wall_vent_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["wall_vent_boxes"]
            ),
            upper_steel_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["upper_steel_boxes"]
            ),
            min_brick_component_px=int(data["min_brick_component_px"]),
            max_brick_pinhole_px=int(data["max_brick_pinhole_px"]),
            brick_edge_feather_sigma=float(data["brick_edge_feather_sigma"]),
            brick_inpaint_dilate_px=int(data["brick_inpaint_dilate_px"]),
            brick_inpaint_feather_px=float(data["brick_inpaint_feather_px"]),
            tabletop_deck_x_range_px=(int(tdx[0]), int(tdx[1])),
            tabletop_deck_y_range_px=(int(tdy[0]), int(tdy[1])),
            mid_brick_exclusion_dilate_px=int(data["mid_brick_exclusion_dilate_px"]),
            mid_steel_max_sat=float(data["mid_steel_max_sat"]),
            mid_steel_max_r_minus_b=float(data["mid_steel_max_r_minus_b"]),
            mid_steel_max_r_over_g=float(data["mid_steel_max_r_over_g"]),
            mid_steel_min_luma=float(data["mid_steel_min_luma"]),
            mid_recessed_shelf_y_max_px=int(data["mid_recessed_shelf_y_max_px"]),
            mid_recessed_steel_min_luma=float(data["mid_recessed_steel_min_luma"]),
            mid_recessed_steel_max_sat=float(data["mid_recessed_steel_max_sat"]),
            mid_recessed_steel_max_r_over_g=float(data["mid_recessed_steel_max_r_over_g"]),
            mid_horiz_shelf_bridge_px=int(data["mid_horiz_shelf_bridge_px"]),
            mid_vert_divider_bridge_px=int(data["mid_vert_divider_bridge_px"]),
            mid_vert_bridge_min_y_px=int(data["mid_vert_bridge_min_y_px"]),
            mid_max_pinhole_px=int(data["mid_max_pinhole_px"]),
            bg_occ_dilate_px=int(data["bg_occ_dilate_px"]),
            bg_wall_donor_min_luma=float(data["bg_wall_donor_min_luma"]),
            bg_wall_donor_max_sat=float(data["bg_wall_donor_max_sat"]),
            bg_floor_shadow_luma_min=float(data["bg_floor_shadow_luma_min"]),
            bg_floor_shadow_dilate_px=int(data["bg_floor_shadow_dilate_px"]),
            bg_floor_donor_max_sat=float(data["bg_floor_donor_max_sat"]),
            bg_floor_donor_max_abs_rb=float(data["bg_floor_donor_max_abs_rb"]),
            bg_dc_harmonize_sigma_px=float(data["bg_dc_harmonize_sigma_px"]),
            bg_dc_blend_band_px=float(data["bg_dc_blend_band_px"]),
            mirror_authentic_collar=bool(data["mirror_authentic_collar"]),
            healed_steel_max_r_minus_b=float(data["healed_steel_max_r_minus_b"]),
            healed_steel_min_luma=float(data["healed_steel_min_luma"]),
            healed_steel_max_wall_luma=float(data["healed_steel_max_wall_luma"]),
            brick_height_relief_weight=float(data["brick_height_relief_weight"]),
            brick_height_saturation_px=float(data["brick_height_saturation_px"]),
        )


@dataclass(frozen=True)
class TzofenSceneBuildConfig:
    scene_id: str
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
    mid_depth_amplitude: float
    fg_depth_amplitude: float
    room_pin_margin_uv: float
    window_portal_inner_box_px: tuple[float, float, float, float]
    window_frame_sample_box_px: tuple[int, int, int, int]
    portal_z_clearance_behind_frame: float
    portal_recess_depth: float
    sam_prompts: tuple[SamPointGroupPrompt, ...]
    bg_seed_boxes: tuple[BoundingBoxPrompt, ...]
    brick_segmentation: TzofenBrickSegmentationConfig
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
            "scaled_height_px": self.scaled_height_px,
            "inner_y_offset_px": self.inner_y_offset_px,
            "baseboard_inner_y": self.baseboard_inner_y,
            "jpeg_quality": self.jpeg_quality,
            "bg_z": self.bg_z,
            "mid_z": self.mid_z,
            "fg_z": self.fg_z,
            "room_depth_amplitude": self.room_depth_amplitude,
            "mid_depth_amplitude": self.mid_depth_amplitude,
            "fg_depth_amplitude": self.fg_depth_amplitude,
            "room_pin_margin_uv": self.room_pin_margin_uv,
            "window_portal_inner_box_px": list(self.window_portal_inner_box_px),
            "window_frame_sample_box_px": list(self.window_frame_sample_box_px),
            "portal_z_clearance_behind_frame": self.portal_z_clearance_behind_frame,
            "portal_recess_depth": self.portal_recess_depth,
            "sam_prompts": [p.to_dict() for p in self.sam_prompts],
            "bg_seed_boxes": [b.to_dict() for b in self.bg_seed_boxes],
            "brick_segmentation": self.brick_segmentation.to_dict(),
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_contact_solve": self.mid_contact_solve.to_dict(),
            "fg_contact_solve": self.fg_contact_solve.to_dict(),
            "topology_verification": self.topology_verification.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TzofenSceneBuildConfig:
        wp = data["window_portal_inner_box_px"]
        wf = data["window_frame_sample_box_px"]
        return cls(
            scene_id=str(data["scene_id"]),
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
            mid_depth_amplitude=float(data["mid_depth_amplitude"]),
            fg_depth_amplitude=float(data["fg_depth_amplitude"]),
            room_pin_margin_uv=float(data["room_pin_margin_uv"]),
            window_portal_inner_box_px=(float(wp[0]), float(wp[1]), float(wp[2]), float(wp[3])),
            window_frame_sample_box_px=(int(wf[0]), int(wf[1]), int(wf[2]), int(wf[3])),
            portal_z_clearance_behind_frame=float(data["portal_z_clearance_behind_frame"]),
            portal_recess_depth=float(data["portal_recess_depth"]),
            sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["sam_prompts"]),
            bg_seed_boxes=tuple(BoundingBoxPrompt.from_dict(b) for b in data["bg_seed_boxes"]),
            brick_segmentation=TzofenBrickSegmentationConfig.from_dict(
                data["brick_segmentation"]
            ),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
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
    tree_canopy: TreeCanopySceneBuildConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_model": self.depth_model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "inpainting": self.inpainting.to_dict(),
            "tzofen": self.tzofen.to_dict(),
            "tree_canopy": self.tree_canopy.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TzofenTreeBundleConfig:
        return cls(
            depth_model=DepthModelConfig.from_dict(data["depth_model"]),
            segmentation=SegmentationConfig.from_dict(data["segmentation"]),
            inpainting=InpaintingConfig.from_dict(data["inpainting"]),
            tzofen=TzofenSceneBuildConfig.from_dict(data["tzofen"]),
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

    tzofen_brick_prompts = (
        _pg(
            (775, 588, 864, 666),
            [(800, 618), (828, 632), (848, 645), (792, 648)],
            [(768, 625), (870, 625), (820, 582), (820, 672)],
        ),
        _pg(
            (562, 594, 676, 775),
            [(602, 632), (590, 725), (656, 725), (650, 635)],
            [(636, 635), (636, 725), (556, 650), (682, 725)],
        ),
        _pg(
            (904, 586, 1018, 714),
            [(955, 625), (955, 675), (998, 655)],
            [(895, 650), (1026, 650), (955, 722)],
        ),
        _pg(
            (996, 732, 1112, 826),
            [(1050, 775), (1085, 785)],
            [(985, 785), (1120, 785), (1050, 834)],
        ),
        _pg(
            (1160, 798, 1268, 922),
            [(1215, 845), (1215, 895)],
            [(1150, 860), (1276, 860), (1215, 930)],
        ),
        _pg(
            (1400, 500, 1468, 590),
            [(1435, 535), (1435, 568)],
            [(1390, 545), (1474, 545), (1435, 596)],
        ),
        _pg(
            (376, 232, 550, 422),
            [(435, 295), (505, 315), (505, 385)],
            [(365, 355), (558, 355), (465, 224), (415, 365)],
        ),
        _pg(
            (368, 496, 442, 568),
            [(395, 525), (420, 542)],
            [(360, 530), (452, 530), (405, 576)],
        ),
        _pg(
            (494, 496, 550, 572),
            [(515, 528), (532, 550)],
            [(482, 535), (560, 535), (520, 580)],
        ),
        _pg(
            (600, 496, 698, 540),
            [(618, 514), (672, 516)],
            [(590, 516), (638, 516), (705, 545), (650, 548)],
        ),
        _pg(
            (700, 496, 784, 568),
            [(728, 525), (758, 548)],
            [(690, 540), (792, 535), (715, 573), (742, 573), (768, 573), (668, 562)],
        ),
        _pg(
            (850, 496, 916, 566),
            [(860, 512), (898, 525), (900, 552)],
            [(838, 525), (846, 540), (876, 525), (876, 555), (924, 535), (898, 573)],
        ),
        _pg(
            (170, 558, 216, 698),
            [(188, 595), (192, 655)],
            [(162, 625), (224, 625), (192, 708)],
        ),
    )

    tzofen_cfg = TzofenSceneBuildConfig(
        scene_id="tzofen_brick_table",
        source_image_relpath="media/binyamin_gallery/binyamin_צופן.jpg_afbc9c55.jpg",
        inner_dims=inner,
        expanded_dims=expanded,
        scaled_height_px=919,
        inner_y_offset_px=42,
        baseboard_inner_y=484,
        jpeg_quality=93,
        bg_z=-340.0,
        mid_z=-340.0,
        fg_z=-340.0,
        room_depth_amplitude=155.0,
        mid_depth_amplitude=200.0,
        fg_depth_amplitude=230.0,
        room_pin_margin_uv=0.045,
        window_portal_inner_box_px=(975.0, 146.0, 1126.0, 250.0),
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
        brick_segmentation=TzofenBrickSegmentationConfig(
            brick_sam_prompts=tzofen_brick_prompts,
            tabletop_split_y_px=496,
            upper_sat_min=118.0,
            upper_b_star_min=14.0,
            upper_hue_max=25.0,
            upper_r_minus_b_min=32.0,
            upper_shadow_val_max=115.0,
            upper_shadow_r_minus_b_min=26.0,
            upper_shadow_sat_min=128.0,
            upper_shadow_hue_max=26.0,
            lower_core_sat_min=135.0,
            lower_core_a_star_min=7.5,
            lower_core_b_star_min=16.0,
            lower_core_hue_max=25.0,
            lower_core_r_minus_b_min=32.0,
            lower_warm_sat_min=135.0,
            lower_warm_a_star_min=11.0,
            lower_warm_b_star_min=18.0,
            lower_warm_hue_max=25.0,
            lower_shadow_dilate_px=15,
            lower_shadow_val_max=112.0,
            lower_shadow_sat_min=135.0,
            lower_shadow_r_minus_b_min=24.0,
            lower_shadow_hue_max=26.0,
            dark_recessed_y_max_px=785,
            dark_recessed_val_max=105.0,
            dark_recessed_sat_min=158.0,
            dark_recessed_hue_max=24.0,
            dark_recessed_r_over_g_min=1.38,
            dark_recessed_r_minus_b_min=16.0,
            sam_brick_gate_sat_min=145.0,
            sam_brick_gate_r_minus_b_min=20.0,
            sam_brick_gate_luma_max=176.0,
            rim_pale_luma_min=182.0,
            rim_pale_sat_max=110.0,
            rim_neutral_r_minus_b_max=22.0,
            rim_neutral_luma_min=85.0,
            rim_r_over_g_min=1.36,
            aa_erode_bias_px=0.35,
            aa_half_width_px=0.60,
            decontam_band_px=5.0,
            decontam_interior_dist_px=4.0,
            decontam_core_sat_min=155.0,
            decontam_core_r_over_g_min=1.38,
            decontam_min_luma_ratio=0.35,
            decontam_outward_pad_px=4.0,
            warm_stain_r_minus_b_min=28.0,
            warm_stain_sat_min=90.0,
            warm_stain_hue_max=26.0,
            baseboard_exclude_box=BoundingBoxPrompt(1470, 474, 1500, 502),
            left_vent_exclude_box=BoundingBoxPrompt(0, 0, 155, 467),
            wall_vent_boxes=(
                BoundingBoxPrompt(0, 326, 170, 382),
                BoundingBoxPrompt(0, 382, 28, 468),
            ),
            upper_steel_boxes=(
                BoundingBoxPrompt(152, 378, 210, 468),
                BoundingBoxPrompt(242, 134, 398, 438),
                BoundingBoxPrompt(370, 314, 468, 438),
                BoundingBoxPrompt(534, 368, 648, 438),
                BoundingBoxPrompt(670, 194, 778, 438),
                BoundingBoxPrompt(770, 240, 1148, 438),
                BoundingBoxPrompt(960, 136, 1144, 258),
                BoundingBoxPrompt(1095, 316, 1218, 482),
            ),
            min_brick_component_px=45,
            max_brick_pinhole_px=150,
            brick_edge_feather_sigma=0.45,
            brick_inpaint_dilate_px=28,
            brick_inpaint_feather_px=5.0,
            tabletop_deck_x_range_px=(18, 1474),
            tabletop_deck_y_range_px=(440, 497),
            mid_brick_exclusion_dilate_px=0,
            mid_steel_max_sat=135.0,
            mid_steel_max_r_minus_b=85.0,
            mid_steel_max_r_over_g=1.30,
            mid_steel_min_luma=58.0,
            mid_recessed_shelf_y_max_px=595,
            mid_recessed_steel_min_luma=32.0,
            mid_recessed_steel_max_sat=160.0,
            mid_recessed_steel_max_r_over_g=1.35,
            mid_horiz_shelf_bridge_px=180,
            mid_vert_divider_bridge_px=160,
            mid_vert_bridge_min_y_px=497,
            mid_max_pinhole_px=48,
            bg_occ_dilate_px=28,
            bg_wall_donor_min_luma=225.0,
            bg_wall_donor_max_sat=45.0,
            bg_floor_shadow_luma_min=174.0,
            bg_floor_shadow_dilate_px=12,
            bg_floor_donor_max_sat=55.0,
            bg_floor_donor_max_abs_rb=28.0,
            bg_dc_harmonize_sigma_px=18.0,
            bg_dc_blend_band_px=5.0,
            mirror_authentic_collar=True,
            healed_steel_max_r_minus_b=18.0,
            healed_steel_min_luma=135.0,
            healed_steel_max_wall_luma=236.0,
            brick_height_relief_weight=0.58,
            brick_height_saturation_px=95.0,
        ),
        room_depth_clean=RoomDepthCleanConfig(
            alpha_threshold_u8=16,
            mask_dilation_px=28,
            post_smooth_sigma_px=4.0,
            interior_blend_band_px=14.0,
        ),
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
            contact_boxes=(
                ContactAnchorBox(x0=360, y0=616, x1=380, y1=636, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=1032, y0=624, x1=1054, y1=644, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=200, y0=955, x1=226, y1=999, mode="all_opaque", band_px=6),
                ContactAnchorBox(x0=430, y0=955, x1=510, y1=999, mode="all_opaque", band_px=6),
                ContactAnchorBox(x0=624, y0=938, x1=650, y1=960, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=850, y0=934, x1=876, y1=956, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=1242, y0=928, x1=1270, y1=950, mode="bottom_band", band_px=6),
            ),
            neural_relief_amplitude=24.0,
            contact_relief_taper_px=12.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.0,
            interior_smooth_sigma_px=3.5,
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
            component_bridge_px=2,
            min_component_area_px=30,
            auto_floor_band_px=5,
            auto_floor_max_rise_px=165,
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
        tree_canopy=tree_cfg,
    )


def _prune_small_alpha_components(
    alpha_u8: np.ndarray,
    min_area_px: int = 45,
    alpha_threshold_u8: int = 20,
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


def _segment_tzofen_bricks_alpha(
    tz_pad: np.ndarray,
    tz_pad_img: Image.Image,
    total_sculp_alpha_u8: np.ndarray,
    bcfg: TzofenBrickSegmentationConfig,
    portal_box_px: tuple[float, float, float, float],
    alpha_threshold_u8: int,
    sam_ext: SamBoxMaskExtractor,
) -> np.ndarray:
    """
    Segments warm terracotta mud-bricks (`fg`, including dark shadowed recessed bricks under the
    tabletop) cleanly from white/grey steel (`mid`) and gallery wall/floor (`bg`) with strict
    chromatic perimeter tightening and crisp `~1.2px` signed-EDT antialiasing (Feedback #45).
    """
    ih, iw = total_sculp_alpha_u8.shape
    sam_brick_u8 = sam_ext.segment_point_groups(
        tz_pad_img,
        bcfg.brick_sam_prompts,
        (),
        max_pinhole_fill_px=120,
        min_major_component_px=bcfg.min_brick_component_px,
    )

    hsv = cv2.cvtColor(tz_pad, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(tz_pad, cv2.COLOR_RGB2LAB).astype(np.float32)
    rgb_f = tz_pad.astype(np.float32)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    a_star = lab[:, :, 1] - 128.0
    b_star = lab[:, :, 2] - 128.0
    r_minus_b = rgb_f[:, :, 0] - rgb_f[:, :, 2]
    r_minus_g = rgb_f[:, :, 0] - rgb_f[:, :, 1]
    r_over_g = rgb_f[:, :, 0] / np.maximum(1.0, rgb_f[:, :, 1])
    luma = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]

    yy = np.arange(ih)[:, None]
    sculp_mask = (total_sculp_alpha_u8 >= alpha_threshold_u8) | (sam_brick_u8 >= 128)
    upper = yy <= bcfg.tabletop_split_y_px
    lower = yy > bcfg.tabletop_split_y_px

    brick_upper = sculp_mask & upper & (r_over_g >= bcfg.rim_r_over_g_min) & (
        (
            (sat >= bcfg.upper_sat_min)
            & (b_star >= bcfg.upper_b_star_min)
            & (hue <= bcfg.upper_hue_max)
            & (r_minus_b >= bcfg.upper_r_minus_b_min)
        )
        | (
            (val < bcfg.upper_shadow_val_max)
            & (r_minus_b >= bcfg.upper_shadow_r_minus_b_min)
            & (sat >= bcfg.upper_shadow_sat_min)
            & (hue <= bcfg.upper_shadow_hue_max)
        )
    )
    brick_lower_core = sculp_mask & lower & (r_over_g >= bcfg.rim_r_over_g_min) & (
        (
            (sat >= bcfg.lower_core_sat_min)
            & (a_star >= bcfg.lower_core_a_star_min)
            & (b_star >= bcfg.lower_core_b_star_min)
            & (hue <= bcfg.lower_core_hue_max)
            & (r_minus_b >= bcfg.lower_core_r_minus_b_min)
        )
        | (
            (sat >= bcfg.lower_warm_sat_min)
            & (a_star >= bcfg.lower_warm_a_star_min)
            & (b_star >= bcfg.lower_warm_b_star_min)
            & (hue <= bcfg.lower_warm_hue_max)
        )
    )
    dark_recessed_chroma = (
        (yy <= bcfg.dark_recessed_y_max_px)
        & (val <= bcfg.dark_recessed_val_max)
        & (sat >= bcfg.dark_recessed_sat_min)
        & (hue <= bcfg.dark_recessed_hue_max)
        & (r_over_g >= bcfg.dark_recessed_r_over_g_min)
        & (r_minus_b >= bcfg.dark_recessed_r_minus_b_min)
    )
    sam_brick_bool = sculp_mask & (sam_brick_u8 >= 110) & (
        (
            (r_over_g >= bcfg.rim_r_over_g_min)
            & (r_minus_b >= bcfg.sam_brick_gate_r_minus_b_min)
            & (sat >= bcfg.sam_brick_gate_sat_min)
            & (luma <= bcfg.sam_brick_gate_luma_max)
        )
        | dark_recessed_chroma
    )

    kd = bcfg.lower_shadow_dilate_px
    if kd % 2 == 0:
        kd += 1
    lower_near_core = (
        cv2.dilate(
            (brick_lower_core | sam_brick_bool).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)),
        )
        > 0
    )
    brick_lower_shadow = (
        sculp_mask
        & lower
        & lower_near_core
        & (r_minus_g >= 10.0)
        & (r_over_g >= bcfg.rim_r_over_g_min)
        & (val <= bcfg.lower_shadow_val_max)
        & (sat >= bcfg.lower_shadow_sat_min)
        & (r_minus_b >= bcfg.lower_shadow_r_minus_b_min)
        & (hue <= bcfg.lower_shadow_hue_max)
    )

    bricks_bool = brick_upper | brick_lower_core | sam_brick_bool | brick_lower_shadow
    eb = bcfg.baseboard_exclude_box
    bricks_bool[eb.y0 : eb.y1, eb.x0 : eb.x1] = False
    wx0, wy0, wx1, wy1 = portal_box_px
    bricks_bool[int(np.floor(wy0)) : int(np.ceil(wy1)), int(np.floor(wx0)) : int(np.ceil(wx1))] = (
        False
    )

    # Strip any 2.5px outer-rim pixels mixed with white cubby steel or pale neutral wall/floor:
    dist_in_raw = ndi.distance_transform_edt(bricks_bool).astype(np.float32)
    contaminated_rim = (
        ((luma >= bcfg.rim_pale_luma_min) & (sat < bcfg.rim_pale_sat_max))
        | ((r_minus_b < bcfg.rim_neutral_r_minus_b_max) & (luma >= bcfg.rim_neutral_luma_min))
        | (r_over_g < bcfg.rim_r_over_g_min)
    )
    bricks_bool = bricks_bool & ~((dist_in_raw <= 2.5) & contaminated_rim)

    nl, lbls, stats, _ = cv2.connectedComponentsWithStats(
        bricks_bool.astype(np.uint8), connectivity=8
    )
    bricks_clean = np.zeros_like(bricks_bool)
    for cid in range(1, nl):
        if int(stats[cid, cv2.CC_STAT_AREA]) >= bcfg.min_brick_component_px:
            bricks_clean |= lbls == cid

    holes = (~bricks_clean).astype(np.uint8)
    nh, hlbls, hstats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    for cid in range(1, nh):
        area = int(hstats[cid, cv2.CC_STAT_AREA])
        hx0 = int(hstats[cid, cv2.CC_STAT_LEFT])
        hy0 = int(hstats[cid, cv2.CC_STAT_TOP])
        hx1 = hx0 + int(hstats[cid, cv2.CC_STAT_WIDTH])
        hy1 = hy0 + int(hstats[cid, cv2.CC_STAT_HEIGHT])
        if area <= bcfg.max_brick_pinhole_px and hx0 > 0 and hy0 > 0 and hx1 < iw and hy1 < ih:
            bricks_clean |= (hlbls == cid) & sculp_mask & ((r_over_g >= 1.24) | (luma < 85.0))

    # Crisp ~1.2px signed-EDT antialiased boundary slightly biased inward (`aa_erode_bias_px`):
    dist_in = ndi.distance_transform_edt(bricks_clean).astype(np.float32)
    dist_out = ndi.distance_transform_edt(~bricks_clean).astype(np.float32)
    sdf = dist_in - dist_out - bcfg.aa_erode_bias_px
    aa_u8 = np.clip(
        np.round(np.clip(0.5 + sdf / max(0.25, 2.0 * bcfg.aa_half_width_px), 0.0, 1.0) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    sculp_cap_u8 = np.maximum(total_sculp_alpha_u8, sam_brick_u8)
    raw_u8 = np.minimum(sculp_cap_u8, aa_u8)
    raw_u8[0, :] = 0
    raw_u8[-1, :] = 0
    raw_u8[:, 0] = 0
    raw_u8[:, -1] = 0
    return _prune_small_alpha_components(
        raw_u8,
        min_area_px=bcfg.min_brick_component_px,
        alpha_threshold_u8=alpha_threshold_u8,
    )


def _decontaminate_tzofen_brick_rgba(
    rgb_u8: np.ndarray,
    alpha_u8: np.ndarray,
    bcfg: TzofenBrickSegmentationConfig,
) -> np.ndarray:
    """
    Replaces RGB across the outer `decontam_band_px` (`5px`) boundary band of every terracotta
    brick component with the nearest pure interior terracotta pixel's RGB (`dist_in >= 4px`)
    scaled by local luminance ratio clamped `<= 1.0` (so bright white cubby/wall bleed NEVER
    brightens brick edges into pale pink/peach crusts), then pads clean terracotta RGB outward
    across `alpha == 0` pixels so bilinear WebGL texture sampling never pulls white/grey RGB.
    """
    has_alpha = alpha_u8 > 0
    if not np.any(has_alpha):
        return rgb_u8.copy()

    rgb_f = rgb_u8.astype(np.float32)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    r_over_g = rgb_f[:, :, 0] / np.maximum(1.0, rgb_f[:, :, 1])
    luma = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]

    opaque = alpha_u8 >= 128
    dist_in = ndi.distance_transform_edt(opaque).astype(np.float32)
    pure_core = (
        (dist_in >= bcfg.decontam_interior_dist_px)
        & (sat >= bcfg.decontam_core_sat_min)
        & (r_over_g >= bcfg.decontam_core_r_over_g_min)
    )

    # Ensure thin brick components (< 8px wide) still get their own local warm interior donor:
    lbl_comp, n_comp = ndi.label(has_alpha)
    for cid in range(1, n_comp + 1):
        comp = lbl_comp == cid
        if np.any(comp & pure_core):
            continue
        comp_max_d = float(dist_in[comp].max())
        local_seed = (
            comp
            & (dist_in >= max(0.8, 0.55 * comp_max_d))
            & (sat >= 120.0)
            & (r_over_g >= 1.28)
        )
        if np.any(local_seed):
            pure_core |= local_seed

    if not np.any(pure_core):
        return decontaminate_matte_edge_rgb(rgb_u8, alpha_u8)

    _, core_idx = ndi.distance_transform_edt(~pure_core, return_indices=True)
    donor_rgb = rgb_f[core_idx[0], core_idx[1]]
    donor_luma = luma[core_idx[0], core_idx[1]]
    luma_ratio = np.clip(
        luma / np.maximum(donor_luma, 10.0),
        bcfg.decontam_min_luma_ratio,
        1.0,
    )
    recolored_rgb = np.clip(donor_rgb * luma_ratio[:, :, None], 0.0, 255.0)

    band_w = np.clip((bcfg.decontam_band_px - dist_in) / 1.25, 0.0, 1.0)
    desat_bleed = (sat < bcfg.decontam_core_sat_min) | (r_over_g < bcfg.decontam_core_r_over_g_min)
    band_w = np.where(has_alpha & desat_bleed, 1.0, band_w)
    clean_f = np.where(
        has_alpha[:, :, None],
        (1.0 - band_w[:, :, None]) * rgb_f + band_w[:, :, None] * recolored_rgb,
        rgb_f,
    )
    clean_u8 = np.clip(np.round(clean_f), 0, 255).astype(np.uint8)

    # Pad clean brick RGB outward across all `alpha == 0` pixels (including >= 4px bilinear band):
    _, out_idx = ndi.distance_transform_edt(~has_alpha, return_indices=True)
    padded_u8 = clean_u8[out_idx[0], out_idx[1]].copy()
    padded_u8[has_alpha] = clean_u8[has_alpha]
    return padded_u8


def _neutralize_warm_terracotta_bleed_rgb(
    img_rgb_u8: np.ndarray,
    inpaint_mask_bool: np.ndarray,
    neutral_donor_bool: np.ndarray,
) -> np.ndarray:
    """
    Strips residual warm orange/peach `a*, b*` chroma left by `LaMa` inside dilated brick-removal
    masks so healed white steel cubbies (`mid`) and gallery wall/concrete (`bg`) match surrounding
    neutral illumination with zero orange terracotta crusts.
    """
    if not np.any(inpaint_mask_bool) or not np.any(neutral_donor_bool):
        return img_rgb_u8.copy()

    lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    _, near_idx = ndi.distance_transform_edt(~neutral_donor_bool, return_indices=True)
    ref_a = ndi.gaussian_filter(lab[:, :, 1][near_idx[0], near_idx[1]], sigma=8.0)
    ref_b = ndi.gaussian_filter(lab[:, :, 2][near_idx[0], near_idx[1]], sigma=8.0)

    dist_in = ndi.distance_transform_edt(inpaint_mask_bool).astype(np.float32)
    blend = np.clip(dist_in / 4.0, 0.0, 1.0)

    clamped_a = np.minimum(lab[:, :, 1], ref_a + 1.5)
    clamped_b = np.minimum(lab[:, :, 2], ref_b + 2.5)
    lab[:, :, 1] = np.where(inpaint_mask_bool, (1.0 - blend) * lab[:, :, 1] + blend * clamped_a, lab[:, :, 1])
    lab[:, :, 2] = np.where(inpaint_mask_bool, (1.0 - blend) * lab[:, :, 2] + blend * clamped_b, lab[:, :, 2])
    return cv2.cvtColor(np.clip(np.round(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _harmonize_tzofen_room_inpaint_rgb(
    auth_vent_rgb_u8: np.ndarray,
    auth_inpainted_rgb_u8: np.ndarray,
    auth_occ_bool: np.ndarray,
    baseboard_auth_y: int,
    bcfg: TzofenBrickSegmentationConfig,
) -> np.ndarray:
    """
    Harmonizes low-frequency room brightness across `LaMa` holes via multi-scale Laplace membrane
    interpolation separately above and below `baseboard_auth_y`, feathered outward across
    `~auth_occ_bool` (`w == 1.0` everywhere inside `auth_occ_bool` so zero boundary moat forms),
    then strips any warm terracotta Lab chroma bleed.
    """
    ah, aw, _ = auth_vent_rgb_u8.shape
    by = int(np.clip(baseboard_auth_y, 16, ah - 16))
    src_f = auth_vent_rgb_u8.astype(np.float32)
    inp_f = auth_inpainted_rgb_u8.astype(np.float32)
    hsv = cv2.cvtColor(auth_vent_rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    luma = 0.299 * src_f[:, :, 0] + 0.587 * src_f[:, :, 1] + 0.114 * src_f[:, :, 2]

    wall_donor = (
        (~auth_occ_bool[:by])
        & (luma[:by] >= bcfg.bg_wall_donor_min_luma)
        & (sat[:by] <= bcfg.bg_wall_donor_max_sat)
    )
    floor_clean_donor = (
        (~auth_occ_bool[by:])
        & (luma[by:] >= bcfg.bg_floor_shadow_luma_min)
        & (luma[by:] <= 232.0)
        & (sat[by:] <= bcfg.bg_floor_donor_max_sat)
    )
    if not np.any(wall_donor) or not np.any(floor_clean_donor):
        raise RuntimeError("Tzofen room harmonization found empty wall or floor donor mask")

    sig = bcfg.bg_dc_harmonize_sigma_px

    def _harmonic_dc_fill(sub_src_f: np.ndarray, donor_mask: np.ndarray) -> np.ndarray:
        sh, sw, _ = sub_src_f.shape
        _, idx = ndi.distance_transform_edt(~donor_mask, return_indices=True)
        field = sub_src_f[idx[0], idx[1]].copy()
        ds = 8
        fh, fw = max(4, sh // ds), max(4, sw // ds)
        f_small = cv2.resize(field, (fw, fh), interpolation=cv2.INTER_AREA)
        d_small = (
            cv2.resize(donor_mask.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST) > 0
        )
        fixed = f_small.copy()
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
            f_small = np.where(d_small[:, :, None], fixed, nb)
        up = cv2.resize(f_small, (sw, sh), interpolation=cv2.INTER_CUBIC)
        return ndi.gaussian_filter(up, sigma=(sig, sig, 0.0))

    wall_ref_low = _harmonic_dc_fill(src_f[:by], wall_donor)
    wall_inp = inp_f[:by]
    wall_fine_grain = wall_inp - ndi.gaussian_filter(wall_inp, sigma=(2.5, 2.5, 0.0))
    wall_seam_hf = wall_inp - ndi.gaussian_filter(wall_inp, sigma=(sig, sig, 0.0))
    wy = np.arange(by)[:, None]
    seam_t = np.clip((wy - float(by - 10)) / 8.0, 0.0, 1.0)[:, :, None]
    wall_target = wall_ref_low + (1.0 - seam_t) * wall_fine_grain + seam_t * wall_seam_hf

    floor_ref_low = _harmonic_dc_fill(src_f[by:], floor_clean_donor)
    floor_inp = inp_f[by:]
    floor_hf = floor_inp - ndi.gaussian_filter(floor_inp, sigma=(sig, sig, 0.0))
    floor_target = floor_ref_low + floor_hf

    target_full = np.vstack([wall_target, floor_target])
    dist_out = ndi.distance_transform_edt(~auth_occ_bool).astype(np.float32)
    w_dc = np.clip(1.0 - dist_out / max(1.0, bcfg.bg_dc_blend_band_px), 0.0, 1.0)[:, :, None]
    harmonized_u8 = np.clip(
        np.round((1.0 - w_dc) * src_f + w_dc * target_full), 0, 255
    ).astype(np.uint8)

    neutral_donor_full = np.vstack([wall_donor, floor_clean_donor])
    return _neutralize_warm_terracotta_bleed_rgb(
        img_rgb_u8=harmonized_u8,
        inpaint_mask_bool=w_dc[:, :, 0] > 0.02,
        neutral_donor_bool=neutral_donor_full,
    )


def _prune_unanchored_cutout_rgba(
    rgba_u8: np.ndarray,
    solve_cfg: ContactAnchoredLayerDepthConfig,
) -> np.ndarray:
    """
    Guarantees 100% of 4-connected opaque components (`alpha >= solve_cfg.alpha_threshold_u8`)
    satisfy `size >= solve_cfg.min_component_area_px` AND belong to a contact-anchored PDE domain,
    ensuring `max_abs_rim_jump_world == 0.0` with zero floating islands.
    """
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


def _reconstruct_tzofen_white_steel_mid_rgba(
    tz_pad: np.ndarray,
    tz_vent_clean_img: Image.Image,
    total_sculp_alpha_u8: np.ndarray,
    fg_brick_alpha_u8: np.ndarray,
    cfg: TzofenSceneBuildConfig,
    lama: LamaOcclusionInpainter,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs `scene12_tzofen_mid.png` (`1500x1000` RGBA): ONLY the actual white/grey painted
    steel multi-tier table, legs, shelves, cubbies, and upper frames — with ZERO white sticker
    halos around bricks, ZERO ghost freestanding bricks, ZERO filled see-through floor bays,
    `r = 28px` dilated LaMa + Lab chroma-neutralized brick removal, and `cv2.BORDER_REFLECT_101`
    bottom floor-border reflection for border-touching front legs (`leg1`, `leg2`).
    """
    ih, iw = total_sculp_alpha_u8.shape
    bcfg = cfg.brick_segmentation
    alpha_thr = cfg.topology_verification.alpha_threshold_u8
    y0 = cfg.inner_y_offset_px
    y1 = y0 + cfg.scaled_height_px

    vent_rgb = np.asarray(tz_vent_clean_img, dtype=np.uint8)
    vent_f = vent_rgb.astype(np.float32)
    vent_hsv = cv2.cvtColor(vent_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    vent_hue, vent_sat = vent_hsv[:, :, 0], vent_hsv[:, :, 1]
    vent_rb = vent_f[:, :, 0] - vent_f[:, :, 2]
    vent_rg = vent_f[:, :, 0] / np.maximum(1.0, vent_f[:, :, 1])
    vent_luma = (
        0.299 * vent_f[:, :, 0]
        + 0.587 * vent_f[:, :, 1]
        + 0.114 * vent_f[:, :, 2]
    )

    warm_brick_stain = (
        (total_sculp_alpha_u8 >= 6)
        & (vent_rb >= bcfg.warm_stain_r_minus_b_min)
        & (vent_sat >= bcfg.warm_stain_sat_min)
        & (vent_hue <= bcfg.warm_stain_hue_max)
        & (vent_rg >= 1.24)
    )
    kd = bcfg.brick_inpaint_dilate_px * 2 + 1
    brick_collar = (
        cv2.dilate(
            ((fg_brick_alpha_u8 > 0) | warm_brick_stain).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)),
        )
        > 0
    )

    pure_steel = (
        (total_sculp_alpha_u8 >= 175)
        & (~brick_collar)
        & (vent_luma >= bcfg.healed_steel_min_luma)
        & (vent_sat <= 55.0)
        & (vent_rg <= 1.12)
    )
    pure_steel = (
        cv2.erode(
            pure_steel.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        > 0
    )
    if not np.any(pure_steel):
        raise RuntimeError("Empty pure white steel seed mask on Tzofen mid!")

    _, nearest_idx = ndi.distance_transform_edt(~pure_steel, return_indices=True)
    nearest_steel_rgb = vent_rgb[nearest_idx[0], nearest_idx[1]].astype(np.float32)
    smooth_steel_env = ndi.gaussian_filter(nearest_steel_rgb, sigma=(6.0, 6.0, 0.0))
    unoccluded_table = (
        (total_sculp_alpha_u8 >= 120)
        & (~brick_collar)
        & (vent_rg <= 1.26)
        & (vent_sat <= bcfg.mid_steel_max_sat)
    )
    steel_seed_rgb = np.where(
        unoccluded_table[:, :, None],
        vent_f,
        smooth_steel_env,
    )
    steel_seed_img = Image.fromarray(
        np.clip(np.round(steel_seed_rgb), 0, 255).astype(np.uint8), mode="RGB"
    )

    healed_table_img = lama.inpaint_background(
        steel_seed_img,
        (brick_collar.astype(np.uint8)) * 255,
    )
    healed_table_rgb = np.asarray(healed_table_img, dtype=np.float32)

    dist_in_collar = cv2.distanceTransform(brick_collar.astype(np.uint8), cv2.DIST_L2, 5)
    w_heal = np.clip(dist_in_collar / max(1.0, bcfg.brick_inpaint_feather_px), 0.0, 1.0)
    w_heal = (w_heal * w_heal * (3.0 - 2.0 * w_heal))[:, :, None]
    mid_rgb_raw = np.clip(
        np.round((1.0 - w_heal) * vent_f + w_heal * healed_table_rgb),
        0,
        255,
    ).astype(np.uint8)
    mid_rgb_composite = _neutralize_warm_terracotta_bleed_rgb(
        img_rgb_u8=mid_rgb_raw,
        inpaint_mask_bool=brick_collar,
        neutral_donor_bool=pure_steel,
    )

    # Construct crisp white-steel silhouette WITHOUT `sculp_f - brick_f` sticker halos:
    sculp_bool = (total_sculp_alpha_u8 >= alpha_thr).copy()
    eb = bcfg.baseboard_exclude_box
    sculp_bool[eb.y0 : eb.y1, eb.x0 : eb.x1] = False
    sculp_bool[y1:, :] = False

    brick_core = fg_brick_alpha_u8 >= 185
    if bcfg.mid_brick_exclusion_dilate_px > 0:
        kx = bcfg.mid_brick_exclusion_dilate_px * 2 + 1
        brick_excl = (
            cv2.dilate(
                brick_core.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, kx)),
            )
            > 0
        )
    else:
        brick_excl = brick_core.copy()
    upper_steel_allow = np.zeros((ih, iw), dtype=bool)
    for ub in bcfg.upper_steel_boxes:
        upper_steel_allow[ub.y0 : ub.y1, ub.x0 : ub.x1] = True

    yy = np.arange(ih)[:, None]
    in_recess_band = (
        (yy >= bcfg.tabletop_deck_y_range_px[1])
        & (yy <= bcfg.mid_recessed_shelf_y_max_px)
    )
    open_steel_chroma = (
        (vent_sat <= bcfg.mid_steel_max_sat)
        & (vent_rb <= bcfg.mid_steel_max_r_minus_b)
        & (vent_rg <= bcfg.mid_steel_max_r_over_g)
        & (vent_luma >= bcfg.mid_steel_min_luma)
    )
    recessed_steel_chroma = (
        in_recess_band
        & (vent_sat <= bcfg.mid_recessed_steel_max_sat)
        & (vent_rb <= bcfg.mid_steel_max_r_minus_b)
        & (vent_rg <= bcfg.mid_recessed_steel_max_r_over_g)
        & (vent_luma >= bcfg.mid_recessed_steel_min_luma)
    )
    genuine_steel_chroma = open_steel_chroma | recessed_steel_chroma
    visible_steel_bool = (
        sculp_bool
        & (~brick_excl)
        & genuine_steel_chroma
        & ((yy >= bcfg.tabletop_deck_y_range_px[0]) | upper_steel_allow)
    )
    lvb = bcfg.left_vent_exclude_box
    visible_steel_bool[lvb.y0 : lvb.y1, lvb.x0 : lvb.x1] = False

    # 1. Main horizontal tabletop deck (`ty0..ty1`): exact perspective trapezoid from leftmost
    #    to rightmost visible white steel on each row:
    tx0, tx1 = bcfg.tabletop_deck_x_range_px
    ty0, ty1 = bcfg.tabletop_deck_y_range_px
    front_beam_band = np.zeros((ih, iw), dtype=bool)
    for row_y in range(ty0, ty1):
        xs_row = np.where(visible_steel_bool[row_y, tx0:tx1])[0]
        if len(xs_row) >= 2:
            front_beam_band[row_y, tx0 + int(xs_row[0]) : tx0 + int(xs_row[-1]) + 1] = True

    # 2. Orthogonal shelf/divider bridging ONLY across actual brick footprints (`brick_bridge_zone`),
    #    seeded strictly from true horizontal shelves (`>=22px` wide) or vertical columns (`>=22px` tall)
    #    so open see-through floor bays and freestanding brick silhouettes NEVER gain white steel:
    brick_bridge_zone = (
        cv2.dilate(
            (fg_brick_alpha_u8 >= 8).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        > 0
    ) & ((yy >= ty0) | upper_steel_allow)

    horiz_shelf_seeds = cv2.morphologyEx(
        visible_steel_bool.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((1, 22), dtype=np.uint8),
    )
    horiz_shelf_bridged = (
        cv2.morphologyEx(
            horiz_shelf_seeds,
            cv2.MORPH_CLOSE,
            np.ones((1, bcfg.mid_horiz_shelf_bridge_px), dtype=np.uint8),
        )
        > 0
    ) & brick_bridge_zone

    vert_col_seeds = cv2.morphologyEx(
        visible_steel_bool.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((22, 1), dtype=np.uint8),
    )
    vert_col_bridged = (
        cv2.morphologyEx(
            vert_col_seeds,
            cv2.MORPH_CLOSE,
            np.ones((bcfg.mid_vert_divider_bridge_px, 1), dtype=np.uint8),
        )
        > 0
    ) & brick_bridge_zone & ((yy >= bcfg.mid_vert_bridge_min_y_px) | upper_steel_allow)

    # 3. Voronoi / EDT contact partition inside brick footprints: where a brick sits inside or
    #    against a grey steel cubby wall, extend continuous steel behind the steel-contacting portion
    #    of the brick (`d_steel < d_air - 1.5` and `>= 3.5px` inside any open-air edge) so neither
    #    rest-pose rims nor off-axis parallax ever expose a bright `bg` moat around dark bricks:
    open_air_adj = (~sculp_bool) | (
        (~visible_steel_bool) & (~(fg_brick_alpha_u8 >= 8)) & (~front_beam_band)
    )
    d_steel = ndi.distance_transform_edt(~visible_steel_bool).astype(np.float32)
    d_air = ndi.distance_transform_edt(~open_air_adj).astype(np.float32)
    voronoi_steel_behind_brick = (
        (fg_brick_alpha_u8 >= 160)
        & ((yy >= ty0) | upper_steel_allow)
        & (d_steel < (d_air - 1.5))
        & (d_air >= 3.5)
    )

    solid_steel_bool = (
        visible_steel_bool
        | front_beam_band
        | horiz_shelf_bridged
        | vert_col_bridged
        | voronoi_steel_behind_brick
    )
    solid_steel_bool[eb.y0 : eb.y1, eb.x0 : eb.x1] = False

    dist_in_mid = ndi.distance_transform_edt(solid_steel_bool).astype(np.float32)
    dist_out_mid = ndi.distance_transform_edt(~solid_steel_bool).astype(np.float32)
    sdf_mid = dist_in_mid - dist_out_mid - 0.15
    mid_alpha_f = np.clip(0.5 + sdf_mid / 1.2, 0.0, 1.0)

    wx0, wy0, wx1, wy1 = cfg.window_portal_inner_box_px
    px0 = int(np.ceil(wx0)) + 1
    py0 = int(np.ceil(wy0)) + 1
    px1 = int(np.floor(wx1))
    py1 = int(np.floor(wy1))
    mid_alpha_f[py0:py1, px0:px1] = 0.0
    mid_alpha_f[0, :] = 0.0
    mid_alpha_f[y1:, :] = 0.0
    mid_alpha_f[:, 0] = 0.0
    mid_alpha_f[:, -1] = 0.0

    mid_alpha_u8 = _fill_interior_alpha_pinholes(
        np.clip(np.round(mid_alpha_f * 255.0), 0, 255).astype(np.uint8),
        max_hole_area_px=bcfg.mid_max_pinhole_px,
        preserve_open_box_px=cfg.window_portal_inner_box_px,
        alpha_threshold_u8=alpha_thr,
    )
    mid_alpha_u8[py0:py1, px0:px1] = 0

    # Mirror bottom-border-touching front legs (`leg1`, `leg2`) across `y1 - 1 = 960` into `y1..ih-2`
    # via `cv2.BORDER_REFLECT_101` so they reflect seamlessly with the floor collar (Feedback #48):
    if bcfg.mirror_authentic_collar and ih > y1 + 1:
        touch_cols = np.any(mid_alpha_u8[y1 - 3 : y1, :] >= alpha_thr, axis=0)
        touch_cols_dil = (
            cv2.dilate(touch_cols[None, :].astype(np.uint8), np.ones((1, 13), dtype=np.uint8))[0]
            > 0
        )
        n_bot = ih - y1
        refl_rgb = cv2.copyMakeBorder(
            mid_rgb_composite[y0:y1, :], 0, n_bot, 0, 0, borderType=cv2.BORDER_REFLECT_101
        )[y1 - y0 :]
        refl_a = cv2.copyMakeBorder(
            mid_alpha_u8[y0:y1, :], 0, n_bot, 0, 0, borderType=cv2.BORDER_REFLECT_101
        )[y1 - y0 :]
        mid_rgb_composite[y1:, touch_cols_dil] = refl_rgb[:, touch_cols_dil]
        mid_alpha_u8[y1:, touch_cols_dil] = refl_a[:, touch_cols_dil]

    mid_alpha_u8[0, :] = 0
    mid_alpha_u8[-1, :] = 0
    mid_alpha_u8[:, 0] = 0
    mid_alpha_u8[:, -1] = 0

    mid_rgb_clean = decontaminate_matte_edge_rgb(mid_rgb_composite, mid_alpha_u8)
    mid_rgba_pruned = _prune_unanchored_cutout_rgba(
        np.dstack([mid_rgb_clean, mid_alpha_u8]),
        cfg.mid_contact_solve,
    )
    return mid_rgba_pruned, brick_collar


def _compute_brick_tower_protrusion(
    da2_excess: np.ndarray,
    fg_brick_alpha_u8: np.ndarray,
    bcfg: TzofenBrickSegmentationConfig,
    fg_solve_cfg: ContactAnchoredLayerDepthConfig,
) -> np.ndarray:
    """
    Combines room-detrended Depth Anything V2 relief with per-cluster vertical brick-stack
    height relief so every terracotta brick block & tower protrudes 3D-volumetrically from the
    host white steel table (`mid`) while tapering cleanly to 0 at its bottom contact band.
    """
    ih, iw = fg_brick_alpha_u8.shape
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
        ys, xs = np.where(grp)
        x_min, x_max = int(xs.min()), int(xs.max())
        for col in range(x_min, x_max + 1):
            c_ys = np.where(grp[:, col])[0]
            if len(c_ys) == 0:
                continue
            y_bot = float(c_ys.max())
            rise_px = np.maximum(0.0, y_bot - np.arange(ih, dtype=np.float32))
            rise_norm[grp[:, col], col] = np.clip(
                rise_px[grp[:, col]] / max(1.0, bcfg.brick_height_saturation_px), 0.0, 1.0
            )

    edt_in = ndi.distance_transform_edt(opaque).astype(np.float32)
    block_dome = np.clip(edt_in / 14.0, 0.0, 1.0)
    w_h = bcfg.brick_height_relief_weight
    combined = (1.0 - w_h) * da2_norm + w_h * np.maximum(rise_norm, 0.55 * block_dome)
    return np.clip(combined, 0.0, 1.0).astype(np.float32)


def build_tzofen_scene(
    cfg: TzofenSceneBuildConfig,
    repo_root: Path,
    scenes_dir: Path,
    depth_est: DepthAnythingEstimator,
    matte_ext: SalientMatteExtractor,
    sam_ext: SamBoxMaskExtractor,
    lama: LamaOcclusionInpainter,
) -> dict[str, Any]:
    iw, ih = cfg.inner_dims.width, cfg.inner_dims.height
    ew, eh = cfg.expanded_dims.width, cfg.expanded_dims.height
    px, py = (ew - iw) // 2, (eh - ih) // 2
    bcfg = cfg.brick_segmentation

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
    upper_zone = (yy >= 130) & (yy <= 500)
    da2_upper_fg = np.clip((da2_excess - 0.055) / 0.035, 0.0, 1.0) * upper_zone
    sal_upper = sal * upper_zone * (da2_excess > 0.035)
    brick_upper = np.clip((r_minus_b - 42.0) / 18.0, 0.0, 1.0) * upper_zone * (da2_excess > 0.025)

    sam_dil = (
        cv2.dilate(
            (sam_f > 0.35).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        > 0
    )
    sal_lower = sal * (yy > 500) * sam_dil
    da2_lower_fg = np.clip((da2_excess - 0.075) / 0.035, 0.0, 1.0) * (yy > 500) * sam_dil

    alpha_f = np.maximum.reduce(
        [sam_f, da2_upper_fg, sal_upper, brick_upper, sal_lower, da2_lower_fg]
    )
    xx = np.arange(iw)[None, :]
    wall_like = (yy < 476) & (luma > 233.0) & (r_minus_b >= 3.5) & (da2_excess < 0.065)
    right_wall_shadow = (yy < 476) & (xx >= 1422) & (r_minus_b < 28.0)
    floor_glare_sliver = (
        (yy >= 502)
        & (yy <= 565)
        & (xx >= 1160)
        & (xx <= 1235)
        & (luma > 205.0)
        & (r_minus_b < 32.0)
    )
    alpha_f = np.where(wall_like | right_wall_shadow | floor_glare_sliver, 0.0, alpha_f)
    alpha_f = ndi.gaussian_filter(
        np.clip((alpha_f - 0.42) / 0.22, 0.0, 1.0).astype(np.float32), sigma=0.55
    )
    alpha_f[:130, :] = 0.0
    alpha_f[y1:, :] = 0.0
    alpha_f[:, :8] = 0.0
    alpha_f[:, iw - 8 :] = 0.0

    total_sculp_alpha_u8 = _prune_small_alpha_components(
        np.clip(np.round(alpha_f * 255.0), 0, 255).astype(np.uint8),
        min_area_px=48,
        alpha_threshold_u8=cfg.topology_verification.alpha_threshold_u8,
    )
    total_sculp_alpha_u8 = _fill_interior_alpha_pinholes(
        total_sculp_alpha_u8,
        max_hole_area_px=96,
        preserve_open_box_px=cfg.window_portal_inner_box_px,
        alpha_threshold_u8=cfg.topology_verification.alpha_threshold_u8,
    )

    vent_mask = np.zeros((ih, iw), dtype=np.uint8)
    for vb in bcfg.wall_vent_boxes:
        vent_mask[vb.y0 : vb.y1, vb.x0 : vb.x1] = 255
    tz_vent_clean_img = lama.inpaint_background(tz_pad_img, vent_mask)

    fg_brick_alpha_u8 = _segment_tzofen_bricks_alpha(
        tz_pad=tz_pad,
        tz_pad_img=tz_pad_img,
        total_sculp_alpha_u8=total_sculp_alpha_u8,
        bcfg=bcfg,
        portal_box_px=cfg.window_portal_inner_box_px,
        alpha_threshold_u8=cfg.topology_verification.alpha_threshold_u8,
        sam_ext=sam_ext,
    )
    fg_rgb_clean = _decontaminate_tzofen_brick_rgba(
        rgb_u8=tz_pad,
        alpha_u8=fg_brick_alpha_u8,
        bcfg=bcfg,
    )
    fg_rgba = _prune_unanchored_cutout_rgba(
        np.dstack([fg_rgb_clean, fg_brick_alpha_u8]),
        cfg.fg_contact_solve,
    )
    fg_brick_alpha_u8 = fg_rgba[:, :, 3]
    fg_rgb_clean = _decontaminate_tzofen_brick_rgba(
        rgb_u8=tz_pad,
        alpha_u8=fg_brick_alpha_u8,
        bcfg=bcfg,
    )
    fg_rgba = np.dstack([fg_rgb_clean, fg_brick_alpha_u8])

    mid_rgba, brick_collar = _reconstruct_tzofen_white_steel_mid_rgba(
        tz_pad=tz_pad,
        tz_vent_clean_img=tz_vent_clean_img,
        total_sculp_alpha_u8=total_sculp_alpha_u8,
        fg_brick_alpha_u8=fg_brick_alpha_u8,
        cfg=cfg,
        lama=lama,
    )
    mid_alpha_u8 = mid_rgba[:, :, 3]

    for lname, a_chk in (("mid", mid_alpha_u8), ("fg", fg_brick_alpha_u8)):
        bmax = int(max(a_chk[0, :].max(), a_chk[-1, :].max(), a_chk[:, 0].max(), a_chk[:, -1].max()))
        if bmax != 0:
            raise RuntimeError(f"Tzofen {lname} border alpha must be 0, got {bmax}")

    Image.fromarray(mid_rgba, mode="RGBA").save(scenes_dir / "scene12_tzofen_mid.png")
    Image.fromarray(fg_rgba, mode="RGBA").save(scenes_dir / "scene12_tzofen_fg.png")

    # Build `bg` sculpture removal mask = sculpture + mid + fg + portal + bottom leg tips, dilated by
    # `r = bg_occ_dilate_px` (`28px`) before `LaMa` while keeping unmasked wall & floor context intact:
    vent_clean_u8 = np.asarray(tz_vent_clean_img, dtype=np.uint8)
    raw_occ = (
        (total_sculp_alpha_u8 >= 6)
        | (mid_alpha_u8 >= 6)
        | (fg_brick_alpha_u8 >= 6)
    )
    wx0, wy0, wx1, wy1 = cfg.window_portal_inner_box_px
    raw_occ[
        max(0, int(np.floor(wy0)) - 6) : min(ih, int(np.ceil(wy1)) + 6),
        max(0, int(np.floor(wx0)) - 6) : min(iw, int(np.ceil(wx1)) + 6),
    ] = True
    bot_leg_cols = np.any(raw_occ[max(0, y1 - 36) : y1, :], axis=0)
    bot_leg_cols_dil = (
        cv2.dilate(bot_leg_cols[None, :].astype(np.uint8), np.ones((1, 45), dtype=np.uint8))[0] > 0
    )
    raw_occ[max(0, y1 - 12) : y1, bot_leg_cols_dil] = True

    k_bg = bcfg.bg_occ_dilate_px * 2 + 1
    lama_occ_bool = (
        cv2.dilate(
            raw_occ.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bg, k_bg)),
        )
        > 0
    )
    # Two-pass neural `LaMa` room heal on authentic `919x1500` crop `[y0:y1, :]`:
    #   Pass 1 removes the sculpture (`r = 28px`) with ample unmasked room context;
    #   Pass 2 removes residual under-table floor cast-shadow stencils (`<18%` mask) on the
    #   already-sculpture-free room, followed by `cv2.BORDER_REFLECT_101` mirror collar (#48):
    auth_vent_u8 = vent_clean_u8[y0:y1, :]
    auth_occ_bool = lama_occ_bool[y0:y1, :]
    baseboard_auth_y = cfg.baseboard_inner_y - y0
    auth_inp1_rgb = np.asarray(
        lama.inpaint_background(
            Image.fromarray(auth_vent_u8, mode="RGB"),
            (auth_occ_bool.astype(np.uint8)) * 255,
        ),
        dtype=np.uint8,
    )
    auth_pass1_rgb = _harmonize_tzofen_room_inpaint_rgb(
        auth_vent_rgb_u8=auth_vent_u8,
        auth_inpainted_rgb_u8=auth_inp1_rgb,
        auth_occ_bool=auth_occ_bool,
        baseboard_auth_y=baseboard_auth_y,
        bcfg=bcfg,
    )
    p1_f = auth_pass1_rgb.astype(np.float32)
    p1_luma = 0.299 * p1_f[:, :, 0] + 0.587 * p1_f[:, :, 1] + 0.114 * p1_f[:, :, 2]
    p1_rb = p1_f[:, :, 0] - p1_f[:, :, 2]
    p1_sat = cv2.cvtColor(auth_pass1_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)[:, :, 1]
    ay = np.arange(auth_pass1_rgb.shape[0])[:, None]
    floor_shadow_raw = (ay >= baseboard_auth_y + 2) & (
        (p1_luma < bcfg.bg_floor_shadow_luma_min)
        | (p1_rb >= bcfg.bg_floor_donor_max_abs_rb)
        | (p1_sat >= bcfg.bg_floor_donor_max_sat)
    )
    ks2 = bcfg.bg_floor_shadow_dilate_px * 2 + 1
    floor_shadow_occ = (
        cv2.dilate(
            floor_shadow_raw.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks2, ks2)),
        )
        > 0
    )
    floor_shadow_occ[:baseboard_auth_y, :] = False
    if np.any(floor_shadow_occ):
        auth_inp2_rgb = np.asarray(
            lama.inpaint_background(
                Image.fromarray(auth_pass1_rgb, mode="RGB"),
                (floor_shadow_occ.astype(np.uint8)) * 255,
            ),
            dtype=np.uint8,
        )
        auth_clean_rgb = _harmonize_tzofen_room_inpaint_rgb(
            auth_vent_rgb_u8=auth_pass1_rgb,
            auth_inpainted_rgb_u8=auth_inp2_rgb,
            auth_occ_bool=floor_shadow_occ,
            baseboard_auth_y=baseboard_auth_y,
            bcfg=bcfg,
        )
    else:
        auth_clean_rgb = auth_pass1_rgb

    if bcfg.mirror_authentic_collar:
        bg_exp_rgb = cv2.copyMakeBorder(
            auth_clean_rgb,
            py + y0,
            eh - (py + y1),
            px,
            ew - (px + iw),
            borderType=cv2.BORDER_REFLECT_101,
        )
        lama_inner_rgb = bg_exp_rgb[py : py + ih, px : px + iw].copy()
    else:
        lama_inner_rgb = cv2.copyMakeBorder(
            auth_clean_rgb, y0, ih - y1, 0, 0, borderType=cv2.BORDER_REFLECT_101
        )
        seed_exp = cv2.copyMakeBorder(
            lama_inner_rgb, py, py, px, px, borderType=cv2.BORDER_REFLECT_101
        )
        bg_exp_rgb = _heal_expanded_collar_lama(lama_inner_rgb, seed_exp, None, lama)

    Image.fromarray(lama_inner_rgb, mode="RGB").save(
        scenes_dir / "scene12_tzofen_lama_inner.jpg", quality=cfg.jpeg_quality
    )
    bg_exp_img = Image.fromarray(bg_exp_rgb, mode="RGB")
    bg_exp_img.save(scenes_dir / "scene12_tzofen_bg_inpainted.jpg", quality=cfg.jpeg_quality)

    # Predict Depth Anything V2 on clean right-side-up authentic room `919x1500`, clean residual
    # mounds harmonically under `[mid_rgba, fg_rgba]`, then mirror cleanly across `(600, 442, 2100, 1361)`:
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
        raw_room_u8, [mid_rgba, fg_rgba], cfg.room_depth_clean
    )
    if bcfg.mirror_authentic_collar:
        clean_room_u8 = cv2.copyMakeBorder(
            clean_room_pre_u8[py + y0 : py + y1, px : px + iw],
            py + y0,
            eh - (py + y1),
            px,
            ew - (px + iw),
            borderType=cv2.BORDER_REFLECT_101,
        )
    else:
        clean_room_u8 = clean_room_pre_u8
    Image.fromarray(clean_room_u8, mode="L").save(scenes_dir / "scene12_tzofen_depth.png")

    # 1. Solve `mid` (white steel table) depth anchored C^0 at all 7 floor feet to `clean_room_u8`,
    #    after harmonically removing brick depth mounds (`brick_collar`) from `da2_excess` so white
    #    steel shelves/tabletop NEVER bulge toward the camera where bricks used to sit:
    full_dom = np.ones((ih, iw), dtype=bool)
    brick_heal = brick_collar.copy()
    brick_heal[0, :] = False
    brick_heal[-1, :] = False
    brick_heal[:, 0] = False
    brick_heal[:, -1] = False
    da2_excess_brick_free = _solve_harmonic_field_on_domain(
        domain_mask=full_dom,
        dirichlet_mask=~brick_heal,
        dirichlet_values=da2_excess.astype(np.float64),
    ).astype(np.float32)

    mid_excess_pos = np.maximum(0.0, da2_excess_brick_free)
    mid_opaque = mid_alpha_u8 >= cfg.mid_contact_solve.alpha_threshold_u8
    mid_prot_scale = max(1e-5, float(np.percentile(mid_excess_pos[mid_opaque], 98.0)))
    mid_prot_norm = np.clip(mid_excess_pos / mid_prot_scale, 0.0, 1.0).astype(np.float32)

    mid_solve = solve_contact_anchored_layer_depth(
        room_depth_expanded=clean_room_u8,
        cutout_rgba=mid_rgba,
        cfg=cfg.mid_contact_solve,
        neural_protrusion_norm=mid_prot_norm,
    )
    mid_depth_u8 = mid_solve.depth_u8
    Image.fromarray(mid_depth_u8, mode="L").save(scenes_dir / "scene12_tzofen_mid_depth.png")

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
    if not mid_report.all_passed:
        raise RuntimeError(f"Tzofen mid failed depth topology gate: {mid_report.to_dict()}")

    # 2. Embed `mid` world-Z into `2700x1800` host depth and solve `fg` (terracotta mud-bricks)
    mid_host_depth_exp_u8 = cv2.copyMakeBorder(
        mid_depth_u8, py, py, px, px, borderType=cv2.BORDER_REPLICATE
    )
    fg_prot_norm = _compute_brick_tower_protrusion(
        da2_excess=da2_excess,
        fg_brick_alpha_u8=fg_brick_alpha_u8,
        bcfg=bcfg,
        fg_solve_cfg=cfg.fg_contact_solve,
    )
    fg_solve = solve_contact_anchored_layer_depth(
        room_depth_expanded=mid_host_depth_exp_u8,
        cutout_rgba=fg_rgba,
        cfg=cfg.fg_contact_solve,
        neural_protrusion_norm=fg_prot_norm,
    )
    fg_depth_u8 = fg_solve.depth_u8
    Image.fromarray(fg_depth_u8, mode="L").save(scenes_dir / "scene12_tzofen_fg_depth.png")

    fg_report = verify_layer_contact_and_discontinuity(
        room_depth_expanded=mid_host_depth_exp_u8,
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
    if not fg_report.all_passed:
        raise RuntimeError(f"Tzofen fg failed depth topology gate: {fg_report.to_dict()}")

    mid_world_z = mid_solve.world_z_map
    wx0, wy0, wx1, wy1 = cfg.window_frame_sample_box_px
    frame_mask = (mid_alpha_u8[wy0:wy1, wx0:wx1] >= 200) & (
        luma[wy0:wy1, wx0:wx1] > 185.0
    )
    if not np.any(frame_mask):
        raise RuntimeError("Empty white window frame mask on Tzofen mid!")
    frame_z_med = float(np.median(mid_world_z[wy0:wy1, wx0:wx1][frame_mask]))
    portal_world_z = round(frame_z_med - cfg.portal_z_clearance_behind_frame, 2)

    fg_opaque = fg_brick_alpha_u8 >= cfg.fg_contact_solve.alpha_threshold_u8
    brick_relief_over_mid = fg_solve.world_z_map[fg_opaque] - mid_world_z[fg_opaque]

    return {
        "id": cfg.scene_id,
        "midCoverageRatio": round(float((mid_alpha_u8 >= 20).mean()), 4),
        "fgCoverageRatio": round(float((fg_brick_alpha_u8 >= 20).mean()), 4),
        "brickReliefOverTableP50World": round(float(np.percentile(brick_relief_over_mid, 50.0)), 2),
        "brickReliefOverTableP95World": round(float(np.percentile(brick_relief_over_mid, 95.0)), 2),
        "brickReliefOverTableMaxWorld": round(float(brick_relief_over_mid.max()), 2),
        "windowFrameMedianWorldZ": round(frame_z_med, 3),
        "portalWorldZ": portal_world_z,
        "portalInnerBoxPx": list(cfg.window_portal_inner_box_px),
        "portalRecessDepth": cfg.portal_recess_depth,
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

    print("[tzofen_tree_scenes_builder] Building Scene 12: tzofen_brick_table...", flush=True)
    tzofen_meta = build_tzofen_scene(
        cfg.tzofen, repo_root, scenes_dir, depth_est, matte_ext, sam_ext, lama
    )
    print(f"[tzofen_tree_scenes_builder] Tzofen done: {tzofen_meta}", flush=True)

    scenes_payload: dict[str, Any] = {"tzofen_brick_table": tzofen_meta}
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
        specs_out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
        help="Rebuild only Scene 12 (tzofen_brick_table) while keeping Scene 13 intact.",
    )
    args = parser.parse_args()
    run_tzofen_tree_pipeline(specs_out_path=args.specs_out, only_tzofen=bool(args.only_tzofen))


if __name__ == "__main__":
    main()
