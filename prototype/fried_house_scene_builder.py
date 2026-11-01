#!/usr/bin/env python3
"""
prototype/fried_house_scene_builder.py
Offline neural 2.5D spatial asset builder and typed SSOT configuration models for:
  Scene 10 (`fried_house_railing`): "על המעקה בבית פריד בהרצליה — כרית ואבן"
  Merges Tami Sheaffer & sister Er'ela Sadan at the Fried parents' entrance ramp in Herzliya
  (`IMG_HERZ_0397`, 2006) with the Cushion-and-Stone sculpture (`IMG_BINY_0013`, "מגרה – כרית – אבן",
  Binyamin Gallery 2021) placed on the concrete ramp floor at bottom-right between the green
  railing legs (Beta Feedback #18, `fb-20261030-171425-7fba98`).

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
    import torch
    from PIL import Image
except ImportError as exc:
    raise RuntimeError(
        "Missing neural vision dependencies inside virtual environment. Run with: "
        "`/Users/talsheaffer/code/tami/.venv/bin/python -u prototype/fried_house_scene_builder.py`."
    ) from exc

from prototype.depth_topology import (
    ContactAnchorBox,
    ContactAnchoredLayerDepthConfig,
    LayerTopologyVerificationConfig,
    LayerTopologyVerificationReport,
    RoomDepthCleanConfig,
    clean_room_depth_under_cutouts,
    compute_room_relative_neural_protrusion,
    compute_world_z_map,
    solve_contact_anchored_layer_depth,
    verify_layer_contact_and_discontinuity,
)
from prototype.scene_config import (
    BoundingBoxPrompt,
    CanvasDimensionsConfig,
    DepthModelConfig,
    InpaintingConfig,
    OrientedMatteBoxSpec,
    OutpaintingConfig,
    SamPointGroupPrompt,
    SegmentationConfig,
)
from prototype.vision_primitives import (
    DepthAnythingEstimator,
    LamaOcclusionInpainter,
    SalientMatteExtractor,
    SamBoxMaskExtractor,
    compose_outpainted_room,
    decontaminate_matte_edge_rgb,
    merge_sam_and_oriented_mattes_u8,
    select_torch_device,
)


# ============================================================================
# Hierarchical, Serializable Configuration Data-Models (SSOT)
# ============================================================================


@dataclass(frozen=True)
class CushionStonePlacementConfig:
    """Extraction, warm-sun harmonization, shadow synthesis, and ramp placement for IMG_BINY_0013."""

    catalog_id: str
    source_relpath: str
    src_crop_xyxy: tuple[int, int, int, int]
    sam_prompts: tuple[SamPointGroupPrompt, ...]
    sam_exclude_boxes: tuple[BoundingBoxPrompt, ...]
    isnet_matte_box: OrientedMatteBoxSpec
    support_dilate_px: int
    max_pinhole_fill_px: int
    min_major_component_px: int
    alpha_tight_threshold_u8: int
    tight_crop_pad_px: int
    rotation_deg: float
    dest_tight_wh: tuple[int, int]
    dest_bottom_center_inner_xy: tuple[int, int]
    shade_gain_rgb: tuple[float, float, float]
    sun_gain_rgb: tuple[float, float, float]
    sun_bias_rgb: tuple[float, float, float]
    sun_line_inner_y: float
    sun_transition_px: float
    luma_gamma: float
    contact_shadow_rgb: tuple[int, int, int]
    contact_shadow_band_px: int
    contact_shadow_base_height_ratio: float
    contact_shadow_offset_xy: tuple[int, int]
    contact_shadow_sigma_yx: tuple[float, float]
    contact_shadow_opacity: float
    cast_shadow_rgb: tuple[int, int, int]
    cast_shadow_footprint_height_ratio: float
    cast_shadow_offset_xy: tuple[int, int]
    cast_shadow_sigma_yx: tuple[float, float]
    cast_shadow_opacity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "source_relpath": self.source_relpath,
            "src_crop_xyxy": list(self.src_crop_xyxy),
            "sam_prompts": [p.to_dict() for p in self.sam_prompts],
            "sam_exclude_boxes": [b.to_dict() for b in self.sam_exclude_boxes],
            "isnet_matte_box": self.isnet_matte_box.to_dict(),
            "support_dilate_px": self.support_dilate_px,
            "max_pinhole_fill_px": self.max_pinhole_fill_px,
            "min_major_component_px": self.min_major_component_px,
            "alpha_tight_threshold_u8": self.alpha_tight_threshold_u8,
            "tight_crop_pad_px": self.tight_crop_pad_px,
            "rotation_deg": self.rotation_deg,
            "dest_tight_wh": list(self.dest_tight_wh),
            "dest_bottom_center_inner_xy": list(self.dest_bottom_center_inner_xy),
            "shade_gain_rgb": list(self.shade_gain_rgb),
            "sun_gain_rgb": list(self.sun_gain_rgb),
            "sun_bias_rgb": list(self.sun_bias_rgb),
            "sun_line_inner_y": self.sun_line_inner_y,
            "sun_transition_px": self.sun_transition_px,
            "luma_gamma": self.luma_gamma,
            "contact_shadow_rgb": list(self.contact_shadow_rgb),
            "contact_shadow_band_px": self.contact_shadow_band_px,
            "contact_shadow_base_height_ratio": self.contact_shadow_base_height_ratio,
            "contact_shadow_offset_xy": list(self.contact_shadow_offset_xy),
            "contact_shadow_sigma_yx": list(self.contact_shadow_sigma_yx),
            "contact_shadow_opacity": self.contact_shadow_opacity,
            "cast_shadow_rgb": list(self.cast_shadow_rgb),
            "cast_shadow_footprint_height_ratio": self.cast_shadow_footprint_height_ratio,
            "cast_shadow_offset_xy": list(self.cast_shadow_offset_xy),
            "cast_shadow_sigma_yx": list(self.cast_shadow_sigma_yx),
            "cast_shadow_opacity": self.cast_shadow_opacity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CushionStonePlacementConfig:
        sc = data["src_crop_xyxy"]
        dwh = data["dest_tight_wh"]
        dbc = data["dest_bottom_center_inner_xy"]
        shg = data["shade_gain_rgb"]
        sug = data["sun_gain_rgb"]
        sub = data["sun_bias_rgb"]
        csr = data["contact_shadow_rgb"]
        cso = data["contact_shadow_offset_xy"]
        css = data["contact_shadow_sigma_yx"]
        ksr = data["cast_shadow_rgb"]
        kso = data["cast_shadow_offset_xy"]
        kss = data["cast_shadow_sigma_yx"]
        return cls(
            catalog_id=str(data["catalog_id"]),
            source_relpath=str(data["source_relpath"]),
            src_crop_xyxy=(int(sc[0]), int(sc[1]), int(sc[2]), int(sc[3])),
            sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["sam_prompts"]),
            sam_exclude_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data.get("sam_exclude_boxes", ())
            ),
            isnet_matte_box=OrientedMatteBoxSpec.from_dict(data["isnet_matte_box"]),
            support_dilate_px=int(data["support_dilate_px"]),
            max_pinhole_fill_px=int(data["max_pinhole_fill_px"]),
            min_major_component_px=int(data["min_major_component_px"]),
            alpha_tight_threshold_u8=int(data["alpha_tight_threshold_u8"]),
            tight_crop_pad_px=int(data["tight_crop_pad_px"]),
            rotation_deg=float(data["rotation_deg"]),
            dest_tight_wh=(int(dwh[0]), int(dwh[1])),
            dest_bottom_center_inner_xy=(int(dbc[0]), int(dbc[1])),
            shade_gain_rgb=(float(shg[0]), float(shg[1]), float(shg[2])),
            sun_gain_rgb=(float(sug[0]), float(sug[1]), float(sug[2])),
            sun_bias_rgb=(float(sub[0]), float(sub[1]), float(sub[2])),
            sun_line_inner_y=float(data["sun_line_inner_y"]),
            sun_transition_px=float(data["sun_transition_px"]),
            luma_gamma=float(data["luma_gamma"]),
            contact_shadow_rgb=(int(csr[0]), int(csr[1]), int(csr[2])),
            contact_shadow_band_px=int(data["contact_shadow_band_px"]),
            contact_shadow_base_height_ratio=float(data["contact_shadow_base_height_ratio"]),
            contact_shadow_offset_xy=(int(cso[0]), int(cso[1])),
            contact_shadow_sigma_yx=(float(css[0]), float(css[1])),
            contact_shadow_opacity=float(data["contact_shadow_opacity"]),
            cast_shadow_rgb=(int(ksr[0]), int(ksr[1]), int(ksr[2])),
            cast_shadow_footprint_height_ratio=float(data["cast_shadow_footprint_height_ratio"]),
            cast_shadow_offset_xy=(int(kso[0]), int(kso[1])),
            cast_shadow_sigma_yx=(float(kss[0]), float(kss[1])),
            cast_shadow_opacity=float(data["cast_shadow_opacity"]),
        )


@dataclass(frozen=True)
class SistersMidLayerConfig:
    """Native-resolution SAM + IS-Net segmentation and behind-rail LaMa healing for Tami & Er'ela."""

    sam_prompts: tuple[SamPointGroupPrompt, ...]
    sam_exclude_boxes: tuple[BoundingBoxPrompt, ...]
    hair_matte_boxes: tuple[OrientedMatteBoxSpec, ...]
    support_dilate_px: int
    max_pinhole_fill_px: int
    min_major_component_px: int
    rail_disocclusion_roi_box: BoundingBoxPrompt
    rail_disocclusion_dilate_px: int
    behind_rail_fill_boxes: tuple[BoundingBoxPrompt, ...]
    behind_rail_horizontal_bridge_px: int
    behind_rail_dilate_px: int
    behind_rail_feather_sigma: float
    overhanging_hand_prompts: tuple[SamPointGroupPrompt, ...]
    overhanging_hand_feather_y_band: tuple[int, int]
    overhanging_hand_max_pinhole_px: int
    overhanging_hand_min_component_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sam_prompts": [p.to_dict() for p in self.sam_prompts],
            "sam_exclude_boxes": [b.to_dict() for b in self.sam_exclude_boxes],
            "hair_matte_boxes": [m.to_dict() for m in self.hair_matte_boxes],
            "support_dilate_px": self.support_dilate_px,
            "max_pinhole_fill_px": self.max_pinhole_fill_px,
            "min_major_component_px": self.min_major_component_px,
            "rail_disocclusion_roi_box": self.rail_disocclusion_roi_box.to_dict(),
            "rail_disocclusion_dilate_px": self.rail_disocclusion_dilate_px,
            "behind_rail_fill_boxes": [b.to_dict() for b in self.behind_rail_fill_boxes],
            "behind_rail_horizontal_bridge_px": self.behind_rail_horizontal_bridge_px,
            "behind_rail_dilate_px": self.behind_rail_dilate_px,
            "behind_rail_feather_sigma": self.behind_rail_feather_sigma,
            "overhanging_hand_prompts": [p.to_dict() for p in self.overhanging_hand_prompts],
            "overhanging_hand_feather_y_band": list(self.overhanging_hand_feather_y_band),
            "overhanging_hand_max_pinhole_px": self.overhanging_hand_max_pinhole_px,
            "overhanging_hand_min_component_px": self.overhanging_hand_min_component_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SistersMidLayerConfig:
        hfy = data["overhanging_hand_feather_y_band"]
        return cls(
            sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["sam_prompts"]),
            sam_exclude_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data.get("sam_exclude_boxes", ())
            ),
            hair_matte_boxes=tuple(
                OrientedMatteBoxSpec.from_dict(m) for m in data["hair_matte_boxes"]
            ),
            support_dilate_px=int(data["support_dilate_px"]),
            max_pinhole_fill_px=int(data["max_pinhole_fill_px"]),
            min_major_component_px=int(data["min_major_component_px"]),
            rail_disocclusion_roi_box=BoundingBoxPrompt.from_dict(
                data["rail_disocclusion_roi_box"]
            ),
            rail_disocclusion_dilate_px=int(data["rail_disocclusion_dilate_px"]),
            behind_rail_fill_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["behind_rail_fill_boxes"]
            ),
            behind_rail_horizontal_bridge_px=int(data["behind_rail_horizontal_bridge_px"]),
            behind_rail_dilate_px=int(data["behind_rail_dilate_px"]),
            behind_rail_feather_sigma=float(data["behind_rail_feather_sigma"]),
            overhanging_hand_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["overhanging_hand_prompts"]
            ),
            overhanging_hand_feather_y_band=(int(hfy[0]), int(hfy[1])),
            overhanging_hand_max_pinhole_px=int(data["overhanging_hand_max_pinhole_px"]),
            overhanging_hand_min_component_px=int(data["overhanging_hand_min_component_px"]),
        )


@dataclass(frozen=True)
class GreenHandrailLayerConfig:
    """Native-resolution SAM segmentation & clean endpoint termination for the green metal railing."""

    layer_id: str
    label_he: str
    sam_prompts: tuple[SamPointGroupPrompt, ...]
    full_bg_inpaint_extra_prompts: tuple[SamPointGroupPrompt, ...]
    extra_stub_min_component_px: int
    sam_exclude_boxes: tuple[BoundingBoxPrompt, ...]
    max_pinhole_fill_px: int
    min_major_component_px: int
    right_cap_cutoff_x: int
    right_cap_feather_px: int
    left_wall_extension_inner_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "label_he": self.label_he,
            "sam_prompts": [p.to_dict() for p in self.sam_prompts],
            "full_bg_inpaint_extra_prompts": [
                p.to_dict() for p in self.full_bg_inpaint_extra_prompts
            ],
            "extra_stub_min_component_px": self.extra_stub_min_component_px,
            "sam_exclude_boxes": [b.to_dict() for b in self.sam_exclude_boxes],
            "max_pinhole_fill_px": self.max_pinhole_fill_px,
            "min_major_component_px": self.min_major_component_px,
            "right_cap_cutoff_x": self.right_cap_cutoff_x,
            "right_cap_feather_px": self.right_cap_feather_px,
            "left_wall_extension_inner_px": self.left_wall_extension_inner_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GreenHandrailLayerConfig:
        return cls(
            layer_id=str(data["layer_id"]),
            label_he=str(data["label_he"]),
            sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["sam_prompts"]),
            full_bg_inpaint_extra_prompts=tuple(
                SamPointGroupPrompt.from_dict(p)
                for p in data.get("full_bg_inpaint_extra_prompts", ())
            ),
            extra_stub_min_component_px=int(data["extra_stub_min_component_px"]),
            sam_exclude_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data.get("sam_exclude_boxes", ())
            ),
            max_pinhole_fill_px=int(data["max_pinhole_fill_px"]),
            min_major_component_px=int(data["min_major_component_px"]),
            right_cap_cutoff_x=int(data["right_cap_cutoff_x"]),
            right_cap_feather_px=int(data["right_cap_feather_px"]),
            left_wall_extension_inner_px=int(data["left_wall_extension_inner_px"]),
        )


@dataclass(frozen=True)
class FriedHouseDepthContactPinSpec:
    """Smooth elliptical world-Z contact pin on the 1500x1000 inner frame."""

    label: str
    inner_xy: tuple[float, float]
    radii_xy: tuple[float, float]
    feather_px: float
    target_z: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "inner_xy": list(self.inner_xy),
            "radii_xy": list(self.radii_xy),
            "feather_px": self.feather_px,
            "target_z": self.target_z,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FriedHouseDepthContactPinSpec:
        xy = data["inner_xy"]
        rad = data["radii_xy"]
        return cls(
            label=str(data["label"]),
            inner_xy=(float(xy[0]), float(xy[1])),
            radii_xy=(float(rad[0]), float(rad[1])),
            feather_px=float(data["feather_px"]),
            target_z=float(data["target_z"]),
        )


@dataclass(frozen=True)
class FriedHouseDepthCalibrationConfig:
    """Calibrates 2700x1800 Depth Anything V2 room relief into world Z coordinates."""

    scene_bg_z: float
    room_depth_amplitude: float
    da2_gaussian_sigma: float
    da2_z_min: float
    da2_z_max: float
    sky_norm_guard_band: float
    contact_pins: tuple[FriedHouseDepthContactPinSpec, ...]
    final_smooth_sigma: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_bg_z": self.scene_bg_z,
            "room_depth_amplitude": self.room_depth_amplitude,
            "da2_gaussian_sigma": self.da2_gaussian_sigma,
            "da2_z_min": self.da2_z_min,
            "da2_z_max": self.da2_z_max,
            "sky_norm_guard_band": self.sky_norm_guard_band,
            "contact_pins": [p.to_dict() for p in self.contact_pins],
            "final_smooth_sigma": self.final_smooth_sigma,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FriedHouseDepthCalibrationConfig:
        return cls(
            scene_bg_z=float(data["scene_bg_z"]),
            room_depth_amplitude=float(data["room_depth_amplitude"]),
            da2_gaussian_sigma=float(data["da2_gaussian_sigma"]),
            da2_z_min=float(data["da2_z_min"]),
            da2_z_max=float(data["da2_z_max"]),
            sky_norm_guard_band=float(data["sky_norm_guard_band"]),
            contact_pins=tuple(
                FriedHouseDepthContactPinSpec.from_dict(p) for p in data["contact_pins"]
            ),
            final_smooth_sigma=float(data["final_smooth_sigma"]),
        )


@dataclass(frozen=True)
class FriedHouseLayerTopologyConfig:
    """Ghost-free room depth healing and harmonic C^0/C^-1 layer topology calibration."""

    room_pin_margin_uv: float
    room_clean: RoomDepthCleanConfig
    protrusion_clean: RoomDepthCleanConfig
    protrusion_upper_percentile: float
    handrail_topology: ContactAnchoredLayerDepthConfig
    sisters_topology: ContactAnchoredLayerDepthConfig
    cushion_topology: ContactAnchoredLayerDepthConfig
    layer_verification: LayerTopologyVerificationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_pin_margin_uv": self.room_pin_margin_uv,
            "room_clean": self.room_clean.to_dict(),
            "protrusion_clean": self.protrusion_clean.to_dict(),
            "protrusion_upper_percentile": self.protrusion_upper_percentile,
            "handrail_topology": self.handrail_topology.to_dict(),
            "sisters_topology": self.sisters_topology.to_dict(),
            "cushion_topology": self.cushion_topology.to_dict(),
            "layer_verification": self.layer_verification.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FriedHouseLayerTopologyConfig:
        return cls(
            room_pin_margin_uv=float(data["room_pin_margin_uv"]),
            room_clean=RoomDepthCleanConfig.from_dict(data["room_clean"]),
            protrusion_clean=RoomDepthCleanConfig.from_dict(data["protrusion_clean"]),
            protrusion_upper_percentile=float(data["protrusion_upper_percentile"]),
            handrail_topology=ContactAnchoredLayerDepthConfig.from_dict(data["handrail_topology"]),
            sisters_topology=ContactAnchoredLayerDepthConfig.from_dict(data["sisters_topology"]),
            cushion_topology=ContactAnchoredLayerDepthConfig.from_dict(data["cushion_topology"]),
            layer_verification=LayerTopologyVerificationConfig.from_dict(
                data["layer_verification"]
            ),
        )


@dataclass(frozen=True)
class FriedHouseWindowPortalConfig:
    """
    SSOT configuration for the removable left wooden louvered window shutter occluder on the
    cream Fried house facade (`2700x1800` room canvas) revealing the live 3D portal diorama
    into `wandering_drawing_1` (`שיטוט ברישום 1`, Beta Feedback #32).
    """

    portal_id: str
    target_scene_id: str
    label_he: str
    short_badge_he: str
    shutter_label_he: str
    shutter_output_filename: str
    shutter_depth_output_filename: str
    shutter_crop_output_filename: str
    roi_outer_xyxy: tuple[int, int, int, int]
    shutter_sam_prompts: tuple[SamPointGroupPrompt, ...]
    sam_max_pinhole_fill_px: int
    sam_min_component_px: int
    white_frame_luma_guard_start: float
    white_frame_luma_guard_end: float
    shutter_edge_feather_sigma: float
    recess_seam_close_px: int
    recess_shadow_decay_px: float
    recess_grad_y0_roi: float
    recess_grad_span_px: float
    recess_base_dark_rgb: tuple[float, float, float]
    recess_lower_warm_rgb: tuple[float, float, float]
    recess_lintel_dark_rgb: tuple[float, float, float]
    recess_lintel_weight: float
    portal_aperture_outer_xyxy: tuple[int, int, int, int]
    shutter_crop_outer_xyxy: tuple[int, int, int, int]
    recess_depth: float
    rest_frame_opacity: float
    frame_render_order: float
    occluder_render_order: int
    occluder_alpha_hit_threshold: int
    occluder_z_clearance: float
    shutter_segments_x: int
    shutter_segments_y: int
    shutter_topology: ContactAnchoredLayerDepthConfig

    def portal_inner_box_px(
        self, dims: CanvasDimensionsConfig, outpaint: OutpaintingConfig
    ) -> tuple[float, float, float, float]:
        exp_w, exp_h = outpaint.expanded_dimensions(dims)
        pad_x = (exp_w - dims.width) * 0.5
        pad_y = (exp_h - dims.height) * 0.5
        x0, y0, x1, y1 = self.portal_aperture_outer_xyxy
        return (float(x0 - pad_x), float(y0 - pad_y), float(x1 - pad_x), float(y1 - pad_y))

    def shutter_inner_box_px(
        self, dims: CanvasDimensionsConfig, outpaint: OutpaintingConfig
    ) -> tuple[float, float, float, float]:
        exp_w, exp_h = outpaint.expanded_dimensions(dims)
        pad_x = (exp_w - dims.width) * 0.5
        pad_y = (exp_h - dims.height) * 0.5
        x0, y0, x1, y1 = self.shutter_crop_outer_xyxy
        return (float(x0 - pad_x), float(y0 - pad_y), float(x1 - pad_x), float(y1 - pad_y))

    def to_dict(self) -> dict[str, Any]:
        return {
            "portal_id": self.portal_id,
            "target_scene_id": self.target_scene_id,
            "label_he": self.label_he,
            "short_badge_he": self.short_badge_he,
            "shutter_label_he": self.shutter_label_he,
            "shutter_output_filename": self.shutter_output_filename,
            "shutter_depth_output_filename": self.shutter_depth_output_filename,
            "shutter_crop_output_filename": self.shutter_crop_output_filename,
            "roi_outer_xyxy": list(self.roi_outer_xyxy),
            "shutter_sam_prompts": [p.to_dict() for p in self.shutter_sam_prompts],
            "sam_max_pinhole_fill_px": self.sam_max_pinhole_fill_px,
            "sam_min_component_px": self.sam_min_component_px,
            "white_frame_luma_guard_start": self.white_frame_luma_guard_start,
            "white_frame_luma_guard_end": self.white_frame_luma_guard_end,
            "shutter_edge_feather_sigma": self.shutter_edge_feather_sigma,
            "recess_seam_close_px": self.recess_seam_close_px,
            "recess_shadow_decay_px": self.recess_shadow_decay_px,
            "recess_grad_y0_roi": self.recess_grad_y0_roi,
            "recess_grad_span_px": self.recess_grad_span_px,
            "recess_base_dark_rgb": list(self.recess_base_dark_rgb),
            "recess_lower_warm_rgb": list(self.recess_lower_warm_rgb),
            "recess_lintel_dark_rgb": list(self.recess_lintel_dark_rgb),
            "recess_lintel_weight": self.recess_lintel_weight,
            "portal_aperture_outer_xyxy": list(self.portal_aperture_outer_xyxy),
            "shutter_crop_outer_xyxy": list(self.shutter_crop_outer_xyxy),
            "recess_depth": self.recess_depth,
            "rest_frame_opacity": self.rest_frame_opacity,
            "frame_render_order": self.frame_render_order,
            "occluder_render_order": self.occluder_render_order,
            "occluder_alpha_hit_threshold": self.occluder_alpha_hit_threshold,
            "occluder_z_clearance": self.occluder_z_clearance,
            "shutter_segments_x": self.shutter_segments_x,
            "shutter_segments_y": self.shutter_segments_y,
            "shutter_topology": self.shutter_topology.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FriedHouseWindowPortalConfig:
        roi = data["roi_outer_xyxy"]
        ap = data["portal_aperture_outer_xyxy"]
        sc = data["shutter_crop_outer_xyxy"]
        b_rgb = data["recess_base_dark_rgb"]
        lw_rgb = data["recess_lower_warm_rgb"]
        ld_rgb = data["recess_lintel_dark_rgb"]
        return cls(
            portal_id=str(data["portal_id"]),
            target_scene_id=str(data["target_scene_id"]),
            label_he=str(data["label_he"]),
            short_badge_he=str(data["short_badge_he"]),
            shutter_label_he=str(data["shutter_label_he"]),
            shutter_output_filename=str(data["shutter_output_filename"]),
            shutter_depth_output_filename=str(data["shutter_depth_output_filename"]),
            shutter_crop_output_filename=str(data["shutter_crop_output_filename"]),
            roi_outer_xyxy=(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])),
            shutter_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["shutter_sam_prompts"]
            ),
            sam_max_pinhole_fill_px=int(data["sam_max_pinhole_fill_px"]),
            sam_min_component_px=int(data["sam_min_component_px"]),
            white_frame_luma_guard_start=float(data["white_frame_luma_guard_start"]),
            white_frame_luma_guard_end=float(data["white_frame_luma_guard_end"]),
            shutter_edge_feather_sigma=float(data["shutter_edge_feather_sigma"]),
            recess_seam_close_px=int(data["recess_seam_close_px"]),
            recess_shadow_decay_px=float(data["recess_shadow_decay_px"]),
            recess_grad_y0_roi=float(data["recess_grad_y0_roi"]),
            recess_grad_span_px=float(data["recess_grad_span_px"]),
            recess_base_dark_rgb=(float(b_rgb[0]), float(b_rgb[1]), float(b_rgb[2])),
            recess_lower_warm_rgb=(float(lw_rgb[0]), float(lw_rgb[1]), float(lw_rgb[2])),
            recess_lintel_dark_rgb=(float(ld_rgb[0]), float(ld_rgb[1]), float(ld_rgb[2])),
            recess_lintel_weight=float(data["recess_lintel_weight"]),
            portal_aperture_outer_xyxy=(int(ap[0]), int(ap[1]), int(ap[2]), int(ap[3])),
            shutter_crop_outer_xyxy=(int(sc[0]), int(sc[1]), int(sc[2]), int(sc[3])),
            recess_depth=float(data["recess_depth"]),
            rest_frame_opacity=float(data["rest_frame_opacity"]),
            frame_render_order=float(data["frame_render_order"]),
            occluder_render_order=int(data["occluder_render_order"]),
            occluder_alpha_hit_threshold=int(data["occluder_alpha_hit_threshold"]),
            occluder_z_clearance=float(data["occluder_z_clearance"]),
            shutter_segments_x=int(data["shutter_segments_x"]),
            shutter_segments_y=int(data["shutter_segments_y"]),
            shutter_topology=ContactAnchoredLayerDepthConfig.from_dict(data["shutter_topology"]),
        )


@dataclass(frozen=True)
class FriedHouseSceneConfig:
    """Canonical SSOT configuration for Scene 10 (`fried_house_railing`)."""

    scene_id: str
    exhibition_group: str
    title_he: str
    short_label_he: str
    icon: str
    herz_catalog_id: str
    herz_source_relpath: str
    outpaint_raw_relpath: str
    base_inner_outpaint_relpath: str
    lama_inner_output_filename: str
    bg_output_filename: str
    depth_output_filename: str
    mid_output_filename: str
    mid_depth_output_filename: str
    fg_output_filename: str
    fg_depth_output_filename: str
    railing_output_filename: str
    railing_depth_output_filename: str
    dimensions: CanvasDimensionsConfig
    outpainting: OutpaintingConfig
    depth_model: DepthModelConfig
    segmentation: SegmentationConfig
    inpainting: InpaintingConfig
    inner_wing_seam_collar_px: int
    inner_wing_color_band_px: int
    inner_wing_seam_row_sigma: float
    inner_wing_seam_decay_px: float
    min_registration_corr: float
    feather_collar_px: int
    seam_harmonic_downsample: int
    seam_harmonic_iters: int
    bore_sight_at_rest: bool
    scene_bg_z: float
    mid_z: float
    fg_z: float
    railing_z: float
    mid_render_order: int
    fg_render_order: int
    railing_render_order: int
    bg_segments_x: int
    bg_segments_y: int
    mid_segments_x: int
    mid_segments_y: int
    fg_segments_x: int
    fg_segments_y: int
    railing_segments_x: int
    railing_segments_y: int
    initial_camera_y: float
    initial_camera_z: float
    min_camera_z: float
    max_camera_z: float
    max_yaw_deg: float
    max_pitch_deg: float
    max_strafe_x: float
    max_strafe_y: float
    dolly_speed_mult: float
    particle_color_hex: int
    particle_count: int
    sisters_mid: SistersMidLayerConfig
    handrail: GreenHandrailLayerConfig
    cushion_stone_fg: CushionStonePlacementConfig
    window_portal: FriedHouseWindowPortalConfig
    depth_calibration: FriedHouseDepthCalibrationConfig
    topology: FriedHouseLayerTopologyConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "exhibition_group": self.exhibition_group,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "herz_catalog_id": self.herz_catalog_id,
            "herz_source_relpath": self.herz_source_relpath,
            "outpaint_raw_relpath": self.outpaint_raw_relpath,
            "base_inner_outpaint_relpath": self.base_inner_outpaint_relpath,
            "lama_inner_output_filename": self.lama_inner_output_filename,
            "bg_output_filename": self.bg_output_filename,
            "depth_output_filename": self.depth_output_filename,
            "mid_output_filename": self.mid_output_filename,
            "mid_depth_output_filename": self.mid_depth_output_filename,
            "fg_output_filename": self.fg_output_filename,
            "fg_depth_output_filename": self.fg_depth_output_filename,
            "railing_output_filename": self.railing_output_filename,
            "railing_depth_output_filename": self.railing_depth_output_filename,
            "dimensions": self.dimensions.to_dict(),
            "outpainting": self.outpainting.to_dict(),
            "depth_model": self.depth_model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "inpainting": self.inpainting.to_dict(),
            "inner_wing_seam_collar_px": self.inner_wing_seam_collar_px,
            "inner_wing_color_band_px": self.inner_wing_color_band_px,
            "inner_wing_seam_row_sigma": self.inner_wing_seam_row_sigma,
            "inner_wing_seam_decay_px": self.inner_wing_seam_decay_px,
            "min_registration_corr": self.min_registration_corr,
            "feather_collar_px": self.feather_collar_px,
            "seam_harmonic_downsample": self.seam_harmonic_downsample,
            "seam_harmonic_iters": self.seam_harmonic_iters,
            "bore_sight_at_rest": self.bore_sight_at_rest,
            "scene_bg_z": self.scene_bg_z,
            "mid_z": self.mid_z,
            "fg_z": self.fg_z,
            "railing_z": self.railing_z,
            "mid_render_order": self.mid_render_order,
            "fg_render_order": self.fg_render_order,
            "railing_render_order": self.railing_render_order,
            "bg_segments_x": self.bg_segments_x,
            "bg_segments_y": self.bg_segments_y,
            "mid_segments_x": self.mid_segments_x,
            "mid_segments_y": self.mid_segments_y,
            "fg_segments_x": self.fg_segments_x,
            "fg_segments_y": self.fg_segments_y,
            "railing_segments_x": self.railing_segments_x,
            "railing_segments_y": self.railing_segments_y,
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
            "sisters_mid": self.sisters_mid.to_dict(),
            "handrail": self.handrail.to_dict(),
            "cushion_stone_fg": self.cushion_stone_fg.to_dict(),
            "window_portal": self.window_portal.to_dict(),
            "depth_calibration": self.depth_calibration.to_dict(),
            "topology": self.topology.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FriedHouseSceneConfig:
        return cls(
            scene_id=str(data["scene_id"]),
            exhibition_group=str(data["exhibition_group"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            herz_catalog_id=str(data["herz_catalog_id"]),
            herz_source_relpath=str(data["herz_source_relpath"]),
            outpaint_raw_relpath=str(data["outpaint_raw_relpath"]),
            base_inner_outpaint_relpath=str(data["base_inner_outpaint_relpath"]),
            lama_inner_output_filename=str(data["lama_inner_output_filename"]),
            bg_output_filename=str(data["bg_output_filename"]),
            depth_output_filename=str(data["depth_output_filename"]),
            mid_output_filename=str(data["mid_output_filename"]),
            mid_depth_output_filename=str(data["mid_depth_output_filename"]),
            fg_output_filename=str(data["fg_output_filename"]),
            fg_depth_output_filename=str(data["fg_depth_output_filename"]),
            railing_output_filename=str(data["railing_output_filename"]),
            railing_depth_output_filename=str(data["railing_depth_output_filename"]),
            dimensions=CanvasDimensionsConfig.from_dict(data["dimensions"]),
            outpainting=OutpaintingConfig.from_dict(data["outpainting"]),
            depth_model=DepthModelConfig.from_dict(data["depth_model"]),
            segmentation=SegmentationConfig.from_dict(data["segmentation"]),
            inpainting=InpaintingConfig.from_dict(data["inpainting"]),
            inner_wing_seam_collar_px=int(data["inner_wing_seam_collar_px"]),
            inner_wing_color_band_px=int(data["inner_wing_color_band_px"]),
            inner_wing_seam_row_sigma=float(data["inner_wing_seam_row_sigma"]),
            inner_wing_seam_decay_px=float(data["inner_wing_seam_decay_px"]),
            min_registration_corr=float(data["min_registration_corr"]),
            feather_collar_px=int(data["feather_collar_px"]),
            seam_harmonic_downsample=int(data["seam_harmonic_downsample"]),
            seam_harmonic_iters=int(data["seam_harmonic_iters"]),
            bore_sight_at_rest=bool(data["bore_sight_at_rest"]),
            scene_bg_z=float(data["scene_bg_z"]),
            mid_z=float(data["mid_z"]),
            fg_z=float(data["fg_z"]),
            railing_z=float(data["railing_z"]),
            mid_render_order=int(data["mid_render_order"]),
            fg_render_order=int(data["fg_render_order"]),
            railing_render_order=int(data["railing_render_order"]),
            bg_segments_x=int(data["bg_segments_x"]),
            bg_segments_y=int(data["bg_segments_y"]),
            mid_segments_x=int(data["mid_segments_x"]),
            mid_segments_y=int(data["mid_segments_y"]),
            fg_segments_x=int(data["fg_segments_x"]),
            fg_segments_y=int(data["fg_segments_y"]),
            railing_segments_x=int(data["railing_segments_x"]),
            railing_segments_y=int(data["railing_segments_y"]),
            initial_camera_y=float(data["initial_camera_y"]),
            initial_camera_z=float(data["initial_camera_z"]),
            min_camera_z=float(data["min_camera_z"]),
            max_camera_z=float(data["max_camera_z"]),
            max_yaw_deg=float(data["max_yaw_deg"]),
            max_pitch_deg=float(data["max_pitch_deg"]),
            max_strafe_x=float(data["max_strafe_x"]),
            max_strafe_y=float(data["max_strafe_y"]),
            dolly_speed_mult=float(data["dolly_speed_mult"]),
            particle_color_hex=int(data["particle_color_hex"]),
            particle_count=int(data["particle_count"]),
            sisters_mid=SistersMidLayerConfig.from_dict(data["sisters_mid"]),
            handrail=GreenHandrailLayerConfig.from_dict(data["handrail"]),
            cushion_stone_fg=CushionStonePlacementConfig.from_dict(data["cushion_stone_fg"]),
            window_portal=FriedHouseWindowPortalConfig.from_dict(data["window_portal"]),
            depth_calibration=FriedHouseDepthCalibrationConfig.from_dict(data["depth_calibration"]),
            topology=FriedHouseLayerTopologyConfig.from_dict(data["topology"]),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, raw_json: str) -> FriedHouseSceneConfig:
        return cls.from_dict(json.loads(raw_json))


def default_fried_house_scene_config() -> FriedHouseSceneConfig:
    """Returns canonical SSOT configuration for Scene 10 (`fried_house_railing`)."""
    cfg = FriedHouseSceneConfig(
        scene_id="fried_house_railing",
        exhibition_group="family_places",
        title_he="על המעקה בבית פריד בהרצליה — עם ׳כרית ואבן׳ (2006 / 2021)",
        short_label_he="בית פריד: כרית ואבן",
        icon="🏡",
        herz_catalog_id="IMG_HERZ_0397",
        herz_source_relpath="media/herzliya_fried_house/fried2_pict0067_532e7c13.jpg",
        outpaint_raw_relpath="prototype/assets/outpaint_sources/scene10_fried_house_outpaint_raw.jpg",
        base_inner_outpaint_relpath="prototype/assets/outpaint_sources/scene10_fried_house_base_1500x1000.jpg",
        lama_inner_output_filename="scene10_fried_house_lama_inner.jpg",
        bg_output_filename="scene10_fried_house_bg_inpainted.jpg",
        depth_output_filename="scene10_fried_house_depth.png",
        mid_output_filename="scene10_fried_house_mid.png",
        mid_depth_output_filename="scene10_fried_house_mid_depth.png",
        fg_output_filename="scene10_fried_house_fg.png",
        fg_depth_output_filename="scene10_fried_house_fg_depth.png",
        railing_output_filename="scene10_fried_house_railing.png",
        railing_depth_output_filename="scene10_fried_house_railing_depth.png",
        dimensions=CanvasDimensionsConfig(width=1500, height=1000),
        outpainting=OutpaintingConfig(
            expansion_factor=1.80,
            feather_collar_px=36,
            color_match_band_px=48,
            min_registration_corr=0.88,
            gain_clamp_min=0.78,
            gain_clamp_max=1.28,
        ),
        depth_model=DepthModelConfig(
            model_id="depth-anything/Depth-Anything-V2-Small-hf",
            background_gaussian_sigma=14.0,
            depth_discontinuity_percentile_fg=62.0,
            depth_discontinuity_percentile_mid=30.0,
        ),
        segmentation=SegmentationConfig(
            rembg_model_name="isnet-general-use",
            sam_model_id="facebook/sam-vit-base",
            alpha_foreground_threshold=92,
            alpha_full_opaque_threshold=195,
            min_component_area_px=350,
            sam_morph_close_px=5,
            edge_feather_sigma=0.75,
        ),
        inpainting=InpaintingConfig(
            dilation_radius_px=13,
            mask_blur_sigma=2.0,
            jpeg_quality=92,
        ),
        inner_wing_seam_collar_px=24,
        inner_wing_color_band_px=36,
        inner_wing_seam_row_sigma=18.0,
        inner_wing_seam_decay_px=85.0,
        min_registration_corr=0.88,
        feather_collar_px=36,
        seam_harmonic_downsample=8,
        seam_harmonic_iters=240,
        bore_sight_at_rest=True,
        scene_bg_z=-310.0,
        mid_z=-278.0,
        fg_z=-276.0,
        railing_z=-274.0,
        mid_render_order=8,
        fg_render_order=9,
        railing_render_order=10,
        bg_segments_x=224,
        bg_segments_y=150,
        mid_segments_x=224,
        mid_segments_y=150,
        fg_segments_x=224,
        fg_segments_y=150,
        railing_segments_x=224,
        railing_segments_y=150,
        initial_camera_y=-56.0,
        initial_camera_z=865.0,
        min_camera_z=195.0,
        max_camera_z=1050.0,
        max_yaw_deg=34.0,
        max_pitch_deg=24.0,
        max_strafe_x=260.0,
        max_strafe_y=165.0,
        dolly_speed_mult=1.35,
        particle_color_hex=0xEFE2C6,
        particle_count=65,
        sisters_mid=SistersMidLayerConfig(
            sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=410, y0=885, x1=675, y1=1695),
                    points_xy=(
                        (535.0, 975.0),
                        (535.0, 1085.0),
                        (465.0, 1125.0),
                        (575.0, 1192.0),
                        (550.0, 1280.0),
                        (560.0, 1480.0),
                        (605.0, 1655.0),
                        (655.0, 1655.0),
                        (430.0, 1250.0),
                        (465.0, 1620.0),
                        (535.0, 1170.0),
                        (712.0, 1420.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=580, y0=920, x1=880, y1=1740),
                    points_xy=(
                        (685.0, 1005.0),
                        (705.0, 1105.0),
                        (785.0, 1150.0),
                        (800.0, 1215.0),
                        (675.0, 1295.0),
                        (765.0, 1320.0),
                        (665.0, 1480.0),
                        (775.0, 1540.0),
                        (760.0, 1685.0),
                        (825.0, 1650.0),
                        (712.0, 1400.0),
                        (712.0, 1620.0),
                        (685.0, 1186.0),
                        (855.0, 1120.0),
                        (865.0, 1450.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
                ),
            ),
            sam_exclude_boxes=(
                BoundingBoxPrompt(x0=0, y0=0, x1=405, y1=2048),
                BoundingBoxPrompt(x0=885, y0=0, x1=1358, y1=2048),
                BoundingBoxPrompt(x0=0, y0=1745, x1=1358, y1=2048),
            ),
            hair_matte_boxes=(
                OrientedMatteBoxSpec(
                    box=BoundingBoxPrompt(x0=445, y0=875, x1=640, y1=1090),
                    upright_angle_deg=0.0,
                    taper_margin_px=16,
                    sam_core_erode_px=5,
                ),
                OrientedMatteBoxSpec(
                    box=BoundingBoxPrompt(x0=608, y0=912, x1=776, y1=1115),
                    upright_angle_deg=0.0,
                    taper_margin_px=16,
                    sam_core_erode_px=5,
                ),
            ),
            support_dilate_px=35,
            max_pinhole_fill_px=9000,
            min_major_component_px=12000,
            rail_disocclusion_roi_box=BoundingBoxPrompt(x0=405, y0=880, x1=890, y1=1750),
            rail_disocclusion_dilate_px=12,
            behind_rail_fill_boxes=(
                BoundingBoxPrompt(x0=465, y0=1148, x1=782, y1=1215),
                BoundingBoxPrompt(x0=686, y0=1180, x1=736, y1=1702),
            ),
            behind_rail_horizontal_bridge_px=55,
            behind_rail_dilate_px=9,
            behind_rail_feather_sigma=0.65,
            overhanging_hand_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=770, y0=1150, x1=852, y1=1268),
                    points_xy=(
                        (802.0, 1172.0),
                        (802.0, 1205.0),
                        (792.0, 1235.0),
                        (822.0, 1244.0),
                        (776.0, 1188.0),
                        (842.0, 1192.0),
                        (838.0, 1225.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0, 0),
                ),
            ),
            overhanging_hand_feather_y_band=(1150, 1172),
            overhanging_hand_max_pinhole_px=400,
            overhanging_hand_min_component_px=250,
        ),
        handrail=GreenHandrailLayerConfig(
            layer_id="green_handrail",
            label_he="מעקה הצינור הירוק בכניסה לבית פריד",
            sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=0, y0=1085, x1=782, y1=1214),
                    points_xy=(
                        (40.0, 1104.0),
                        (200.0, 1125.0),
                        (350.0, 1145.0),
                        (430.0, 1161.0),
                        (485.0, 1166.0),
                        (535.0, 1170.0),
                        (575.0, 1174.0),
                        (620.0, 1179.0),
                        (685.0, 1186.0),
                        (750.0, 1194.0),
                        (430.0, 1146.0),
                        (485.0, 1148.0),
                        (535.0, 1150.0),
                        (620.0, 1156.0),
                        (685.0, 1162.0),
                        (485.0, 1184.0),
                        (535.0, 1188.0),
                        (575.0, 1192.0),
                        (620.0, 1198.0),
                        (685.0, 1206.0),
                        (792.0, 1198.0),
                    ),
                    point_labels=(
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=820, y0=1178, x1=1292, y1=1272),
                    points_xy=(
                        (855.0, 1200.0),
                        (980.0, 1215.0),
                        (1120.0, 1230.0),
                        (1245.0, 1243.0),
                        (1272.0, 1246.0),
                        (980.0, 1186.0),
                        (980.0, 1245.0),
                        (802.0, 1205.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 0, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=260, y0=1125, x1=335, y1=1665),
                    points_xy=(
                        (292.0, 1170.0),
                        (295.0, 1300.0),
                        (298.0, 1450.0),
                        (300.0, 1600.0),
                        (295.0, 1648.0),
                        (268.0, 1350.0),
                        (325.0, 1350.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=685, y0=1175, x1=742, y1=1792),
                    points_xy=(
                        (712.0, 1210.0),
                        (712.0, 1350.0),
                        (715.0, 1500.0),
                        (716.0, 1650.0),
                        (716.0, 1765.0),
                        (685.0, 1400.0),
                        (740.0, 1400.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=1225, y0=1218, x1=1284, y1=1935),
                    points_xy=(
                        (1250.0, 1244.0),
                        (1266.0, 1250.0),
                        (1248.0, 1268.0),
                        (1248.0, 1420.0),
                        (1250.0, 1600.0),
                        (1252.0, 1780.0),
                        (1254.0, 1915.0),
                        (1222.0, 1500.0),
                        (1284.0, 1500.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 0, 0),
                ),
            ),
            full_bg_inpaint_extra_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=1260, y0=1220, x1=1358, y1=1275),
                    points_xy=(
                        (1285.0, 1244.0),
                        (1325.0, 1249.0),
                        (1350.0, 1252.0),
                        (1320.0, 1272.0),
                    ),
                    point_labels=(1, 1, 1, 0),
                ),
            ),
            extra_stub_min_component_px=150,
            sam_exclude_boxes=(
                BoundingBoxPrompt(x0=776, y0=1162, x1=842, y1=1268),
                BoundingBoxPrompt(x0=1286, y0=1255, x1=1358, y1=1450),
            ),
            max_pinhole_fill_px=250,
            min_major_component_px=800,
            right_cap_cutoff_x=1282,
            right_cap_feather_px=6,
            left_wall_extension_inner_px=5,
        ),
        cushion_stone_fg=CushionStonePlacementConfig(
            catalog_id="IMG_BINY_0013",
            source_relpath="media/binyamin_gallery/binyamin_מגרה_כרית_אבן.jpg_6ea63fc4.jpg",
            src_crop_xyxy=(700, 800, 1800, 1960),
            sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=65, y0=60, x1=1015, y1=1115),
                    points_xy=(
                        (180.0, 180.0),
                        (450.0, 320.0),
                        (680.0, 520.0),
                        (820.0, 780.0),
                        (550.0, 880.0),
                        (490.0, 1040.0),
                        (910.0, 930.0),
                        (260.0, 760.0),
                        (650.0, 1080.0),
                        (850.0, 1040.0),
                        (180.0, 1040.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=150, y0=465, x1=465, y1=1005),
                    points_xy=(
                        (300.0, 560.0),
                        (270.0, 720.0),
                        (360.0, 760.0),
                        (240.0, 910.0),
                        (375.0, 935.0),
                        (480.0, 720.0),
                        (260.0, 1025.0),
                        (380.0, 1025.0),
                        (130.0, 850.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 0, 0, 0, 0),
                ),
            ),
            sam_exclude_boxes=(BoundingBoxPrompt(x0=0, y0=1108, x1=1100, y1=1160),),
            isnet_matte_box=OrientedMatteBoxSpec(
                box=BoundingBoxPrompt(x0=40, y0=40, x1=1040, y1=1108),
                upright_angle_deg=0.0,
                taper_margin_px=12,
                sam_core_erode_px=4,
            ),
            support_dilate_px=18,
            max_pinhole_fill_px=4000,
            min_major_component_px=10000,
            alpha_tight_threshold_u8=24,
            tight_crop_pad_px=4,
            rotation_deg=-3.0,
            dest_tight_wh=(136, 140),
            dest_bottom_center_inner_xy=(932, 858),
            shade_gain_rgb=(0.97, 0.96, 0.95),
            sun_gain_rgb=(1.09, 1.05, 0.96),
            sun_bias_rgb=(10.0, 7.0, 1.0),
            sun_line_inner_y=830.0,
            sun_transition_px=26.0,
            luma_gamma=0.98,
            contact_shadow_rgb=(42, 40, 35),
            contact_shadow_band_px=9,
            contact_shadow_base_height_ratio=0.68,
            contact_shadow_offset_xy=(-2, 2),
            contact_shadow_sigma_yx=(2.4, 4.2),
            contact_shadow_opacity=0.66,
            cast_shadow_rgb=(54, 52, 46),
            cast_shadow_footprint_height_ratio=0.22,
            cast_shadow_offset_xy=(-12, -3),
            cast_shadow_sigma_yx=(6.0, 11.5),
            cast_shadow_opacity=0.33,
        ),
        window_portal=FriedHouseWindowPortalConfig(
            portal_id="fried_house_left_window_portal",
            target_scene_id="wandering_drawing_1",
            label_he="🪟 חלון העץ בבית פריד · שער אל ׳שיטוט ברישום 1׳",
            short_badge_he="✏️ שיטוט ברישום 1",
            shutter_label_he="תריס העץ החום בחלון בית פריד",
            shutter_output_filename="scene10_fried_house_window_shutter.png",
            shutter_depth_output_filename="scene10_fried_house_window_shutter_depth.png",
            shutter_crop_output_filename="scene10_fried_house_window_shutter_crop.png",
            roi_outer_xyxy=(300, 500, 820, 940),
            shutter_sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=78, y0=80, x1=260, y1=358),
                    points_xy=(
                        (115.0, 118.0),
                        (205.0, 128.0),
                        (125.0, 220.0),
                        (215.0, 225.0),
                        (125.0, 325.0),
                        (215.0, 325.0),
                        (70.0, 100.0),
                        (72.0, 200.0),
                        (74.0, 310.0),
                        (150.0, 74.0),
                        (160.0, 364.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=252, y0=92, x1=442, y1=356),
                    points_xy=(
                        (290.0, 135.0),
                        (385.0, 145.0),
                        (295.0, 230.0),
                        (395.0, 235.0),
                        (295.0, 325.0),
                        (400.0, 325.0),
                        (448.0, 140.0),
                        (450.0, 250.0),
                        (450.0, 345.0),
                        (340.0, 88.0),
                        (340.0, 362.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
                ),
            ),
            sam_max_pinhole_fill_px=2500,
            sam_min_component_px=800,
            white_frame_luma_guard_start=136.0,
            white_frame_luma_guard_end=158.0,
            shutter_edge_feather_sigma=0.65,
            recess_seam_close_px=7,
            recess_shadow_decay_px=14.0,
            recess_grad_y0_roi=85.0,
            recess_grad_span_px=270.0,
            recess_base_dark_rgb=(28.0, 26.0, 25.0),
            recess_lower_warm_rgb=(40.0, 37.0, 34.0),
            recess_lintel_dark_rgb=(17.0, 16.0, 15.0),
            recess_lintel_weight=0.55,
            portal_aperture_outer_xyxy=(397, 619, 725, 846),
            shutter_crop_outer_xyxy=(374, 581, 742, 859),
            recess_depth=118.0,
            rest_frame_opacity=0.0,
            frame_render_order=6.5,
            occluder_render_order=6,
            occluder_alpha_hit_threshold=24,
            occluder_z_clearance=0.35,
            shutter_segments_x=224,
            shutter_segments_y=150,
            shutter_topology=ContactAnchoredLayerDepthConfig(
                bg_z=-310.0,
                room_depth_amplitude=210.0,
                room_pin_margin_uv=0.045,
                layer_z=-224.0,
                layer_depth_amplitude=24.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=12,
                min_component_area_px=200,
                auto_floor_band_px=6,
                auto_floor_max_rise_px=24,
                contact_boxes=(
                    ContactAnchorBox(
                        x0=370, y0=578, x1=746, y1=862, mode="perimeter_rim", band_px=3
                    ),
                ),
                neural_relief_amplitude=1.6,
                contact_relief_taper_px=6.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.25,
                interior_smooth_sigma_px=1.4,
                overflow_tolerance_world=0.5,
            ),
        ),
        depth_calibration=FriedHouseDepthCalibrationConfig(
            scene_bg_z=-310.0,
            room_depth_amplitude=210.0,
            da2_gaussian_sigma=14.0,
            da2_z_min=-302.0,
            da2_z_max=-108.0,
            sky_norm_guard_band=0.04,
            contact_pins=(
                FriedHouseDepthContactPinSpec(
                    label="left_window_portal_wall",
                    inner_xy=(-42.0, 320.0),
                    radii_xy=(210.0, 170.0),
                    feather_px=75.0,
                    target_z=-221.0,
                ),
            ),
            final_smooth_sigma=4.0,
        ),
        topology=FriedHouseLayerTopologyConfig(
            room_pin_margin_uv=0.045,
            room_clean=RoomDepthCleanConfig(
                alpha_threshold_u8=20,
                mask_dilation_px=18,
                extra_clean_boxes_expanded=(),
                preserve_boxes_expanded=(),
                post_smooth_sigma_px=3.0,
                interior_blend_band_px=10.0,
            ),
            protrusion_clean=RoomDepthCleanConfig(
                alpha_threshold_u8=20,
                mask_dilation_px=16,
                extra_clean_boxes_expanded=(),
                preserve_boxes_expanded=(),
                post_smooth_sigma_px=3.0,
                interior_blend_band_px=10.0,
            ),
            protrusion_upper_percentile=98.0,
            handrail_topology=ContactAnchoredLayerDepthConfig(
                bg_z=-310.0,
                room_depth_amplitude=210.0,
                room_pin_margin_uv=0.045,
                layer_z=-274.0,
                layer_depth_amplitude=148.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=16,
                min_component_area_px=1,
                auto_floor_band_px=8,
                auto_floor_max_rise_px=48,
                contact_boxes=(
                    ContactAnchorBox(
                        x0=412, y0=528, x1=430, y1=550, mode="left_band", band_px=10
                    ),
                    ContactAnchorBox(
                        x0=548, y0=810, x1=578, y1=842, mode="bottom_band", band_px=10
                    ),
                    ContactAnchorBox(
                        x0=752, y0=832, x1=782, y1=862, mode="bottom_band", band_px=10
                    ),
                    ContactAnchorBox(
                        x0=1016, y0=918, x1=1046, y1=948, mode="bottom_band", band_px=10
                    ),
                ),
                neural_relief_amplitude=2.5,
                contact_relief_taper_px=8.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.0,
                interior_smooth_sigma_px=1.5,
                overflow_tolerance_world=0.5,
            ),
            sisters_topology=ContactAnchoredLayerDepthConfig(
                bg_z=-310.0,
                room_depth_amplitude=210.0,
                room_pin_margin_uv=0.045,
                layer_z=-278.0,
                layer_depth_amplitude=148.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=16,
                min_component_area_px=1,
                auto_floor_band_px=10,
                auto_floor_max_rise_px=48,
                contact_boxes=(
                    ContactAnchorBox(
                        x0=698, y0=785, x1=768, y1=816, mode="bottom_band", band_px=10
                    ),
                    ContactAnchorBox(
                        x0=770, y0=820, x1=840, y1=848, mode="bottom_band", band_px=10
                    ),
                ),
                neural_relief_amplitude=5.5,
                contact_relief_taper_px=10.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.0,
                interior_smooth_sigma_px=2.0,
                overflow_tolerance_world=0.5,
            ),
            cushion_topology=ContactAnchoredLayerDepthConfig(
                bg_z=-310.0,
                room_depth_amplitude=210.0,
                room_pin_margin_uv=0.045,
                layer_z=-276.0,
                layer_depth_amplitude=148.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=16,
                min_component_area_px=1,
                auto_floor_band_px=10,
                auto_floor_max_rise_px=48,
                contact_boxes=(
                    ContactAnchorBox(
                        x0=855, y0=815, x1=1003, y1=866, mode="bottom_band", band_px=10
                    ),
                ),
                neural_relief_amplitude=14.0,
                contact_relief_taper_px=10.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.0,
                interior_smooth_sigma_px=1.8,
                overflow_tolerance_world=0.5,
            ),
            layer_verification=LayerTopologyVerificationConfig(
                alpha_threshold_u8=20,
                rim_erosion_px=2,
                rim_exterior_probe_px=8,
                max_allowed_contact_dz_world=2.0,
                max_allowed_p95_contact_dz_world=1.2,
                max_allowed_rim_jump_world=2.5,
                host_monotonicity_tolerance_world=2.0,
            ),
        ),
    )
    round_trip = FriedHouseSceneConfig.from_json(cfg.to_json())
    if round_trip != cfg:
        raise RuntimeError("FriedHouseSceneConfig failed lossless JSON round-trip verification.")
    return cfg


# ============================================================================
# Modular Vision & Spatial Asset Synthesis Helpers
# ============================================================================


def compute_herz_inner_placement(
    herz_size: tuple[int, int], dims: CanvasDimensionsConfig
) -> tuple[int, int, int, int]:
    """
    Computes isotropic full-height placement `(x0, y0, width, height)` of portrait `IMG_HERZ_0397`
    inside the `1500x1000` inner WebGL canvas so 100% of vertical house + ramp legs are preserved.
    """
    src_w, src_h = herz_size
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source image size: {herz_size}")
    placed_h = dims.height
    placed_w = int(round(src_w * (float(placed_h) / float(src_h))))
    if placed_w % 2 == 1:
        placed_w += 1
    x0 = (dims.width - placed_w) // 2
    y0 = 0
    return x0, y0, placed_w, placed_h


def _place_native_rgba_into_inner_canvas(
    native_rgb_u8: np.ndarray,
    native_alpha_u8: np.ndarray,
    dims: CanvasDimensionsConfig,
) -> Image.Image:
    """Resizes native `1358x2048` RGBA cutout isotropically into centered `1500x1000` canvas."""
    h_nat, w_nat = native_alpha_u8.shape
    x0, y0, pw, ph = compute_herz_inner_placement((w_nat, h_nat), dims)
    patch = Image.fromarray(np.dstack([native_rgb_u8, native_alpha_u8]), mode="RGBA").resize(
        (pw, ph), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGBA", (dims.width, dims.height), (0, 0, 0, 0))
    canvas.alpha_composite(patch, dest=(x0, y0))
    return canvas


def _bridge_behind_rail_silhouette_gaps(
    sisters_alpha_u8: np.ndarray,
    rail_alpha_u8: np.ndarray,
    sisters_cfg: SistersMidLayerConfig,
) -> np.ndarray:
    """
    Fills narrow horizontal-bar and vertical-post occlusion slits inside Tami & Er'ela's body
    silhouette where the green handrail crosses in front of their clothes, without expanding
    their outer body contour into background pixels.
    """
    bridged = sisters_alpha_u8.copy()
    bw = sisters_cfg.behind_rail_horizontal_bridge_px
    if bw % 2 == 0:
        bw += 1
    iso_close = cv2.morphologyEx(
        (sisters_alpha_u8 > 64).astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bw, bw)),
    )
    body_envelope = iso_close > 0
    rz_k = sisters_cfg.behind_rail_dilate_px
    if rz_k % 2 == 0:
        rz_k += 1
    rail_zone = cv2.dilate(
        (rail_alpha_u8 > 16).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rz_k, rz_k)),
    ) > 0

    fill_region = np.zeros_like(body_envelope)
    for b in sisters_cfg.behind_rail_fill_boxes:
        fill_region[b.y0 : b.y1, b.x0 : b.x1] = True

    heal_mask = body_envelope & rail_zone & fill_region
    bridged = np.where(heal_mask, 255, bridged).astype(np.uint8)
    bridged_soft = ndi.gaussian_filter(
        bridged.astype(np.float32), sigma=sisters_cfg.behind_rail_feather_sigma
    )
    return np.where(heal_mask, np.clip(np.round(bridged_soft), 0, 255), sisters_alpha_u8).astype(
        np.uint8
    )


def _segment_point_groups_with_embeddings(
    sam_extractor: SamBoxMaskExtractor,
    rgb_image: Image.Image,
    image_embeddings: torch.Tensor,
    point_groups: Sequence[SamPointGroupPrompt],
    exclude_boxes: Sequence[BoundingBoxPrompt],
    max_pinhole_fill_px: int,
    min_major_component_px: int,
) -> np.ndarray:
    """Decodes SAM masks reusing precomputed ViT `image_embeddings` to avoid redundant encoder passes."""
    width, height = rgb_image.size
    union_bool = np.zeros((height, width), dtype=bool)
    for pg in point_groups:
        box_xyxy = [float(v) for v in pg.box.as_xyxy_list()]
        pts = [[float(px), float(py)] for px, py in pg.points_xy]
        lbls = [int(lbl) for lbl in pg.point_labels]
        inp = sam_extractor.processor(
            rgb_image,
            input_boxes=[[box_xyxy]],
            input_points=[[pts]],
            input_labels=[[lbls]],
            return_tensors="pt",
        ).to(sam_extractor.device)
        with torch.no_grad():
            out = sam_extractor.model(
                image_embeddings=image_embeddings,
                input_points=inp["input_points"],
                input_labels=inp["input_labels"],
                input_boxes=inp["input_boxes"],
                multimask_output=True,
            )
        masks = sam_extractor.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu(),
        )[0]
        scores = out.iou_scores.cpu()[0, 0]
        best_idx = int(scores.argmax().item())
        union_bool |= masks[0, best_idx].numpy()

    combined = (union_bool.astype(np.uint8)) * 255
    for eb in exclude_boxes:
        combined[eb.y0 : eb.y1, eb.x0 : eb.x1] = 0

    if sam_extractor.cfg.sam_morph_close_px > 1:
        k = sam_extractor.cfg.sam_morph_close_px
        if k % 2 == 0:
            k += 1
        combined = cv2.morphologyEx(
            combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )

    if max_pinhole_fill_px > 0:
        inv = (combined < 128).astype(np.uint8)
        nh, h_lbls, h_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        for hid in range(1, nh):
            hx = int(h_stats[hid, cv2.CC_STAT_LEFT])
            hy = int(h_stats[hid, cv2.CC_STAT_TOP])
            hw = int(h_stats[hid, cv2.CC_STAT_WIDTH])
            hh = int(h_stats[hid, cv2.CC_STAT_HEIGHT])
            touches = hx == 0 or hy == 0 or (hx + hw >= width) or (hy + hh >= height)
            if not touches and int(h_stats[hid, cv2.CC_STAT_AREA]) < max_pinhole_fill_px:
                combined[h_lbls == hid] = 255

    if min_major_component_px > 0:
        fg_bin = (combined >= 128).astype(np.uint8)
        nl, lbls, stats, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=8)
        clean_bin = np.zeros_like(combined)
        for cid in range(1, nl):
            if int(stats[cid, cv2.CC_STAT_AREA]) >= min_major_component_px:
                clean_bin[lbls == cid] = 255
        combined = clean_bin

    alpha_f = combined.astype(np.float32)
    if sam_extractor.cfg.edge_feather_sigma > 0:
        alpha_f = ndi.gaussian_filter(alpha_f, sigma=sam_extractor.cfg.edge_feather_sigma)
    return np.clip(np.round(alpha_f), 0, 255).astype(np.uint8)


def _extract_overhanging_hands_native_rgba(
    herz_rgb: Image.Image,
    image_embeddings: torch.Tensor,
    sisters_cfg: SistersMidLayerConfig,
    sam_extractor: SamBoxMaskExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts Tami's overhanging left hand/cigarette resting over the front face of the green
    handrail so it stays locked 1:1 to the rail plane under 6-DOF camera motion.
    """
    _, h_nat = herz_rgb.size
    hand_sam_u8 = _segment_point_groups_with_embeddings(
        sam_extractor=sam_extractor,
        rgb_image=herz_rgb,
        image_embeddings=image_embeddings,
        point_groups=sisters_cfg.overhanging_hand_prompts,
        exclude_boxes=(),
        max_pinhole_fill_px=sisters_cfg.overhanging_hand_max_pinhole_px,
        min_major_component_px=sisters_cfg.overhanging_hand_min_component_px,
    )
    hy0, hy1 = sisters_cfg.overhanging_hand_feather_y_band
    hand_f = hand_sam_u8.astype(np.float32)
    yy = np.arange(h_nat, dtype=np.float32)[:, None]
    hand_ramp = np.clip((yy - float(hy0)) / max(1.0, float(hy1 - hy0)), 0.0, 1.0)
    hand_ramp_s = hand_ramp * hand_ramp * (3.0 - 2.0 * hand_ramp)
    hand_f *= hand_ramp_s

    over_alpha_u8 = np.clip(np.round(hand_f), 0, 255).astype(np.uint8)
    over_rgb_u8 = decontaminate_matte_edge_rgb(np.asarray(herz_rgb, dtype=np.uint8), over_alpha_u8)
    return over_rgb_u8, over_alpha_u8


def build_sisters_railing_and_clean_herz(
    herz_rgb: Image.Image,
    cfg: FriedHouseSceneConfig,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """
    Synthesizes from native `IMG_HERZ_0397` (`1358x2048`):
      1. `mid_rgba` (`1500x1000`): Tami & Er'ela with IS-Net hair matting and LaMa-healed garments
         behind the green railing post & top bar.
      2. `railing_rgba` (`1500x1000`): Green metal handrail + 3 vertical legs + overhanging hands.
      3. `herz_clean_inner_rgb` (`664x1000`): LaMa-inpainted person-free, railing-free entrance plate.
    """
    w_nat, h_nat = herz_rgb.size
    herz_np = np.asarray(herz_rgb, dtype=np.uint8)

    base_inputs = sam_extractor.processor(herz_rgb, return_tensors="pt").to(sam_extractor.device)
    with torch.no_grad():
        herz_embeddings = sam_extractor.model.get_image_embeddings(base_inputs["pixel_values"])

    sisters_sam_u8 = _segment_point_groups_with_embeddings(
        sam_extractor=sam_extractor,
        rgb_image=herz_rgb,
        image_embeddings=herz_embeddings,
        point_groups=cfg.sisters_mid.sam_prompts,
        exclude_boxes=cfg.sisters_mid.sam_exclude_boxes,
        max_pinhole_fill_px=cfg.sisters_mid.max_pinhole_fill_px,
        min_major_component_px=cfg.sisters_mid.min_major_component_px,
    )
    sisters_alpha_u8 = merge_sam_and_oriented_mattes_u8(
        rgb_image=herz_rgb,
        sam_u8=sisters_sam_u8,
        oriented_boxes=cfg.sisters_mid.hair_matte_boxes,
        support_dilate_px=cfg.sisters_mid.support_dilate_px,
        right_edge_fade_x_start=None,
        rembg_session=matte_extractor.session,
    )
    for eb in cfg.sisters_mid.sam_exclude_boxes:
        sisters_alpha_u8[eb.y0 : eb.y1, eb.x0 : eb.x1] = 0

    rail_sam_u8 = _segment_point_groups_with_embeddings(
        sam_extractor=sam_extractor,
        rgb_image=herz_rgb,
        image_embeddings=herz_embeddings,
        point_groups=cfg.handrail.sam_prompts,
        exclude_boxes=cfg.handrail.sam_exclude_boxes,
        max_pinhole_fill_px=cfg.handrail.max_pinhole_fill_px,
        min_major_component_px=cfg.handrail.min_major_component_px,
    )

    # Smooth rounded right cap just past the rightmost vertical railing leg
    cap_x = cfg.handrail.right_cap_cutoff_x
    cap_f = max(2, cfg.handrail.right_cap_feather_px)
    xx_nat = np.arange(w_nat, dtype=np.float32)[None, :]
    cap_ramp = np.clip((float(cap_x) - xx_nat) / float(cap_f), 0.0, 1.0)
    cap_env = cap_ramp * cap_ramp * (3.0 - 2.0 * cap_ramp)
    rail_alpha_u8 = np.clip(np.round(rail_sam_u8.astype(np.float32) * cap_env), 0, 255).astype(
        np.uint8
    )

    # Full rail mask (including rightmost stub to x=1358) for complete background LaMa removal
    if cfg.handrail.full_bg_inpaint_extra_prompts:
        extra_stub_u8 = _segment_point_groups_with_embeddings(
            sam_extractor=sam_extractor,
            rgb_image=herz_rgb,
            image_embeddings=herz_embeddings,
            point_groups=cfg.handrail.full_bg_inpaint_extra_prompts,
            exclude_boxes=cfg.handrail.sam_exclude_boxes,
            max_pinhole_fill_px=cfg.handrail.max_pinhole_fill_px,
            min_major_component_px=cfg.handrail.extra_stub_min_component_px,
        )
        full_rail_for_bg_u8 = np.maximum(rail_sam_u8, extra_stub_u8)
    else:
        full_rail_for_bg_u8 = rail_sam_u8

    # Heal green handrail & middle vertical post across Tami & Er'ela's garments inside tight ROI
    roi = cfg.sisters_mid.rail_disocclusion_roi_box
    roi_rgb = herz_rgb.crop((roi.x0, roi.y0, roi.x1, roi.y1))
    roi_rail_mask = np.where(rail_alpha_u8[roi.y0 : roi.y1, roi.x0 : roi.x1] > 8, 255, 0).astype(
        np.uint8
    )
    roi_healed = lama_inpainter.inpaint_background(
        roi_rgb,
        roi_rail_mask,
        dilation_radius_px=cfg.sisters_mid.rail_disocclusion_dilate_px,
    )
    sisters_disocc_np = herz_np.copy()
    sisters_disocc_np[roi.y0 : roi.y1, roi.x0 : roi.x1, :] = np.asarray(roi_healed, dtype=np.uint8)

    sisters_bridged_alpha_u8 = _bridge_behind_rail_silhouette_gaps(
        sisters_alpha_u8=sisters_alpha_u8,
        rail_alpha_u8=rail_alpha_u8,
        sisters_cfg=cfg.sisters_mid,
    )
    sisters_decontam_rgb = decontaminate_matte_edge_rgb(sisters_disocc_np, sisters_bridged_alpha_u8)
    mid_rgba = _place_native_rgba_into_inner_canvas(
        native_rgb_u8=sisters_decontam_rgb,
        native_alpha_u8=sisters_bridged_alpha_u8,
        dims=cfg.dimensions,
    )

    # Build green handrail layer + overhanging hands resting on top of the bar
    over_rgb_u8, over_alpha_u8 = _extract_overhanging_hands_native_rgba(
        herz_rgb=herz_rgb,
        image_embeddings=herz_embeddings,
        sisters_cfg=cfg.sisters_mid,
        sam_extractor=sam_extractor,
    )
    rail_clean_alpha_u8 = np.where(over_alpha_u8 > 180, 0, rail_alpha_u8).astype(np.uint8)
    rail_decontam_rgb = decontaminate_matte_edge_rgb(herz_np, rail_clean_alpha_u8)

    railing_rgba = _place_native_rgba_into_inner_canvas(
        native_rgb_u8=rail_decontam_rgb,
        native_alpha_u8=rail_clean_alpha_u8,
        dims=cfg.dimensions,
    )
    over_inner_rgba = _place_native_rgba_into_inner_canvas(
        native_rgb_u8=over_rgb_u8,
        native_alpha_u8=over_alpha_u8,
        dims=cfg.dimensions,
    )
    railing_rgba.alpha_composite(over_inner_rgba)

    # Extend left wall-mount flange slightly into the front house wing corner so panning shows zero gap
    x0_inner, _, pw, ph = compute_herz_inner_placement((w_nat, h_nat), cfg.dimensions)
    ext_px = cfg.handrail.left_wall_extension_inner_px
    if ext_px > 0 and x0_inner >= ext_px:
        rail_arr = np.asarray(railing_rgba, dtype=np.uint8).copy()
        src_col = rail_arr[:, x0_inner : x0_inner + 2, :]
        active_rows = np.where(src_col[:, :, 3].max(axis=1) > 48)[0]
        if active_rows.size > 0:
            ry0 = max(0, int(active_rows.min()) - 1)
            ry1 = min(cfg.dimensions.height, int(active_rows.max()) + 2)
            col_rgba = rail_arr[ry0:ry1, x0_inner + 1 : x0_inner + 2, :]
            for dx in range(1, ext_px + 1):
                rail_arr[ry0:ry1, x0_inner - dx : x0_inner - dx + 1, :] = col_rgba
            railing_rgba = Image.fromarray(rail_arr, mode="RGBA")

    # Inner target-resolution LaMa background inpainting (664x1000 exact 1:1 WebGL grid)
    union_bg_mask_u8 = np.where(
        (sisters_bridged_alpha_u8 > 8) | (full_rail_for_bg_u8 > 8), 255, 0
    ).astype(np.uint8)
    herz_inner_orig = herz_rgb.resize((pw, ph), Image.Resampling.LANCZOS)
    mask_inner_u8 = np.where(
        np.asarray(Image.fromarray(union_bg_mask_u8).resize((pw, ph), Image.Resampling.BILINEAR))
        > 16,
        255,
        0,
    ).astype(np.uint8)
    herz_clean_inner_rgb = lama_inpainter.inpaint_background(
        herz_inner_orig,
        mask_inner_u8,
        dilation_radius_px=cfg.inpainting.dilation_radius_px,
        mask_blur_sigma=cfg.inpainting.mask_blur_sigma,
    )

    return mid_rgba, railing_rgba, herz_clean_inner_rgb


def _compose_inner_canvas_with_outpaint_wings(
    center_portrait_rgb: Image.Image,
    outpaint_inner_crop_rgb: Image.Image,
    x0: int,
    collar_px: int,
    color_band_px: int,
    seam_row_sigma: float,
    seam_decay_px: float,
    gain_bounds: tuple[float, float],
) -> Image.Image:
    """
    Harmonizes the left (`x < x0`) and right (`x > x0 + pw`) 3:2 outpaint wings to the exact
    vertical border strips of `center_portrait_rgb` (`664x1000`) and cosine-feathers the join.
    """
    pw, ph = center_portrait_rgb.size
    iw, ih = outpaint_inner_crop_rgb.size
    if ih != ph:
        raise ValueError(f"Height mismatch: portrait {ph} vs inner canvas {ih}")

    port_np = np.asarray(center_portrait_rgb.convert("RGB"), dtype=np.float32)
    wing_np = np.asarray(outpaint_inner_crop_rgb.convert("RGB"), dtype=np.float32).copy()

    band = min(color_band_px, pw // 4)
    g_min, g_max = gain_bounds

    # Harmonize left wing (0 .. x0 + collar_px) to left border of portrait
    left_ref = port_np[:, :band, :]
    left_src = wing_np[:, x0 : x0 + band, :]
    for ch in range(3):
        gain = float(np.clip(np.std(left_ref[:, :, ch]) / max(1.0, np.std(left_src[:, :, ch])), g_min, g_max))
        bias = float(np.mean(left_ref[:, :, ch])) - gain * float(np.mean(left_src[:, :, ch]))
        wing_np[:, : x0 + band, ch] = wing_np[:, : x0 + band, ch] * gain + bias

    # Harmonize right wing (x0 + pw - collar_px .. iw) to right border of portrait
    right_ref = port_np[:, pw - band :, :]
    right_src = wing_np[:, x0 + pw - band : x0 + pw, :]
    for ch in range(3):
        gain = float(np.clip(np.std(right_ref[:, :, ch]) / max(1.0, np.std(right_src[:, :, ch])), g_min, g_max))
        bias = float(np.mean(right_ref[:, :, ch])) - gain * float(np.mean(right_src[:, :, ch]))
        wing_np[:, x0 + pw - band :, ch] = wing_np[:, x0 + pw - band :, ch] * gain + bias

    # Per-row smooth bias continuity across left and right vertical seams
    left_row_diff = ndi.gaussian_filter1d(
        port_np[:, 0, :] - wing_np[:, x0, :], sigma=seam_row_sigma, axis=0
    )
    right_row_diff = ndi.gaussian_filter1d(
        port_np[:, pw - 1, :] - wing_np[:, x0 + pw - 1, :], sigma=seam_row_sigma, axis=0
    )
    left_decay = np.exp(
        -np.maximum(0.0, float(x0) - np.arange(x0 + collar_px, dtype=np.float32))
        / max(1.0, seam_decay_px)
    )[None, :, None]
    wing_np[:, : x0 + collar_px, :] += left_row_diff[:, None, :] * left_decay

    right_len = iw - (x0 + pw - collar_px)
    right_decay = np.exp(
        -np.maximum(0.0, np.arange(right_len, dtype=np.float32) - float(collar_px))
        / max(1.0, seam_decay_px)
    )[None, :, None]
    wing_np[:, x0 + pw - collar_px :, :] += right_row_diff[:, None, :] * right_decay

    alpha_x = np.ones(pw, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, collar_px, dtype=np.float32))
    alpha_x[:collar_px] = ramp
    alpha_x[-collar_px:] = ramp[::-1]
    w_port = alpha_x[None, :, None]

    out_np = wing_np.copy()
    out_np[:, x0 : x0 + pw, :] = w_port * port_np + (1.0 - w_port) * wing_np[:, x0 : x0 + pw, :]
    return Image.fromarray(np.clip(np.round(out_np), 0, 255).astype(np.uint8), mode="RGB")


def build_fried_house_background_plates(
    herz_orig_rgb: Image.Image,
    herz_clean_inner_rgb: Image.Image,
    raw_outpaint_rgb: Image.Image,
    cfg: FriedHouseSceneConfig,
) -> tuple[Image.Image, Image.Image, Image.Image, float]:
    """
    Builds:
      1. `base_inner_rgb` (`1500x1000`): archival `IMG_HERZ_0397` + seamless 3:2 outpaint wings.
      2. `lama_inner_rgb` (`1500x1000`): LaMa-cleaned `IMG_HERZ_0397` + seamless 3:2 outpaint wings.
      3. `expanded_bg_rgb` (`2700x1800`): center-locked, Poisson/Laplacian seam-harmonized room plate.
      4. `reg_corr`: registration correlation verified via `compose_outpainted_room`.
    """
    exp_w, exp_h = cfg.outpainting.expanded_dimensions(cfg.dimensions)
    pad_x = (exp_w - cfg.dimensions.width) // 2
    pad_y = (exp_h - cfg.dimensions.height) // 2

    outpaint_exp = raw_outpaint_rgb.convert("RGB").resize((exp_w, exp_h), Image.Resampling.LANCZOS)
    outpaint_inner_crop = outpaint_exp.crop(
        (pad_x, pad_y, pad_x + cfg.dimensions.width, pad_y + cfg.dimensions.height)
    )

    x0, _, pw, ph = compute_herz_inner_placement(herz_orig_rgb.size, cfg.dimensions)
    herz_orig_inner = herz_orig_rgb.resize((pw, ph), Image.Resampling.LANCZOS)
    gain_bounds = (cfg.outpainting.gain_clamp_min, cfg.outpainting.gain_clamp_max)

    base_inner_rgb = _compose_inner_canvas_with_outpaint_wings(
        center_portrait_rgb=herz_orig_inner,
        outpaint_inner_crop_rgb=outpaint_inner_crop,
        x0=x0,
        collar_px=cfg.inner_wing_seam_collar_px,
        color_band_px=cfg.inner_wing_color_band_px,
        seam_row_sigma=cfg.inner_wing_seam_row_sigma,
        seam_decay_px=cfg.inner_wing_seam_decay_px,
        gain_bounds=gain_bounds,
    )
    lama_inner_rgb = _compose_inner_canvas_with_outpaint_wings(
        center_portrait_rgb=herz_clean_inner_rgb,
        outpaint_inner_crop_rgb=outpaint_inner_crop,
        x0=x0,
        collar_px=cfg.inner_wing_seam_collar_px,
        color_band_px=cfg.inner_wing_color_band_px,
        seam_row_sigma=cfg.inner_wing_seam_row_sigma,
        seam_decay_px=cfg.inner_wing_seam_decay_px,
        gain_bounds=gain_bounds,
    )

    expanded_bg_rgb, reg_corr = compose_outpainted_room(
        inner_rgb=lama_inner_rgb,
        raw_outpaint_rgb=raw_outpaint_rgb,
        dims=cfg.dimensions,
        outpaint_cfg=cfg.outpainting,
        min_registration_corr=cfg.min_registration_corr,
        feather_collar_px=cfg.feather_collar_px,
        harmonize_seam_field=True,
        seam_harmonic_downsample=cfg.seam_harmonic_downsample,
        seam_harmonic_iters=cfg.seam_harmonic_iters,
    )
    return base_inner_rgb, lama_inner_rgb, expanded_bg_rgb, reg_corr


def build_cushion_stone_fg_layer(
    project_root: Path,
    cfg: FriedHouseSceneConfig,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
) -> Image.Image:
    """
    Extracts `"מגרה – כרית – אבן"` (`IMG_BINY_0013`) via hybrid SAM + IS-Net matting, harmonizes its
    studio lighting to the dappled Herzliya afternoon sunlight on the lower-right concrete ramp,
    and composites a two-layer physical contact + directional cast shadow onto `1500x1000`.
    """
    cs_cfg = cfg.cushion_stone_fg
    src_path = project_root / cs_cfg.source_relpath
    if not src_path.is_file():
        raise FileNotFoundError(f"Required Cushion-and-Stone source image missing: {src_path}")

    full_biny = Image.open(src_path).convert("RGB")
    cx0, cy0, cx1, cy1 = cs_cfg.src_crop_xyxy
    crop_rgb = full_biny.crop((cx0, cy0, cx1, cy1))

    sam_u8 = sam_extractor.segment_point_groups(
        rgb_image=crop_rgb,
        point_groups=cs_cfg.sam_prompts,
        exclude_boxes=cs_cfg.sam_exclude_boxes,
        max_pinhole_fill_px=cs_cfg.max_pinhole_fill_px,
        min_major_component_px=cs_cfg.min_major_component_px,
    )
    hybrid_alpha_u8 = merge_sam_and_oriented_mattes_u8(
        rgb_image=crop_rgb,
        sam_u8=sam_u8,
        oriented_boxes=(cs_cfg.isnet_matte_box,),
        support_dilate_px=cs_cfg.support_dilate_px,
        right_edge_fade_x_start=None,
        rembg_session=matte_extractor.session,
    )
    for eb in cs_cfg.sam_exclude_boxes:
        hybrid_alpha_u8[eb.y0 : eb.y1, eb.x0 : eb.x1] = 0

    decontam_rgb = decontaminate_matte_edge_rgb(np.asarray(crop_rgb, dtype=np.uint8), hybrid_alpha_u8)
    rgba_crop = Image.fromarray(np.dstack([decontam_rgb, hybrid_alpha_u8]), mode="RGBA")
    if abs(cs_cfg.rotation_deg) > 0.05:
        rgba_crop = rgba_crop.rotate(
            cs_cfg.rotation_deg, resample=Image.Resampling.BICUBIC, expand=True
        )

    rot_arr = np.asarray(rgba_crop, dtype=np.uint8)
    ys, xs = np.where(rot_arr[:, :, 3] >= cs_cfg.alpha_tight_threshold_u8)
    if ys.size == 0:
        raise RuntimeError("Cushion-and-Stone neural segmentation returned an empty foreground mask.")

    pad = cs_cfg.tight_crop_pad_px
    ty0 = max(0, int(ys.min()) - pad)
    ty1 = min(rot_arr.shape[0], int(ys.max()) + pad + 1)
    tx0 = max(0, int(xs.min()) - pad)
    tx1 = min(rot_arr.shape[1], int(xs.max()) + pad + 1)
    tight_rgba = rgba_crop.crop((tx0, ty0, tx1, ty1))

    tw, th = cs_cfg.dest_tight_wh
    scaled_rgba = tight_rgba.resize((tw, th), Image.Resampling.LANCZOS)
    sc_arr = np.asarray(scaled_rgba, dtype=np.float32)
    sc_rgb = sc_arr[:, :, :3]
    sc_alpha = sc_arr[:, :, 3]

    dest_cx, dest_bot_y = cs_cfg.dest_bottom_center_inner_xy
    dest_x0 = int(round(dest_cx - tw * 0.5))
    dest_y0 = int(round(dest_bot_y - th))

    # Spatially-graded open-shade -> afternoon-sunbeam tonal harmonization
    canvas_ys = float(dest_y0) + np.arange(th, dtype=np.float32)[:, None]
    u_sun = np.clip(
        (canvas_ys - (cs_cfg.sun_line_inner_y - cs_cfg.sun_transition_px * 0.5))
        / max(1.0, cs_cfg.sun_transition_px),
        0.0,
        1.0,
    )
    w_sun = (u_sun * u_sun * (3.0 - 2.0 * u_sun))[:, :, None]

    shade_g = np.array(cs_cfg.shade_gain_rgb, dtype=np.float32)[None, None, :]
    sun_g = np.array(cs_cfg.sun_gain_rgb, dtype=np.float32)[None, None, :]
    sun_b = np.array(cs_cfg.sun_bias_rgb, dtype=np.float32)[None, None, :]

    gamma_rgb = 255.0 * np.power(np.clip(sc_rgb / 255.0, 0.0, 1.0), cs_cfg.luma_gamma)
    shade_rgb = gamma_rgb * shade_g
    sun_rgb = gamma_rgb * sun_g + sun_b
    relit_rgb_u8 = np.clip(np.round((1.0 - w_sun) * shade_rgb + w_sun * sun_rgb), 0, 255).astype(
        np.uint8
    )
    relit_patch = Image.fromarray(
        np.dstack([relit_rgb_u8, np.clip(np.round(sc_alpha), 0, 255).astype(np.uint8)]), mode="RGBA"
    )

    # Synthesize grounded concrete-ramp cast shadow + tight contact occlusion shadow
    fg_canvas = Image.new("RGBA", (cfg.dimensions.width, cfg.dimensions.height), (0, 0, 0, 0))
    alpha_norm = sc_alpha / 255.0

    # 1. Soft directional cast shadow from lower footprint projected slightly back-left onto concrete
    foot_h = max(8, int(round(th * cs_cfg.cast_shadow_footprint_height_ratio)))
    cast_mask_full = np.zeros((cfg.dimensions.height, cfg.dimensions.width), dtype=np.float32)
    kx, ky = cs_cfg.cast_shadow_offset_xy
    cy_start = max(0, dest_y0 + th - foot_h + ky)
    cy_end = min(cfg.dimensions.height, dest_y0 + th + ky)
    cx_start = max(0, dest_x0 + kx)
    cx_end = min(cfg.dimensions.width, dest_x0 + tw + kx)
    if cy_end > cy_start and cx_end > cx_start:
        src_foot = alpha_norm[th - (cy_end - cy_start) : th, : (cx_end - cx_start)]
        cast_mask_full[cy_start:cy_end, cx_start:cx_end] = src_foot
        cast_blur = ndi.gaussian_filter(cast_mask_full, sigma=cs_cfg.cast_shadow_sigma_yx)
        if float(cast_blur.max()) > 1e-4:
            cast_blur = cast_blur / float(cast_blur.max())
        cast_a_u8 = np.clip(np.round(cast_blur * cs_cfg.cast_shadow_opacity * 255.0), 0, 255).astype(
            np.uint8
        )
        cast_rgb = np.zeros((cfg.dimensions.height, cfg.dimensions.width, 3), dtype=np.uint8)
        cast_rgb[:, :] = cs_cfg.cast_shadow_rgb
        fg_canvas.alpha_composite(Image.fromarray(np.dstack([cast_rgb, cast_a_u8]), mode="RGBA"))

    # 2. Tight ambient-occlusion contact crease right along the per-column bottom edge of stone & cushion
    contact_mask_full = np.zeros((cfg.dimensions.height, cfg.dimensions.width), dtype=np.float32)
    band_px = cs_cfg.contact_shadow_band_px
    ox, oy = cs_cfg.contact_shadow_offset_xy
    for col_idx in range(tw):
        col_ys = np.where(alpha_norm[:, col_idx] > 0.35)[0]
        if col_ys.size == 0:
            continue
        bot_r = int(col_ys.max())
        # Restrict contact crease to real floor-touching base rows
        if bot_r < int(round(th * cs_cfg.contact_shadow_base_height_ratio)):
            continue
        gy = dest_y0 + bot_r + oy
        gx = dest_x0 + col_idx + ox
        if 0 <= gx < cfg.dimensions.width:
            y_lo = max(0, gy - band_px // 2)
            y_hi = min(cfg.dimensions.height, gy + (band_px - band_px // 2))
            contact_mask_full[y_lo:y_hi, gx] = 1.0

    contact_blur = ndi.gaussian_filter(contact_mask_full, sigma=cs_cfg.contact_shadow_sigma_yx)
    if float(contact_blur.max()) > 1e-4:
        contact_blur = contact_blur / float(contact_blur.max())
    contact_a_u8 = np.clip(
        np.round(contact_blur * cs_cfg.contact_shadow_opacity * 255.0), 0, 255
    ).astype(np.uint8)
    contact_rgb = np.zeros((cfg.dimensions.height, cfg.dimensions.width, 3), dtype=np.uint8)
    contact_rgb[:, :] = cs_cfg.contact_shadow_rgb
    fg_canvas.alpha_composite(Image.fromarray(np.dstack([contact_rgb, contact_a_u8]), mode="RGBA"))

    # 3. Composite harmonized Cushion-and-Stone cleanly on top of its floor shadows
    fg_canvas.alpha_composite(relit_patch, dest=(dest_x0, dest_y0))
    return fg_canvas


def build_calibrated_fried_house_depth_map(
    expanded_bg_rgb: Image.Image,
    cutout_layers: Sequence[Image.Image],
    cfg: FriedHouseSceneConfig,
    depth_estimator: DepthAnythingEstimator,
) -> Image.Image:
    """
    Computes a ghost-free, calibrated `2700x1800` 8-bit room depth map (`0=far .. 255=near`) from
    real `DepthAnythingEstimator` monocular depth healed beneath all foreground/midground/railing
    cutouts via `clean_room_depth_under_cutouts` so the stucco wall and concrete steps contain
    zero ghost foreground depth bulges.
    """
    dc = cfg.depth_calibration
    exp_w, exp_h = expanded_bg_rgb.size
    pad_x = (exp_w - cfg.dimensions.width) * 0.5
    pad_y = (exp_h - cfg.dimensions.height) * 0.5

    raw_da2 = depth_estimator.predict_depth(expanded_bg_rgb)
    raw_min = float(raw_da2.min())
    raw_max = float(raw_da2.max())
    raw_u8 = np.rint(
        np.clip((raw_da2 - raw_min) / max(1e-6, raw_max - raw_min), 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    cutout_arrays = [np.asarray(im.convert("RGBA"), dtype=np.uint8) for im in cutout_layers]
    cleaned_raw_u8 = clean_room_depth_under_cutouts(
        room_depth=raw_u8,
        cutout_layers=cutout_arrays,
        cfg=cfg.topology.room_clean,
    )

    smooth_da2 = ndi.gaussian_filter(
        cleaned_raw_u8.astype(np.float32), sigma=dc.da2_gaussian_sigma
    )
    d_min = float(smooth_da2.min())
    d_max = float(smooth_da2.max())
    norm_da2 = (smooth_da2 - d_min) / max(1e-6, d_max - d_min)

    z_field = dc.da2_z_min + norm_da2 * (dc.da2_z_max - dc.da2_z_min)

    if dc.contact_pins:
        xx_inner = (np.arange(exp_w, dtype=np.float32) - pad_x)[None, :]
        yy_inner = (np.arange(exp_h, dtype=np.float32) - pad_y)[:, None]
        sky_guard = np.clip(
            (norm_da2 - dc.sky_norm_guard_band) / max(1e-4, dc.sky_norm_guard_band), 0.0, 1.0
        )
        for pin in dc.contact_pins:
            px, py = pin.inner_xy
            rx, ry = pin.radii_xy
            ell_r = np.hypot((xx_inner - px) / max(1.0, rx), (yy_inner - py) / max(1.0, ry))
            geom_mean_r = float(np.sqrt(max(1.0, rx * ry)))
            excess_px = np.maximum(0.0, ell_r - 1.0) * geom_mean_r
            u_pin = np.clip(excess_px / max(1.0, pin.feather_px), 0.0, 1.0)
            w_pin = (1.0 - u_pin * u_pin * (3.0 - 2.0 * u_pin)) * sky_guard
            z_field = z_field + w_pin * (pin.target_z - z_field)

    if dc.final_smooth_sigma > 0.0:
        z_field = ndi.gaussian_filter(z_field, sigma=dc.final_smooth_sigma)

    norm_u8 = np.clip(
        (z_field - dc.scene_bg_z) / max(1e-4, dc.room_depth_amplitude),
        0.0,
        1.0,
    )
    depth_u8 = np.clip(np.round(norm_u8 * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(depth_u8, mode="L")


def extract_window_shutter_and_open_recess_plates(
    expanded_bg_rgb: Image.Image,
    lama_inner_rgb: Image.Image,
    cfg: FriedHouseSceneConfig,
    sam_extractor: SamBoxMaskExtractor,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """
    Segments the brown two-leaf wooden louvered window shutter inside the white outer window frame
    on the left cream house wall (`2700x1800` room canvas) into:
      1) `open_expanded_bg_rgb` (`2700x1800`) and `open_lama_inner_rgb` (`1500x1000`): 100% intact
         white outer window frame & sill framing a clean, softly shaded dark architectural window
         recess ready for the live 3D portal diorama into `wandering_drawing_1`.
      2) `shutter_full_rgba` (`2700x1800` RGBA) and `shutter_crop_rgba` (tight box-cropped RGBA)
         for the removable wooden shutter occluder.
    """
    wp = cfg.window_portal
    exp_w, exp_h = expanded_bg_rgb.size
    iw, ih = cfg.dimensions.width, cfg.dimensions.height
    pad_x = (exp_w - iw) // 2
    pad_y = (exp_h - ih) // 2

    rx0, ry0, rx1, ry1 = wp.roi_outer_xyxy
    roi_rgb = expanded_bg_rgb.crop((rx0, ry0, rx1, ry1))
    roi_np = np.asarray(roi_rgb, dtype=np.uint8)
    roi_f = roi_np.astype(np.float32)

    sam_u8 = sam_extractor.segment_point_groups(
        roi_rgb,
        wp.shutter_sam_prompts,
        (),
        wp.sam_max_pinhole_fill_px,
        wp.sam_min_component_px,
    )
    roi_luma = 0.299 * roi_f[:, :, 0] + 0.587 * roi_f[:, :, 1] + 0.114 * roi_f[:, :, 2]
    guard_span = max(1e-4, wp.white_frame_luma_guard_end - wp.white_frame_luma_guard_start)
    white_guard = 1.0 - np.clip(
        (roi_luma - wp.white_frame_luma_guard_start) / guard_span, 0.0, 1.0
    )

    k_close = max(3, int(wp.recess_seam_close_px) | 1)
    seam_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    core_binary = ((sam_u8.astype(np.float32) * white_guard) >= 96.0).astype(np.uint8) * 255
    closed_binary = cv2.morphologyEx(core_binary, cv2.MORPH_CLOSE, seam_kernel)
    filled_bool = ndi.binary_fill_holes(closed_binary > 127)

    # Protect outer white window frame perimeter without punching pinholes inside filled shutter interior
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    solid_interior = cv2.erode(filled_bool.astype(np.uint8), erode_k, iterations=1) > 0
    perimeter_guard = np.where(solid_interior, 1.0, white_guard)
    unified_mask_f = filled_bool.astype(np.float32) * perimeter_guard

    if wp.shutter_edge_feather_sigma > 0.0:
        feathered = ndi.gaussian_filter(unified_mask_f * 255.0, sigma=wp.shutter_edge_feather_sigma)
    else:
        feathered = unified_mask_f * 255.0
    shutter_alpha_roi_f = np.clip(feathered * perimeter_guard, 0.0, 255.0)
    shutter_alpha_roi_f[solid_interior] = 255.0
    shutter_alpha_roi_u8 = np.clip(np.round(shutter_alpha_roi_f), 0, 255).astype(np.uint8)

    decontam_roi_u8 = decontaminate_matte_edge_rgb(roi_np, shutter_alpha_roi_u8)
    shutter_full_np = np.zeros((exp_h, exp_w, 4), dtype=np.uint8)
    shutter_full_np[ry0:ry1, rx0:rx1, :3] = decontam_roi_u8
    shutter_full_np[ry0:ry1, rx0:rx1, 3] = shutter_alpha_roi_u8

    ax0, ay0, ax1, ay1 = wp.portal_aperture_outer_xyxy
    min_ap_alpha = int(shutter_full_np[ay0:ay1, ax0:ax1, 3].min())
    if min_ap_alpha < 250:
        raise RuntimeError(
            f"[{cfg.scene_id}] Shutter alpha inside portal aperture {wp.portal_aperture_outer_xyxy} "
            f"has min alpha={min_ap_alpha} (< 250); portal would leak through closed shutter!"
        )

    cx0, cy0, cx1, cy1 = wp.shutter_crop_outer_xyxy
    shutter_full_rgba = Image.fromarray(shutter_full_np, mode="RGBA")
    shutter_crop_rgba = Image.fromarray(shutter_full_np[cy0:cy1, cx0:cx1].copy(), mode="RGBA")

    # Synthesize softly shaded dark architectural window recess inside the white frame on layer0
    recess_weight = (shutter_alpha_roi_u8.astype(np.float32) / 255.0)[:, :, None]
    rh_roi, rw_roi = roi_np.shape[:2]
    yy_r = np.arange(rh_roi, dtype=np.float32)[:, None]
    v_grad = np.clip(
        (yy_r - wp.recess_grad_y0_roi) / max(1.0, wp.recess_grad_span_px), 0.0, 1.0
    )[:, :, None]

    base_dark = np.array(wp.recess_base_dark_rgb, dtype=np.float32)[None, None, :]
    lower_warm = np.array(wp.recess_lower_warm_rgb, dtype=np.float32)[None, None, :]
    lintel_dark = np.array(wp.recess_lintel_dark_rgb, dtype=np.float32)[None, None, :]

    dist_in = ndi.distance_transform_edt(shutter_alpha_roi_u8 >= 96).astype(np.float32)
    lintel_reveal = np.exp(-dist_in / max(1.0, wp.recess_shadow_decay_px))[:, :, None]
    recess_rgb = (1.0 - v_grad) * base_dark + v_grad * lower_warm
    recess_rgb = (
        recess_rgb * (1.0 - wp.recess_lintel_weight * lintel_reveal)
        + lintel_dark * (wp.recess_lintel_weight * lintel_reveal)
    )

    open_roi_u8 = np.clip(
        np.round((1.0 - recess_weight) * roi_f + recess_weight * recess_rgb), 0, 255
    ).astype(np.uint8)

    open_exp_np = np.asarray(expanded_bg_rgb, dtype=np.uint8).copy()
    open_exp_np[ry0:ry1, rx0:rx1] = open_roi_u8
    open_expanded_bg_rgb = Image.fromarray(open_exp_np, mode="RGB")

    # Propagate open window recess into the overlapping left edge of `lama_inner_rgb` (`1500x1000`)
    open_inner_np = np.asarray(lama_inner_rgb, dtype=np.uint8).copy()
    ix0 = max(0, rx0 - pad_x)
    iy0 = max(0, ry0 - pad_y)
    ix1 = min(iw, rx1 - pad_x)
    iy1 = min(ih, ry1 - pad_y)
    if ix1 > ix0 and iy1 > iy0:
        sx0 = (pad_x + ix0) - rx0
        sy0 = (pad_y + iy0) - ry0
        sx1 = sx0 + (ix1 - ix0)
        sy1 = sy0 + (iy1 - iy0)
        w_in = recess_weight[sy0:sy1, sx0:sx1]
        rec_in = recess_rgb[sy0:sy1, sx0:sx1]
        orig_in = open_inner_np[iy0:iy1, ix0:ix1].astype(np.float32)
        open_inner_np[iy0:iy1, ix0:ix1] = np.clip(
            np.round((1.0 - w_in) * orig_in + w_in * rec_in), 0, 255
        ).astype(np.uint8)
    open_lama_inner_rgb = Image.fromarray(open_inner_np, mode="RGB")

    return open_expanded_bg_rgb, open_lama_inner_rgb, shutter_full_rgba, shutter_crop_rgba


def compose_populated_expanded_rgb(
    expanded_bg_rgb: Image.Image,
    cutout_layers: Sequence[Image.Image],
    inner_dims: CanvasDimensionsConfig,
) -> Image.Image:
    """Alpha-composites ordered `1500x1000` or `2700x1800` RGBA cutout layers onto `2700x1800` room plate."""
    exp_w, exp_h = expanded_bg_rgb.size
    pad_x = (exp_w - inner_dims.width) // 2
    pad_y = (exp_h - inner_dims.height) // 2
    comp = expanded_bg_rgb.convert("RGBA")
    for layer in cutout_layers:
        if layer.size == (inner_dims.width, inner_dims.height):
            comp.alpha_composite(layer.convert("RGBA"), dest=(pad_x, pad_y))
        elif layer.size == (exp_w, exp_h):
            comp.alpha_composite(layer.convert("RGBA"), dest=(0, 0))
        else:
            raise ValueError(
                f"Expected cutout size {(inner_dims.width, inner_dims.height)} or {(exp_w, exp_h)}, "
                f"got {layer.size}."
            )
    return comp.convert("RGB")


def solve_and_verify_fried_house_layer_depths(
    expanded_bg_rgb: Image.Image,
    room_depth_img: Image.Image,
    shutter_rgba: Image.Image,
    mid_rgba: Image.Image,
    fg_rgba: Image.Image,
    railing_rgba: Image.Image,
    cfg: FriedHouseSceneConfig,
    depth_estimator: DepthAnythingEstimator,
) -> tuple[
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    dict[str, LayerTopologyVerificationReport],
]:
    """
    Solves harmonic contact-anchored depth maps (`pinMarginUv = 0.0`, outward `EDT` rim
    extrapolation) for `window_shutter` (flush with cream left wall), `mid` (Tami & Er'ela),
    `fg` (`כרית ואבן`), and `green_handrail` (bolted to upper-left stucco wall + step posts),
    and verifies C^0/C^-1 topology invariants on all 4 layers.
    """
    topo = cfg.topology
    room_depth_u8 = np.asarray(room_depth_img.convert("L"), dtype=np.uint8)
    populated_rgb = compose_populated_expanded_rgb(
        expanded_bg_rgb=expanded_bg_rgb,
        cutout_layers=(shutter_rgba, mid_rgba, fg_rgba, railing_rgba),
        inner_dims=cfg.dimensions,
    )
    raw_pop = depth_estimator.predict_depth(populated_rgb)
    pop_min = float(raw_pop.min())
    pop_max = float(raw_pop.max())
    raw_pop_u8 = np.rint(
        np.clip((raw_pop - pop_min) / max(1e-6, pop_max - pop_min), 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    specs: tuple[tuple[str, Image.Image, ContactAnchoredLayerDepthConfig], ...] = (
        ("window_shutter", shutter_rgba, cfg.window_portal.shutter_topology),
        ("mid", mid_rgba, topo.sisters_topology),
        ("fg", fg_rgba, topo.cushion_topology),
        (cfg.handrail.layer_id, railing_rgba, topo.handrail_topology),
    )

    solved_images: dict[str, Image.Image] = {}
    reports: dict[str, LayerTopologyVerificationReport] = {}
    for layer_key, rgba_img, layer_cfg in specs:
        rgba_np = np.asarray(rgba_img.convert("RGBA"), dtype=np.uint8)
        protrusion = compute_room_relative_neural_protrusion(
            raw_neural_depth_expanded=raw_pop_u8,
            cutout_rgba=rgba_np,
            clean_cfg=topo.protrusion_clean,
            upper_percentile=topo.protrusion_upper_percentile,
        )
        solved = solve_contact_anchored_layer_depth(
            room_depth_expanded=room_depth_u8,
            cutout_rgba=rgba_np,
            cfg=layer_cfg,
            neural_protrusion_norm=protrusion,
        )
        report = verify_layer_contact_and_discontinuity(
            room_depth_expanded=room_depth_u8,
            layer_rgba=rgba_np,
            layer_depth=solved.depth_u8,
            contact_mask=solved.contact_mask,
            bg_z=layer_cfg.bg_z,
            room_depth_amplitude=layer_cfg.room_depth_amplitude,
            room_pin_margin_uv=layer_cfg.room_pin_margin_uv,
            layer_z=layer_cfg.layer_z,
            layer_depth_amplitude=layer_cfg.layer_depth_amplitude,
            layer_pin_margin_uv=layer_cfg.layer_pin_margin_uv,
            verify_cfg=topo.layer_verification,
        )
        if not report.all_passed:
            raise RuntimeError(
                f"[{cfg.scene_id}] Layer {layer_key!r} failed 2.5D depth topology verification: "
                f"{report.to_json()}"
            )
        solved_images[layer_key] = Image.fromarray(solved.depth_u8, mode="L")
        reports[layer_key] = report

    return (
        solved_images["window_shutter"],
        solved_images["mid"],
        solved_images["fg"],
        solved_images[cfg.handrail.layer_id],
        reports,
    )


def build_parent_integration_payload(
    cfg: FriedHouseSceneConfig,
    reg_corr: float,
    mid_cov: float,
    fg_cov: float,
    rail_cov: float,
    shutter_cov: float,
    portal_world_z: float,
    occluder_world_z: float,
    topology_reports: dict[str, LayerTopologyVerificationReport],
) -> dict[str, Any]:
    """Builds the complete typed manifest + JS runtime + topology report for parent integration."""
    topo = cfg.topology
    wp = cfg.window_portal
    portal_inner_box = list(wp.portal_inner_box_px(cfg.dimensions, cfg.outpainting))
    shutter_inner_box = list(wp.shutter_inner_box_px(cfg.dimensions, cfg.outpainting))

    portal_shelf_entry = {
        "id": wp.portal_id,
        "targetSceneId": wp.target_scene_id,
        "labelHe": wp.label_he,
        "shortBadgeHe": wp.short_badge_he,
        "innerBoxPx": portal_inner_box,
        "worldZ": round(portal_world_z, 2),
        "recessDepth": wp.recess_depth,
        "restFrameOpacity": wp.rest_frame_opacity,
        "frameRenderOrder": wp.frame_render_order,
        "occluder": {
            "texture": f"assets/scenes/{wp.shutter_crop_output_filename}",
            "fullCanvasTexture": f"assets/scenes/{wp.shutter_output_filename}",
            "depthTexture": f"assets/scenes/{wp.shutter_depth_output_filename}",
            "innerBoxPx": shutter_inner_box,
            "fullInnerCanvas": False,
            "expanded": True,
            "zPos": wp.shutter_topology.layer_z,
            "depthAmplitude": wp.shutter_topology.layer_depth_amplitude,
            "pinMarginUv": wp.shutter_topology.layer_pin_margin_uv,
            "segmentsX": wp.shutter_segments_x,
            "segmentsY": wp.shutter_segments_y,
            "worldZ": round(occluder_world_z, 2),
            "renderOrder": wp.occluder_render_order,
            "baseOpacity": 1.0,
            "alphaHitThreshold": wp.occluder_alpha_hit_threshold,
            "labelHe": wp.shutter_label_he,
        },
    }

    extra_layer_manifest = {
        "id": cfg.handrail.layer_id,
        "labelHe": cfg.handrail.label_he,
        "texture": f"assets/scenes/{cfg.railing_output_filename}",
        "depthTexture": f"assets/scenes/{cfg.railing_depth_output_filename}",
        "zPos": cfg.railing_z,
        "renderOrder": cfg.railing_render_order,
        "expanded": False,
        "baseOpacity": 1.0,
        "excursionBoost": 0.0,
        "depthAmplitude": topo.handrail_topology.layer_depth_amplitude,
        "pinMarginUv": topo.handrail_topology.layer_pin_margin_uv,
        "alphaCutout": 0.0,
        "segmentsX": cfg.railing_segments_x,
        "segmentsY": cfg.railing_segments_y,
        "depthTest": False,
        "depthWrite": False,
    }
    exp_w, exp_h = cfg.outpainting.expanded_dimensions(cfg.dimensions)
    manifest_entry = {
        "id": cfg.scene_id,
        "title": cfg.title_he,
        "shortLabel": cfg.short_label_he,
        "shortLabelHe": cfg.short_label_he,
        "icon": cfg.icon,
        "exhibitionGroup": cfg.exhibition_group,
        "sourceImage": cfg.herz_source_relpath,
        "secondarySourceImage": cfg.cushion_stone_fg.source_relpath,
        "sourceCatalogIds": [cfg.herz_catalog_id, cfg.cushion_stone_fg.catalog_id],
        "outpaintSource": cfg.outpaint_raw_relpath,
        "segmentationBackend": "fried_house_hybrid_sam_isnet",
        "depthBackend": "fried_house_harmonic_contact_anchored_da2",
        "innerWidth": cfg.dimensions.width,
        "innerHeight": cfg.dimensions.height,
        "bgExpandedWidth": exp_w,
        "bgExpandedHeight": exp_h,
        "bgExpansionFactor": cfg.outpainting.expansion_factor,
        "fgExpanded": False,
        "boreSightAtRest": cfg.bore_sight_at_rest,
        "outpaintRegistrationCorr": round(reg_corr, 4),
        "roomDepthAmplitude": cfg.depth_calibration.room_depth_amplitude,
        "sceneBgZ": cfg.scene_bg_z,
        "midZ": cfg.mid_z,
        "midDepthAmplitude": topo.sisters_topology.layer_depth_amplitude,
        "midPinMarginUv": topo.sisters_topology.layer_pin_margin_uv,
        "midSegmentsX": cfg.mid_segments_x,
        "midSegmentsY": cfg.mid_segments_y,
        "fgZ": cfg.fg_z,
        "fgDepthAmplitude": topo.cushion_topology.layer_depth_amplitude,
        "fgPinMarginUv": topo.cushion_topology.layer_pin_margin_uv,
        "fgSegmentsX": cfg.fg_segments_x,
        "fgSegmentsY": cfg.fg_segments_y,
        "bgSegmentsX": cfg.bg_segments_x,
        "bgSegmentsY": cfg.bg_segments_y,
        "initialCameraY": cfg.initial_camera_y,
        "initialCameraZ": cfg.initial_camera_z,
        "minCameraZ": cfg.min_camera_z,
        "maxCameraZ": cfg.max_camera_z,
        "maxYawDeg": cfg.max_yaw_deg,
        "maxPitchDeg": cfg.max_pitch_deg,
        "maxStrafeX": cfg.max_strafe_x,
        "maxStrafeY": cfg.max_strafe_y,
        "dollySpeedMult": cfg.dolly_speed_mult,
        "midRenderOrder": cfg.mid_render_order,
        "fgRenderOrder": cfg.fg_render_order,
        "particleColorHex": cfg.particle_color_hex,
        "particleCount": cfg.particle_count,
        "bgTexture": f"assets/scenes/{cfg.bg_output_filename}",
        "midTexture": f"assets/scenes/{cfg.mid_output_filename}",
        "midDepthTexture": f"assets/scenes/{cfg.mid_depth_output_filename}",
        "fgTexture": f"assets/scenes/{cfg.fg_output_filename}",
        "fgDepthTexture": f"assets/scenes/{cfg.fg_depth_output_filename}",
        "depthTexture": f"assets/scenes/{cfg.depth_output_filename}",
        "extraLayers": [extra_layer_manifest],
        "portalShelves": [portal_shelf_entry],
        "reflectionTexture": None,
        "hasReflection": False,
        "midCoverageRatio": round(mid_cov, 4),
        "fgCoverageRatio": round(fg_cov, 4),
        "railingCoverageRatio": round(rail_cov, 4),
        "shutterCoverageRatio": round(shutter_cov, 4),
    }
    js_scene_config = {
        "id": cfg.scene_id,
        "title": cfg.title_he,
        "shortLabel": cfg.short_label_he,
        "icon": cfg.icon,
        "exhibitionGroup": cfg.exhibition_group,
        "bgTexture": f"assets/scenes/{cfg.bg_output_filename}",
        "depthTexture": f"assets/scenes/{cfg.depth_output_filename}",
        "midTexture": f"assets/scenes/{cfg.mid_output_filename}",
        "midDepthTexture": f"assets/scenes/{cfg.mid_depth_output_filename}",
        "midDepthAmplitude": topo.sisters_topology.layer_depth_amplitude,
        "midPinMarginUv": topo.sisters_topology.layer_pin_margin_uv,
        "midSegmentsX": cfg.mid_segments_x,
        "midSegmentsY": cfg.mid_segments_y,
        "fgTexture": f"assets/scenes/{cfg.fg_output_filename}",
        "fgDepthTexture": f"assets/scenes/{cfg.fg_depth_output_filename}",
        "fgDepthAmplitude": topo.cushion_topology.layer_depth_amplitude,
        "fgPinMarginUv": topo.cushion_topology.layer_pin_margin_uv,
        "fgSegmentsX": cfg.fg_segments_x,
        "fgSegmentsY": cfg.fg_segments_y,
        "boreSightAtRest": cfg.bore_sight_at_rest,
        "bgZ": cfg.scene_bg_z,
        "midZ": cfg.mid_z,
        "fgZ": cfg.fg_z,
        "bgSegmentsX": cfg.bg_segments_x,
        "bgSegmentsY": cfg.bg_segments_y,
        "roomDepthAmplitude": cfg.depth_calibration.room_depth_amplitude,
        "initialCameraY": cfg.initial_camera_y,
        "initialCameraZ": cfg.initial_camera_z,
        "minCameraZ": cfg.min_camera_z,
        "maxCameraZ": cfg.max_camera_z,
        "maxYawDeg": cfg.max_yaw_deg,
        "maxPitchDeg": cfg.max_pitch_deg,
        "maxStrafeX": cfg.max_strafe_x,
        "maxStrafeY": cfg.max_strafe_y,
        "dollySpeedMult": cfg.dolly_speed_mult,
        "midRenderOrder": cfg.mid_render_order,
        "fgRenderOrder": cfg.fg_render_order,
        "extraLayers": [extra_layer_manifest],
        "portalShelves": [portal_shelf_entry],
        "particleColor": f"0x{cfg.particle_color_hex:06X}",
        "particleCount": cfg.particle_count,
        "hasReflection": False,
    }
    return {
        "scene_id": cfg.scene_id,
        "exhibition_group": cfg.exhibition_group,
        "feedback_id": "fb-20261031-223109-d53d13",
        "manifest_entry": manifest_entry,
        "js_scene_config": js_scene_config,
        "topology_verification": {k: v.to_dict() for k, v in topology_reports.items()},
        "ssot_config": cfg.to_dict(),
    }


def build_fried_house_scene(
    project_root: Path,
    output_dir: Path,
    cfg: FriedHouseSceneConfig | None = None,
    export_spec_json: Path | None = None,
) -> dict[str, Any]:
    """End-to-end neural synthesis for Scene 10 (`fried_house_railing`)."""
    scene_cfg = cfg or default_fried_house_scene_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    herz_path = project_root / scene_cfg.herz_source_relpath
    if not herz_path.is_file():
        raise FileNotFoundError(f"Missing Fried house source photograph: {herz_path}")

    outpaint_raw_path = project_root / scene_cfg.outpaint_raw_relpath
    if not outpaint_raw_path.is_file():
        raise FileNotFoundError(f"Missing 3:2 Fried house outpaint plate: {outpaint_raw_path}")

    device = select_torch_device()
    print(f"[{scene_cfg.scene_id}] Initializing neural vision models on device={device}...")
    depth_estimator = DepthAnythingEstimator(scene_cfg.depth_model, device=device)
    sam_extractor = SamBoxMaskExtractor(scene_cfg.segmentation)
    matte_extractor = SalientMatteExtractor(scene_cfg.segmentation)
    lama_inpainter = LamaOcclusionInpainter(scene_cfg.inpainting)

    herz_rgb = Image.open(herz_path).convert("RGB")
    raw_outpaint_rgb = Image.open(outpaint_raw_path).convert("RGB")

    print(
        f"[{scene_cfg.scene_id}] Segmenting Tami & Er'ela (MID Z={scene_cfg.mid_z}) + "
        f"Green Handrail (Z={scene_cfg.railing_z}) + LaMa native background..."
    )
    mid_rgba, railing_rgba, herz_clean_inner_rgb = build_sisters_railing_and_clean_herz(
        herz_rgb=herz_rgb,
        cfg=scene_cfg,
        sam_extractor=sam_extractor,
        matte_extractor=matte_extractor,
        lama_inpainter=lama_inpainter,
    )

    print(
        f"[{scene_cfg.scene_id}] Extracting & harmonizing Cushion-and-Stone "
        f"({scene_cfg.cushion_stone_fg.catalog_id}, FG Z={scene_cfg.fg_z})..."
    )
    fg_rgba = build_cushion_stone_fg_layer(
        project_root=project_root,
        cfg=scene_cfg,
        sam_extractor=sam_extractor,
        matte_extractor=matte_extractor,
    )

    print(f"[{scene_cfg.scene_id}] Composing & registering 2700x1800 Fried house background...")
    base_inner_rgb, closed_lama_inner_rgb, closed_expanded_bg_rgb, reg_corr = (
        build_fried_house_background_plates(
            herz_orig_rgb=herz_rgb,
            herz_clean_inner_rgb=herz_clean_inner_rgb,
            raw_outpaint_rgb=raw_outpaint_rgb,
            cfg=scene_cfg,
        )
    )

    print(
        f"[{scene_cfg.scene_id}] Segmenting removable wooden window shutter & open window recess "
        f"(portal -> {scene_cfg.window_portal.target_scene_id!r})..."
    )
    expanded_bg_rgb, lama_inner_rgb, shutter_full_rgba, shutter_crop_rgba = (
        extract_window_shutter_and_open_recess_plates(
            expanded_bg_rgb=closed_expanded_bg_rgb,
            lama_inner_rgb=closed_lama_inner_rgb,
            cfg=scene_cfg,
            sam_extractor=sam_extractor,
        )
    )

    print(
        f"[{scene_cfg.scene_id}] Synthesizing ghost-free 2700x1800 Depth Anything V2 room depth..."
    )
    depth_img = build_calibrated_fried_house_depth_map(
        expanded_bg_rgb=expanded_bg_rgb,
        cutout_layers=(shutter_full_rgba, mid_rgba, fg_rgba, railing_rgba),
        cfg=scene_cfg,
        depth_estimator=depth_estimator,
    )

    print(
        f"[{scene_cfg.scene_id}] Solving & verifying C^0/C^-1 contact-anchored layer depth maps..."
    )
    shutter_depth_img, mid_depth_img, fg_depth_img, railing_depth_img, topology_reports = (
        solve_and_verify_fried_house_layer_depths(
            expanded_bg_rgb=expanded_bg_rgb,
            room_depth_img=depth_img,
            shutter_rgba=shutter_full_rgba,
            mid_rgba=mid_rgba,
            fg_rgba=fg_rgba,
            railing_rgba=railing_rgba,
            cfg=scene_cfg,
            depth_estimator=depth_estimator,
        )
    )
    for lkey, rep in topology_reports.items():
        print(
            f"  -> [{scene_cfg.scene_id}.{lkey}] max|ΔZ_contact|={rep.max_abs_contact_dz_world:.3f}, "
            f"p95|ΔZ_contact|={rep.p95_abs_contact_dz_world:.3f}, "
            f"max|ΔZ_rim|={rep.max_abs_rim_jump_world:.3f}, passed={rep.all_passed}"
        )

    room_world_z = compute_world_z_map(
        depth_map=np.asarray(depth_img.convert("L"), dtype=np.uint8),
        plane_z=scene_cfg.scene_bg_z,
        depth_amplitude=scene_cfg.depth_calibration.room_depth_amplitude,
        pin_margin_uv=scene_cfg.topology.room_pin_margin_uv,
    )
    ax0, ay0, ax1, ay1 = scene_cfg.window_portal.portal_aperture_outer_xyxy
    portal_world_z = float(np.median(room_world_z[ay0:ay1, ax0:ax1]))
    occluder_world_z = portal_world_z + scene_cfg.window_portal.occluder_z_clearance

    base_inner_path = project_root / scene_cfg.base_inner_outpaint_relpath
    base_inner_path.parent.mkdir(parents=True, exist_ok=True)
    base_inner_rgb.save(
        base_inner_path, format="JPEG", quality=scene_cfg.inpainting.jpeg_quality, optimize=True
    )

    lama_inner_path = output_dir / scene_cfg.lama_inner_output_filename
    bg_path = output_dir / scene_cfg.bg_output_filename
    depth_path = output_dir / scene_cfg.depth_output_filename
    mid_path = output_dir / scene_cfg.mid_output_filename
    mid_depth_path = output_dir / scene_cfg.mid_depth_output_filename
    fg_path = output_dir / scene_cfg.fg_output_filename
    fg_depth_path = output_dir / scene_cfg.fg_depth_output_filename
    rail_path = output_dir / scene_cfg.railing_output_filename
    rail_depth_path = output_dir / scene_cfg.railing_depth_output_filename
    shutter_path = output_dir / scene_cfg.window_portal.shutter_output_filename
    shutter_depth_path = output_dir / scene_cfg.window_portal.shutter_depth_output_filename
    shutter_crop_path = output_dir / scene_cfg.window_portal.shutter_crop_output_filename

    lama_inner_rgb.save(
        lama_inner_path, format="JPEG", quality=scene_cfg.inpainting.jpeg_quality, optimize=True
    )
    expanded_bg_rgb.save(
        bg_path, format="JPEG", quality=scene_cfg.inpainting.jpeg_quality, optimize=True
    )
    depth_img.save(depth_path, format="PNG", optimize=True)
    mid_rgba.save(mid_path, format="PNG", optimize=True)
    mid_depth_img.save(mid_depth_path, format="PNG", optimize=True)
    fg_rgba.save(fg_path, format="PNG", optimize=True)
    fg_depth_img.save(fg_depth_path, format="PNG", optimize=True)
    railing_rgba.save(rail_path, format="PNG", optimize=True)
    railing_depth_img.save(rail_depth_path, format="PNG", optimize=True)
    shutter_full_rgba.save(shutter_path, format="PNG", optimize=True)
    shutter_depth_img.save(shutter_depth_path, format="PNG", optimize=True)
    shutter_crop_rgba.save(shutter_crop_path, format="PNG", optimize=True)

    mid_cov = float((np.asarray(mid_rgba)[:, :, 3] > 16).mean())
    fg_cov = float((np.asarray(fg_rgba)[:, :, 3] > 16).mean())
    rail_cov = float((np.asarray(railing_rgba)[:, :, 3] > 16).mean())
    shutter_cov = float((np.asarray(shutter_full_rgba)[:, :, 3] > 16).mean())

    payload = build_parent_integration_payload(
        cfg=scene_cfg,
        reg_corr=reg_corr,
        mid_cov=mid_cov,
        fg_cov=fg_cov,
        rail_cov=rail_cov,
        shutter_cov=shutter_cov,
        portal_world_z=portal_world_z,
        occluder_world_z=occluder_world_z,
        topology_reports=topology_reports,
    )

    if export_spec_json is not None:
        export_spec_json.parent.mkdir(parents=True, exist_ok=True)
        export_spec_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  -> Exported parent integration spec: {export_spec_json}")

    print(
        f"[{scene_cfg.scene_id}] Completed! "
        f"bg={expanded_bg_rgb.size} (corr={reg_corr:.4f}), "
        f"portal_world_z={portal_world_z:.2f}, "
        f"mid_cov={mid_cov:.4f}, fg_cov={fg_cov:.4f}, rail_cov={rail_cov:.4f}, shutter_cov={shutter_cov:.4f}"
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Scene 10 (`fried_house_railing`) neural 2.5D spatial WebGL assets."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "prototype" / "assets" / "scenes",
        help="Output directory for WebGL scene textures.",
    )
    parser.add_argument(
        "--export-spec-json",
        type=Path,
        default=None,
        help="Optional path to export full manifest + JS scene config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_fried_house_scene(
        project_root=args.project_root.resolve(),
        output_dir=args.output_dir.resolve(),
        export_spec_json=args.export_spec_json.resolve() if args.export_spec_json else None,
    )


if __name__ == "__main__":
    main()
