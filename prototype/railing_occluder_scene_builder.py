#!/usr/bin/env python3
"""
prototype/railing_occluder_scene_builder.py
Offline neural 2.5D spatial asset builder and typed SSOT configuration models for:
  Scene (`fried_yard_installation`): "בחצר בית פריד בהרצליה — ׳מרחב לא מוגן 7׳ לפני המעקה הירוק"
  Fuses the wide Herzliya parents' yard & green entrance railing (`IMG_WORD_0215`, 2007) with
  Tamar Sheaffer's stepped green installation (`IMG_PORT_0062`, "מרחב לא מוגן 7", 2022) placed as a
  depth-displaced clickable foreground occluder in front of the green railing, concealing Tami
  Sheaffer & sister Er'ela Sadan (`IMG_HERZ_0397`, "אראלה סדן") standing by the railing and
  unlocking a seamless portal transition into `fried_house_railing`
  (Beta Feedback #26, `fb-20261030-234247-bcec61`).

Strict No-Fallback Guarantee: executes real neural computer vision models (`Depth Anything V2`,
`Meta SAM`, `IS-Net` via `rembg`, and `LaMa` via `simple-lama-inpainting`) inside `.venv`.
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
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from PIL import Image
except ImportError as exc:
    raise RuntimeError(
        "Missing neural vision / sparse PDE dependencies inside virtual environment. Run with: "
        "`/Users/talsheaffer/code/tami/.venv/bin/python -u prototype/railing_occluder_scene_builder.py`."
    ) from exc

from prototype.fried_house_scene_builder import (
    FriedHouseDepthCalibrationConfig,
    FriedHouseDepthContactPinSpec,
    SistersMidLayerConfig,
    default_fried_house_scene_config,
)
from prototype.scene_config import (
    BoundingBoxPrompt,
    CanvasDimensionsConfig,
    DepthModelConfig,
    InpaintingConfig,
    OutpaintingConfig,
    SamPointGroupPrompt,
    SegmentationConfig,
)
from prototype.vision_primitives import (
    DepthAnythingEstimator,
    LamaOcclusionInpainter,
    SalientMatteExtractor,
    SamBoxMaskExtractor,
    decontaminate_matte_edge_rgb,
    extract_oriented_isnet_roi_alpha_f32,
    merge_sam_and_oriented_mattes_u8,
    select_torch_device,
)


# ============================================================================
# Hierarchical, Serializable Configuration Data-Models (SSOT)
# ============================================================================


@dataclass(frozen=True)
class YardSourceFramingConfig:
    """Affine framing from native 3008x2000 IMG_WORD_0215 onto 1500x1000 inner & 2700x1800 expanded."""

    catalog_id: str
    source_relpath: str
    src_center_xy: tuple[float, float]
    src_inner_wh: tuple[float, float]
    bystander_crop_xyxy: tuple[int, int, int, int]
    bystander_sam_prompts: tuple[SamPointGroupPrompt, ...]
    bystander_dilate_src_px: int
    outpaint_seed_blur_sigma: float
    outpaint_seam_dilate_exp_px: int
    lama_global_downsample_wh: tuple[int, int]
    lama_railing_roi_pad_exp_px: int
    lama_feather_sigma_exp_px: float
    grain_patch_exp_xyxy: tuple[int, int, int, int]
    grain_highpass_sigma_px: float
    grain_strength: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "source_relpath": self.source_relpath,
            "src_center_xy": list(self.src_center_xy),
            "src_inner_wh": list(self.src_inner_wh),
            "bystander_crop_xyxy": list(self.bystander_crop_xyxy),
            "bystander_sam_prompts": [p.to_dict() for p in self.bystander_sam_prompts],
            "bystander_dilate_src_px": self.bystander_dilate_src_px,
            "outpaint_seed_blur_sigma": self.outpaint_seed_blur_sigma,
            "outpaint_seam_dilate_exp_px": self.outpaint_seam_dilate_exp_px,
            "lama_global_downsample_wh": list(self.lama_global_downsample_wh),
            "lama_railing_roi_pad_exp_px": self.lama_railing_roi_pad_exp_px,
            "lama_feather_sigma_exp_px": self.lama_feather_sigma_exp_px,
            "grain_patch_exp_xyxy": list(self.grain_patch_exp_xyxy),
            "grain_highpass_sigma_px": self.grain_highpass_sigma_px,
            "grain_strength": self.grain_strength,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> YardSourceFramingConfig:
        sc = data["src_center_xy"]
        swh = data["src_inner_wh"]
        bc = data["bystander_crop_xyxy"]
        lds = data["lama_global_downsample_wh"]
        gp = data["grain_patch_exp_xyxy"]
        return cls(
            catalog_id=str(data["catalog_id"]),
            source_relpath=str(data["source_relpath"]),
            src_center_xy=(float(sc[0]), float(sc[1])),
            src_inner_wh=(float(swh[0]), float(swh[1])),
            bystander_crop_xyxy=(int(bc[0]), int(bc[1]), int(bc[2]), int(bc[3])),
            bystander_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["bystander_sam_prompts"]
            ),
            bystander_dilate_src_px=int(data["bystander_dilate_src_px"]),
            outpaint_seed_blur_sigma=float(data["outpaint_seed_blur_sigma"]),
            outpaint_seam_dilate_exp_px=int(data["outpaint_seam_dilate_exp_px"]),
            lama_global_downsample_wh=(int(lds[0]), int(lds[1])),
            lama_railing_roi_pad_exp_px=int(data["lama_railing_roi_pad_exp_px"]),
            lama_feather_sigma_exp_px=float(data["lama_feather_sigma_exp_px"]),
            grain_patch_exp_xyxy=(int(gp[0]), int(gp[1]), int(gp[2]), int(gp[3])),
            grain_highpass_sigma_px=float(data["grain_highpass_sigma_px"]),
            grain_strength=float(data["grain_strength"]),
        )


@dataclass(frozen=True)
class SeparatedYardLayerConfig:
    """SAM-segmented foreground/midground depth layer extracted from IMG_WORD_0215."""

    layer_id: str
    label_he: str
    output_filename: str
    depth_output_filename: str | None
    is_expanded: bool
    z_pos: float
    render_order: int
    depth_amplitude: float
    pin_margin_uv: float
    segments_xy: tuple[int, int]
    src_crop_xyxy: tuple[int, int, int, int]
    sam_prompts: tuple[SamPointGroupPrompt, ...]
    sam_exclude_boxes_local: tuple[BoundingBoxPrompt, ...]
    max_pinhole_fill_px: int
    min_major_component_px: int
    bg_inpaint_dilate_exp_px: int
    edge_feather_sigma: float
    depth_smooth_sigma: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "label_he": self.label_he,
            "output_filename": self.output_filename,
            "depth_output_filename": self.depth_output_filename,
            "is_expanded": self.is_expanded,
            "z_pos": self.z_pos,
            "render_order": self.render_order,
            "depth_amplitude": self.depth_amplitude,
            "pin_margin_uv": self.pin_margin_uv,
            "segments_xy": list(self.segments_xy),
            "src_crop_xyxy": list(self.src_crop_xyxy),
            "sam_prompts": [p.to_dict() for p in self.sam_prompts],
            "sam_exclude_boxes_local": [b.to_dict() for b in self.sam_exclude_boxes_local],
            "max_pinhole_fill_px": self.max_pinhole_fill_px,
            "min_major_component_px": self.min_major_component_px,
            "bg_inpaint_dilate_exp_px": self.bg_inpaint_dilate_exp_px,
            "edge_feather_sigma": self.edge_feather_sigma,
            "depth_smooth_sigma": self.depth_smooth_sigma,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeparatedYardLayerConfig:
        sxy = data["segments_xy"]
        sc = data["src_crop_xyxy"]
        return cls(
            layer_id=str(data["layer_id"]),
            label_he=str(data["label_he"]),
            output_filename=str(data["output_filename"]),
            depth_output_filename=(
                str(data["depth_output_filename"])
                if data.get("depth_output_filename") is not None
                else None
            ),
            is_expanded=bool(data["is_expanded"]),
            z_pos=float(data["z_pos"]),
            render_order=int(data["render_order"]),
            depth_amplitude=float(data["depth_amplitude"]),
            pin_margin_uv=float(data["pin_margin_uv"]),
            segments_xy=(int(sxy[0]), int(sxy[1])),
            src_crop_xyxy=(int(sc[0]), int(sc[1]), int(sc[2]), int(sc[3])),
            sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["sam_prompts"]),
            sam_exclude_boxes_local=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data.get("sam_exclude_boxes_local", ())
            ),
            max_pinhole_fill_px=int(data["max_pinhole_fill_px"]),
            min_major_component_px=int(data["min_major_component_px"]),
            bg_inpaint_dilate_exp_px=int(data["bg_inpaint_dilate_exp_px"]),
            edge_feather_sigma=float(data["edge_feather_sigma"]),
            depth_smooth_sigma=float(data["depth_smooth_sigma"]),
        )


@dataclass(frozen=True)
class SistersRevealPlacementConfig:
    """Placement, shaded-yard harmonization, and contact shadow for Tami & Er'ela Sadan."""

    herz_catalog_id: str
    herz_source_relpath: str
    verified_mid_rgba_relpath: str
    sisters_extraction: SistersMidLayerConfig
    alpha_tight_threshold_u8: int
    dest_tight_wh: tuple[int, int]
    dest_bottom_center_inner_xy: tuple[int, int]
    shade_gain_rgb: tuple[float, float, float]
    shade_bias_rgb: tuple[float, float, float]
    luma_gamma: float
    contact_shadow_rgb: tuple[int, int, int]
    contact_shadow_footprint_ratio: float
    contact_shadow_offset_xy: tuple[int, int]
    contact_shadow_sigma_yx: tuple[float, float]
    contact_shadow_opacity: float
    portal_id: str
    portal_target_scene_id: str
    portal_label_he: str
    portal_short_badge_he: str
    portal_show_micro_badge: bool
    portal_hit_pad_inner_px: int
    portal_recess_depth: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "herz_catalog_id": self.herz_catalog_id,
            "herz_source_relpath": self.herz_source_relpath,
            "verified_mid_rgba_relpath": self.verified_mid_rgba_relpath,
            "sisters_extraction": self.sisters_extraction.to_dict(),
            "alpha_tight_threshold_u8": self.alpha_tight_threshold_u8,
            "dest_tight_wh": list(self.dest_tight_wh),
            "dest_bottom_center_inner_xy": list(self.dest_bottom_center_inner_xy),
            "shade_gain_rgb": list(self.shade_gain_rgb),
            "shade_bias_rgb": list(self.shade_bias_rgb),
            "luma_gamma": self.luma_gamma,
            "contact_shadow_rgb": list(self.contact_shadow_rgb),
            "contact_shadow_footprint_ratio": self.contact_shadow_footprint_ratio,
            "contact_shadow_offset_xy": list(self.contact_shadow_offset_xy),
            "contact_shadow_sigma_yx": list(self.contact_shadow_sigma_yx),
            "contact_shadow_opacity": self.contact_shadow_opacity,
            "portal_id": self.portal_id,
            "portal_target_scene_id": self.portal_target_scene_id,
            "portal_label_he": self.portal_label_he,
            "portal_short_badge_he": self.portal_short_badge_he,
            "portal_show_micro_badge": self.portal_show_micro_badge,
            "portal_hit_pad_inner_px": self.portal_hit_pad_inner_px,
            "portal_recess_depth": self.portal_recess_depth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SistersRevealPlacementConfig:
        dwh = data["dest_tight_wh"]
        dbc = data["dest_bottom_center_inner_xy"]
        sg = data["shade_gain_rgb"]
        sb = data["shade_bias_rgb"]
        csr = data["contact_shadow_rgb"]
        cso = data["contact_shadow_offset_xy"]
        css = data["contact_shadow_sigma_yx"]
        return cls(
            herz_catalog_id=str(data["herz_catalog_id"]),
            herz_source_relpath=str(data["herz_source_relpath"]),
            verified_mid_rgba_relpath=str(data["verified_mid_rgba_relpath"]),
            sisters_extraction=SistersMidLayerConfig.from_dict(data["sisters_extraction"]),
            alpha_tight_threshold_u8=int(data["alpha_tight_threshold_u8"]),
            dest_tight_wh=(int(dwh[0]), int(dwh[1])),
            dest_bottom_center_inner_xy=(int(dbc[0]), int(dbc[1])),
            shade_gain_rgb=(float(sg[0]), float(sg[1]), float(sg[2])),
            shade_bias_rgb=(float(sb[0]), float(sb[1]), float(sb[2])),
            luma_gamma=float(data["luma_gamma"]),
            contact_shadow_rgb=(int(csr[0]), int(csr[1]), int(csr[2])),
            contact_shadow_footprint_ratio=float(data["contact_shadow_footprint_ratio"]),
            contact_shadow_offset_xy=(int(cso[0]), int(cso[1])),
            contact_shadow_sigma_yx=(float(css[0]), float(css[1])),
            contact_shadow_opacity=float(data["contact_shadow_opacity"]),
            portal_id=str(data["portal_id"]),
            portal_target_scene_id=str(data["portal_target_scene_id"]),
            portal_label_he=str(data["portal_label_he"]),
            portal_short_badge_he=str(data["portal_short_badge_he"]),
            portal_show_micro_badge=bool(data["portal_show_micro_badge"]),
            portal_hit_pad_inner_px=int(data["portal_hit_pad_inner_px"]),
            portal_recess_depth=float(data["portal_recess_depth"]),
        )


@dataclass(frozen=True)
class InstallationOccluderConfig:
    """Hybrid SAM + IS-Net extraction, 3D relief depth, and yard placement for IMG_PORT_0062."""

    catalog_id: str
    source_relpath: str
    label_he: str
    output_filename: str
    depth_output_filename: str
    world_z: float
    render_order: int
    depth_amplitude: float
    pin_margin_uv: float
    segments_xy: tuple[int, int]
    alpha_hit_threshold_u8: int
    sam_prompts: tuple[SamPointGroupPrompt, ...]
    cutout_window_boxes: tuple[BoundingBoxPrompt, ...]
    cutout_rembg_max_u8: int
    foliage_refine_boxes: tuple[BoundingBoxPrompt, ...]
    support_dilate_px: int
    foliage_sam_floor_weight: float
    solid_rembg_weight: float
    max_pinhole_fill_px: int
    min_component_area_px: int
    edge_feather_sigma: float
    alpha_tight_threshold_u8: int
    dest_tight_wh: tuple[int, int]
    dest_bottom_center_inner_xy: tuple[int, int]
    yard_gain_rgb: tuple[float, float, float]
    yard_bias_rgb: tuple[float, float, float]
    luma_gamma: float
    contact_shadow_rgb: tuple[int, int, int]
    contact_shadow_band_ratio: float
    contact_shadow_offset_xy: tuple[int, int]
    contact_shadow_sigma_yx: tuple[float, float]
    contact_shadow_opacity: float
    depth_smooth_sigma: float
    min_sisters_occlusion_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "source_relpath": self.source_relpath,
            "label_he": self.label_he,
            "output_filename": self.output_filename,
            "depth_output_filename": self.depth_output_filename,
            "world_z": self.world_z,
            "render_order": self.render_order,
            "depth_amplitude": self.depth_amplitude,
            "pin_margin_uv": self.pin_margin_uv,
            "segments_xy": list(self.segments_xy),
            "alpha_hit_threshold_u8": self.alpha_hit_threshold_u8,
            "sam_prompts": [p.to_dict() for p in self.sam_prompts],
            "cutout_window_boxes": [b.to_dict() for b in self.cutout_window_boxes],
            "cutout_rembg_max_u8": self.cutout_rembg_max_u8,
            "foliage_refine_boxes": [b.to_dict() for b in self.foliage_refine_boxes],
            "support_dilate_px": self.support_dilate_px,
            "foliage_sam_floor_weight": self.foliage_sam_floor_weight,
            "solid_rembg_weight": self.solid_rembg_weight,
            "max_pinhole_fill_px": self.max_pinhole_fill_px,
            "min_component_area_px": self.min_component_area_px,
            "edge_feather_sigma": self.edge_feather_sigma,
            "alpha_tight_threshold_u8": self.alpha_tight_threshold_u8,
            "dest_tight_wh": list(self.dest_tight_wh),
            "dest_bottom_center_inner_xy": list(self.dest_bottom_center_inner_xy),
            "yard_gain_rgb": list(self.yard_gain_rgb),
            "yard_bias_rgb": list(self.yard_bias_rgb),
            "luma_gamma": self.luma_gamma,
            "contact_shadow_rgb": list(self.contact_shadow_rgb),
            "contact_shadow_band_ratio": self.contact_shadow_band_ratio,
            "contact_shadow_offset_xy": list(self.contact_shadow_offset_xy),
            "contact_shadow_sigma_yx": list(self.contact_shadow_sigma_yx),
            "contact_shadow_opacity": self.contact_shadow_opacity,
            "depth_smooth_sigma": self.depth_smooth_sigma,
            "min_sisters_occlusion_coverage": self.min_sisters_occlusion_coverage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstallationOccluderConfig:
        sxy = data["segments_xy"]
        dwh = data["dest_tight_wh"]
        dbc = data["dest_bottom_center_inner_xy"]
        yg = data["yard_gain_rgb"]
        yb = data["yard_bias_rgb"]
        csr = data["contact_shadow_rgb"]
        cso = data["contact_shadow_offset_xy"]
        css = data["contact_shadow_sigma_yx"]
        return cls(
            catalog_id=str(data["catalog_id"]),
            source_relpath=str(data["source_relpath"]),
            label_he=str(data["label_he"]),
            output_filename=str(data["output_filename"]),
            depth_output_filename=str(data["depth_output_filename"]),
            world_z=float(data["world_z"]),
            render_order=int(data["render_order"]),
            depth_amplitude=float(data["depth_amplitude"]),
            pin_margin_uv=float(data["pin_margin_uv"]),
            segments_xy=(int(sxy[0]), int(sxy[1])),
            alpha_hit_threshold_u8=int(data["alpha_hit_threshold_u8"]),
            sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["sam_prompts"]),
            cutout_window_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["cutout_window_boxes"]
            ),
            cutout_rembg_max_u8=int(data["cutout_rembg_max_u8"]),
            foliage_refine_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["foliage_refine_boxes"]
            ),
            support_dilate_px=int(data["support_dilate_px"]),
            foliage_sam_floor_weight=float(data["foliage_sam_floor_weight"]),
            solid_rembg_weight=float(data["solid_rembg_weight"]),
            max_pinhole_fill_px=int(data["max_pinhole_fill_px"]),
            min_component_area_px=int(data["min_component_area_px"]),
            edge_feather_sigma=float(data["edge_feather_sigma"]),
            alpha_tight_threshold_u8=int(data["alpha_tight_threshold_u8"]),
            dest_tight_wh=(int(dwh[0]), int(dwh[1])),
            dest_bottom_center_inner_xy=(int(dbc[0]), int(dbc[1])),
            yard_gain_rgb=(float(yg[0]), float(yg[1]), float(yg[2])),
            yard_bias_rgb=(float(yb[0]), float(yb[1]), float(yb[2])),
            luma_gamma=float(data["luma_gamma"]),
            contact_shadow_rgb=(int(csr[0]), int(csr[1]), int(csr[2])),
            contact_shadow_band_ratio=float(data["contact_shadow_band_ratio"]),
            contact_shadow_offset_xy=(int(cso[0]), int(cso[1])),
            contact_shadow_sigma_yx=(float(css[0]), float(css[1])),
            contact_shadow_opacity=float(data["contact_shadow_opacity"]),
            depth_smooth_sigma=float(data["depth_smooth_sigma"]),
            min_sisters_occlusion_coverage=float(data["min_sisters_occlusion_coverage"]),
        )


@dataclass(frozen=True)
class HarmonicContactAnchorConfig:
    """2.5D Depth Topology parameters: C0 harmonic contact continuity + C^-1 outward EDT discontinuity."""

    room_pin_margin_uv: float
    active_alpha_threshold_u8: int
    solid_contact_alpha_u8: int
    graph_connect_dilate_px: int
    neutral_backdrop_gray_u8: int
    da2_percentile_low: float
    da2_percentile_high: float
    railing_post_min_vertical_span_px: int
    railing_post_base_contact_band_px: int
    railing_left_bush_contact_band_px: int
    railing_min_detected_posts: int
    installation_relief_amplitude_z: float
    installation_rear_support_min_y_px: int
    installation_rear_support_max_x_px: int
    installation_front_step_min_y_px: int
    installation_contact_band_px: int
    hedge_solve_downsample: int
    hedge_relief_amplitude_z: float
    hedge_contact_band_exp_px: int
    hedge_contact_max_exp_x_px: int
    sisters_shoe_contact_band_px: int
    fg_ground_contact_band_exp_px: int
    max_contact_delta_z: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_pin_margin_uv": self.room_pin_margin_uv,
            "active_alpha_threshold_u8": self.active_alpha_threshold_u8,
            "solid_contact_alpha_u8": self.solid_contact_alpha_u8,
            "graph_connect_dilate_px": self.graph_connect_dilate_px,
            "neutral_backdrop_gray_u8": self.neutral_backdrop_gray_u8,
            "da2_percentile_low": self.da2_percentile_low,
            "da2_percentile_high": self.da2_percentile_high,
            "railing_post_min_vertical_span_px": self.railing_post_min_vertical_span_px,
            "railing_post_base_contact_band_px": self.railing_post_base_contact_band_px,
            "railing_left_bush_contact_band_px": self.railing_left_bush_contact_band_px,
            "railing_min_detected_posts": self.railing_min_detected_posts,
            "installation_relief_amplitude_z": self.installation_relief_amplitude_z,
            "installation_rear_support_min_y_px": self.installation_rear_support_min_y_px,
            "installation_rear_support_max_x_px": self.installation_rear_support_max_x_px,
            "installation_front_step_min_y_px": self.installation_front_step_min_y_px,
            "installation_contact_band_px": self.installation_contact_band_px,
            "hedge_solve_downsample": self.hedge_solve_downsample,
            "hedge_relief_amplitude_z": self.hedge_relief_amplitude_z,
            "hedge_contact_band_exp_px": self.hedge_contact_band_exp_px,
            "hedge_contact_max_exp_x_px": self.hedge_contact_max_exp_x_px,
            "sisters_shoe_contact_band_px": self.sisters_shoe_contact_band_px,
            "fg_ground_contact_band_exp_px": self.fg_ground_contact_band_exp_px,
            "max_contact_delta_z": self.max_contact_delta_z,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HarmonicContactAnchorConfig:
        return cls(
            room_pin_margin_uv=float(data["room_pin_margin_uv"]),
            active_alpha_threshold_u8=int(data["active_alpha_threshold_u8"]),
            solid_contact_alpha_u8=int(data["solid_contact_alpha_u8"]),
            graph_connect_dilate_px=int(data["graph_connect_dilate_px"]),
            neutral_backdrop_gray_u8=int(data["neutral_backdrop_gray_u8"]),
            da2_percentile_low=float(data["da2_percentile_low"]),
            da2_percentile_high=float(data["da2_percentile_high"]),
            railing_post_min_vertical_span_px=int(data["railing_post_min_vertical_span_px"]),
            railing_post_base_contact_band_px=int(data["railing_post_base_contact_band_px"]),
            railing_left_bush_contact_band_px=int(data["railing_left_bush_contact_band_px"]),
            railing_min_detected_posts=int(data["railing_min_detected_posts"]),
            installation_relief_amplitude_z=float(data["installation_relief_amplitude_z"]),
            installation_rear_support_min_y_px=int(data["installation_rear_support_min_y_px"]),
            installation_rear_support_max_x_px=int(data["installation_rear_support_max_x_px"]),
            installation_front_step_min_y_px=int(data["installation_front_step_min_y_px"]),
            installation_contact_band_px=int(data["installation_contact_band_px"]),
            hedge_solve_downsample=int(data["hedge_solve_downsample"]),
            hedge_relief_amplitude_z=float(data["hedge_relief_amplitude_z"]),
            hedge_contact_band_exp_px=int(data["hedge_contact_band_exp_px"]),
            hedge_contact_max_exp_x_px=int(data["hedge_contact_max_exp_x_px"]),
            sisters_shoe_contact_band_px=int(data["sisters_shoe_contact_band_px"]),
            fg_ground_contact_band_exp_px=int(data["fg_ground_contact_band_exp_px"]),
            max_contact_delta_z=float(data["max_contact_delta_z"]),
        )


@dataclass(frozen=True)
class RailingOccluderSceneConfig:
    """Canonical SSOT configuration for `fried_yard_installation` (Feedback #26)."""

    scene_id: str
    exhibition_group: str
    title_he: str
    short_label_he: str
    icon: str
    bg_output_filename: str
    depth_output_filename: str
    mid_output_filename: str
    fg_output_filename: str
    dimensions: CanvasDimensionsConfig
    outpainting: OutpaintingConfig
    depth_model: DepthModelConfig
    segmentation: SegmentationConfig
    inpainting: InpaintingConfig
    bore_sight_at_rest: bool
    fg_is_expanded: bool
    scene_bg_z: float
    mid_z: float
    fg_z: float
    mid_render_order: int
    fg_render_order: int
    bg_segments_x: int
    bg_segments_y: int
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
    portal_hit_box_inner_px: tuple[float, float, float, float]
    occluder_hit_box_inner_px: tuple[float, float, float, float]
    occluder_coverage_ratio: float
    yard_framing: YardSourceFramingConfig
    hedge_layer: SeparatedYardLayerConfig
    railing_layer: SeparatedYardLayerConfig
    brush_pile_fg: SeparatedYardLayerConfig
    sisters_reveal: SistersRevealPlacementConfig
    installation_occluder: InstallationOccluderConfig
    depth_calibration: FriedHouseDepthCalibrationConfig
    contact_topology: HarmonicContactAnchorConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "exhibition_group": self.exhibition_group,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "bg_output_filename": self.bg_output_filename,
            "depth_output_filename": self.depth_output_filename,
            "mid_output_filename": self.mid_output_filename,
            "fg_output_filename": self.fg_output_filename,
            "dimensions": self.dimensions.to_dict(),
            "outpainting": self.outpainting.to_dict(),
            "depth_model": self.depth_model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "inpainting": self.inpainting.to_dict(),
            "bore_sight_at_rest": self.bore_sight_at_rest,
            "fg_is_expanded": self.fg_is_expanded,
            "scene_bg_z": self.scene_bg_z,
            "mid_z": self.mid_z,
            "fg_z": self.fg_z,
            "mid_render_order": self.mid_render_order,
            "fg_render_order": self.fg_render_order,
            "bg_segments_x": self.bg_segments_x,
            "bg_segments_y": self.bg_segments_y,
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
            "portal_hit_box_inner_px": list(self.portal_hit_box_inner_px),
            "occluder_hit_box_inner_px": list(self.occluder_hit_box_inner_px),
            "occluder_coverage_ratio": self.occluder_coverage_ratio,
            "yard_framing": self.yard_framing.to_dict(),
            "hedge_layer": self.hedge_layer.to_dict(),
            "railing_layer": self.railing_layer.to_dict(),
            "brush_pile_fg": self.brush_pile_fg.to_dict(),
            "sisters_reveal": self.sisters_reveal.to_dict(),
            "installation_occluder": self.installation_occluder.to_dict(),
            "depth_calibration": self.depth_calibration.to_dict(),
            "contact_topology": self.contact_topology.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RailingOccluderSceneConfig:
        phb = data["portal_hit_box_inner_px"]
        ohb = data["occluder_hit_box_inner_px"]
        return cls(
            scene_id=str(data["scene_id"]),
            exhibition_group=str(data["exhibition_group"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            bg_output_filename=str(data["bg_output_filename"]),
            depth_output_filename=str(data["depth_output_filename"]),
            mid_output_filename=str(data["mid_output_filename"]),
            fg_output_filename=str(data["fg_output_filename"]),
            dimensions=CanvasDimensionsConfig.from_dict(data["dimensions"]),
            outpainting=OutpaintingConfig.from_dict(data["outpainting"]),
            depth_model=DepthModelConfig.from_dict(data["depth_model"]),
            segmentation=SegmentationConfig.from_dict(data["segmentation"]),
            inpainting=InpaintingConfig.from_dict(data["inpainting"]),
            bore_sight_at_rest=bool(data["bore_sight_at_rest"]),
            fg_is_expanded=bool(data["fg_is_expanded"]),
            scene_bg_z=float(data["scene_bg_z"]),
            mid_z=float(data["mid_z"]),
            fg_z=float(data["fg_z"]),
            mid_render_order=int(data["mid_render_order"]),
            fg_render_order=int(data["fg_render_order"]),
            bg_segments_x=int(data["bg_segments_x"]),
            bg_segments_y=int(data["bg_segments_y"]),
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
            portal_hit_box_inner_px=(
                float(phb[0]),
                float(phb[1]),
                float(phb[2]),
                float(phb[3]),
            ),
            occluder_hit_box_inner_px=(
                float(ohb[0]),
                float(ohb[1]),
                float(ohb[2]),
                float(ohb[3]),
            ),
            occluder_coverage_ratio=float(data["occluder_coverage_ratio"]),
            yard_framing=YardSourceFramingConfig.from_dict(data["yard_framing"]),
            hedge_layer=SeparatedYardLayerConfig.from_dict(data["hedge_layer"]),
            railing_layer=SeparatedYardLayerConfig.from_dict(data["railing_layer"]),
            brush_pile_fg=SeparatedYardLayerConfig.from_dict(data["brush_pile_fg"]),
            sisters_reveal=SistersRevealPlacementConfig.from_dict(data["sisters_reveal"]),
            installation_occluder=InstallationOccluderConfig.from_dict(
                data["installation_occluder"]
            ),
            depth_calibration=FriedHouseDepthCalibrationConfig.from_dict(
                data["depth_calibration"]
            ),
            contact_topology=HarmonicContactAnchorConfig.from_dict(data["contact_topology"]),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, raw_json: str) -> RailingOccluderSceneConfig:
        return cls.from_dict(json.loads(raw_json))


def default_railing_occluder_scene_config() -> RailingOccluderSceneConfig:
    """Returns canonical SSOT configuration for `fried_yard_installation`."""
    fried_ref = default_fried_house_scene_config()
    return RailingOccluderSceneConfig(
        scene_id="fried_yard_installation",
        exhibition_group="family_places",
        title_he="בחצר בית פריד בהרצליה — ׳מרחב לא מוגן 7׳ לפני המעקה הירוק",
        short_label_he="חצר בית פריד: מרחב לא מוגן",
        icon="🌿",
        bg_output_filename="scene_railing_occluder_bg_inpainted.jpg",
        depth_output_filename="scene_railing_occluder_depth.png",
        mid_output_filename="scene_railing_occluder_mid.png",
        fg_output_filename="scene_railing_occluder_fg.png",
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
            background_gaussian_sigma=11.0,
            depth_discontinuity_percentile_fg=62.0,
            depth_discontinuity_percentile_mid=30.0,
        ),
        segmentation=SegmentationConfig(
            rembg_model_name="isnet-general-use",
            sam_model_id="facebook/sam-vit-base",
            alpha_foreground_threshold=92,
            alpha_full_opaque_threshold=195,
            min_component_area_px=220,
            sam_morph_close_px=5,
            edge_feather_sigma=0.65,
        ),
        inpainting=InpaintingConfig(
            dilation_radius_px=27,
            mask_blur_sigma=2.0,
            jpeg_quality=92,
        ),
        bore_sight_at_rest=True,
        fg_is_expanded=True,
        scene_bg_z=-360.0,
        mid_z=-303.44,
        fg_z=-229.07,
        mid_render_order=3,
        fg_render_order=8,
        bg_segments_x=224,
        bg_segments_y=150,
        initial_camera_y=0.0,
        initial_camera_z=755.0,
        min_camera_z=205.0,
        max_camera_z=985.0,
        max_yaw_deg=34.0,
        max_pitch_deg=24.0,
        max_strafe_x=255.0,
        max_strafe_y=160.0,
        dolly_speed_mult=1.35,
        particle_color_hex=0xE6DFC6,
        particle_count=70,
        portal_hit_box_inner_px=(515.0, 468.0, 705.0, 788.0),
        occluder_hit_box_inner_px=(436.0, 94.0, 792.0, 930.0),
        occluder_coverage_ratio=0.9809,
        yard_framing=YardSourceFramingConfig(
            catalog_id="IMG_WORD_0215",
            source_relpath="media/wordpress_blog/wp_wide_pict2739_4637ae7c.jpg",
            src_center_xy=(1100.0, 1100.0),
            src_inner_wh=(1830.0, 1220.0),
            bystander_crop_xyxy=(0, 900, 180, 1300),
            bystander_sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=0, y0=945, x1=95, y1=1225),
                    points_xy=((40.0, 1000.0), (40.0, 1100.0), (45.0, 1180.0)),
                    point_labels=(1, 1, 1),
                ),
            ),
            bystander_dilate_src_px=26,
            outpaint_seed_blur_sigma=3.5,
            outpaint_seam_dilate_exp_px=28,
            lama_global_downsample_wh=(1080, 720),
            lama_railing_roi_pad_exp_px=95,
            lama_feather_sigma_exp_px=5.5,
            grain_patch_exp_xyxy=(980, 1360, 1260, 1580),
            grain_highpass_sigma_px=2.2,
            grain_strength=0.45,
        ),
        hedge_layer=SeparatedYardLayerConfig(
            layer_id="yard_hedge_bank",
            label_he="משוכת הגדר החיה והסבך בחצר בית פריד",
            output_filename="scene_railing_occluder_hedge.png",
            depth_output_filename="scene_railing_occluder_hedge_depth.png",
            is_expanded=True,
            z_pos=-356.0,
            render_order=2,
            depth_amplitude=200.0,
            pin_margin_uv=0.0,
            segments_xy=(192, 128),
            src_crop_xyxy=(1120, 940, 3008, 2000),
            sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=1180, y0=1020, x1=3008, y1=2000),
                    points_xy=(
                        (1500.0, 1350.0),
                        (1900.0, 1400.0),
                        (2300.0, 1500.0),
                        (2700.0, 1500.0),
                        (2000.0, 1700.0),
                        (1600.0, 1550.0),
                        (2450.0, 1180.0),
                        (2100.0, 1140.0),
                        (1300.0, 1800.0),
                        (1240.0, 1120.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
                ),
            ),
            sam_exclude_boxes_local=(),
            max_pinhole_fill_px=4500,
            min_major_component_px=8000,
            bg_inpaint_dilate_exp_px=52,
            edge_feather_sigma=1.15,
            depth_smooth_sigma=6.5,
        ),
        railing_layer=SeparatedYardLayerConfig(
            layer_id="green_railing",
            label_he="מעקה הצינור הירוק בשביל הכניסה לבית פריד",
            output_filename="scene_railing_occluder_railing.png",
            depth_output_filename="scene_railing_occluder_railing_depth.png",
            is_expanded=False,
            z_pos=-312.0,
            render_order=4,
            depth_amplitude=48.0,
            pin_margin_uv=0.0,
            segments_xy=(256, 170),
            src_crop_xyxy=(600, 1080, 1000, 1600),
            sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=875, y0=1215, x1=915, y1=1560),
                    points_xy=(
                        (895.0, 1300.0),
                        (895.0, 1420.0),
                        (895.0, 1520.0),
                        (878.0, 1360.0),
                        (912.0, 1360.0),
                    ),
                    point_labels=(1, 1, 1, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=782, y0=1185, x1=822, y1=1510),
                    points_xy=(
                        (801.0, 1260.0),
                        (801.0, 1380.0),
                        (801.0, 1480.0),
                        (785.0, 1340.0),
                        (818.0, 1340.0),
                    ),
                    point_labels=(1, 1, 1, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=712, y0=1160, x1=750, y1=1460),
                    points_xy=(
                        (731.0, 1230.0),
                        (731.0, 1330.0),
                        (731.0, 1420.0),
                        (715.0, 1310.0),
                        (747.0, 1310.0),
                    ),
                    point_labels=(1, 1, 1, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=662, y0=1142, x1=694, y1=1400),
                    points_xy=(
                        (676.0, 1200.0),
                        (676.0, 1290.0),
                        (676.0, 1370.0),
                        (664.0, 1260.0),
                        (691.0, 1260.0),
                    ),
                    point_labels=(1, 1, 1, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=692, y0=1122, x1=952, y1=1235),
                    points_xy=(
                        (715.0, 1134.0),
                        (760.0, 1150.0),
                        (840.0, 1181.0),
                        (920.0, 1212.0),
                        (715.0, 1118.0),
                        (760.0, 1130.0),
                        (840.0, 1160.0),
                        (760.0, 1175.0),
                        (840.0, 1205.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0, 0, 0, 0),
                ),
            ),
            sam_exclude_boxes_local=(BoundingBoxPrompt(x0=0, y0=0, x1=64, y1=520),),
            max_pinhole_fill_px=80,
            min_major_component_px=150,
            bg_inpaint_dilate_exp_px=26,
            edge_feather_sigma=0.65,
            depth_smooth_sigma=0.0,
        ),
        brush_pile_fg=SeparatedYardLayerConfig(
            layer_id="foreground_brush_pile",
            label_he="ערימת הגזם והענפים בקדמת החצר",
            output_filename="scene_railing_occluder_fg.png",
            depth_output_filename=None,
            is_expanded=True,
            z_pos=-229.07,
            render_order=8,
            depth_amplitude=0.0,
            pin_margin_uv=0.0,
            segments_xy=(1, 1),
            src_crop_xyxy=(0, 1300, 860, 1850),
            sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=0, y0=1380, x1=755, y1=1800),
                    points_xy=(
                        (300.0, 1550.0),
                        (500.0, 1560.0),
                        (150.0, 1600.0),
                        (600.0, 1620.0),
                        (400.0, 1785.0),
                        (710.0, 1460.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0),
                ),
            ),
            sam_exclude_boxes_local=(),
            max_pinhole_fill_px=1800,
            min_major_component_px=3500,
            bg_inpaint_dilate_exp_px=27,
            edge_feather_sigma=1.25,
            depth_smooth_sigma=0.0,
        ),
        sisters_reveal=SistersRevealPlacementConfig(
            herz_catalog_id=fried_ref.herz_catalog_id,
            herz_source_relpath=fried_ref.herz_source_relpath,
            verified_mid_rgba_relpath="prototype/assets/scenes/scene10_fried_house_mid.png",
            sisters_extraction=fried_ref.sisters_mid,
            alpha_tight_threshold_u8=32,
            dest_tight_wh=(162, 292),
            dest_bottom_center_inner_xy=(610, 774),
            shade_gain_rgb=(0.88, 0.90, 0.87),
            shade_bias_rgb=(3.0, 4.0, 2.0),
            luma_gamma=1.06,
            contact_shadow_rgb=(28, 30, 24),
            contact_shadow_footprint_ratio=0.14,
            contact_shadow_offset_xy=(4, 5),
            contact_shadow_sigma_yx=(5.5, 13.0),
            contact_shadow_opacity=0.42,
            portal_id="yard_railing_portal",
            portal_target_scene_id="fried_house_railing",
            portal_label_he="🏡 תמי שפר ואראלה סדן על המעקה הירוק בבית פריד",
            portal_short_badge_he="🏡 תמי ואראלה במעקה",
            portal_show_micro_badge=False,
            portal_hit_pad_inner_px=14,
            portal_recess_depth=0.0,
        ),
        installation_occluder=InstallationOccluderConfig(
            catalog_id="IMG_PORT_0062",
            source_relpath="media/portfolio_site/מרחב-לא-מוגן7-scaled.jpg_ac17985c.jpg",
            label_he="המיצב ׳מרחב לא מוגן 7׳",
            output_filename="scene_railing_occluder_installation.png",
            depth_output_filename="scene_railing_occluder_installation_depth.png",
            world_z=-302.0,
            render_order=6,
            depth_amplitude=64.0,
            pin_margin_uv=0.0,
            segments_xy=(192, 128),
            alpha_hit_threshold_u8=24,
            sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=745, y0=95, x1=835, y1=340),
                    points_xy=(
                        (782.0, 128.0),
                        (784.0, 195.0),
                        (785.0, 275.0),
                        (755.0, 180.0),
                        (815.0, 180.0),
                    ),
                    point_labels=(1, 1, 1, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=630, y0=235, x1=1025, y1=795),
                    points_xy=(
                        (740.0, 290.0),
                        (840.0, 320.0),
                        (695.0, 500.0),
                        (700.0, 680.0),
                        (880.0, 480.0),
                        (835.0, 620.0),
                        (905.0, 740.0),
                        (660.0, 320.0),
                        (990.0, 340.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=490, y0=688, x1=665, y1=722),
                    points_xy=(
                        (525.0, 704.0),
                        (580.0, 704.0),
                        (635.0, 704.0),
                        (560.0, 690.0),
                        (560.0, 718.0),
                    ),
                    point_labels=(1, 1, 1, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=960, y0=650, x1=1270, y1=785),
                    points_xy=(
                        (1150.0, 675.0),
                        (1040.0, 705.0),
                        (1035.0, 745.0),
                        (1210.0, 703.0),
                        (1120.0, 660.0),
                        (1150.0, 735.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=475, y0=720, x1=1135, y1=1350),
                    points_xy=(
                        (580.0, 760.0),
                        (620.0, 920.0),
                        (680.0, 1060.0),
                        (800.0, 900.0),
                        (805.0, 1040.0),
                        (960.0, 960.0),
                        (1040.0, 975.0),
                        (600.0, 1220.0),
                        (760.0, 1220.0),
                        (890.0, 1200.0),
                        (674.0, 885.0),
                        (1022.0, 1060.0),
                        (520.0, 1080.0),
                        (520.0, 1310.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=920, y0=1030, x1=1220, y1=1695),
                    points_xy=(
                        (1080.0, 1140.0),
                        (1045.0, 1295.0),
                        (1118.0, 1290.0),
                        (1085.0, 1410.0),
                        (1152.0, 1490.0),
                        (1090.0, 1545.0),
                        (1122.0, 1630.0),
                        (1185.0, 1300.0),
                        (1055.0, 1630.0),
                        (1105.0, 1645.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=600, y0=1310, x1=1288, y1=2245),
                    points_xy=(
                        (830.0, 1380.0),
                        (850.0, 1460.0),
                        (945.0, 1440.0),
                        (690.0, 1365.0),
                        (750.0, 1405.0),
                        (885.0, 1550.0),
                        (900.0, 1635.0),
                        (690.0, 1660.0),
                        (672.0, 1605.0),
                        (720.0, 1840.0),
                        (660.0, 1940.0),
                        (830.0, 1880.0),
                        (935.0, 1765.0),
                        (1020.0, 1905.0),
                        (1105.0, 2060.0),
                        (1165.0, 2135.0),
                        (1022.0, 2165.0),
                        (585.0, 1520.0),
                        (760.0, 2030.0),
                        (920.0, 2060.0),
                        (1220.0, 1880.0),
                    ),
                    point_labels=(
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                    ),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=375, y0=1275, x1=575, y1=1840),
                    points_xy=(
                        (465.0, 1380.0),
                        (455.0, 1515.0),
                        (492.0, 1625.0),
                        (495.0, 1745.0),
                        (528.0, 1805.0),
                        (415.0, 1680.0),
                        (548.0, 1560.0),
                        (555.0, 1680.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 0, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=468, y0=1825, x1=638, y1=1998),
                    points_xy=(
                        (515.0, 1905.0),
                        (555.0, 1945.0),
                        (595.0, 1965.0),
                        (485.0, 1982.0),
                        (625.0, 1992.0),
                    ),
                    point_labels=(1, 1, 1, 0, 0),
                ),
            ),
            cutout_window_boxes=(
                BoundingBoxPrompt(x0=650, y0=790, x1=698, y1=995),
                BoundingBoxPrompt(x0=1002, y0=1010, x1=1042, y1=1128),
            ),
            cutout_rembg_max_u8=64,
            foliage_refine_boxes=(
                BoundingBoxPrompt(x0=735, y0=90, x1=845, y1=360),
                BoundingBoxPrompt(x0=760, y0=400, x1=1035, y1=640),
                BoundingBoxPrompt(x0=955, y0=1025, x1=1225, y1=1265),
                BoundingBoxPrompt(x0=370, y0=1265, x1=565, y1=1505),
                BoundingBoxPrompt(x0=1020, y0=1475, x1=1205, y1=1695),
            ),
            support_dilate_px=29,
            foliage_sam_floor_weight=0.35,
            solid_rembg_weight=0.85,
            max_pinhole_fill_px=1200,
            min_component_area_px=120,
            edge_feather_sigma=0.55,
            alpha_tight_threshold_u8=24,
            dest_tight_wh=(356, 836),
            dest_bottom_center_inner_xy=(614, 930),
            yard_gain_rgb=(0.95, 0.97, 0.93),
            yard_bias_rgb=(1.0, 2.5, 0.5),
            luma_gamma=1.03,
            contact_shadow_rgb=(26, 28, 22),
            contact_shadow_band_ratio=0.18,
            contact_shadow_offset_xy=(4, 7),
            contact_shadow_sigma_yx=(8.5, 20.0),
            contact_shadow_opacity=0.45,
            depth_smooth_sigma=3.2,
            min_sisters_occlusion_coverage=0.72,
        ),
        depth_calibration=FriedHouseDepthCalibrationConfig(
            scene_bg_z=-360.0,
            room_depth_amplitude=215.0,
            da2_gaussian_sigma=13.0,
            da2_z_min=-360.0,
            da2_z_max=-145.0,
            sky_norm_guard_band=0.05,
            contact_pins=(
                FriedHouseDepthContactPinSpec(
                    label="sisters_walkway_footprint",
                    inner_xy=(610.0, 768.0),
                    radii_xy=(46.0, 14.0),
                    feather_px=30.0,
                    target_z=-303.7,
                ),
            ),
            final_smooth_sigma=8.0,
        ),
        contact_topology=HarmonicContactAnchorConfig(
            room_pin_margin_uv=0.04,
            active_alpha_threshold_u8=16,
            solid_contact_alpha_u8=48,
            graph_connect_dilate_px=1,
            neutral_backdrop_gray_u8=128,
            da2_percentile_low=4.0,
            da2_percentile_high=98.0,
            railing_post_min_vertical_span_px=90,
            railing_post_base_contact_band_px=10,
            railing_left_bush_contact_band_px=8,
            railing_min_detected_posts=4,
            installation_relief_amplitude_z=16.0,
            installation_rear_support_min_y_px=808,
            installation_rear_support_max_x_px=580,
            installation_front_step_min_y_px=890,
            installation_contact_band_px=12,
            hedge_solve_downsample=2,
            hedge_relief_amplitude_z=24.0,
            hedge_contact_band_exp_px=18,
            hedge_contact_max_exp_x_px=2540,
            sisters_shoe_contact_band_px=10,
            fg_ground_contact_band_exp_px=20,
            max_contact_delta_z=2.5,
        ),
    )


# ============================================================================
# Coordinate Transforms & Neural Synthesis Helpers
# ============================================================================


def _compute_yard_expanded_mapping(
    cfg: RailingOccluderSceneConfig,
) -> tuple[int, int, int, int, float]:
    """Returns (exp_w, exp_h, pad_x, pad_y, scale_exp_per_src)."""
    inner_w, inner_h = cfg.dimensions.width, cfg.dimensions.height
    exp_w, exp_h = cfg.outpainting.expanded_dimensions(cfg.dimensions)
    pad_x = (exp_w - inner_w) // 2
    pad_y = (exp_h - inner_h) // 2
    scale = float(inner_w) / float(cfg.yard_framing.src_inner_wh[0])
    return exp_w, exp_h, pad_x, pad_y, scale


def _warp_source_to_expanded_canvas(
    word_rgb: Image.Image,
    cfg: RailingOccluderSceneConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Maps native 3008x2000 yard photo onto the 2700x1800 expanded canvas around `src_center_xy`,
    returning `(seeded_rgb_u8, missing_mask_u8)` where out-of-bounds margins are mirror-seeded
    prior to neural LaMa outpainting.
    """
    src_np = np.asarray(word_rgb.convert("RGB"), dtype=np.uint8)
    src_h, src_w, _ = src_np.shape
    exp_w, exp_h, _, _, scale = _compute_yard_expanded_mapping(cfg)
    cx, cy = cfg.yard_framing.src_center_xy

    ex_grid = (np.arange(exp_w, dtype=np.float32) - exp_w * 0.5) / scale + cx
    ey_grid = (np.arange(exp_h, dtype=np.float32) - exp_h * 0.5) / scale + cy
    map_x, map_y = np.meshgrid(ex_grid, ey_grid)

    missing = (
        (map_x < 0.0)
        | (map_x > float(src_w - 1))
        | (map_y < 0.0)
        | (map_y > float(src_h - 1))
    )
    warped = cv2.remap(
        src_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if cfg.yard_framing.outpaint_seed_blur_sigma > 0.0 and missing.any():
        blurred = ndi.gaussian_filter(
            warped.astype(np.float32),
            sigma=(
                cfg.yard_framing.outpaint_seed_blur_sigma,
                cfg.yard_framing.outpaint_seed_blur_sigma,
                0.0,
            ),
        )
        warped = np.where(missing[:, :, None], np.clip(np.round(blurred), 0, 255), warped).astype(
            np.uint8
        )
    return warped, (missing.astype(np.uint8)) * 255


def _warp_source_mask_to_expanded(
    src_mask_u8: np.ndarray,
    cfg: RailingOccluderSceneConfig,
) -> np.ndarray:
    """Projects a native 3008x2000 mask onto the 2700x1800 expanded grid."""
    exp_w, exp_h, _, _, scale = _compute_yard_expanded_mapping(cfg)
    cx, cy = cfg.yard_framing.src_center_xy
    ex_grid = (np.arange(exp_w, dtype=np.float32) - exp_w * 0.5) / scale + cx
    ey_grid = (np.arange(exp_h, dtype=np.float32) - exp_h * 0.5) / scale + cy
    map_x, map_y = np.meshgrid(ex_grid, ey_grid)
    return cv2.remap(
        src_mask_u8,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _segment_source_crop_mask_u8(
    word_rgb: Image.Image,
    crop_xyxy: tuple[int, int, int, int],
    prompts: tuple[SamPointGroupPrompt, ...],
    exclude_local: tuple[BoundingBoxPrompt, ...],
    max_pinhole_fill_px: int,
    min_major_component_px: int,
    edge_feather_sigma: float,
    sam_extractor: SamBoxMaskExtractor,
) -> np.ndarray:
    """Runs Meta SAM on a high-resolution crop of IMG_WORD_0215 and embeds into 3008x2000."""
    src_w, src_h = word_rgb.size
    cx0, cy0, cx1, cy1 = crop_xyxy
    sub_rgb = word_rgb.crop((cx0, cy0, cx1, cy1))
    local_prompts = tuple(
        SamPointGroupPrompt(
            box=BoundingBoxPrompt(
                x0=max(0, p.box.x0 - cx0),
                y0=max(0, p.box.y0 - cy0),
                x1=min(cx1 - cx0, p.box.x1 - cx0),
                y1=min(cy1 - cy0, p.box.y1 - cy0),
            ),
            points_xy=tuple((px - cx0, py - cy0) for px, py in p.points_xy),
            point_labels=p.point_labels,
        )
        for p in prompts
    )
    crop_alpha = sam_extractor.segment_point_groups(
        rgb_image=sub_rgb,
        point_groups=local_prompts,
        exclude_boxes=exclude_local,
        max_pinhole_fill_px=max_pinhole_fill_px,
        min_major_component_px=min_major_component_px,
    )
    if edge_feather_sigma > 0.0:
        crop_alpha = np.clip(
            np.round(ndi.gaussian_filter(crop_alpha.astype(np.float32), sigma=edge_feather_sigma)),
            0,
            255,
        ).astype(np.uint8)
    full_alpha = np.zeros((src_h, src_w), dtype=np.uint8)
    full_alpha[cy0:cy1, cx0:cx1] = crop_alpha
    return full_alpha


def _glsl_smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Exact NumPy mirror of WebGL GLSL `smoothstep(edge0, edge1, x)`."""
    t = np.clip((x - edge0) / max(1e-7, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def evaluate_layer_world_z(
    depth_u8: np.ndarray,
    z_pos: float,
    depth_amplitude: float,
    pin_margin_uv: float,
) -> np.ndarray:
    """
    Evaluates the exact WebGL world-space depth `Z_world(x, y)` matching `ROOM_DEPTH_VERTEX_SHADER`:
      `Z_world(u, v) = z_pos + depth_amplitude * (depth_u8 / 255.0) * pin_uv(u, v)`.
    """
    h, w = depth_u8.shape
    raw_d = depth_u8.astype(np.float32) / 255.0
    if pin_margin_uv > 1e-5:
        u = (np.arange(w, dtype=np.float32) + 0.5) / float(w)
        v = 1.0 - (np.arange(h, dtype=np.float32) + 0.5) / float(h)
        pin_x = _glsl_smoothstep(0.0, pin_margin_uv, u) * _glsl_smoothstep(
            0.0, pin_margin_uv, 1.0 - u
        )
        pin_y = _glsl_smoothstep(0.0, pin_margin_uv, v) * _glsl_smoothstep(
            0.0, pin_margin_uv, 1.0 - v
        )
        raw_d = raw_d * (pin_y[:, None] * pin_x[None, :])
    return float(z_pos) + float(depth_amplitude) * raw_d


def _build_connected_active_mask(
    alpha_u8: np.ndarray,
    topo_cfg: HarmonicContactAnchorConfig,
) -> np.ndarray:
    """Builds the active silhouette domain `Omega` including anti-aliased edge pixels."""
    active = alpha_u8 >= topo_cfg.active_alpha_threshold_u8
    if topo_cfg.graph_connect_dilate_px > 0:
        k = 2 * topo_cfg.graph_connect_dilate_px + 1
        active = cv2.dilate(active.astype(np.uint8), np.ones((k, k), dtype=np.uint8)).astype(bool)
    if not np.any(active):
        raise RuntimeError("Active silhouette mask is empty in harmonic contact solver.")
    return active


def _solve_harmonic_contact_delta_field(
    active_mask: np.ndarray,
    dirichlet_mask: np.ndarray,
    dirichlet_values: np.ndarray,
) -> np.ndarray:
    """
    Solves the mixed Dirichlet-Neumann Laplace PDE `nabla^2 delta = 0` on the 4-connected graph
    of `active_mask` (`Omega`) with exact Dirichlet boundary `delta = dirichlet_values` on
    `dirichlet_mask` (`dOmega_contact`) and zero-flux Neumann boundary `d(delta)/dn = 0` on all
    free silhouette edges (`dOmega_free`), inheriting nearest solved offset on detached islands.
    """
    active = active_mask.astype(bool)
    dirich = dirichlet_mask.astype(bool) & active
    if not np.any(dirich):
        raise RuntimeError("Cannot solve harmonic contact anchoring without Dirichlet contact pixels.")

    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    lbl, _ = ndi.label(active, structure=struct4)
    anchored_comps = np.unique(lbl[dirich])
    anchored_comps = anchored_comps[anchored_comps > 0]
    omega_conn = np.isin(lbl, anchored_comps)

    free = omega_conn & ~dirich
    n_free = int(free.sum())
    solved = np.zeros(active.shape, dtype=np.float32)
    solved[dirich] = dirichlet_values[dirich].astype(np.float32)

    if n_free > 0:
        idx_map = -np.ones(active.shape, dtype=np.int32)
        idx_map[free] = np.arange(n_free, dtype=np.int32)
        fys, fxs = np.where(free)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        rhs = np.zeros(n_free, dtype=np.float64)
        diag = np.zeros(n_free, dtype=np.float64)
        h, w = active.shape

        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = fys + dy
            nx = fxs + dx
            valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            v_idx = np.where(valid)[0]
            ny_v = ny[v_idx]
            nx_v = nx[v_idx]
            in_om = omega_conn[ny_v, nx_v]
            o_idx = v_idx[in_om]
            ny_o = ny_v[in_om]
            nx_o = nx_v[in_om]
            diag[o_idx] += 1.0
            n_id = idx_map[ny_o, nx_o]
            is_free_nb = n_id >= 0
            if np.any(is_free_nb):
                rows.append(o_idx[is_free_nb])
                cols.append(n_id[is_free_nb])
                vals.append(-np.ones(int(is_free_nb.sum()), dtype=np.float64))
            is_dir_nb = ~is_free_nb
            if np.any(is_dir_nb):
                np.add.at(
                    rhs,
                    o_idx[is_dir_nb],
                    dirichlet_values[ny_o[is_dir_nb], nx_o[is_dir_nb]].astype(np.float64),
                )

        rows.append(np.arange(n_free, dtype=np.int32))
        cols.append(np.arange(n_free, dtype=np.int32))
        vals.append(diag)
        lap_csc = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_free, n_free),
        ).tocsc()
        sol = spla.spsolve(lap_csc, rhs)
        if not np.all(np.isfinite(sol)):
            raise RuntimeError("Harmonic contact sparse solver produced non-finite values.")
        solved[free] = sol.astype(np.float32)

    _, edt_idx = ndi.distance_transform_edt(~omega_conn, return_indices=True)
    return np.where(omega_conn, solved, solved[edt_idx[0], edt_idx[1]]).astype(np.float32)


def _compute_masked_da2_relief_norm(
    rgb_canvas: np.ndarray,
    alpha_u8: np.ndarray,
    depth_estimator: DepthAnythingEstimator,
    smooth_sigma: float,
    topo_cfg: HarmonicContactAnchorConfig,
) -> np.ndarray:
    """
    Computes normalized Depth Anything V2 relief inside `alpha_u8` over a neutral backdrop with
    normalized-convolution interior smoothing so exterior backdrop never bleeds into the silhouette.
    """
    a_f = (alpha_u8.astype(np.float32) / 255.0)[:, :, None]
    bg_gray = float(topo_cfg.neutral_backdrop_gray_u8)
    comp_rgb = np.clip(
        np.round(rgb_canvas.astype(np.float32) * a_f + bg_gray * (1.0 - a_f)),
        0,
        255,
    ).astype(np.uint8)
    raw_depth = depth_estimator.predict_depth(Image.fromarray(comp_rgb, mode="RGB"))
    solid = alpha_u8 >= topo_cfg.solid_contact_alpha_u8
    if not np.any(solid):
        raise RuntimeError("Cannot compute neural DA2 relief on an empty foreground matte.")
    vals = raw_depth[solid]
    p_lo = float(np.percentile(vals, topo_cfg.da2_percentile_low))
    p_hi = float(np.percentile(vals, topo_cfg.da2_percentile_high))
    norm = np.clip((raw_depth - p_lo) / max(1e-5, p_hi - p_lo), 0.0, 1.0).astype(np.float32)
    if smooth_sigma > 0.0:
        mask_f = (alpha_u8 >= topo_cfg.active_alpha_threshold_u8).astype(np.float32)
        num = ndi.gaussian_filter(norm * mask_f, sigma=smooth_sigma)
        den = np.maximum(1e-5, ndi.gaussian_filter(mask_f, sigma=smooth_sigma))
        norm = np.where(mask_f > 0.0, num / den, norm)
    return norm


def _detect_railing_contact_mask(
    rail_alpha_inner_u8: np.ndarray,
    topo_cfg: HarmonicContactAnchorConfig,
) -> tuple[np.ndarray, int]:
    """Detects all vertical post ground-contact bases + left ivy-bush anchor on 1500x1000 inner."""
    solid = rail_alpha_inner_u8 >= topo_cfg.solid_contact_alpha_u8
    if not np.any(solid):
        raise RuntimeError("Empty green railing alpha matte when detecting ground contacts.")
    inner_w = rail_alpha_inner_u8.shape[1]
    ymax_col = np.array(
        [np.where(solid[:, x])[0].max() if solid[:, x].any() else -1 for x in range(inner_w)],
        dtype=np.int32,
    )
    ymin_col = np.array(
        [np.where(solid[:, x])[0].min() if solid[:, x].any() else 99999 for x in range(inner_w)],
        dtype=np.int32,
    )
    in_post = (ymax_col >= 0) & (
        (ymax_col - ymin_col) >= topo_cfg.railing_post_min_vertical_span_px
    )
    n_lbl, lbl = cv2.connectedComponents(in_post[:, None].astype(np.uint8))
    post_count = int(n_lbl - 1)
    if post_count < topo_cfg.railing_min_detected_posts:
        raise RuntimeError(
            f"Detected only {post_count} green railing posts "
            f"(minimum required {topo_cfg.railing_min_detected_posts})."
        )
    contact = np.zeros_like(solid, dtype=bool)
    for cid in range(1, n_lbl):
        pxs = np.where(lbl[:, 0] == cid)[0]
        sub_ys, sub_xs = np.where(solid[:, pxs])
        y_bot = int(sub_ys.max())
        sel = sub_ys >= (y_bot - topo_cfg.railing_post_base_contact_band_px)
        contact[sub_ys[sel], pxs[sub_xs[sel]]] = True

    all_xs = np.where(solid)[1]
    left_limit_x = min(inner_w, int(all_xs.min()) + topo_cfg.railing_left_bush_contact_band_px)
    contact[:, :left_limit_x] |= solid[:, :left_limit_x]
    return contact, post_count


def _detect_installation_contact_mask(
    inst_alpha_inner_u8: np.ndarray,
    topo_cfg: HarmonicContactAnchorConfig,
) -> np.ndarray:
    """Detects rear stone/platform ground support + front bottom staircase step on 1500x1000 inner."""
    solid = inst_alpha_inner_u8 >= topo_cfg.solid_contact_alpha_u8
    if not np.any(solid):
        raise RuntimeError("Empty installation alpha matte when detecting ground contacts.")
    inner_w = inst_alpha_inner_u8.shape[1]
    contact = np.zeros_like(solid, dtype=bool)
    for x in range(inner_w):
        ys = np.where(solid[:, x])[0]
        if ys.size == 0:
            continue
        ym = int(ys.max())
        is_rear_support = (
            ym >= topo_cfg.installation_rear_support_min_y_px
            and x <= topo_cfg.installation_rear_support_max_x_px
        )
        is_front_step = ym >= topo_cfg.installation_front_step_min_y_px
        if is_rear_support or is_front_step:
            sel = ys >= (ym - topo_cfg.installation_contact_band_px)
            contact[ys[sel], x] = True
    if not np.any(contact):
        raise RuntimeError("Failed to detect ground-contact footprint on installation occluder.")
    return contact


def _detect_hedge_contact_mask(
    hedge_alpha_exp_u8: np.ndarray,
    topo_cfg: HarmonicContactAnchorConfig,
) -> np.ndarray:
    """Detects the bottom walkway/ground contact skirt of the sweeping hedge bank on 2700x1800."""
    solid = hedge_alpha_exp_u8 >= topo_cfg.solid_contact_alpha_u8
    if not np.any(solid):
        raise RuntimeError("Empty hedge alpha matte when detecting ground contacts.")
    max_x = min(hedge_alpha_exp_u8.shape[1], topo_cfg.hedge_contact_max_exp_x_px)
    contact = np.zeros_like(solid, dtype=bool)
    for x in range(max_x):
        ys = np.where(solid[:, x])[0]
        if ys.size == 0:
            continue
        ym = int(ys.max())
        sel = ys >= (ym - topo_cfg.hedge_contact_band_exp_px)
        contact[ys[sel], x] = True
    if not np.any(contact):
        raise RuntimeError("Failed to detect ground-contact skirt on sweeping hedge bank.")
    return contact


def _encode_extrapolated_world_z_u8(
    layer_label: str,
    world_z_interior: np.ndarray,
    active_mask: np.ndarray,
    alpha_u8: np.ndarray,
    z_pos: float,
    depth_amplitude: float,
    contact_mask: np.ndarray | None = None,
) -> Image.Image:
    """
    Extrapolates interior world depth outward from the eroded opaque silhouette (`alpha_u8 >= 20`)
    via Euclidean Distance Transform (`C^-1` free-silhouette discontinuity with zero rubber-sheeting
    or feathered-rim backdrop bleed), while preserving `contact_mask` anchors verbatim.
    """
    if depth_amplitude <= 0.0:
        raise ValueError(f"Layer '{layer_label}' requires positive depth_amplitude.")
    opaque = (alpha_u8 >= 20) & active_mask
    if not np.any(opaque):
        opaque = active_mask
    eroded = ndi.binary_erosion(opaque, iterations=2)
    seed_mask = eroded if np.any(eroded) else opaque
    if contact_mask is not None:
        seed_mask = seed_mask | (contact_mask & opaque)
    _, edt_idx = ndi.distance_transform_edt(~seed_mask, return_indices=True)
    world_z_full = np.where(
        seed_mask,
        world_z_interior,
        world_z_interior[edt_idx[0], edt_idx[1]],
    ).astype(np.float32)

    norm_d = (world_z_full - float(z_pos)) / float(depth_amplitude)
    min_norm = float(norm_d.min())
    max_norm = float(norm_d.max())
    if min_norm < 0.0 or max_norm > 1.0:
        raise RuntimeError(
            f"Layer '{layer_label}' world depth range [{world_z_full.min():.2f}, {world_z_full.max():.2f}] "
            f"exceeds configured [z_pos={z_pos:.2f}, z_pos+amp={z_pos + depth_amplitude:.2f}]."
        )
    depth_u8 = np.clip(np.round(norm_d * 255.0), 0, 255).astype(np.uint8)
    transparent = alpha_u8 == 0
    if np.any(transparent) and int(depth_u8[transparent].min()) == 0:
        raise RuntimeError(
            f"Layer '{layer_label}' contains zero-depth cliff pixels outside its alpha silhouette."
        )
    return Image.fromarray(depth_u8, mode="L")


def _harmonize_rgb_u8(
    rgb_u8: np.ndarray,
    gain_rgb: tuple[float, float, float],
    bias_rgb: tuple[float, float, float],
    luma_gamma: float,
) -> np.ndarray:
    """Applies calibrated gamma + RGB daylight harmonization without touching alpha."""
    arr = rgb_u8.astype(np.float32) / 255.0
    if abs(luma_gamma - 1.0) > 1e-4:
        arr = np.power(np.clip(arr, 0.0, 1.0), luma_gamma)
    gain = np.array(gain_rgb, dtype=np.float32)[None, None, :]
    bias = np.array(bias_rgb, dtype=np.float32)[None, None, :]
    out = arr * 255.0 * gain + bias
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _composite_grounded_rgba_on_inner(
    subject_rgb_u8: np.ndarray,
    subject_alpha_u8: np.ndarray,
    tight_thresh_u8: int,
    dest_tight_wh: tuple[int, int],
    dest_bottom_center_xy: tuple[int, int],
    inner_wh: tuple[int, int],
    shadow_rgb: tuple[int, int, int],
    shadow_band_ratio: float,
    shadow_offset_xy: tuple[int, int],
    shadow_sigma_yx: tuple[float, float],
    shadow_opacity: float,
) -> tuple[Image.Image, np.ndarray, tuple[float, float, float, float]]:
    """
    Resizes a tightly cropped neural RGBA cutout onto the 1500x1000 inner canvas with an anchored
    ground contact shadow baked beneath its base, returning `(rgba_image, subject_only_alpha_u8, bbox_xyxy)`.
    """
    ys, xs = np.where(subject_alpha_u8 >= tight_thresh_u8)
    if xs.size == 0:
        raise RuntimeError("Empty subject alpha matte in _composite_grounded_rgba_on_inner.")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    crop_rgba = Image.fromarray(
        np.dstack([subject_rgb_u8[y0:y1, x0:x1], subject_alpha_u8[y0:y1, x0:x1]]),
        mode="RGBA",
    )
    tw, th = dest_tight_wh
    scaled_rgba = np.asarray(crop_rgba.resize((tw, th), Image.Resampling.LANCZOS), dtype=np.uint8)

    inner_w, inner_h = inner_wh
    bx, by = dest_bottom_center_xy
    dx0 = int(round(bx - tw * 0.5))
    dy0 = int(round(by - th))
    dx1 = dx0 + tw
    dy1 = dy0 + th

    sx0 = max(0, -dx0)
    sy0 = max(0, -dy0)
    sx1 = tw - max(0, dx1 - inner_w)
    sy1 = th - max(0, dy1 - inner_h)
    tx0, ty0 = max(0, dx0), max(0, dy0)
    tx1, ty1 = tx0 + (sx1 - sx0), ty0 + (sy1 - sy0)
    if tx1 <= tx0 or ty1 <= ty0:
        raise ValueError("Grounded subject placement falls completely outside inner canvas.")

    subj_rgb_canvas = np.zeros((inner_h, inner_w, 3), dtype=np.uint8)
    subj_alpha_canvas = np.zeros((inner_h, inner_w), dtype=np.uint8)
    subj_rgb_canvas[ty0:ty1, tx0:tx1] = scaled_rgba[sy0:sy1, sx0:sx1, :3]
    subj_alpha_canvas[ty0:ty1, tx0:tx1] = scaled_rgba[sy0:sy1, sx0:sx1, 3]

    # Synthesize grounded contact shadow from the lower support band of the alpha silhouette
    band_top = max(0, ty1 - max(8, int(round((ty1 - ty0) * shadow_band_ratio))))
    foot_mask = np.zeros((inner_h, inner_w), dtype=np.float32)
    foot_mask[band_top:ty1, tx0:tx1] = (
        subj_alpha_canvas[band_top:ty1, tx0:tx1].astype(np.float32) / 255.0
    )
    ox, oy = shadow_offset_xy
    shifted_foot = ndi.shift(foot_mask, shift=(float(oy), float(ox)), order=1, mode="constant")
    shadow_alpha = (
        ndi.gaussian_filter(shifted_foot, sigma=shadow_sigma_yx) * float(shadow_opacity)
    )
    shadow_alpha = np.clip(shadow_alpha, 0.0, 1.0)

    subj_a_f = subj_alpha_canvas.astype(np.float32) / 255.0
    out_a_f = subj_a_f + shadow_alpha * (1.0 - subj_a_f)
    sh_col = np.array(shadow_rgb, dtype=np.float32)[None, None, :]
    num_rgb = (
        subj_rgb_canvas.astype(np.float32) * subj_a_f[:, :, None]
        + sh_col * (shadow_alpha * (1.0 - subj_a_f))[:, :, None]
    )
    out_rgb_u8 = np.where(
        out_a_f[:, :, None] > 1e-4,
        np.clip(np.round(num_rgb / np.maximum(1e-4, out_a_f[:, :, None])), 0, 255),
        0.0,
    ).astype(np.uint8)
    out_a_u8 = np.clip(np.round(out_a_f * 255.0), 0, 255).astype(np.uint8)

    ay, ax = np.where(subj_alpha_canvas >= tight_thresh_u8)
    bbox = (float(ax.min()), float(ay.min()), float(ax.max() + 1), float(ay.max() + 1))
    return Image.fromarray(np.dstack([out_rgb_u8, out_a_u8]), mode="RGBA"), subj_alpha_canvas, bbox


def _extract_sisters_rgba(
    project_root: Path,
    cfg: RailingOccluderSceneConfig,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    """Loads verified neural sisters cutout or synthesizes directly from IMG_HERZ_0397 via SAM+IS-Net."""
    verified_path = project_root / cfg.sisters_reveal.verified_mid_rgba_relpath
    if verified_path.exists():
        rgba = np.asarray(Image.open(verified_path).convert("RGBA"), dtype=np.uint8)
        if int((rgba[:, :, 3] > 64).sum()) > 5000:
            return rgba[:, :, :3], rgba[:, :, 3]

    herz_path = project_root / cfg.sisters_reveal.herz_source_relpath
    if not herz_path.exists():
        raise RuntimeError(f"Missing sisters archival source: {herz_path}")
    herz_rgb = Image.open(herz_path).convert("RGB")
    ext = cfg.sisters_reveal.sisters_extraction
    sam_u8 = sam_extractor.segment_point_groups(
        rgb_image=herz_rgb,
        point_groups=ext.sam_prompts,
        exclude_boxes=ext.sam_exclude_boxes,
        max_pinhole_fill_px=ext.max_pinhole_fill_px,
        min_major_component_px=ext.min_major_component_px,
    )
    roi_mattes = [
        extract_oriented_isnet_roi_alpha_f32(herz_rgb, mb, matte_extractor)
        for mb in ext.hair_matte_boxes
    ]
    merged_u8 = merge_sam_and_oriented_mattes_u8(
        sam_u8,
        roi_mattes,
        ext.hair_matte_boxes,
        support_dilate_px=ext.support_dilate_px,
        edge_feather_sigma=cfg.segmentation.edge_feather_sigma,
    )
    clean_rgb = decontaminate_matte_edge_rgb(np.asarray(herz_rgb, dtype=np.uint8), merged_u8)
    return clean_rgb, merged_u8


def _extract_installation_cutout(
    project_root: Path,
    occ_cfg: InstallationOccluderConfig,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    """Segments IMG_PORT_0062 ('מרחב לא מוגן 7') via hybrid Meta SAM + IS-Net matting."""
    port_path = project_root / occ_cfg.source_relpath
    if not port_path.exists():
        raise RuntimeError(f"Missing installation source image: {port_path}")
    port_rgb = Image.open(port_path).convert("RGB")

    sam_alpha = sam_extractor.segment_point_groups(
        rgb_image=port_rgb,
        point_groups=occ_cfg.sam_prompts,
        exclude_boxes=(),
        max_pinhole_fill_px=occ_cfg.max_pinhole_fill_px,
        min_major_component_px=occ_cfg.min_component_area_px,
    )
    rembg_alpha = matte_extractor.extract_alpha(port_rgb)

    ksup = (
        occ_cfg.support_dilate_px
        if occ_cfg.support_dilate_px % 2 == 1
        else occ_cfg.support_dilate_px + 1
    )
    support = cv2.dilate(
        (sam_alpha > 96).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksup, ksup)),
    )

    cutout_holes = np.zeros_like(sam_alpha, dtype=bool)
    for wb in occ_cfg.cutout_window_boxes:
        cutout_holes[wb.y0 : wb.y1, wb.x0 : wb.x1] |= (
            rembg_alpha[wb.y0 : wb.y1, wb.x0 : wb.x1] < occ_cfg.cutout_rembg_max_u8
        )

    foliage_zones = np.zeros_like(sam_alpha, dtype=bool)
    for fb in occ_cfg.foliage_refine_boxes:
        foliage_zones[fb.y0 : fb.y1, fb.x0 : fb.x1] = True

    sam_f = sam_alpha.astype(np.float32)
    rem_f = rembg_alpha.astype(np.float32) * support.astype(np.float32)
    fused = np.where(
        foliage_zones,
        np.maximum(sam_f * occ_cfg.foliage_sam_floor_weight, rem_f),
        np.maximum(sam_f, rem_f * occ_cfg.solid_rembg_weight),
    )
    fused[cutout_holes] = 0.0
    if occ_cfg.edge_feather_sigma > 0.0:
        fused = ndi.gaussian_filter(fused, sigma=occ_cfg.edge_feather_sigma)
    fused_u8 = np.clip(np.round(fused), 0, 255).astype(np.uint8)

    nl, lbls, stats, _ = cv2.connectedComponentsWithStats(
        (fused_u8 >= occ_cfg.alpha_tight_threshold_u8).astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros_like(fused_u8, dtype=bool)
    for cid in range(1, nl):
        if int(stats[cid, cv2.CC_STAT_AREA]) >= occ_cfg.min_component_area_px:
            keep |= lbls == cid
    fused_u8 = np.where(keep, fused_u8, 0).astype(np.uint8)

    clean_rgb = decontaminate_matte_edge_rgb(np.asarray(port_rgb, dtype=np.uint8), fused_u8)
    harm_rgb = _harmonize_rgb_u8(
        clean_rgb,
        gain_rgb=occ_cfg.yard_gain_rgb,
        bias_rgb=occ_cfg.yard_bias_rgb,
        luma_gamma=occ_cfg.luma_gamma,
    )
    return harm_rgb, fused_u8


def _calibrate_yard_depth_u8(
    clean_bg_rgb: Image.Image,
    cfg: RailingOccluderSceneConfig,
    depth_estimator: DepthAnythingEstimator,
) -> Image.Image:
    """Computes calibrated 2700x1800 Depth Anything V2 relief map with smooth ramp contact pins."""
    dc = cfg.depth_calibration
    exp_w, exp_h, pad_x, pad_y, _ = _compute_yard_expanded_mapping(cfg)
    raw = depth_estimator.predict_depth(clean_bg_rgb)
    smooth = ndi.gaussian_filter(raw, sigma=dc.da2_gaussian_sigma)

    p_lo = float(np.percentile(smooth, dc.sky_norm_guard_band * 100.0))
    p_hi = float(np.percentile(smooth, 99.2))
    norm = np.clip((smooth - p_lo) / max(1e-5, p_hi - p_lo), 0.0, 1.0)
    world_z = dc.da2_z_min + norm * (dc.da2_z_max - dc.da2_z_min)

    yy, xx = np.mgrid[0:exp_h, 0:exp_w].astype(np.float32)
    for pin in dc.contact_pins:
        px = float(pad_x) + pin.inner_xy[0]
        py = float(pad_y) + pin.inner_xy[1]
        rx = max(4.0, pin.radii_xy[0])
        ry = max(4.0, pin.radii_xy[1])
        dist = np.sqrt(((xx - px) / rx) ** 2 + ((yy - py) / ry) ** 2)
        fw = max(0.15, pin.feather_px / np.sqrt(rx * ry))
        t = np.clip((dist - 1.0) / fw, 0.0, 1.0)
        w = 1.0 - t * t * (3.0 - 2.0 * t)
        world_z = world_z * (1.0 - w) + float(pin.target_z) * w

    if dc.final_smooth_sigma > 0.0:
        world_z = ndi.gaussian_filter(world_z, sigma=dc.final_smooth_sigma)
    disp = np.clip((world_z - dc.scene_bg_z) / max(1e-5, dc.room_depth_amplitude), 0.0, 1.0)
    return Image.fromarray(np.clip(np.round(disp * 255.0), 0, 255).astype(np.uint8), mode="L")


# ============================================================================
# Main Scene Builder & SSOT Manifest Exporter
# ============================================================================


def _build_separated_layer_manifest_dict(spec: SeparatedYardLayerConfig) -> dict[str, Any]:
    """Serializes a separated yard layer into the runtime `extraLayers` manifest format."""
    entry: dict[str, Any] = {
        "id": spec.layer_id,
        "labelHe": spec.label_he,
        "texture": f"assets/scenes/{spec.output_filename}",
        "zPos": spec.z_pos,
        "renderOrder": spec.render_order,
        "expanded": spec.is_expanded,
        "baseOpacity": 1.0,
        "excursionBoost": 0.0,
        "depthTest": False,
    }
    if spec.depth_output_filename is not None and spec.depth_amplitude > 0.0:
        entry["depthTexture"] = f"assets/scenes/{spec.depth_output_filename}"
        entry["depthAmplitude"] = spec.depth_amplitude
        entry["pinMarginUv"] = spec.pin_margin_uv
        entry["segmentsX"] = spec.segments_xy[0]
        entry["segmentsY"] = spec.segments_xy[1]
    return entry


def build_railing_occluder_manifest_entry(cfg: RailingOccluderSceneConfig) -> dict[str, Any]:
    """Exports the SSOT runtime scene manifest dictionary for `fried_yard_installation`."""
    exp_w, exp_h = cfg.outpainting.expanded_dimensions(cfg.dimensions)
    sr = cfg.sisters_reveal
    occ = cfg.installation_occluder
    return {
        "id": cfg.scene_id,
        "title": cfg.title_he,
        "shortLabelHe": cfg.short_label_he,
        "icon": cfg.icon,
        "exhibitionGroup": cfg.exhibition_group,
        "sourceCatalogIds": [
            cfg.yard_framing.catalog_id,
            occ.catalog_id,
            sr.herz_catalog_id,
        ],
        "segmentationBackend": "railing_occluder_hybrid_sam_isnet",
        "depthBackend": "railing_occluder_harmonic_contact_da2",
        "innerWidth": cfg.dimensions.width,
        "innerHeight": cfg.dimensions.height,
        "bgExpandedWidth": exp_w,
        "bgExpandedHeight": exp_h,
        "bgExpansionFactor": cfg.outpainting.expansion_factor,
        "fgExpanded": cfg.fg_is_expanded,
        "boreSightAtRest": cfg.bore_sight_at_rest,
        "roomDepthAmplitude": cfg.depth_calibration.room_depth_amplitude,
        "sceneBgZ": cfg.scene_bg_z,
        "midZ": cfg.mid_z,
        "fgZ": cfg.fg_z,
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
        "fgTexture": f"assets/scenes/{cfg.fg_output_filename}",
        "depthTexture": f"assets/scenes/{cfg.depth_output_filename}",
        "extraLayers": [
            _build_separated_layer_manifest_dict(cfg.hedge_layer),
            _build_separated_layer_manifest_dict(cfg.railing_layer),
        ],
        "portalShelves": [
            {
                "id": sr.portal_id,
                "targetSceneId": sr.portal_target_scene_id,
                "labelHe": sr.portal_label_he,
                "shortBadgeHe": sr.portal_short_badge_he,
                "showMicroBadge": sr.portal_show_micro_badge,
                "innerBoxPx": [round(v, 2) for v in cfg.portal_hit_box_inner_px],
                "worldZ": cfg.mid_z,
                "recessDepth": sr.portal_recess_depth,
                "embedDiorama": False,
                "showFrame": False,
                "occluder": {
                    "texture": f"assets/scenes/{occ.output_filename}",
                    "depthTexture": f"assets/scenes/{occ.depth_output_filename}",
                    "depthAmplitude": occ.depth_amplitude,
                    "pinMarginUv": occ.pin_margin_uv,
                    "segmentsX": occ.segments_xy[0],
                    "segmentsY": occ.segments_xy[1],
                    "worldZ": occ.world_z,
                    "renderOrder": occ.render_order,
                    "fullInnerCanvas": True,
                    "alphaHitThreshold": occ.alpha_hit_threshold_u8,
                    "innerBoxPx": [round(v, 2) for v in cfg.occluder_hit_box_inner_px],
                    "labelHe": occ.label_he,
                    "coverageRatio": round(cfg.occluder_coverage_ratio, 4),
                    "locksPortal": True,
                },
            }
        ],
        "reflectionTexture": None,
        "hasReflection": False,
    }


def verify_railing_occluder_depth_topology(
    cfg: RailingOccluderSceneConfig,
    project_root: Path = _REPO_ROOT,
) -> dict[str, float]:
    """
    Programmatically verifies 2.5D Depth Topology laws on `fried_yard_installation`:
      - Law 1 (C0 Contact Continuity): `max |Z_world_layer - Z_world_bg| <= max_contact_delta_z`
        across all 4 railing post bases + left bush anchor, installation ground supports,
        hedge walkway skirt, and Tami & Er'ela Sadan's shoes on `layer0`.
      - Law 2 (C^-1 Free-Silhouette Discontinuity): `pinMarginUv == 0.0` on all cutout depth layers
        and zero `depth_u8 == 0` cliff pixels outside alpha silhouettes (`EDT` extrapolation).
      - Physical Front-to-Back Clearance: green railing stays strictly in front of Tami & Er'ela.
    """
    if not cfg.bore_sight_at_rest:
        raise RuntimeError(f"[{cfg.scene_id}] bore_sight_at_rest must be True.")
    for name, pin_val in (
        ("hedge_layer", cfg.hedge_layer.pin_margin_uv),
        ("railing_layer", cfg.railing_layer.pin_margin_uv),
        ("installation_occluder", cfg.installation_occluder.pin_margin_uv),
    ):
        if abs(pin_val) > 1e-7:
            raise RuntimeError(f"[{cfg.scene_id}] {name}.pin_margin_uv must be 0.0, got {pin_val}.")
    if cfg.railing_layer.depth_output_filename is None or cfg.hedge_layer.depth_output_filename is None:
        raise RuntimeError(f"[{cfg.scene_id}] Both hedge_layer and railing_layer require depth maps.")

    scenes_dir = project_root / "prototype" / "assets" / "scenes"
    _, _, pad_x, pad_y, _ = _compute_yard_expanded_mapping(cfg)
    inner_w, inner_h = cfg.dimensions.width, cfg.dimensions.height
    topo = cfg.contact_topology

    bg_d = np.asarray(Image.open(scenes_dir / cfg.depth_output_filename), dtype=np.uint8)
    z_bg_exp = evaluate_layer_world_z(
        bg_d,
        z_pos=cfg.scene_bg_z,
        depth_amplitude=cfg.depth_calibration.room_depth_amplitude,
        pin_margin_uv=topo.room_pin_margin_uv,
    )
    z_bg_inner = z_bg_exp[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]

    # 1. Green Railing contact continuity + EDT extrapolation
    rail_a = np.asarray(Image.open(scenes_dir / cfg.railing_layer.output_filename))[:, :, 3]
    rail_d = np.asarray(Image.open(scenes_dir / cfg.railing_layer.depth_output_filename), dtype=np.uint8)
    if int(rail_d[rail_a == 0].min()) == 0:
        raise RuntimeError("Railing depth map contains unextrapolated 0-depth pixels outside alpha.")
    z_rail = evaluate_layer_world_z(
        rail_d,
        z_pos=cfg.railing_layer.z_pos,
        depth_amplitude=cfg.railing_layer.depth_amplitude,
        pin_margin_uv=cfg.railing_layer.pin_margin_uv,
    )
    rail_contact, post_count = _detect_railing_contact_mask(rail_a, topo)
    rail_max_dz = float(np.abs(z_rail[rail_contact] - z_bg_inner[rail_contact]).max())

    # 2. Installation occluder contact continuity + EDT extrapolation
    occ = cfg.installation_occluder
    inst_a = np.asarray(Image.open(scenes_dir / occ.output_filename))[:, :, 3]
    inst_d = np.asarray(Image.open(scenes_dir / occ.depth_output_filename), dtype=np.uint8)
    if int(inst_d[inst_a == 0].min()) == 0:
        raise RuntimeError("Installation depth map contains unextrapolated 0-depth pixels outside alpha.")
    z_inst = evaluate_layer_world_z(
        inst_d,
        z_pos=occ.world_z,
        depth_amplitude=occ.depth_amplitude,
        pin_margin_uv=occ.pin_margin_uv,
    )
    inst_contact = _detect_installation_contact_mask(inst_a, topo)
    inst_max_dz = float(np.abs(z_inst[inst_contact] - z_bg_inner[inst_contact]).max())

    # 3. Hedge bank contact continuity + EDT extrapolation
    hedge_a = np.asarray(Image.open(scenes_dir / cfg.hedge_layer.output_filename))[:, :, 3]
    hedge_d = np.asarray(Image.open(scenes_dir / cfg.hedge_layer.depth_output_filename), dtype=np.uint8)
    if int(hedge_d[hedge_a == 0].min()) == 0:
        raise RuntimeError("Hedge depth map contains unextrapolated 0-depth pixels outside alpha.")
    z_hedge = evaluate_layer_world_z(
        hedge_d,
        z_pos=cfg.hedge_layer.z_pos,
        depth_amplitude=cfg.hedge_layer.depth_amplitude,
        pin_margin_uv=cfg.hedge_layer.pin_margin_uv,
    )
    hedge_contact = _detect_hedge_contact_mask(hedge_a, topo)
    hedge_max_dz = float(np.abs(z_hedge[hedge_contact] - z_bg_exp[hedge_contact]).max())

    # 4. Sisters (`mid`) shoe contact & railing clearance
    mid_a = np.asarray(Image.open(scenes_dir / cfg.mid_output_filename))[:, :, 3]
    mid_solid = mid_a >= topo.solid_contact_alpha_u8
    mys, _ = np.where(mid_solid)
    if mys.size == 0:
        raise RuntimeError("Empty sisters mid layer alpha matte during topology verification.")
    shoe_y_min = int(mys.max()) - topo.sisters_shoe_contact_band_px
    shoe_mask = mid_solid.copy()
    shoe_mask[:shoe_y_min, :] = False
    sisters_shoe_max_dz = float(np.abs(float(cfg.mid_z) - z_bg_inner[shoe_mask]).max())

    rail_sisters_overlap = (rail_a >= topo.solid_contact_alpha_u8) & mid_solid
    if not np.any(rail_sisters_overlap):
        raise RuntimeError("Expected green railing to cross in front of sisters mid silhouette.")
    min_rail_sisters_clearance_z = float((z_rail[rail_sisters_overlap] - float(cfg.mid_z)).min())
    if min_rail_sisters_clearance_z <= 5.0:
        raise RuntimeError(
            f"Green railing fails front-of-sisters clearance: min dZ={min_rail_sisters_clearance_z:.2f}"
        )

    for label, dz in (
        ("railing_posts", rail_max_dz),
        ("installation_base", inst_max_dz),
        ("hedge_skirt", hedge_max_dz),
        ("sisters_shoes", sisters_shoe_max_dz),
    ):
        if dz > topo.max_contact_delta_z:
            raise RuntimeError(
                f"[{cfg.scene_id}] C0 contact continuity violated on '{label}': "
                f"max |dZ_world|={dz:.3f} > {topo.max_contact_delta_z:.3f}"
            )

    return {
        "railing_post_count": float(post_count),
        "railing_max_contact_dz": rail_max_dz,
        "installation_max_contact_dz": inst_max_dz,
        "hedge_max_contact_dz": hedge_max_dz,
        "sisters_shoe_max_dz": sisters_shoe_max_dz,
        "min_rail_sisters_clearance_z": min_rail_sisters_clearance_z,
    }


def build_railing_occluder_scene(
    cfg: RailingOccluderSceneConfig | None = None,
    project_root: Path = _REPO_ROOT,
) -> tuple[RailingOccluderSceneConfig, dict[str, Any]]:
    """Synthesizes all neural layers for `fried_yard_installation` and returns updated config + manifest."""
    active_cfg = cfg or default_railing_occluder_scene_config()
    scenes_dir = project_root / "prototype" / "assets" / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    topo = active_cfg.contact_topology

    word_path = project_root / active_cfg.yard_framing.source_relpath
    if not word_path.exists():
        raise RuntimeError(f"Missing yard source image: {word_path}")
    word_rgb = Image.open(word_path).convert("RGB")

    device = select_torch_device()
    print(f"[{active_cfg.scene_id}] Initializing neural vision models on device={device}...")
    sam_extractor = SamBoxMaskExtractor(active_cfg.segmentation)
    matte_extractor = SalientMatteExtractor(active_cfg.segmentation)
    depth_estimator = DepthAnythingEstimator(active_cfg.depth_model, device=device)
    lama_inpainter = LamaOcclusionInpainter(active_cfg.inpainting)

    exp_w, exp_h, pad_x, pad_y, _ = _compute_yard_expanded_mapping(active_cfg)
    inner_w, inner_h = active_cfg.dimensions.width, active_cfg.dimensions.height

    # 1. Warp source yard onto 2700x1800 expanded canvas & segment multi-plane elements with SAM
    print(f"[{active_cfg.scene_id}] Segmenting yard layers (hedge, green railing, brush pile, bystander)...")
    seeded_exp_rgb, missing_exp_u8 = _warp_source_to_expanded_canvas(word_rgb, active_cfg)

    bystander_src_u8 = _segment_source_crop_mask_u8(
        word_rgb=word_rgb,
        crop_xyxy=active_cfg.yard_framing.bystander_crop_xyxy,
        prompts=active_cfg.yard_framing.bystander_sam_prompts,
        exclude_local=(),
        max_pinhole_fill_px=400,
        min_major_component_px=200,
        edge_feather_sigma=0.0,
        sam_extractor=sam_extractor,
    )
    kb = 2 * active_cfg.yard_framing.bystander_dilate_src_px + 1
    bystander_src_u8 = cv2.dilate(
        bystander_src_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
    )
    bystander_exp_u8 = _warp_source_mask_to_expanded(bystander_src_u8, active_cfg)

    def _extract_yard_layer(spec: SeparatedYardLayerConfig) -> tuple[np.ndarray, np.ndarray]:
        src_alpha = _segment_source_crop_mask_u8(
            word_rgb=word_rgb,
            crop_xyxy=spec.src_crop_xyxy,
            prompts=spec.sam_prompts,
            exclude_local=spec.sam_exclude_boxes_local,
            max_pinhole_fill_px=spec.max_pinhole_fill_px,
            min_major_component_px=spec.min_major_component_px,
            edge_feather_sigma=spec.edge_feather_sigma,
            sam_extractor=sam_extractor,
        )
        exp_alpha = _warp_source_mask_to_expanded(src_alpha, active_cfg)
        clean_exp_rgb = decontaminate_matte_edge_rgb(seeded_exp_rgb, exp_alpha)
        return clean_exp_rgb, exp_alpha

    hedge_rgb_exp, hedge_alpha_exp = _extract_yard_layer(active_cfg.hedge_layer)
    rail_rgb_exp, rail_alpha_exp = _extract_yard_layer(active_cfg.railing_layer)
    pile_rgb_exp, pile_alpha_exp = _extract_yard_layer(active_cfg.brush_pile_fg)
    # Prevent outpaint-reflected brush pile pixels in the far-left synthetic strip from floating in fg
    pile_alpha_exp[missing_exp_u8 > 0] = 0

    # Save foreground brush pile RGBA (2700x1800 expanded)
    fg_path = scenes_dir / active_cfg.fg_output_filename
    Image.fromarray(np.dstack([pile_rgb_exp, pile_alpha_exp]), mode="RGBA").save(
        fg_path, format="PNG", optimize=True
    )

    # Save sweeping hedge bank RGBA (2700x1800 expanded)
    hedge_path = scenes_dir / active_cfg.hedge_layer.output_filename
    Image.fromarray(np.dstack([hedge_rgb_exp, hedge_alpha_exp]), mode="RGBA").save(
        hedge_path, format="PNG", optimize=True
    )

    # Save green handrail RGBA (1500x1000 inner)
    rail_inner_rgb = rail_rgb_exp[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]
    rail_inner_alpha = rail_alpha_exp[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]
    rail_inner_rgba = np.dstack([rail_inner_rgb, rail_inner_alpha])
    rail_path = scenes_dir / active_cfg.railing_layer.output_filename
    Image.fromarray(rail_inner_rgba, mode="RGBA").save(rail_path, format="PNG", optimize=True)

    # 2. Neural LaMa background inpainting (global dilated mask + full-res local railing ROI pass)
    print(f"[{active_cfg.scene_id}] Synthesizing clean occluded yard & outpaint margins via LaMa...")

    def _dilate_exp(mask_u8: np.ndarray, rad_px: int) -> np.ndarray:
        k = 2 * max(25, rad_px) + 1
        return cv2.dilate(
            (mask_u8 > 24).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        )

    global_hole_u8 = np.maximum.reduce(
        [
            _dilate_exp(missing_exp_u8, active_cfg.yard_framing.outpaint_seam_dilate_exp_px),
            _dilate_exp(bystander_exp_u8, active_cfg.inpainting.dilation_radius_px),
            _dilate_exp(hedge_alpha_exp, active_cfg.hedge_layer.bg_inpaint_dilate_exp_px),
            _dilate_exp(rail_alpha_exp, active_cfg.railing_layer.bg_inpaint_dilate_exp_px),
            _dilate_exp(pile_alpha_exp, active_cfg.brush_pile_fg.bg_inpaint_dilate_exp_px),
        ]
    )

    ds_w, ds_h = active_cfg.yard_framing.lama_global_downsample_wh
    seeded_pil = Image.fromarray(seeded_exp_rgb, mode="RGB")
    inpainted_ds = lama_inpainter.inpaint_background(
        rgb_image=seeded_pil.resize((ds_w, ds_h), Image.Resampling.LANCZOS),
        binary_mask_u8=np.where(
            cv2.resize(global_hole_u8, (ds_w, ds_h), interpolation=cv2.INTER_NEAREST) > 32,
            255,
            0,
        ).astype(np.uint8),
        dilation_radius_px=2,
        mask_blur_sigma=0.0,
    )
    inpainted_f32 = np.asarray(
        inpainted_ds.resize((exp_w, exp_h), Image.Resampling.BICUBIC), dtype=np.float32
    )

    gx0, gy0, gx1, gy1 = active_cfg.yard_framing.grain_patch_exp_xyxy
    grain_patch = seeded_exp_rgb[gy0:gy1, gx0:gx1].astype(np.float32)
    grain_hp = grain_patch - ndi.gaussian_filter(
        grain_patch,
        sigma=(
            active_cfg.yard_framing.grain_highpass_sigma_px,
            active_cfg.yard_framing.grain_highpass_sigma_px,
            0.0,
        ),
    )
    tiled_grain = np.tile(
        grain_hp,
        (exp_h // grain_hp.shape[0] + 1, exp_w // grain_hp.shape[1] + 1, 1),
    )[:exp_h, :exp_w]
    inpainted_f32 = np.clip(
        inpainted_f32 + tiled_grain * active_cfg.yard_framing.grain_strength,
        0.0,
        255.0,
    )

    blend_w = np.clip(
        ndi.gaussian_filter(
            (global_hole_u8 > 0).astype(np.float32),
            sigma=active_cfg.yard_framing.lama_feather_sigma_exp_px,
        ),
        0.0,
        1.0,
    )[:, :, None]
    bg_exp_f32 = seeded_exp_rgb.astype(np.float32) * (1.0 - blend_w) + inpainted_f32 * blend_w

    # Full-resolution local LaMa refinement behind the thin green railing posts & handrail
    rys, rxs = np.where(rail_alpha_exp > 16)
    if rxs.size > 0:
        rpad = active_cfg.yard_framing.lama_railing_roi_pad_exp_px
        rx0 = max(0, int(rxs.min()) - rpad)
        ry0 = max(0, int(rys.min()) - rpad)
        rx1 = min(exp_w, int(rxs.max()) + rpad)
        ry1 = min(exp_h, int(rys.max()) + rpad)
        rail_hole_local = _dilate_exp(
            rail_alpha_exp[ry0:ry1, rx0:rx1],
            active_cfg.railing_layer.bg_inpaint_dilate_exp_px,
        )
        roi_seed = Image.fromarray(
            np.clip(np.round(bg_exp_f32[ry0:ry1, rx0:rx1]), 0, 255).astype(np.uint8),
            mode="RGB",
        )
        roi_orig = seeded_exp_rgb[ry0:ry1, rx0:rx1].copy()
        roi_inp_seed = np.where(
            rail_hole_local[:, :, None] > 0,
            np.asarray(roi_seed, dtype=np.uint8),
            roi_orig,
        )
        local_inp = lama_inpainter.inpaint_background(
            rgb_image=Image.fromarray(roi_inp_seed, mode="RGB"),
            binary_mask_u8=rail_hole_local,
            dilation_radius_px=3,
            mask_blur_sigma=1.0,
        )
        local_w = np.clip(
            ndi.gaussian_filter((rail_hole_local > 0).astype(np.float32), sigma=3.0),
            0.0,
            1.0,
        )[:, :, None]
        bg_exp_f32[ry0:ry1, rx0:rx1] = (
            bg_exp_f32[ry0:ry1, rx0:rx1] * (1.0 - local_w)
            + np.asarray(local_inp, dtype=np.float32) * local_w
        )

    clean_bg_rgb = Image.fromarray(
        np.clip(np.round(bg_exp_f32), 0, 255).astype(np.uint8), mode="RGB"
    )
    bg_path = scenes_dir / active_cfg.bg_output_filename
    clean_bg_rgb.save(
        bg_path, format="JPEG", quality=active_cfg.inpainting.jpeg_quality, optimize=True
    )

    print(f"[{active_cfg.scene_id}] Computing calibrated ghost-free Depth Anything V2 yard depth...")
    bg_depth_img = _calibrate_yard_depth_u8(clean_bg_rgb, active_cfg, depth_estimator)
    depth_path = scenes_dir / active_cfg.depth_output_filename
    bg_depth_img.save(depth_path, format="PNG", optimize=True)

    bg_depth_u8 = np.asarray(bg_depth_img, dtype=np.uint8)
    z_world_bg_exp = evaluate_layer_world_z(
        bg_depth_u8,
        z_pos=active_cfg.scene_bg_z,
        depth_amplitude=active_cfg.depth_calibration.room_depth_amplitude,
        pin_margin_uv=topo.room_pin_margin_uv,
    )
    z_world_bg_inner = z_world_bg_exp[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]

    # 3. Synthesize C0 harmonic contact-anchored + C^-1 EDT depth maps for hedge & green railing
    print(f"[{active_cfg.scene_id}] Solving harmonic contact-anchored depth for hedge & green railing...")
    if active_cfg.hedge_layer.depth_output_filename is None:
        raise ValueError("hedge_layer requires depth_output_filename.")
    hedge_active = _build_connected_active_mask(hedge_alpha_exp, topo)
    hedge_contact = _detect_hedge_contact_mask(hedge_alpha_exp, topo)
    hedge_da2_norm = _compute_masked_da2_relief_norm(
        rgb_canvas=hedge_rgb_exp,
        alpha_u8=hedge_alpha_exp,
        depth_estimator=depth_estimator,
        smooth_sigma=active_cfg.hedge_layer.depth_smooth_sigma,
        topo_cfg=topo,
    )
    z_free_hedge = topo.hedge_relief_amplitude_z * (
        hedge_da2_norm - float(hedge_da2_norm[hedge_contact].mean())
    )
    target_delta_hedge = z_world_bg_exp - z_free_hedge
    ds_h = max(1, topo.hedge_solve_downsample)
    if ds_h > 1:
        delta_hedge_ds = _solve_harmonic_contact_delta_field(
            active_mask=hedge_active[::ds_h, ::ds_h],
            dirichlet_mask=hedge_contact[::ds_h, ::ds_h],
            dirichlet_values=target_delta_hedge[::ds_h, ::ds_h],
        )
        delta_hedge = cv2.resize(
            delta_hedge_ds,
            (exp_w, exp_h),
            interpolation=cv2.INTER_LINEAR,
        )
        delta_hedge[hedge_contact] = target_delta_hedge[hedge_contact]
    else:
        delta_hedge = _solve_harmonic_contact_delta_field(
            active_mask=hedge_active,
            dirichlet_mask=hedge_contact,
            dirichlet_values=target_delta_hedge,
        )
    hedge_depth_img = _encode_extrapolated_world_z_u8(
        layer_label=active_cfg.hedge_layer.layer_id,
        world_z_interior=z_free_hedge + delta_hedge,
        active_mask=hedge_active,
        alpha_u8=hedge_alpha_exp,
        z_pos=active_cfg.hedge_layer.z_pos,
        depth_amplitude=active_cfg.hedge_layer.depth_amplitude,
        contact_mask=hedge_contact,
    )
    hedge_depth_img.save(
        scenes_dir / active_cfg.hedge_layer.depth_output_filename,
        format="PNG",
        optimize=True,
    )

    if active_cfg.railing_layer.depth_output_filename is None:
        raise ValueError("railing_layer requires depth_output_filename.")
    rail_active = _build_connected_active_mask(rail_inner_alpha, topo)
    rail_contact, post_count = _detect_railing_contact_mask(rail_inner_alpha, topo)
    z_rail_interior = _solve_harmonic_contact_delta_field(
        active_mask=rail_active,
        dirichlet_mask=rail_contact,
        dirichlet_values=z_world_bg_inner,
    )
    rail_depth_img = _encode_extrapolated_world_z_u8(
        layer_label=active_cfg.railing_layer.layer_id,
        world_z_interior=z_rail_interior,
        active_mask=rail_active,
        alpha_u8=rail_inner_alpha,
        z_pos=active_cfg.railing_layer.z_pos,
        depth_amplitude=active_cfg.railing_layer.depth_amplitude,
        contact_mask=rail_contact,
    )
    rail_depth_img.save(
        scenes_dir / active_cfg.railing_layer.depth_output_filename,
        format="PNG",
        optimize=True,
    )
    print(f"[{active_cfg.scene_id}] Anchored {post_count} green railing posts cleanly to walkway.")

    # Measure grounded fg_z along the foreground brush pile base inside the inner horizontal span
    pile_solid_in = pile_alpha_exp[:, pad_x : pad_x + inner_w] >= topo.solid_contact_alpha_u8
    fg_contact_vals: list[float] = []
    for x_in in range(inner_w):
        pys = np.where(pile_solid_in[:, x_in])[0]
        if pys.size > 0:
            ym_p = int(pys.max())
            sel_p = pys >= (ym_p - topo.fg_ground_contact_band_exp_px)
            fg_contact_vals.extend(z_world_bg_exp[pys[sel_p], pad_x + x_in].tolist())
    calibrated_fg_z = (
        round(float(np.mean(fg_contact_vals)), 2) if fg_contact_vals else active_cfg.fg_z
    )

    # 4. Synthesize Tami Sheaffer & Er'ela Sadan mid-layer anchored at walkway shoe contact
    print(f"[{active_cfg.scene_id}] Placing Tami Sheaffer & Er'ela Sadan by the green railing...")
    sis_rgb, sis_alpha = _extract_sisters_rgba(
        project_root, active_cfg, sam_extractor, matte_extractor
    )
    sis_harm_rgb = _harmonize_rgb_u8(
        sis_rgb,
        gain_rgb=active_cfg.sisters_reveal.shade_gain_rgb,
        bias_rgb=active_cfg.sisters_reveal.shade_bias_rgb,
        luma_gamma=active_cfg.sisters_reveal.luma_gamma,
    )
    sr = active_cfg.sisters_reveal
    mid_rgba_img, sisters_inner_alpha, sis_bbox = _composite_grounded_rgba_on_inner(
        subject_rgb_u8=sis_harm_rgb,
        subject_alpha_u8=sis_alpha,
        tight_thresh_u8=sr.alpha_tight_threshold_u8,
        dest_tight_wh=sr.dest_tight_wh,
        dest_bottom_center_xy=sr.dest_bottom_center_inner_xy,
        inner_wh=(inner_w, inner_h),
        shadow_rgb=sr.contact_shadow_rgb,
        shadow_band_ratio=sr.contact_shadow_footprint_ratio,
        shadow_offset_xy=sr.contact_shadow_offset_xy,
        shadow_sigma_yx=sr.contact_shadow_sigma_yx,
        shadow_opacity=sr.contact_shadow_opacity,
    )
    mid_path = scenes_dir / active_cfg.mid_output_filename
    mid_rgba_img.save(mid_path, format="PNG", optimize=True)

    mid_saved_alpha = np.asarray(mid_rgba_img, dtype=np.uint8)[:, :, 3]
    sis_solid = mid_saved_alpha >= topo.solid_contact_alpha_u8
    sys_idx, _ = np.where(sis_solid)
    shoe_band_mask = sis_solid.copy()
    shoe_band_mask[: int(sys_idx.max()) - topo.sisters_shoe_contact_band_px, :] = False
    calibrated_mid_z = round(float(z_world_bg_inner[shoe_band_mask].mean()), 2)

    # 5. Synthesize clickable C0-anchored + C^-1 EDT installation occluder (IMG_PORT_0062)
    print(f"[{active_cfg.scene_id}] Segmenting & harmonic-anchoring installation occluder (IMG_PORT_0062)...")
    occ = active_cfg.installation_occluder
    inst_rgb, inst_alpha = _extract_installation_cutout(
        project_root, occ, sam_extractor, matte_extractor
    )
    inst_rgba_img, inst_inner_alpha, inst_bbox = _composite_grounded_rgba_on_inner(
        subject_rgb_u8=inst_rgb,
        subject_alpha_u8=inst_alpha,
        tight_thresh_u8=occ.alpha_tight_threshold_u8,
        dest_tight_wh=occ.dest_tight_wh,
        dest_bottom_center_xy=occ.dest_bottom_center_inner_xy,
        inner_wh=(inner_w, inner_h),
        shadow_rgb=occ.contact_shadow_rgb,
        shadow_band_ratio=occ.contact_shadow_band_ratio,
        shadow_offset_xy=occ.contact_shadow_offset_xy,
        shadow_sigma_yx=occ.contact_shadow_sigma_yx,
        shadow_opacity=occ.contact_shadow_opacity,
    )
    inst_path = scenes_dir / occ.output_filename
    inst_rgba_img.save(inst_path, format="PNG", optimize=True)

    inst_np = np.asarray(inst_rgba_img, dtype=np.uint8)
    inst_saved_alpha = inst_np[:, :, 3]
    inst_active = _build_connected_active_mask(inst_saved_alpha, topo)
    inst_contact = _detect_installation_contact_mask(inst_saved_alpha, topo)
    inst_da2_norm = _compute_masked_da2_relief_norm(
        rgb_canvas=inst_np[:, :, :3],
        alpha_u8=inst_inner_alpha,
        depth_estimator=depth_estimator,
        smooth_sigma=occ.depth_smooth_sigma,
        topo_cfg=topo,
    )
    z_free_inst = topo.installation_relief_amplitude_z * (
        inst_da2_norm - float(inst_da2_norm[inst_contact].mean())
    )
    delta_inst = _solve_harmonic_contact_delta_field(
        active_mask=inst_active,
        dirichlet_mask=inst_contact,
        dirichlet_values=z_world_bg_inner - z_free_inst,
    )
    inst_depth_img = _encode_extrapolated_world_z_u8(
        layer_label="installation_occluder",
        world_z_interior=z_free_inst + delta_inst,
        active_mask=inst_active,
        alpha_u8=inst_saved_alpha,
        z_pos=occ.world_z,
        depth_amplitude=occ.depth_amplitude,
        contact_mask=inst_contact,
    )
    inst_depth_path = scenes_dir / occ.depth_output_filename
    inst_depth_img.save(inst_depth_path, format="PNG", optimize=True)

    sis_active = sisters_inner_alpha > 64
    occ_active = inst_inner_alpha > 64
    occ_cov = float((sis_active & occ_active).sum()) / float(max(1, int(sis_active.sum())))
    print(
        f"[{active_cfg.scene_id}] Verified sisters occlusion coverage={occ_cov:.3f} "
        f"(sis_bbox={sis_bbox}, inst_bbox={inst_bbox})"
    )
    if occ_cov < occ.min_sisters_occlusion_coverage:
        raise RuntimeError(
            f"Installation occluder covers only {occ_cov:.3f} of sisters silhouette "
            f"(minimum required {occ.min_sisters_occlusion_coverage:.3f})."
        )

    pad_px = float(sr.portal_hit_pad_inner_px)
    portal_box = (
        max(0.0, sis_bbox[0] - pad_px),
        max(0.0, sis_bbox[1] - pad_px),
        min(float(inner_w), sis_bbox[2] + pad_px),
        min(float(inner_h), sis_bbox[3] + pad_px),
    )
    base_dict = active_cfg.to_dict()
    base_dict["mid_z"] = calibrated_mid_z
    base_dict["fg_z"] = calibrated_fg_z
    base_dict["brush_pile_fg"]["z_pos"] = calibrated_fg_z
    base_dict["portal_hit_box_inner_px"] = list(portal_box)
    base_dict["occluder_hit_box_inner_px"] = list(inst_bbox)
    base_dict["occluder_coverage_ratio"] = occ_cov
    updated_cfg = RailingOccluderSceneConfig.from_dict(base_dict)

    topo_metrics = verify_railing_occluder_depth_topology(updated_cfg, project_root=project_root)
    print(f"[{active_cfg.scene_id}] 2.5D Depth Topology verified: {topo_metrics}")

    manifest_entry = build_railing_occluder_manifest_entry(updated_cfg)
    return updated_cfg, manifest_entry


def update_manifest_scene_key_atomically(
    manifest_path: Path,
    scene_id: str,
    manifest_entry: dict[str, Any],
) -> None:
    """Re-reads `scene_manifest.json` immediately before updating ONLY `scene_id`."""
    if not manifest_path.exists():
        raise RuntimeError(f"Scene manifest does not exist: {manifest_path}")
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_data.get("scenes"), dict):
        raise RuntimeError("scene_manifest.json 'scenes' must be a JSON object.")
    manifest_data["scenes"][scene_id] = manifest_entry
    manifest_path.write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize neural 2.5D layers for fried_yard_installation"
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Atomically update fried_yard_installation inside prototype/assets/scenes/scene_manifest.json",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional scratch directory for pre-erase and post-erase visual verification composites",
    )
    args = parser.parse_args()

    updated_cfg, entry = build_railing_occluder_scene()
    if args.update_manifest:
        manifest_path = _REPO_ROOT / "prototype" / "assets" / "scenes" / "scene_manifest.json"
        update_manifest_scene_key_atomically(manifest_path, updated_cfg.scene_id, entry)
        print(f"[{updated_cfg.scene_id}] Updated scene_manifest.json key '{updated_cfg.scene_id}'.")

    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir = _REPO_ROOT / "prototype" / "assets" / "scenes"
        exp_w, exp_h, pad_x, pad_y, _ = _compute_yard_expanded_mapping(updated_cfg)
        iw, ih = updated_cfg.dimensions.width, updated_cfg.dimensions.height
        bg = Image.open(scenes_dir / updated_cfg.bg_output_filename).convert("RGBA")
        hedge = Image.open(scenes_dir / updated_cfg.hedge_layer.output_filename).convert("RGBA")
        fg = Image.open(scenes_dir / updated_cfg.fg_output_filename).convert("RGBA")
        mid = Image.open(scenes_dir / updated_cfg.mid_output_filename).convert("RGBA")
        rail = Image.open(scenes_dir / updated_cfg.railing_layer.output_filename).convert("RGBA")
        inst = Image.open(
            scenes_dir / updated_cfg.installation_occluder.output_filename
        ).convert("RGBA")

        base_exp = Image.alpha_composite(bg, hedge)
        inner_reveal = base_exp.crop((pad_x, pad_y, pad_x + iw, pad_y + ih))
        inner_reveal = Image.alpha_composite(inner_reveal, mid)
        inner_reveal = Image.alpha_composite(inner_reveal, rail)
        fg_inner = fg.crop((pad_x, pad_y, pad_x + iw, pad_y + ih))
        post_erase = Image.alpha_composite(inner_reveal, fg_inner)
        pre_erase = Image.alpha_composite(
            Image.alpha_composite(inner_reveal, inst),
            fg_inner,
        )
        pre_erase.convert("RGB").save(args.preview_dir / "preview_pre_erase.jpg", quality=92)
        post_erase.convert("RGB").save(args.preview_dir / "preview_post_erase.jpg", quality=92)
        print(f"[{updated_cfg.scene_id}] Saved previews in {args.preview_dir}")


if __name__ == "__main__":
    main()
