#!/usr/bin/env python3
"""
prototype/table_installation_scene_builder.py
Offline neural 2.5D spatial asset builder and typed SSOT configuration models for:
  Scene 11 (`fried_table_hybrid`): "שולחן האוכל בבית פריד בהרצליה — עם ׳היבריד׳ במהופך"
  Isolates Tamar "Tami" Sheaffer's mixed-media installation "היבריד" (`IMG_BINY_0010`, Binyamin
  Gallery 2021), mirrors it horizontally left<->right, and composites it with 3D relief depth,
  warm indoor illumination, crimson tablecloth radiosity bounce, and physical contact/cast shadows
  along the perspective diagonal of the Fried parents' dining table in Herzliya (`IMG_WORD_0200`)
  between Grandfather Michael + grandson (`mid`) and Grandmother Ahuva + grandson (`fg`)
  (Beta Feedback #25, `fb-20261030-233158-3eb7d7`).

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
    from PIL import Image, ImageOps
    from rembg import remove
except ImportError as exc:
    raise RuntimeError(
        "Missing neural vision dependencies inside virtual environment. Run with: "
        "`/Users/talsheaffer/code/tami/.venv/bin/python -u prototype/table_installation_scene_builder.py`."
    ) from exc

from prototype.depth_topology import (
    ContactAnchorBox,
    ContactAnchoredLayerDepthConfig,
    LayerTopologyVerificationConfig,
    LayerTopologyVerificationReport,
    RoomDepthCleanConfig,
    clean_room_depth_under_cutouts,
    compute_perimeter_pin_weights,
    compute_room_relative_neural_protrusion,
    extrapolate_depth_outward_edt,
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
    decontaminate_matte_edge_rgb,
    merge_sam_and_oriented_mattes_u8,
    select_torch_device,
)


# ============================================================================
# Hierarchical, Serializable Configuration Data-Models (SSOT)
# ============================================================================


@dataclass(frozen=True)
class TableDepthContactPinSpec:
    """Smooth elliptical Z-depth pin on the 1500x1000 inner coordinate grid."""

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
    def from_dict(cls, data: dict[str, Any]) -> TableDepthContactPinSpec:
        ixy = data["inner_xy"]
        rxy = data["radii_xy"]
        return cls(
            label=str(data["label"]),
            inner_xy=(float(ixy[0]), float(ixy[1])),
            radii_xy=(float(rxy[0]), float(rxy[1])),
            feather_px=float(data["feather_px"]),
            target_z=float(data["target_z"]),
        )


@dataclass(frozen=True)
class DiningFamilyLayerConfig:
    """Native-resolution SAM + IS-Net segmentation spec for a dining-room person/furniture layer."""

    layer_id: str
    label_he: str
    sam_prompts: tuple[SamPointGroupPrompt, ...]
    sam_exclude_boxes: tuple[BoundingBoxPrompt, ...]
    oriented_matte_boxes: tuple[OrientedMatteBoxSpec, ...]
    support_dilate_px: int
    max_pinhole_fill_px: int
    min_major_component_px: int
    top_clear_y: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "label_he": self.label_he,
            "sam_prompts": [p.to_dict() for p in self.sam_prompts],
            "sam_exclude_boxes": [b.to_dict() for b in self.sam_exclude_boxes],
            "oriented_matte_boxes": [m.to_dict() for m in self.oriented_matte_boxes],
            "support_dilate_px": self.support_dilate_px,
            "max_pinhole_fill_px": self.max_pinhole_fill_px,
            "min_major_component_px": self.min_major_component_px,
            "top_clear_y": self.top_clear_y,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiningFamilyLayerConfig:
        return cls(
            layer_id=str(data["layer_id"]),
            label_he=str(data["label_he"]),
            sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["sam_prompts"]),
            sam_exclude_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data.get("sam_exclude_boxes", ())
            ),
            oriented_matte_boxes=tuple(
                OrientedMatteBoxSpec.from_dict(m) for m in data.get("oriented_matte_boxes", ())
            ),
            support_dilate_px=int(data["support_dilate_px"]),
            max_pinhole_fill_px=int(data["max_pinhole_fill_px"]),
            min_major_component_px=int(data["min_major_component_px"]),
            top_clear_y=int(data["top_clear_y"]),
        )


@dataclass(frozen=True)
class DiningRoomBackgroundConfig:
    """Registration, occlusion healing, and authentic-border mirror reflection config for IMG_WORD_0200."""

    inner_scaled_wh: tuple[int, int]
    inner_x0: int
    template_inner_xyxy: tuple[int, int, int, int]
    scale_search_range_step: tuple[int, int, int]
    table_protect_sam_prompts: tuple[SamPointGroupPrompt, ...]
    table_items_sam_prompts: tuple[SamPointGroupPrompt, ...]
    table_protect_max_pinhole_px: int
    table_protect_min_component_px: int
    table_items_max_pinhole_px: int
    table_items_min_component_px: int
    table_protect_alpha_u8: int
    table_items_alpha_u8: int
    table_wall_min_rgb_exclude: float
    table_hole_fill_max_y_native: int
    cutout_alpha_threshold_u8: int
    emp_table_suppress_boxes_inner: tuple[BoundingBoxPrompt, ...]
    occlusion_dilate_inner_px: int
    lowfreq_harmonize_sigma: float
    occ_lama_dilate_px: int
    occ_lama_blur_sigma: float
    occ_interior_feather_px: float
    occ_table_guard_dist_px: float
    mirror_authentic_borders: bool
    expand_border_touching_cutouts: bool
    border_touch_margin_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "inner_scaled_wh": list(self.inner_scaled_wh),
            "inner_x0": self.inner_x0,
            "template_inner_xyxy": list(self.template_inner_xyxy),
            "scale_search_range_step": list(self.scale_search_range_step),
            "table_protect_sam_prompts": [p.to_dict() for p in self.table_protect_sam_prompts],
            "table_items_sam_prompts": [p.to_dict() for p in self.table_items_sam_prompts],
            "table_protect_max_pinhole_px": self.table_protect_max_pinhole_px,
            "table_protect_min_component_px": self.table_protect_min_component_px,
            "table_items_max_pinhole_px": self.table_items_max_pinhole_px,
            "table_items_min_component_px": self.table_items_min_component_px,
            "table_protect_alpha_u8": self.table_protect_alpha_u8,
            "table_items_alpha_u8": self.table_items_alpha_u8,
            "table_wall_min_rgb_exclude": self.table_wall_min_rgb_exclude,
            "table_hole_fill_max_y_native": self.table_hole_fill_max_y_native,
            "cutout_alpha_threshold_u8": self.cutout_alpha_threshold_u8,
            "emp_table_suppress_boxes_inner": [
                b.to_dict() for b in self.emp_table_suppress_boxes_inner
            ],
            "occlusion_dilate_inner_px": self.occlusion_dilate_inner_px,
            "lowfreq_harmonize_sigma": self.lowfreq_harmonize_sigma,
            "occ_lama_dilate_px": self.occ_lama_dilate_px,
            "occ_lama_blur_sigma": self.occ_lama_blur_sigma,
            "occ_interior_feather_px": self.occ_interior_feather_px,
            "occ_table_guard_dist_px": self.occ_table_guard_dist_px,
            "mirror_authentic_borders": self.mirror_authentic_borders,
            "expand_border_touching_cutouts": self.expand_border_touching_cutouts,
            "border_touch_margin_px": self.border_touch_margin_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiningRoomBackgroundConfig:
        isw = data["inner_scaled_wh"]
        tpl = data["template_inner_xyxy"]
        ssr = data["scale_search_range_step"]
        return cls(
            inner_scaled_wh=(int(isw[0]), int(isw[1])),
            inner_x0=int(data["inner_x0"]),
            template_inner_xyxy=(int(tpl[0]), int(tpl[1]), int(tpl[2]), int(tpl[3])),
            scale_search_range_step=(int(ssr[0]), int(ssr[1]), int(ssr[2])),
            table_protect_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["table_protect_sam_prompts"]
            ),
            table_items_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["table_items_sam_prompts"]
            ),
            table_protect_max_pinhole_px=int(data["table_protect_max_pinhole_px"]),
            table_protect_min_component_px=int(data["table_protect_min_component_px"]),
            table_items_max_pinhole_px=int(data["table_items_max_pinhole_px"]),
            table_items_min_component_px=int(data["table_items_min_component_px"]),
            table_protect_alpha_u8=int(data["table_protect_alpha_u8"]),
            table_items_alpha_u8=int(data["table_items_alpha_u8"]),
            table_wall_min_rgb_exclude=float(data["table_wall_min_rgb_exclude"]),
            table_hole_fill_max_y_native=int(data["table_hole_fill_max_y_native"]),
            cutout_alpha_threshold_u8=int(data["cutout_alpha_threshold_u8"]),
            emp_table_suppress_boxes_inner=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["emp_table_suppress_boxes_inner"]
            ),
            occlusion_dilate_inner_px=int(data["occlusion_dilate_inner_px"]),
            lowfreq_harmonize_sigma=float(data["lowfreq_harmonize_sigma"]),
            occ_lama_dilate_px=int(data["occ_lama_dilate_px"]),
            occ_lama_blur_sigma=float(data["occ_lama_blur_sigma"]),
            occ_interior_feather_px=float(data["occ_interior_feather_px"]),
            occ_table_guard_dist_px=float(data["occ_table_guard_dist_px"]),
            mirror_authentic_borders=bool(data["mirror_authentic_borders"]),
            expand_border_touching_cutouts=bool(data["expand_border_touching_cutouts"]),
            border_touch_margin_px=int(data["border_touch_margin_px"]),
        )


@dataclass(frozen=True)
class HybridInstallationPlacementConfig:
    """Extraction, horizontal mirror, table relighting, shadow, and 3D relief spec for IMG_BINY_0010."""

    layer_id: str
    label_he: str
    catalog_id: str
    source_relpath: str
    mirror_horizontal: bool
    src_crop_xyxy: tuple[int, int, int, int]
    antenna_subcrop_xyxy: tuple[int, int, int, int]
    spheres_subcrop_xyxy: tuple[int, int, int, int]
    baseboard_suppress_crop_xyxy: tuple[int, int, int, int]
    baseboard_sam_keep_threshold: float
    pod_zone_crop_xyxy: tuple[int, int, int, int]
    pod_zone_feather_sigma: float
    sam_prompts: tuple[SamPointGroupPrompt, ...]
    sam_max_pinhole_fill_px: int
    sam_min_major_component_px: int
    thread_local_bg_sigma: float
    thread_luma_delta_floor: float
    thread_boost_gain: float
    thread_support_dilate_px: int
    thread_support_alpha_threshold: float
    thread_roi_boxes: tuple[BoundingBoxPrompt, ...]
    pod_shadow_suppress_box: BoundingBoxPrompt
    pod_shadow_luma_max: float
    pod_shadow_sam_keep_threshold: float
    cutout_alpha_threshold: float
    cutout_min_component_px: int
    cutout_keep_feather_sigma: float
    cutout_keep_gain: float
    tight_alpha_threshold_u8: int
    tight_crop_pad_px: int
    rotation_deg: float
    rotated_alpha_threshold_u8: int
    rotated_crop_pad_px: int
    dest_width_px: int
    dest_left_inner_x: int
    dest_bottom_inner_y: int
    warm_key_gain_rgb: tuple[float, float, float]
    table_contact_v_start: float
    table_contact_v_ramp: float
    table_base_alpha_threshold: float
    table_base_smooth_sigma: float
    table_base_fallback_v_ratio: float
    underside_decay_px: float
    crimson_bounce_rgb: tuple[float, float, float]
    crimson_bounce_strength: float
    footprint_decay_px: float
    footprint_v_start: float
    footprint_v_ramp: float
    cast_shadow_shift_yx: tuple[float, float]
    cast_shadow_sigma_yx: tuple[float, float]
    cast_shadow_gain: float
    cast_shadow_max_alpha: float
    contact_shadow_shift_yx: tuple[float, float]
    contact_shadow_sigma_yx: tuple[float, float]
    contact_shadow_gain: float
    contact_shadow_max_alpha: float
    combined_shadow_max_alpha: float
    shadow_rgb: tuple[float, float, float]
    tabletop_front_crease_p0_xy: tuple[float, float]
    tabletop_front_crease_p1_xy: tuple[float, float]
    tabletop_front_margin_px: float
    tabletop_bounds_xyxy: tuple[float, float, float, float]
    tabletop_mask_feather_sigma: float
    z_pos: float
    depth_amplitude: float
    render_order: int
    segments_x: int
    segments_y: int
    pin_margin_uv: float
    alpha_cutout: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "label_he": self.label_he,
            "catalog_id": self.catalog_id,
            "source_relpath": self.source_relpath,
            "mirror_horizontal": self.mirror_horizontal,
            "src_crop_xyxy": list(self.src_crop_xyxy),
            "antenna_subcrop_xyxy": list(self.antenna_subcrop_xyxy),
            "spheres_subcrop_xyxy": list(self.spheres_subcrop_xyxy),
            "baseboard_suppress_crop_xyxy": list(self.baseboard_suppress_crop_xyxy),
            "baseboard_sam_keep_threshold": self.baseboard_sam_keep_threshold,
            "pod_zone_crop_xyxy": list(self.pod_zone_crop_xyxy),
            "pod_zone_feather_sigma": self.pod_zone_feather_sigma,
            "sam_prompts": [p.to_dict() for p in self.sam_prompts],
            "sam_max_pinhole_fill_px": self.sam_max_pinhole_fill_px,
            "sam_min_major_component_px": self.sam_min_major_component_px,
            "thread_local_bg_sigma": self.thread_local_bg_sigma,
            "thread_luma_delta_floor": self.thread_luma_delta_floor,
            "thread_boost_gain": self.thread_boost_gain,
            "thread_support_dilate_px": self.thread_support_dilate_px,
            "thread_support_alpha_threshold": self.thread_support_alpha_threshold,
            "thread_roi_boxes": [b.to_dict() for b in self.thread_roi_boxes],
            "pod_shadow_suppress_box": self.pod_shadow_suppress_box.to_dict(),
            "pod_shadow_luma_max": self.pod_shadow_luma_max,
            "pod_shadow_sam_keep_threshold": self.pod_shadow_sam_keep_threshold,
            "cutout_alpha_threshold": self.cutout_alpha_threshold,
            "cutout_min_component_px": self.cutout_min_component_px,
            "cutout_keep_feather_sigma": self.cutout_keep_feather_sigma,
            "cutout_keep_gain": self.cutout_keep_gain,
            "tight_alpha_threshold_u8": self.tight_alpha_threshold_u8,
            "tight_crop_pad_px": self.tight_crop_pad_px,
            "rotation_deg": self.rotation_deg,
            "rotated_alpha_threshold_u8": self.rotated_alpha_threshold_u8,
            "rotated_crop_pad_px": self.rotated_crop_pad_px,
            "dest_width_px": self.dest_width_px,
            "dest_left_inner_x": self.dest_left_inner_x,
            "dest_bottom_inner_y": self.dest_bottom_inner_y,
            "warm_key_gain_rgb": list(self.warm_key_gain_rgb),
            "table_contact_v_start": self.table_contact_v_start,
            "table_contact_v_ramp": self.table_contact_v_ramp,
            "table_base_alpha_threshold": self.table_base_alpha_threshold,
            "table_base_smooth_sigma": self.table_base_smooth_sigma,
            "table_base_fallback_v_ratio": self.table_base_fallback_v_ratio,
            "underside_decay_px": self.underside_decay_px,
            "crimson_bounce_rgb": list(self.crimson_bounce_rgb),
            "crimson_bounce_strength": self.crimson_bounce_strength,
            "footprint_decay_px": self.footprint_decay_px,
            "footprint_v_start": self.footprint_v_start,
            "footprint_v_ramp": self.footprint_v_ramp,
            "cast_shadow_shift_yx": list(self.cast_shadow_shift_yx),
            "cast_shadow_sigma_yx": list(self.cast_shadow_sigma_yx),
            "cast_shadow_gain": self.cast_shadow_gain,
            "cast_shadow_max_alpha": self.cast_shadow_max_alpha,
            "contact_shadow_shift_yx": list(self.contact_shadow_shift_yx),
            "contact_shadow_sigma_yx": list(self.contact_shadow_sigma_yx),
            "contact_shadow_gain": self.contact_shadow_gain,
            "contact_shadow_max_alpha": self.contact_shadow_max_alpha,
            "combined_shadow_max_alpha": self.combined_shadow_max_alpha,
            "shadow_rgb": list(self.shadow_rgb),
            "tabletop_front_crease_p0_xy": list(self.tabletop_front_crease_p0_xy),
            "tabletop_front_crease_p1_xy": list(self.tabletop_front_crease_p1_xy),
            "tabletop_front_margin_px": self.tabletop_front_margin_px,
            "tabletop_bounds_xyxy": list(self.tabletop_bounds_xyxy),
            "tabletop_mask_feather_sigma": self.tabletop_mask_feather_sigma,
            "z_pos": self.z_pos,
            "depth_amplitude": self.depth_amplitude,
            "render_order": self.render_order,
            "segments_x": self.segments_x,
            "segments_y": self.segments_y,
            "pin_margin_uv": self.pin_margin_uv,
            "alpha_cutout": self.alpha_cutout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HybridInstallationPlacementConfig:
        sc = data["src_crop_xyxy"]
        ant = data["antenna_subcrop_xyxy"]
        sph = data["spheres_subcrop_xyxy"]
        bbs = data["baseboard_suppress_crop_xyxy"]
        pzc = data["pod_zone_crop_xyxy"]
        wkg = data["warm_key_gain_rgb"]
        cbr = data["crimson_bounce_rgb"]
        ksh = data["cast_shadow_shift_yx"]
        kss = data["cast_shadow_sigma_yx"]
        csh = data["contact_shadow_shift_yx"]
        css = data["contact_shadow_sigma_yx"]
        srgb = data["shadow_rgb"]
        tp0 = data["tabletop_front_crease_p0_xy"]
        tp1 = data["tabletop_front_crease_p1_xy"]
        tbb = data["tabletop_bounds_xyxy"]
        return cls(
            layer_id=str(data["layer_id"]),
            label_he=str(data["label_he"]),
            catalog_id=str(data["catalog_id"]),
            source_relpath=str(data["source_relpath"]),
            mirror_horizontal=bool(data["mirror_horizontal"]),
            src_crop_xyxy=(int(sc[0]), int(sc[1]), int(sc[2]), int(sc[3])),
            antenna_subcrop_xyxy=(int(ant[0]), int(ant[1]), int(ant[2]), int(ant[3])),
            spheres_subcrop_xyxy=(int(sph[0]), int(sph[1]), int(sph[2]), int(sph[3])),
            baseboard_suppress_crop_xyxy=(int(bbs[0]), int(bbs[1]), int(bbs[2]), int(bbs[3])),
            baseboard_sam_keep_threshold=float(data["baseboard_sam_keep_threshold"]),
            pod_zone_crop_xyxy=(int(pzc[0]), int(pzc[1]), int(pzc[2]), int(pzc[3])),
            pod_zone_feather_sigma=float(data["pod_zone_feather_sigma"]),
            sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["sam_prompts"]),
            sam_max_pinhole_fill_px=int(data["sam_max_pinhole_fill_px"]),
            sam_min_major_component_px=int(data["sam_min_major_component_px"]),
            thread_local_bg_sigma=float(data["thread_local_bg_sigma"]),
            thread_luma_delta_floor=float(data["thread_luma_delta_floor"]),
            thread_boost_gain=float(data["thread_boost_gain"]),
            thread_support_dilate_px=int(data["thread_support_dilate_px"]),
            thread_support_alpha_threshold=float(data["thread_support_alpha_threshold"]),
            thread_roi_boxes=tuple(BoundingBoxPrompt.from_dict(b) for b in data["thread_roi_boxes"]),
            pod_shadow_suppress_box=BoundingBoxPrompt.from_dict(data["pod_shadow_suppress_box"]),
            pod_shadow_luma_max=float(data["pod_shadow_luma_max"]),
            pod_shadow_sam_keep_threshold=float(data["pod_shadow_sam_keep_threshold"]),
            cutout_alpha_threshold=float(data["cutout_alpha_threshold"]),
            cutout_min_component_px=int(data["cutout_min_component_px"]),
            cutout_keep_feather_sigma=float(data["cutout_keep_feather_sigma"]),
            cutout_keep_gain=float(data["cutout_keep_gain"]),
            tight_alpha_threshold_u8=int(data["tight_alpha_threshold_u8"]),
            tight_crop_pad_px=int(data["tight_crop_pad_px"]),
            rotation_deg=float(data["rotation_deg"]),
            rotated_alpha_threshold_u8=int(data["rotated_alpha_threshold_u8"]),
            rotated_crop_pad_px=int(data["rotated_crop_pad_px"]),
            dest_width_px=int(data["dest_width_px"]),
            dest_left_inner_x=int(data["dest_left_inner_x"]),
            dest_bottom_inner_y=int(data["dest_bottom_inner_y"]),
            warm_key_gain_rgb=(float(wkg[0]), float(wkg[1]), float(wkg[2])),
            table_contact_v_start=float(data["table_contact_v_start"]),
            table_contact_v_ramp=float(data["table_contact_v_ramp"]),
            table_base_alpha_threshold=float(data["table_base_alpha_threshold"]),
            table_base_smooth_sigma=float(data["table_base_smooth_sigma"]),
            table_base_fallback_v_ratio=float(data["table_base_fallback_v_ratio"]),
            underside_decay_px=float(data["underside_decay_px"]),
            crimson_bounce_rgb=(float(cbr[0]), float(cbr[1]), float(cbr[2])),
            crimson_bounce_strength=float(data["crimson_bounce_strength"]),
            footprint_decay_px=float(data["footprint_decay_px"]),
            footprint_v_start=float(data["footprint_v_start"]),
            footprint_v_ramp=float(data["footprint_v_ramp"]),
            cast_shadow_shift_yx=(float(ksh[0]), float(ksh[1])),
            cast_shadow_sigma_yx=(float(kss[0]), float(kss[1])),
            cast_shadow_gain=float(data["cast_shadow_gain"]),
            cast_shadow_max_alpha=float(data["cast_shadow_max_alpha"]),
            contact_shadow_shift_yx=(float(csh[0]), float(csh[1])),
            contact_shadow_sigma_yx=(float(css[0]), float(css[1])),
            contact_shadow_gain=float(data["contact_shadow_gain"]),
            contact_shadow_max_alpha=float(data["contact_shadow_max_alpha"]),
            combined_shadow_max_alpha=float(data["combined_shadow_max_alpha"]),
            shadow_rgb=(float(srgb[0]), float(srgb[1]), float(srgb[2])),
            tabletop_front_crease_p0_xy=(float(tp0[0]), float(tp0[1])),
            tabletop_front_crease_p1_xy=(float(tp1[0]), float(tp1[1])),
            tabletop_front_margin_px=float(data["tabletop_front_margin_px"]),
            tabletop_bounds_xyxy=(float(tbb[0]), float(tbb[1]), float(tbb[2]), float(tbb[3])),
            tabletop_mask_feather_sigma=float(data["tabletop_mask_feather_sigma"]),
            z_pos=float(data["z_pos"]),
            depth_amplitude=float(data["depth_amplitude"]),
            render_order=int(data["render_order"]),
            segments_x=int(data["segments_x"]),
            segments_y=int(data["segments_y"]),
            pin_margin_uv=float(data["pin_margin_uv"]),
            alpha_cutout=float(data["alpha_cutout"]),
        )


@dataclass(frozen=True)
class TableInstallationDepthCalibrationConfig:
    """Ghost-free dining-room depth mapping and C^0/C^-1 contact-anchored layer calibration."""

    scene_bg_z: float
    room_depth_amplitude: float
    room_pin_margin_uv: float
    da2_gaussian_sigma: float
    da2_z_min: float
    da2_z_max: float
    contact_pins: tuple[TableDepthContactPinSpec, ...]
    final_smooth_sigma: float
    room_clean: RoomDepthCleanConfig
    protrusion_clean: RoomDepthCleanConfig
    protrusion_upper_percentile: float
    mid_topology: ContactAnchoredLayerDepthConfig
    installation_topology: ContactAnchoredLayerDepthConfig
    fg_topology: ContactAnchoredLayerDepthConfig
    front_chair_topology: ContactAnchoredLayerDepthConfig
    layer_verification: LayerTopologyVerificationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_bg_z": self.scene_bg_z,
            "room_depth_amplitude": self.room_depth_amplitude,
            "room_pin_margin_uv": self.room_pin_margin_uv,
            "da2_gaussian_sigma": self.da2_gaussian_sigma,
            "da2_z_min": self.da2_z_min,
            "da2_z_max": self.da2_z_max,
            "contact_pins": [p.to_dict() for p in self.contact_pins],
            "final_smooth_sigma": self.final_smooth_sigma,
            "room_clean": self.room_clean.to_dict(),
            "protrusion_clean": self.protrusion_clean.to_dict(),
            "protrusion_upper_percentile": self.protrusion_upper_percentile,
            "mid_topology": self.mid_topology.to_dict(),
            "installation_topology": self.installation_topology.to_dict(),
            "fg_topology": self.fg_topology.to_dict(),
            "front_chair_topology": self.front_chair_topology.to_dict(),
            "layer_verification": self.layer_verification.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableInstallationDepthCalibrationConfig:
        return cls(
            scene_bg_z=float(data["scene_bg_z"]),
            room_depth_amplitude=float(data["room_depth_amplitude"]),
            room_pin_margin_uv=float(data["room_pin_margin_uv"]),
            da2_gaussian_sigma=float(data["da2_gaussian_sigma"]),
            da2_z_min=float(data["da2_z_min"]),
            da2_z_max=float(data["da2_z_max"]),
            contact_pins=tuple(
                TableDepthContactPinSpec.from_dict(p) for p in data["contact_pins"]
            ),
            final_smooth_sigma=float(data["final_smooth_sigma"]),
            room_clean=RoomDepthCleanConfig.from_dict(data["room_clean"]),
            protrusion_clean=RoomDepthCleanConfig.from_dict(data["protrusion_clean"]),
            protrusion_upper_percentile=float(data["protrusion_upper_percentile"]),
            mid_topology=ContactAnchoredLayerDepthConfig.from_dict(data["mid_topology"]),
            installation_topology=ContactAnchoredLayerDepthConfig.from_dict(
                data["installation_topology"]
            ),
            fg_topology=ContactAnchoredLayerDepthConfig.from_dict(data["fg_topology"]),
            front_chair_topology=ContactAnchoredLayerDepthConfig.from_dict(
                data["front_chair_topology"]
            ),
            layer_verification=LayerTopologyVerificationConfig.from_dict(
                data["layer_verification"]
            ),
        )


@dataclass(frozen=True)
class FriedTableHybridSceneConfig:
    """Canonical SSOT configuration for Scene 11 (`fried_table_hybrid`)."""

    scene_id: str
    exhibition_group: str
    title_he: str
    short_label_he: str
    icon: str
    dining_catalog_id: str
    dining_source_relpath: str
    outpaint_raw_relpath: str
    lama_inner_output_filename: str
    bg_output_filename: str
    depth_output_filename: str
    mid_output_filename: str
    mid_depth_output_filename: str
    fg_output_filename: str
    fg_depth_output_filename: str
    installation_output_filename: str
    installation_depth_output_filename: str
    front_chair_output_filename: str
    front_chair_depth_output_filename: str
    dimensions: CanvasDimensionsConfig
    outpainting: OutpaintingConfig
    depth_model: DepthModelConfig
    segmentation: SegmentationConfig
    inpainting: InpaintingConfig
    min_registration_corr: float
    bore_sight_at_rest: bool
    scene_bg_z: float
    mid_z: float
    fg_z: float
    front_chair_z: float
    mid_render_order: int
    fg_render_order: int
    front_chair_render_order: int
    bg_segments_x: int
    bg_segments_y: int
    mid_segments_x: int
    mid_segments_y: int
    fg_segments_x: int
    fg_segments_y: int
    front_chair_segments_x: int
    front_chair_segments_y: int
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
    room_background: DiningRoomBackgroundConfig
    michael_mid: DiningFamilyLayerConfig
    ahuva_fg: DiningFamilyLayerConfig
    front_chair: DiningFamilyLayerConfig
    hybrid_installation: HybridInstallationPlacementConfig
    depth_calibration: TableInstallationDepthCalibrationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "exhibition_group": self.exhibition_group,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "dining_catalog_id": self.dining_catalog_id,
            "dining_source_relpath": self.dining_source_relpath,
            "outpaint_raw_relpath": self.outpaint_raw_relpath,
            "lama_inner_output_filename": self.lama_inner_output_filename,
            "bg_output_filename": self.bg_output_filename,
            "depth_output_filename": self.depth_output_filename,
            "mid_output_filename": self.mid_output_filename,
            "mid_depth_output_filename": self.mid_depth_output_filename,
            "fg_output_filename": self.fg_output_filename,
            "fg_depth_output_filename": self.fg_depth_output_filename,
            "installation_output_filename": self.installation_output_filename,
            "installation_depth_output_filename": self.installation_depth_output_filename,
            "front_chair_output_filename": self.front_chair_output_filename,
            "front_chair_depth_output_filename": self.front_chair_depth_output_filename,
            "dimensions": self.dimensions.to_dict(),
            "outpainting": self.outpainting.to_dict(),
            "depth_model": self.depth_model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "inpainting": self.inpainting.to_dict(),
            "min_registration_corr": self.min_registration_corr,
            "bore_sight_at_rest": self.bore_sight_at_rest,
            "scene_bg_z": self.scene_bg_z,
            "mid_z": self.mid_z,
            "fg_z": self.fg_z,
            "front_chair_z": self.front_chair_z,
            "mid_render_order": self.mid_render_order,
            "fg_render_order": self.fg_render_order,
            "front_chair_render_order": self.front_chair_render_order,
            "bg_segments_x": self.bg_segments_x,
            "bg_segments_y": self.bg_segments_y,
            "mid_segments_x": self.mid_segments_x,
            "mid_segments_y": self.mid_segments_y,
            "fg_segments_x": self.fg_segments_x,
            "fg_segments_y": self.fg_segments_y,
            "front_chair_segments_x": self.front_chair_segments_x,
            "front_chair_segments_y": self.front_chair_segments_y,
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
            "room_background": self.room_background.to_dict(),
            "michael_mid": self.michael_mid.to_dict(),
            "ahuva_fg": self.ahuva_fg.to_dict(),
            "front_chair": self.front_chair.to_dict(),
            "hybrid_installation": self.hybrid_installation.to_dict(),
            "depth_calibration": self.depth_calibration.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FriedTableHybridSceneConfig:
        return cls(
            scene_id=str(data["scene_id"]),
            exhibition_group=str(data["exhibition_group"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            dining_catalog_id=str(data["dining_catalog_id"]),
            dining_source_relpath=str(data["dining_source_relpath"]),
            outpaint_raw_relpath=str(data["outpaint_raw_relpath"]),
            lama_inner_output_filename=str(data["lama_inner_output_filename"]),
            bg_output_filename=str(data["bg_output_filename"]),
            depth_output_filename=str(data["depth_output_filename"]),
            mid_output_filename=str(data["mid_output_filename"]),
            mid_depth_output_filename=str(data["mid_depth_output_filename"]),
            fg_output_filename=str(data["fg_output_filename"]),
            fg_depth_output_filename=str(data["fg_depth_output_filename"]),
            installation_output_filename=str(data["installation_output_filename"]),
            installation_depth_output_filename=str(data["installation_depth_output_filename"]),
            front_chair_output_filename=str(data["front_chair_output_filename"]),
            front_chair_depth_output_filename=str(data["front_chair_depth_output_filename"]),
            dimensions=CanvasDimensionsConfig.from_dict(data["dimensions"]),
            outpainting=OutpaintingConfig.from_dict(data["outpainting"]),
            depth_model=DepthModelConfig.from_dict(data["depth_model"]),
            segmentation=SegmentationConfig.from_dict(data["segmentation"]),
            inpainting=InpaintingConfig.from_dict(data["inpainting"]),
            min_registration_corr=float(data["min_registration_corr"]),
            bore_sight_at_rest=bool(data["bore_sight_at_rest"]),
            scene_bg_z=float(data["scene_bg_z"]),
            mid_z=float(data["mid_z"]),
            fg_z=float(data["fg_z"]),
            front_chair_z=float(data["front_chair_z"]),
            mid_render_order=int(data["mid_render_order"]),
            fg_render_order=int(data["fg_render_order"]),
            front_chair_render_order=int(data["front_chair_render_order"]),
            bg_segments_x=int(data["bg_segments_x"]),
            bg_segments_y=int(data["bg_segments_y"]),
            mid_segments_x=int(data["mid_segments_x"]),
            mid_segments_y=int(data["mid_segments_y"]),
            fg_segments_x=int(data["fg_segments_x"]),
            fg_segments_y=int(data["fg_segments_y"]),
            front_chair_segments_x=int(data["front_chair_segments_x"]),
            front_chair_segments_y=int(data["front_chair_segments_y"]),
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
            room_background=DiningRoomBackgroundConfig.from_dict(data["room_background"]),
            michael_mid=DiningFamilyLayerConfig.from_dict(data["michael_mid"]),
            ahuva_fg=DiningFamilyLayerConfig.from_dict(data["ahuva_fg"]),
            front_chair=DiningFamilyLayerConfig.from_dict(data["front_chair"]),
            hybrid_installation=HybridInstallationPlacementConfig.from_dict(
                data["hybrid_installation"]
            ),
            depth_calibration=TableInstallationDepthCalibrationConfig.from_dict(
                data["depth_calibration"]
            ),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, raw_json: str) -> FriedTableHybridSceneConfig:
        return cls.from_dict(json.loads(raw_json))


def default_fried_table_hybrid_scene_config() -> FriedTableHybridSceneConfig:
    """Returns canonical SSOT configuration for Scene 11 (`fried_table_hybrid`)."""
    cfg = FriedTableHybridSceneConfig(
        scene_id="fried_table_hybrid",
        exhibition_group="family_places",
        title_he="שולחן האוכל בבית פריד בהרצליה — עם ׳היבריד׳ במהופך (2006 / 2021)",
        short_label_he="שולחן בית פריד: היבריד",
        icon="🍽️",
        dining_catalog_id="IMG_WORD_0200",
        dining_source_relpath="media/wordpress_blog/wp_wide_img_0293_e24e32b8.jpg",
        outpaint_raw_relpath="prototype/assets/outpaint_sources/scene11_fried_table_hybrid_outpaint_raw.jpg",
        lama_inner_output_filename="scene11_fried_table_hybrid_lama_inner.jpg",
        bg_output_filename="scene11_fried_table_hybrid_bg_inpainted.jpg",
        depth_output_filename="scene11_fried_table_hybrid_depth.png",
        mid_output_filename="scene11_fried_table_hybrid_mid.png",
        mid_depth_output_filename="scene11_fried_table_hybrid_mid_depth.png",
        fg_output_filename="scene11_fried_table_hybrid_fg.png",
        fg_depth_output_filename="scene11_fried_table_hybrid_fg_depth.png",
        installation_output_filename="scene11_fried_table_hybrid_installation.png",
        installation_depth_output_filename="scene11_fried_table_hybrid_installation_depth.png",
        front_chair_output_filename="scene11_fried_table_hybrid_front_chair.png",
        front_chair_depth_output_filename="scene11_fried_table_hybrid_front_chair_depth.png",
        dimensions=CanvasDimensionsConfig(width=1500, height=1000),
        outpainting=OutpaintingConfig(
            expansion_factor=1.80,
            feather_collar_px=24,
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
            dilation_radius_px=14,
            mask_blur_sigma=2.5,
            jpeg_quality=92,
        ),
        min_registration_corr=0.88,
        bore_sight_at_rest=True,
        scene_bg_z=-320.0,
        mid_z=-298.0,
        fg_z=-320.0,
        front_chair_z=-320.0,
        mid_render_order=2,
        fg_render_order=4,
        front_chair_render_order=5,
        bg_segments_x=288,
        bg_segments_y=192,
        mid_segments_x=224,
        mid_segments_y=150,
        fg_segments_x=320,
        fg_segments_y=214,
        front_chair_segments_x=320,
        front_chair_segments_y=214,
        initial_camera_y=0.0,
        initial_camera_z=680.0,
        min_camera_z=185.0,
        max_camera_z=960.0,
        max_yaw_deg=32.0,
        max_pitch_deg=22.0,
        max_strafe_x=240.0,
        max_strafe_y=150.0,
        dolly_speed_mult=1.30,
        particle_color_hex=0xEFE4CF,
        particle_count=60,
        room_background=DiningRoomBackgroundConfig(
            inner_scaled_wh=(1334, 1000),
            inner_x0=83,
            template_inner_xyxy=(200, 40, 1380, 282),
            scale_search_range_step=(1460, 1640, 6),
            table_protect_sam_prompts=(
                # Front-left wooden table leg
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=238, y0=880, x1=310, y1=1136),
                    points_xy=(
                        (268.0, 920.0),
                        (272.0, 980.0),
                        (276.0, 1045.0),
                        (282.0, 1105.0),
                        (230.0, 1000.0),
                        (316.0, 1000.0),
                        (282.0, 1144.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0, 0),
                ),
                # Front-right wooden table leg
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=720, y0=990, x1=792, y1=1200),
                    points_xy=(
                        (750.0, 1020.0),
                        (750.0, 1085.0),
                        (750.0, 1150.0),
                        (750.0, 1190.0),
                        (710.0, 1085.0),
                        (798.0, 1085.0),
                        (710.0, 1160.0),
                        (800.0, 1160.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0, 0, 0),
                ),
                # Full red tablecloth + green paisley runner + front crimson skirt & wooden apron rail
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=194, y0=632, x1=1024, y1=1082),
                    points_xy=(
                        (550.0, 675.0),
                        (650.0, 665.0),
                        (700.0, 725.0),
                        (825.0, 745.0),
                        (500.0, 815.0),
                        (350.0, 895.0),
                        (550.0, 915.0),
                        (725.0, 955.0),
                        (778.0, 1035.0),
                        (910.0, 785.0),
                        (955.0, 705.0),
                        (218.0, 925.0),
                        (235.0, 865.0),
                        (345.0, 760.0),
                        (360.0, 935.0),
                        (560.0, 975.0),
                        (650.0, 625.0),
                        (850.0, 645.0),
                        (500.0, 1030.0),
                        (650.0, 1060.0),
                        (182.0, 880.0),
                        (1035.0, 720.0),
                    ),
                    point_labels=(
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0,
                    ),
                ),
            ),
            table_items_sam_prompts=(
                # White hardcover book on green paisley runner / red tablecloth
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=616, y0=754, x1=788, y1=848),
                    points_xy=(
                        (665.0, 804.0),
                        (715.0, 810.0),
                        (752.0, 818.0),
                        (695.0, 784.0),
                        (608.0, 825.0),
                        (786.0, 845.0),
                        (700.0, 748.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0, 0),
                ),
                # Yellow/white document folder on upper-right tabletop
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=760, y0=668, x1=936, y1=746),
                    points_xy=(
                        (815.0, 702.0),
                        (865.0, 706.0),
                        (902.0, 712.0),
                        (845.0, 692.0),
                        (752.0, 705.0),
                        (850.0, 662.0),
                        (850.0, 752.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0, 0),
                ),
            ),
            table_protect_max_pinhole_px=35000,
            table_protect_min_component_px=400,
            table_items_max_pinhole_px=8000,
            table_items_min_component_px=200,
            table_protect_alpha_u8=96,
            table_items_alpha_u8=64,
            table_wall_min_rgb_exclude=180.0,
            table_hole_fill_max_y_native=1090,
            cutout_alpha_threshold_u8=8,
            emp_table_suppress_boxes_inner=(
                BoundingBoxPrompt(x0=205, y0=678, x1=254, y1=786),
                BoundingBoxPrompt(x0=225, y0=785, x1=860, y1=1000),
                BoundingBoxPrompt(x0=815, y0=535, x1=985, y1=855),
            ),
            occlusion_dilate_inner_px=15,
            lowfreq_harmonize_sigma=48.0,
            occ_lama_dilate_px=10,
            occ_lama_blur_sigma=2.0,
            occ_interior_feather_px=4.0,
            occ_table_guard_dist_px=4.0,
            mirror_authentic_borders=True,
            expand_border_touching_cutouts=True,
            border_touch_margin_px=2,
        ),
        michael_mid=DiningFamilyLayerConfig(
            layer_id="michael_mid",
            label_he="סבא מיכאל פריד והנכד מצד שמאל של השולחן",
            sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=250, y0=356, x1=515, y1=685),
                    points_xy=(
                        (395.0, 368.0),
                        (395.0, 425.0),
                        (325.0, 475.0),
                        (465.0, 478.0),
                        (295.0, 560.0),
                        (500.0, 585.0),
                        (395.0, 348.0),
                        (335.0, 348.0),
                        (455.0, 348.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 0, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=190, y0=490, x1=575, y1=875),
                    points_xy=(
                        (405.0, 545.0),
                        (350.0, 665.0),
                        (270.0, 665.0),
                        (425.0, 690.0),
                        (485.0, 675.0),
                        (535.0, 680.0),
                        (505.0, 618.0),
                        (218.0, 745.0),
                        (232.0, 818.0),
                        (355.0, 795.0),
                        (475.0, 775.0),
                        (545.0, 765.0),
                        (175.0, 680.0),
                        (535.0, 560.0),
                        (265.0, 815.0),
                        (205.0, 860.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
                ),
            ),
            sam_exclude_boxes=(
                BoundingBoxPrompt(x0=0, y0=0, x1=1600, y1=356),
                BoundingBoxPrompt(x0=0, y0=0, x1=186, y1=1200),
                BoundingBoxPrompt(x0=578, y0=0, x1=1600, y1=1200),
                BoundingBoxPrompt(x0=0, y0=872, x1=1600, y1=1200),
            ),
            oriented_matte_boxes=(
                OrientedMatteBoxSpec(
                    box=BoundingBoxPrompt(x0=340, y0=356, x1=468, y1=468),
                    upright_angle_deg=0.0,
                    taper_margin_px=14,
                    sam_core_erode_px=4,
                    exclude_boxes=(BoundingBoxPrompt(x0=0, y0=0, x1=1600, y1=356),),
                ),
                OrientedMatteBoxSpec(
                    box=BoundingBoxPrompt(x0=340, y0=485, x1=475, y1=635),
                    upright_angle_deg=0.0,
                    taper_margin_px=16,
                    sam_core_erode_px=4,
                ),
            ),
            support_dilate_px=30,
            max_pinhole_fill_px=8000,
            min_major_component_px=5000,
            top_clear_y=356,
        ),
        ahuva_fg=DiningFamilyLayerConfig(
            layer_id="ahuva_fg",
            label_he="סבתא אהובה פריד והנכד מצד ימין של השולחן",
            sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=765, y0=545, x1=1265, y1=1200),
                    points_xy=(
                        (1165.0, 615.0),
                        (1120.0, 725.0),
                        (1195.0, 760.0),
                        (1055.0, 765.0),
                        (1035.0, 800.0),
                        (1015.0, 835.0),
                        (1080.0, 865.0),
                        (955.0, 960.0),
                        (855.0, 1085.0),
                        (815.0, 1184.0),
                        (860.0, 1184.0),
                        (1035.0, 1140.0),
                        (960.0, 705.0),
                        (1035.0, 680.0),
                        (860.0, 760.0),
                        (815.0, 920.0),
                        (755.0, 1120.0),
                        (992.0, 760.0),
                        (988.0, 795.0),
                        (1012.0, 755.0),
                        (1320.0, 950.0),
                    ),
                    point_labels=(
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=1045, y0=460, x1=1325, y1=1200),
                    points_xy=(
                        (1120.0, 515.0),
                        (1120.0, 585.0),
                        (1265.0, 695.0),
                        (1285.0, 755.0),
                        (1255.0, 880.0),
                        (1255.0, 1045.0),
                        (1235.0, 1175.0),
                        (1220.0, 495.0),
                        (1320.0, 620.0),
                        (1315.0, 950.0),
                        (1165.0, 1165.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=1075, y0=755, x1=1265, y1=1200),
                    points_xy=(
                        (1215.0, 825.0),
                        (1185.0, 955.0),
                        (1148.0, 1075.0),
                        (1155.0, 1155.0),
                        (1265.0, 950.0),
                        (1080.0, 1180.0),
                        (992.0, 760.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0, 0),
                ),
            ),
            sam_exclude_boxes=(
                BoundingBoxPrompt(x0=0, y0=0, x1=765, y1=1200),
                BoundingBoxPrompt(x0=0, y0=0, x1=1600, y1=455),
                BoundingBoxPrompt(x0=1326, y0=0, x1=1600, y1=1200),
            ),
            oriented_matte_boxes=(
                OrientedMatteBoxSpec(
                    box=BoundingBoxPrompt(x0=1045, y0=455, x1=1195, y1=635),
                    upright_angle_deg=0.0,
                    taper_margin_px=16,
                    sam_core_erode_px=4,
                ),
                OrientedMatteBoxSpec(
                    box=BoundingBoxPrompt(x0=1095, y0=545, x1=1265, y1=715),
                    upright_angle_deg=0.0,
                    taper_margin_px=16,
                    sam_core_erode_px=4,
                ),
            ),
            support_dilate_px=32,
            max_pinhole_fill_px=1500,
            min_major_component_px=4000,
            top_clear_y=455,
        ),
        front_chair=DiningFamilyLayerConfig(
            layer_id="front_chair",
            label_he="כיסא העץ הקדמי משמאל לשולחן האוכל",
            sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=42, y0=668, x1=575, y1=1200),
                    points_xy=(
                        (75.0, 745.0),
                        (118.0, 825.0),
                        (155.0, 960.0),
                        (212.0, 1085.0),
                        (315.0, 1085.0),
                        (155.0, 1155.0),
                        (350.0, 1145.0),
                        (485.0, 1165.0),
                        (275.0, 1150.0),
                        (260.0, 820.0),
                        (340.0, 920.0),
                        (272.0, 980.0),
                        (85.0, 1155.0),
                        (190.0, 920.0),
                        (465.0, 1050.0),
                        (355.0, 1050.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
                ),
            ),
            sam_exclude_boxes=(
                BoundingBoxPrompt(x0=0, y0=0, x1=1600, y1=666),
                BoundingBoxPrompt(x0=578, y0=0, x1=1600, y1=1200),
                BoundingBoxPrompt(x0=255, y0=666, x1=578, y1=1058),
                BoundingBoxPrompt(x0=0, y0=1020, x1=122, y1=1200),
            ),
            oriented_matte_boxes=(),
            support_dilate_px=0,
            max_pinhole_fill_px=700,
            min_major_component_px=4500,
            top_clear_y=666,
        ),
        hybrid_installation=HybridInstallationPlacementConfig(
            layer_id="table_installation",
            label_he="מיצב ׳היבריד׳ (2021) במהופך לאורך שולחן האוכל בבית פריד",
            catalog_id="IMG_BINY_0010",
            source_relpath="media/binyamin_gallery/binyamin_היבריד.jpg_396684e5.jpg",
            mirror_horizontal=True,
            src_crop_xyxy=(290, 220, 1745, 1345),
            antenna_subcrop_xyxy=(0, 0, 640, 500),
            spheres_subcrop_xyxy=(580, 470, 1455, 1125),
            baseboard_suppress_crop_xyxy=(725, 495, 805, 595),
            baseboard_sam_keep_threshold=96.0,
            pod_zone_crop_xyxy=(285, 410, 775, 675),
            pod_zone_feather_sigma=8.0,
            sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=275, y0=405, x1=785, y1=690),
                    points_xy=(
                        (480.0, 520.0),
                        (360.0, 560.0),
                        (620.0, 550.0),
                        (710.0, 595.0),
                        (530.0, 635.0),
                        (295.0, 430.0),
                        (440.0, 675.0),
                        (755.0, 535.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 0, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=15, y0=20, x1=610, y1=480),
                    points_xy=(
                        (325.0, 162.0),
                        (325.0, 300.0),
                        (325.0, 412.0),
                        (170.0, 165.0),
                        (480.0, 162.0),
                        (235.0, 120.0),
                        (455.0, 115.0),
                        (325.0, 60.0),
                        (250.0, 320.0),
                        (400.0, 320.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
                ),
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=565, y0=490, x1=1425, y1=1100),
                    points_xy=(
                        (855.0, 555.0),
                        (655.0, 665.0),
                        (705.0, 745.0),
                        (865.0, 815.0),
                        (1005.0, 720.0),
                        (1175.0, 760.0),
                        (1255.0, 915.0),
                        (1125.0, 1025.0),
                        (1335.0, 1025.0),
                        (620.0, 890.0),
                        (920.0, 995.0),
                        (1240.0, 620.0),
                        (755.0, 548.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
                ),
            ),
            sam_max_pinhole_fill_px=2500,
            sam_min_major_component_px=400,
            thread_local_bg_sigma=6.0,
            thread_luma_delta_floor=14.0,
            thread_boost_gain=9.0,
            thread_support_dilate_px=95,
            thread_support_alpha_threshold=64.0,
            thread_roi_boxes=(
                BoundingBoxPrompt(x0=620, y0=575, x1=1440, y1=1110),
                BoundingBoxPrompt(x0=815, y0=495, x1=960, y1=575),
            ),
            pod_shadow_suppress_box=BoundingBoxPrompt(x0=280, y0=575, x1=695, y1=720),
            pod_shadow_luma_max=85.0,
            pod_shadow_sam_keep_threshold=64.0,
            cutout_alpha_threshold=36.0,
            cutout_min_component_px=120,
            cutout_keep_feather_sigma=0.8,
            cutout_keep_gain=1.4,
            tight_alpha_threshold_u8=20,
            tight_crop_pad_px=8,
            rotation_deg=-3.2,
            rotated_alpha_threshold_u8=16,
            rotated_crop_pad_px=4,
            dest_width_px=428,
            dest_left_inner_x=452,
            dest_bottom_inner_y=740,
            warm_key_gain_rgb=(1.035, 0.992, 0.915),
            table_contact_v_start=0.38,
            table_contact_v_ramp=0.10,
            table_base_alpha_threshold=0.22,
            table_base_smooth_sigma=6.0,
            table_base_fallback_v_ratio=0.62,
            underside_decay_px=18.0,
            crimson_bounce_rgb=(176.0, 56.0, 62.0),
            crimson_bounce_strength=0.16,
            footprint_decay_px=14.0,
            footprint_v_start=0.40,
            footprint_v_ramp=0.08,
            cast_shadow_shift_yx=(11.0, 9.0),
            cast_shadow_sigma_yx=(10.5, 16.0),
            cast_shadow_gain=0.46,
            cast_shadow_max_alpha=0.36,
            contact_shadow_shift_yx=(3.0, 2.0),
            contact_shadow_sigma_yx=(2.8, 4.8),
            contact_shadow_gain=0.92,
            contact_shadow_max_alpha=0.68,
            combined_shadow_max_alpha=0.72,
            shadow_rgb=(28.0, 7.0, 11.0),
            tabletop_front_crease_p0_xy=(263.0, 721.0),
            tabletop_front_crease_p1_xy=(758.0, 823.0),
            tabletop_front_margin_px=8.0,
            tabletop_bounds_xyxy=(280.0, 530.0, 935.0, 835.0),
            tabletop_mask_feather_sigma=4.0,
            z_pos=-284.0,
            depth_amplitude=125.0,
            render_order=3,
            segments_x=224,
            segments_y=150,
            pin_margin_uv=0.0,
            alpha_cutout=0.0,
        ),
        depth_calibration=TableInstallationDepthCalibrationConfig(
            scene_bg_z=-320.0,
            room_depth_amplitude=220.0,
            room_pin_margin_uv=0.045,
            da2_gaussian_sigma=10.0,
            da2_z_min=-314.0,
            da2_z_max=-195.0,
            contact_pins=(),
            final_smooth_sigma=0.0,
            room_clean=RoomDepthCleanConfig(
                alpha_threshold_u8=20,
                mask_dilation_px=14,
                extra_clean_boxes_expanded=(),
                preserve_boxes_expanded=((880, 915, 1495, 1255),),
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
            mid_topology=ContactAnchoredLayerDepthConfig(
                bg_z=-320.0,
                room_depth_amplitude=220.0,
                room_pin_margin_uv=0.045,
                layer_z=-298.0,
                layer_depth_amplitude=125.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=16,
                min_component_area_px=1,
                auto_floor_band_px=10,
                auto_floor_max_rise_px=48,
                contact_boxes=(
                    ContactAnchorBox(
                        x0=250, y0=575, x1=560, y1=730, mode="bottom_band", band_px=10
                    ),
                ),
                neural_relief_amplitude=8.0,
                contact_relief_taper_px=8.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.0,
                interior_smooth_sigma_px=2.0,
                overflow_tolerance_world=0.5,
            ),
            installation_topology=ContactAnchoredLayerDepthConfig(
                bg_z=-320.0,
                room_depth_amplitude=220.0,
                room_pin_margin_uv=0.045,
                layer_z=-284.0,
                layer_depth_amplitude=125.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=28,
                min_component_area_px=1,
                auto_floor_band_px=10,
                auto_floor_max_rise_px=48,
                contact_boxes=(
                    ContactAnchorBox(
                        x0=450, y0=560, x1=838, y1=750, mode="bottom_band", band_px=10
                    ),
                ),
                neural_relief_amplitude=14.0,
                contact_relief_taper_px=8.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.0,
                interior_smooth_sigma_px=1.8,
                overflow_tolerance_world=0.5,
            ),
            fg_topology=ContactAnchoredLayerDepthConfig(
                bg_z=-320.0,
                room_depth_amplitude=220.0,
                room_pin_margin_uv=0.045,
                layer_z=-320.0,
                layer_depth_amplitude=175.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=16,
                min_component_area_px=1,
                auto_floor_band_px=12,
                auto_floor_max_rise_px=48,
                contact_boxes=(
                    ContactAnchorBox(
                        x0=720, y0=955, x1=1165, y1=1000, mode="bottom_band", band_px=12
                    ),
                ),
                neural_relief_amplitude=10.0,
                contact_relief_taper_px=6.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.0,
                interior_smooth_sigma_px=2.0,
                overflow_tolerance_world=0.5,
            ),
            front_chair_topology=ContactAnchoredLayerDepthConfig(
                bg_z=-320.0,
                room_depth_amplitude=220.0,
                room_pin_margin_uv=0.045,
                layer_z=-320.0,
                layer_depth_amplitude=175.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=16,
                min_component_area_px=1,
                auto_floor_band_px=12,
                auto_floor_max_rise_px=48,
                contact_boxes=(
                    ContactAnchorBox(
                        x0=170, y0=975, x1=558, y1=1000, mode="bottom_band", band_px=12
                    ),
                ),
                neural_relief_amplitude=8.0,
                contact_relief_taper_px=4.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.0,
                interior_smooth_sigma_px=2.0,
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
    round_trip = FriedTableHybridSceneConfig.from_json(cfg.to_json())
    if round_trip != cfg:
        raise RuntimeError("FriedTableHybridSceneConfig failed lossless JSON round-trip check.")
    return cfg


# ============================================================================
# Modular Vision & Spatial Layer Synthesis Primitives
# ============================================================================


def _segment_dining_layer_native(
    word_rgb: Image.Image,
    layer_cfg: DiningFamilyLayerConfig,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
) -> np.ndarray:
    """Segments a 1600x1200 person/furniture layer via promptable SAM + optional IS-Net matting."""
    alpha_u8 = sam_extractor.segment_point_groups(
        rgb_image=word_rgb,
        point_groups=layer_cfg.sam_prompts,
        exclude_boxes=layer_cfg.sam_exclude_boxes,
        max_pinhole_fill_px=layer_cfg.max_pinhole_fill_px,
        min_major_component_px=layer_cfg.min_major_component_px,
    )
    if layer_cfg.oriented_matte_boxes:
        alpha_u8 = merge_sam_and_oriented_mattes_u8(
            rgb_image=word_rgb,
            sam_u8=alpha_u8,
            oriented_boxes=layer_cfg.oriented_matte_boxes,
            support_dilate_px=layer_cfg.support_dilate_px,
            right_edge_fade_x_start=None,
            rembg_session=matte_extractor.session,
        )
    if layer_cfg.top_clear_y > 0:
        alpha_u8[: layer_cfg.top_clear_y, :] = 0
    return alpha_u8


def _native_layer_to_inner_rgba(
    word_np: np.ndarray,
    alpha_native_u8: np.ndarray,
    inner_scaled_wh: tuple[int, int],
    inner_x0: int,
    dims: CanvasDimensionsConfig,
) -> Image.Image:
    """Decontaminates edge RGB on native 1600x1200 and maps isotropically onto 1500x1000 RGBA."""
    rgb_clean = decontaminate_matte_edge_rgb(word_np, alpha_native_u8)
    rgba_scaled = Image.fromarray(np.dstack([rgb_clean, alpha_native_u8]), mode="RGBA").resize(
        inner_scaled_wh, Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGBA", (dims.width, dims.height), (0, 0, 0, 0))
    canvas.paste(rgba_scaled, (inner_x0, 0))
    return canvas


def segment_dining_family_layers(
    word_rgb: Image.Image,
    cfg: FriedTableHybridSceneConfig,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
) -> tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    """
    Segments `michael_mid`, `ahuva_fg`, and `front_chair` on native 1600x1200 `IMG_WORD_0200`
    and returns both 1500x1000 RGBA layers and native alpha masks for background healing.
    """
    chair_a = _segment_dining_layer_native(
        word_rgb=word_rgb,
        layer_cfg=cfg.front_chair,
        sam_extractor=sam_extractor,
        matte_extractor=matte_extractor,
    )
    mid_a = _segment_dining_layer_native(
        word_rgb=word_rgb,
        layer_cfg=cfg.michael_mid,
        sam_extractor=sam_extractor,
        matte_extractor=matte_extractor,
    )
    mid_a = np.where(chair_a > 128, 0, mid_a).astype(np.uint8)

    fg_a = _segment_dining_layer_native(
        word_rgb=word_rgb,
        layer_cfg=cfg.ahuva_fg,
        sam_extractor=sam_extractor,
        matte_extractor=matte_extractor,
    )

    word_np = np.asarray(word_rgb, dtype=np.uint8)
    bg_cfg = cfg.room_background
    mid_rgba = _native_layer_to_inner_rgba(
        word_np=word_np,
        alpha_native_u8=mid_a,
        inner_scaled_wh=bg_cfg.inner_scaled_wh,
        inner_x0=bg_cfg.inner_x0,
        dims=cfg.dimensions,
    )
    fg_rgba = _native_layer_to_inner_rgba(
        word_np=word_np,
        alpha_native_u8=fg_a,
        inner_scaled_wh=bg_cfg.inner_scaled_wh,
        inner_x0=bg_cfg.inner_x0,
        dims=cfg.dimensions,
    )
    chair_rgba = _native_layer_to_inner_rgba(
        word_np=word_np,
        alpha_native_u8=chair_a,
        inner_scaled_wh=bg_cfg.inner_scaled_wh,
        inner_x0=bg_cfg.inner_x0,
        dims=cfg.dimensions,
    )
    return mid_rgba, fg_rgba, chair_rgba, mid_a, fg_a, chair_a


def _register_empty_room_to_inner(
    word_scaled_rgb: Image.Image,
    empty_room_rgb: Image.Image,
    bg_cfg: DiningRoomBackgroundConfig,
    dims: CanvasDimensionsConfig,
    min_registration_corr: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Registers the 3:2 empty-room plate onto the 1500x1000 inner frame via static upper-wall landmarks."""
    target_canvas = np.zeros((dims.height, dims.width, 3), dtype=np.float32)
    sw_in, sh_in = bg_cfg.inner_scaled_wh
    target_canvas[:sh_in, bg_cfg.inner_x0 : bg_cfg.inner_x0 + sw_in] = np.asarray(
        word_scaled_rgb, dtype=np.float32
    )
    tgt_gray = cv2.cvtColor(target_canvas.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    tx0, ty0, tx1, ty1 = bg_cfg.template_inner_xyxy
    tpl = tgt_gray[ty0:ty1, tx0:tx1]

    emp_np = np.asarray(empty_room_rgb, dtype=np.uint8)
    emp_h, emp_w, _ = emp_np.shape
    emp_gray = cv2.cvtColor(emp_np, cv2.COLOR_RGB2GRAY)

    best_corr = -1.0
    best_params: tuple[int, int, int, int] | None = None
    s_min, s_max, s_step = bg_cfg.scale_search_range_step
    for scaled_w in range(s_min, s_max, s_step):
        scaled_h = int(round(scaled_w * emp_h / emp_w))
        resized_gray = cv2.resize(
            emp_gray,
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_AREA if scaled_w < emp_w else cv2.INTER_LINEAR,
        ).astype(np.float32)
        if resized_gray.shape[0] <= (ty1 - ty0) + 4 or resized_gray.shape[1] <= (tx1 - tx0) + 4:
            continue
        res = cv2.matchTemplate(resized_gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if float(max_val) > best_corr:
            best_corr = float(max_val)
            best_params = (scaled_w, scaled_h, tx0 - int(max_loc[0]), ty0 - int(max_loc[1]))

    if best_params is None or best_corr < min_registration_corr:
        raise RuntimeError(
            f"Empty dining room registration correlation {best_corr:.4f} below required "
            f"threshold {min_registration_corr:.4f}."
        )

    sw, sh, dx, dy = best_params
    resized_rgb = np.asarray(
        empty_room_rgb.resize((sw, sh), Image.Resampling.LANCZOS), dtype=np.float32
    )
    aligned = np.zeros((dims.height, dims.width, 3), dtype=np.float32)
    valid = np.zeros((dims.height, dims.width), dtype=bool)

    sx0 = max(0, -dx)
    sy0 = max(0, -dy)
    dx0 = max(0, dx)
    dy0 = max(0, dy)
    copy_w = min(sw - sx0, dims.width - dx0)
    copy_h = min(sh - sy0, dims.height - dy0)
    aligned[dy0 : dy0 + copy_h, dx0 : dx0 + copy_w] = resized_rgb[
        sy0 : sy0 + copy_h, sx0 : sx0 + copy_w
    ]
    valid[dy0 : dy0 + copy_h, dx0 : dx0 + copy_w] = True
    return aligned, valid, best_corr


def authentic_photo_box_expanded(cfg: FriedTableHybridSceneConfig) -> tuple[int, int, int, int]:
    """
    Returns exact `(x0_auth, y0_auth, x1_auth, y1_auth)` of unpadded `IMG_WORD_0200` inside the
    `2700x1800` expanded canvas (`(683, 400, 2017, 1400)` for `1334x1000` at `inner_x0=83`).
    """
    iw, ih = cfg.dimensions.width, cfg.dimensions.height
    exp_w, exp_h = cfg.outpainting.expanded_dimensions(cfg.dimensions)
    px = (exp_w - iw) // 2
    py = (exp_h - ih) // 2
    sw_in, sh_in = cfg.room_background.inner_scaled_wh
    x0_auth = px + cfg.room_background.inner_x0
    y0_auth = py
    return (x0_auth, y0_auth, x0_auth + sw_in, y0_auth + sh_in)


def mirror_authentic_region_to_expanded(
    auth_arr: np.ndarray,
    auth_box_exp: tuple[int, int, int, int],
    exp_wh: tuple[int, int],
) -> np.ndarray:
    """Applies optical `cv2.BORDER_REFLECT_101` mirror reflection outside `auth_box_exp`."""
    x0_auth, y0_auth, x1_auth, y1_auth = auth_box_exp
    exp_w, exp_h = exp_wh
    expected_hw = (y1_auth - y0_auth, x1_auth - x0_auth)
    if auth_arr.shape[:2] != expected_hw:
        raise ValueError(
            f"Expected authentic array shape {expected_hw} for box {auth_box_exp}, got {auth_arr.shape[:2]}."
        )
    return cv2.copyMakeBorder(
        auth_arr,
        y0_auth,
        exp_h - y1_auth,
        x0_auth,
        exp_w - x1_auth,
        borderType=cv2.BORDER_REFLECT_101,
    )


def cutout_touches_authentic_border_inner(
    rgba_inner: np.ndarray,
    bg_cfg: DiningRoomBackgroundConfig,
    alpha_threshold_u8: int,
) -> bool:
    """Checks whether an `inner` (`1500x1000`) RGBA cutout touches the authentic photo border."""
    sw_in, sh_in = bg_cfg.inner_scaled_wh
    x0 = bg_cfg.inner_x0
    x1 = x0 + sw_in
    m = max(1, bg_cfg.border_touch_margin_px)
    opaque = rgba_inner[:sh_in, x0:x1, 3] >= alpha_threshold_u8
    if not np.any(opaque):
        return False
    return bool(
        np.any(opaque[:m, :])
        or np.any(opaque[sh_in - m :, :])
        or np.any(opaque[:, :m])
        or np.any(opaque[:, sw_in - m :])
    )


def _extract_unoccluded_table_mask_native(
    word_rgb: Image.Image,
    mid_a: np.ndarray,
    fg_a: np.ndarray,
    chair_a: np.ndarray,
    bg_cfg: DiningRoomBackgroundConfig,
    sam_extractor: SamBoxMaskExtractor,
) -> np.ndarray:
    """
    Segments the entire visible dining table (red tablecloth, green paisley runner, white book,
    yellow document folder, crimson skirt, and dark wooden legs) on `word_rgb` (`1600x1200`)
    outside the people/front-chair cutouts `(mid_a <= 8) & (fg_a <= 8) & (chair_a <= 8)` so
    100% of the unoccluded dining table remains untouched original `IMG_WORD_0200`.
    """
    word_np = np.asarray(word_rgb, dtype=np.uint8)
    prot_u8 = sam_extractor.segment_point_groups(
        word_rgb,
        bg_cfg.table_protect_sam_prompts,
        (),
        bg_cfg.table_protect_max_pinhole_px,
        bg_cfg.table_protect_min_component_px,
    )
    items_u8 = sam_extractor.segment_point_groups(
        word_rgb,
        bg_cfg.table_items_sam_prompts,
        (),
        bg_cfg.table_items_max_pinhole_px,
        bg_cfg.table_items_min_component_px,
    )

    wall_bright = (word_np.min(axis=2) > bg_cfg.table_wall_min_rgb_exclude) & (
        items_u8 < bg_cfg.table_items_alpha_u8
    )
    prot_bool = (
        (prot_u8 >= bg_cfg.table_protect_alpha_u8) | (items_u8 >= bg_cfg.table_items_alpha_u8)
    ) & (~wall_bright)
    fill_max_y = min(prot_bool.shape[0], bg_cfg.table_hole_fill_max_y_native)
    prot_bool[:fill_max_y, :] = ndi.binary_fill_holes(prot_bool[:fill_max_y, :])

    occ_thr = bg_cfg.cutout_alpha_threshold_u8
    unoccluded_table_native = (
        prot_bool & (mid_a <= occ_thr) & (fg_a <= occ_thr) & (chair_a <= occ_thr)
    )
    return unoccluded_table_native


def build_dining_room_background_plates(
    word_rgb: Image.Image,
    raw_outpaint_rgb: Image.Image,
    mid_a: np.ndarray,
    fg_a: np.ndarray,
    chair_a: np.ndarray,
    cfg: FriedTableHybridSceneConfig,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> tuple[Image.Image, Image.Image, float]:
    """
    Synthesizes:
      1) Authentic interior (`1334x1000` at `(683, 400, 2017, 1400)` in `2700x1800`): 100%
         untouched original `IMG_WORD_0200` on every unoccluded pixel (preserving the entire
         visible dining table, red tablecloth, paisley runner, book, yellow folder, wooden legs,
         doorway, and room walls), healing ONLY occluded pixels behind `(mid_a > 8) | (fg_a > 8) |
         (chair_a > 8)` via neural `LaMa` FFC with clean non-table room context.
      2) `expanded_bg_rgb` (`2700x1800`) and `lama_inner_rgb` (`1500x1000`): crisp optical
         `cv2.BORDER_REFLECT_101` mirror reflection of the LaMa-cleaned authentic interior across
         `(x0_auth, y0_auth, x1_auth, y1_auth)` (Feedback #52).
    """
    bg_cfg = cfg.room_background
    iw, ih = cfg.dimensions.width, cfg.dimensions.height
    sw_in, sh_in = bg_cfg.inner_scaled_wh
    x0 = bg_cfg.inner_x0
    x1 = x0 + sw_in

    unoccluded_table_native = _extract_unoccluded_table_mask_native(
        word_rgb=word_rgb,
        mid_a=mid_a,
        fg_a=fg_a,
        chair_a=chair_a,
        bg_cfg=bg_cfg,
        sam_extractor=sam_extractor,
    )
    word_scaled = word_rgb.resize((sw_in, sh_in), Image.Resampling.LANCZOS)
    word_scaled_np = np.asarray(word_scaled, dtype=np.float32)

    aligned_emp, emp_valid, reg_corr = _register_empty_room_to_inner(
        word_scaled_rgb=word_scaled,
        empty_room_rgb=raw_outpaint_rgb,
        bg_cfg=bg_cfg,
        dims=cfg.dimensions,
        min_registration_corr=cfg.min_registration_corr,
    )
    emp_valid = emp_valid.copy()
    for sbox in bg_cfg.emp_table_suppress_boxes_inner:
        emp_valid[sbox.y0 : sbox.y1, sbox.x0 : sbox.x1] = False

    occ_thr = bg_cfg.cutout_alpha_threshold_u8
    occ_native = ((mid_a > occ_thr) | (fg_a > occ_thr) | (chair_a > occ_thr)).astype(np.uint8) * 255
    occ_scaled = cv2.resize(occ_native, (sw_in, sh_in), interpolation=cv2.INTER_NEAREST)
    k_dil = bg_cfg.occlusion_dilate_inner_px
    if k_dil % 2 == 0:
        k_dil += 1
    occ_scaled = cv2.dilate(
        occ_scaled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil, k_dil))
    )

    unoccluded_table_scaled = (
        cv2.resize(
            unoccluded_table_native.astype(np.uint8) * 255,
            (sw_in, sh_in),
            interpolation=cv2.INTER_NEAREST,
        )
        > 127
    )
    occ_scaled[unoccluded_table_scaled] = 0

    word_frame_mask = np.zeros((ih, iw), dtype=bool)
    word_frame_mask[:sh_in, x0:x1] = True
    word_canvas = np.zeros((ih, iw, 3), dtype=np.float32)
    word_valid = np.zeros((ih, iw), dtype=bool)
    unoccluded_table_inner = np.zeros((ih, iw), dtype=bool)
    word_canvas[:sh_in, x0:x1] = word_scaled_np
    word_valid[:sh_in, x0:x1] = occ_scaled == 0
    unoccluded_table_inner[:sh_in, x0:x1] = unoccluded_table_scaled

    occ_inner = word_frame_mask & (~word_valid)
    non_table_room_valid = word_valid & (~unoccluded_table_inner)

    static_overlap = non_table_room_valid & emp_valid
    diff = np.zeros((ih, iw, 3), dtype=np.float32)
    w_ov = static_overlap.astype(np.float32)
    diff[static_overlap] = word_canvas[static_overlap] - aligned_emp[static_overlap]
    sig = bg_cfg.lowfreq_harmonize_sigma
    num_blur = ndi.gaussian_filter(diff * w_ov[:, :, None], sigma=(sig, sig, 0.0))
    den_blur = np.maximum(1e-4, ndi.gaussian_filter(w_ov, sigma=sig))[:, :, None]
    emp_harmonized = np.clip(aligned_emp + (num_blur / den_blur), 0.0, 255.0)

    idx_non_table = ndi.distance_transform_edt(
        ~non_table_room_valid, return_distances=False, return_indices=True
    )
    nearest_room_rgb = word_canvas[idx_non_table[0], idx_non_table[1]]

    # Build temporary LaMa context seed with 100% of the red tablecloth replaced by clean room/floor
    # so LaMa never smears crimson cloth onto the wall behind Michael or floor behind Ahuva/chair.
    room_seed_inner = np.where(emp_valid[:, :, None], emp_harmonized, nearest_room_rgb)
    room_seed_inner[non_table_room_valid] = word_canvas[non_table_room_valid]
    room_seed_inner[unoccluded_table_inner] = nearest_room_rgb[unoccluded_table_inner]
    room_seed_auth = np.clip(np.round(room_seed_inner[:sh_in, x0:x1]), 0, 255).astype(np.uint8)
    seed_inner_u8 = cv2.copyMakeBorder(
        room_seed_auth,
        0,
        ih - sh_in,
        x0,
        iw - x1,
        borderType=cv2.BORDER_REFLECT_101,
    )

    healed_inner_img = lama_inpainter.inpaint_background(
        Image.fromarray(seed_inner_u8, mode="RGB"),
        (occ_inner.astype(np.uint8)) * 255,
        dilation_radius_px=bg_cfg.occ_lama_dilate_px,
        mask_blur_sigma=bg_cfg.occ_lama_blur_sigma,
    )
    healed_inner_np = np.asarray(healed_inner_img, dtype=np.float32)

    dist_inside_occ = cv2.distanceTransform(occ_inner.astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_table = cv2.distanceTransform((~unoccluded_table_inner).astype(np.uint8), cv2.DIST_L2, 5)
    w_occ = np.clip(dist_inside_occ / max(1.0, bg_cfg.occ_interior_feather_px), 0.0, 1.0)
    w_occ = w_occ * w_occ * (3.0 - 2.0 * w_occ)
    near_table = occ_inner & (dist_to_table <= bg_cfg.occ_table_guard_dist_px)
    w_occ[near_table] = 1.0
    w_occ[~occ_inner] = 0.0
    w_occ_3 = w_occ[:, :, None]
    inner_comp = (1.0 - w_occ_3) * word_canvas + w_occ_3 * healed_inner_np

    # Guarantee exact pixel identity on every unoccluded pixel inside `[0:sh_in, x0:x1]`
    word_scaled_u8 = np.asarray(word_scaled, dtype=np.uint8)
    auth_clean_u8 = np.clip(np.round(inner_comp[:sh_in, x0:x1]), 0, 255).astype(np.uint8)
    keep_exact = occ_scaled == 0
    auth_clean_u8[keep_exact] = word_scaled_u8[keep_exact]

    if not bg_cfg.mirror_authentic_borders:
        raise ValueError("DiningRoomBackgroundConfig requires mirror_authentic_borders=True.")

    exp_w, exp_h = cfg.outpainting.expanded_dimensions(cfg.dimensions)
    px = (exp_w - iw) // 2
    py = (exp_h - ih) // 2
    auth_box_exp = authentic_photo_box_expanded(cfg)
    final_2700 = mirror_authentic_region_to_expanded(
        auth_arr=auth_clean_u8,
        auth_box_exp=auth_box_exp,
        exp_wh=(exp_w, exp_h),
    )
    inner_mirrored_u8 = final_2700[py : py + ih, px : px + iw].copy()
    return (
        Image.fromarray(inner_mirrored_u8, mode="RGB"),
        Image.fromarray(final_2700, mode="RGB"),
        reg_corr,
    )


def extract_hybrid_installation_cutout(
    biny_rgb: Image.Image,
    inst_cfg: HybridInstallationPlacementConfig,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
) -> Image.Image:
    """Extracts museum-grade RGBA cutout of `IMG_BINY_0010` ('היבריד') via IS-Net + Meta SAM."""
    cx0, cy0, cx1, cy1 = inst_cfg.src_crop_xyxy
    crop = biny_rgb.crop((cx0, cy0, cx1, cy1))
    cw, ch = crop.size

    isnet_full = matte_extractor.extract_alpha(crop).astype(np.float32)

    ax0, ay0, ax1, ay1 = inst_cfg.antenna_subcrop_xyxy
    ant_a = np.asarray(
        remove(crop.crop((ax0, ay0, ax1, ay1)), session=matte_extractor.session, only_mask=False).convert(
            "RGBA"
        ),
        dtype=np.float32,
    )[:, :, 3]

    sx0, sy0, sx1, sy1 = inst_cfg.spheres_subcrop_xyxy
    sph_a = np.asarray(
        remove(crop.crop((sx0, sy0, sx1, sy1)), session=matte_extractor.session, only_mask=False).convert(
            "RGBA"
        ),
        dtype=np.float32,
    )[:, :, 3]

    sam_a = sam_extractor.segment_point_groups(
        rgb_image=crop,
        point_groups=inst_cfg.sam_prompts,
        exclude_boxes=(),
        max_pinhole_fill_px=inst_cfg.sam_max_pinhole_fill_px,
        min_major_component_px=inst_cfg.sam_min_major_component_px,
    ).astype(np.float32)

    combined_a = isnet_full.copy()
    combined_a[ay0:ay1, ax0:ax1] = np.maximum(combined_a[ay0:ay1, ax0:ax1], ant_a)

    bx0, by0, bx1, by1 = inst_cfg.baseboard_suppress_crop_xyxy
    sph_clean = sph_a.copy()
    sph_clean[by0 - sy0 : by1 - sy0, bx0 - sx0 : bx1 - sx0] = np.minimum(
        sph_clean[by0 - sy0 : by1 - sy0, bx0 - sx0 : bx1 - sx0],
        sam_a[by0:by1, bx0:bx1],
    )
    combined_a[sy0:sy1, sx0:sx1] = np.maximum(combined_a[sy0:sy1, sx0:sx1], sph_clean)

    px0, py0, px1, py1 = inst_cfg.pod_zone_crop_xyxy
    pod_zone = np.zeros((ch, cw), dtype=np.float32)
    pod_zone[py0:py1, px0:px1] = 1.0
    pod_zone = ndi.gaussian_filter(pod_zone, sigma=inst_cfg.pod_zone_feather_sigma)
    combined_a = np.maximum(combined_a, sam_a * pod_zone)

    crop_np = np.asarray(crop, dtype=np.float32)
    lum = 0.299 * crop_np[:, :, 0] + 0.587 * crop_np[:, :, 1] + 0.114 * crop_np[:, :, 2]
    local_bg = ndi.gaussian_filter(lum, sigma=inst_cfg.thread_local_bg_sigma)
    thread_boost = np.clip(
        (lum - local_bg - inst_cfg.thread_luma_delta_floor) * inst_cfg.thread_boost_gain,
        0.0,
        255.0,
    )
    kd = inst_cfg.thread_support_dilate_px
    if kd % 2 == 0:
        kd += 1
    sph_support = (
        cv2.dilate(
            (combined_a > inst_cfg.thread_support_alpha_threshold).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)),
        )
        > 0
    )
    sph_roi = np.zeros((ch, cw), dtype=bool)
    for rbox in inst_cfg.thread_roi_boxes:
        sph_roi[rbox.y0 : rbox.y1, rbox.x0 : rbox.x1] = True
    combined_a = np.where(sph_support & sph_roi, np.maximum(combined_a, thread_boost), combined_a)

    notch = np.zeros((ch, cw), dtype=bool)
    notch[by0:by1, bx0:bx1] = True
    combined_a = np.where(
        notch & (sam_a < inst_cfg.baseboard_sam_keep_threshold), 0.0, combined_a
    )

    sbox = inst_cfg.pod_shadow_suppress_box
    shadow_zone = np.zeros((ch, cw), dtype=bool)
    shadow_zone[sbox.y0 : sbox.y1, sbox.x0 : sbox.x1] = True
    dark_shadow = (lum < inst_cfg.pod_shadow_luma_max) & (
        sam_a < inst_cfg.pod_shadow_sam_keep_threshold
    )
    combined_a = np.where(dark_shadow & shadow_zone, 0.0, combined_a)

    fg_bin = (combined_a >= inst_cfg.cutout_alpha_threshold).astype(np.uint8)
    nl, lbls, stats, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=8)
    keep = np.zeros((ch, cw), dtype=bool)
    for cid in range(1, nl):
        if int(stats[cid, cv2.CC_STAT_AREA]) >= inst_cfg.cutout_min_component_px:
            keep |= lbls == cid
    keep_soft = ndi.gaussian_filter(keep.astype(np.float32), sigma=inst_cfg.cutout_keep_feather_sigma)
    alpha_u8 = np.clip(
        np.round(combined_a * np.clip(keep_soft * inst_cfg.cutout_keep_gain, 0.0, 1.0)), 0, 255
    ).astype(np.uint8)
    rgb_clean = decontaminate_matte_edge_rgb(np.asarray(crop, dtype=np.uint8), alpha_u8)
    return Image.fromarray(np.dstack([rgb_clean, alpha_u8]), mode="RGBA")


def build_installation_rgba_layer(
    biny_rgb: Image.Image,
    cfg: FriedTableHybridSceneConfig,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
) -> Image.Image:
    """
    Builds `installation_rgba` (`1500x1000` RGBA): horizontally mirrored `IMG_BINY_0010` placed
    along the dining table diagonal with warm tungsten/flash relighting, crimson tablecloth
    radiosity bounce, and physical tight contact + directional cast shadows.
    """
    inst_cfg = cfg.hybrid_installation
    iw, ih = cfg.dimensions.width, cfg.dimensions.height

    cutout_rgba = extract_hybrid_installation_cutout(
        biny_rgb=biny_rgb,
        inst_cfg=inst_cfg,
        sam_extractor=sam_extractor,
        matte_extractor=matte_extractor,
    )
    oriented = ImageOps.mirror(cutout_rgba) if inst_cfg.mirror_horizontal else cutout_rgba

    arr = np.asarray(oriented, dtype=np.uint8)
    ys, xs = np.where(arr[:, :, 3] > inst_cfg.tight_alpha_threshold_u8)
    pad0 = inst_cfg.tight_crop_pad_px
    tight = oriented.crop(
        (
            max(0, int(xs.min()) - pad0),
            max(0, int(ys.min()) - pad0),
            min(arr.shape[1], int(xs.max()) + pad0 + 1),
            min(arr.shape[0], int(ys.max()) + pad0 + 1),
        )
    )
    rotated = tight.rotate(inst_cfg.rotation_deg, resample=Image.Resampling.BICUBIC, expand=True)
    rarr = np.asarray(rotated, dtype=np.uint8)
    rys, rxs = np.where(rarr[:, :, 3] > inst_cfg.rotated_alpha_threshold_u8)
    pad1 = inst_cfg.rotated_crop_pad_px
    rcrop = rotated.crop(
        (
            max(0, int(rxs.min()) - pad1),
            max(0, int(rys.min()) - pad1),
            min(rarr.shape[1], int(rxs.max()) + pad1 + 1),
            min(rarr.shape[0], int(rys.max()) + pad1 + 1),
        )
    )

    target_w = inst_cfg.dest_width_px
    target_h = int(round(target_w * rcrop.height / rcrop.width))
    scaled = rcrop.resize((target_w, target_h), Image.Resampling.LANCZOS)
    s_np = np.asarray(scaled, dtype=np.float32)
    rgb = s_np[:, :, :3]
    alpha = s_np[:, :, 3] / 255.0

    sh, _ = alpha.shape
    vv = np.linspace(0.0, 1.0, sh, dtype=np.float32)[:, None]

    warm_gain = np.array(inst_cfg.warm_key_gain_rgb, dtype=np.float32)[None, None, :]
    relit = np.clip(rgb * warm_gain, 0.0, 255.0)

    table_contact_zone = vv > inst_cfg.table_contact_v_start
    y_idx = np.arange(sh, dtype=np.float32)[:, None]
    valid_col = (alpha > inst_cfg.table_base_alpha_threshold) & table_contact_zone
    y_bot_col = np.max(np.where(valid_col, y_idx, -1.0), axis=0)
    y_bot_smooth = ndi.gaussian_filter1d(
        np.where(y_bot_col > 0, y_bot_col, float(sh) * inst_cfg.table_base_fallback_v_ratio),
        sigma=inst_cfg.table_base_smooth_sigma,
    )[None, :]
    dist_from_base = np.maximum(0.0, y_bot_smooth - y_idx)
    underside_w = (
        np.exp(-dist_from_base / inst_cfg.underside_decay_px)
        * (valid_col.any(axis=0)[None, :].astype(np.float32))
        * np.clip(
            (vv - inst_cfg.table_contact_v_start) / inst_cfg.table_contact_v_ramp, 0.0, 1.0
        )
    )
    crimson_bounce = np.array(inst_cfg.crimson_bounce_rgb, dtype=np.float32)[None, None, :]
    bounce_w = (underside_w * inst_cfg.crimson_bounce_strength)[:, :, None]
    relit = (1.0 - bounce_w) * relit + bounce_w * crimson_bounce

    dest_x = inst_cfg.dest_left_inner_x
    dest_y = inst_cfg.dest_bottom_inner_y - target_h

    footprint = np.zeros((ih, iw), dtype=np.float32)
    contact_band = (
        alpha
        * np.exp(-dist_from_base / inst_cfg.footprint_decay_px)
        * np.clip((vv - inst_cfg.footprint_v_start) / inst_cfg.footprint_v_ramp, 0.0, 1.0)
    )
    footprint[dest_y : dest_y + target_h, dest_x : dest_x + target_w] = contact_band

    cast_sh = ndi.shift(
        footprint, shift=inst_cfg.cast_shadow_shift_yx, order=1, mode="constant", cval=0.0
    )
    cast_sh = ndi.gaussian_filter(cast_sh, sigma=inst_cfg.cast_shadow_sigma_yx)
    cast_sh = np.clip(cast_sh * inst_cfg.cast_shadow_gain, 0.0, inst_cfg.cast_shadow_max_alpha)

    cont_sh = ndi.shift(
        footprint, shift=inst_cfg.contact_shadow_shift_yx, order=1, mode="constant", cval=0.0
    )
    cont_sh = ndi.gaussian_filter(cont_sh, sigma=inst_cfg.contact_shadow_sigma_yx)
    cont_sh = np.clip(cont_sh * inst_cfg.contact_shadow_gain, 0.0, inst_cfg.contact_shadow_max_alpha)

    yy_i, xx_i = np.mgrid[0:ih, 0:iw].astype(np.float32)
    p0x, p0y = inst_cfg.tabletop_front_crease_p0_xy
    p1x, p1y = inst_cfg.tabletop_front_crease_p1_xy
    top_front_y = p0y + (xx_i - p0x) * ((p1y - p0y) / max(1.0, p1x - p0x))
    tbx0, tby0, tbx1, tby1 = inst_cfg.tabletop_bounds_xyxy
    on_table = (
        (yy_i <= top_front_y + inst_cfg.tabletop_front_margin_px)
        & (yy_i >= tby0)
        & (yy_i <= tby1)
        & (xx_i >= tbx0)
        & (xx_i <= tbx1)
    )
    on_table_soft = ndi.gaussian_filter(
        on_table.astype(np.float32), sigma=inst_cfg.tabletop_mask_feather_sigma
    )

    sh_alpha = (
        np.clip(1.0 - (1.0 - cast_sh) * (1.0 - cont_sh), 0.0, inst_cfg.combined_shadow_max_alpha)
        * on_table_soft
    )
    shadow_rgb = np.zeros((ih, iw, 3), dtype=np.float32)
    shadow_rgb[:, :] = np.array(inst_cfg.shadow_rgb, dtype=np.float32)

    installation_rgba = Image.fromarray(
        np.dstack(
            [
                np.round(shadow_rgb).astype(np.uint8),
                np.round(sh_alpha * 255.0).astype(np.uint8),
            ]
        ),
        mode="RGBA",
    )
    sculp_patch = Image.fromarray(
        np.dstack(
            [
                np.clip(np.round(relit), 0, 255).astype(np.uint8),
                np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8),
            ]
        ),
        mode="RGBA",
    )
    installation_rgba.alpha_composite(sculp_patch, dest=(dest_x, dest_y))
    return installation_rgba


def build_calibrated_dining_room_depth_map(
    expanded_bg_rgb: Image.Image,
    cutout_layers: Sequence[Image.Image],
    cfg: FriedTableHybridSceneConfig,
    depth_estimator: DepthAnythingEstimator,
) -> Image.Image:
    """
    Computes a ghost-free, calibrated `2700x1800` 8-bit dining-room depth map via
    `DepthAnythingEstimator` healed under native family/chair cutouts via
    `clean_room_depth_under_cutouts` (preserving the unoccluded dining tabletop) and reflected via
    `cv2.BORDER_REFLECT_101` across the authentic photo rectangle `(683, 400, 2017, 1400)` so
    ground and surround depth match the mirrored RGB background seamlessly (Feedback #52).
    """
    dc = cfg.depth_calibration
    exp_w, exp_h = expanded_bg_rgb.size
    pad_x = (exp_w - cfg.dimensions.width) * 0.5
    pad_y = (exp_h - cfg.dimensions.height) * 0.5
    x0_auth, y0_auth, x1_auth, y1_auth = authentic_photo_box_expanded(cfg)

    exp_np = np.asarray(expanded_bg_rgb.convert("RGB"), dtype=np.uint8)
    auth_rgb_img = Image.fromarray(exp_np[y0_auth:y1_auth, x0_auth:x1_auth], mode="RGB")
    raw_da2_auth = depth_estimator.predict_depth(auth_rgb_img)
    raw_min = float(raw_da2_auth.min())
    raw_max = float(raw_da2_auth.max())
    raw_u8_auth = np.rint(
        np.clip((raw_da2_auth - raw_min) / max(1e-6, raw_max - raw_min), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    raw_u8 = mirror_authentic_region_to_expanded(
        auth_arr=raw_u8_auth,
        auth_box_exp=(x0_auth, y0_auth, x1_auth, y1_auth),
        exp_wh=(exp_w, exp_h),
    )

    cutout_arrays = [np.asarray(im.convert("RGBA"), dtype=np.uint8) for im in cutout_layers]
    cleaned_raw_u8 = clean_room_depth_under_cutouts(
        room_depth=raw_u8,
        cutout_layers=cutout_arrays,
        cfg=dc.room_clean,
    )

    smooth_da2 = ndi.gaussian_filter(
        cleaned_raw_u8.astype(np.float32), sigma=dc.da2_gaussian_sigma
    )
    auth_smooth = smooth_da2[y0_auth:y1_auth, x0_auth:x1_auth]
    d_min = float(auth_smooth.min())
    d_max = float(auth_smooth.max())
    norm_da2 = np.clip((smooth_da2 - d_min) / max(1e-6, d_max - d_min), 0.0, 1.0)
    z_field = dc.da2_z_min + norm_da2 * (dc.da2_z_max - dc.da2_z_min)

    xx_inner = (np.arange(exp_w, dtype=np.float32) - pad_x)[None, :]
    yy_inner = (np.arange(exp_h, dtype=np.float32) - pad_y)[:, None]

    for pin in dc.contact_pins:
        px, py = pin.inner_xy
        rx, ry = pin.radii_xy
        ell_r = np.hypot((xx_inner - px) / max(1.0, rx), (yy_inner - py) / max(1.0, ry))
        geom_mean_r = float(np.sqrt(max(1.0, rx * ry)))
        excess_px = np.maximum(0.0, ell_r - 1.0) * geom_mean_r
        u_pin = np.clip(excess_px / max(1.0, pin.feather_px), 0.0, 1.0)
        w_pin = 1.0 - u_pin * u_pin * (3.0 - 2.0 * u_pin)
        z_field = z_field + w_pin * (pin.target_z - z_field)

    if dc.final_smooth_sigma > 0.0:
        z_field = ndi.gaussian_filter(z_field, sigma=dc.final_smooth_sigma)

    norm_u8 = np.clip(
        (z_field - dc.scene_bg_z) / max(1e-4, dc.room_depth_amplitude),
        0.0,
        1.0,
    )
    depth_u8_full = np.clip(np.round(norm_u8 * 255.0), 0, 255).astype(np.uint8)
    depth_u8_auth = depth_u8_full[y0_auth:y1_auth, x0_auth:x1_auth]
    depth_u8 = mirror_authentic_region_to_expanded(
        auth_arr=depth_u8_auth,
        auth_box_exp=(x0_auth, y0_auth, x1_auth, y1_auth),
        exp_wh=(exp_w, exp_h),
    )
    return Image.fromarray(depth_u8, mode="L")


def compose_populated_dining_expanded_rgb(
    expanded_bg_rgb: Image.Image,
    cutout_layers: Sequence[Image.Image],
    inner_dims: CanvasDimensionsConfig,
) -> Image.Image:
    """Alpha-composites ordered `1500x1000` dining cutouts onto `2700x1800` room plate for DA2."""
    exp_w, exp_h = expanded_bg_rgb.size
    pad_x = (exp_w - inner_dims.width) // 2
    pad_y = (exp_h - inner_dims.height) // 2
    comp = expanded_bg_rgb.convert("RGBA")
    for layer in cutout_layers:
        if layer.size != (inner_dims.width, inner_dims.height):
            raise ValueError(
                f"Expected inner cutout size {(inner_dims.width, inner_dims.height)}, got {layer.size}."
            )
        comp.alpha_composite(layer.convert("RGBA"), dest=(pad_x, pad_y))
    return comp.convert("RGB")


def _expand_border_touching_cutout_and_depth(
    rgba_inner_np: np.ndarray,
    world_z_inner: np.ndarray,
    contact_inner: np.ndarray,
    cfg: FriedTableHybridSceneConfig,
    layer_cfg: ContactAnchoredLayerDepthConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mirrors a border-touching `1500x1000` cutout + solved world-Z + contact mask across the
    authentic photo rectangle `(683, 400, 2017, 1400)` onto `2700x1800` via `cv2.BORDER_REFLECT_101`,
    applies the host room's outer perimeter pin taper (`room_pin_margin_uv = 0.045`) so reflected
    floor contacts remain exact (`ΔZ_world == 0`) all the way to the `2700x1800` perimeter, and
    extrapolates interior world-Z outward via Voronoi `EDT` (`pinMarginUv = 0.0`).
    """
    bg_cfg = cfg.room_background
    sw_in, sh_in = bg_cfg.inner_scaled_wh
    x0 = bg_cfg.inner_x0
    x1 = x0 + sw_in
    exp_wh = cfg.outpainting.expanded_dimensions(cfg.dimensions)
    exp_w, exp_h = exp_wh
    auth_box_exp = authentic_photo_box_expanded(cfg)

    rgba_exp = mirror_authentic_region_to_expanded(
        auth_arr=rgba_inner_np[:sh_in, x0:x1],
        auth_box_exp=auth_box_exp,
        exp_wh=exp_wh,
    ).copy()
    rgba_exp[0, :, 3] = 0
    rgba_exp[-1, :, 3] = 0
    rgba_exp[:, 0, 3] = 0
    rgba_exp[:, -1, 3] = 0

    world_z_mirrored = mirror_authentic_region_to_expanded(
        auth_arr=world_z_inner[:sh_in, x0:x1].astype(np.float32),
        auth_box_exp=auth_box_exp,
        exp_wh=exp_wh,
    )
    pin_room = compute_perimeter_pin_weights(exp_h, exp_w, layer_cfg.room_pin_margin_uv)
    world_z_pinned = (
        float(layer_cfg.bg_z) + (world_z_mirrored - float(layer_cfg.bg_z)) * pin_room
    ).astype(np.float32)

    contact_exp = (
        mirror_authentic_region_to_expanded(
            auth_arr=contact_inner[:sh_in, x0:x1].astype(np.uint8),
            auth_box_exp=auth_box_exp,
            exp_wh=exp_wh,
        )
        > 0
    )
    opaque_exp = rgba_exp[:, :, 3] >= layer_cfg.alpha_threshold_u8
    contact_exp &= opaque_exp
    world_z_full = extrapolate_depth_outward_edt(world_z_pinned, opaque_exp)

    d_norm = (world_z_full - float(layer_cfg.layer_z)) / float(layer_cfg.layer_depth_amplitude)
    if float(d_norm.min()) < -0.005 or float(d_norm.max()) > 1.005:
        raise ValueError(
            f"Expanded mirrored layer world-Z [{float(world_z_full.min()):.2f}, "
            f"{float(world_z_full.max()):.2f}] overflows [{layer_cfg.layer_z:.2f}, "
            f"{layer_cfg.layer_z + layer_cfg.layer_depth_amplitude:.2f}]."
        )
    depth_exp = np.rint(np.clip(d_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgba_exp, depth_exp, contact_exp


def solve_and_verify_dining_table_layer_depths(
    expanded_bg_rgb: Image.Image,
    room_depth_img: Image.Image,
    mid_rgba: Image.Image,
    inst_rgba: Image.Image,
    fg_rgba: Image.Image,
    chair_rgba: Image.Image,
    cfg: FriedTableHybridSceneConfig,
    depth_estimator: DepthAnythingEstimator,
) -> tuple[
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    dict[str, LayerTopologyVerificationReport],
]:
    """
    Solves harmonic contact-anchored depth maps (`pinMarginUv = 0.0`, outward `EDT` rim
    extrapolation) for all 4 cutout layers in `fried_table_hybrid`, and expands any border-touching
    cutout layers (`fg` and `front_chair`) to `2700x1800` via `cv2.BORDER_REFLECT_101` across the
    authentic photo rectangle `(683, 400, 2017, 1400)` so chair legs, clothing, sandals, and ground
    reflect seamlessly across `y1_auth` (Feedback #52).
    """
    dc = cfg.depth_calibration
    bg_cfg = cfg.room_background
    room_depth_u8 = np.asarray(room_depth_img.convert("L"), dtype=np.uint8)
    populated_rgb = compose_populated_dining_expanded_rgb(
        expanded_bg_rgb=expanded_bg_rgb,
        cutout_layers=(mid_rgba, inst_rgba, fg_rgba, chair_rgba),
        inner_dims=cfg.dimensions,
    )
    pop_np = np.asarray(populated_rgb, dtype=np.uint8)
    auth_box_exp = authentic_photo_box_expanded(cfg)
    x0_auth, y0_auth, x1_auth, y1_auth = auth_box_exp
    raw_pop_auth = depth_estimator.predict_depth(
        Image.fromarray(pop_np[y0_auth:y1_auth, x0_auth:x1_auth], mode="RGB")
    )
    pop_min = float(raw_pop_auth.min())
    pop_max = float(raw_pop_auth.max())
    raw_pop_auth_u8 = np.rint(
        np.clip((raw_pop_auth - pop_min) / max(1e-6, pop_max - pop_min), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    raw_pop_u8 = mirror_authentic_region_to_expanded(
        auth_arr=raw_pop_auth_u8,
        auth_box_exp=auth_box_exp,
        exp_wh=expanded_bg_rgb.size,
    )

    specs: tuple[tuple[str, Image.Image, ContactAnchoredLayerDepthConfig], ...] = (
        ("mid", mid_rgba, dc.mid_topology),
        (cfg.hybrid_installation.layer_id, inst_rgba, dc.installation_topology),
        ("fg", fg_rgba, dc.fg_topology),
        (cfg.front_chair.layer_id, chair_rgba, dc.front_chair_topology),
    )

    final_rgba_images: dict[str, Image.Image] = {}
    solved_images: dict[str, Image.Image] = {}
    reports: dict[str, LayerTopologyVerificationReport] = {}
    for layer_key, rgba_img, layer_cfg in specs:
        rgba_np = np.asarray(rgba_img.convert("RGBA"), dtype=np.uint8)
        protrusion = compute_room_relative_neural_protrusion(
            raw_neural_depth_expanded=raw_pop_u8,
            cutout_rgba=rgba_np,
            clean_cfg=dc.protrusion_clean,
            upper_percentile=dc.protrusion_upper_percentile,
        )
        solved = solve_contact_anchored_layer_depth(
            room_depth_expanded=room_depth_u8,
            cutout_rgba=rgba_np,
            cfg=layer_cfg,
            neural_protrusion_norm=protrusion,
        )
        final_rgba_np = rgba_np
        final_depth_u8 = solved.depth_u8
        final_contact = solved.contact_mask
        if bg_cfg.expand_border_touching_cutouts and cutout_touches_authentic_border_inner(
            rgba_inner=rgba_np,
            bg_cfg=bg_cfg,
            alpha_threshold_u8=layer_cfg.alpha_threshold_u8,
        ):
            final_rgba_np, final_depth_u8, final_contact = _expand_border_touching_cutout_and_depth(
                rgba_inner_np=rgba_np,
                world_z_inner=solved.world_z_map,
                contact_inner=solved.contact_mask,
                cfg=cfg,
                layer_cfg=layer_cfg,
            )

        report = verify_layer_contact_and_discontinuity(
            room_depth_expanded=room_depth_u8,
            layer_rgba=final_rgba_np,
            layer_depth=final_depth_u8,
            contact_mask=final_contact,
            bg_z=layer_cfg.bg_z,
            room_depth_amplitude=layer_cfg.room_depth_amplitude,
            room_pin_margin_uv=layer_cfg.room_pin_margin_uv,
            layer_z=layer_cfg.layer_z,
            layer_depth_amplitude=layer_cfg.layer_depth_amplitude,
            layer_pin_margin_uv=layer_cfg.layer_pin_margin_uv,
            verify_cfg=dc.layer_verification,
        )
        if not report.all_passed:
            raise RuntimeError(
                f"[{cfg.scene_id}] Layer {layer_key!r} failed 2.5D depth topology verification: "
                f"{report.to_json()}"
            )
        final_rgba_images[layer_key] = Image.fromarray(final_rgba_np, mode="RGBA")
        solved_images[layer_key] = Image.fromarray(final_depth_u8, mode="L")
        reports[layer_key] = report

    return (
        final_rgba_images["fg"],
        final_rgba_images[cfg.front_chair.layer_id],
        solved_images["mid"],
        solved_images[cfg.hybrid_installation.layer_id],
        solved_images["fg"],
        solved_images[cfg.front_chair.layer_id],
        reports,
    )


def _build_fried_table_extra_layers_spec(cfg: FriedTableHybridSceneConfig) -> list[dict[str, Any]]:
    inst = cfg.hybrid_installation
    dc = cfg.depth_calibration
    expand_cutouts = cfg.room_background.expand_border_touching_cutouts
    return [
        {
            "id": inst.layer_id,
            "labelHe": inst.label_he,
            "texture": f"assets/scenes/{cfg.installation_output_filename}",
            "depthTexture": f"assets/scenes/{cfg.installation_depth_output_filename}",
            "zPos": inst.z_pos,
            "renderOrder": inst.render_order,
            "expanded": False,
            "baseOpacity": 1.0,
            "excursionBoost": 0.0,
            "depthTest": False,
            "depthWrite": False,
            "depthAmplitude": inst.depth_amplitude,
            "pinMarginUv": inst.pin_margin_uv,
            "alphaCutout": inst.alpha_cutout,
            "segmentsX": inst.segments_x,
            "segmentsY": inst.segments_y,
        },
        {
            "id": cfg.front_chair.layer_id,
            "labelHe": cfg.front_chair.label_he,
            "texture": f"assets/scenes/{cfg.front_chair_output_filename}",
            "depthTexture": f"assets/scenes/{cfg.front_chair_depth_output_filename}",
            "zPos": cfg.front_chair_z,
            "renderOrder": cfg.front_chair_render_order,
            "expanded": expand_cutouts,
            "baseOpacity": 1.0,
            "excursionBoost": 0.0,
            "depthTest": False,
            "depthWrite": False,
            "depthAmplitude": dc.front_chair_topology.layer_depth_amplitude,
            "pinMarginUv": dc.front_chair_topology.layer_pin_margin_uv,
            "alphaCutout": 0.0,
            "segmentsX": cfg.front_chair_segments_x,
            "segmentsY": cfg.front_chair_segments_y,
        },
    ]


def build_fried_table_hybrid_manifest_entry(
    cfg: FriedTableHybridSceneConfig,
    reg_corr: float | None = None,
) -> dict[str, Any]:
    """Builds the canonical `scene_manifest.json` record for Scene 11 (`fried_table_hybrid`)."""
    inst = cfg.hybrid_installation
    dc = cfg.depth_calibration
    exp_w, exp_h = cfg.outpainting.expanded_dimensions(cfg.dimensions)
    extra_layers = _build_fried_table_extra_layers_spec(cfg)
    entry: dict[str, Any] = {
        "id": cfg.scene_id,
        "title": cfg.title_he,
        "shortLabel": cfg.short_label_he,
        "shortLabelHe": cfg.short_label_he,
        "icon": cfg.icon,
        "exhibitionGroup": cfg.exhibition_group,
        "sourceImage": cfg.dining_source_relpath,
        "secondarySourceImage": inst.source_relpath,
        "sourceCatalogIds": [cfg.dining_catalog_id, inst.catalog_id],
        "outpaintSource": cfg.outpaint_raw_relpath,
        "segmentationBackend": "fried_table_hybrid_sam_isnet",
        "depthBackend": "fried_table_harmonic_contact_anchored_da2",
        "innerWidth": cfg.dimensions.width,
        "innerHeight": cfg.dimensions.height,
        "bgExpandedWidth": exp_w,
        "bgExpandedHeight": exp_h,
        "bgExpansionFactor": cfg.outpainting.expansion_factor,
        "fgExpanded": cfg.room_background.expand_border_touching_cutouts,
        "authenticPhotoBoxExpanded": list(authentic_photo_box_expanded(cfg)),
        "boreSightAtRest": cfg.bore_sight_at_rest,
        "roomDepthAmplitude": dc.room_depth_amplitude,
        "sceneBgZ": cfg.scene_bg_z,
        "midZ": cfg.mid_z,
        "midDepthAmplitude": dc.mid_topology.layer_depth_amplitude,
        "midPinMarginUv": dc.mid_topology.layer_pin_margin_uv,
        "midSegmentsX": cfg.mid_segments_x,
        "midSegmentsY": cfg.mid_segments_y,
        "fgZ": cfg.fg_z,
        "fgDepthAmplitude": dc.fg_topology.layer_depth_amplitude,
        "fgPinMarginUv": dc.fg_topology.layer_pin_margin_uv,
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
        "extraLayers": extra_layers,
        "reflectionTexture": None,
        "hasReflection": False,
    }
    if reg_corr is not None:
        entry["outpaintRegistrationCorr"] = round(reg_corr, 4)
    return entry


def build_fried_table_hybrid_js_config(cfg: FriedTableHybridSceneConfig) -> dict[str, Any]:
    """Builds the canonical `scene_manager.js` config object for Scene 11 (`fried_table_hybrid`)."""
    dc = cfg.depth_calibration
    return {
        "id": cfg.scene_id,
        "title": cfg.title_he,
        "shortLabel": cfg.short_label_he,
        "icon": cfg.icon,
        "exhibitionGroup": cfg.exhibition_group,
        "bgTexture": f"assets/scenes/{cfg.bg_output_filename}",
        "depthTexture": f"assets/scenes/{cfg.depth_output_filename}",
        "midTexture": f"assets/scenes/{cfg.mid_output_filename}",
        "midDepthTexture": f"assets/scenes/{cfg.mid_depth_output_filename}",
        "midDepthAmplitude": dc.mid_topology.layer_depth_amplitude,
        "midPinMarginUv": dc.mid_topology.layer_pin_margin_uv,
        "midSegmentsX": cfg.mid_segments_x,
        "midSegmentsY": cfg.mid_segments_y,
        "fgTexture": f"assets/scenes/{cfg.fg_output_filename}",
        "fgDepthTexture": f"assets/scenes/{cfg.fg_depth_output_filename}",
        "fgExpanded": cfg.room_background.expand_border_touching_cutouts,
        "fgDepthAmplitude": dc.fg_topology.layer_depth_amplitude,
        "fgPinMarginUv": dc.fg_topology.layer_pin_margin_uv,
        "fgSegmentsX": cfg.fg_segments_x,
        "fgSegmentsY": cfg.fg_segments_y,
        "boreSightAtRest": cfg.bore_sight_at_rest,
        "bgZ": cfg.scene_bg_z,
        "midZ": cfg.mid_z,
        "fgZ": cfg.fg_z,
        "bgSegmentsX": cfg.bg_segments_x,
        "bgSegmentsY": cfg.bg_segments_y,
        "roomDepthAmplitude": dc.room_depth_amplitude,
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
        "extraLayers": _build_fried_table_extra_layers_spec(cfg),
        "particleColor": f"0x{cfg.particle_color_hex:06X}",
        "particleCount": cfg.particle_count,
        "hasReflection": False,
    }


def build_fried_table_hybrid_scene(
    project_root: Path,
    output_dir: Path,
    cfg: FriedTableHybridSceneConfig | None = None,
    export_spec_json: Path | None = None,
    preview_dir: Path | None = None,
) -> dict[str, Any]:
    """End-to-end neural 2.5D asset synthesis for Scene 11 (`fried_table_hybrid`)."""
    scene_cfg = cfg or default_fried_table_hybrid_scene_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    dining_path = project_root / scene_cfg.dining_source_relpath
    if not dining_path.is_file():
        raise FileNotFoundError(f"Missing Fried dining room source photograph: {dining_path}")

    biny_path = project_root / scene_cfg.hybrid_installation.source_relpath
    if not biny_path.is_file():
        raise FileNotFoundError(f"Missing 'היבריד' installation photograph: {biny_path}")

    outpaint_raw_path = project_root / scene_cfg.outpaint_raw_relpath
    if not outpaint_raw_path.is_file():
        raise FileNotFoundError(f"Missing 3:2 empty dining room source plate: {outpaint_raw_path}")

    device = select_torch_device()
    print(f"[{scene_cfg.scene_id}] Initializing neural vision models on device={device}...")
    depth_estimator = DepthAnythingEstimator(scene_cfg.depth_model, device=device)
    sam_extractor = SamBoxMaskExtractor(scene_cfg.segmentation)
    matte_extractor = SalientMatteExtractor(scene_cfg.segmentation)
    lama_inpainter = LamaOcclusionInpainter(scene_cfg.inpainting)

    word_rgb = Image.open(dining_path).convert("RGB")
    biny_rgb = Image.open(biny_path).convert("RGB")
    raw_outpaint_rgb = Image.open(outpaint_raw_path).convert("RGB")

    print(
        f"[{scene_cfg.scene_id}] Segmenting Grandfather Michael (MID Z={scene_cfg.mid_z}), "
        f"Grandmother Ahuva (FG Z={scene_cfg.fg_z}), and Front Chair (Z={scene_cfg.front_chair_z})..."
    )
    mid_rgba, fg_rgba, chair_rgba, mid_a, fg_a, chair_a = segment_dining_family_layers(
        word_rgb=word_rgb,
        cfg=scene_cfg,
        sam_extractor=sam_extractor,
        matte_extractor=matte_extractor,
    )

    print(
        f"[{scene_cfg.scene_id}] Synthesizing 1500x1000 & 2700x1800 dining room background plates..."
    )
    lama_inner_rgb, expanded_bg_rgb, reg_corr = build_dining_room_background_plates(
        word_rgb=word_rgb,
        raw_outpaint_rgb=raw_outpaint_rgb,
        mid_a=mid_a,
        fg_a=fg_a,
        chair_a=chair_a,
        cfg=scene_cfg,
        sam_extractor=sam_extractor,
        lama_inpainter=lama_inpainter,
    )

    print(
        f"[{scene_cfg.scene_id}] Extracting, mirroring & relighting 'היבריד' "
        f"({scene_cfg.hybrid_installation.catalog_id}, Z={scene_cfg.hybrid_installation.z_pos})..."
    )
    inst_rgba = build_installation_rgba_layer(
        biny_rgb=biny_rgb,
        cfg=scene_cfg,
        sam_extractor=sam_extractor,
        matte_extractor=matte_extractor,
    )

    print(
        f"[{scene_cfg.scene_id}] Synthesizing ghost-free 2700x1800 Depth Anything V2 room depth..."
    )
    depth_img = build_calibrated_dining_room_depth_map(
        expanded_bg_rgb=expanded_bg_rgb,
        cutout_layers=(mid_rgba, fg_rgba, chair_rgba),
        cfg=scene_cfg,
        depth_estimator=depth_estimator,
    )

    print(
        f"[{scene_cfg.scene_id}] Solving & verifying C^0/C^-1 contact-anchored layer depth maps..."
    )
    (
        fg_rgba,
        chair_rgba,
        mid_depth_img,
        inst_depth_img,
        fg_depth_img,
        chair_depth_img,
        topology_reports,
    ) = solve_and_verify_dining_table_layer_depths(
        expanded_bg_rgb=expanded_bg_rgb,
        room_depth_img=depth_img,
        mid_rgba=mid_rgba,
        inst_rgba=inst_rgba,
        fg_rgba=fg_rgba,
        chair_rgba=chair_rgba,
        cfg=scene_cfg,
        depth_estimator=depth_estimator,
    )
    for lkey, rep in topology_reports.items():
        print(
            f"  -> [{scene_cfg.scene_id}.{lkey}] max|ΔZ_contact|={rep.max_abs_contact_dz_world:.3f}, "
            f"p95|ΔZ_contact|={rep.p95_abs_contact_dz_world:.3f}, "
            f"max|ΔZ_rim|={rep.max_abs_rim_jump_world:.3f}, passed={rep.all_passed}"
        )

    lama_inner_rgb.save(
        output_dir / scene_cfg.lama_inner_output_filename,
        format="JPEG",
        quality=scene_cfg.inpainting.jpeg_quality,
        optimize=True,
    )
    expanded_bg_rgb.save(
        output_dir / scene_cfg.bg_output_filename,
        format="JPEG",
        quality=scene_cfg.inpainting.jpeg_quality,
        optimize=True,
    )
    depth_img.save(output_dir / scene_cfg.depth_output_filename, format="PNG", optimize=True)
    mid_rgba.save(output_dir / scene_cfg.mid_output_filename, format="PNG", optimize=True)
    mid_depth_img.save(
        output_dir / scene_cfg.mid_depth_output_filename, format="PNG", optimize=True
    )
    fg_rgba.save(output_dir / scene_cfg.fg_output_filename, format="PNG", optimize=True)
    fg_depth_img.save(
        output_dir / scene_cfg.fg_depth_output_filename, format="PNG", optimize=True
    )
    inst_rgba.save(
        output_dir / scene_cfg.installation_output_filename, format="PNG", optimize=True
    )
    inst_depth_img.save(
        output_dir / scene_cfg.installation_depth_output_filename, format="PNG", optimize=True
    )
    chair_rgba.save(
        output_dir / scene_cfg.front_chair_output_filename, format="PNG", optimize=True
    )
    chair_depth_img.save(
        output_dir / scene_cfg.front_chair_depth_output_filename, format="PNG", optimize=True
    )

    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        expanded_bg_rgb.resize((1350, 900), Image.Resampling.LANCZOS).save(
            preview_dir / "scene11_outpaint_preview_v2.jpg", format="JPEG", quality=92
        )
        exp_w, exp_h = expanded_bg_rgb.size
        pad_x = (exp_w - scene_cfg.dimensions.width) // 2
        pad_y = (exp_h - scene_cfg.dimensions.height) // 2
        comp_exp = expanded_bg_rgb.convert("RGBA")
        for layer_img in (mid_rgba, inst_rgba, fg_rgba, chair_rgba):
            dest = (0, 0) if layer_img.size == (exp_w, exp_h) else (pad_x, pad_y)
            comp_exp.alpha_composite(layer_img, dest=dest)
        comp_exp.crop((pad_x, pad_y, pad_x + scene_cfg.dimensions.width, pad_y + scene_cfg.dimensions.height)).convert(
            "RGB"
        ).save(preview_dir / "scene11_full_comp_preview_v2.jpg", format="JPEG", quality=93)

    manifest_entry = build_fried_table_hybrid_manifest_entry(scene_cfg, reg_corr=reg_corr)
    js_scene_config = build_fried_table_hybrid_js_config(scene_cfg)
    payload = {
        "scene_id": scene_cfg.scene_id,
        "feedback_id": "fb-20261102-100309-7eadc1",
        "authentic_photo_box_expanded": list(authentic_photo_box_expanded(scene_cfg)),
        "manifest_entry": manifest_entry,
        "js_scene_config": js_scene_config,
        "topology_verification": {k: v.to_dict() for k, v in topology_reports.items()},
        "ssot_config": scene_cfg.to_dict(),
    }
    if export_spec_json is not None:
        export_spec_json.parent.mkdir(parents=True, exist_ok=True)
        export_spec_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(
        f"[{scene_cfg.scene_id}] Completed! bg={expanded_bg_rgb.size} (corr={reg_corr:.4f}), "
        f"fg={fg_rgba.size}, chair={chair_rgba.size}, inst={inst_rgba.size}, depth={depth_img.size}"
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Scene 11 (`fried_table_hybrid`) neural 2.5D spatial WebGL assets."
    )
    parser.add_argument("--project-root", type=Path, default=_REPO_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=_REPO_ROOT / "prototype" / "assets" / "scenes"
    )
    parser.add_argument("--export-spec-json", type=Path, default=None)
    parser.add_argument("--preview-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_fried_table_hybrid_scene(
        project_root=args.project_root.resolve(),
        output_dir=args.output_dir.resolve(),
        export_spec_json=args.export_spec_json.resolve() if args.export_spec_json else None,
        preview_dir=args.preview_dir.resolve() if args.preview_dir else None,
    )


if __name__ == "__main__":
    main()
