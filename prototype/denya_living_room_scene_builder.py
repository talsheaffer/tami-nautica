#!/usr/bin/env python3
"""
prototype/denya_living_room_scene_builder.py
Offline neural 2.5D spatial asset builder and typed SSOT configuration models for:
  1. `denya_living_room`: "ארוחה בסלון דניה — שער אל הספה והגיטרה ואל ׳שיטוט 2׳"
     Dining-table scene (`IMG_DROP_0106`) with the entire plaid dining table and seated family
     (Naama, Ya'ara, Carmel, Tami, Gadi) unified on a single continuous contact-anchored 3D
     depth-displaced foreground mesh (`fg` + `fgDepthTexture`, `pinMarginUv = 0.0`, outward EDT),
     an unoccluded mid-depth center coffee table & director's chair (`mid` + `midDepthTexture`)
     with a neural-LaMa downward apron behind the far tablecloth edge, Tal, Ya'ara & Carmel
     playing the guitar (`IMG_DROP_0104`) planted and contact-anchored onto the background gray
     sofa (`planted_couch_trio` + `depthTexture`), and a flush wall-anchored `'wandering 2'`
     (`wandering_drawing`) picture-frame portal on the salon gallery wall.
  2. `denya_couch_guitar`: "ספת הסלון בדניה — נגינה בגיטרה תחת קיר העבודות ושער ׳שיטוט 2׳"
     Close-up gray-couch 2.5D scene (`IMG_DROP_0104`) unifying the gray sofa, throw pillows,
     plush toy, and seated guitar trio (Tal, Ya'ara, Carmel + guitar) onto a single continuous
     wall-and-floor contact-anchored 3D mesh (`mid` + `midDepthTexture`, `pinMarginUv = 0.0`,
     outward EDT) with zero cushion halos, keeping the entire oak coffee table and all tabletop
     objects unified on a floor-leg-anchored 3D foreground mesh (`fg` + `fgDepthTexture`), and
     sharing the flush wall-anchored `'wandering 2'` portal.

Strict No-Fallback Guarantee: executes real neural models (`Depth Anything V2`, `Meta SAM`,
`IS-Net` via `rembg`, and `LaMa` via `simple-lama-inpainting`) inside `.venv`. Enforces Laws 1–5
of `.agents/skills/coding-guidelines/references/depth_topology_principles.md` via
`prototype.depth_topology` with zero silent fallbacks or parameter clamping.
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
except ImportError as exc:
    raise RuntimeError(
        "Missing neural vision dependencies inside virtual environment. Run with: "
        "`/Users/talsheaffer/code/tami/.venv/bin/python -u prototype/denya_living_room_scene_builder.py`."
    ) from exc

from prototype.depth_topology import (
    ContactAnchorBox,
    ContactAnchoredLayerDepthConfig,
    ContactAnchoredSolveResult,
    LayerTopologyVerificationConfig,
    LayerTopologyVerificationReport,
    RoomDepthCleanConfig,
    clean_room_depth_under_cutouts,
    compute_world_z_map,
    extract_contact_anchor_mask,
    extract_layer_window_from_expanded,
    normalize_depth_array,
    solve_contact_anchored_layer_depth,
    verify_layer_contact_and_discontinuity,
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
    select_torch_device,
)


# ============================================================================
# Hierarchical, Serializable Configuration Data-Models (SSOT)
# ============================================================================


@dataclass(frozen=True)
class PointCorrespondencePair:
    """2D pixel correspondence `(src_xy -> dst_xy)` in `1500x1000` inner coordinates."""

    src_xy: tuple[float, float]
    dst_xy: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_xy": list(self.src_xy),
            "dst_xy": list(self.dst_xy),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointCorrespondencePair:
        sxy = data["src_xy"]
        dxy = data["dst_xy"]
        return cls(
            src_xy=(float(sxy[0]), float(sxy[1])),
            dst_xy=(float(dxy[0]), float(dxy[1])),
        )


@dataclass(frozen=True)
class ProgressiveLamaOutpaintConfig:
    """Multi-ring LaMa architectural periphery continuation without hallucinated objects."""

    resized_source_wh: tuple[int, int]
    inner_crop_y0: int
    working_scale: float
    ring_step_px: int
    ring_dilation_px: int
    seam_feather_px: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "resized_source_wh": list(self.resized_source_wh),
            "inner_crop_y0": self.inner_crop_y0,
            "working_scale": self.working_scale,
            "ring_step_px": self.ring_step_px,
            "ring_dilation_px": self.ring_dilation_px,
            "seam_feather_px": self.seam_feather_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressiveLamaOutpaintConfig:
        rwh = data["resized_source_wh"]
        return cls(
            resized_source_wh=(int(rwh[0]), int(rwh[1])),
            inner_crop_y0=int(data["inner_crop_y0"]),
            working_scale=float(data["working_scale"]),
            ring_step_px=int(data["ring_step_px"]),
            ring_dilation_px=int(data["ring_dilation_px"]),
            seam_feather_px=float(data["seam_feather_px"]),
        )


@dataclass(frozen=True)
class WallDepthFlattenPinConfig:
    """Smooth elliptical Z-depth pin keeping the salon gallery wall planar in 3D."""

    inner_xy: tuple[float, float]
    radii_xy: tuple[float, float]
    feather_px: float
    target_norm_depth: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "inner_xy": list(self.inner_xy),
            "radii_xy": list(self.radii_xy),
            "feather_px": self.feather_px,
            "target_norm_depth": self.target_norm_depth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WallDepthFlattenPinConfig:
        ixy = data["inner_xy"]
        rxy = data["radii_xy"]
        return cls(
            inner_xy=(float(ixy[0]), float(ixy[1])),
            radii_xy=(float(rxy[0]), float(rxy[1])),
            feather_px=float(data["feather_px"]),
            target_norm_depth=float(data["target_norm_depth"]),
        )


@dataclass(frozen=True)
class DownwardOcclusionApronConfig:
    """Neural-LaMa downward continuation apron behind a foreground occluder to prevent parallax gaps."""

    downward_extension_px: int
    max_depth_into_occluder_px: float
    bottom_feather_px: float
    seam_blend_px: float
    alpha_threshold_u8: int
    occluder_threshold_u8: int
    inpaint_dilation_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "downward_extension_px": self.downward_extension_px,
            "max_depth_into_occluder_px": self.max_depth_into_occluder_px,
            "bottom_feather_px": self.bottom_feather_px,
            "seam_blend_px": self.seam_blend_px,
            "alpha_threshold_u8": self.alpha_threshold_u8,
            "occluder_threshold_u8": self.occluder_threshold_u8,
            "inpaint_dilation_px": self.inpaint_dilation_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DownwardOcclusionApronConfig:
        return cls(
            downward_extension_px=int(data["downward_extension_px"]),
            max_depth_into_occluder_px=float(data["max_depth_into_occluder_px"]),
            bottom_feather_px=float(data["bottom_feather_px"]),
            seam_blend_px=float(data["seam_blend_px"]),
            alpha_threshold_u8=int(data["alpha_threshold_u8"]),
            occluder_threshold_u8=int(data["occluder_threshold_u8"]),
            inpaint_dilation_px=int(data["inpaint_dilation_px"]),
        )


@dataclass(frozen=True)
class SourceProtrusionReliefConfig:
    """Parameters for extracting genuine neural 3D sculptural/perspective relief from source DA2."""

    da2_smooth_sigma_px: float
    anchor_baseline_percentile: float
    upper_percentile: float
    core_erode_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "da2_smooth_sigma_px": self.da2_smooth_sigma_px,
            "anchor_baseline_percentile": self.anchor_baseline_percentile,
            "upper_percentile": self.upper_percentile,
            "core_erode_px": self.core_erode_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceProtrusionReliefConfig:
        return cls(
            da2_smooth_sigma_px=float(data["da2_smooth_sigma_px"]),
            anchor_baseline_percentile=float(data["anchor_baseline_percentile"]),
            upper_percentile=float(data["upper_percentile"]),
            core_erode_px=int(data["core_erode_px"]),
        )


@dataclass(frozen=True)
class FramedPaintingPortalConfig:
    """Framed `'wandering 2'` (`wandering_drawing`) gallery artwork & flush 3D wall portal spec."""

    target_scene_id: str
    portal_id_in_dining: str
    portal_id_in_couch: str
    label_he: str
    short_badge_he: str
    source_bg_relpath: str
    source_mid_relpath: str
    source_fg_relpath: str
    couch_outer_box_xyxy: tuple[int, int, int, int]
    couch_moulding_px: int
    couch_matboard_px: int
    couch_bevel_px: int
    oak_base_rgb: tuple[int, int, int]
    oak_light_rgb: tuple[int, int, int]
    oak_dark_rgb: tuple[int, int, int]
    couch_mat_rgb: tuple[int, int, int]
    dining_mat_rgb: tuple[int, int, int]
    bevel_shadow_rgb: tuple[int, int, int]
    bevel_highlight_rgb: tuple[int, int, int]
    drop_shadow_offset_couch_xy: tuple[int, int]
    drop_shadow_sigma_couch: float
    drop_shadow_opacity: float
    glass_warm_tint_rgb: tuple[float, float, float]
    glass_contrast: float
    recess_depth: float
    moulding_relief_norm: float
    matboard_relief_norm: float
    frame_min_component_px: int
    badge_offset_y_factor: float

    def couch_inner_aperture_xyxy(self) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.couch_outer_box_xyxy
        inset = self.couch_moulding_px + self.couch_matboard_px
        return (x0 + inset, y0 + inset, x1 - inset, y1 - inset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_scene_id": self.target_scene_id,
            "portal_id_in_dining": self.portal_id_in_dining,
            "portal_id_in_couch": self.portal_id_in_couch,
            "label_he": self.label_he,
            "short_badge_he": self.short_badge_he,
            "source_bg_relpath": self.source_bg_relpath,
            "source_mid_relpath": self.source_mid_relpath,
            "source_fg_relpath": self.source_fg_relpath,
            "couch_outer_box_xyxy": list(self.couch_outer_box_xyxy),
            "couch_moulding_px": self.couch_moulding_px,
            "couch_matboard_px": self.couch_matboard_px,
            "couch_bevel_px": self.couch_bevel_px,
            "oak_base_rgb": list(self.oak_base_rgb),
            "oak_light_rgb": list(self.oak_light_rgb),
            "oak_dark_rgb": list(self.oak_dark_rgb),
            "couch_mat_rgb": list(self.couch_mat_rgb),
            "dining_mat_rgb": list(self.dining_mat_rgb),
            "bevel_shadow_rgb": list(self.bevel_shadow_rgb),
            "bevel_highlight_rgb": list(self.bevel_highlight_rgb),
            "drop_shadow_offset_couch_xy": list(self.drop_shadow_offset_couch_xy),
            "drop_shadow_sigma_couch": self.drop_shadow_sigma_couch,
            "drop_shadow_opacity": self.drop_shadow_opacity,
            "glass_warm_tint_rgb": list(self.glass_warm_tint_rgb),
            "glass_contrast": self.glass_contrast,
            "recess_depth": self.recess_depth,
            "moulding_relief_norm": self.moulding_relief_norm,
            "matboard_relief_norm": self.matboard_relief_norm,
            "frame_min_component_px": self.frame_min_component_px,
            "badge_offset_y_factor": self.badge_offset_y_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FramedPaintingPortalConfig:
        ob = data["couch_outer_box_xyxy"]
        obr = data["oak_base_rgb"]
        olr = data["oak_light_rgb"]
        odr = data["oak_dark_rgb"]
        cmr = data["couch_mat_rgb"]
        dmr = data["dining_mat_rgb"]
        bsr = data["bevel_shadow_rgb"]
        bhr = data["bevel_highlight_rgb"]
        dso = data["drop_shadow_offset_couch_xy"]
        gwt = data["glass_warm_tint_rgb"]
        return cls(
            target_scene_id=str(data["target_scene_id"]),
            portal_id_in_dining=str(data["portal_id_in_dining"]),
            portal_id_in_couch=str(data["portal_id_in_couch"]),
            label_he=str(data["label_he"]),
            short_badge_he=str(data["short_badge_he"]),
            source_bg_relpath=str(data["source_bg_relpath"]),
            source_mid_relpath=str(data["source_mid_relpath"]),
            source_fg_relpath=str(data["source_fg_relpath"]),
            couch_outer_box_xyxy=(int(ob[0]), int(ob[1]), int(ob[2]), int(ob[3])),
            couch_moulding_px=int(data["couch_moulding_px"]),
            couch_matboard_px=int(data["couch_matboard_px"]),
            couch_bevel_px=int(data["couch_bevel_px"]),
            oak_base_rgb=(int(obr[0]), int(obr[1]), int(obr[2])),
            oak_light_rgb=(int(olr[0]), int(olr[1]), int(olr[2])),
            oak_dark_rgb=(int(odr[0]), int(odr[1]), int(odr[2])),
            couch_mat_rgb=(int(cmr[0]), int(cmr[1]), int(cmr[2])),
            dining_mat_rgb=(int(dmr[0]), int(dmr[1]), int(dmr[2])),
            bevel_shadow_rgb=(int(bsr[0]), int(bsr[1]), int(bsr[2])),
            bevel_highlight_rgb=(int(bhr[0]), int(bhr[1]), int(bhr[2])),
            drop_shadow_offset_couch_xy=(int(dso[0]), int(dso[1])),
            drop_shadow_sigma_couch=float(data["drop_shadow_sigma_couch"]),
            drop_shadow_opacity=float(data["drop_shadow_opacity"]),
            glass_warm_tint_rgb=(float(gwt[0]), float(gwt[1]), float(gwt[2])),
            glass_contrast=float(data["glass_contrast"]),
            recess_depth=float(data["recess_depth"]),
            moulding_relief_norm=float(data["moulding_relief_norm"]),
            matboard_relief_norm=float(data["matboard_relief_norm"]),
            frame_min_component_px=int(data["frame_min_component_px"]),
            badge_offset_y_factor=float(data["badge_offset_y_factor"]),
        )


@dataclass(frozen=True)
class PlantedCouchTrioConfig:
    """Homography, daylight harmonization, and sofa-anchored 3D depth for Tal, Ya'ara & Carmel."""

    layer_id: str
    label_he: str
    depth_output_filename: str
    depth_solve: ContactAnchoredLayerDepthConfig
    protrusion: SourceProtrusionReliefConfig
    segments_x: int
    segments_y: int
    render_order: int
    min_warped_component_px: int
    people_plane_pairs: tuple[PointCorrespondencePair, ...]
    relight_blend: float
    gain_clamp_range: tuple[float, float]
    sofa_shadow_rgb: tuple[int, int, int]
    sofa_shadow_offset_xy: tuple[int, int]
    sofa_shadow_sigma_yx: tuple[float, float]
    sofa_shadow_opacity: float
    wall_shadow_offset_xy: tuple[int, int]
    wall_shadow_sigma_yx: tuple[float, float]
    wall_shadow_opacity: float
    portal_id_to_couch: str
    portal_label_he: str
    portal_short_badge_he: str
    portal_inner_box_xyxy: tuple[float, float, float, float]
    portal_fade_start_progress: float
    badge_offset_y_factor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "label_he": self.label_he,
            "depth_output_filename": self.depth_output_filename,
            "depth_solve": self.depth_solve.to_dict(),
            "protrusion": self.protrusion.to_dict(),
            "segments_x": self.segments_x,
            "segments_y": self.segments_y,
            "render_order": self.render_order,
            "min_warped_component_px": self.min_warped_component_px,
            "people_plane_pairs": [p.to_dict() for p in self.people_plane_pairs],
            "relight_blend": self.relight_blend,
            "gain_clamp_range": list(self.gain_clamp_range),
            "sofa_shadow_rgb": list(self.sofa_shadow_rgb),
            "sofa_shadow_offset_xy": list(self.sofa_shadow_offset_xy),
            "sofa_shadow_sigma_yx": list(self.sofa_shadow_sigma_yx),
            "sofa_shadow_opacity": self.sofa_shadow_opacity,
            "wall_shadow_offset_xy": list(self.wall_shadow_offset_xy),
            "wall_shadow_sigma_yx": list(self.wall_shadow_sigma_yx),
            "wall_shadow_opacity": self.wall_shadow_opacity,
            "portal_id_to_couch": self.portal_id_to_couch,
            "portal_label_he": self.portal_label_he,
            "portal_short_badge_he": self.portal_short_badge_he,
            "portal_inner_box_xyxy": list(self.portal_inner_box_xyxy),
            "portal_fade_start_progress": self.portal_fade_start_progress,
            "badge_offset_y_factor": self.badge_offset_y_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlantedCouchTrioConfig:
        gcr = data["gain_clamp_range"]
        ssr = data["sofa_shadow_rgb"]
        sso = data["sofa_shadow_offset_xy"]
        sss = data["sofa_shadow_sigma_yx"]
        wso = data["wall_shadow_offset_xy"]
        wss = data["wall_shadow_sigma_yx"]
        pib = data["portal_inner_box_xyxy"]
        return cls(
            layer_id=str(data["layer_id"]),
            label_he=str(data["label_he"]),
            depth_output_filename=str(data["depth_output_filename"]),
            depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["depth_solve"]),
            protrusion=SourceProtrusionReliefConfig.from_dict(data["protrusion"]),
            segments_x=int(data["segments_x"]),
            segments_y=int(data["segments_y"]),
            render_order=int(data["render_order"]),
            min_warped_component_px=int(data["min_warped_component_px"]),
            people_plane_pairs=tuple(
                PointCorrespondencePair.from_dict(p) for p in data["people_plane_pairs"]
            ),
            relight_blend=float(data["relight_blend"]),
            gain_clamp_range=(float(gcr[0]), float(gcr[1])),
            sofa_shadow_rgb=(int(ssr[0]), int(ssr[1]), int(ssr[2])),
            sofa_shadow_offset_xy=(int(sso[0]), int(sso[1])),
            sofa_shadow_sigma_yx=(float(sss[0]), float(sss[1])),
            sofa_shadow_opacity=float(data["sofa_shadow_opacity"]),
            wall_shadow_offset_xy=(int(wso[0]), int(wso[1])),
            wall_shadow_sigma_yx=(float(wss[0]), float(wss[1])),
            wall_shadow_opacity=float(data["wall_shadow_opacity"]),
            portal_id_to_couch=str(data["portal_id_to_couch"]),
            portal_label_he=str(data["portal_label_he"]),
            portal_short_badge_he=str(data["portal_short_badge_he"]),
            portal_inner_box_xyxy=(
                float(pib[0]),
                float(pib[1]),
                float(pib[2]),
                float(pib[3]),
            ),
            portal_fade_start_progress=float(data["portal_fade_start_progress"]),
            badge_offset_y_factor=float(data["badge_offset_y_factor"]),
        )


@dataclass(frozen=True)
class DenyaDiningViewSceneConfig:
    """Scene configuration for `denya_living_room` (`IMG_DROP_0106` dining-table portal view)."""

    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    exhibition_group: str
    catalog_id: str
    source_relpath: str
    bg_output_filename: str
    depth_output_filename: str
    mid_output_filename: str
    mid_depth_output_filename: str
    fg_output_filename: str
    fg_depth_output_filename: str
    planted_people_output_filename: str
    frame_output_filename: str
    frame_depth_output_filename: str
    fg_sam_prompts: tuple[SamPointGroupPrompt, ...]
    fg_exclude_boxes: tuple[BoundingBoxPrompt, ...]
    fg_max_pinhole_fill_px: int
    fg_min_component_px: int
    bottom_fg_extension_x0: int
    right_fg_extension_y0: int
    mid_sam_prompts: tuple[SamPointGroupPrompt, ...]
    mid_max_pinhole_fill_px: int
    mid_min_component_px: int
    mid_apron: DownwardOcclusionApronConfig
    sofa_ref_box: BoundingBoxPrompt
    perched_backrest_pillow_box: BoundingBoxPrompt
    wall_flatten_pin: WallDepthFlattenPinConfig
    da2_gaussian_sigma: float
    room_depth_clean: RoomDepthCleanConfig
    frame_depth_solve: ContactAnchoredLayerDepthConfig
    mid_depth_solve: ContactAnchoredLayerDepthConfig
    mid_protrusion: SourceProtrusionReliefConfig
    fg_depth_solve: ContactAnchoredLayerDepthConfig
    fg_protrusion: SourceProtrusionReliefConfig
    scene_bg_z: float
    room_depth_amplitude: float
    bg_segments_x: int
    bg_segments_y: int
    frame_segments_x: int
    frame_segments_y: int
    mid_segments_x: int
    mid_segments_y: int
    fg_segments_x: int
    fg_segments_y: int
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
    frame_render_order: int
    mid_render_order: int
    fg_render_order: int
    fg_bleed_guard_min_z: float
    particle_color_hex: int
    particle_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "exhibition_group": self.exhibition_group,
            "catalog_id": self.catalog_id,
            "source_relpath": self.source_relpath,
            "bg_output_filename": self.bg_output_filename,
            "depth_output_filename": self.depth_output_filename,
            "mid_output_filename": self.mid_output_filename,
            "mid_depth_output_filename": self.mid_depth_output_filename,
            "fg_output_filename": self.fg_output_filename,
            "fg_depth_output_filename": self.fg_depth_output_filename,
            "planted_people_output_filename": self.planted_people_output_filename,
            "frame_output_filename": self.frame_output_filename,
            "frame_depth_output_filename": self.frame_depth_output_filename,
            "fg_sam_prompts": [p.to_dict() for p in self.fg_sam_prompts],
            "fg_exclude_boxes": [b.to_dict() for b in self.fg_exclude_boxes],
            "fg_max_pinhole_fill_px": self.fg_max_pinhole_fill_px,
            "fg_min_component_px": self.fg_min_component_px,
            "bottom_fg_extension_x0": self.bottom_fg_extension_x0,
            "right_fg_extension_y0": self.right_fg_extension_y0,
            "mid_sam_prompts": [p.to_dict() for p in self.mid_sam_prompts],
            "mid_max_pinhole_fill_px": self.mid_max_pinhole_fill_px,
            "mid_min_component_px": self.mid_min_component_px,
            "mid_apron": self.mid_apron.to_dict(),
            "sofa_ref_box": self.sofa_ref_box.to_dict(),
            "perched_backrest_pillow_box": self.perched_backrest_pillow_box.to_dict(),
            "wall_flatten_pin": self.wall_flatten_pin.to_dict(),
            "da2_gaussian_sigma": self.da2_gaussian_sigma,
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "frame_depth_solve": self.frame_depth_solve.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "mid_protrusion": self.mid_protrusion.to_dict(),
            "fg_depth_solve": self.fg_depth_solve.to_dict(),
            "fg_protrusion": self.fg_protrusion.to_dict(),
            "scene_bg_z": self.scene_bg_z,
            "room_depth_amplitude": self.room_depth_amplitude,
            "bg_segments_x": self.bg_segments_x,
            "bg_segments_y": self.bg_segments_y,
            "frame_segments_x": self.frame_segments_x,
            "frame_segments_y": self.frame_segments_y,
            "mid_segments_x": self.mid_segments_x,
            "mid_segments_y": self.mid_segments_y,
            "fg_segments_x": self.fg_segments_x,
            "fg_segments_y": self.fg_segments_y,
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
            "frame_render_order": self.frame_render_order,
            "mid_render_order": self.mid_render_order,
            "fg_render_order": self.fg_render_order,
            "fg_bleed_guard_min_z": self.fg_bleed_guard_min_z,
            "particle_color_hex": self.particle_color_hex,
            "particle_count": self.particle_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DenyaDiningViewSceneConfig:
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            exhibition_group=str(data["exhibition_group"]),
            catalog_id=str(data["catalog_id"]),
            source_relpath=str(data["source_relpath"]),
            bg_output_filename=str(data["bg_output_filename"]),
            depth_output_filename=str(data["depth_output_filename"]),
            mid_output_filename=str(data["mid_output_filename"]),
            mid_depth_output_filename=str(data["mid_depth_output_filename"]),
            fg_output_filename=str(data["fg_output_filename"]),
            fg_depth_output_filename=str(data["fg_depth_output_filename"]),
            planted_people_output_filename=str(data["planted_people_output_filename"]),
            frame_output_filename=str(data["frame_output_filename"]),
            frame_depth_output_filename=str(data["frame_depth_output_filename"]),
            fg_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["fg_sam_prompts"]),
            fg_exclude_boxes=tuple(BoundingBoxPrompt.from_dict(b) for b in data["fg_exclude_boxes"]),
            fg_max_pinhole_fill_px=int(data["fg_max_pinhole_fill_px"]),
            fg_min_component_px=int(data["fg_min_component_px"]),
            bottom_fg_extension_x0=int(data["bottom_fg_extension_x0"]),
            right_fg_extension_y0=int(data["right_fg_extension_y0"]),
            mid_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["mid_sam_prompts"]
            ),
            mid_max_pinhole_fill_px=int(data["mid_max_pinhole_fill_px"]),
            mid_min_component_px=int(data["mid_min_component_px"]),
            mid_apron=DownwardOcclusionApronConfig.from_dict(data["mid_apron"]),
            sofa_ref_box=BoundingBoxPrompt.from_dict(data["sofa_ref_box"]),
            perched_backrest_pillow_box=BoundingBoxPrompt.from_dict(
                data["perched_backrest_pillow_box"]
            ),
            wall_flatten_pin=WallDepthFlattenPinConfig.from_dict(data["wall_flatten_pin"]),
            da2_gaussian_sigma=float(data["da2_gaussian_sigma"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            frame_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["frame_depth_solve"]),
            mid_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_depth_solve"]),
            mid_protrusion=SourceProtrusionReliefConfig.from_dict(data["mid_protrusion"]),
            fg_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["fg_depth_solve"]),
            fg_protrusion=SourceProtrusionReliefConfig.from_dict(data["fg_protrusion"]),
            scene_bg_z=float(data["scene_bg_z"]),
            room_depth_amplitude=float(data["room_depth_amplitude"]),
            bg_segments_x=int(data["bg_segments_x"]),
            bg_segments_y=int(data["bg_segments_y"]),
            frame_segments_x=int(data["frame_segments_x"]),
            frame_segments_y=int(data["frame_segments_y"]),
            mid_segments_x=int(data["mid_segments_x"]),
            mid_segments_y=int(data["mid_segments_y"]),
            fg_segments_x=int(data["fg_segments_x"]),
            fg_segments_y=int(data["fg_segments_y"]),
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
            frame_render_order=int(data["frame_render_order"]),
            mid_render_order=int(data["mid_render_order"]),
            fg_render_order=int(data["fg_render_order"]),
            fg_bleed_guard_min_z=float(data["fg_bleed_guard_min_z"]),
            particle_color_hex=int(data["particle_color_hex"]),
            particle_count=int(data["particle_count"]),
        )


@dataclass(frozen=True)
class DenyaCouchViewSceneConfig:
    """Scene configuration for `denya_couch_guitar` (`IMG_DROP_0104` gray-couch guitar view)."""

    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    exhibition_group: str
    catalog_id: str
    source_relpath: str
    bg_output_filename: str
    depth_output_filename: str
    mid_output_filename: str
    mid_depth_output_filename: str
    fg_output_filename: str
    fg_depth_output_filename: str
    frame_output_filename: str
    frame_depth_output_filename: str
    people_sam_prompts: tuple[SamPointGroupPrompt, ...]
    people_exclude_boxes: tuple[BoundingBoxPrompt, ...]
    people_max_pinhole_fill_px: int
    people_min_component_px: int
    people_head_salient_max_y: int
    coffee_table_full_sam_prompts: tuple[SamPointGroupPrompt, ...]
    coffee_table_morph_close_px: int
    coffee_table_edge_feather_sigma: float
    coffee_table_max_pinhole_fill_px: int
    coffee_table_min_component_px: int
    bottom_chair_tips_boxes_full: tuple[BoundingBoxPrompt, ...]
    mid_apron: DownwardOcclusionApronConfig
    wall_flatten_pin: WallDepthFlattenPinConfig
    da2_gaussian_sigma: float
    room_depth_clean: RoomDepthCleanConfig
    frame_depth_solve: ContactAnchoredLayerDepthConfig
    mid_depth_solve: ContactAnchoredLayerDepthConfig
    mid_protrusion: SourceProtrusionReliefConfig
    fg_depth_solve: ContactAnchoredLayerDepthConfig
    fg_protrusion: SourceProtrusionReliefConfig
    scene_bg_z: float
    room_depth_amplitude: float
    bg_segments_x: int
    bg_segments_y: int
    frame_segments_x: int
    frame_segments_y: int
    mid_segments_x: int
    mid_segments_y: int
    fg_segments_x: int
    fg_segments_y: int
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
    frame_render_order: int
    mid_render_order: int
    fg_render_order: int
    return_portal_id: str
    return_portal_label_he: str
    return_portal_short_badge_he: str
    return_portal_inner_box_xyxy: tuple[float, float, float, float]
    particle_color_hex: int
    particle_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "exhibition_group": self.exhibition_group,
            "catalog_id": self.catalog_id,
            "source_relpath": self.source_relpath,
            "bg_output_filename": self.bg_output_filename,
            "depth_output_filename": self.depth_output_filename,
            "mid_output_filename": self.mid_output_filename,
            "mid_depth_output_filename": self.mid_depth_output_filename,
            "fg_output_filename": self.fg_output_filename,
            "fg_depth_output_filename": self.fg_depth_output_filename,
            "frame_output_filename": self.frame_output_filename,
            "frame_depth_output_filename": self.frame_depth_output_filename,
            "people_sam_prompts": [p.to_dict() for p in self.people_sam_prompts],
            "people_exclude_boxes": [b.to_dict() for b in self.people_exclude_boxes],
            "people_max_pinhole_fill_px": self.people_max_pinhole_fill_px,
            "people_min_component_px": self.people_min_component_px,
            "people_head_salient_max_y": self.people_head_salient_max_y,
            "coffee_table_full_sam_prompts": [
                p.to_dict() for p in self.coffee_table_full_sam_prompts
            ],
            "coffee_table_morph_close_px": self.coffee_table_morph_close_px,
            "coffee_table_edge_feather_sigma": self.coffee_table_edge_feather_sigma,
            "coffee_table_max_pinhole_fill_px": self.coffee_table_max_pinhole_fill_px,
            "coffee_table_min_component_px": self.coffee_table_min_component_px,
            "bottom_chair_tips_boxes_full": [b.to_dict() for b in self.bottom_chair_tips_boxes_full],
            "mid_apron": self.mid_apron.to_dict(),
            "wall_flatten_pin": self.wall_flatten_pin.to_dict(),
            "da2_gaussian_sigma": self.da2_gaussian_sigma,
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "frame_depth_solve": self.frame_depth_solve.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "mid_protrusion": self.mid_protrusion.to_dict(),
            "fg_depth_solve": self.fg_depth_solve.to_dict(),
            "fg_protrusion": self.fg_protrusion.to_dict(),
            "scene_bg_z": self.scene_bg_z,
            "room_depth_amplitude": self.room_depth_amplitude,
            "bg_segments_x": self.bg_segments_x,
            "bg_segments_y": self.bg_segments_y,
            "frame_segments_x": self.frame_segments_x,
            "frame_segments_y": self.frame_segments_y,
            "mid_segments_x": self.mid_segments_x,
            "mid_segments_y": self.mid_segments_y,
            "fg_segments_x": self.fg_segments_x,
            "fg_segments_y": self.fg_segments_y,
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
            "frame_render_order": self.frame_render_order,
            "mid_render_order": self.mid_render_order,
            "fg_render_order": self.fg_render_order,
            "return_portal_id": self.return_portal_id,
            "return_portal_label_he": self.return_portal_label_he,
            "return_portal_short_badge_he": self.return_portal_short_badge_he,
            "return_portal_inner_box_xyxy": list(self.return_portal_inner_box_xyxy),
            "particle_color_hex": self.particle_color_hex,
            "particle_count": self.particle_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DenyaCouchViewSceneConfig:
        rpb = data["return_portal_inner_box_xyxy"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            exhibition_group=str(data["exhibition_group"]),
            catalog_id=str(data["catalog_id"]),
            source_relpath=str(data["source_relpath"]),
            bg_output_filename=str(data["bg_output_filename"]),
            depth_output_filename=str(data["depth_output_filename"]),
            mid_output_filename=str(data["mid_output_filename"]),
            mid_depth_output_filename=str(data["mid_depth_output_filename"]),
            fg_output_filename=str(data["fg_output_filename"]),
            fg_depth_output_filename=str(data["fg_depth_output_filename"]),
            frame_output_filename=str(data["frame_output_filename"]),
            frame_depth_output_filename=str(data["frame_depth_output_filename"]),
            people_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["people_sam_prompts"]
            ),
            people_exclude_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["people_exclude_boxes"]
            ),
            people_max_pinhole_fill_px=int(data["people_max_pinhole_fill_px"]),
            people_min_component_px=int(data["people_min_component_px"]),
            people_head_salient_max_y=int(data["people_head_salient_max_y"]),
            coffee_table_full_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["coffee_table_full_sam_prompts"]
            ),
            coffee_table_morph_close_px=int(data["coffee_table_morph_close_px"]),
            coffee_table_edge_feather_sigma=float(data["coffee_table_edge_feather_sigma"]),
            coffee_table_max_pinhole_fill_px=int(data["coffee_table_max_pinhole_fill_px"]),
            coffee_table_min_component_px=int(data["coffee_table_min_component_px"]),
            bottom_chair_tips_boxes_full=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["bottom_chair_tips_boxes_full"]
            ),
            mid_apron=DownwardOcclusionApronConfig.from_dict(data["mid_apron"]),
            wall_flatten_pin=WallDepthFlattenPinConfig.from_dict(data["wall_flatten_pin"]),
            da2_gaussian_sigma=float(data["da2_gaussian_sigma"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            frame_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["frame_depth_solve"]),
            mid_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_depth_solve"]),
            mid_protrusion=SourceProtrusionReliefConfig.from_dict(data["mid_protrusion"]),
            fg_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["fg_depth_solve"]),
            fg_protrusion=SourceProtrusionReliefConfig.from_dict(data["fg_protrusion"]),
            scene_bg_z=float(data["scene_bg_z"]),
            room_depth_amplitude=float(data["room_depth_amplitude"]),
            bg_segments_x=int(data["bg_segments_x"]),
            bg_segments_y=int(data["bg_segments_y"]),
            frame_segments_x=int(data["frame_segments_x"]),
            frame_segments_y=int(data["frame_segments_y"]),
            mid_segments_x=int(data["mid_segments_x"]),
            mid_segments_y=int(data["mid_segments_y"]),
            fg_segments_x=int(data["fg_segments_x"]),
            fg_segments_y=int(data["fg_segments_y"]),
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
            frame_render_order=int(data["frame_render_order"]),
            mid_render_order=int(data["mid_render_order"]),
            fg_render_order=int(data["fg_render_order"]),
            return_portal_id=str(data["return_portal_id"]),
            return_portal_label_he=str(data["return_portal_label_he"]),
            return_portal_short_badge_he=str(data["return_portal_short_badge_he"]),
            return_portal_inner_box_xyxy=(
                float(rpb[0]),
                float(rpb[1]),
                float(rpb[2]),
                float(rpb[3]),
            ),
            particle_color_hex=int(data["particle_color_hex"]),
            particle_count=int(data["particle_count"]),
        )


@dataclass(frozen=True)
class DenyaLivingRoomBundleConfig:
    """Root SSOT configuration bundle for the Denya living-room portal pair (#19)."""

    feedback_id: str
    spec_output_relpath: str
    dimensions: CanvasDimensionsConfig
    outpainting: OutpaintingConfig
    progressive_lama: ProgressiveLamaOutpaintConfig
    depth_model: DepthModelConfig
    segmentation: SegmentationConfig
    salient_edge_support_px: int
    alpha_clean_threshold_u8: int
    interior_solidify_erode_px: int
    interior_solidify_threshold_u8: int
    inpainting: InpaintingConfig
    topology_verification: LayerTopologyVerificationConfig
    wall_homography_pairs: tuple[PointCorrespondencePair, ...]
    wall_color_ref_box_couch: BoundingBoxPrompt
    wall_color_ref_box_dining: BoundingBoxPrompt
    framed_painting: FramedPaintingPortalConfig
    planted_trio: PlantedCouchTrioConfig
    dining_scene: DenyaDiningViewSceneConfig
    couch_scene: DenyaCouchViewSceneConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "spec_output_relpath": self.spec_output_relpath,
            "dimensions": self.dimensions.to_dict(),
            "outpainting": self.outpainting.to_dict(),
            "progressive_lama": self.progressive_lama.to_dict(),
            "depth_model": self.depth_model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "salient_edge_support_px": self.salient_edge_support_px,
            "alpha_clean_threshold_u8": self.alpha_clean_threshold_u8,
            "interior_solidify_erode_px": self.interior_solidify_erode_px,
            "interior_solidify_threshold_u8": self.interior_solidify_threshold_u8,
            "inpainting": self.inpainting.to_dict(),
            "topology_verification": self.topology_verification.to_dict(),
            "wall_homography_pairs": [p.to_dict() for p in self.wall_homography_pairs],
            "wall_color_ref_box_couch": self.wall_color_ref_box_couch.to_dict(),
            "wall_color_ref_box_dining": self.wall_color_ref_box_dining.to_dict(),
            "framed_painting": self.framed_painting.to_dict(),
            "planted_trio": self.planted_trio.to_dict(),
            "dining_scene": self.dining_scene.to_dict(),
            "couch_scene": self.couch_scene.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DenyaLivingRoomBundleConfig:
        return cls(
            feedback_id=str(data["feedback_id"]),
            spec_output_relpath=str(data["spec_output_relpath"]),
            dimensions=CanvasDimensionsConfig.from_dict(data["dimensions"]),
            outpainting=OutpaintingConfig.from_dict(data["outpainting"]),
            progressive_lama=ProgressiveLamaOutpaintConfig.from_dict(data["progressive_lama"]),
            depth_model=DepthModelConfig.from_dict(data["depth_model"]),
            segmentation=SegmentationConfig.from_dict(data["segmentation"]),
            salient_edge_support_px=int(data["salient_edge_support_px"]),
            alpha_clean_threshold_u8=int(data["alpha_clean_threshold_u8"]),
            interior_solidify_erode_px=int(data["interior_solidify_erode_px"]),
            interior_solidify_threshold_u8=int(data["interior_solidify_threshold_u8"]),
            inpainting=InpaintingConfig.from_dict(data["inpainting"]),
            topology_verification=LayerTopologyVerificationConfig.from_dict(
                data["topology_verification"]
            ),
            wall_homography_pairs=tuple(
                PointCorrespondencePair.from_dict(p) for p in data["wall_homography_pairs"]
            ),
            wall_color_ref_box_couch=BoundingBoxPrompt.from_dict(data["wall_color_ref_box_couch"]),
            wall_color_ref_box_dining=BoundingBoxPrompt.from_dict(
                data["wall_color_ref_box_dining"]
            ),
            framed_painting=FramedPaintingPortalConfig.from_dict(data["framed_painting"]),
            planted_trio=PlantedCouchTrioConfig.from_dict(data["planted_trio"]),
            dining_scene=DenyaDiningViewSceneConfig.from_dict(data["dining_scene"]),
            couch_scene=DenyaCouchViewSceneConfig.from_dict(data["couch_scene"]),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> DenyaLivingRoomBundleConfig:
        return cls.from_dict(json.loads(payload))


def _sam_group(
    box: tuple[int, int, int, int],
    pos: Sequence[tuple[int, int]],
    neg: Sequence[tuple[int, int]] = (),
) -> SamPointGroupPrompt:
    pts = tuple(pos) + tuple(neg)
    labels = tuple([1] * len(pos) + [0] * len(neg))
    return SamPointGroupPrompt(BoundingBoxPrompt(*box), pts, labels)


def default_denya_living_room_bundle_config() -> DenyaLivingRoomBundleConfig:
    """Returns canonical SSOT configuration for `denya_living_room` and `denya_couch_guitar`."""
    wall_pairs = (
        PointCorrespondencePair((192.0, 107.0), (465.0, 208.0)),
        PointCorrespondencePair((79.0, 101.0), (427.0, 207.0)),
        PointCorrespondencePair((362.0, 40.0), (517.0, 173.0)),
        PointCorrespondencePair((472.0, 49.0), (566.0, 174.0)),
        PointCorrespondencePair((364.0, 88.0), (521.0, 194.0)),
        PointCorrespondencePair((472.0, 89.0), (567.0, 195.0)),
        PointCorrespondencePair((486.0, 235.0), (571.0, 260.0)),
        PointCorrespondencePair((384.0, 237.0), (526.0, 259.0)),
        PointCorrespondencePair((628.0, 38.0), (628.0, 166.0)),
        PointCorrespondencePair((684.0, 40.0), (653.0, 167.0)),
        PointCorrespondencePair((683.0, 122.0), (653.0, 204.0)),
        PointCorrespondencePair((627.0, 120.0), (628.0, 203.0)),
        PointCorrespondencePair((771.0, 39.0), (692.0, 160.0)),
        PointCorrespondencePair((895.0, 42.0), (750.0, 159.0)),
        PointCorrespondencePair((886.0, 146.0), (750.0, 208.0)),
        PointCorrespondencePair((768.0, 144.0), (693.0, 208.0)),
        PointCorrespondencePair((987.0, 32.0), (801.0, 152.0)),
        PointCorrespondencePair((1058.0, 37.0), (839.0, 152.0)),
        PointCorrespondencePair((1051.0, 94.0), (838.0, 180.0)),
        PointCorrespondencePair((981.0, 91.0), (801.0, 180.0)),
        PointCorrespondencePair((981.0, 116.0), (802.0, 192.0)),
        PointCorrespondencePair((1040.0, 119.0), (834.0, 192.0)),
        PointCorrespondencePair((1034.0, 174.0), (834.0, 221.0)),
        PointCorrespondencePair((976.0, 173.0), (802.0, 221.0)),
    )

    people_plane_pairs = (
        PointCorrespondencePair((272.0, 338.0), (492.0, 290.0)),
        PointCorrespondencePair((965.0, 342.0), (785.0, 298.0)),
        PointCorrespondencePair((272.0, 530.0), (490.0, 358.0)),
        PointCorrespondencePair((965.0, 530.0), (786.0, 366.0)),
        PointCorrespondencePair((272.0, 635.0), (486.0, 394.0)),
        PointCorrespondencePair((965.0, 635.0), (786.0, 402.0)),
    )

    room_pin_uv = 0.045
    din_bg_z = -385.0
    din_room_amp = 110.0
    couch_bg_z = -350.0
    couch_room_amp = 102.0

    return DenyaLivingRoomBundleConfig(
        feedback_id="fb-20261030-231405-f5e3f7",
        spec_output_relpath="prototype/assets/scenes/denya_living_bundle_spec.json",
        dimensions=CanvasDimensionsConfig(width=1500, height=1000),
        outpainting=OutpaintingConfig(
            expansion_factor=1.80,
            feather_collar_px=48,
            color_match_band_px=40,
            min_registration_corr=0.92,
        ),
        progressive_lama=ProgressiveLamaOutpaintConfig(
            resized_source_wh=(1500, 1125),
            inner_crop_y0=62,
            working_scale=0.50,
            ring_step_px=72,
            ring_dilation_px=2,
            seam_feather_px=6.0,
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
            min_component_area_px=320,
            sam_morph_close_px=5,
            edge_feather_sigma=0.75,
        ),
        salient_edge_support_px=3,
        alpha_clean_threshold_u8=20,
        interior_solidify_erode_px=4,
        interior_solidify_threshold_u8=96,
        inpainting=InpaintingConfig(
            dilation_radius_px=26,
            mask_blur_sigma=2.5,
            jpeg_quality=92,
        ),
        topology_verification=LayerTopologyVerificationConfig(
            alpha_threshold_u8=20,
            rim_erosion_px=2,
            rim_exterior_probe_px=8,
            max_allowed_contact_dz_world=2.0,
            max_allowed_p95_contact_dz_world=1.2,
            max_allowed_rim_jump_world=2.5,
            host_monotonicity_tolerance_world=2.0,
        ),
        wall_homography_pairs=wall_pairs,
        wall_color_ref_box_couch=BoundingBoxPrompt(500, 145, 600, 248),
        wall_color_ref_box_dining=BoundingBoxPrompt(575, 208, 616, 268),
        framed_painting=FramedPaintingPortalConfig(
            target_scene_id="wandering_drawing",
            portal_id_in_dining="portal_denya_dining_wandering_2",
            portal_id_in_couch="portal_denya_couch_wandering_2",
            label_he="✏️ ׳שיטוט 2׳ על קיר הסלון · איים מרחפים",
            short_badge_he="✏️ שיטוט 2",
            source_bg_relpath="prototype/assets/scenes/scene5_wandering_bg_inpainted.jpg",
            source_mid_relpath="prototype/assets/scenes/scene5_wandering_mid.png",
            source_fg_relpath="prototype/assets/scenes/scene5_wandering_fg.png",
            couch_outer_box_xyxy=(608, 148, 764, 252),
            couch_moulding_px=6,
            couch_matboard_px=9,
            couch_bevel_px=2,
            oak_base_rgb=(152, 125, 94),
            oak_light_rgb=(176, 149, 116),
            oak_dark_rgb=(112, 88, 63),
            couch_mat_rgb=(213, 211, 202),
            dining_mat_rgb=(199, 194, 179),
            bevel_shadow_rgb=(162, 158, 148),
            bevel_highlight_rgb=(228, 226, 216),
            drop_shadow_offset_couch_xy=(-4, 4),
            drop_shadow_sigma_couch=3.2,
            drop_shadow_opacity=0.34,
            glass_warm_tint_rgb=(0.985, 0.978, 0.945),
            glass_contrast=0.94,
            recess_depth=96.0,
            moulding_relief_norm=1.0,
            matboard_relief_norm=0.38,
            frame_min_component_px=200,
            badge_offset_y_factor=-0.68,
        ),
        planted_trio=PlantedCouchTrioConfig(
            layer_id="planted_couch_trio",
            label_he="טל, יערה וכרמל עם הגיטרה על הספה האפורה בסלון דניה",
            depth_output_filename="denya_living_dining_couch_people_depth.png",
            depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=din_bg_z,
                room_depth_amplitude=din_room_amp,
                room_pin_margin_uv=room_pin_uv,
                layer_z=-384.0,
                layer_depth_amplitude=68.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=8,
                min_component_area_px=200,
                contact_boxes=(
                    ContactAnchorBox(480, 350, 825, 435, mode="bottom_band", band_px=6),
                ),
                neural_relief_amplitude=12.0,
                contact_relief_taper_px=8.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.5,
                interior_smooth_sigma_px=3.5,
                overflow_tolerance_world=0.5,
            ),
            protrusion=SourceProtrusionReliefConfig(
                da2_smooth_sigma_px=6.0,
                anchor_baseline_percentile=8.0,
                upper_percentile=98.0,
                core_erode_px=2,
            ),
            segments_x=192,
            segments_y=128,
            render_order=7,
            min_warped_component_px=2500,
            people_plane_pairs=people_plane_pairs,
            relight_blend=0.76,
            gain_clamp_range=(0.86, 1.12),
            sofa_shadow_rgb=(64, 60, 54),
            sofa_shadow_offset_xy=(-4, 4),
            sofa_shadow_sigma_yx=(5.0, 6.0),
            sofa_shadow_opacity=0.32,
            wall_shadow_offset_xy=(-5, 2),
            wall_shadow_sigma_yx=(6.5, 7.5),
            wall_shadow_opacity=0.12,
            portal_id_to_couch="portal_denya_to_couch",
            portal_label_he="🎸 ספת הסלון בדניה · טל, יערה וכרמל עם הגיטרה",
            portal_short_badge_he="🎸 לספה והגיטרה",
            portal_inner_box_xyxy=(435.0, 266.0, 815.0, 422.0),
            portal_fade_start_progress=0.24,
            badge_offset_y_factor=-0.35,
        ),
        dining_scene=DenyaDiningViewSceneConfig(
            scene_id="denya_living_room",
            title_he="ארוחה בסלון דניה — שער אל הספה והגיטרה ואל ׳שיטוט 2׳",
            short_label_he="🍽️ סלון דניה: שולחן ושערים",
            icon="🍽️",
            exhibition_group="bene_beitcha_home",
            catalog_id="IMG_DROP_0106",
            source_relpath="media/drop_folder_imports/denya_2026_08_17_dining_table_tami_gadi_a75e3694.jpg",
            bg_output_filename="denya_living_dining_bg.jpg",
            depth_output_filename="denya_living_dining_depth.png",
            mid_output_filename="denya_living_dining_mid.png",
            mid_depth_output_filename="denya_living_dining_mid_depth.png",
            fg_output_filename="denya_living_dining_fg.png",
            fg_depth_output_filename="denya_living_dining_fg_depth.png",
            planted_people_output_filename="denya_living_dining_couch_people.png",
            frame_output_filename="denya_living_dining_wandering2_frame.png",
            frame_depth_output_filename="denya_living_dining_wandering2_frame_depth.png",
            fg_sam_prompts=(
                # Naama (left black sleeveless top + arm resting behind Carmel's chair)
                _sam_group(
                    (110, 240, 445, 760),
                    [(345, 285), (345, 390), (320, 520), (245, 485), (175, 620), (145, 710)],
                    [(430, 330), (220, 310), (105, 500)],
                ),
                # Ya'ara at far end of dining table (white tee + outstretched left arm)
                _sam_group(
                    (395, 376, 658, 630),
                    [(498, 392), (480, 445), (475, 535), (585, 490), (632, 492), (425, 565)],
                    [(560, 415), (405, 405), (610, 440), (498, 368)],
                ),
                # Carmel at front-left + wooden chair back
                _sam_group(
                    (60, 528, 570, 965),
                    [(285, 580), (295, 665), (245, 760), (465, 695), (150, 595), (115, 785), (125, 865)],
                    [(55, 620), (55, 760), (140, 500)],
                ),
                # Tami (blue shirt)
                _sam_group(
                    (970, 358, 1255, 880),
                    [(1095, 405), (1095, 485), (1125, 575), (1025, 540), (1140, 685), (1095, 765), (1125, 815)],
                    [(945, 420), (960, 515), (1010, 365)],
                ),
                # Gadi (white shirt + dark shorts/lap in right foreground)
                _sam_group(
                    (1105, 262, 1500, 1000),
                    [
                        (1315, 315),
                        (1285, 430),
                        (1320, 690),
                        (1440, 650),
                        (1220, 860),
                        (1420, 920),
                        (1145, 860),
                        (1158, 935),
                        (1175, 985),
                        (1245, 985),
                    ],
                    [(1160, 310), (1155, 430), (1475, 340)],
                ),
                # Beige ceramic mug + water glass on back edge of dining table (strictly below chair rail)
                _sam_group(
                    (732, 466, 786, 535),
                    [(764, 476), (764, 504)],
                    [(760, 456), (715, 472), (800, 472)],
                ),
                # Far-edge white napkin/plate rim just right of the water glass
                _sam_group(
                    (790, 468, 862, 532),
                    [(815, 488), (835, 505), (802, 512)],
                    [(820, 458), (868, 472)],
                ),
                # Bottom-left plaid tablecloth corner tip
                _sam_group(
                    (38, 920, 215, 1000),
                    [(62, 988), (95, 965), (140, 990)],
                    [(42, 918), (28, 955)],
                ),
                # Dining table plaid tablecloth + dishes/glasses/pitcher
                _sam_group(
                    (125, 448, 1210, 1000),
                    [
                        (700, 820),
                        (850, 550),
                        (300, 950),
                        (580, 850),
                        (660, 700),
                        (680, 475),
                        (840, 480),
                        (815, 492),
                        (545, 530),
                        (355, 760),
                        (755, 625),
                        (435, 785),
                        (190, 965),
                        (905, 520),
                    ],
                    [(110, 520), (415, 460), (650, 435), (850, 445), (955, 485)],
                ),
            ),
            fg_exclude_boxes=(
                BoundingBoxPrompt(455, 325, 535, 378),
            ),
            fg_max_pinhole_fill_px=85000,
            fg_min_component_px=10000,
            bottom_fg_extension_x0=65,
            right_fg_extension_y0=490,
            mid_sam_prompts=(
                # Center wooden coffee table & tree-stump stool directly in front of planted couch trio
                _sam_group(
                    (548, 385, 708, 474),
                    [(590, 412), (635, 412), (680, 415), (658, 456)],
                    [(635, 368), (532, 395), (565, 462)],
                ),
                # Center white canvas director's chair directly in front of planted couch trio
                _sam_group(
                    (694, 366, 856, 478),
                    [(772, 412), (842, 382), (712, 425), (772, 456), (825, 452)],
                    [(755, 352), (720, 352), (805, 352), (868, 410)],
                ),
            ),
            mid_max_pinhole_fill_px=4500,
            mid_min_component_px=1500,
            mid_apron=DownwardOcclusionApronConfig(
                downward_extension_px=24,
                max_depth_into_occluder_px=26.0,
                bottom_feather_px=6.0,
                seam_blend_px=4.0,
                alpha_threshold_u8=20,
                occluder_threshold_u8=64,
                inpaint_dilation_px=20,
            ),
            sofa_ref_box=BoundingBoxPrompt(435, 286, 815, 405),
            perched_backrest_pillow_box=BoundingBoxPrompt(625, 236, 700, 305),
            wall_flatten_pin=WallDepthFlattenPinConfig(
                inner_xy=(635.0, 215.0),
                radii_xy=(260.0, 120.0),
                feather_px=95.0,
                target_norm_depth=0.08,
            ),
            da2_gaussian_sigma=12.0,
            room_depth_clean=RoomDepthCleanConfig(
                alpha_threshold_u8=16,
                mask_dilation_px=16,
                extra_clean_boxes_expanded=(),
                preserve_boxes_expanded=(
                    (600, 340, 2100, 805),
                    (600, 340, 985, 950),
                    (985, 765, 1146, 875),
                    (1458, 704, 1630, 965),
                    (1630, 340, 2100, 950),
                ),
                post_smooth_sigma_px=2.5,
                interior_blend_band_px=10.0,
            ),
            frame_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=din_bg_z,
                room_depth_amplitude=din_room_amp,
                room_pin_margin_uv=room_pin_uv,
                layer_z=-385.0,
                layer_depth_amplitude=26.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=4,
                min_component_area_px=100,
                contact_boxes=(
                    ContactAnchorBox(0, 0, 1500, 1000, mode="perimeter_rim", band_px=2),
                ),
                neural_relief_amplitude=1.2,
                contact_relief_taper_px=3.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.5,
                interior_smooth_sigma_px=0.8,
                overflow_tolerance_world=0.5,
            ),
            mid_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=din_bg_z,
                room_depth_amplitude=din_room_amp,
                room_pin_margin_uv=room_pin_uv,
                layer_z=-382.0,
                layer_depth_amplitude=74.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=8,
                min_component_area_px=200,
                contact_boxes=(
                    ContactAnchorBox(545, 440, 860, 512, mode="bottom_band", band_px=6),
                ),
                neural_relief_amplitude=15.0,
                contact_relief_taper_px=8.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.5,
                interior_smooth_sigma_px=1.5,
                overflow_tolerance_world=0.5,
            ),
            mid_protrusion=SourceProtrusionReliefConfig(
                da2_smooth_sigma_px=2.0,
                anchor_baseline_percentile=10.0,
                upper_percentile=98.0,
                core_erode_px=1,
            ),
            fg_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=din_bg_z,
                room_depth_amplitude=din_room_amp,
                room_pin_margin_uv=room_pin_uv,
                layer_z=-374.0,
                layer_depth_amplitude=175.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=8,
                min_component_area_px=500,
                contact_boxes=(
                    ContactAnchorBox(72, 555, 160, 675, mode="left_band", band_px=6),
                    ContactAnchorBox(55, 795, 132, 945, mode="left_band", band_px=6),
                ),
                neural_relief_amplitude=98.0,
                contact_relief_taper_px=12.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.5,
                interior_smooth_sigma_px=2.0,
                overflow_tolerance_world=0.5,
            ),
            fg_protrusion=SourceProtrusionReliefConfig(
                da2_smooth_sigma_px=2.5,
                anchor_baseline_percentile=6.0,
                upper_percentile=98.5,
                core_erode_px=1,
            ),
            scene_bg_z=din_bg_z,
            room_depth_amplitude=din_room_amp,
            bg_segments_x=224,
            bg_segments_y=150,
            frame_segments_x=160,
            frame_segments_y=108,
            mid_segments_x=192,
            mid_segments_y=128,
            fg_segments_x=224,
            fg_segments_y=150,
            min_inner_scale=2.12,
            initial_camera_y=0.0,
            initial_camera_z=460.0,
            min_camera_z=-120.0,
            max_camera_z=640.0,
            max_yaw_deg=18.5,
            max_pitch_deg=12.5,
            max_strafe_x=68.0,
            max_strafe_y=46.0,
            dolly_speed_mult=1.35,
            frame_render_order=6,
            mid_render_order=8,
            fg_render_order=10,
            fg_bleed_guard_min_z=210.0,
            particle_color_hex=0xE8DCC4,
            particle_count=65,
        ),
        couch_scene=DenyaCouchViewSceneConfig(
            scene_id="denya_couch_guitar",
            title_he="ספת הסלון בדניה — נגינה בגיטרה תחת קיר העבודות ושער ׳שיטוט 2׳",
            short_label_he="🎸 ספת דניה: גיטרה ושיטוט",
            icon="🎸",
            exhibition_group="bene_beitcha_home",
            catalog_id="IMG_DROP_0104",
            source_relpath="media/drop_folder_imports/denya_2025_12_27_guitar_living_room_artworks_35f16626.jpg",
            bg_output_filename="denya_living_couch_bg.jpg",
            depth_output_filename="denya_living_couch_depth.png",
            mid_output_filename="denya_living_couch_mid.png",
            mid_depth_output_filename="denya_living_couch_mid_depth.png",
            fg_output_filename="denya_living_couch_fg.png",
            fg_depth_output_filename="denya_living_couch_fg_depth.png",
            frame_output_filename="denya_living_couch_wandering2_frame.png",
            frame_depth_output_filename="denya_living_couch_wandering2_frame_depth.png",
            people_sam_prompts=(
                # Tal seated in profile on left-center of gray couch (`c_in` 1500x1000) — clean nose & glasses silhouette
                _sam_group(
                    (440, 245, 650, 682),
                    [
                        (575, 290),
                        (575, 345),
                        (610, 365),
                        (618, 382),
                        (595, 410),
                        (535, 345),
                        (495, 455),
                        (560, 480),
                        (600, 600),
                        (520, 585),
                        (640, 580),
                    ],
                    [
                        (638, 355),
                        (642, 378),
                        (636, 405),
                        (620, 435),
                        (608, 472),
                        (612, 488),
                        (616, 502),
                        (450, 355),
                        (438, 440),
                        (432, 485),
                        (595, 686),
                        (485, 675),
                    ],
                ),
                # Classical acoustic guitar (body + bridge + full fretboard + headstock)
                _sam_group(
                    (636, 418, 1028, 588),
                    [
                        (700, 450),
                        (712, 485),
                        (700, 548),
                        (775, 495),
                        (865, 486),
                        (920, 484),
                        (968, 482),
                        (1005, 480),
                    ],
                    [(680, 600), (885, 455), (885, 518), (985, 450), (985, 515)],
                ),
                # Ya'ara in rose sweater & pink pants
                _sam_group(
                    (722, 240, 892, 670),
                    [(825, 275), (825, 315), (815, 365), (795, 410), (775, 625), (855, 615)],
                    [(725, 370), (890, 405), (820, 676)],
                ),
                # Carmel in cream ribbed sweater & gray leggings
                _sam_group(
                    (858, 305, 1025, 650),
                    [(965, 382), (955, 422), (935, 500), (930, 545), (915, 608), (968, 600)],
                    [(890, 385), (992, 330), (1030, 480)],
                ),
            ),
            people_exclude_boxes=(
                BoundingBoxPrompt(445, 652, 525, 705),
                BoundingBoxPrompt(400, 405, 452, 520),
                BoundingBoxPrompt(602, 458, 636, 513),
            ),
            people_max_pinhole_fill_px=4500,
            people_min_component_px=2500,
            people_head_salient_max_y=430,
            # Coordinates in FULL 1500x1125 resized photo (y_full = y_inner + 62) so entire coffee table + all objects + legs stay unified
            coffee_table_full_sam_prompts=(
                # Main oak coffee table slab & both wooden legs
                _sam_group(
                    (340, 688, 1320, 1072),
                    [
                        (550, 860),
                        (750, 800),
                        (820, 845),
                        (950, 820),
                        (1080, 875),
                        (380, 960),
                        (370, 995),
                        (530, 1015),
                        (535, 1052),
                        (1135, 925),
                        (1130, 962),
                        (1260, 875),
                        (1295, 860),
                        (410, 760),
                        (660, 745),
                        (800, 735),
                        (965, 720),
                        (1095, 715),
                    ],
                    [
                        (290, 850),
                        (450, 1025),
                        (750, 980),
                        (960, 960),
                        (1045, 975),
                        (1230, 945),
                        (1345, 860),
                        (620, 695),
                        (800, 690),
                    ],
                ),
                # Left woven lidded baskets + coaster on table edge
                _sam_group(
                    (345, 795, 552, 986),
                    [(385, 895), (465, 890), (460, 945), (455, 818)],
                    [(315, 895)],
                ),
                # Open blue/white book + mouse + black phone + clear water glass
                _sam_group(
                    (452, 718, 652, 875),
                    [(485, 742), (535, 805), (595, 732), (622, 745), (622, 785)],
                    [(535, 700)],
                ),
                # Front-center square ceramic coaster
                _sam_group(
                    (655, 838, 736, 906),
                    [(695, 872)],
                    [(635, 872), (755, 872)],
                ),
                # Round terracotta nut bowl at back edge
                _sam_group(
                    (686, 696, 758, 764),
                    [(720, 725), (720, 748)],
                    [(672, 705), (772, 705)],
                ),
                # Back-center-right pair of square ceramic coasters (eliminates tabletop cutout hole!)
                _sam_group(
                    (826, 694, 946, 748),
                    [(862, 716), (912, 715), (885, 736)],
                    [(810, 692), (885, 686), (956, 690)],
                ),
                # Back-right dark-blue fabric pouch
                _sam_group(
                    (1082, 678, 1206, 760),
                    [(1122, 698), (1152, 720), (1176, 742), (1105, 732)],
                    [(1065, 678), (1212, 692), (1185, 680)],
                ),
                # Right-side books, white paper sheet, tissue roll, keys & phone
                _sam_group(
                    (960, 702, 1308, 868),
                    [(1025, 732), (1055, 768), (1135, 765), (1228, 752), (1228, 785), (1262, 825), (1240, 808)],
                    [(1030, 688), (1320, 780), (1275, 735)],
                ),
            ),
            coffee_table_morph_close_px=7,
            coffee_table_edge_feather_sigma=0.85,
            coffee_table_max_pinhole_fill_px=45000,
            coffee_table_min_component_px=10000,
            bottom_chair_tips_boxes_full=(
                BoundingBoxPrompt(350, 1015, 480, 1125),
                BoundingBoxPrompt(735, 1015, 1165, 1125),
                BoundingBoxPrompt(985, 955, 1155, 1125),
            ),
            mid_apron=DownwardOcclusionApronConfig(
                downward_extension_px=48,
                max_depth_into_occluder_px=50.0,
                bottom_feather_px=7.0,
                seam_blend_px=4.0,
                alpha_threshold_u8=20,
                occluder_threshold_u8=64,
                inpaint_dilation_px=22,
            ),
            wall_flatten_pin=WallDepthFlattenPinConfig(
                inner_xy=(650.0, 145.0),
                radii_xy=(520.0, 195.0),
                feather_px=120.0,
                target_norm_depth=0.10,
            ),
            da2_gaussian_sigma=12.0,
            room_depth_clean=RoomDepthCleanConfig(
                alpha_threshold_u8=16,
                mask_dilation_px=18,
                extra_clean_boxes_expanded=(
                    (950, 1353, 1080, 1465),
                    (1335, 1353, 1765, 1465),
                ),
                preserve_boxes_expanded=(
                    (600, 340, 2100, 1045),
                ),
                post_smooth_sigma_px=2.5,
                interior_blend_band_px=10.0,
            ),
            frame_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=couch_bg_z,
                room_depth_amplitude=couch_room_amp,
                room_pin_margin_uv=room_pin_uv,
                layer_z=-348.0,
                layer_depth_amplitude=24.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=4,
                min_component_area_px=100,
                contact_boxes=(
                    ContactAnchorBox(0, 0, 1500, 1000, mode="perimeter_rim", band_px=2),
                ),
                neural_relief_amplitude=1.4,
                contact_relief_taper_px=3.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.5,
                interior_smooth_sigma_px=0.8,
                overflow_tolerance_world=0.5,
            ),
            mid_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=couch_bg_z,
                room_depth_amplitude=couch_room_amp,
                room_pin_margin_uv=room_pin_uv,
                layer_z=-342.0,
                layer_depth_amplitude=82.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=8,
                min_component_area_px=500,
                contact_boxes=(
                    ContactAnchorBox(445, 555, 1025, 725, mode="bottom_band", band_px=8),
                ),
                neural_relief_amplitude=14.0,
                contact_relief_taper_px=12.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.5,
                interior_smooth_sigma_px=4.0,
                overflow_tolerance_world=0.5,
            ),
            mid_protrusion=SourceProtrusionReliefConfig(
                da2_smooth_sigma_px=7.0,
                anchor_baseline_percentile=10.0,
                upper_percentile=98.0,
                core_erode_px=2,
            ),
            fg_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=couch_bg_z,
                room_depth_amplitude=couch_room_amp,
                room_pin_margin_uv=room_pin_uv,
                layer_z=-322.0,
                layer_depth_amplitude=105.0,
                layer_pin_margin_uv=0.0,
                alpha_threshold_u8=20,
                component_bridge_px=8,
                min_component_area_px=500,
                contact_boxes=(
                    ContactAnchorBox(1080, 1320, 1180, 1420, mode="bottom_band", band_px=6),
                    ContactAnchorBox(1680, 1220, 1785, 1328, mode="bottom_band", band_px=6),
                ),
                neural_relief_amplitude=64.0,
                contact_relief_taper_px=10.0,
                enforce_front_of_host=True,
                host_clearance_tolerance_world=0.5,
                interior_smooth_sigma_px=2.0,
                overflow_tolerance_world=0.5,
            ),
            fg_protrusion=SourceProtrusionReliefConfig(
                da2_smooth_sigma_px=2.5,
                anchor_baseline_percentile=8.0,
                upper_percentile=98.5,
                core_erode_px=1,
            ),
            scene_bg_z=couch_bg_z,
            room_depth_amplitude=couch_room_amp,
            bg_segments_x=224,
            bg_segments_y=150,
            frame_segments_x=160,
            frame_segments_y=108,
            mid_segments_x=224,
            mid_segments_y=150,
            fg_segments_x=224,
            fg_segments_y=150,
            min_inner_scale=1.98,
            initial_camera_y=0.0,
            initial_camera_z=455.0,
            min_camera_z=-90.0,
            max_camera_z=650.0,
            max_yaw_deg=28.0,
            max_pitch_deg=20.0,
            max_strafe_x=220.0,
            max_strafe_y=140.0,
            dolly_speed_mult=1.30,
            frame_render_order=6,
            mid_render_order=8,
            fg_render_order=10,
            return_portal_id="portal_denya_back_to_dining",
            return_portal_label_he="🍽️ מבט משולחן האוכל · תמי, גדי והמשפחה",
            return_portal_short_badge_he="🍽️ לשולחן האוכל",
            return_portal_inner_box_xyxy=(540.0, 580.0, 760.0, 648.0),
            particle_color_hex=0xE2D7C5,
            particle_count=60,
        ),
    )


# ============================================================================
# Vision, Homography, Framing & Contact-Anchored Depth Synthesis Primitives
# ============================================================================


def load_source_full_and_inner(
    image_path: Path,
    prog_cfg: ProgressiveLamaOutpaintConfig,
    inner_dims: CanvasDimensionsConfig,
) -> tuple[Image.Image, Image.Image]:
    """Loads EXIF-upright 4:3 photograph, resizes to `1500x1125`, and extracts `1500x1000` inner crop."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing Denya living-room source photograph: {image_path}")
    src = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    rw, rh = prog_cfg.resized_source_wh
    full_rgb = src.resize((rw, rh), Image.Resampling.LANCZOS)
    y0 = prog_cfg.inner_crop_y0
    inner_rgb = full_rgb.crop((0, y0, inner_dims.width, y0 + inner_dims.height))
    return full_rgb, inner_rgb


def compute_homography_from_pairs(pairs: tuple[PointCorrespondencePair, ...]) -> np.ndarray:
    """Computes least-squares planar projective homography `H` (`3x3` float64) from SSOT point pairs."""
    if len(pairs) < 4:
        raise ValueError(f"At least 4 point pairs are required for homography; got {len(pairs)}")
    src_pts = np.float32([p.src_xy for p in pairs])
    dst_pts = np.float32([p.dst_xy for p in pairs])
    H_mat, _ = cv2.findHomography(src_pts, dst_pts, method=0)
    if H_mat is None:
        raise RuntimeError("cv2.findHomography failed to solve planar homography.")
    return H_mat.astype(np.float64)


def fill_enclosed_pinholes_u8(alpha_u8: np.ndarray, max_hole_area_px: int) -> np.ndarray:
    """Fills strictly interior enclosed transparent holes up to `max_hole_area_px`."""
    if max_hole_area_px <= 0:
        return alpha_u8
    fg_bool = alpha_u8 >= 128
    filled_bool = ndi.binary_fill_holes(fg_bool)
    holes_bool = filled_bool & (~fg_bool)
    if not holes_bool.any():
        return alpha_u8
    labeled, num_features = ndi.label(holes_bool)
    if num_features == 0:
        return alpha_u8
    sizes = ndi.sum(holes_bool, labeled, index=range(1, num_features + 1))
    out = alpha_u8.copy()
    for idx, area in enumerate(sizes, start=1):
        if float(area) <= float(max_hole_area_px):
            out[labeled == idx] = 255
    return out


def solidify_interior_alpha_u8(
    alpha_u8: np.ndarray,
    max_hole_area_px: int,
    erode_px: int,
    threshold_u8: int,
) -> np.ndarray:
    """
    Sets strictly interior pixels (more than `erode_px` inside the outer silhouette boundary)
    to fully opaque `255`, eliminating semi-transparent 1–2 px seam speckles between adjacent
    SAM prompt groups while leaving outer feathered silhouette edges untouched.
    """
    filled_u8 = fill_enclosed_pinholes_u8(alpha_u8, max_hole_area_px)
    if erode_px <= 0:
        return filled_u8
    fg_bool = filled_u8 >= threshold_u8
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    interior = cv2.erode(fg_bool.astype(np.uint8), k, iterations=1) > 0
    out = filled_u8.copy()
    out[interior] = 255
    return out


def seal_right_bottom_crop_notches_u8(
    alpha_u8: np.ndarray,
    bottom_x0: int,
    right_y0: int,
    max_hole_area_px: int,
) -> np.ndarray:
    """
    Seals border-touching SAM/IS-Net alpha notches along right (`x = W - 1`, `y >= right_y0`)
    and bottom (`y = H - 1`, `x >= bottom_x0`) crop borders by temporarily closing the crop
    boundary with a 1-pixel solid frame before `fill_enclosed_pinholes_u8`.
    """
    h, w = alpha_u8.shape
    padded = np.zeros((h + 1, w + 1), dtype=np.uint8)
    padded[:h, :w] = alpha_u8
    padded[h, bottom_x0 : w + 1] = 255
    padded[right_y0 : h + 1, w] = 255
    sealed = fill_enclosed_pinholes_u8(padded, max_hole_area_px)
    return sealed[:h, :w]


def filter_min_components_u8(
    alpha_u8: np.ndarray,
    min_area_px: int,
    threshold_u8: int,
) -> np.ndarray:
    """Keeps connected components (`alpha >= threshold_u8`) of size `>= min_area_px`, zeroing dust."""
    fg = alpha_u8 >= threshold_u8
    labeled, num_features = ndi.label(fg)
    if num_features == 0:
        return np.zeros_like(alpha_u8)
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_area_px
    keep[0] = False
    valid_mask = keep[labeled]
    return np.where(valid_mask, alpha_u8, 0).astype(np.uint8)


def refine_alpha_with_salient_edges(
    rgb_image: Image.Image,
    sam_alpha_u8: np.ndarray,
    matte_extractor: SalientMatteExtractor,
    support_px: int,
    edge_feather_sigma: float,
) -> np.ndarray:
    """Adds fine IS-Net hair wisps within `support_px` of SAM boundaries without attenuating SAM masks."""
    sal_u8 = matte_extractor.extract_alpha(rgb_image)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * support_px + 1, 2 * support_px + 1))
    dilated = cv2.dilate(sam_alpha_u8, k, iterations=1)
    sal_near_edge = np.where(dilated > 24, sal_u8, 0)
    combined = np.maximum(sam_alpha_u8, sal_near_edge).astype(np.float32)
    if edge_feather_sigma > 0:
        combined = ndi.gaussian_filter(combined, sigma=edge_feather_sigma * 0.65)
    return np.clip(np.round(combined), 0, 255).astype(np.uint8)


def build_downward_apron_rgba(
    authentic_inner_rgb: Image.Image,
    healed_behind_occluder_inner_rgb: Image.Image,
    base_alpha_u8: np.ndarray,
    occluder_inner_u8: np.ndarray,
    apron_cfg: DownwardOcclusionApronConfig,
    min_component_px: int,
) -> Image.Image:
    """
    Constructs an RGBA mid-layer cutout that is 100% authentic un-inpainted pixels above the
    foreground occluder (`boreSightAtRest: true`) while extending smoothly downward into a thin
    neural-LaMa-healed apron behind the top edge of `occluder_inner_u8` so zero parallax gap can
    ever open under vertical pitch/strafe.
    """
    auth_np = np.asarray(authentic_inner_rgb, dtype=np.uint8)
    healed_np = np.asarray(healed_behind_occluder_inner_rgb, dtype=np.uint8)

    ext = apron_cfg.downward_extension_px
    v_kernel = np.zeros((2 * ext + 1, 5), dtype=np.uint8)
    v_kernel[ext:, :] = 1
    base_fg = (base_alpha_u8 >= apron_cfg.alpha_threshold_u8).astype(np.uint8)
    extended_down = cv2.dilate(base_fg, v_kernel, iterations=1) > 0

    occ_bool = occluder_inner_u8 >= apron_cfg.occluder_threshold_u8
    dist_into_occ = ndi.distance_transform_edt(occ_bool).astype(np.float32)
    dist_outside_occ = ndi.distance_transform_edt(~occ_bool).astype(np.float32)

    apron_taper = np.clip(
        (apron_cfg.max_depth_into_occluder_px - dist_into_occ)
        / max(1.0, apron_cfg.bottom_feather_px),
        0.0,
        1.0,
    )
    apron_alpha_u8 = np.where(
        occ_bool & extended_down & (dist_into_occ <= apron_cfg.max_depth_into_occluder_px),
        np.round(255.0 * apron_taper),
        0.0,
    ).astype(np.uint8)

    unoccluded_alpha_u8 = np.where(~occ_bool, base_alpha_u8, 0).astype(np.uint8)
    combined_alpha_u8 = np.maximum(unoccluded_alpha_u8, apron_alpha_u8)
    combined_alpha_u8 = filter_min_components_u8(
        combined_alpha_u8, min_component_px, apron_cfg.alpha_threshold_u8
    )

    decontam_auth = decontaminate_matte_edge_rgb(auth_np, base_alpha_u8)
    # Blend smoothly from authentic pixels outside occluder to LaMa-healed pixels inside occluder
    w_healed = np.where(
        occ_bool,
        np.clip(dist_into_occ / max(1.0, apron_cfg.seam_blend_px), 0.25, 1.0),
        np.clip(1.0 - dist_outside_occ / max(1.0, apron_cfg.seam_blend_px), 0.0, 0.25),
    )[:, :, None]
    # Keep strictly unoccluded interior (>2px away from occluder) 100% authentic
    w_healed = np.where((~occ_bool)[:, :, None] & (dist_outside_occ[:, :, None] > 2.0), 0.0, w_healed)
    blended_rgb = np.clip(
        np.round((1.0 - w_healed) * decontam_auth.astype(np.float32) + w_healed * healed_np.astype(np.float32)),
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(np.dstack([blended_rgb, combined_alpha_u8]), mode="RGBA")


def compute_source_da2_protrusion_norm(
    raw_source_da2_layer: np.ndarray,
    cutout_rgba: np.ndarray,
    solve_cfg: ContactAnchoredLayerDepthConfig,
    relief_cfg: SourceProtrusionReliefConfig,
) -> np.ndarray:
    """
    Computes normalized `[0, 1]` 3D sculptural/perspective relief from the un-inpainted source
    photograph's Depth Anything V2 prediction relative to the physical contact anchor baseline.
    """
    opaque = cutout_rgba[:, :, 3] >= solve_cfg.alpha_threshold_u8
    if not np.any(opaque):
        return np.zeros(raw_source_da2_layer.shape, dtype=np.float32)

    raw_f = raw_source_da2_layer.astype(np.float32)
    if relief_cfg.core_erode_px > 0:
        er_r = int(relief_cfg.core_erode_px)
        k_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er_r + 1, 2 * er_r + 1))
        core_mask = cv2.erode(opaque.astype(np.uint8), k_core, iterations=1) > 0
        if np.any(core_mask):
            idx_edt = ndi.distance_transform_edt(
                ~core_mask, return_distances=False, return_indices=True
            )
            raw_f = raw_f[idx_edt[0], idx_edt[1]]

    smooth = ndi.gaussian_filter(raw_f, sigma=relief_cfg.da2_smooth_sigma_px)
    d_norm = normalize_depth_array(smooth)

    contact = extract_contact_anchor_mask(opaque, solve_cfg)
    baseline = float(np.percentile(d_norm[contact], relief_cfg.anchor_baseline_percentile))
    excess = np.maximum(0.0, d_norm - baseline)
    scale = float(np.percentile(excess[opaque], relief_cfg.upper_percentile))
    if scale <= 1e-5:
        return np.zeros_like(d_norm, dtype=np.float32)
    return np.clip(excess / scale, 0.0, 1.0).astype(np.float32)


def progressive_lama_outpaint_canvas(
    full_clean_rgb: Image.Image,
    inner_dims: CanvasDimensionsConfig,
    outpaint_cfg: OutpaintingConfig,
    prog_cfg: ProgressiveLamaOutpaintConfig,
    lama_inpainter: LamaOcclusionInpainter,
) -> Image.Image:
    """
    Extends the `1500x1125` clean architectural room plate seamlessly to `2700x1800` via
    progressive multi-ring Fast Fourier Convolution LaMa continuation — zero hallucinated
    objects or synthetic furniture.
    """
    exp_w, exp_h = outpaint_cfg.expanded_dimensions(inner_dims)
    pad_x = (exp_w - inner_dims.width) // 2
    pad_y = (exp_h - inner_dims.height) // 2
    rw, rh = full_clean_rgb.size
    y_start = pad_y - prog_cfg.inner_crop_y0

    canvas = np.zeros((exp_h, exp_w, 3), dtype=np.uint8)
    known_full = np.zeros((exp_h, exp_w), dtype=bool)
    canvas[y_start : y_start + rh, pad_x : pad_x + rw] = np.asarray(full_clean_rgb, dtype=np.uint8)
    known_full[y_start : y_start + rh, pad_x : pad_x + rw] = True

    work_w = int(round(exp_w * prog_cfg.working_scale))
    work_h = int(round(exp_h * prog_cfg.working_scale))
    work_rgb = np.asarray(
        Image.fromarray(canvas, mode="RGB").resize((work_w, work_h), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    ).copy()
    known_work = (
        np.asarray(
            Image.fromarray((known_full.astype(np.uint8) * 255), mode="L").resize(
                (work_w, work_h), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        > 127
    )

    ring = prog_cfg.ring_step_px
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * ring + 1, 2 * ring + 1))
    while not bool(known_work.all()):
        grown = cv2.dilate(known_work.astype(np.uint8), kernel, iterations=1) > 0
        unknown_u8 = np.where(known_work, 0, 255).astype(np.uint8)
        inpainted = np.asarray(
            lama_inpainter.inpaint_background(
                Image.fromarray(work_rgb, mode="RGB"),
                unknown_u8,
                dilation_radius_px=prog_cfg.ring_dilation_px,
                mask_blur_sigma=0.0,
            ),
            dtype=np.uint8,
        )
        new_band = grown & (~known_work)
        work_rgb[new_band] = inpainted[new_band]
        known_work = grown

    upscaled = np.asarray(
        Image.fromarray(work_rgb, mode="RGB").resize((exp_w, exp_h), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    dist_inside = ndi.distance_transform_edt(known_full).astype(np.float32)
    alpha_real = np.clip(dist_inside / max(1.0, prog_cfg.seam_feather_px), 0.0, 1.0)[:, :, None]
    composed = alpha_real * canvas.astype(np.float32) + (1.0 - alpha_real) * upscaled
    return Image.fromarray(np.clip(np.round(composed), 0, 255).astype(np.uint8), mode="RGB")


def synthesize_wandering2_framed_assets(
    project_root: Path,
    fp_cfg: FramedPaintingPortalConfig,
    inner_dims: CanvasDimensionsConfig,
    H_wall_couch_to_dining: np.ndarray,
    alpha_clean_threshold_u8: int,
) -> tuple[
    Image.Image,
    Image.Image,
    np.ndarray,
    Image.Image,
    Image.Image,
    np.ndarray,
    tuple[float, float, float, float],
]:
    """
    Builds the physical natural-oak & cream-matboard `'wandering 2'` picture frame and normalized
    wood-moulding relief maps for BOTH views (`couch` and homography-warped `dining`).
    """
    bg_img = Image.open(project_root / fp_cfg.source_bg_relpath).convert("RGB")
    mid_img = Image.open(project_root / fp_cfg.source_mid_relpath).convert("RGBA")
    fg_img = Image.open(project_root / fp_cfg.source_fg_relpath).convert("RGBA")

    bw, bh = bg_img.size
    px = (bw - inner_dims.width) // 2
    py = (bh - inner_dims.height) // 2
    art_comp = bg_img.crop((px, py, px + inner_dims.width, py + inner_dims.height)).convert("RGBA")
    art_comp.alpha_composite(mid_img)
    art_comp.alpha_composite(fg_img)

    art_np = np.asarray(art_comp.convert("RGB"), dtype=np.float32)
    mean_luma = float(art_np.mean())
    art_np = (art_np - mean_luma) * fp_cfg.glass_contrast + mean_luma
    art_np *= np.array(fp_cfg.glass_warm_tint_rgb, dtype=np.float32)[None, None, :]
    art_rgb_img = Image.fromarray(np.clip(np.round(art_np), 0, 255).astype(np.uint8), mode="RGB")

    ox0, oy0, ox1, oy1 = fp_cfg.couch_outer_box_xyxy
    ix0, iy0, ix1, iy1 = fp_cfg.couch_inner_aperture_xyxy()
    m_px = fp_cfg.couch_moulding_px
    b_px = fp_cfg.couch_bevel_px
    iw, ih = ix1 - ix0, iy1 - iy0

    W, H = inner_dims.width, inner_dims.height
    couch_relief = np.zeros((H, W), dtype=np.float32)
    couch_relief[oy0:oy1, ox0:ox1] = fp_cfg.moulding_relief_norm
    mx0_r, my0_r, mx1_r, my1_r = ox0 + m_px, oy0 + m_px, ox1 - m_px, oy1 - m_px
    couch_relief[my0_r:my1_r, mx0_r:mx1_r] = fp_cfg.matboard_relief_norm
    couch_relief[iy0:iy1, ix0:ix1] = 0.0

    def _build_couch_frame_rgba(mat_rgb: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        box_mask = np.zeros((H, W), dtype=np.float32)
        sx, sy = fp_cfg.drop_shadow_offset_couch_xy
        sy0 = max(0, oy0 + sy)
        sy1 = min(H, oy1 + sy)
        sx0 = max(0, ox0 + sx)
        sx1 = min(W, ox1 + sx)
        box_mask[sy0:sy1, sx0:sx1] = 1.0
        sh_blur = ndi.gaussian_filter(box_mask, sigma=fp_cfg.drop_shadow_sigma_couch)
        sh_blur[oy0:oy1, ox0:ox1] = 0.0
        sh_a = np.clip(sh_blur * fp_cfg.drop_shadow_opacity * 255.0, 0, 255)

        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[:, :, 0] = 42.0
        rgba[:, :, 1] = 40.0
        rgba[:, :, 2] = 36.0
        rgba[:, :, 3] = sh_a

        rgba[oy0:oy1, ox0:ox1, :3] = np.array(fp_cfg.oak_base_rgb, dtype=np.float32)
        rgba[oy0:oy1, ox0:ox1, 3] = 255.0
        rgba[oy0 : oy0 + 2, ox0:ox1, :3] = np.array(fp_cfg.oak_light_rgb, dtype=np.float32)
        rgba[oy0:oy1, ox1 - 2 : ox1, :3] = np.array(fp_cfg.oak_light_rgb, dtype=np.float32)
        rgba[oy0:oy1, ox0 : ox0 + 2, :3] = np.array(fp_cfg.oak_dark_rgb, dtype=np.float32)
        rgba[oy1 - 2 : oy1, ox0:ox1, :3] = np.array(fp_cfg.oak_dark_rgb, dtype=np.float32)

        mx0, my0, mx1, my1 = ox0 + m_px, oy0 + m_px, ox1 - m_px, oy1 - m_px
        rgba[my0:my1, mx0:mx1, :3] = np.array(mat_rgb, dtype=np.float32)
        rgba[my0:my1, mx0:mx1, 3] = 255.0

        bx0, by0, bx1, by1 = ix0 - b_px, iy0 - b_px, ix1 + b_px, iy1 + b_px
        rgba[by0:iy0, bx0:bx1, :3] = np.array(fp_cfg.bevel_shadow_rgb, dtype=np.float32)
        rgba[by0:by1, bx1 - b_px : bx1, :3] = np.array(fp_cfg.bevel_shadow_rgb, dtype=np.float32)
        rgba[iy1:by1, bx0:bx1, :3] = np.array(fp_cfg.bevel_highlight_rgb, dtype=np.float32)
        rgba[by0:by1, bx0:ix0, :3] = np.array(fp_cfg.bevel_highlight_rgb, dtype=np.float32)

        frame_hole = rgba.copy()
        frame_hole[iy0:iy1, ix0:ix1, :] = 0.0

        art_patch = np.asarray(
            art_rgb_img.resize((iw, ih), Image.Resampling.LANCZOS), dtype=np.float32
        )
        full_preview = rgba.copy()
        full_preview[iy0:iy1, ix0:ix1, :3] = art_patch
        full_preview[iy0:iy1, ix0:ix1, 3] = 255.0
        return (
            np.clip(np.round(full_preview), 0, 255).astype(np.uint8),
            np.clip(np.round(frame_hole), 0, 255).astype(np.uint8),
        )

    couch_prev_u8, couch_hole_u8 = _build_couch_frame_rgba(fp_cfg.couch_mat_rgb)
    couch_hole_u8[:, :, 3] = filter_min_components_u8(
        couch_hole_u8[:, :, 3], fp_cfg.frame_min_component_px, alpha_clean_threshold_u8
    )
    couch_for_din_prev_u8, couch_for_din_hole_u8 = _build_couch_frame_rgba(fp_cfg.dining_mat_rgb)

    def _warp_premultiplied_rgba(src_rgba_u8: np.ndarray, H_mat: np.ndarray) -> np.ndarray:
        H_out, W_out = inner_dims.height, inner_dims.width
        src_f = src_rgba_u8.astype(np.float32)
        alpha_norm = src_f[:, :, 3:4] / 255.0
        premult_rgb = src_f[:, :, :3] * alpha_norm
        warped_pm = cv2.warpPerspective(
            premult_rgb,
            H_mat,
            (W_out, H_out),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0),
        )
        warped_a = cv2.warpPerspective(
            src_f[:, :, 3],
            H_mat,
            (W_out, H_out),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        wa_norm = np.clip(warped_a[:, :, None] / 255.0, 0.0, 1.0)
        unmult_rgb = np.where(wa_norm > 0.02, warped_pm / np.maximum(wa_norm, 0.02), 0.0)
        out = np.zeros((H_out, W_out, 4), dtype=np.uint8)
        out[:, :, :3] = np.clip(np.round(unmult_rgb), 0, 255).astype(np.uint8)
        out[:, :, 3] = np.clip(np.round(warped_a), 0, 255).astype(np.uint8)
        return out

    din_prev_u8 = _warp_premultiplied_rgba(couch_for_din_prev_u8, H_wall_couch_to_dining)
    din_hole_u8 = _warp_premultiplied_rgba(couch_for_din_hole_u8, H_wall_couch_to_dining)
    din_hole_u8[:, :, 3] = filter_min_components_u8(
        din_hole_u8[:, :, 3], fp_cfg.frame_min_component_px, alpha_clean_threshold_u8
    )
    din_relief = np.clip(
        cv2.warpPerspective(
            couch_relief,
            H_wall_couch_to_dining,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ),
        0.0,
        1.0,
    ).astype(np.float32)

    ap_corners = np.float32([[ix0, iy0], [ix1, iy0], [ix1, iy1], [ix0, iy1]]).reshape(-1, 1, 2)
    din_ap_pts = cv2.perspectiveTransform(ap_corners, H_wall_couch_to_dining).reshape(-1, 2)
    din_ax0 = float(max(din_ap_pts[0, 0], din_ap_pts[3, 0]) - 1.5)
    din_ay0 = float(max(din_ap_pts[0, 1], din_ap_pts[1, 1]) - 1.5)
    din_ax1 = float(min(din_ap_pts[1, 0], din_ap_pts[2, 0]) + 1.5)
    din_ay1 = float(min(din_ap_pts[2, 1], din_ap_pts[3, 1]) + 1.5)

    return (
        Image.fromarray(couch_prev_u8, mode="RGBA"),
        Image.fromarray(couch_hole_u8, mode="RGBA"),
        couch_relief,
        Image.fromarray(din_prev_u8, mode="RGBA"),
        Image.fromarray(din_hole_u8, mode="RGBA"),
        din_relief,
        (round(din_ax0, 2), round(din_ay0, 2), round(din_ax1, 2), round(din_ay1, 2)),
    )


def synthesize_planted_couch_trio_and_shadow(
    couch_inner_rgb: Image.Image,
    couch_people_alpha_u8: np.ndarray,
    couch_inner_da2: np.ndarray,
    dining_clean_inner_rgb: Image.Image,
    dining_sofa_mask_u8: np.ndarray,
    cfg: DenyaLivingRoomBundleConfig,
) -> tuple[Image.Image, Image.Image, np.ndarray]:
    """
    Warps segmented Tal, Ya'ara & Carmel with the guitar and their neural DA2 depth relief from
    `couch_inner` onto the gray couch in `dining_inner` via `H_people`. Separates the crisp
    1-component cutout RGBA (`planted_trio_cutout_rgba`) from the ambient wall/sofa contact shadow
    (`ambient_shadow_rgba`, composited directly onto `layer0` so zero floating shadow fringe exists).
    """
    trio_cfg = cfg.planted_trio
    W, H = cfg.dimensions.width, cfg.dimensions.height
    H_people = compute_homography_from_pairs(trio_cfg.people_plane_pairs)

    couch_np = np.asarray(couch_inner_rgb, dtype=np.float32)
    din_np = np.asarray(dining_clean_inner_rgb, dtype=np.float32)

    cb = cfg.wall_color_ref_box_couch
    db = cfg.wall_color_ref_box_dining
    src_wall_med = np.median(couch_np[cb.y0 : cb.y1, cb.x0 : cb.x1], axis=(0, 1))
    dst_wall_med = np.median(din_np[db.y0 : db.y1, db.x0 : db.x1], axis=(0, 1))
    raw_gains = dst_wall_med / np.maximum(1.0, src_wall_med)
    g_lo, g_hi = trio_cfg.gain_clamp_range
    clamped_gains = np.clip(raw_gains, g_lo, g_hi)
    eff_gains = 1.0 + trio_cfg.relight_blend * (clamped_gains - 1.0)

    decontam_u8 = decontaminate_matte_edge_rgb(
        np.clip(np.round(couch_np * eff_gains[None, None, :]), 0, 255).astype(np.uint8),
        couch_people_alpha_u8,
    )
    alpha_f = couch_people_alpha_u8.astype(np.float32) / 255.0
    premult_rgb = decontam_u8.astype(np.float32) * alpha_f[:, :, None]

    warped_pm = cv2.warpPerspective(
        premult_rgb,
        H_people,
        (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )
    warped_a = cv2.warpPerspective(
        couch_people_alpha_u8.astype(np.float32),
        H_people,
        (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_da2 = cv2.warpPerspective(
        couch_inner_da2.astype(np.float32),
        H_people,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    wa_norm = np.clip(warped_a[:, :, None] / 255.0, 0.0, 1.0)
    warped_rgb_u8 = np.clip(
        np.round(np.where(wa_norm > 0.02, warped_pm / np.maximum(wa_norm, 0.02), 0.0)),
        0,
        255,
    ).astype(np.uint8)
    warped_a_u8 = filter_min_components_u8(
        np.clip(np.round(warped_a), 0, 255).astype(np.uint8),
        trio_cfg.min_warped_component_px,
        cfg.alpha_clean_threshold_u8,
    )

    shadow_canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sil_f = warped_a_u8.astype(np.float32) / 255.0
    sofa_gate_f = ndi.gaussian_filter((dining_sofa_mask_u8 > 96).astype(np.float32), sigma=2.5)

    wx, wy = trio_cfg.wall_shadow_offset_xy
    wall_shift = ndi.shift(sil_f, shift=(wy, wx), order=1, mode="constant", cval=0.0)
    wall_blur = ndi.gaussian_filter(wall_shift, sigma=trio_cfg.wall_shadow_sigma_yx)
    wall_a_u8 = np.clip(
        np.round(wall_blur * (1.0 - sofa_gate_f) * trio_cfg.wall_shadow_opacity * 255.0),
        0,
        255,
    ).astype(np.uint8)
    sh_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    sh_rgb[:, :] = trio_cfg.sofa_shadow_rgb
    shadow_canvas.alpha_composite(Image.fromarray(np.dstack([sh_rgb, wall_a_u8]), mode="RGBA"))

    sx, sy = trio_cfg.sofa_shadow_offset_xy
    sofa_shift = ndi.shift(sil_f, shift=(sy, sx), order=1, mode="constant", cval=0.0)
    sofa_blur = ndi.gaussian_filter(sofa_shift, sigma=trio_cfg.sofa_shadow_sigma_yx)
    sofa_a_u8 = np.clip(
        np.round(sofa_blur * sofa_gate_f * trio_cfg.sofa_shadow_opacity * 255.0),
        0,
        255,
    ).astype(np.uint8)
    shadow_canvas.alpha_composite(Image.fromarray(np.dstack([sh_rgb, sofa_a_u8]), mode="RGBA"))

    cutout_rgba = Image.fromarray(np.dstack([warped_rgb_u8, warped_a_u8]), mode="RGBA")
    return cutout_rgba, shadow_canvas, warped_da2


def build_calibrated_wall_pinned_depth(
    expanded_bg_rgb: Image.Image,
    inner_dims: CanvasDimensionsConfig,
    wall_pin: WallDepthFlattenPinConfig,
    da2_sigma: float,
    depth_estimator: DepthAnythingEstimator,
) -> tuple[np.ndarray, float]:
    """
    Predicts 8-bit `2700x1800` room depth via Depth Anything V2 and gently pins the flat salon
    gallery wall to a uniform planar depth so the framed `'wandering 2'` portal locks flush to
    the wall under 6-DOF camera motion.
    """
    exp_w, exp_h = expanded_bg_rgb.size
    pad_x = (exp_w - inner_dims.width) * 0.5
    pad_y = (exp_h - inner_dims.height) * 0.5

    raw_da2 = depth_estimator.predict_depth(expanded_bg_rgb)
    smooth_da2 = ndi.gaussian_filter(raw_da2, sigma=da2_sigma)
    d_min = float(smooth_da2.min())
    d_max = float(smooth_da2.max())
    norm_da2 = (smooth_da2 - d_min) / max(1e-6, d_max - d_min)

    xx_inner = (np.arange(exp_w, dtype=np.float32) - pad_x)[None, :]
    yy_inner = (np.arange(exp_h, dtype=np.float32) - pad_y)[:, None]
    px, py = wall_pin.inner_xy
    rx, ry = wall_pin.radii_xy
    ell_r = np.hypot((xx_inner - px) / max(1.0, rx), (yy_inner - py) / max(1.0, ry))
    geom_r = float(np.sqrt(max(1.0, rx * ry)))
    excess_px = np.maximum(0.0, ell_r - 1.0) * geom_r
    u_pin = np.clip(excess_px / max(1.0, wall_pin.feather_px), 0.0, 1.0)
    w_pin = 1.0 - u_pin * u_pin * (3.0 - 2.0 * u_pin)

    wall_sample_mask = ell_r <= 0.85
    measured_wall_med = float(np.median(norm_da2[wall_sample_mask]))
    target_d = 0.5 * measured_wall_med + 0.5 * wall_pin.target_norm_depth
    pinned = np.clip(norm_da2 + w_pin * (target_d - norm_da2), 0.0, 1.0)

    depth_u8 = np.clip(np.round(pinned * 255.0), 0, 255).astype(np.uint8)
    final_wall_norm = float(np.median(depth_u8[wall_sample_mask]) / 255.0)
    return depth_u8, final_wall_norm


def _solve_and_verify_cutout_layer(
    label: str,
    room_depth_clean_u8: np.ndarray,
    cutout_rgba_img: Image.Image,
    solve_cfg: ContactAnchoredLayerDepthConfig,
    verify_cfg: LayerTopologyVerificationConfig,
    protrusion_norm: np.ndarray,
) -> tuple[ContactAnchoredSolveResult, LayerTopologyVerificationReport]:
    rgba_np = np.asarray(cutout_rgba_img, dtype=np.uint8)
    solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=room_depth_clean_u8,
        cutout_rgba=rgba_np,
        cfg=solve_cfg,
        neural_protrusion_norm=protrusion_norm,
    )
    report = verify_layer_contact_and_discontinuity(
        room_depth_expanded=room_depth_clean_u8,
        layer_rgba=rgba_np,
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
            f"[Depth Topology Gate FAILED] {label}: {report.to_json()}"
        )
    print(
        f"  [Depth Topology OK] {label:34s} | "
        f"contact max={report.max_abs_contact_dz_world:.2f} p95={report.p95_abs_contact_dz_world:.2f} | "
        f"rim max={report.max_abs_rim_jump_world:.2f} | "
        f"min_dz={report.min_signed_dz_vs_host_world:.2f} | "
        f"Z=[{float(solved.world_z_map[solved.opaque_mask].min()):.1f}, "
        f"{float(solved.world_z_map[solved.opaque_mask].max()):.1f}]"
    )
    return solved, report


def _mean_world_z_in_inner_box(
    world_z_map: np.ndarray,
    inner_box_xyxy: Sequence[float],
    mask: np.ndarray | None = None,
) -> float:
    """Computes mean world-Z inside an inner-coordinate `[x0, y0, x1, y1]` box (`1500x1000` or `2700x1800`)."""
    win = extract_layer_window_from_expanded(world_z_map, (1000, 1500)) if world_z_map.shape != (1000, 1500) else world_z_map
    x0 = max(0, min(1499, int(round(inner_box_xyxy[0]))))
    y0 = max(0, min(999, int(round(inner_box_xyxy[1]))))
    x1 = max(x0 + 1, min(1500, int(round(inner_box_xyxy[2]))))
    y1 = max(y0 + 1, min(1000, int(round(inner_box_xyxy[3]))))
    patch = win[y0:y1, x0:x1]
    if mask is not None:
        m_win = extract_layer_window_from_expanded(mask.astype(np.float32), (1000, 1500)) > 0.5 if mask.shape != (1000, 1500) else mask
        m_patch = m_win[y0:y1, x0:x1]
        if np.any(m_patch):
            return round(float(np.median(patch[m_patch])), 2)
    return round(float(np.median(patch)), 2)


def build_denya_living_room_scenes(
    project_root: Path,
    output_dir: Path,
    cfg: DenyaLivingRoomBundleConfig | None = None,
    export_spec_json: Path | None = None,
) -> dict[str, Any]:
    """Synthesizes all neural layers, contact-anchored depth maps, and SSOT manifest records."""
    bundle = cfg or default_denya_living_room_bundle_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    din_cfg = bundle.dining_scene
    couch_cfg = bundle.couch_scene
    fp_cfg = bundle.framed_painting
    trio_cfg = bundle.planted_trio
    y0 = bundle.progressive_lama.inner_crop_y0
    W, H = bundle.dimensions.width, bundle.dimensions.height
    exp_w, exp_h = bundle.outpainting.expanded_dimensions(bundle.dimensions)
    pad_x = (exp_w - W) // 2
    pad_y = (exp_h - H) // 2
    fy_start = pad_y - y0

    device = select_torch_device()
    print(f"[Denya Living Bundle] Loading neural vision models on device={device}...")
    depth_estimator = DepthAnythingEstimator(bundle.depth_model, device=device)
    sam_extractor = SamBoxMaskExtractor(bundle.segmentation)
    matte_extractor = SalientMatteExtractor(bundle.segmentation)
    lama_inpainter = LamaOcclusionInpainter(bundle.inpainting)

    din_full_rgb, din_inner_rgb = load_source_full_and_inner(
        project_root / din_cfg.source_relpath, bundle.progressive_lama, bundle.dimensions
    )
    couch_full_rgb, couch_inner_rgb = load_source_full_and_inner(
        project_root / couch_cfg.source_relpath, bundle.progressive_lama, bundle.dimensions
    )

    # Predict authentic source Depth Anything V2 maps BEFORE any foreground erasure
    din_source_da2_full = depth_estimator.predict_depth(din_full_rgb)
    din_source_da2_inner = din_source_da2_full[y0 : y0 + H, :]

    couch_source_da2_full = depth_estimator.predict_depth(couch_full_rgb)
    couch_source_da2_inner = couch_source_da2_full[y0 : y0 + H, :]
    couch_source_da2_exp = np.zeros((exp_h, exp_w), dtype=np.float32)
    rh_full = couch_full_rgb.size[1]
    couch_source_da2_exp[fy_start : fy_start + rh_full, pad_x : pad_x + W] = couch_source_da2_full

    H_wall = compute_homography_from_pairs(bundle.wall_homography_pairs)
    (
        couch_art_prev_rgba,
        couch_frame_hole_rgba,
        couch_frame_relief_norm,
        din_art_prev_rgba,
        din_frame_hole_rgba,
        din_frame_relief_norm,
        din_frame_aperture_xyxy,
    ) = synthesize_wandering2_framed_assets(
        project_root=project_root,
        fp_cfg=fp_cfg,
        inner_dims=bundle.dimensions,
        H_wall_couch_to_dining=H_wall,
        alpha_clean_threshold_u8=bundle.alpha_clean_threshold_u8,
    )

    # -------------------------------------------------------------------------
    # Part 1: `denya_couch_guitar` — Trio+Guitar MID (`C^-1` over sofa) & Coffee-Table FG
    # -------------------------------------------------------------------------
    print(f"[{couch_cfg.scene_id}] Segmenting seated trio+guitar MID (preserving gray sofa on layer0) and full coffee-table FG...")
    people_sam_u8 = sam_extractor.segment_point_groups(
        couch_inner_rgb,
        couch_cfg.people_sam_prompts,
        couch_cfg.people_exclude_boxes,
        couch_cfg.people_max_pinhole_fill_px,
        couch_cfg.people_min_component_px,
    )
    people_sal_u8 = refine_alpha_with_salient_edges(
        couch_inner_rgb,
        people_sam_u8,
        matte_extractor,
        support_px=bundle.salient_edge_support_px,
        edge_feather_sigma=bundle.segmentation.edge_feather_sigma,
    )
    head_max_y = couch_cfg.people_head_salient_max_y
    people_alpha_u8 = people_sam_u8.copy()
    people_alpha_u8[:head_max_y, :] = people_sal_u8[:head_max_y, :]
    if bundle.segmentation.edge_feather_sigma > 0.0:
        body_soft = np.clip(
            np.round(
                ndi.gaussian_filter(
                    people_sam_u8.astype(np.float32),
                    sigma=bundle.segmentation.edge_feather_sigma * 0.5,
                )
            ),
            0,
            255,
        ).astype(np.uint8)
        people_alpha_u8[head_max_y:, :] = body_soft[head_max_y:, :]
    for eb in couch_cfg.people_exclude_boxes:
        people_alpha_u8[eb.y0 : eb.y1, eb.x0 : eb.x1] = 0
    people_alpha_u8 = solidify_interior_alpha_u8(
        people_alpha_u8,
        max_hole_area_px=couch_cfg.people_max_pinhole_fill_px,
        erode_px=bundle.interior_solidify_erode_px,
        threshold_u8=bundle.interior_solidify_threshold_u8,
    )
    for eb in couch_cfg.people_exclude_boxes:
        people_alpha_u8[eb.y0 : eb.y1, eb.x0 : eb.x1] = 0
    people_alpha_u8 = filter_min_components_u8(
        people_alpha_u8, couch_cfg.people_min_component_px, bundle.alpha_clean_threshold_u8
    )

    # Full 1500x1125 coffee-table + all tabletop objects + legs
    table_full_u8 = sam_extractor.segment_point_groups(
        couch_full_rgb,
        couch_cfg.coffee_table_full_sam_prompts,
        (),
        couch_cfg.coffee_table_max_pinhole_fill_px,
        bundle.segmentation.min_component_area_px,
    )
    k_table = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (couch_cfg.coffee_table_morph_close_px, couch_cfg.coffee_table_morph_close_px),
    )
    table_full_u8 = cv2.morphologyEx(table_full_u8, cv2.MORPH_CLOSE, k_table)
    table_full_u8 = solidify_interior_alpha_u8(
        table_full_u8,
        max_hole_area_px=couch_cfg.coffee_table_max_pinhole_fill_px,
        erode_px=bundle.interior_solidify_erode_px,
        threshold_u8=bundle.interior_solidify_threshold_u8,
    )
    if couch_cfg.coffee_table_edge_feather_sigma > 0.0:
        blurred_table = ndi.gaussian_filter(
            table_full_u8.astype(np.float32), sigma=couch_cfg.coffee_table_edge_feather_sigma
        )
        lo_t = float(bundle.segmentation.alpha_foreground_threshold)
        hi_t = float(bundle.segmentation.alpha_full_opaque_threshold)
        table_full_u8 = np.clip(
            np.round((blurred_table - lo_t) * (255.0 / max(1.0, hi_t - lo_t))), 0, 255
        ).astype(np.uint8)
        table_full_u8 = solidify_interior_alpha_u8(
            table_full_u8,
            max_hole_area_px=couch_cfg.coffee_table_max_pinhole_fill_px,
            erode_px=bundle.interior_solidify_erode_px,
            threshold_u8=bundle.interior_solidify_threshold_u8,
        )
    table_full_u8 = filter_min_components_u8(
        table_full_u8, couch_cfg.coffee_table_min_component_px, bundle.alpha_clean_threshold_u8
    )
    table_inner_u8 = table_full_u8[y0 : y0 + H, :]

    # Heal ONLY the coffee table on `couch_full_rgb` to synthesize a clean downward knees apron
    table_and_tips_u8 = table_full_u8.copy()
    for b_tip in couch_cfg.bottom_chair_tips_boxes_full:
        table_and_tips_u8[b_tip.y0 : b_tip.y1, b_tip.x0 : b_tip.x1] = 255
    couch_without_table_full_rgb = lama_inpainter.inpaint_background(
        couch_full_rgb,
        np.where(table_and_tips_u8 > 24, 255, 0).astype(np.uint8),
        dilation_radius_px=couch_cfg.mid_apron.inpaint_dilation_px,
    )
    couch_without_table_inner_rgb = couch_without_table_full_rgb.crop((0, y0, W, y0 + H))

    couch_mid_rgba = build_downward_apron_rgba(
        authentic_inner_rgb=couch_inner_rgb,
        healed_behind_occluder_inner_rgb=couch_without_table_inner_rgb,
        base_alpha_u8=people_alpha_u8,
        occluder_inner_u8=table_inner_u8,
        apron_cfg=couch_cfg.mid_apron,
        min_component_px=couch_cfg.people_min_component_px,
    )

    # Expanded 2700x1800 coffee-table foreground (`fgExpanded: true`)
    table_full_decontam = decontaminate_matte_edge_rgb(
        np.asarray(couch_full_rgb, dtype=np.uint8), table_full_u8
    )
    couch_fg_exp = np.zeros((exp_h, exp_w, 4), dtype=np.uint8)
    couch_fg_exp[fy_start : fy_start + rh_full, pad_x : pad_x + W, :3] = table_full_decontam
    couch_fg_exp[fy_start : fy_start + rh_full, pad_x : pad_x + W, 3] = table_full_u8
    couch_fg_rgba = Image.fromarray(couch_fg_exp, mode="RGBA")

    # Erase ONLY people+guitar + coffee table + bottom foreground chair arms from `layer0` so the
    # gray sofa, pillows, and plush toy remain intact on `layer0` (`C^-1` behind Tal's profile!)
    couch_occ_full_u8 = table_and_tips_u8.copy()
    couch_occ_full_u8[y0 : y0 + H, :] = np.maximum(
        couch_occ_full_u8[y0 : y0 + H, :], people_alpha_u8
    )
    couch_clean_full_rgb = lama_inpainter.inpaint_background(
        couch_full_rgb,
        np.where(couch_occ_full_u8 > 24, 255, 0).astype(np.uint8),
        dilation_radius_px=bundle.inpainting.dilation_radius_px,
    )
    couch_clean_full_rgba = couch_clean_full_rgb.convert("RGBA")
    couch_clean_full_rgba.alpha_composite(couch_art_prev_rgba, dest=(0, y0))
    couch_bg_exp_rgb = progressive_lama_outpaint_canvas(
        full_clean_rgb=couch_clean_full_rgba.convert("RGB"),
        inner_dims=bundle.dimensions,
        outpaint_cfg=bundle.outpainting,
        prog_cfg=bundle.progressive_lama,
        lama_inpainter=lama_inpainter,
    )
    couch_raw_depth_u8, _ = build_calibrated_wall_pinned_depth(
        expanded_bg_rgb=couch_bg_exp_rgb,
        inner_dims=bundle.dimensions,
        wall_pin=couch_cfg.wall_flatten_pin,
        da2_sigma=couch_cfg.da2_gaussian_sigma,
        depth_estimator=depth_estimator,
    )
    couch_clean_depth_u8 = clean_room_depth_under_cutouts(
        room_depth=couch_raw_depth_u8,
        cutout_layers=[np.asarray(couch_fg_rgba)],
        cfg=couch_cfg.room_depth_clean,
    )
    couch_depth_img = Image.fromarray(couch_clean_depth_u8, mode="L")

    # Solve & verify all 3 displaced cutout layers in `denya_couch_guitar`
    couch_frame_solved, couch_frame_report = _solve_and_verify_cutout_layer(
        label="denya_couch_guitar.wandering2_wall_frame",
        room_depth_clean_u8=couch_clean_depth_u8,
        cutout_rgba_img=couch_frame_hole_rgba,
        solve_cfg=couch_cfg.frame_depth_solve,
        verify_cfg=bundle.topology_verification,
        protrusion_norm=couch_frame_relief_norm,
    )
    couch_mid_protrusion = compute_source_da2_protrusion_norm(
        raw_source_da2_layer=couch_source_da2_inner,
        cutout_rgba=np.asarray(couch_mid_rgba),
        solve_cfg=couch_cfg.mid_depth_solve,
        relief_cfg=couch_cfg.mid_protrusion,
    )
    couch_mid_solved, couch_mid_report = _solve_and_verify_cutout_layer(
        label="denya_couch_guitar.mid_trio_guitar",
        room_depth_clean_u8=couch_clean_depth_u8,
        cutout_rgba_img=couch_mid_rgba,
        solve_cfg=couch_cfg.mid_depth_solve,
        verify_cfg=bundle.topology_verification,
        protrusion_norm=couch_mid_protrusion,
    )
    couch_fg_protrusion = compute_source_da2_protrusion_norm(
        raw_source_da2_layer=couch_source_da2_exp,
        cutout_rgba=np.asarray(couch_fg_rgba),
        solve_cfg=couch_cfg.fg_depth_solve,
        relief_cfg=couch_cfg.fg_protrusion,
    )
    couch_fg_solved, couch_fg_report = _solve_and_verify_cutout_layer(
        label="denya_couch_guitar.fg_coffee_table",
        room_depth_clean_u8=couch_clean_depth_u8,
        cutout_rgba_img=couch_fg_rgba,
        solve_cfg=couch_cfg.fg_depth_solve,
        verify_cfg=bundle.topology_verification,
        protrusion_norm=couch_fg_protrusion,
    )

    couch_room_world_z_exp = compute_world_z_map(
        depth_map=couch_clean_depth_u8,
        plane_z=couch_cfg.scene_bg_z,
        depth_amplitude=couch_cfg.room_depth_amplitude,
        pin_margin_uv=couch_cfg.frame_depth_solve.room_pin_margin_uv,
    )
    couch_ap = fp_cfg.couch_inner_aperture_xyxy()
    couch_wall_world_z = _mean_world_z_in_inner_box(couch_room_world_z_exp, couch_ap)
    couch_return_portal_world_z = _mean_world_z_in_inner_box(
        couch_fg_solved.world_z_map,
        couch_cfg.return_portal_inner_box_xyxy,
        mask=couch_fg_solved.opaque_mask,
    )

    # -------------------------------------------------------------------------
    # Part 2: `denya_living_room` — Unified Dining FG, Center MID & Planted Trio
    # -------------------------------------------------------------------------
    print(f"[{din_cfg.scene_id}] Segmenting unified dining FG, center coffee-table MID & planted trio...")
    din_fg_sam_u8 = sam_extractor.segment_point_groups(
        din_inner_rgb,
        din_cfg.fg_sam_prompts,
        din_cfg.fg_exclude_boxes,
        din_cfg.fg_max_pinhole_fill_px,
        bundle.segmentation.min_component_area_px,
    )
    din_fg_filled_u8 = seal_right_bottom_crop_notches_u8(
        din_fg_sam_u8,
        bottom_x0=din_cfg.bottom_fg_extension_x0,
        right_y0=din_cfg.right_fg_extension_y0,
        max_hole_area_px=din_cfg.fg_max_pinhole_fill_px,
    )
    din_fg_alpha_u8 = refine_alpha_with_salient_edges(
        din_inner_rgb,
        din_fg_filled_u8,
        matte_extractor,
        support_px=bundle.salient_edge_support_px,
        edge_feather_sigma=bundle.segmentation.edge_feather_sigma,
    )
    for eb in din_cfg.fg_exclude_boxes:
        din_fg_alpha_u8[eb.y0 : eb.y1, eb.x0 : eb.x1] = 0
    din_fg_alpha_u8 = seal_right_bottom_crop_notches_u8(
        din_fg_alpha_u8,
        bottom_x0=din_cfg.bottom_fg_extension_x0,
        right_y0=din_cfg.right_fg_extension_y0,
        max_hole_area_px=din_cfg.fg_max_pinhole_fill_px,
    )
    din_fg_alpha_u8 = solidify_interior_alpha_u8(
        din_fg_alpha_u8,
        max_hole_area_px=din_cfg.fg_max_pinhole_fill_px,
        erode_px=bundle.interior_solidify_erode_px,
        threshold_u8=bundle.interior_solidify_threshold_u8,
    )
    din_fg_alpha_u8 = filter_min_components_u8(
        din_fg_alpha_u8, din_cfg.fg_min_component_px, bundle.alpha_clean_threshold_u8
    )

    din_fg_rgb_u8 = decontaminate_matte_edge_rgb(
        np.asarray(din_inner_rgb, dtype=np.uint8), din_fg_alpha_u8
    )
    din_fg_rgba = Image.fromarray(np.dstack([din_fg_rgb_u8, din_fg_alpha_u8]), mode="RGBA")

    din_mid_sam_u8 = sam_extractor.segment_point_groups(
        din_inner_rgb,
        din_cfg.mid_sam_prompts,
        (),
        din_cfg.mid_max_pinhole_fill_px,
        din_cfg.mid_min_component_px,
    )
    din_mid_sam_u8 = fill_enclosed_pinholes_u8(din_mid_sam_u8, din_cfg.mid_max_pinhole_fill_px)
    din_mid_sam_u8 = filter_min_components_u8(
        din_mid_sam_u8, din_cfg.mid_min_component_px, bundle.alpha_clean_threshold_u8
    )

    # Heal ONLY `din_fg` first so `din_mid` gets a seamless downward apron behind the far tablecloth edge
    din_fg_only_full_u8 = np.zeros((din_full_rgb.size[1], W), dtype=np.uint8)
    din_fg_only_full_u8[y0 : y0 + H, :] = din_fg_alpha_u8
    din_fg_only_full_u8[y0 + H :, :] = din_fg_alpha_u8[H - 1 : H, :]
    din_fg_only_full_u8[y0 + H :, din_cfg.bottom_fg_extension_x0 :] = 255
    din_without_fg_full_rgb = lama_inpainter.inpaint_background(
        din_full_rgb,
        np.where(din_fg_only_full_u8 > 24, 255, 0).astype(np.uint8),
        dilation_radius_px=din_cfg.mid_apron.inpaint_dilation_px,
    )
    din_without_fg_inner_rgb = din_without_fg_full_rgb.crop((0, y0, W, y0 + H))

    din_mid_rgba = build_downward_apron_rgba(
        authentic_inner_rgb=din_inner_rgb,
        healed_behind_occluder_inner_rgb=din_without_fg_inner_rgb,
        base_alpha_u8=din_mid_sam_u8,
        occluder_inner_u8=din_fg_alpha_u8,
        apron_cfg=din_cfg.mid_apron,
        min_component_px=din_cfg.mid_min_component_px,
    )
    din_mid_alpha_u8 = np.asarray(din_mid_rgba, dtype=np.uint8)[:, :, 3]

    din_sofa_mask_u8 = sam_extractor.segment_box_group(din_inner_rgb, [din_cfg.sofa_ref_box])

    # Build LaMa occlusion mask for dining room `layer0` background: FG + center MID + perched pillow
    din_occ_full_u8 = din_fg_only_full_u8.copy()
    din_occ_inner = np.maximum(din_fg_alpha_u8, din_mid_alpha_u8)
    pb = din_cfg.perched_backrest_pillow_box
    din_occ_inner[pb.y0 : pb.y1, pb.x0 : pb.x1] = 255
    din_occ_full_u8[y0 : y0 + H, :] = np.maximum(din_occ_full_u8[y0 : y0 + H, :], din_occ_inner)

    din_clean_full_rgb = lama_inpainter.inpaint_background(
        din_full_rgb,
        np.where(din_occ_full_u8 > 24, 255, 0).astype(np.uint8),
        dilation_radius_px=bundle.inpainting.dilation_radius_px,
    )
    din_clean_inner_rgb = din_clean_full_rgb.crop((0, y0, W, y0 + H))

    planted_trio_rgba, planted_trio_shadow_rgba, planted_trio_warped_da2 = (
        synthesize_planted_couch_trio_and_shadow(
            couch_inner_rgb=couch_inner_rgb,
            couch_people_alpha_u8=people_alpha_u8,
            couch_inner_da2=couch_source_da2_inner,
            dining_clean_inner_rgb=din_clean_inner_rgb,
            dining_sofa_mask_u8=din_sofa_mask_u8,
            cfg=bundle,
        )
    )

    # Composite ambient sofa/wall shadow + static glazed `'wandering 2'` preview onto `layer0`
    din_clean_full_rgba = din_clean_full_rgb.convert("RGBA")
    din_clean_full_rgba.alpha_composite(planted_trio_shadow_rgba, dest=(0, y0))
    din_clean_full_rgba.alpha_composite(din_art_prev_rgba, dest=(0, y0))
    din_bg_exp_rgb = progressive_lama_outpaint_canvas(
        full_clean_rgb=din_clean_full_rgba.convert("RGB"),
        inner_dims=bundle.dimensions,
        outpaint_cfg=bundle.outpainting,
        prog_cfg=bundle.progressive_lama,
        lama_inpainter=lama_inpainter,
    )
    din_raw_depth_u8, _ = build_calibrated_wall_pinned_depth(
        expanded_bg_rgb=din_bg_exp_rgb,
        inner_dims=bundle.dimensions,
        wall_pin=din_cfg.wall_flatten_pin,
        da2_sigma=din_cfg.da2_gaussian_sigma,
        depth_estimator=depth_estimator,
    )
    din_clean_depth_u8 = clean_room_depth_under_cutouts(
        room_depth=din_raw_depth_u8,
        cutout_layers=[np.asarray(din_mid_rgba), np.asarray(din_fg_rgba)],
        cfg=din_cfg.room_depth_clean,
    )
    din_depth_img = Image.fromarray(din_clean_depth_u8, mode="L")

    # Solve & verify all 4 displaced cutout layers in `denya_living_room`
    din_frame_solved, din_frame_report = _solve_and_verify_cutout_layer(
        label="denya_living_room.wandering2_wall_frame",
        room_depth_clean_u8=din_clean_depth_u8,
        cutout_rgba_img=din_frame_hole_rgba,
        solve_cfg=din_cfg.frame_depth_solve,
        verify_cfg=bundle.topology_verification,
        protrusion_norm=din_frame_relief_norm,
    )
    trio_protrusion = compute_source_da2_protrusion_norm(
        raw_source_da2_layer=planted_trio_warped_da2,
        cutout_rgba=np.asarray(planted_trio_rgba),
        solve_cfg=trio_cfg.depth_solve,
        relief_cfg=trio_cfg.protrusion,
    )
    trio_solved, trio_report = _solve_and_verify_cutout_layer(
        label="denya_living_room.planted_couch_trio",
        room_depth_clean_u8=din_clean_depth_u8,
        cutout_rgba_img=planted_trio_rgba,
        solve_cfg=trio_cfg.depth_solve,
        verify_cfg=bundle.topology_verification,
        protrusion_norm=trio_protrusion,
    )
    din_mid_protrusion = compute_source_da2_protrusion_norm(
        raw_source_da2_layer=din_source_da2_inner,
        cutout_rgba=np.asarray(din_mid_rgba),
        solve_cfg=din_cfg.mid_depth_solve,
        relief_cfg=din_cfg.mid_protrusion,
    )
    din_mid_solved, din_mid_report = _solve_and_verify_cutout_layer(
        label="denya_living_room.mid_coffee_table_chair",
        room_depth_clean_u8=din_clean_depth_u8,
        cutout_rgba_img=din_mid_rgba,
        solve_cfg=din_cfg.mid_depth_solve,
        verify_cfg=bundle.topology_verification,
        protrusion_norm=din_mid_protrusion,
    )
    din_fg_protrusion = compute_source_da2_protrusion_norm(
        raw_source_da2_layer=din_source_da2_inner,
        cutout_rgba=np.asarray(din_fg_rgba),
        solve_cfg=din_cfg.fg_depth_solve,
        relief_cfg=din_cfg.fg_protrusion,
    )
    din_fg_solved, din_fg_report = _solve_and_verify_cutout_layer(
        label="denya_living_room.fg_unified_dining_table_family",
        room_depth_clean_u8=din_clean_depth_u8,
        cutout_rgba_img=din_fg_rgba,
        solve_cfg=din_cfg.fg_depth_solve,
        verify_cfg=bundle.topology_verification,
        protrusion_norm=din_fg_protrusion,
    )

    din_room_world_z_exp = compute_world_z_map(
        depth_map=din_clean_depth_u8,
        plane_z=din_cfg.scene_bg_z,
        depth_amplitude=din_cfg.room_depth_amplitude,
        pin_margin_uv=din_cfg.frame_depth_solve.room_pin_margin_uv,
    )
    din_wall_world_z = _mean_world_z_in_inner_box(din_room_world_z_exp, din_frame_aperture_xyxy)
    din_trio_portal_world_z = _mean_world_z_in_inner_box(
        trio_solved.world_z_map,
        trio_cfg.portal_inner_box_xyxy,
        mask=trio_solved.opaque_mask,
    )

    # -------------------------------------------------------------------------
    # Part 3: Save All Color & Depth Textures + Assemble SSOT Manifest Records
    # -------------------------------------------------------------------------
    jq = bundle.inpainting.jpeg_quality
    couch_bg_exp_rgb.save(output_dir / couch_cfg.bg_output_filename, format="JPEG", quality=jq, optimize=True)
    couch_depth_img.save(output_dir / couch_cfg.depth_output_filename, format="PNG", optimize=True)
    couch_mid_rgba.save(output_dir / couch_cfg.mid_output_filename, format="PNG", optimize=True)
    Image.fromarray(couch_mid_solved.depth_u8, mode="L").save(
        output_dir / couch_cfg.mid_depth_output_filename, format="PNG", optimize=True
    )
    couch_fg_rgba.save(output_dir / couch_cfg.fg_output_filename, format="PNG", optimize=True)
    Image.fromarray(couch_fg_solved.depth_u8, mode="L").save(
        output_dir / couch_cfg.fg_depth_output_filename, format="PNG", optimize=True
    )
    couch_frame_hole_rgba.save(output_dir / couch_cfg.frame_output_filename, format="PNG", optimize=True)
    Image.fromarray(couch_frame_solved.depth_u8, mode="L").save(
        output_dir / couch_cfg.frame_depth_output_filename, format="PNG", optimize=True
    )

    din_bg_exp_rgb.save(output_dir / din_cfg.bg_output_filename, format="JPEG", quality=jq, optimize=True)
    din_depth_img.save(output_dir / din_cfg.depth_output_filename, format="PNG", optimize=True)
    din_mid_rgba.save(output_dir / din_cfg.mid_output_filename, format="PNG", optimize=True)
    Image.fromarray(din_mid_solved.depth_u8, mode="L").save(
        output_dir / din_cfg.mid_depth_output_filename, format="PNG", optimize=True
    )
    din_fg_rgba.save(output_dir / din_cfg.fg_output_filename, format="PNG", optimize=True)
    Image.fromarray(din_fg_solved.depth_u8, mode="L").save(
        output_dir / din_cfg.fg_depth_output_filename, format="PNG", optimize=True
    )
    planted_trio_rgba.save(output_dir / din_cfg.planted_people_output_filename, format="PNG", optimize=True)
    Image.fromarray(trio_solved.depth_u8, mode="L").save(
        output_dir / trio_cfg.depth_output_filename, format="PNG", optimize=True
    )
    din_frame_hole_rgba.save(output_dir / din_cfg.frame_output_filename, format="PNG", optimize=True)
    Image.fromarray(din_frame_solved.depth_u8, mode="L").save(
        output_dir / din_cfg.frame_depth_output_filename, format="PNG", optimize=True
    )

    dining_portals = [
        {
            "id": fp_cfg.portal_id_in_dining,
            "targetSceneId": fp_cfg.target_scene_id,
            "portalMode": "aperture",
            "labelHe": fp_cfg.label_he,
            "shortBadgeHe": fp_cfg.short_badge_he,
            "innerBoxPx": list(din_frame_aperture_xyxy),
            "worldZ": din_wall_world_z,
            "recessDepth": fp_cfg.recess_depth,
            "restFrameOpacity": 0.0,
            "badgeOffsetYFactor": fp_cfg.badge_offset_y_factor,
        },
        {
            "id": trio_cfg.portal_id_to_couch,
            "targetSceneId": couch_cfg.scene_id,
            "portalMode": "crossfade",
            "labelHe": trio_cfg.portal_label_he,
            "shortBadgeHe": trio_cfg.portal_short_badge_he,
            "innerBoxPx": list(trio_cfg.portal_inner_box_xyxy),
            "worldZ": din_trio_portal_world_z,
            "recessDepth": 0.0,
            "restFrameOpacity": 0.0,
            "fadeStartProgress": trio_cfg.portal_fade_start_progress,
            "badgeOffsetYFactor": trio_cfg.badge_offset_y_factor,
        },
    ]

    couch_portals = [
        {
            "id": fp_cfg.portal_id_in_couch,
            "targetSceneId": fp_cfg.target_scene_id,
            "portalMode": "aperture",
            "labelHe": fp_cfg.label_he,
            "shortBadgeHe": fp_cfg.short_badge_he,
            "innerBoxPx": [float(v) for v in couch_ap],
            "worldZ": couch_wall_world_z,
            "recessDepth": fp_cfg.recess_depth,
            "restFrameOpacity": 0.0,
            "badgeOffsetYFactor": fp_cfg.badge_offset_y_factor,
        },
        {
            "id": couch_cfg.return_portal_id,
            "targetSceneId": din_cfg.scene_id,
            "portalMode": "reverse_transition",
            "labelHe": couch_cfg.return_portal_label_he,
            "shortBadgeHe": couch_cfg.return_portal_short_badge_he,
            "innerBoxPx": list(couch_cfg.return_portal_inner_box_xyxy),
            "worldZ": couch_return_portal_world_z,
            "recessDepth": 0.0,
            "restFrameOpacity": 0.0,
            "badgeOffsetYFactor": 0.0,
        },
    ]

    din_extra_layers = [
        {
            "id": "wandering2_wall_frame",
            "labelHe": "מסגרת עץ אלון על קיר הסלון בדניה — שער אל ׳שיטוט 2׳",
            "texture": f"assets/scenes/{din_cfg.frame_output_filename}",
            "depthTexture": f"assets/scenes/{din_cfg.frame_depth_output_filename}",
            "zPos": din_cfg.frame_depth_solve.layer_z,
            "depthAmplitude": din_cfg.frame_depth_solve.layer_depth_amplitude,
            "pinMarginUv": 0.0,
            "segmentsX": din_cfg.frame_segments_x,
            "segmentsY": din_cfg.frame_segments_y,
            "renderOrder": din_cfg.frame_render_order,
            "expanded": False,
            "baseOpacity": 1.0,
            "excursionBoost": 0.0,
            "depthTest": False,
            "depthWrite": False,
        },
        {
            "id": trio_cfg.layer_id,
            "labelHe": trio_cfg.label_he,
            "texture": f"assets/scenes/{din_cfg.planted_people_output_filename}",
            "depthTexture": f"assets/scenes/{trio_cfg.depth_output_filename}",
            "zPos": trio_cfg.depth_solve.layer_z,
            "depthAmplitude": trio_cfg.depth_solve.layer_depth_amplitude,
            "pinMarginUv": 0.0,
            "segmentsX": trio_cfg.segments_x,
            "segmentsY": trio_cfg.segments_y,
            "renderOrder": trio_cfg.render_order,
            "expanded": False,
            "baseOpacity": 1.0,
            "excursionBoost": 0.0,
            "depthTest": False,
            "depthWrite": False,
        },
    ]

    couch_extra_layers = [
        {
            "id": "wandering2_wall_frame",
            "labelHe": "מסגרת עץ אלון על קיר הסלון בדניה — שער אל ׳שיטוט 2׳",
            "texture": f"assets/scenes/{couch_cfg.frame_output_filename}",
            "depthTexture": f"assets/scenes/{couch_cfg.frame_depth_output_filename}",
            "zPos": couch_cfg.frame_depth_solve.layer_z,
            "depthAmplitude": couch_cfg.frame_depth_solve.layer_depth_amplitude,
            "pinMarginUv": 0.0,
            "segmentsX": couch_cfg.frame_segments_x,
            "segmentsY": couch_cfg.frame_segments_y,
            "renderOrder": couch_cfg.frame_render_order,
            "expanded": False,
            "baseOpacity": 1.0,
            "excursionBoost": 0.0,
            "depthTest": False,
            "depthWrite": False,
        },
    ]

    verification_reports = {
        "denya_living_room": {
            "wandering2_wall_frame": din_frame_report.to_dict(),
            "planted_couch_trio": trio_report.to_dict(),
            "mid": din_mid_report.to_dict(),
            "fg": din_fg_report.to_dict(),
        },
        "denya_couch_guitar": {
            "wandering2_wall_frame": couch_frame_report.to_dict(),
            "mid": couch_mid_report.to_dict(),
            "fg": couch_fg_report.to_dict(),
        },
    }

    couch_mid_alpha_u8 = np.asarray(couch_mid_rgba, dtype=np.uint8)[:, :, 3]
    din_manifest = {
        "id": din_cfg.scene_id,
        "title": din_cfg.title_he,
        "shortLabelHe": din_cfg.short_label_he,
        "icon": din_cfg.icon,
        "exhibitionGroup": din_cfg.exhibition_group,
        "sourceCatalogIds": [din_cfg.catalog_id, couch_cfg.catalog_id],
        "sourceImage": din_cfg.source_relpath,
        "secondarySourceImage": couch_cfg.source_relpath,
        "segmentationBackend": "denya_living_sam_isnet_lama_c0_edt",
        "depthBackend": "denya_living_contact_anchored_da2_edt",
        "innerWidth": W,
        "innerHeight": H,
        "bgExpandedWidth": exp_w,
        "bgExpandedHeight": exp_h,
        "bgExpansionFactor": bundle.outpainting.expansion_factor,
        "fgExpanded": False,
        "fgBleedEdges": {
            "left": False,
            "right": True,
            "top": False,
            "bottom": True,
            "guardMinZ": din_cfg.fg_bleed_guard_min_z,
        },
        "boreSightAtRest": True,
        "minInnerScale": din_cfg.min_inner_scale,
        "roomDepthAmplitude": din_cfg.room_depth_amplitude,
        "sceneBgZ": din_cfg.scene_bg_z,
        "midZ": din_cfg.mid_depth_solve.layer_z,
        "midDepthAmplitude": din_cfg.mid_depth_solve.layer_depth_amplitude,
        "midPinMarginUv": 0.0,
        "midSegmentsX": din_cfg.mid_segments_x,
        "midSegmentsY": din_cfg.mid_segments_y,
        "midDepthTest": False,
        "midDepthWrite": False,
        "fgZ": din_cfg.fg_depth_solve.layer_z,
        "fgDepthAmplitude": din_cfg.fg_depth_solve.layer_depth_amplitude,
        "fgPinMarginUv": 0.0,
        "fgSegmentsX": din_cfg.fg_segments_x,
        "fgSegmentsY": din_cfg.fg_segments_y,
        "fgDepthTest": False,
        "fgDepthWrite": False,
        "bgSegmentsX": din_cfg.bg_segments_x,
        "bgSegmentsY": din_cfg.bg_segments_y,
        "initialCameraY": din_cfg.initial_camera_y,
        "initialCameraZ": din_cfg.initial_camera_z,
        "minCameraZ": din_cfg.min_camera_z,
        "maxCameraZ": din_cfg.max_camera_z,
        "maxYawDeg": din_cfg.max_yaw_deg,
        "maxPitchDeg": din_cfg.max_pitch_deg,
        "maxStrafeX": din_cfg.max_strafe_x,
        "maxStrafeY": din_cfg.max_strafe_y,
        "dollySpeedMult": din_cfg.dolly_speed_mult,
        "midRenderOrder": din_cfg.mid_render_order,
        "fgRenderOrder": din_cfg.fg_render_order,
        "particleColorHex": din_cfg.particle_color_hex,
        "particleCount": din_cfg.particle_count,
        "bgTexture": f"assets/scenes/{din_cfg.bg_output_filename}",
        "depthTexture": f"assets/scenes/{din_cfg.depth_output_filename}",
        "midTexture": f"assets/scenes/{din_cfg.mid_output_filename}",
        "midDepthTexture": f"assets/scenes/{din_cfg.mid_depth_output_filename}",
        "fgTexture": f"assets/scenes/{din_cfg.fg_output_filename}",
        "fgDepthTexture": f"assets/scenes/{din_cfg.fg_depth_output_filename}",
        "extraLayers": din_extra_layers,
        "portalShelves": dining_portals,
        "reflectionTexture": None,
        "hasReflection": False,
        "fgCoverageRatio": round(float((din_fg_alpha_u8 > 16).mean()), 4),
        "midCoverageRatio": round(float((din_mid_alpha_u8 > 16).mean()), 4),
        "plantedTrioCoverageRatio": round(
            float((np.asarray(planted_trio_rgba)[:, :, 3] > 16).mean()), 4
        ),
        "topologyVerification": verification_reports["denya_living_room"],
    }

    couch_manifest = {
        "id": couch_cfg.scene_id,
        "title": couch_cfg.title_he,
        "shortLabelHe": couch_cfg.short_label_he,
        "icon": couch_cfg.icon,
        "exhibitionGroup": couch_cfg.exhibition_group,
        "sourceCatalogIds": [couch_cfg.catalog_id],
        "sourceImage": couch_cfg.source_relpath,
        "segmentationBackend": "denya_living_sam_isnet_lama_c0_edt",
        "depthBackend": "denya_living_contact_anchored_da2_edt",
        "innerWidth": W,
        "innerHeight": H,
        "bgExpandedWidth": exp_w,
        "bgExpandedHeight": exp_h,
        "bgExpansionFactor": bundle.outpainting.expansion_factor,
        "fgExpanded": True,
        "boreSightAtRest": True,
        "minInnerScale": couch_cfg.min_inner_scale,
        "roomDepthAmplitude": couch_cfg.room_depth_amplitude,
        "sceneBgZ": couch_cfg.scene_bg_z,
        "midZ": couch_cfg.mid_depth_solve.layer_z,
        "midDepthAmplitude": couch_cfg.mid_depth_solve.layer_depth_amplitude,
        "midPinMarginUv": 0.0,
        "midSegmentsX": couch_cfg.mid_segments_x,
        "midSegmentsY": couch_cfg.mid_segments_y,
        "midDepthTest": False,
        "midDepthWrite": False,
        "fgZ": couch_cfg.fg_depth_solve.layer_z,
        "fgDepthAmplitude": couch_cfg.fg_depth_solve.layer_depth_amplitude,
        "fgPinMarginUv": 0.0,
        "fgSegmentsX": couch_cfg.fg_segments_x,
        "fgSegmentsY": couch_cfg.fg_segments_y,
        "fgDepthTest": False,
        "fgDepthWrite": False,
        "bgSegmentsX": couch_cfg.bg_segments_x,
        "bgSegmentsY": couch_cfg.bg_segments_y,
        "initialCameraY": couch_cfg.initial_camera_y,
        "initialCameraZ": couch_cfg.initial_camera_z,
        "minCameraZ": couch_cfg.min_camera_z,
        "maxCameraZ": couch_cfg.max_camera_z,
        "maxYawDeg": couch_cfg.max_yaw_deg,
        "maxPitchDeg": couch_cfg.max_pitch_deg,
        "maxStrafeX": couch_cfg.max_strafe_x,
        "maxStrafeY": couch_cfg.max_strafe_y,
        "dollySpeedMult": couch_cfg.dolly_speed_mult,
        "midRenderOrder": couch_cfg.mid_render_order,
        "fgRenderOrder": couch_cfg.fg_render_order,
        "particleColorHex": couch_cfg.particle_color_hex,
        "particleCount": couch_cfg.particle_count,
        "bgTexture": f"assets/scenes/{couch_cfg.bg_output_filename}",
        "depthTexture": f"assets/scenes/{couch_cfg.depth_output_filename}",
        "midTexture": f"assets/scenes/{couch_cfg.mid_output_filename}",
        "midDepthTexture": f"assets/scenes/{couch_cfg.mid_depth_output_filename}",
        "fgTexture": f"assets/scenes/{couch_cfg.fg_output_filename}",
        "fgDepthTexture": f"assets/scenes/{couch_cfg.fg_depth_output_filename}",
        "extraLayers": couch_extra_layers,
        "portalShelves": couch_portals,
        "reflectionTexture": None,
        "hasReflection": False,
        "fgCoverageRatio": round(float((couch_fg_exp[:, :, 3] > 16).mean()), 4),
        "midCoverageRatio": round(float((couch_mid_alpha_u8 > 16).mean()), 4),
        "topologyVerification": verification_reports["denya_couch_guitar"],
    }

    payload = {
        "feedback_id": bundle.feedback_id,
        "scenes": {
            din_cfg.scene_id: din_manifest,
            couch_cfg.scene_id: couch_manifest,
        },
        "verification_reports": verification_reports,
        "ssot_config": bundle.to_dict(),
    }

    spec_out = export_spec_json or (project_root / bundle.spec_output_relpath)
    spec_out.parent.mkdir(parents=True, exist_ok=True)
    spec_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[Denya Living Bundle] Saved spec -> {spec_out}\n"
        f"  dining wallZ={din_wall_world_z}, trioPortalZ={din_trio_portal_world_z}, aperture={din_frame_aperture_xyxy}\n"
        f"  couch  wallZ={couch_wall_world_z}, returnPortalZ={couch_return_portal_world_z}, aperture={couch_ap}"
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Denya living-room portal scenes (`denya_living_room` + `denya_couch_guitar`)."
    )
    parser.add_argument("--project-root", type=Path, default=_REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "prototype" / "assets" / "scenes",
    )
    parser.add_argument("--export-spec-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_denya_living_room_scenes(
        project_root=args.project_root.resolve(),
        output_dir=args.output_dir.resolve(),
        export_spec_json=args.export_spec_json.resolve() if args.export_spec_json else None,
    )


if __name__ == "__main__":
    main()
