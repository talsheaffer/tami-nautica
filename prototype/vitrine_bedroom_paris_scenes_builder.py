#!/usr/bin/env python3
"""
prototype/vitrine_bedroom_paris_scenes_builder.py
Offline neural 2.5D spatial asset builder and typed SSOT configuration models for:
  1. Scene 20 (`vitrine_keys_pen` / `scene20_vitrine_*`) [Feedback #42]:
     Tami's museum vitrine installation (`IMG_MUSE_0258`, *"בנה ביתך"*, 2026) decomposed into
     3D outer dark-wood box frame + brass glass knob (`fg`), 3D pink shelf + ceramic holder +
     upright pink pen + key bundle (`mid`), and recessed dark-blue vitrine cavity + gallery wall (`bg`).
  2. Scene 21 (`bar_giora_bedroom` / `scene21_bar_giora_*`) [Feedback #42]:
     Bar Giora bedroom in Haifa (`IMG_WORD_0211`) with 3D wardrobe & armchair (`fg`), panda-quilt
     bed & oval coffee table (`mid`), and Tami's vitrine (`mounted_wall_vitrine`) mounted on the
     wall below the framed painting rotated 130° clockwise (`rotationDeg: -130`, `flightRollDeg: -130`)
     so its cast shadow reconciles with the right-hand bedroom window light.
  3. Scene 22 (`paris_cite_white_table` / `scene22_paris_table_*`) [Feedback #43]:
     Tami's Cité Internationale des Arts studio in Paris (`IMG_WORD_0153`) with 3D wooden folding
     chair (`fg`), continuous chair-healed white tabletop + metal legs + flowerpot cabinet (`mid`),
     and the 1958 childhood photograph (`IMG_WHAT_0424`) perspective-placed on the white tabletop.
  4. Scene 23 (`toddler_tami_1958` / `scene23_tami_1958_*`) [Feedback #43]:
     Perspective-rectified zoomed-in 1958 portrait (`IMG_WHAT_0424`) of toddler Tami in 2.5D relief.

Strict No-Fallback Guarantee: executes real neural models (`Depth Anything V2`, `Meta SAM`,
`IS-Net` via `rembg`, and `LaMa` via `simple-lama-inpainting`) inside `.venv`.
"""

from __future__ import annotations

import argparse
import json
import math
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
        "`/Users/talsheaffer/code/tami/.venv/bin/python -u prototype/vitrine_bedroom_paris_scenes_builder.py`."
    ) from exc

from prototype.depth_topology import (
    ContactAnchorBox,
    ContactAnchoredLayerDepthConfig,
    LayerTopologyVerificationConfig,
    RoomDepthCleanConfig,
    clean_room_depth_under_cutouts,
    compute_room_relative_neural_protrusion,
    compute_world_z_map,
    extrapolate_depth_outward_edt,
    extract_contact_anchor_mask,
    extract_layer_window_from_expanded,
    glsl_smoothstep,
    solve_contact_anchored_layer_depth,
)
from prototype.new_scenes_builder import (
    _inpaint_masked_plate_lama,
    _prune_tiny_alpha_specks_rgba,
)
from prototype.scene_config import (
    BoundingBoxPrompt,
    CanvasDimensionsConfig,
    DepthModelConfig,
    InpaintingConfig,
    SamPointGroupPrompt,
    SegmentationConfig,
)
from prototype.scroll_purim_carpet_scenes_builder import (
    SceneCameraRigSpec,
    _assert_border_alpha_contract,
    _compute_edt_rim_p99_jump_world,
    _keep_largest_cc_u8,
    _mirror_reflect_rect_to_canvas,
    _pg,
    _prune_unanchored_cutout_islands,
    _segment_prompt_union_u8,
    _tight_alpha_bbox,
    _verify_and_gate_layer_topology,
)
from prototype.vision_primitives import (
    DepthAnythingEstimator,
    LamaOcclusionInpainter,
    SalientMatteExtractor,
    SamBoxMaskExtractor,
    decontaminate_matte_edge_rgb,
)


# ============================================================================
# Hierarchical, Serializable Configuration Data-Models (SSOT)
# ============================================================================


@dataclass(frozen=True)
class VitrineKeysPenSceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    asset_prefix: str
    source_relpath: str
    pen_shadow_heal_box_inner: tuple[int, int, int, int]
    plaster_wall_luma_exclude: float
    fg_sam_prompts: tuple[SamPointGroupPrompt, ...]
    mid_sam_prompts: tuple[SamPointGroupPrompt, ...]
    bg_inpaint_dilation_px: int
    bg_inpaint_blur_sigma: float
    bg_jpeg_quality: int
    knob_box_inner: tuple[int, int, int, int]
    frame_protrusion_norm: float
    knob_protrusion_norm: float
    room_depth_clean: RoomDepthCleanConfig
    mid_depth_solve: ContactAnchoredLayerDepthConfig
    fg_depth_solve: ContactAnchoredLayerDepthConfig
    camera_rig: SceneCameraRigSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "asset_prefix": self.asset_prefix,
            "source_relpath": self.source_relpath,
            "pen_shadow_heal_box_inner": list(self.pen_shadow_heal_box_inner),
            "plaster_wall_luma_exclude": self.plaster_wall_luma_exclude,
            "fg_sam_prompts": [p.to_dict() for p in self.fg_sam_prompts],
            "mid_sam_prompts": [p.to_dict() for p in self.mid_sam_prompts],
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "bg_inpaint_blur_sigma": self.bg_inpaint_blur_sigma,
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "knob_box_inner": list(self.knob_box_inner),
            "frame_protrusion_norm": self.frame_protrusion_norm,
            "knob_protrusion_norm": self.knob_protrusion_norm,
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "fg_depth_solve": self.fg_depth_solve.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VitrineKeysPenSceneSpec:
        ps = data["pen_shadow_heal_box_inner"]
        kb = data["knob_box_inner"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            asset_prefix=str(data["asset_prefix"]),
            source_relpath=str(data["source_relpath"]),
            pen_shadow_heal_box_inner=(int(ps[0]), int(ps[1]), int(ps[2]), int(ps[3])),
            plaster_wall_luma_exclude=float(data["plaster_wall_luma_exclude"]),
            fg_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["fg_sam_prompts"]),
            mid_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["mid_sam_prompts"]),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            bg_inpaint_blur_sigma=float(data["bg_inpaint_blur_sigma"]),
            bg_jpeg_quality=int(data["bg_jpeg_quality"]),
            knob_box_inner=(int(kb[0]), int(kb[1]), int(kb[2]), int(kb[3])),
            frame_protrusion_norm=float(data["frame_protrusion_norm"]),
            knob_protrusion_norm=float(data["knob_protrusion_norm"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            mid_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_depth_solve"]),
            fg_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["fg_depth_solve"]),
            camera_rig=SceneCameraRigSpec.from_dict(data["camera_rig"]),
        )


@dataclass(frozen=True)
class BarGioraBedroomSceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    asset_prefix: str
    bedroom_source_relpath: str
    vitrine_source_relpath: str
    scaled_full_wh: tuple[int, int]
    crop_y_range: tuple[int, int]
    vitrine_wall_center_xy: tuple[float, float]
    vitrine_wall_side_px: float
    vitrine_rotation_deg: float
    vitrine_flight_roll_deg: float
    wall_camera_focal_px: float
    wall_vanishing_point_xy: tuple[float, float]
    vitrine_indoor_rgb_scale: tuple[float, float, float]
    vitrine_shadow_ray_offset_src_xy: tuple[float, float]
    vitrine_shadow_attenuation_floor: float
    vitrine_shadow_gain: float
    vitrine_shadow_max_alpha_norm: float
    vitrine_shadow_blur_sigma_px: float
    vitrine_shadow_handoff_margin_px: int
    vitrine_shadow_plaster_rgb_scale: tuple[float, float, float]
    vitrine_frame_relief_world: float
    vitrine_knob_relief_world: float
    vitrine_cavity_back_relief_world: float
    vitrine_items_min_relief_world: float
    vitrine_items_max_relief_world: float
    vitrine_contact_rim_taper_px: float
    portal_id: str
    target_scene_id: str
    portal_label_he: str
    portal_short_badge_he: str
    fg_full_sam_prompts: tuple[SamPointGroupPrompt, ...]
    mid_sam_prompts: tuple[SamPointGroupPrompt, ...]
    bg_inpaint_dilation_px: int
    bg_inpaint_blur_sigma: float
    bg_jpeg_quality: int
    room_depth_clean: RoomDepthCleanConfig
    mid_depth_solve: ContactAnchoredLayerDepthConfig
    fg_depth_solve: ContactAnchoredLayerDepthConfig
    vitrine_depth_solve: ContactAnchoredLayerDepthConfig
    camera_rig: SceneCameraRigSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "asset_prefix": self.asset_prefix,
            "bedroom_source_relpath": self.bedroom_source_relpath,
            "vitrine_source_relpath": self.vitrine_source_relpath,
            "scaled_full_wh": list(self.scaled_full_wh),
            "crop_y_range": list(self.crop_y_range),
            "vitrine_wall_center_xy": list(self.vitrine_wall_center_xy),
            "vitrine_wall_side_px": self.vitrine_wall_side_px,
            "vitrine_rotation_deg": self.vitrine_rotation_deg,
            "vitrine_flight_roll_deg": self.vitrine_flight_roll_deg,
            "wall_camera_focal_px": self.wall_camera_focal_px,
            "wall_vanishing_point_xy": list(self.wall_vanishing_point_xy),
            "vitrine_indoor_rgb_scale": list(self.vitrine_indoor_rgb_scale),
            "vitrine_shadow_ray_offset_src_xy": list(self.vitrine_shadow_ray_offset_src_xy),
            "vitrine_shadow_attenuation_floor": self.vitrine_shadow_attenuation_floor,
            "vitrine_shadow_gain": self.vitrine_shadow_gain,
            "vitrine_shadow_max_alpha_norm": self.vitrine_shadow_max_alpha_norm,
            "vitrine_shadow_blur_sigma_px": self.vitrine_shadow_blur_sigma_px,
            "vitrine_shadow_handoff_margin_px": self.vitrine_shadow_handoff_margin_px,
            "vitrine_shadow_plaster_rgb_scale": list(self.vitrine_shadow_plaster_rgb_scale),
            "vitrine_frame_relief_world": self.vitrine_frame_relief_world,
            "vitrine_knob_relief_world": self.vitrine_knob_relief_world,
            "vitrine_cavity_back_relief_world": self.vitrine_cavity_back_relief_world,
            "vitrine_items_min_relief_world": self.vitrine_items_min_relief_world,
            "vitrine_items_max_relief_world": self.vitrine_items_max_relief_world,
            "vitrine_contact_rim_taper_px": self.vitrine_contact_rim_taper_px,
            "portal_id": self.portal_id,
            "target_scene_id": self.target_scene_id,
            "portal_label_he": self.portal_label_he,
            "portal_short_badge_he": self.portal_short_badge_he,
            "fg_full_sam_prompts": [p.to_dict() for p in self.fg_full_sam_prompts],
            "mid_sam_prompts": [p.to_dict() for p in self.mid_sam_prompts],
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "bg_inpaint_blur_sigma": self.bg_inpaint_blur_sigma,
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "fg_depth_solve": self.fg_depth_solve.to_dict(),
            "vitrine_depth_solve": self.vitrine_depth_solve.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BarGioraBedroomSceneSpec:
        sf = data["scaled_full_wh"]
        cy = data["crop_y_range"]
        mc = data["vitrine_wall_center_xy"]
        vp = data["wall_vanishing_point_xy"]
        irs = data["vitrine_indoor_rgb_scale"]
        sro = data["vitrine_shadow_ray_offset_src_xy"]
        spr = data["vitrine_shadow_plaster_rgb_scale"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            asset_prefix=str(data["asset_prefix"]),
            bedroom_source_relpath=str(data["bedroom_source_relpath"]),
            vitrine_source_relpath=str(data["vitrine_source_relpath"]),
            scaled_full_wh=(int(sf[0]), int(sf[1])),
            crop_y_range=(int(cy[0]), int(cy[1])),
            vitrine_wall_center_xy=(float(mc[0]), float(mc[1])),
            vitrine_wall_side_px=float(data["vitrine_wall_side_px"]),
            vitrine_rotation_deg=float(data["vitrine_rotation_deg"]),
            vitrine_flight_roll_deg=float(data["vitrine_flight_roll_deg"]),
            wall_camera_focal_px=float(data["wall_camera_focal_px"]),
            wall_vanishing_point_xy=(float(vp[0]), float(vp[1])),
            vitrine_indoor_rgb_scale=(float(irs[0]), float(irs[1]), float(irs[2])),
            vitrine_shadow_ray_offset_src_xy=(float(sro[0]), float(sro[1])),
            vitrine_shadow_attenuation_floor=float(data["vitrine_shadow_attenuation_floor"]),
            vitrine_shadow_gain=float(data["vitrine_shadow_gain"]),
            vitrine_shadow_max_alpha_norm=float(data["vitrine_shadow_max_alpha_norm"]),
            vitrine_shadow_blur_sigma_px=float(data["vitrine_shadow_blur_sigma_px"]),
            vitrine_shadow_handoff_margin_px=int(data["vitrine_shadow_handoff_margin_px"]),
            vitrine_shadow_plaster_rgb_scale=(float(spr[0]), float(spr[1]), float(spr[2])),
            vitrine_frame_relief_world=float(data["vitrine_frame_relief_world"]),
            vitrine_knob_relief_world=float(data["vitrine_knob_relief_world"]),
            vitrine_cavity_back_relief_world=float(data["vitrine_cavity_back_relief_world"]),
            vitrine_items_min_relief_world=float(data["vitrine_items_min_relief_world"]),
            vitrine_items_max_relief_world=float(data["vitrine_items_max_relief_world"]),
            vitrine_contact_rim_taper_px=float(data["vitrine_contact_rim_taper_px"]),
            portal_id=str(data["portal_id"]),
            target_scene_id=str(data["target_scene_id"]),
            portal_label_he=str(data["portal_label_he"]),
            portal_short_badge_he=str(data["portal_short_badge_he"]),
            fg_full_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["fg_full_sam_prompts"]
            ),
            mid_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["mid_sam_prompts"]),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            bg_inpaint_blur_sigma=float(data["bg_inpaint_blur_sigma"]),
            bg_jpeg_quality=int(data["bg_jpeg_quality"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            mid_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_depth_solve"]),
            fg_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["fg_depth_solve"]),
            vitrine_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["vitrine_depth_solve"]),
            camera_rig=SceneCameraRigSpec.from_dict(data["camera_rig"]),
        )


@dataclass(frozen=True)
class ParisCiteWhiteTableSceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    asset_prefix: str
    studio_source_relpath: str
    photo_source_relpath: str
    scaled_widescreen_wh: tuple[int, int]
    pad_top_bottom_px: int
    chair_y_bounds: tuple[int, int]
    chair_white_bar_luma_min: float
    chair_white_bar_rmb_max: float
    chair_grey_bar_luma_min: float
    chair_grey_bar_rmb_max: float
    chair_dark_floor_luma_max: float
    chair_dark_floor_rmb_max: float
    chair_inpaint_dilate_px: int
    fg_chair_sam_prompts: tuple[SamPointGroupPrompt, ...]
    mid_table_sam_prompts: tuple[SamPointGroupPrompt, ...]
    photo_src_crop_box: tuple[int, int, int, int]
    photo_dst_quad_inner_xy: tuple[tuple[float, float], ...]
    photo_contact_shadow_max_alpha_u8: int
    photo_contact_shadow_sigma_px: float
    photo_tabletop_clearance_world: float
    portal_id: str
    target_scene_id: str
    portal_label_he: str
    portal_short_badge_he: str
    bg_inpaint_dilation_px: int
    bg_inpaint_blur_sigma: float
    bg_jpeg_quality: int
    room_depth_clean: RoomDepthCleanConfig
    mid_depth_solve: ContactAnchoredLayerDepthConfig
    fg_depth_solve: ContactAnchoredLayerDepthConfig
    camera_rig: SceneCameraRigSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "asset_prefix": self.asset_prefix,
            "studio_source_relpath": self.studio_source_relpath,
            "photo_source_relpath": self.photo_source_relpath,
            "scaled_widescreen_wh": list(self.scaled_widescreen_wh),
            "pad_top_bottom_px": self.pad_top_bottom_px,
            "chair_y_bounds": list(self.chair_y_bounds),
            "chair_white_bar_luma_min": self.chair_white_bar_luma_min,
            "chair_white_bar_rmb_max": self.chair_white_bar_rmb_max,
            "chair_grey_bar_luma_min": self.chair_grey_bar_luma_min,
            "chair_grey_bar_rmb_max": self.chair_grey_bar_rmb_max,
            "chair_dark_floor_luma_max": self.chair_dark_floor_luma_max,
            "chair_dark_floor_rmb_max": self.chair_dark_floor_rmb_max,
            "chair_inpaint_dilate_px": self.chair_inpaint_dilate_px,
            "fg_chair_sam_prompts": [p.to_dict() for p in self.fg_chair_sam_prompts],
            "mid_table_sam_prompts": [p.to_dict() for p in self.mid_table_sam_prompts],
            "photo_src_crop_box": list(self.photo_src_crop_box),
            "photo_dst_quad_inner_xy": [list(pt) for pt in self.photo_dst_quad_inner_xy],
            "photo_contact_shadow_max_alpha_u8": self.photo_contact_shadow_max_alpha_u8,
            "photo_contact_shadow_sigma_px": self.photo_contact_shadow_sigma_px,
            "photo_tabletop_clearance_world": self.photo_tabletop_clearance_world,
            "portal_id": self.portal_id,
            "target_scene_id": self.target_scene_id,
            "portal_label_he": self.portal_label_he,
            "portal_short_badge_he": self.portal_short_badge_he,
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "bg_inpaint_blur_sigma": self.bg_inpaint_blur_sigma,
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "fg_depth_solve": self.fg_depth_solve.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParisCiteWhiteTableSceneSpec:
        sw = data["scaled_widescreen_wh"]
        cy = data["chair_y_bounds"]
        pc = data["photo_src_crop_box"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            asset_prefix=str(data["asset_prefix"]),
            studio_source_relpath=str(data["studio_source_relpath"]),
            photo_source_relpath=str(data["photo_source_relpath"]),
            scaled_widescreen_wh=(int(sw[0]), int(sw[1])),
            pad_top_bottom_px=int(data["pad_top_bottom_px"]),
            chair_y_bounds=(int(cy[0]), int(cy[1])),
            chair_white_bar_luma_min=float(data["chair_white_bar_luma_min"]),
            chair_white_bar_rmb_max=float(data["chair_white_bar_rmb_max"]),
            chair_grey_bar_luma_min=float(data["chair_grey_bar_luma_min"]),
            chair_grey_bar_rmb_max=float(data["chair_grey_bar_rmb_max"]),
            chair_dark_floor_luma_max=float(data["chair_dark_floor_luma_max"]),
            chair_dark_floor_rmb_max=float(data["chair_dark_floor_rmb_max"]),
            chair_inpaint_dilate_px=int(data["chair_inpaint_dilate_px"]),
            fg_chair_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["fg_chair_sam_prompts"]
            ),
            mid_table_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["mid_table_sam_prompts"]
            ),
            photo_src_crop_box=(int(pc[0]), int(pc[1]), int(pc[2]), int(pc[3])),
            photo_dst_quad_inner_xy=tuple(
                (float(pt[0]), float(pt[1])) for pt in data["photo_dst_quad_inner_xy"]
            ),
            photo_contact_shadow_max_alpha_u8=int(data["photo_contact_shadow_max_alpha_u8"]),
            photo_contact_shadow_sigma_px=float(data["photo_contact_shadow_sigma_px"]),
            photo_tabletop_clearance_world=float(data["photo_tabletop_clearance_world"]),
            portal_id=str(data["portal_id"]),
            target_scene_id=str(data["target_scene_id"]),
            portal_label_he=str(data["portal_label_he"]),
            portal_short_badge_he=str(data["portal_short_badge_he"]),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            bg_inpaint_blur_sigma=float(data["bg_inpaint_blur_sigma"]),
            bg_jpeg_quality=int(data["bg_jpeg_quality"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            mid_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_depth_solve"]),
            fg_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["fg_depth_solve"]),
            camera_rig=SceneCameraRigSpec.from_dict(data["camera_rig"]),
        )


@dataclass(frozen=True)
class ToddlerTami1958SceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    asset_prefix: str
    source_relpath: str
    src_inner_quad_xy: tuple[tuple[float, float], ...]
    dst_inner_quad_xy: tuple[tuple[float, float], ...]
    authentic_crop_margins_ltrb_px: tuple[int, int, int, int]
    sam_support_dilate_px: int
    cutout_core_side_top_margin_px: int
    mid_sam_prompts: tuple[SamPointGroupPrompt, ...]
    bg_inpaint_dilation_px: int
    bg_inpaint_blur_sigma: float
    bg_jpeg_quality: int
    room_depth_clean: RoomDepthCleanConfig
    mid_depth_solve: ContactAnchoredLayerDepthConfig
    camera_rig: SceneCameraRigSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "asset_prefix": self.asset_prefix,
            "source_relpath": self.source_relpath,
            "src_inner_quad_xy": [list(pt) for pt in self.src_inner_quad_xy],
            "dst_inner_quad_xy": [list(pt) for pt in self.dst_inner_quad_xy],
            "authentic_crop_margins_ltrb_px": list(self.authentic_crop_margins_ltrb_px),
            "sam_support_dilate_px": self.sam_support_dilate_px,
            "cutout_core_side_top_margin_px": self.cutout_core_side_top_margin_px,
            "mid_sam_prompts": [p.to_dict() for p in self.mid_sam_prompts],
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "bg_inpaint_blur_sigma": self.bg_inpaint_blur_sigma,
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToddlerTami1958SceneSpec:
        am = data["authentic_crop_margins_ltrb_px"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            asset_prefix=str(data["asset_prefix"]),
            source_relpath=str(data["source_relpath"]),
            src_inner_quad_xy=tuple((float(pt[0]), float(pt[1])) for pt in data["src_inner_quad_xy"]),
            dst_inner_quad_xy=tuple((float(pt[0]), float(pt[1])) for pt in data["dst_inner_quad_xy"]),
            authentic_crop_margins_ltrb_px=(int(am[0]), int(am[1]), int(am[2]), int(am[3])),
            sam_support_dilate_px=int(data["sam_support_dilate_px"]),
            cutout_core_side_top_margin_px=int(data["cutout_core_side_top_margin_px"]),
            mid_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["mid_sam_prompts"]),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            bg_inpaint_blur_sigma=float(data["bg_inpaint_blur_sigma"]),
            bg_jpeg_quality=int(data["bg_jpeg_quality"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            mid_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_depth_solve"]),
            camera_rig=SceneCameraRigSpec.from_dict(data["camera_rig"]),
        )


@dataclass(frozen=True)
class VitrineBedroomParisBundleConfig:
    inner_dimensions: CanvasDimensionsConfig
    expanded_dimensions: CanvasDimensionsConfig
    inner_offset_xy: tuple[int, int]
    scenes_output_relpath: str
    spec_output_abspath: str
    depth_model: DepthModelConfig
    segmentation: SegmentationConfig
    scene_inpainting: InpaintingConfig
    lama_max_working_side_px: int
    lama_roi_context_margin_px: int
    edt_rim_probe_px: int
    max_edt_rim_jump_world: float
    topology_verification: LayerTopologyVerificationConfig
    vitrine_keys_pen: VitrineKeysPenSceneSpec
    bar_giora_bedroom: BarGioraBedroomSceneSpec
    paris_cite_white_table: ParisCiteWhiteTableSceneSpec
    toddler_tami_1958: ToddlerTami1958SceneSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "inner_dimensions": self.inner_dimensions.to_dict(),
            "expanded_dimensions": self.expanded_dimensions.to_dict(),
            "inner_offset_xy": list(self.inner_offset_xy),
            "scenes_output_relpath": self.scenes_output_relpath,
            "spec_output_abspath": self.spec_output_abspath,
            "depth_model": self.depth_model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "scene_inpainting": self.scene_inpainting.to_dict(),
            "lama_max_working_side_px": self.lama_max_working_side_px,
            "lama_roi_context_margin_px": self.lama_roi_context_margin_px,
            "edt_rim_probe_px": self.edt_rim_probe_px,
            "max_edt_rim_jump_world": self.max_edt_rim_jump_world,
            "topology_verification": self.topology_verification.to_dict(),
            "vitrine_keys_pen": self.vitrine_keys_pen.to_dict(),
            "bar_giora_bedroom": self.bar_giora_bedroom.to_dict(),
            "paris_cite_white_table": self.paris_cite_white_table.to_dict(),
            "toddler_tami_1958": self.toddler_tami_1958.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VitrineBedroomParisBundleConfig:
        io = data["inner_offset_xy"]
        return cls(
            inner_dimensions=CanvasDimensionsConfig.from_dict(data["inner_dimensions"]),
            expanded_dimensions=CanvasDimensionsConfig.from_dict(data["expanded_dimensions"]),
            inner_offset_xy=(int(io[0]), int(io[1])),
            scenes_output_relpath=str(data["scenes_output_relpath"]),
            spec_output_abspath=str(data["spec_output_abspath"]),
            depth_model=DepthModelConfig.from_dict(data["depth_model"]),
            segmentation=SegmentationConfig.from_dict(data["segmentation"]),
            scene_inpainting=InpaintingConfig.from_dict(data["scene_inpainting"]),
            lama_max_working_side_px=int(data["lama_max_working_side_px"]),
            lama_roi_context_margin_px=int(data["lama_roi_context_margin_px"]),
            edt_rim_probe_px=int(data["edt_rim_probe_px"]),
            max_edt_rim_jump_world=float(data["max_edt_rim_jump_world"]),
            topology_verification=LayerTopologyVerificationConfig.from_dict(
                data["topology_verification"]
            ),
            vitrine_keys_pen=VitrineKeysPenSceneSpec.from_dict(data["vitrine_keys_pen"]),
            bar_giora_bedroom=BarGioraBedroomSceneSpec.from_dict(data["bar_giora_bedroom"]),
            paris_cite_white_table=ParisCiteWhiteTableSceneSpec.from_dict(
                data["paris_cite_white_table"]
            ),
            toddler_tami_1958=ToddlerTami1958SceneSpec.from_dict(data["toddler_tami_1958"]),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, raw_json: str) -> VitrineBedroomParisBundleConfig:
        return cls.from_dict(json.loads(raw_json))


# ============================================================================
# Scene 20 Builder: `vitrine_keys_pen` (`scene20_vitrine_*`)
# ============================================================================


def build_vitrine_keys_pen_scene(
    project_root: Path,
    cfg: VitrineBedroomParisBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    spec = cfg.vitrine_keys_pen
    iw, ih = cfg.inner_dimensions.width, cfg.inner_dimensions.height
    ew, eh = cfg.expanded_dimensions.width, cfg.expanded_dimensions.height
    ox, oy = cfg.inner_offset_xy
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Scene 20: {spec.scene_id}] Building 3-layer museum vitrine scene (mirror edges)...")
    src_img = (
        Image.open(project_root / spec.source_relpath)
        .convert("RGB")
        .resize((iw, ih), Image.Resampling.LANCZOS)
    )
    src_np = np.asarray(src_img, dtype=np.uint8)

    mid_u8 = _segment_prompt_union_u8(
        rgb_image=src_img,
        prompts=spec.mid_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=60,
        max_pinhole_fill_px=80,
    )
    fg_raw_u8 = _segment_prompt_union_u8(
        rgb_image=src_img,
        prompts=spec.fg_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=80,
        max_pinhole_fill_px=120,
    )
    luma = 0.299 * src_np[:, :, 0] + 0.587 * src_np[:, :, 1] + 0.114 * src_np[:, :, 2]
    fg_u8 = np.where((luma > spec.plaster_wall_luma_exclude) | (mid_u8 >= 128), np.uint8(0), fg_raw_u8)

    mid_rgb = decontaminate_matte_edge_rgb(src_np, mid_u8)
    mid_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            Image.fromarray(np.dstack([mid_rgb, mid_u8]), mode="RGBA"),
            min_area_px=spec.mid_depth_solve.min_component_area_px,
            alpha_thresh_u8=spec.mid_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.mid_depth_solve,
    )
    fg_rgb = decontaminate_matte_edge_rgb(src_np, fg_u8)
    fg_inner_canvas = _prune_tiny_alpha_specks_rgba(
        Image.fromarray(np.dstack([fg_rgb, fg_u8]), mode="RGBA"),
        min_area_px=spec.fg_depth_solve.min_component_area_px,
        alpha_thresh_u8=spec.fg_depth_solve.alpha_threshold_u8,
    )

    _assert_border_alpha_contract(mid_canvas, f"{spec.scene_id}.mid")
    _assert_border_alpha_contract(fg_inner_canvas, f"{spec.scene_id}.fg_inner")
    mid_np = np.asarray(mid_canvas, dtype=np.uint8)
    fg_inner_np = np.asarray(fg_inner_canvas, dtype=np.uint8)

    fg_2700_np = _mirror_reflect_rect_to_canvas(fg_inner_np, (ew, eh), (ox, oy))
    fg_2700_np[:2, :, :] = 0
    fg_2700_np[-2:, :, :] = 0
    fg_2700_np[:, :2, :] = 0
    fg_2700_np[:, -2:, :] = 0
    fg_canvas = Image.fromarray(fg_2700_np, mode="RGBA")
    _assert_border_alpha_contract(fg_canvas, f"{spec.scene_id}.fg")
    fg_np = np.asarray(fg_canvas, dtype=np.uint8)

    # Inpaint pink shelf, pen, keys, pen cast shadow, and outer frame out of inner cavity/wall
    bg_hole = np.where((mid_np[:, :, 3] >= 16) | (fg_inner_np[:, :, 3] >= 16), 255, 0).astype(np.uint8)
    sx0, sy0, sx1, sy1 = spec.pen_shadow_heal_box_inner
    shadow_dark = (luma[sy0:sy1, sx0:sx1] < 38.0) & (src_np[sy0:sy1, sx0:sx1, 2] < 62)
    bg_hole[sy0:sy1, sx0:sx1] = np.where(shadow_dark, 255, bg_hole[sy0:sy1, sx0:sx1])

    clean_inner_rgb = _inpaint_masked_plate_lama(
        src_rgb=src_img,
        binary_mask_u8=bg_hole,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.bg_inpaint_dilation_px,
        blur_sigma=spec.bg_inpaint_blur_sigma,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )
    clean_inner_np = np.asarray(clean_inner_rgb, dtype=np.uint8)
    bg_inpainted_rgb = Image.fromarray(
        _mirror_reflect_rect_to_canvas(clean_inner_np, (ew, eh), (ox, oy)),
        mode="RGB",
    )

    master_2700_rgb = Image.fromarray(
        _mirror_reflect_rect_to_canvas(src_np, (ew, eh), (ox, oy)),
        mode="RGB",
    )
    raw_intact_depth_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(master_2700_rgb) * 255.0), 0, 255
    ).astype(np.uint8)

    inpainted_room_depth_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(bg_inpainted_rgb), dtype=np.uint8
    )
    bg_depth_healed_u8 = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(mid_np, fg_np),
        cfg=spec.room_depth_clean,
    )
    bg_depth_clean_u8 = _mirror_reflect_rect_to_canvas(
        bg_depth_healed_u8[oy : oy + ih, ox : ox + iw],
        (ew, eh),
        (ox, oy),
    )

    mid_da2 = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_depth_u8,
        cutout_rgba=mid_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.mid_depth_solve.alpha_threshold_u8,
    )
    mid_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=mid_np,
        cfg=spec.mid_depth_solve,
        neural_protrusion_norm=np.clip(0.55 * mid_da2 + 0.45, 0.0, 1.0).astype(np.float32),
    )

    fg_prot = np.full((ih, iw), spec.frame_protrusion_norm, dtype=np.float32)
    kx0, ky0, kx1, ky1 = spec.knob_box_inner
    fg_prot[ky0:ky1, kx0:kx1] = spec.knob_protrusion_norm
    fg_inner_solve_cfg = ContactAnchoredLayerDepthConfig(
        bg_z=spec.fg_depth_solve.bg_z,
        room_depth_amplitude=spec.fg_depth_solve.room_depth_amplitude,
        room_pin_margin_uv=spec.fg_depth_solve.room_pin_margin_uv,
        layer_z=spec.fg_depth_solve.layer_z,
        layer_depth_amplitude=spec.fg_depth_solve.layer_depth_amplitude,
        layer_pin_margin_uv=0.0,
        contact_boxes=tuple(
            ContactAnchorBox(
                x0=b.x0 - ox,
                y0=b.y0 - oy,
                x1=b.x1 - ox,
                y1=b.y1 - oy,
                mode=b.mode,
                band_px=b.band_px,
            )
            for b in spec.fg_depth_solve.contact_boxes
        ),
        neural_relief_amplitude=spec.fg_depth_solve.neural_relief_amplitude,
        contact_relief_taper_px=spec.fg_depth_solve.contact_relief_taper_px,
        enforce_front_of_host=spec.fg_depth_solve.enforce_front_of_host,
        alpha_threshold_u8=spec.fg_depth_solve.alpha_threshold_u8,
        min_component_area_px=spec.fg_depth_solve.min_component_area_px,
    )
    fg_inner_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=fg_inner_np,
        cfg=fg_inner_solve_cfg,
        neural_protrusion_norm=fg_prot,
    )
    fg_mirrored_d_norm = (
        _mirror_reflect_rect_to_canvas(fg_inner_solved.depth_u8, (ew, eh), (ox, oy)).astype(np.float32)
        / 255.0
    )
    fg_opaque_2700 = fg_np[:, :, 3] >= spec.fg_depth_solve.alpha_threshold_u8
    fg_depth_2700_u8 = np.rint(
        extrapolate_depth_outward_edt(fg_mirrored_d_norm, fg_opaque_2700) * 255.0
    ).astype(np.uint8)
    fg_contact_2700 = extract_contact_anchor_mask(fg_opaque_2700, spec.fg_depth_solve)

    mid_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.mid",
        bg_depth_clean_u8,
        mid_np,
        mid_solved.depth_u8,
        mid_solved.contact_mask,
        spec.mid_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )
    fg_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.fg",
        bg_depth_clean_u8,
        fg_np,
        fg_depth_2700_u8,
        fg_contact_2700,
        spec.fg_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )

    pfx = spec.asset_prefix
    bg_inpainted_rgb.save(scenes_dir / f"{pfx}_bg_inpainted.jpg", quality=spec.bg_jpeg_quality, optimize=True)
    Image.fromarray(bg_depth_clean_u8, mode="L").save(scenes_dir / f"{pfx}_depth.png")
    mid_canvas.save(scenes_dir / f"{pfx}_mid.png")
    Image.fromarray(mid_solved.depth_u8, mode="L").save(scenes_dir / f"{pfx}_mid_depth.png")
    fg_canvas.save(scenes_dir / f"{pfx}_fg.png")
    Image.fromarray(fg_depth_2700_u8, mode="L").save(scenes_dir / f"{pfx}_fg_depth.png")

    rig = spec.camera_rig
    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "bgExpandedWidth": ew,
        "bgExpandedHeight": eh,
        "bgTexture": f"assets/scenes/{pfx}_bg_inpainted.jpg",
        "depthTexture": f"assets/scenes/{pfx}_depth.png",
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
        "fgExpanded": True,
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
        "portraitInitialYawDeg": rig.portrait_initial_yaw_deg,
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
        "midCoverageRatio": round(float((mid_np[:, :, 3] >= 20).mean()), 4),
        "fgCoverageRatio": round(float((fg_np[:, :, 3] >= 20).mean()), 4),
        "depthTopologyVerification": {"mid": mid_verify, "fg": fg_verify},
    }


# ============================================================================
# Scene 21 Builder: `bar_giora_bedroom` (`scene21_bar_giora_*`)
# ============================================================================


def _fit_quadratic_wall_illumination(
    luma_f32: np.ndarray,
    unshadowed_mask: np.ndarray,
) -> np.ndarray:
    """Fits a smooth 2D quadratic polynomial illumination field to unshadowed museum plaster."""
    h, w = luma_f32.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xn = (xx.astype(np.float64) - 0.5 * w) / float(w)
    yn = (yy.astype(np.float64) - 0.5 * h) / float(h)
    design_full = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], axis=-1)
    coeffs, _, _, _ = np.linalg.lstsq(
        design_full[unshadowed_mask], luma_f32[unshadowed_mask].astype(np.float64), rcond=None
    )
    fitted = np.tensordot(design_full, coeffs, axes=([-1], [0]))
    return np.clip(fitted, 1.0, 255.0).astype(np.float32)


def _build_wall_perspective_homography(
    src_center_xy: tuple[float, float],
    src_box_side_px: float,
    inner_wh: tuple[int, int],
    spec: BarGioraBedroomSceneSpec,
) -> tuple[np.ndarray, float]:
    """
    Constructs the exact 3x3 pinhole perspective homography mapping the padded museum vitrine
    plane (`src_center_xy`) onto the receding 3D Bar Giora bedroom wall plane at
    `spec.vitrine_wall_center_xy`, rotated in-plane by `spec.vitrine_rotation_deg` (`-130°` CW)
    and foreshortened toward the wall's horizontal vanishing point `spec.wall_vanishing_point_xy`.
    Returns `(H_wall_3x3, horiz_foreshortening_factor)`.
    """
    iw, _ = inner_wh
    f = float(spec.wall_camera_focal_px)
    vp_x, vp_y = spec.wall_vanishing_point_xy
    cx_cam = 0.5 * float(iw)
    cy_cam = float(vp_y)

    k_mat = np.array([[f, 0.0, cx_cam], [0.0, f, cy_cam], [0.0, 0.0, 1.0]], dtype=np.float64)
    k_inv = np.linalg.inv(k_mat)

    u_raw = k_inv @ np.array([vp_x, vp_y, 1.0], dtype=np.float64)
    u_hat = u_raw / np.linalg.norm(u_raw)
    v_hat = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    dst_cx, dst_cy = spec.vitrine_wall_center_xy
    p0 = k_inv @ np.array([dst_cx, dst_cy, 1.0], dtype=np.float64)

    s_world = float(spec.vitrine_wall_side_px) / (f * float(src_box_side_px))
    theta = math.radians(float(spec.vitrine_rotation_deg))
    alpha = math.cos(theta)
    beta = math.sin(theta)

    # Matches OpenCV getRotationMatrix2D(center, angle_deg, 1.0) on wall tangent plane (u_hat, v_hat)
    col_x = s_world * (alpha * u_hat - beta * v_hat)
    col_y = s_world * (beta * u_hat + alpha * v_hat)
    sx0, sy0 = src_center_xy
    col_t = p0 - col_x * sx0 - col_y * sy0

    h_wall = k_mat @ np.column_stack([col_x, col_y, col_t])
    h_wall = h_wall / h_wall[2, 2]
    horiz_foreshortening = float(u_hat[0] - p0[0] * u_hat[2])
    return h_wall.astype(np.float64), horiz_foreshortening


def _build_flush_wall_perspective_vitrine(
    vitrine_src_1500_img: Image.Image,
    clean_bedroom_inner_np: np.ndarray,
    bg_depth_clean_u8: np.ndarray,
    cfg: VitrineBedroomParisBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
) -> tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray, list[float], float]:
    """
    Synthesizes Scene 21's wall-mounted vitrine (`#53` & `#54`):
      1. Segments the dark-wood vitrine cleanly via SAM + edge decontamination (zero white museum plaster rim).
      2. Extracts the authentic museum cast wall shadow via quadratic plaster illumination + swept-box
         completion past the bottom-right DSLR crop so zero straight crop edge exists, tinted with
         darkened local Bar Giora bedroom plaster RGB for seamless blending (`#53`).
      3. Perspective-warps box + shadow + 3D relief onto the receding 3D Bar Giora wall plane (`H_wall`).
      4. Solves world-Z depth directly on top of `Z_world_wall(x, y)` (`ΔZ = 0` across 100% of shadow
         and 360° wooden box contact rim) so the vitrine is flush against and parallel to the wall (`#54`).
    """
    spec = cfg.bar_giora_bedroom
    vspec = cfg.vitrine_keys_pen
    iw, ih = cfg.inner_dimensions.width, cfg.inner_dimensions.height

    vsrc_np = np.asarray(vitrine_src_1500_img, dtype=np.uint8)
    vluma = (
        0.299 * vsrc_np[:, :, 0].astype(np.float32)
        + 0.587 * vsrc_np[:, :, 1].astype(np.float32)
        + 0.114 * vsrc_np[:, :, 2].astype(np.float32)
    )

    vit_mid_u8 = _segment_prompt_union_u8(
        rgb_image=vitrine_src_1500_img,
        prompts=vspec.mid_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=60,
        max_pinhole_fill_px=80,
    )
    vit_fg_raw_u8 = _segment_prompt_union_u8(
        rgb_image=vitrine_src_1500_img,
        prompts=vspec.fg_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=80,
        max_pinhole_fill_px=120,
    )
    vit_fg_u8 = np.where(vluma > vspec.plaster_wall_luma_exclude, np.uint8(0), vit_fg_raw_u8)

    frame_closed = cv2.morphologyEx(
        ((vit_fg_u8 >= 96) | (vit_mid_u8 >= 96)).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    box_solid_src = ndi.binary_fill_holes(frame_closed)
    if float(box_solid_src.mean()) < 0.42:
        raise RuntimeError(
            f"Vitrine solid cavity fill failed (coverage={float(box_solid_src.mean()):.3f} < 0.42)."
        )

    box_inner_core = ndi.binary_erosion(box_solid_src, iterations=2)
    box_alpha_src = np.where(
        box_inner_core,
        np.uint8(255),
        np.where(box_solid_src, np.maximum(vit_fg_u8, np.uint8(210)), vit_fg_u8),
    )
    box_rgb_decontam = decontaminate_matte_edge_rgb(vsrc_np, box_alpha_src).astype(np.float32)
    rgb_scale = np.asarray(spec.vitrine_indoor_rgb_scale, dtype=np.float32)[None, None, :]
    box_rgb_src = np.clip(np.rint(box_rgb_decontam * rgb_scale), 0.0, 255.0).astype(np.uint8)

    bys, bxs = np.where(box_solid_src)
    box_cx_src = 0.5 * float(int(bxs.min()) + int(bxs.max()))
    box_cy_src = 0.5 * float(int(bys.min()) + int(bys.max()))
    src_box_side_px = 0.5 * float((int(bxs.max()) - int(bxs.min())) + (int(bys.max()) - int(bys.min())))

    # Measure authentic directional cast wall shadow on museum plaster & complete bottom-right DSLR crop
    yy_s, xx_s = np.mgrid[0:ih, 0:iw]
    unshadowed_plaster = (
        (xx_s < 296)
        | (yy_s < 64)
        | ((xx_s > 1200) & (yy_s < 192))
        | ((yy_s > 952) & (xx_s < 545))
    ) & (~ndi.binary_dilation(box_solid_src, iterations=8))
    wall_ref_luma = _fit_quadratic_wall_illumination(vluma, unshadowed_plaster)
    atten_meas_src = np.clip((wall_ref_luma - vluma) / np.maximum(wall_ref_luma, 1.0), 0.0, 1.0)
    atten_meas_src[ndi.binary_dilation(box_solid_src, iterations=4)] = 0.0

    pad_tl = 120
    pad_br_x, pad_br_y = 480, 480
    pw, ph = iw + pad_tl + pad_br_x, ih + pad_tl + pad_br_y

    box_solid_pad = np.zeros((ph, pw), dtype=np.bool_)
    box_solid_pad[pad_tl : pad_tl + ih, pad_tl : pad_tl + iw] = box_solid_src
    box_rgb_pad = np.zeros((ph, pw, 3), dtype=np.uint8)
    box_rgb_pad[pad_tl : pad_tl + ih, pad_tl : pad_tl + iw] = box_rgb_src
    box_alpha_pad = np.zeros((ph, pw), dtype=np.uint8)
    box_alpha_pad[pad_tl : pad_tl + ih, pad_tl : pad_tl + iw] = box_alpha_src

    dx_ray, dy_ray = spec.vitrine_shadow_ray_offset_src_xy
    swept_env = np.zeros((ph, pw), dtype=np.float32)
    box_solid_f = box_solid_pad.astype(np.float32)
    n_sweep = 48
    for step_idx in range(1, n_sweep + 1):
        s_frac = float(step_idx) / float(n_sweep)
        m_shift = np.array(
            [[1.0, 0.0, s_frac * dx_ray], [0.0, 1.0, s_frac * dy_ray]], dtype=np.float32
        )
        shifted = cv2.warpAffine(
            box_solid_f, m_shift, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        swept_env = np.maximum(swept_env, shifted * (1.0 - 0.34 * s_frac))
    swept_env = ndi.gaussian_filter(swept_env, sigma=24.0).astype(np.float32)

    swept_inner = swept_env[pad_tl : pad_tl + ih, pad_tl : pad_tl + iw]
    calib_zone = (
        (swept_inner > 0.30)
        & (~ndi.binary_dilation(box_solid_src, iterations=8))
        & (xx_s >= 600)
        & (xx_s <= iw - 40)
        & (yy_s >= 230)
        & (yy_s <= ih - 30)
    )
    if not np.any(calib_zone):
        raise RuntimeError("Vitrine shadow calibration zone is empty.")
    calib_scale = float(np.median(atten_meas_src[calib_zone] / np.maximum(swept_inner[calib_zone], 1e-4)))
    swept_atten_pad = np.clip(swept_env * calib_scale, 0.0, 1.0)

    atten_meas_pad = np.zeros((ph, pw), dtype=np.float32)
    atten_meas_pad[pad_tl : pad_tl + ih, pad_tl : pad_tl + iw] = ndi.gaussian_filter(
        atten_meas_src, sigma=14.0
    )

    # Smooth cosine handoff from real measured museum plaster shadow to calibrated swept completion
    hm = float(max(8, spec.vitrine_shadow_handoff_margin_px))
    rx = np.clip((float(iw - 1) - xx_s.astype(np.float32)) / hm, 0.0, 1.0)
    ry = np.clip((float(ih - 1) - yy_s.astype(np.float32)) / hm, 0.0, 1.0)
    dist_from_box = ndi.distance_transform_edt(~box_solid_src).astype(np.float32)
    r_box = np.clip((dist_from_box - 4.0) / 6.0, 0.0, 1.0)
    w_inner = (0.5 - 0.5 * np.cos(np.pi * rx)) * (0.5 - 0.5 * np.cos(np.pi * ry)) * (
        0.5 - 0.5 * np.cos(np.pi * r_box)
    )
    w_inner = np.where(swept_inner > 0.012, w_inner, 0.0).astype(np.float32)
    w_meas_pad = np.zeros((ph, pw), dtype=np.float32)
    w_meas_pad[pad_tl : pad_tl + ih, pad_tl : pad_tl + iw] = w_inner

    atten_combined = w_meas_pad * atten_meas_pad + (1.0 - w_meas_pad) * swept_atten_pad
    soft_toe = glsl_smoothstep(0.004, spec.vitrine_shadow_attenuation_floor, atten_combined)
    dir_shadow_pad = np.clip(
        atten_combined * soft_toe * spec.vitrine_shadow_gain,
        0.0,
        spec.vitrine_shadow_max_alpha_norm,
    ).astype(np.float32)
    # Tight 1.4px wall-contact crevice occlusion around all 4 wooden sides so unshadowed rim sits flush
    dist_box_pad = ndi.distance_transform_edt(~box_solid_pad).astype(np.float32)
    crevice_ao_pad = (0.28 * np.exp(-0.5 * (dist_box_pad / 9.5) ** 2)).astype(np.float32)

    # 3D relief map inside the vitrine (world Z units forward of the bedroom wall plane)
    vit_da2 = depth_estimator.predict_depth(vitrine_src_1500_img).astype(np.float32)
    mid_mask_src = vit_mid_u8 >= 96
    if np.any(mid_mask_src):
        d_lo, d_hi = np.percentile(vit_da2[mid_mask_src], [10.0, 95.0])
        mid_norm = np.clip((vit_da2 - float(d_lo)) / max(1e-4, float(d_hi - d_lo)), 0.0, 1.0)
    else:
        mid_norm = np.zeros_like(vit_da2)

    relief_src = np.zeros((ih, iw), dtype=np.float32)
    relief_src[box_solid_src] = spec.vitrine_cavity_back_relief_world
    items_relief = spec.vitrine_items_min_relief_world + mid_norm * (
        spec.vitrine_items_max_relief_world - spec.vitrine_items_min_relief_world
    )
    w_mid = (vit_mid_u8.astype(np.float32) / 255.0)
    relief_src = np.where(
        box_solid_src,
        (1.0 - w_mid) * relief_src + w_mid * items_relief,
        0.0,
    )
    frame_only_w = np.clip((vit_fg_u8.astype(np.float32) - vit_mid_u8.astype(np.float32)) / 255.0, 0.0, 1.0)
    relief_src = np.where(
        box_solid_src,
        np.maximum(relief_src, frame_only_w * spec.vitrine_frame_relief_world),
        0.0,
    )
    kx0, ky0, kx1, ky1 = vspec.knob_box_inner
    knob_w = (vit_fg_u8[ky0:ky1, kx0:kx1].astype(np.float32) / 255.0)
    relief_src[ky0:ky1, kx0:kx1] = np.maximum(
        relief_src[ky0:ky1, kx0:kx1], knob_w * spec.vitrine_knob_relief_world
    )
    relief_src = ndi.gaussian_filter(relief_src, sigma=1.8).astype(np.float32)
    relief_pad = np.zeros((ph, pw), dtype=np.float32)
    relief_pad[pad_tl : pad_tl + ih, pad_tl : pad_tl + iw] = relief_src

    # Perspective-warp vitrine box + cast shadow + relief onto receding 3D Bar Giora wall plane
    h_wall, horiz_foreshortening = _build_wall_perspective_homography(
        src_center_xy=(pad_tl + box_cx_src, pad_tl + box_cy_src),
        src_box_side_px=src_box_side_px,
        inner_wh=(iw, ih),
        spec=spec,
    )
    warped_box_rgb = cv2.warpPerspective(
        box_rgb_pad, h_wall, (iw, ih), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    warped_box_a_u8 = cv2.warpPerspective(
        box_alpha_pad, h_wall, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    warped_dir_sh = cv2.warpPerspective(
        dir_shadow_pad, h_wall, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    warped_crevice_ao = cv2.warpPerspective(
        crevice_ao_pad, h_wall, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    warped_relief = cv2.warpPerspective(
        relief_pad, h_wall, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )

    if spec.vitrine_shadow_blur_sigma_px > 0.0:
        warped_dir_sh = ndi.gaussian_filter(warped_dir_sh, sigma=spec.vitrine_shadow_blur_sigma_px).astype(
            np.float32
        )
    warped_crevice_ao = ndi.gaussian_filter(warped_crevice_ao, sigma=0.7).astype(np.float32)
    warped_sh_norm = np.clip(np.maximum(warped_dir_sh, warped_crevice_ao), 0.0, 0.46).astype(np.float32)

    # Local darkened Bar Giora plaster RGB for 100% seamless alpha transition into the wall (#53)
    smooth_plaster = ndi.gaussian_filter(clean_bedroom_inner_np.astype(np.float32), sigma=(4.0, 4.0, 0.0))
    sh_rgb_scale = np.asarray(spec.vitrine_shadow_plaster_rgb_scale, dtype=np.float32)[None, None, :]
    shadow_rgb = np.clip(smooth_plaster * sh_rgb_scale, 0.0, 255.0)

    a_box = (warped_box_a_u8.astype(np.float32) / 255.0)[:, :, None]
    a_sh = np.clip(warped_sh_norm, 0.0, 1.0)[:, :, None]
    a_out = a_box + a_sh * (1.0 - a_box)
    rgb_out = (a_box * warped_box_rgb.astype(np.float32) + a_sh * (1.0 - a_box) * shadow_rgb) / np.maximum(
        a_out, 1e-6
    )

    out_rgba = np.zeros((ih, iw, 4), dtype=np.uint8)
    out_rgba[:, :, :3] = np.clip(np.rint(rgb_out), 0.0, 255.0).astype(np.uint8)
    out_rgba[:, :, 3] = np.clip(np.rint(a_out[:, :, 0] * 255.0), 0.0, 255.0).astype(np.uint8)
    out_rgba[:2, :, :] = 0
    out_rgba[-2:, :, :] = 0
    out_rgba[:, :2, :] = 0
    out_rgba[:, -2:, :] = 0

    # Solve 3D depth flush & parallel to the receding bedroom wall Z_world_wall(x, y) (#54)
    vsolve = spec.vitrine_depth_solve
    room_z_exp = compute_world_z_map(
        bg_depth_clean_u8, vsolve.bg_z, vsolve.room_depth_amplitude, vsolve.room_pin_margin_uv
    )
    z_world_wall = extract_layer_window_from_expanded(room_z_exp, (ih, iw))

    warped_box_solid = warped_box_a_u8 >= 140
    eroded_box = ndi.binary_erosion(warped_box_solid, iterations=3)
    dist_inside_rim = ndi.distance_transform_edt(eroded_box).astype(np.float32)
    w_box_contact = glsl_smoothstep(0.0, float(spec.vitrine_contact_rim_taper_px), dist_inside_rim)

    z_world_vit = z_world_wall + 0.05 + w_box_contact * warped_relief
    d_norm = np.clip(
        (z_world_vit - float(vsolve.layer_z)) / float(vsolve.layer_depth_amplitude),
        0.0,
        1.0,
    ).astype(np.float32)
    active_mask = out_rgba[:, :, 3] >= vsolve.alpha_threshold_u8
    vit_depth_u8 = np.rint(extrapolate_depth_outward_edt(d_norm, active_mask) * 255.0).astype(np.uint8)
    vit_contact_mask = extract_contact_anchor_mask(active_mask, vsolve)
    vit_world_z = compute_world_z_map(vit_depth_u8, vsolve.layer_z, vsolve.layer_depth_amplitude, 0.0)

    cx_dst, cy_dst = spec.vitrine_wall_center_xy
    half_w = round(0.5 * spec.vitrine_wall_side_px * horiz_foreshortening + 2.0, 1)
    half_h = round(0.5 * spec.vitrine_wall_side_px + 2.0, 1)
    portal_box = [
        round(cx_dst - half_w, 1),
        round(cy_dst - half_h, 1),
        round(cx_dst + half_w, 1),
        round(cy_dst + half_h, 1),
    ]
    portal_world_z = round(float(np.median(vit_world_z[warped_box_solid])), 2)

    return (
        Image.fromarray(out_rgba, mode="RGBA"),
        vit_depth_u8,
        vit_contact_mask,
        vit_world_z,
        portal_box,
        portal_world_z,
    )


def build_bar_giora_bedroom_scene(
    project_root: Path,
    cfg: VitrineBedroomParisBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    spec = cfg.bar_giora_bedroom
    iw, ih = cfg.inner_dimensions.width, cfg.inner_dimensions.height
    ew, eh = cfg.expanded_dimensions.width, cfg.expanded_dimensions.height
    ox, oy = cfg.inner_offset_xy
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Scene 21: {spec.scene_id}] Building Bar Giora bedroom + flush wall-perspective vitrine & shadow..."
    )
    fw, fh = spec.scaled_full_wh
    cy0, cy1 = spec.crop_y_range
    dest_full_xy = (ox, oy - cy0)
    full_img = (
        Image.open(project_root / spec.bedroom_source_relpath)
        .convert("RGB")
        .resize((fw, fh), Image.Resampling.LANCZOS)
    )
    full_np = np.asarray(full_img, dtype=np.uint8)
    inner_np = full_np[cy0:cy1, :, :]
    inner_img = Image.fromarray(inner_np, mode="RGB")

    # 1. Segment wardrobe + armchair on full 1500x1125 bedroom plate so mirror reflection is authentic
    fg_full_u8 = _segment_prompt_union_u8(
        rgb_image=full_img,
        prompts=spec.fg_full_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=250,
        max_pinhole_fill_px=180,
    )
    fg_full_rgb = decontaminate_matte_edge_rgb(full_np, fg_full_u8)
    fg_full_solve_cfg = ContactAnchoredLayerDepthConfig(
        bg_z=spec.fg_depth_solve.bg_z,
        room_depth_amplitude=spec.fg_depth_solve.room_depth_amplitude,
        room_pin_margin_uv=spec.fg_depth_solve.room_pin_margin_uv,
        layer_z=spec.fg_depth_solve.layer_z,
        layer_depth_amplitude=spec.fg_depth_solve.layer_depth_amplitude,
        layer_pin_margin_uv=0.0,
        contact_boxes=tuple(
            ContactAnchorBox(
                x0=b.x0 - dest_full_xy[0],
                y0=b.y0 - dest_full_xy[1],
                x1=b.x1 - dest_full_xy[0],
                y1=b.y1 - dest_full_xy[1],
                mode=b.mode,
                band_px=b.band_px,
            )
            for b in spec.fg_depth_solve.contact_boxes
        ),
        neural_relief_amplitude=spec.fg_depth_solve.neural_relief_amplitude,
        contact_relief_taper_px=spec.fg_depth_solve.contact_relief_taper_px,
        enforce_front_of_host=spec.fg_depth_solve.enforce_front_of_host,
        alpha_threshold_u8=spec.fg_depth_solve.alpha_threshold_u8,
        min_component_area_px=spec.fg_depth_solve.min_component_area_px,
    )
    fg_full_clean = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            Image.fromarray(np.dstack([fg_full_rgb, fg_full_u8]), mode="RGBA"),
            min_area_px=fg_full_solve_cfg.min_component_area_px,
            alpha_thresh_u8=fg_full_solve_cfg.alpha_threshold_u8,
        ),
        solve_cfg=fg_full_solve_cfg,
    )
    fg_full_rgba_np = np.asarray(fg_full_clean, dtype=np.uint8)

    # 2. Segment bed + coffee table (`mid`) on inner 1500x1000 window
    mid_raw_u8 = _segment_prompt_union_u8(
        rgb_image=inner_img,
        prompts=spec.mid_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=250,
        max_pinhole_fill_px=180,
    )
    mid_filled = ndi.binary_fill_holes(mid_raw_u8 >= 96)
    mid_raw_u8 = np.where(mid_filled & (mid_raw_u8 < 255), np.uint8(255), mid_raw_u8)
    mid_u8 = np.where(fg_full_rgba_np[cy0:cy1, :, 3] >= 128, np.uint8(0), mid_raw_u8)
    mid_u8[0, :] = 0
    mid_u8[-1, :] = 0
    mid_u8[:, 0] = 0
    mid_u8[:, -1] = 0
    mid_u8[-3:, :] = np.minimum(
        mid_u8[-3:, :],
        np.array([[180], [90], [0]], dtype=np.uint8),
    )

    mid_rgb = decontaminate_matte_edge_rgb(inner_np, mid_u8)
    mid_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            Image.fromarray(np.dstack([mid_rgb, mid_u8]), mode="RGBA"),
            min_area_px=spec.mid_depth_solve.min_component_area_px,
            alpha_thresh_u8=spec.mid_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.mid_depth_solve,
    )
    _assert_border_alpha_contract(mid_canvas, f"{spec.scene_id}.mid")
    mid_np = np.asarray(mid_canvas, dtype=np.uint8)

    # 3. Mirror fg_full_rgba onto 2700x1800 (`fgExpanded: True`)
    fg_2700_np = _mirror_reflect_rect_to_canvas(fg_full_rgba_np, (ew, eh), dest_full_xy)
    fg_2700_np[:2, :, :] = 0
    fg_2700_np[-2:, :, :] = 0
    fg_2700_np[:, :2, :] = 0
    fg_2700_np[:, -2:, :] = 0
    fg_canvas = Image.fromarray(fg_2700_np, mode="RGBA")
    _assert_border_alpha_contract(fg_canvas, f"{spec.scene_id}.fg")
    fg_np = np.asarray(fg_canvas, dtype=np.uint8)

    # 4. Inpaint fg + mid out of full 1500x1125 bedroom plate and mirror onto 2700x1800 (#48)
    full_hole = (fg_full_rgba_np[:, :, 3] >= 16).astype(np.uint8) * 255
    full_hole[cy0:cy1, :] = np.where(mid_np[:, :, 3] >= 16, np.uint8(255), full_hole[cy0:cy1, :])
    full_hole[:cy0, :265] = 255
    full_hole[cy1:, :265] = 255
    full_hole[cy1:, 1335:] = 255
    full_hole[cy1:, 240:660] = 255

    clean_full_bedroom = _inpaint_masked_plate_lama(
        src_rgb=full_img,
        binary_mask_u8=full_hole,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.bg_inpaint_dilation_px,
        blur_sigma=spec.bg_inpaint_blur_sigma,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )
    clean_full_np = np.asarray(clean_full_bedroom, dtype=np.uint8)
    bg_inpainted_rgb = Image.fromarray(
        _mirror_reflect_rect_to_canvas(clean_full_np, (ew, eh), dest_full_xy),
        mode="RGB",
    )

    master_2700_rgb = Image.fromarray(
        _mirror_reflect_rect_to_canvas(full_np, (ew, eh), dest_full_xy),
        mode="RGB",
    )
    raw_intact_depth_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(master_2700_rgb) * 255.0), 0, 255
    ).astype(np.uint8)
    raw_full_depth_u8 = raw_intact_depth_u8[
        dest_full_xy[1] : dest_full_xy[1] + fh,
        dest_full_xy[0] : dest_full_xy[0] + fw,
    ]

    inpainted_room_depth_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(bg_inpainted_rgb), dtype=np.uint8
    )
    bg_depth_healed_u8 = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(mid_np, fg_np),
        cfg=spec.room_depth_clean,
    )
    bg_depth_clean_u8 = _mirror_reflect_rect_to_canvas(
        bg_depth_healed_u8[dest_full_xy[1] : dest_full_xy[1] + fh, dest_full_xy[0] : dest_full_xy[0] + fw],
        (ew, eh),
        dest_full_xy,
    )

    # 5. Solve mid depth (1500x1000) and mirrored fg depth (1500x1125 -> 2700x1800)
    mid_prot = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_depth_u8,
        cutout_rgba=mid_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.mid_depth_solve.alpha_threshold_u8,
    )
    mid_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=mid_np,
        cfg=spec.mid_depth_solve,
        neural_protrusion_norm=mid_prot,
    )

    # Solve fg on centered full_img window (using exact unpinned room_z in [338:1463, 600:2100]), then mirror
    room_z_exp_fg = compute_world_z_map(
        bg_depth_clean_u8,
        spec.fg_depth_solve.bg_z,
        spec.fg_depth_solve.room_depth_amplitude,
        spec.fg_depth_solve.room_pin_margin_uv,
    )
    full_host_z = room_z_exp_fg[dest_full_xy[1] : dest_full_xy[1] + fh, dest_full_xy[0] : dest_full_xy[0] + fw]
    # Center-embed full_img (1500x1125) into a temporary (2700x1801) canvas with mirrored surround
    tmp_exp_h = fh + 2 * dest_full_xy[1]
    full_host_u8 = np.clip(
        np.rint(
            ((full_host_z - spec.fg_depth_solve.bg_z) / spec.fg_depth_solve.room_depth_amplitude) * 255.0
        ),
        0,
        255,
    ).astype(np.uint8)
    tmp_room_u8 = _mirror_reflect_rect_to_canvas(full_host_u8, (ew, tmp_exp_h), dest_full_xy)
    tmp_raw_u8 = _mirror_reflect_rect_to_canvas(raw_full_depth_u8, (ew, tmp_exp_h), dest_full_xy)
    fg_full_prot = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=tmp_raw_u8,
        cutout_rgba=fg_full_rgba_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=fg_full_solve_cfg.alpha_threshold_u8,
    )
    fg_full_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=tmp_room_u8,
        cutout_rgba=fg_full_rgba_np,
        cfg=ContactAnchoredLayerDepthConfig(
            bg_z=fg_full_solve_cfg.bg_z,
            room_depth_amplitude=fg_full_solve_cfg.room_depth_amplitude,
            room_pin_margin_uv=0.0,
            layer_z=fg_full_solve_cfg.layer_z,
            layer_depth_amplitude=fg_full_solve_cfg.layer_depth_amplitude,
            layer_pin_margin_uv=0.0,
            contact_boxes=fg_full_solve_cfg.contact_boxes,
            neural_relief_amplitude=fg_full_solve_cfg.neural_relief_amplitude,
            contact_relief_taper_px=fg_full_solve_cfg.contact_relief_taper_px,
            enforce_front_of_host=fg_full_solve_cfg.enforce_front_of_host,
            alpha_threshold_u8=fg_full_solve_cfg.alpha_threshold_u8,
            min_component_area_px=fg_full_solve_cfg.min_component_area_px,
        ),
        neural_protrusion_norm=fg_full_prot,
    )
    fg_mirrored_d_norm = (
        _mirror_reflect_rect_to_canvas(fg_full_solved.depth_u8, (ew, eh), dest_full_xy).astype(np.float32)
        / 255.0
    )
    fg_opaque_2700 = fg_np[:, :, 3] >= spec.fg_depth_solve.alpha_threshold_u8
    fg_depth_2700_u8 = np.rint(
        extrapolate_depth_outward_edt(fg_mirrored_d_norm, fg_opaque_2700) * 255.0
    ).astype(np.uint8)
    fg_contact_2700 = extract_contact_anchor_mask(fg_opaque_2700, spec.fg_depth_solve)

    # 6. Build flush wall-perspective vitrine + authentic cast shadow (#53 & #54)
    vitrine_src_1500_img = (
        Image.open(project_root / spec.vitrine_source_relpath)
        .convert("RGB")
        .resize((iw, ih), Image.Resampling.LANCZOS)
    )
    (
        vitrine_canvas,
        vit_depth_u8,
        vit_contact_mask,
        _vit_world_z,
        portal_box,
        portal_world_z,
    ) = _build_flush_wall_perspective_vitrine(
        vitrine_src_1500_img=vitrine_src_1500_img,
        clean_bedroom_inner_np=clean_full_np[cy0:cy1, :, :],
        bg_depth_clean_u8=bg_depth_clean_u8,
        cfg=cfg,
        depth_estimator=depth_estimator,
        sam_extractor=sam_extractor,
    )
    _assert_border_alpha_contract(vitrine_canvas, f"{spec.scene_id}.mounted_wall_vitrine")
    vitrine_np = np.asarray(vitrine_canvas, dtype=np.uint8)

    mid_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.mid",
        bg_depth_clean_u8,
        mid_np,
        mid_solved.depth_u8,
        mid_solved.contact_mask,
        spec.mid_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )
    fg_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.fg",
        bg_depth_clean_u8,
        fg_np,
        fg_depth_2700_u8,
        fg_contact_2700,
        spec.fg_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )
    vit_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.mounted_wall_vitrine",
        bg_depth_clean_u8,
        vitrine_np,
        vit_depth_u8,
        vit_contact_mask,
        spec.vitrine_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )

    pfx = spec.asset_prefix
    bg_inpainted_rgb.save(scenes_dir / f"{pfx}_bg_inpainted.jpg", quality=spec.bg_jpeg_quality, optimize=True)
    Image.fromarray(bg_depth_clean_u8, mode="L").save(scenes_dir / f"{pfx}_depth.png")
    mid_canvas.save(scenes_dir / f"{pfx}_mid.png")
    Image.fromarray(mid_solved.depth_u8, mode="L").save(scenes_dir / f"{pfx}_mid_depth.png")
    fg_canvas.save(scenes_dir / f"{pfx}_fg.png")
    Image.fromarray(fg_depth_2700_u8, mode="L").save(scenes_dir / f"{pfx}_fg_depth.png")
    vitrine_canvas.save(scenes_dir / f"{pfx}_vitrine.png")
    Image.fromarray(vit_depth_u8, mode="L").save(scenes_dir / f"{pfx}_vitrine_depth.png")

    vit_tight_box = _tight_alpha_bbox(vitrine_np, 20)

    extra_layers = [
        {
            "id": "mounted_wall_vitrine",
            "labelHe": "תיבת הוויטרינה התלויה צמוד לקיר החדר (מסובבת 130° עם צל מקורי)",
            "texture": f"assets/scenes/{pfx}_vitrine.png",
            "depthTexture": f"assets/scenes/{pfx}_vitrine_depth.png",
            "zPos": spec.vitrine_depth_solve.layer_z,
            "depthAmplitude": spec.vitrine_depth_solve.layer_depth_amplitude,
            "pinMarginUv": 0.0,
            "segmentsX": 224,
            "segmentsY": 150,
            "renderOrder": 5.5,
            "expanded": False,
            "baseOpacity": 1.0,
            "excursionBoost": 0.0,
            "depthTest": False,
            "depthWrite": False,
        }
    ]
    portal_shelves = [
        {
            "id": spec.portal_id,
            "targetSceneId": spec.target_scene_id,
            "portalMode": "crossfade",
            "labelHe": spec.portal_label_he,
            "shortBadgeHe": spec.portal_short_badge_he,
            "showMicroBadge": False,
            "innerBoxPx": portal_box,
            "worldZ": portal_world_z,
            "recessDepth": 0.0,
            "embedDiorama": False,
            "showFrame": False,
            "restFrameOpacity": 0.0,
            "frameRenderOrder": 5.8,
            "rotationDeg": spec.vitrine_rotation_deg,
            "flightRollDeg": spec.vitrine_flight_roll_deg,
        }
    ]

    rig = spec.camera_rig
    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "bgExpandedWidth": ew,
        "bgExpandedHeight": eh,
        "bgTexture": f"assets/scenes/{pfx}_bg_inpainted.jpg",
        "depthTexture": f"assets/scenes/{pfx}_depth.png",
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
        "fgExpanded": True,
        "fgBleedEdges": None,
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
        "portraitInitialYawDeg": rig.portrait_initial_yaw_deg,
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
        "midCoverageRatio": round(float((mid_np[:, :, 3] >= 20).mean()), 4),
        "fgCoverageRatio": round(float((fg_np[:, :, 3] >= 20).mean()), 4),
        "mountedVitrineTightBoxPx": vit_tight_box,
        "extraLayers": extra_layers,
        "portalShelves": portal_shelves,
        "depthTopologyVerification": {
            "mid": mid_verify,
            "fg": fg_verify,
            "mounted_wall_vitrine": vit_verify,
        },
    }


# ============================================================================
# Scene 22 Builder: `paris_cite_white_table` (`scene22_paris_table_*`)
# ============================================================================


def _build_tabletop_1958_photo_rgba(
    photo_src_rgb: Image.Image,
    inner_wh: tuple[int, int],
    spec: ParisCiteWhiteTableSceneSpec,
) -> Image.Image:
    iw, ih = inner_wh
    cx0, cy0, cx1, cy1 = spec.photo_src_crop_box
    card_np = np.asarray(photo_src_rgb.crop((cx0, cy0, cx1, cy1)), dtype=np.uint8)
    ch, cw, _ = card_np.shape
    src_q = np.array(
        [[0.0, 0.0], [float(cw - 1), 0.0], [float(cw - 1), float(ch - 1)], [0.0, float(ch - 1)]],
        dtype=np.float32,
    )
    dst_q = np.asarray(spec.photo_dst_quad_inner_xy, dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(src_q, dst_q)

    warped_rgb = cv2.warpPerspective(
        card_np, h_mat, (iw, ih), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
    )
    warped_a = cv2.warpPerspective(
        np.full((ch, cw), 255, dtype=np.uint8),
        h_mat,
        (iw, ih),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    card_aa = np.where(warped_a >= 20, warped_a, 0).astype(np.uint8)

    solid = (card_aa >= 96).astype(np.float32)
    shadow_seed = ndi.shift(solid, shift=(1.6, 1.8), order=1)
    shadow_blur = ndi.gaussian_filter(shadow_seed, sigma=spec.photo_contact_shadow_sigma_px)
    if float(shadow_blur.max()) > 1e-5:
        shadow_blur = shadow_blur / float(shadow_blur.max())
    shadow_a = np.clip(
        np.rint(shadow_blur * float(spec.photo_contact_shadow_max_alpha_u8)),
        0,
        spec.photo_contact_shadow_max_alpha_u8,
    ).astype(np.uint8)

    out_rgba = np.zeros((ih, iw, 4), dtype=np.uint8)
    sh_only = (card_aa == 0) & (shadow_a > 0)
    out_rgba[sh_only, :3] = 35
    out_rgba[sh_only, 3] = shadow_a[sh_only]
    fg_px = card_aa > 0
    out_rgba[fg_px, :3] = warped_rgb[fg_px]
    out_rgba[fg_px, 3] = card_aa[fg_px]
    return Image.fromarray(out_rgba, mode="RGBA")


def build_paris_cite_white_table_scene(
    project_root: Path,
    cfg: VitrineBedroomParisBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    spec = cfg.paris_cite_white_table
    iw, ih = cfg.inner_dimensions.width, cfg.inner_dimensions.height
    ew, eh = cfg.expanded_dimensions.width, cfg.expanded_dimensions.height
    ox, oy = cfg.inner_offset_xy
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Scene 22: {spec.scene_id}] Building Paris Cité studio white table + 1958 photo (mirror edges)..."
    )
    sw, sh = spec.scaled_widescreen_wh
    pad_tb = spec.pad_top_bottom_px
    dest_w153_inner_xy = (0, pad_tb)
    dest_w153_exp_xy = (ox, oy + pad_tb)

    w153 = np.asarray(
        Image.open(project_root / spec.studio_source_relpath)
        .convert("RGB")
        .resize((sw, sh), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    studio_1500_np = _mirror_reflect_rect_to_canvas(w153, (iw, ih), dest_w153_inner_xy)
    studio_1500_img = Image.fromarray(studio_1500_np, mode="RGB")

    # 1. Segment wooden folding chair (`fg`) cleanly separated from white table bars & dark floor
    chair_sam = _segment_prompt_union_u8(
        rgb_image=studio_1500_img,
        prompts=spec.fg_chair_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=80,
        max_pinhole_fill_px=0,
    )
    rgb_f = studio_1500_np.astype(np.float32)
    r_minus_b = rgb_f[:, :, 0] - rgb_f[:, :, 2]
    luma = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    not_white_bar = ~(
        (luma > spec.chair_white_bar_luma_min) & (r_minus_b < spec.chair_white_bar_rmb_max)
    )
    not_grey_bar = ~(
        (luma > spec.chair_grey_bar_luma_min) & (r_minus_b < spec.chair_grey_bar_rmb_max)
    )
    not_dark_floor = ~(
        (luma < spec.chair_dark_floor_luma_max) & (r_minus_b < spec.chair_dark_floor_rmb_max)
    )
    chair_u8 = np.where(
        (chair_sam >= 128) & not_white_bar & not_grey_bar & not_dark_floor,
        chair_sam,
        np.uint8(0),
    )
    cy_min, cy_max = spec.chair_y_bounds
    chair_u8[:cy_min, :] = 0
    chair_u8[cy_max:, :] = 0

    chair_rgb = decontaminate_matte_edge_rgb(studio_1500_np, chair_u8)
    fg_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            Image.fromarray(np.dstack([chair_rgb, chair_u8]), mode="RGBA"),
            min_area_px=spec.fg_depth_solve.min_component_area_px,
            alpha_thresh_u8=spec.fg_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.fg_depth_solve,
    )
    _assert_border_alpha_contract(fg_canvas, f"{spec.scene_id}.fg")
    fg_np = np.asarray(fg_canvas, dtype=np.uint8)

    # 2. Inpaint chair out of authentic 1500x844 w153 plate so tabletop & front legs are continuous
    k_ch = 2 * spec.chair_inpaint_dilate_px + 1
    chair_core_mask = cv2.dilate(
        (fg_np[pad_tb : pad_tb + sh, :, 3] >= 16).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_ch, k_ch)),
    )
    no_chair_w153_img = lama_inpainter.inpaint_background(
        Image.fromarray(w153, mode="RGB"), chair_core_mask
    )
    no_chair_w153_np = np.asarray(no_chair_w153_img, dtype=np.uint8)
    no_chair_1500_np = _mirror_reflect_rect_to_canvas(no_chair_w153_np, (iw, ih), dest_w153_inner_xy)
    no_chair_1500_img = Image.fromarray(no_chair_1500_np, mode="RGB")

    # 3. Segment cabinet + flowerpots + continuous white tabletop + front-left metal T-leg (`mid`)
    mid_u8 = _segment_prompt_union_u8(
        rgb_image=no_chair_1500_img,
        prompts=spec.mid_table_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=120,
        max_pinhole_fill_px=60,
    )
    mid_core_u8 = mid_u8[pad_tb : pad_tb + sh, :].copy()
    mid_core_u8[:, 0] = 0
    mid_core_u8[:, -1] = 0
    mid_core_rgb = decontaminate_matte_edge_rgb(no_chair_w153_np, mid_core_u8)
    mid_mirrored_rgba = _mirror_reflect_rect_to_canvas(
        np.dstack([mid_core_rgb, mid_core_u8]), (iw, ih), dest_w153_inner_xy
    )
    mid_mirrored_rgba[0, :, :] = 0
    mid_mirrored_rgba[-1, :, :] = 0
    mid_mirrored_rgba[:, 0, :] = 0
    mid_mirrored_rgba[:, -1, :] = 0
    mid_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            Image.fromarray(mid_mirrored_rgba, mode="RGBA"),
            min_area_px=spec.mid_depth_solve.min_component_area_px,
            alpha_thresh_u8=spec.mid_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.mid_depth_solve,
    )
    _assert_border_alpha_contract(mid_canvas, f"{spec.scene_id}.mid")
    mid_np = np.asarray(mid_canvas, dtype=np.uint8)

    # 4. Build perspective-placed 1958 photo on the white tabletop
    photo_src_rgb = Image.open(project_root / spec.photo_source_relpath).convert("RGB")
    photo_canvas = _build_tabletop_1958_photo_rgba(photo_src_rgb, (iw, ih), spec)
    _assert_border_alpha_contract(photo_canvas, f"{spec.scene_id}.placed_1958_photo")
    photo_np = np.asarray(photo_canvas, dtype=np.uint8)

    # 5. Inpaint mid | fg out of 1500x844 w153 core and mirror across all 4 edges onto 2700x1800 (#48)
    bg_hole_core = np.where(
        (mid_np[pad_tb : pad_tb + sh, :, 3] >= 16) | (fg_np[pad_tb : pad_tb + sh, :, 3] >= 16),
        255,
        0,
    ).astype(np.uint8)
    clean_studio_w153 = _inpaint_masked_plate_lama(
        src_rgb=no_chair_w153_img,
        binary_mask_u8=bg_hole_core,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.bg_inpaint_dilation_px,
        blur_sigma=spec.bg_inpaint_blur_sigma,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )
    clean_w153_np = np.asarray(clean_studio_w153, dtype=np.uint8)
    bg_inpainted_rgb = Image.fromarray(
        _mirror_reflect_rect_to_canvas(clean_w153_np, (ew, eh), dest_w153_exp_xy),
        mode="RGB",
    )

    master_2700_rgb = Image.fromarray(
        _mirror_reflect_rect_to_canvas(w153, (ew, eh), dest_w153_exp_xy),
        mode="RGB",
    )
    raw_intact_depth_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(master_2700_rgb) * 255.0), 0, 255
    ).astype(np.uint8)

    inpainted_room_depth_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(bg_inpainted_rgb), dtype=np.uint8
    )
    bg_depth_healed_u8 = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(mid_np, fg_np),
        cfg=spec.room_depth_clean,
    )
    bg_depth_clean_u8 = _mirror_reflect_rect_to_canvas(
        bg_depth_healed_u8[
            dest_w153_exp_xy[1] : dest_w153_exp_xy[1] + sh,
            dest_w153_exp_xy[0] : dest_w153_exp_xy[0] + sw,
        ],
        (ew, eh),
        dest_w153_exp_xy,
    )

    mid_prot = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_depth_u8,
        cutout_rgba=mid_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.mid_depth_solve.alpha_threshold_u8,
    )
    mid_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=mid_np,
        cfg=spec.mid_depth_solve,
        neural_protrusion_norm=mid_prot,
    )

    fg_prot = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_depth_u8,
        cutout_rgba=fg_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.fg_depth_solve.alpha_threshold_u8,
    )
    fg_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=fg_np,
        cfg=spec.fg_depth_solve,
        neural_protrusion_norm=fg_prot,
    )

    # Lock `placed_1958_photo` $C^0$ directly onto the 3D `mid` white tabletop surface + outward EDT!
    photo_opaque = photo_np[:, :, 3] >= 20
    photo_target_z = mid_solved.world_z_map + spec.photo_tabletop_clearance_world
    photo_d_norm = np.clip(
        (photo_target_z - spec.mid_depth_solve.layer_z) / spec.mid_depth_solve.layer_depth_amplitude,
        0.0,
        1.0,
    ).astype(np.float32)
    photo_depth_u8 = np.rint(extrapolate_depth_outward_edt(photo_d_norm, photo_opaque) * 255.0).astype(
        np.uint8
    )
    photo_rim_jump = _compute_edt_rim_p99_jump_world(
        photo_np,
        photo_depth_u8,
        spec.mid_depth_solve.layer_depth_amplitude,
        20,
        cfg.edt_rim_probe_px,
    )
    if photo_rim_jump > cfg.max_edt_rim_jump_world:
        raise RuntimeError(f"{spec.scene_id}.placed_1958_photo rim jump {photo_rim_jump:.3f} > max")

    photo_obj_z = compute_world_z_map(
        photo_depth_u8, spec.mid_depth_solve.layer_z, spec.mid_depth_solve.layer_depth_amplitude, 0.0
    )
    tabletop_dz = np.abs(photo_obj_z[photo_opaque] - mid_solved.world_z_map[photo_opaque])
    if float(np.percentile(tabletop_dz, 95.0)) > 1.5:
        raise RuntimeError(f"{spec.scene_id}.placed_1958_photo detached from tabletop!")

    mid_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.mid",
        bg_depth_clean_u8,
        mid_np,
        mid_solved.depth_u8,
        mid_solved.contact_mask,
        spec.mid_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )
    fg_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.fg",
        bg_depth_clean_u8,
        fg_np,
        fg_solved.depth_u8,
        fg_solved.contact_mask,
        spec.fg_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )

    pfx = spec.asset_prefix
    bg_inpainted_rgb.save(scenes_dir / f"{pfx}_bg_inpainted.jpg", quality=spec.bg_jpeg_quality, optimize=True)
    Image.fromarray(bg_depth_clean_u8, mode="L").save(scenes_dir / f"{pfx}_depth.png")
    mid_canvas.save(scenes_dir / f"{pfx}_mid.png")
    Image.fromarray(mid_solved.depth_u8, mode="L").save(scenes_dir / f"{pfx}_mid_depth.png")
    fg_canvas.save(scenes_dir / f"{pfx}_fg.png")
    Image.fromarray(fg_solved.depth_u8, mode="L").save(scenes_dir / f"{pfx}_fg_depth.png")
    photo_canvas.save(scenes_dir / f"{pfx}_photo.png")
    Image.fromarray(photo_depth_u8, mode="L").save(scenes_dir / f"{pfx}_photo_depth.png")

    photo_box = [float(v) for v in _tight_alpha_bbox(photo_np, 20)]
    portal_world_z = round(float(np.median(photo_obj_z[photo_opaque])), 2)

    extra_layers = [
        {
            "id": "placed_1958_photo",
            "labelHe": "תצלום תמי הפעוטה מ-1958 מונח על השולחן הלבן — לחצו להתקרבות",
            "texture": f"assets/scenes/{pfx}_photo.png",
            "depthTexture": f"assets/scenes/{pfx}_photo_depth.png",
            "zPos": spec.mid_depth_solve.layer_z,
            "depthAmplitude": spec.mid_depth_solve.layer_depth_amplitude,
            "pinMarginUv": 0.0,
            "segmentsX": 224,
            "segmentsY": 150,
            "renderOrder": 6.5,
            "expanded": False,
            "baseOpacity": 1.0,
            "excursionBoost": 0.0,
            "depthTest": False,
            "depthWrite": False,
        }
    ]
    portal_shelves = [
        {
            "id": spec.portal_id,
            "targetSceneId": spec.target_scene_id,
            "portalMode": "crossfade",
            "labelHe": spec.portal_label_he,
            "shortBadgeHe": spec.portal_short_badge_he,
            "showMicroBadge": False,
            "innerBoxPx": photo_box,
            "worldZ": portal_world_z,
            "recessDepth": 0.0,
            "embedDiorama": False,
            "showFrame": False,
            "restFrameOpacity": 0.0,
            "frameRenderOrder": 6.8,
        }
    ]

    rig = spec.camera_rig
    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "bgExpandedWidth": ew,
        "bgExpandedHeight": eh,
        "bgTexture": f"assets/scenes/{pfx}_bg_inpainted.jpg",
        "depthTexture": f"assets/scenes/{pfx}_depth.png",
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
        "portraitInitialYawDeg": rig.portrait_initial_yaw_deg,
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
        "midCoverageRatio": round(float((mid_np[:, :, 3] >= 20).mean()), 4),
        "fgCoverageRatio": round(float((fg_np[:, :, 3] >= 20).mean()), 4),
        "extraLayers": extra_layers,
        "portalShelves": portal_shelves,
        "depthTopologyVerification": {
            "mid": mid_verify,
            "fg": fg_verify,
            "placed_1958_photo": {
                "tabletopP95DzWorld": round(float(np.percentile(tabletop_dz, 95.0)), 4),
                "liveGateP99RimJumpWorld": round(photo_rim_jump, 4),
                "all_passed": True,
            },
        },
    }


# ============================================================================
# Scene 23 Builder: `toddler_tami_1958` (`scene23_tami_1958_*`)
# ============================================================================


def build_toddler_tami_1958_scene(
    project_root: Path,
    cfg: VitrineBedroomParisBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    spec = cfg.toddler_tami_1958
    iw, ih = cfg.inner_dimensions.width, cfg.inner_dimensions.height
    ew, eh = cfg.expanded_dimensions.width, cfg.expanded_dimensions.height
    ox, oy = cfg.inner_offset_xy
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Scene 23: {spec.scene_id}] Building zoomed-in 1958 toddler Tami portrait (mirror edges)...")
    raw_np = np.asarray(Image.open(project_root / spec.source_relpath).convert("RGB"), dtype=np.uint8)
    src_q = np.asarray(spec.src_inner_quad_xy, dtype=np.float32)
    dst_q = np.asarray(spec.dst_inner_quad_xy, dtype=np.float32)
    rect_inner = cv2.warpPerspective(
        raw_np,
        cv2.getPerspectiveTransform(src_q, dst_q),
        (iw, ih),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    ml, mt, mr, mb = spec.authentic_crop_margins_ltrb_px
    auth_core_np = rect_inner[mt : ih - mb, ml : iw - mr, :].copy()
    ch_c, cw_c, _ = auth_core_np.shape
    dest_core_inner_xy = (ml, mt)
    dest_core_exp_xy = (ox + ml, oy + mt)

    mirrored_inner_np = _mirror_reflect_rect_to_canvas(auth_core_np, (iw, ih), dest_core_inner_xy)
    mirrored_inner_img = Image.fromarray(mirrored_inner_np, mode="RGB")

    tami_sam = _segment_prompt_union_u8(
        rgb_image=mirrored_inner_img,
        prompts=spec.mid_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=400,
        max_pinhole_fill_px=400,
    )
    isnet_a = np.asarray(remove(mirrored_inner_img, session=matte_extractor.session), dtype=np.uint8)[
        :, :, 3
    ]
    k_sup = 2 * spec.sam_support_dilate_px + 1
    sam_sup = (
        cv2.dilate(
            (tami_sam >= 128).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sup, k_sup)),
        )
        > 0
    )
    fused_a = np.maximum(tami_sam, np.where(sam_sup, isnet_a, np.uint8(0)))

    core_a = fused_a[mt : ih - mb, ml : iw - mr].copy()
    m_st = spec.cutout_core_side_top_margin_px
    core_a[:m_st, :] = 0
    core_a[:, :m_st] = 0
    core_a[:, cw_c - m_st :] = 0
    # Retain ONLY the single main connected component of toddler Tami (reaching bottom edge cleanly)
    clean_core_a = _keep_largest_cc_u8(core_a, thresh=spec.mid_depth_solve.alpha_threshold_u8)
    core_rgb = decontaminate_matte_edge_rgb(auth_core_np, clean_core_a)

    mirrored_tami_rgba = _mirror_reflect_rect_to_canvas(
        np.dstack([core_rgb, clean_core_a]), (iw, ih), dest_core_inner_xy
    )
    mirrored_tami_rgba[0, :, :] = 0
    mirrored_tami_rgba[:, 0, :] = 0
    mirrored_tami_rgba[:, -1, :] = 0
    mirrored_tami_rgba[-3:, :, 3] = np.minimum(
        mirrored_tami_rgba[-3:, :, 3],
        np.array([[180], [90], [0]], dtype=np.uint8),
    )

    mid_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            Image.fromarray(mirrored_tami_rgba, mode="RGBA"),
            min_area_px=spec.mid_depth_solve.min_component_area_px,
            alpha_thresh_u8=spec.mid_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.mid_depth_solve,
    )
    _assert_border_alpha_contract(mid_canvas, f"{spec.scene_id}.mid")
    mid_np = np.asarray(mid_canvas, dtype=np.uint8)

    # Build 2700x1800 mirrored Tami backing (`fgExpanded: True`, `fgRenderOrder: 5.8` right behind `mid`)
    # so mirror reflection across y=1000 continues Tami seamlessly with zero cutoff (#48)!
    fg_2700_np = _mirror_reflect_rect_to_canvas(
        np.dstack([core_rgb, clean_core_a]), (ew, eh), dest_core_exp_xy
    )
    fg_2700_np[:2, :, :] = 0
    fg_2700_np[-2:, :, :] = 0
    fg_2700_np[:, :2, :] = 0
    fg_2700_np[:, -2:, :] = 0
    fg_canvas = Image.fromarray(fg_2700_np, mode="RGBA")
    _assert_border_alpha_contract(fg_canvas, f"{spec.scene_id}.fg")
    fg_np = np.asarray(fg_canvas, dtype=np.uint8)

    hole_core_u8 = np.where(clean_core_a >= 16, 255, 0).astype(np.uint8)
    clean_core_rgb = _inpaint_masked_plate_lama(
        src_rgb=Image.fromarray(auth_core_np, mode="RGB"),
        binary_mask_u8=hole_core_u8,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.bg_inpaint_dilation_px,
        blur_sigma=spec.bg_inpaint_blur_sigma,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )
    clean_core_np = np.asarray(clean_core_rgb, dtype=np.uint8)
    bg_inpainted_rgb = Image.fromarray(
        _mirror_reflect_rect_to_canvas(clean_core_np, (ew, eh), dest_core_exp_xy),
        mode="RGB",
    )

    master_2700_rgb = Image.fromarray(
        _mirror_reflect_rect_to_canvas(auth_core_np, (ew, eh), dest_core_exp_xy),
        mode="RGB",
    )
    raw_intact_depth_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(master_2700_rgb) * 255.0),
        0,
        255,
    ).astype(np.uint8)

    inpainted_room_depth_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(bg_inpainted_rgb), dtype=np.uint8
    )
    bg_depth_healed_u8 = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(fg_np,),
        cfg=spec.room_depth_clean,
    )
    bg_depth_clean_u8 = _mirror_reflect_rect_to_canvas(
        bg_depth_healed_u8[
            dest_core_exp_xy[1] : dest_core_exp_xy[1] + ch_c,
            dest_core_exp_xy[0] : dest_core_exp_xy[0] + cw_c,
        ],
        (ew, eh),
        dest_core_exp_xy,
    )

    mid_prot = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_depth_u8,
        cutout_rgba=mid_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.mid_depth_solve.alpha_threshold_u8,
    )
    mid_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=mid_np,
        cfg=spec.mid_depth_solve,
        neural_protrusion_norm=mid_prot,
    )
    mid_core_d_u8 = mid_solved.depth_u8[mt : ih - mb, ml : iw - mr]
    fg_mirrored_d_norm = (
        _mirror_reflect_rect_to_canvas(mid_core_d_u8, (ew, eh), dest_core_exp_xy).astype(np.float32)
        / 255.0
    )
    fg_opaque_2700 = fg_np[:, :, 3] >= spec.mid_depth_solve.alpha_threshold_u8
    fg_depth_2700_u8 = np.rint(
        extrapolate_depth_outward_edt(fg_mirrored_d_norm, fg_opaque_2700) * 255.0
    ).astype(np.uint8)
    fg_rim_jump = _compute_edt_rim_p99_jump_world(
        fg_np,
        fg_depth_2700_u8,
        spec.mid_depth_solve.layer_depth_amplitude,
        20,
        cfg.edt_rim_probe_px,
    )
    if fg_rim_jump > cfg.max_edt_rim_jump_world:
        raise RuntimeError(f"Scene 23 fg mirror EDT rim jump {fg_rim_jump:.3f} exceeds max")

    mid_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.mid",
        bg_depth_clean_u8,
        mid_np,
        mid_solved.depth_u8,
        mid_solved.contact_mask,
        spec.mid_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )

    pfx = spec.asset_prefix
    bg_inpainted_rgb.save(scenes_dir / f"{pfx}_bg_inpainted.jpg", quality=spec.bg_jpeg_quality, optimize=True)
    Image.fromarray(bg_depth_clean_u8, mode="L").save(scenes_dir / f"{pfx}_depth.png")
    mid_canvas.save(scenes_dir / f"{pfx}_mid.png")
    Image.fromarray(mid_solved.depth_u8, mode="L").save(scenes_dir / f"{pfx}_mid_depth.png")
    fg_canvas.save(scenes_dir / f"{pfx}_fg.png")
    Image.fromarray(fg_depth_2700_u8, mode="L").save(scenes_dir / f"{pfx}_fg_depth.png")

    rig = spec.camera_rig
    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "bgExpandedWidth": ew,
        "bgExpandedHeight": eh,
        "bgTexture": f"assets/scenes/{pfx}_bg_inpainted.jpg",
        "depthTexture": f"assets/scenes/{pfx}_depth.png",
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
        "fgDepthAmplitude": spec.mid_depth_solve.layer_depth_amplitude,
        "fgPinMarginUv": 0.0,
        "fgSegmentsX": 224,
        "fgSegmentsY": 150,
        "fgDepthTest": False,
        "fgDepthWrite": False,
        "fgExpanded": True,
        "boreSightAtRest": True,
        "bgZ": spec.mid_depth_solve.bg_z,
        "midZ": spec.mid_depth_solve.layer_z,
        "fgZ": spec.mid_depth_solve.layer_z,
        "minInnerScale": rig.min_inner_scale,
        "bgSegmentsX": 224,
        "bgSegmentsY": 150,
        "roomDepthAmplitude": spec.mid_depth_solve.room_depth_amplitude,
        "initialCameraY": rig.initial_camera_y,
        "initialCameraZ": rig.initial_camera_z,
        "portraitInitialYawDeg": rig.portrait_initial_yaw_deg,
        "minCameraZ": rig.min_camera_z,
        "maxCameraZ": rig.max_camera_z,
        "maxYawDeg": rig.max_yaw_deg,
        "maxPitchDeg": rig.max_pitch_deg,
        "maxStrafeX": rig.max_strafe_x,
        "maxStrafeY": rig.max_strafe_y,
        "dollySpeedMult": rig.dolly_speed_mult,
        "midRenderOrder": 6,
        "fgRenderOrder": 5.8,
        "particleColor": rig.particle_color_hex,
        "particleCount": rig.particle_count,
        "hasReflection": False,
        "midCoverageRatio": round(float((mid_np[:, :, 3] >= 20).mean()), 4),
        "fgCoverageRatio": round(float((fg_np[:, :, 3] >= 20).mean()), 4),
        "depthTopologyVerification": {
            "mid": mid_verify,
            "fg": {"liveGateP99RimJumpWorld": round(fg_rim_jump, 4), "all_passed": True},
        },
    }


# ============================================================================
# Canonical SSOT Bundle Configuration
# ============================================================================


def _map21_pt(pt: tuple[float, float]) -> tuple[float, float]:
    scale = 1500.0 / 1334.0
    return (
        round(float(np.clip((pt[0] - 83.0) * scale, 1.0, 1499.0)), 2),
        round(float(np.clip(pt[1] * scale - 62.0, 1.0, 999.0)), 2),
    )


def _map21_full_pt(pt: tuple[float, float]) -> tuple[float, float]:
    scale = 1500.0 / 1334.0
    return (
        round(float(np.clip((pt[0] - 83.0) * scale, 1.0, 1499.0)), 2),
        round(float(np.clip(pt[1] * scale, 1.0, 1124.0)), 2),
    )


def _map21_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    scale = 1500.0 / 1334.0
    x0 = int(np.clip(round((box[0] - 83.0) * scale), 0, 1499))
    y0 = int(np.clip(round(box[1] * scale - 62.0), 0, 999))
    x1 = int(np.clip(round((box[2] - 83.0) * scale), x0 + 2, 1500))
    y1 = int(np.clip(round(box[3] * scale - 62.0), y0 + 2, 1000))
    return (x0, y0, x1, y1)


def default_vitrine_bedroom_paris_bundle_config() -> VitrineBedroomParisBundleConfig:
    room_pin = 0.045
    return VitrineBedroomParisBundleConfig(
        inner_dimensions=CanvasDimensionsConfig(width=1500, height=1000),
        expanded_dimensions=CanvasDimensionsConfig(width=2700, height=1800),
        inner_offset_xy=(600, 400),
        scenes_output_relpath="prototype/assets/scenes",
        spec_output_abspath=(
            "/Users/talsheaffer/.gemini/antigravity/brain/"
            "235713ae-3cfb-48f0-94e4-2012c0824fd5/scratch/sub2_vitrine_spec.json"
        ),
        depth_model=DepthModelConfig(
            model_id="depth-anything/Depth-Anything-V2-Small-hf",
            background_gaussian_sigma=2.2,
            depth_discontinuity_percentile_fg=82.0,
            depth_discontinuity_percentile_mid=62.0,
        ),
        segmentation=SegmentationConfig(
            rembg_model_name="isnet-general-use",
            sam_model_id="facebook/sam-vit-base",
            alpha_foreground_threshold=92,
            alpha_full_opaque_threshold=195,
            min_component_area_px=80,
            sam_morph_close_px=3,
            edge_feather_sigma=0.65,
        ),
        scene_inpainting=InpaintingConfig(
            dilation_radius_px=22,
            mask_blur_sigma=3.0,
            jpeg_quality=94,
        ),
        lama_max_working_side_px=1600,
        lama_roi_context_margin_px=160,
        edt_rim_probe_px=6,
        max_edt_rim_jump_world=2.0,
        topology_verification=LayerTopologyVerificationConfig(
            alpha_threshold_u8=20,
            rim_erosion_px=2,
            rim_exterior_probe_px=6,
            max_allowed_contact_dz_world=2.4,
            max_allowed_p95_contact_dz_world=1.6,
            max_allowed_rim_jump_world=2.0,
            host_monotonicity_tolerance_world=2.0,
        ),
        vitrine_keys_pen=VitrineKeysPenSceneSpec(
            scene_id="vitrine_keys_pen",
            title_he="תיבת הוויטרינה — מדף ורוד, עט וצרור מפתחות (׳בנה ביתך׳, 2026)",
            short_label_he="ויטרינה, מפתחות ועט",
            asset_prefix="scene20_vitrine",
            source_relpath="media/museum_press/rgma_bene_beitcha_vitrine_keys_pen_2026_e5e92bc1.jpg",
            pen_shadow_heal_box_inner=(620, 248, 835, 535),
            plaster_wall_luma_exclude=155.0,
            fg_sam_prompts=(
                _pg(
                    (308, 72, 396, 946),
                    [(348, 120), (350, 320), (350, 520), (350, 720), (348, 900)],
                    [(292, 300), (292, 700), (412, 300), (412, 750)],
                ),
                _pg(
                    (1102, 72, 1190, 946),
                    [(1146, 120), (1146, 320), (1146, 520), (1146, 720), (1146, 900)],
                    [(1088, 300), (1088, 700), (1205, 300), (1205, 700)],
                ),
                _pg(
                    (380, 76, 1116, 160),
                    [(450, 116), (620, 116), (780, 116), (950, 116), (1070, 116)],
                    [(500, 62), (800, 62), (500, 178), (850, 178)],
                ),
                _pg(
                    (380, 860, 1116, 944),
                    [(450, 902), (620, 902), (780, 902), (950, 902), (1070, 902)],
                    [(500, 958), (800, 958), (500, 845), (850, 845)],
                ),
                _pg(
                    (1024, 464, 1103, 544),
                    [(1064, 504), (1050, 490), (1080, 518), (1050, 518), (1080, 490)],
                    [(1015, 504), (1064, 455), (1064, 552)],
                ),
            ),
            mid_sam_prompts=(
                _pg(
                    (566, 236, 644, 514),
                    [(616, 260), (610, 320), (602, 395), (593, 470)],
                    [(555, 350), (650, 350), (680, 330)],
                ),
                _pg(
                    (522, 504, 636, 592),
                    [(555, 545), (595, 545), (575, 520), (575, 575)],
                    [(505, 540), (636, 490)],
                ),
                _pg(
                    (584, 494, 874, 626),
                    [
                        (655, 520),
                        (668, 535),
                        (615, 585),
                        (640, 595),
                        (702, 530),
                        (722, 530),
                        (718, 590),
                        (752, 592),
                        (752, 515),
                        (795, 545),
                        (845, 548),
                    ],
                    [(660, 480), (750, 480), (830, 510), (885, 550)],
                ),
                _pg(
                    (392, 564, 884, 652),
                    [
                        (435, 585),
                        (505, 585),
                        (435, 622),
                        (550, 622),
                        (665, 622),
                        (780, 622),
                        (860, 622),
                        (855, 585),
                    ],
                    [(460, 545), (460, 675), (640, 675), (898, 610)],
                ),
                _pg(
                    (718, 644, 804, 804),
                    [(740, 665), (782, 665), (760, 710), (758, 765), (756, 792)],
                    [(705, 700), (812, 700), (705, 775), (810, 775)],
                ),
            ),
            bg_inpaint_dilation_px=18,
            bg_inpaint_blur_sigma=3.0,
            bg_jpeg_quality=94,
            knob_box_inner=(1020, 460, 1106, 548),
            frame_protrusion_norm=0.78,
            knob_protrusion_norm=1.0,
            room_depth_clean=RoomDepthCleanConfig(
                alpha_threshold_u8=20,
                mask_dilation_px=16,
                post_smooth_sigma_px=3.0,
                interior_blend_band_px=12.0,
            ),
            mid_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=-350.0,
                room_depth_amplitude=96.0,
                room_pin_margin_uv=room_pin,
                layer_z=-350.0,
                layer_depth_amplitude=175.0,
                layer_pin_margin_uv=0.0,
                contact_boxes=(
                    ContactAnchorBox(x0=390, y0=562, x1=428, y1=652, mode="left_band", band_px=10),
                    ContactAnchorBox(x0=716, y0=755, x1=806, y1=806, mode="bottom_band", band_px=10),
                ),
                neural_relief_amplitude=22.0,
                contact_relief_taper_px=10.0,
                enforce_front_of_host=True,
                alpha_threshold_u8=20,
                min_component_area_px=120,
            ),
            fg_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=-350.0,
                room_depth_amplitude=96.0,
                room_pin_margin_uv=room_pin,
                layer_z=-350.0,
                layer_depth_amplitude=195.0,
                layer_pin_margin_uv=0.0,
                contact_boxes=(
                    ContactAnchorBox(x0=906, y0=472, x1=936, y1=1346, mode="left_band", band_px=10),
                    ContactAnchorBox(x0=1760, y0=472, x1=1792, y1=1346, mode="right_band", band_px=10),
                ),
                neural_relief_amplitude=42.0,
                contact_relief_taper_px=10.0,
                enforce_front_of_host=True,
                alpha_threshold_u8=20,
                min_component_area_px=80,
            ),
            camera_rig=SceneCameraRigSpec(
                min_inner_scale=1.94,
                initial_camera_y=-4.0,
                initial_camera_z=525.0,
                min_camera_z=-85.0,
                max_camera_z=685.0,
                max_yaw_deg=22.0,
                max_pitch_deg=15.5,
                max_strafe_x=125.0,
                max_strafe_y=80.0,
                dolly_speed_mult=1.3,
                particle_color_hex="0xDFD5C5",
                particle_count=55,
            ),
        ),
        bar_giora_bedroom=BarGioraBedroomSceneSpec(
            scene_id="bar_giora_bedroom",
            title_he="חדר השינה בבר גיורא (חיפה) — וויטרינת המפתחות והעט תלויה מתחת לציור",
            short_label_he="בר גיורא והוויטרינה",
            asset_prefix="scene21_bar_giora",
            bedroom_source_relpath="media/wordpress_blog/wp_wide_p1050497_51a8f72e.jpg",
            vitrine_source_relpath="media/museum_press/rgma_bene_beitcha_vitrine_keys_pen_2026_e5e92bc1.jpg",
            scaled_full_wh=(1500, 1125),
            crop_y_range=(62, 1062),
            vitrine_wall_center_xy=(555.0, 424.0),
            vitrine_wall_side_px=118.0,
            vitrine_rotation_deg=-130.0,
            vitrine_flight_roll_deg=-130.0,
            wall_camera_focal_px=1167.0,
            wall_vanishing_point_xy=(2104.0, 298.0),
            vitrine_indoor_rgb_scale=(0.88, 0.90, 0.92),
            vitrine_shadow_ray_offset_src_xy=(265.0, 302.0),
            vitrine_shadow_attenuation_floor=0.042,
            vitrine_shadow_gain=1.10,
            vitrine_shadow_max_alpha_norm=0.42,
            vitrine_shadow_blur_sigma_px=3.6,
            vitrine_shadow_handoff_margin_px=32,
            vitrine_shadow_plaster_rgb_scale=(0.22, 0.21, 0.20),
            vitrine_frame_relief_world=6.2,
            vitrine_knob_relief_world=7.5,
            vitrine_cavity_back_relief_world=1.4,
            vitrine_items_min_relief_world=2.8,
            vitrine_items_max_relief_world=4.8,
            vitrine_contact_rim_taper_px=3.5,
            portal_id="portal_bar_giora_vitrine",
            target_scene_id="vitrine_keys_pen",
            portal_label_he="תיבת הוויטרינה מ׳בנה ביתך׳ · לחצו להתקרבות וסיבוב אל תוך הוויטרינה",
            portal_short_badge_he="אל הוויטרינה",
            fg_full_sam_prompts=(
                _pg(
                    (0, 0, 258, 1125),
                    [
                        _map21_full_pt(p)
                        for p in [
                            (150, 120),
                            (210, 250),
                            (160, 500),
                            (240, 650),
                            (180, 880),
                            (275, 950),
                        ]
                    ]
                    + [
                        (25.0, 30.0),
                        (140.0, 35.0),
                        (25.0, 260.0),
                        (25.0, 810.0),
                        (25.0, 1105.0),
                        (150.0, 1105.0),
                        (225.0, 1105.0),
                    ],
                    [_map21_full_pt(p) for p in [(255, 120), (275, 450), (312, 750), (320, 950)]]
                    + [(230.0, 35.0), (290.0, 1100.0)],
                ),
                _pg(
                    (1340, 755, 1500, 1125),
                    [
                        _map21_full_pt(p)
                        for p in [
                            (1390, 725),
                            (1385, 810),
                            (1315, 815),
                            (1355, 895),
                            (1395, 955),
                            (1302, 950),
                        ]
                    ],
                    [_map21_full_pt(p) for p in [(1335, 720), (1272, 820), (1270, 940)]],
                ),
            ),
            mid_sam_prompts=(
                _pg(
                    (215, 646, 740, 1000),
                    [
                        _map21_pt(p)
                        for p in [
                            (340, 720),
                            (480, 680),
                            (620, 710),
                            (540, 860),
                            (420, 940),
                            (640, 880),
                        ]
                    ]
                    + [
                        (245.0, 715.0),
                        (275.0, 780.0),
                        (350.0, 820.0),
                        (365.0, 675.0),
                        (410.0, 656.0),
                        (455.0, 656.0),
                        (390.0, 685.0),
                        (435.0, 685.0),
                        (460.0, 695.0),
                        (410.0, 715.0),
                        (445.0, 715.0),
                    ],
                    [
                        (285.0, 655.0),
                        (335.0, 658.0),
                        (370.0, 650.0),
                        (415.0, 640.0),
                        (455.0, 640.0),
                        _map21_pt((720, 960)),
                    ],
                ),
                _pg(
                    (465, 526, 952, 1000),
                    [
                        _map21_pt(p)
                        for p in [
                            (480, 680),
                            (620, 710),
                            (720, 560),
                            (850, 595),
                            (760, 660),
                            (540, 860),
                            (640, 880),
                            (810, 780),
                            (925, 675),
                        ]
                    ]
                    + [
                        (655.0, 615.0),
                        (740.0, 548.0),
                        (825.0, 558.0),
                        (895.0, 572.0),
                        (924.0, 586.0),
                        (820.0, 585.0),
                        (875.0, 608.0),
                        (915.0, 622.0),
                        (922.0, 640.0),
                    ],
                    [
                        _map21_pt(p)
                        for p in [
                            (560, 540),
                            (660, 510),
                            (850, 920),
                            (720, 960),
                        ]
                    ]
                    + [
                        (480.0, 560.0),
                        (480.0, 620.0),
                        (550.0, 560.0),
                        (735.0, 528.0),
                        (790.0, 529.0),
                        (845.0, 528.0),
                        (900.0, 535.0),
                        (948.0, 565.0),
                        (948.0, 615.0),
                        (948.0, 655.0),
                    ],
                ),
                _pg(
                    _map21_box((874, 628, 1154, 988)),
                    [
                        _map21_pt(p)
                        for p in [
                            (1035, 660),
                            (1085, 700),
                            (920, 745),
                            (980, 750),
                            (1080, 775),
                            (940, 820),
                            (1045, 845),
                            (1028, 915),
                            (1042, 965),
                        ]
                    ],
                    [_map21_pt(p) for p in [(1010, 635), (1135, 680), (1165, 780), (1135, 890), (935, 905)]],
                ),
            ),
            bg_inpaint_dilation_px=22,
            bg_inpaint_blur_sigma=3.2,
            bg_jpeg_quality=94,
            room_depth_clean=RoomDepthCleanConfig(
                alpha_threshold_u8=20,
                mask_dilation_px=16,
                post_smooth_sigma_px=3.0,
                interior_blend_band_px=12.0,
            ),
            mid_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=-350.0,
                room_depth_amplitude=118.0,
                room_pin_margin_uv=room_pin,
                layer_z=-350.0,
                layer_depth_amplitude=175.0,
                layer_pin_margin_uv=0.0,
                contact_boxes=(
                    ContactAnchorBox(x0=235, y0=930, x1=690, y1=998, mode="bottom_band", band_px=12),
                    ContactAnchorBox(x0=650, y0=520, x1=965, y1=680, mode="top_band", band_px=10),
                    ContactAnchorBox(x0=930, y0=860, x1=1145, y1=998, mode="bottom_band", band_px=12),
                ),
                neural_relief_amplitude=20.0,
                contact_relief_taper_px=30.0,
                enforce_front_of_host=True,
                alpha_threshold_u8=20,
                min_component_area_px=5000,
            ),
            fg_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=-350.0,
                room_depth_amplitude=118.0,
                room_pin_margin_uv=room_pin,
                layer_z=-350.0,
                layer_depth_amplitude=195.0,
                layer_pin_margin_uv=0.0,
                contact_boxes=(
                    ContactAnchorBox(x0=720, y0=1318, x1=858, y1=1463, mode="bottom_band", band_px=12),
                    ContactAnchorBox(x0=1940, y0=1328, x1=2100, y1=1463, mode="bottom_band", band_px=12),
                ),
                neural_relief_amplitude=26.0,
                contact_relief_taper_px=34.0,
                enforce_front_of_host=True,
                alpha_threshold_u8=20,
                min_component_area_px=300,
            ),
            vitrine_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=-350.0,
                room_depth_amplitude=118.0,
                room_pin_margin_uv=room_pin,
                layer_z=-350.0,
                layer_depth_amplitude=175.0,
                layer_pin_margin_uv=0.0,
                contact_boxes=(
                    ContactAnchorBox(x0=415, y0=320, x1=655, y1=535, mode="perimeter_rim", band_px=3),
                ),
                neural_relief_amplitude=6.5,
                contact_relief_taper_px=12.0,
                enforce_front_of_host=True,
                alpha_threshold_u8=20,
                min_component_area_px=80,
            ),
            camera_rig=SceneCameraRigSpec(
                min_inner_scale=2.12,
                initial_camera_y=0.0,
                initial_camera_z=470.0,
                min_camera_z=-110.0,
                max_camera_z=635.0,
                max_yaw_deg=19.0,
                max_pitch_deg=14.0,
                max_strafe_x=72.0,
                max_strafe_y=48.0,
                dolly_speed_mult=1.28,
                particle_color_hex="0xEAE0CF",
                particle_count=55,
                portrait_initial_yaw_deg=7.5,
            ),
        ),
        paris_cite_white_table=ParisCiteWhiteTableSceneSpec(
            scene_id="paris_cite_white_table",
            title_he="הסטודיו ב-Cité des Arts בפריז — תצלום תמי מ-1958 על השולחן הלבן",
            short_label_he="השולחן הלבן בפריז",
            asset_prefix="scene22_paris_table",
            studio_source_relpath="media/wordpress_blog/wp_house_img-20160207-wa0007_f05b5bf0.jpg",
            photo_source_relpath=(
                "media/_upright_derivatives/whatsapp_family/"
                "wa_ada_wertheimer_1958_tami_standing_530bc045_cw270.jpg"
            ),
            scaled_widescreen_wh=(1500, 844),
            pad_top_bottom_px=78,
            chair_y_bounds=(348, 908),
            chair_white_bar_luma_min=135.0,
            chair_white_bar_rmb_max=24.0,
            chair_grey_bar_luma_min=50.0,
            chair_grey_bar_rmb_max=15.5,
            chair_dark_floor_luma_max=52.0,
            chair_dark_floor_rmb_max=14.0,
            chair_inpaint_dilate_px=5,
            fg_chair_sam_prompts=(
                _pg(
                    (534, 348, 908, 908),
                    [
                        (666, 395),
                        (760, 405),
                        (885, 380),
                        (755, 456),
                        (640, 495),
                        (585, 565),
                        (680, 575),
                        (750, 555),
                        (845, 515),
                        (598, 675),
                        (555, 785),
                        (680, 700),
                        (708, 865),
                        (775, 665),
                        (868, 675),
                        (888, 785),
                        (785, 785),
                        (638, 726),
                    ],
                    [
                        (750, 430),
                        (745, 505),
                        (635, 650),
                        (632, 658),
                        (650, 790),
                        (760, 625),
                        (798, 708),
                        (848, 689),
                        (790, 725),
                        (795, 840),
                        (565, 480),
                        (885, 480),
                        (820, 365),
                        (625, 382),
                        (710, 368),
                        (715, 506),
                        (820, 645),
                    ],
                ),
            ),
            mid_table_sam_prompts=(
                _pg(
                    (4, 305, 332, 922),
                    [
                        (75, 365),
                        (155, 365),
                        (145, 425),
                        (65, 520),
                        (165, 520),
                        (165, 680),
                        (65, 760),
                        (165, 845),
                        (285, 825),
                        (105, 895),
                    ],
                    [(225, 330), (265, 445), (255, 780), (45, 905)],
                ),
                _pg(
                    (22, 302, 298, 468),
                    [(68, 380), (70, 435), (102, 345), (150, 380), (152, 430), (182, 390)],
                    [(18, 360), (225, 325), (218, 435)],
                ),
                _pg(
                    (215, 298, 954, 428),
                    [
                        (305, 348),
                        (290, 385),
                        (410, 385),
                        (535, 368),
                        (660, 352),
                        (780, 348),
                        (895, 350),
                        (405, 412),
                    ],
                    [(450, 315), (620, 315), (480, 435), (650, 398), (820, 382), (880, 385)],
                ),
                _pg(
                    (298, 412, 568, 908),
                    [
                        (422, 465),
                        (432, 595),
                        (438, 725),
                        (444, 835),
                        (365, 875),
                        (515, 825),
                        (355, 592),
                        (315, 735),
                    ],
                    [
                        (365, 505),
                        (375, 680),
                        (385, 795),
                        (495, 520),
                        (505, 720),
                        (535, 865),
                    ],
                ),
            ),
            photo_src_crop_box=(118, 96, 1610, 1058),
            photo_dst_quad_inner_xy=(
                (420.0, 368.0),
                (565.0, 353.0),
                (666.0, 364.0),
                (516.0, 389.0),
            ),
            photo_contact_shadow_max_alpha_u8=16,
            photo_contact_shadow_sigma_px=2.2,
            photo_tabletop_clearance_world=0.6,
            portal_id="portal_paris_table_tami_1958",
            target_scene_id="toddler_tami_1958",
            portal_label_he="תמי הפעוטה (1958) · לחצו על התצלום שעל השולחן הלבן להתקרבות",
            portal_short_badge_he="אל תמי 1958",
            bg_inpaint_dilation_px=20,
            bg_inpaint_blur_sigma=3.0,
            bg_jpeg_quality=94,
            room_depth_clean=RoomDepthCleanConfig(
                alpha_threshold_u8=20,
                mask_dilation_px=16,
                post_smooth_sigma_px=3.0,
                interior_blend_band_px=12.0,
            ),
            mid_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=-350.0,
                room_depth_amplitude=118.0,
                room_pin_margin_uv=room_pin,
                layer_z=-350.0,
                layer_depth_amplitude=185.0,
                layer_pin_margin_uv=0.0,
                contact_boxes=(
                    ContactAnchorBox(x0=10, y0=815, x1=332, y1=930, mode="bottom_band", band_px=12),
                    ContactAnchorBox(x0=300, y0=800, x1=545, y1=915, mode="bottom_band", band_px=10),
                    ContactAnchorBox(x0=880, y0=332, x1=956, y1=372, mode="right_band", band_px=10),
                ),
                neural_relief_amplitude=18.0,
                contact_relief_taper_px=30.0,
                enforce_front_of_host=True,
                alpha_threshold_u8=20,
                min_component_area_px=150,
            ),
            fg_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=-350.0,
                room_depth_amplitude=118.0,
                room_pin_margin_uv=room_pin,
                layer_z=-350.0,
                layer_depth_amplitude=195.0,
                layer_pin_margin_uv=0.0,
                contact_boxes=(
                    ContactAnchorBox(x0=545, y0=755, x1=905, y1=910, mode="bottom_band", band_px=12),
                ),
                neural_relief_amplitude=22.0,
                contact_relief_taper_px=28.0,
                enforce_front_of_host=True,
                alpha_threshold_u8=20,
                min_component_area_px=120,
            ),
            camera_rig=SceneCameraRigSpec(
                min_inner_scale=1.94,
                initial_camera_y=-4.0,
                initial_camera_z=525.0,
                min_camera_z=-85.0,
                max_camera_z=685.0,
                max_yaw_deg=22.0,
                max_pitch_deg=15.5,
                max_strafe_x=125.0,
                max_strafe_y=80.0,
                dolly_speed_mult=1.3,
                particle_color_hex="0xE3E6E8",
                particle_count=55,
                portrait_initial_yaw_deg=7.5,
            ),
        ),
        toddler_tami_1958=ToddlerTami1958SceneSpec(
            scene_id="toddler_tami_1958",
            title_he="תמי הפעוטה עומדת (1958) — מבט מקרוב מתוך התצלום המשפחתי",
            short_label_he="תמי הפעוטה (1958)",
            asset_prefix="scene23_tami_1958",
            source_relpath=(
                "media/_upright_derivatives/whatsapp_family/"
                "wa_ada_wertheimer_1958_tami_standing_530bc045_cw270.jpg"
            ),
            src_inner_quad_xy=((292.0, 102.0), (1434.0, 122.0), (1580.0, 1024.0), (146.0, 1006.0)),
            dst_inner_quad_xy=((24.0, 24.0), (1476.0, 24.0), (1476.0, 984.0), (24.0, 984.0)),
            authentic_crop_margins_ltrb_px=(32, 32, 32, 24),
            sam_support_dilate_px=9,
            cutout_core_side_top_margin_px=8,
            mid_sam_prompts=(
                _pg(
                    (265, 58, 1138, 986),
                    [
                        (780, 118),
                        (715, 160),
                        (865, 165),
                        (935, 285),
                        (780, 315),
                        (680, 560),
                        (920, 585),
                        (1025, 735),
                        (480, 640),
                        (375, 730),
                        (445, 835),
                        (680, 820),
                        (895, 865),
                        (560, 935),
                        (760, 945),
                        (385, 955),
                    ],
                    [
                        (585, 130),
                        (985, 130),
                        (565, 330),
                        (1035, 360),
                        (435, 480),
                        (1125, 580),
                        (1145, 840),
                        (235, 780),
                        (215, 935),
                    ],
                ),
            ),
            bg_inpaint_dilation_px=24,
            bg_inpaint_blur_sigma=3.2,
            bg_jpeg_quality=94,
            room_depth_clean=RoomDepthCleanConfig(
                alpha_threshold_u8=20,
                mask_dilation_px=18,
                post_smooth_sigma_px=3.2,
                interior_blend_band_px=12.0,
            ),
            mid_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=-350.0,
                room_depth_amplitude=104.0,
                room_pin_margin_uv=room_pin,
                layer_z=-350.0,
                layer_depth_amplitude=165.0,
                layer_pin_margin_uv=0.0,
                contact_boxes=(
                    ContactAnchorBox(x0=260, y0=966, x1=1120, y1=984, mode="all_opaque", band_px=10),
                    ContactAnchorBox(x0=260, y0=935, x1=1120, y1=999, mode="bottom_band", band_px=10),
                ),
                neural_relief_amplitude=22.0,
                contact_relief_taper_px=30.0,
                enforce_front_of_host=True,
                alpha_threshold_u8=20,
                min_component_area_px=5000,
            ),
            camera_rig=SceneCameraRigSpec(
                min_inner_scale=1.94,
                initial_camera_y=-4.0,
                initial_camera_z=530.0,
                min_camera_z=-85.0,
                max_camera_z=690.0,
                max_yaw_deg=22.0,
                max_pitch_deg=15.5,
                max_strafe_x=130.0,
                max_strafe_y=82.0,
                dolly_speed_mult=1.3,
                particle_color_hex="0xE6DFD5",
                particle_count=52,
            ),
        ),
    )


def build_all_vitrine_bedroom_paris_scenes(
    project_root: Path,
    cfg: VitrineBedroomParisBundleConfig | None = None,
    only_scene: str | None = None,
) -> dict[str, Any]:
    active_cfg = cfg or default_vitrine_bedroom_paris_bundle_config()
    depth_estimator = DepthAnythingEstimator(active_cfg.depth_model)
    sam_extractor = SamBoxMaskExtractor(active_cfg.segmentation)
    matte_extractor = (
        SalientMatteExtractor(active_cfg.segmentation)
        if only_scene in (None, "toddler_tami_1958")
        else None
    )
    lama_inpainter = LamaOcclusionInpainter(active_cfg.scene_inpainting)

    spec_out = Path(active_cfg.spec_output_abspath)
    existing_scenes: dict[str, Any] = {}
    if only_scene is not None and spec_out.exists():
        existing_scenes = dict(json.loads(spec_out.read_text(encoding="utf-8")).get("scenes", {}))

    scenes_map: dict[str, Any] = dict(existing_scenes)
    if only_scene in (None, "vitrine_keys_pen"):
        s20 = build_vitrine_keys_pen_scene(
            project_root, active_cfg, depth_estimator, sam_extractor, lama_inpainter
        )
        scenes_map[s20["id"]] = s20
    if only_scene in (None, "bar_giora_bedroom"):
        s21 = build_bar_giora_bedroom_scene(
            project_root, active_cfg, depth_estimator, sam_extractor, lama_inpainter
        )
        scenes_map[s21["id"]] = s21
    if only_scene in (None, "paris_cite_white_table"):
        s22 = build_paris_cite_white_table_scene(
            project_root, active_cfg, depth_estimator, sam_extractor, lama_inpainter
        )
        scenes_map[s22["id"]] = s22
    if only_scene in (None, "toddler_tami_1958"):
        assert matte_extractor is not None
        s23 = build_toddler_tami_1958_scene(
            project_root, active_cfg, depth_estimator, sam_extractor, matte_extractor, lama_inpainter
        )
        scenes_map[s23["id"]] = s23

    spec_payload = {
        "schemaVersion": "1.0",
        "workstream": "Scenes 20-23 — Vitrine + Bar Giora Bedroom (#42) & Paris White Table + Toddler Tami 1958 (#43)",
        "generatedBy": "prototype/vitrine_bedroom_paris_scenes_builder.py",
        "ssotConfig": active_cfg.to_dict(),
        "scenes": scenes_map,
    }
    spec_out.parent.mkdir(parents=True, exist_ok=True)
    spec_out.write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[Done] Saved SSOT scene spec to {spec_out}")
    return spec_payload


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2.5D spatial scenes 20-23 (#42 & #43).")
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Verify lossless SSOT config JSON roundtrip and exit without running neural models.",
    )
    parser.add_argument(
        "--only-scene",
        choices=("vitrine_keys_pen", "bar_giora_bedroom", "paris_cite_white_table", "toddler_tami_1958"),
        default=None,
        help="Rebuild only a single scene from the bundle and update spec JSON in place.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_cli_args(argv)
    cfg = default_vitrine_bedroom_paris_bundle_config()
    roundtripped = VitrineBedroomParisBundleConfig.from_json(cfg.to_json())
    if roundtripped != cfg:
        raise RuntimeError("VitrineBedroomParisBundleConfig failed lossless JSON round-trip check!")
    if args.dry_run_config:
        print("VitrineBedroomParisBundleConfig lossless JSON round-trip OK.")
        return
    build_all_vitrine_bedroom_paris_scenes(_REPO_ROOT, cfg, only_scene=args.only_scene)


if __name__ == "__main__":
    main()
