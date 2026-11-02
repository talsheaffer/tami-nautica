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
    extract_layer_window_from_expanded,
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
    _expand_plate_reflect_and_heal_collar_2700,
    _keep_largest_cc_u8,
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
    collar_heal_half_width_px: int
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
            "collar_heal_half_width_px": self.collar_heal_half_width_px,
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
            collar_heal_half_width_px=int(data["collar_heal_half_width_px"]),
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
    vitrine_src_crop_box_1500: tuple[int, int, int, int]
    vitrine_mount_side_px: int
    vitrine_mount_center_xy: tuple[float, float]
    vitrine_rotation_deg: float
    vitrine_flight_roll_deg: float
    vitrine_indoor_luma_scale: float
    vitrine_shadow_shift_xy: tuple[float, float]
    vitrine_shadow_sigma_px: float
    vitrine_shadow_max_alpha_u8: int
    portal_id: str
    target_scene_id: str
    portal_label_he: str
    portal_short_badge_he: str
    portal_half_box_px: float
    fg_sam_prompts: tuple[SamPointGroupPrompt, ...]
    mid_sam_prompts: tuple[SamPointGroupPrompt, ...]
    bg_inpaint_dilation_px: int
    bg_inpaint_blur_sigma: float
    collar_heal_half_width_px: int
    bg_jpeg_quality: int
    fg_bleed_guard_min_z: float
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
            "vitrine_src_crop_box_1500": list(self.vitrine_src_crop_box_1500),
            "vitrine_mount_side_px": self.vitrine_mount_side_px,
            "vitrine_mount_center_xy": list(self.vitrine_mount_center_xy),
            "vitrine_rotation_deg": self.vitrine_rotation_deg,
            "vitrine_flight_roll_deg": self.vitrine_flight_roll_deg,
            "vitrine_indoor_luma_scale": self.vitrine_indoor_luma_scale,
            "vitrine_shadow_shift_xy": list(self.vitrine_shadow_shift_xy),
            "vitrine_shadow_sigma_px": self.vitrine_shadow_sigma_px,
            "vitrine_shadow_max_alpha_u8": self.vitrine_shadow_max_alpha_u8,
            "portal_id": self.portal_id,
            "target_scene_id": self.target_scene_id,
            "portal_label_he": self.portal_label_he,
            "portal_short_badge_he": self.portal_short_badge_he,
            "portal_half_box_px": self.portal_half_box_px,
            "fg_sam_prompts": [p.to_dict() for p in self.fg_sam_prompts],
            "mid_sam_prompts": [p.to_dict() for p in self.mid_sam_prompts],
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "bg_inpaint_blur_sigma": self.bg_inpaint_blur_sigma,
            "collar_heal_half_width_px": self.collar_heal_half_width_px,
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "fg_bleed_guard_min_z": self.fg_bleed_guard_min_z,
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
        vc = data["vitrine_src_crop_box_1500"]
        mc = data["vitrine_mount_center_xy"]
        ss = data["vitrine_shadow_shift_xy"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            asset_prefix=str(data["asset_prefix"]),
            bedroom_source_relpath=str(data["bedroom_source_relpath"]),
            vitrine_source_relpath=str(data["vitrine_source_relpath"]),
            scaled_full_wh=(int(sf[0]), int(sf[1])),
            crop_y_range=(int(cy[0]), int(cy[1])),
            vitrine_src_crop_box_1500=(int(vc[0]), int(vc[1]), int(vc[2]), int(vc[3])),
            vitrine_mount_side_px=int(data["vitrine_mount_side_px"]),
            vitrine_mount_center_xy=(float(mc[0]), float(mc[1])),
            vitrine_rotation_deg=float(data["vitrine_rotation_deg"]),
            vitrine_flight_roll_deg=float(data["vitrine_flight_roll_deg"]),
            vitrine_indoor_luma_scale=float(data["vitrine_indoor_luma_scale"]),
            vitrine_shadow_shift_xy=(float(ss[0]), float(ss[1])),
            vitrine_shadow_sigma_px=float(data["vitrine_shadow_sigma_px"]),
            vitrine_shadow_max_alpha_u8=int(data["vitrine_shadow_max_alpha_u8"]),
            portal_id=str(data["portal_id"]),
            target_scene_id=str(data["target_scene_id"]),
            portal_label_he=str(data["portal_label_he"]),
            portal_short_badge_he=str(data["portal_short_badge_he"]),
            portal_half_box_px=float(data["portal_half_box_px"]),
            fg_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["fg_sam_prompts"]),
            mid_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["mid_sam_prompts"]),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            bg_inpaint_blur_sigma=float(data["bg_inpaint_blur_sigma"]),
            collar_heal_half_width_px=int(data["collar_heal_half_width_px"]),
            bg_jpeg_quality=int(data["bg_jpeg_quality"]),
            fg_bleed_guard_min_z=float(data["fg_bleed_guard_min_z"]),
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
    top_bottom_inpaint_band_px: int
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
    collar_heal_half_width_px: int
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
            "top_bottom_inpaint_band_px": self.top_bottom_inpaint_band_px,
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
            "collar_heal_half_width_px": self.collar_heal_half_width_px,
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
            top_bottom_inpaint_band_px=int(data["top_bottom_inpaint_band_px"]),
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
            collar_heal_half_width_px=int(data["collar_heal_half_width_px"]),
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
    border_heal_margins_ltrb_px: tuple[int, int, int, int]
    sam_support_dilate_px: int
    cutout_safe_margins_ltrb_px: tuple[int, int, int, int]
    mid_sam_prompts: tuple[SamPointGroupPrompt, ...]
    bg_inpaint_dilation_px: int
    bg_inpaint_blur_sigma: float
    collar_heal_half_width_px: int
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
            "border_heal_margins_ltrb_px": list(self.border_heal_margins_ltrb_px),
            "sam_support_dilate_px": self.sam_support_dilate_px,
            "cutout_safe_margins_ltrb_px": list(self.cutout_safe_margins_ltrb_px),
            "mid_sam_prompts": [p.to_dict() for p in self.mid_sam_prompts],
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "bg_inpaint_blur_sigma": self.bg_inpaint_blur_sigma,
            "collar_heal_half_width_px": self.collar_heal_half_width_px,
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToddlerTami1958SceneSpec:
        bm = data["border_heal_margins_ltrb_px"]
        cm = data["cutout_safe_margins_ltrb_px"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            asset_prefix=str(data["asset_prefix"]),
            source_relpath=str(data["source_relpath"]),
            src_inner_quad_xy=tuple((float(pt[0]), float(pt[1])) for pt in data["src_inner_quad_xy"]),
            dst_inner_quad_xy=tuple((float(pt[0]), float(pt[1])) for pt in data["dst_inner_quad_xy"]),
            border_heal_margins_ltrb_px=(int(bm[0]), int(bm[1]), int(bm[2]), int(bm[3])),
            sam_support_dilate_px=int(data["sam_support_dilate_px"]),
            cutout_safe_margins_ltrb_px=(int(cm[0]), int(cm[1]), int(cm[2]), int(cm[3])),
            mid_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["mid_sam_prompts"]),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            bg_inpaint_blur_sigma=float(data["bg_inpaint_blur_sigma"]),
            collar_heal_half_width_px=int(data["collar_heal_half_width_px"]),
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


def _expand_vitrine_gallery_wall_2700(
    clean_inner_u8: np.ndarray,
    target_wh: tuple[int, int],
    dest_xy: tuple[int, int],
    collar_band_px: int,
    lama_inpainter: LamaOcclusionInpainter,
) -> Image.Image:
    """
    Expands Scene 20's `1500x1000` vitrine plate to `2700x1800` pure museum plaster wall
    without reflecting the dark-blue vitrine cavity into the outer expansion ring.
    """
    ih, iw, _ = clean_inner_u8.shape
    ox, oy = dest_xy
    plaster_donor = clean_inner_u8.astype(np.float32).copy()
    x_l, x_r = 275, 1225
    y_t, y_b = 64, 952
    left_col = np.mean(plaster_donor[y_t:y_b, x_l - 24 : x_l, :], axis=1, keepdims=True)
    right_col = np.mean(plaster_donor[y_t:y_b, x_r : x_r + 24, :], axis=1, keepdims=True)
    alpha_x = np.linspace(0.0, 1.0, x_r - x_l, dtype=np.float32)[None, :, None]
    base_wall = (1.0 - alpha_x) * left_col + alpha_x * right_col

    strip = plaster_donor[y_t:y_b, 24:264, :]
    strip_hp = strip - ndi.gaussian_filter(strip, sigma=(7.0, 7.0, 0.0))
    tiled_hp = np.tile(
        np.concatenate([strip_hp, strip_hp[:, ::-1, :]], axis=1),
        (1, ((x_r - x_l) // (2 * strip_hp.shape[1])) + 2, 1),
    )[:, : (x_r - x_l), :]
    plaster_donor[y_t:y_b, x_l:x_r, :] = np.clip(base_wall + tiled_hp, 0.0, 255.0)
    plaster_donor_u8 = np.rint(plaster_donor).astype(np.uint8)

    expanded_wall_img = _expand_plate_reflect_and_heal_collar_2700(
        core_rgb_u8=plaster_donor_u8,
        target_wh=target_wh,
        dest_xy=dest_xy,
        collar_band_px=collar_band_px,
        lama_inpainter=lama_inpainter,
    )
    out_np = np.asarray(expanded_wall_img, dtype=np.uint8).copy()
    # Inside [y_t-24:y_b+24, x_l-24:x_r+24] restore the crisp LaMa-cleaned blue vitrine cavity
    vy0, vy1 = y_t - 24, y_b + 24
    vx0, vx1 = x_l - 24, x_r + 24
    out_np[oy + vy0 : oy + vy1, ox + vx0 : ox + vx1, :] = clean_inner_u8[vy0:vy1, vx0:vx1, :]
    return Image.fromarray(out_np, mode="RGB")


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

    print(f"\n[Scene 20: {spec.scene_id}] Building 3-layer museum vitrine scene...")
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
    fg_canvas = _prune_tiny_alpha_specks_rgba(
        Image.fromarray(np.dstack([fg_rgb, fg_u8]), mode="RGBA"),
        min_area_px=spec.fg_depth_solve.min_component_area_px,
        alpha_thresh_u8=spec.fg_depth_solve.alpha_threshold_u8,
    )

    _assert_border_alpha_contract(mid_canvas, f"{spec.scene_id}.mid")
    _assert_border_alpha_contract(fg_canvas, f"{spec.scene_id}.fg")
    mid_np = np.asarray(mid_canvas, dtype=np.uint8)
    fg_np = np.asarray(fg_canvas, dtype=np.uint8)

    # Inpaint pink shelf, pen, keys, pen cast shadow, and outer frame out of inner cavity/wall
    bg_hole = np.where((mid_np[:, :, 3] >= 16) | (fg_np[:, :, 3] >= 16), 255, 0).astype(np.uint8)
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
    bg_inpainted_rgb = _expand_vitrine_gallery_wall_2700(
        clean_inner_u8=np.asarray(clean_inner_rgb, dtype=np.uint8),
        target_wh=(ew, eh),
        dest_xy=(ox, oy),
        collar_band_px=spec.collar_heal_half_width_px,
        lama_inpainter=lama_inpainter,
    )

    master_2700_rgba = bg_inpainted_rgb.convert("RGBA")
    master_2700_rgba.alpha_composite(Image.fromarray(src_np, mode="RGB").convert("RGBA"), dest=(ox, oy))
    master_2700_rgb = master_2700_rgba.convert("RGB")

    raw_intact_depth_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(master_2700_rgb) * 255.0), 0, 255
    ).astype(np.uint8)
    inpainted_room_depth_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(bg_inpainted_rgb), dtype=np.uint8
    )
    bg_depth_clean_u8 = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(mid_np, fg_np),
        cfg=spec.room_depth_clean,
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
    fg_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=fg_np,
        cfg=spec.fg_depth_solve,
        neural_protrusion_norm=fg_prot,
    )

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
        "depthTopologyVerification": {"mid": mid_verify, "fg": fg_verify},
    }


# ============================================================================
# Scene 21 Builder: `bar_giora_bedroom` (`scene21_bar_giora_*`)
# ============================================================================


def _build_mounted_rotated_vitrine_rgba(
    vitrine_src_1500_np: np.ndarray,
    inner_wh: tuple[int, int],
    spec: BarGioraBedroomSceneSpec,
) -> Image.Image:
    iw, ih = inner_wh
    vx0, vy0, vx1, vy1 = spec.vitrine_src_crop_box_1500
    side = spec.vitrine_mount_side_px
    vit_crop = Image.fromarray(vitrine_src_1500_np[vy0:vy1, vx0:vx1], mode="RGB").resize(
        (side, side), Image.Resampling.LANCZOS
    )
    vit_np = np.clip(
        np.rint(np.asarray(vit_crop, dtype=np.float32) * spec.vitrine_indoor_luma_scale),
        0,
        255,
    ).astype(np.uint8)

    pad = side
    work_dim = side + 2 * pad
    work_rgb = np.zeros((work_dim, work_dim, 3), dtype=np.uint8)
    work_mask = np.zeros((work_dim, work_dim), dtype=np.uint8)
    work_rgb[pad : pad + side, pad : pad + side] = vit_np
    work_mask[pad : pad + side, pad : pad + side] = 255

    # OpenCV getRotationMatrix2D: negative angle = clockwise rotation on screen!
    rot_m = cv2.getRotationMatrix2D((work_dim * 0.5, work_dim * 0.5), spec.vitrine_rotation_deg, 1.0)
    rot_rgb = cv2.warpAffine(
        work_rgb, rot_m, (work_dim, work_dim), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
    )
    rot_alpha = cv2.warpAffine(
        work_mask, rot_m, (work_dim, work_dim), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    box_aa = np.where(rot_alpha >= 20, rot_alpha, 0).astype(np.uint8)

    cx, cy = spec.vitrine_mount_center_xy
    dx = int(round(cx - work_dim * 0.5))
    dy = int(round(cy - work_dim * 0.5))

    canvas_rgb = np.zeros((ih, iw, 3), dtype=np.uint8)
    canvas_a = np.zeros((ih, iw), dtype=np.uint8)
    y0, y1 = max(0, dy), min(ih, dy + work_dim)
    x0, x1 = max(0, dx), min(iw, dx + work_dim)
    sy0, sy1 = y0 - dy, y1 - dy
    sx0, sx1 = x0 - dx, x1 - dx
    canvas_rgb[y0:y1, x0:x1] = rot_rgb[sy0:sy1, sx0:sx1]
    canvas_a[y0:y1, x0:x1] = box_aa[sy0:sy1, sx0:sx1]

    # Directional cast shadow down-left matching right-hand bedroom window illumination
    sh_dx, sh_dy = spec.vitrine_shadow_shift_xy
    solid = (canvas_a >= 96).astype(np.float32)
    shifted = ndi.shift(solid, shift=(sh_dy, sh_dx), order=1)
    shadow_blur = ndi.gaussian_filter(shifted, sigma=spec.vitrine_shadow_sigma_px)
    if float(shadow_blur.max()) > 1e-5:
        shadow_blur = shadow_blur / float(shadow_blur.max())
    shadow_a = np.clip(
        np.rint(shadow_blur * float(spec.vitrine_shadow_max_alpha_u8)),
        0,
        spec.vitrine_shadow_max_alpha_u8,
    ).astype(np.uint8)

    out_rgba = np.zeros((ih, iw, 4), dtype=np.uint8)
    sh_only = (canvas_a == 0) & (shadow_a > 0)
    out_rgba[sh_only, :3] = 28
    out_rgba[sh_only, 3] = shadow_a[sh_only]
    fg_px = canvas_a > 0
    out_rgba[fg_px, :3] = canvas_rgb[fg_px]
    out_rgba[fg_px, 3] = canvas_a[fg_px]
    return Image.fromarray(out_rgba, mode="RGBA")


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

    print(f"\n[Scene 21: {spec.scene_id}] Building Bar Giora bedroom + rotated wall vitrine...")
    fw, fh = spec.scaled_full_wh
    cy0, cy1 = spec.crop_y_range
    full_img = (
        Image.open(project_root / spec.bedroom_source_relpath)
        .convert("RGB")
        .resize((fw, fh), Image.Resampling.LANCZOS)
    )
    full_np = np.asarray(full_img, dtype=np.uint8)
    inner_np = full_np[cy0:cy1, :, :]
    inner_img = Image.fromarray(inner_np, mode="RGB")

    fg_u8 = _segment_prompt_union_u8(
        rgb_image=inner_img,
        prompts=spec.fg_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=250,
        max_pinhole_fill_px=180,
    )
    mid_raw_u8 = _segment_prompt_union_u8(
        rgb_image=inner_img,
        prompts=spec.mid_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=250,
        max_pinhole_fill_px=180,
    )
    mid_filled = ndi.binary_fill_holes(mid_raw_u8 >= 96)
    mid_raw_u8 = np.where(mid_filled & (mid_raw_u8 < 255), np.uint8(255), mid_raw_u8)
    mid_u8 = np.where(fg_u8 >= 128, np.uint8(0), mid_raw_u8)
    # Keep mid strictly inside 1px inner border so mid has 0 border alpha
    mid_u8[0, :] = 0
    mid_u8[-1, :] = 0
    mid_u8[:, 0] = 0
    mid_u8[:, -1] = 0
    mid_u8[-3:, :] = np.minimum(
        mid_u8[-3:, :],
        np.array([[180], [90], [0]], dtype=np.uint8),
    )

    fg_rgb = decontaminate_matte_edge_rgb(inner_np, fg_u8)
    fg_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            Image.fromarray(np.dstack([fg_rgb, fg_u8]), mode="RGBA"),
            min_area_px=spec.fg_depth_solve.min_component_area_px,
            alpha_thresh_u8=spec.fg_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.fg_depth_solve,
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
    fg_np = np.asarray(fg_canvas, dtype=np.uint8)
    mid_np = np.asarray(mid_canvas, dtype=np.uint8)

    vitrine_src_1500_np = np.asarray(
        Image.open(project_root / spec.vitrine_source_relpath)
        .convert("RGB")
        .resize((iw, ih), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    vitrine_canvas = _build_mounted_rotated_vitrine_rgba(vitrine_src_1500_np, (iw, ih), spec)
    _assert_border_alpha_contract(vitrine_canvas, f"{spec.scene_id}.mounted_wall_vitrine")
    vitrine_np = np.asarray(vitrine_canvas, dtype=np.uint8)

    # Inpaint fg + mid out of the full 1500x1125 bedroom plate so ceiling & carpet extend natively
    full_hole = np.zeros((fh, fw), dtype=np.uint8)
    full_hole[cy0:cy1, :] = np.where((fg_np[:, :, 3] >= 16) | (mid_np[:, :, 3] >= 16), 255, 0).astype(
        np.uint8
    )
    # Extend wardrobe & armchair holes to top/bottom of 1500x1125 plate so no stub remains outside cy0..cy1
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
    bg_inpainted_rgb = _expand_plate_reflect_and_heal_collar_2700(
        core_rgb_u8=np.asarray(clean_full_bedroom, dtype=np.uint8),
        target_wh=(ew, eh),
        dest_xy=(ox, oy - cy0),
        collar_band_px=spec.collar_heal_half_width_px,
        lama_inpainter=lama_inpainter,
    )

    master_2700_rgba = bg_inpainted_rgb.convert("RGBA")
    master_2700_rgba.alpha_composite(full_img.convert("RGBA"), dest=(ox, oy - cy0))
    master_2700_rgb = master_2700_rgba.convert("RGB")

    raw_intact_depth_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(master_2700_rgb) * 255.0), 0, 255
    ).astype(np.uint8)
    inpainted_room_depth_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(bg_inpainted_rgb), dtype=np.uint8
    )
    bg_depth_clean_u8 = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(mid_np, fg_np),
        cfg=spec.room_depth_clean,
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

    vit_opa = vitrine_np[:, :, 3] >= spec.vitrine_depth_solve.alpha_threshold_u8
    vit_dist = ndi.distance_transform_edt(vit_opa).astype(np.float32)
    vit_prot = np.clip(vit_dist / max(1.0, float(vit_dist.max()) * 0.35), 0.0, 1.0).astype(np.float32)
    vit_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=vitrine_np,
        cfg=spec.vitrine_depth_solve,
        neural_protrusion_norm=vit_prot,
    )

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
    vit_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.mounted_wall_vitrine",
        bg_depth_clean_u8,
        vitrine_np,
        vit_solved.depth_u8,
        vit_solved.contact_mask,
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
    Image.fromarray(fg_solved.depth_u8, mode="L").save(scenes_dir / f"{pfx}_fg_depth.png")
    vitrine_canvas.save(scenes_dir / f"{pfx}_vitrine.png")
    Image.fromarray(vit_solved.depth_u8, mode="L").save(scenes_dir / f"{pfx}_vitrine_depth.png")

    portal_world_z = round(float(np.median(vit_solved.world_z_map[vit_opa])), 2)
    cx, cy = spec.vitrine_mount_center_xy
    hb = spec.portal_half_box_px
    portal_box = [round(cx - hb, 1), round(cy - hb, 1), round(cx + hb, 1), round(cy + hb, 1)]
    vit_tight_box = _tight_alpha_bbox(vitrine_np, 20)

    extra_layers = [
        {
            "id": "mounted_wall_vitrine",
            "labelHe": "תיבת הוויטרינה התלויה על קיר החדר (מסובבת 130° להתאמת אור החלון)",
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
        "fgExpanded": False,
        "fgBleedEdges": {
            "left": True,
            "right": True,
            "top": True,
            "bottom": True,
            "guardMinZ": spec.fg_bleed_guard_min_z,
        },
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

    print(f"\n[Scene 22: {spec.scene_id}] Building Paris Cité studio white table + 1958 photo...")
    sw, sh = spec.scaled_widescreen_wh
    pad_tb = spec.pad_top_bottom_px
    tb_band = spec.top_bottom_inpaint_band_px

    w153 = np.asarray(
        Image.open(project_root / spec.studio_source_relpath)
        .convert("RGB")
        .resize((sw, sh), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    seed_1500 = cv2.copyMakeBorder(w153, pad_tb, pad_tb, 0, 0, borderType=cv2.BORDER_REPLICATE)
    tb_mask = np.zeros((ih, iw), dtype=np.uint8)
    tb_mask[:tb_band, :] = 255
    tb_mask[ih - tb_band :, :] = 255
    healed_tb = np.array(
        lama_inpainter.inpaint_background(Image.fromarray(seed_1500, mode="RGB"), tb_mask),
        dtype=np.uint8,
        copy=True,
    )
    keep_margin = tb_band - pad_tb - 2
    healed_tb[pad_tb + keep_margin : ih - pad_tb - keep_margin, :] = w153[
        keep_margin : sh - keep_margin, :
    ]
    studio_1500_img = Image.fromarray(healed_tb, mode="RGB")

    # 1. Segment wooden folding chair (`fg`) cleanly separated from white table bars & dark floor
    chair_sam = _segment_prompt_union_u8(
        rgb_image=studio_1500_img,
        prompts=spec.fg_chair_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=80,
        max_pinhole_fill_px=0,
    )
    rgb_f = healed_tb.astype(np.float32)
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

    chair_rgb = decontaminate_matte_edge_rgb(healed_tb, chair_u8)
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

    # 2. Inpaint chair out of studio_1500_img so white tabletop & front legs are 100% continuous behind chair
    k_ch = 2 * spec.chair_inpaint_dilate_px + 1
    chair_inp_mask = cv2.dilate(
        (fg_np[:, :, 3] >= 16).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_ch, k_ch)),
    )
    no_chair_img = lama_inpainter.inpaint_background(studio_1500_img, chair_inp_mask)
    no_chair_np = np.asarray(no_chair_img, dtype=np.uint8)

    # 3. Segment cabinet + flowerpots + continuous white tabletop + front-left metal T-leg (`mid`)
    mid_u8 = _segment_prompt_union_u8(
        rgb_image=no_chair_img,
        prompts=spec.mid_table_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=120,
        max_pinhole_fill_px=60,
    )
    mid_u8[0, :] = 0
    mid_u8[-1, :] = 0
    mid_u8[:, 0] = 0
    mid_u8[:, -1] = 0
    mid_rgb = decontaminate_matte_edge_rgb(no_chair_np, mid_u8)
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

    # 4. Build perspective-placed 1958 photo on the white tabletop
    photo_src_rgb = Image.open(project_root / spec.photo_source_relpath).convert("RGB")
    photo_canvas = _build_tabletop_1958_photo_rgba(photo_src_rgb, (iw, ih), spec)
    _assert_border_alpha_contract(photo_canvas, f"{spec.scene_id}.placed_1958_photo")
    photo_np = np.asarray(photo_canvas, dtype=np.uint8)

    # 5. Inpaint mid out of no_chair_img and expand to 2700x1800 clean studio room
    bg_hole = np.where((mid_np[:, :, 3] >= 16) | (fg_np[:, :, 3] >= 16), 255, 0).astype(np.uint8)
    clean_studio_inner = _inpaint_masked_plate_lama(
        src_rgb=no_chair_img,
        binary_mask_u8=bg_hole,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.bg_inpaint_dilation_px,
        blur_sigma=spec.bg_inpaint_blur_sigma,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )
    bg_inpainted_rgb = _expand_plate_reflect_and_heal_collar_2700(
        core_rgb_u8=np.asarray(clean_studio_inner, dtype=np.uint8),
        target_wh=(ew, eh),
        dest_xy=(ox, oy),
        collar_band_px=spec.collar_heal_half_width_px,
        lama_inpainter=lama_inpainter,
    )

    master_2700_rgba = bg_inpainted_rgb.convert("RGBA")
    master_2700_rgba.alpha_composite(studio_1500_img.convert("RGBA"), dest=(ox, oy))
    master_2700_rgb = master_2700_rgba.convert("RGB")

    raw_intact_depth_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(master_2700_rgb) * 255.0), 0, 255
    ).astype(np.uint8)
    inpainted_room_depth_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(bg_inpainted_rgb), dtype=np.uint8
    )
    bg_depth_clean_u8 = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(mid_np, fg_np),
        cfg=spec.room_depth_clean,
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

    print(f"\n[Scene 23: {spec.scene_id}] Building zoomed-in 1958 toddler Tami portrait scene...")
    raw_np = np.asarray(Image.open(project_root / spec.source_relpath).convert("RGB"), dtype=np.uint8)
    src_q = np.asarray(spec.src_inner_quad_xy, dtype=np.float32)
    dst_q = np.asarray(spec.dst_inner_quad_xy, dtype=np.float32)
    rect_inner = cv2.warpPerspective(
        raw_np, cv2.getPerspectiveTransform(src_q, dst_q), (iw, ih), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
    )

    ml, mt, mr, mb = spec.border_heal_margins_ltrb_px
    b_mask = np.zeros((ih, iw), dtype=np.uint8)
    b_mask[:mt, :] = 255
    b_mask[ih - mb :, :] = 255
    b_mask[:, :ml] = 255
    b_mask[:, iw - mr :] = 255
    healed_np = np.array(
        lama_inpainter.inpaint_background(Image.fromarray(rect_inner, mode="RGB"), b_mask),
        dtype=np.uint8,
        copy=True,
    )
    healed_np[mt - 2 : ih - mb + 2, ml - 2 : iw - mr + 2] = rect_inner[
        mt - 2 : ih - mb + 2, ml - 2 : iw - mr + 2
    ]
    healed_img = Image.fromarray(healed_np, mode="RGB")

    tami_sam = _segment_prompt_union_u8(
        rgb_image=healed_img,
        prompts=spec.mid_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=400,
        max_pinhole_fill_px=400,
    )
    isnet_a = np.asarray(remove(healed_img, session=matte_extractor.session), dtype=np.uint8)[:, :, 3]
    k_sup = 2 * spec.sam_support_dilate_px + 1
    sam_sup = cv2.dilate(
        (tami_sam >= 128).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sup, k_sup)),
    ) > 0
    fused_a = np.maximum(tami_sam, np.where(sam_sup, isnet_a, np.uint8(0)))
    cml, cmt, cmr, cmb = spec.cutout_safe_margins_ltrb_px
    fused_a[:cmt, :] = 0
    fused_a[ih - cmb :, :] = 0
    fused_a[:, :cml] = 0
    fused_a[:, iw - cmr :] = 0
    # Strictly retain ONLY the single main connected component of toddler Tami (zero head specks!)
    clean_tami_a = _keep_largest_cc_u8(fused_a, thresh=spec.mid_depth_solve.alpha_threshold_u8)

    mid_rgb = decontaminate_matte_edge_rgb(healed_np, clean_tami_a)
    mid_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            Image.fromarray(np.dstack([mid_rgb, clean_tami_a]), mode="RGBA"),
            min_area_px=spec.mid_depth_solve.min_component_area_px,
            alpha_thresh_u8=spec.mid_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.mid_depth_solve,
    )
    fg_empty = Image.fromarray(np.zeros((ih, iw, 4), dtype=np.uint8), mode="RGBA")
    _assert_border_alpha_contract(mid_canvas, f"{spec.scene_id}.mid")
    _assert_border_alpha_contract(fg_empty, f"{spec.scene_id}.fg")
    mid_np = np.asarray(mid_canvas, dtype=np.uint8)

    hole_u8 = np.where(mid_np[:, :, 3] >= 16, 255, 0).astype(np.uint8)
    clean_bg_inner = _inpaint_masked_plate_lama(
        src_rgb=healed_img,
        binary_mask_u8=hole_u8,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.bg_inpaint_dilation_px,
        blur_sigma=spec.bg_inpaint_blur_sigma,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )
    bg_inpainted_rgb = _expand_plate_reflect_and_heal_collar_2700(
        core_rgb_u8=np.asarray(clean_bg_inner, dtype=np.uint8),
        target_wh=(ew, eh),
        dest_xy=(ox, oy),
        collar_band_px=spec.collar_heal_half_width_px,
        lama_inpainter=lama_inpainter,
    )

    master_2700_rgba = bg_inpainted_rgb.convert("RGBA")
    master_2700_rgba.alpha_composite(healed_img.convert("RGBA"), dest=(ox, oy))
    master_2700_rgb = master_2700_rgba.convert("RGB")

    raw_intact_depth_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(master_2700_rgb) * 255.0), 0, 255
    ).astype(np.uint8)
    inpainted_room_depth_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(bg_inpainted_rgb), dtype=np.uint8
    )
    bg_depth_clean_u8 = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(mid_np,),
        cfg=spec.room_depth_clean,
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
    fg_empty.save(scenes_dir / f"{pfx}_fg.png")

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
        "fgDepthTexture": None,
        "fgExpanded": False,
        "boreSightAtRest": True,
        "bgZ": spec.mid_depth_solve.bg_z,
        "midZ": spec.mid_depth_solve.layer_z,
        "fgZ": -180.0,
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
        "fgCoverageRatio": 0.0,
        "depthTopologyVerification": {"mid": mid_verify},
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
            "235713ae-3cfb-48f0-94e4-2012c0824fd5/scratch/subagent_c_scenes.json"
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
            collar_heal_half_width_px=28,
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
                contact_relief_taper_px=26.0,
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
                    ContactAnchorBox(x0=306, y0=72, x1=336, y1=946, mode="left_band", band_px=10),
                    ContactAnchorBox(x0=1160, y0=72, x1=1192, y1=946, mode="right_band", band_px=10),
                ),
                neural_relief_amplitude=42.0,
                contact_relief_taper_px=24.0,
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
            vitrine_src_crop_box_1500=(312, 74, 1188, 944),
            vitrine_mount_side_px=122,
            vitrine_mount_center_xy=(555.0, 424.0),
            vitrine_rotation_deg=-130.0,
            vitrine_flight_roll_deg=-130.0,
            vitrine_indoor_luma_scale=0.92,
            vitrine_shadow_shift_xy=(-16.0, 12.0),
            vitrine_shadow_sigma_px=8.0,
            vitrine_shadow_max_alpha_u8=18,
            portal_id="portal_bar_giora_vitrine",
            target_scene_id="vitrine_keys_pen",
            portal_label_he="תיבת הוויטרינה מ׳בנה ביתך׳ · לחצו להתקרבות וסיבוב אל תוך הוויטרינה",
            portal_short_badge_he="אל הוויטרינה",
            portal_half_box_px=63.0,
            fg_sam_prompts=(
                _pg(
                    (0, 0, 255, 1000),
                    [
                        _map21_pt(p)
                        for p in [
                            (150, 120),
                            (210, 250),
                            (160, 500),
                            (240, 650),
                            (180, 880),
                            (275, 950),
                        ]
                    ]
                    + [(25.0, 200.0), (25.0, 750.0)],
                    [_map21_pt(p) for p in [(255, 120), (275, 450), (312, 750), (320, 950)]],
                ),
                _pg(
                    (1340, 695, 1500, 1000),
                    [
                        _map21_pt(p)
                        for p in [
                            (1390, 725),
                            (1385, 810),
                            (1315, 815),
                            (1355, 895),
                            (1395, 955),
                            (1302, 950),
                        ]
                    ],
                    [_map21_pt(p) for p in [(1335, 720), (1272, 820), (1270, 940)]],
                ),
            ),
            mid_sam_prompts=(
                _pg(
                    (205, 526, 952, 1000),
                    [
                        _map21_pt(p)
                        for p in [
                            (340, 720),
                            (480, 680),
                            (620, 710),
                            (720, 560),
                            (850, 595),
                            (760, 660),
                            (540, 860),
                            (420, 940),
                            (640, 880),
                            (810, 780),
                            (925, 675),
                        ]
                    ]
                    + [
                        (245.0, 715.0),
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
                            (420, 580),
                            (560, 540),
                            (660, 510),
                            (850, 920),
                            (720, 960),
                        ]
                    ]
                    + [
                        (235.0, 555.0),
                        (295.0, 645.0),
                        (350.0, 625.0),
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
            collar_heal_half_width_px=28,
            bg_jpeg_quality=94,
            fg_bleed_guard_min_z=210.0,
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
                min_component_area_px=300,
            ),
            fg_depth_solve=ContactAnchoredLayerDepthConfig(
                bg_z=-350.0,
                room_depth_amplitude=118.0,
                room_pin_margin_uv=room_pin,
                layer_z=-350.0,
                layer_depth_amplitude=195.0,
                layer_pin_margin_uv=0.0,
                contact_boxes=(
                    ContactAnchorBox(x0=120, y0=920, x1=258, y1=1000, mode="bottom_band", band_px=12),
                    ContactAnchorBox(x0=1340, y0=935, x1=1500, y1=1000, mode="bottom_band", band_px=12),
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
                    ContactAnchorBox(x0=465, y0=335, x1=645, y1=372, mode="top_band", band_px=8),
                    ContactAnchorBox(x0=465, y0=476, x1=645, y1=515, mode="bottom_band", band_px=8),
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
            top_bottom_inpaint_band_px=86,
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
            collar_heal_half_width_px=28,
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
                    ContactAnchorBox(x0=10, y0=815, x1=332, y1=924, mode="bottom_band", band_px=12),
                    ContactAnchorBox(x0=300, y0=800, x1=545, y1=910, mode="bottom_band", band_px=10),
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
            border_heal_margins_ltrb_px=(30, 30, 30, 22),
            sam_support_dilate_px=9,
            cutout_safe_margins_ltrb_px=(40, 40, 40, 14),
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
            collar_heal_half_width_px=28,
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
                    ContactAnchorBox(x0=260, y0=935, x1=1120, y1=988, mode="bottom_band", band_px=12),
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
