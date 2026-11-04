#!/usr/bin/env python3
"""
prototype/scroll_purim_carpet_scenes_builder.py
Offline neural 2.5D spatial asset builder and typed SSOT configuration models for:
  1. Scene 17 (`bene_beitcha_scroll_foot` / `scene17_scroll_foot_*`) [Feedback #40]:
     Tami's museum installation *"מגילה וכף רגל"* (`IMG_MUSE_0256`) with vintage 1966 Purim
     photograph (`IMG_WHAT_0433`) pinned beneath the sculpted plaster foot on the scroll tongue,
     interactive vanishing foot occluder, and sideways-falling black rod revealing the portal.
  2. Scene 18 (`purim_costume_1966` / `scene18_purim_1966_*`) [Feedback #40]:
     Perspective-rectified 1966 childhood Purim photograph (`IMG_WHAT_0433`) of young Tami and
     Ada kneeling in costume with a woven straw basket in a sunlit vintage garden.
  3. Scene 19 (`herzliya_lawn_carpet` / `scene19_herz_carpet_*`) [Feedback #41]:
     Tal and Guy on the sunlit Herzliya lawn (`IMG_HERZ_0399`) fused with Tami's rolled carpet
     and seamlessly extended crumbly earth trail from *"הגלגול הבא"* (`IMG_BINY_0009`), unrolling
     across the foreground grass from lower-left (`x=0` off-image) to lower-right.

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
        "`/Users/talsheaffer/code/tami/.venv/bin/python -u prototype/scroll_purim_carpet_scenes_builder.py`."
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
    extrapolate_depth_outward_edt,
    extract_contact_anchor_mask,
    extract_layer_window_from_expanded,
    solve_contact_anchored_layer_depth,
    verify_layer_contact_and_discontinuity,
)
from prototype.new_scenes_builder import (
    _cosine_feather_2d,
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
    pos: Sequence[tuple[float, float]],
    neg: Sequence[tuple[float, float]],
) -> SamPointGroupPrompt:
    return SamPointGroupPrompt(
        box=BoundingBoxPrompt(*box),
        points_xy=tuple((float(px), float(py)) for px, py in pos)
        + tuple((float(nx), float(ny)) for nx, ny in neg),
        point_labels=tuple([1] * len(pos) + [0] * len(neg)),
    )


# ============================================================================
# Hierarchical, Serializable Configuration Data-Models (SSOT)
# ============================================================================


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
    portrait_initial_yaw_deg: float = 0.0

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
            "portrait_initial_yaw_deg": self.portrait_initial_yaw_deg,
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
            portrait_initial_yaw_deg=float(data["portrait_initial_yaw_deg"]),
        )


@dataclass(frozen=True)
class ScrollFootSceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    asset_prefix: str
    scroll_source_relpath: str
    photo_source_relpath: str
    plate_wh: tuple[int, int]
    inner_dest_xy: tuple[int, int]
    foot_supersample_factor: int
    rod_crop_norm: tuple[float, float, float, float]
    rod_work_width_px: int
    foot_tuck_erode_px: int
    scroll_sam_prompts: tuple[SamPointGroupPrompt, ...]
    rod_foot_sam_prompts: tuple[SamPointGroupPrompt, ...]
    foot_3x_sam_prompts: tuple[SamPointGroupPrompt, ...]
    scroll_rod_heal_dilation_px: int
    scroll_rod_heal_blur_sigma: float
    bg_inpaint_dilation_px: int
    bg_inpaint_blur_sigma: float
    bg_jpeg_quality: int
    baseboard_fit_left_x_range: tuple[int, int]
    baseboard_fit_right_x_range: tuple[int, int]
    baseboard_left_y_window: tuple[int, int]
    baseboard_right_y_window: tuple[int, int]
    baseboard_band_half_px: int
    harmonic_lowpass_sigma_px: float
    harmonic_downsample_factor: int
    grain_highpass_sigma_px: float
    quilt_tile_px: int
    quilt_overlap_px: int
    wall_grain_box_plate: tuple[int, int, int, int]
    floor_grain_box_plate: tuple[int, int, int, int]
    scroll_ambient_shadow_strength: float
    scroll_ambient_shadow_sigma_px: float
    scroll_catenary_peak_y_px: float
    scroll_catenary_sigma_y_px: float
    scroll_catenary_relief_world: float
    photo_src_outer_quad_xy: tuple[tuple[float, float], ...]
    photo_src_glare_heal_box: tuple[int, int, int, int]
    photo_work_wh: tuple[int, int]
    photo_dst_quad_inner_xy: tuple[tuple[float, float], ...]
    photo_contact_shadow_max_alpha_u8: int
    photo_contact_shadow_sigma_px: float
    rod_fall_angle_deg: float
    rod_min_clearance_over_scroll_world: float
    portal_id: str
    target_scene_id: str
    portal_label_he: str
    portal_short_badge_he: str
    occluder_label_he: str
    room_depth_clean: RoomDepthCleanConfig
    mid_depth_solve: ContactAnchoredLayerDepthConfig
    photo_depth_solve: ContactAnchoredLayerDepthConfig
    rod_depth_solve: ContactAnchoredLayerDepthConfig
    occluder_depth_solve: ContactAnchoredLayerDepthConfig
    camera_rig: SceneCameraRigSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "asset_prefix": self.asset_prefix,
            "scroll_source_relpath": self.scroll_source_relpath,
            "photo_source_relpath": self.photo_source_relpath,
            "plate_wh": list(self.plate_wh),
            "inner_dest_xy": list(self.inner_dest_xy),
            "foot_supersample_factor": self.foot_supersample_factor,
            "rod_crop_norm": list(self.rod_crop_norm),
            "rod_work_width_px": self.rod_work_width_px,
            "foot_tuck_erode_px": self.foot_tuck_erode_px,
            "scroll_sam_prompts": [p.to_dict() for p in self.scroll_sam_prompts],
            "rod_foot_sam_prompts": [p.to_dict() for p in self.rod_foot_sam_prompts],
            "foot_3x_sam_prompts": [p.to_dict() for p in self.foot_3x_sam_prompts],
            "scroll_rod_heal_dilation_px": self.scroll_rod_heal_dilation_px,
            "scroll_rod_heal_blur_sigma": self.scroll_rod_heal_blur_sigma,
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "bg_inpaint_blur_sigma": self.bg_inpaint_blur_sigma,
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "baseboard_fit_left_x_range": list(self.baseboard_fit_left_x_range),
            "baseboard_fit_right_x_range": list(self.baseboard_fit_right_x_range),
            "baseboard_left_y_window": list(self.baseboard_left_y_window),
            "baseboard_right_y_window": list(self.baseboard_right_y_window),
            "baseboard_band_half_px": self.baseboard_band_half_px,
            "harmonic_lowpass_sigma_px": self.harmonic_lowpass_sigma_px,
            "harmonic_downsample_factor": self.harmonic_downsample_factor,
            "grain_highpass_sigma_px": self.grain_highpass_sigma_px,
            "quilt_tile_px": self.quilt_tile_px,
            "quilt_overlap_px": self.quilt_overlap_px,
            "wall_grain_box_plate": list(self.wall_grain_box_plate),
            "floor_grain_box_plate": list(self.floor_grain_box_plate),
            "scroll_ambient_shadow_strength": self.scroll_ambient_shadow_strength,
            "scroll_ambient_shadow_sigma_px": self.scroll_ambient_shadow_sigma_px,
            "scroll_catenary_peak_y_px": self.scroll_catenary_peak_y_px,
            "scroll_catenary_sigma_y_px": self.scroll_catenary_sigma_y_px,
            "scroll_catenary_relief_world": self.scroll_catenary_relief_world,
            "photo_src_outer_quad_xy": [list(pt) for pt in self.photo_src_outer_quad_xy],
            "photo_src_glare_heal_box": list(self.photo_src_glare_heal_box),
            "photo_work_wh": list(self.photo_work_wh),
            "photo_dst_quad_inner_xy": [list(pt) for pt in self.photo_dst_quad_inner_xy],
            "photo_contact_shadow_max_alpha_u8": self.photo_contact_shadow_max_alpha_u8,
            "photo_contact_shadow_sigma_px": self.photo_contact_shadow_sigma_px,
            "rod_fall_angle_deg": self.rod_fall_angle_deg,
            "rod_min_clearance_over_scroll_world": self.rod_min_clearance_over_scroll_world,
            "portal_id": self.portal_id,
            "target_scene_id": self.target_scene_id,
            "portal_label_he": self.portal_label_he,
            "portal_short_badge_he": self.portal_short_badge_he,
            "occluder_label_he": self.occluder_label_he,
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "photo_depth_solve": self.photo_depth_solve.to_dict(),
            "rod_depth_solve": self.rod_depth_solve.to_dict(),
            "occluder_depth_solve": self.occluder_depth_solve.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScrollFootSceneSpec:
        pw = data["plate_wh"]
        idx = data["inner_dest_xy"]
        rcn = data["rod_crop_norm"]
        blx = data["baseboard_fit_left_x_range"]
        brx = data["baseboard_fit_right_x_range"]
        bly = data["baseboard_left_y_window"]
        bry = data["baseboard_right_y_window"]
        wgb = data["wall_grain_box_plate"]
        fgb = data["floor_grain_box_plate"]
        pgh = data["photo_src_glare_heal_box"]
        pww = data["photo_work_wh"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            asset_prefix=str(data["asset_prefix"]),
            scroll_source_relpath=str(data["scroll_source_relpath"]),
            photo_source_relpath=str(data["photo_source_relpath"]),
            plate_wh=(int(pw[0]), int(pw[1])),
            inner_dest_xy=(int(idx[0]), int(idx[1])),
            foot_supersample_factor=int(data["foot_supersample_factor"]),
            rod_crop_norm=(float(rcn[0]), float(rcn[1]), float(rcn[2]), float(rcn[3])),
            rod_work_width_px=int(data["rod_work_width_px"]),
            foot_tuck_erode_px=int(data["foot_tuck_erode_px"]),
            scroll_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["scroll_sam_prompts"]),
            rod_foot_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["rod_foot_sam_prompts"]),
            foot_3x_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["foot_3x_sam_prompts"]),
            scroll_rod_heal_dilation_px=int(data["scroll_rod_heal_dilation_px"]),
            scroll_rod_heal_blur_sigma=float(data["scroll_rod_heal_blur_sigma"]),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            bg_inpaint_blur_sigma=float(data["bg_inpaint_blur_sigma"]),
            bg_jpeg_quality=int(data["bg_jpeg_quality"]),
            baseboard_fit_left_x_range=(int(blx[0]), int(blx[1])),
            baseboard_fit_right_x_range=(int(brx[0]), int(brx[1])),
            baseboard_left_y_window=(int(bly[0]), int(bly[1])),
            baseboard_right_y_window=(int(bry[0]), int(bry[1])),
            baseboard_band_half_px=int(data["baseboard_band_half_px"]),
            harmonic_lowpass_sigma_px=float(data["harmonic_lowpass_sigma_px"]),
            harmonic_downsample_factor=int(data["harmonic_downsample_factor"]),
            grain_highpass_sigma_px=float(data["grain_highpass_sigma_px"]),
            quilt_tile_px=int(data["quilt_tile_px"]),
            quilt_overlap_px=int(data["quilt_overlap_px"]),
            wall_grain_box_plate=(int(wgb[0]), int(wgb[1]), int(wgb[2]), int(wgb[3])),
            floor_grain_box_plate=(int(fgb[0]), int(fgb[1]), int(fgb[2]), int(fgb[3])),
            scroll_ambient_shadow_strength=float(data["scroll_ambient_shadow_strength"]),
            scroll_ambient_shadow_sigma_px=float(data["scroll_ambient_shadow_sigma_px"]),
            scroll_catenary_peak_y_px=float(data["scroll_catenary_peak_y_px"]),
            scroll_catenary_sigma_y_px=float(data["scroll_catenary_sigma_y_px"]),
            scroll_catenary_relief_world=float(data["scroll_catenary_relief_world"]),
            photo_src_outer_quad_xy=tuple((float(pt[0]), float(pt[1])) for pt in data["photo_src_outer_quad_xy"]),
            photo_src_glare_heal_box=(int(pgh[0]), int(pgh[1]), int(pgh[2]), int(pgh[3])),
            photo_work_wh=(int(pww[0]), int(pww[1])),
            photo_dst_quad_inner_xy=tuple((float(pt[0]), float(pt[1])) for pt in data["photo_dst_quad_inner_xy"]),
            photo_contact_shadow_max_alpha_u8=int(data["photo_contact_shadow_max_alpha_u8"]),
            photo_contact_shadow_sigma_px=float(data["photo_contact_shadow_sigma_px"]),
            rod_fall_angle_deg=float(data["rod_fall_angle_deg"]),
            rod_min_clearance_over_scroll_world=float(data["rod_min_clearance_over_scroll_world"]),
            portal_id=str(data["portal_id"]),
            target_scene_id=str(data["target_scene_id"]),
            portal_label_he=str(data["portal_label_he"]),
            portal_short_badge_he=str(data["portal_short_badge_he"]),
            occluder_label_he=str(data["occluder_label_he"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            mid_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_depth_solve"]),
            photo_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["photo_depth_solve"]),
            rod_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["rod_depth_solve"]),
            occluder_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["occluder_depth_solve"]),
            camera_rig=SceneCameraRigSpec.from_dict(data["camera_rig"]),
        )


@dataclass(frozen=True)
class PurimCostumeSceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    asset_prefix: str
    source_relpath: str
    src_inner_quad_xy: tuple[tuple[float, float], ...]
    rectified_full_wh: tuple[int, int]
    rectified_crop_y_range: tuple[int, int]
    inner_dest_xy: tuple[int, int]
    glare_heal_box_rect: tuple[int, int, int, int]
    mid_sam_prompts: tuple[SamPointGroupPrompt, ...]
    sam_support_dilate_px: int
    sam_core_erode_px: int
    min_component_area_px: int
    bg_inpaint_dilation_px: int
    bg_inpaint_blur_sigma: float
    mirror_inset_px: int
    glare_dehaze_width_px: int
    glare_dehaze_y_center_px: float
    glare_dehaze_y_scale_px: float
    glare_dehaze_strength: float
    bg_jpeg_quality: int
    room_depth_clean: RoomDepthCleanConfig
    mid_depth_solve: ContactAnchoredLayerDepthConfig
    camera_rig: SceneCameraRigSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "asset_prefix": self.asset_prefix,
            "source_relpath": self.source_relpath,
            "src_inner_quad_xy": [list(pt) for pt in self.src_inner_quad_xy],
            "rectified_full_wh": list(self.rectified_full_wh),
            "rectified_crop_y_range": list(self.rectified_crop_y_range),
            "inner_dest_xy": list(self.inner_dest_xy),
            "glare_heal_box_rect": list(self.glare_heal_box_rect),
            "mid_sam_prompts": [p.to_dict() for p in self.mid_sam_prompts],
            "sam_support_dilate_px": self.sam_support_dilate_px,
            "sam_core_erode_px": self.sam_core_erode_px,
            "min_component_area_px": self.min_component_area_px,
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "bg_inpaint_blur_sigma": self.bg_inpaint_blur_sigma,
            "mirror_inset_px": self.mirror_inset_px,
            "glare_dehaze_width_px": self.glare_dehaze_width_px,
            "glare_dehaze_y_center_px": self.glare_dehaze_y_center_px,
            "glare_dehaze_y_scale_px": self.glare_dehaze_y_scale_px,
            "glare_dehaze_strength": self.glare_dehaze_strength,
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PurimCostumeSceneSpec:
        rfw = data["rectified_full_wh"]
        rcy = data["rectified_crop_y_range"]
        idx = data["inner_dest_xy"]
        ghb = data["glare_heal_box_rect"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            asset_prefix=str(data["asset_prefix"]),
            source_relpath=str(data["source_relpath"]),
            src_inner_quad_xy=tuple((float(pt[0]), float(pt[1])) for pt in data["src_inner_quad_xy"]),
            rectified_full_wh=(int(rfw[0]), int(rfw[1])),
            rectified_crop_y_range=(int(rcy[0]), int(rcy[1])),
            inner_dest_xy=(int(idx[0]), int(idx[1])),
            glare_heal_box_rect=(int(ghb[0]), int(ghb[1]), int(ghb[2]), int(ghb[3])),
            mid_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["mid_sam_prompts"]),
            sam_support_dilate_px=int(data["sam_support_dilate_px"]),
            sam_core_erode_px=int(data["sam_core_erode_px"]),
            min_component_area_px=int(data["min_component_area_px"]),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            bg_inpaint_blur_sigma=float(data["bg_inpaint_blur_sigma"]),
            mirror_inset_px=int(data["mirror_inset_px"]),
            glare_dehaze_width_px=int(data["glare_dehaze_width_px"]),
            glare_dehaze_y_center_px=float(data["glare_dehaze_y_center_px"]),
            glare_dehaze_y_scale_px=float(data["glare_dehaze_y_scale_px"]),
            glare_dehaze_strength=float(data["glare_dehaze_strength"]),
            bg_jpeg_quality=int(data["bg_jpeg_quality"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            mid_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_depth_solve"]),
            camera_rig=SceneCameraRigSpec.from_dict(data["camera_rig"]),
        )


@dataclass(frozen=True)
class HerzliyaCarpetSceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    asset_prefix: str
    lawn_source_relpath: str
    carpet_source_relpath: str
    tal_split_x_px: int
    tal_sam_prompts: tuple[SamPointGroupPrompt, ...]
    guy_sam_prompts: tuple[SamPointGroupPrompt, ...]
    sam_support_dilate_px: int
    sam_core_erode_px: int
    min_component_area_px: int
    hose_heal_box_inner: tuple[int, int, int, int]
    bg_inpaint_dilation_px: int
    bg_inpaint_blur_sigma: float
    bg_jpeg_quality: int
    carpet_sam_prompts: tuple[SamPointGroupPrompt, ...]
    earth_sam_prompts: tuple[SamPointGroupPrompt, ...]
    carpet_scale: float
    start_center_x_px: float
    end_center_x_px: float
    roll_bottom_y_px: int
    earth_end_x_px: int
    earth_top_y_px: int
    earth_bottom_y_px: int
    soil_src_core_box: tuple[int, int, int, int]
    soil_quilt_tile_w: int
    soil_quilt_overlap_px: int
    soil_seam_heal_half_px: int
    outdoor_rgb_gain: tuple[float, float, float]
    grass_contact_shadow_max_alpha_u8: int
    grass_contact_shadow_sigma_px: float
    room_depth_clean: RoomDepthCleanConfig
    mid_depth_solve: ContactAnchoredLayerDepthConfig
    earth_depth_solve: ContactAnchoredLayerDepthConfig
    roll_depth_solve: ContactAnchoredLayerDepthConfig
    camera_rig: SceneCameraRigSpec
    reflect_left_mirror: bool = True
    unroll_duration_ms: int = 1450

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "asset_prefix": self.asset_prefix,
            "lawn_source_relpath": self.lawn_source_relpath,
            "carpet_source_relpath": self.carpet_source_relpath,
            "tal_split_x_px": self.tal_split_x_px,
            "tal_sam_prompts": [p.to_dict() for p in self.tal_sam_prompts],
            "guy_sam_prompts": [p.to_dict() for p in self.guy_sam_prompts],
            "sam_support_dilate_px": self.sam_support_dilate_px,
            "sam_core_erode_px": self.sam_core_erode_px,
            "min_component_area_px": self.min_component_area_px,
            "hose_heal_box_inner": list(self.hose_heal_box_inner),
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "bg_inpaint_blur_sigma": self.bg_inpaint_blur_sigma,
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "carpet_sam_prompts": [p.to_dict() for p in self.carpet_sam_prompts],
            "earth_sam_prompts": [p.to_dict() for p in self.earth_sam_prompts],
            "carpet_scale": self.carpet_scale,
            "start_center_x_px": self.start_center_x_px,
            "end_center_x_px": self.end_center_x_px,
            "roll_bottom_y_px": self.roll_bottom_y_px,
            "earth_end_x_px": self.earth_end_x_px,
            "earth_top_y_px": self.earth_top_y_px,
            "earth_bottom_y_px": self.earth_bottom_y_px,
            "soil_src_core_box": list(self.soil_src_core_box),
            "soil_quilt_tile_w": self.soil_quilt_tile_w,
            "soil_quilt_overlap_px": self.soil_quilt_overlap_px,
            "soil_seam_heal_half_px": self.soil_seam_heal_half_px,
            "outdoor_rgb_gain": list(self.outdoor_rgb_gain),
            "grass_contact_shadow_max_alpha_u8": self.grass_contact_shadow_max_alpha_u8,
            "grass_contact_shadow_sigma_px": self.grass_contact_shadow_sigma_px,
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "earth_depth_solve": self.earth_depth_solve.to_dict(),
            "roll_depth_solve": self.roll_depth_solve.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
            "reflect_left_mirror": self.reflect_left_mirror,
            "unroll_duration_ms": self.unroll_duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HerzliyaCarpetSceneSpec:
        hhb = data["hose_heal_box_inner"]
        ssc = data["soil_src_core_box"]
        org = data["outdoor_rgb_gain"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            asset_prefix=str(data["asset_prefix"]),
            lawn_source_relpath=str(data["lawn_source_relpath"]),
            carpet_source_relpath=str(data["carpet_source_relpath"]),
            tal_split_x_px=int(data["tal_split_x_px"]),
            tal_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["tal_sam_prompts"]),
            guy_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["guy_sam_prompts"]),
            sam_support_dilate_px=int(data["sam_support_dilate_px"]),
            sam_core_erode_px=int(data["sam_core_erode_px"]),
            min_component_area_px=int(data["min_component_area_px"]),
            hose_heal_box_inner=(int(hhb[0]), int(hhb[1]), int(hhb[2]), int(hhb[3])),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            bg_inpaint_blur_sigma=float(data["bg_inpaint_blur_sigma"]),
            bg_jpeg_quality=int(data["bg_jpeg_quality"]),
            carpet_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["carpet_sam_prompts"]),
            earth_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["earth_sam_prompts"]),
            carpet_scale=float(data["carpet_scale"]),
            start_center_x_px=float(data["start_center_x_px"]),
            end_center_x_px=float(data["end_center_x_px"]),
            roll_bottom_y_px=int(data["roll_bottom_y_px"]),
            earth_end_x_px=int(data["earth_end_x_px"]),
            earth_top_y_px=int(data["earth_top_y_px"]),
            earth_bottom_y_px=int(data["earth_bottom_y_px"]),
            soil_src_core_box=(int(ssc[0]), int(ssc[1]), int(ssc[2]), int(ssc[3])),
            soil_quilt_tile_w=int(data["soil_quilt_tile_w"]),
            soil_quilt_overlap_px=int(data["soil_quilt_overlap_px"]),
            soil_seam_heal_half_px=int(data["soil_seam_heal_half_px"]),
            outdoor_rgb_gain=(float(org[0]), float(org[1]), float(org[2])),
            grass_contact_shadow_max_alpha_u8=int(data["grass_contact_shadow_max_alpha_u8"]),
            grass_contact_shadow_sigma_px=float(data["grass_contact_shadow_sigma_px"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            mid_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_depth_solve"]),
            earth_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["earth_depth_solve"]),
            roll_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["roll_depth_solve"]),
            camera_rig=SceneCameraRigSpec.from_dict(data["camera_rig"]),
            reflect_left_mirror=bool(data["reflect_left_mirror"]),
            unroll_duration_ms=int(data["unroll_duration_ms"]),
        )


@dataclass(frozen=True)
class ScrollPurimCarpetBundleConfig:
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
    bene_beitcha_scroll_foot: ScrollFootSceneSpec
    purim_costume_1966: PurimCostumeSceneSpec
    herzliya_lawn_carpet: HerzliyaCarpetSceneSpec

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
            "bene_beitcha_scroll_foot": self.bene_beitcha_scroll_foot.to_dict(),
            "purim_costume_1966": self.purim_costume_1966.to_dict(),
            "herzliya_lawn_carpet": self.herzliya_lawn_carpet.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScrollPurimCarpetBundleConfig:
        off = data["inner_offset_xy"]
        return cls(
            inner_dimensions=CanvasDimensionsConfig.from_dict(data["inner_dimensions"]),
            expanded_dimensions=CanvasDimensionsConfig.from_dict(data["expanded_dimensions"]),
            inner_offset_xy=(int(off[0]), int(off[1])),
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
            bene_beitcha_scroll_foot=ScrollFootSceneSpec.from_dict(
                data["bene_beitcha_scroll_foot"]
            ),
            purim_costume_1966=PurimCostumeSceneSpec.from_dict(data["purim_costume_1966"]),
            herzliya_lawn_carpet=HerzliyaCarpetSceneSpec.from_dict(
                data["herzliya_lawn_carpet"]
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> ScrollPurimCarpetBundleConfig:
        return cls.from_dict(json.loads(raw))


# ============================================================================
# Shared Vision, Outpainting, & Depth Topology Verification Primitives
# ============================================================================


def _assert_border_alpha_contract(
    layer_rgba: Image.Image,
    label: str,
    allow_left_bleed: bool = False,
) -> None:
    alpha = np.asarray(layer_rgba.convert("RGBA"), dtype=np.uint8)[:, :, 3]
    edges = [
        int(alpha[:, -1].max()),
        int(alpha[0, :].max()),
        int(alpha[-1, :].max()),
    ]
    if not allow_left_bleed:
        edges.append(int(alpha[:, 0].max()))
    max_edge = max(edges)
    if max_edge > 0:
        raise RuntimeError(f"{label} violated border-alpha contract (max_edge={max_edge}).")


def _compute_edt_rim_p99_jump_world(
    rgba_u8: np.ndarray,
    depth_u8: np.ndarray,
    layer_amplitude: float,
    alpha_threshold_u8: int,
    rim_probe_px: int,
) -> float:
    opaque = rgba_u8[:, :, 3] >= alpha_threshold_u8
    if not np.any(opaque) or bool(np.all(opaque)):
        return 0.0
    dist_out, nearest_in = ndi.distance_transform_edt(~opaque, return_indices=True)
    rim = (~opaque) & (dist_out <= float(rim_probe_px))
    if not np.any(rim):
        return 0.0
    ry, rx = np.where(rim)
    ny, nx = nearest_in[0, ry, rx], nearest_in[1, ry, rx]
    dz = (
        np.abs(depth_u8[ry, rx].astype(np.float32) - depth_u8[ny, nx].astype(np.float32))
        / 255.0
        * float(layer_amplitude)
    )
    return float(np.percentile(dz, 99.0))


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
    if not report.contact_passed or not report.discontinuity_passed or not report.monotonicity_passed:
        raise RuntimeError(f"{label} failed depth topology verification: {report.to_json()}")
    if p99_rim_jump > max_edt_rim_jump_world:
        raise RuntimeError(
            f"{label} failed outward EDT rim gate: p99={p99_rim_jump:.3f} > {max_edt_rim_jump_world:.3f}"
        )
    out = report.to_dict()
    out["liveGateP99RimJumpWorld"] = round(p99_rim_jump, 4)
    return out


def _prune_unanchored_cutout_islands(
    layer_rgba: Image.Image,
    solve_cfg: ContactAnchoredLayerDepthConfig,
) -> Image.Image:
    arr = np.asarray(layer_rgba.convert("RGBA"), dtype=np.uint8).copy()
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

    return Image.fromarray(arr, mode="RGBA")


def _segment_prompt_union_u8(
    rgb_image: Image.Image,
    prompts: Sequence[SamPointGroupPrompt],
    sam_extractor: SamBoxMaskExtractor,
    min_component_area_px: int,
    max_pinhole_fill_px: int = 1800,
) -> np.ndarray:
    w, h = rgb_image.size
    union_u8 = np.zeros((h, w), dtype=np.uint8)
    for prompt in prompts:
        mask_u8 = sam_extractor.segment_point_groups(
            rgb_image=rgb_image,
            point_groups=(prompt,),
            exclude_boxes=(),
            max_pinhole_fill_px=max_pinhole_fill_px,
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


def _keep_largest_cc_u8(mask_u8: np.ndarray, thresh: int = 96) -> np.ndarray:
    nl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 >= thresh).astype(np.uint8), connectivity=8
    )
    if nl <= 1:
        return mask_u8
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    support = cv2.dilate(
        (lbl == largest).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return np.where(support > 0, mask_u8, 0).astype(np.uint8)


def _tight_alpha_bbox(rgba_u8: np.ndarray, alpha_thresh: int = 20) -> list[int]:
    ys, xs = np.where(rgba_u8[:, :, 3] >= alpha_thresh)
    if len(xs) == 0:
        raise ValueError("Cannot compute tight bounding box on empty RGBA layer.")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _quilt_nonperiodic_grain_field(
    grain_patch_f32: np.ndarray,
    target_wh: tuple[int, int],
    tile_px: int,
    overlap_px: int,
    seed: int,
) -> np.ndarray:
    tw, th = target_wh
    ph, pw, _ = grain_patch_f32.shape
    t_sz = min(tile_px, pw - 4, ph - 4)
    step = max(24, t_sz - overlap_px)
    acc = np.zeros((th, tw, 3), dtype=np.float32)
    w_acc = np.zeros((th, tw, 1), dtype=np.float32)
    win = np.maximum(
        _cosine_feather_2d(
            height=t_sz,
            width=t_sz,
            left_px=overlap_px,
            top_px=overlap_px,
            right_px=overlap_px,
            bottom_px=overlap_px,
            elliptical=False,
        )[:, :, None],
        1e-3,
    )
    variants = (
        grain_patch_f32,
        grain_patch_f32[:, ::-1, :],
        grain_patch_f32[::-1, :, :],
        grain_patch_f32[::-1, ::-1, :],
    )
    rng = np.random.default_rng(seed)
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


def _min_error_vertical_cut_mask(
    left_ov_f32: np.ndarray, right_ov_f32: np.ndarray, feather_px: float = 4.0
) -> np.ndarray:
    """
    Efros-Freeman dynamic-programming minimum-error vertical seam mask in `[0, 1]`,
    where `0.0` selects `left_ov_f32` and `1.0` selects `right_ov_f32` (`shape = (H, W, 1)`).
    """
    h_ov, w_ov, _ = left_ov_f32.shape
    if w_ov <= 2 or h_ov <= 1:
        return np.linspace(0.0, 1.0, max(1, w_ov), dtype=np.float32)[None, :, None]
    err = np.mean((left_ov_f32 - right_ov_f32) ** 2, axis=2)
    dp = err.copy()
    for y in range(1, h_ov):
        prev = dp[y - 1]
        best = prev.copy()
        best[1:] = np.minimum(best[1:], prev[:-1])
        best[:-1] = np.minimum(best[:-1], prev[1:])
        dp[y] += best
    seam_x = np.zeros(h_ov, dtype=np.int32)
    seam_x[-1] = int(np.argmin(dp[-1]))
    for y in range(h_ov - 2, -1, -1):
        xp = int(seam_x[y + 1])
        lo = max(0, xp - 1)
        hi = min(w_ov, xp + 2)
        seam_x[y] = lo + int(np.argmin(dp[y, lo:hi]))
    xx = np.arange(w_ov, dtype=np.float32)[None, :]
    w_right = np.clip((xx - seam_x[:, None].astype(np.float32)) / max(1.0, feather_px) + 0.5, 0.0, 1.0)
    return w_right[:, :, None]


def _mirror_reflect_rect_to_canvas(
    core_arr: np.ndarray,
    target_wh: tuple[int, int],
    dest_xy: tuple[int, int],
) -> np.ndarray:
    """
    Expands an authentic rectangular plate or depth map `core_arr` (`ch x cw [x C]`)
    placed at `dest_xy = (dx, dy)` onto `target_wh = (tw, th)` using exact analytic
    optical mirror reflection (`cv2.BORDER_REFLECT_101`, period `2 * (N - 1)`),
    preserving 100% crisp native pixels inside the core and $C^0$ continuity across
    every mirror fold line without synthetic outpainting or collar blurring.
    """
    tw, th = target_wh
    ch, cw = core_arr.shape[:2]
    if cw < 2 or ch < 2:
        raise ValueError(f"core_arr must be at least 2x2 for mirror reflection, got {(cw, ch)}.")
    dx, dy = int(dest_xy[0]), int(dest_xy[1])
    qx = np.arange(tw, dtype=np.int64) - dx
    px = 2 * (cw - 1)
    rx = np.mod(qx, px)
    ix = np.where(rx < cw, rx, px - rx)

    qy = np.arange(th, dtype=np.int64) - dy
    py = 2 * (ch - 1)
    ry = np.mod(qy, py)
    iy = np.where(ry < ch, ry, py - ry)
    return np.ascontiguousarray(core_arr[np.ix_(iy, ix)])


def _clean_and_dehaze_purim_garden_core(
    clean_full_garden_np: np.ndarray,
    spec: PurimCostumeSceneSpec,
) -> np.ndarray:
    """
    Crops `spec.mirror_inset_px` inside the rectified `1008 x 1400` 1966 garden plate
    to exclude raw paper-edge specks and gently dehazes the lower-left phone-flash glare
    against the clean right garden edge so mirror folds have uniform vintage illumination.
    """
    rh, rw, _ = clean_full_garden_np.shape
    inset = spec.mirror_inset_px
    core_f = clean_full_garden_np[inset : rh - inset, inset : rw - inset].astype(np.float32)
    ch, cw, _ = core_f.shape

    dh_w = min(spec.glare_dehaze_width_px, cw // 3)
    ref_w = min(150, cw // 4)
    ref_edge = core_f[:, cw - ref_w - 4 : cw - 4]
    donor = core_f[:, :dh_w]
    sigma_y = 56.0
    ref_mu = ndi.gaussian_filter(np.mean(ref_edge, axis=1, keepdims=True), sigma=(sigma_y, 0.0, 0.0))
    don_mu = ndi.gaussian_filter(np.mean(donor, axis=1, keepdims=True), sigma=(sigma_y, 0.0, 0.0))
    ref_var = ndi.gaussian_filter(
        np.mean((ref_edge - ref_mu) ** 2, axis=1, keepdims=True), sigma=(sigma_y, 0.0, 0.0)
    )
    don_var = ndi.gaussian_filter(
        np.mean((donor - don_mu) ** 2, axis=1, keepdims=True), sigma=(sigma_y, 0.0, 0.0)
    )
    scale_std = np.clip(np.sqrt((ref_var + 16.0) / (don_var + 16.0)), 0.72, 1.48)
    dehazed_left = np.clip((donor - don_mu) * scale_std + ref_mu, 0.0, 255.0)

    wx_haze = (
        0.5 + 0.5 * np.cos(np.pi * np.clip(np.arange(dh_w, dtype=np.float32) / float(dh_w), 0.0, 1.0))
    )[None, :, None]
    wy_haze = (
        0.5
        * (
            1.0
            + np.tanh(
                (np.arange(ch, dtype=np.float32) - spec.glare_dehaze_y_center_px)
                / spec.glare_dehaze_y_scale_px
            )
        )
    )[:, None, None]
    w_dehaze = spec.glare_dehaze_strength * wx_haze * wy_haze
    core_f[:, :dh_w] = (1.0 - w_dehaze) * donor + w_dehaze * dehazed_left
    return np.clip(np.rint(core_f), 0, 255).astype(np.uint8)


# ============================================================================
# Task 1A: Scene 17 Builder — `bene_beitcha_scroll_foot` (`scene17_scroll_foot_*`)
# ============================================================================


def _clean_scroll_museum_plate(
    plate_rgb_u8: np.ndarray,
    occluder_union_plate_u8: np.ndarray,
    spec: ScrollFootSceneSpec,
    lama_inpainter: LamaOcclusionInpainter,
    max_working_side_px: int,
    roi_margin_px: int,
) -> np.ndarray:
    """
    Cleans the scroll, black rod, and sculpted foot (`occluder_union_plate_u8`) out of
    the authentic `667 x 1000` museum plate (`plate_rgb_u8`) using perspective baseboard
    continuity, 2D harmonic wall/floor illumination, authentic micro-grain, and LaMa
    mid-frequency detail strictly inside the occluded hole while preserving 100% native
    gallery pixels everywhere outside the hole up to all 4 authentic plate borders.
    """
    ph, pw, _ = plate_rgb_u8.shape
    plate_f = plate_rgb_u8.astype(np.float32)
    luma = 0.299 * plate_f[:, :, 0] + 0.587 * plate_f[:, :, 1] + 0.114 * plate_f[:, :, 2]

    # 1. Fit exact colinear perspective baseboard line y_plate = intercept + slope * x_plate
    xs_fit: list[float] = []
    ys_fit: list[float] = []
    lx0, lx1 = spec.baseboard_fit_left_x_range
    rx0, rx1 = spec.baseboard_fit_right_x_range
    ly0, ly1 = spec.baseboard_left_y_window
    ry0, ry1 = spec.baseboard_right_y_window
    for x in range(lx0, lx1, 4):
        yb = ly0 + int(np.argmin(luma[ly0:ly1, x]))
        xs_fit.append(float(x))
        ys_fit.append(float(yb))
    for x in range(rx0, rx1, 4):
        yb = ry0 + int(np.argmin(luma[ry0:ry1, x]))
        xs_fit.append(float(x))
        ys_fit.append(float(yb))
    slope, intercept = np.polyfit(
        np.asarray(xs_fit, dtype=np.float64),
        np.asarray(ys_fit, dtype=np.float64),
        1,
    )

    xx_plate = np.arange(pw, dtype=np.float32)[None, :]
    yy_plate = np.arange(ph, dtype=np.float32)[:, None]
    y_base_plate = float(intercept) + float(slope) * xx_plate

    occ_dil = (
        cv2.dilate(
            (occluder_union_plate_u8 >= 16).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
        )
        > 0
    )
    bh = spec.baseboard_band_half_px
    clean_wall_plate = (~occ_dil) & (yy_plate < (y_base_plate - float(bh)))
    clean_floor_plate = (~occ_dil) & (yy_plate > (y_base_plate + float(bh)))

    # 2. Solve separate 2D harmonic fields for museum wall and concrete floor on 667x1000
    def _solve_surface_field_plate(clean_mask_p: np.ndarray) -> np.ndarray:
        lp_sig = spec.harmonic_lowpass_sigma_px
        w_c = np.maximum(1e-4, ndi.gaussian_filter(clean_mask_p.astype(np.float32), sigma=lp_sig))
        low_p = np.stack(
            [
                ndi.gaussian_filter(plate_f[:, :, c] * clean_mask_p, sigma=lp_sig) / w_c
                for c in range(3)
            ],
            axis=2,
        )
        cy_idx, cx_idx = np.where(clean_mask_p)
        cy_s, cx_s = cy_idx[::4], cx_idx[::4]
        tx_pts = np.tanh(((cx_s.astype(np.float32) - 0.5 * pw) / (0.5 * pw)) * 1.15)
        ty_pts = np.tanh(((cy_s.astype(np.float32) - 0.5 * ph) / (0.5 * ph)) * 1.15)
        design = np.column_stack([np.ones_like(tx_pts), tx_pts, ty_pts])

        tx_full = np.tanh(((np.arange(pw, dtype=np.float32) - 0.5 * pw) / (0.5 * pw)) * 1.15)[None, :]
        ty_full = np.tanh(((np.arange(ph, dtype=np.float32) - 0.5 * ph) / (0.5 * ph)) * 1.15)[:, None]
        dir_vals = np.zeros((ph, pw, 3), dtype=np.float32)
        for ch_i in range(3):
            vals = plate_f[cy_s, cx_s, ch_i]
            lo_b = float(np.percentile(vals, 3.0)) * 0.96
            hi_b = float(np.percentile(vals, 97.0)) * 1.03
            coeffs, _, _, _ = np.linalg.lstsq(design, vals, rcond=None)
            pred = coeffs[0] + coeffs[1] * tx_full + coeffs[2] * ty_full
            dir_vals[:, :, ch_i] = np.clip(pred, lo_b, hi_b)

        dir_vals = np.where(clean_mask_p[:, :, None], low_p, dir_vals)
        dir_mask = clean_mask_p.copy()
        dir_mask[:4, :] = True
        dir_mask[-4:, :] = True
        dir_mask[:, :4] = True
        dir_mask[:, -4:] = True

        ds = max(1, spec.harmonic_downsample_factor)
        gh, gw = max(8, ph // ds), max(8, pw // ds)
        dom_ds = np.ones((gh, gw), dtype=bool)
        dmask_ds = cv2.resize(dir_mask.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) > 0
        out = np.zeros((ph, pw, 3), dtype=np.float32)
        for ch_i in range(3):
            dv_ds = cv2.resize(dir_vals[:, :, ch_i], (gw, gh), interpolation=cv2.INTER_AREA).astype(
                np.float64
            )
            sol = _solve_harmonic_field_on_domain(dom_ds, dmask_ds, dv_ds).astype(np.float32)
            out[:, :, ch_i] = cv2.resize(sol, (pw, ph), interpolation=cv2.INTER_CUBIC)
        return out

    wall_harm = _solve_surface_field_plate(clean_wall_plate)
    floor_harm = _solve_surface_field_plate(clean_floor_plate)

    # 3. Sample authentic vertical baseboard residual profile [-bh..+bh] from left & right strips
    offsets = np.arange(-bh, bh + 1, dtype=np.float32)
    prof_left = np.zeros((len(offsets), 3), dtype=np.float32)
    prof_right = np.zeros((len(offsets), 3), dtype=np.float32)
    for idx_o, dy_o in enumerate(offsets):
        l_samples = []
        for x in range(lx0 + 8, lx1 - 8, 2):
            yc = int(round(float(intercept) + float(slope) * x + float(dy_o)))
            y_ref_w = int(round(float(intercept) + float(slope) * x - float(bh + 6)))
            y_ref_f = int(round(float(intercept) + float(slope) * x + float(bh + 6)))
            ref = plate_f[y_ref_w, x] if dy_o < 0 else plate_f[y_ref_f, x]
            l_samples.append(plate_f[yc, x] - ref)
        prof_left[idx_o] = np.median(np.stack(l_samples, axis=0), axis=0)

        r_samples = []
        for x in range(rx0 + 8, rx1 - 8, 2):
            yc = int(round(float(intercept) + float(slope) * x + float(dy_o)))
            y_ref_w = int(round(float(intercept) + float(slope) * x - float(bh + 6)))
            y_ref_f = int(round(float(intercept) + float(slope) * x + float(bh + 6)))
            ref = plate_f[y_ref_w, x] if dy_o < 0 else plate_f[y_ref_f, x]
            r_samples.append(plate_f[yc, x] - ref)
        prof_right[idx_o] = np.median(np.stack(r_samples, axis=0), axis=0)

    rel_y_p = yy_plate - y_base_plate
    floor_weight = 0.5 * (1.0 + np.tanh(rel_y_p / 1.35))[:, :, None]
    base_harm = (1.0 - floor_weight) * wall_harm + floor_weight * floor_harm

    lr_blend = np.clip((xx_plate - float(lx1)) / max(1.0, float(rx0 - lx1)), 0.0, 1.0)
    in_band = np.abs(rel_y_p) <= float(bh)
    if np.any(in_band):
        by, bx = np.where(in_band)
        frac_idx = np.clip(rel_y_p[by, bx] + float(bh), 0.0, float(2 * bh))
        i0 = np.floor(frac_idx).astype(np.int32)
        i1 = np.minimum(2 * bh, i0 + 1)
        wf = (frac_idx - i0.astype(np.float32))[:, None]
        w_lr = lr_blend[0, bx][:, None]
        p_l = (1.0 - wf) * prof_left[i0] + wf * prof_left[i1]
        p_r = (1.0 - wf) * prof_right[i0] + wf * prof_right[i1]
        taper_b = np.cos(0.5 * np.pi * np.clip(np.abs(rel_y_p[by, bx]) / float(bh + 1), 0.0, 1.0))[
            :, None
        ]
        base_harm[by, bx] += taper_b * ((1.0 - w_lr) * p_l + w_lr * p_r)

    wx0, wy0, wx1, wy1 = spec.wall_grain_box_plate
    fx0, fy0, fx1, fy1 = spec.floor_grain_box_plate
    wall_patch = plate_f[wy0:wy1, wx0:wx1]
    wall_grain_patch = wall_patch - ndi.gaussian_filter(
        wall_patch, sigma=(spec.grain_highpass_sigma_px, spec.grain_highpass_sigma_px, 0.0)
    )
    floor_patch = plate_f[fy0:fy1, fx0:fx1]
    floor_grain_patch = floor_patch - ndi.gaussian_filter(
        floor_patch, sigma=(spec.grain_highpass_sigma_px, spec.grain_highpass_sigma_px, 0.0)
    )
    wall_grain_p = _quilt_nonperiodic_grain_field(
        wall_grain_patch, (pw, ph), spec.quilt_tile_px, spec.quilt_overlap_px, seed=7017
    )
    floor_grain_p = _quilt_nonperiodic_grain_field(
        floor_grain_patch, (pw, ph), spec.quilt_tile_px, spec.quilt_overlap_px, seed=7018
    )
    grain_band_suppress = 1.0 - np.exp(-0.5 * (rel_y_p / 3.0) ** 2)[:, :, None]
    synth_plate_f = np.clip(
        base_harm
        + grain_band_suppress * ((1.0 - floor_weight) * wall_grain_p + floor_weight * floor_grain_p),
        0.0,
        255.0,
    )

    occ_plate = np.where(occluder_union_plate_u8 >= 16, 255, 0).astype(np.uint8)
    lama_raw = _inpaint_masked_plate_lama(
        src_rgb=Image.fromarray(plate_rgb_u8, mode="RGB"),
        binary_mask_u8=occ_plate,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.bg_inpaint_dilation_px,
        blur_sigma=spec.bg_inpaint_blur_sigma,
        max_working_side_px=max_working_side_px,
        roi_context_margin_px=roi_margin_px,
    )
    lama_f = np.asarray(lama_raw, dtype=np.float32)
    lama_low = ndi.gaussian_filter(lama_f, sigma=(16.0, 16.0, 0.0))
    lama_midfreq = np.clip(lama_f - lama_low, -14.0, 14.0) * grain_band_suppress

    hole_dil = cv2.dilate(
        occ_plate,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * spec.bg_inpaint_dilation_px + 1, 2 * spec.bg_inpaint_dilation_px + 1),
        ),
    )
    hole_w = ndi.gaussian_filter((hole_dil > 0).astype(np.float32), sigma=9.0)[:, :, None]
    clean_bg_hole_f = np.clip(synth_plate_f + 0.55 * lama_midfreq, 0.0, 255.0)
    bg_clean_f = (1.0 - hole_w) * plate_f + hole_w * clean_bg_hole_f

    scroll_soft = ndi.gaussian_filter(
        (occ_plate > 0).astype(np.float32), sigma=spec.scroll_ambient_shadow_sigma_px
    )
    if float(scroll_soft.max()) > 1e-5:
        scroll_soft = scroll_soft / float(scroll_soft.max())
    bg_clean_f = np.clip(
        bg_clean_f * (1.0 - spec.scroll_ambient_shadow_strength * scroll_soft[:, :, None]),
        0.0,
        255.0,
    )
    return np.rint(bg_clean_f).astype(np.uint8)


def _build_pinned_purim_photo_rgba(
    photo_src_rgb: Image.Image,
    inner_dims: CanvasDimensionsConfig,
    spec: ScrollFootSceneSpec,
    lama_inpainter: LamaOcclusionInpainter,
) -> Image.Image:
    """
    Extracts the physical cream-bordered 1966 Purim photograph from `IMG_WHAT_0433`
    (LaMa-healing the tiny bottom-left camera glare corner), perspective-warps it onto
    `photo_dst_quad_inner_xy` on the flat lower scroll tongue, and adds a subtle
    sub-threshold (`alpha <= 15 < 20`) ambient paper shadow around its rim.
    """
    src_np = np.asarray(photo_src_rgb, dtype=np.uint8)
    gx0, gy0, gx1, gy1 = spec.photo_src_glare_heal_box
    glare_mask = np.zeros(src_np.shape[:2], dtype=np.uint8)
    glare_mask[gy0:gy1, gx0:gx1] = 255
    healed_src = _inpaint_masked_plate_lama(
        src_rgb=photo_src_rgb,
        binary_mask_u8=glare_mask,
        lama_inpainter=lama_inpainter,
        dilation_px=8,
        blur_sigma=2.0,
        max_working_side_px=1350,
        roi_context_margin_px=120,
    )

    ww, wh = spec.photo_work_wh
    src_q = np.asarray(spec.photo_src_outer_quad_xy, dtype=np.float32)
    work_q = np.asarray(
        [[0.0, 0.0], [float(ww - 1), 0.0], [float(ww - 1), float(wh - 1)], [0.0, float(wh - 1)]],
        dtype=np.float32,
    )
    rect_card = cv2.warpPerspective(
        np.asarray(healed_src, dtype=np.uint8),
        cv2.getPerspectiveTransform(src_q, work_q),
        (ww, wh),
        flags=cv2.INTER_LANCZOS4,
    )
    # Harmonize slightly to gallery concrete/paper illumination (warm cream border ~228, rich tonal contrast)
    card_f = rect_card.astype(np.float32) * np.array([0.96, 0.96, 0.95], dtype=np.float32)
    rect_card_u8 = np.clip(np.rint(card_f), 0, 255).astype(np.uint8)

    iw, ih = inner_dims.width, inner_dims.height
    dst_q = np.asarray(spec.photo_dst_quad_inner_xy, dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(work_q, dst_q)
    warped_rgb = cv2.warpPerspective(
        rect_card_u8,
        h_mat,
        (iw, ih),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    card_mask_work = np.full((wh, ww), 255, dtype=np.uint8)
    card_alpha = cv2.warpPerspective(
        card_mask_work,
        h_mat,
        (iw, ih),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Ensure crisp card silhouette above alpha_threshold_u8=20, plus soft sub-threshold shadow (<20)
    card_solid = card_alpha >= 96
    card_aa = np.where(card_alpha >= 20, card_alpha, 0).astype(np.uint8)

    shadow_seed = ndi.shift(card_solid.astype(np.float32), shift=(1.8, 1.5), order=1)
    shadow_blur = ndi.gaussian_filter(shadow_seed, sigma=spec.photo_contact_shadow_sigma_px)
    if float(shadow_blur.max()) > 1e-5:
        shadow_blur = shadow_blur / float(shadow_blur.max())
    shadow_alpha = np.clip(
        np.rint(shadow_blur * float(spec.photo_contact_shadow_max_alpha_u8)),
        0,
        spec.photo_contact_shadow_max_alpha_u8,
    ).astype(np.uint8)

    out_rgba = np.zeros((ih, iw, 4), dtype=np.uint8)
    shadow_only = (card_aa == 0) & (shadow_alpha > 0)
    out_rgba[shadow_only, :3] = 28
    out_rgba[shadow_only, 3] = shadow_alpha[shadow_only]

    card_px = card_aa > 0
    out_rgba[card_px, :3] = warped_rgb[card_px]
    out_rgba[card_px, 3] = card_aa[card_px]
    return Image.fromarray(out_rgba, mode="RGBA")


def build_bene_beitcha_scroll_foot_scene(
    project_root: Path,
    cfg: ScrollPurimCarpetBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    spec = cfg.bene_beitcha_scroll_foot
    inner_dims = cfg.inner_dimensions
    exp_dims = cfg.expanded_dimensions
    ox_in, oy_in = cfg.inner_offset_xy
    dx_in, dy_in = spec.inner_dest_xy
    ox_exp, oy_exp = ox_in + dx_in, oy_in + dy_in
    pw, ph = spec.plate_wh
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Scene 17: {spec.scene_id}] Building museum scroll, pinned Purim photo, "
        f"falling black rod & sculpted foot occluder..."
    )

    muse_orig = Image.open(project_root / spec.scroll_source_relpath).convert("RGB")
    orig_w, orig_h = muse_orig.size
    plate_rgb = muse_orig.resize((pw, ph), Image.Resampling.LANCZOS)
    plate_np = np.asarray(plate_rgb, dtype=np.uint8)

    # 1. Segment Scroll on 667x1000 plate
    scroll_plate_u8 = _keep_largest_cc_u8(
        _segment_prompt_union_u8(
            rgb_image=plate_rgb,
            prompts=spec.scroll_sam_prompts,
            sam_extractor=sam_extractor,
            min_component_area_px=20000,
            max_pinhole_fill_px=5000,
        ),
        thresh=96,
    )

    # 2. Segment full connected `rod + foot` on proven 800x1631 crop, then map onto 667x1000 & 3x plate
    cn_x0, cn_y0, cn_x1, cn_y1 = spec.rod_crop_norm
    cx0, cy0 = int(round(cn_x0 * orig_w)), int(round(cn_y0 * orig_h))
    cx1, cy1 = int(round(cn_x1 * orig_w)), int(round(cn_y1 * orig_h))
    rod_crop_rgb = muse_orig.crop((cx0, cy0, cx1, cy1))
    rw = spec.rod_work_width_px
    rh = int(round(rw * (cy1 - cy0) / max(1, cx1 - cx0)))
    rod_work_rgb = rod_crop_rgb.resize((rw, rh), Image.Resampling.LANCZOS)

    rod_foot_work_u8 = _keep_largest_cc_u8(
        _segment_prompt_union_u8(
            rgb_image=rod_work_rgb,
            prompts=spec.rod_foot_sam_prompts,
            sam_extractor=sam_extractor,
            min_component_area_px=200,
            max_pinhole_fill_px=1800,
        ),
        thresh=64,
    )

    ss = spec.foot_supersample_factor
    pw_3x, ph_3x = pw * ss, ph * ss
    muse_3x = muse_orig.resize((pw_3x, ph_3x), Image.Resampling.LANCZOS)
    muse_3x_np = np.asarray(muse_3x, dtype=np.uint8)

    # Map rod_foot_work_u8 (rw x rh) -> 3x plate (pw_3x x ph_3x)
    grid_x_3x = (
        (np.arange(pw_3x, dtype=np.float32) / float(pw_3x) * float(orig_w) - float(cx0))
        / float(cx1 - cx0)
        * float(rw)
    )
    grid_y_3x = (
        (np.arange(ph_3x, dtype=np.float32) / float(ph_3x) * float(orig_h) - float(cy0))
        / float(cy1 - cy0)
        * float(rh)
    )
    map_x_3x, map_y_3x = np.meshgrid(grid_x_3x, grid_y_3x)
    rod_foot_3x_u8 = cv2.remap(
        rod_foot_work_u8,
        map_x_3x,
        map_y_3x,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 3. Segment sculpted plaster foot at 3x supersampled resolution (2001x3000)
    foot_3x_u8 = _keep_largest_cc_u8(
        _segment_prompt_union_u8(
            rgb_image=muse_3x,
            prompts=spec.foot_3x_sam_prompts,
            sam_extractor=sam_extractor,
            min_component_area_px=800,
            max_pinhole_fill_px=400,
        ),
        thresh=96,
    )

    # Separate isolated black rod + silver ankle shaft (tucking cleanly 3px behind foot ankle socket)
    k_tuck = 2 * spec.foot_tuck_erode_px * ss + 1
    foot_core_3x = cv2.erode(
        (foot_3x_u8 >= 128).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_tuck, k_tuck)),
    )
    rod_only_3x_u8 = _keep_largest_cc_u8(
        np.where(foot_core_3x > 0, 0, rod_foot_3x_u8).astype(np.uint8),
        thresh=80,
    )

    # Decontaminate & downsample Foot (`occluder`) and Rod (`scroll_black_rod`) to 667x1000 -> 1500x1000
    foot_rgb_3x = decontaminate_matte_edge_rgb(muse_3x_np, foot_3x_u8)
    foot_plate_rgba = Image.fromarray(np.dstack([foot_rgb_3x, foot_3x_u8]), mode="RGBA").resize(
        (pw, ph), Image.Resampling.LANCZOS
    )
    rod_rgb_3x = decontaminate_matte_edge_rgb(muse_3x_np, rod_only_3x_u8)
    rod_plate_rgba = Image.fromarray(np.dstack([rod_rgb_3x, rod_only_3x_u8]), mode="RGBA").resize(
        (pw, ph), Image.Resampling.LANCZOS
    )

    # 4. Heal (rod | foot) out of the 3x scroll image via neural LaMa so scroll drawing is 100% clean
    rod_foot_union_3x = np.where((rod_foot_3x_u8 >= 24) | (foot_3x_u8 >= 24), 255, 0).astype(
        np.uint8
    )
    print("  -> Healing sculpted foot & black rod on 3x supersampled architectural scroll via LaMa...")
    healed_muse_3x = _inpaint_masked_plate_lama(
        src_rgb=muse_3x,
        binary_mask_u8=rod_foot_union_3x,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.scroll_rod_heal_dilation_px * ss,
        blur_sigma=spec.scroll_rod_heal_blur_sigma * ss,
        max_working_side_px=1600,
        roi_context_margin_px=180,
    )
    healed_plate_rgb = healed_muse_3x.resize((pw, ph), Image.Resampling.LANCZOS)
    healed_plate_np = np.asarray(healed_plate_rgb, dtype=np.uint8)

    rod_foot_union_1k = np.asarray(
        Image.fromarray(rod_foot_union_3x, mode="L").resize((pw, ph), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    # Trim any tiny heel/toe protrusion that poked outside the clean scroll paper edge
    scroll_eroded = cv2.erode(
        (scroll_plate_u8 >= 128).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    outside_paper = (rod_foot_union_1k >= 128) & (scroll_eroded == 0)
    clean_scroll_alpha = np.where(outside_paper, 0, scroll_plate_u8).astype(np.uint8)
    clean_scroll_rgb = decontaminate_matte_edge_rgb(healed_plate_np, clean_scroll_alpha)
    scroll_plate_rgba = Image.fromarray(
        np.dstack([clean_scroll_rgb, clean_scroll_alpha]), mode="RGBA"
    )

    # Embed mid (scroll), rod, and foot occluder on 1500x1000 inner canvases
    def _embed_on_inner(patch_rgba: Image.Image, solve_c: ContactAnchoredLayerDepthConfig) -> Image.Image:
        canvas = Image.fromarray(
            np.zeros((inner_dims.height, inner_dims.width, 4), dtype=np.uint8), mode="RGBA"
        )
        canvas.alpha_composite(patch_rgba, dest=(dx_in, dy_in))
        pruned = _prune_unanchored_cutout_islands(
            _prune_tiny_alpha_specks_rgba(
                canvas,
                min_area_px=solve_c.min_component_area_px,
                alpha_thresh_u8=solve_c.alpha_threshold_u8,
            ),
            solve_cfg=solve_c,
        )
        return pruned

    mid_canvas = _embed_on_inner(scroll_plate_rgba, spec.mid_depth_solve)
    rod_canvas = _embed_on_inner(rod_plate_rgba, spec.rod_depth_solve)
    occ_canvas = _embed_on_inner(foot_plate_rgba, spec.occluder_depth_solve)
    fg_empty_canvas = Image.fromarray(
        np.zeros((inner_dims.height, inner_dims.width, 4), dtype=np.uint8), mode="RGBA"
    )

    photo_src_rgb = Image.open(project_root / spec.photo_source_relpath).convert("RGB")
    photo_canvas = _build_pinned_purim_photo_rgba(
        photo_src_rgb=photo_src_rgb,
        inner_dims=inner_dims,
        spec=spec,
        lama_inpainter=lama_inpainter,
    )

    _assert_border_alpha_contract(mid_canvas, f"{spec.scene_id}.mid")
    _assert_border_alpha_contract(rod_canvas, f"{spec.scene_id}.rod")
    _assert_border_alpha_contract(occ_canvas, f"{spec.scene_id}.occluder")
    _assert_border_alpha_contract(photo_canvas, f"{spec.scene_id}.photo")
    _assert_border_alpha_contract(fg_empty_canvas, f"{spec.scene_id}.fg")

    mid_np = np.asarray(mid_canvas, dtype=np.uint8)
    rod_np = np.asarray(rod_canvas, dtype=np.uint8)
    occ_np = np.asarray(occ_canvas, dtype=np.uint8)
    photo_np = np.asarray(photo_canvas, dtype=np.uint8)

    # 5. Clean 667x1000 museum plate inside scroll/rod/foot hole, then mirror-reflect onto 2700x1800
    occ_union_plate = np.maximum(scroll_plate_u8, rod_foot_union_1k)
    print(
        "  -> Cleaning 667x1000 museum plate & mirror-reflecting wall, baseboard & concrete floor to 2700x1800..."
    )
    clean_plate_u8 = _clean_scroll_museum_plate(
        plate_rgb_u8=plate_np,
        occluder_union_plate_u8=occ_union_plate,
        spec=spec,
        lama_inpainter=lama_inpainter,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_margin_px=cfg.lama_roi_context_margin_px,
    )
    exp_wh = (exp_dims.width, exp_dims.height)
    bg_inpainted_rgb = Image.fromarray(
        _mirror_reflect_rect_to_canvas(clean_plate_u8, exp_wh, (ox_exp, oy_exp)),
        mode="RGB",
    )

    plate_intact_d_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(plate_rgb) * 255.0), 0, 255
    ).astype(np.uint8)
    raw_intact_depth_u8 = _mirror_reflect_rect_to_canvas(
        plate_intact_d_u8, exp_wh, (ox_exp, oy_exp)
    )
    plate_clean_d_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(
            Image.fromarray(clean_plate_u8, mode="RGB")
        ),
        dtype=np.uint8,
    )
    inpainted_room_depth_u8 = _mirror_reflect_rect_to_canvas(
        plate_clean_d_u8, exp_wh, (ox_exp, oy_exp)
    )
    room_d_cleaned = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(mid_np, rod_np, occ_np),
        cfg=spec.room_depth_clean,
    )
    bg_depth_clean_u8 = _mirror_reflect_rect_to_canvas(
        room_d_cleaned[oy_exp : oy_exp + ph, ox_exp : ox_exp + pw],
        exp_wh,
        (ox_exp, oy_exp),
    )

    # 6. Solve Law 1 ($C^0$ contact) + Law 2 ($C^{-1}$ EDT) depths for mid, photo, rod, and occluder
    yy_in = np.arange(inner_dims.height, dtype=np.float32)[:, None]
    catenary_profile = np.broadcast_to(
        np.exp(-0.5 * ((yy_in - spec.scroll_catenary_peak_y_px) / spec.scroll_catenary_sigma_y_px) ** 2),
        (inner_dims.height, inner_dims.width),
    ).astype(np.float32)
    mid_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=mid_np,
        cfg=spec.mid_depth_solve,
        neural_protrusion_norm=catenary_profile,
    )

    photo_protrusion = np.ones((inner_dims.height, inner_dims.width), dtype=np.float32)
    photo_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=photo_np,
        cfg=spec.photo_depth_solve,
        neural_protrusion_norm=photo_protrusion,
    )

    # Cylindrical cross-axis relief on rod + strict clearance over curved scroll (`mid`)
    rod_opaque = rod_np[:, :, 3] >= spec.rod_depth_solve.alpha_threshold_u8
    rod_dist_in = ndi.distance_transform_edt(rod_opaque).astype(np.float32)
    rod_cyl_norm = np.clip(rod_dist_in / max(1.0, float(np.percentile(rod_dist_in[rod_opaque], 95.0))), 0.0, 1.0)
    rod_solved_raw = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=rod_np,
        cfg=spec.rod_depth_solve,
        neural_protrusion_norm=rod_cyl_norm,
    )
    # Guarantee rod shaft stays strictly in front of curved `mid` scroll paper everywhere away from anchors
    dist_rod_contact = ndi.distance_transform_edt(~rod_solved_raw.contact_mask).astype(np.float32)
    taper_rod = np.clip(dist_rod_contact / spec.rod_depth_solve.contact_relief_taper_px, 0.0, 1.0)
    rod_target_z = np.where(
        rod_opaque & (~rod_solved_raw.contact_mask),
        (1.0 - taper_rod) * rod_solved_raw.world_z_map
        + taper_rod
        * np.maximum(
            rod_solved_raw.world_z_map,
            mid_solved.world_z_map + spec.rod_min_clearance_over_scroll_world,
        ),
        rod_solved_raw.world_z_map,
    )
    rod_d_int = np.clip(
        (rod_target_z - spec.rod_depth_solve.layer_z) / spec.rod_depth_solve.layer_depth_amplitude,
        0.0,
        1.0,
    ).astype(np.float32)
    rod_depth_u8 = np.rint(
        extrapolate_depth_outward_edt(rod_d_int, rod_solved_raw.opaque_mask) * 255.0
    ).astype(np.uint8)

    # Sculpted plaster foot (`occluder`): combine room-relative Depth Anything V2 + sculptural dome
    occ_da2_prot = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_depth_u8,
        cutout_rgba=occ_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.occluder_depth_solve.alpha_threshold_u8,
    )
    occ_opaque = occ_np[:, :, 3] >= spec.occluder_depth_solve.alpha_threshold_u8
    occ_dist = ndi.distance_transform_edt(occ_opaque).astype(np.float32)
    occ_dome = np.power(
        np.clip(occ_dist / max(1.0, float(occ_dist.max())), 0.0, 1.0), 0.65
    ).astype(np.float32)
    occ_protrusion = np.clip(0.55 * occ_da2_prot + 0.45 * occ_dome, 0.0, 1.0).astype(np.float32)
    occ_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=occ_np,
        cfg=spec.occluder_depth_solve,
        neural_protrusion_norm=occ_protrusion,
    )

    # 7. Programmatic topology gates on all 4 displaced layers
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
    photo_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.pinned_purim_photo",
        bg_depth_clean_u8,
        photo_np,
        photo_solved.depth_u8,
        photo_solved.contact_mask,
        spec.photo_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )
    rod_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.scroll_black_rod",
        bg_depth_clean_u8,
        rod_np,
        rod_depth_u8,
        rod_solved_raw.contact_mask,
        spec.rod_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )
    occ_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.occluder",
        bg_depth_clean_u8,
        occ_np,
        occ_solved.depth_u8,
        occ_solved.contact_mask,
        spec.occluder_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )

    # Compute exact rod pivot & tight bounding boxes
    ry_all, rx_all = np.where(rod_opaque)
    y_piv_max = int(ry_all.max())
    bottom_tip_xs = rx_all[ry_all >= y_piv_max - 4]
    rod_pivot_inner_px = [round(float(np.mean(bottom_tip_xs)), 1), float(y_piv_max)]
    rod_inner_box_px = _tight_alpha_bbox(rod_np, 20)
    foot_inner_box_px = _tight_alpha_bbox(occ_np, 20)
    photo_inner_box_px = _tight_alpha_bbox(photo_np, 20)

    # Save all 11 Scene 17 files
    pfx = spec.asset_prefix
    bg_inpainted_rgb.save(
        scenes_dir / f"{pfx}_bg_inpainted.jpg",
        format="JPEG",
        quality=spec.bg_jpeg_quality,
        subsampling=0,
        optimize=True,
    )
    Image.fromarray(bg_depth_clean_u8, mode="L").save(scenes_dir / f"{pfx}_depth.png", format="PNG")
    mid_canvas.save(scenes_dir / f"{pfx}_mid.png", format="PNG")
    Image.fromarray(mid_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_mid_depth.png", format="PNG"
    )
    fg_empty_canvas.save(scenes_dir / f"{pfx}_fg.png", format="PNG")
    photo_canvas.save(scenes_dir / f"{pfx}_photo.png", format="PNG")
    Image.fromarray(photo_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_photo_depth.png", format="PNG"
    )
    rod_canvas.save(scenes_dir / f"{pfx}_rod.png", format="PNG")
    Image.fromarray(rod_depth_u8, mode="L").save(scenes_dir / f"{pfx}_rod_depth.png", format="PNG")
    occ_canvas.save(scenes_dir / f"{pfx}_occluder.png", format="PNG")
    Image.fromarray(occ_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_occluder_depth.png", format="PNG"
    )

    mid_cov = float((mid_np[:, :, 3] >= 20).mean())
    fx0, fy0, fx1, fy1 = foot_inner_box_px
    occ_box_cov = float((occ_np[fy0:fy1, fx0:fx1, 3] >= 24).mean())
    print(
        f"  -> Saved {pfx}_* (mid_cov={mid_cov:.4f}, rod_pivot={rod_pivot_inner_px}, "
        f"photo_box={photo_inner_box_px}, foot_box={foot_inner_box_px}, "
        f"occ_p95_dz={occ_verify['p95_abs_contact_dz_world']:.3f})"
    )

    rig = spec.camera_rig
    extra_layers = [
        {
            "id": "pinned_purim_photo",
            "labelHe": "תצלום פורים הישן (תמי ועדה, ~1966) מוצמד מתחת לכף הרגל",
            "texture": f"assets/scenes/{pfx}_photo.png",
            "depthTexture": f"assets/scenes/{pfx}_photo_depth.png",
            "zPos": spec.photo_depth_solve.layer_z,
            "depthAmplitude": spec.photo_depth_solve.layer_depth_amplitude,
            "pinMarginUv": 0.0,
            "segmentsX": 192,
            "segmentsY": 128,
            "renderOrder": 6.5,
            "expanded": False,
            "baseOpacity": 1.0,
            "excursionBoost": 0.0,
            "depthTest": False,
            "depthWrite": False,
        },
        {
            "id": "scroll_black_rod",
            "labelHe": "המוט השחור הנטוי על המגילה",
            "texture": f"assets/scenes/{pfx}_rod.png",
            "depthTexture": f"assets/scenes/{pfx}_rod_depth.png",
            "zPos": spec.rod_depth_solve.layer_z,
            "depthAmplitude": spec.rod_depth_solve.layer_depth_amplitude,
            "pinMarginUv": 0.0,
            "segmentsX": 192,
            "segmentsY": 128,
            "renderOrder": 7.2,
            "expanded": False,
            "baseOpacity": 1.0,
            "excursionBoost": 0.0,
            "depthTest": False,
            "depthWrite": False,
            "rodPivotInnerPx": rod_pivot_inner_px,
            "rodInnerBoxPx": rod_inner_box_px,
            "rodFallAngleDeg": spec.rod_fall_angle_deg,
        },
    ]
    px_piv, py_piv = rod_pivot_inner_px
    piv_d_u8 = float(rod_depth_u8[min(inner_dims.height - 1, max(0, int(round(py_piv)))), min(inner_dims.width - 1, max(0, int(round(px_piv))))])
    piv_world_z = round(float(spec.rod_depth_solve.layer_z + spec.rod_depth_solve.layer_depth_amplitude * (piv_d_u8 / 255.0)), 3)
    photo_cx = int(round(0.5 * (photo_inner_box_px[0] + photo_inner_box_px[2])))
    photo_cy = int(round(0.5 * (photo_inner_box_px[1] + photo_inner_box_px[3])))
    photo_d_u8 = float(photo_solved.depth_u8[min(inner_dims.height - 1, max(0, photo_cy)), min(inner_dims.width - 1, max(0, photo_cx))])
    photo_world_z = round(float(spec.photo_depth_solve.layer_z + spec.photo_depth_solve.layer_depth_amplitude * (photo_d_u8 / 255.0)), 2)

    portal_shelves = [
        {
            "id": spec.portal_id,
            "targetSceneId": spec.target_scene_id,
            "portalMode": "crossfade",
            "labelHe": spec.portal_label_he,
            "shortBadgeHe": spec.portal_short_badge_he,
            "showMicroBadge": False,
            "innerBoxPx": photo_inner_box_px,
            "worldZ": photo_world_z,
            "recessDepth": 0.0,
            "embedDiorama": False,
            "showFrame": False,
            "restFrameOpacity": 0.0,
            "frameRenderOrder": 6.8,
            "occluder": {
                "texture": f"assets/scenes/{pfx}_occluder.png",
                "depthTexture": f"assets/scenes/{pfx}_occluder_depth.png",
                "worldZ": spec.occluder_depth_solve.layer_z,
                "depthAmplitude": spec.occluder_depth_solve.layer_depth_amplitude,
                "pinMarginUv": 0.0,
                "segmentsX": 192,
                "segmentsY": 128,
                "renderOrder": 7.8,
                "fullInnerCanvas": True,
                "alphaHitThreshold": 24,
                "innerBoxPx": foot_inner_box_px,
                "labelHe": spec.occluder_label_he,
                "coverageRatio": round(occ_box_cov, 4),
                "locksPortal": True,
                "companionFall": {
                    "extraLayerId": "scroll_black_rod",
                    "pivotInnerPx": rod_pivot_inner_px,
                    "pivotWorldZ": piv_world_z,
                    "fallAngleDeg": spec.rod_fall_angle_deg,
                    "shiftWorldX": 0.0,
                    "shiftWorldY": 0.0,
                },
            },
        }
    ]

    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "icon": spec.icon,
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
        "fgRenderOrder": 9,
        "particleColor": rig.particle_color_hex,
        "particleCount": rig.particle_count,
        "hasReflection": False,
        "midCoverageRatio": round(mid_cov, 4),
        "fgCoverageRatio": 0.0,
        "photoInnerBoxPx": photo_inner_box_px,
        "footInnerBoxPx": foot_inner_box_px,
        "rodInnerBoxPx": rod_inner_box_px,
        "rodPivotInnerPx": rod_pivot_inner_px,
        "rodFallAngleDeg": spec.rod_fall_angle_deg,
        "rodFallConvention": "Three.js right-handed +Z rotation in degrees (negative = clockwise on screen onto open right concrete floor along perspective iso-depth line)",
        "extraLayers": extra_layers,
        "portalShelves": portal_shelves,
        "depthTopologyVerification": {
            "mid": mid_verify,
            "pinned_purim_photo": photo_verify,
            "scroll_black_rod": rod_verify,
            "occluder": occ_verify,
        },
    }


# ============================================================================
# Task 1B: Scene 18 Builder — `purim_costume_1966` (`scene18_purim_1966_*`)
# ============================================================================


def build_purim_costume_1966_scene(
    project_root: Path,
    cfg: ScrollPurimCarpetBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    spec = cfg.purim_costume_1966
    inner_dims = cfg.inner_dimensions
    exp_dims = cfg.expanded_dimensions
    ox_in, oy_in = cfg.inner_offset_xy
    dx_in, dy_in = spec.inner_dest_xy
    ox_exp, oy_exp = ox_in + dx_in, oy_in + dy_in
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Scene 18: {spec.scene_id}] Building 1966 Purim garden scene "
        f"(inner={inner_dims.width}x{inner_dims.height}, expanded={exp_dims.width}x{exp_dims.height})..."
    )

    src_orig = np.asarray(Image.open(project_root / spec.source_relpath).convert("RGB"), dtype=np.uint8)
    rw_full, rh_full = spec.rectified_full_wh
    src_q = np.asarray(spec.src_inner_quad_xy, dtype=np.float32)
    dst_q = np.asarray(
        [
            [0.0, 0.0],
            [float(rw_full - 1), 0.0],
            [float(rw_full - 1), float(rh_full - 1)],
            [0.0, float(rh_full - 1)],
        ],
        dtype=np.float32,
    )
    rect_full = cv2.warpPerspective(
        src_orig,
        cv2.getPerspectiveTransform(src_q, dst_q),
        (rw_full, rh_full),
        flags=cv2.INTER_LANCZOS4,
    )
    # Heal washed-out flash glare streak along the lower-left edge of the full rectified garden print
    gx0, gy0, gx1, gy1 = spec.glare_heal_box_rect
    glare_mask_full = np.zeros(rect_full.shape[:2], dtype=np.uint8)
    glare_mask_full[gy0:gy1, gx0:gx1] = 255
    rect_full_clean_img = _inpaint_masked_plate_lama(
        src_rgb=Image.fromarray(rect_full, mode="RGB"),
        binary_mask_u8=glare_mask_full,
        lama_inpainter=lama_inpainter,
        dilation_px=10,
        blur_sigma=2.5,
        max_working_side_px=1500,
        roi_context_margin_px=140,
    )
    rect_full_clean_np = np.asarray(rect_full_clean_img, dtype=np.uint8)

    cy0, cy1 = spec.rectified_crop_y_range
    plate_np = rect_full_clean_np[cy0:cy1, :, :]
    plate_clean_corner_img = Image.fromarray(plate_np, mode="RGB")

    # 1. Segment young Tami, Ada, and the woven straw basket via SAM + IS-Net fusion
    sam_u8 = _segment_prompt_union_u8(
        rgb_image=plate_clean_corner_img,
        prompts=spec.mid_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=spec.min_component_area_px,
        max_pinhole_fill_px=900,
    )
    isnet_a = np.asarray(remove(plate_clean_corner_img, session=matte_extractor.session), dtype=np.uint8)[
        :, :, 3
    ]
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
    fused_alpha = np.where(sam_support > 0, np.maximum(isnet_a, sam_core), 0).astype(np.uint8)
    solid_filled = ndi.binary_fill_holes(fused_alpha >= 96)
    fused_alpha = np.where(solid_filled & (fused_alpha < 255), np.uint8(255), fused_alpha)
    decontam_rgb = decontaminate_matte_edge_rgb(plate_np, fused_alpha)
    mid_patch_rgba = Image.fromarray(np.dstack([decontam_rgb, fused_alpha]), mode="RGBA")

    mid_canvas_raw = Image.fromarray(
        np.zeros((inner_dims.height, inner_dims.width, 4), dtype=np.uint8), mode="RGBA"
    )
    mid_canvas_raw.alpha_composite(mid_patch_rgba, dest=(dx_in, dy_in))
    mid_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            mid_canvas_raw,
            min_area_px=spec.min_component_area_px,
            alpha_thresh_u8=spec.mid_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.mid_depth_solve,
    )
    fg_empty_canvas = Image.fromarray(
        np.zeros((inner_dims.height, inner_dims.width, 4), dtype=np.uint8), mode="RGBA"
    )
    _assert_border_alpha_contract(mid_canvas, f"{spec.scene_id}.mid")
    _assert_border_alpha_contract(fg_empty_canvas, f"{spec.scene_id}.fg")
    mid_np = np.asarray(mid_canvas, dtype=np.uint8)

    # 2. LaMa-inpaint Tami, Ada & basket out of the FULL rectified 1966 garden plate (`rw_full x rh_full`)
    # so upper tree canopy (`y < cy0`) and lower garden ground (`y > cy1`) extend into `2700x1800` natively!
    print("  -> Inpainting figures & basket out of full 1966 garden plate via LaMa...")
    full_hole_u8 = np.zeros((rh_full, rw_full), dtype=np.uint8)
    full_hole_u8[cy0:cy1, :] = np.where(fused_alpha >= 16, 255, 0).astype(np.uint8)
    clean_full_garden_img = _inpaint_masked_plate_lama(
        src_rgb=rect_full_clean_img,
        binary_mask_u8=full_hole_u8,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.bg_inpaint_dilation_px,
        blur_sigma=spec.bg_inpaint_blur_sigma,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )
    clean_full_garden_np = np.asarray(clean_full_garden_img, dtype=np.uint8)

    # 3. Mirror-reflect authentic 1966 garden plate & depth across all 4 photo edges onto 2700x1800
    print("  -> Mirror-reflecting clean 1966 garden plate & depth across all 4 photo edges to 2700x1800...")
    inset = spec.mirror_inset_px
    core_dest_xy = (ox_exp + inset, oy_exp - cy0 + inset)
    exp_wh = (exp_dims.width, exp_dims.height)
    clean_core_u8 = _clean_and_dehaze_purim_garden_core(clean_full_garden_np, spec)
    ch_core, cw_core = clean_core_u8.shape[:2]
    bg_inpainted_rgb = Image.fromarray(
        _mirror_reflect_rect_to_canvas(clean_core_u8, exp_wh, core_dest_xy),
        mode="RGB",
    )

    intact_core_u8 = rect_full_clean_np[inset : rh_full - inset, inset : rw_full - inset]
    intact_core_d_u8 = np.clip(
        np.rint(
            depth_estimator.predict_depth(Image.fromarray(intact_core_u8, mode="RGB")) * 255.0
        ),
        0,
        255,
    ).astype(np.uint8)
    raw_intact_depth_u8 = _mirror_reflect_rect_to_canvas(intact_core_d_u8, exp_wh, core_dest_xy)

    clean_core_d_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(Image.fromarray(clean_core_u8, mode="RGB")),
        dtype=np.uint8,
    )
    inpainted_room_depth_u8 = _mirror_reflect_rect_to_canvas(clean_core_d_u8, exp_wh, core_dest_xy)
    room_d_cleaned = clean_room_depth_under_cutouts(
        room_depth=inpainted_room_depth_u8,
        cutout_layers=(mid_np,),
        cfg=spec.room_depth_clean,
    )
    bg_depth_clean_u8 = _mirror_reflect_rect_to_canvas(
        room_d_cleaned[
            core_dest_xy[1] : core_dest_xy[1] + ch_core,
            core_dest_xy[0] : core_dest_xy[0] + cw_core,
        ],
        exp_wh,
        core_dest_xy,
    )

    mid_protrusion = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_depth_u8,
        cutout_rgba=mid_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.mid_depth_solve.alpha_threshold_u8,
    )
    mid_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=mid_np,
        cfg=spec.mid_depth_solve,
        neural_protrusion_norm=mid_protrusion,
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
    bg_inpainted_rgb.save(
        scenes_dir / f"{pfx}_bg_inpainted.jpg",
        format="JPEG",
        quality=spec.bg_jpeg_quality,
        subsampling=0,
        optimize=True,
    )
    Image.fromarray(bg_depth_clean_u8, mode="L").save(scenes_dir / f"{pfx}_depth.png", format="PNG")
    mid_canvas.save(scenes_dir / f"{pfx}_mid.png", format="PNG")
    Image.fromarray(mid_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_mid_depth.png", format="PNG"
    )
    fg_empty_canvas.save(scenes_dir / f"{pfx}_fg.png", format="PNG")

    mid_cov = float((mid_np[:, :, 3] >= 20).mean())
    print(
        f"  -> Saved {pfx}_* (mid_cov={mid_cov:.4f}, "
        f"mid_p95_dz={mid_verify['p95_abs_contact_dz_world']:.3f})"
    )

    rig = spec.camera_rig
    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "icon": spec.icon,
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
        "midCoverageRatio": round(mid_cov, 4),
        "fgCoverageRatio": 0.0,
        "depthTopologyVerification": {
            "mid": mid_verify,
        },
    }


# ============================================================================
# Task 2: Scene 19 Builder — `herzliya_lawn_carpet` (`scene19_herz_carpet_*`)
# ============================================================================


def _synthesize_extended_earth_trail_rgba(
    biny_np: np.ndarray,
    earth_sam_u8: np.ndarray,
    inner_dims: CanvasDimensionsConfig,
    roll_alpha_inner_u8: np.ndarray,
    spec: HerzliyaCarpetSceneSpec,
    lama_inpainter: LamaOcclusionInpainter,
) -> Image.Image:
    """
    Synthesizes a long, non-repeating, photorealistic brown crumbly soil strip from ONLY
    the isolated earth region of `IMG_BINY_0009` via staggered native-resolution soil patch
    quilting + neural LaMa seam healing + authentic organic crumbly top/bottom edge profiles,
    spanning `x = 0 .. spec.earth_end_x_px` (solidly bleeding off the lower-left edge `x = 0`
    and tucking cleanly behind the solid center of `carpet_roll` across its full unroll).
    """
    iw, ih = inner_dims.width, inner_dims.height
    target_w = spec.earth_end_x_px
    target_y0 = spec.earth_top_y_px
    target_y1 = spec.earth_bottom_y_px
    target_h = target_y1 - target_y0

    inv_scale = 1.0 / spec.carpet_scale
    work_w = int(round(target_w * inv_scale))
    work_h = int(round(target_h * inv_scale))

    sx0, sy0, sx1, sy1 = spec.soil_src_core_box
    core_patch = biny_np[sy0:sy1, sx0:sx1].astype(np.float32)
    ch, cw, _ = core_patch.shape

    # 1. Quilt full-bleed soil texture at native gallery resolution (`work_w x work_h`)
    # using Efros-Freeman minimum-error vertical cuts + golden-ratio horizontal offsets (zero mirror flips!)
    tile_w = min(spec.soil_quilt_tile_w, cw - 8)
    ov_x = min(spec.soil_quilt_overlap_px, tile_w // 2)
    step_x = max(64, tile_w - ov_x)

    rng = np.random.default_rng(7019)
    soil_canvas_f = np.zeros((work_h, work_w, 3), dtype=np.float32)
    seam_centers_x: list[int] = []
    golden = 0.6180339887498949

    crop_h = min(ch, max(180, int(round(ch * 0.86))))
    for t_idx, gx in enumerate(range(0, work_w, step_x)):
        u_q = ((t_idx + 1) * golden + float(rng.uniform(-0.08, 0.08))) % 1.0
        ox = int(round(u_q * float(max(1, cw - tile_w - 1))))
        oy = int(rng.integers(0, max(1, ch - crop_h)))
        sub = core_patch[oy : oy + crop_h, ox : ox + tile_w]
        sub_resized = cv2.resize(sub, (tile_w, work_h), interpolation=cv2.INTER_LANCZOS4)
        x1 = min(work_w, gx + tile_w)
        sw_c = x1 - gx
        sub_c = sub_resized[:, :sw_c]
        if t_idx == 0:
            soil_canvas_f[:, gx:x1] = sub_c
        else:
            w_ox = min(ov_x, sw_c)
            cut_mask = _min_error_vertical_cut_mask(
                soil_canvas_f[:, gx : gx + w_ox], sub_c[:, :w_ox], feather_px=4.0
            )
            blend_1d = np.ones((work_h, sw_c, 1), dtype=np.float32)
            blend_1d[:, :w_ox] = cut_mask
            dst_s = soil_canvas_f[:, gx:x1]
            soil_canvas_f[:, gx:x1] = dst_s * (1.0 - blend_1d) + sub_c * blend_1d
            seam_centers_x.append(gx + w_ox // 2)

    quilted_soil_u8 = np.clip(np.rint(soil_canvas_f), 0, 255).astype(np.uint8)

    # 2. Neural LaMa seam healing across the quilted soil joins (in manageable chunks <= 1920px wide)
    seam_half = spec.soil_seam_heal_half_px
    if seam_centers_x and seam_half > 0:
        chunk_w = 1800
        chunk_step = 1500
        soil_healed_np = quilted_soil_u8.copy()
        for cx0 in range(0, work_w, chunk_step):
            cx1 = min(work_w, cx0 + chunk_w)
            local_mask = np.zeros((work_h, cx1 - cx0), dtype=np.uint8)
            for scx in seam_centers_x:
                if cx0 + 40 <= scx <= cx1 - 40:
                    lx0 = max(0, scx - cx0 - seam_half)
                    lx1 = min(cx1 - cx0, scx - cx0 + seam_half)
                    local_mask[:, lx0:lx1] = 255
            if np.any(local_mask):
                healed_chunk = _inpaint_masked_plate_lama(
                    src_rgb=Image.fromarray(soil_healed_np[:, cx0:cx1], mode="RGB"),
                    binary_mask_u8=local_mask,
                    lama_inpainter=lama_inpainter,
                    dilation_px=4,
                    blur_sigma=2.0,
                    max_working_side_px=1800,
                    roi_context_margin_px=96,
                )
                soil_healed_np[:, cx0:cx1] = np.asarray(healed_chunk, dtype=np.uint8)
        quilted_soil_u8 = soil_healed_np

    # 3. Extract authentic irregular crumbly top & bottom edge offsets from `earth_sam_u8` (`x = 10..760`)
    top_offsets_src: list[float] = []
    bot_offsets_src: list[float] = []
    for x in range(10, 760):
        ys = np.where(earth_sam_u8[:, x] >= 128)[0]
        if len(ys) >= 40:
            top_offsets_src.append(float(ys.min()))
            bot_offsets_src.append(float(ys.max()))
    top_arr = np.asarray(top_offsets_src, dtype=np.float32)
    bot_arr = np.asarray(bot_offsets_src, dtype=np.float32)
    # Detrend linear perspective slope to retain pure organic soil-clod edge roughness
    xs_prof = np.arange(len(top_arr), dtype=np.float32)
    top_jitter = top_arr - np.polyval(np.polyfit(xs_prof, top_arr, 1), xs_prof)
    bot_jitter = bot_arr - np.polyval(np.polyfit(xs_prof, bot_arr, 1), xs_prof)

    def _tile_1d_profile(prof: np.ndarray, length: int, r_seed: int) -> np.ndarray:
        r_loc = np.random.default_rng(r_seed)
        seg_len = min(360, len(prof) - 8)
        ov_1d = 90
        st_1d = seg_len - ov_1d
        a1 = np.zeros(length, dtype=np.float32)
        w1 = np.zeros(length, dtype=np.float32)
        t_win = np.ones(seg_len, dtype=np.float32)
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, ov_1d, dtype=np.float32))
        t_win[:ov_1d] = ramp
        t_win[-ov_1d:] = ramp[::-1]
        for gx in range(0, length, st_1d):
            o = int(r_loc.integers(0, max(1, len(prof) - seg_len)))
            seg = prof[o : o + seg_len]
            if bool(r_loc.integers(0, 2)):
                seg = seg[::-1]
            x1 = min(length, gx + seg_len)
            n = x1 - gx
            wc = t_win[:n].copy()
            if gx == 0:
                wc[:ov_1d] = 1.0
            a1[gx:x1] += seg[:n] * wc
            w1[gx:x1] += wc
        return a1 / np.maximum(1e-4, w1)

    top_jit_w = np.clip(_tile_1d_profile(top_jitter, work_w, 8101) * 1.15, -18.0, 18.0)
    bot_jit_w = np.clip(_tile_1d_profile(bot_jitter, work_w, 8102) * 1.15, -18.0, 18.0)

    yy_w = np.arange(work_h, dtype=np.float32)[:, None]
    y_top_bound = (22.0 + top_jit_w)[None, :]
    y_bot_bound = (float(work_h - 22) + bot_jit_w)[None, :]

    # Right-end taper inside `[earth_end_x_px - 32 .. earth_end_x_px]` (hidden behind carpet at endCenterX)
    xx_w = np.arange(work_w, dtype=np.float32)[None, :]
    right_taper = np.clip((float(work_w - 6) - xx_w) / 95.0, 0.0, 1.0)
    mask_f = (
        np.clip((yy_w - y_top_bound) / 3.2, 0.0, 1.0)
        * np.clip((y_bot_bound - yy_w) / 3.2, 0.0, 1.0)
        * right_taper
    )
    work_alpha_u8 = np.clip(np.rint(mask_f * 255.0), 0, 255).astype(np.uint8)
    # Guarantee 100% solid left-edge bleed at x=0 across core soil rows
    work_alpha_u8[:, 0] = work_alpha_u8[:, 4]

    # 4. Harmonize soil lighting to warm outdoor Herzliya sunlight + subtle vertical mound shading
    gains = np.asarray(spec.outdoor_rgb_gain, dtype=np.float32)[None, None, :]
    v_norm = yy_w / max(1.0, float(work_h - 1))
    sun_curve = (1.06 - 0.14 * (v_norm ** 1.35))[:, :, None]
    soil_lit_u8 = np.clip(
        np.rint(quilted_soil_u8.astype(np.float32) * gains * sun_curve), 0, 255
    ).astype(np.uint8)

    strip_rgba = Image.fromarray(np.dstack([soil_lit_u8, work_alpha_u8]), mode="RGBA").resize(
        (target_w, target_h), Image.Resampling.LANCZOS
    )
    strip_np = np.asarray(strip_rgba, dtype=np.uint8).copy()

    # Ensure vertical span of `earth_full` stays strictly inside the longest contiguous solid run
    # (`alpha >= 245`) of `carpet_roll` at `start_center_x_px` (ignoring any detached bottom fringe tassel)
    # so the `uClipUvMaxX` vertical cut is 100% hidden behind solid carpet at every `progress in [0, 1]`!
    cx_start = int(round(spec.start_center_x_px))
    col_solid = roll_alpha_inner_u8[:, cx_start] >= 245
    best_run_y0, best_run_y1 = -1, -1
    cur_y0 = -1
    for y_i in range(ih):
        if bool(col_solid[y_i]):
            if cur_y0 < 0:
                cur_y0 = y_i
            if (y_i - cur_y0) > (best_run_y1 - best_run_y0):
                best_run_y0, best_run_y1 = cur_y0, y_i
        else:
            cur_y0 = -1
    if best_run_y0 < 0 or (best_run_y1 - best_run_y0) < 40:
        raise RuntimeError(f"carpet_roll lacks a solid contiguous core at start_center_x_px={cx_start}!")
    roll_y_min = best_run_y0 + 4
    roll_y_max = best_run_y1 - 4

    earth_canvas = np.zeros((ih, iw, 4), dtype=np.uint8)
    earth_canvas[target_y0:target_y1, :target_w] = strip_np
    if roll_y_min > 0:
        earth_canvas[:roll_y_min, :, 3] = 0
    if roll_y_max + 1 < ih:
        earth_canvas[roll_y_max + 1 :, :, 3] = 0

    # Clean crisp threshold >= 20 + subtle sub-threshold grass contact shadow (<= 15 < 20)
    solid_soil = earth_canvas[:, :, 3] >= 20
    earth_canvas[:, :, 3] = np.where(solid_soil, earth_canvas[:, :, 3], 0)
    # Ensure left column x=0 has opaque soil pixels (`allow_left_bleed=True`)
    if not np.any(earth_canvas[:, 0, 3] >= 200):
        raise RuntimeError("Extended earth trail failed to reach left image border x=0 solidly.")

    sh_seed = ndi.shift(solid_soil.astype(np.float32), shift=(2.5, 0.0), order=1)
    sh_blur = ndi.gaussian_filter(sh_seed, sigma=spec.grass_contact_shadow_sigma_px)
    if float(sh_blur.max()) > 1e-5:
        sh_blur = sh_blur / float(sh_blur.max())
    sh_alpha = np.clip(
        np.rint(sh_blur * float(spec.grass_contact_shadow_max_alpha_u8)),
        0,
        spec.grass_contact_shadow_max_alpha_u8,
    ).astype(np.uint8)
    sh_only = (~solid_soil) & (sh_alpha > 0)
    earth_canvas[sh_only, :3] = (24, 34, 18)
    earth_canvas[sh_only, 3] = sh_alpha[sh_only]
    # Zero top/bottom/right 1px canvas borders
    earth_canvas[0, :, :] = 0
    earth_canvas[-1, :, :] = 0
    earth_canvas[:, -1, :] = 0

    return Image.fromarray(earth_canvas, mode="RGBA")


def build_herzliya_lawn_carpet_scene(
    project_root: Path,
    cfg: ScrollPurimCarpetBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    spec = cfg.herzliya_lawn_carpet
    inner_dims = cfg.inner_dimensions
    exp_dims = cfg.expanded_dimensions
    ox_in, oy_in = cfg.inner_offset_xy
    iw, ih = inner_dims.width, inner_dims.height
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Scene 19: {spec.scene_id}] Building Herzliya lawn + unrolling carpet & extended earth trail..."
    )

    herz_img = (
        Image.open(project_root / spec.lawn_source_relpath)
        .convert("RGB")
        .resize((iw, ih), Image.Resampling.LANCZOS)
    )
    herz_np = np.asarray(herz_img, dtype=np.uint8)

    # 1. Segment Tal (left, standing) and Guy (right, lying on grass) via SAM + IS-Net fusion
    tal_sam_u8 = _keep_largest_cc_u8(
        _segment_prompt_union_u8(
            rgb_image=herz_img,
            prompts=spec.tal_sam_prompts,
            sam_extractor=sam_extractor,
            min_component_area_px=spec.min_component_area_px,
            max_pinhole_fill_px=600,
        ),
        thresh=96,
    )
    guy_sam_u8 = _keep_largest_cc_u8(
        _segment_prompt_union_u8(
            rgb_image=herz_img,
            prompts=spec.guy_sam_prompts,
            sam_extractor=sam_extractor,
            min_component_area_px=spec.min_component_area_px,
            max_pinhole_fill_px=800,
        ),
        thresh=96,
    )
    herz_isnet_a = np.asarray(remove(herz_img, session=matte_extractor.session), dtype=np.uint8)[:, :, 3]

    k_sup = 2 * spec.sam_support_dilate_px + 1
    k_ero = 2 * spec.sam_core_erode_px + 1
    union_sam = np.maximum(tal_sam_u8, guy_sam_u8)
    sam_sup = cv2.dilate(
        (union_sam >= 96).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sup, k_sup)),
    )
    sam_core = (
        cv2.erode(
            (union_sam >= 160).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_ero, k_ero)),
        )
        * 255
    )
    mid_alpha_raw = np.where(sam_sup > 0, np.maximum(herz_isnet_a, sam_core), 0).astype(np.uint8)
    mid_rgb_decontam = decontaminate_matte_edge_rgb(herz_np, mid_alpha_raw)
    mid_canvas = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            Image.fromarray(np.dstack([mid_rgb_decontam, mid_alpha_raw]), mode="RGBA"),
            min_area_px=spec.min_component_area_px,
            alpha_thresh_u8=spec.mid_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.mid_depth_solve,
    )
    fg_empty_canvas = Image.fromarray(np.zeros((ih, iw, 4), dtype=np.uint8), mode="RGBA")
    _assert_border_alpha_contract(mid_canvas, f"{spec.scene_id}.mid")
    _assert_border_alpha_contract(fg_empty_canvas, f"{spec.scene_id}.fg")
    mid_np = np.asarray(mid_canvas, dtype=np.uint8)

    # 2. Inpaint Tal, Guy, and bottom-right corner garden hose out of 1500x1000, then outpaint to 2700x1800
    print("  -> Inpainting Tal, Guy & corner garden hose out of Herzliya lawn via LaMa...")
    inner_hole_u8 = np.where(mid_np[:, :, 3] >= 14, 255, 0).astype(np.uint8)
    hx0, hy0, hx1, hy1 = spec.hose_heal_box_inner
    inner_hole_u8[hy0:hy1, hx0:hx1] = 255
    clean_inner_lawn_img = _inpaint_masked_plate_lama(
        src_rgb=herz_img,
        binary_mask_u8=inner_hole_u8,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.bg_inpaint_dilation_px,
        blur_sigma=spec.bg_inpaint_blur_sigma,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )
    clean_inner_lawn_np = np.asarray(clean_inner_lawn_img, dtype=np.uint8)

    print("  -> Mirror-reflecting sunlit Herzliya yard & lawn across authentic edges to 2700x1800...")
    exp_wh = (exp_dims.width, exp_dims.height)
    bg_inpainted_rgb = Image.fromarray(
        _mirror_reflect_rect_to_canvas(clean_inner_lawn_np, exp_wh, (ox_in, oy_in)),
        mode="RGB",
    )

    # 3. Depth Anything V2 on authentic 1500x1000 Herzliya yard + horizontal leveling of foreground lawn
    # (`y_inner >= 720`) + exact mirror reflection onto 2700x1800!
    intact_inner_d_u8 = np.clip(
        np.rint(depth_estimator.predict_depth(herz_img) * 255.0), 0, 255
    ).astype(np.uint8)
    raw_intact_depth_u8 = _mirror_reflect_rect_to_canvas(
        intact_inner_d_u8, exp_wh, (ox_in, oy_in)
    )
    clean_inner_d_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(clean_inner_lawn_img), dtype=np.uint8
    )
    room_d_raw = _mirror_reflect_rect_to_canvas(clean_inner_d_u8, exp_wh, (ox_in, oy_in))
    room_d_cleaned = clean_room_depth_under_cutouts(
        room_depth=room_d_raw,
        cutout_layers=(mid_np,),
        cfg=spec.room_depth_clean,
    )
    d_inner_clean = room_d_cleaned[oy_in : oy_in + ih, ox_in : ox_in + iw].astype(np.float32)
    yy_in_f = np.arange(ih, dtype=np.float32)[:, None]
    lawn_level_w = 0.5 * (1.0 + np.tanh((yy_in_f - 735.0) / 36.0))
    row_med = np.median(d_inner_clean[:, 80 : iw - 80], axis=1)[:, None]
    d_inner_leveled_u8 = np.clip(
        np.rint((1.0 - lawn_level_w) * d_inner_clean + lawn_level_w * row_med),
        0,
        255,
    ).astype(np.uint8)
    bg_depth_clean_u8 = _mirror_reflect_rect_to_canvas(
        d_inner_leveled_u8, exp_wh, (ox_in, oy_in)
    )

    # 4. Segment Rolled Carpet Cylinder & Earth Trail from `IMG_BINY_0009` (1800x1200)
    biny_img = Image.open(project_root / spec.carpet_source_relpath).convert("RGB")
    biny_np = np.asarray(biny_img, dtype=np.uint8)
    biny_da2 = depth_estimator.predict_depth(biny_img)

    carpet_sam_raw = _keep_largest_cc_u8(
        _segment_prompt_union_u8(
            rgb_image=biny_img,
            prompts=spec.carpet_sam_prompts,
            sam_extractor=sam_extractor,
            min_component_area_px=25000,
            max_pinhole_fill_px=4500,
        ),
        thresh=96,
    )
    carpet_filled_core = ndi.binary_fill_holes(carpet_sam_raw >= 128)
    carpet_sam_u8 = np.maximum(carpet_sam_raw, carpet_filled_core.astype(np.uint8) * 255)
    earth_sam_u8 = _keep_largest_cc_u8(
        _segment_prompt_union_u8(
            rgb_image=biny_img,
            prompts=spec.earth_sam_prompts,
            sam_extractor=sam_extractor,
            min_component_area_px=25000,
            max_pinhole_fill_px=2000,
        ),
        thresh=96,
    )

    # Decontaminate & outdoor-harmonize rolled carpet cylinder
    carpet_decontam_rgb = decontaminate_matte_edge_rgb(biny_np, carpet_sam_u8)
    c_gains = np.asarray((1.03, 1.01, 0.94), dtype=np.float32)[None, None, :]
    carpet_lit_rgb = np.clip(np.rint(carpet_decontam_rgb.astype(np.float32) * c_gains), 0, 255).astype(
        np.uint8
    )
    cy_idx, cx_idx = np.where(carpet_sam_u8 >= 20)
    cb_x0, cb_y0 = int(cx_idx.min()), int(cy_idx.min())
    cb_x1, cb_y1 = int(cx_idx.max()) + 1, int(cy_idx.max()) + 1

    crop_roll_rgba = Image.fromarray(
        np.dstack([carpet_lit_rgb[cb_y0:cb_y1, cb_x0:cb_x1], carpet_sam_u8[cb_y0:cb_y1, cb_x0:cb_x1]]),
        mode="RGBA",
    )
    crop_roll_da2 = biny_da2[cb_y0:cb_y1, cb_x0:cb_x1]

    rw_scaled = int(round((cb_x1 - cb_x0) * spec.carpet_scale))
    rh_scaled = int(round((cb_y1 - cb_y0) * spec.carpet_scale))
    roll_scaled_rgba = crop_roll_rgba.resize((rw_scaled, rh_scaled), Image.Resampling.LANCZOS)
    roll_scaled_da2 = cv2.resize(crop_roll_da2, (rw_scaled, rh_scaled), interpolation=cv2.INTER_CUBIC)

    roll_dx = int(round(spec.start_center_x_px - 0.5 * rw_scaled))
    roll_dy = spec.roll_bottom_y_px - rh_scaled

    roll_raw_canvas = Image.fromarray(np.zeros((ih, iw, 4), dtype=np.uint8), mode="RGBA")
    roll_raw_canvas.alpha_composite(roll_scaled_rgba, dest=(roll_dx, roll_dy))
    roll_pruned = _prune_unanchored_cutout_islands(
        _prune_tiny_alpha_specks_rgba(
            roll_raw_canvas,
            min_area_px=spec.roll_depth_solve.min_component_area_px,
            alpha_thresh_u8=spec.roll_depth_solve.alpha_threshold_u8,
        ),
        solve_cfg=spec.roll_depth_solve,
    )
    roll_np = np.asarray(roll_pruned, dtype=np.uint8).copy()
    # Add subtle sub-threshold (<20) grass contact shadow under the rolled carpet
    roll_solid = roll_np[:, :, 3] >= 20
    roll_sh_seed = ndi.shift(roll_solid.astype(np.float32), shift=(3.0, 1.5), order=1)
    roll_sh = ndi.gaussian_filter(roll_sh_seed, sigma=spec.grass_contact_shadow_sigma_px)
    if float(roll_sh.max()) > 1e-5:
        roll_sh = roll_sh / float(roll_sh.max())
    roll_sh_a = np.clip(
        np.rint(roll_sh * float(spec.grass_contact_shadow_max_alpha_u8)),
        0,
        spec.grass_contact_shadow_max_alpha_u8,
    ).astype(np.uint8)
    r_sh_only = (~roll_solid) & (roll_sh_a > 0)
    roll_np[r_sh_only, :3] = (22, 32, 16)
    roll_np[r_sh_only, 3] = roll_sh_a[r_sh_only]
    roll_canvas = Image.fromarray(roll_np, mode="RGBA")
    _assert_border_alpha_contract(roll_canvas, f"{spec.scene_id}.roll")

    # 5. Synthesize full-width extended crumbly earth trail (`x = 0 .. earth_end_x_px`)
    print("  -> Quilting & LaMa-healing extended crumbly earth trail from IMG_BINY_0009...")
    earth_canvas = _synthesize_extended_earth_trail_rgba(
        biny_np=biny_np,
        earth_sam_u8=earth_sam_u8,
        inner_dims=inner_dims,
        roll_alpha_inner_u8=roll_np[:, :, 3],
        spec=spec,
        lama_inpainter=lama_inpainter,
    )
    _assert_border_alpha_contract(earth_canvas, f"{spec.scene_id}.earth_full", allow_left_bleed=True)
    earth_np = np.asarray(earth_canvas, dtype=np.uint8)

    # Verify that at `start_center_x_px`, 100% of opaque earth pixels are occluded behind solid `carpet_roll`
    cx_s = int(round(spec.start_center_x_px))
    exposed_clip = (earth_np[:, cx_s, 3] >= 20) & (roll_np[:, cx_s, 3] < 220)
    if np.any(exposed_clip):
        raise RuntimeError(
            f"Earth vertical clip edge at x={cx_s} is not 100% hidden behind carpet_roll!"
        )

    # Verify Tal & Guy sit strictly behind (`y_max < earth_y_min`) the foreground carpet & earth track
    mid_box = _tight_alpha_bbox(mid_np, 20)
    earth_box = _tight_alpha_bbox(earth_np, 20)
    roll_box = _tight_alpha_bbox(roll_np, 20)
    if mid_box[3] > roll_box[1] + 4:
        raise RuntimeError(
            f"Midground figures ({mid_box}) overlap foreground carpet roll ({roll_box})!"
        )

    # 6. Solve Law 1 ($C^0$ lawn contact) + Law 2 ($C^{-1}$ EDT) depths for `mid`, `earth_full`, and `roll`
    mid_protrusion = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_depth_u8,
        cutout_rgba=mid_np,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.mid_depth_solve.alpha_threshold_u8,
    )
    mid_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=mid_np,
        cfg=spec.mid_depth_solve,
        neural_protrusion_norm=mid_protrusion,
    )

    # Crumbly soil mound relief (`+5.8` world units) modulated by soil pebble luminance & distance from edge
    earth_opaque = earth_np[:, :, 3] >= spec.earth_depth_solve.alpha_threshold_u8
    earth_dist = ndi.distance_transform_edt(earth_opaque).astype(np.float32)
    earth_mound = np.clip(earth_dist / 16.0, 0.0, 1.0)
    soil_luma = (
        0.299 * earth_np[:, :, 0].astype(np.float32)
        + 0.587 * earth_np[:, :, 1].astype(np.float32)
        + 0.114 * earth_np[:, :, 2].astype(np.float32)
    ) / 255.0
    earth_prot = np.clip(earth_mound * (0.72 + 0.28 * soil_luma), 0.0, 1.0).astype(np.float32)
    earth_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=earth_np,
        cfg=spec.earth_depth_solve,
        neural_protrusion_norm=earth_prot,
    )

    # Rolled carpet 3D cylindrical + spiral cross-section relief (`+26.0` world units)
    roll_da2_canvas = np.zeros((ih, iw), dtype=np.float32)
    roll_da2_canvas[roll_dy : roll_dy + rh_scaled, roll_dx : roll_dx + rw_scaled] = roll_scaled_da2
    roll_vals = roll_da2_canvas[roll_solid]
    r_lo, r_hi = float(np.percentile(roll_vals, 5.0)), float(np.percentile(roll_vals, 98.0))
    roll_da2_norm = np.clip((roll_da2_canvas - r_lo) / max(1e-5, r_hi - r_lo), 0.0, 1.0)
    roll_dist = ndi.distance_transform_edt(roll_solid).astype(np.float32)
    roll_dome = np.power(np.clip(roll_dist / max(1.0, float(roll_dist.max())), 0.0, 1.0), 0.6)
    roll_prot = np.clip(0.55 * roll_da2_norm + 0.45 * roll_dome, 0.0, 1.0).astype(np.float32)
    roll_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=roll_np,
        cfg=spec.roll_depth_solve,
        neural_protrusion_norm=roll_prot,
    )

    # Verify & gate all 3 displaced cutout layers
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
    earth_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.carpet_earth_trail",
        bg_depth_clean_u8,
        earth_np,
        earth_solved.depth_u8,
        earth_solved.contact_mask,
        spec.earth_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )
    roll_verify = _verify_and_gate_layer_topology(
        f"{spec.scene_id}.carpet_roll",
        bg_depth_clean_u8,
        roll_np,
        roll_solved.depth_u8,
        roll_solved.contact_mask,
        spec.roll_depth_solve,
        cfg.topology_verification,
        cfg.edt_rim_probe_px,
        cfg.max_edt_rim_jump_world,
    )

    # Measure lawn world-Z along the horizontal unroll track (startCenterX -> endCenterX)
    room_world_z_exp = compute_world_z_map(
        depth_map=bg_depth_clean_u8,
        plane_z=spec.roll_depth_solve.bg_z,
        depth_amplitude=spec.roll_depth_solve.room_depth_amplitude,
        pin_margin_uv=spec.roll_depth_solve.room_pin_margin_uv,
    )
    host_world_z_in = extract_layer_window_from_expanded(room_world_z_exp, layer_shape=(ih, iw))
    y_contact_probe = min(ih - 2, spec.roll_bottom_y_px - 3)
    start_cx_i = int(round(spec.start_center_x_px))
    end_cx_i = int(round(spec.end_center_x_px))
    start_lawn_z = float(
        np.median(host_world_z_in[y_contact_probe - 4 : y_contact_probe + 2, start_cx_i - 30 : start_cx_i + 30])
    )
    end_lawn_z = float(
        np.median(host_world_z_in[y_contact_probe - 4 : y_contact_probe + 2, end_cx_i - 30 : end_cx_i + 30])
    )
    delta_world_z = round(end_lawn_z - start_lawn_z, 4)

    # Save all 9 Scene 19 files
    pfx = spec.asset_prefix
    bg_inpainted_rgb.save(
        scenes_dir / f"{pfx}_bg_inpainted.jpg",
        format="JPEG",
        quality=spec.bg_jpeg_quality,
        subsampling=0,
        optimize=True,
    )
    Image.fromarray(bg_depth_clean_u8, mode="L").save(scenes_dir / f"{pfx}_depth.png", format="PNG")
    mid_canvas.save(scenes_dir / f"{pfx}_mid.png", format="PNG")
    Image.fromarray(mid_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_mid_depth.png", format="PNG"
    )
    fg_empty_canvas.save(scenes_dir / f"{pfx}_fg.png", format="PNG")
    earth_canvas.save(scenes_dir / f"{pfx}_earth_full.png", format="PNG")
    Image.fromarray(earth_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_earth_full_depth.png", format="PNG"
    )
    roll_canvas.save(scenes_dir / f"{pfx}_roll.png", format="PNG")
    Image.fromarray(roll_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_roll_depth.png", format="PNG"
    )

    mid_cov = float((mid_np[:, :, 3] >= 20).mean())
    start_uv_x = round(spec.start_center_x_px / float(iw), 6)
    end_uv_x = round(float(spec.earth_end_x_px) / float(iw), 6)
    delta_inner_px = round(spec.end_center_x_px - spec.start_center_x_px, 2)
    roll_core_depth_u8 = float(np.median(roll_solved.depth_u8[roll_solid]))
    roll_track_world_z = round(
        spec.roll_depth_solve.layer_z
        + spec.roll_depth_solve.layer_depth_amplitude * (roll_core_depth_u8 / 255.0),
        3,
    )

    carpet_unroll_spec = {
        "earthLayerId": "carpet_earth_trail",
        "rollLayerId": "carpet_roll",
        "reflectedEarthLayerId": "carpet_earth_trail_reflection" if spec.reflect_left_mirror else None,
        "reflectedRollLayerId": "carpet_roll_reflection" if spec.reflect_left_mirror else None,
        "startCenterX": spec.start_center_x_px,
        "endCenterX": spec.end_center_x_px,
        "deltaInnerPx": delta_inner_px,
        "startUvX": start_uv_x,
        "endUvX": end_uv_x,
        "rollTrackWorldZ": roll_track_world_z,
        "trailTrackWorldZ": round(start_lawn_z, 3),
        "durationMs": spec.unroll_duration_ms,
        "rollBoxInnerPx": roll_box,
        "earthBoxInnerPx": earth_box,
        "startLawnWorldZ": round(start_lawn_z, 3),
        "endLawnWorldZ": round(end_lawn_z, 3),
        "deltaWorldZ": delta_world_z,
    }
    carpet_unroll_rig = {
        "trailLayerId": "carpet_earth_trail",
        "rollLayerId": "carpet_roll",
        "reflectedTrailLayerId": "carpet_earth_trail_reflection" if spec.reflect_left_mirror else None,
        "reflectedRollLayerId": "carpet_roll_reflection" if spec.reflect_left_mirror else None,
        "clipUvMinX": 0.0,
        "initialClipUvMaxX": start_uv_x,
        "unrolledClipUvMaxX": end_uv_x,
        "rollShiftInnerPxX": delta_inner_px,
        "rollShiftInnerPxY": 0.0,
        "rollShiftWorldZ": 0.0,
        "rollTrackWorldZ": roll_track_world_z,
        "trailTrackWorldZ": round(start_lawn_z, 3),
        "durationMs": spec.unroll_duration_ms,
    }
    print(
        f"  -> Saved {pfx}_* (mid_cov={mid_cov:.4f}, roll_box={roll_box}, "
        f"earth_box={earth_box}, deltaWorldZ={delta_world_z})"
    )

    rig = spec.camera_rig
    extra_layers = [
        {
            "id": "carpet_earth_trail",
            "labelHe": "שובל האדמה המתמשך על הדשא — ׳הגלגול הבא׳",
            "texture": f"assets/scenes/{pfx}_earth_full.png",
            "depthTexture": f"assets/scenes/{pfx}_earth_full_depth.png",
            "zPos": spec.earth_depth_solve.layer_z,
            "depthAmplitude": spec.earth_depth_solve.layer_depth_amplitude,
            "pinMarginUv": 0.0,
            "segmentsX": 224,
            "segmentsY": 150,
            "renderOrder": 7.2,
            "expanded": False,
            "mirrorLeftEdge": False,
            "baseOpacity": 1.0,
            "excursionBoost": 0.0,
            "depthTest": False,
            "depthWrite": False,
            "clipUvMinX": 0.0,
            "clipUvMaxX": start_uv_x,
            "initialClipUvMaxX": start_uv_x,
        },
        {
            "id": "carpet_roll",
            "labelHe": "גליל השטיח והאדמה — לחצו לגלגול על הדשא",
            "texture": f"assets/scenes/{pfx}_roll.png",
            "depthTexture": f"assets/scenes/{pfx}_roll_depth.png",
            "zPos": spec.roll_depth_solve.layer_z,
            "depthAmplitude": spec.roll_depth_solve.layer_depth_amplitude,
            "pinMarginUv": 0.0,
            "segmentsX": 224,
            "segmentsY": 150,
            "renderOrder": 7.6,
            "expanded": False,
            "mirrorLeftEdge": False,
            "baseOpacity": 1.0,
            "excursionBoost": 0.0,
            "depthTest": False,
            "depthWrite": False,
            "rollBoxInnerPx": roll_box,
        },
    ]
    if spec.reflect_left_mirror:
        extra_layers.extend(
            [
                {
                    "id": "carpet_earth_trail_reflection",
                    "labelHe": "השתקפות שובל האדמה המתמשך מעבר לשפת המראה השמאלית",
                    "texture": f"assets/scenes/{pfx}_earth_full.png",
                    "depthTexture": f"assets/scenes/{pfx}_earth_full_depth.png",
                    "zPos": spec.earth_depth_solve.layer_z,
                    "depthAmplitude": spec.earth_depth_solve.layer_depth_amplitude,
                    "pinMarginUv": 0.0,
                    "segmentsX": 224,
                    "segmentsY": 150,
                    "renderOrder": 7.2,
                    "expanded": False,
                    "mirrorLeftEdge": True,
                    "baseOpacity": 1.0,
                    "excursionBoost": 0.0,
                    "depthTest": False,
                    "depthWrite": False,
                    "clipUvMinX": 0.0,
                    "clipUvMaxX": start_uv_x,
                    "initialClipUvMaxX": start_uv_x,
                },
                {
                    "id": "carpet_roll_reflection",
                    "labelHe": "השתקפות גליל השטיח מעבר לשפת המראה השמאלית",
                    "texture": f"assets/scenes/{pfx}_roll.png",
                    "depthTexture": f"assets/scenes/{pfx}_roll_depth.png",
                    "zPos": spec.roll_depth_solve.layer_z,
                    "depthAmplitude": spec.roll_depth_solve.layer_depth_amplitude,
                    "pinMarginUv": 0.0,
                    "segmentsX": 224,
                    "segmentsY": 150,
                    "renderOrder": 7.6,
                    "expanded": False,
                    "mirrorLeftEdge": True,
                    "baseOpacity": 1.0,
                    "excursionBoost": 0.0,
                    "depthTest": False,
                    "depthWrite": False,
                    "rollBoxInnerPx": roll_box,
                },
            ]
        )

    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "icon": spec.icon,
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
        "midCoverageRatio": round(mid_cov, 4),
        "fgCoverageRatio": 0.0,
        "carpetUnrollSpec": carpet_unroll_spec,
        "carpetUnrollRig": carpet_unroll_rig,
        "extraLayers": extra_layers,
        "depthTopologyVerification": {
            "mid": mid_verify,
            "carpet_earth_trail": earth_verify,
            "carpet_roll": roll_verify,
        },
    }


# ============================================================================
# Canonical Default Configuration & CLI Orchestrator
# ============================================================================


def default_scroll_purim_carpet_bundle_config() -> ScrollPurimCarpetBundleConfig:
    room_pin = 0.045

    # Helper to convert Scene 17 inner coords -> 3x supersampled 2001x3000 plate coords
    def _s17_3x_pt(xi: float, yi: float) -> tuple[float, float]:
        return ((xi - 416.0) * 3.0, yi * 3.0)

    def _s17_3x_box(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        return ((x0 - 416) * 3, y0 * 3, (x1 - 416) * 3, y1 * 3)

    s17_bg_z = -350.0
    s17_room_amp = 118.0
    scroll_foot_spec = ScrollFootSceneSpec(
        scene_id="bene_beitcha_scroll_foot",
        title_he="מגילה וכף רגל — תצלום פורים מוצמד מתחת לכף הרגל (׳בנה ביתך׳, 2026)",
        short_label_he="מגילה וכף רגל",
        icon="",
        asset_prefix="scene17_scroll_foot",
        scroll_source_relpath="media/museum_press/rgma_bene_beitcha_scroll_drawing_foot_2026_331d870d.jpg",
        photo_source_relpath="media/whatsapp_family/wa_gadi_childhood_ca1966_tami_ada_purim_kneeling_c4a60365.jpg",
        plate_wh=(667, 1000),
        inner_dest_xy=(416, 0),
        foot_supersample_factor=3,
        rod_crop_norm=(0.25, 0.02, 0.95, 0.97),
        rod_work_width_px=800,
        foot_tuck_erode_px=3,
        scroll_sam_prompts=(
            _pg(
                (185, 38, 616, 962),
                [
                    (210, 60),
                    (415, 75),
                    (310, 220),
                    (240, 450),
                    (415, 450),
                    (260, 680),
                    (410, 690),
                    (320, 820),
                    (480, 820),
                    (350, 935),
                    (575, 868),
                ],
                [
                    (150, 120),
                    (175, 450),
                    (200, 720),
                    (260, 880),
                    (310, 965),
                    (520, 930),
                    (580, 760),
                    (465, 620),
                    (460, 350),
                    (460, 120),
                ],
            ),
        ),
        rod_foot_sam_prompts=(
            _pg(
                (240, 452, 468, 1384),
                [
                    (262.0, 476.0),
                    (272.0, 520.0),
                    (282.0, 575.0),
                    (296.0, 635.0),
                    (308.0, 700.0),
                    (322.0, 770.0),
                    (336.0, 840.0),
                    (350.0, 910.0),
                    (364.0, 980.0),
                    (376.0, 1035.0),
                    (388.0, 1085.0),
                    (398.0, 1130.0),
                    (405.0, 1152.0),
                    (414.0, 1190.0),
                    (425.0, 1245.0),
                    (434.0, 1295.0),
                    (410.0, 1320.0),
                    (382.0, 1315.0),
                    (436.0, 1345.0),
                    (445.0, 1368.0),
                ],
                [
                    (225.0, 490.0),
                    (245.0, 600.0),
                    (270.0, 740.0),
                    (290.0, 860.0),
                    (315.0, 990.0),
                    (345.0, 1100.0),
                    (365.0, 1210.0),
                    (335.0, 1315.0),
                    (375.0, 1365.0),
                    (305.0, 490.0),
                    (325.0, 610.0),
                    (348.0, 740.0),
                    (370.0, 860.0),
                    (395.0, 990.0),
                    (425.0, 1100.0),
                    (445.0, 1200.0),
                    (465.0, 1295.0),
                    (472.0, 1355.0),
                    (258.0, 438.0),
                    (440.0, 1395.0),
                ],
            ),
        ),
        foot_3x_sam_prompts=(
            _pg(
                _s17_3x_box(784, 750, 845, 818),
                [
                    _s17_3x_pt(794, 764),
                    _s17_3x_pt(802, 782),
                    _s17_3x_pt(816, 796),
                    _s17_3x_pt(830, 808),
                    _s17_3x_pt(836, 792),
                    _s17_3x_pt(816, 774),
                ],
                [
                    _s17_3x_pt(825, 755),
                    _s17_3x_pt(828, 768),
                    _s17_3x_pt(780, 775),
                    _s17_3x_pt(805, 812),
                    _s17_3x_pt(848, 802),
                    _s17_3x_pt(806, 752),
                ],
            ),
        ),
        scroll_rod_heal_dilation_px=9,
        scroll_rod_heal_blur_sigma=2.0,
        bg_inpaint_dilation_px=26,
        bg_inpaint_blur_sigma=3.5,
        bg_jpeg_quality=94,
        baseboard_fit_left_x_range=(12, 196),
        baseboard_fit_right_x_range=(468, 656),
        baseboard_left_y_window=(700, 770),
        baseboard_right_y_window=(620, 680),
        baseboard_band_half_px=14,
        harmonic_lowpass_sigma_px=18.0,
        harmonic_downsample_factor=3,
        grain_highpass_sigma_px=9.0,
        quilt_tile_px=128,
        quilt_overlap_px=48,
        wall_grain_box_plate=(16, 80, 176, 620),
        floor_grain_box_plate=(16, 780, 236, 985),
        scroll_ambient_shadow_strength=0.045,
        scroll_ambient_shadow_sigma_px=16.0,
        scroll_catenary_peak_y_px=645.0,
        scroll_catenary_sigma_y_px=85.0,
        scroll_catenary_relief_world=9.0,
        photo_src_outer_quad_xy=(
            (70.0, 48.0),
            (934.0, 24.0),
            (1006.0, 1252.0),
            (60.0, 1266.0),
        ),
        photo_src_glare_heal_box=(55, 1085, 315, 1265),
        photo_work_wh=(480, 620),
        photo_dst_quad_inner_xy=(
            (706.0, 804.0),
            (822.0, 774.0),
            (882.0, 904.0),
            (758.0, 938.0),
        ),
        photo_contact_shadow_max_alpha_u8=15,
        photo_contact_shadow_sigma_px=2.4,
        rod_fall_angle_deg=-95.0,
        rod_min_clearance_over_scroll_world=1.4,
        portal_id="portal_scroll_foot_purim_1966",
        target_scene_id="purim_costume_1966",
        portal_label_he="תמי ועדה מחופשות בפורים (~1966) · לחצו על התצלום להיכנס לגינה",
        portal_short_badge_he="אל פורים 1966",
        occluder_label_he="כף הרגל המפוסלת המצמידה את פינת התצלום",
        room_depth_clean=RoomDepthCleanConfig(
            alpha_threshold_u8=16,
            mask_dilation_px=22,
            post_smooth_sigma_px=3.0,
            interior_blend_band_px=12.0,
        ),
        mid_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=s17_bg_z,
            room_depth_amplitude=s17_room_amp,
            room_pin_margin_uv=room_pin,
            layer_z=-350.0,
            layer_depth_amplitude=165.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=6,
            min_component_area_px=400,
            contact_boxes=(
                ContactAnchorBox(x0=600, y0=38, x1=860, y1=76, mode="top_band", band_px=8),
                ContactAnchorBox(x0=690, y0=754, x1=1035, y1=962, mode="all_opaque", band_px=6),
            ),
            neural_relief_amplitude=9.0,
            contact_relief_taper_px=12.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.2,
            interior_smooth_sigma_px=2.0,
        ),
        photo_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=s17_bg_z,
            room_depth_amplitude=s17_room_amp,
            room_pin_margin_uv=room_pin,
            layer_z=-350.0,
            layer_depth_amplitude=165.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=4,
            min_component_area_px=200,
            contact_boxes=(
                ContactAnchorBox(x0=700, y0=768, x1=890, y1=945, mode="perimeter_rim", band_px=4),
            ),
            neural_relief_amplitude=1.2,
            contact_relief_taper_px=4.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.2,
            interior_smooth_sigma_px=1.0,
        ),
        rod_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=s17_bg_z,
            room_depth_amplitude=s17_room_amp,
            room_pin_margin_uv=room_pin,
            layer_z=-350.0,
            layer_depth_amplitude=165.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=8,
            min_component_area_px=120,
            contact_boxes=(
                ContactAnchorBox(x0=715, y0=278, x1=746, y1=312, mode="top_band", band_px=6),
                ContactAnchorBox(x0=815, y0=758, x1=842, y1=786, mode="bottom_band", band_px=6),
            ),
            neural_relief_amplitude=10.5,
            contact_relief_taper_px=8.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.2,
            interior_smooth_sigma_px=1.2,
        ),
        occluder_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=s17_bg_z,
            room_depth_amplitude=s17_room_amp,
            room_pin_margin_uv=room_pin,
            layer_z=-350.0,
            layer_depth_amplitude=165.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=5,
            min_component_area_px=150,
            contact_boxes=(
                ContactAnchorBox(x0=780, y0=748, x1=848, y1=820, mode="bottom_band", band_px=5),
            ),
            neural_relief_amplitude=18.0,
            contact_relief_taper_px=6.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.2,
            interior_smooth_sigma_px=1.2,
        ),
        camera_rig=SceneCameraRigSpec(
            min_inner_scale=1.94,
            initial_camera_y=-8.0,
            initial_camera_z=535.0,
            min_camera_z=-90.0,
            max_camera_z=695.0,
            max_yaw_deg=23.0,
            max_pitch_deg=16.0,
            max_strafe_x=135.0,
            max_strafe_y=85.0,
            dolly_speed_mult=1.3,
            particle_color_hex="0xE4DDD2",
            particle_count=52,
        ),
    )

    # Convert Scene 18 prompts from 720x1000 probe coords -> 1008x1000 (cropped y=255..1255 of 1008x1400)
    # Isotropic 1.40x scale: x_1008 = x_720 * 1.40, y_crop = y_720 * 1.40 - 255.0
    def _s18_pt(x720: float, y720: float) -> tuple[float, float]:
        return (round(x720 * 1.40, 1), round(y720 * 1.40 - 255.0, 1))

    def _s18_box(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        return (
            int(round(x0 * 1.40)),
            max(0, int(round(y0 * 1.40 - 255.0))),
            min(1008, int(round(x1 * 1.40))),
            min(1000, int(round(y1 * 1.40 - 255.0))),
        )

    s18_bg_z = -350.0
    s18_room_amp = 108.0
    purim_spec = PurimCostumeSceneSpec(
        scene_id="purim_costume_1966",
        title_he="תמי ועדה בפורים עם סלסלת הקש (~1966)",
        short_label_he="פורים 1966",
        icon="",
        asset_prefix="scene18_purim_1966",
        source_relpath="media/whatsapp_family/wa_gadi_childhood_ca1966_tami_ada_purim_kneeling_c4a60365.jpg",
        src_inner_quad_xy=(
            (126.0, 96.0),
            (856.0, 78.0),
            (918.0, 1182.0),
            (122.0, 1190.0),
        ),
        rectified_full_wh=(1008, 1400),
        rectified_crop_y_range=(255, 1255),
        inner_dest_xy=(246, 0),
        glare_heal_box_rect=(0, 1278, 165, 1400),
        mid_sam_prompts=(
            _pg(
                _s18_box(162, 345, 395, 812),
                [
                    _s18_pt(330, 380),
                    _s18_pt(335, 435),
                    _s18_pt(335, 515),
                    _s18_pt(345, 575),
                    _s18_pt(350, 625),
                    _s18_pt(345, 665),
                    _s18_pt(250, 560),
                    _s18_pt(215, 680),
                    _s18_pt(275, 660),
                    _s18_pt(240, 780),
                    _s18_pt(335, 675),
                ],
                [
                    _s18_pt(250, 380),
                    _s18_pt(150, 550),
                    _s18_pt(145, 720),
                    _s18_pt(235, 835),
                    _s18_pt(385, 365),
                ],
            ),
            _pg(
                _s18_box(330, 365, 595, 845),
                [
                    _s18_pt(480, 405),
                    _s18_pt(480, 455),
                    _s18_pt(480, 560),
                    _s18_pt(415, 560),
                    _s18_pt(350, 600),
                    _s18_pt(352, 650),
                    _s18_pt(545, 585),
                    _s18_pt(555, 690),
                    _s18_pt(485, 710),
                    _s18_pt(485, 800),
                    _s18_pt(565, 772),
                ],
                [
                    _s18_pt(415, 390),
                    _s18_pt(565, 420),
                    _s18_pt(605, 580),
                    _s18_pt(605, 740),
                    _s18_pt(485, 860),
                    _s18_pt(575, 830),
                ],
            ),
            _pg(
                _s18_box(320, 670, 430, 846),
                [
                    _s18_pt(376, 695),
                    _s18_pt(365, 765),
                    _s18_pt(395, 785),
                    _s18_pt(368, 822),
                ],
                [
                    _s18_pt(312, 790),
                    _s18_pt(368, 858),
                    _s18_pt(432, 825),
                ],
            ),
        ),
        sam_support_dilate_px=3,
        sam_core_erode_px=4,
        min_component_area_px=1200,
        bg_inpaint_dilation_px=26,
        bg_inpaint_blur_sigma=3.5,
        mirror_inset_px=16,
        glare_dehaze_width_px=210,
        glare_dehaze_y_center_px=440.0,
        glare_dehaze_y_scale_px=95.0,
        glare_dehaze_strength=0.82,
        bg_jpeg_quality=94,
        room_depth_clean=RoomDepthCleanConfig(
            alpha_threshold_u8=16,
            mask_dilation_px=22,
            post_smooth_sigma_px=3.0,
            interior_blend_band_px=12.0,
        ),
        mid_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=s18_bg_z,
            room_depth_amplitude=s18_room_amp,
            room_pin_margin_uv=room_pin,
            layer_z=-350.0,
            layer_depth_amplitude=165.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=12,
            min_component_area_px=600,
            auto_floor_band_px=7,
            auto_floor_max_rise_px=68,
            neural_relief_amplitude=22.0,
            contact_relief_taper_px=10.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.2,
            interior_smooth_sigma_px=1.6,
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
            particle_color_hex="0xEAD8BE",
            particle_count=58,
        ),
    )

    s19_bg_z = -350.0
    s19_room_amp = 114.0
    herz_carpet_spec = HerzliyaCarpetSceneSpec(
        scene_id="herzliya_lawn_carpet",
        title_he="דשא הרצליה ו׳הגלגול הבא׳ — טל וגיא מול שטיח האדמה המתגלגל",
        short_label_he="השטיח על הדשא",
        icon="",
        asset_prefix="scene19_herz_carpet",
        lawn_source_relpath="media/herzliya_fried_house/fried2_pict0069_419e57a6.jpg",
        carpet_source_relpath="media/binyamin_gallery/binyamin_הגלגול_הבא_מבט_צד.jpg_9568681a.jpg",
        tal_split_x_px=550,
        tal_sam_prompts=(
            _pg(
                (366, 170, 516, 754),
                [
                    (462, 210),
                    (435, 315),
                    (430, 425),
                    (415, 550),
                    (478, 565),
                    (402, 685),
                    (465, 685),
                    (412, 732),
                    (482, 708),
                ],
                [
                    (355, 250),
                    (520, 250),
                    (355, 500),
                    (525, 500),
                    (355, 720),
                    (530, 715),
                    (420, 762),
                ],
            ),
        ),
        guy_sam_prompts=(
            _pg(
                (592, 504, 988, 660),
                [
                    (705, 535),
                    (725, 595),
                    (645, 615),
                    (635, 640),
                    (795, 622),
                    (885, 622),
                    (950, 595),
                    (965, 615),
                ],
                [
                    (635, 555),
                    (785, 560),
                    (900, 555),
                    (635, 665),
                    (780, 662),
                    (920, 658),
                ],
            ),
        ),
        sam_support_dilate_px=4,
        sam_core_erode_px=4,
        min_component_area_px=1000,
        hose_heal_box_inner=(1368, 868, 1500, 1000),
        bg_inpaint_dilation_px=26,
        bg_inpaint_blur_sigma=3.5,
        bg_jpeg_quality=94,
        carpet_sam_prompts=(
            _pg(
                (750, 294, 1776, 1138),
                [
                    (980, 340),
                    (1380, 360),
                    (850, 520),
                    (1150, 550),
                    (1580, 560),
                    (910, 720),
                    (1350, 720),
                    (1680, 760),
                    (980, 890),
                    (1280, 940),
                    (1580, 960),
                    (1025, 1035),
                    (1140, 1065),
                    (1225, 1110),
                    (1360, 1095),
                    (1495, 1092),
                    (1595, 1058),
                    (1685, 985),
                ],
                [
                    (720, 500),
                    (740, 780),
                    (820, 960),
                    (910, 1080),
                    (1040, 1150),
                    (1225, 1165),
                    (1420, 1160),
                    (1620, 1145),
                    (1785, 960),
                    (1760, 420),
                    (1350, 260),
                ],
            ),
        ),
        earth_sam_prompts=(
            _pg(
                (0, 700, 1195, 1185),
                [
                    (60, 760),
                    (350, 760),
                    (680, 760),
                    (100, 920),
                    (450, 920),
                    (780, 920),
                    (120, 1100),
                    (480, 1090),
                    (820, 1075),
                    (1020, 1085),
                ],
                [
                    (100, 680),
                    (450, 680),
                    (740, 685),
                    (860, 760),
                    (930, 880),
                    (1020, 980),
                    (1140, 1040),
                    (1230, 1110),
                    (900, 1150),
                    (500, 1175),
                    (150, 1190),
                ],
            ),
        ),
        carpet_scale=0.236,
        start_center_x_px=235.0,
        end_center_x_px=1315.0,
        roll_bottom_y_px=946,
        earth_end_x_px=1355,
        earth_top_y_px=834,
        earth_bottom_y_px=930,
        soil_src_core_box=(12, 748, 792, 1108),
        soil_quilt_tile_w=420,
        soil_quilt_overlap_px=150,
        soil_seam_heal_half_px=18,
        outdoor_rgb_gain=(1.05, 1.02, 0.91),
        grass_contact_shadow_max_alpha_u8=15,
        grass_contact_shadow_sigma_px=3.0,
        room_depth_clean=RoomDepthCleanConfig(
            alpha_threshold_u8=16,
            mask_dilation_px=20,
            post_smooth_sigma_px=2.8,
            interior_blend_band_px=11.0,
        ),
        mid_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=s19_bg_z,
            room_depth_amplitude=s19_room_amp,
            room_pin_margin_uv=room_pin,
            layer_z=-350.0,
            layer_depth_amplitude=165.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=8,
            min_component_area_px=600,
            contact_boxes=(
                ContactAnchorBox(x0=365, y0=680, x1=520, y1=756, mode="bottom_band", band_px=7),
                ContactAnchorBox(x0=590, y0=590, x1=990, y1=662, mode="bottom_band", band_px=7),
            ),
            neural_relief_amplitude=22.0,
            contact_relief_taper_px=9.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.2,
            interior_smooth_sigma_px=1.6,
        ),
        earth_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=s19_bg_z,
            room_depth_amplitude=s19_room_amp,
            room_pin_margin_uv=room_pin,
            layer_z=-350.0,
            layer_depth_amplitude=165.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=6,
            min_component_area_px=800,
            contact_boxes=(
                ContactAnchorBox(x0=0, y0=824, x1=1365, y1=950, mode="bottom_band", band_px=5),
                ContactAnchorBox(x0=0, y0=824, x1=1365, y1=950, mode="top_band", band_px=4),
            ),
            neural_relief_amplitude=5.8,
            contact_relief_taper_px=6.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.2,
            interior_smooth_sigma_px=1.2,
        ),
        roll_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=s19_bg_z,
            room_depth_amplitude=s19_room_amp,
            room_pin_margin_uv=room_pin,
            layer_z=-350.0,
            layer_depth_amplitude=165.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=6,
            min_component_area_px=500,
            contact_boxes=(
                ContactAnchorBox(x0=105, y0=905, x1=365, y1=952, mode="bottom_band", band_px=6),
            ),
            neural_relief_amplitude=26.0,
            contact_relief_taper_px=8.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.2,
            interior_smooth_sigma_px=1.4,
        ),
        camera_rig=SceneCameraRigSpec(
            min_inner_scale=1.94,
            initial_camera_y=-6.0,
            initial_camera_z=535.0,
            min_camera_z=-85.0,
            max_camera_z=695.0,
            max_yaw_deg=26.0,
            max_pitch_deg=16.0,
            max_strafe_x=135.0,
            max_strafe_y=85.0,
            dolly_speed_mult=1.3,
            particle_color_hex="0xE8DFC2",
            particle_count=60,
            portrait_initial_yaw_deg=21.5,
        ),
    )

    return ScrollPurimCarpetBundleConfig(
        inner_dimensions=CanvasDimensionsConfig(width=1500, height=1000),
        expanded_dimensions=CanvasDimensionsConfig(width=2700, height=1800),
        inner_offset_xy=(600, 400),
        scenes_output_relpath="prototype/assets/scenes",
        spec_output_abspath="/Users/talsheaffer/.gemini/antigravity/brain/235713ae-3cfb-48f0-94e4-2012c0824fd5/scratch/sub2_scroll_purim_spec.json",
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
            min_component_area_px=350,
            sam_morph_close_px=5,
            edge_feather_sigma=0.65,
        ),
        scene_inpainting=InpaintingConfig(
            dilation_radius_px=26,
            mask_blur_sigma=3.5,
            jpeg_quality=94,
        ),
        lama_max_working_side_px=1620,
        lama_roi_context_margin_px=180,
        edt_rim_probe_px=6,
        max_edt_rim_jump_world=2.0,
        topology_verification=LayerTopologyVerificationConfig(
            alpha_threshold_u8=20,
            rim_erosion_px=2,
            rim_exterior_probe_px=6,
            max_allowed_contact_dz_world=2.2,
            max_allowed_p95_contact_dz_world=1.4,
            max_allowed_rim_jump_world=2.0,
            host_monotonicity_tolerance_world=1.8,
        ),
        bene_beitcha_scroll_foot=scroll_foot_spec,
        purim_costume_1966=purim_spec,
        herzliya_lawn_carpet=herz_carpet_spec,
    )


def build_all_scroll_purim_carpet_scenes(
    project_root: Path | None = None,
    config: ScrollPurimCarpetBundleConfig | None = None,
    only_scene: str | None = None,
) -> dict[str, Any]:
    root = project_root or _REPO_ROOT
    cfg = config or default_scroll_purim_carpet_bundle_config()

    roundtrip = ScrollPurimCarpetBundleConfig.from_json(cfg.to_json())
    if roundtrip != cfg:
        raise RuntimeError("ScrollPurimCarpetBundleConfig failed lossless JSON round-trip check.")

    device = select_torch_device()
    print(f"Initializing neural vision models on device='{device}'...", flush=True)
    depth_estimator = DepthAnythingEstimator(cfg.depth_model, device=device)
    sam_extractor = SamBoxMaskExtractor(cfg.segmentation)
    matte_extractor = SalientMatteExtractor(cfg.segmentation)
    lama_inpainter = LamaOcclusionInpainter(cfg.scene_inpainting)

    spec_out = Path(cfg.spec_output_abspath)
    existing_scenes: dict[str, Any] = {}
    if only_scene and spec_out.is_file():
        prev = json.loads(spec_out.read_text(encoding="utf-8"))
        existing_scenes = dict(prev.get("scenes", {}))

    scenes_out: dict[str, Any] = dict(existing_scenes)
    if only_scene in (None, "bene_beitcha_scroll_foot"):
        s17 = build_bene_beitcha_scroll_foot_scene(
            project_root=root,
            cfg=cfg,
            depth_estimator=depth_estimator,
            sam_extractor=sam_extractor,
            lama_inpainter=lama_inpainter,
        )
        scenes_out[s17["id"]] = s17

    if only_scene in (None, "purim_costume_1966"):
        s18 = build_purim_costume_1966_scene(
            project_root=root,
            cfg=cfg,
            depth_estimator=depth_estimator,
            sam_extractor=sam_extractor,
            matte_extractor=matte_extractor,
            lama_inpainter=lama_inpainter,
        )
        scenes_out[s18["id"]] = s18

    if only_scene in (None, "herzliya_lawn_carpet"):
        s19 = build_herzliya_lawn_carpet_scene(
            project_root=root,
            cfg=cfg,
            depth_estimator=depth_estimator,
            sam_extractor=sam_extractor,
            matte_extractor=matte_extractor,
            lama_inpainter=lama_inpainter,
        )
        scenes_out[s19["id"]] = s19

    payload = {
        "schemaVersion": "1.0",
        "workstream": "Subagent B — Scroll Foot + Purim 1966 (#40) & Herzliya Lawn Unrolling Carpet (#41)",
        "generatedBy": "prototype/scroll_purim_carpet_scenes_builder.py",
        "ssotConfig": cfg.to_dict(),
        "scenes": scenes_out,
        "manifestEntries": scenes_out,
        "sceneConfigs": scenes_out,
    }
    spec_out.parent.mkdir(parents=True, exist_ok=True)
    spec_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] Exported Subagent B scene specs & topology metrics -> {spec_out}", flush=True)
    return payload


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Scenes 17, 18, and 19 neural 2.5D spatial assets and SSOT specs."
    )
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Validate lossless SSOT config serialization without running heavy neural inference.",
    )
    parser.add_argument(
        "--only-scene",
        type=str,
        default=None,
        choices=["bene_beitcha_scroll_foot", "purim_costume_1966", "herzliya_lawn_carpet"],
        help="Optionally rebuild only a single scene while preserving existing entries.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli_args()
    cfg_main = default_scroll_purim_carpet_bundle_config()
    if ScrollPurimCarpetBundleConfig.from_json(cfg_main.to_json()) != cfg_main:
        raise RuntimeError("ScrollPurimCarpetBundleConfig failed lossless JSON round-trip check.")
    if args.dry_run_config:
        print("ScrollPurimCarpetBundleConfig lossless JSON round-trip OK.")
    else:
        build_all_scroll_purim_carpet_scenes(
            config=cfg_main,
            only_scene=args.only_scene,
        )
