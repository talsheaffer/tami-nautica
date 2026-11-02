#!/usr/bin/env python3
"""
prototype/humpty_booth_scene_builder.py
Offline neural 2.5D spatial asset builder and typed SSOT configuration models for Scene 16:
  - Surgical bottom-right Gemini sparkle watermark removal via neural `LaMa`:
      `/Users/talsheaffer/Downloads/Gemini_Generated_Image_ (2).png`
      -> `prototype/assets/sources/humpty_dumpty_booth_clean.png` (>99.8% bit-identical).
  - Scene 16 (`humpty_dumpty_booth` / `scene16_humpty_booth_*`, inner `1500x1000`, expanded `2700x1800`):
      Tamar "Tami" Sheaffer's *המפטי דמפטי בתא האפור* (Humpty Dumpty seated at a pink-gingham
      teacup table inside a dark-gray wooden booth with a rear metal cage fan in the studio).
      Decomposed into genuine 3-layer 2.5D depth topology (`boreSightAtRest: true`):
        1. `bg` (`2700x1800`): Recessed dark-gray booth interior cavity + rear metal cage fan +
           seamlessly extended studio wall, wood rail, ceiling, and floor via neural `LaMa`.
        2. `mid` (`1500x1000`): Humpty Dumpty + carved wooden chair + pink-gingham table, plate,
           teacup, and oak table leg/apron (`midPinMarginUv = 0.0`), $C^0$ contact-anchored at
           chair/table feet & booth wall with sculptural 3D relief and Voronoi `EDT` extrapolation.
        3. `fg` (`1500x1000`): Front outer wooden booth frame/aperture (`fgPinMarginUv = 0.0`),
           $C^0$ contact-anchored at left & right frame posts to the booth threshold depth.

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
        "`/Users/talsheaffer/code/tami/.venv/bin/python -u prototype/humpty_booth_scene_builder.py`."
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
    extract_contact_anchor_mask,
    extract_layer_window_from_expanded,
    solve_contact_anchored_layer_depth,
    verify_layer_contact_and_discontinuity,
)
from prototype.new_scenes_builder import _inpaint_masked_plate_lama
from prototype.scene_config import (
    BoundingBoxPrompt,
    CanvasDimensionsConfig,
    DepthModelConfig,
    InpaintingConfig,
    SamPointGroupPrompt,
    SegmentationConfig,
)
from prototype.scroll_humpty_scenes_builder import (
    SceneCameraRigSpec,
    WatermarkCleanTargetSpec,
    _compute_auto_floor_p95_dz_world,
    _compute_edt_rim_p99_jump_world,
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
class BoothFrameGeometrySpec:
    """Exact carpentry corridor & aperture bounds on the `1500x1000` inner canvas."""

    left_outer_top_xy: tuple[float, float]
    left_outer_bot_xy: tuple[float, float]
    left_inner_top_xy: tuple[float, float]
    left_inner_bot_xy: tuple[float, float]
    lintel_outer_left_xy: tuple[float, float]
    lintel_outer_right_xy: tuple[float, float]
    lintel_inner_left_xy: tuple[float, float]
    lintel_inner_right_xy: tuple[float, float]
    right_outer_top_xy: tuple[float, float]
    right_outer_bot_xy: tuple[float, float]
    right_inner_top_xy: tuple[float, float]
    right_inner_bot_xy: tuple[float, float]
    ceiling_bright_luma_min: float
    studio_wall_luma_min: float
    edge_feather_sigma_px: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_outer_top_xy": list(self.left_outer_top_xy),
            "left_outer_bot_xy": list(self.left_outer_bot_xy),
            "left_inner_top_xy": list(self.left_inner_top_xy),
            "left_inner_bot_xy": list(self.left_inner_bot_xy),
            "lintel_outer_left_xy": list(self.lintel_outer_left_xy),
            "lintel_outer_right_xy": list(self.lintel_outer_right_xy),
            "lintel_inner_left_xy": list(self.lintel_inner_left_xy),
            "lintel_inner_right_xy": list(self.lintel_inner_right_xy),
            "right_outer_top_xy": list(self.right_outer_top_xy),
            "right_outer_bot_xy": list(self.right_outer_bot_xy),
            "right_inner_top_xy": list(self.right_inner_top_xy),
            "right_inner_bot_xy": list(self.right_inner_bot_xy),
            "ceiling_bright_luma_min": self.ceiling_bright_luma_min,
            "studio_wall_luma_min": self.studio_wall_luma_min,
            "edge_feather_sigma_px": self.edge_feather_sigma_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoothFrameGeometrySpec:
        def _pt(key: str) -> tuple[float, float]:
            v = data[key]
            return (float(v[0]), float(v[1]))

        return cls(
            left_outer_top_xy=_pt("left_outer_top_xy"),
            left_outer_bot_xy=_pt("left_outer_bot_xy"),
            left_inner_top_xy=_pt("left_inner_top_xy"),
            left_inner_bot_xy=_pt("left_inner_bot_xy"),
            lintel_outer_left_xy=_pt("lintel_outer_left_xy"),
            lintel_outer_right_xy=_pt("lintel_outer_right_xy"),
            lintel_inner_left_xy=_pt("lintel_inner_left_xy"),
            lintel_inner_right_xy=_pt("lintel_inner_right_xy"),
            right_outer_top_xy=_pt("right_outer_top_xy"),
            right_outer_bot_xy=_pt("right_outer_bot_xy"),
            right_inner_top_xy=_pt("right_inner_top_xy"),
            right_inner_bot_xy=_pt("right_inner_bot_xy"),
            ceiling_bright_luma_min=float(data["ceiling_bright_luma_min"]),
            studio_wall_luma_min=float(data["studio_wall_luma_min"]),
            edge_feather_sigma_px=float(data["edge_feather_sigma_px"]),
        )


@dataclass(frozen=True)
class HumptyBoothSceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    asset_prefix: str
    clean_source_relpath: str
    scaled_source_size_wh: tuple[int, int]
    dest_xy_inner: tuple[int, int]
    bg_jpeg_quality: int
    dado_rail_y_inner: tuple[int, int]
    studio_floor_horizon_y_inner: int
    fan_preserve_box_inner: BoundingBoxPrompt
    humpty_head_crown_box_inner: BoundingBoxPrompt
    humpty_head_min_r_minus_b: float
    under_table_cavity_box_inner: BoundingBoxPrompt
    cavity_dark_luma_max: float
    cavity_dark_sat_max: float
    shoe_protect_boxes_inner: tuple[BoundingBoxPrompt, ...]
    fg_frame_geometry: BoothFrameGeometrySpec
    fg_sam_prompts: tuple[SamPointGroupPrompt, ...]
    mid_sam_prompts: tuple[SamPointGroupPrompt, ...]
    min_component_area_px: int
    bg_inpaint_dilation_px: int
    outpaint_seam_halfwidth_px: int
    back_wall_depth_u8: float
    fan_relief_depth_u8: float
    booth_threshold_depth_u8: float
    studio_wall_depth_u8: float
    studio_floor_front_depth_u8: float
    da2_room_blend_weight: float
    room_depth_clean: RoomDepthCleanConfig
    mid_depth_solve: ContactAnchoredLayerDepthConfig
    fg_depth_solve: ContactAnchoredLayerDepthConfig
    camera_rig: SceneCameraRigSpec
    fg_bleed_edges: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "asset_prefix": self.asset_prefix,
            "clean_source_relpath": self.clean_source_relpath,
            "scaled_source_size_wh": list(self.scaled_source_size_wh),
            "dest_xy_inner": list(self.dest_xy_inner),
            "bg_jpeg_quality": self.bg_jpeg_quality,
            "dado_rail_y_inner": list(self.dado_rail_y_inner),
            "studio_floor_horizon_y_inner": self.studio_floor_horizon_y_inner,
            "fan_preserve_box_inner": self.fan_preserve_box_inner.to_dict(),
            "humpty_head_crown_box_inner": self.humpty_head_crown_box_inner.to_dict(),
            "humpty_head_min_r_minus_b": self.humpty_head_min_r_minus_b,
            "under_table_cavity_box_inner": self.under_table_cavity_box_inner.to_dict(),
            "cavity_dark_luma_max": self.cavity_dark_luma_max,
            "cavity_dark_sat_max": self.cavity_dark_sat_max,
            "shoe_protect_boxes_inner": [b.to_dict() for b in self.shoe_protect_boxes_inner],
            "fg_frame_geometry": self.fg_frame_geometry.to_dict(),
            "fg_sam_prompts": [p.to_dict() for p in self.fg_sam_prompts],
            "mid_sam_prompts": [p.to_dict() for p in self.mid_sam_prompts],
            "min_component_area_px": self.min_component_area_px,
            "bg_inpaint_dilation_px": self.bg_inpaint_dilation_px,
            "outpaint_seam_halfwidth_px": self.outpaint_seam_halfwidth_px,
            "back_wall_depth_u8": self.back_wall_depth_u8,
            "fan_relief_depth_u8": self.fan_relief_depth_u8,
            "booth_threshold_depth_u8": self.booth_threshold_depth_u8,
            "studio_wall_depth_u8": self.studio_wall_depth_u8,
            "studio_floor_front_depth_u8": self.studio_floor_front_depth_u8,
            "da2_room_blend_weight": self.da2_room_blend_weight,
            "room_depth_clean": self.room_depth_clean.to_dict(),
            "mid_depth_solve": self.mid_depth_solve.to_dict(),
            "fg_depth_solve": self.fg_depth_solve.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
            "fg_bleed_edges": list(self.fg_bleed_edges),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumptyBoothSceneSpec:
        ss = data["scaled_source_size_wh"]
        di = data["dest_xy_inner"]
        dr = data["dado_rail_y_inner"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            asset_prefix=str(data["asset_prefix"]),
            clean_source_relpath=str(data["clean_source_relpath"]),
            scaled_source_size_wh=(int(ss[0]), int(ss[1])),
            dest_xy_inner=(int(di[0]), int(di[1])),
            bg_jpeg_quality=int(data["bg_jpeg_quality"]),
            dado_rail_y_inner=(int(dr[0]), int(dr[1])),
            studio_floor_horizon_y_inner=int(data["studio_floor_horizon_y_inner"]),
            fan_preserve_box_inner=BoundingBoxPrompt.from_dict(data["fan_preserve_box_inner"]),
            humpty_head_crown_box_inner=BoundingBoxPrompt.from_dict(
                data["humpty_head_crown_box_inner"]
            ),
            humpty_head_min_r_minus_b=float(data["humpty_head_min_r_minus_b"]),
            under_table_cavity_box_inner=BoundingBoxPrompt.from_dict(
                data["under_table_cavity_box_inner"]
            ),
            cavity_dark_luma_max=float(data["cavity_dark_luma_max"]),
            cavity_dark_sat_max=float(data["cavity_dark_sat_max"]),
            shoe_protect_boxes_inner=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["shoe_protect_boxes_inner"]
            ),
            fg_frame_geometry=BoothFrameGeometrySpec.from_dict(data["fg_frame_geometry"]),
            fg_sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["fg_sam_prompts"]),
            mid_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["mid_sam_prompts"]
            ),
            min_component_area_px=int(data["min_component_area_px"]),
            bg_inpaint_dilation_px=int(data["bg_inpaint_dilation_px"]),
            outpaint_seam_halfwidth_px=int(data["outpaint_seam_halfwidth_px"]),
            back_wall_depth_u8=float(data["back_wall_depth_u8"]),
            fan_relief_depth_u8=float(data["fan_relief_depth_u8"]),
            booth_threshold_depth_u8=float(data["booth_threshold_depth_u8"]),
            studio_wall_depth_u8=float(data["studio_wall_depth_u8"]),
            studio_floor_front_depth_u8=float(data["studio_floor_front_depth_u8"]),
            da2_room_blend_weight=float(data["da2_room_blend_weight"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
            mid_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["mid_depth_solve"]),
            fg_depth_solve=ContactAnchoredLayerDepthConfig.from_dict(data["fg_depth_solve"]),
            camera_rig=SceneCameraRigSpec.from_dict(data["camera_rig"]),
            fg_bleed_edges=tuple(str(e) for e in data["fg_bleed_edges"]),
        )


@dataclass(frozen=True)
class HumptyBoothBundleConfig:
    scenes_output_relpath: str
    inner_dimensions: CanvasDimensionsConfig
    expanded_dimensions: CanvasDimensionsConfig
    lama_max_working_side_px: int
    lama_roi_context_margin_px: int
    edt_rim_probe_px: int
    max_edt_rim_jump_world: float
    depth_model: DepthModelConfig
    segmentation: SegmentationConfig
    watermark_inpainting: InpaintingConfig
    scene_inpainting: InpaintingConfig
    topology_verification: LayerTopologyVerificationConfig
    watermark_target: WatermarkCleanTargetSpec
    scene: HumptyBoothSceneSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenes_output_relpath": self.scenes_output_relpath,
            "inner_dimensions": self.inner_dimensions.to_dict(),
            "expanded_dimensions": self.expanded_dimensions.to_dict(),
            "lama_max_working_side_px": self.lama_max_working_side_px,
            "lama_roi_context_margin_px": self.lama_roi_context_margin_px,
            "edt_rim_probe_px": self.edt_rim_probe_px,
            "max_edt_rim_jump_world": self.max_edt_rim_jump_world,
            "depth_model": self.depth_model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "watermark_inpainting": self.watermark_inpainting.to_dict(),
            "scene_inpainting": self.scene_inpainting.to_dict(),
            "topology_verification": self.topology_verification.to_dict(),
            "watermark_target": self.watermark_target.to_dict(),
            "scene": self.scene.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumptyBoothBundleConfig:
        return cls(
            scenes_output_relpath=str(data["scenes_output_relpath"]),
            inner_dimensions=CanvasDimensionsConfig.from_dict(data["inner_dimensions"]),
            expanded_dimensions=CanvasDimensionsConfig.from_dict(data["expanded_dimensions"]),
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
            watermark_target=WatermarkCleanTargetSpec.from_dict(data["watermark_target"]),
            scene=HumptyBoothSceneSpec.from_dict(data["scene"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> HumptyBoothBundleConfig:
        return cls.from_dict(json.loads(raw))


def _pg(
    box: tuple[int, int, int, int],
    pos: Sequence[tuple[float, float]],
    neg: Sequence[tuple[float, float]],
) -> SamPointGroupPrompt:
    return SamPointGroupPrompt(
        box=BoundingBoxPrompt(*box),
        points_xy=tuple(pos) + tuple(neg),
        point_labels=tuple([1] * len(pos) + [0] * len(neg)),
    )


def default_humpty_booth_bundle_config() -> HumptyBoothBundleConfig:
    inner = CanvasDimensionsConfig(width=1500, height=1000)
    expanded = CanvasDimensionsConfig(width=2700, height=1800)

    wm_target = WatermarkCleanTargetSpec(
        name="humpty_dumpty_booth_clean",
        source_abspath="/Users/talsheaffer/Downloads/Gemini_Generated_Image_ (2).png",
        output_relpath="prototype/assets/sources/humpty_dumpty_booth_clean.png",
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
    )

    frame_geom = BoothFrameGeometrySpec(
        left_outer_top_xy=(405.0, 0.0),
        left_outer_bot_xy=(416.0, 1000.0),
        left_inner_top_xy=(497.0, 73.0),
        left_inner_bot_xy=(515.0, 1000.0),
        lintel_outer_left_xy=(524.0, 0.0),
        lintel_outer_right_xy=(1052.0, 148.0),
        lintel_inner_left_xy=(497.0, 73.0),
        lintel_inner_right_xy=(1026.0, 178.0),
        right_outer_top_xy=(1052.0, 148.0),
        right_outer_bot_xy=(974.0, 1000.0),
        right_inner_top_xy=(1026.0, 178.0),
        right_inner_bot_xy=(954.0, 1000.0),
        ceiling_bright_luma_min=168.0,
        studio_wall_luma_min=136.0,
        edge_feather_sigma_px=0.55,
    )

    fg_prompts = (
        # Left exterior brown wood side panel + front-left vertical dark-gray booth post
        _pg(
            (395, 0, 520, 1000),
            [
                (418.0, 120.0),
                (465.0, 120.0),
                (422.0, 420.0),
                (475.0, 420.0),
                (430.0, 720.0),
                (485.0, 720.0),
                (435.0, 955.0),
                (492.0, 955.0),
            ],
            [
                (390.0, 910.0),
                (535.0, 260.0),
                (535.0, 550.0),
                (535.0, 720.0),
                (535.0, 920.0),
            ],
        ),
        # Top slanted dark-gray lintel beam & roof lip
        _pg(
            (485, 0, 1060, 192),
            [
                (515.0, 35.0),
                (635.0, 62.0),
                (765.0, 95.0),
                (895.0, 128.0),
                (1015.0, 158.0),
            ],
            [
                (680.0, 18.0),
                (850.0, 45.0),
                (995.0, 80.0),
                (650.0, 150.0),
                (850.0, 175.0),
            ],
        ),
        # Front-right vertical dark-gray booth frame post
        _pg(
            (948, 142, 1056, 1000),
            [
                (1038.0, 185.0),
                (1022.0, 360.0),
                (1004.0, 540.0),
                (988.0, 720.0),
                (974.0, 885.0),
                (966.0, 975.0),
            ],
            [
                (935.0, 320.0),
                (935.0, 540.0),
                (935.0, 850.0),
                (1054.0, 260.0),
                (1032.0, 550.0),
                (1004.0, 800.0),
                (990.0, 950.0),
            ],
        ),
    )

    mid_prompts = (
        # Humpty Dumpty: sculptural beige egg head, blue vest, white sleeves, hands, silver fork
        _pg(
            (548, 442, 766, 655),
            [
                (645.0, 475.0),
                (615.0, 515.0),
                (665.0, 520.0),
                (648.0, 560.0),
                (585.0, 585.0),
                (645.0, 595.0),
                (725.0, 585.0),
                (612.0, 625.0),
                (705.0, 615.0),
            ],
            [
                (565.0, 448.0),
                (542.0, 468.0),
                (715.0, 470.0),
                (755.0, 520.0),
            ],
        ),
        # Carved dark-brown wooden chair backrest (strictly below rear cage fan)
        _pg(
            (506, 480, 604, 642),
            [
                (576.0, 502.0),
                (545.0, 528.0),
                (524.0, 562.0),
                (555.0, 565.0),
                (526.0, 612.0),
            ],
            [
                (555.0, 462.0),
                (525.0, 492.0),
                (492.0, 560.0),
            ],
        ),
        # Pink-and-white gingham tablecloth, porcelain plate, teacup, oak apron & front-right leg
        _pg(
            (508, 590, 938, 1000),
            [
                (555.0, 665.0),
                (665.0, 642.0),
                (775.0, 625.0),
                (855.0, 650.0),
                (914.0, 705.0),
                (560.0, 745.0),
                (715.0, 718.0),
                (848.0, 755.0),
                (842.0, 865.0),
                (836.0, 965.0),
                (516.0, 825.0),
            ],
            [
                (655.0, 845.0),
                (755.0, 875.0),
                (555.0, 915.0),
                (915.0, 845.0),
                (655.0, 955.0),
            ],
        ),
        # Humpty Dumpty's dangling legs, black glossy shoes, & curved wooden chair legs
        _pg(
            (576, 730, 742, 962),
            [
                (626.0, 795.0),
                (632.0, 842.0),
                (638.0, 868.0),
                (692.0, 768.0),
                (704.0, 808.0),
                (712.0, 824.0),
                (594.0, 835.0),
                (594.0, 918.0),
                (595.0, 948.0),
                (690.0, 858.0),
                (692.0, 892.0),
            ],
            [
                (562.0, 860.0),
                (612.0, 895.0),
                (662.0, 825.0),
                (662.0, 910.0),
                (735.0, 885.0),
            ],
        ),
    )

    scene_spec = HumptyBoothSceneSpec(
        scene_id="humpty_dumpty_booth",
        title_he="המפטי דמפטי בתא האפור — סעודת יחיד בסטודיו",
        short_label_he="המפטי דמפטי בתא",
        icon="",
        asset_prefix="scene16_humpty_booth",
        clean_source_relpath="prototype/assets/sources/humpty_dumpty_booth_clean.png",
        scaled_source_size_wh=(730, 1000),
        dest_xy_inner=(385, 0),
        bg_jpeg_quality=95,
        dado_rail_y_inner=(504, 540),
        studio_floor_horizon_y_inner=740,
        fan_preserve_box_inner=BoundingBoxPrompt(x0=504, y0=272, x1=648, y1=480),
        humpty_head_crown_box_inner=BoundingBoxPrompt(x0=582, y0=442, x1=708, y1=484),
        humpty_head_min_r_minus_b=24.0,
        under_table_cavity_box_inner=BoundingBoxPrompt(x0=524, y0=748, x1=812, y1=1000),
        cavity_dark_luma_max=24.0,
        cavity_dark_sat_max=68.0,
        shoe_protect_boxes_inner=(
            BoundingBoxPrompt(x0=616, y0=832, x1=656, y1=886),
            BoundingBoxPrompt(x0=684, y0=792, x1=734, y1=846),
        ),
        fg_frame_geometry=frame_geom,
        fg_sam_prompts=fg_prompts,
        mid_sam_prompts=mid_prompts,
        min_component_area_px=120,
        bg_inpaint_dilation_px=16,
        outpaint_seam_halfwidth_px=26,
        back_wall_depth_u8=26.0,
        fan_relief_depth_u8=54.0,
        booth_threshold_depth_u8=194.0,
        studio_wall_depth_u8=168.0,
        studio_floor_front_depth_u8=222.0,
        da2_room_blend_weight=0.22,
        room_depth_clean=RoomDepthCleanConfig(
            alpha_threshold_u8=16,
            mask_dilation_px=16,
            extra_clean_boxes_expanded=(),
            preserve_boxes_expanded=((1100, 675, 1232, 878),),
            post_smooth_sigma_px=3.0,
            interior_blend_band_px=10.0,
        ),
        mid_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=-350.0,
            room_depth_amplitude=145.0,
            room_pin_margin_uv=0.045,
            layer_z=-350.0,
            layer_depth_amplitude=175.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=18,
            min_component_area_px=90,
            auto_floor_band_px=6,
            auto_floor_max_rise_px=48,
            contact_boxes=(
                ContactAnchorBox(x0=580, y0=936, x1=610, y1=962, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=718, y0=874, x1=748, y1=904, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=800, y0=980, x1=884, y1=1000, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=506, y0=636, x1=528, y1=804, mode="left_band", band_px=6),
            ),
            neural_relief_amplitude=36.0,
            contact_relief_taper_px=2.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.0,
            interior_smooth_sigma_px=2.2,
            overflow_tolerance_world=0.5,
        ),
        fg_depth_solve=ContactAnchoredLayerDepthConfig(
            bg_z=-350.0,
            room_depth_amplitude=145.0,
            room_pin_margin_uv=0.045,
            layer_z=-350.0,
            layer_depth_amplitude=195.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            component_bridge_px=16,
            min_component_area_px=120,
            auto_floor_band_px=6,
            auto_floor_max_rise_px=36,
            contact_boxes=(
                ContactAnchorBox(x0=412, y0=978, x1=518, y1=1000, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=950, y0=978, x1=978, y1=1000, mode="bottom_band", band_px=6),
            ),
            neural_relief_amplitude=28.0,
            contact_relief_taper_px=2.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.0,
            interior_smooth_sigma_px=2.5,
            overflow_tolerance_world=0.5,
        ),
        camera_rig=SceneCameraRigSpec(
            min_inner_scale=1.94,
            initial_camera_y=-6.0,
            initial_camera_z=525.0,
            min_camera_z=-90.0,
            max_camera_z=685.0,
            max_yaw_deg=22.0,
            max_pitch_deg=15.0,
            max_strafe_x=125.0,
            max_strafe_y=78.0,
            dolly_speed_mult=1.32,
            particle_color_hex="0xE2D7C5",
            particle_count=55,
        ),
        fg_bleed_edges=("top", "bottom"),
    )

    return HumptyBoothBundleConfig(
        scenes_output_relpath="prototype/assets/scenes",
        inner_dimensions=inner,
        expanded_dimensions=expanded,
        lama_max_working_side_px=1350,
        lama_roi_context_margin_px=180,
        edt_rim_probe_px=6,
        max_edt_rim_jump_world=2.5,
        depth_model=DepthModelConfig(
            model_id="depth-anything/Depth-Anything-V2-Small-hf",
            background_gaussian_sigma=10.0,
            depth_discontinuity_percentile_fg=62.0,
            depth_discontinuity_percentile_mid=30.0,
        ),
        segmentation=SegmentationConfig(
            rembg_model_name="isnet-general-use",
            sam_model_id="facebook/sam-vit-base",
            alpha_foreground_threshold=42,
            alpha_full_opaque_threshold=185,
            min_component_area_px=120,
            sam_morph_close_px=3,
            edge_feather_sigma=0.80,
        ),
        watermark_inpainting=InpaintingConfig(
            dilation_radius_px=8,
            mask_blur_sigma=2.0,
            jpeg_quality=95,
        ),
        scene_inpainting=InpaintingConfig(
            dilation_radius_px=16,
            mask_blur_sigma=3.5,
            jpeg_quality=95,
        ),
        topology_verification=LayerTopologyVerificationConfig(
            alpha_threshold_u8=20,
            rim_erosion_px=2,
            rim_exterior_probe_px=6,
            max_allowed_contact_dz_world=2.5,
            max_allowed_p95_contact_dz_world=1.8,
            max_allowed_rim_jump_world=2.5,
            host_monotonicity_tolerance_world=2.0,
        ),
        watermark_target=wm_target,
        scene=scene_spec,
    )


# ============================================================================
# Step 1: Surgical Bottom-Right Gemini Sparkle Watermark Healing
# ============================================================================


def clean_humpty_booth_watermark(
    project_root: Path,
    target: WatermarkCleanTargetSpec,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    src_path = Path(target.source_abspath)
    if not src_path.is_file():
        raise FileNotFoundError(f"Missing Humpty Booth source image: {src_path}")

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
        ndi.gaussian_filter(dil.astype(np.float32) / 255.0, sigma=target.inpaint_blur_sigma) * 1.12,
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
    Image.fromarray(out, mode="RGB").save(out_path, format="PNG")
    print(
        f"[Watermark Clean] {target.name} ({w}x{h}) -> {target.output_relpath} "
        f"(bit_identical={identical_ratio * 100.0:.3f}%)"
    )
    return {
        "outputRelpath": target.output_relpath,
        "width": w,
        "height": h,
        "bitIdenticalPixelRatio": round(identical_ratio, 6),
    }


# ============================================================================
# Step 2: Geometry & Segmentation Primitives (`fg` Booth Frame & `mid` Humpty)
# ============================================================================


def _prune_unanchored_cutout_rgba(
    rgba_u8: np.ndarray,
    solve_cfg: ContactAnchoredLayerDepthConfig,
) -> np.ndarray:
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
            max_pinhole_fill_px=1600,
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


def _line_x_at_y(p0: tuple[float, float], p1: tuple[float, float], yy: np.ndarray) -> np.ndarray:
    dy = p1[1] - p0[1]
    if abs(dy) < 1e-6:
        return np.full_like(yy, p0[0], dtype=np.float32)
    return (p0[0] + (yy - p0[1]) * ((p1[0] - p0[0]) / dy)).astype(np.float32)


def _line_y_at_x(p0: tuple[float, float], p1: tuple[float, float], xx: np.ndarray) -> np.ndarray:
    dx = p1[0] - p0[0]
    if abs(dx) < 1e-6:
        return np.full_like(xx, p0[1], dtype=np.float32)
    return (p0[1] + (xx - p0[0]) * ((p1[1] - p0[1]) / dx)).astype(np.float32)


def _segment_booth_outer_frame_fg(
    inner_rgb: Image.Image,
    spec: HumptyBoothSceneSpec,
    sam_extractor: SamBoxMaskExtractor,
) -> np.ndarray:
    """
    Segments `scene16_humpty_booth_fg.png` (`1500x1000` RGBA): the front outer wooden booth frame
    (left brown exterior panel + front-left vertical post, slanted dark-gray top lintel, and
    front-right vertical dark-gray frame stile) using Meta SAM fused with exact carpentry aperture
    corridors so zero interior fan/table pixels or outer studio wall pipes leak onto `fg`.
    """
    iw, ih = inner_rgb.size
    inner_np = np.asarray(inner_rgb, dtype=np.uint8)
    luma = (
        0.299 * inner_np[:, :, 0].astype(np.float32)
        + 0.587 * inner_np[:, :, 1].astype(np.float32)
        + 0.114 * inner_np[:, :, 2].astype(np.float32)
    )

    sam_fg_u8 = _segment_prompt_union_u8(
        rgb_image=inner_rgb,
        prompts=spec.fg_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=spec.min_component_area_px,
    )

    g = spec.fg_frame_geometry
    yy = np.arange(ih, dtype=np.float32)[:, None]
    xx = np.arange(iw, dtype=np.float32)[None, :]

    # 1. Left brown exterior panel + front-left vertical dark-gray post
    x_left_out = _line_x_at_y(g.left_outer_top_xy, g.left_outer_bot_xy, yy)
    x_left_in = _line_x_at_y(g.left_inner_top_xy, g.left_inner_bot_xy, yy)
    left_corridor = (xx >= x_left_out) & (xx <= x_left_in)
    left_post = left_corridor & ((sam_fg_u8 >= 96) | (luma <= 160.0))

    # 2. Top slanted dark-gray lintel beam & roof fascia (strictly below bright studio ceiling)
    y_lintel_top = _line_y_at_x(g.lintel_outer_left_xy, g.lintel_outer_right_xy, xx)
    y_lintel_bot = _line_y_at_x(g.lintel_inner_left_xy, g.lintel_inner_right_xy, xx)
    lintel_corridor = (
        (xx >= g.left_inner_top_xy[0] - 4.0)
        & (xx <= g.right_outer_top_xy[0] + 1.0)
        & (yy >= np.maximum(0.0, y_lintel_top - 4.0))
        & (yy <= y_lintel_bot)
    )
    lintel_beam = lintel_corridor & (luma < g.ceiling_bright_luma_min) & (
        (sam_fg_u8 >= 80) | (yy >= y_lintel_top)
    )

    # 3. Front-right vertical dark-gray booth frame stile (razor-sharp linear outer edge)
    x_right_in = _line_x_at_y(g.right_inner_top_xy, g.right_inner_bot_xy, yy)
    x_right_out = _line_x_at_y(g.right_outer_top_xy, g.right_outer_bot_xy, yy)
    right_corridor = (
        (yy >= g.right_outer_top_xy[1] - 2.0)
        & (xx >= x_right_in)
        & (xx <= x_right_out)
    )
    right_post = right_corridor & ((sam_fg_u8 >= 96) | (luma < g.studio_wall_luma_min))

    frame_bool = left_post | lintel_beam | right_post
    # Explicitly ensure zero overlap with the recessed booth opening interior
    interior_open = (
        (xx > x_left_in + 0.5)
        & (xx < x_right_in - 0.5)
        & (yy > y_lintel_bot + 0.5)
    )
    frame_bool = frame_bool & (~interior_open)

    soft_a = ndi.gaussian_filter(frame_bool.astype(np.float32), sigma=g.edge_feather_sigma_px)
    fg_alpha_u8 = np.clip(
        np.round(np.clip((soft_a - 0.28) / 0.44, 0.0, 1.0) * 255.0), 0, 255
    ).astype(np.uint8)
    fg_alpha_u8[:, 0] = 0
    fg_alpha_u8[:, -1] = 0

    fg_rgb_clean = decontaminate_matte_edge_rgb(inner_np, fg_alpha_u8)
    return _prune_unanchored_cutout_rgba(
        np.dstack([fg_rgb_clean, fg_alpha_u8]),
        spec.fg_depth_solve,
    )


def _segment_humpty_table_chair_mid(
    inner_rgb: Image.Image,
    fg_rgba_u8: np.ndarray,
    spec: HumptyBoothSceneSpec,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
) -> np.ndarray:
    """
    Segments `scene16_humpty_booth_mid.png` (`1500x1000` RGBA): Humpty Dumpty + carved wooden
    chair + pink-gingham table, porcelain plate, teacup, and light-oak apron/leg, strictly
    excluding the front booth frame (`fg`) and the rear metal cage fan (`bg`).
    """
    iw, ih = inner_rgb.size
    inner_np = np.asarray(inner_rgb, dtype=np.uint8)
    rgb_f = inner_np.astype(np.float32)
    luma = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    r_minus_b = rgb_f[:, :, 0] - rgb_f[:, :, 2]
    hsv = cv2.cvtColor(inner_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1]

    sam_mid_u8 = _segment_prompt_union_u8(
        rgb_image=inner_rgb,
        prompts=spec.mid_sam_prompts,
        sam_extractor=sam_extractor,
        min_component_area_px=spec.min_component_area_px,
    )

    # Fuse IS-Net fine edge matting strictly within a tight collar around SAM support
    isnet_rgba = remove(inner_rgb, session=matte_extractor.session)
    isnet_a = np.asarray(isnet_rgba, dtype=np.uint8)[:, :, 3]
    sam_support = (
        cv2.dilate(
            (sam_mid_u8 >= 96).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        > 0
    )
    sam_core = (
        cv2.erode(
            (sam_mid_u8 >= 150).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        * 255
    )
    mid_raw_u8 = np.where(
        sam_support,
        np.maximum(sam_mid_u8, np.minimum(isnet_a, np.uint8(255))),
        np.uint8(0),
    )
    mid_raw_u8 = np.maximum(mid_raw_u8, sam_core)

    # Strictly protect the rear metallic cage fan (`bg`), while preserving Humpty's warm egg crown
    fb = spec.fan_preserve_box_inner
    hb = spec.humpty_head_crown_box_inner
    fan_zone = np.zeros((ih, iw), dtype=bool)
    fan_zone[fb.y0 : fb.y1, fb.x0 : fb.x1] = True
    head_crown = np.zeros((ih, iw), dtype=bool)
    head_crown[hb.y0 : hb.y1, hb.x0 : hb.x1] = (
        r_minus_b[hb.y0 : hb.y1, hb.x0 : hb.x1] >= spec.humpty_head_min_r_minus_b
    )
    mid_raw_u8[fan_zone & (~head_crown)] = 0
    mid_raw_u8[: hb.y0, :] = 0

    # Keep the dark-charcoal booth interior cavity underneath the oak table apron cleanly
    # transparent between carved brown chair legs, Humpty's warm trousers/ankles, and his two
    # glossy black oval shoes (suppressing noisy `isnet_a` dust in the dark under-table shadow)
    yy = np.arange(ih, dtype=np.float32)[:, None]
    xx = np.arange(iw, dtype=np.float32)[None, :]
    y_apron_bot = 801.0 - 0.267 * (xx - 525.0)
    under_apron_zone = (xx >= 515.0) & (xx <= 806.0) & (yy > y_apron_bot + 2.0)

    warm_wood_or_trousers = (
        (r_minus_b >= 11.5)
        & (luma >= 16.0)
        & (xx >= 516.0)
        & (xx <= 752.0)
        & (yy <= 962.0)
    )
    shoe_ovals = np.zeros((ih, iw), dtype=bool)
    for sb in spec.shoe_protect_boxes_inner:
        scx = 0.5 * (sb.x0 + sb.x1)
        scy = 0.5 * (sb.y0 + sb.y1)
        srx = max(4.0, 0.42 * (sb.x1 - sb.x0))
        sry = max(4.0, 0.44 * (sb.y1 - sb.y0))
        shoe_ovals |= (((xx - scx) / srx) ** 2 + ((yy - scy) / sry) ** 2) <= 1.0

    under_keep = warm_wood_or_trousers | (shoe_ovals & ((sam_mid_u8 >= 96) | (isnet_a >= 96)))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    under_keep_u8 = cv2.morphologyEx(
        cv2.morphologyEx(under_keep.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3),
        cv2.MORPH_CLOSE,
        k3,
    )
    mid_raw_u8 = np.where(under_apron_zone, under_keep_u8, mid_raw_u8)

    # Clip strictly inside the booth horizontal corridor `[left_inner - 1 .. 946]` & subtract `fg`
    g = spec.fg_frame_geometry
    x_left_in = _line_x_at_y(g.left_inner_top_xy, g.left_inner_bot_xy, yy)
    corridor_ok = (xx >= x_left_in - 1.5) & (xx <= 946.0)
    fg_occ_f = fg_rgba_u8[:, :, 3].astype(np.float32) / 255.0
    mid_alpha_f = (mid_raw_u8.astype(np.float32) / 255.0) * (1.0 - fg_occ_f) * corridor_ok
    mid_alpha_f = ndi.gaussian_filter(
        np.clip((mid_alpha_f - 0.30) / 0.42, 0.0, 1.0).astype(np.float32), sigma=0.50
    )
    mid_alpha_u8 = np.clip(np.round(mid_alpha_f * 255.0), 0, 255).astype(np.uint8)
    mid_alpha_u8[0, :] = 0
    mid_alpha_u8[:, 0] = 0
    mid_alpha_u8[:, -1] = 0

    mid_rgb_clean = decontaminate_matte_edge_rgb(inner_np, mid_alpha_u8)
    return _prune_unanchored_cutout_rgba(
        np.dstack([mid_rgb_clean, mid_alpha_u8]),
        spec.mid_depth_solve,
    )


# ============================================================================
# Step 3: Studio Surround Outpainting & Recessed Booth Cavity Inpainting (`bg`)
# ============================================================================


def _synthesize_uncropped_studio_inner_and_expanded(
    clean_src_rgb: Image.Image,
    spec: HumptyBoothSceneSpec,
    inner_dims: CanvasDimensionsConfig,
    exp_dims: CanvasDimensionsConfig,
    lama_inpainter: LamaOcclusionInpainter,
    max_working_side_px: int,
    roi_context_margin_px: int,
) -> tuple[Image.Image, np.ndarray]:
    """
    Places the uncropped `730x1000` scaled booth photograph onto `1500x1000` (`inner_rgb`) and
    synthesizes a seamless `2700x1800` studio wall/dado-rail/ceiling/floor surround (`seed_2700`)
    healed across boundary seams via neural `LaMa`.
    """
    iw, ih = inner_dims.width, inner_dims.height
    ew, eh = exp_dims.width, exp_dims.height
    ox, oy = (ew - iw) // 2, (eh - ih) // 2

    sw, sh = spec.scaled_source_size_wh
    dx_in, dy_in = spec.dest_xy_inner
    scaled_src = np.asarray(
        clean_src_rgb.resize((sw, sh), Image.Resampling.LANCZOS), dtype=np.uint8
    )

    px0 = ox + dx_in
    py0 = oy + dy_in
    g = spec.fg_frame_geometry
    yy_local = np.arange(sh, dtype=np.float32)[:, None]
    xx_in_sub = (np.arange(sw, dtype=np.float32) + float(dx_in))[None, :]
    x_left_out_sub = _line_x_at_y(g.left_outer_top_xy, g.left_outer_bot_xy, yy_local)
    x_right_out_sub = _line_x_at_y(g.right_outer_top_xy, g.right_outer_bot_xy, yy_local)

    # First heal the lower-right sliced storage-crate corner (`y >= 676` outside `x_right_out`)
    # directly on `scaled_src` via neural LaMa so the right-side studio strip (`sw-152 : sw-4`)
    # has 100% authentic 2D camera-captured plaster, wood dado rail, wainscoting, and floor.
    healed_scaled = scaled_src.copy()
    sliced_crate_zone = (xx_in_sub > x_right_out_sub + 2.0) & (yy_local >= 676.0)
    floor_seed_rgb = np.array([154, 147, 137], dtype=np.uint8)
    healed_scaled[sliced_crate_zone] = floor_seed_rgb
    crate_roi_y0 = 520
    crate_roi_x0 = max(0, int(np.floor(x_right_out_sub.min())) - dx_in - 48)
    crate_roi_mask = np.zeros((sh - crate_roi_y0, sw - crate_roi_x0), dtype=np.uint8)
    crate_roi_mask[sliced_crate_zone[crate_roi_y0:, crate_roi_x0:]] = 255
    # Also mask small isolated wall-hardware outlier spots on the right studio plaster strip
    # strictly outside the slanted outer booth stile + wall hardware (`x > max(x_right_out_sub) + 19`)
    strip_x0 = min(sw - 16, int(np.ceil(float((x_right_out_sub - float(dx_in)).max()))) + 20)
    strip_x1 = sw - 2
    strip_f = scaled_src[:, strip_x0:strip_x1, :].astype(np.float32)
    strip_ref = np.percentile(strip_f, 82.0, axis=1, keepdims=True)
    spot_diff = np.max(np.abs(strip_f - strip_ref), axis=2)
    rail_band = (np.arange(sh)[:, None] >= 494) & (np.arange(sh)[:, None] <= 550)
    spot_mask_strip = (spot_diff > 18.0) & (~rail_band) & (np.arange(sh)[:, None] < 676)
    if np.any(spot_mask_strip):
        spot_full = np.zeros((sh, sw), dtype=np.uint8)
        spot_full[:, strip_x0:strip_x1][spot_mask_strip] = 255
        healed_spots = np.asarray(
            lama_inpainter.inpaint_background(
                Image.fromarray(healed_scaled, mode="RGB"),
                spot_full,
                dilation_radius_px=5,
                mask_blur_sigma=1.5,
            ),
            dtype=np.uint8,
        )
        spot_dil = cv2.dilate(spot_full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        healed_scaled[spot_dil] = healed_spots[spot_dil]

    healed_crate_roi = np.asarray(
        lama_inpainter.inpaint_background(
            Image.fromarray(healed_scaled[crate_roi_y0:, crate_roi_x0:], mode="RGB"),
            crate_roi_mask,
            dilation_radius_px=8,
            mask_blur_sigma=2.5,
        ),
        dtype=np.uint8,
    )
    crate_dil = (
        cv2.dilate(crate_roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    )
    sub_crate = healed_scaled[crate_roi_y0:, crate_roi_x0:]
    sub_crate[crate_dil] = healed_crate_roi[crate_dil]
    healed_scaled[crate_roi_y0:, crate_roi_x0:] = sub_crate

    # Build the full `2700x1800` studio backdrop purely via 2D `cv2.BORDER_REFLECT_101` optical
    # mirror reflection of the authentic `148x1000` right-side studio strip anchored at its exact
    # native coordinates `[py0:py0+sh, px0+strip_x0:px0+strip_x1]` (zero 1D column extrusion!).
    strip_2d = healed_scaled[:, strip_x0:strip_x1, :]
    pad_bot = eh - py0 - sh
    top_plaster_refl = cv2.copyMakeBorder(
        strip_2d[:420, :, :], py0, 0, 0, 0, borderType=cv2.BORDER_REFLECT_101
    )[:py0, :, :]
    bot_floor_refl = cv2.copyMakeBorder(
        strip_2d[700:, :, :], 0, pad_bot, 0, 0, borderType=cv2.BORDER_REFLECT_101
    )[sh - 700 :, :, :]
    strip_full_h = np.concatenate([top_plaster_refl, strip_2d, bot_floor_refl], axis=0)

    anchor_x0 = px0 + strip_x0
    anchor_x1 = px0 + strip_x1
    studio_canvas = cv2.copyMakeBorder(
        strip_full_h,
        0,
        0,
        anchor_x0,
        ew - anchor_x1,
        borderType=cv2.BORDER_REFLECT_101,
    )

    keep_photo_mask = (xx_in_sub >= x_left_out_sub) & (xx_in_sub < float(dx_in + strip_x1))
    sub_view = studio_canvas[py0 : py0 + sh, px0 : px0 + sw]
    sub_view[keep_photo_mask] = healed_scaled[keep_photo_mask]
    studio_canvas[py0 : py0 + sh, px0 : px0 + sw] = sub_view

    # Heal ONLY the narrow boundary transitions at the top-above-booth lintel and left booth edge
    # while keeping 100% of the booth AND the crisp 2D BORDER_REFLECT_101 right/outer walls intact!
    hw = spec.outpaint_seam_halfwidth_px
    yy_exp = np.arange(eh, dtype=np.float32)[:, None]
    xx_exp = np.arange(ew, dtype=np.float32)[None, :]
    seam_mask = np.zeros((eh, ew), dtype=np.uint8)
    seam_mask[py0 - hw : py0 + 12, ox + 420 : anchor_x0 + 8] = 255
    x_left_out_exp = float(ox) + _line_x_at_y(
        g.left_outer_top_xy, g.left_outer_bot_xy, yy_exp - float(oy)
    )
    left_seam = (
        (yy_exp >= float(py0 - hw))
        & (yy_exp <= float(py0 + sh + hw))
        & (xx_exp >= x_left_out_exp - float(hw))
        & (xx_exp <= x_left_out_exp - 1.0)
    )
    seam_mask[left_seam] = 255

    healed_2700_img = _inpaint_masked_plate_lama(
        src_rgb=Image.fromarray(studio_canvas, mode="RGB"),
        binary_mask_u8=seam_mask,
        lama_inpainter=lama_inpainter,
        dilation_px=8,
        blur_sigma=3.5,
        max_working_side_px=max_working_side_px,
        roi_context_margin_px=roi_context_margin_px,
    )
    healed_2700_u8 = np.asarray(healed_2700_img, dtype=np.uint8).copy()
    # Guarantee 100% bit-identical original pixels across the entire booth & interior!
    booth_protect = (
        (xx_in_sub >= x_left_out_sub)
        & (
            xx_in_sub
            <= np.where(yy_local < 674.0, float(dx_in + strip_x0), x_right_out_sub + 1.0)
        )
    )
    sub_healed = healed_2700_u8[py0 : py0 + sh, px0 : px0 + sw]
    sub_healed[booth_protect] = scaled_src[booth_protect]
    healed_2700_u8[py0 : py0 + sh, px0 : px0 + sw] = sub_healed

    inner_rgb = Image.fromarray(
        healed_2700_u8[oy : oy + ih, ox : ox + iw],
        mode="RGB",
    )
    return inner_rgb, healed_2700_u8


def _inpaint_booth_cavity_and_frame_bg(
    master_2700_u8: np.ndarray,
    mid_rgba_u8: np.ndarray,
    fg_rgba_u8: np.ndarray,
    spec: HumptyBoothSceneSpec,
    inner_dims: CanvasDimensionsConfig,
    exp_dims: CanvasDimensionsConfig,
    lama_inpainter: LamaOcclusionInpainter,
    max_working_side_px: int,
    roi_context_margin_px: int,
) -> Image.Image:
    """
    Synthesizes `scene16_humpty_booth_bg_inpainted.jpg` (`2700x1800` RGB):
      - Keeps 100% of the rear metallic cage fan & wall plate inside the dark-gray booth untouched.
      - Removes `mid` (Humpty + chair + gingham table) from the dark booth interior via cavity-seeded
        neural `LaMa` so zero pink gingham, blue vest, or light-oak wood bleeds into the dark cavity.
      - Removes `fg` (front outer wooden booth frame) cleanly so orbiting around the frame posts
        reveals recessed dark booth walls on the inside and warm studio wall on the outside.
    """
    iw, ih = inner_dims.width, inner_dims.height
    ew, eh = exp_dims.width, exp_dims.height
    ox, oy = (ew - iw) // 2, (eh - ih) // 2

    work_2700 = master_2700_u8.copy()
    inner_view = work_2700[oy : oy + ih, ox : ox + iw].copy()

    g = spec.fg_frame_geometry
    yy_in = np.arange(ih, dtype=np.float32)[:, None]
    xx_in = np.arange(iw, dtype=np.float32)[None, :]
    x_left_in = _line_x_at_y(g.left_inner_top_xy, g.left_inner_bot_xy, yy_in)
    x_right_in = _line_x_at_y(g.right_inner_top_xy, g.right_inner_bot_xy, yy_in)
    y_lintel_bot = _line_y_at_x(g.lintel_inner_left_xy, g.lintel_inner_right_xy, xx_in)

    # Cavity interior region (including half the frame thickness so behind-frame parallax sees dark wall)
    cavity_domain = (
        (xx_in >= x_left_in - 32.0)
        & (xx_in <= x_right_in + 18.0)
        & (yy_in >= y_lintel_bot - 24.0)
    )

    fb = spec.fan_preserve_box_inner
    fan_box = np.zeros((ih, iw), dtype=bool)
    fan_box[fb.y0 : fb.y1, fb.x0 : fb.x1] = True

    mid_occ = mid_rgba_u8[:, :, 3] >= 8
    fg_occ = fg_rgba_u8[:, :, 3] >= 8
    kd = 2 * spec.bg_inpaint_dilation_px + 1
    occ_inner = (
        cv2.dilate(
            (mid_occ | fg_occ).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)),
        )
        > 0
    )
    # Never mask the unoccluded rear metallic cage fan!
    fan_safe = fan_box & (~cv2.dilate(mid_occ.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool))
    occ_inner[fan_safe] = False

    # Pre-seed inside `occ_inner & cavity_domain` with smooth authentic dark-charcoal booth interior
    # donor pixels, and `occ_inner & (~cavity_domain)` with clean studio wall/ceiling/floor donors!
    clean_cavity_donor = (
        cavity_domain
        & (~occ_inner)
        & (~fan_box)
        & (
            (
                0.299 * inner_view[:, :, 0].astype(np.float32)
                + 0.587 * inner_view[:, :, 1].astype(np.float32)
                + 0.114 * inner_view[:, :, 2].astype(np.float32)
            )
            <= 82.0
        )
    )
    _, idx_cav = ndi.distance_transform_edt(~clean_cavity_donor, return_indices=True)
    cav_fill = ndi.gaussian_filter(
        inner_view[idx_cav[0], idx_cav[1]].astype(np.float32),
        sigma=(12.0, 12.0, 0.0),
    )

    clean_studio_donor = (~cavity_domain) & (~occ_inner)
    _, idx_stu = ndi.distance_transform_edt(~clean_studio_donor, return_indices=True)
    stu_fill = ndi.gaussian_filter(
        inner_view[idx_stu[0], idx_stu[1]].astype(np.float32),
        sigma=(12.0, 12.0, 0.0),
    )

    seeded_inner = inner_view.copy()
    cav_seed_mask = occ_inner & cavity_domain
    stu_seed_mask = occ_inner & (~cavity_domain)
    seeded_inner[cav_seed_mask] = np.clip(np.round(cav_fill[cav_seed_mask]), 0, 255).astype(
        np.uint8
    )
    seeded_inner[stu_seed_mask] = np.clip(np.round(stu_fill[stu_seed_mask]), 0, 255).astype(
        np.uint8
    )

    work_2700[oy : oy + ih, ox : ox + iw] = seeded_inner

    # Extend a realistic 34px booth floor plinth directly below `y_inner = 1000` (`y_exp = 1400..1434`)
    # resting cleanly on the warm studio floor (`y_exp >= 1434`), instead of a deep black bar!
    booth_bot_strip = np.median(seeded_inner[965:998, 520:945, :], axis=(0, 1)).astype(np.float32)
    plinth_h = 34
    yy_plinth = np.linspace(0.0, 1.0, plinth_h, dtype=np.float32)[:, None, None]
    work_2700[oy + ih : oy + ih + plinth_h, ox + 495 : ox + 960] = np.clip(
        np.round(booth_bot_strip[None, None, :] * (1.0 - 0.10 * yy_plinth)), 0, 255
    ).astype(np.uint8)

    mask_2700 = np.zeros((eh, ew), dtype=np.uint8)
    mask_2700[oy : oy + ih, ox : ox + iw] = (occ_inner.astype(np.uint8)) * 255
    # Smoothly heal studio floor contact along the booth base (`y_exp = 1426..1458`)
    mask_2700[oy + ih + plinth_h - 8 : oy + ih + plinth_h + 24, ox + 405 : ox + 990] = 255
    mask_2700[oy : oy + ih, ox : ox + iw][fan_safe] = 0

    healed_bg_img = _inpaint_masked_plate_lama(
        src_rgb=Image.fromarray(work_2700, mode="RGB"),
        binary_mask_u8=mask_2700,
        lama_inpainter=lama_inpainter,
        dilation_px=6,
        blur_sigma=3.0,
        max_working_side_px=max_working_side_px,
        roi_context_margin_px=roi_context_margin_px,
    )
    healed_bg_u8 = np.asarray(healed_bg_img, dtype=np.uint8).copy()
    # Guarantee 100% bit-exact preservation of the rear metal cage fan & unoccluded booth cavity!
    unmasked = mask_2700 == 0
    healed_bg_u8[unmasked] = work_2700[unmasked]
    return Image.fromarray(healed_bg_u8, mode="RGB")


def _build_calibrated_booth_room_depth_u8(
    bg_inpainted_rgb: Image.Image,
    mid_rgba_u8: np.ndarray,
    fg_rgba_u8: np.ndarray,
    spec: HumptyBoothSceneSpec,
    inner_dims: CanvasDimensionsConfig,
    exp_dims: CanvasDimensionsConfig,
    depth_estimator: DepthAnythingEstimator,
) -> np.ndarray:
    """
    Synthesizes `scene16_humpty_booth_depth.png` (`2700x1800` uint8):
    Fuses Depth Anything V2 room depth with calibrated 3D recessed booth-cavity geometry (deep back
    wall + protruding rear cage fan + perspective side walls & floor ramping smoothly forward to
    the booth threshold) and cleans under cutouts (`clean_room_depth_under_cutouts`).
    """
    iw, ih = inner_dims.width, inner_dims.height
    ew, eh = exp_dims.width, exp_dims.height
    ox, oy = (ew - iw) // 2, (eh - ih) // 2

    da2_room_f = depth_estimator.predict_depth(bg_inpainted_rgb).astype(np.float32) * 255.0

    yy = np.arange(eh, dtype=np.float32)[:, None]
    xx = np.arange(ew, dtype=np.float32)[None, :]
    yy_in = yy - float(oy)
    xx_in = xx - float(ox)

    # Outer studio wall (`y_in <= 740`) + studio floor perspective ramp (`y_in > 740`)
    y_hor = float(spec.studio_floor_horizon_y_inner)
    t_stu_floor = np.clip((yy_in - y_hor) / max(1.0, float(eh - oy) - y_hor), 0.0, 1.0)
    studio_depth = (
        spec.studio_wall_depth_u8
        + (spec.studio_floor_front_depth_u8 - spec.studio_wall_depth_u8) * t_stu_floor
        - 10.0 * np.clip((float(oy) - yy) / float(oy), 0.0, 1.0)
    )

    # Recessed dark-gray booth box interior cavity (`x_in ≈ 485..975, y_in >= 73`):
    # Back wall (`x_in ≈ 535..895, y_in ≈ 195..795`) sits at `back_wall_depth_u8`, while left wall,
    # right wall, ceiling soffit, and booth floor (`y_in = 795..1000+`) ramp smoothly forward to
    # `booth_threshold_depth_u8` at the front frame plane!
    g = spec.fg_frame_geometry
    x_left_in = _line_x_at_y(g.left_inner_top_xy, g.left_inner_bot_xy, yy_in)
    x_right_in = _line_x_at_y(g.right_inner_top_xy, g.right_inner_bot_xy, yy_in)
    y_lintel_bot = _line_y_at_x(g.lintel_inner_left_xy, g.lintel_inner_right_xy, xx_in)

    # Side-wall & soffit perspective ramps from front aperture inward to the recessed back wall
    left_wall_t = np.clip(1.0 - (xx_in - x_left_in) / 52.0, 0.0, 1.0)
    right_wall_t = np.clip(1.0 - (x_right_in - xx_in) / 88.0, 0.0, 1.0)
    soffit_t = np.clip(1.0 - (yy_in - y_lintel_bot) / 115.0, 0.0, 1.0)
    # Booth floor ramp (`y_in = 785 -> 1000`): chair rear leg (`y=898`), chair front leg (`y=956`),
    # and front oak table leg / frame posts (`y=996`) sit at smoothly increasing forward Z!
    booth_floor_t = np.clip((yy_in - 785.0) / (996.0 - 785.0), 0.0, 1.18)

    forward_t = np.clip(
        np.maximum(np.maximum(left_wall_t, right_wall_t), np.maximum(soffit_t, booth_floor_t)),
        0.0,
        1.12,
    )
    cavity_depth = (
        spec.back_wall_depth_u8
        + (spec.booth_threshold_depth_u8 - spec.back_wall_depth_u8) * forward_t
    )

    # Protruding rear metallic cage fan inside `fan_preserve_box_inner`
    fb = spec.fan_preserve_box_inner
    fcx = float(ox) + 0.5 * (fb.x0 + fb.x1)
    fcy = float(oy) + 0.5 * (fb.y0 + 62 + fb.y1)
    fr_rad = np.sqrt(((xx - fcx) / 66.0) ** 2 + ((yy - fcy) / 68.0) ** 2)
    fan_dome = np.clip(1.0 - fr_rad**2, 0.0, 1.0)
    cavity_depth = cavity_depth + (spec.fan_relief_depth_u8 - spec.back_wall_depth_u8) * fan_dome

    # Smooth transition between recessed booth interior and outer studio walls/threshold
    in_cavity_f = (
        (xx_in >= x_left_in - 10.0)
        & (xx_in <= x_right_in + 10.0)
        & (yy_in >= y_lintel_bot - 8.0)
    ).astype(np.float32)
    in_cavity_w = ndi.gaussian_filter(in_cavity_f, sigma=9.0)
    # Ensure bottom threshold under left & right frame posts matches `booth_threshold_depth_u8`
    post_foot_w = ndi.gaussian_filter(
        (
            (yy_in >= 960.0)
            & (yy_in <= 1020.0)
            & (
                ((xx_in >= 392.0) & (xx_in <= 525.0))
                | ((xx_in >= 945.0) & (xx_in <= 1005.0))
            )
        ).astype(np.float32),
        sigma=7.0,
    )

    geom_depth = (1.0 - in_cavity_w) * studio_depth + in_cavity_w * cavity_depth
    geom_depth = (1.0 - post_foot_w) * geom_depth + post_foot_w * spec.booth_threshold_depth_u8

    w_da2 = spec.da2_room_blend_weight
    da2_smooth = ndi.gaussian_filter(da2_room_f, sigma=6.0)
    fused_u8 = np.clip(
        np.round((1.0 - w_da2) * geom_depth + w_da2 * da2_smooth), 0, 255
    ).astype(np.uint8)

    return clean_room_depth_under_cutouts(
        room_depth=fused_u8,
        cutout_layers=(mid_rgba_u8, fg_rgba_u8),
        cfg=spec.room_depth_clean,
    )


# ============================================================================
# Step 4: Verification, Composite Previews, & Top-Level Scene Builder
# ============================================================================


def _verify_and_gate_layer(
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
    if not report.all_passed:
        raise RuntimeError(f"{label} failed depth topology verification: {report.to_json()}")
    if p99_rim_jump > max_edt_rim_jump_world:
        raise RuntimeError(
            f"{label} failed outward EDT rim gate: {p99_rim_jump:.3f} > {max_edt_rim_jump_world:.3f}"
        )
    if auto_floor_p95_dz > verify_cfg.max_allowed_contact_dz_world:
        raise RuntimeError(
            f"{label} failed auto-floor contact gate: {auto_floor_p95_dz:.3f} > "
            f"{verify_cfg.max_allowed_contact_dz_world:.3f}"
        )
    out = report.to_dict()
    out["liveGateP99RimJumpWorld"] = round(p99_rim_jump, 4)
    out["autoFloorP95AbsDzWorld"] = round(auto_floor_p95_dz, 4)
    return out


def build_humpty_dumpty_booth_scene(
    project_root: Path | None = None,
    config: HumptyBoothBundleConfig | None = None,
    specs_out_path: Path | None = None,
    preview_dir: Path | None = None,
) -> dict[str, Any]:
    root = project_root or _REPO_ROOT
    cfg = config or default_humpty_booth_bundle_config()

    roundtrip = HumptyBoothBundleConfig.from_json(cfg.to_json())
    if roundtrip != cfg:
        raise RuntimeError("HumptyBoothBundleConfig failed lossless JSON round-trip check.")

    device = select_torch_device()
    print(f"[humpty_booth_scene_builder] Initializing neural models on device='{device}'...")
    depth_estimator = DepthAnythingEstimator(cfg.depth_model, device=device)
    sam_extractor = SamBoxMaskExtractor(cfg.segmentation)
    matte_extractor = SalientMatteExtractor(cfg.segmentation)
    wm_lama = LamaOcclusionInpainter(cfg.watermark_inpainting)
    scene_lama = LamaOcclusionInpainter(cfg.scene_inpainting)

    # 1. Clean bottom-right Gemini sparkle watermark into `prototype/assets/sources/humpty_dumpty_booth_clean.png`
    wm_report = clean_humpty_booth_watermark(
        project_root=root,
        target=cfg.watermark_target,
        lama_inpainter=wm_lama,
    )

    spec = cfg.scene
    inner_dims = cfg.inner_dimensions
    exp_dims = cfg.expanded_dimensions
    ox = (exp_dims.width - inner_dims.width) // 2
    oy = (exp_dims.height - inner_dims.height) // 2
    scenes_dir = root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    clean_src_img = Image.open(root / spec.clean_source_relpath).convert("RGB")

    # 2. Uncropped `1500x1000` inner composition + `2700x1800` studio wall/ceiling/floor outpainting
    print("[humpty_booth_scene_builder] Synthesizing uncropped 1500x1000 & 2700x1800 studio surround...")
    inner_rgb, master_2700_u8 = _synthesize_uncropped_studio_inner_and_expanded(
        clean_src_rgb=clean_src_img,
        spec=spec,
        inner_dims=inner_dims,
        exp_dims=exp_dims,
        lama_inpainter=scene_lama,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )

    # 3. Segment `fg` (front outer wooden booth frame) & `mid` (Humpty + chair + gingham table)
    print("[humpty_booth_scene_builder] Segmenting `fg` booth frame & `mid` Humpty/chair/table...")
    fg_rgba_u8 = _segment_booth_outer_frame_fg(
        inner_rgb=inner_rgb,
        spec=spec,
        sam_extractor=sam_extractor,
    )
    mid_rgba_u8 = _segment_humpty_table_chair_mid(
        inner_rgb=inner_rgb,
        fg_rgba_u8=fg_rgba_u8,
        spec=spec,
        sam_extractor=sam_extractor,
        matte_extractor=matte_extractor,
    )

    # 4. Inpaint `bg` (`2700x1800`) preserving rear cage fan & studio room, and build calibrated room depth
    print("[humpty_booth_scene_builder] Inpainting recessed booth cavity (`bg`) via neural LaMa...")
    bg_inpainted_rgb = _inpaint_booth_cavity_and_frame_bg(
        master_2700_u8=master_2700_u8,
        mid_rgba_u8=mid_rgba_u8,
        fg_rgba_u8=fg_rgba_u8,
        spec=spec,
        inner_dims=inner_dims,
        exp_dims=exp_dims,
        lama_inpainter=scene_lama,
        max_working_side_px=cfg.lama_max_working_side_px,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )

    print("[humpty_booth_scene_builder] Solving calibrated 2.5D depth maps & verifying topology...")
    bg_depth_clean_u8 = _build_calibrated_booth_room_depth_u8(
        bg_inpainted_rgb=bg_inpainted_rgb,
        mid_rgba_u8=mid_rgba_u8,
        fg_rgba_u8=fg_rgba_u8,
        spec=spec,
        inner_dims=inner_dims,
        exp_dims=exp_dims,
        depth_estimator=depth_estimator,
    )

    raw_intact_da2_u8 = np.clip(
        np.rint(
            depth_estimator.predict_depth(Image.fromarray(master_2700_u8, mode="RGB")) * 255.0
        ),
        0,
        255,
    ).astype(np.uint8)

    mid_protrusion = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_da2_u8,
        cutout_rgba=mid_rgba_u8,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.mid_depth_solve.alpha_threshold_u8,
    )
    mid_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=mid_rgba_u8,
        cfg=spec.mid_depth_solve,
        neural_protrusion_norm=mid_protrusion,
    )

    fg_protrusion = compute_room_relative_neural_protrusion(
        raw_neural_depth_expanded=raw_intact_da2_u8,
        cutout_rgba=fg_rgba_u8,
        clean_cfg=spec.room_depth_clean,
        alpha_threshold_u8=spec.fg_depth_solve.alpha_threshold_u8,
    )
    fg_solved = solve_contact_anchored_layer_depth(
        room_depth_expanded=bg_depth_clean_u8,
        cutout_rgba=fg_rgba_u8,
        cfg=spec.fg_depth_solve,
        neural_protrusion_norm=fg_protrusion,
    )

    mid_verify = _verify_and_gate_layer(
        label=f"{spec.scene_id}.mid",
        bg_depth_u8=bg_depth_clean_u8,
        layer_rgba_u8=mid_rgba_u8,
        layer_depth_u8=mid_solved.depth_u8,
        contact_mask=mid_solved.contact_mask,
        solve_cfg=spec.mid_depth_solve,
        verify_cfg=cfg.topology_verification,
        edt_rim_probe_px=cfg.edt_rim_probe_px,
        max_edt_rim_jump_world=cfg.max_edt_rim_jump_world,
    )
    fg_verify = _verify_and_gate_layer(
        label=f"{spec.scene_id}.fg",
        bg_depth_u8=bg_depth_clean_u8,
        layer_rgba_u8=fg_rgba_u8,
        layer_depth_u8=fg_solved.depth_u8,
        contact_mask=fg_solved.contact_mask,
        solve_cfg=spec.fg_depth_solve,
        verify_cfg=cfg.topology_verification,
        edt_rim_probe_px=cfg.edt_rim_probe_px,
        max_edt_rim_jump_world=cfg.max_edt_rim_jump_world,
    )

    # 5. Materialize Scene 16 assets to `prototype/assets/scenes/scene16_humpty_booth_*`
    pfx = spec.asset_prefix
    bg_inpainted_rgb.save(
        scenes_dir / f"{pfx}_bg_inpainted.jpg",
        format="JPEG",
        quality=spec.bg_jpeg_quality,
        subsampling=0,
        optimize=True,
    )
    Image.fromarray(bg_depth_clean_u8, mode="L").save(
        scenes_dir / f"{pfx}_depth.png", format="PNG"
    )
    Image.fromarray(mid_rgba_u8, mode="RGBA").save(
        scenes_dir / f"{pfx}_mid.png", format="PNG"
    )
    Image.fromarray(mid_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_mid_depth.png", format="PNG"
    )
    Image.fromarray(fg_rgba_u8, mode="RGBA").save(
        scenes_dir / f"{pfx}_fg.png", format="PNG"
    )
    Image.fromarray(fg_solved.depth_u8, mode="L").save(
        scenes_dir / f"{pfx}_fg_depth.png", format="PNG"
    )

    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        rest_comp = bg_inpainted_rgb.crop(
            (ox, oy, ox + inner_dims.width, oy + inner_dims.height)
        ).convert("RGBA")
        rest_comp.alpha_composite(Image.fromarray(mid_rgba_u8, mode="RGBA"))
        rest_comp.alpha_composite(Image.fromarray(fg_rgba_u8, mode="RGBA"))
        rest_comp.convert("RGB").save(
            preview_dir / "scene16_humpty_booth_rest_comp.jpg", quality=93
        )

    mid_cov = float((mid_rgba_u8[:, :, 3] >= 20).mean())
    fg_cov = float((fg_rgba_u8[:, :, 3] >= 20).mean())
    rig = spec.camera_rig

    manifest_entry = {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "icon": spec.icon,
        "bgTexture": f"assets/scenes/{pfx}_bg_inpainted.jpg",
        "depthTexture": f"assets/scenes/{pfx}_depth.png",
        "midTexture": f"assets/scenes/{pfx}_mid.png",
        "midDepthTexture": f"assets/scenes/{pfx}_mid_depth.png",
        "fgTexture": f"assets/scenes/{pfx}_fg.png",
        "fgDepthTexture": f"assets/scenes/{pfx}_fg_depth.png",
        "boreSightAtRest": True,
        "bgZ": spec.mid_depth_solve.bg_z,
        "midZ": spec.mid_depth_solve.layer_z,
        "fgZ": spec.fg_depth_solve.layer_z,
        "roomDepthAmplitude": spec.mid_depth_solve.room_depth_amplitude,
        "midDepthAmplitude": spec.mid_depth_solve.layer_depth_amplitude,
        "fgDepthAmplitude": spec.fg_depth_solve.layer_depth_amplitude,
        "midPinMarginUv": 0.0,
        "fgPinMarginUv": 0.0,
        "bgSegmentsX": 224,
        "bgSegmentsY": 150,
        "midSegmentsX": 224,
        "midSegmentsY": 150,
        "fgSegmentsX": 224,
        "fgSegmentsY": 150,
        "midDepthTest": False,
        "midDepthWrite": False,
        "fgDepthTest": False,
        "fgDepthWrite": False,
        "fgExpanded": False,
        "fgBleedEdges": {
            edge: True for edge in spec.fg_bleed_edges
        }
        | {"guardMinZ": 210.0},
        "minInnerScale": rig.min_inner_scale,
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
    }

    result = {
        "watermarkCleaning": wm_report,
        "scene": manifest_entry,
        "topologyVerification": {
            "mid": mid_verify,
            "fg": fg_verify,
        },
    }
    if specs_out_path is not None:
        specs_out_path.parent.mkdir(parents=True, exist_ok=True)
        specs_out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"[humpty_booth_scene_builder] Done! mid_cov={mid_cov:.4f} (p95_dz={mid_verify['p95_abs_contact_dz_world']:.3f}), "
        f"fg_cov={fg_cov:.4f} (p95_dz={fg_verify['p95_abs_contact_dz_world']:.3f})"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Scene 16 (`humpty_dumpty_booth` / `scene16_humpty_booth_*`)."
    )
    parser.add_argument(
        "--specs-out",
        type=Path,
        default=None,
        help="Optional JSON output path for Scene 16 manifest & verification specs.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional directory for rest-pose composite preview.",
    )
    args = parser.parse_args()
    build_humpty_dumpty_booth_scene(
        specs_out_path=args.specs_out,
        preview_dir=args.preview_dir,
    )


if __name__ == "__main__":
    main()
