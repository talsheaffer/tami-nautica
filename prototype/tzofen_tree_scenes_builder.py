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
    fg_z: float
    mid_z: float
    room_depth_amplitude: float
    fg_depth_amplitude: float
    room_pin_margin_uv: float
    window_portal_inner_box_px: tuple[float, float, float, float]
    window_frame_sample_box_px: tuple[int, int, int, int]
    portal_z_clearance_behind_frame: float
    portal_recess_depth: float
    sam_prompts: tuple[SamPointGroupPrompt, ...]
    bg_seed_boxes: tuple[BoundingBoxPrompt, ...]
    room_depth_clean: RoomDepthCleanConfig
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
            "fg_z": self.fg_z,
            "mid_z": self.mid_z,
            "room_depth_amplitude": self.room_depth_amplitude,
            "fg_depth_amplitude": self.fg_depth_amplitude,
            "room_pin_margin_uv": self.room_pin_margin_uv,
            "window_portal_inner_box_px": list(self.window_portal_inner_box_px),
            "window_frame_sample_box_px": list(self.window_frame_sample_box_px),
            "portal_z_clearance_behind_frame": self.portal_z_clearance_behind_frame,
            "portal_recess_depth": self.portal_recess_depth,
            "sam_prompts": [p.to_dict() for p in self.sam_prompts],
            "bg_seed_boxes": [b.to_dict() for b in self.bg_seed_boxes],
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
            source_image_relpath=str(data["source_image_relpath"]),
            inner_dims=CanvasDimensionsConfig.from_dict(data["inner_dims"]),
            expanded_dims=CanvasDimensionsConfig.from_dict(data["expanded_dims"]),
            scaled_height_px=int(data["scaled_height_px"]),
            inner_y_offset_px=int(data["inner_y_offset_px"]),
            baseboard_inner_y=int(data["baseboard_inner_y"]),
            jpeg_quality=int(data["jpeg_quality"]),
            bg_z=float(data["bg_z"]),
            fg_z=float(data["fg_z"]),
            mid_z=float(data["mid_z"]),
            room_depth_amplitude=float(data["room_depth_amplitude"]),
            fg_depth_amplitude=float(data["fg_depth_amplitude"]),
            room_pin_margin_uv=float(data["room_pin_margin_uv"]),
            window_portal_inner_box_px=(float(wp[0]), float(wp[1]), float(wp[2]), float(wp[3])),
            window_frame_sample_box_px=(int(wf[0]), int(wf[1]), int(wf[2]), int(wf[3])),
            portal_z_clearance_behind_frame=float(data["portal_z_clearance_behind_frame"]),
            portal_recess_depth=float(data["portal_recess_depth"]),
            sam_prompts=tuple(SamPointGroupPrompt.from_dict(p) for p in data["sam_prompts"]),
            bg_seed_boxes=tuple(BoundingBoxPrompt.from_dict(b) for b in data["bg_seed_boxes"]),
            room_depth_clean=RoomDepthCleanConfig.from_dict(data["room_depth_clean"]),
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
            [(8, 455), (1492, 465), (100, 515), (1320, 520)],
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
            [(150, 750), (310, 770), (310, 910), (565, 910), (565, 795), (410, 660)],
        ),
        _pg(
            (562, 495, 898, 955),
            [
                (636, 560),
                (740, 575),
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
            [(735, 755), (745, 850), (595, 805), (595, 900)],
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
            [(1140, 610), (920, 920), (1310, 750), (1080, 950)],
        ),
        _pg(
            (358, 495, 550, 634),
            [(369, 555), (369, 618), (420, 528), (520, 535)],
            [(348, 580), (430, 585), (395, 625)],
        ),
        _pg(
            (885, 495, 1055, 640),
            [(1042, 540), (1042, 626), (975, 565), (920, 530)],
            [(985, 525), (985, 610), (1065, 580)],
        ),
        _pg(
            (1350, 492, 1472, 602),
            [(1365, 530), (1380, 545), (1440, 545), (1452, 545)],
            [(1330, 550), (1485, 550), (1410, 620)],
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
        bg_z=-350.0,
        fg_z=-350.0,
        mid_z=-260.0,
        room_depth_amplitude=128.0,
        fg_depth_amplitude=175.0,
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
        room_depth_clean=RoomDepthCleanConfig(
            alpha_threshold_u8=16,
            mask_dilation_px=24,
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
            contact_boxes=(
                ContactAnchorBox(x0=360, y0=616, x1=380, y1=636, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=1032, y0=624, x1=1054, y1=644, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=200, y0=942, x1=226, y1=962, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=430, y0=942, x1=510, y1=962, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=624, y0=938, x1=650, y1=960, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=850, y0=934, x1=876, y1=956, mode="bottom_band", band_px=6),
                ContactAnchorBox(x0=1242, y0=928, x1=1270, y1=950, mode="bottom_band", band_px=6),
            ),
            neural_relief_amplitude=34.0,
            contact_relief_taper_px=12.0,
            enforce_front_of_host=True,
            host_clearance_tolerance_world=0.0,
            interior_smooth_sigma_px=2.2,
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
    sal_scaled = matte_ext.extract_alpha(Image.fromarray(tz_scaled, mode="RGB")).astype(np.float32) / 255.0
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
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
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
    alpha_f[y1 - 1 :, :] = 0.0
    alpha_f[:, :8] = 0.0
    alpha_f[:, iw - 8 :] = 0.0

    alpha_u8 = _prune_small_alpha_components(
        np.clip(np.round(alpha_f * 255.0), 0, 255).astype(np.uint8),
        min_area_px=48,
        alpha_threshold_u8=cfg.topology_verification.alpha_threshold_u8,
    )
    alpha_u8 = _fill_interior_alpha_pinholes(
        alpha_u8,
        max_hole_area_px=110,
        preserve_open_box_px=cfg.window_portal_inner_box_px,
        alpha_threshold_u8=cfg.topology_verification.alpha_threshold_u8,
    )
    border_max = int(
        max(
            alpha_u8[0, :].max(),
            alpha_u8[-1, :].max(),
            alpha_u8[:, 0].max(),
            alpha_u8[:, -1].max(),
        )
    )
    if border_max != 0:
        raise RuntimeError(f"Tzofen fg border alpha must be 0, got {border_max}")

    fg_rgb_clean = decontaminate_matte_edge_rgb(tz_pad, alpha_u8)
    fg_rgba = np.dstack([fg_rgb_clean, alpha_u8])
    Image.fromarray(fg_rgba, mode="RGBA").save(scenes_dir / "scene12_tzofen_fg.png")
    Image.fromarray(np.zeros((ih, iw, 4), dtype=np.uint8), mode="RGBA").save(
        scenes_dir / "scene12_tzofen_mid.png"
    )

    vent_mask = np.zeros((ih, iw), dtype=np.uint8)
    vent_mask[326:402, :195] = 255
    tz_vent_clean_img = lama.inpaint_background(tz_pad_img, vent_mask)

    lama_mask = np.where(alpha_u8 >= 10, np.uint8(255), np.uint8(0))
    lama_inner_img = lama.inpaint_background(tz_vent_clean_img, lama_mask)
    lama_inner_img.save(scenes_dir / "scene12_tzofen_lama_inner.jpg", quality=cfg.jpeg_quality)
    lama_inner_rgb = np.asarray(lama_inner_img, dtype=np.uint8)

    seed_exp = cv2.copyMakeBorder(
        lama_inner_rgb, py, py, px, px, borderType=cv2.BORDER_REFLECT_101
    )
    bg_exp_rgb = _heal_expanded_collar_lama(lama_inner_rgb, seed_exp, None, lama)
    bg_exp_img = Image.fromarray(bg_exp_rgb, mode="RGB")
    bg_exp_img.save(scenes_dir / "scene12_tzofen_bg_inpainted.jpg", quality=cfg.jpeg_quality)

    raw_room_da2 = depth_est.predict_depth(bg_exp_img)
    raw_room_u8 = np.clip(np.round(raw_room_da2 * 255.0), 0, 255).astype(np.uint8)
    clean_room_u8 = clean_room_depth_under_cutouts(
        raw_room_u8, [fg_rgba], cfg.room_depth_clean
    )
    Image.fromarray(clean_room_u8, mode="L").save(scenes_dir / "scene12_tzofen_depth.png")

    tz_excess_pos = np.maximum(0.0, da2_excess)
    tz_opaque = alpha_u8 >= cfg.fg_contact_solve.alpha_threshold_u8
    tz_prot_scale = max(1e-5, float(np.percentile(tz_excess_pos[tz_opaque], 98.0)))
    tz_prot_norm = np.clip(tz_excess_pos / tz_prot_scale, 0.0, 1.0).astype(np.float32)

    fg_solve = solve_contact_anchored_layer_depth(
        room_depth_expanded=clean_room_u8,
        cutout_rgba=fg_rgba,
        cfg=cfg.fg_contact_solve,
        neural_protrusion_norm=tz_prot_norm,
    )
    fg_depth_u8 = fg_solve.depth_u8
    Image.fromarray(fg_depth_u8, mode="L").save(scenes_dir / "scene12_tzofen_fg_depth.png")

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
    if not fg_report.all_passed:
        raise RuntimeError(f"Tzofen fg failed depth topology gate: {fg_report.to_dict()}")

    fg_world_z = fg_solve.world_z_map
    wx0, wy0, wx1, wy1 = cfg.window_frame_sample_box_px
    frame_mask = (alpha_u8[wy0:wy1, wx0:wx1] >= 200) & (
        luma[wy0:wy1, wx0:wx1] > 185.0
    )
    if not np.any(frame_mask):
        raise RuntimeError("Empty white window frame mask on Tzofen!")
    frame_z_med = float(np.median(fg_world_z[wy0:wy1, wx0:wx1][frame_mask]))
    portal_world_z = round(frame_z_med - cfg.portal_z_clearance_behind_frame, 2)

    return {
        "id": cfg.scene_id,
        "fgCoverageRatio": round(float((alpha_u8 >= 20).mean()), 4),
        "windowFrameMedianWorldZ": round(frame_z_med, 3),
        "portalWorldZ": portal_world_z,
        "portalInnerBoxPx": list(cfg.window_portal_inner_box_px),
        "portalRecessDepth": cfg.portal_recess_depth,
        "topologyVerification": {"fg": fg_report.to_dict()},
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
