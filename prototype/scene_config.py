#!/usr/bin/env python3
"""
prototype/scene_config.py
Canonical, hierarchical, serializable configuration data-models (SSOT)
for the offline neural 2.5D spatial asset pipeline and WebGL runtime contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from prototype.depth_topology import (
    ContactAnchorBox,
    ContactAnchoredLayerDepthConfig,
    LayerTopologyVerificationConfig,
    RoomDepthCleanConfig,
)

VALID_SEGMENTATION_BACKENDS = frozenset(
    {"isnet", "sam_boxes", "sam_point_groups", "graphite_paper"}
)
VALID_DEPTH_BACKENDS = frozenset({"depth_anything", "drawing_dome_relief", "pipe_tunnel_relief"})
VALID_DIFFUSE_APERTURE_ROLES = frozenset({"chamber_halo", "subject_back", "subject_front"})
VALID_OVERLAY_EXTRACTION_MODES = frozenset(
    {
        "darc_oval_island",
        "graphite_crop",
        "sam_crop",
        "rgba_asset",
        "graphite_unmix_island",
        "isnet_dehalo_capsule",
    }
)


@dataclass(frozen=True)
class CanvasDimensionsConfig:
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CanvasDimensionsConfig:
        return cls(width=int(data["width"]), height=int(data["height"]))


@dataclass(frozen=True)
class OutpaintingConfig:
    expansion_factor: float
    feather_collar_px: int
    color_match_band_px: int
    min_registration_corr: float
    homography_patch_width: int = 300
    homography_patch_height: int = 220
    homography_search_margin_px: int = 90
    homography_min_patch_corr: float = 0.70
    similarity_salient_grad_percentile: float = 82.0
    similarity_coarse_downsample: float = 0.35
    similarity_coarse_scale_min: float = 0.42
    similarity_coarse_scale_max: float = 1.15
    similarity_coarse_steps: int = 48
    similarity_fine_half_span: float = 0.045
    similarity_fine_steps: int = 19
    similarity_fine_margin_px: int = 60
    gain_clamp_min: float = 0.82
    gain_clamp_max: float = 1.22
    max_warp_border_pull_px: float = 18.0

    @property
    def expanded_scale(self) -> float:
        return self.expansion_factor

    def expanded_dimensions(self, inner: CanvasDimensionsConfig) -> tuple[int, int]:
        return (
            int(round(inner.width * self.expansion_factor)),
            int(round(inner.height * self.expansion_factor)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutpaintingConfig:
        return cls(
            expansion_factor=float(data["expansion_factor"]),
            feather_collar_px=int(data["feather_collar_px"]),
            color_match_band_px=int(data["color_match_band_px"]),
            min_registration_corr=float(data["min_registration_corr"]),
            homography_patch_width=int(data.get("homography_patch_width", 300)),
            homography_patch_height=int(data.get("homography_patch_height", 220)),
            homography_search_margin_px=int(data.get("homography_search_margin_px", 90)),
            homography_min_patch_corr=float(data.get("homography_min_patch_corr", 0.70)),
            similarity_salient_grad_percentile=float(
                data.get("similarity_salient_grad_percentile", 82.0)
            ),
            similarity_coarse_downsample=float(data.get("similarity_coarse_downsample", 0.35)),
            similarity_coarse_scale_min=float(data.get("similarity_coarse_scale_min", 0.42)),
            similarity_coarse_scale_max=float(data.get("similarity_coarse_scale_max", 1.15)),
            similarity_coarse_steps=int(data.get("similarity_coarse_steps", 48)),
            similarity_fine_half_span=float(data.get("similarity_fine_half_span", 0.045)),
            similarity_fine_steps=int(data.get("similarity_fine_steps", 19)),
            similarity_fine_margin_px=int(data.get("similarity_fine_margin_px", 60)),
            gain_clamp_min=float(data.get("gain_clamp_min", 0.82)),
            gain_clamp_max=float(data.get("gain_clamp_max", 1.22)),
            max_warp_border_pull_px=float(data.get("max_warp_border_pull_px", 18.0)),
        )


@dataclass(frozen=True)
class DepthModelConfig:
    model_id: str
    background_gaussian_sigma: float
    depth_discontinuity_percentile_fg: float
    depth_discontinuity_percentile_mid: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepthModelConfig:
        return cls(
            model_id=str(data["model_id"]),
            background_gaussian_sigma=float(data["background_gaussian_sigma"]),
            depth_discontinuity_percentile_fg=float(data["depth_discontinuity_percentile_fg"]),
            depth_discontinuity_percentile_mid=float(data["depth_discontinuity_percentile_mid"]),
        )


@dataclass(frozen=True)
class BoundingBoxPrompt:
    x0: int
    y0: int
    x1: int
    y1: int

    def as_xyxy_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    def scaled(self, scale: float) -> BoundingBoxPrompt:
        if abs(scale - 1.0) <= 1e-6:
            return self
        return BoundingBoxPrompt(
            x0=int(round(self.x0 * scale)),
            y0=int(round(self.y0 * scale)),
            x1=int(round(self.x1 * scale)),
            y1=int(round(self.y1 * scale)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundingBoxPrompt:
        return cls(
            x0=int(data["x0"]),
            y0=int(data["y0"]),
            x1=int(data["x1"]),
            y1=int(data["y1"]),
        )


@dataclass(frozen=True)
class InpaintRegionDilationOverride:
    """Per-box LaMa mask dilation radius override to preserve nearby wall relief details."""

    box: BoundingBoxPrompt
    dilation_radius_px: int

    def scaled(self, scale: float) -> InpaintRegionDilationOverride:
        if abs(scale - 1.0) <= 1e-6:
            return self
        return InpaintRegionDilationOverride(
            box=self.box.scaled(scale),
            dilation_radius_px=max(1, int(round(self.dilation_radius_px * scale))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "dilation_radius_px": self.dilation_radius_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InpaintRegionDilationOverride:
        return cls(
            box=BoundingBoxPrompt.from_dict(data["box"]),
            dilation_radius_px=int(data["dilation_radius_px"]),
        )


@dataclass(frozen=True)
class SamPointGroupPrompt:
    box: BoundingBoxPrompt
    points_xy: tuple[tuple[float, float], ...]
    point_labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.points_xy) != len(self.point_labels) or len(self.points_xy) == 0:
            raise ValueError(
                "SamPointGroupPrompt requires non-empty points_xy and point_labels of equal length."
            )

    def scaled(self, scale: float) -> SamPointGroupPrompt:
        if abs(scale - 1.0) <= 1e-6:
            return self
        return SamPointGroupPrompt(
            box=self.box.scaled(scale),
            points_xy=tuple((px * scale, py * scale) for px, py in self.points_xy),
            point_labels=self.point_labels,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "points_xy": [[float(px), float(py)] for px, py in self.points_xy],
            "point_labels": [int(lbl) for lbl in self.point_labels],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamPointGroupPrompt:
        return cls(
            box=BoundingBoxPrompt.from_dict(data["box"]),
            points_xy=tuple((float(pt[0]), float(pt[1])) for pt in data["points_xy"]),
            point_labels=tuple(int(lbl) for lbl in data["point_labels"]),
        )


@dataclass(frozen=True)
class OrientedMatteBoxSpec:
    box: BoundingBoxPrompt
    upright_angle_deg: float
    taper_margin_px: int = 10
    sam_core_erode_px: int = 9
    exclude_boxes: tuple[BoundingBoxPrompt, ...] = ()
    suppress_warm_box: BoundingBoxPrompt | None = None
    suppress_warm_rg_diff: float = 21.0
    suppress_warm_sigma: float = 2.2
    suppress_warm_gain: float = 1.35

    def scaled(self, scale: float) -> OrientedMatteBoxSpec:
        if abs(scale - 1.0) <= 1e-6:
            return self
        return OrientedMatteBoxSpec(
            box=self.box.scaled(scale),
            upright_angle_deg=self.upright_angle_deg,
            taper_margin_px=max(2, int(round(self.taper_margin_px * scale))),
            sam_core_erode_px=max(1, int(round(self.sam_core_erode_px * scale))),
            exclude_boxes=tuple(eb.scaled(scale) for eb in self.exclude_boxes),
            suppress_warm_box=(
                self.suppress_warm_box.scaled(scale)
                if self.suppress_warm_box is not None
                else None
            ),
            suppress_warm_rg_diff=self.suppress_warm_rg_diff,
            suppress_warm_sigma=self.suppress_warm_sigma * scale,
            suppress_warm_gain=self.suppress_warm_gain,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "upright_angle_deg": self.upright_angle_deg,
            "taper_margin_px": self.taper_margin_px,
            "sam_core_erode_px": self.sam_core_erode_px,
            "exclude_boxes": [eb.to_dict() for eb in self.exclude_boxes],
            "suppress_warm_box": (
                self.suppress_warm_box.to_dict() if self.suppress_warm_box is not None else None
            ),
            "suppress_warm_rg_diff": self.suppress_warm_rg_diff,
            "suppress_warm_sigma": self.suppress_warm_sigma,
            "suppress_warm_gain": self.suppress_warm_gain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrientedMatteBoxSpec:
        raw_swb = data.get("suppress_warm_box")
        return cls(
            box=BoundingBoxPrompt.from_dict(data["box"]),
            upright_angle_deg=float(data["upright_angle_deg"]),
            taper_margin_px=int(data.get("taper_margin_px", 10)),
            sam_core_erode_px=int(data.get("sam_core_erode_px", 9)),
            exclude_boxes=tuple(
                BoundingBoxPrompt.from_dict(eb) for eb in data.get("exclude_boxes", ())
            ),
            suppress_warm_box=(
                BoundingBoxPrompt.from_dict(raw_swb) if raw_swb is not None else None
            ),
            suppress_warm_rg_diff=float(data.get("suppress_warm_rg_diff", 21.0)),
            suppress_warm_sigma=float(data.get("suppress_warm_sigma", 2.2)),
            suppress_warm_gain=float(data.get("suppress_warm_gain", 1.35)),
        )


@dataclass(frozen=True)
class DrawingStrokeBox:
    x0: int
    y0: int
    x1: int
    y1: int
    min_ink_delta: float
    morph_close_px: int
    max_hole_fill_area_px: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DrawingStrokeBox:
        return cls(
            x0=int(data["x0"]),
            y0=int(data["y0"]),
            x1=int(data["x1"]),
            y1=int(data["y1"]),
            min_ink_delta=float(data["min_ink_delta"]),
            morph_close_px=int(data["morph_close_px"]),
            max_hole_fill_area_px=int(data["max_hole_fill_area_px"]),
        )


@dataclass(frozen=True)
class GraphiteDrawingConfig:
    paper_downsample_width: int
    paper_downsample_height: int
    paper_bg_percentile: int
    paper_filter_size: int
    frame_border_ignore_px: int
    min_component_area_px: int
    edge_feather_sigma: float
    lama_dilation_radius_px: int
    lama_mask_blur_sigma: float
    dome_bg_percentile: float
    dome_ink_delta_threshold: float
    dome_morph_close_px: int
    dome_power_exponent: float
    dome_gaussian_sigma: float
    stage_y_weight: float
    stage_y_exponent: float
    stage_x_weight: float
    depth_dome_weight: float
    depth_stage_weight: float
    depth_neural_weight: float
    depth_final_sigma: float
    oval_board_scaled_width: int
    oval_board_scaled_height: int
    oval_center_xy: tuple[float, float]
    oval_radii_xy: tuple[float, float]
    oval_min_component_area_px: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["oval_center_xy"] = list(self.oval_center_xy)
        d["oval_radii_xy"] = list(self.oval_radii_xy)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphiteDrawingConfig:
        oc = data["oval_center_xy"]
        orad = data["oval_radii_xy"]
        return cls(
            paper_downsample_width=int(data["paper_downsample_width"]),
            paper_downsample_height=int(data["paper_downsample_height"]),
            paper_bg_percentile=int(data["paper_bg_percentile"]),
            paper_filter_size=int(data["paper_filter_size"]),
            frame_border_ignore_px=int(data["frame_border_ignore_px"]),
            min_component_area_px=int(data["min_component_area_px"]),
            edge_feather_sigma=float(data["edge_feather_sigma"]),
            lama_dilation_radius_px=int(data["lama_dilation_radius_px"]),
            lama_mask_blur_sigma=float(data["lama_mask_blur_sigma"]),
            dome_bg_percentile=float(data["dome_bg_percentile"]),
            dome_ink_delta_threshold=float(data["dome_ink_delta_threshold"]),
            dome_morph_close_px=int(data["dome_morph_close_px"]),
            dome_power_exponent=float(data["dome_power_exponent"]),
            dome_gaussian_sigma=float(data["dome_gaussian_sigma"]),
            stage_y_weight=float(data["stage_y_weight"]),
            stage_y_exponent=float(data["stage_y_exponent"]),
            stage_x_weight=float(data["stage_x_weight"]),
            depth_dome_weight=float(data["depth_dome_weight"]),
            depth_stage_weight=float(data["depth_stage_weight"]),
            depth_neural_weight=float(data["depth_neural_weight"]),
            depth_final_sigma=float(data["depth_final_sigma"]),
            oval_board_scaled_width=int(data["oval_board_scaled_width"]),
            oval_board_scaled_height=int(data["oval_board_scaled_height"]),
            oval_center_xy=(float(oc[0]), float(oc[1])),
            oval_radii_xy=(float(orad[0]), float(orad[1])),
            oval_min_component_area_px=int(data["oval_min_component_area_px"]),
        )


@dataclass(frozen=True)
class LayerOverlaySpec:
    source_image_relpath: str
    extraction_mode: str
    src_crop_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)
    dest_xy: tuple[int, int] = (0, 0)
    dest_wh: tuple[int, int] = (64, 64)
    min_delta: float = 22.0
    opacity: float = 1.0
    src_crop_norm: tuple[float, float, float, float] | None = None
    dest_center_xy: tuple[int, int] | None = None
    gamma: float = 0.48
    warm_sun_harmony: float = 0.14
    shadow_pocket_lighting: bool = False
    trim_upper_pale_wisps: bool = False
    keep_largest_component_only: bool = False
    suppress_thin_tethers: bool = True
    trim_vertical_tether_ends: bool = False

    def __post_init__(self) -> None:
        if self.extraction_mode not in VALID_OVERLAY_EXTRACTION_MODES:
            raise ValueError(
                f"Unsupported LayerOverlaySpec extraction_mode '{self.extraction_mode}'. "
                f"Valid options: {sorted(VALID_OVERLAY_EXTRACTION_MODES)}"
            )

    def scaled(self, scale: float) -> LayerOverlaySpec:
        if abs(scale - 1.0) <= 1e-6:
            return self
        return LayerOverlaySpec(
            source_image_relpath=self.source_image_relpath,
            extraction_mode=self.extraction_mode,
            src_crop_xyxy=self.src_crop_xyxy,
            dest_xy=(int(round(self.dest_xy[0] * scale)), int(round(self.dest_xy[1] * scale))),
            dest_wh=(
                max(16, int(round(self.dest_wh[0] * scale))),
                max(16, int(round(self.dest_wh[1] * scale))),
            ),
            min_delta=self.min_delta,
            opacity=self.opacity,
            src_crop_norm=self.src_crop_norm,
            dest_center_xy=(
                (
                    int(round(self.dest_center_xy[0] * scale)),
                    int(round(self.dest_center_xy[1] * scale)),
                )
                if self.dest_center_xy is not None
                else None
            ),
            gamma=self.gamma,
            warm_sun_harmony=self.warm_sun_harmony,
            shadow_pocket_lighting=self.shadow_pocket_lighting,
            trim_upper_pale_wisps=self.trim_upper_pale_wisps,
            keep_largest_component_only=self.keep_largest_component_only,
            suppress_thin_tethers=self.suppress_thin_tethers,
            trim_vertical_tether_ends=self.trim_vertical_tether_ends,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_image_relpath": self.source_image_relpath,
            "extraction_mode": self.extraction_mode,
            "src_crop_xyxy": list(self.src_crop_xyxy),
            "dest_xy": list(self.dest_xy),
            "dest_wh": list(self.dest_wh),
            "min_delta": self.min_delta,
            "opacity": self.opacity,
            "src_crop_norm": list(self.src_crop_norm) if self.src_crop_norm is not None else None,
            "dest_center_xy": (
                list(self.dest_center_xy) if self.dest_center_xy is not None else None
            ),
            "gamma": self.gamma,
            "warm_sun_harmony": self.warm_sun_harmony,
            "shadow_pocket_lighting": self.shadow_pocket_lighting,
            "trim_upper_pale_wisps": self.trim_upper_pale_wisps,
            "keep_largest_component_only": self.keep_largest_component_only,
            "suppress_thin_tethers": self.suppress_thin_tethers,
            "trim_vertical_tether_ends": self.trim_vertical_tether_ends,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerOverlaySpec:
        sc = data.get("src_crop_xyxy", [0, 0, 0, 0])
        dx = data.get("dest_xy", [0, 0])
        dw = data.get("dest_wh", [64, 64])
        sc_norm = data.get("src_crop_norm")
        dc_xy = data.get("dest_center_xy")
        return cls(
            source_image_relpath=str(data["source_image_relpath"]),
            extraction_mode=str(data["extraction_mode"]),
            src_crop_xyxy=(int(sc[0]), int(sc[1]), int(sc[2]), int(sc[3])),
            dest_xy=(int(dx[0]), int(dx[1])),
            dest_wh=(int(dw[0]), int(dw[1])),
            min_delta=float(data.get("min_delta", 22.0)),
            opacity=float(data.get("opacity", 1.0)),
            src_crop_norm=(
                (float(sc_norm[0]), float(sc_norm[1]), float(sc_norm[2]), float(sc_norm[3]))
                if sc_norm is not None
                else None
            ),
            dest_center_xy=(int(dc_xy[0]), int(dc_xy[1])) if dc_xy is not None else None,
            gamma=float(data.get("gamma", 0.48)),
            warm_sun_harmony=float(data.get("warm_sun_harmony", 0.14)),
            shadow_pocket_lighting=bool(data.get("shadow_pocket_lighting", False)),
            trim_upper_pale_wisps=bool(data.get("trim_upper_pale_wisps", False)),
            keep_largest_component_only=bool(data.get("keep_largest_component_only", False)),
            suppress_thin_tethers=bool(data.get("suppress_thin_tethers", True)),
            trim_vertical_tether_ends=bool(data.get("trim_vertical_tether_ends", False)),
        )


@dataclass(frozen=True)
class FloatingElementExtractionConfig:
    paper_downsample_divisor: int
    paper_percentile: int
    paper_filter_size: int
    paper_sample_max_diff: float
    min_paper_sigma: float
    noise_floor_sigma_mult: float
    min_delta_floor: float
    ink_upper_percentile: float
    min_ink_span: float
    tether_binary_threshold: float
    tether_window_hw: tuple[int, int]
    tether_min_density: float
    tether_dilate_px: int
    min_graphite_alpha: float
    min_graphite_component_px: int
    stroke_presence_boost: float
    unmix_alpha_floor: float
    graphite_tint_rgb: tuple[float, float, float]
    preserve_shading_blend: float
    max_shaded_lum: float
    capsule_context_pad_px: int
    wall_bg_max_alpha: float
    wall_bg_percentile: float
    fallback_wall_rgb: tuple[float, float, float]
    solid_core_min_alpha: float
    solid_core_min_warmth: float
    solid_core_dark_lum: float
    porthole_close_px: int
    porthole_max_area_px: int
    pale_bleed_max_warmth: float
    pale_bleed_min_lum: float
    pale_bleed_max_wall_dist: float
    capsule_alpha_low: float
    capsule_alpha_high: float
    min_capsule_component_px: int
    capsule_alpha_sigma: float
    capsule_unmix_alpha_floor: float
    wisp_x_range_norm: tuple[float, float]
    wisp_y_max_norm: float
    wisp_max_warmth: float
    wisp_min_lum: float
    shadow_pocket_cool_rgb: tuple[float, float, float]
    shadow_pocket_warm_rgb: tuple[float, float, float]
    mirror_steel_tint_rgb: tuple[float, float, float]
    mirror_blur_base_sigma: float
    mirror_blur_dist_gain: float
    mirror_bg_dark_lum: float
    mirror_bg_lum_span: float
    mirror_illum_base: float
    mirror_illum_gain: float
    mirror_illum_min: float
    mirror_illum_max: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tether_window_hw"] = list(self.tether_window_hw)
        d["graphite_tint_rgb"] = list(self.graphite_tint_rgb)
        d["fallback_wall_rgb"] = list(self.fallback_wall_rgb)
        d["wisp_x_range_norm"] = list(self.wisp_x_range_norm)
        d["shadow_pocket_cool_rgb"] = list(self.shadow_pocket_cool_rgb)
        d["shadow_pocket_warm_rgb"] = list(self.shadow_pocket_warm_rgb)
        d["mirror_steel_tint_rgb"] = list(self.mirror_steel_tint_rgb)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FloatingElementExtractionConfig:
        tw = data["tether_window_hw"]
        gt = data["graphite_tint_rgb"]
        fw = data["fallback_wall_rgb"]
        wx = data["wisp_x_range_norm"]
        sc = data["shadow_pocket_cool_rgb"]
        sw = data["shadow_pocket_warm_rgb"]
        ms = data["mirror_steel_tint_rgb"]
        return cls(
            paper_downsample_divisor=int(data["paper_downsample_divisor"]),
            paper_percentile=int(data["paper_percentile"]),
            paper_filter_size=int(data["paper_filter_size"]),
            paper_sample_max_diff=float(data["paper_sample_max_diff"]),
            min_paper_sigma=float(data["min_paper_sigma"]),
            noise_floor_sigma_mult=float(data["noise_floor_sigma_mult"]),
            min_delta_floor=float(data["min_delta_floor"]),
            ink_upper_percentile=float(data["ink_upper_percentile"]),
            min_ink_span=float(data["min_ink_span"]),
            tether_binary_threshold=float(data["tether_binary_threshold"]),
            tether_window_hw=(int(tw[0]), int(tw[1])),
            tether_min_density=float(data["tether_min_density"]),
            tether_dilate_px=int(data["tether_dilate_px"]),
            min_graphite_alpha=float(data["min_graphite_alpha"]),
            min_graphite_component_px=int(data["min_graphite_component_px"]),
            stroke_presence_boost=float(data["stroke_presence_boost"]),
            unmix_alpha_floor=float(data["unmix_alpha_floor"]),
            graphite_tint_rgb=(float(gt[0]), float(gt[1]), float(gt[2])),
            preserve_shading_blend=float(data["preserve_shading_blend"]),
            max_shaded_lum=float(data["max_shaded_lum"]),
            capsule_context_pad_px=int(data["capsule_context_pad_px"]),
            wall_bg_max_alpha=float(data["wall_bg_max_alpha"]),
            wall_bg_percentile=float(data["wall_bg_percentile"]),
            fallback_wall_rgb=(float(fw[0]), float(fw[1]), float(fw[2])),
            solid_core_min_alpha=float(data["solid_core_min_alpha"]),
            solid_core_min_warmth=float(data["solid_core_min_warmth"]),
            solid_core_dark_lum=float(data["solid_core_dark_lum"]),
            porthole_close_px=int(data["porthole_close_px"]),
            porthole_max_area_px=int(data["porthole_max_area_px"]),
            pale_bleed_max_warmth=float(data["pale_bleed_max_warmth"]),
            pale_bleed_min_lum=float(data["pale_bleed_min_lum"]),
            pale_bleed_max_wall_dist=float(data["pale_bleed_max_wall_dist"]),
            capsule_alpha_low=float(data["capsule_alpha_low"]),
            capsule_alpha_high=float(data["capsule_alpha_high"]),
            min_capsule_component_px=int(data["min_capsule_component_px"]),
            capsule_alpha_sigma=float(data["capsule_alpha_sigma"]),
            capsule_unmix_alpha_floor=float(data["capsule_unmix_alpha_floor"]),
            wisp_x_range_norm=(float(wx[0]), float(wx[1])),
            wisp_y_max_norm=float(data["wisp_y_max_norm"]),
            wisp_max_warmth=float(data["wisp_max_warmth"]),
            wisp_min_lum=float(data["wisp_min_lum"]),
            shadow_pocket_cool_rgb=(float(sc[0]), float(sc[1]), float(sc[2])),
            shadow_pocket_warm_rgb=(float(sw[0]), float(sw[1]), float(sw[2])),
            mirror_steel_tint_rgb=(float(ms[0]), float(ms[1]), float(ms[2])),
            mirror_blur_base_sigma=float(data["mirror_blur_base_sigma"]),
            mirror_blur_dist_gain=float(data["mirror_blur_dist_gain"]),
            mirror_bg_dark_lum=float(data["mirror_bg_dark_lum"]),
            mirror_bg_lum_span=float(data["mirror_bg_lum_span"]),
            mirror_illum_base=float(data["mirror_illum_base"]),
            mirror_illum_gain=float(data["mirror_illum_gain"]),
            mirror_illum_min=float(data["mirror_illum_min"]),
            mirror_illum_max=float(data["mirror_illum_max"]),
        )


@dataclass(frozen=True)
class MirrorReflectedItemSpec:
    overlay: LayerOverlaySpec
    height_norm: float
    steel_reflectance: float
    overlay_is_expanded: bool = False

    def scaled(self, scale: float) -> MirrorReflectedItemSpec:
        if abs(scale - 1.0) <= 1e-6:
            return self
        return MirrorReflectedItemSpec(
            overlay=self.overlay.scaled(scale),
            height_norm=self.height_norm,
            steel_reflectance=self.steel_reflectance,
            overlay_is_expanded=self.overlay_is_expanded,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overlay": self.overlay.to_dict(),
            "height_norm": self.height_norm,
            "steel_reflectance": self.steel_reflectance,
            "overlay_is_expanded": self.overlay_is_expanded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MirrorReflectedItemSpec:
        return cls(
            overlay=LayerOverlaySpec.from_dict(data["overlay"]),
            height_norm=float(data["height_norm"]),
            steel_reflectance=float(data["steel_reflectance"]),
            overlay_is_expanded=bool(data.get("overlay_is_expanded", False)),
        )


@dataclass(frozen=True)
class CeilingMirrorReflectionConfig:
    vanishing_point_xy: tuple[float, float]
    base_opacity: float
    excursion_boost: float
    items: tuple[MirrorReflectedItemSpec, ...]

    def scaled(self, scale: float) -> CeilingMirrorReflectionConfig:
        if abs(scale - 1.0) <= 1e-6:
            return self
        return CeilingMirrorReflectionConfig(
            vanishing_point_xy=(
                self.vanishing_point_xy[0] * scale,
                self.vanishing_point_xy[1] * scale,
            ),
            base_opacity=self.base_opacity,
            excursion_boost=self.excursion_boost,
            items=tuple(it.scaled(scale) for it in self.items),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vanishing_point_xy": list(self.vanishing_point_xy),
            "base_opacity": self.base_opacity,
            "excursion_boost": self.excursion_boost,
            "items": [it.to_dict() for it in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CeilingMirrorReflectionConfig:
        vp = data["vanishing_point_xy"]
        return cls(
            vanishing_point_xy=(float(vp[0]), float(vp[1])),
            base_opacity=float(data["base_opacity"]),
            excursion_boost=float(data["excursion_boost"]),
            items=tuple(MirrorReflectedItemSpec.from_dict(it) for it in data["items"]),
        )


@dataclass(frozen=True)
class ContactDepthPinSpec:
    inner_xy: tuple[float, float]
    target_z: float
    radius_px: float
    feather_px: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "inner_xy": list(self.inner_xy),
            "target_z": self.target_z,
            "radius_px": self.radius_px,
            "feather_px": self.feather_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContactDepthPinSpec:
        xy = data["inner_xy"]
        return cls(
            inner_xy=(float(xy[0]), float(xy[1])),
            target_z=float(data["target_z"]),
            radius_px=float(data["radius_px"]),
            feather_px=float(data["feather_px"]),
        )


@dataclass(frozen=True)
class PipeTunnelReliefConfig:
    front_center_xy: tuple[float, float]
    front_radii_xy: tuple[float, float]
    rear_center_xy: tuple[float, float]
    rear_radii_xy: tuple[float, float]
    scene_bg_z: float
    pipe_depth_amplitude: float
    reference_camera_z: float
    front_rim_z: float
    rim_face_width_px: float
    rear_rim_z: float
    rear_chamber_z: float
    rear_chamber_taper_r: float
    rear_chamber_core_clamp_ratio: float
    mouth_join_blend_px: float
    bell_sam_prompt: SamPointGroupPrompt
    conduit_sam_prompt: SamPointGroupPrompt
    bell_sam_max_pinhole_px: int
    bell_sam_min_component_px: int
    conduit_sam_max_pinhole_px: int
    conduit_sam_min_component_px: int
    polar_angle_samples: int
    polar_sam_probe_inward_px: float
    polar_sam_probe_outward_px: float
    polar_refine_inward_px: float
    polar_refine_outward_px: float
    polar_da2_edge_weight: float
    wall_bright_lum_threshold: float
    wall_bright_penalty_weight: float
    rim_edge_inward_bias_px: float
    polar_smooth_sigma_samples: float
    left_join_angle_range_deg: tuple[float, float]
    left_pipe_search_end_exp_x: int
    left_pipe_join_exp_x: int
    left_pipe_top_clamp_exp_y: tuple[int, int]
    left_pipe_bot_clamp_exp_y: tuple[int, int]
    left_pipe_snap_window_px: tuple[int, int]
    left_pipe_elbow_exp_x: float
    left_pipe_corner_shadow_halfwidth_px: float
    left_pipe_corner_shadow_weight: float
    left_pipe_irls_iters: int
    left_pipe_irls_huber_px: float
    left_pipe_hinge_smooth_sigma_px: float
    left_pipe_left_edge_z: float
    left_pipe_elbow_z: float
    left_pipe_mouth_join_z: float
    left_pipe_left_cyl_radius_z: float
    left_pipe_join_cyl_radius_z: float
    shell_alpha_aa_halfwidth_px: float
    shell_pure_concrete_erode_px: int
    shell_edge_pad_outward_px: int
    neural_relief_amplitude_z: float
    neural_highpass_sigma: float
    interior_smooth_sigma_px: float
    bisection_iters: int
    wall_inpaint_downsample_wh: tuple[int, int]
    wall_inpaint_dilation_px: int
    wall_inpaint_extra_boxes_exp: tuple[BoundingBoxPrompt, ...]
    wall_inpaint_feather_px: float
    wall_grain_patch_exp_xyxy: tuple[int, int, int, int]
    wall_grain_highpass_sigma_px: float
    wall_grain_strength: float
    flat_wall_depth_max_u8: int
    conduit_bell_vert_taper_px: float = 36.0
    rim_arc_taper_deg: float = 10.0
    contact_pins: tuple[ContactDepthPinSpec, ...] = ()
    outer_wall_collar_px: float = 24.0
    outer_wall_depth_zero_px: float = 12.0
    outer_wall_inner_bevel_px: float = 6.0
    outer_wall_shade_mult: float = 0.84

    def to_dict(self) -> dict[str, Any]:
        return {
            "front_center_xy": list(self.front_center_xy),
            "front_radii_xy": list(self.front_radii_xy),
            "rear_center_xy": list(self.rear_center_xy),
            "rear_radii_xy": list(self.rear_radii_xy),
            "scene_bg_z": self.scene_bg_z,
            "pipe_depth_amplitude": self.pipe_depth_amplitude,
            "reference_camera_z": self.reference_camera_z,
            "front_rim_z": self.front_rim_z,
            "rim_face_width_px": self.rim_face_width_px,
            "rear_rim_z": self.rear_rim_z,
            "rear_chamber_z": self.rear_chamber_z,
            "rear_chamber_taper_r": self.rear_chamber_taper_r,
            "rear_chamber_core_clamp_ratio": self.rear_chamber_core_clamp_ratio,
            "mouth_join_blend_px": self.mouth_join_blend_px,
            "bell_sam_prompt": self.bell_sam_prompt.to_dict(),
            "conduit_sam_prompt": self.conduit_sam_prompt.to_dict(),
            "bell_sam_max_pinhole_px": self.bell_sam_max_pinhole_px,
            "bell_sam_min_component_px": self.bell_sam_min_component_px,
            "conduit_sam_max_pinhole_px": self.conduit_sam_max_pinhole_px,
            "conduit_sam_min_component_px": self.conduit_sam_min_component_px,
            "polar_angle_samples": self.polar_angle_samples,
            "polar_sam_probe_inward_px": self.polar_sam_probe_inward_px,
            "polar_sam_probe_outward_px": self.polar_sam_probe_outward_px,
            "polar_refine_inward_px": self.polar_refine_inward_px,
            "polar_refine_outward_px": self.polar_refine_outward_px,
            "polar_da2_edge_weight": self.polar_da2_edge_weight,
            "wall_bright_lum_threshold": self.wall_bright_lum_threshold,
            "wall_bright_penalty_weight": self.wall_bright_penalty_weight,
            "rim_edge_inward_bias_px": self.rim_edge_inward_bias_px,
            "polar_smooth_sigma_samples": self.polar_smooth_sigma_samples,
            "left_join_angle_range_deg": list(self.left_join_angle_range_deg),
            "left_pipe_search_end_exp_x": self.left_pipe_search_end_exp_x,
            "left_pipe_join_exp_x": self.left_pipe_join_exp_x,
            "left_pipe_top_clamp_exp_y": list(self.left_pipe_top_clamp_exp_y),
            "left_pipe_bot_clamp_exp_y": list(self.left_pipe_bot_clamp_exp_y),
            "left_pipe_snap_window_px": list(self.left_pipe_snap_window_px),
            "left_pipe_elbow_exp_x": self.left_pipe_elbow_exp_x,
            "left_pipe_corner_shadow_halfwidth_px": self.left_pipe_corner_shadow_halfwidth_px,
            "left_pipe_corner_shadow_weight": self.left_pipe_corner_shadow_weight,
            "left_pipe_irls_iters": self.left_pipe_irls_iters,
            "left_pipe_irls_huber_px": self.left_pipe_irls_huber_px,
            "left_pipe_hinge_smooth_sigma_px": self.left_pipe_hinge_smooth_sigma_px,
            "left_pipe_left_edge_z": self.left_pipe_left_edge_z,
            "left_pipe_elbow_z": self.left_pipe_elbow_z,
            "left_pipe_mouth_join_z": self.left_pipe_mouth_join_z,
            "left_pipe_left_cyl_radius_z": self.left_pipe_left_cyl_radius_z,
            "left_pipe_join_cyl_radius_z": self.left_pipe_join_cyl_radius_z,
            "shell_alpha_aa_halfwidth_px": self.shell_alpha_aa_halfwidth_px,
            "shell_pure_concrete_erode_px": self.shell_pure_concrete_erode_px,
            "shell_edge_pad_outward_px": self.shell_edge_pad_outward_px,
            "neural_relief_amplitude_z": self.neural_relief_amplitude_z,
            "neural_highpass_sigma": self.neural_highpass_sigma,
            "interior_smooth_sigma_px": self.interior_smooth_sigma_px,
            "bisection_iters": self.bisection_iters,
            "wall_inpaint_downsample_wh": list(self.wall_inpaint_downsample_wh),
            "wall_inpaint_dilation_px": self.wall_inpaint_dilation_px,
            "wall_inpaint_extra_boxes_exp": [
                b.to_dict() for b in self.wall_inpaint_extra_boxes_exp
            ],
            "wall_inpaint_feather_px": self.wall_inpaint_feather_px,
            "wall_grain_patch_exp_xyxy": list(self.wall_grain_patch_exp_xyxy),
            "wall_grain_highpass_sigma_px": self.wall_grain_highpass_sigma_px,
            "wall_grain_strength": self.wall_grain_strength,
            "flat_wall_depth_max_u8": self.flat_wall_depth_max_u8,
            "conduit_bell_vert_taper_px": self.conduit_bell_vert_taper_px,
            "rim_arc_taper_deg": self.rim_arc_taper_deg,
            "contact_pins": [p.to_dict() for p in self.contact_pins],
            "outer_wall_collar_px": self.outer_wall_collar_px,
            "outer_wall_depth_zero_px": self.outer_wall_depth_zero_px,
            "outer_wall_inner_bevel_px": self.outer_wall_inner_bevel_px,
            "outer_wall_shade_mult": self.outer_wall_shade_mult,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipeTunnelReliefConfig:
        fc = data["front_center_xy"]
        fr = data["front_radii_xy"]
        rc = data["rear_center_xy"]
        rr = data["rear_radii_xy"]
        lja = data["left_join_angle_range_deg"]
        lpt = data["left_pipe_top_clamp_exp_y"]
        lpb = data["left_pipe_bot_clamp_exp_y"]
        lsw = data["left_pipe_snap_window_px"]
        wid = data["wall_inpaint_downsample_wh"]
        wgp = data["wall_grain_patch_exp_xyxy"]
        return cls(
            front_center_xy=(float(fc[0]), float(fc[1])),
            front_radii_xy=(float(fr[0]), float(fr[1])),
            rear_center_xy=(float(rc[0]), float(rc[1])),
            rear_radii_xy=(float(rr[0]), float(rr[1])),
            scene_bg_z=float(data["scene_bg_z"]),
            pipe_depth_amplitude=float(data["pipe_depth_amplitude"]),
            reference_camera_z=float(data["reference_camera_z"]),
            front_rim_z=float(data["front_rim_z"]),
            rim_face_width_px=float(data["rim_face_width_px"]),
            rear_rim_z=float(data["rear_rim_z"]),
            rear_chamber_z=float(data["rear_chamber_z"]),
            rear_chamber_taper_r=float(data["rear_chamber_taper_r"]),
            rear_chamber_core_clamp_ratio=float(data["rear_chamber_core_clamp_ratio"]),
            mouth_join_blend_px=float(data["mouth_join_blend_px"]),
            bell_sam_prompt=SamPointGroupPrompt.from_dict(data["bell_sam_prompt"]),
            conduit_sam_prompt=SamPointGroupPrompt.from_dict(data["conduit_sam_prompt"]),
            bell_sam_max_pinhole_px=int(data["bell_sam_max_pinhole_px"]),
            bell_sam_min_component_px=int(data["bell_sam_min_component_px"]),
            conduit_sam_max_pinhole_px=int(data["conduit_sam_max_pinhole_px"]),
            conduit_sam_min_component_px=int(data["conduit_sam_min_component_px"]),
            polar_angle_samples=int(data["polar_angle_samples"]),
            polar_sam_probe_inward_px=float(data["polar_sam_probe_inward_px"]),
            polar_sam_probe_outward_px=float(data["polar_sam_probe_outward_px"]),
            polar_refine_inward_px=float(data["polar_refine_inward_px"]),
            polar_refine_outward_px=float(data["polar_refine_outward_px"]),
            polar_da2_edge_weight=float(data["polar_da2_edge_weight"]),
            wall_bright_lum_threshold=float(data["wall_bright_lum_threshold"]),
            wall_bright_penalty_weight=float(data["wall_bright_penalty_weight"]),
            rim_edge_inward_bias_px=float(data["rim_edge_inward_bias_px"]),
            polar_smooth_sigma_samples=float(data["polar_smooth_sigma_samples"]),
            left_join_angle_range_deg=(float(lja[0]), float(lja[1])),
            left_pipe_search_end_exp_x=int(data["left_pipe_search_end_exp_x"]),
            left_pipe_join_exp_x=int(data["left_pipe_join_exp_x"]),
            left_pipe_top_clamp_exp_y=(int(lpt[0]), int(lpt[1])),
            left_pipe_bot_clamp_exp_y=(int(lpb[0]), int(lpb[1])),
            left_pipe_snap_window_px=(int(lsw[0]), int(lsw[1])),
            left_pipe_elbow_exp_x=float(data["left_pipe_elbow_exp_x"]),
            left_pipe_corner_shadow_halfwidth_px=float(
                data["left_pipe_corner_shadow_halfwidth_px"]
            ),
            left_pipe_corner_shadow_weight=float(data["left_pipe_corner_shadow_weight"]),
            left_pipe_irls_iters=int(data["left_pipe_irls_iters"]),
            left_pipe_irls_huber_px=float(data["left_pipe_irls_huber_px"]),
            left_pipe_hinge_smooth_sigma_px=float(data["left_pipe_hinge_smooth_sigma_px"]),
            left_pipe_left_edge_z=float(data["left_pipe_left_edge_z"]),
            left_pipe_elbow_z=float(data["left_pipe_elbow_z"]),
            left_pipe_mouth_join_z=float(data["left_pipe_mouth_join_z"]),
            left_pipe_left_cyl_radius_z=float(data["left_pipe_left_cyl_radius_z"]),
            left_pipe_join_cyl_radius_z=float(data["left_pipe_join_cyl_radius_z"]),
            shell_alpha_aa_halfwidth_px=float(data["shell_alpha_aa_halfwidth_px"]),
            shell_pure_concrete_erode_px=int(data["shell_pure_concrete_erode_px"]),
            shell_edge_pad_outward_px=int(data["shell_edge_pad_outward_px"]),
            neural_relief_amplitude_z=float(data["neural_relief_amplitude_z"]),
            neural_highpass_sigma=float(data["neural_highpass_sigma"]),
            interior_smooth_sigma_px=float(data["interior_smooth_sigma_px"]),
            bisection_iters=int(data["bisection_iters"]),
            wall_inpaint_downsample_wh=(int(wid[0]), int(wid[1])),
            wall_inpaint_dilation_px=int(data["wall_inpaint_dilation_px"]),
            wall_inpaint_extra_boxes_exp=tuple(
                BoundingBoxPrompt.from_dict(b)
                for b in data.get("wall_inpaint_extra_boxes_exp", ())
            ),
            wall_inpaint_feather_px=float(data["wall_inpaint_feather_px"]),
            wall_grain_patch_exp_xyxy=(int(wgp[0]), int(wgp[1]), int(wgp[2]), int(wgp[3])),
            wall_grain_highpass_sigma_px=float(data["wall_grain_highpass_sigma_px"]),
            wall_grain_strength=float(data["wall_grain_strength"]),
            flat_wall_depth_max_u8=int(data["flat_wall_depth_max_u8"]),
            conduit_bell_vert_taper_px=float(data.get("conduit_bell_vert_taper_px", 36.0)),
            rim_arc_taper_deg=float(data.get("rim_arc_taper_deg", 10.0)),
            contact_pins=tuple(
                ContactDepthPinSpec.from_dict(p) for p in data.get("contact_pins", ())
            ),
            outer_wall_collar_px=float(data.get("outer_wall_collar_px", 24.0)),
            outer_wall_depth_zero_px=float(data.get("outer_wall_depth_zero_px", 12.0)),
            outer_wall_inner_bevel_px=float(data.get("outer_wall_inner_bevel_px", 6.0)),
            outer_wall_shade_mult=float(data.get("outer_wall_shade_mult", 0.84)),
        )


@dataclass(frozen=True)
class TunnelBoreTargetSpec:
    inner_norm_x: float
    inner_norm_y: float
    target_z: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TunnelBoreTargetSpec:
        return cls(
            inner_norm_x=float(data["inner_norm_x"]),
            inner_norm_y=float(data["inner_norm_y"]),
            target_z=float(data["target_z"]),
        )


@dataclass(frozen=True)
class DiffuseApertureFusionSpec:
    source_image_relpath: str
    src_crop_xyxy: tuple[int, int, int, int]
    dest_center_inner_xy: tuple[float, float]
    dest_inner_wh: tuple[float, float]
    canvas_scale_mult: float
    occluder_remove_boxes: tuple[BoundingBoxPrompt, ...]
    front_subject_boxes: tuple[BoundingBoxPrompt, ...]
    front_exclude_boxes: tuple[BoundingBoxPrompt, ...]
    back_subject_boxes: tuple[BoundingBoxPrompt, ...]
    back_hair_matte_box: BoundingBoxPrompt
    shadow_rgb: tuple[float, float, float]
    midtone_rgb: tuple[float, float, float]
    highlight_rgb: tuple[float, float, float]
    mist_glow_rgb: tuple[float, float, float]
    chroma_retention: float
    shadow_lift: float
    luma_gamma: float
    mist_veil_core: float
    mist_veil_edge: float
    halo_rx_ratio: float
    halo_ry_ratio: float
    halo_feather_power: float
    halo_max_alpha: float
    halo_patch_scale_wh: tuple[float, float]
    halo_offset_inner_xy: tuple[float, float]
    halo_luminous_lift: float
    halo_bg_blur_sigma_px: float
    tabletop_source_band_ratios: tuple[float, float]
    tabletop_blend_band_px: int
    tabletop_alpha_bridge_sigma_px: float
    tabletop_rgb_diffuse_sigma_yx: tuple[float, float]
    arch_center_x_ratio: float
    arch_half_span_x_ratio: float
    bottom_arch_curvature: float
    organic_wave_amp_ratio: float
    depth_contour_amp_ratio: float
    depth_contour_smooth_sigma_px: float
    bottom_fade_start_ratio: float
    bottom_fade_end_ratio: float
    bottom_fade_power: float
    top_fade_end_ratio: float
    left_fade_end_ratio: float
    right_fade_start_ratio: float
    right_fade_end_ratio: float
    lower_bloom_strength: float
    lower_bloom_gamma: float
    edge_light_wrap_sigma_px: float
    edge_light_wrap_strength: float
    aperture_glow_center_inner_xy: tuple[float, float]
    aperture_glow_radii_inner_xy: tuple[float, float]
    aperture_glow_core_rgb: tuple[float, float, float]
    aperture_glow_max_strength: float
    aperture_glow_feather_power: float
    aperture_grain_preserve_sigma_px: float
    aperture_grain_preserve_weight: float
    lama_dilation_px: int
    warm_spill_rgb: tuple[float, float, float] = (248.0, 236.0, 214.0)
    warm_rim_wrap_strength: float = 0.42
    lower_warm_pool_strength: float = 0.66
    tabletop_preserve_real_rgb_weight: float = 0.78
    clip_to_rear_bore: bool = True
    rear_bore_center_inner_xy: tuple[float, float] = (878.0, 480.0)
    rear_bore_radii_inner_xy: tuple[float, float] = (152.0, 168.0)
    rear_bore_feather_start_ratio: float = 0.84

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_image_relpath": self.source_image_relpath,
            "src_crop_xyxy": list(self.src_crop_xyxy),
            "dest_center_inner_xy": list(self.dest_center_inner_xy),
            "dest_inner_wh": list(self.dest_inner_wh),
            "canvas_scale_mult": self.canvas_scale_mult,
            "occluder_remove_boxes": [b.to_dict() for b in self.occluder_remove_boxes],
            "front_subject_boxes": [b.to_dict() for b in self.front_subject_boxes],
            "front_exclude_boxes": [b.to_dict() for b in self.front_exclude_boxes],
            "back_subject_boxes": [b.to_dict() for b in self.back_subject_boxes],
            "back_hair_matte_box": self.back_hair_matte_box.to_dict(),
            "shadow_rgb": list(self.shadow_rgb),
            "midtone_rgb": list(self.midtone_rgb),
            "highlight_rgb": list(self.highlight_rgb),
            "mist_glow_rgb": list(self.mist_glow_rgb),
            "chroma_retention": self.chroma_retention,
            "shadow_lift": self.shadow_lift,
            "luma_gamma": self.luma_gamma,
            "mist_veil_core": self.mist_veil_core,
            "mist_veil_edge": self.mist_veil_edge,
            "halo_rx_ratio": self.halo_rx_ratio,
            "halo_ry_ratio": self.halo_ry_ratio,
            "halo_feather_power": self.halo_feather_power,
            "halo_max_alpha": self.halo_max_alpha,
            "halo_patch_scale_wh": list(self.halo_patch_scale_wh),
            "halo_offset_inner_xy": list(self.halo_offset_inner_xy),
            "halo_luminous_lift": self.halo_luminous_lift,
            "halo_bg_blur_sigma_px": self.halo_bg_blur_sigma_px,
            "tabletop_source_band_ratios": list(self.tabletop_source_band_ratios),
            "tabletop_blend_band_px": self.tabletop_blend_band_px,
            "tabletop_alpha_bridge_sigma_px": self.tabletop_alpha_bridge_sigma_px,
            "tabletop_rgb_diffuse_sigma_yx": list(self.tabletop_rgb_diffuse_sigma_yx),
            "arch_center_x_ratio": self.arch_center_x_ratio,
            "arch_half_span_x_ratio": self.arch_half_span_x_ratio,
            "bottom_arch_curvature": self.bottom_arch_curvature,
            "organic_wave_amp_ratio": self.organic_wave_amp_ratio,
            "depth_contour_amp_ratio": self.depth_contour_amp_ratio,
            "depth_contour_smooth_sigma_px": self.depth_contour_smooth_sigma_px,
            "bottom_fade_start_ratio": self.bottom_fade_start_ratio,
            "bottom_fade_end_ratio": self.bottom_fade_end_ratio,
            "bottom_fade_power": self.bottom_fade_power,
            "top_fade_end_ratio": self.top_fade_end_ratio,
            "left_fade_end_ratio": self.left_fade_end_ratio,
            "right_fade_start_ratio": self.right_fade_start_ratio,
            "right_fade_end_ratio": self.right_fade_end_ratio,
            "lower_bloom_strength": self.lower_bloom_strength,
            "lower_bloom_gamma": self.lower_bloom_gamma,
            "edge_light_wrap_sigma_px": self.edge_light_wrap_sigma_px,
            "edge_light_wrap_strength": self.edge_light_wrap_strength,
            "aperture_glow_center_inner_xy": list(self.aperture_glow_center_inner_xy),
            "aperture_glow_radii_inner_xy": list(self.aperture_glow_radii_inner_xy),
            "aperture_glow_core_rgb": list(self.aperture_glow_core_rgb),
            "aperture_glow_max_strength": self.aperture_glow_max_strength,
            "aperture_glow_feather_power": self.aperture_glow_feather_power,
            "aperture_grain_preserve_sigma_px": self.aperture_grain_preserve_sigma_px,
            "aperture_grain_preserve_weight": self.aperture_grain_preserve_weight,
            "lama_dilation_px": self.lama_dilation_px,
            "warm_spill_rgb": list(self.warm_spill_rgb),
            "warm_rim_wrap_strength": self.warm_rim_wrap_strength,
            "lower_warm_pool_strength": self.lower_warm_pool_strength,
            "tabletop_preserve_real_rgb_weight": self.tabletop_preserve_real_rgb_weight,
            "clip_to_rear_bore": self.clip_to_rear_bore,
            "rear_bore_center_inner_xy": list(self.rear_bore_center_inner_xy),
            "rear_bore_radii_inner_xy": list(self.rear_bore_radii_inner_xy),
            "rear_bore_feather_start_ratio": self.rear_bore_feather_start_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiffuseApertureFusionSpec:
        sc = data["src_crop_xyxy"]
        dc = data["dest_center_inner_xy"]
        dw = data["dest_inner_wh"]
        sh = data["shadow_rgb"]
        md = data["midtone_rgb"]
        hi = data["highlight_rgb"]
        mg = data["mist_glow_rgb"]
        ws = data["warm_spill_rgb"]
        hps = data["halo_patch_scale_wh"]
        hoi = data["halo_offset_inner_xy"]
        tsb = data["tabletop_source_band_ratios"]
        trd = data["tabletop_rgb_diffuse_sigma_yx"]
        agc = data["aperture_glow_center_inner_xy"]
        agr = data["aperture_glow_radii_inner_xy"]
        agrgb = data["aperture_glow_core_rgb"]
        rbc = data.get("rear_bore_center_inner_xy", (878.0, 480.0))
        rbr = data.get("rear_bore_radii_inner_xy", (152.0, 168.0))
        return cls(
            source_image_relpath=str(data["source_image_relpath"]),
            src_crop_xyxy=(int(sc[0]), int(sc[1]), int(sc[2]), int(sc[3])),
            dest_center_inner_xy=(float(dc[0]), float(dc[1])),
            dest_inner_wh=(float(dw[0]), float(dw[1])),
            canvas_scale_mult=float(data["canvas_scale_mult"]),
            occluder_remove_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data.get("occluder_remove_boxes", ())
            ),
            front_subject_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["front_subject_boxes"]
            ),
            front_exclude_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data.get("front_exclude_boxes", ())
            ),
            back_subject_boxes=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["back_subject_boxes"]
            ),
            back_hair_matte_box=BoundingBoxPrompt.from_dict(data["back_hair_matte_box"]),
            shadow_rgb=(float(sh[0]), float(sh[1]), float(sh[2])),
            midtone_rgb=(float(md[0]), float(md[1]), float(md[2])),
            highlight_rgb=(float(hi[0]), float(hi[1]), float(hi[2])),
            mist_glow_rgb=(float(mg[0]), float(mg[1]), float(mg[2])),
            chroma_retention=float(data["chroma_retention"]),
            shadow_lift=float(data["shadow_lift"]),
            luma_gamma=float(data["luma_gamma"]),
            mist_veil_core=float(data["mist_veil_core"]),
            mist_veil_edge=float(data["mist_veil_edge"]),
            halo_rx_ratio=float(data["halo_rx_ratio"]),
            halo_ry_ratio=float(data["halo_ry_ratio"]),
            halo_feather_power=float(data["halo_feather_power"]),
            halo_max_alpha=float(data["halo_max_alpha"]),
            halo_patch_scale_wh=(float(hps[0]), float(hps[1])),
            halo_offset_inner_xy=(float(hoi[0]), float(hoi[1])),
            halo_luminous_lift=float(data["halo_luminous_lift"]),
            halo_bg_blur_sigma_px=float(data["halo_bg_blur_sigma_px"]),
            tabletop_source_band_ratios=(float(tsb[0]), float(tsb[1])),
            tabletop_blend_band_px=int(data["tabletop_blend_band_px"]),
            tabletop_alpha_bridge_sigma_px=float(data["tabletop_alpha_bridge_sigma_px"]),
            tabletop_rgb_diffuse_sigma_yx=(float(trd[0]), float(trd[1])),
            arch_center_x_ratio=float(data["arch_center_x_ratio"]),
            arch_half_span_x_ratio=float(data["arch_half_span_x_ratio"]),
            bottom_arch_curvature=float(data["bottom_arch_curvature"]),
            organic_wave_amp_ratio=float(data["organic_wave_amp_ratio"]),
            depth_contour_amp_ratio=float(data["depth_contour_amp_ratio"]),
            depth_contour_smooth_sigma_px=float(data["depth_contour_smooth_sigma_px"]),
            bottom_fade_start_ratio=float(data["bottom_fade_start_ratio"]),
            bottom_fade_end_ratio=float(data["bottom_fade_end_ratio"]),
            bottom_fade_power=float(data["bottom_fade_power"]),
            top_fade_end_ratio=float(data["top_fade_end_ratio"]),
            left_fade_end_ratio=float(data["left_fade_end_ratio"]),
            right_fade_start_ratio=float(data["right_fade_start_ratio"]),
            right_fade_end_ratio=float(data["right_fade_end_ratio"]),
            lower_bloom_strength=float(data["lower_bloom_strength"]),
            lower_bloom_gamma=float(data["lower_bloom_gamma"]),
            edge_light_wrap_sigma_px=float(data["edge_light_wrap_sigma_px"]),
            edge_light_wrap_strength=float(data["edge_light_wrap_strength"]),
            aperture_glow_center_inner_xy=(float(agc[0]), float(agc[1])),
            aperture_glow_radii_inner_xy=(float(agr[0]), float(agr[1])),
            aperture_glow_core_rgb=(float(agrgb[0]), float(agrgb[1]), float(agrgb[2])),
            aperture_glow_max_strength=float(data["aperture_glow_max_strength"]),
            aperture_glow_feather_power=float(data["aperture_glow_feather_power"]),
            aperture_grain_preserve_sigma_px=float(data["aperture_grain_preserve_sigma_px"]),
            aperture_grain_preserve_weight=float(data["aperture_grain_preserve_weight"]),
            lama_dilation_px=int(data["lama_dilation_px"]),
            warm_spill_rgb=(float(ws[0]), float(ws[1]), float(ws[2])),
            warm_rim_wrap_strength=float(data["warm_rim_wrap_strength"]),
            lower_warm_pool_strength=float(data["lower_warm_pool_strength"]),
            tabletop_preserve_real_rgb_weight=float(data["tabletop_preserve_real_rgb_weight"]),
            clip_to_rear_bore=bool(data.get("clip_to_rear_bore", True)),
            rear_bore_center_inner_xy=(float(rbc[0]), float(rbc[1])),
            rear_bore_radii_inner_xy=(float(rbr[0]), float(rbr[1])),
            rear_bore_feather_start_ratio=float(data.get("rear_bore_feather_start_ratio", 0.84)),
        )


@dataclass(frozen=True)
class NeuralPortraitReliefConfig:
    """SSOT configuration for Depth Anything V2 volumetric portrait relief + outward EDT."""

    layer_z: float
    layer_depth_amplitude: float
    layer_pin_margin_uv: float = 0.0
    alpha_threshold_u8: int = 20
    min_component_area_px: int = 50
    lower_percentile: float = 5.0
    upper_percentile: float = 98.0
    interior_smooth_sigma_px: float = 2.4
    pin_edges: tuple[bool, bool, bool, bool] = (False, False, False, False)
    border_band_px: int = 6
    border_taper_px: float = 72.0

    def __post_init__(self) -> None:
        if self.layer_depth_amplitude <= 0.0:
            raise ValueError(
                f"layer_depth_amplitude must be > 0.0, got {self.layer_depth_amplitude}."
            )
        if self.layer_pin_margin_uv != 0.0:
            raise ValueError(
                f"Law 2 requires layer_pin_margin_uv == 0.0 on cutout layers, "
                f"got {self.layer_pin_margin_uv}."
            )
        if not (1 <= self.alpha_threshold_u8 <= 255):
            raise ValueError(
                f"alpha_threshold_u8 must be in [1, 255], got {self.alpha_threshold_u8}."
            )
        if self.min_component_area_px < 1:
            raise ValueError(
                f"min_component_area_px must be >= 1, got {self.min_component_area_px}."
            )
        if not (0.0 <= self.lower_percentile < self.upper_percentile <= 100.0):
            raise ValueError(
                f"Expected 0 <= lower_percentile < upper_percentile <= 100, got "
                f"({self.lower_percentile}, {self.upper_percentile})."
            )
        if self.interior_smooth_sigma_px < 0.0:
            raise ValueError(
                f"interior_smooth_sigma_px must be >= 0.0, got {self.interior_smooth_sigma_px}."
            )
        if len(self.pin_edges) != 4:
            raise ValueError(
                f"pin_edges must be a 4-tuple (left, right, top, bottom), got {self.pin_edges!r}."
            )
        if self.border_band_px < 1:
            raise ValueError(f"border_band_px must be >= 1, got {self.border_band_px}.")
        if self.border_taper_px <= 0.0:
            raise ValueError(f"border_taper_px must be > 0.0, got {self.border_taper_px}.")

    def scaled(self, scale: float) -> NeuralPortraitReliefConfig:
        if abs(scale - 1.0) <= 1e-6:
            return self
        return NeuralPortraitReliefConfig(
            layer_z=self.layer_z,
            layer_depth_amplitude=self.layer_depth_amplitude,
            layer_pin_margin_uv=self.layer_pin_margin_uv,
            alpha_threshold_u8=self.alpha_threshold_u8,
            min_component_area_px=max(1, int(round(self.min_component_area_px * scale * scale))),
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
            interior_smooth_sigma_px=self.interior_smooth_sigma_px * scale,
            pin_edges=self.pin_edges,
            border_band_px=max(2, int(round(self.border_band_px * scale))),
            border_taper_px=self.border_taper_px * scale,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_z": float(self.layer_z),
            "layer_depth_amplitude": float(self.layer_depth_amplitude),
            "layer_pin_margin_uv": float(self.layer_pin_margin_uv),
            "alpha_threshold_u8": int(self.alpha_threshold_u8),
            "min_component_area_px": int(self.min_component_area_px),
            "lower_percentile": float(self.lower_percentile),
            "upper_percentile": float(self.upper_percentile),
            "interior_smooth_sigma_px": float(self.interior_smooth_sigma_px),
            "pin_edges": [bool(e) for e in self.pin_edges],
            "border_band_px": int(self.border_band_px),
            "border_taper_px": float(self.border_taper_px),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeuralPortraitReliefConfig:
        pe = data["pin_edges"]
        return cls(
            layer_z=float(data["layer_z"]),
            layer_depth_amplitude=float(data["layer_depth_amplitude"]),
            layer_pin_margin_uv=float(data["layer_pin_margin_uv"]),
            alpha_threshold_u8=int(data["alpha_threshold_u8"]),
            min_component_area_px=int(data["min_component_area_px"]),
            lower_percentile=float(data["lower_percentile"]),
            upper_percentile=float(data["upper_percentile"]),
            interior_smooth_sigma_px=float(data["interior_smooth_sigma_px"]),
            pin_edges=(bool(pe[0]), bool(pe[1]), bool(pe[2]), bool(pe[3])),
            border_band_px=int(data["border_band_px"]),
            border_taper_px=float(data["border_taper_px"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> NeuralPortraitReliefConfig:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class ExtraDepthLayerSpec:
    layer_id: str
    label_he: str
    output_filename: str
    z_pos: float
    render_order: int
    overlays: tuple[LayerOverlaySpec, ...] = ()
    is_expanded: bool = False
    sam_boxes: tuple[BoundingBoxPrompt, ...] = ()
    sam_point_groups: tuple[SamPointGroupPrompt, ...] = ()
    oriented_matte_boxes: tuple[OrientedMatteBoxSpec, ...] = ()
    exclude_boxes: tuple[BoundingBoxPrompt, ...] = ()
    max_pinhole_fill_px: int = 14000
    min_major_component_px: int = 20000
    occluder_inpaint_boxes: tuple[BoundingBoxPrompt, ...] = ()
    right_shoulder_extend_box: BoundingBoxPrompt | None = None
    right_shoulder_extend_px: int = 0
    pre_lama_dilation_px: int | None = None
    pre_lama_blur_sigma: float | None = None
    ceiling_mirror_reflection: CeilingMirrorReflectionConfig | None = None
    base_opacity: float = 1.0
    excursion_boost: float = 0.0
    depth_test: bool = False
    depth_write: bool = False
    diffuse_aperture_role: str | None = None
    pipe_shell_mode: bool = False
    depth_output_filename: str | None = None
    depth_amplitude: float | None = None
    alpha_cutout: float | None = None
    pin_margin_uv: float | None = None
    segments_x: int | None = None
    segments_y: int | None = None
    contact_depth_solve: ContactAnchoredLayerDepthConfig | None = None
    portrait_relief: NeuralPortraitReliefConfig | None = None

    def __post_init__(self) -> None:
        if (
            self.diffuse_aperture_role is not None
            and self.diffuse_aperture_role not in VALID_DIFFUSE_APERTURE_ROLES
        ):
            raise ValueError(
                f"ExtraDepthLayerSpec '{self.layer_id}' has unsupported diffuse_aperture_role "
                f"'{self.diffuse_aperture_role}'."
            )
        if self.pipe_shell_mode and self.depth_output_filename is None:
            raise ValueError(
                f"ExtraDepthLayerSpec '{self.layer_id}' sets pipe_shell_mode=True without depth_output_filename."
            )
        if self.contact_depth_solve is not None and self.portrait_relief is not None:
            raise ValueError(
                f"ExtraDepthLayerSpec '{self.layer_id}' cannot set both contact_depth_solve and portrait_relief."
            )
        if self.contact_depth_solve is not None:
            if self.depth_output_filename is None or self.depth_amplitude is None:
                raise ValueError(
                    f"ExtraDepthLayerSpec '{self.layer_id}' sets contact_depth_solve without "
                    f"depth_output_filename / depth_amplitude."
                )
            if (
                abs(self.z_pos - self.contact_depth_solve.layer_z) > 1e-4
                or abs(self.depth_amplitude - self.contact_depth_solve.layer_depth_amplitude) > 1e-4
            ):
                raise ValueError(
                    f"ExtraDepthLayerSpec '{self.layer_id}' z_pos/depth_amplitude must match "
                    f"contact_depth_solve.layer_z/layer_depth_amplitude."
                )
        if self.portrait_relief is not None:
            if self.depth_output_filename is None or self.depth_amplitude is None:
                raise ValueError(
                    f"ExtraDepthLayerSpec '{self.layer_id}' sets portrait_relief without "
                    f"depth_output_filename / depth_amplitude."
                )
            if (
                abs(self.z_pos - self.portrait_relief.layer_z) > 1e-4
                or abs(self.depth_amplitude - self.portrait_relief.layer_depth_amplitude) > 1e-4
            ):
                raise ValueError(
                    f"ExtraDepthLayerSpec '{self.layer_id}' z_pos/depth_amplitude must match "
                    f"portrait_relief.layer_z/layer_depth_amplitude."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "label_he": self.label_he,
            "output_filename": self.output_filename,
            "z_pos": self.z_pos,
            "render_order": self.render_order,
            "overlays": [ov.to_dict() for ov in self.overlays],
            "is_expanded": self.is_expanded,
            "sam_boxes": [b.to_dict() for b in self.sam_boxes],
            "sam_point_groups": [pg.to_dict() for pg in self.sam_point_groups],
            "oriented_matte_boxes": [omb.to_dict() for omb in self.oriented_matte_boxes],
            "exclude_boxes": [eb.to_dict() for eb in self.exclude_boxes],
            "max_pinhole_fill_px": self.max_pinhole_fill_px,
            "min_major_component_px": self.min_major_component_px,
            "occluder_inpaint_boxes": [ob.to_dict() for ob in self.occluder_inpaint_boxes],
            "right_shoulder_extend_box": (
                self.right_shoulder_extend_box.to_dict()
                if self.right_shoulder_extend_box is not None
                else None
            ),
            "right_shoulder_extend_px": self.right_shoulder_extend_px,
            "pre_lama_dilation_px": self.pre_lama_dilation_px,
            "pre_lama_blur_sigma": self.pre_lama_blur_sigma,
            "ceiling_mirror_reflection": (
                self.ceiling_mirror_reflection.to_dict()
                if self.ceiling_mirror_reflection is not None
                else None
            ),
            "base_opacity": self.base_opacity,
            "excursion_boost": self.excursion_boost,
            "depth_test": self.depth_test,
            "depth_write": self.depth_write,
            "diffuse_aperture_role": self.diffuse_aperture_role,
            "pipe_shell_mode": self.pipe_shell_mode,
            "depth_output_filename": self.depth_output_filename,
            "depth_amplitude": self.depth_amplitude,
            "alpha_cutout": self.alpha_cutout,
            "pin_margin_uv": self.pin_margin_uv,
            "segments_x": self.segments_x,
            "segments_y": self.segments_y,
            "contact_depth_solve": (
                self.contact_depth_solve.to_dict()
                if self.contact_depth_solve is not None
                else None
            ),
            "portrait_relief": (
                self.portrait_relief.to_dict() if self.portrait_relief is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtraDepthLayerSpec:
        raw_rseb = data.get("right_shoulder_extend_box")
        raw_pld = data.get("pre_lama_dilation_px")
        raw_plb = data.get("pre_lama_blur_sigma")
        raw_cmr = data.get("ceiling_mirror_reflection")
        raw_da = data.get("depth_amplitude")
        raw_ac = data.get("alpha_cutout")
        raw_pm = data.get("pin_margin_uv")
        raw_sx = data.get("segments_x")
        raw_sy = data.get("segments_y")
        raw_cds = data.get("contact_depth_solve")
        raw_pr = data.get("portrait_relief")
        return cls(
            layer_id=str(data["layer_id"]),
            label_he=str(data["label_he"]),
            output_filename=str(data["output_filename"]),
            z_pos=float(data["z_pos"]),
            render_order=int(data["render_order"]),
            overlays=tuple(LayerOverlaySpec.from_dict(ov) for ov in data.get("overlays", ())),
            is_expanded=bool(data.get("is_expanded", False)),
            sam_boxes=tuple(BoundingBoxPrompt.from_dict(b) for b in data.get("sam_boxes", ())),
            sam_point_groups=tuple(
                SamPointGroupPrompt.from_dict(pg) for pg in data.get("sam_point_groups", ())
            ),
            oriented_matte_boxes=tuple(
                OrientedMatteBoxSpec.from_dict(omb)
                for omb in data.get("oriented_matte_boxes", ())
            ),
            exclude_boxes=tuple(
                BoundingBoxPrompt.from_dict(eb) for eb in data.get("exclude_boxes", ())
            ),
            max_pinhole_fill_px=int(data.get("max_pinhole_fill_px", 14000)),
            min_major_component_px=int(data.get("min_major_component_px", 20000)),
            occluder_inpaint_boxes=tuple(
                BoundingBoxPrompt.from_dict(ob) for ob in data.get("occluder_inpaint_boxes", ())
            ),
            right_shoulder_extend_box=(
                BoundingBoxPrompt.from_dict(raw_rseb) if raw_rseb is not None else None
            ),
            right_shoulder_extend_px=int(data.get("right_shoulder_extend_px", 0)),
            pre_lama_dilation_px=int(raw_pld) if raw_pld is not None else None,
            pre_lama_blur_sigma=float(raw_plb) if raw_plb is not None else None,
            ceiling_mirror_reflection=(
                CeilingMirrorReflectionConfig.from_dict(raw_cmr)
                if raw_cmr is not None
                else None
            ),
            base_opacity=float(data.get("base_opacity", 1.0)),
            excursion_boost=float(data.get("excursion_boost", 0.0)),
            depth_test=bool(data.get("depth_test", False)),
            depth_write=bool(data.get("depth_write", False)),
            diffuse_aperture_role=data.get("diffuse_aperture_role"),
            pipe_shell_mode=bool(data.get("pipe_shell_mode", False)),
            depth_output_filename=data.get("depth_output_filename"),
            depth_amplitude=float(raw_da) if raw_da is not None else None,
            alpha_cutout=float(raw_ac) if raw_ac is not None else None,
            pin_margin_uv=float(raw_pm) if raw_pm is not None else None,
            segments_x=int(raw_sx) if raw_sx is not None else None,
            segments_y=int(raw_sy) if raw_sy is not None else None,
            contact_depth_solve=(
                ContactAnchoredLayerDepthConfig.from_dict(raw_cds)
                if raw_cds is not None
                else None
            ),
            portrait_relief=(
                NeuralPortraitReliefConfig.from_dict(raw_pr) if raw_pr is not None else None
            ),
        )


@dataclass(frozen=True)
class SegmentationConfig:
    rembg_model_name: str
    sam_model_id: str
    alpha_foreground_threshold: int
    alpha_full_opaque_threshold: int
    min_component_area_px: int
    sam_morph_close_px: int
    edge_feather_sigma: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentationConfig:
        return cls(
            rembg_model_name=str(data["rembg_model_name"]),
            sam_model_id=str(data["sam_model_id"]),
            alpha_foreground_threshold=int(data["alpha_foreground_threshold"]),
            alpha_full_opaque_threshold=int(data["alpha_full_opaque_threshold"]),
            min_component_area_px=int(data["min_component_area_px"]),
            sam_morph_close_px=int(data["sam_morph_close_px"]),
            edge_feather_sigma=float(data["edge_feather_sigma"]),
        )


@dataclass(frozen=True)
class InpaintingConfig:
    dilation_radius_px: int
    mask_blur_sigma: float
    jpeg_quality: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InpaintingConfig:
        return cls(
            dilation_radius_px=int(data["dilation_radius_px"]),
            mask_blur_sigma=float(data["mask_blur_sigma"]),
            jpeg_quality=int(data["jpeg_quality"]),
        )


@dataclass(frozen=True)
class ApertureBox:
    x0: int
    y0: int
    x1: int
    y1: int
    max_alpha: int
    feather_sigma: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApertureBox:
        return cls(
            x0=int(data["x0"]),
            y0=int(data["y0"]),
            x1=int(data["x1"]),
            y1=int(data["y1"]),
            max_alpha=int(data["max_alpha"]),
            feather_sigma=float(data["feather_sigma"]),
        )


@dataclass(frozen=True)
class SceneBuildSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    source_image_relpath: str
    outpaint_raw_relpath: str
    reflection_source_relpath: str | None
    lama_inner_output_filename: str
    bg_output_filename: str
    mid_output_filename: str
    fg_output_filename: str
    depth_output_filename: str
    reflection_output_filename: str | None
    segmentation_backend: str
    depth_backend: str
    min_registration_corr: float
    room_depth_amplitude: float
    particle_color_hex: int
    particle_count: int
    fg_x_range_norm: tuple[float, float]
    fg_min_depth_quantile: float
    sam_fg_boxes: tuple[BoundingBoxPrompt, ...]
    sam_mid_boxes: tuple[BoundingBoxPrompt, ...]
    drawing_fg_boxes: tuple[DrawingStrokeBox, ...] = ()
    drawing_mid_boxes: tuple[DrawingStrokeBox, ...] = ()
    fg_overlays: tuple[LayerOverlaySpec, ...] = ()
    mid_overlays: tuple[LayerOverlaySpec, ...] = ()
    reflection_apertures: tuple[ApertureBox, ...] = ()
    feather_collar_px: int | None = None
    registration_core_margin_xy: tuple[int, int] = (0, 0)
    initial_camera_y: float = 0.0
    initial_camera_z: float = 400.0
    min_camera_z: float = 120.0
    max_camera_z: float = 685.0
    scene_bg_z: float = -350.0
    min_inner_scale: float | None = None
    max_yaw_deg: float = 34.0
    max_pitch_deg: float = 24.0
    max_strafe_x: float = 250.0
    max_strafe_y: float = 120.0
    dolly_speed_mult: float = 1.0
    fg_render_order: int = 3
    mid_render_order: int = 2
    fg_z: float = 0.0
    mid_z: float = -150.0
    bg_segments_x: int | None = None
    bg_segments_y: int | None = None
    harmonize_seam_field: bool = False
    seam_harmonic_downsample: int = 8
    seam_harmonic_iters: int = 240
    rear_chamber_soften_box: BoundingBoxPrompt | None = None
    pipe_tunnel_relief: PipeTunnelReliefConfig | None = None
    tunnel_bore_target: TunnelBoreTargetSpec | None = None
    diffuse_aperture_fusion: DiffuseApertureFusionSpec | None = None
    fg_expanded_raw_relpath: str | None = None
    fg_is_expanded: bool = False
    sam_fg_point_groups: tuple[SamPointGroupPrompt, ...] = ()
    fg_oriented_matte_boxes: tuple[OrientedMatteBoxSpec, ...] = ()
    sam_core_support_dilate_px: int = 85
    sam_fg_exclude_boxes: tuple[BoundingBoxPrompt, ...] = ()
    sam_fg_right_edge_fade_x_start: int | None = None
    sam_max_pinhole_fill_px: int = 14000
    sam_min_major_component_px: int = 25000
    extra_layers: tuple[ExtraDepthLayerSpec, ...] = ()
    ceiling_mirror_reflection: CeilingMirrorReflectionConfig | None = None
    bore_sight_at_rest: bool = True
    room_depth_clean: RoomDepthCleanConfig | None = None
    room_depth_clean_layers: tuple[str, ...] = ()
    topology_verification: LayerTopologyVerificationConfig | None = None
    mid_depth_output_filename: str | None = None
    mid_depth_amplitude: float | None = None
    mid_pin_margin_uv: float | None = None
    mid_segments_x: int | None = None
    mid_segments_y: int | None = None
    mid_contact_depth_solve: ContactAnchoredLayerDepthConfig | None = None
    fg_depth_output_filename: str | None = None
    fg_depth_amplitude: float | None = None
    fg_pin_margin_uv: float | None = None
    fg_segments_x: int | None = None
    fg_segments_y: int | None = None
    fg_contact_depth_solve: ContactAnchoredLayerDepthConfig | None = None
    fg_portrait_relief: NeuralPortraitReliefConfig | None = None
    sam_mid_point_groups: tuple[SamPointGroupPrompt, ...] = ()
    sam_mid_exclude_boxes: tuple[BoundingBoxPrompt, ...] = ()
    sam_mid_max_pinhole_fill_px: int = 4000
    sam_mid_min_major_component_px: int = 25
    inpaint_region_dilation_overrides: tuple[InpaintRegionDilationOverride, ...] = ()
    inpaint_from_outpaint_inner: bool = False
    inpainting_jpeg_quality_override: int | None = None
    canvas_scale_mult: float = 1.0
    floating_overlay_scale_mult: float = 1.0
    mirror_blur_override: tuple[float, float] | None = None
    use_mirror_outpaint: bool = False
    mid_depth_test: bool = False
    fg_depth_test: bool = False

    def __post_init__(self) -> None:
        if self.segmentation_backend not in VALID_SEGMENTATION_BACKENDS:
            raise ValueError(
                f"Scene '{self.scene_id}': unsupported segmentation_backend "
                f"'{self.segmentation_backend}'. Valid options: {sorted(VALID_SEGMENTATION_BACKENDS)}"
            )
        if self.depth_backend not in VALID_DEPTH_BACKENDS:
            raise ValueError(
                f"Scene '{self.scene_id}': unsupported depth_backend "
                f"'{self.depth_backend}'. Valid options: {sorted(VALID_DEPTH_BACKENDS)}"
            )
        if self.depth_backend == "pipe_tunnel_relief" and self.pipe_tunnel_relief is None:
            raise ValueError(
                f"Scene '{self.scene_id}': depth_backend='pipe_tunnel_relief' requires pipe_tunnel_relief config."
            )
        if self.canvas_scale_mult <= 0.0:
            raise ValueError(
                f"Scene '{self.scene_id}': canvas_scale_mult must be > 0.0, got {self.canvas_scale_mult}."
            )
        if self.floating_overlay_scale_mult <= 0.0:
            raise ValueError(
                f"Scene '{self.scene_id}': floating_overlay_scale_mult must be > 0.0, "
                f"got {self.floating_overlay_scale_mult}."
            )
        if self.inpainting_jpeg_quality_override is not None and not (
            1 <= self.inpainting_jpeg_quality_override <= 100
        ):
            raise ValueError(
                f"Scene '{self.scene_id}': inpainting_jpeg_quality_override must be in [1, 100], "
                f"got {self.inpainting_jpeg_quality_override}."
            )
        if self.mirror_blur_override is not None and (
            len(self.mirror_blur_override) != 2
            or self.mirror_blur_override[0] < 0.0
            or self.mirror_blur_override[1] < 0.0
        ):
            raise ValueError(
                f"Scene '{self.scene_id}': mirror_blur_override must be non-negative (base, gain), "
                f"got {self.mirror_blur_override!r}."
            )
        if self.sam_mid_max_pinhole_fill_px < 0 or self.sam_mid_min_major_component_px < 1:
            raise ValueError(
                f"Scene '{self.scene_id}': invalid sam_mid pinhole/component thresholds "
                f"({self.sam_mid_max_pinhole_fill_px}, {self.sam_mid_min_major_component_px})."
            )
        if self.fg_contact_depth_solve is not None and self.fg_portrait_relief is not None:
            raise ValueError(
                f"Scene '{self.scene_id}': cannot set both fg_contact_depth_solve and fg_portrait_relief."
            )
        if self.mid_contact_depth_solve is not None:
            if self.mid_depth_output_filename is None or self.mid_depth_amplitude is None:
                raise ValueError(
                    f"Scene '{self.scene_id}': mid_contact_depth_solve requires "
                    f"mid_depth_output_filename and mid_depth_amplitude."
                )
            if (
                abs(self.mid_z - self.mid_contact_depth_solve.layer_z) > 1e-4
                or abs(self.mid_depth_amplitude - self.mid_contact_depth_solve.layer_depth_amplitude)
                > 1e-4
            ):
                raise ValueError(
                    f"Scene '{self.scene_id}': mid_z/mid_depth_amplitude must match "
                    f"mid_contact_depth_solve.layer_z/layer_depth_amplitude."
                )
        if self.fg_contact_depth_solve is not None:
            if self.fg_depth_output_filename is None or self.fg_depth_amplitude is None:
                raise ValueError(
                    f"Scene '{self.scene_id}': fg_contact_depth_solve requires "
                    f"fg_depth_output_filename and fg_depth_amplitude."
                )
            if (
                abs(self.fg_z - self.fg_contact_depth_solve.layer_z) > 1e-4
                or abs(self.fg_depth_amplitude - self.fg_contact_depth_solve.layer_depth_amplitude)
                > 1e-4
            ):
                raise ValueError(
                    f"Scene '{self.scene_id}': fg_z/fg_depth_amplitude must match "
                    f"fg_contact_depth_solve.layer_z/layer_depth_amplitude."
                )
        if self.fg_portrait_relief is not None:
            if self.fg_depth_output_filename is None or self.fg_depth_amplitude is None:
                raise ValueError(
                    f"Scene '{self.scene_id}': fg_portrait_relief requires "
                    f"fg_depth_output_filename and fg_depth_amplitude."
                )
            if (
                abs(self.fg_z - self.fg_portrait_relief.layer_z) > 1e-4
                or abs(self.fg_depth_amplitude - self.fg_portrait_relief.layer_depth_amplitude)
                > 1e-4
            ):
                raise ValueError(
                    f"Scene '{self.scene_id}': fg_z/fg_depth_amplitude must match "
                    f"fg_portrait_relief.layer_z/layer_depth_amplitude."
                )

    def scene_dimensions(self, base_dims: CanvasDimensionsConfig) -> CanvasDimensionsConfig:
        """Returns this scene's inner viewport pixel dimensions."""
        return CanvasDimensionsConfig(
            width=int(round(base_dims.width * self.canvas_scale_mult)),
            height=int(round(base_dims.height * self.canvas_scale_mult)),
        )

    def expanded_dimensions(
        self,
        base_dims: CanvasDimensionsConfig,
        outpaint_cfg: OutpaintingConfig,
    ) -> CanvasDimensionsConfig:
        """Returns this scene's expanded outpaint backdrop pixel dimensions."""
        sdims = self.scene_dimensions(base_dims)
        return CanvasDimensionsConfig(
            width=int(round(sdims.width * outpaint_cfg.expansion_factor)),
            height=int(round(sdims.height * outpaint_cfg.expansion_factor)),
        )

    def floating_overlay_dimensions(
        self,
        base_dims: CanvasDimensionsConfig,
    ) -> CanvasDimensionsConfig:
        """Returns pixel dimensions for pure non-expanded floating/reflected artwork layers."""
        return CanvasDimensionsConfig(
            width=int(round(base_dims.width * self.floating_overlay_scale_mult)),
            height=int(round(base_dims.height * self.floating_overlay_scale_mult)),
        )

    def uses_floating_overlay_canvas_for_mid(self) -> bool:
        """True when `mid` is a pure non-depth-meshed overlay layer rendered at `floating_overlay_scale_mult`."""
        return (
            abs(self.floating_overlay_scale_mult - 1.0) > 1e-6
            and len(self.mid_overlays) > 0
            and len(self.sam_mid_boxes) == 0
            and len(self.sam_mid_point_groups) == 0
            and len(self.drawing_mid_boxes) == 0
            and self.segmentation_backend not in ("isnet", "graphite_drawing")
            and self.mid_depth_output_filename is None
        )

    def uses_floating_overlay_canvas_for_extra_layer(self, el: ExtraDepthLayerSpec) -> bool:
        """True when an extra layer is a non-expanded floating or mirror-reflected artwork overlay."""
        return (
            not el.is_expanded
            and abs(self.floating_overlay_scale_mult - 1.0) > 1e-6
            and len(el.sam_boxes) == 0
            and len(el.sam_point_groups) == 0
            and el.diffuse_aperture_role is None
            and not el.pipe_shell_mode
            and el.depth_output_filename is None
            and (len(el.overlays) > 0 or el.ceiling_mirror_reflection is not None)
        )

    def expected_bg_dimensions(
        self,
        base_dims: CanvasDimensionsConfig,
        outpaint_cfg: OutpaintingConfig,
    ) -> tuple[int, int]:
        edims = self.expanded_dimensions(base_dims, outpaint_cfg)
        return (edims.width, edims.height)

    def expected_mid_dimensions(self, base_dims: CanvasDimensionsConfig) -> tuple[int, int]:
        dims = (
            self.floating_overlay_dimensions(base_dims)
            if self.uses_floating_overlay_canvas_for_mid()
            else self.scene_dimensions(base_dims)
        )
        return (dims.width, dims.height)

    def expected_fg_dimensions(
        self,
        base_dims: CanvasDimensionsConfig,
        outpaint_cfg: OutpaintingConfig,
    ) -> tuple[int, int]:
        dims = (
            self.expanded_dimensions(base_dims, outpaint_cfg)
            if self.fg_is_expanded
            else self.scene_dimensions(base_dims)
        )
        return (dims.width, dims.height)

    def expected_extra_layer_dimensions(
        self,
        el: ExtraDepthLayerSpec,
        base_dims: CanvasDimensionsConfig,
        outpaint_cfg: OutpaintingConfig,
    ) -> tuple[int, int]:
        if el.is_expanded:
            dims = self.expanded_dimensions(base_dims, outpaint_cfg)
        elif self.diffuse_aperture_fusion is not None and (
            el.diffuse_aperture_role is not None
            or el.layer_id in ("yaara_front", "tami_back", "chamber_halo")
        ):
            sdims = self.scene_dimensions(base_dims)
            mult = self.diffuse_aperture_fusion.canvas_scale_mult
            dims = CanvasDimensionsConfig(
                width=max(sdims.width, int(round(sdims.width * mult))),
                height=max(sdims.height, int(round(sdims.height * mult))),
            )
        elif self.uses_floating_overlay_canvas_for_extra_layer(el):
            dims = self.floating_overlay_dimensions(base_dims)
        else:
            dims = self.scene_dimensions(base_dims)
        return (dims.width, dims.height)

    def expected_reflection_dimensions(self, base_dims: CanvasDimensionsConfig) -> tuple[int, int]:
        dims = self.scene_dimensions(base_dims)
        return (dims.width, dims.height)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fg_x_range_norm"] = list(self.fg_x_range_norm)
        d["sam_fg_boxes"] = [b.to_dict() for b in self.sam_fg_boxes]
        d["sam_mid_boxes"] = [b.to_dict() for b in self.sam_mid_boxes]
        d["sam_mid_point_groups"] = [pg.to_dict() for pg in self.sam_mid_point_groups]
        d["sam_mid_exclude_boxes"] = [eb.to_dict() for eb in self.sam_mid_exclude_boxes]
        d["inpaint_region_dilation_overrides"] = [
            ov.to_dict() for ov in self.inpaint_region_dilation_overrides
        ]
        d["drawing_fg_boxes"] = [b.to_dict() for b in self.drawing_fg_boxes]
        d["drawing_mid_boxes"] = [b.to_dict() for b in self.drawing_mid_boxes]
        d["fg_overlays"] = [o.to_dict() for o in self.fg_overlays]
        d["mid_overlays"] = [o.to_dict() for o in self.mid_overlays]
        d["reflection_apertures"] = [ap.to_dict() for ap in self.reflection_apertures]
        d["registration_core_margin_xy"] = list(self.registration_core_margin_xy)
        d["sam_fg_point_groups"] = [pg.to_dict() for pg in self.sam_fg_point_groups]
        d["fg_oriented_matte_boxes"] = [omb.to_dict() for omb in self.fg_oriented_matte_boxes]
        d["sam_fg_exclude_boxes"] = [eb.to_dict() for eb in self.sam_fg_exclude_boxes]
        d["extra_layers"] = [el.to_dict() for el in self.extra_layers]
        d["ceiling_mirror_reflection"] = (
            self.ceiling_mirror_reflection.to_dict()
            if self.ceiling_mirror_reflection is not None
            else None
        )
        d["rear_chamber_soften_box"] = (
            self.rear_chamber_soften_box.to_dict()
            if self.rear_chamber_soften_box is not None
            else None
        )
        d["pipe_tunnel_relief"] = (
            self.pipe_tunnel_relief.to_dict() if self.pipe_tunnel_relief is not None else None
        )
        d["tunnel_bore_target"] = (
            self.tunnel_bore_target.to_dict() if self.tunnel_bore_target is not None else None
        )
        d["diffuse_aperture_fusion"] = (
            self.diffuse_aperture_fusion.to_dict()
            if self.diffuse_aperture_fusion is not None
            else None
        )
        d["room_depth_clean"] = (
            self.room_depth_clean.to_dict() if self.room_depth_clean is not None else None
        )
        d["room_depth_clean_layers"] = list(self.room_depth_clean_layers)
        d["topology_verification"] = (
            self.topology_verification.to_dict()
            if self.topology_verification is not None
            else None
        )
        d["mid_contact_depth_solve"] = (
            self.mid_contact_depth_solve.to_dict()
            if self.mid_contact_depth_solve is not None
            else None
        )
        d["fg_contact_depth_solve"] = (
            self.fg_contact_depth_solve.to_dict()
            if self.fg_contact_depth_solve is not None
            else None
        )
        d["fg_portrait_relief"] = (
            self.fg_portrait_relief.to_dict() if self.fg_portrait_relief is not None else None
        )
        d["mirror_blur_override"] = (
            list(self.mirror_blur_override) if self.mirror_blur_override is not None else None
        )
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneBuildSpec:
        xr = data["fg_x_range_norm"]
        rcm = data.get("registration_core_margin_xy", [0, 0])
        raw_collar = data.get("feather_collar_px")
        raw_fade_x = data.get("sam_fg_right_edge_fade_x_start")
        raw_cmr = data.get("ceiling_mirror_reflection")
        raw_rcsb = data.get("rear_chamber_soften_box")
        raw_ptr = data.get("pipe_tunnel_relief")
        raw_tbt = data.get("tunnel_bore_target")
        raw_daf = data.get("diffuse_aperture_fusion")
        raw_seg_x = data.get("bg_segments_x")
        raw_seg_y = data.get("bg_segments_y")
        raw_min_is = data.get("min_inner_scale")
        raw_rdc = data.get("room_depth_clean")
        raw_tv = data.get("topology_verification")
        raw_mda = data.get("mid_depth_amplitude")
        raw_mpm = data.get("mid_pin_margin_uv")
        raw_msx = data.get("mid_segments_x")
        raw_msy = data.get("mid_segments_y")
        raw_mcds = data.get("mid_contact_depth_solve")
        raw_fda = data.get("fg_depth_amplitude")
        raw_fpm = data.get("fg_pin_margin_uv")
        raw_fsx = data.get("fg_segments_x")
        raw_fsy = data.get("fg_segments_y")
        raw_fcds = data.get("fg_contact_depth_solve")
        raw_fpr = data.get("fg_portrait_relief")
        raw_ijqo = data.get("inpainting_jpeg_quality_override")
        raw_mbo = data.get("mirror_blur_override")
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            source_image_relpath=str(data["source_image_relpath"]),
            outpaint_raw_relpath=str(data["outpaint_raw_relpath"]),
            reflection_source_relpath=data.get("reflection_source_relpath"),
            lama_inner_output_filename=str(data["lama_inner_output_filename"]),
            bg_output_filename=str(data["bg_output_filename"]),
            mid_output_filename=str(data["mid_output_filename"]),
            fg_output_filename=str(data["fg_output_filename"]),
            depth_output_filename=str(data["depth_output_filename"]),
            reflection_output_filename=data.get("reflection_output_filename"),
            segmentation_backend=str(data["segmentation_backend"]),
            depth_backend=str(data["depth_backend"]),
            min_registration_corr=float(data["min_registration_corr"]),
            room_depth_amplitude=float(data["room_depth_amplitude"]),
            particle_color_hex=int(data["particle_color_hex"]),
            particle_count=int(data["particle_count"]),
            fg_x_range_norm=(float(xr[0]), float(xr[1])),
            fg_min_depth_quantile=float(data["fg_min_depth_quantile"]),
            sam_fg_boxes=tuple(BoundingBoxPrompt.from_dict(b) for b in data["sam_fg_boxes"]),
            sam_mid_boxes=tuple(BoundingBoxPrompt.from_dict(b) for b in data["sam_mid_boxes"]),
            drawing_fg_boxes=tuple(
                DrawingStrokeBox.from_dict(b) for b in data.get("drawing_fg_boxes", ())
            ),
            drawing_mid_boxes=tuple(
                DrawingStrokeBox.from_dict(b) for b in data.get("drawing_mid_boxes", ())
            ),
            fg_overlays=tuple(
                LayerOverlaySpec.from_dict(o) for o in data.get("fg_overlays", ())
            ),
            mid_overlays=tuple(
                LayerOverlaySpec.from_dict(o) for o in data.get("mid_overlays", ())
            ),
            reflection_apertures=tuple(
                ApertureBox.from_dict(ap) for ap in data.get("reflection_apertures", ())
            ),
            feather_collar_px=int(raw_collar) if raw_collar is not None else None,
            registration_core_margin_xy=(int(rcm[0]), int(rcm[1])),
            initial_camera_y=float(data.get("initial_camera_y", 0.0)),
            initial_camera_z=float(data.get("initial_camera_z", 400.0)),
            min_camera_z=float(data.get("min_camera_z", 120.0)),
            max_camera_z=float(data.get("max_camera_z", 685.0)),
            scene_bg_z=float(data.get("scene_bg_z", -350.0)),
            min_inner_scale=float(raw_min_is) if raw_min_is is not None else None,
            max_yaw_deg=float(data.get("max_yaw_deg", 34.0)),
            max_pitch_deg=float(data.get("max_pitch_deg", 24.0)),
            max_strafe_x=float(data.get("max_strafe_x", 250.0)),
            max_strafe_y=float(data.get("max_strafe_y", 120.0)),
            dolly_speed_mult=float(data.get("dolly_speed_mult", 1.0)),
            fg_render_order=int(data.get("fg_render_order", 3)),
            mid_render_order=int(data.get("mid_render_order", 2)),
            fg_z=float(data.get("fg_z", 0.0)),
            mid_z=float(data.get("mid_z", -150.0)),
            bg_segments_x=int(raw_seg_x) if raw_seg_x is not None else None,
            bg_segments_y=int(raw_seg_y) if raw_seg_y is not None else None,
            harmonize_seam_field=bool(data.get("harmonize_seam_field", False)),
            seam_harmonic_downsample=int(data.get("seam_harmonic_downsample", 8)),
            seam_harmonic_iters=int(data.get("seam_harmonic_iters", 240)),
            rear_chamber_soften_box=(
                BoundingBoxPrompt.from_dict(raw_rcsb) if raw_rcsb is not None else None
            ),
            pipe_tunnel_relief=(
                PipeTunnelReliefConfig.from_dict(raw_ptr) if raw_ptr is not None else None
            ),
            tunnel_bore_target=(
                TunnelBoreTargetSpec.from_dict(raw_tbt) if raw_tbt is not None else None
            ),
            diffuse_aperture_fusion=(
                DiffuseApertureFusionSpec.from_dict(raw_daf) if raw_daf is not None else None
            ),
            fg_expanded_raw_relpath=data.get("fg_expanded_raw_relpath"),
            fg_is_expanded=bool(data.get("fg_is_expanded", False)),
            sam_fg_point_groups=tuple(
                SamPointGroupPrompt.from_dict(pg) for pg in data.get("sam_fg_point_groups", ())
            ),
            fg_oriented_matte_boxes=tuple(
                OrientedMatteBoxSpec.from_dict(omb)
                for omb in data.get("fg_oriented_matte_boxes", ())
            ),
            sam_core_support_dilate_px=int(data.get("sam_core_support_dilate_px", 85)),
            sam_fg_exclude_boxes=tuple(
                BoundingBoxPrompt.from_dict(eb) for eb in data.get("sam_fg_exclude_boxes", ())
            ),
            sam_fg_right_edge_fade_x_start=int(raw_fade_x) if raw_fade_x is not None else None,
            sam_max_pinhole_fill_px=int(data.get("sam_max_pinhole_fill_px", 14000)),
            sam_min_major_component_px=int(data.get("sam_min_major_component_px", 25000)),
            extra_layers=tuple(
                ExtraDepthLayerSpec.from_dict(el) for el in data.get("extra_layers", ())
            ),
            ceiling_mirror_reflection=(
                CeilingMirrorReflectionConfig.from_dict(raw_cmr)
                if raw_cmr is not None
                else None
            ),
            bore_sight_at_rest=bool(data.get("bore_sight_at_rest", True)),
            room_depth_clean=(
                RoomDepthCleanConfig.from_dict(raw_rdc) if raw_rdc is not None else None
            ),
            room_depth_clean_layers=tuple(
                str(lid) for lid in data.get("room_depth_clean_layers", ())
            ),
            topology_verification=(
                LayerTopologyVerificationConfig.from_dict(raw_tv)
                if raw_tv is not None
                else None
            ),
            mid_depth_output_filename=data.get("mid_depth_output_filename"),
            mid_depth_amplitude=float(raw_mda) if raw_mda is not None else None,
            mid_pin_margin_uv=float(raw_mpm) if raw_mpm is not None else None,
            mid_segments_x=int(raw_msx) if raw_msx is not None else None,
            mid_segments_y=int(raw_msy) if raw_msy is not None else None,
            mid_contact_depth_solve=(
                ContactAnchoredLayerDepthConfig.from_dict(raw_mcds)
                if raw_mcds is not None
                else None
            ),
            fg_depth_output_filename=data.get("fg_depth_output_filename"),
            fg_depth_amplitude=float(raw_fda) if raw_fda is not None else None,
            fg_pin_margin_uv=float(raw_fpm) if raw_fpm is not None else None,
            fg_segments_x=int(raw_fsx) if raw_fsx is not None else None,
            fg_segments_y=int(raw_fsy) if raw_fsy is not None else None,
            fg_contact_depth_solve=(
                ContactAnchoredLayerDepthConfig.from_dict(raw_fcds)
                if raw_fcds is not None
                else None
            ),
            fg_portrait_relief=(
                NeuralPortraitReliefConfig.from_dict(raw_fpr) if raw_fpr is not None else None
            ),
            sam_mid_point_groups=tuple(
                SamPointGroupPrompt.from_dict(pg) for pg in data.get("sam_mid_point_groups", ())
            ),
            sam_mid_exclude_boxes=tuple(
                BoundingBoxPrompt.from_dict(eb) for eb in data.get("sam_mid_exclude_boxes", ())
            ),
            sam_mid_max_pinhole_fill_px=int(data.get("sam_mid_max_pinhole_fill_px", 4000)),
            sam_mid_min_major_component_px=int(data.get("sam_mid_min_major_component_px", 25)),
            inpaint_region_dilation_overrides=tuple(
                InpaintRegionDilationOverride.from_dict(ov)
                for ov in data.get("inpaint_region_dilation_overrides", ())
            ),
            inpaint_from_outpaint_inner=bool(data.get("inpaint_from_outpaint_inner", False)),
            inpainting_jpeg_quality_override=(int(raw_ijqo) if raw_ijqo is not None else None),
            canvas_scale_mult=float(data.get("canvas_scale_mult", 1.0)),
            floating_overlay_scale_mult=float(data.get("floating_overlay_scale_mult", 1.0)),
            mirror_blur_override=(
                (float(raw_mbo[0]), float(raw_mbo[1])) if raw_mbo is not None else None
            ),
            use_mirror_outpaint=bool(data.get("use_mirror_outpaint", False)),
            mid_depth_test=bool(data.get("mid_depth_test", False)),
            fg_depth_test=bool(data.get("fg_depth_test", False)),
        )


@dataclass(frozen=True)
class CameraRigConfig:
    """Canonical SSOT configuration for WebGL camera navigation, gyroscope, and portal flight."""

    desktop_fov_deg: float
    portrait_fov_deg: float
    key_move_speed_per_sec: float
    key_elevation_speed_per_sec: float
    gyro_gain: float
    gyro_max_yaw_deg: float
    gyro_max_pitch_deg: float
    gyro_smoothing_rate: float
    gyro_recenter_threshold_deg: float
    gyro_parallax_shift_x: float
    gyro_parallax_shift_y: float
    gyro_hint_dismiss_yaw_deg: float
    gyro_hint_dismiss_pitch_deg: float
    gyro_no_events_timeout_ms: int
    portal_flight_duration_ms: int
    portal_fade_start_progress: float
    portal_dust_render_order: float
    clear_color_hex: int
    portal_occluder_regen_delay_ms: int = 60000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraRigConfig:
        return cls(
            desktop_fov_deg=float(data["desktop_fov_deg"]),
            portrait_fov_deg=float(data["portrait_fov_deg"]),
            key_move_speed_per_sec=float(data["key_move_speed_per_sec"]),
            key_elevation_speed_per_sec=float(data["key_elevation_speed_per_sec"]),
            gyro_gain=float(data["gyro_gain"]),
            gyro_max_yaw_deg=float(data["gyro_max_yaw_deg"]),
            gyro_max_pitch_deg=float(data["gyro_max_pitch_deg"]),
            gyro_smoothing_rate=float(data["gyro_smoothing_rate"]),
            gyro_recenter_threshold_deg=float(data["gyro_recenter_threshold_deg"]),
            gyro_parallax_shift_x=float(data["gyro_parallax_shift_x"]),
            gyro_parallax_shift_y=float(data["gyro_parallax_shift_y"]),
            gyro_hint_dismiss_yaw_deg=float(data["gyro_hint_dismiss_yaw_deg"]),
            gyro_hint_dismiss_pitch_deg=float(data["gyro_hint_dismiss_pitch_deg"]),
            gyro_no_events_timeout_ms=int(data["gyro_no_events_timeout_ms"]),
            portal_flight_duration_ms=int(data["portal_flight_duration_ms"]),
            portal_fade_start_progress=float(data["portal_fade_start_progress"]),
            portal_dust_render_order=float(data["portal_dust_render_order"]),
            clear_color_hex=int(data["clear_color_hex"]),
            portal_occluder_regen_delay_ms=int(data.get("portal_occluder_regen_delay_ms", 60000)),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, raw_json: str) -> CameraRigConfig:
        return cls.from_dict(json.loads(raw_json))


def default_camera_rig_config() -> CameraRigConfig:
    return CameraRigConfig(
        desktop_fov_deg=45.0,
        portrait_fov_deg=56.0,
        key_move_speed_per_sec=270.0,
        key_elevation_speed_per_sec=180.0,
        gyro_gain=1.25,
        gyro_max_yaw_deg=24.0,
        gyro_max_pitch_deg=18.0,
        gyro_smoothing_rate=14.0,
        gyro_recenter_threshold_deg=26.0,
        gyro_parallax_shift_x=24.0,
        gyro_parallax_shift_y=15.0,
        gyro_hint_dismiss_yaw_deg=1.8,
        gyro_hint_dismiss_pitch_deg=1.5,
        gyro_no_events_timeout_ms=1500,
        portal_flight_duration_ms=1150,
        portal_fade_start_progress=0.75,
        portal_dust_render_order=13.5,
        clear_color_hex=0x121314,
        portal_occluder_regen_delay_ms=60000,
    )


@dataclass(frozen=True)
class SpatialPipelineConfig:
    output_dir_relpath: str
    manifest_filename: str
    dimensions: CanvasDimensionsConfig
    outpainting: OutpaintingConfig
    depth_model: DepthModelConfig
    segmentation: SegmentationConfig
    inpainting: InpaintingConfig
    graphite_drawing: GraphiteDrawingConfig
    floating_extraction: FloatingElementExtractionConfig
    camera_rig: CameraRigConfig
    scenes: tuple[SceneBuildSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir_relpath": self.output_dir_relpath,
            "manifest_filename": self.manifest_filename,
            "dimensions": self.dimensions.to_dict(),
            "outpainting": self.outpainting.to_dict(),
            "depth_model": self.depth_model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "inpainting": self.inpainting.to_dict(),
            "graphite_drawing": self.graphite_drawing.to_dict(),
            "floating_extraction": self.floating_extraction.to_dict(),
            "camera_rig": self.camera_rig.to_dict(),
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialPipelineConfig:
        raw_fe = data.get("floating_extraction")
        return cls(
            output_dir_relpath=str(data["output_dir_relpath"]),
            manifest_filename=str(data["manifest_filename"]),
            dimensions=CanvasDimensionsConfig.from_dict(data["dimensions"]),
            outpainting=OutpaintingConfig.from_dict(data["outpainting"]),
            depth_model=DepthModelConfig.from_dict(data["depth_model"]),
            segmentation=SegmentationConfig.from_dict(data["segmentation"]),
            inpainting=InpaintingConfig.from_dict(data["inpainting"]),
            graphite_drawing=GraphiteDrawingConfig.from_dict(data["graphite_drawing"]),
            floating_extraction=(
                FloatingElementExtractionConfig.from_dict(raw_fe)
                if raw_fe is not None
                else default_floating_element_extraction_config()
            ),
            camera_rig=CameraRigConfig.from_dict(data["camera_rig"]),
            scenes=tuple(SceneBuildSpec.from_dict(s) for s in data["scenes"]),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, raw_json: str) -> SpatialPipelineConfig:
        return cls.from_dict(json.loads(raw_json))

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> SpatialPipelineConfig:
        return cls.from_json(path.read_text(encoding="utf-8"))


def default_graphite_drawing_config() -> GraphiteDrawingConfig:
    return GraphiteDrawingConfig(
        paper_downsample_width=188,
        paper_downsample_height=125,
        paper_bg_percentile=88,
        paper_filter_size=11,
        frame_border_ignore_px=20,
        min_component_area_px=35,
        edge_feather_sigma=0.70,
        lama_dilation_radius_px=10,
        lama_mask_blur_sigma=1.5,
        dome_bg_percentile=85.0,
        dome_ink_delta_threshold=11.0,
        dome_morph_close_px=21,
        dome_power_exponent=0.65,
        dome_gaussian_sigma=14.0,
        stage_y_weight=0.35,
        stage_y_exponent=1.35,
        stage_x_weight=0.12,
        depth_dome_weight=0.62,
        depth_stage_weight=0.24,
        depth_neural_weight=0.14,
        depth_final_sigma=10.0,
        oval_board_scaled_width=1200,
        oval_board_scaled_height=500,
        oval_center_xy=(600.0, 252.0),
        oval_radii_xy=(578.0, 236.0),
        oval_min_component_area_px=95,
    )


def default_floating_element_extraction_config() -> FloatingElementExtractionConfig:
    return FloatingElementExtractionConfig(
        paper_downsample_divisor=10,
        paper_percentile=90,
        paper_filter_size=7,
        paper_sample_max_diff=14.0,
        min_paper_sigma=2.1,
        noise_floor_sigma_mult=3.0,
        min_delta_floor=10.5,
        ink_upper_percentile=90.0,
        min_ink_span=30.0,
        tether_binary_threshold=0.10,
        tether_window_hw=(9, 19),
        tether_min_density=0.135,
        tether_dilate_px=11,
        min_graphite_alpha=0.06,
        min_graphite_component_px=18,
        stroke_presence_boost=1.18,
        unmix_alpha_floor=0.24,
        graphite_tint_rgb=(22.0, 21.0, 20.0),
        preserve_shading_blend=0.28,
        max_shaded_lum=95.0,
        capsule_context_pad_px=60,
        wall_bg_max_alpha=0.08,
        wall_bg_percentile=75.0,
        fallback_wall_rgb=(214.0, 214.0, 214.0),
        solid_core_min_alpha=0.68,
        solid_core_min_warmth=22.0,
        solid_core_dark_lum=95.0,
        porthole_close_px=9,
        porthole_max_area_px=14000,
        pale_bleed_max_warmth=21.0,
        pale_bleed_min_lum=118.0,
        pale_bleed_max_wall_dist=95.0,
        capsule_alpha_low=0.30,
        capsule_alpha_high=0.78,
        min_capsule_component_px=180,
        capsule_alpha_sigma=0.50,
        capsule_unmix_alpha_floor=0.32,
        wisp_x_range_norm=(0.40, 0.82),
        wisp_y_max_norm=0.54,
        wisp_max_warmth=25.0,
        wisp_min_lum=115.0,
        shadow_pocket_cool_rgb=(0.74, 0.78, 0.84),
        shadow_pocket_warm_rgb=(1.02, 0.98, 0.90),
        mirror_steel_tint_rgb=(0.94, 0.97, 1.00),
        mirror_blur_base_sigma=0.36,
        mirror_blur_dist_gain=0.18,
        mirror_bg_dark_lum=55.0,
        mirror_bg_lum_span=95.0,
        mirror_illum_base=0.64,
        mirror_illum_gain=0.38,
        mirror_illum_min=0.60,
        mirror_illum_max=1.04,
    )


def _marseille_mirror_v1_scene_spec() -> SceneBuildSpec:
    """Archived Version 1 specification for Marseille Vieux-Port mirror pavilion."""
    capsules_rel = "media/binyamin_gallery/binyamin_אולי_שם_יהיה_יותר_טוב.jpg_a23a554b.jpg"
    wandering_rel = "media/binyamin_gallery/binyamin_שיטוט_2.jpg_9af0fd97.jpg"

    pod_between_gadi_guy = LayerOverlaySpec(
        source_image_relpath=capsules_rel,
        extraction_mode="isnet_dehalo_capsule",
        src_crop_norm=(0.285, 0.155, 0.415, 0.312),
        dest_center_xy=(525, 675),
        dest_wh=(118, 118),
        warm_sun_harmony=0.04,
        shadow_pocket_lighting=True,
    )
    pencil_between_guy_tami = LayerOverlaySpec(
        source_image_relpath=wandering_rel,
        extraction_mode="graphite_unmix_island",
        src_crop_norm=(0.456, 0.022, 0.668, 0.228),
        dest_center_xy=(912, 685),
        dest_wh=(138, 138),
        gamma=0.46,
        suppress_thin_tethers=True,
    )
    capsule_porthole_low_mid = LayerOverlaySpec(
        source_image_relpath=capsules_rel,
        extraction_mode="isnet_dehalo_capsule",
        src_crop_norm=(0.575, 0.308, 0.878, 0.492),
        dest_center_xy=(912, 508),
        dest_wh=(180, 180),
        warm_sun_harmony=0.14,
        trim_upper_pale_wisps=True,
        keep_largest_component_only=True,
    )
    pencil_branched_upper_mid = LayerOverlaySpec(
        source_image_relpath=wandering_rel,
        extraction_mode="graphite_unmix_island",
        src_crop_norm=(0.088, 0.155, 0.312, 0.362),
        dest_center_xy=(1015, 360),
        dest_wh=(184, 184),
        gamma=0.45,
        suppress_thin_tethers=True,
    )
    boat_spires_high_mirror = LayerOverlaySpec(
        source_image_relpath=capsules_rel,
        extraction_mode="isnet_dehalo_capsule",
        src_crop_norm=(0.760, 0.468, 0.938, 0.638),
        dest_center_xy=(815, 255),
        dest_wh=(138, 138),
        warm_sun_harmony=0.12,
        keep_largest_component_only=True,
    )

    return SceneBuildSpec(
        scene_id="marseille_mirror_v1",
        title_he="תחת תקרת המראה במרסיי (גרסה 1 שמורה)",
        short_label_he="מרסיי (גרסה 1)",
        icon="🪞",
        source_image_relpath="prototype/assets/outpaint_sources/scene6_marseille_base_1500x1000.jpg",
        outpaint_raw_relpath="prototype/assets/outpaint_sources/scene6_marseille_outpaint_raw.jpg",
        reflection_source_relpath=None,
        lama_inner_output_filename="scene6_marseille_v1_lama_inner.jpg",
        bg_output_filename="scene6_marseille_v1_bg_inpainted.jpg",
        mid_output_filename="scene6_marseille_v1_mid.png",
        fg_output_filename="scene6_marseille_v1_fg.png",
        depth_output_filename="scene6_marseille_v1_depth.png",
        reflection_output_filename="scene6_marseille_v1_reflections.png",
        segmentation_backend="sam_point_groups",
        depth_backend="depth_anything",
        min_registration_corr=0.88,
        room_depth_amplitude=68.0,
        particle_color_hex=0xE5DEC9,
        particle_count=70,
        fg_x_range_norm=(0.0, 1.0),
        fg_min_depth_quantile=0.50,
        sam_fg_boxes=(),
        sam_mid_boxes=(),
        mid_overlays=(capsule_porthole_low_mid,),
        feather_collar_px=36,
        initial_camera_y=-185.0,
        initial_camera_z=855.0,
        max_camera_z=925.0,
        fg_render_order=6,
        mid_render_order=4,
        fg_expanded_raw_relpath="prototype/assets/outpaint_sources/scene6_marseille_people_expanded_raw.jpg",
        fg_is_expanded=True,
        sam_fg_point_groups=(
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=0, y0=0, x1=1308, y1=1340),
                points_xy=(
                    (320.0, 360.0),
                    (260.0, 820.0),
                    (680.0, 560.0),
                    (920.0, 480.0),
                    (1000.0, 60.0),
                    (1095.0, 300.0),
                    (1155.0, 385.0),
                    (1165.0, 465.0),
                    (740.0, 880.0),
                    (1165.0, 765.0),
                    (1225.0, 815.0),
                    (1215.0, 715.0),
                    (980.0, 980.0),
                    (620.0, 1220.0),
                    (1155.0, 935.0),
                    (1275.0, 885.0),
                    (1335.0, 815.0),
                    (1225.0, 580.0),
                    (1235.0, 460.0),
                    (1160.0, 160.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=920, y0=0, x1=1235, y1=605),
                points_xy=(
                    (995.0, 35.0),
                    (1035.0, 70.0),
                    (1070.0, 185.0),
                    (1095.0, 300.0),
                    (1155.0, 380.0),
                    (1175.0, 460.0),
                    (1140.0, 505.0),
                    (1090.0, 545.0),
                    (1120.0, 80.0),
                    (1165.0, 210.0),
                    (1215.0, 350.0),
                    (1225.0, 470.0),
                    (1205.0, 565.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=1055, y0=650, x1=1305, y1=902),
                points_xy=(
                    (1145.0, 765.0),
                    (1215.0, 795.0),
                    (1235.0, 835.0),
                    (1215.0, 705.0),
                    (1135.0, 888.0),
                    (1275.0, 875.0),
                    (1288.0, 735.0),
                ),
                point_labels=(1, 1, 1, 1, 0, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=760, y0=930, x1=1515, y1=1800),
                points_xy=(
                    (1335.0, 985.0),
                    (1335.0, 1055.0),
                    (1330.0, 1135.0),
                    (1315.0, 1225.0),
                    (1060.0, 1280.0),
                    (1220.0, 1340.0),
                    (1240.0, 1620.0),
                    (895.0, 1580.0),
                    (935.0, 1385.0),
                    (1150.0, 1060.0),
                    (1225.0, 1060.0),
                    (960.0, 1160.0),
                    (795.0, 1520.0),
                    (1445.0, 1060.0),
                    (1465.0, 1205.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=1225, y0=932, x1=1418, y1=1265),
                points_xy=(
                    (1335.0, 975.0),
                    (1335.0, 1045.0),
                    (1330.0, 1125.0),
                    (1320.0, 1210.0),
                    (1242.0, 1050.0),
                    (1408.0, 1050.0),
                    (1242.0, 950.0),
                ),
                point_labels=(1, 1, 1, 1, 0, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=1435, y0=1175, x1=2102, y1=1800),
                points_xy=(
                    (1565.0, 1235.0),
                    (1535.0, 1305.0),
                    (1600.0, 1390.0),
                    (1675.0, 1315.0),
                    (1662.0, 1328.0),
                    (1890.0, 1355.0),
                    (1835.0, 1415.0),
                    (1785.0, 1485.0),
                    (1855.0, 1555.0),
                    (1635.0, 1560.0),
                    (1755.0, 1665.0),
                    (1580.0, 1155.0),
                    (1655.0, 1185.0),
                    (1820.0, 1235.0),
                    (1955.0, 1690.0),
                    (2015.0, 1660.0),
                    (1950.0, 1605.0),
                    (2065.0, 1520.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
            ),
        ),
        sam_fg_exclude_boxes=(
            BoundingBoxPrompt(x0=1300, y0=750, x1=1402, y1=875),
            BoundingBoxPrompt(x0=1545, y0=905, x1=1745, y1=1030),
        ),
        sam_fg_right_edge_fade_x_start=2068,
        sam_max_pinhole_fill_px=14000,
        sam_min_major_component_px=25000,
        extra_layers=(
            ExtraDepthLayerSpec(
                layer_id="between_people",
                label_he="מרחב קרוב בין הדמויות (קפסולה ואי רישום)",
                output_filename="scene6_marseille_v1_between.png",
                z_pos=-70.0,
                render_order=5,
                overlays=(pod_between_gadi_guy, pencil_between_guy_tami),
            ),
            ExtraDepthLayerSpec(
                layer_id="upper_mid",
                label_he="מרחב עליון מעל הדמויות (אי רישום מסועף)",
                output_filename="scene6_marseille_v1_upper_mid.png",
                z_pos=-215.0,
                render_order=3,
                overlays=(pencil_branched_upper_mid,),
            ),
            ExtraDepthLayerSpec(
                layer_id="high_mirror",
                label_he="מרחב גבוה סמוך לתקרת המראה (סירת צריחים מנייר)",
                output_filename="scene6_marseille_v1_high.png",
                z_pos=-275.0,
                render_order=2,
                overlays=(boat_spires_high_mirror,),
            ),
        ),
        ceiling_mirror_reflection=CeilingMirrorReflectionConfig(
            vanishing_point_xy=(737.2, 434.6),
            base_opacity=0.92,
            excursion_boost=0.08,
            items=(
                MirrorReflectedItemSpec(
                    overlay=pod_between_gadi_guy, height_norm=0.44, steel_reflectance=0.78
                ),
                MirrorReflectedItemSpec(
                    overlay=pencil_between_guy_tami, height_norm=0.48, steel_reflectance=0.84
                ),
                MirrorReflectedItemSpec(
                    overlay=capsule_porthole_low_mid, height_norm=0.50, steel_reflectance=0.78
                ),
                MirrorReflectedItemSpec(
                    overlay=pencil_branched_upper_mid, height_norm=0.64, steel_reflectance=0.84
                ),
                MirrorReflectedItemSpec(
                    overlay=boat_spires_high_mirror, height_norm=0.68, steel_reflectance=0.78
                ),
            ),
        ),
    )


def _marseille_mirror_scene_spec() -> SceneBuildSpec:
    """
    Canonical Version 2 SSOT specification for Marseille Vieux-Port mirror pavilion:
    1. Upright-oriented IS-Net + SAM hybrid hair & eyeglass matting on Gadi, Tami, and Guy.
    2. Multi-person 3D depth separation: Gadi & Tami at Z=0, sculptural refuge capsule floating
       in front of Guy's chest/abdomen at Z=-28, pod next to Guy's head at Z=-54, Guy at Z=-76.
    3. Multi-plane 3D virtual reflections behind Z_mirror=-310: reflected artworks at Z=-355,
       -435, -505, SAM-segmented reflected people at Z=-555, and virtual plaza relief at Z=-700.
    4. Expanded wide-angle zoom-out (Z up to 1620) & deep pass-through zoom-in to mirror (Z=-340).
    """
    capsules_rel = "media/binyamin_gallery/binyamin_אולי_שם_יהיה_יותר_טוב.jpg_a23a554b.jpg"
    wandering_rel = "media/binyamin_gallery/binyamin_שיטוט_2.jpg_9af0fd97.jpg"
    vp_xy = (737.2, 434.6)

    capsule_chest_between_people = LayerOverlaySpec(
        source_image_relpath=capsules_rel,
        extraction_mode="isnet_dehalo_capsule",
        src_crop_norm=(0.238, 0.365, 0.440, 0.505),
        dest_center_xy=(1215, 1308),
        dest_wh=(182, 182),
        warm_sun_harmony=0.10,
        keep_largest_component_only=True,
        trim_vertical_tether_ends=True,
    )
    pod_between_gadi_guy = LayerOverlaySpec(
        source_image_relpath=capsules_rel,
        extraction_mode="isnet_dehalo_capsule",
        src_crop_norm=(0.285, 0.155, 0.415, 0.312),
        dest_center_xy=(525, 675),
        dest_wh=(118, 118),
        warm_sun_harmony=0.04,
        shadow_pocket_lighting=True,
    )
    pencil_between_guy_tami = LayerOverlaySpec(
        source_image_relpath=wandering_rel,
        extraction_mode="graphite_unmix_island",
        src_crop_norm=(0.456, 0.022, 0.668, 0.228),
        dest_center_xy=(912, 685),
        dest_wh=(138, 138),
        gamma=0.46,
        suppress_thin_tethers=True,
    )
    capsule_porthole_low_mid = LayerOverlaySpec(
        source_image_relpath=capsules_rel,
        extraction_mode="isnet_dehalo_capsule",
        src_crop_norm=(0.575, 0.308, 0.878, 0.492),
        dest_center_xy=(912, 508),
        dest_wh=(180, 180),
        warm_sun_harmony=0.14,
        trim_upper_pale_wisps=True,
        keep_largest_component_only=True,
    )
    pencil_branched_upper_mid = LayerOverlaySpec(
        source_image_relpath=wandering_rel,
        extraction_mode="graphite_unmix_island",
        src_crop_norm=(0.088, 0.155, 0.312, 0.362),
        dest_center_xy=(1015, 360),
        dest_wh=(184, 184),
        gamma=0.45,
        suppress_thin_tethers=True,
    )
    boat_spires_high_mirror = LayerOverlaySpec(
        source_image_relpath=capsules_rel,
        extraction_mode="isnet_dehalo_capsule",
        src_crop_norm=(0.760, 0.468, 0.938, 0.638),
        dest_center_xy=(815, 255),
        dest_wh=(138, 138),
        warm_sun_harmony=0.12,
        keep_largest_component_only=True,
    )

    gadi_refl_excl_1 = BoundingBoxPrompt(x0=1272, y0=725, x1=1405, y1=852)
    gadi_refl_excl_2 = BoundingBoxPrompt(x0=1262, y0=852, x1=1405, y1=935)
    tami_pole_excl_1 = BoundingBoxPrompt(x0=1580, y0=1105, x1=1745, y1=1166)
    tami_pole_excl_2 = BoundingBoxPrompt(x0=1628, y0=1166, x1=1745, y1=1194)
    tami_pole_excl_3 = BoundingBoxPrompt(x0=1668, y0=1194, x1=1745, y1=1222)
    tami_pole_excl_4 = BoundingBoxPrompt(x0=1698, y0=1222, x1=1745, y1=1250)
    tami_left_shadow_excl_1 = BoundingBoxPrompt(x0=1395, y0=1295, x1=1424, y1=1325)
    tami_left_shadow_excl_2 = BoundingBoxPrompt(x0=1395, y0=1325, x1=1450, y1=1364)
    tami_left_shadow_excl_3 = BoundingBoxPrompt(x0=1395, y0=1364, x1=1456, y1=1402)
    tami_dup_thumb_sliver_excl = BoundingBoxPrompt(x0=1844, y0=1602, x1=1905, y1=1628)
    tami_dup_arm_right_excl = BoundingBoxPrompt(x0=2102, y0=1500, x1=2700, y1=1800)

    return SceneBuildSpec(
        scene_id="marseille_mirror",
        title_he="תחת תקרת המראה במרסיי — קפסולות ואיים מרחפים",
        short_label_he="מרסיי: מתחת למראה",
        icon="🪞",
        source_image_relpath="prototype/assets/outpaint_sources/scene6_marseille_base_1500x1000.jpg",
        outpaint_raw_relpath="prototype/assets/outpaint_sources/scene6_marseille_outpaint_raw.jpg",
        reflection_source_relpath=None,
        lama_inner_output_filename="scene6_marseille_lama_inner.jpg",
        bg_output_filename="scene6_marseille_bg_inpainted.jpg",
        mid_output_filename="scene6_marseille_mid.png",
        fg_output_filename="scene6_marseille_fg.png",
        depth_output_filename="scene6_marseille_depth.png",
        reflection_output_filename=None,
        segmentation_backend="sam_point_groups",
        depth_backend="depth_anything",
        min_registration_corr=0.88,
        room_depth_amplitude=105.0,
        particle_color_hex=0xE5DEC9,
        particle_count=70,
        fg_x_range_norm=(0.0, 1.0),
        fg_min_depth_quantile=0.50,
        sam_fg_boxes=(),
        sam_mid_boxes=(),
        mid_overlays=(capsule_porthole_low_mid,),
        feather_collar_px=36,
        initial_camera_y=-239.0,
        initial_camera_z=855.0,
        min_camera_z=-340.0,
        max_camera_z=1420.0,
        scene_bg_z=-700.0,
        max_yaw_deg=58.0,
        max_pitch_deg=44.0,
        max_strafe_x=560.0,
        max_strafe_y=420.0,
        dolly_speed_mult=1.75,
        fg_render_order=10,
        mid_render_order=7,
        fg_z=-24.0,
        fg_depth_output_filename="scene6_marseille_fg_depth.png",
        fg_depth_amplitude=24.0,
        fg_pin_margin_uv=0.0,
        fg_segments_x=224,
        fg_segments_y=150,
        fg_portrait_relief=NeuralPortraitReliefConfig(
            layer_z=-24.0,
            layer_depth_amplitude=24.0,
            layer_pin_margin_uv=0.0,
            alpha_threshold_u8=20,
            min_component_area_px=50,
            lower_percentile=5.0,
            upper_percentile=98.0,
            interior_smooth_sigma_px=2.4,
            pin_edges=(True, False, True, True),
            border_band_px=6,
            border_taper_px=72.0,
        ),
        room_depth_clean=RoomDepthCleanConfig(
            alpha_threshold_u8=16,
            mask_dilation_px=20,
            preserve_boxes_expanded=((0, 0, 2700, 680),),
            post_smooth_sigma_px=3.0,
            interior_blend_band_px=12.0,
        ),
        room_depth_clean_layers=("refl_people",),
        topology_verification=LayerTopologyVerificationConfig(),
        fg_expanded_raw_relpath="prototype/assets/outpaint_sources/scene6_marseille_people_expanded_raw.jpg",
        fg_is_expanded=True,
        sam_fg_point_groups=(
            # Gadi: fleece hoodie + backpack + head/glasses
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=0, y0=0, x1=1308, y1=1340),
                points_xy=(
                    (320.0, 360.0),
                    (260.0, 820.0),
                    (680.0, 560.0),
                    (920.0, 480.0),
                    (1000.0, 60.0),
                    (1095.0, 300.0),
                    (1155.0, 385.0),
                    (1165.0, 465.0),
                    (740.0, 880.0),
                    (1165.0, 765.0),
                    (1225.0, 815.0),
                    (1215.0, 715.0),
                    (980.0, 980.0),
                    (620.0, 1220.0),
                    (1155.0, 935.0),
                    (1275.0, 885.0),
                    (1335.0, 815.0),
                    (1225.0, 580.0),
                    (1235.0, 460.0),
                    (1160.0, 160.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            ),
            # Gadi upper hood & backpack flap fine detail
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=920, y0=0, x1=1235, y1=605),
                points_xy=(
                    (995.0, 35.0),
                    (1035.0, 70.0),
                    (1070.0, 185.0),
                    (1095.0, 300.0),
                    (1155.0, 380.0),
                    (1175.0, 460.0),
                    (1140.0, 505.0),
                    (1090.0, 545.0),
                    (1120.0, 80.0),
                    (1165.0, 210.0),
                    (1215.0, 350.0),
                    (1225.0, 470.0),
                    (1205.0, 565.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
            ),
            # Gadi head & glasses fine detail
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=1055, y0=640, x1=1305, y1=915),
                points_xy=(
                    (1145.0, 765.0),
                    (1215.0, 795.0),
                    (1235.0, 835.0),
                    (1215.0, 705.0),
                    (1135.0, 898.0),
                    (1288.0, 885.0),
                    (1292.0, 735.0),
                ),
                point_labels=(1, 1, 1, 1, 0, 0, 0),
            ),
            # Tami upper: silver hair crown + glasses + laughing face + real phone + raised right arm
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=1415, y0=1085, x1=2102, y1=1558),
                points_xy=(
                    (1565.0, 1235.0),
                    (1605.0, 1205.0),
                    (1640.0, 1230.0),
                    (1535.0, 1305.0),
                    (1600.0, 1390.0),
                    (1675.0, 1315.0),
                    (1662.0, 1328.0),
                    (1890.0, 1355.0),
                    (1835.0, 1415.0),
                    (1985.0, 1455.0),
                    (2045.0, 1495.0),
                    (1785.0, 1485.0),
                    (1495.0, 1420.0),
                    (1498.0, 1465.0),
                    (1512.0, 1520.0),
                    (1430.0, 1350.0),
                    (1445.0, 1480.0),
                    (1455.0, 1535.0),
                    (1720.0, 1135.0),
                    (1820.0, 1235.0),
                    (1940.0, 1540.0),
                    (1990.0, 1548.0),
                    (1895.0, 1548.0),
                ),
                point_labels=(
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ),
            ),
            # Tami lower: left silver hair tips + olive-linen shirt torso (excluding duplicate hand+phone)
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=1415, y0=1380, x1=1902, y1=1800),
                points_xy=(
                    (1600.0, 1430.0),
                    (1498.0, 1465.0),
                    (1512.0, 1520.0),
                    (1528.0, 1565.0),
                    (1785.0, 1485.0),
                    (1830.0, 1495.0),
                    (1635.0, 1560.0),
                    (1785.0, 1565.0),
                    (1755.0, 1665.0),
                    (1560.0, 1740.0),
                    (1755.0, 1745.0),
                    (1545.0, 1788.0),
                    (1640.0, 1788.0),
                    (1755.0, 1788.0),
                    (1835.0, 1788.0),
                    (1445.0, 1480.0),
                    (1455.0, 1545.0),
                    (1475.0, 1615.0),
                    (1882.0, 1565.0),
                    (1870.0, 1606.0),
                    (1848.0, 1665.0),
                    (1868.0, 1720.0),
                    (1892.0, 1765.0),
                ),
                point_labels=(
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ),
            ),
        ),
        fg_oriented_matte_boxes=(
            OrientedMatteBoxSpec(
                box=BoundingBoxPrompt(x0=940, y0=560, x1=1340, y1=935),
                upright_angle_deg=125.0,
                taper_margin_px=10,
                sam_core_erode_px=9,
                exclude_boxes=(gadi_refl_excl_1, gadi_refl_excl_2),
            ),
            OrientedMatteBoxSpec(
                box=BoundingBoxPrompt(x0=1395, y0=1060, x1=1815, y1=1605),
                upright_angle_deg=-25.0,
                taper_margin_px=10,
                sam_core_erode_px=9,
                exclude_boxes=(
                    tami_pole_excl_1,
                    tami_pole_excl_2,
                    tami_pole_excl_3,
                    tami_pole_excl_4,
                    tami_left_shadow_excl_1,
                    tami_left_shadow_excl_2,
                    tami_left_shadow_excl_3,
                ),
                suppress_warm_box=BoundingBoxPrompt(x0=1390, y0=1330, x1=1462, y1=1412),
                suppress_warm_rg_diff=21.0,
                suppress_warm_sigma=2.2,
                suppress_warm_gain=1.35,
            ),
        ),
        sam_core_support_dilate_px=85,
        sam_fg_exclude_boxes=(
            gadi_refl_excl_1,
            gadi_refl_excl_2,
            tami_pole_excl_1,
            tami_pole_excl_2,
            tami_pole_excl_3,
            tami_pole_excl_4,
            tami_left_shadow_excl_1,
            tami_left_shadow_excl_2,
            tami_left_shadow_excl_3,
            tami_dup_thumb_sliver_excl,
            tami_dup_arm_right_excl,
            BoundingBoxPrompt(x0=1545, y0=905, x1=1745, y1=1030),
        ),
        sam_fg_right_edge_fade_x_start=2068,
        sam_max_pinhole_fill_px=14000,
        sam_min_major_component_px=25000,
        extra_layers=(
            ExtraDepthLayerSpec(
                layer_id="between_chest",
                label_he="קפסולת מפלט מרחפת בחלל שבין הדמויות (מול החזה של גיא)",
                output_filename="scene6_marseille_between_chest.png",
                z_pos=-28.0,
                render_order=9,
                overlays=(capsule_chest_between_people,),
                is_expanded=True,
            ),
            ExtraDepthLayerSpec(
                layer_id="between_people",
                label_he="מרחב קרוב לצד ראשו של גיא ובין הדמויות",
                output_filename="scene6_marseille_between.png",
                z_pos=-54.0,
                render_order=9,
                overlays=(pod_between_gadi_guy, pencil_between_guy_tami),
            ),
            ExtraDepthLayerSpec(
                layer_id="guy_mid_fg",
                label_he="דמותו של גיא בעומק ביניים (מאחורי קפסולת המפלט)",
                output_filename="scene6_marseille_guy.png",
                z_pos=-92.0,
                render_order=8,
                is_expanded=True,
                depth_test=False,
                depth_write=False,
                depth_output_filename="scene6_marseille_guy_depth.png",
                depth_amplitude=36.0,
                pin_margin_uv=0.0,
                segments_x=224,
                segments_y=150,
                portrait_relief=NeuralPortraitReliefConfig(
                    layer_z=-92.0,
                    layer_depth_amplitude=36.0,
                    layer_pin_margin_uv=0.0,
                    alpha_threshold_u8=20,
                    min_component_area_px=50,
                    lower_percentile=5.0,
                    upper_percentile=98.0,
                    interior_smooth_sigma_px=2.4,
                    pin_edges=(False, False, False, True),
                    border_band_px=6,
                    border_taper_px=72.0,
                ),
                sam_point_groups=(
                    SamPointGroupPrompt(
                        box=BoundingBoxPrompt(x0=760, y0=915, x1=1515, y1=1800),
                        points_xy=(
                            (1335.0, 985.0),
                            (1335.0, 1055.0),
                            (1330.0, 1135.0),
                            (1315.0, 1225.0),
                            (1060.0, 1280.0),
                            (1220.0, 1340.0),
                            (1240.0, 1620.0),
                            (895.0, 1580.0),
                            (935.0, 1385.0),
                            (880.0, 1788.0),
                            (1080.0, 1788.0),
                            (1280.0, 1788.0),
                            (1410.0, 1788.0),
                            (1150.0, 1060.0),
                            (1225.0, 1060.0),
                            (960.0, 1160.0),
                            (795.0, 1520.0),
                            (1445.0, 1060.0),
                            (1475.0, 1205.0),
                        ),
                        point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
                    ),
                    SamPointGroupPrompt(
                        box=BoundingBoxPrompt(x0=1220, y0=915, x1=1422, y1=1265),
                        points_xy=(
                            (1335.0, 975.0),
                            (1335.0, 1045.0),
                            (1330.0, 1125.0),
                            (1320.0, 1210.0),
                            (1242.0, 1050.0),
                            (1412.0, 1050.0),
                            (1242.0, 945.0),
                        ),
                        point_labels=(1, 1, 1, 1, 0, 0, 0),
                    ),
                ),
                oriented_matte_boxes=(
                    OrientedMatteBoxSpec(
                        box=BoundingBoxPrompt(x0=1195, y0=890, x1=1445, y1=1275),
                        upright_angle_deg=0.0,
                        taper_margin_px=10,
                        sam_core_erode_px=9,
                        exclude_boxes=(BoundingBoxPrompt(x0=1195, y0=1122, x1=1228, y1=1185),),
                    ),
                ),
                min_major_component_px=20000,
                occluder_inpaint_boxes=(BoundingBoxPrompt(x0=1445, y0=1265, x1=1565, y1=1800),),
                right_shoulder_extend_box=BoundingBoxPrompt(x0=1415, y0=1305, x1=1545, y1=1800),
                right_shoulder_extend_px=38,
            ),
            ExtraDepthLayerSpec(
                layer_id="upper_mid",
                label_he="מרחב עליון מעל הדמויות (אי רישום מסועף)",
                output_filename="scene6_marseille_upper_mid.png",
                z_pos=-215.0,
                render_order=6,
                overlays=(pencil_branched_upper_mid,),
            ),
            ExtraDepthLayerSpec(
                layer_id="high_mirror",
                label_he="מרחב גבוה סמוך לתקרת המראה (סירת צריחים מנייר)",
                output_filename="scene6_marseille_high.png",
                z_pos=-268.0,
                render_order=5,
                overlays=(boat_spires_high_mirror,),
            ),
            ExtraDepthLayerSpec(
                layer_id="refl_high",
                label_he="השתקפות קרובה במראה (סירת צריחים מנייר)",
                output_filename="scene6_marseille_refl_high.png",
                z_pos=-355.0,
                render_order=4,
                base_opacity=0.92,
                excursion_boost=0.08,
                ceiling_mirror_reflection=CeilingMirrorReflectionConfig(
                    vanishing_point_xy=vp_xy,
                    base_opacity=0.92,
                    excursion_boost=0.08,
                    items=(
                        MirrorReflectedItemSpec(
                            overlay=boat_spires_high_mirror,
                            height_norm=0.68,
                            steel_reflectance=0.78,
                        ),
                    ),
                ),
            ),
            ExtraDepthLayerSpec(
                layer_id="refl_mid",
                label_he="השתקפות אמצעית במראה (אי מסועף וקפסולת אשנב)",
                output_filename="scene6_marseille_refl_mid.png",
                z_pos=-435.0,
                render_order=3,
                base_opacity=0.92,
                excursion_boost=0.08,
                ceiling_mirror_reflection=CeilingMirrorReflectionConfig(
                    vanishing_point_xy=vp_xy,
                    base_opacity=0.92,
                    excursion_boost=0.08,
                    items=(
                        MirrorReflectedItemSpec(
                            overlay=pencil_branched_upper_mid,
                            height_norm=0.64,
                            steel_reflectance=0.84,
                        ),
                        MirrorReflectedItemSpec(
                            overlay=capsule_porthole_low_mid,
                            height_norm=0.50,
                            steel_reflectance=0.78,
                        ),
                    ),
                ),
            ),
            ExtraDepthLayerSpec(
                layer_id="refl_between",
                label_he="השתקפות עמוקה במראה (קפסולות ואי בין הדמויות)",
                output_filename="scene6_marseille_refl_between.png",
                z_pos=-505.0,
                render_order=2,
                base_opacity=0.92,
                excursion_boost=0.08,
                ceiling_mirror_reflection=CeilingMirrorReflectionConfig(
                    vanishing_point_xy=vp_xy,
                    base_opacity=0.92,
                    excursion_boost=0.08,
                    items=(
                        MirrorReflectedItemSpec(
                            overlay=pod_between_gadi_guy,
                            height_norm=0.44,
                            steel_reflectance=0.78,
                        ),
                        MirrorReflectedItemSpec(
                            overlay=pencil_between_guy_tami,
                            height_norm=0.48,
                            steel_reflectance=0.84,
                        ),
                        MirrorReflectedItemSpec(
                            overlay=capsule_chest_between_people,
                            height_norm=0.26,
                            steel_reflectance=0.78,
                            overlay_is_expanded=True,
                        ),
                    ),
                ),
            ),
            ExtraDepthLayerSpec(
                layer_id="refl_people",
                label_he="השתקפות תלת-ממדית של הדמויות בכיכר האבן",
                output_filename="scene6_marseille_refl_people.png",
                z_pos=-675.0,
                render_order=1,
                depth_test=False,
                depth_write=False,
                depth_output_filename="scene6_marseille_refl_people_depth.png",
                depth_amplitude=60.0,
                pin_margin_uv=0.0,
                segments_x=192,
                segments_y=128,
                contact_depth_solve=ContactAnchoredLayerDepthConfig(
                    bg_z=-700.0,
                    room_depth_amplitude=105.0,
                    room_pin_margin_uv=0.045,
                    layer_z=-675.0,
                    layer_depth_amplitude=60.0,
                    layer_pin_margin_uv=0.0,
                    alpha_threshold_u8=20,
                    component_bridge_px=12,
                    min_component_area_px=1,
                    contact_boxes=(
                        ContactAnchorBox(
                            x0=675, y0=344, x1=735, y1=392, mode="top_band", band_px=6
                        ),
                        ContactAnchorBox(
                            x0=992, y0=524, x1=1068, y1=608, mode="left_band", band_px=6
                        ),
                    ),
                    neural_relief_amplitude=14.0,
                    contact_relief_taper_px=6.0,
                    enforce_front_of_host=True,
                    host_clearance_tolerance_world=0.0,
                    interior_smooth_sigma_px=1.2,
                ),
                sam_boxes=(
                    BoundingBoxPrompt(x0=662, y0=340, x1=790, y1=468),
                    BoundingBoxPrompt(x0=990, y0=520, x1=1142, y1=626),
                ),
                pre_lama_dilation_px=8,
                pre_lama_blur_sigma=1.2,
            ),
        ),
        ceiling_mirror_reflection=None,
        inpaint_from_outpaint_inner=True,
        inpainting_jpeg_quality_override=96,
        canvas_scale_mult=1.4,
        floating_overlay_scale_mult=2.0,
        mirror_blur_override=(0.08, 0.04),
    )


def _default_plan_b_pipe_tunnel_relief() -> PipeTunnelReliefConfig:
    """
    True two-surface 3D occlusion-split configuration for Plan B (v2 & v3):
    1) Rigid concrete pipe bell + corner-hinged horizontal conduit shell (`pipe_shell` at Z=+8..-395)
       with EDT-extrapolated exterior depth so straddling triangles stay 100% undeformed.
    2) Clean LaMa-inpainted Binyamin Gallery wall, corner, and floor backdrop (`layer0` at Z=-430).
    """
    return PipeTunnelReliefConfig(
        front_center_xy=(807.5, 510.5),
        front_radii_xy=(421.0, 410.5),
        rear_center_xy=(878.0, 480.0),
        rear_radii_xy=(152.0, 168.0),
        scene_bg_z=-430.0,
        pipe_depth_amplitude=470.0,
        reference_camera_z=400.0,
        front_rim_z=8.0,
        rim_face_width_px=38.0,
        rear_rim_z=-265.0,
        rear_chamber_z=-395.0,
        rear_chamber_taper_r=0.88,
        rear_chamber_core_clamp_ratio=0.94,
        mouth_join_blend_px=76.0,
        bell_sam_prompt=SamPointGroupPrompt(
            box=BoundingBoxPrompt(x0=968, y0=482, x1=1848, y1=1338),
            points_xy=(
                (1408.0, 910.0),
                (1030.0, 910.0),
                (1800.0, 910.0),
                (1408.0, 520.0),
                (1408.0, 1295.0),
                (1125.0, 625.0),
                (1690.0, 625.0),
                (1125.0, 1195.0),
                (1690.0, 1195.0),
                (1408.0, 455.0),
                (1408.0, 1352.0),
                (1862.0, 910.0),
                (1772.0, 560.0),
                (1772.0, 1265.0),
                (1160.0, 1290.0),
            ),
            point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
        ),
        conduit_sam_prompt=SamPointGroupPrompt(
            box=BoundingBoxPrompt(x0=0, y0=600, x1=1120, y1=1105),
            points_xy=(
                (60.0, 840.0),
                (260.0, 840.0),
                (520.0, 850.0),
                (760.0, 860.0),
                (940.0, 740.0),
                (940.0, 985.0),
                (60.0, 660.0),
                (260.0, 675.0),
                (60.0, 1050.0),
                (300.0, 1035.0),
                (40.0, 608.0),
                (180.0, 618.0),
                (360.0, 638.0),
                (615.0, 662.0),
                (880.0, 660.0),
                (40.0, 1102.0),
                (260.0, 1084.0),
                (615.0, 1056.0),
                (880.0, 1070.0),
            ),
            point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ),
        bell_sam_max_pinhole_px=950000,
        bell_sam_min_component_px=25000,
        conduit_sam_max_pinhole_px=200000,
        conduit_sam_min_component_px=15000,
        polar_angle_samples=720,
        polar_sam_probe_inward_px=36.0,
        polar_sam_probe_outward_px=28.0,
        polar_refine_inward_px=8.0,
        polar_refine_outward_px=5.0,
        polar_da2_edge_weight=0.45,
        wall_bright_lum_threshold=165.0,
        wall_bright_penalty_weight=0.35,
        rim_edge_inward_bias_px=0.8,
        polar_smooth_sigma_samples=4.0,
        left_join_angle_range_deg=(146.0, 226.0),
        left_pipe_search_end_exp_x=985,
        left_pipe_join_exp_x=1095,
        left_pipe_top_clamp_exp_y=(580, 740),
        left_pipe_bot_clamp_exp_y=(980, 1130),
        left_pipe_snap_window_px=(6, 9),
        left_pipe_elbow_exp_x=615.0,
        left_pipe_corner_shadow_halfwidth_px=60.0,
        left_pipe_corner_shadow_weight=0.10,
        left_pipe_irls_iters=12,
        left_pipe_irls_huber_px=1.5,
        left_pipe_hinge_smooth_sigma_px=10.0,
        left_pipe_left_edge_z=-285.0,
        left_pipe_elbow_z=-355.0,
        left_pipe_mouth_join_z=-18.0,
        left_pipe_left_cyl_radius_z=24.0,
        left_pipe_join_cyl_radius_z=36.0,
        shell_alpha_aa_halfwidth_px=0.65,
        shell_pure_concrete_erode_px=3,
        shell_edge_pad_outward_px=10,
        neural_relief_amplitude_z=16.0,
        neural_highpass_sigma=36.0,
        interior_smooth_sigma_px=2.8,
        bisection_iters=24,
        wall_inpaint_downsample_wh=(1080, 720),
        wall_inpaint_dilation_px=28,
        wall_inpaint_extra_boxes_exp=(
            BoundingBoxPrompt(x0=1458, y0=1308, x1=1492, y1=1406),
        ),
        wall_inpaint_feather_px=10.0,
        wall_grain_patch_exp_xyxy=(900, 120, 1800, 420),
        wall_grain_highpass_sigma_px=2.2,
        wall_grain_strength=0.85,
        flat_wall_depth_max_u8=12,
        contact_pins=(
            ContactDepthPinSpec(
                inner_xy=(714.0, 792.0),
                target_z=-35.0,
                radius_px=28.0,
                feather_px=44.0,
            ),
            ContactDepthPinSpec(
                inner_xy=(691.0, 288.0),
                target_z=-95.0,
                radius_px=26.0,
                feather_px=42.0,
            ),
            ContactDepthPinSpec(
                inner_xy=(882.0, 669.0),
                target_z=-180.0,
                radius_px=26.0,
                feather_px=40.0,
            ),
            ContactDepthPinSpec(
                inner_xy=(946.0, 646.0),
                target_z=-180.0,
                radius_px=28.0,
                feather_px=42.0,
            ),
            ContactDepthPinSpec(
                inner_xy=(894.0, 334.0),
                target_z=-265.0,
                radius_px=22.0,
                feather_px=36.0,
            ),
        ),
    )


def _plan_b_pipe_v1_scene_spec() -> SceneBuildSpec:
    """Archived Version 1 specification for Plan B (preserved per beta feedback #10)."""
    return SceneBuildSpec(
        scene_id="plan_b_pipe_v1",
        title_he="Plan B — מבעד לצינור המלט והנייר (גרסה 1 שמורה)",
        short_label_he="Plan B (גרסה 1)",
        icon="🕳️",
        source_image_relpath="media/binyamin_gallery/binyamin_Plan_B_פרט.jpg_14b611ac.jpg",
        outpaint_raw_relpath="prototype/assets/outpaint_sources/scene4_plan_b_outpaint_raw.jpg",
        reflection_source_relpath="media/darchei_etzim_pdfs/darchei_etzim_2011_plate_035.jpg_d20d1be9.jpg",
        lama_inner_output_filename="scene4_plan_b_v1_lama_inner.jpg",
        bg_output_filename="scene4_plan_b_v1_bg_inpainted.jpg",
        mid_output_filename="scene4_plan_b_v1_mid.png",
        fg_output_filename="scene4_plan_b_v1_fg.png",
        depth_output_filename="scene4_plan_b_v1_depth.png",
        reflection_output_filename="scene4_plan_b_v1_reflections.png",
        segmentation_backend="sam_boxes",
        depth_backend="depth_anything",
        min_registration_corr=0.92,
        room_depth_amplitude=104.0,
        particle_color_hex=0x9FB7CC,
        particle_count=75,
        fg_x_range_norm=(0.40, 0.56),
        fg_min_depth_quantile=0.55,
        sam_fg_boxes=(
            BoundingBoxPrompt(x0=620, y0=284, x1=766, y1=552),
            BoundingBoxPrompt(x0=658, y0=588, x1=788, y1=796),
        ),
        sam_mid_boxes=(
            BoundingBoxPrompt(x0=868, y0=330, x1=925, y1=438),
            BoundingBoxPrompt(x0=866, y0=500, x1=904, y1=672),
            BoundingBoxPrompt(x0=892, y0=560, x1=998, y1=668),
        ),
        reflection_apertures=(
            ApertureBox(x0=768, y0=340, x1=968, y1=575, max_alpha=72, feather_sigma=14.0),
        ),
    )


def _plan_b_pipe_v2_scene_spec() -> SceneBuildSpec:
    """
    Version 2 of Plan B: true two-surface 3D occlusion split between the flat Binyamin Gallery
    wall (`layer0` at Z=-430, roomDepthAmplitude=0) and the rigid 3D concrete pipe shell + tunnel
    (`pipe_shell` at Z=+8..-395 with alphaCutout=0.5 and EDT-extrapolated exterior depth).
    """
    ptr_cfg = _default_plan_b_pipe_tunnel_relief()
    return SceneBuildSpec(
        scene_id="plan_b_pipe",
        title_he="Plan B — מבעד לצינור המלט והנייר (גלריה בנימין)",
        short_label_he="צינור Plan B",
        icon="🕳️",
        source_image_relpath="media/binyamin_gallery/binyamin_Plan_B_פרט.jpg_14b611ac.jpg",
        outpaint_raw_relpath="prototype/assets/outpaint_sources/scene4_plan_b_outpaint_raw.jpg",
        reflection_source_relpath=None,
        lama_inner_output_filename="scene4_plan_b_lama_inner.jpg",
        bg_output_filename="scene4_plan_b_bg_inpainted.jpg",
        mid_output_filename="scene4_plan_b_mid.png",
        fg_output_filename="scene4_plan_b_fg.png",
        depth_output_filename="scene4_plan_b_depth.png",
        reflection_output_filename=None,
        segmentation_backend="sam_boxes",
        depth_backend="pipe_tunnel_relief",
        min_registration_corr=0.92,
        room_depth_amplitude=0.0,
        particle_color_hex=0x9FB7CC,
        particle_count=65,
        fg_x_range_norm=(0.40, 0.56),
        fg_min_depth_quantile=0.55,
        sam_fg_boxes=(
            BoundingBoxPrompt(x0=658, y0=588, x1=788, y1=796),
        ),
        sam_mid_boxes=(
            BoundingBoxPrompt(x0=866, y0=500, x1=904, y1=672),
            BoundingBoxPrompt(x0=892, y0=560, x1=998, y1=668),
        ),
        feather_collar_px=40,
        initial_camera_y=0.0,
        initial_camera_z=400.0,
        min_camera_z=25.0,
        max_camera_z=685.0,
        scene_bg_z=-520.0,
        min_inner_scale=2.0783,
        max_yaw_deg=34.0,
        max_pitch_deg=24.0,
        max_strafe_x=250.0,
        max_strafe_y=120.0,
        dolly_speed_mult=1.20,
        fg_render_order=6,
        mid_render_order=4,
        fg_z=-35.0,
        mid_z=-180.0,
        mid_depth_output_filename="scene4_plan_b_mid_depth.png",
        mid_depth_amplitude=1.0,
        mid_pin_margin_uv=0.0,
        mid_segments_x=64,
        mid_segments_y=44,
        mid_depth_test=True,
        fg_depth_output_filename="scene4_plan_b_fg_depth.png",
        fg_depth_amplitude=1.0,
        fg_pin_margin_uv=0.0,
        fg_segments_x=64,
        fg_segments_y=44,
        fg_depth_test=True,
        use_mirror_outpaint=True,
        bg_segments_x=256,
        bg_segments_y=172,
        harmonize_seam_field=True,
        seam_harmonic_downsample=8,
        seam_harmonic_iters=240,
        pipe_tunnel_relief=ptr_cfg,
        tunnel_bore_target=TunnelBoreTargetSpec(
            inner_norm_x=879.0 / 1500.0,
            inner_norm_y=462.0 / 1000.0,
            target_z=-330.0,
        ),
        extra_layers=(
            ExtraDepthLayerSpec(
                layer_id="pipe_shell",
                label_he="גוף צינור המלט והמנהרה התלת-ממדית",
                output_filename="scene4_plan_b_pipe_shell.png",
                depth_output_filename="scene4_plan_b_pipe_shell_depth.png",
                z_pos=-430.0,
                render_order=1,
                is_expanded=True,
                depth_test=True,
                depth_write=True,
                pipe_shell_mode=True,
                depth_amplitude=ptr_cfg.pipe_depth_amplitude,
                alpha_cutout=0.5,
                pin_margin_uv=0.0,
                segments_x=256,
                segments_y=172,
            ),
            ExtraDepthLayerSpec(
                layer_id="hanging_ring",
                label_he="טבעת המלט התלויה מתקרת הצינור",
                output_filename="scene4_plan_b_hanging_ring.png",
                z_pos=-95.0,
                render_order=5,
                depth_test=True,
                sam_boxes=(
                    BoundingBoxPrompt(x0=620, y0=284, x1=766, y1=552),
                ),
            ),
            ExtraDepthLayerSpec(
                layer_id="rear_disc",
                label_he="דיסקית התלייה בפתח האחורי של הצינור",
                output_filename="scene4_plan_b_rear_disc.png",
                z_pos=-265.0,
                render_order=3,
                depth_test=True,
                sam_boxes=(
                    BoundingBoxPrompt(x0=868, y0=330, x1=925, y1=438),
                ),
            ),
        ),
    )


def _plan_b_pipe_v3_scene_spec() -> SceneBuildSpec:
    """
    Version 3 of Plan B: fuses IMG_DROP_0098 (Tami & toddler Ya'ara reading at the twilight window)
    into the far diffuse bluish-white opening of Version 2's true two-surface 3D concrete pipe split,
    with clean gallery-wall disocclusion behind the pipe rim and bore-guided fly-through navigation.
    """
    ptr_cfg = _default_plan_b_pipe_tunnel_relief()
    return SceneBuildSpec(
        scene_id="plan_b_pipe_v3",
        title_he="Plan B (גרסה 3) — תמי ויערה באור הדיפוזי בקצה הצינור",
        short_label_he="Plan B: תמי ויערה",
        icon="🕳️",
        source_image_relpath="media/binyamin_gallery/binyamin_Plan_B_פרט.jpg_14b611ac.jpg",
        outpaint_raw_relpath="prototype/assets/outpaint_sources/scene4_plan_b_outpaint_raw.jpg",
        reflection_source_relpath=None,
        lama_inner_output_filename="scene4_plan_b_v3_lama_inner.jpg",
        bg_output_filename="scene4_plan_b_v3_bg_inpainted.jpg",
        mid_output_filename="scene4_plan_b_v3_mid.png",
        fg_output_filename="scene4_plan_b_v3_fg.png",
        depth_output_filename="scene4_plan_b_v3_depth.png",
        reflection_output_filename=None,
        segmentation_backend="sam_boxes",
        depth_backend="pipe_tunnel_relief",
        min_registration_corr=0.92,
        room_depth_amplitude=0.0,
        particle_color_hex=0xA8C0DC,
        particle_count=65,
        fg_x_range_norm=(0.40, 0.56),
        fg_min_depth_quantile=0.55,
        sam_fg_boxes=(
            BoundingBoxPrompt(x0=658, y0=588, x1=788, y1=796),
        ),
        sam_mid_boxes=(
            BoundingBoxPrompt(x0=866, y0=500, x1=904, y1=672),
            BoundingBoxPrompt(x0=892, y0=560, x1=998, y1=668),
        ),
        feather_collar_px=40,
        initial_camera_y=0.0,
        initial_camera_z=400.0,
        min_camera_z=-235.0,
        max_camera_z=685.0,
        scene_bg_z=-520.0,
        min_inner_scale=2.0783,
        max_yaw_deg=36.0,
        max_pitch_deg=26.0,
        max_strafe_x=280.0,
        max_strafe_y=140.0,
        dolly_speed_mult=1.50,
        fg_render_order=8,
        mid_render_order=6,
        fg_z=-35.0,
        mid_z=-180.0,
        mid_depth_output_filename="scene4_plan_b_v3_mid_depth.png",
        mid_depth_amplitude=1.0,
        mid_pin_margin_uv=0.0,
        mid_segments_x=64,
        mid_segments_y=44,
        mid_depth_test=True,
        fg_depth_output_filename="scene4_plan_b_v3_fg_depth.png",
        fg_depth_amplitude=1.0,
        fg_pin_margin_uv=0.0,
        fg_segments_x=64,
        fg_segments_y=44,
        fg_depth_test=True,
        use_mirror_outpaint=True,
        bg_segments_x=256,
        bg_segments_y=172,
        harmonize_seam_field=True,
        seam_harmonic_downsample=8,
        seam_harmonic_iters=240,
        rear_chamber_soften_box=BoundingBoxPrompt(x0=748, y0=346, x1=988, y1=642),
        pipe_tunnel_relief=ptr_cfg,
        tunnel_bore_target=TunnelBoreTargetSpec(
            inner_norm_x=878.0 / 1500.0,
            inner_norm_y=488.0 / 1000.0,
            target_z=-332.0,
        ),
        diffuse_aperture_fusion=DiffuseApertureFusionSpec(
            source_image_relpath="media/drop_folder_imports/denya_2020_05_08_twilight_windows_tami_yaara_e83c3fa9.jpg",
            src_crop_xyxy=(760, 1560, 2220, 2548),
            dest_center_inner_xy=(878.0, 492.0),
            dest_inner_wh=(248.0, 178.0),
            canvas_scale_mult=2.4,
            occluder_remove_boxes=(
                BoundingBoxPrompt(x0=0, y0=640, x1=270, y1=988),
            ),
            front_subject_boxes=(
                BoundingBoxPrompt(x0=12, y0=540, x1=860, y1=885),
                BoundingBoxPrompt(x0=215, y0=385, x1=795, y1=855),
            ),
            front_exclude_boxes=(
                BoundingBoxPrompt(x0=846, y0=685, x1=1245, y1=885),
            ),
            back_subject_boxes=(
                BoundingBoxPrompt(x0=635, y0=145, x1=1245, y1=885),
            ),
            back_hair_matte_box=BoundingBoxPrompt(x0=680, y0=145, x1=1130, y1=640),
            shadow_rgb=(84.0, 95.0, 116.0),
            midtone_rgb=(156.0, 168.0, 192.0),
            highlight_rgb=(234.0, 240.0, 250.0),
            mist_glow_rgb=(236.0, 241.0, 249.0),
            chroma_retention=0.46,
            shadow_lift=0.20,
            luma_gamma=0.80,
            mist_veil_core=0.09,
            mist_veil_edge=0.38,
            halo_rx_ratio=0.49,
            halo_ry_ratio=0.49,
            halo_feather_power=1.65,
            halo_max_alpha=0.92,
            halo_patch_scale_wh=(1.28, 1.48),
            halo_offset_inner_xy=(0.0, 16.0),
            halo_luminous_lift=0.60,
            halo_bg_blur_sigma_px=28.0,
            tabletop_source_band_ratios=(0.825, 0.870),
            tabletop_blend_band_px=36,
            tabletop_alpha_bridge_sigma_px=20.0,
            tabletop_rgb_diffuse_sigma_yx=(12.0, 20.0),
            arch_center_x_ratio=0.455,
            arch_half_span_x_ratio=0.445,
            bottom_arch_curvature=0.075,
            organic_wave_amp_ratio=0.020,
            depth_contour_amp_ratio=0.045,
            depth_contour_smooth_sigma_px=28.0,
            bottom_fade_start_ratio=0.79,
            bottom_fade_end_ratio=0.985,
            bottom_fade_power=1.15,
            top_fade_end_ratio=0.11,
            left_fade_end_ratio=0.21,
            right_fade_start_ratio=0.72,
            right_fade_end_ratio=0.860,
            lower_bloom_strength=0.48,
            lower_bloom_gamma=0.95,
            edge_light_wrap_sigma_px=24.0,
            edge_light_wrap_strength=0.38,
            aperture_glow_center_inner_xy=(878.0, 506.0),
            aperture_glow_radii_inner_xy=(148.0, 176.0),
            aperture_glow_core_rgb=(236.0, 238.0, 244.0),
            aperture_glow_max_strength=0.88,
            aperture_glow_feather_power=1.65,
            aperture_grain_preserve_sigma_px=7.0,
            aperture_grain_preserve_weight=0.48,
            lama_dilation_px=16,
            warm_spill_rgb=(248.0, 236.0, 214.0),
            warm_rim_wrap_strength=0.55,
            lower_warm_pool_strength=0.62,
            tabletop_preserve_real_rgb_weight=0.78,
        ),
        extra_layers=(
            ExtraDepthLayerSpec(
                layer_id="pipe_shell",
                label_he="גוף צינור המלט והמנהרה התלת-ממדית",
                output_filename="scene4_plan_b_v3_pipe_shell.png",
                depth_output_filename="scene4_plan_b_v3_pipe_shell_depth.png",
                z_pos=-430.0,
                render_order=1,
                is_expanded=True,
                depth_test=True,
                depth_write=True,
                pipe_shell_mode=True,
                depth_amplitude=ptr_cfg.pipe_depth_amplitude,
                alpha_cutout=0.5,
                pin_margin_uv=0.0,
                segments_x=256,
                segments_y=172,
            ),
            ExtraDepthLayerSpec(
                layer_id="hanging_ring",
                label_he="טבעת המלט התלויה מתקרת הצינור",
                output_filename="scene4_plan_b_v3_hanging_ring.png",
                z_pos=-95.0,
                render_order=7,
                depth_test=True,
                sam_boxes=(
                    BoundingBoxPrompt(x0=620, y0=284, x1=766, y1=552),
                ),
            ),
            ExtraDepthLayerSpec(
                layer_id="rear_disc",
                label_he="דיסקית התלייה בפתח האחורי של הצינור",
                output_filename="scene4_plan_b_v3_rear_disc.png",
                z_pos=-265.0,
                render_order=5,
                depth_test=True,
                sam_boxes=(
                    BoundingBoxPrompt(x0=868, y0=330, x1=925, y1=438),
                ),
            ),
            ExtraDepthLayerSpec(
                layer_id="yaara_front",
                label_he="יערה והספר הפתוח באור הדיפוזי בקצה הצינור",
                output_filename="scene4_plan_b_v3_yaara_front.png",
                z_pos=-322.0,
                render_order=4,
                depth_test=True,
                diffuse_aperture_role="subject_front",
            ),
            ExtraDepthLayerSpec(
                layer_id="tami_back",
                label_he="תמי קוראת עם יערה באור הדיפוזי בקצה הצינור",
                output_filename="scene4_plan_b_v3_tami_back.png",
                z_pos=-342.0,
                render_order=3,
                depth_test=True,
                diffuse_aperture_role="subject_back",
            ),
            ExtraDepthLayerSpec(
                layer_id="chamber_halo",
                label_he="הילת החלון הדיפוזית המתמזגת לעומק הצינור",
                output_filename="scene4_plan_b_v3_chamber_halo.png",
                z_pos=-376.0,
                render_order=2,
                depth_test=True,
                diffuse_aperture_role="chamber_halo",
            ),
        ),
    )


def default_spatial_pipeline_config() -> SpatialPipelineConfig:
    """Returns the canonical SSOT spatial asset synthesis configuration."""
    return SpatialPipelineConfig(
        output_dir_relpath="prototype/assets/scenes",
        manifest_filename="scene_manifest.json",
        dimensions=CanvasDimensionsConfig(width=1500, height=1000),
        outpainting=OutpaintingConfig(
            expansion_factor=1.80,
            feather_collar_px=24,
            color_match_band_px=64,
            min_registration_corr=0.92,
        ),
        depth_model=DepthModelConfig(
            model_id="depth-anything/Depth-Anything-V2-Small-hf",
            background_gaussian_sigma=18.0,
            depth_discontinuity_percentile_fg=62.0,
            depth_discontinuity_percentile_mid=30.0,
        ),
        segmentation=SegmentationConfig(
            rembg_model_name="isnet-general-use",
            sam_model_id="facebook/sam-vit-base",
            alpha_foreground_threshold=92,
            alpha_full_opaque_threshold=195,
            min_component_area_px=350,
            sam_morph_close_px=7,
            edge_feather_sigma=0.85,
        ),
        inpainting=InpaintingConfig(
            dilation_radius_px=26,
            mask_blur_sigma=3.0,
            jpeg_quality=91,
        ),
        graphite_drawing=default_graphite_drawing_config(),
        floating_extraction=default_floating_element_extraction_config(),
        camera_rig=default_camera_rig_config(),
        scenes=(
            SceneBuildSpec(
                scene_id="paper_capsules",
                title_he="קפסולות נייר תלויות — ׳אולי שם יהיה יותר טוב׳ (גלריה בנימין)",
                short_label_he="קפסולות נייר",
                icon="🫧",
                source_image_relpath="media/binyamin_gallery/binyamin_אולי_שם_יהיה_יותר_טוב.jpg_a23a554b.jpg",
                outpaint_raw_relpath="prototype/assets/outpaint_sources/scene1_herzliya_outpaint_raw.jpg",
                reflection_source_relpath=None,
                lama_inner_output_filename="scene1_herzliya_lama_inner.jpg",
                bg_output_filename="scene1_herzliya_bg_inpainted.jpg",
                mid_output_filename="scene1_herzliya_mid.png",
                fg_output_filename="scene1_herzliya_fg.png",
                depth_output_filename="scene1_herzliya_depth.png",
                reflection_output_filename=None,
                segmentation_backend="isnet",
                depth_backend="depth_anything",
                min_registration_corr=0.92,
                room_depth_amplitude=92.0,
                particle_color_hex=0xEAD8BE,
                particle_count=85,
                fg_x_range_norm=(0.22, 0.62),
                fg_min_depth_quantile=0.48,
                sam_fg_boxes=(),
                sam_mid_boxes=(
                    BoundingBoxPrompt(x0=194, y0=488, x1=274, y1=604),
                ),
            ),
            SceneBuildSpec(
                scene_id="bene_beitcha",
                title_he="תערוכת ׳בנה ביתך׳ — ארכיטקטורות של שבריריות (מוזיאון רמת גן)",
                short_label_he="׳בנה ביתך׳",
                icon="🏛️",
                source_image_relpath="media/museum_press/rgma_bene_beitcha_TAMAR-SHEFER-YUVAL-HAI-1.jpg_a3cb100a.jpg",
                outpaint_raw_relpath="prototype/assets/outpaint_sources/scene2_denya_outpaint_raw.jpg",
                reflection_source_relpath="media/darchei_etzim_pdfs/darchei_etzim_2011_plate_003.jpg_32d2f7b3.jpg",
                lama_inner_output_filename="scene2_denya_lama_inner.jpg",
                bg_output_filename="scene2_denya_bg_inpainted.jpg",
                mid_output_filename="scene2_denya_mid.png",
                fg_output_filename="scene2_denya_fg.png",
                depth_output_filename="scene2_denya_depth.png",
                reflection_output_filename="scene2_reflections.png",
                segmentation_backend="sam_boxes",
                depth_backend="depth_anything",
                min_registration_corr=0.92,
                room_depth_amplitude=92.0,
                particle_color_hex=0xB8C7CF,
                particle_count=90,
                fg_x_range_norm=(0.01, 0.45),
                fg_min_depth_quantile=0.52,
                sam_fg_boxes=(
                    BoundingBoxPrompt(x0=336, y0=74, x1=456, y1=376),
                    BoundingBoxPrompt(x0=188, y0=330, x1=246, y1=446),
                    BoundingBoxPrompt(x0=112, y0=300, x1=480, y1=842),
                    BoundingBoxPrompt(x0=364, y0=412, x1=642, y1=846),
                    BoundingBoxPrompt(x0=44, y0=660, x1=306, y1=922),
                ),
                sam_mid_boxes=(
                    BoundingBoxPrompt(x0=656, y0=490, x1=724, y1=654),
                    BoundingBoxPrompt(x0=766, y0=448, x1=1006, y1=660),
                    BoundingBoxPrompt(x0=982, y0=676, x1=1106, y1=718),
                ),
                sam_mid_point_groups=(
                    # Cream fireplace upper & middle body + recessed hearth (full right column)
                    SamPointGroupPrompt(
                        box=BoundingBoxPrompt(x0=768, y0=527, x1=931, y1=651),
                        points_xy=(
                            (776.0, 545.0),
                            (776.0, 600.0),
                            (776.0, 642.0),
                            (850.0, 555.0),
                            (805.0, 610.0),
                            (842.0, 610.0),
                            (885.0, 610.0),
                            (924.0, 536.0),
                            (924.0, 558.0),
                            (924.0, 579.0),
                            (910.0, 588.0),
                            (910.0, 606.0),
                            (910.0, 623.0),
                            (765.0, 560.0),
                            (765.0, 625.0),
                            (933.0, 545.0),
                            (933.0, 575.0),
                            (920.0, 590.0),
                            (920.0, 604.0),
                            (945.0, 605.0),
                        ),
                        point_labels=(
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 0, 0,
                        ),
                    ),
                    # Lower right cream plinth (bright white top face + warm cream front face to crisp right corner)
                    SamPointGroupPrompt(
                        box=BoundingBoxPrompt(x0=855, y0=624, x1=971, y1=650),
                        points_xy=(
                            (875.0, 631.0),
                            (900.0, 631.0),
                            (942.0, 631.0),
                            (958.0, 632.0),
                            (880.0, 641.0),
                            (915.0, 640.0),
                            (948.0, 639.0),
                            (964.0, 638.0),
                            (885.0, 620.0),
                            (948.0, 622.0),
                            (964.0, 625.0),
                            (900.0, 651.0),
                            (950.0, 648.0),
                            (972.0, 639.0),
                        ),
                        point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
                    ),
                    # Cluster of stones sitting on right lower cream plinth
                    SamPointGroupPrompt(
                        box=BoundingBoxPrompt(x0=905, y0=609, x1=938, y1=637),
                        points_xy=(
                            (918.0, 622.0),
                            (912.0, 627.0),
                            (926.0, 620.0),
                            (930.0, 628.0),
                            (920.0, 632.0),
                            (923.0, 615.0),
                            (903.0, 616.0),
                            (918.0, 607.0),
                            (939.0, 618.0),
                            (939.0, 626.0),
                        ),
                        point_labels=(1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
                    ),
                ),
                sam_mid_exclude_boxes=(
                    BoundingBoxPrompt(x0=692, y0=465, x1=748, y1=515),
                    BoundingBoxPrompt(x0=764, y0=488, x1=777, y1=528),
                    BoundingBoxPrompt(x0=940, y0=537, x1=994, y1=545),
                ),
                inpaint_region_dilation_overrides=(
                    InpaintRegionDilationOverride(
                        box=BoundingBoxPrompt(x0=688, y0=390, x1=885, y1=496),
                        dilation_radius_px=5,
                    ),
                ),
                reflection_apertures=(
                    ApertureBox(x0=1218, y0=500, x1=1282, y1=570, max_alpha=125, feather_sigma=4.5),
                ),
                room_depth_clean=RoomDepthCleanConfig(
                    alpha_threshold_u8=16,
                    mask_dilation_px=22,
                    post_smooth_sigma_px=3.0,
                    interior_blend_band_px=12.0,
                ),
                room_depth_clean_layers=("fg", "mid"),
                topology_verification=LayerTopologyVerificationConfig(),
                mid_z=-350.0,
                mid_depth_output_filename="scene2_denya_mid_depth.png",
                mid_depth_amplitude=48.0,
                mid_pin_margin_uv=0.0,
                mid_segments_x=224,
                mid_segments_y=150,
                mid_contact_depth_solve=ContactAnchoredLayerDepthConfig(
                    bg_z=-350.0,
                    room_depth_amplitude=92.0,
                    room_pin_margin_uv=0.045,
                    layer_z=-350.0,
                    layer_depth_amplitude=48.0,
                    layer_pin_margin_uv=0.0,
                    alpha_threshold_u8=20,
                    component_bridge_px=20,
                    min_component_area_px=1,
                    contact_boxes=(
                        ContactAnchorBox(
                            x0=656, y0=638, x1=726, y1=656, mode="bottom_band", band_px=6
                        ),
                        ContactAnchorBox(
                            x0=766, y0=634, x1=972, y1=661, mode="bottom_band", band_px=6
                        ),
                        ContactAnchorBox(
                            x0=982, y0=694, x1=1106, y1=718, mode="bottom_band", band_px=6
                        ),
                    ),
                    neural_relief_amplitude=20.0,
                    contact_relief_taper_px=8.0,
                    enforce_front_of_host=True,
                    host_clearance_tolerance_world=0.0,
                    interior_smooth_sigma_px=1.5,
                ),
                fg_z=-336.0,
                fg_depth_output_filename="scene2_denya_fg_depth.png",
                fg_depth_amplitude=64.0,
                fg_pin_margin_uv=0.0,
                fg_segments_x=224,
                fg_segments_y=150,
                fg_contact_depth_solve=ContactAnchoredLayerDepthConfig(
                    bg_z=-350.0,
                    room_depth_amplitude=92.0,
                    room_pin_margin_uv=0.045,
                    layer_z=-336.0,
                    layer_depth_amplitude=64.0,
                    layer_pin_margin_uv=0.0,
                    alpha_threshold_u8=20,
                    component_bridge_px=12,
                    min_component_area_px=1,
                    contact_boxes=(
                        ContactAnchorBox(
                            x0=46, y0=902, x1=148, y1=922, mode="bottom_band", band_px=6
                        ),
                        ContactAnchorBox(
                            x0=148, y0=794, x1=640, y1=874, mode="bottom_band", band_px=6
                        ),
                    ),
                    neural_relief_amplitude=32.0,
                    contact_relief_taper_px=10.0,
                    enforce_front_of_host=True,
                    host_clearance_tolerance_world=0.0,
                    interior_smooth_sigma_px=1.5,
                ),
            ),
            _plan_b_pipe_v2_scene_spec(),
            _plan_b_pipe_v3_scene_spec(),
            SceneBuildSpec(
                scene_id="wandering_drawing",
                title_he="׳שיטוט 2׳ — רישום איים מרחפים ותבליט דמיוני על נייר",
                short_label_he="שיטוט ברישום",
                icon="✏️",
                source_image_relpath="media/binyamin_gallery/binyamin_שיטוט_2.jpg_9af0fd97.jpg",
                outpaint_raw_relpath="prototype/assets/outpaint_sources/scene5_wandering_outpaint_raw.jpg",
                reflection_source_relpath=None,
                lama_inner_output_filename="scene5_wandering_lama_inner.jpg",
                bg_output_filename="scene5_wandering_bg_inpainted.jpg",
                mid_output_filename="scene5_wandering_mid.png",
                fg_output_filename="scene5_wandering_fg.png",
                depth_output_filename="scene5_wandering_depth.png",
                reflection_output_filename=None,
                segmentation_backend="graphite_paper",
                depth_backend="drawing_dome_relief",
                min_registration_corr=0.80,
                room_depth_amplitude=118.0,
                particle_color_hex=0xD8CFC2,
                particle_count=65,
                fg_x_range_norm=(0.04, 0.92),
                fg_min_depth_quantile=0.50,
                sam_fg_boxes=(),
                sam_mid_boxes=(),
                drawing_fg_boxes=(
                    DrawingStrokeBox(55, 688, 384, 980, 22.0, 5, 420),
                    DrawingStrokeBox(546, 618, 608, 720, 38.0, 5, 220),
                    DrawingStrokeBox(680, 344, 822, 720, 42.0, 5, 320),
                    DrawingStrokeBox(396, 712, 852, 980, 22.0, 5, 500),
                    DrawingStrokeBox(1176, 720, 1368, 980, 32.0, 5, 320),
                ),
                drawing_mid_boxes=(
                    DrawingStrokeBox(132, 162, 444, 688, 20.0, 5, 420),
                    DrawingStrokeBox(292, 680, 355, 980, 22.0, 3, 0),
                    DrawingStrokeBox(690, 28, 996, 232, 20.0, 5, 420),
                    DrawingStrokeBox(850, 228, 874, 980, 32.0, 3, 0),
                ),
                mid_overlays=(
                    LayerOverlaySpec(
                        source_image_relpath="media/darchei_etzim_pdfs/darchei_etzim_2011_plate_035.jpg_d20d1be9.jpg",
                        extraction_mode="darc_oval_island",
                        src_crop_xyxy=(775, 42, 1068, 98),
                        dest_xy=(1055, 78),
                        dest_wh=(320, 72),
                        min_delta=22.0,
                        opacity=0.96,
                    ),
                    LayerOverlaySpec(
                        source_image_relpath="media/darchei_etzim_pdfs/darchei_etzim_2011_plate_035.jpg_d20d1be9.jpg",
                        extraction_mode="darc_oval_island",
                        src_crop_xyxy=(838, 212, 1026, 294),
                        dest_xy=(488, 225),
                        dest_wh=(210, 92),
                        min_delta=24.0,
                        opacity=0.95,
                    ),
                    LayerOverlaySpec(
                        source_image_relpath="media/darchei_etzim_pdfs/darchei_etzim_2011_plate_035.jpg_d20d1be9.jpg",
                        extraction_mode="darc_oval_island",
                        src_crop_xyxy=(1002, 320, 1146, 406),
                        dest_xy=(52, 515),
                        dest_wh=(155, 95),
                        min_delta=24.0,
                        opacity=0.96,
                    ),
                ),
                initial_camera_y=0.0,
                initial_camera_z=525.0,
                max_camera_z=780.0,
                use_mirror_outpaint=True,
            ),
            _marseille_mirror_scene_spec(),
            _plan_b_pipe_v1_scene_spec(),
            _marseille_mirror_v1_scene_spec(),
        ),
    )
