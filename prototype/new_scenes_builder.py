#!/usr/bin/env python3
"""
prototype/new_scenes_builder.py
Offline neural 2.5D spatial asset builder and typed SSOT configuration models for:
  1. Scene 7 (`wandering_drawing_1`): "שיטוט ברישום 1" — Naama's 9 deduplicated drawings
     composed seamlessly across 3 depth layers (`bg`, `mid`, `fg`) + 9 480px web thumbnails.
  2. Scene 8 (`denya_model`): "דגם בית דניה על לבד אפור" — Tami's physical architectural
     maquette of 80 Greenboim St (Denya) over a seamless LaMa-healed moving-blanket gray felt backdrop.
  3. Scene 9 (`shelf_portal`): "קיר המדפים — שער בין עולמות" — Modular concrete & wood cubby
     cabinet (`IMG_BINY_0007`) with 8 SAM-segmented alpha-cut portal apertures and WebGL coordinates.

Strict No-Fallback Guarantee: executes real neural models (`Depth Anything V2`, `Meta SAM`,
`IS-Net` via `rembg`, and `LaMa` via `simple-lama-inpainting`) inside `.venv`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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
    from rembg import remove
except ImportError as exc:
    raise RuntimeError(
        "Missing neural vision dependencies inside virtual environment. Run with: "
        "`/Users/talsheaffer/code/tami/.venv/bin/python -u prototype/new_scenes_builder.py`."
    ) from exc

from prototype.depth_topology import extrapolate_depth_outward_edt
from prototype.scene_config import (
    BoundingBoxPrompt,
    CanvasDimensionsConfig,
    DepthModelConfig,
    GraphiteDrawingConfig,
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


# ============================================================================
# Hierarchical, Serializable Configuration Data-Models (SSOT)
# ============================================================================


@dataclass(frozen=True)
class WanderingArtworkEntrySpec:
    catalog_id: str
    dropped_duplicate_ids: tuple[str, ...]
    source_relpath: str
    thumbnail_filename: str
    title_he: str
    year_label: str
    assigned_layer: str
    placement_role: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dropped_duplicate_ids"] = list(self.dropped_duplicate_ids)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WanderingArtworkEntrySpec:
        return cls(
            catalog_id=str(data["catalog_id"]),
            dropped_duplicate_ids=tuple(str(x) for x in data.get("dropped_duplicate_ids", ())),
            source_relpath=str(data["source_relpath"]),
            thumbnail_filename=str(data["thumbnail_filename"]),
            title_he=str(data["title_he"]),
            year_label=str(data["year_label"]),
            assigned_layer=str(data["assigned_layer"]),
            placement_role=str(data["placement_role"]),
        )


@dataclass(frozen=True)
class PaperTransmittanceZoneSpec:
    catalog_id: str
    source_relpath: str
    src_crop_norm: tuple[float, float, float, float]
    dest_box_xyxy: tuple[int, int, int, int]
    feather_left_px: int
    feather_top_px: int
    feather_right_px: int
    feather_bottom_px: int
    elliptical_falloff: bool
    transmittance_strength: float
    paper_percentile: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["src_crop_norm"] = list(self.src_crop_norm)
        d["dest_box_xyxy"] = list(self.dest_box_xyxy)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperTransmittanceZoneSpec:
        sc = data["src_crop_norm"]
        db = data["dest_box_xyxy"]
        return cls(
            catalog_id=str(data["catalog_id"]),
            source_relpath=str(data["source_relpath"]),
            src_crop_norm=(float(sc[0]), float(sc[1]), float(sc[2]), float(sc[3])),
            dest_box_xyxy=(int(db[0]), int(db[1]), int(db[2]), int(db[3])),
            feather_left_px=int(data["feather_left_px"]),
            feather_top_px=int(data["feather_top_px"]),
            feather_right_px=int(data["feather_right_px"]),
            feather_bottom_px=int(data["feather_bottom_px"]),
            elliptical_falloff=bool(data["elliptical_falloff"]),
            transmittance_strength=float(data["transmittance_strength"]),
            paper_percentile=float(data["paper_percentile"]),
        )


@dataclass(frozen=True)
class WanderingLayerIslandSpec:
    catalog_id: str
    source_relpath: str
    target_layer: str
    extraction_method: str
    src_crop_norm: tuple[float, float, float, float]
    dest_xy: tuple[int, int]
    dest_wh: tuple[int, int]
    paper_percentile: float
    lum_delta_floor: float
    lum_delta_full: float
    chroma_weight: float
    alpha_gamma: float
    perimeter_taper_px: int
    bottom_taper_px: int
    min_component_area_px: int
    max_interior_hole_fill_px: int = 0
    lama_heal_boxes_px: tuple[BoundingBoxPrompt, ...] = ()
    sam_prompts: tuple[SamPointGroupPrompt, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "source_relpath": self.source_relpath,
            "target_layer": self.target_layer,
            "extraction_method": self.extraction_method,
            "src_crop_norm": list(self.src_crop_norm),
            "dest_xy": list(self.dest_xy),
            "dest_wh": list(self.dest_wh),
            "paper_percentile": self.paper_percentile,
            "lum_delta_floor": self.lum_delta_floor,
            "lum_delta_full": self.lum_delta_full,
            "chroma_weight": self.chroma_weight,
            "alpha_gamma": self.alpha_gamma,
            "perimeter_taper_px": self.perimeter_taper_px,
            "bottom_taper_px": self.bottom_taper_px,
            "min_component_area_px": self.min_component_area_px,
            "max_interior_hole_fill_px": self.max_interior_hole_fill_px,
            "lama_heal_boxes_px": [b.to_dict() for b in self.lama_heal_boxes_px],
            "sam_prompts": [p.to_dict() for p in self.sam_prompts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WanderingLayerIslandSpec:
        sc = data["src_crop_norm"]
        dxy = data["dest_xy"]
        dwh = data["dest_wh"]
        return cls(
            catalog_id=str(data["catalog_id"]),
            source_relpath=str(data["source_relpath"]),
            target_layer=str(data["target_layer"]),
            extraction_method=str(data["extraction_method"]),
            src_crop_norm=(float(sc[0]), float(sc[1]), float(sc[2]), float(sc[3])),
            dest_xy=(int(dxy[0]), int(dxy[1])),
            dest_wh=(int(dwh[0]), int(dwh[1])),
            paper_percentile=float(data["paper_percentile"]),
            lum_delta_floor=float(data["lum_delta_floor"]),
            lum_delta_full=float(data["lum_delta_full"]),
            chroma_weight=float(data["chroma_weight"]),
            alpha_gamma=float(data["alpha_gamma"]),
            perimeter_taper_px=int(data["perimeter_taper_px"]),
            bottom_taper_px=int(data["bottom_taper_px"]),
            min_component_area_px=int(data["min_component_area_px"]),
            max_interior_hole_fill_px=int(data.get("max_interior_hole_fill_px", 0)),
            lama_heal_boxes_px=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data.get("lama_heal_boxes_px", ())
            ),
            sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data.get("sam_prompts", ())
            ),
        )


@dataclass(frozen=True)
class WanderingScrollPortalSpec:
    portal_id: str
    target_scene_id: str
    label_he: str
    short_badge_he: str
    foot_layer_id: str
    foot_label_he: str
    scroll_catalog_id: str
    inner_box_px: tuple[int, int, int, int]
    veil_feather_px: int
    mid_z: float
    foot_z_pos: float
    foot_depth_amplitude: float
    foot_depth_min_norm: float
    foot_depth_max_norm: float
    foot_heal_dilation_px: int
    foot_heal_blur_sigma: float
    foot_min_component_area_px: int
    foot_sam_prompts: tuple[SamPointGroupPrompt, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "portal_id": self.portal_id,
            "target_scene_id": self.target_scene_id,
            "label_he": self.label_he,
            "short_badge_he": self.short_badge_he,
            "foot_layer_id": self.foot_layer_id,
            "foot_label_he": self.foot_label_he,
            "scroll_catalog_id": self.scroll_catalog_id,
            "inner_box_px": list(self.inner_box_px),
            "veil_feather_px": self.veil_feather_px,
            "mid_z": self.mid_z,
            "foot_z_pos": self.foot_z_pos,
            "foot_depth_amplitude": self.foot_depth_amplitude,
            "foot_depth_min_norm": self.foot_depth_min_norm,
            "foot_depth_max_norm": self.foot_depth_max_norm,
            "foot_heal_dilation_px": self.foot_heal_dilation_px,
            "foot_heal_blur_sigma": self.foot_heal_blur_sigma,
            "foot_min_component_area_px": self.foot_min_component_area_px,
            "foot_sam_prompts": [p.to_dict() for p in self.foot_sam_prompts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WanderingScrollPortalSpec:
        ib = data["inner_box_px"]
        return cls(
            portal_id=str(data["portal_id"]),
            target_scene_id=str(data["target_scene_id"]),
            label_he=str(data["label_he"]),
            short_badge_he=str(data["short_badge_he"]),
            foot_layer_id=str(data["foot_layer_id"]),
            foot_label_he=str(data["foot_label_he"]),
            scroll_catalog_id=str(data["scroll_catalog_id"]),
            inner_box_px=(int(ib[0]), int(ib[1]), int(ib[2]), int(ib[3])),
            veil_feather_px=int(data["veil_feather_px"]),
            mid_z=float(data["mid_z"]),
            foot_z_pos=float(data["foot_z_pos"]),
            foot_depth_amplitude=float(data["foot_depth_amplitude"]),
            foot_depth_min_norm=float(data["foot_depth_min_norm"]),
            foot_depth_max_norm=float(data["foot_depth_max_norm"]),
            foot_heal_dilation_px=int(data["foot_heal_dilation_px"]),
            foot_heal_blur_sigma=float(data["foot_heal_blur_sigma"]),
            foot_min_component_area_px=int(data["foot_min_component_area_px"]),
            foot_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["foot_sam_prompts"]
            ),
        )


@dataclass(frozen=True)
class WanderingDrawing1Spec:
    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    asset_prefix: str
    thumbnail_height_px: int
    thumbnail_jpeg_quality: int
    paper_base_rgb: tuple[float, float, float]
    paper_grain_source_relpath: str
    paper_grain_crop_norm: tuple[float, float, float, float]
    paper_grain_amplitude: float
    paper_vignette_strength: float
    paper_downsample_divisor: int
    paper_filter_size: int
    unmix_alpha_floor: float
    layer_depth_min_u8: int
    layer_depth_collar_sigma: float
    lama_heal_dilation_px: int
    lama_heal_blur_sigma: float
    artworks: tuple[WanderingArtworkEntrySpec, ...]
    bg_zones: tuple[PaperTransmittanceZoneSpec, ...]
    islands: tuple[WanderingLayerIslandSpec, ...]
    scroll_portal: WanderingScrollPortalSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "asset_prefix": self.asset_prefix,
            "thumbnail_height_px": self.thumbnail_height_px,
            "thumbnail_jpeg_quality": self.thumbnail_jpeg_quality,
            "paper_base_rgb": list(self.paper_base_rgb),
            "paper_grain_source_relpath": self.paper_grain_source_relpath,
            "paper_grain_crop_norm": list(self.paper_grain_crop_norm),
            "paper_grain_amplitude": self.paper_grain_amplitude,
            "paper_vignette_strength": self.paper_vignette_strength,
            "paper_downsample_divisor": self.paper_downsample_divisor,
            "paper_filter_size": self.paper_filter_size,
            "unmix_alpha_floor": self.unmix_alpha_floor,
            "layer_depth_min_u8": self.layer_depth_min_u8,
            "layer_depth_collar_sigma": self.layer_depth_collar_sigma,
            "lama_heal_dilation_px": self.lama_heal_dilation_px,
            "lama_heal_blur_sigma": self.lama_heal_blur_sigma,
            "artworks": [a.to_dict() for a in self.artworks],
            "bg_zones": [z.to_dict() for z in self.bg_zones],
            "islands": [isl.to_dict() for isl in self.islands],
            "scroll_portal": self.scroll_portal.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WanderingDrawing1Spec:
        pb = data["paper_base_rgb"]
        pgc = data["paper_grain_crop_norm"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            asset_prefix=str(data["asset_prefix"]),
            thumbnail_height_px=int(data["thumbnail_height_px"]),
            thumbnail_jpeg_quality=int(data["thumbnail_jpeg_quality"]),
            paper_base_rgb=(float(pb[0]), float(pb[1]), float(pb[2])),
            paper_grain_source_relpath=str(data["paper_grain_source_relpath"]),
            paper_grain_crop_norm=(float(pgc[0]), float(pgc[1]), float(pgc[2]), float(pgc[3])),
            paper_grain_amplitude=float(data["paper_grain_amplitude"]),
            paper_vignette_strength=float(data["paper_vignette_strength"]),
            paper_downsample_divisor=int(data["paper_downsample_divisor"]),
            paper_filter_size=int(data["paper_filter_size"]),
            unmix_alpha_floor=float(data["unmix_alpha_floor"]),
            layer_depth_min_u8=int(data["layer_depth_min_u8"]),
            layer_depth_collar_sigma=float(data["layer_depth_collar_sigma"]),
            lama_heal_dilation_px=int(data["lama_heal_dilation_px"]),
            lama_heal_blur_sigma=float(data["lama_heal_blur_sigma"]),
            artworks=tuple(WanderingArtworkEntrySpec.from_dict(a) for a in data["artworks"]),
            bg_zones=tuple(PaperTransmittanceZoneSpec.from_dict(z) for z in data["bg_zones"]),
            islands=tuple(WanderingLayerIslandSpec.from_dict(i) for i in data["islands"]),
            scroll_portal=WanderingScrollPortalSpec.from_dict(data["scroll_portal"]),
        )


@dataclass(frozen=True)
class FeltPatchBoxSpec:
    patch_id: str
    box_xyxy: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {"patch_id": self.patch_id, "box_xyxy": list(self.box_xyxy)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeltPatchBoxSpec:
        bx = data["box_xyxy"]
        return cls(
            patch_id=str(data["patch_id"]),
            box_xyxy=(int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])),
        )


@dataclass(frozen=True)
class DenyaModelSceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    asset_prefix: str
    primary_model_catalog_id: str
    primary_model_relpath: str
    secondary_reference_catalog_id: str
    secondary_reference_relpath: str
    felt_source_catalog_id: str
    felt_source_relpath: str
    felt_patches: tuple[FeltPatchBoxSpec, ...]
    felt_tile_draw_wh: tuple[int, int]
    felt_seam_half_width_px: int
    felt_target_mean_rgb: tuple[float, float, float]
    felt_target_std_rgb: tuple[float, float, float]
    felt_spotlight_gain: float
    felt_vignette_drop: float
    felt_lama_dilation_px: int
    felt_lama_blur_sigma: float
    felt_depth_sigma: float
    pad_margin_ltrb: tuple[int, int, int, int]
    studio_cloth_fill_rgb: tuple[int, int, int]
    border_completion_boxes_padded: tuple[BoundingBoxPrompt, ...]
    border_lama_dilation_px: int
    border_lama_blur_sigma: float
    padded_sam_prompts: tuple[SamPointGroupPrompt, ...]
    pergola_cloth_hole_box_padded: BoundingBoxPrompt
    blue_cloth_excess_thresh: float
    blue_cloth_max_lum: float
    maquette_dest_center_xy: tuple[int, int]
    maquette_dest_width: int
    unsharp_sigma: float
    unsharp_amount: float
    contact_shadow_offset_xy: tuple[int, int]
    contact_shadow_blur_sigma: float
    contact_shadow_strength: float
    fg_corner_felt_fold_box: tuple[int, int, int, int]
    fg_vignette_max_alpha: float
    fg_vignette_inner_radius_norm: float
    fg_vignette_outer_radius_norm: float
    layer_depth_min_u8: int
    layer_depth_collar_sigma: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "asset_prefix": self.asset_prefix,
            "primary_model_catalog_id": self.primary_model_catalog_id,
            "primary_model_relpath": self.primary_model_relpath,
            "secondary_reference_catalog_id": self.secondary_reference_catalog_id,
            "secondary_reference_relpath": self.secondary_reference_relpath,
            "felt_source_catalog_id": self.felt_source_catalog_id,
            "felt_source_relpath": self.felt_source_relpath,
            "felt_patches": [p.to_dict() for p in self.felt_patches],
            "felt_tile_draw_wh": list(self.felt_tile_draw_wh),
            "felt_seam_half_width_px": self.felt_seam_half_width_px,
            "felt_target_mean_rgb": list(self.felt_target_mean_rgb),
            "felt_target_std_rgb": list(self.felt_target_std_rgb),
            "felt_spotlight_gain": self.felt_spotlight_gain,
            "felt_vignette_drop": self.felt_vignette_drop,
            "felt_lama_dilation_px": self.felt_lama_dilation_px,
            "felt_lama_blur_sigma": self.felt_lama_blur_sigma,
            "felt_depth_sigma": self.felt_depth_sigma,
            "pad_margin_ltrb": list(self.pad_margin_ltrb),
            "studio_cloth_fill_rgb": list(self.studio_cloth_fill_rgb),
            "border_completion_boxes_padded": [
                b.to_dict() for b in self.border_completion_boxes_padded
            ],
            "border_lama_dilation_px": self.border_lama_dilation_px,
            "border_lama_blur_sigma": self.border_lama_blur_sigma,
            "padded_sam_prompts": [p.to_dict() for p in self.padded_sam_prompts],
            "pergola_cloth_hole_box_padded": self.pergola_cloth_hole_box_padded.to_dict(),
            "blue_cloth_excess_thresh": self.blue_cloth_excess_thresh,
            "blue_cloth_max_lum": self.blue_cloth_max_lum,
            "maquette_dest_center_xy": list(self.maquette_dest_center_xy),
            "maquette_dest_width": self.maquette_dest_width,
            "unsharp_sigma": self.unsharp_sigma,
            "unsharp_amount": self.unsharp_amount,
            "contact_shadow_offset_xy": list(self.contact_shadow_offset_xy),
            "contact_shadow_blur_sigma": self.contact_shadow_blur_sigma,
            "contact_shadow_strength": self.contact_shadow_strength,
            "fg_corner_felt_fold_box": list(self.fg_corner_felt_fold_box),
            "fg_vignette_max_alpha": self.fg_vignette_max_alpha,
            "fg_vignette_inner_radius_norm": self.fg_vignette_inner_radius_norm,
            "fg_vignette_outer_radius_norm": self.fg_vignette_outer_radius_norm,
            "layer_depth_min_u8": self.layer_depth_min_u8,
            "layer_depth_collar_sigma": self.layer_depth_collar_sigma,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DenyaModelSceneSpec:
        ftw = data["felt_tile_draw_wh"]
        fmean = data["felt_target_mean_rgb"]
        fstd = data["felt_target_std_rgb"]
        pm = data["pad_margin_ltrb"]
        sc = data["studio_cloth_fill_rgb"]
        mc = data["maquette_dest_center_xy"]
        cso = data["contact_shadow_offset_xy"]
        fcf = data["fg_corner_felt_fold_box"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            asset_prefix=str(data["asset_prefix"]),
            primary_model_catalog_id=str(data["primary_model_catalog_id"]),
            primary_model_relpath=str(data["primary_model_relpath"]),
            secondary_reference_catalog_id=str(data["secondary_reference_catalog_id"]),
            secondary_reference_relpath=str(data["secondary_reference_relpath"]),
            felt_source_catalog_id=str(data["felt_source_catalog_id"]),
            felt_source_relpath=str(data["felt_source_relpath"]),
            felt_patches=tuple(FeltPatchBoxSpec.from_dict(p) for p in data["felt_patches"]),
            felt_tile_draw_wh=(int(ftw[0]), int(ftw[1])),
            felt_seam_half_width_px=int(data["felt_seam_half_width_px"]),
            felt_target_mean_rgb=(float(fmean[0]), float(fmean[1]), float(fmean[2])),
            felt_target_std_rgb=(float(fstd[0]), float(fstd[1]), float(fstd[2])),
            felt_spotlight_gain=float(data["felt_spotlight_gain"]),
            felt_vignette_drop=float(data["felt_vignette_drop"]),
            felt_lama_dilation_px=int(data["felt_lama_dilation_px"]),
            felt_lama_blur_sigma=float(data["felt_lama_blur_sigma"]),
            felt_depth_sigma=float(data["felt_depth_sigma"]),
            pad_margin_ltrb=(int(pm[0]), int(pm[1]), int(pm[2]), int(pm[3])),
            studio_cloth_fill_rgb=(int(sc[0]), int(sc[1]), int(sc[2])),
            border_completion_boxes_padded=tuple(
                BoundingBoxPrompt.from_dict(b) for b in data["border_completion_boxes_padded"]
            ),
            border_lama_dilation_px=int(data["border_lama_dilation_px"]),
            border_lama_blur_sigma=float(data["border_lama_blur_sigma"]),
            padded_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["padded_sam_prompts"]
            ),
            pergola_cloth_hole_box_padded=BoundingBoxPrompt.from_dict(
                data["pergola_cloth_hole_box_padded"]
            ),
            blue_cloth_excess_thresh=float(data["blue_cloth_excess_thresh"]),
            blue_cloth_max_lum=float(data["blue_cloth_max_lum"]),
            maquette_dest_center_xy=(int(mc[0]), int(mc[1])),
            maquette_dest_width=int(data["maquette_dest_width"]),
            unsharp_sigma=float(data["unsharp_sigma"]),
            unsharp_amount=float(data["unsharp_amount"]),
            contact_shadow_offset_xy=(int(cso[0]), int(cso[1])),
            contact_shadow_blur_sigma=float(data["contact_shadow_blur_sigma"]),
            contact_shadow_strength=float(data["contact_shadow_strength"]),
            fg_corner_felt_fold_box=(int(fcf[0]), int(fcf[1]), int(fcf[2]), int(fcf[3])),
            fg_vignette_max_alpha=float(data["fg_vignette_max_alpha"]),
            fg_vignette_inner_radius_norm=float(data["fg_vignette_inner_radius_norm"]),
            fg_vignette_outer_radius_norm=float(data["fg_vignette_outer_radius_norm"]),
            layer_depth_min_u8=int(data["layer_depth_min_u8"]),
            layer_depth_collar_sigma=float(data["layer_depth_collar_sigma"]),
        )


@dataclass(frozen=True)
class PortalCubbySpec:
    cubby_id: str
    runtime_shelf_id: str
    row_index: int
    col_index: int
    target_scene_id: str
    label_he: str
    aperture_bounds_px: BoundingBoxPrompt
    backplane_sam_prompt: SamPointGroupPrompt
    occluder_sam_prompt: SamPointGroupPrompt | None = None
    occluder_label_he: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cubby_id": self.cubby_id,
            "runtime_shelf_id": self.runtime_shelf_id,
            "row_index": self.row_index,
            "col_index": self.col_index,
            "target_scene_id": self.target_scene_id,
            "label_he": self.label_he,
            "aperture_bounds_px": self.aperture_bounds_px.to_dict(),
            "backplane_sam_prompt": self.backplane_sam_prompt.to_dict(),
            "occluder_sam_prompt": (
                self.occluder_sam_prompt.to_dict()
                if self.occluder_sam_prompt is not None
                else None
            ),
            "occluder_label_he": self.occluder_label_he,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PortalCubbySpec:
        raw_occ = data.get("occluder_sam_prompt")
        raw_lbl = data.get("occluder_label_he")
        return cls(
            cubby_id=str(data["cubby_id"]),
            runtime_shelf_id=str(data["runtime_shelf_id"]),
            row_index=int(data["row_index"]),
            col_index=int(data["col_index"]),
            target_scene_id=str(data["target_scene_id"]),
            label_he=str(data["label_he"]),
            aperture_bounds_px=BoundingBoxPrompt.from_dict(data["aperture_bounds_px"]),
            backplane_sam_prompt=SamPointGroupPrompt.from_dict(data["backplane_sam_prompt"]),
            occluder_sam_prompt=(
                SamPointGroupPrompt.from_dict(raw_occ) if raw_occ is not None else None
            ),
            occluder_label_he=str(raw_lbl) if raw_lbl is not None else None,
        )


@dataclass(frozen=True)
class ShelfPortalSceneSpec:
    scene_id: str
    title_he: str
    short_label_he: str
    icon: str
    asset_prefix: str
    occluder_asset_prefix: str
    portal_layout_version: str
    source_catalog_id: str
    source_relpath: str
    gallery_room_catalog_id: str
    gallery_room_relpath: str
    gallery_room_crop_xyxy: tuple[int, int, int, int]
    gallery_room_heal_sam_prompts: tuple[SamPointGroupPrompt, ...]
    gallery_room_heal_max_x_px: int
    source_crop_xyxy: tuple[int, int, int, int]
    cabinet_body_box_xyxy: tuple[int, int, int, int]
    cabinet_side_cheek_width_px: int
    cabinet_side_cheek_start_src_y: int
    concrete_beam_sample_xyxy: tuple[int, int, int, int]
    cabinet_side_cheek_gain: float
    outer_edge_feather_px: int
    mid_perimeter_guard_px: int
    inner_box_rim_tuck_px: float
    occluder_sprite_pad_px: int
    aperture_interior_inset_px: int
    aperture_close_kernel_px: int
    portal_world_z: float
    portal_recess_depth: float
    webgl_plane_width: float
    webgl_plane_height: float
    portal_plane_bleed_px: int
    cutout_edge_feather_sigma: float
    crown_ceiling_cutout_sam_prompts: tuple[SamPointGroupPrompt, ...]
    crown_ceiling_close_kernel_px: int
    recessed_mid_sam_prompts: tuple[SamPointGroupPrompt, ...]
    portal_cubbies: tuple[PortalCubbySpec, ...]
    bg_dark_felt_gain: float
    bg_lama_dilation_px: int
    bg_lama_overlap_outer_pad_px: int
    bg_lama_blur_sigma: float
    bg_lama_color_match_sigma: float
    layer_depth_min_u8: int
    layer_depth_collar_sigma: float
    scene_bg_z: float
    scene_mid_z: float
    scene_fg_z: float
    room_depth_amplitude: float
    initial_camera_y: float
    initial_camera_z: float
    min_camera_z: float
    max_camera_z: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title_he": self.title_he,
            "short_label_he": self.short_label_he,
            "icon": self.icon,
            "asset_prefix": self.asset_prefix,
            "occluder_asset_prefix": self.occluder_asset_prefix,
            "portal_layout_version": self.portal_layout_version,
            "source_catalog_id": self.source_catalog_id,
            "source_relpath": self.source_relpath,
            "gallery_room_catalog_id": self.gallery_room_catalog_id,
            "gallery_room_relpath": self.gallery_room_relpath,
            "gallery_room_crop_xyxy": list(self.gallery_room_crop_xyxy),
            "gallery_room_heal_sam_prompts": [
                p.to_dict() for p in self.gallery_room_heal_sam_prompts
            ],
            "gallery_room_heal_max_x_px": self.gallery_room_heal_max_x_px,
            "source_crop_xyxy": list(self.source_crop_xyxy),
            "cabinet_body_box_xyxy": list(self.cabinet_body_box_xyxy),
            "cabinet_side_cheek_width_px": self.cabinet_side_cheek_width_px,
            "cabinet_side_cheek_start_src_y": self.cabinet_side_cheek_start_src_y,
            "concrete_beam_sample_xyxy": list(self.concrete_beam_sample_xyxy),
            "cabinet_side_cheek_gain": self.cabinet_side_cheek_gain,
            "outer_edge_feather_px": self.outer_edge_feather_px,
            "mid_perimeter_guard_px": self.mid_perimeter_guard_px,
            "inner_box_rim_tuck_px": self.inner_box_rim_tuck_px,
            "occluder_sprite_pad_px": self.occluder_sprite_pad_px,
            "aperture_interior_inset_px": self.aperture_interior_inset_px,
            "aperture_close_kernel_px": self.aperture_close_kernel_px,
            "portal_world_z": self.portal_world_z,
            "portal_recess_depth": self.portal_recess_depth,
            "webgl_plane_width": self.webgl_plane_width,
            "webgl_plane_height": self.webgl_plane_height,
            "portal_plane_bleed_px": self.portal_plane_bleed_px,
            "cutout_edge_feather_sigma": self.cutout_edge_feather_sigma,
            "crown_ceiling_cutout_sam_prompts": [
                p.to_dict() for p in self.crown_ceiling_cutout_sam_prompts
            ],
            "crown_ceiling_close_kernel_px": self.crown_ceiling_close_kernel_px,
            "recessed_mid_sam_prompts": [p.to_dict() for p in self.recessed_mid_sam_prompts],
            "portal_cubbies": [c.to_dict() for c in self.portal_cubbies],
            "bg_dark_felt_gain": self.bg_dark_felt_gain,
            "bg_lama_dilation_px": self.bg_lama_dilation_px,
            "bg_lama_overlap_outer_pad_px": self.bg_lama_overlap_outer_pad_px,
            "bg_lama_blur_sigma": self.bg_lama_blur_sigma,
            "bg_lama_color_match_sigma": self.bg_lama_color_match_sigma,
            "layer_depth_min_u8": self.layer_depth_min_u8,
            "layer_depth_collar_sigma": self.layer_depth_collar_sigma,
            "scene_bg_z": self.scene_bg_z,
            "scene_mid_z": self.scene_mid_z,
            "scene_fg_z": self.scene_fg_z,
            "room_depth_amplitude": self.room_depth_amplitude,
            "initial_camera_y": self.initial_camera_y,
            "initial_camera_z": self.initial_camera_z,
            "min_camera_z": self.min_camera_z,
            "max_camera_z": self.max_camera_z,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShelfPortalSceneSpec:
        sc = data["source_crop_xyxy"]
        cbb = data["cabinet_body_box_xyxy"]
        grc = data["gallery_room_crop_xyxy"]
        cbs = data["concrete_beam_sample_xyxy"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
            short_label_he=str(data["short_label_he"]),
            icon=str(data["icon"]),
            asset_prefix=str(data["asset_prefix"]),
            occluder_asset_prefix=str(data["occluder_asset_prefix"]),
            portal_layout_version=str(data["portal_layout_version"]),
            source_catalog_id=str(data["source_catalog_id"]),
            source_relpath=str(data["source_relpath"]),
            gallery_room_catalog_id=str(data["gallery_room_catalog_id"]),
            gallery_room_relpath=str(data["gallery_room_relpath"]),
            gallery_room_crop_xyxy=(int(grc[0]), int(grc[1]), int(grc[2]), int(grc[3])),
            gallery_room_heal_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p)
                for p in data["gallery_room_heal_sam_prompts"]
            ),
            gallery_room_heal_max_x_px=int(data["gallery_room_heal_max_x_px"]),
            source_crop_xyxy=(int(sc[0]), int(sc[1]), int(sc[2]), int(sc[3])),
            cabinet_body_box_xyxy=(int(cbb[0]), int(cbb[1]), int(cbb[2]), int(cbb[3])),
            cabinet_side_cheek_width_px=int(data["cabinet_side_cheek_width_px"]),
            cabinet_side_cheek_start_src_y=int(data["cabinet_side_cheek_start_src_y"]),
            concrete_beam_sample_xyxy=(int(cbs[0]), int(cbs[1]), int(cbs[2]), int(cbs[3])),
            cabinet_side_cheek_gain=float(data["cabinet_side_cheek_gain"]),
            outer_edge_feather_px=int(data["outer_edge_feather_px"]),
            mid_perimeter_guard_px=int(data["mid_perimeter_guard_px"]),
            inner_box_rim_tuck_px=float(data["inner_box_rim_tuck_px"]),
            occluder_sprite_pad_px=int(data["occluder_sprite_pad_px"]),
            aperture_interior_inset_px=int(data["aperture_interior_inset_px"]),
            aperture_close_kernel_px=int(data["aperture_close_kernel_px"]),
            portal_world_z=float(data["portal_world_z"]),
            portal_recess_depth=float(data["portal_recess_depth"]),
            webgl_plane_width=float(data["webgl_plane_width"]),
            webgl_plane_height=float(data["webgl_plane_height"]),
            portal_plane_bleed_px=int(data["portal_plane_bleed_px"]),
            cutout_edge_feather_sigma=float(data["cutout_edge_feather_sigma"]),
            crown_ceiling_cutout_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p)
                for p in data["crown_ceiling_cutout_sam_prompts"]
            ),
            crown_ceiling_close_kernel_px=int(data["crown_ceiling_close_kernel_px"]),
            recessed_mid_sam_prompts=tuple(
                SamPointGroupPrompt.from_dict(p) for p in data["recessed_mid_sam_prompts"]
            ),
            portal_cubbies=tuple(PortalCubbySpec.from_dict(c) for c in data["portal_cubbies"]),
            bg_dark_felt_gain=float(data["bg_dark_felt_gain"]),
            bg_lama_dilation_px=int(data["bg_lama_dilation_px"]),
            bg_lama_overlap_outer_pad_px=int(data["bg_lama_overlap_outer_pad_px"]),
            bg_lama_blur_sigma=float(data["bg_lama_blur_sigma"]),
            bg_lama_color_match_sigma=float(data["bg_lama_color_match_sigma"]),
            layer_depth_min_u8=int(data["layer_depth_min_u8"]),
            layer_depth_collar_sigma=float(data["layer_depth_collar_sigma"]),
            scene_bg_z=float(data["scene_bg_z"]),
            scene_mid_z=float(data["scene_mid_z"]),
            scene_fg_z=float(data["scene_fg_z"]),
            room_depth_amplitude=float(data["room_depth_amplitude"]),
            initial_camera_y=float(data["initial_camera_y"]),
            initial_camera_z=float(data["initial_camera_z"]),
            min_camera_z=float(data["min_camera_z"]),
            max_camera_z=float(data["max_camera_z"]),
        )


@dataclass(frozen=True)
class NewScenesBundleConfig:
    scenes_output_relpath: str
    drawings_output_relpath: str
    spec_output_abspath: str
    dimensions: CanvasDimensionsConfig
    inner_dimensions: CanvasDimensionsConfig
    expanded_bg_scale: float
    lama_max_working_side_px: int
    lama_roi_context_margin_px: int
    depth_model: DepthModelConfig
    segmentation: SegmentationConfig
    inpainting: InpaintingConfig
    graphite_drawing: GraphiteDrawingConfig
    wandering_drawing_1: WanderingDrawing1Spec
    denya_model: DenyaModelSceneSpec
    shelf_portal: ShelfPortalSceneSpec

    @property
    def expanded_dimensions(self) -> CanvasDimensionsConfig:
        return CanvasDimensionsConfig(
            width=int(round(self.inner_dimensions.width * self.expanded_bg_scale)),
            height=int(round(self.inner_dimensions.height * self.expanded_bg_scale)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenes_output_relpath": self.scenes_output_relpath,
            "drawings_output_relpath": self.drawings_output_relpath,
            "spec_output_abspath": self.spec_output_abspath,
            "dimensions": self.dimensions.to_dict(),
            "inner_dimensions": self.inner_dimensions.to_dict(),
            "expanded_bg_scale": self.expanded_bg_scale,
            "lama_max_working_side_px": self.lama_max_working_side_px,
            "lama_roi_context_margin_px": self.lama_roi_context_margin_px,
            "depth_model": self.depth_model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "inpainting": self.inpainting.to_dict(),
            "graphite_drawing": self.graphite_drawing.to_dict(),
            "wandering_drawing_1": self.wandering_drawing_1.to_dict(),
            "denya_model": self.denya_model.to_dict(),
            "shelf_portal": self.shelf_portal.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NewScenesBundleConfig:
        return cls(
            scenes_output_relpath=str(data["scenes_output_relpath"]),
            drawings_output_relpath=str(data["drawings_output_relpath"]),
            spec_output_abspath=str(data["spec_output_abspath"]),
            dimensions=CanvasDimensionsConfig.from_dict(data["dimensions"]),
            inner_dimensions=CanvasDimensionsConfig.from_dict(data["inner_dimensions"]),
            expanded_bg_scale=float(data["expanded_bg_scale"]),
            lama_max_working_side_px=int(data["lama_max_working_side_px"]),
            lama_roi_context_margin_px=int(data["lama_roi_context_margin_px"]),
            depth_model=DepthModelConfig.from_dict(data["depth_model"]),
            segmentation=SegmentationConfig.from_dict(data["segmentation"]),
            inpainting=InpaintingConfig.from_dict(data["inpainting"]),
            graphite_drawing=GraphiteDrawingConfig.from_dict(data["graphite_drawing"]),
            wandering_drawing_1=WanderingDrawing1Spec.from_dict(data["wandering_drawing_1"]),
            denya_model=DenyaModelSceneSpec.from_dict(data["denya_model"]),
            shelf_portal=ShelfPortalSceneSpec.from_dict(data["shelf_portal"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw_json: str) -> NewScenesBundleConfig:
        return cls.from_dict(json.loads(raw_json))


# ============================================================================
# Canonical Configuration Factory (Zero Magic Numbers Inline)
# ============================================================================


def default_new_scenes_bundle_config() -> NewScenesBundleConfig:
    dims = CanvasDimensionsConfig(width=1800, height=1200)
    depth_cfg = DepthModelConfig(
        model_id="depth-anything/Depth-Anything-V2-Small-hf",
        background_gaussian_sigma=14.0,
        depth_discontinuity_percentile_fg=62.0,
        depth_discontinuity_percentile_mid=30.0,
    )
    seg_cfg = SegmentationConfig(
        rembg_model_name="isnet-general-use",
        sam_model_id="facebook/sam-vit-base",
        alpha_foreground_threshold=48,
        alpha_full_opaque_threshold=185,
        min_component_area_px=120,
        sam_morph_close_px=5,
        edge_feather_sigma=0.85,
    )
    inpaint_cfg = InpaintingConfig(
        dilation_radius_px=14,
        mask_blur_sigma=2.0,
        jpeg_quality=92,
    )
    draw_cfg = GraphiteDrawingConfig(
        paper_downsample_width=225,
        paper_downsample_height=150,
        paper_bg_percentile=88,
        paper_filter_size=11,
        frame_border_ignore_px=18,
        min_component_area_px=35,
        edge_feather_sigma=0.75,
        lama_dilation_radius_px=10,
        lama_mask_blur_sigma=1.5,
        dome_bg_percentile=86.0,
        dome_ink_delta_threshold=11.0,
        dome_morph_close_px=21,
        dome_power_exponent=0.65,
        dome_gaussian_sigma=14.0,
        stage_y_weight=0.34,
        stage_y_exponent=1.30,
        stage_x_weight=0.12,
        depth_dome_weight=0.46,
        depth_stage_weight=0.22,
        depth_neural_weight=0.32,
        depth_final_sigma=9.0,
        oval_board_scaled_width=1200,
        oval_board_scaled_height=500,
        oval_center_xy=(600.0, 252.0),
        oval_radii_xy=(578.0, 236.0),
        oval_min_component_area_px=95,
    )

    wandering1 = WanderingDrawing1Spec(
        scene_id="wandering_drawing_1",
        title_he="׳שיטוט ברישום 1׳ — מרחב רישומי משולב (בחירת נעמה)",
        short_label_he="שיטוט ברישום 1",
        icon="🖋️",
        asset_prefix="scene7_wandering1",
        thumbnail_height_px=480,
        thumbnail_jpeg_quality=92,
        paper_base_rgb=(238.0, 236.0, 230.0),
        paper_grain_source_relpath="media/portfolio_site/IMG_7167B-m-scaled.jpg_bef5ac69.jpg",
        paper_grain_crop_norm=(0.06, 0.02, 0.94, 0.19),
        paper_grain_amplitude=4.5,
        paper_vignette_strength=0.055,
        paper_downsample_divisor=12,
        paper_filter_size=9,
        unmix_alpha_floor=0.22,
        layer_depth_min_u8=42,
        layer_depth_collar_sigma=6.0,
        lama_heal_dilation_px=8,
        lama_heal_blur_sigma=1.5,
        artworks=(
            WanderingArtworkEntrySpec(
                catalog_id="IMG_PORT_0259",
                dropped_duplicate_ids=(),
                source_relpath="media/portfolio_site/portfolio_drawing_broken_screen_2021_1823cbfe.jpg",
                thumbnail_filename="thumb_w1_broken_screen_0259.jpg",
                title_he="מסך שבור — אכסדרת פרספקטיבה בדיו ועיפרון",
                year_label="2021",
                assigned_layer="bg",
                placement_role="full_width_upper_architectural_colonnade",
            ),
            WanderingArtworkEntrySpec(
                catalog_id="IMG_PORT_0260",
                dropped_duplicate_ids=("IMG_PORT_0047",),
                source_relpath="media/portfolio_site/portfolio_drawing_reflection_floor_1_2021_ea1865cb.jpg",
                thumbnail_filename="thumb_w1_reflection_floor_0260.jpg",
                title_he="השתקפות רצפה 1 — רישום פרספקטיבה והשתקפויות",
                year_label="2021",
                assigned_layer="bg",
                placement_role="full_width_lower_reflection_floor",
            ),
            WanderingArtworkEntrySpec(
                catalog_id="IMG_PORT_0042",
                dropped_duplicate_ids=(),
                source_relpath="media/portfolio_site/IMG_7135B-m-scaled.jpg_78f770a3.jpg",
                thumbnail_filename="thumb_w1_golden_palms_0042.jpg",
                title_he="צמרת דקל וגגות בעיפרון צבעוני וגרפיט",
                year_label="2020–2022",
                assigned_layer="bg",
                placement_role="right_center_horizon_palm_and_rooftops",
            ),
            WanderingArtworkEntrySpec(
                catalog_id="IMG_PORT_0045",
                dropped_duplicate_ids=("IMG_PORT_0046",),
                source_relpath="media/portfolio_site/IMG_7148B-m.jpg_ca9ddebb.jpg",
                thumbnail_filename="thumb_w1_vertical_refl_0045.jpg",
                title_he="השתקפות חלון אנכית וחוטי חשמל",
                year_label="2021",
                assigned_layer="bg",
                placement_role="left_center_architectural_window_reflection",
            ),
            WanderingArtworkEntrySpec(
                catalog_id="IMG_PORT_0043",
                dropped_duplicate_ids=("IMG_PORT_0036",),
                source_relpath="media/portfolio_site/IMG_7141-m-scaled.jpg_732de4b2.jpg",
                thumbnail_filename="thumb_w1_road_ribbons_0043.jpg",
                title_he="סרטי כביש מקופלים — תבליט נייר גזור על קיר",
                year_label="2021",
                assigned_layer="mid",
                placement_role="center_left_floating_road_ribbons_sculpture",
            ),
            WanderingArtworkEntrySpec(
                catalog_id="IMG_MUSE_0256",
                dropped_duplicate_ids=(),
                source_relpath="media/museum_press/rgma_bene_beitcha_scroll_drawing_foot_2026_331d870d.jpg",
                thumbnail_filename="thumb_w1_scroll_foot_0256.jpg",
                title_he="מגילת רישום בוטנית וכף רגל מפוסלת (׳בנה ביתך׳)",
                year_label="2026",
                assigned_layer="mid",
                placement_role="right_cascading_scroll_and_sculpted_foot",
            ),
            WanderingArtworkEntrySpec(
                catalog_id="IMG_BINY_0018",
                dropped_duplicate_ids=(),
                source_relpath="media/binyamin_gallery/binyamin_פורום_פרט.jpg_4e23d30b.jpg",
                thumbnail_filename="thumb_w1_forum_detail_0018.jpg",
                title_he="פורום (פרט) — דיוקן מכני, בועות מחשבה ומדף עשב",
                year_label="2022",
                assigned_layer="fg",
                placement_role="left_foreground_wire_profile_and_grass_shelf",
            ),
            WanderingArtworkEntrySpec(
                catalog_id="IMG_PORT_0049",
                dropped_duplicate_ids=(),
                source_relpath="media/portfolio_site/IMG_7167B-m-scaled.jpg_bef5ac69.jpg",
                thumbnail_filename="thumb_w1_yellow_crane_0049.jpg",
                title_he="מגדל עגורן וחלונות צהובים ברישום ודיו",
                year_label="2021",
                assigned_layer="fg",
                placement_role="lower_center_left_crane_and_yellow_windows",
            ),
            WanderingArtworkEntrySpec(
                catalog_id="IMG_PORT_0044",
                dropped_duplicate_ids=(),
                source_relpath="media/portfolio_site/IMG_7147B-m.jpg_850eca39.jpg",
                thumbnail_filename="thumb_w1_potted_stool_0044.jpg",
                title_he="עציץ על שרפרף ברישום גרפיט",
                year_label="2021",
                assigned_layer="fg",
                placement_role="lower_center_right_potted_plant_on_stool",
            ),
        ),
        bg_zones=(
            PaperTransmittanceZoneSpec(
                catalog_id="IMG_PORT_0259",
                source_relpath="media/portfolio_site/portfolio_drawing_broken_screen_2021_1823cbfe.jpg",
                src_crop_norm=(0.0, 0.0, 1.0, 1.0),
                dest_box_xyxy=(210, 140, 2490, 1180),
                feather_left_px=210,
                feather_top_px=150,
                feather_right_px=210,
                feather_bottom_px=230,
                elliptical_falloff=False,
                transmittance_strength=0.96,
                paper_percentile=88.0,
            ),
            PaperTransmittanceZoneSpec(
                catalog_id="IMG_PORT_0260",
                source_relpath="media/portfolio_site/portfolio_drawing_reflection_floor_1_2021_ea1865cb.jpg",
                src_crop_norm=(0.0, 0.0, 1.0, 1.0),
                dest_box_xyxy=(210, 910, 2490, 1660),
                feather_left_px=210,
                feather_top_px=220,
                feather_right_px=210,
                feather_bottom_px=150,
                elliptical_falloff=False,
                transmittance_strength=0.92,
                paper_percentile=88.0,
            ),
            PaperTransmittanceZoneSpec(
                catalog_id="IMG_PORT_0042",
                source_relpath="media/portfolio_site/IMG_7135B-m-scaled.jpg_78f770a3.jpg",
                src_crop_norm=(0.02, 0.45, 0.98, 0.99),
                dest_box_xyxy=(1217, 608, 1721, 1079),
                feather_left_px=110,
                feather_top_px=105,
                feather_right_px=110,
                feather_bottom_px=110,
                elliptical_falloff=True,
                transmittance_strength=0.90,
                paper_percentile=88.0,
            ),
            PaperTransmittanceZoneSpec(
                catalog_id="IMG_PORT_0045",
                source_relpath="media/portfolio_site/IMG_7148B-m.jpg_ca9ddebb.jpg",
                src_crop_norm=(0.03, 0.03, 0.97, 0.97),
                dest_box_xyxy=(721, 462, 1142, 1046),
                feather_left_px=96,
                feather_top_px=80,
                feather_right_px=105,
                feather_bottom_px=105,
                elliptical_falloff=True,
                transmittance_strength=0.82,
                paper_percentile=88.0,
            ),
        ),
        islands=(
            WanderingLayerIslandSpec(
                catalog_id="IMG_PORT_0043",
                source_relpath="media/portfolio_site/IMG_7141-m-scaled.jpg_732de4b2.jpg",
                target_layer="mid",
                extraction_method="wall_cutout_unmix",
                src_crop_norm=(0.01, 0.04, 0.72, 0.96),
                dest_xy=(235, 115),
                dest_wh=(815, 595),
                paper_percentile=86.0,
                lum_delta_floor=16.0,
                lum_delta_full=56.0,
                chroma_weight=0.55,
                alpha_gamma=0.58,
                perimeter_taper_px=18,
                bottom_taper_px=18,
                min_component_area_px=65,
                max_interior_hole_fill_px=2200,
                lama_heal_boxes_px=(
                    BoundingBoxPrompt(x0=0, y0=225, x1=94, y1=418),
                    BoundingBoxPrompt(x0=878, y0=560, x1=1030, y1=712),
                    BoundingBoxPrompt(x0=1124, y0=556, x1=1246, y1=712),
                    BoundingBoxPrompt(x0=1330, y0=554, x1=1564, y1=712),
                ),
            ),
            WanderingLayerIslandSpec(
                catalog_id="IMG_MUSE_0256",
                source_relpath="media/museum_press/rgma_bene_beitcha_scroll_drawing_foot_2026_331d870d.jpg",
                target_layer="mid",
                extraction_method="sam_scroll_sculpture",
                src_crop_norm=(0.25, 0.02, 0.95, 0.97),
                dest_xy=(1208, 22),
                dest_wh=(565, 1152),
                paper_percentile=88.0,
                lum_delta_floor=12.0,
                lum_delta_full=64.0,
                chroma_weight=0.50,
                alpha_gamma=0.75,
                perimeter_taper_px=12,
                bottom_taper_px=12,
                min_component_area_px=4000,
                max_interior_hole_fill_px=18000,
                sam_prompts=(
                    SamPointGroupPrompt(
                        box=BoundingBoxPrompt(x0=36, y0=12, x1=768, y1=1568),
                        points_xy=(
                            (260.0, 120.0),
                            (285.0, 380.0),
                            (295.0, 680.0),
                            (315.0, 980.0),
                            (380.0, 1260.0),
                            (235.0, 1290.0),
                            (465.0, 1410.0),
                            (580.0, 1380.0),
                            (45.0, 820.0),
                            (62.0, 1340.0),
                            (675.0, 420.0),
                            (715.0, 980.0),
                            (185.0, 1525.0),
                        ),
                        point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
                    ),
                ),
            ),
            WanderingLayerIslandSpec(
                catalog_id="IMG_BINY_0018",
                source_relpath="media/binyamin_gallery/binyamin_פורום_פרט.jpg_4e23d30b.jpg",
                target_layer="fg",
                extraction_method="optical_pigment_unmix",
                src_crop_norm=(0.18, 0.08, 0.875, 0.98),
                dest_xy=(22, 235),
                dest_wh=(690, 625),
                paper_percentile=88.0,
                lum_delta_floor=14.0,
                lum_delta_full=68.0,
                chroma_weight=0.95,
                alpha_gamma=0.62,
                perimeter_taper_px=38,
                bottom_taper_px=38,
                min_component_area_px=28,
            ),
            WanderingLayerIslandSpec(
                catalog_id="IMG_PORT_0049",
                source_relpath="media/portfolio_site/IMG_7167B-m-scaled.jpg_bef5ac69.jpg",
                target_layer="fg",
                extraction_method="optical_pigment_unmix",
                src_crop_norm=(0.02, 0.19, 0.98, 1.00),
                dest_xy=(475, 578),
                dest_wh=(395, 600),
                paper_percentile=88.0,
                lum_delta_floor=10.5,
                lum_delta_full=66.0,
                chroma_weight=1.35,
                alpha_gamma=0.68,
                perimeter_taper_px=64,
                bottom_taper_px=48,
                min_component_area_px=36,
            ),
            WanderingLayerIslandSpec(
                catalog_id="IMG_PORT_0044",
                source_relpath="media/portfolio_site/IMG_7147B-m.jpg_850eca39.jpg",
                target_layer="fg",
                extraction_method="optical_pigment_unmix",
                src_crop_norm=(0.02, 0.07, 0.94, 1.00),
                dest_xy=(855, 598),
                dest_wh=(435, 580),
                paper_percentile=88.0,
                lum_delta_floor=10.5,
                lum_delta_full=64.0,
                chroma_weight=0.75,
                alpha_gamma=0.68,
                perimeter_taper_px=62,
                bottom_taper_px=48,
                min_component_area_px=36,
            ),
        ),
        scroll_portal=WanderingScrollPortalSpec(
            portal_id="portal_wandering1_scroll_palm",
            target_scene_id="scroll_palm_world",
            label_he="🌴 עולם הדקל והמוטות בתוך המגילה · ׳מגילה וכף רגל׳",
            short_badge_he="🌴 אל תוך המגילה",
            foot_layer_id="scroll_foot",
            foot_label_he="כף הרגל הקרמית והמוט העטוף בחזית ׳מגילה וכף רגל׳",
            scroll_catalog_id="IMG_MUSE_0256",
            inner_box_px=(1070, 408, 1258, 734),
            veil_feather_px=20,
            mid_z=-80.0,
            foot_z_pos=-48.0,
            foot_depth_amplitude=28.0,
            foot_depth_min_norm=0.22,
            foot_depth_max_norm=0.92,
            foot_heal_dilation_px=9,
            foot_heal_blur_sigma=2.0,
            foot_min_component_area_px=120,
            foot_sam_prompts=(
                SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=240, y0=452, x1=468, y1=1384),
                    points_xy=(
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
                    ),
                    point_labels=(
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0,
                    ),
                ),
            ),
        ),
    )

    denya_spec = DenyaModelSceneSpec(
        scene_id="denya_model",
        title_he="דגם בית דניה על לבד אפור (רחוב גרינבוים 80, חיפה)",
        short_label_he="דגם בית דניה",
        icon="🏡",
        asset_prefix="scene8_denya_model",
        primary_model_catalog_id="IMG_WORD_0189",
        primary_model_relpath="media/wordpress_blog/wp_wide_d79ed795d793d79c_a8d33353.jpg",
        secondary_reference_catalog_id="IMG_WORD_0148",
        secondary_reference_relpath="media/wordpress_blog/wp_house_d7a1d7a8d799d7a7d794-3_388207f4.jpg",
        felt_source_catalog_id="IMG_BINY_0007",
        felt_source_relpath="media/binyamin_gallery/binyamin_ארון_קיר_פרט_צד_שני.jpg_f7e9020f.jpg",
        felt_patches=(
            FeltPatchBoxSpec("r3_c1", (46, 506, 142, 620)),
            FeltPatchBoxSpec("r4_c4", (674, 778, 778, 878)),
            FeltPatchBoxSpec("r5_c1", (46, 948, 142, 1050)),
            FeltPatchBoxSpec("r6_c2", (170, 1124, 268, 1214)),
            FeltPatchBoxSpec("r7_c1", (46, 1301, 132, 1382)),
            FeltPatchBoxSpec("r7_c2", (292, 1301, 390, 1382)),
            FeltPatchBoxSpec("r7_c4", (821, 1301, 902, 1382)),
        ),
        felt_tile_draw_wh=(360, 360),
        felt_seam_half_width_px=48,
        felt_target_mean_rgb=(53.0, 50.5, 52.5),
        felt_target_std_rgb=(17.0, 16.0, 16.5),
        felt_spotlight_gain=0.20,
        felt_vignette_drop=0.22,
        felt_lama_dilation_px=4,
        felt_lama_blur_sigma=1.2,
        felt_depth_sigma=12.0,
        pad_margin_ltrb=(72, 36, 56, 68),
        studio_cloth_fill_rgb=(9, 14, 25),
        border_completion_boxes_padded=(
            BoundingBoxPrompt(x0=20, y0=205, x1=80, y1=428),
            BoundingBoxPrompt(x0=680, y0=618, x1=884, y1=668),
            BoundingBoxPrompt(x0=854, y0=514, x1=894, y1=644),
        ),
        border_lama_dilation_px=4,
        border_lama_blur_sigma=1.2,
        padded_sam_prompts=(
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=18, y0=148, x1=892, y1=666),
                points_xy=(
                    (336.0, 290.0),
                    (536.0, 320.0),
                    (726.0, 345.0),
                    (381.0, 172.0),
                    (656.0, 210.0),
                    (197.0, 238.0),
                    (211.0, 435.0),
                    (491.0, 520.0),
                    (761.0, 505.0),
                    (776.0, 595.0),
                    (111.0, 285.0),
                    (58.0, 295.0),
                    (10.0, 250.0),
                    (10.0, 365.0),
                    (196.0, 110.0),
                    (476.0, 105.0),
                    (776.0, 125.0),
                    (440.0, 185.0),
                    (851.0, 435.0),
                    (181.0, 575.0),
                    (411.0, 625.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            ),
        ),
        pergola_cloth_hole_box_padded=BoundingBoxPrompt(x0=420, y0=168, x1=464, y1=204),
        blue_cloth_excess_thresh=7.5,
        blue_cloth_max_lum=58.0,
        maquette_dest_center_xy=(900, 625),
        maquette_dest_width=1580,
        unsharp_sigma=1.15,
        unsharp_amount=0.38,
        contact_shadow_offset_xy=(16, 28),
        contact_shadow_blur_sigma=26.0,
        contact_shadow_strength=0.52,
        fg_corner_felt_fold_box=(536, 496, 674, 622),
        fg_vignette_max_alpha=0.0,
        fg_vignette_inner_radius_norm=0.62,
        fg_vignette_outer_radius_norm=0.72,
        layer_depth_min_u8=48,
        layer_depth_collar_sigma=6.0,
    )

    shelf_spec = ShelfPortalSceneSpec(
        scene_id="shelf_portal",
        title_he="קיר המדפים — שער בין עולמות היצירה",
        short_label_he="קיר המדפים (שער)",
        icon="🧱",
        asset_prefix="scene9_shelf_portal",
        occluder_asset_prefix="scene9_occluder",
        portal_layout_version="full_wall_v4",
        source_catalog_id="IMG_BINY_0007",
        source_relpath="media/binyamin_gallery/binyamin_ארון_קיר_פרט_צד_שני.jpg_f7e9020f.jpg",
        gallery_room_catalog_id="IMG_BINY_0015",
        gallery_room_relpath="media/binyamin_gallery/binyamin_מרכז_העיר_פינת_לב_הבית_מראה_הצבה.jpg_f27ae486.jpg",
        gallery_room_crop_xyxy=(0, 0, 2790, 1860),
        gallery_room_heal_sam_prompts=(
            # Steel U-handrail + grey cloth on back gallery wall (preserving wall/skirting inside the arch)
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=1142, y0=922, x1=1375, y1=1178),
                points_xy=(
                    (1174.0, 995.0),
                    (1174.0, 1085.0),
                    (1174.0, 1140.0),
                    (1212.0, 945.0),
                    (1242.0, 948.0),
                    (1290.0, 942.0),
                    (1338.0, 995.0),
                    (1338.0, 1085.0),
                    (1338.0, 1162.0),
                    (1255.0, 1015.0),
                    (1255.0, 1065.0),
                    (1255.0, 1096.0),
                    (1255.0, 1140.0),
                    (1135.0, 1020.0),
                    (1382.0, 1020.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
            ),
            # Protruding tow-hitch tongue + red cloth left of green trolley deck (`x < 1656`)
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=1494, y0=1184, x1=1656, y1=1308),
                points_xy=(
                    (1538.0, 1202.0),
                    (1516.0, 1225.0),
                    (1558.0, 1235.0),
                    (1572.0, 1278.0),
                    (1625.0, 1252.0),
                    (1510.0, 1275.0),
                    (1615.0, 1215.0),
                    (1680.0, 1255.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 0, 0, 0),
            ),
        ),
        gallery_room_heal_max_x_px=1654,
        source_crop_xyxy=(0, 0, 1001, 1472),
        cabinet_body_box_xyxy=(1062, 396, 1638, 1244),
        cabinet_side_cheek_width_px=5,
        cabinet_side_cheek_start_src_y=88,
        concrete_beam_sample_xyxy=(20, 662, 980, 725),
        cabinet_side_cheek_gain=0.90,
        outer_edge_feather_px=2,
        mid_perimeter_guard_px=18,
        inner_box_rim_tuck_px=4.0,
        occluder_sprite_pad_px=4,
        aperture_interior_inset_px=5,
        aperture_close_kernel_px=19,
        portal_world_z=-284.0,
        portal_recess_depth=160.0,
        webgl_plane_width=1260.0,
        webgl_plane_height=840.0,
        portal_plane_bleed_px=10,
        cutout_edge_feather_sigma=0.85,
        crown_ceiling_cutout_sam_prompts=(
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=562, y0=0, x1=724, y1=48),
                points_xy=(
                    (585.0, 18.0),
                    (645.0, 18.0),
                    (705.0, 18.0),
                    (585.0, 36.0),
                    (645.0, 36.0),
                    (700.0, 36.0),
                    (550.0, 18.0),
                    (550.0, 38.0),
                    (735.0, 18.0),
                    (730.0, 38.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=562, y0=40, x1=718, y1=56),
                points_xy=(
                    (580.0, 48.0),
                    (625.0, 48.0),
                    (670.0, 48.0),
                    (705.0, 48.0),
                    (550.0, 48.0),
                    (725.0, 48.0),
                    (625.0, 34.0),
                    (625.0, 62.0),
                ),
                point_labels=(1, 1, 1, 1, 0, 0, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=560, y0=50, x1=716, y1=81),
                points_xy=(
                    (580.0, 62.0),
                    (620.0, 62.0),
                    (680.0, 55.0),
                    (580.0, 74.0),
                    (630.0, 74.0),
                    (548.0, 65.0),
                    (675.0, 68.0),
                    (705.0, 68.0),
                    (605.0, 86.0),
                    (665.0, 86.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=846, y0=0, x1=889, y1=25),
                points_xy=(
                    (860.0, 12.0),
                    (878.0, 12.0),
                    (868.0, 20.0),
                    (838.0, 12.0),
                    (896.0, 12.0),
                    (868.0, 30.0),
                ),
                point_labels=(1, 1, 1, 0, 0, 0),
            ),
        ),
        crown_ceiling_close_kernel_px=5,
        recessed_mid_sam_prompts=(
            # 3 circular holes inside the Row-4 hollow concrete block (native 1001x1472 coords)
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=392, y0=774, x1=448, y1=830),
                points_xy=((420.0, 802.0), (410.0, 802.0), (384.0, 802.0), (455.0, 802.0)),
                point_labels=(1, 1, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=456, y0=774, x1=512, y1=830),
                points_xy=((484.0, 802.0), (474.0, 802.0), (448.0, 802.0), (519.0, 802.0)),
                point_labels=(1, 1, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=520, y0=774, x1=576, y1=830),
                points_xy=((548.0, 802.0), (538.0, 802.0), (512.0, 802.0), (584.0, 802.0)),
                point_labels=(1, 1, 0, 0),
            ),
            # Empty dark-felt cubbies recessed to `mid` for 2.5D cabinet depth
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=267, y0=112, x1=417, y1=281),
                points_xy=(
                    (305.0, 150.0),
                    (380.0, 150.0),
                    (342.0, 200.0),
                    (305.0, 250.0),
                    (380.0, 250.0),
                    (256.0, 195.0),
                    (426.0, 195.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=3, y0=492, x1=150, y1=638),
                points_xy=(
                    (42.0, 525.0),
                    (115.0, 525.0),
                    (78.0, 568.0),
                    (42.0, 612.0),
                    (115.0, 612.0),
                    (160.0, 568.0),
                    (78.0, 480.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 0, 0),
            ),
            SamPointGroupPrompt(
                box=BoundingBoxPrompt(x0=268, y0=1286, x1=415, y1=1452),
                points_xy=(
                    (305.0, 1320.0),
                    (380.0, 1320.0),
                    (342.0, 1370.0),
                    (305.0, 1425.0),
                    (380.0, 1425.0),
                    (256.0, 1370.0),
                    (425.0, 1370.0),
                ),
                point_labels=(1, 1, 1, 1, 1, 0, 0),
            ),
        ),
        portal_cubbies=(
            PortalCubbySpec(
                cubby_id="row1_col1",
                runtime_shelf_id="cubby_scroll_foot",
                row_index=0,
                col_index=0,
                target_scene_id="bene_beitcha_scroll_foot",
                label_he="📜 מגילה וכף רגל · ׳בנה ביתך׳",
                aperture_bounds_px=BoundingBoxPrompt(x0=3, y0=112, x1=148, y1=281),
                backplane_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=2, y0=110, x1=150, y1=283),
                    points_xy=(
                        (38.0, 138.0),
                        (112.0, 138.0),
                        (78.0, 175.0),
                        (122.0, 212.0),
                        (62.0, 238.0),
                        (32.0, 202.0),
                        (156.0, 175.0),
                        (75.0, 98.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0, 0, 0),
                ),
                occluder_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=2, y0=190, x1=149, y1=280),
                    points_xy=(
                        (32.0, 204.0),
                        (22.0, 232.0),
                        (14.0, 266.0),
                        (62.0, 236.0),
                        (95.0, 245.0),
                        (125.0, 264.0),
                        (72.0, 268.0),
                        (55.0, 196.0),
                        (82.0, 206.0),
                        (118.0, 214.0),
                        (136.0, 232.0),
                        (156.0, 235.0),
                        (75.0, 286.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
                ),
                occluder_label_he="שברי פורצלן ואבן כורכר",
            ),
            PortalCubbySpec(
                cubby_id="row1_col3",
                runtime_shelf_id="cubby_capsules",
                row_index=0,
                col_index=2,
                target_scene_id="paper_capsules",
                label_he="🫧 קפסולות נייר · אולי שם יהיה יותר טוב",
                aperture_bounds_px=BoundingBoxPrompt(x0=531, y0=110, x1=681, y1=281),
                backplane_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=529, y0=108, x1=683, y1=283),
                    points_xy=(
                        (562.0, 138.0),
                        (648.0, 138.0),
                        (608.0, 168.0),
                        (652.0, 215.0),
                        (592.0, 212.0),
                        (644.0, 258.0),
                        (520.0, 175.0),
                        (690.0, 175.0),
                        (605.0, 98.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0, 0, 0, 0),
                ),
                occluder_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=530, y0=192, x1=680, y1=280),
                    points_xy=(
                        (592.0, 214.0),
                        (556.0, 230.0),
                        (578.0, 262.0),
                        (546.0, 268.0),
                        (612.0, 268.0),
                        (644.0, 260.0),
                        (662.0, 268.0),
                        (626.0, 254.0),
                        (548.0, 200.0),
                        (636.0, 204.0),
                        (664.0, 224.0),
                        (640.0, 234.0),
                        (520.0, 235.0),
                        (688.0, 235.0),
                        (605.0, 286.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
                ),
                occluder_label_he="לוח בטון ופקעת חוטים צבעונית",
            ),
            PortalCubbySpec(
                cubby_id="row2_col1",
                runtime_shelf_id="cubby_wandering_2",
                row_index=1,
                col_index=0,
                target_scene_id="wandering_drawing",
                label_he="✏️ שיטוט 2 · איים מרחפים",
                aperture_bounds_px=BoundingBoxPrompt(x0=127, y0=307, x1=277, y1=465),
                backplane_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=125, y0=305, x1=279, y1=467),
                    points_xy=(
                        (158.0, 338.0),
                        (242.0, 338.0),
                        (202.0, 378.0),
                        (222.0, 415.0),
                        (168.0, 435.0),
                        (258.0, 438.0),
                        (116.0, 385.0),
                        (286.0, 385.0),
                        (202.0, 298.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0, 0, 0, 0),
                ),
                occluder_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=132, y0=408, x1=276, y1=461),
                    points_xy=(
                        (168.0, 428.0),
                        (152.0, 448.0),
                        (195.0, 446.0),
                        (232.0, 452.0),
                        (258.0, 434.0),
                        (266.0, 450.0),
                        (165.0, 398.0),
                        (235.0, 405.0),
                        (205.0, 414.0),
                        (165.0, 464.0),
                        (240.0, 464.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
                ),
                occluder_label_he="קפלי פורצלן לבנים",
            ),
            PortalCubbySpec(
                cubby_id="row2_col3",
                runtime_shelf_id="cubby_bene_beitcha",
                row_index=1,
                col_index=2,
                target_scene_id="bene_beitcha",
                label_he="🏛️ בנה ביתך · מוזיאון רמת גן",
                aperture_bounds_px=BoundingBoxPrompt(x0=656, y0=306, x1=806, y1=465),
                backplane_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=654, y0=304, x1=808, y1=467),
                    points_xy=(
                        (686.0, 338.0),
                        (772.0, 338.0),
                        (732.0, 375.0),
                        (782.0, 395.0),
                        (686.0, 438.0),
                        (758.0, 442.0),
                        (646.0, 380.0),
                        (816.0, 380.0),
                        (730.0, 296.0),
                    ),
                    point_labels=(1, 1, 1, 1, 0, 0, 0, 0, 0),
                ),
                occluder_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=656, y0=402, x1=796, y1=461),
                    points_xy=(
                        (686.0, 418.0),
                        (714.0, 416.0),
                        (682.0, 446.0),
                        (715.0, 448.0),
                        (758.0, 442.0),
                        (776.0, 426.0),
                        (685.0, 395.0),
                        (755.0, 398.0),
                        (735.0, 405.0),
                        (685.0, 464.0),
                        (755.0, 464.0),
                        (808.0, 435.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
                ),
                occluder_label_he="אבני חול וכורכר חמות",
            ),
            PortalCubbySpec(
                cubby_id="row3_col2",
                runtime_shelf_id="cubby_fried_house",
                row_index=2,
                col_index=1,
                target_scene_id="fried_house_railing",
                label_he="🪨 בית פריד · כרית ואבן על הרמפה",
                aperture_bounds_px=BoundingBoxPrompt(x0=266, y0=492, x1=413, y1=638),
                backplane_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=264, y0=490, x1=415, y1=640),
                    points_xy=(
                        (295.0, 518.0),
                        (385.0, 518.0),
                        (286.0, 568.0),
                        (396.0, 568.0),
                        (276.0, 622.0),
                        (404.0, 622.0),
                        (342.0, 572.0),
                        (345.0, 618.0),
                        (256.0, 565.0),
                        (422.0, 565.0),
                        (340.0, 482.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
                ),
                occluder_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=274, y0=545, x1=407, y1=640),
                    points_xy=(
                        (336.0, 565.0),
                        (358.0, 582.0),
                        (308.0, 616.0),
                        (348.0, 618.0),
                        (386.0, 618.0),
                        (292.0, 545.0),
                        (388.0, 545.0),
                        (278.0, 588.0),
                        (406.0, 585.0),
                        (340.0, 646.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
                ),
                occluder_label_he="אבן זהובה מונחת על חלוק אבן",
            ),
            PortalCubbySpec(
                cubby_id="row3_col3",
                runtime_shelf_id="cubby_denya_model",
                row_index=2,
                col_index=2,
                target_scene_id="denya_model",
                label_he="🏠 דגם בית דניה · קרטון ועץ",
                aperture_bounds_px=BoundingBoxPrompt(x0=529, y0=491, x1=681, y1=638),
                backplane_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=527, y0=489, x1=683, y1=640),
                    points_xy=(
                        (565.0, 508.0),
                        (645.0, 508.0),
                        (605.0, 542.0),
                        (565.0, 580.0),
                        (645.0, 580.0),
                        (605.0, 618.0),
                        (518.0, 565.0),
                        (692.0, 565.0),
                        (605.0, 480.0),
                        (605.0, 646.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
                ),
                occluder_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=527, y0=489, x1=683, y1=640),
                    points_xy=(
                        (565.0, 508.0),
                        (645.0, 508.0),
                        (605.0, 542.0),
                        (565.0, 580.0),
                        (645.0, 580.0),
                        (605.0, 618.0),
                        (550.0, 624.0),
                        (660.0, 624.0),
                        (518.0, 565.0),
                        (692.0, 565.0),
                        (605.0, 480.0),
                        (605.0, 646.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
                ),
                occluder_label_he="שמיכת לבד אפורה מקופלת",
            ),
            PortalCubbySpec(
                cubby_id="row4_col3",
                runtime_shelf_id="cubby_bar_giora",
                row_index=3,
                col_index=2,
                target_scene_id="bar_giora_bedroom",
                label_he="🕳️ חדר שינה בבר גיורא · ויטרינה ומנורת מפרש",
                aperture_bounds_px=BoundingBoxPrompt(x0=661, y0=763, x1=808, y1=905),
                backplane_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=659, y0=761, x1=810, y1=907),
                    points_xy=(
                        (692.0, 792.0),
                        (776.0, 792.0),
                        (735.0, 834.0),
                        (692.0, 878.0),
                        (776.0, 878.0),
                        (648.0, 834.0),
                        (820.0, 834.0),
                        (735.0, 752.0),
                        (735.0, 914.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 0, 0, 0, 0),
                ),
            ),
            PortalCubbySpec(
                cubby_id="row7_col1",
                runtime_shelf_id="cubby_paris_table",
                row_index=6,
                col_index=0,
                target_scene_id="paris_cite_white_table",
                label_he="🕳️ פאריס · שולחן לבן בסיטה",
                aperture_bounds_px=BoundingBoxPrompt(x0=268, y0=1286, x1=415, y1=1452),
                backplane_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=268, y0=1286, x1=415, y1=1452),
                    points_xy=((340.0, 1370.0),),
                    point_labels=(1,),
                ),
            ),
            PortalCubbySpec(
                cubby_id="row7_col0",
                runtime_shelf_id="cubby_tzofen",
                row_index=6,
                col_index=0,
                target_scene_id="tzofen_brick_table",
                label_he="🧱 ׳צופן׳ · שולחן הברזל ולבני הבוץ",
                aperture_bounds_px=BoundingBoxPrompt(x0=11, y0=1289, x1=163, y1=1451),
                backplane_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=11, y0=1289, x1=163, y1=1451),
                    points_xy=((87.0, 1370.0),),
                    point_labels=(1,),
                ),
            ),
            PortalCubbySpec(
                cubby_id="row5_green_center",
                runtime_shelf_id="cubby_plan_b_v3",
                row_index=4,
                col_index=1,
                target_scene_id="plan_b_pipe_v3",
                label_he="🕳️ Plan B · תמי ויערה באור הדיפוזי",
                aperture_bounds_px=BoundingBoxPrompt(x0=450, y0=938, x1=759, y1=1116),
                backplane_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=448, y0=936, x1=761, y1=1118),
                    points_xy=(
                        (488.0, 968.0),
                        (605.0, 968.0),
                        (722.0, 968.0),
                        (475.0, 1045.0),
                        (732.0, 1045.0),
                        (472.0, 1095.0),
                        (732.0, 1095.0),
                        (595.0, 1040.0),
                        (600.0, 1085.0),
                        (438.0, 1015.0),
                        (772.0, 1015.0),
                        (605.0, 926.0),
                        (605.0, 1135.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
                ),
                occluder_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=486, y0=1018, x1=706, y1=1122),
                    points_xy=(
                        (595.0, 1038.0),
                        (578.0, 1046.0),
                        (614.0, 1046.0),
                        (525.0, 1068.0),
                        (600.0, 1068.0),
                        (675.0, 1068.0),
                        (525.0, 1098.0),
                        (600.0, 1102.0),
                        (675.0, 1098.0),
                        (525.0, 1005.0),
                        (675.0, 1005.0),
                        (472.0, 1075.0),
                        (728.0, 1075.0),
                        (550.0, 1132.0),
                        (650.0, 1132.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
                ),
                occluder_label_he="פרח לוטוס ורוד ואריג מפוספס",
            ),
            PortalCubbySpec(
                cubby_id="row7_green_center",
                runtime_shelf_id="cubby_marseille",
                row_index=6,
                col_index=1,
                target_scene_id="marseille_mirror",
                label_he="🪞 מרסיי · מתחת לתקרת המראה",
                aperture_bounds_px=BoundingBoxPrompt(x0=450, y0=1266, x1=759, y1=1452),
                backplane_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=448, y0=1264, x1=761, y1=1454),
                    points_xy=(
                        (478.0, 1295.0),
                        (725.0, 1295.0),
                        (478.0, 1365.0),
                        (725.0, 1365.0),
                        (478.0, 1432.0),
                        (725.0, 1432.0),
                        (598.0, 1300.0),
                        (598.0, 1375.0),
                        (598.0, 1435.0),
                        (436.0, 1360.0),
                        (772.0, 1360.0),
                        (600.0, 1250.0),
                        (600.0, 1462.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
                ),
                occluder_sam_prompt=SamPointGroupPrompt(
                    box=BoundingBoxPrompt(x0=502, y0=1266, x1=694, y1=1455),
                    points_xy=(
                        (578.0, 1280.0),
                        (532.0, 1300.0),
                        (625.0, 1306.0),
                        (662.0, 1308.0),
                        (545.0, 1335.0),
                        (645.0, 1335.0),
                        (545.0, 1382.0),
                        (645.0, 1382.0),
                        (545.0, 1425.0),
                        (645.0, 1425.0),
                        (598.0, 1444.0),
                        (478.0, 1315.0),
                        (725.0, 1315.0),
                        (478.0, 1405.0),
                        (725.0, 1405.0),
                        (598.0, 1252.0),
                        (598.0, 1462.0),
                    ),
                    point_labels=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
                ),
                occluder_label_he="כלי טרקוטה ולוחות בטון על לבד אדום",
            ),
        ),
        bg_dark_felt_gain=0.54,
        bg_lama_dilation_px=26,
        bg_lama_overlap_outer_pad_px=22,
        bg_lama_blur_sigma=10.0,
        bg_lama_color_match_sigma=36.0,
        layer_depth_min_u8=46,
        layer_depth_collar_sigma=5.0,
        scene_bg_z=-350.0,
        scene_mid_z=-296.0,
        scene_fg_z=-280.0,
        room_depth_amplitude=358.0,
        initial_camera_y=-36.0,
        initial_camera_z=1740.0,
        min_camera_z=-260.0,
        max_camera_z=2040.0,
    )

    return NewScenesBundleConfig(
        scenes_output_relpath="prototype/assets/scenes",
        drawings_output_relpath="prototype/assets/drawings",
        spec_output_abspath=(
            "/Users/talsheaffer/.gemini/antigravity/brain/"
            "235713ae-3cfb-48f0-94e4-2012c0824fd5/scratch/subagent3_new_scenes_spec.json"
        ),
        dimensions=dims,
        inner_dimensions=CanvasDimensionsConfig(width=1500, height=1000),
        expanded_bg_scale=1.80,
        lama_max_working_side_px=1350,
        lama_roi_context_margin_px=192,
        depth_model=depth_cfg,
        segmentation=seg_cfg,
        inpainting=inpaint_cfg,
        graphite_drawing=draw_cfg,
        wandering_drawing_1=wandering1,
        denya_model=denya_spec,
        shelf_portal=shelf_spec,
    )


# ============================================================================
# Shared Vision & Geometry Helpers
# ============================================================================


def _crop_norm_rgb(img: Image.Image, crop_norm: tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    x0 = max(0, min(w - 1, int(round(crop_norm[0] * w))))
    y0 = max(0, min(h - 1, int(round(crop_norm[1] * h))))
    x1 = max(x0 + 1, min(w, int(round(crop_norm[2] * w))))
    y1 = max(y0 + 1, min(h, int(round(crop_norm[3] * h))))
    return img.convert("RGB").crop((x0, y0, x1, y1))


def _estimate_smooth_paper_rgb(
    rgb_f32: np.ndarray,
    downsample_divisor: int,
    percentile: float,
    filter_size: int,
) -> np.ndarray:
    h, w, _ = rgb_f32.shape
    ds_w = max(16, w // max(1, downsample_divisor))
    ds_h = max(16, h // max(1, downsample_divisor))
    small = cv2.resize(rgb_f32, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
    return np.stack(
        [
            cv2.resize(
                ndi.percentile_filter(small[:, :, ch], percentile=percentile, size=filter_size),
                (w, h),
                interpolation=cv2.INTER_CUBIC,
            )
            for ch in range(3)
        ],
        axis=2,
    )


def _cosine_feather_2d(
    height: int,
    width: int,
    left_px: int,
    top_px: int,
    right_px: int,
    bottom_px: int,
    elliptical: bool,
) -> np.ndarray:
    if elliptical:
        yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        rad = np.hypot(xx, yy)
        avg_f = max(0.08, (left_px + right_px) / max(1.0, float(width)))
        inner_r = max(0.25, 1.0 - avg_f)
        t = np.clip((rad - inner_r) / max(0.08, 1.0 - inner_r), 0.0, 1.0)
        return (0.5 + 0.5 * np.cos(np.pi * t)).astype(np.float32)

    wx = np.ones(width, dtype=np.float32)
    wy = np.ones(height, dtype=np.float32)
    if left_px > 1:
        span = min(width, left_px)
        wx[:span] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, span, dtype=np.float32))
    if right_px > 1:
        span = min(width, right_px)
        wx[-span:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, span, dtype=np.float32))
    if top_px > 1:
        span = min(height, top_px)
        wy[:span] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, span, dtype=np.float32))
    if bottom_px > 1:
        span = min(height, bottom_px)
        wy[-span:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, span, dtype=np.float32))
    return np.outer(wy, wx).astype(np.float32)


def _compute_alpha_supported_depth_u8(
    layer_rgba: Image.Image,
    neutral_backdrop_rgb: tuple[float, float, float],
    depth_estimator: DepthAnythingEstimator,
    min_fg_depth_u8: int,
    collar_sigma: float,
) -> Image.Image:
    """
    Predicts real neural Depth Anything V2 3D surface relief for an RGBA layer (`0=far, 255=near`),
    normalizing foreground depth into `[min_fg_depth_u8, 255]` on opaque/semi-transparent support
    with a smooth collar transition to `0` across empty background space.
    """
    rgba_np = np.asarray(layer_rgba.convert("RGBA"), dtype=np.float32)
    rgb = rgba_np[:, :, :3]
    alpha = rgba_np[:, :, 3] / 255.0
    if not np.any(alpha > 0.02):
        h, w = alpha.shape
        return Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L")

    bg_arr = np.array(neutral_backdrop_rgb, dtype=np.float32)[None, None, :]
    comp_rgb = np.clip(
        np.round(alpha[:, :, None] * rgb + (1.0 - alpha[:, :, None]) * bg_arr), 0, 255
    ).astype(np.uint8)
    raw_depth = depth_estimator.predict_depth(Image.fromarray(comp_rgb, mode="RGB"))

    fg_vals = raw_depth[alpha > 0.08]
    p_lo = float(np.percentile(fg_vals, 3.0))
    p_hi = float(np.percentile(fg_vals, 99.0))
    norm_fg = np.clip((raw_depth - p_lo) / max(1e-5, p_hi - p_lo), 0.0, 1.0)
    scaled_fg = float(min_fg_depth_u8) + norm_fg * (255.0 - float(min_fg_depth_u8))

    soft_support = np.clip(
        np.maximum(alpha, ndi.gaussian_filter(alpha, sigma=collar_sigma)), 0.0, 1.0
    )
    depth_u8 = np.clip(np.round(scaled_fg * soft_support), 0, 255).astype(np.uint8)
    return Image.fromarray(depth_u8, mode="L")


# ============================================================================
# Scene A Builder: `wandering_drawing_1` ("שיטוט ברישום 1")
# ============================================================================


def _build_archival_paper_ground(
    project_root: Path,
    dims: CanvasDimensionsConfig,
    spec: WanderingDrawing1Spec,
) -> np.ndarray:
    grain_src = Image.open(project_root / spec.paper_grain_source_relpath).convert("RGB")
    grain_crop = _crop_norm_rgb(grain_src, spec.paper_grain_crop_norm).resize(
        (dims.width, dims.height), Image.Resampling.LANCZOS
    )
    grain_f = np.asarray(grain_crop, dtype=np.float32)
    low_freq = ndi.gaussian_filter(grain_f, sigma=(18.0, 18.0, 0.0))
    high_freq = grain_f - low_freq
    std_val = max(1.0, float(np.std(high_freq)))
    norm_grain = (high_freq / std_val) * spec.paper_grain_amplitude

    yy = np.linspace(-1.0, 1.0, dims.height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, dims.width, dtype=np.float32)[None, :]
    vignette = 1.0 - spec.paper_vignette_strength * np.clip(0.55 * xx * xx + 0.65 * yy * yy, 0.0, 1.4)
    base = np.array(spec.paper_base_rgb, dtype=np.float32)[None, None, :] * vignette[:, :, None]
    return np.clip(base + norm_grain, 0.0, 255.0)


def _fuse_paper_transmittance_zone(
    canvas_f32: np.ndarray,
    project_root: Path,
    zone: PaperTransmittanceZoneSpec,
    spec: WanderingDrawing1Spec,
) -> np.ndarray:
    src_img = Image.open(project_root / zone.source_relpath).convert("RGB")
    x0, y0, x1, y1 = zone.dest_box_xyxy
    zw, zh = x1 - x0, y1 - y0
    cropped = _crop_norm_rgb(src_img, zone.src_crop_norm).resize(
        (zw, zh), Image.Resampling.LANCZOS
    )
    patch_f = np.asarray(cropped, dtype=np.float32)
    paper_env = _estimate_smooth_paper_rgb(
        patch_f,
        downsample_divisor=spec.paper_downsample_divisor,
        percentile=zone.paper_percentile,
        filter_size=spec.paper_filter_size,
    )
    transmittance = np.clip(patch_f / np.maximum(paper_env, 32.0), 0.0, 1.03)
    weight = (
        _cosine_feather_2d(
            height=zh,
            width=zw,
            left_px=zone.feather_left_px,
            top_px=zone.feather_top_px,
            right_px=zone.feather_right_px,
            bottom_px=zone.feather_bottom_px,
            elliptical=zone.elliptical_falloff,
        )[:, :, None]
        * zone.transmittance_strength
    )
    eff_trans = 1.0 - weight * (1.0 - np.minimum(transmittance, 1.0))
    out = canvas_f32.copy()
    out[y0:y1, x0:x1] = np.clip(out[y0:y1, x0:x1] * eff_trans, 0.0, 255.0)
    return out


def _inpaint_masked_plate_lama(
    src_rgb: Image.Image,
    binary_mask_u8: np.ndarray,
    lama_inpainter: LamaOcclusionInpainter,
    dilation_px: int,
    blur_sigma: float,
    max_working_side_px: int,
    roi_context_margin_px: int,
) -> Image.Image:
    """
    Executes real neural LaMa Fast Fourier Convolution inpainting while keeping unmasked
    high-resolution (`2700x1800`) pixels 100% native crisp:
      - If the dilated mask bounding box + context margin fits inside `max_working_side_px`,
        inpaints that native-resolution 1:1 ROI directly.
      - Otherwise (e.g. sparse seam knots across a `2700x1800` plate), executes neural LaMa at
        `max_working_side_px` and alpha-composites strictly inside the soft-dilated mask.
    """
    ys, xs = np.where(binary_mask_u8 > 0)
    if xs.size == 0:
        return src_rgb

    w, h = src_rgb.size
    src_np = np.asarray(src_rgb.convert("RGB"), dtype=np.float32)

    k_rad = max(1, int(dilation_px))
    dilated_u8 = cv2.dilate(
        (binary_mask_u8 > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_rad + 1, 2 * k_rad + 1)),
    )
    feather_sigma = max(1.0, float(blur_sigma))
    blend_alpha = np.clip(
        ndi.gaussian_filter(dilated_u8.astype(np.float32) / 255.0, sigma=feather_sigma) * 1.18,
        0.0,
        1.0,
    )[:, :, None]

    rx0 = max(0, int(xs.min()) - roi_context_margin_px)
    ry0 = max(0, int(ys.min()) - roi_context_margin_px)
    rx1 = min(w, int(xs.max()) + 1 + roi_context_margin_px)
    ry1 = min(h, int(ys.max()) + 1 + roi_context_margin_px)
    rw, rh = rx1 - rx0, ry1 - ry0

    if max(rw, rh) <= max_working_side_px:
        roi_rgb = src_rgb.crop((rx0, ry0, rx1, ry1))
        roi_mask = binary_mask_u8[ry0:ry1, rx0:rx1]
        healed_roi = np.asarray(
            lama_inpainter.inpaint_background(
                roi_rgb,
                roi_mask,
                dilation_radius_px=dilation_px,
                mask_blur_sigma=blur_sigma,
            ),
            dtype=np.float32,
        )
        out_np = src_np.copy()
        roi_a = blend_alpha[ry0:ry1, rx0:rx1]
        out_np[ry0:ry1, rx0:rx1] = roi_a * healed_roi + (1.0 - roi_a) * src_np[ry0:ry1, rx0:rx1]
        return Image.fromarray(np.clip(np.round(out_np), 0, 255).astype(np.uint8), mode="RGB")

    scale = min(1.0, float(max_working_side_px) / float(max(w, h)))
    work_w = max(64, int(round(w * scale)))
    work_h = max(64, int(round(h * scale)))
    work_rgb = src_rgb.resize((work_w, work_h), Image.Resampling.LANCZOS)
    work_mask = cv2.resize(binary_mask_u8, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
    work_dil = max(1, int(round(dilation_px * scale)))
    work_blur = max(0.75, float(blur_sigma) * scale)
    healed_work = lama_inpainter.inpaint_background(
        work_rgb,
        work_mask,
        dilation_radius_px=work_dil,
        mask_blur_sigma=work_blur,
    )
    healed_up = np.asarray(
        healed_work.resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32
    )
    out_np = blend_alpha * healed_up + (1.0 - blend_alpha) * src_np
    return Image.fromarray(np.clip(np.round(out_np), 0, 255).astype(np.uint8), mode="RGB")


def _heal_boxes_with_lama(
    src_rgb: Image.Image,
    boxes: Sequence[BoundingBoxPrompt],
    lama_inpainter: LamaOcclusionInpainter,
    dilation_px: int,
    blur_sigma: float,
    max_working_side_px: int = 1350,
    roi_context_margin_px: int = 192,
) -> Image.Image:
    if not boxes:
        return src_rgb
    w, h = src_rgb.size
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        mask[max(0, b.y0) : min(h, b.y1), max(0, b.x0) : min(w, b.x1)] = 255
    return _inpaint_masked_plate_lama(
        src_rgb=src_rgb,
        binary_mask_u8=mask,
        lama_inpainter=lama_inpainter,
        dilation_px=dilation_px,
        blur_sigma=blur_sigma,
        max_working_side_px=max_working_side_px,
        roi_context_margin_px=roi_context_margin_px,
    )


def _estimate_global_wall_illumination_rgb(
    rgb_f32: np.ndarray,
    wall_percentile: float,
    blur_sigma: float,
) -> np.ndarray:
    """
    Estimates smooth gallery-wall illumination solely from bright background pixels
    (`lum >= percentile`), preventing wide dark cut-paper ribbons from pulling down the
    estimated background envelope and bleaching their interiors.
    """
    lum = 0.299 * rgb_f32[:, :, 0] + 0.587 * rgb_f32[:, :, 1] + 0.114 * rgb_f32[:, :, 2]
    thresh = float(np.percentile(lum, max(55.0, wall_percentile - 14.0)))
    wall_weight = (lum >= thresh).astype(np.float32)
    w_blur = np.maximum(1e-4, ndi.gaussian_filter(wall_weight, sigma=blur_sigma))
    return np.stack(
        [
            ndi.gaussian_filter(rgb_f32[:, :, ch] * wall_weight, sigma=blur_sigma) / w_blur
            for ch in range(3)
        ],
        axis=2,
    )


def _extract_pigment_or_wall_island_rgba(
    src_rgb: Image.Image,
    isl: WanderingLayerIslandSpec,
    spec: WanderingDrawing1Spec,
) -> Image.Image:
    cropped = _crop_norm_rgb(src_rgb, isl.src_crop_norm)
    rgb_f = np.asarray(cropped, dtype=np.float32)
    ch, cw, _ = rgb_f.shape

    if isl.extraction_method == "wall_cutout_unmix":
        paper_env = _estimate_global_wall_illumination_rgb(
            rgb_f,
            wall_percentile=isl.paper_percentile,
            blur_sigma=max(48.0, float(min(cw, ch)) * 0.12),
        )
    else:
        paper_env = _estimate_smooth_paper_rgb(
            rgb_f,
            downsample_divisor=spec.paper_downsample_divisor,
            percentile=isl.paper_percentile,
            filter_size=spec.paper_filter_size,
        )

    lum = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    paper_lum = (
        0.299 * paper_env[:, :, 0] + 0.587 * paper_env[:, :, 1] + 0.114 * paper_env[:, :, 2]
    )
    lum_delta = np.maximum(0.0, paper_lum - lum)

    vec_dist = np.linalg.norm(rgb_f - paper_env, axis=2)
    yellow_warmth = np.maximum(
        0.0,
        (rgb_f[:, :, 0] + rgb_f[:, :, 1]) * 0.5
        - rgb_f[:, :, 2]
        - ((paper_env[:, :, 0] + paper_env[:, :, 1]) * 0.5 - paper_env[:, :, 2]),
    )
    pigment_signal = np.maximum(
        lum_delta, isl.chroma_weight * np.maximum(vec_dist * 0.65, yellow_warmth)
    )

    span = max(8.0, isl.lum_delta_full - isl.lum_delta_floor)
    norm_sig = np.clip((pigment_signal - isl.lum_delta_floor) / span, 0.0, 1.0)
    alpha_raw = np.power(norm_sig, isl.alpha_gamma)

    if isl.max_interior_hole_fill_px > 0:
        # Preserve crisp white highway lane dashes inside opaque asphalt road ribbons while keeping large wire loops open
        solid_ribbon = (alpha_raw > 0.35).astype(np.uint8)
        inv = (solid_ribbon == 0).astype(np.uint8)
        nh, h_lbls, h_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        dash_fill = np.zeros_like(solid_ribbon, dtype=bool)
        for hid in range(1, nh):
            hx = int(h_stats[hid, cv2.CC_STAT_LEFT])
            hy = int(h_stats[hid, cv2.CC_STAT_TOP])
            hw = int(h_stats[hid, cv2.CC_STAT_WIDTH])
            hh = int(h_stats[hid, cv2.CC_STAT_HEIGHT])
            touches = hx == 0 or hy == 0 or (hx + hw >= cw) or (hy + hh >= ch)
            if not touches and int(h_stats[hid, cv2.CC_STAT_AREA]) < isl.max_interior_hole_fill_px:
                dash_fill |= h_lbls == hid
        alpha_raw = np.where(dash_fill, 1.0, alpha_raw)

    fg_bin = (alpha_raw > 0.07).astype(np.uint8)
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=8)
    keep = np.zeros_like(fg_bin, dtype=bool)
    for cid in range(1, n_lbl):
        if int(stats[cid, cv2.CC_STAT_AREA]) >= isl.min_component_area_px:
            keep |= lbls == cid
    alpha_clean = np.where(keep, alpha_raw, 0.0)

    taper = _cosine_feather_2d(
        height=ch,
        width=cw,
        left_px=isl.perimeter_taper_px,
        top_px=isl.perimeter_taper_px,
        right_px=isl.perimeter_taper_px,
        bottom_px=isl.bottom_taper_px,
        elliptical=False,
    )
    alpha_final = np.clip(alpha_clean * taper, 0.0, 1.0)

    a_safe = np.maximum(alpha_final[:, :, None], spec.unmix_alpha_floor)
    unmixed_rgb = np.where(
        (alpha_final[:, :, None] > 0.02) & (alpha_final[:, :, None] < 0.94),
        np.clip((rgb_f - (1.0 - a_safe) * paper_env) / a_safe, 0.0, 255.0),
        rgb_f,
    )
    rgba = np.dstack(
        [
            np.round(unmixed_rgb).astype(np.uint8),
            np.round(alpha_final * 255.0).astype(np.uint8),
        ]
    )
    return Image.fromarray(rgba, mode="RGBA").resize(isl.dest_wh, Image.Resampling.LANCZOS)


def _extract_sam_scroll_sculpture_rgba(
    src_rgb: Image.Image,
    isl: WanderingLayerIslandSpec,
    sam_extractor: SamBoxMaskExtractor,
) -> Image.Image:
    cropped = _crop_norm_rgb(src_rgb, isl.src_crop_norm)
    target_h = max(64, int(round(800 * cropped.height / max(1, cropped.width))))
    work_rgb = cropped.resize((800, target_h), Image.Resampling.LANCZOS)
    alpha_u8 = sam_extractor.segment_point_groups(
        rgb_image=work_rgb,
        point_groups=isl.sam_prompts,
        exclude_boxes=(),
        max_pinhole_fill_px=max(2400, isl.max_interior_hole_fill_px),
        min_major_component_px=isl.min_component_area_px,
        right_edge_fade_x_start=None,
    )
    # Keep strictly the primary continuous scroll sculpture component
    fg_bin = (alpha_u8 >= 96).astype(np.uint8)
    nl, lbls, stats, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=8)
    if nl > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        support = cv2.dilate(
            (lbls == largest).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        alpha_u8 = np.where(support > 0, alpha_u8, 0).astype(np.uint8)

    rgb_u8 = decontaminate_matte_edge_rgb(np.asarray(work_rgb, dtype=np.uint8), alpha_u8)
    taper = _cosine_feather_2d(
        height=target_h,
        width=800,
        left_px=isl.perimeter_taper_px,
        top_px=isl.perimeter_taper_px,
        right_px=isl.perimeter_taper_px,
        bottom_px=isl.bottom_taper_px,
        elliptical=False,
    )
    alpha_tapered = np.clip(np.round(alpha_u8.astype(np.float32) * taper), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb_u8, alpha_tapered]), mode="RGBA").resize(
        isl.dest_wh, Image.Resampling.LANCZOS
    )


def _prune_tiny_alpha_specks_rgba(layer_rgba: Image.Image, min_area_px: int, alpha_thresh_u8: int = 20) -> Image.Image:
    """
    Removes isolated sub-threshold alpha specks so every pixel with `alpha >= alpha_thresh_u8`
    belongs to a genuine coherent component, guaranteeing clean outward EDT rim continuity.
    """
    arr = np.asarray(layer_rgba.convert("RGBA"), dtype=np.uint8).copy()
    fg_bin = (arr[:, :, 3] >= alpha_thresh_u8).astype(np.uint8)
    nl, lbls, stats, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=8)
    keep = np.zeros_like(fg_bin, dtype=bool)
    for cid in range(1, nl):
        if int(stats[cid, cv2.CC_STAT_AREA]) >= min_area_px:
            keep |= lbls == cid
    support = cv2.dilate(
        keep.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    arr[:, :, 3] = np.where(support > 0, arr[:, :, 3], 0).astype(np.uint8)
    arr[arr[:, :, 3] == 0, :3] = 0
    return Image.fromarray(arr, mode="RGBA")


def _compute_edt_extrapolated_layer_depth_u8(
    layer_rgba: Image.Image,
    neutral_backdrop_rgb: tuple[float, float, float],
    depth_estimator: DepthAnythingEstimator,
    depth_min_norm: float,
    depth_max_norm: float,
    alpha_threshold_u8: int = 20,
) -> Image.Image:
    """
    Computes genuine neural Depth Anything V2 sculptural relief on opaque pixels (`alpha >= alpha_threshold_u8`)
    normalized to `[depth_min_norm, depth_max_norm]` and extrapolates outward across all transparent
    pixels via Euclidean Distance Transform (`extrapolate_depth_outward_edt`, Law 2 $C^{-1}$ silhouette).
    """
    rgba_np = np.asarray(layer_rgba.convert("RGBA"), dtype=np.uint8)
    alpha_u8 = rgba_np[:, :, 3]
    opaque = alpha_u8 >= alpha_threshold_u8
    if not np.any(opaque):
        return Image.fromarray(np.zeros(alpha_u8.shape, dtype=np.uint8), mode="L")

    alpha_f = alpha_u8.astype(np.float32) / 255.0
    bg_arr = np.array(neutral_backdrop_rgb, dtype=np.float32)[None, None, :]
    comp_rgb = np.clip(
        np.round(alpha_f[:, :, None] * rgba_np[:, :, :3].astype(np.float32) + (1.0 - alpha_f[:, :, None]) * bg_arr),
        0,
        255,
    ).astype(np.uint8)
    raw_depth = depth_estimator.predict_depth(Image.fromarray(comp_rgb, mode="RGB"))
    fg_vals = raw_depth[opaque]
    p_lo = float(np.percentile(fg_vals, 3.0))
    p_hi = float(np.percentile(fg_vals, 99.0))
    norm_fg = np.clip((raw_depth - p_lo) / max(1e-5, p_hi - p_lo), 0.0, 1.0)
    interior_d = (float(depth_min_norm) + norm_fg * float(depth_max_norm - depth_min_norm)).astype(np.float32)
    interior_d = ndi.gaussian_filter(interior_d, sigma=1.1).astype(np.float32)
    extrap_d = extrapolate_depth_outward_edt(interior_d, opaque)
    depth_u8 = np.clip(np.rint(extrap_d * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(depth_u8, mode="L")


def _extract_scroll_paper_and_foot_patches(
    src_rgb: Image.Image,
    isl: WanderingLayerIslandSpec,
    portal_spec: WanderingScrollPortalSpec,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> tuple[Image.Image, Image.Image]:
    """
    Separates `IMG_MUSE_0256` into:
      1. `healed_scroll_patch_rgba`: the paper scroll with the 3D ceramic foot + wrapped pole
         neural-LaMa healed on the underlying paper so zero gap/duplicate pole appears under parallax.
      2. `foot_patch_rgba`: the 3D sculpted ceramic foot + metallic ankle + dark wrapped pole for
         mounting on a dedicated forward `extraLayer`.
    """
    cropped = _crop_norm_rgb(src_rgb, isl.src_crop_norm)
    target_h = max(64, int(round(800 * cropped.height / max(1, cropped.width))))
    work_rgb = cropped.resize((800, target_h), Image.Resampling.LANCZOS)

    scroll_alpha_u8 = sam_extractor.segment_point_groups(
        rgb_image=work_rgb,
        point_groups=isl.sam_prompts,
        exclude_boxes=(),
        max_pinhole_fill_px=max(2400, isl.max_interior_hole_fill_px),
        min_major_component_px=isl.min_component_area_px,
        right_edge_fade_x_start=None,
    )
    fg_bin = (scroll_alpha_u8 >= 96).astype(np.uint8)
    nl, lbls, stats, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=8)
    if nl > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        support = cv2.dilate(
            (lbls == largest).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        scroll_alpha_u8 = np.where(support > 0, scroll_alpha_u8, 0).astype(np.uint8)

    foot_alpha_u8 = sam_extractor.segment_point_groups(
        rgb_image=work_rgb,
        point_groups=portal_spec.foot_sam_prompts,
        exclude_boxes=(),
        max_pinhole_fill_px=1800,
        min_major_component_px=portal_spec.foot_min_component_area_px,
        right_edge_fade_x_start=None,
    )
    # Constrain foot_alpha strictly inside the SAM bounding box & keep only the primary connected 3D pole+foot
    if portal_spec.foot_sam_prompts and portal_spec.foot_sam_prompts[0].box is not None:
        b = portal_spec.foot_sam_prompts[0].box
        box_mask = np.zeros_like(foot_alpha_u8)
        box_mask[max(0, b.y0) : min(target_h, b.y1), max(0, b.x0) : min(800, b.x1)] = 255
        foot_alpha_u8 = np.minimum(foot_alpha_u8, box_mask)

    nl_f, lbl_f, stats_f, _ = cv2.connectedComponentsWithStats(
        (foot_alpha_u8 >= 64).astype(np.uint8), connectivity=8
    )
    if nl_f > 1:
        largest_foot = 1 + int(np.argmax(stats_f[1:, cv2.CC_STAT_AREA]))
        foot_support = cv2.dilate(
            (lbl_f == largest_foot).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        foot_alpha_u8 = np.where(foot_support > 0, foot_alpha_u8, 0).astype(np.uint8)

    taper = _cosine_feather_2d(
        height=target_h,
        width=800,
        left_px=isl.perimeter_taper_px,
        top_px=isl.perimeter_taper_px,
        right_px=isl.perimeter_taper_px,
        bottom_px=isl.bottom_taper_px,
        elliptical=False,
    )

    foot_rgb_u8 = decontaminate_matte_edge_rgb(np.asarray(work_rgb, dtype=np.uint8), foot_alpha_u8)
    foot_alpha_tapered = np.clip(
        np.round(foot_alpha_u8.astype(np.float32) * taper), 0, 255
    ).astype(np.uint8)
    foot_patch_rgba = Image.fromarray(
        np.dstack([foot_rgb_u8, foot_alpha_tapered]), mode="RGBA"
    ).resize(isl.dest_wh, Image.Resampling.LANCZOS)

    # Heal the 3D foot & wrapped pole strip on the underlying scroll paper via neural LaMa
    foot_bin_u8 = np.where(foot_alpha_u8 >= 32, 255, 0).astype(np.uint8)
    healed_work_rgb = _inpaint_masked_plate_lama(
        src_rgb=work_rgb,
        binary_mask_u8=foot_bin_u8,
        lama_inpainter=lama_inpainter,
        dilation_px=portal_spec.foot_heal_dilation_px,
        blur_sigma=portal_spec.foot_heal_blur_sigma,
        max_working_side_px=1350,
        roi_context_margin_px=140,
    )
    # Subtract any foot protrusion that stuck outside the scroll paper silhouette
    scroll_eroded = cv2.erode(
        (scroll_alpha_u8 >= 128).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    outside_paper_foot = (foot_alpha_u8 >= 128) & (scroll_eroded == 0)
    healed_scroll_alpha = np.where(outside_paper_foot, 0, scroll_alpha_u8).astype(np.uint8)
    healed_scroll_rgb = decontaminate_matte_edge_rgb(
        np.asarray(healed_work_rgb, dtype=np.uint8), healed_scroll_alpha
    )
    healed_scroll_alpha_tapered = np.clip(
        np.round(healed_scroll_alpha.astype(np.float32) * taper), 0, 255
    ).astype(np.uint8)
    healed_scroll_patch_rgba = Image.fromarray(
        np.dstack([healed_scroll_rgb, healed_scroll_alpha_tapered]), mode="RGBA"
    ).resize(isl.dest_wh, Image.Resampling.LANCZOS)

    return healed_scroll_patch_rgba, foot_patch_rgba


def _partition_scroll_veil_rgba(
    mid_healed_rgba: Image.Image,
    inner_box_px: tuple[int, int, int, int],
    feather_px: int,
) -> tuple[Image.Image, Image.Image]:
    """
    Exact quantized straight-alpha `over` partition of `mid_healed_rgba` across `inner_box_px`:
      - `alpha_composite(mid_cut_rgba, scroll_veil_rgba)` reconstructs `mid_healed_rgba` seamlessly at rest.
      - When `scroll_veil_rgba` dissolves (`1 -> 0`), `mid_cut_rgba` opens a soft-feathered window
        strictly inside `inner_box_px` (`alpha == 0` across the core aperture).
    """
    orig_u8 = np.asarray(mid_healed_rgba.convert("RGBA"), dtype=np.uint8)
    h, w, _ = orig_u8.shape
    x0, y0, x1, y1 = inner_box_px
    bw, bh = x1 - x0, y1 - y0
    if bw <= 2 * feather_px or bh <= 2 * feather_px:
        raise ValueError(f"Invalid scroll portal inner_box_px={inner_box_px} for feather_px={feather_px}.")

    w_veil = np.zeros((h, w), dtype=np.float32)
    w_veil[y0:y1, x0:x1] = _cosine_feather_2d(
        height=bh,
        width=bw,
        left_px=feather_px,
        top_px=feather_px,
        right_px=feather_px,
        bottom_px=feather_px,
        elliptical=False,
    )

    a_orig_u8 = orig_u8[:, :, 3]
    a_orig_f = a_orig_u8.astype(np.float32)
    rgb_orig_u8 = orig_u8[:, :, :3]

    veil_a_u8 = np.minimum(
        a_orig_u8,
        np.clip(np.rint(w_veil * a_orig_f), 0, 255).astype(np.uint8),
    )
    veil_a_f = veil_a_u8.astype(np.float32)
    mid_a_u8 = np.where(
        (veil_a_u8 >= a_orig_u8) | (veil_a_u8 == 255),
        0,
        np.clip(
            np.rint(255.0 * (a_orig_f - veil_a_f) / np.maximum(1.0, 255.0 - veil_a_f)),
            0,
            255,
        ),
    ).astype(np.uint8)

    mid_u8 = np.zeros((h, w, 4), dtype=np.uint8)
    mid_u8[:, :, 3] = mid_a_u8
    mid_nz = mid_a_u8 > 0
    mid_u8[mid_nz, :3] = rgb_orig_u8[mid_nz]

    veil_u8 = np.zeros((h, w, 4), dtype=np.uint8)
    veil_u8[:, :, 3] = veil_a_u8
    veil_nz = veil_a_u8 > 0
    veil_u8[veil_nz, :3] = rgb_orig_u8[veil_nz]

    return Image.fromarray(mid_u8, mode="RGBA"), Image.fromarray(veil_u8, mode="RGBA")


def build_wandering_drawing_1_scroll_portal_layers(
    project_root: Path,
    cfg: NewScenesBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
    healed_cache: dict[str, Image.Image] | None = None,
) -> dict[str, Any]:
    """
    Synthesizes the upgraded `wandering_drawing_1` MID + forward `scroll_foot` extraLayer +
    dissolvable `scroll_veil` portal aperture assets (`scene7_wandering1_{mid,mid_depth,scroll_foot,scroll_foot_depth,scroll_veil}.png`).
    """
    spec = cfg.wandering_drawing_1
    portal_spec = spec.scroll_portal
    work_dims = cfg.dimensions
    inner_dims = cfg.inner_dimensions
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    local_cache: dict[str, Image.Image] = dict(healed_cache) if healed_cache is not None else {}
    for isl in spec.islands:
        if isl.target_layer == "mid" and isl.lama_heal_boxes_px and isl.catalog_id not in local_cache:
            raw_img = Image.open(project_root / isl.source_relpath).convert("RGB")
            local_cache[isl.catalog_id] = _heal_boxes_with_lama(
                src_rgb=raw_img,
                boxes=isl.lama_heal_boxes_px,
                lama_inpainter=lama_inpainter,
                dilation_px=spec.lama_heal_dilation_px,
                blur_sigma=spec.lama_heal_blur_sigma,
            )

    mid_work = Image.fromarray(
        np.zeros((work_dims.height, work_dims.width, 4), dtype=np.uint8), mode="RGBA"
    )
    foot_work = Image.fromarray(
        np.zeros((work_dims.height, work_dims.width, 4), dtype=np.uint8), mode="RGBA"
    )

    for isl in spec.islands:
        if isl.target_layer != "mid":
            continue
        src_img = local_cache.get(isl.catalog_id) or Image.open(
            project_root / isl.source_relpath
        ).convert("RGB")
        if isl.extraction_method == "sam_scroll_sculpture" and isl.catalog_id == portal_spec.scroll_catalog_id:
            scroll_patch_rgba, foot_patch_rgba = _extract_scroll_paper_and_foot_patches(
                src_rgb=src_img,
                isl=isl,
                portal_spec=portal_spec,
                sam_extractor=sam_extractor,
                lama_inpainter=lama_inpainter,
            )
            mid_work.alpha_composite(scroll_patch_rgba, dest=isl.dest_xy)
            foot_work.alpha_composite(foot_patch_rgba, dest=isl.dest_xy)
        elif isl.extraction_method == "sam_scroll_sculpture":
            patch_rgba = _extract_sam_scroll_sculpture_rgba(src_img, isl, sam_extractor)
            mid_work.alpha_composite(patch_rgba, dest=isl.dest_xy)
        elif isl.extraction_method in ("wall_cutout_unmix", "optical_pigment_unmix"):
            patch_rgba = _extract_pigment_or_wall_island_rgba(src_img, isl, spec)
            mid_work.alpha_composite(patch_rgba, dest=isl.dest_xy)
        else:
            raise ValueError(f"Unsupported mid island extraction_method: {isl.extraction_method}")

    mid_healed_canvas = mid_work.resize(
        (inner_dims.width, inner_dims.height), Image.Resampling.LANCZOS
    )
    foot_canvas = _prune_tiny_alpha_specks_rgba(
        foot_work.resize((inner_dims.width, inner_dims.height), Image.Resampling.LANCZOS),
        min_area_px=80,
    )

    mid_cut_canvas, scroll_veil_canvas = _partition_scroll_veil_rgba(
        mid_healed_rgba=mid_healed_canvas,
        inner_box_px=portal_spec.inner_box_px,
        feather_px=portal_spec.veil_feather_px,
    )

    _assert_zero_border_alpha(mid_cut_canvas, f"{spec.scene_id}.mid")
    _assert_zero_border_alpha(foot_canvas, f"{spec.scene_id}.scroll_foot")
    _assert_zero_border_alpha(scroll_veil_canvas, f"{spec.scene_id}.scroll_veil")

    # Verify lossless optical rest-pose reconstruction in premultiplied RGBA space
    # (ignores unpremultiplied RGB residue at 100% transparent alpha == 0 pixels)
    recomp = np.asarray(Image.alpha_composite(mid_cut_canvas, scroll_veil_canvas), dtype=np.float32)
    orig_h = np.asarray(mid_healed_canvas, dtype=np.float32)
    recomp_premult = np.dstack([recomp[:, :, :3] * (recomp[:, :, 3:4] / 255.0), recomp[:, :, 3:4]])
    orig_premult = np.dstack([orig_h[:, :, :3] * (orig_h[:, :, 3:4] / 255.0), orig_h[:, :, 3:4]])
    max_recomp_err = float(np.max(np.abs(recomp_premult - orig_premult)))
    if max_recomp_err > 2.0:
        raise RuntimeError(
            f"Scroll veil partition failed exact rest-pose reconstruction (max_err={max_recomp_err:.2f})."
        )

    mid_depth_img = _compute_edt_extrapolated_layer_depth_u8(
        layer_rgba=mid_cut_canvas,
        neutral_backdrop_rgb=spec.paper_base_rgb,
        depth_estimator=depth_estimator,
        depth_min_norm=float(spec.layer_depth_min_u8) / 255.0,
        depth_max_norm=1.0,
    )
    foot_depth_img = _compute_edt_extrapolated_layer_depth_u8(
        layer_rgba=foot_canvas,
        neutral_backdrop_rgb=spec.paper_base_rgb,
        depth_estimator=depth_estimator,
        depth_min_norm=portal_spec.foot_depth_min_norm,
        depth_max_norm=portal_spec.foot_depth_max_norm,
    )

    pfx = spec.asset_prefix
    mid_cut_canvas.save(scenes_dir / f"{pfx}_mid.png", format="PNG")
    mid_depth_img.save(scenes_dir / f"{pfx}_mid_depth.png", format="PNG")
    foot_canvas.save(scenes_dir / f"{pfx}_scroll_foot.png", format="PNG")
    foot_depth_img.save(scenes_dir / f"{pfx}_scroll_foot_depth.png", format="PNG")
    scroll_veil_canvas.save(scenes_dir / f"{pfx}_scroll_veil.png", format="PNG")

    x0, y0, x1, y1 = portal_spec.inner_box_px
    veil_box_alpha = np.asarray(scroll_veil_canvas)[:, :, 3][y0:y1, x0:x1]
    veil_cov = float((veil_box_alpha >= 24).mean())
    foot_cov = float((np.asarray(foot_canvas)[:, :, 3] >= 20).mean())
    mid_cov = float((np.asarray(mid_cut_canvas)[:, :, 3] > 16).mean())

    return {
        "midTexture": f"assets/scenes/{pfx}_mid.png",
        "midDepthTexture": f"assets/scenes/{pfx}_mid_depth.png",
        "scrollFootTexture": f"assets/scenes/{pfx}_scroll_foot.png",
        "scrollFootDepthTexture": f"assets/scenes/{pfx}_scroll_foot_depth.png",
        "scrollVeilTexture": f"assets/scenes/{pfx}_scroll_veil.png",
        "midZ": portal_spec.mid_z,
        "scrollFootZPos": portal_spec.foot_z_pos,
        "scrollFootDepthAmplitude": portal_spec.foot_depth_amplitude,
        "portalInnerBoxPx": list(portal_spec.inner_box_px),
        "veilFeatherPx": portal_spec.veil_feather_px,
        "veilCoverageRatio": round(veil_cov, 4),
        "scrollFootCoverageRatio": round(foot_cov, 4),
        "midCoverageRatio": round(mid_cov, 4),
        "maxRestPoseRecompErrorU8": round(max_recomp_err, 3),
    }


def _assert_zero_border_alpha(layer_rgba: Image.Image, label: str) -> None:
    alpha = np.asarray(layer_rgba.convert("RGBA"), dtype=np.uint8)[:, :, 3]
    max_edge = int(
        max(
            alpha[:, 0].max(),
            alpha[:, -1].max(),
            alpha[0, :].max(),
            alpha[-1, :].max(),
        )
    )
    if max_edge > 0:
        raise RuntimeError(f"{label} violated zero-border-alpha invariant (max_edge={max_edge}).")


def build_wandering_drawing_1_scene(
    project_root: Path,
    cfg: NewScenesBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    spec = cfg.wandering_drawing_1
    work_dims = cfg.dimensions
    inner_dims = cfg.inner_dimensions
    exp_dims = cfg.expanded_dimensions
    scenes_dir = project_root / cfg.scenes_output_relpath
    drawings_dir = project_root / cfg.drawings_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)
    drawings_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Scene 7: {spec.scene_id}] Synthesizing 9-artwork spatial drawing world "
        f"(inner={inner_dims.width}x{inner_dims.height}, expanded_bg={exp_dims.width}x{exp_dims.height})..."
    )
    if len(spec.artworks) != 9 or len({a.catalog_id for a in spec.artworks}) != 9:
        raise ValueError("wandering_drawing_1 requires exact 9 unique deduplicated artworks.")

    # Cache healed sources (specifically IMG_PORT_0043 after removing embedded mini-squares)
    healed_cache: dict[str, Image.Image] = {}
    for isl in spec.islands:
        if isl.lama_heal_boxes_px:
            raw_img = Image.open(project_root / isl.source_relpath).convert("RGB")
            print(
                f"  -> Healing {len(isl.lama_heal_boxes_px)} embedded duplicate mini-frames "
                f"in {isl.catalog_id} via neural LaMa..."
            )
            healed_cache[isl.catalog_id] = _heal_boxes_with_lama(
                src_rgb=raw_img,
                boxes=isl.lama_heal_boxes_px,
                lama_inpainter=lama_inpainter,
                dilation_px=spec.lama_heal_dilation_px,
                blur_sigma=spec.lama_heal_blur_sigma,
            )

    # 1. Export 9 crisp 480px-high thumbnails
    thumb_records: list[dict[str, Any]] = []
    for art in spec.artworks:
        src_img = healed_cache.get(art.catalog_id) or Image.open(
            project_root / art.source_relpath
        ).convert("RGB")
        th_h = spec.thumbnail_height_px
        th_w = max(32, int(round(src_img.width * th_h / max(1, src_img.height))))
        thumb_img = src_img.resize((th_w, th_h), Image.Resampling.LANCZOS)
        thumb_path = drawings_dir / art.thumbnail_filename
        thumb_img.save(thumb_path, format="JPEG", quality=spec.thumbnail_jpeg_quality, optimize=True)
        thumb_records.append(
            {
                "catalogId": art.catalog_id,
                "droppedDuplicateIds": list(art.dropped_duplicate_ids),
                "titleHe": art.title_he,
                "year": art.year_label,
                "assignedLayer": art.assigned_layer,
                "placementRole": art.placement_role,
                "thumbnailRelpath": f"assets/drawings/{art.thumbnail_filename}",
                "thumbnailWidth": th_w,
                "thumbnailHeight": th_h,
                "sourceRelpath": art.source_relpath,
            }
        )

    # 2. Synthesize seamless 1.80x expanded archival paper BG (2700x1800)
    bg_f32 = _build_archival_paper_ground(project_root, exp_dims, spec)
    for zone in spec.bg_zones:
        bg_f32 = _fuse_paper_transmittance_zone(bg_f32, project_root, zone, spec)
    bg_img = Image.fromarray(np.round(bg_f32).astype(np.uint8), mode="RGB")
    bg_depth_img = depth_estimator.build_drawing_dome_relief_depth_uint8(
        bg_img, cfg.graphite_drawing
    )

    # 3. Synthesize MID + scroll_foot + scroll_veil via shared reproducible portal helper, plus FG
    portal_summary = build_wandering_drawing_1_scroll_portal_layers(
        project_root=project_root,
        cfg=cfg,
        depth_estimator=depth_estimator,
        sam_extractor=sam_extractor,
        lama_inpainter=lama_inpainter,
        healed_cache=healed_cache,
    )

    fg_work = Image.fromarray(
        np.zeros((work_dims.height, work_dims.width, 4), dtype=np.uint8), mode="RGBA"
    )
    for isl in spec.islands:
        if isl.target_layer != "fg":
            continue
        src_img = healed_cache.get(isl.catalog_id) or Image.open(
            project_root / isl.source_relpath
        ).convert("RGB")
        if isl.extraction_method in ("wall_cutout_unmix", "optical_pigment_unmix"):
            patch_rgba = _extract_pigment_or_wall_island_rgba(src_img, isl, spec)
            fg_work.alpha_composite(patch_rgba, dest=isl.dest_xy)
        else:
            raise ValueError(f"Unsupported fg island extraction_method: {isl.extraction_method}")

    fg_canvas = fg_work.resize((inner_dims.width, inner_dims.height), Image.Resampling.LANCZOS)
    _assert_zero_border_alpha(fg_canvas, f"{spec.scene_id}.fg")

    fg_depth_img = _compute_edt_extrapolated_layer_depth_u8(
        layer_rgba=fg_canvas,
        neutral_backdrop_rgb=spec.paper_base_rgb,
        depth_estimator=depth_estimator,
        depth_min_norm=float(spec.layer_depth_min_u8) / 255.0,
        depth_max_norm=1.0,
    )

    pfx = spec.asset_prefix
    outputs = {
        "bg": scenes_dir / f"{pfx}_bg.png",
        "bg_depth": scenes_dir / f"{pfx}_bg_depth.png",
        "fg": scenes_dir / f"{pfx}_fg.png",
        "fg_depth": scenes_dir / f"{pfx}_fg_depth.png",
    }
    bg_img.save(outputs["bg"], format="PNG")
    bg_depth_img.save(outputs["bg_depth"], format="PNG")
    fg_canvas.save(outputs["fg"], format="PNG")
    fg_depth_img.save(outputs["fg_depth"], format="PNG")

    fg_cov = float((np.asarray(fg_canvas)[:, :, 3] > 16).mean())
    mid_cov = float(portal_summary["midCoverageRatio"])
    print(f"  -> Saved {pfx}_*.png (fg_cov={fg_cov:.4f}, mid_cov={mid_cov:.4f}) + 9 480px thumbnails.")

    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "icon": spec.icon,
        "innerWidth": inner_dims.width,
        "innerHeight": inner_dims.height,
        "expandedWidth": exp_dims.width,
        "expandedHeight": exp_dims.height,
        "bgExpansionFactor": cfg.expanded_bg_scale,
        "bgTexture": f"assets/scenes/{pfx}_bg.png",
        "depthTexture": f"assets/scenes/{pfx}_bg_depth.png",
        "midTexture": portal_summary["midTexture"],
        "midDepthTexture": portal_summary["midDepthTexture"],
        "scrollFootTexture": portal_summary["scrollFootTexture"],
        "scrollFootDepthTexture": portal_summary["scrollFootDepthTexture"],
        "scrollVeilTexture": portal_summary["scrollVeilTexture"],
        "fgTexture": f"assets/scenes/{pfx}_fg.png",
        "fgDepthTexture": f"assets/scenes/{pfx}_fg_depth.png",
        "fgCoverageRatio": round(fg_cov, 4),
        "midCoverageRatio": round(mid_cov, 4),
        "scrollPortal": portal_summary,
        "deduplicatedArtworks": thumb_records,
    }


# ============================================================================
# Scene B Builder: `denya_model` ("דגם בית דניה על לבד אפור")
# ============================================================================


def _synthesize_seamless_felt_backdrop_lama(
    project_root: Path,
    dims: CanvasDimensionsConfig,
    spec: DenyaModelSceneSpec,
    lama_inpainter: LamaOcclusionInpainter,
    max_working_side_px: int = 1350,
    roi_context_margin_px: int = 192,
) -> Image.Image:
    """
    Synthesizes a continuous, non-periodic moving-blanket gray felt backdrop from real `IMG_BINY_0007`
    felt patches using staggered 2D cosine overlap quilting + sparse circular LaMa knot healing
    (avoiding rectilinear grid masks that cause architectural FFC models to hallucinate mortar joints).
    """
    raw_shelf = np.asarray(
        Image.open(project_root / spec.felt_source_relpath).convert("RGB"), dtype=np.float32
    )
    target_mean = np.array(spec.felt_target_mean_rgb, dtype=np.float32)
    target_std = np.array(spec.felt_target_std_rgb, dtype=np.float32)

    tw, th = spec.felt_tile_draw_wh
    overlap = max(24, spec.felt_seam_half_width_px * 2)
    step_x = max(64, tw - overlap)
    step_y = max(64, th - overlap)

    norm_patches: list[np.ndarray] = []
    for p_spec in spec.felt_patches:
        x0, y0, x1, y1 = p_spec.box_xyxy
        patch = raw_shelf[y0:y1, x0:x1].copy()
        low_illum = ndi.gaussian_filter(patch, sigma=(14.0, 14.0, 0.0))
        flat_patch = patch - low_illum + np.mean(patch, axis=(0, 1), keepdims=True)
        p_mean = np.mean(flat_patch, axis=(0, 1), keepdims=True)
        p_std = np.maximum(3.0, np.std(flat_patch, axis=(0, 1), keepdims=True))
        harmonized = (flat_patch - p_mean) * (target_std[None, None, :] / p_std) + target_mean[
            None, None, :
        ]
        resized_tile = cv2.resize(
            np.clip(harmonized, 0.0, 255.0), (tw, th), interpolation=cv2.INTER_LANCZOS4
        )
        norm_patches.append(resized_tile)

    tile_win = _cosine_feather_2d(
        height=th,
        width=tw,
        left_px=overlap,
        top_px=overlap,
        right_px=overlap,
        bottom_px=overlap,
        elliptical=False,
    )[:, :, None]

    pad_w = dims.width + tw
    pad_h = dims.height + th
    acc_rgb = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    acc_w = np.zeros((pad_h, pad_w, 1), dtype=np.float32)
    knot_mask = np.zeros((pad_h, pad_w), dtype=np.uint8)

    n_rows = (pad_h - overlap) // step_y + 2
    n_cols = (pad_w - overlap) // step_x + 2
    for r in range(n_rows):
        row_shift = (r % 2) * (step_x // 2)
        for c in range(n_cols):
            py = r * step_y
            px = c * step_x - row_shift
            if px + tw <= 0 or py + th <= 0 or px >= pad_w or py >= pad_h:
                continue
            tile = norm_patches[(r * 3 + c * 5 + r) % len(norm_patches)].copy()
            if (r + c) % 2 == 1:
                tile = tile[:, ::-1, :]
            if (r * 2 + c) % 3 == 1:
                tile = tile[::-1, :, :]
            if (r + c * 2) % 4 == 2:
                tile = np.swapaxes(tile, 0, 1)

            sx0 = max(0, -px)
            sy0 = max(0, -py)
            dx0 = max(0, px)
            dy0 = max(0, py)
            w_sub = min(tw - sx0, pad_w - dx0)
            h_sub = min(th - sy0, pad_h - dy0)
            if w_sub <= 0 or h_sub <= 0:
                continue

            w_patch = tile_win[sy0 : sy0 + h_sub, sx0 : sx0 + w_sub]
            acc_rgb[dy0 : dy0 + h_sub, dx0 : dx0 + w_sub] += (
                tile[sy0 : sy0 + h_sub, sx0 : sx0 + w_sub] * w_patch
            )
            acc_w[dy0 : dy0 + h_sub, dx0 : dx0 + w_sub] += w_patch

            if r > 0 and c > 0 and (r + c) % 2 == 0:
                kr = max(8, spec.felt_lama_dilation_px * 4)
                kx = min(pad_w - kr - 4, max(kr + 4, dx0 + overlap // 2))
                ky = min(pad_h - kr - 4, max(kr + 4, dy0 + overlap // 2))
                yy_k, xx_k = np.ogrid[ky - kr : ky + kr + 1, kx - kr : kx + kr + 1]
                disk = ((xx_k - kx) ** 2 + (yy_k - ky) ** 2) <= (kr * kr)
                knot_mask[ky - kr : ky + kr + 1, kx - kr : kx + kr + 1][disk] = 255

    ox = overlap // 2
    oy = overlap // 2
    blended = acc_rgb[oy : oy + dims.height, ox : ox + dims.width] / np.maximum(
        1e-4, acc_w[oy : oy + dims.height, ox : ox + dims.width]
    )
    blended_u8 = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    knot_crop = knot_mask[oy : oy + dims.height, ox : ox + dims.width]

    healed_rgb = _inpaint_masked_plate_lama(
        src_rgb=Image.fromarray(blended_u8, mode="RGB"),
        binary_mask_u8=knot_crop,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.felt_lama_dilation_px,
        blur_sigma=spec.felt_lama_blur_sigma,
        max_working_side_px=max_working_side_px,
        roi_context_margin_px=roi_context_margin_px,
    )
    felt_f = np.asarray(healed_rgb, dtype=np.float32)

    yy = np.linspace(-1.0, 1.0, dims.height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, dims.width, dtype=np.float32)[None, :]
    rad2 = 0.62 * xx * xx + 0.78 * (yy + 0.08) * (yy + 0.08)
    spotlight = 1.0 + spec.felt_spotlight_gain * np.exp(-2.1 * rad2) - spec.felt_vignette_drop * np.clip(
        rad2, 0.0, 1.25
    )
    lit_felt = np.clip(np.round(felt_f * spotlight[:, :, None]), 0, 255).astype(np.uint8)
    return Image.fromarray(lit_felt, mode="RGB")


def _complete_and_segment_denya_maquette_rgba(
    project_root: Path,
    work_dims: CanvasDimensionsConfig,
    inner_dims: CanvasDimensionsConfig,
    spec: DenyaModelSceneSpec,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> tuple[Image.Image, np.ndarray]:
    src_img = Image.open(project_root / spec.primary_model_relpath).convert("RGB")
    sw, sh = src_img.size
    pl, pt, pr, pb = spec.pad_margin_ltrb
    pw, ph = sw + pl + pr, sh + pt + pb

    src_np = np.asarray(src_img, dtype=np.uint8)
    padded_np = np.zeros((ph, pw, 3), dtype=np.uint8)
    cloth_rgb = np.array(spec.studio_cloth_fill_rgb, dtype=np.uint8)
    padded_np[:, :] = cloth_rgb[None, None, :]
    padded_np[pt : pt + sh, pl : pl + sw] = src_np

    # Seed natural papier-mâché diagonal hillside ridge on the left margin (guided by IMG_WORD_0148)
    # so neural LaMa heals the seam cleanly into a freestanding physical model edge.
    if pl > 0:
        mirror_strip = src_np[:, : min(sw, pl)][:, ::-1, :].copy()
        for row_idx in range(sh):
            py_row = pt + row_idx
            # Physical maquette left cardboard base slopes down-right from (x=28, y=212) to (x=72, y=396)
            if 208 <= py_row <= 405:
                diag_x = int(round(28.0 + (py_row - 208.0) * ((pl - 28.0) / (405.0 - 208.0))))
                diag_x = max(24, min(pl, diag_x))
                span_w = pl - diag_x
                if span_w > 0:
                    padded_np[py_row, diag_x:pl] = mirror_strip[row_idx, -span_w:]

    border_mask = np.zeros((ph, pw), dtype=np.uint8)
    if pl > 0:
        # Heal strictly along the vertical graft seam (pl-10..pl+10) and right border boxes
        border_mask[204:412, max(0, pl - 10) : min(pw, pl + 10)] = 255
    for b in spec.border_completion_boxes_padded[1 if pl > 0 else 0 :]:
        border_mask[max(0, b.y0) : min(ph, b.y1), max(0, b.x0) : min(pw, b.x1)] = 255

    completed_rgb = _inpaint_masked_plate_lama(
        src_rgb=Image.fromarray(padded_np, mode="RGB"),
        binary_mask_u8=border_mask,
        lama_inpainter=lama_inpainter,
        dilation_px=spec.border_lama_dilation_px,
        blur_sigma=spec.border_lama_blur_sigma,
        max_working_side_px=1350,
        roi_context_margin_px=192,
    )

    sam_alpha = sam_extractor.segment_point_groups(
        rgb_image=completed_rgb,
        point_groups=spec.padded_sam_prompts,
        exclude_boxes=(),
        max_pinhole_fill_px=900,
        min_major_component_px=2500,
        right_edge_fade_x_start=None,
    )

    cutout_isnet = remove(completed_rgb, session=matte_extractor.session, only_mask=False)
    if not isinstance(cutout_isnet, Image.Image):
        raise RuntimeError("IS-Net rembg failed on Denya architectural maquette.")
    isnet_alpha = np.asarray(cutout_isnet.convert("RGBA"), dtype=np.float32)[:, :, 3]

    support = (
        cv2.dilate(
            (sam_alpha > 120).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        > 0
    )
    eroded_sam = cv2.erode(
        sam_alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ).astype(np.float32)
    alpha_f = np.where(support, np.maximum(eroded_sam, isnet_alpha), 0.0)

    comp_np = np.asarray(completed_rgb, dtype=np.float32)
    lum = 0.299 * comp_np[:, :, 0] + 0.587 * comp_np[:, :, 1] + 0.114 * comp_np[:, :, 2]
    blue_excess = comp_np[:, :, 2] - np.maximum(comp_np[:, :, 0], comp_np[:, :, 1] * 0.92)
    navy_cloth = (blue_excess > spec.blue_cloth_excess_thresh) & (lum < spec.blue_cloth_max_lum)

    phb = spec.pergola_cloth_hole_box_padded
    pergola_zone = np.zeros((ph, pw), dtype=bool)
    pergola_zone[phb.y0 : phb.y1, phb.x0 : phb.x1] = True
    alpha_f = np.where(navy_cloth & (pergola_zone | (sam_alpha < 128)), 0.0, alpha_f)

    # Enforce clean diagonal physical boundary on completed left hillside tip so zero cloth bleeds in
    if pl > 0:
        yy_grid, xx_grid = np.mgrid[0:ph, 0:pw]
        outside_left_tip = (xx_grid < pl) & (
            (yy_grid < 208)
            | (yy_grid > 406)
            | (xx_grid < (26.0 + (yy_grid - 208.0) * ((pl - 26.0) / (406.0 - 208.0))))
        )
        soft_tip_keep = 1.0 - ndi.gaussian_filter(outside_left_tip.astype(np.float32), sigma=1.2)
        alpha_f = alpha_f * np.clip(soft_tip_keep, 0.0, 1.0)

    # Fill any unintended interior pinholes inside the solid maquette body (preserving the open pergola aperture)
    solid_body = (alpha_f > 96.0).astype(np.uint8)
    inv_body = (solid_body == 0).astype(np.uint8)
    nh, h_lbls, h_stats, _ = cv2.connectedComponentsWithStats(inv_body, connectivity=4)
    for hid in range(1, nh):
        hx = int(h_stats[hid, cv2.CC_STAT_LEFT])
        hy = int(h_stats[hid, cv2.CC_STAT_TOP])
        hw = int(h_stats[hid, cv2.CC_STAT_WIDTH])
        hh = int(h_stats[hid, cv2.CC_STAT_HEIGHT])
        touches_border = hx == 0 or hy == 0 or (hx + hw >= pw) or (hy + hh >= ph)
        hole_mask = h_lbls == hid
        in_pergola = bool(np.any(hole_mask & pergola_zone))
        if not touches_border and not in_pergola and int(h_stats[hid, cv2.CC_STAT_AREA]) < 4500:
            alpha_f = np.where(hole_mask, 255.0, alpha_f)

    fg_bin = (alpha_f > 32.0).astype(np.uint8)
    nl, lbls, stats, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=8)
    if nl <= 1:
        raise RuntimeError("Denya model segmentation produced zero foreground components.")
    largest_cid = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    alpha_u8 = np.where(lbls == largest_cid, np.clip(np.round(alpha_f), 0, 255), 0).astype(np.uint8)

    raw_u8 = np.asarray(completed_rgb, dtype=np.uint8)
    decontam_u8 = decontaminate_matte_edge_rgb(raw_u8, alpha_u8)
    # Keep bright white foamcore roof edges neutral so unmixing dark-blue cloth never creates warm/magenta overshoot
    bright_roof = (comp_np[:, :, 0] > 160.0) & (comp_np[:, :, 1] > 152.0) & (lum > 165.0)
    neutral_roof = np.repeat(np.round(lum).astype(np.uint8)[:, :, None], 3, axis=2)
    decontam_u8 = np.where(
        bright_roof[:, :, None] & (alpha_u8[:, :, None] < 250), neutral_roof, decontam_u8
    )

    ys, xs = np.where(alpha_u8 > 8)
    bx0 = max(0, int(xs.min()) - 6)
    by0 = max(0, int(ys.min()) - 6)
    bx1, by1 = min(pw, int(xs.max()) + 7), min(ph, int(ys.max()) + 7)
    tight_rgba = Image.fromarray(
        np.dstack([decontam_u8[by0:by1, bx0:bx1], alpha_u8[by0:by1, bx0:bx1]]), mode="RGBA"
    )

    dest_w = spec.maquette_dest_width
    dest_h = max(64, int(round(dest_w * tight_rgba.height / max(1, tight_rgba.width))))
    scaled_rgba = tight_rgba.resize((dest_w, dest_h), Image.Resampling.LANCZOS)
    sc_np = np.asarray(scaled_rgba, dtype=np.float32)
    blur_rgb = ndi.gaussian_filter(sc_np[:, :, :3], sigma=(spec.unsharp_sigma, spec.unsharp_sigma, 0.0))
    sharp_rgb = np.clip(
        sc_np[:, :, :3] + spec.unsharp_amount * (sc_np[:, :, :3] - blur_rgb), 0.0, 255.0
    )

    work_canvas = Image.fromarray(
        np.zeros((work_dims.height, work_dims.width, 4), dtype=np.uint8), mode="RGBA"
    )
    scaled_clean = Image.fromarray(
        np.dstack([np.round(sharp_rgb).astype(np.uint8), np.round(sc_np[:, :, 3]).astype(np.uint8)]),
        mode="RGBA",
    )
    dx = int(round(spec.maquette_dest_center_xy[0] - dest_w * 0.5))
    dy = int(round(spec.maquette_dest_center_xy[1] - dest_h * 0.5))
    work_canvas.alpha_composite(scaled_clean, dest=(dx, dy))

    inner_canvas = work_canvas.resize((inner_dims.width, inner_dims.height), Image.Resampling.LANCZOS)
    _assert_zero_border_alpha(inner_canvas, f"{spec.scene_id}.maquette_inner")
    return inner_canvas, np.asarray(inner_canvas)[:, :, 3]


def _split_denya_maquette_fg_mid_layers(
    full_maquette_rgba: Image.Image,
    maquette_depth_u8: Image.Image,
    spec: DenyaModelSceneSpec,
    lama_inpainter: LamaOcclusionInpainter,
) -> tuple[Image.Image, Image.Image]:
    """
    Splits the freestanding Denya architectural maquette into two 3D relief tiers using real
    Depth Anything V2 predictions:
      - `fg`: nearest cantilevered living-room volume, balcony & green garden lawn (`border_alpha=0`)
      - `mid`: full maquette with LaMa-healed interior behind `fg`'s solid core so parallax never ghosts
    """
    rgba_np = np.asarray(full_maquette_rgba.convert("RGBA"), dtype=np.float32)
    rgb_u8 = np.round(rgba_np[:, :, :3]).astype(np.uint8)
    alpha_f = rgba_np[:, :, 3] / 255.0
    depth_f = np.asarray(maquette_depth_u8, dtype=np.float32) / 255.0

    active = alpha_f > 0.05
    if not np.any(active):
        raise RuntimeError("Denya maquette alpha is empty during FG/MID split.")
    d_lo = float(np.percentile(depth_f[active], spec.fg_vignette_inner_radius_norm * 100.0))
    d_hi = float(np.percentile(depth_f[active], spec.fg_vignette_outer_radius_norm * 100.0))
    t_near = np.clip((depth_f - d_lo) / max(0.03, d_hi - d_lo), 0.0, 1.0)
    near_smooth = ndi.gaussian_filter(t_near * t_near * (3.0 - 2.0 * t_near), sigma=3.5)

    fg_alpha_u8 = np.clip(np.round(alpha_f * near_smooth * 255.0), 0, 255).astype(np.uint8)
    fg_rgba = Image.fromarray(np.dstack([rgb_u8, fg_alpha_u8]), mode="RGBA")
    _assert_zero_border_alpha(fg_rgba, f"{spec.scene_id}.fg")

    # Heal behind opaque foreground core on `mid` so parallax never reveals duplicate window mullions
    fg_core = cv2.erode(
        (fg_alpha_u8 > 215).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
    )
    if np.any(fg_core > 0):
        neutral_bg = np.array(spec.felt_target_mean_rgb, dtype=np.float32)[None, None, :]
        seeded_mid = np.clip(
            np.round(alpha_f[:, :, None] * rgba_np[:, :, :3] + (1.0 - alpha_f[:, :, None]) * neutral_bg),
            0,
            255,
        ).astype(np.uint8)
        healed_mid_rgb = np.asarray(
            _inpaint_masked_plate_lama(
                src_rgb=Image.fromarray(seeded_mid, mode="RGB"),
                binary_mask_u8=fg_core,
                lama_inpainter=lama_inpainter,
                dilation_px=6,
                blur_sigma=2.0,
                max_working_side_px=1350,
                roi_context_margin_px=192,
            ),
            dtype=np.uint8,
        )
        blend_core = (ndi.gaussian_filter(fg_core.astype(np.float32) / 255.0, sigma=4.0))[:, :, None]
        mid_rgb_u8 = np.clip(
            np.round(blend_core * healed_mid_rgb + (1.0 - blend_core) * rgb_u8), 0, 255
        ).astype(np.uint8)
    else:
        mid_rgb_u8 = rgb_u8

    mid_rgba = Image.fromarray(
        np.dstack([mid_rgb_u8, np.round(rgba_np[:, :, 3]).astype(np.uint8)]), mode="RGBA"
    )
    _assert_zero_border_alpha(mid_rgba, f"{spec.scene_id}.mid")
    return mid_rgba, fg_rgba


def _apply_maquette_contact_shadow_to_expanded_felt(
    felt_bg_rgb: Image.Image,
    inner_maquette_alpha_u8: np.ndarray,
    exp_dims: CanvasDimensionsConfig,
    inner_dims: CanvasDimensionsConfig,
    spec: DenyaModelSceneSpec,
) -> Image.Image:
    pad_x = (exp_dims.width - inner_dims.width) // 2
    pad_y = (exp_dims.height - inner_dims.height) // 2
    full_alpha = np.zeros((exp_dims.height, exp_dims.width), dtype=np.uint8)
    full_alpha[pad_y : pad_y + inner_dims.height, pad_x : pad_x + inner_dims.width] = (
        inner_maquette_alpha_u8
    )

    ox, oy = spec.contact_shadow_offset_xy
    shifted = np.zeros_like(full_alpha, dtype=np.float32)
    h, w = full_alpha.shape
    sy0, dy0 = max(0, -oy), max(0, oy)
    sx0, dx0 = max(0, -ox), max(0, ox)
    sh_h = h - abs(oy)
    sh_w = w - abs(ox)
    shifted[dy0 : dy0 + sh_h, dx0 : dx0 + sh_w] = (
        full_alpha[sy0 : sy0 + sh_h, sx0 : sx0 + sh_w].astype(np.float32) / 255.0
    )
    shadow = ndi.gaussian_filter(shifted, sigma=spec.contact_shadow_blur_sigma)
    atten = 1.0 - spec.contact_shadow_strength * np.clip(shadow, 0.0, 1.0)
    bg_f = np.asarray(felt_bg_rgb, dtype=np.float32) * atten[:, :, None]
    return Image.fromarray(np.clip(np.round(bg_f), 0, 255).astype(np.uint8), mode="RGB")


def build_denya_model_scene(
    project_root: Path,
    cfg: NewScenesBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> dict[str, Any]:
    spec = cfg.denya_model
    work_dims = cfg.dimensions
    inner_dims = cfg.inner_dimensions
    exp_dims = cfg.expanded_dimensions
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Scene 8: {spec.scene_id}] Synthesizing Denya house maquette on 1.80x gray felt "
        f"(inner={inner_dims.width}x{inner_dims.height}, expanded_bg={exp_dims.width}x{exp_dims.height})..."
    )
    felt_bg_clean = _synthesize_seamless_felt_backdrop_lama(
        project_root=project_root,
        dims=exp_dims,
        spec=spec,
        lama_inpainter=lama_inpainter,
    )
    full_maquette_rgba, maquette_alpha_u8 = _complete_and_segment_denya_maquette_rgba(
        project_root=project_root,
        work_dims=work_dims,
        inner_dims=inner_dims,
        spec=spec,
        sam_extractor=sam_extractor,
        matte_extractor=matte_extractor,
        lama_inpainter=lama_inpainter,
    )
    bg_img = _apply_maquette_contact_shadow_to_expanded_felt(
        felt_bg_rgb=felt_bg_clean,
        inner_maquette_alpha_u8=maquette_alpha_u8,
        exp_dims=exp_dims,
        inner_dims=inner_dims,
        spec=spec,
    )

    raw_felt_depth = depth_estimator.predict_depth(felt_bg_clean)
    smooth_felt_depth = ndi.gaussian_filter(raw_felt_depth, sigma=spec.felt_depth_sigma)
    fd_min, fd_max = float(smooth_felt_depth.min()), float(smooth_felt_depth.max())
    norm_fd = (smooth_felt_depth - fd_min) / max(1e-6, fd_max - fd_min)
    bg_depth_img = Image.fromarray(
        np.clip(np.round(norm_fd * 255.0), 0, 255).astype(np.uint8), mode="L"
    )

    full_maquette_depth_img = _compute_alpha_supported_depth_u8(
        layer_rgba=full_maquette_rgba,
        neutral_backdrop_rgb=spec.felt_target_mean_rgb,
        depth_estimator=depth_estimator,
        min_fg_depth_u8=spec.layer_depth_min_u8,
        collar_sigma=spec.layer_depth_collar_sigma,
    )
    mid_rgba, fg_rgba = _split_denya_maquette_fg_mid_layers(
        full_maquette_rgba=full_maquette_rgba,
        maquette_depth_u8=full_maquette_depth_img,
        spec=spec,
        lama_inpainter=lama_inpainter,
    )
    mid_depth_img = full_maquette_depth_img
    fg_depth_img = _compute_alpha_supported_depth_u8(
        layer_rgba=fg_rgba,
        neutral_backdrop_rgb=spec.felt_target_mean_rgb,
        depth_estimator=depth_estimator,
        min_fg_depth_u8=spec.layer_depth_min_u8,
        collar_sigma=spec.layer_depth_collar_sigma,
    )

    pfx = spec.asset_prefix
    bg_img.save(scenes_dir / f"{pfx}_bg.png", format="PNG")
    bg_depth_img.save(scenes_dir / f"{pfx}_bg_depth.png", format="PNG")
    mid_rgba.save(scenes_dir / f"{pfx}_mid.png", format="PNG")
    mid_depth_img.save(scenes_dir / f"{pfx}_mid_depth.png", format="PNG")
    fg_rgba.save(scenes_dir / f"{pfx}_fg.png", format="PNG")
    fg_depth_img.save(scenes_dir / f"{pfx}_fg_depth.png", format="PNG")

    fg_cov = float((np.asarray(fg_rgba)[:, :, 3] > 16).mean())
    mid_cov = float((np.asarray(mid_rgba)[:, :, 3] > 16).mean())
    print(f"  -> Saved {pfx}_*.png (mid_cov={mid_cov:.4f}, fg_cov={fg_cov:.4f}).")

    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "icon": spec.icon,
        "primaryModelCatalogId": spec.primary_model_catalog_id,
        "primaryModelRelpath": spec.primary_model_relpath,
        "secondaryReferenceCatalogId": spec.secondary_reference_catalog_id,
        "secondaryReferenceRelpath": spec.secondary_reference_relpath,
        "feltSourceCatalogId": spec.felt_source_catalog_id,
        "innerWidth": inner_dims.width,
        "innerHeight": inner_dims.height,
        "expandedWidth": exp_dims.width,
        "expandedHeight": exp_dims.height,
        "bgExpansionFactor": cfg.expanded_bg_scale,
        "bgTexture": f"assets/scenes/{pfx}_bg.png",
        "depthTexture": f"assets/scenes/{pfx}_bg_depth.png",
        "midTexture": f"assets/scenes/{pfx}_mid.png",
        "midDepthTexture": f"assets/scenes/{pfx}_mid_depth.png",
        "fgTexture": f"assets/scenes/{pfx}_fg.png",
        "fgDepthTexture": f"assets/scenes/{pfx}_fg_depth.png",
        "fgCoverageRatio": round(fg_cov, 4),
        "midCoverageRatio": round(mid_cov, 4),
    }


# ============================================================================
# Scene C Builder: `shelf_portal` ("קיר המדפים — שער בין עולמות")
# ============================================================================


def _src_box_to_expanded_xyxy(
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
    src_w: float,
    src_h: float,
    cabinet_body_box_xyxy: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    cab_x0, cab_y0, cab_x1, cab_y1 = cabinet_body_box_xyxy
    cab_w = float(cab_x1 - cab_x0)
    cab_h = float(cab_y1 - cab_y0)
    return (
        float(cab_x0) + (float(bx0) / src_w) * cab_w,
        float(cab_y0) + (float(by0) / src_h) * cab_h,
        float(cab_x0) + (float(bx1) / src_w) * cab_w,
        float(cab_y0) + (float(by1) / src_h) * cab_h,
    )


def _compute_portal_cubby_webgl_coordinates(
    cubby: PortalCubbySpec,
    actual_cutout_mask: np.ndarray,
    src_w: int,
    src_h: int,
    inner_dims: CanvasDimensionsConfig,
    exp_dims: CanvasDimensionsConfig,
    spec: ShelfPortalSceneSpec,
    occluder_runtime_meta: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ys, xs = np.where(actual_cutout_mask > 128)
    if xs.size > 20:
        x0_s, y0_s = int(xs.min()), int(ys.min())
        x1_s, y1_s = int(xs.max()) + 1, int(ys.max()) + 1
    else:
        b = cubby.aperture_bounds_px
        x0_s, y0_s, x1_s, y1_s = b.x0, b.y0, b.x1, b.y1

    pad_x = 0.5 * float(exp_dims.width - inner_dims.width)
    pad_y = 0.5 * float(exp_dims.height - inner_dims.height)
    w_inner = float(inner_dims.width)
    h_inner = float(inner_dims.height)
    w_exp = float(exp_dims.width)
    h_exp = float(exp_dims.height)

    pb = cubby.aperture_bounds_px
    pb_exp = _src_box_to_expanded_xyxy(
        pb.x0, pb.y0, pb.x1, pb.y1, float(src_w), float(src_h), spec.cabinet_body_box_xyxy
    )
    ap_exp = _src_box_to_expanded_xyxy(
        x0_s, y0_s, x1_s, y1_s, float(src_w), float(src_h), spec.cabinet_body_box_xyxy
    )
    tuck = float(spec.inner_box_rim_tuck_px)
    bleed = float(spec.portal_plane_bleed_px)

    inner_box_px = [
        round(pb_exp[0] - pad_x - tuck, 2),
        round(pb_exp[1] - pad_y - tuck, 2),
        round(pb_exp[2] - pad_x + tuck, 2),
        round(pb_exp[3] - pad_y + tuck, 2),
    ]

    def _exp_box_to_world(ex0: float, ey0: float, ex1: float, ey1: float) -> dict[str, float]:
        ix0, iy0 = ex0 - pad_x, ey0 - pad_y
        ix1, iy1 = ex1 - pad_x, ey1 - pad_y
        cx = 0.5 * (ix0 + ix1)
        cy = 0.5 * (iy0 + iy1)
        return {
            "worldX": round((cx / w_inner - 0.5) * spec.webgl_plane_width, 3),
            "worldY": round((0.5 - cy / h_inner) * spec.webgl_plane_height, 3),
            "width": round(((ix1 - ix0) / w_inner) * spec.webgl_plane_width, 3),
            "height": round(((iy1 - iy0) / h_inner) * spec.webgl_plane_height, 3),
        }

    runtime_shelf_spec: dict[str, Any] = {
        "id": cubby.runtime_shelf_id,
        "targetSceneId": cubby.target_scene_id,
        "labelHe": cubby.label_he,
        "innerBoxPx": inner_box_px,
        "worldZ": spec.portal_world_z,
        "recessDepth": spec.portal_recess_depth,
    }
    if occluder_runtime_meta is not None:
        runtime_shelf_spec["occluder"] = occluder_runtime_meta

    coord_record: dict[str, Any] = {
        "cubbyId": cubby.cubby_id,
        "runtimeShelfId": cubby.runtime_shelf_id,
        "rowIndex": cubby.row_index,
        "colIndex": cubby.col_index,
        "targetSceneId": cubby.target_scene_id,
        "labelHe": cubby.label_he,
        "hasRemovableOccluder": occluder_runtime_meta is not None,
        "innerBoxPx": inner_box_px,
        "boxPx2700": [round(v, 2) for v in pb_exp],
        "pixelBoundsSource": {"x0": x0_s, "y0": y0_s, "x1": x1_s, "y1": y1_s},
        "fullCubbyBackplaneBoundsPxSource": {
            "x0": pb.x0,
            "y0": pb.y0,
            "x1": pb.x1,
            "y1": pb.y1,
        },
        "uvExpandedTopLeftOrigin": {
            "u0": round(ap_exp[0] / w_exp, 6),
            "v0": round(ap_exp[1] / h_exp, 6),
            "u1": round(ap_exp[2] / w_exp, 6),
            "v1": round(ap_exp[3] / h_exp, 6),
        },
        "uvExpandedWebglBottomLeftOrigin": {
            "u0": round(ap_exp[0] / w_exp, 6),
            "v0": round(1.0 - ap_exp[3] / h_exp, 6),
            "u1": round(ap_exp[2] / w_exp, 6),
            "v1": round(1.0 - ap_exp[1] / h_exp, 6),
        },
        "worldRectAperture": _exp_box_to_world(*ap_exp),
        "worldRectFullBackplane": _exp_box_to_world(*pb_exp),
        "worldRectBleedBehindRim": _exp_box_to_world(
            pb_exp[0] - bleed, pb_exp[1] - bleed, pb_exp[2] + bleed, pb_exp[3] + bleed
        ),
    }
    if occluder_runtime_meta is not None:
        coord_record["occluder"] = occluder_runtime_meta
    return coord_record, runtime_shelf_spec


def _export_portal_occluder_sprite(
    cubby: PortalCubbySpec,
    shelf_np: np.ndarray,
    occluder_alpha_u8: np.ndarray,
    portal_mask_u8: np.ndarray,
    scenes_dir: Path,
    inner_dims: CanvasDimensionsConfig,
    exp_dims: CanvasDimensionsConfig,
    spec: ShelfPortalSceneSpec,
) -> dict[str, Any]:
    src_h, src_w = occluder_alpha_u8.shape
    ys, xs = np.where(occluder_alpha_u8 > 8)
    if xs.size < 25:
        raise RuntimeError(
            f"SAM occluder segmentation for cubby '{cubby.runtime_shelf_id}' produced empty mask ({xs.size} px)."
        )

    pad = spec.occluder_sprite_pad_px
    ox0 = max(0, int(xs.min()) - pad)
    oy0 = max(0, int(ys.min()) - pad)
    ox1 = min(src_w, int(xs.max()) + 1 + pad)
    oy1 = min(src_h, int(ys.max()) + 1 + pad)

    clean_rgb = decontaminate_matte_edge_rgb(shelf_np, occluder_alpha_u8)
    sprite_rgba_np = np.dstack([clean_rgb[oy0:oy1, ox0:ox1], occluder_alpha_u8[oy0:oy1, ox0:ox1]])
    # Ensure 1px transparent border around sprite so GPU bilinear filtering never bleeds at UV clamp edges
    sprite_rgba_np[0, :, 3] = 0
    sprite_rgba_np[-1, :, 3] = 0
    sprite_rgba_np[:, 0, 3] = 0
    sprite_rgba_np[:, -1, 3] = 0

    sprite_filename = f"{spec.occluder_asset_prefix}_{cubby.runtime_shelf_id}.png"
    Image.fromarray(sprite_rgba_np, mode="RGBA").save(scenes_dir / sprite_filename, format="PNG")

    occ_exp = _src_box_to_expanded_xyxy(
        ox0, oy0, ox1, oy1, float(src_w), float(src_h), spec.cabinet_body_box_xyxy
    )
    pad_x = 0.5 * float(exp_dims.width - inner_dims.width)
    pad_y = 0.5 * float(exp_dims.height - inner_dims.height)
    occ_inner_box_px = [
        round(occ_exp[0] - pad_x, 2),
        round(occ_exp[1] - pad_y, 2),
        round(occ_exp[2] - pad_x, 2),
        round(occ_exp[3] - pad_y, 2),
    ]

    ab = cubby.aperture_bounds_px
    occ_area = float((occluder_alpha_u8[ab.y0 : ab.y1, ab.x0 : ab.x1] > 64).sum())
    ap_area = max(1.0, float((portal_mask_u8[ab.y0 : ab.y1, ab.x0 : ab.x1] > 64).sum()))
    cov_ratio = min(1.0, occ_area / ap_area)

    return {
        "texture": f"assets/scenes/{sprite_filename}",
        "innerBoxPx": occ_inner_box_px,
        "boxPx2700": [round(v, 2) for v in occ_exp],
        "spriteWidth": int(ox1 - ox0),
        "spriteHeight": int(oy1 - oy0),
        "labelHe": cubby.occluder_label_he or cubby.label_he,
        "coverageRatio": round(cov_ratio, 4),
        "locksPortal": True,
    }


def _encode_sam_image_embeddings(
    sam_extractor: SamBoxMaskExtractor,
    rgb_image: Image.Image,
) -> torch.Tensor:
    base_inputs = sam_extractor.processor(rgb_image, return_tensors="pt").to(sam_extractor.device)
    with torch.no_grad():
        return sam_extractor.model.get_image_embeddings(base_inputs["pixel_values"])


def _decode_sam_with_cached_embeddings(
    sam_extractor: SamBoxMaskExtractor,
    rgb_image: Image.Image,
    image_embeddings: torch.Tensor,
    point_groups: Sequence[SamPointGroupPrompt],
    max_pinhole_fill_px: int,
    min_major_component_px: int,
) -> np.ndarray:
    width, height = rgb_image.size
    if not point_groups:
        return np.zeros((height, width), dtype=np.uint8)
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


def _resize_rgba_premultiplied(rgba_np: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    rgb_f = rgba_np[:, :, :3].astype(np.float32)
    a_f = rgba_np[:, :, 3:4].astype(np.float32) / 255.0
    premult = np.clip(np.round(rgb_f * a_f), 0, 255).astype(np.uint8)
    alpha_u8 = rgba_np[:, :, 3]
    pm_resized = np.asarray(
        Image.fromarray(premult, mode="RGB").resize(target_wh, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    a_resized = np.asarray(
        Image.fromarray(alpha_u8, mode="L").resize(target_wh, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    a_norm = a_resized[:, :, None] / 255.0
    rgb_unmult = np.where(a_norm > 0.02, pm_resized / np.maximum(a_norm, 0.02), pm_resized)
    out = np.zeros((target_wh[1], target_wh[0], 4), dtype=np.uint8)
    out[:, :, :3] = np.clip(np.round(rgb_unmult), 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(np.round(a_resized), 0, 255).astype(np.uint8)
    return out


def _build_expanded_shelf_cabinet_fg_rgba(
    raw_shelf_img: Image.Image,
    fg_body_source_rgba: np.ndarray,
    src_h: int,
    exp_dims: CanvasDimensionsConfig,
    spec: ShelfPortalSceneSpec,
) -> Image.Image:
    """
    Constructs the full-perspective 2700x1800 (`fgExpanded: true`) freestanding concrete shelf wall:
      - Places the ENTIRE 7-row `IMG_BINY_0007` cabinet (`y_src = 0..1472`, including the yellow
        architectural model crown with SAM-cut ceiling pockets, all 7 concrete & green shelf rows,
        wooden chest, down to the plinth) inside `cabinet_body_box_xyxy` with zero top frame/halo.
      - Adds a thin (`5px`) uniform poured-concrete side cheek (rotated 90 deg from the uniform
        middle concrete beam) strictly below the yellow rooftop crown (`y_src >= 82`) so left-edge
        diorama apertures have a solid physical side wall without boxing in the yellow crown.
    """
    cab_x0, cab_y0, cab_x1, cab_y1 = spec.cabinet_body_box_xyxy
    cab_w = cab_x1 - cab_x0
    cab_h = cab_y1 - cab_y0
    cheek_w = spec.cabinet_side_cheek_width_px
    cheek_y0 = cab_y0 + int(
        round(float(cab_h) * float(spec.cabinet_side_cheek_start_src_y) / float(max(1, src_h)))
    )
    cheek_h = max(1, cab_y1 - cheek_y0)

    body_resized = _resize_rgba_premultiplied(fg_body_source_rgba, (cab_w, cab_h)).astype(np.float32)
    feather_px = max(1, spec.outer_edge_feather_px)
    crown_h = max(1, cheek_y0 - cab_y0)
    # Trim the camera-cropped yellow upper fragment (`y_src < cabinet_side_cheek_start_src_y = 88`)
    # so the freestanding shelf wall terminates cleanly at its physical horizontal cream top cornice.
    body_resized[:crown_h, :, 3] = 0.0
    cornice_top_taper = _cosine_feather_2d(
        height=cheek_h,
        width=cab_w,
        left_px=0,
        top_px=feather_px,
        right_px=0,
        bottom_px=0,
        elliptical=False,
    )
    body_resized[crown_h:, :, 3] *= cornice_top_taper

    out_rgba = np.zeros((exp_dims.height, exp_dims.width, 4), dtype=np.float32)
    out_rgba[cab_y0:cab_y1, cab_x0:cab_x1] = body_resized

    if cheek_w > 0:
        beam_vert = raw_shelf_img.crop(spec.concrete_beam_sample_xyxy).transpose(
            Image.Transpose.ROTATE_90
        )
        cheek_rgb = np.clip(
            np.asarray(
                beam_vert.resize((cheek_w, cheek_h), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            * spec.cabinet_side_cheek_gain,
            0.0,
            255.0,
        )
        cheek_alpha = (
            _cosine_feather_2d(
                height=cheek_h,
                width=cheek_w,
                left_px=feather_px,
                top_px=feather_px,
                right_px=0,
                bottom_px=feather_px,
                elliptical=False,
            )
            * 255.0
        )
        full_x0 = cab_x0 - cheek_w
        full_x1 = cab_x1 + cheek_w
        out_rgba[cheek_y0:cab_y1, full_x0:cab_x0, :3] = cheek_rgb
        out_rgba[cheek_y0:cab_y1, full_x0:cab_x0, 3] = cheek_alpha
        out_rgba[cheek_y0:cab_y1, cab_x1:full_x1, :3] = cheek_rgb[:, ::-1, :]
        out_rgba[cheek_y0:cab_y1, cab_x1:full_x1, 3] = cheek_alpha[:, ::-1]
    else:
        full_x0 = cab_x0
        full_x1 = cab_x1

    # Softly anti-alias bottom plinth contact row (`cab_y1 - 1`)
    out_rgba[cab_y1 - 1, full_x0:full_x1, 3] = np.minimum(
        out_rgba[cab_y1 - 1, full_x0:full_x1, 3], 180.0
    )

    return Image.fromarray(np.clip(np.round(out_rgba), 0, 255).astype(np.uint8), mode="RGBA")


def _build_expanded_shelf_gallery_room_bg_rgb(
    project_root: Path,
    exp_dims: CanvasDimensionsConfig,
    spec: ShelfPortalSceneSpec,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
    roi_context_margin_px: int,
) -> Image.Image:
    """
    Constructs the full-perspective 2700x1800 Gallery Binyamin room background (`IMG_BINY_0015`):
      - Segments ONLY the central steel U-handrail and protruding tow-hitch tongue/red cloth via
        Meta SAM (`gallery_room_heal_sam_prompts`), preserving the wall/baseboard inside the arch
        and Tami's green trolley plank deck (`x >= gallery_room_heal_max_x_px`).
      - Heals at native 1:1 ROI resolution via neural LaMa with an overlap-collar low-frequency
        luminance/chroma harmonizer and zero baked rectangular wall/floor shadow boxes, eliminating
        all rectangular compositing halos around the freestanding cabinet.
    """
    room_raw = Image.open(project_root / spec.gallery_room_relpath).convert("RGB")
    room_2700 = room_raw.crop(spec.gallery_room_crop_xyxy).resize(
        (exp_dims.width, exp_dims.height), Image.Resampling.LANCZOS
    )
    if not spec.gallery_room_heal_sam_prompts:
        return room_2700

    room_embeddings = _encode_sam_image_embeddings(sam_extractor, room_2700)
    raw_sam_u8 = _decode_sam_with_cached_embeddings(
        sam_extractor=sam_extractor,
        rgb_image=room_2700,
        image_embeddings=room_embeddings,
        point_groups=spec.gallery_room_heal_sam_prompts,
        max_pinhole_fill_px=600,
        min_major_component_px=120,
    )
    prompt_gate = np.zeros((exp_dims.height, exp_dims.width), dtype=bool)
    for pg in spec.gallery_room_heal_sam_prompts:
        prompt_gate[pg.box.y0 : pg.box.y1, pg.box.x0 : pg.box.x1] = True
    prompt_gate[:, spec.gallery_room_heal_max_x_px :] = False
    obj_mask_u8 = np.where(prompt_gate & (raw_sam_u8 >= 96), 255, 0).astype(np.uint8)

    r_core = max(1, spec.bg_lama_dilation_px)
    r_lama = r_core + max(4, spec.bg_lama_overlap_outer_pad_px)
    core_mask_u8 = cv2.dilate(
        obj_mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_core + 1, 2 * r_core + 1)),
    )
    lama_mask_u8 = cv2.dilate(
        obj_mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_lama + 1, 2 * r_lama + 1)),
    )
    core_mask_u8[:, spec.gallery_room_heal_max_x_px :] = 0
    lama_mask_u8[:, spec.gallery_room_heal_max_x_px :] = 0

    ys, xs = np.where(lama_mask_u8 > 0)
    if xs.size == 0:
        return room_2700

    rx0 = max(0, int(xs.min()) - roi_context_margin_px)
    ry0 = max(0, int(ys.min()) - roi_context_margin_px)
    rx1 = min(exp_dims.width, int(xs.max()) + 1 + roi_context_margin_px)
    ry1 = min(exp_dims.height, int(ys.max()) + 1 + roi_context_margin_px)

    src_np = np.asarray(room_2700, dtype=np.float32)
    src_roi = src_np[ry0:ry1, rx0:rx1]
    core_roi = core_mask_u8[ry0:ry1, rx0:rx1]
    lama_roi = lama_mask_u8[ry0:ry1, rx0:rx1]

    healed_roi = np.asarray(
        lama_inpainter.inpaint_background(
            room_2700.crop((rx0, ry0, rx1, ry1)),
            lama_roi,
            dilation_radius_px=0,
            mask_blur_sigma=0.0,
        ),
        dtype=np.float32,
    )

    # Harmonize low-frequency tone/color across the clean overlap collar (`core_roi == 0` inside `lama_roi > 0`)
    overlap_w = ((lama_roi > 0) & (core_roi == 0)).astype(np.float32)
    if spec.bg_lama_color_match_sigma > 0.0 and np.any(overlap_w > 0.0):
        denom = np.maximum(
            1e-4, ndi.gaussian_filter(overlap_w, sigma=spec.bg_lama_color_match_sigma)
        )
        delta_smooth = np.stack(
            [
                ndi.gaussian_filter(
                    (src_roi[:, :, ch] - healed_roi[:, :, ch]) * overlap_w,
                    sigma=spec.bg_lama_color_match_sigma,
                )
                / denom
                for ch in range(3)
            ],
            axis=2,
        )
        healed_roi = np.clip(healed_roi + delta_smooth, 0.0, 255.0)

    eroded_lama = cv2.erode(
        lama_roi,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    blend_a = ndi.gaussian_filter(
        core_roi.astype(np.float32) / 255.0,
        sigma=max(1.0, spec.bg_lama_blur_sigma),
    )
    blend_a = np.where(eroded_lama > 0, np.clip(blend_a * 1.16, 0.0, 1.0), 0.0)[:, :, None]

    out_np = src_np.copy()
    out_np[ry0:ry1, rx0:rx1] = blend_a * healed_roi + (1.0 - blend_a) * src_roi
    return Image.fromarray(np.clip(np.round(out_np), 0, 255).astype(np.uint8), mode="RGB")


def build_shelf_portal_scene(
    project_root: Path,
    cfg: NewScenesBundleConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
    seamless_felt_reference_rgb: Image.Image,
) -> dict[str, Any]:
    spec = cfg.shelf_portal
    inner_dims = cfg.inner_dimensions
    exp_dims = cfg.expanded_dimensions
    scenes_dir = project_root / cfg.scenes_output_relpath
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Scene 9: {spec.scene_id}] Segmenting {len(spec.portal_cubbies)} full-wall SAM portal cubbies + "
        f"removable occluders (cabinet={spec.cabinet_body_box_xyxy} on {exp_dims.width}x{exp_dims.height})..."
    )
    raw_img = Image.open(project_root / spec.source_relpath).convert("RGB")
    shelf_rgb = raw_img.crop(spec.source_crop_xyxy)
    src_w, src_h = shelf_rgb.size
    shelf_np = np.asarray(shelf_rgb, dtype=np.uint8)
    shelf_embeddings = _encode_sam_image_embeddings(sam_extractor, shelf_rgb)

    fg_cutout_union = np.zeros((src_h, src_w), dtype=np.float32)
    portal_union_bool = np.zeros((src_h, src_w), dtype=bool)
    cubby_coord_records: list[dict[str, Any]] = []
    runtime_portal_shelves: list[dict[str, Any]] = []

    close_k = spec.aperture_close_kernel_px
    if close_k % 2 == 0:
        close_k += 1
    close_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    inset_px = spec.aperture_interior_inset_px

    # 1. Segment each of the 9 portal cubbies AND removable obscuring objects via Meta SAM
    for cubby in spec.portal_cubbies:
        ab = cubby.aperture_bounds_px
        roi_gate = np.zeros((src_h, src_w), dtype=bool)
        roi_gate[ab.y0 : ab.y1, ab.x0 : ab.x1] = True

        back_sam_u8 = _decode_sam_with_cached_embeddings(
            sam_extractor=sam_extractor,
            rgb_image=shelf_rgb,
            image_embeddings=shelf_embeddings,
            point_groups=(cubby.backplane_sam_prompt,),
            max_pinhole_fill_px=1800,
            min_major_component_px=220,
        )

        occ_u8 = np.zeros((src_h, src_w), dtype=np.uint8)
        if cubby.occluder_sam_prompt is not None:
            raw_occ_u8 = _decode_sam_with_cached_embeddings(
                sam_extractor=sam_extractor,
                rgb_image=shelf_rgb,
                image_embeddings=shelf_embeddings,
                point_groups=(cubby.occluder_sam_prompt,),
                max_pinhole_fill_px=1400,
                min_major_component_px=140,
            )
            ob = cubby.occluder_sam_prompt.box
            occ_gate = np.zeros((src_h, src_w), dtype=bool)
            gx0 = max(0, max(ob.x0, ab.x0 - 2))
            gy0 = max(0, max(ob.y0, ab.y0 - 2))
            gx1 = min(src_w, min(ob.x1, ab.x1 + 2))
            gy1 = min(src_h, min(ob.y1, ab.y1 + 3))
            occ_gate[gy0:gy1, gx0:gx1] = True
            occ_u8 = np.where(occ_gate, raw_occ_u8, 0).astype(np.uint8)

        # Cut BOTH the felt/green backing AND the obscuring object out of `fg`, closing any shadow seam
        combined_inside = np.where(roi_gate, np.maximum(back_sam_u8, occ_u8), 0).astype(np.uint8)
        closed_inside = cv2.morphologyEx(combined_inside, cv2.MORPH_CLOSE, close_se)
        # Guarantee zero stray shadow/dust islands inside the strict interior of the cubby aperture
        if (ab.y1 - ab.y0) > 2 * inset_px and (ab.x1 - ab.x0) > 2 * inset_px:
            closed_inside[
                ab.y0 + inset_px : ab.y1 - inset_px, ab.x0 + inset_px : ab.x1 - inset_px
            ] = 255
        cutout_f = np.where(roi_gate, closed_inside.astype(np.float32), 0.0)
        portal_mask_u8 = np.clip(np.round(cutout_f), 0, 255).astype(np.uint8)

        occluder_meta: dict[str, Any] | None = None
        if cubby.occluder_sam_prompt is not None:
            occluder_meta = _export_portal_occluder_sprite(
                cubby=cubby,
                shelf_np=shelf_np,
                occluder_alpha_u8=occ_u8,
                portal_mask_u8=portal_mask_u8,
                scenes_dir=scenes_dir,
                inner_dims=inner_dims,
                exp_dims=exp_dims,
                spec=spec,
            )

        fg_cutout_union = np.maximum(fg_cutout_union, cutout_f)
        portal_union_bool |= portal_mask_u8 > 64

        coord_rec, runtime_spec = _compute_portal_cubby_webgl_coordinates(
            cubby=cubby,
            actual_cutout_mask=portal_mask_u8,
            src_w=src_w,
            src_h=src_h,
            inner_dims=inner_dims,
            exp_dims=exp_dims,
            spec=spec,
            occluder_runtime_meta=occluder_meta,
        )
        cubby_coord_records.append(coord_rec)
        runtime_portal_shelves.append(runtime_spec)

    # 2. Segment non-portal recessed pockets (3 hollow-block holes + empty felt cubbies)
    recessed_sam_u8 = _decode_sam_with_cached_embeddings(
        sam_extractor=sam_extractor,
        rgb_image=shelf_rgb,
        image_embeddings=shelf_embeddings,
        point_groups=spec.recessed_mid_sam_prompts,
        max_pinhole_fill_px=600,
        min_major_component_px=80,
    )
    recessed_gate = np.zeros((src_h, src_w), dtype=bool)
    for rp in spec.recessed_mid_sam_prompts:
        recessed_gate[rp.box.y0 : rp.box.y1, rp.box.x0 : rp.box.x1] = True
    recessed_cutout_f = np.where(
        recessed_gate & (~portal_union_bool), recessed_sam_u8.astype(np.float32), 0.0
    )

    # 2b. Cut distant ceiling/spotlight pockets behind the yellow rooftop architectural model crown
    crown_cutout_f = np.zeros((src_h, src_w), dtype=np.float32)
    if spec.crown_ceiling_cutout_sam_prompts:
        crown_sam_u8 = _decode_sam_with_cached_embeddings(
            sam_extractor=sam_extractor,
            rgb_image=shelf_rgb,
            image_embeddings=shelf_embeddings,
            point_groups=spec.crown_ceiling_cutout_sam_prompts,
            max_pinhole_fill_px=1200,
            min_major_component_px=40,
        )
        crown_gate = np.zeros((src_h, src_w), dtype=bool)
        for cp in spec.crown_ceiling_cutout_sam_prompts:
            crown_gate[cp.box.y0 : cp.box.y1, cp.box.x0 : cp.box.x1] = True
        crown_k = spec.crown_ceiling_close_kernel_px
        if crown_k % 2 == 0:
            crown_k += 1
        crown_closed = cv2.morphologyEx(
            np.where(crown_gate, crown_sam_u8, 0).astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (crown_k, crown_k)),
        )
        crown_cutout_f = np.where(crown_gate, crown_closed.astype(np.float32), 0.0)

    total_fg_cutout = np.maximum(np.maximum(fg_cutout_union, recessed_cutout_f), crown_cutout_f)
    if spec.cutout_edge_feather_sigma > 0:
        total_fg_cutout = ndi.gaussian_filter(total_fg_cutout, sigma=spec.cutout_edge_feather_sigma)
    fg_alpha_src_u8 = np.clip(np.round(255.0 - total_fg_cutout), 0, 255).astype(np.uint8)
    fg_rgb_clean_src = decontaminate_matte_edge_rgb(shelf_np, fg_alpha_src_u8)
    fg_body_source_rgba = np.dstack([fg_rgb_clean_src, fg_alpha_src_u8])

    # 3. Build `fg` RGBA (2700x1800, `fgExpanded: true`, freestanding full-height cabinet)
    fg_rgba = _build_expanded_shelf_cabinet_fg_rgba(
        raw_shelf_img=raw_img,
        fg_body_source_rgba=fg_body_source_rgba,
        src_h=src_h,
        exp_dims=exp_dims,
        spec=spec,
    )

    # 4. Build `mid` RGBA (1500x1000 center window of 2700x1800, `border_alpha = 0` on all 4 edges)
    mid_support_src = cv2.dilate(
        np.clip(np.round(recessed_cutout_f), 0, 255).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
    )
    portal_guard_src = cv2.dilate(
        (portal_union_bool.astype(np.uint8)) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    mid_alpha_src_u8 = np.where(portal_guard_src > 0, 0, mid_support_src).astype(np.uint8)
    p_guard = spec.mid_perimeter_guard_px
    if p_guard > 0:
        mid_alpha_src_u8[:, :p_guard] = 0
        mid_alpha_src_u8[:, max(0, src_w - p_guard) :] = 0
        mid_alpha_src_u8[:p_guard, :] = 0
        mid_alpha_src_u8[max(0, src_h - p_guard) :, :] = 0

    dark_felt_src = np.clip(
        np.asarray(
            seamless_felt_reference_rgb.resize((src_w, src_h), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        * spec.bg_dark_felt_gain,
        0.0,
        255.0,
    ).astype(np.uint8)
    mid_rgb_src = np.where((recessed_cutout_f > 96.0)[:, :, None], shelf_np, dark_felt_src)
    cab_x0, cab_y0, cab_x1, cab_y1 = spec.cabinet_body_box_xyxy
    mid_body_resized = _resize_rgba_premultiplied(
        np.dstack([mid_rgb_src, mid_alpha_src_u8]),
        (cab_x1 - cab_x0, cab_y1 - cab_y0),
    )
    mid_exp_canvas = np.zeros((exp_dims.height, exp_dims.width, 4), dtype=np.uint8)
    mid_exp_canvas[cab_y0:cab_y1, cab_x0:cab_x1] = mid_body_resized
    pad_x = (exp_dims.width - inner_dims.width) // 2
    pad_y = (exp_dims.height - inner_dims.height) // 2
    mid_inner_np = mid_exp_canvas[pad_y : pad_y + inner_dims.height, pad_x : pad_x + inner_dims.width].copy()
    mid_taper = _cosine_feather_2d(
        height=inner_dims.height,
        width=inner_dims.width,
        left_px=16,
        top_px=16,
        right_px=16,
        bottom_px=16,
        elliptical=False,
    )
    mid_inner_np[:, :, 3] = np.clip(
        np.round(mid_inner_np[:, :, 3].astype(np.float32) * mid_taper), 0, 255
    ).astype(np.uint8)
    mid_rgba = Image.fromarray(mid_inner_np, mode="RGBA")
    _assert_zero_border_alpha(mid_rgba, f"{spec.scene_id}.mid")

    # 5. Build `bg` RGB (2700x1800 wide-angle Gallery Binyamin room, SAM+LaMa healed with zero shadow box)
    bg_img = _build_expanded_shelf_gallery_room_bg_rgb(
        project_root=project_root,
        exp_dims=exp_dims,
        spec=spec,
        sam_extractor=sam_extractor,
        lama_inpainter=lama_inpainter,
        roi_context_margin_px=cfg.lama_roi_context_margin_px,
    )

    # 6. Predict real Depth Anything V2 maps (`0=far, 255=near`)
    fg_depth_img = _compute_alpha_supported_depth_u8(
        layer_rgba=fg_rgba,
        neutral_backdrop_rgb=(128.0, 126.0, 124.0),
        depth_estimator=depth_estimator,
        min_fg_depth_u8=spec.layer_depth_min_u8,
        collar_sigma=spec.layer_depth_collar_sigma,
    )
    mid_depth_img = _compute_alpha_supported_depth_u8(
        layer_rgba=mid_rgba,
        neutral_backdrop_rgb=(28.0, 27.0, 29.0),
        depth_estimator=depth_estimator,
        min_fg_depth_u8=spec.layer_depth_min_u8,
        collar_sigma=spec.layer_depth_collar_sigma,
    )
    bg_depth_raw_u8 = np.asarray(
        depth_estimator.build_smoothed_room_depth_uint8(bg_img), dtype=np.float32
    )
    # Law 1 & Law 4: Guarantee `layer0` meets the cabinet floor plinth (`cab_y1`) at `Z_world == scene_fg_z`
    # while remaining strictly behind `scene_mid_z` (`-296.0`) inside the upper cabinet footprint.
    plinth_target_u8 = float(
        np.clip(
            ((spec.scene_fg_z - spec.scene_bg_z) / max(1e-5, spec.room_depth_amplitude)) * 255.0,
            0.0,
            255.0,
        )
    )
    upper_max_u8 = float(
        np.clip(
            (((spec.scene_mid_z - 8.0) - spec.scene_bg_z) / max(1e-5, spec.room_depth_amplitude))
            * 255.0,
            0.0,
            255.0,
        )
    )
    cheek_w = spec.cabinet_side_cheek_width_px
    cheek_y0 = cab_y0 + int(
        round(float(cab_y1 - cab_y0) * float(spec.cabinet_side_cheek_start_src_y) / float(max(1, src_h)))
    )
    cab_full_x0 = max(0, cab_x0 - cheek_w)
    cab_full_x1 = min(exp_dims.width, cab_x1 + cheek_w)

    upper_gate = np.zeros((exp_dims.height, exp_dims.width), dtype=np.float32)
    upper_gate[cheek_y0 : max(cheek_y0 + 1, cab_y1 - 18), cab_full_x0:cab_full_x1] = 1.0
    upper_gate = ndi.gaussian_filter(upper_gate, sigma=8.0)
    bg_depth_adj = upper_gate * np.minimum(bg_depth_raw_u8, upper_max_u8) + (
        1.0 - upper_gate
    ) * bg_depth_raw_u8

    plinth_w = np.zeros((exp_dims.height, exp_dims.width), dtype=np.float32)
    plinth_w[max(0, cab_y1 - 4) : min(exp_dims.height, cab_y1 + 5), cab_full_x0:cab_full_x1] = 1.0
    plinth_w = ndi.gaussian_filter(plinth_w, sigma=5.0)
    plinth_w /= max(1e-5, float(plinth_w[cab_y1 - 1, (cab_full_x0 + cab_full_x1) // 2]))
    plinth_w = np.clip(plinth_w, 0.0, 1.0)
    bg_depth_adj = plinth_w * plinth_target_u8 + (1.0 - plinth_w) * bg_depth_adj
    bg_depth_img = Image.fromarray(np.clip(np.round(bg_depth_adj), 0, 255).astype(np.uint8), mode="L")

    pfx = spec.asset_prefix
    bg_img.save(scenes_dir / f"{pfx}_bg.png", format="PNG")
    bg_depth_img.save(scenes_dir / f"{pfx}_bg_depth.png", format="PNG")
    mid_rgba.save(scenes_dir / f"{pfx}_mid.png", format="PNG")
    mid_depth_img.save(scenes_dir / f"{pfx}_mid_depth.png", format="PNG")
    fg_rgba.save(scenes_dir / f"{pfx}_fg.png", format="PNG")
    fg_depth_img.save(scenes_dir / f"{pfx}_fg_depth.png", format="PNG")

    fg_exp_alpha = np.asarray(fg_rgba)[:, :, 3]
    fg_cov = float((fg_exp_alpha > 16).mean())
    cab_alpha = fg_exp_alpha[cab_y0:cab_y1, cab_x0:cab_x1]
    portal_open_ratio = float((cab_alpha < 16).mean())
    mid_cov = float((np.asarray(mid_rgba)[:, :, 3] > 16).mean())
    locked_count = sum(1 for s in runtime_portal_shelves if "occluder" in s)
    print(
        f"  -> Saved {pfx}_*.png + {locked_count} occluder sprites "
        f"(fg_exp_cov={fg_cov:.4f}, cabinet_open_ratio={portal_open_ratio:.4f}, mid_cov={mid_cov:.4f})."
    )

    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "shortLabelHe": spec.short_label_he,
        "icon": spec.icon,
        "portalLayoutVersion": spec.portal_layout_version,
        "sourceCatalogId": spec.source_catalog_id,
        "sourceRelpath": spec.source_relpath,
        "galleryRoomCatalogId": spec.gallery_room_catalog_id,
        "galleryRoomRelpath": spec.gallery_room_relpath,
        "sourceCropXYXY": list(spec.source_crop_xyxy),
        "cabinetBodyBoxXYXY": list(spec.cabinet_body_box_xyxy),
        "innerWidth": inner_dims.width,
        "innerHeight": inner_dims.height,
        "expandedWidth": exp_dims.width,
        "expandedHeight": exp_dims.height,
        "bgExpandedWidth": exp_dims.width,
        "bgExpandedHeight": exp_dims.height,
        "bgExpansionFactor": cfg.expanded_bg_scale,
        "fgExpanded": True,
        "boreSightAtRest": True,
        "sceneBgZ": spec.scene_bg_z,
        "midZ": spec.scene_mid_z,
        "fgZ": spec.scene_fg_z,
        "roomDepthAmplitude": spec.room_depth_amplitude,
        "initialCameraY": spec.initial_camera_y,
        "initialCameraZ": spec.initial_camera_z,
        "minCameraZ": spec.min_camera_z,
        "maxCameraZ": spec.max_camera_z,
        "referenceWebglPlane": {
            "width": spec.webgl_plane_width,
            "height": spec.webgl_plane_height,
        },
        "bgTexture": f"assets/scenes/{pfx}_bg.png",
        "depthTexture": f"assets/scenes/{pfx}_bg_depth.png",
        "midTexture": f"assets/scenes/{pfx}_mid.png",
        "midDepthTexture": f"assets/scenes/{pfx}_mid_depth.png",
        "fgTexture": f"assets/scenes/{pfx}_fg.png",
        "fgDepthTexture": f"assets/scenes/{pfx}_fg_depth.png",
        "fgCoverageRatio": round(fg_cov, 4),
        "midCoverageRatio": round(mid_cov, 4),
        "portalTransparentRatio": round(portal_open_ratio, 4),
        "portalShelves": runtime_portal_shelves,
        "portalCubbies": cubby_coord_records,
    }


# ============================================================================
# Top-Level Pipeline Orchestrator
# ============================================================================


_ALL_SCENE_IDS: tuple[str, ...] = ("wandering_drawing_1", "denya_model", "shelf_portal")


def build_all_new_scenes(
    project_root: Path | None = None,
    bundle_cfg: NewScenesBundleConfig | None = None,
    selected_scenes: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parent.parent
    cfg = bundle_cfg or default_new_scenes_bundle_config()

    active_ids = tuple(selected_scenes) if selected_scenes else _ALL_SCENE_IDS
    unknown_ids = [sid for sid in active_ids if sid not in _ALL_SCENE_IDS]
    if unknown_ids:
        raise ValueError(
            f"Unknown scene IDs {unknown_ids}. Valid IDs: {list(_ALL_SCENE_IDS)}."
        )

    # Verify lossless round-trip serialization of typed config tree before running heavy models
    roundtrip_cfg = NewScenesBundleConfig.from_json(cfg.to_json())
    if roundtrip_cfg != cfg:
        raise RuntimeError("NewScenesBundleConfig failed lossless JSON round-trip check.")

    spec_out_path = Path(cfg.spec_output_abspath)
    merged_scenes: dict[str, Any] = {}
    if spec_out_path.exists():
        prev_payload = json.loads(spec_out_path.read_text(encoding="utf-8"))
        if isinstance(prev_payload.get("scenes"), dict):
            merged_scenes.update(prev_payload["scenes"])

    device = select_torch_device()
    print(f"Initializing neural computer-vision models on device='{device}' (scenes={list(active_ids)})...")
    depth_estimator = DepthAnythingEstimator(cfg.depth_model, device=device)
    sam_extractor = SamBoxMaskExtractor(cfg.segmentation)
    lama_inpainter = LamaOcclusionInpainter(cfg.inpainting)

    if "wandering_drawing_1" in active_ids:
        scene7_summary = build_wandering_drawing_1_scene(
            project_root=root,
            cfg=cfg,
            depth_estimator=depth_estimator,
            sam_extractor=sam_extractor,
            lama_inpainter=lama_inpainter,
        )
        merged_scenes[scene7_summary["id"]] = scene7_summary

    if "denya_model" in active_ids:
        matte_extractor = SalientMatteExtractor(cfg.segmentation)
        scene8_summary = build_denya_model_scene(
            project_root=root,
            cfg=cfg,
            depth_estimator=depth_estimator,
            sam_extractor=sam_extractor,
            matte_extractor=matte_extractor,
            lama_inpainter=lama_inpainter,
        )
        merged_scenes[scene8_summary["id"]] = scene8_summary

    if "shelf_portal" in active_ids:
        felt_ref_path = (
            root / cfg.scenes_output_relpath / f"{cfg.denya_model.asset_prefix}_bg.png"
        )
        if felt_ref_path.exists():
            seamless_felt_ref = Image.open(felt_ref_path).convert("RGB")
        else:
            seamless_felt_ref = _synthesize_seamless_felt_backdrop_lama(
                project_root=root,
                dims=cfg.expanded_dimensions,
                spec=cfg.denya_model,
                lama_inpainter=lama_inpainter,
            )
        scene9_summary = build_shelf_portal_scene(
            project_root=root,
            cfg=cfg,
            depth_estimator=depth_estimator,
            sam_extractor=sam_extractor,
            lama_inpainter=lama_inpainter,
            seamless_felt_reference_rgb=seamless_felt_ref,
        )
        merged_scenes[scene9_summary["id"]] = scene9_summary

    spec_payload = {
        "schemaVersion": "1.0",
        "generatedBy": "prototype/new_scenes_builder.py",
        "bundleConfig": cfg.to_dict(),
        "scenes": merged_scenes,
    }

    spec_out_path.parent.mkdir(parents=True, exist_ok=True)
    spec_out_path.write_text(
        json.dumps(spec_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nExported typed JSON scene spec & portal coordinates -> {spec_out_path}")
    return spec_payload


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build neural 2.5D spatial scene layers and WebGL portal specs."
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="all",
        help="Comma-separated scene IDs to build (`wandering_drawing_1,denya_model,shelf_portal`) or `all`.",
    )
    parser.add_argument(
        "--only-shelf-portal",
        action="store_true",
        help="Fast rebuild of `shelf_portal` (`scene9_shelf_portal_*` + SAM occluder sprites) only.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = _parse_cli_args()
    if cli_args.only_shelf_portal:
        selected: Sequence[str] | None = ("shelf_portal",)
    elif cli_args.scenes.strip().lower() == "all":
        selected = None
    else:
        selected = [s.strip() for s in cli_args.scenes.split(",") if s.strip()]
    build_all_new_scenes(selected_scenes=selected)
