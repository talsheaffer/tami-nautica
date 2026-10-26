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

VALID_SEGMENTATION_BACKENDS = frozenset({"isnet", "sam_boxes"})


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
    fg_x_range_norm: tuple[float, float]
    fg_min_depth_quantile: float
    sam_fg_boxes: tuple[BoundingBoxPrompt, ...]
    sam_mid_boxes: tuple[BoundingBoxPrompt, ...]
    reflection_apertures: tuple[ApertureBox, ...]

    def __post_init__(self) -> None:
        if self.segmentation_backend not in VALID_SEGMENTATION_BACKENDS:
            raise ValueError(
                f"Scene '{self.scene_id}': unsupported segmentation_backend "
                f"'{self.segmentation_backend}'. Valid options: {sorted(VALID_SEGMENTATION_BACKENDS)}"
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fg_x_range_norm"] = list(self.fg_x_range_norm)
        d["sam_fg_boxes"] = [b.to_dict() for b in self.sam_fg_boxes]
        d["sam_mid_boxes"] = [b.to_dict() for b in self.sam_mid_boxes]
        d["reflection_apertures"] = [ap.to_dict() for ap in self.reflection_apertures]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneBuildSpec:
        xr = data["fg_x_range_norm"]
        return cls(
            scene_id=str(data["scene_id"]),
            title_he=str(data["title_he"]),
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
            fg_x_range_norm=(float(xr[0]), float(xr[1])),
            fg_min_depth_quantile=float(data["fg_min_depth_quantile"]),
            sam_fg_boxes=tuple(BoundingBoxPrompt.from_dict(b) for b in data["sam_fg_boxes"]),
            sam_mid_boxes=tuple(BoundingBoxPrompt.from_dict(b) for b in data["sam_mid_boxes"]),
            reflection_apertures=tuple(
                ApertureBox.from_dict(ap) for ap in data["reflection_apertures"]
            ),
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
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialPipelineConfig:
        return cls(
            output_dir_relpath=str(data["output_dir_relpath"]),
            manifest_filename=str(data["manifest_filename"]),
            dimensions=CanvasDimensionsConfig.from_dict(data["dimensions"]),
            outpainting=OutpaintingConfig.from_dict(data["outpainting"]),
            depth_model=DepthModelConfig.from_dict(data["depth_model"]),
            segmentation=SegmentationConfig.from_dict(data["segmentation"]),
            inpainting=InpaintingConfig.from_dict(data["inpainting"]),
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
        scenes=(
            SceneBuildSpec(
                scene_id="herzliya",
                title_he="חצר הילדות בהרצליה — קפסולות הנייר (׳אולי שם יהיה יותר טוב׳)",
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
                fg_x_range_norm=(0.22, 0.62),
                fg_min_depth_quantile=0.48,
                sam_fg_boxes=(),
                sam_mid_boxes=(
                    BoundingBoxPrompt(x0=194, y0=488, x1=274, y1=604),
                ),
                reflection_apertures=(),
            ),
            SceneBuildSpec(
                scene_id="denya",
                title_he="הסטודיו בדניה ותערוכת ׳בנה ביתך׳ (מוזיאון רמת גן)",
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
                reflection_apertures=(
                    ApertureBox(x0=1218, y0=500, x1=1282, y1=570, max_alpha=125, feather_sigma=4.5),
                ),
            ),
        ),
    )
