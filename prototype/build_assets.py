#!/usr/bin/env python3
"""
prototype/build_assets.py
Offline neural 2.5D spatial asset builder for Tamar "Tami" Sheaffer's 70th Birthday Prototype.
Executes real Depth Anything V2 monocular depth estimation, IS-Net & Meta SAM neural sculpture
segmentation, LaMa Fast Fourier Convolution occlusion inpainting, and homographic wide-angle
outpaint registration & center-locked blending.

Usage:
    /Users/talsheaffer/code/tami/.venv/bin/python prototype/build_assets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from prototype.scene_config import (
    SceneBuildSpec,
    SpatialPipelineConfig,
    default_spatial_pipeline_config,
)
from prototype.vision_primitives import (
    DepthAnythingEstimator,
    LamaOcclusionInpainter,
    SalientMatteExtractor,
    SamBoxMaskExtractor,
    build_aperture_reflection_layer,
    compose_outpainted_room,
    partition_sculpture_layers,
)


def build_single_scene(
    project_root: Path,
    output_dir: Path,
    cfg: SpatialPipelineConfig,
    spec: SceneBuildSpec,
    depth_estimator: DepthAnythingEstimator,
    matte_extractor: SalientMatteExtractor | None,
    sam_extractor: SamBoxMaskExtractor | None,
    lama_inpainter: LamaOcclusionInpainter | None,
    reuse_inner_layers: bool = False,
) -> dict[str, Any]:
    """Synthesizes all verified multi-plane, outpainted room, and depth textures for a single scene."""
    src_path = project_root / spec.source_image_relpath
    if not src_path.is_file():
        raise FileNotFoundError(f"Required source photograph does not exist: {src_path}")

    outpaint_raw_path = project_root / spec.outpaint_raw_relpath
    if not outpaint_raw_path.is_file():
        raise FileNotFoundError(f"Required wide-angle outpaint source missing: {outpaint_raw_path}")

    fg_path = output_dir / spec.fg_output_filename
    mid_path = output_dir / spec.mid_output_filename
    lama_inner_path = output_dir / spec.lama_inner_output_filename

    if reuse_inner_layers and fg_path.is_file() and mid_path.is_file() and lama_inner_path.is_file():
        print(f"[{spec.scene_id}] Reusing verified inner sculpture cutouts & LaMa room ({lama_inner_path.name})")
        fg_img = Image.open(fg_path).convert("RGBA")
        mid_img = Image.open(mid_path).convert("RGBA")
        lama_inner_rgb = Image.open(lama_inner_path).convert("RGB")
    else:
        if matte_extractor is None or sam_extractor is None or lama_inpainter is None:
            raise RuntimeError("Segmentation/LaMa models required when reuse_inner_layers is not satisfied.")

        target_size = (cfg.dimensions.width, cfg.dimensions.height)
        print(f"[{spec.scene_id}] Loading source photograph: {src_path.name}")
        base_rgb = Image.open(src_path).convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
        rgb_np = np.asarray(base_rgb, dtype=np.uint8)

        if spec.segmentation_backend == "isnet":
            print(f"[{spec.scene_id}] Running Depth Anything V2 ({cfg.depth_model.model_id})...")
            scene_depth = depth_estimator.predict_depth(base_rgb)
            print(f"[{spec.scene_id}] Running IS-Net neural matting ({cfg.segmentation.rembg_model_name})...")
            alpha_u8 = matte_extractor.extract_alpha(base_rgb)
            print(f"[{spec.scene_id}] Partitioning Foreground (Z=0) & Midground (Z=-150) planes...")
            fg_img, mid_img, union_mask_u8 = partition_sculpture_layers(
                rgb_np=rgb_np,
                alpha_u8=alpha_u8,
                depth_norm=scene_depth,
                seg_cfg=cfg.segmentation,
                spec=spec,
            )
            if spec.sam_fg_boxes or spec.sam_mid_boxes:
                print(
                    f"[{spec.scene_id}] Supplementing translucent details via Meta SAM "
                    f"({cfg.segmentation.sam_model_id})..."
                )
                fg_img, mid_img, union_mask_u8 = sam_extractor.supplement_rgba_layers(
                    base_rgb, fg_img, mid_img, union_mask_u8, spec
                )
        elif spec.segmentation_backend == "sam_boxes":
            print(
                f"[{spec.scene_id}] Running Meta SAM promptable neural segmentation "
                f"({cfg.segmentation.sam_model_id})..."
            )
            fg_img, mid_img, union_mask_u8 = sam_extractor.extract_fg_mid_layers(base_rgb, spec)
        else:
            raise ValueError(f"Unsupported segmentation_backend: {spec.segmentation_backend}")

        fg_img.save(fg_path, format="PNG", optimize=True)
        mid_img.save(mid_path, format="PNG", optimize=True)

        print(
            f"[{spec.scene_id}] Running neural LaMa FFC inpainting "
            f"(dilation r={cfg.inpainting.dilation_radius_px}px)..."
        )
        lama_inner_rgb = lama_inpainter.inpaint_background(base_rgb, union_mask_u8)
        lama_inner_rgb.save(
            lama_inner_path, format="JPEG", quality=cfg.inpainting.jpeg_quality, optimize=True
        )
        print(f"  -> Saved {lama_inner_path.name}")

    fg_coverage = float((np.asarray(fg_img)[:, :, 3] > 16).mean())
    mid_coverage = float((np.asarray(mid_img)[:, :, 3] > 16).mean())

    print(
        f"[{spec.scene_id}] Registering & center-locking {cfg.outpainting.expansion_factor:.2f}x "
        f"outpainted gallery room ({outpaint_raw_path.name})..."
    )
    raw_outpaint_rgb = Image.open(outpaint_raw_path).convert("RGB")
    expanded_bg_rgb, reg_corr = compose_outpainted_room(
        inner_rgb=lama_inner_rgb,
        raw_outpaint_rgb=raw_outpaint_rgb,
        dims=cfg.dimensions,
        outpaint_cfg=cfg.outpainting,
    )
    bg_path = output_dir / spec.bg_output_filename
    expanded_bg_rgb.save(bg_path, format="JPEG", quality=cfg.inpainting.jpeg_quality, optimize=True)
    exp_w, exp_h = expanded_bg_rgb.size
    print(f"  -> Saved {bg_path.name} ({exp_w}x{exp_h}, homography corr={reg_corr:.4f})")

    print(f"[{spec.scene_id}] Computing Depth Anything V2 room geometry depth on {exp_w}x{exp_h} room...")
    room_depth_img = depth_estimator.build_smoothed_room_depth_uint8(expanded_bg_rgb)
    depth_path = output_dir / spec.depth_output_filename
    room_depth_img.save(depth_path, format="PNG", optimize=True)
    print(f"  -> Saved {depth_path.name} ({room_depth_img.size[0]}x{room_depth_img.size[1]})")

    reflection_filename: str | None = None
    if spec.reflection_output_filename and spec.reflection_source_relpath:
        ref_src_path = project_root / spec.reflection_source_relpath
        if not ref_src_path.is_file():
            raise FileNotFoundError(f"Required reflection source image missing: {ref_src_path}")
        ref_src_img = Image.open(ref_src_path)
        ref_layer_img = build_aperture_reflection_layer(
            canvas_width=cfg.dimensions.width,
            canvas_height=cfg.dimensions.height,
            reflection_src=ref_src_img,
            apertures=spec.reflection_apertures,
        )
        ref_out_path = output_dir / spec.reflection_output_filename
        ref_layer_img.save(ref_out_path, format="PNG", optimize=True)
        reflection_filename = spec.reflection_output_filename
        print(f"  -> Saved {ref_out_path.name} ({len(spec.reflection_apertures)} bounded apertures)")

    return {
        "id": spec.scene_id,
        "title": spec.title_he,
        "segmentationBackend": spec.segmentation_backend,
        "innerWidth": cfg.dimensions.width,
        "innerHeight": cfg.dimensions.height,
        "bgExpandedWidth": exp_w,
        "bgExpandedHeight": exp_h,
        "bgExpansionFactor": cfg.outpainting.expansion_factor,
        "outpaintRegistrationCorr": round(reg_corr, 4),
        "bgTexture": f"assets/scenes/{spec.bg_output_filename}",
        "midTexture": f"assets/scenes/{spec.mid_output_filename}",
        "fgTexture": f"assets/scenes/{spec.fg_output_filename}",
        "depthTexture": f"assets/scenes/{spec.depth_output_filename}",
        "reflectionTexture": f"assets/scenes/{reflection_filename}" if reflection_filename else None,
        "fgCoverageRatio": round(fg_coverage, 4),
        "midCoverageRatio": round(mid_coverage, 4),
    }


def run_pipeline(
    cfg: SpatialPipelineConfig,
    project_root: Path = PROJECT_ROOT,
    reuse_inner_layers: bool = False,
    rebuild_inner_scenes: frozenset[str] = frozenset(),
) -> Path:
    output_dir = project_root / cfg.output_dir_relpath
    output_dir.mkdir(parents=True, exist_ok=True)

    need_full_models = (not reuse_inner_layers) or bool(rebuild_inner_scenes)
    depth_estimator = DepthAnythingEstimator(cfg.depth_model)
    matte_extractor = SalientMatteExtractor(cfg.segmentation) if need_full_models else None
    sam_extractor = SamBoxMaskExtractor(cfg.segmentation) if need_full_models else None
    lama_inpainter = LamaOcclusionInpainter(cfg.inpainting) if need_full_models else None

    scene_records: dict[str, Any] = {}
    for spec in cfg.scenes:
        scene_reuse = reuse_inner_layers and (spec.scene_id not in rebuild_inner_scenes)
        scene_records[spec.scene_id] = build_single_scene(
            project_root=project_root,
            output_dir=output_dir,
            cfg=cfg,
            spec=spec,
            depth_estimator=depth_estimator,
            matte_extractor=matte_extractor,
            sam_extractor=sam_extractor,
            lama_inpainter=lama_inpainter,
            reuse_inner_layers=scene_reuse,
        )

    manifest = {
        "pipeline_config": cfg.to_dict(),
        "scenes": scene_records,
    }
    manifest_path = output_dir / cfg.manifest_filename
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved SSOT scene manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize neural 2.5D spatial scene layers")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a JSON SpatialPipelineConfig file",
    )
    parser.add_argument(
        "--reuse-inner-layers",
        action="store_true",
        help="Reuse existing verified inner sculpture PNGs and LaMa room plates to rebuild outpaint + depth",
    )
    parser.add_argument(
        "--rebuild-inner-scene",
        action="append",
        default=[],
        help="Scene ID whose inner segmentation and LaMa room must be rebuilt even with --reuse-inner-layers",
    )
    args = parser.parse_args()

    cfg = (
        SpatialPipelineConfig.load_json(args.config)
        if args.config is not None
        else default_spatial_pipeline_config()
    )
    run_pipeline(
        cfg,
        reuse_inner_layers=args.reuse_inner_layers,
        rebuild_inner_scenes=frozenset(args.rebuild_inner_scene),
    )


if __name__ == "__main__":
    main()
