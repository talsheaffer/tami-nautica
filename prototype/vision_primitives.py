#!/usr/bin/env python3
"""
prototype/vision_primitives.py
Upstream neural computer-vision primitives for 2.5D Layered Depth Image (LDI) synthesis:
1. Monocular depth estimation via Depth Anything V2 (HuggingFace Transformers + PyTorch MPS/CPU).
2. Filamentary salient sculpture matting via IS-Net (`rembg`).
3. Promptable architectural sculpture segmentation via Meta Segment Anything Model (`SamModel`).
4. Connected-component depth/spatial layer partitioning into Foreground & Midground RGBA planes.
5. Morphological mask dilation and neural Fast Fourier Convolution inpainting via LaMa (`big-lama.pt`).
6. Homographic wide-angle outpainting registration, affine seam color harmonization, and center-lock blending.

Strict No-Fallback Guarantee: raises explicit RuntimeError/ValueError if required neural packages,
weights, segmentation contracts, or outpainting registration correlations fail.
"""

from __future__ import annotations

from typing import Sequence

try:
    import cv2
    import numpy as np
    import scipy.ndimage as ndi
    import torch
    from PIL import Image
    from rembg import new_session, remove
    from simple_lama_inpainting.models.model import LAMA_MODEL_URL
    from simple_lama_inpainting.utils import download_model, prepare_img_and_mask
    from transformers import (
        AutoImageProcessor,
        AutoModelForDepthEstimation,
        SamModel,
        SamProcessor,
    )
except ImportError as exc:
    raise RuntimeError(
        "Missing neural vision dependencies. Execute asset synthesis using the project virtual "
        "environment interpreter: `/Users/talsheaffer/code/tami/.venv/bin/python prototype/build_assets.py`."
    ) from exc

from prototype.scene_config import (
    ApertureBox,
    BoundingBoxPrompt,
    CanvasDimensionsConfig,
    DepthModelConfig,
    InpaintingConfig,
    OutpaintingConfig,
    SceneBuildSpec,
    SegmentationConfig,
)


def select_torch_device() -> torch.device:
    """Selects Apple Silicon MPS accelerator when available, otherwise CUDA or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DepthAnythingEstimator:
    """Computes normalized per-pixel relative depth maps in [0.0, 1.0] (1.0 = nearest)."""

    def __init__(self, cfg: DepthModelConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = device or select_torch_device()
        self.processor = AutoImageProcessor.from_pretrained(cfg.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(cfg.model_id).to(self.device)
        self.model.eval()

    def predict_depth(self, rgb_image: Image.Image) -> np.ndarray:
        """Returns float32 array of shape (H, W) normalized to [0.0, 1.0]."""
        width, height = rgb_image.size
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        interpolated = torch.nn.functional.interpolate(
            predicted_depth.detach().cpu().float().unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_np = interpolated.numpy().astype(np.float32)
        d_min = float(depth_np.min())
        d_max = float(depth_np.max())
        if d_max - d_min <= 1e-6:
            raise RuntimeError("Depth Anything V2 produced a degenerate constant depth map.")
        return (depth_np - d_min) / (d_max - d_min)

    def build_smoothed_room_depth_uint8(self, inpainted_room_rgb: Image.Image) -> Image.Image:
        """Predicts smooth architectural room depth on the expanded/inpainted room plate."""
        raw_depth = self.predict_depth(inpainted_room_rgb)
        smooth_depth = ndi.gaussian_filter(raw_depth, sigma=self.cfg.background_gaussian_sigma)
        s_min = float(smooth_depth.min())
        s_max = float(smooth_depth.max())
        norm_depth = (smooth_depth - s_min) / max(1e-6, s_max - s_min)
        depth_u8 = np.clip(np.round(norm_depth * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(depth_u8, mode="L")


class SalientMatteExtractor:
    """Extracts full-resolution neural alpha mattes of filamentary/suspended sculptures via IS-Net."""

    def __init__(self, cfg: SegmentationConfig) -> None:
        self.cfg = cfg
        self.session = new_session(cfg.rembg_model_name)

    def extract_alpha(self, rgb_image: Image.Image) -> np.ndarray:
        """Returns contrast-cleaned uint8 alpha matte of shape (H, W) in [0, 255]."""
        cutout_rgba = remove(rgb_image, session=self.session, only_mask=False)
        if not isinstance(cutout_rgba, Image.Image):
            raise RuntimeError("rembg failed to return a PIL Image.")
        raw_alpha = np.asarray(cutout_rgba.convert("RGBA"))[:, :, 3].astype(np.float32)

        low = float(self.cfg.alpha_foreground_threshold)
        high = float(max(self.cfg.alpha_foreground_threshold + 1, self.cfg.alpha_full_opaque_threshold))
        ramped = np.clip((raw_alpha - low) / (high - low), 0.0, 1.0) * 255.0
        alpha = np.round(ramped).astype(np.uint8)

        if int((alpha > 0).sum()) < self.cfg.min_component_area_px:
            raise RuntimeError(
                f"Neural segmentation ({self.cfg.rembg_model_name}) yielded insufficient foreground "
                f"area (< {self.cfg.min_component_area_px} px)."
            )
        return alpha


class SamBoxMaskExtractor:
    """Segments multi-part architectural sculptures via Meta Segment Anything (`SamModel`)."""

    def __init__(self, cfg: SegmentationConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.processor = SamProcessor.from_pretrained(cfg.sam_model_id)
        self.model = SamModel.from_pretrained(cfg.sam_model_id).to(self.device)
        self.model.eval()

    def _segment_box_group(
        self, rgb_image: Image.Image, boxes: Sequence[BoundingBoxPrompt]
    ) -> np.ndarray:
        width, height = rgb_image.size
        if not boxes:
            raise ValueError("SamBoxMaskExtractor requires at least one BoundingBoxPrompt.")

        box_list = [b.as_xyxy_list() for b in boxes]
        inputs = self.processor(rgb_image, input_boxes=[box_list], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=True)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        scores = outputs.iou_scores.cpu()[0]

        union_bool = np.zeros((height, width), dtype=bool)
        for idx in range(len(box_list)):
            best_m = int(scores[idx].argmax().item())
            union_bool |= masks[idx, best_m].numpy()

        binary_u8 = (union_bool.astype(np.uint8)) * 255
        if self.cfg.sam_morph_close_px > 1:
            k = self.cfg.sam_morph_close_px
            if k % 2 == 0:
                k += 1
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, close_kernel)

        alpha_f = binary_u8.astype(np.float32)
        if self.cfg.edge_feather_sigma > 0:
            alpha_f = ndi.gaussian_filter(alpha_f, sigma=self.cfg.edge_feather_sigma)
        return np.clip(np.round(alpha_f), 0, 255).astype(np.uint8)

    def extract_fg_mid_layers(
        self, rgb_image: Image.Image, spec: SceneBuildSpec
    ) -> tuple[Image.Image, Image.Image, np.ndarray]:
        rgb_np = np.asarray(rgb_image, dtype=np.uint8)
        alpha_fg = self._segment_box_group(rgb_image, spec.sam_fg_boxes)
        alpha_mid = self._segment_box_group(rgb_image, spec.sam_mid_boxes)
        alpha_mid = np.where(alpha_fg > 128, 0, alpha_mid).astype(np.uint8)

        fg_rgba = np.dstack([rgb_np, alpha_fg])
        mid_rgba = np.dstack([rgb_np, alpha_mid])
        combined_mask_u8 = np.where((alpha_fg > 8) | (alpha_mid > 8), 255, 0).astype(np.uint8)
        return (
            Image.fromarray(fg_rgba, mode="RGBA"),
            Image.fromarray(mid_rgba, mode="RGBA"),
            combined_mask_u8,
        )

    def supplement_rgba_layers(
        self,
        rgb_image: Image.Image,
        fg_img: Image.Image,
        mid_img: Image.Image,
        union_mask_u8: np.ndarray,
        spec: SceneBuildSpec,
    ) -> tuple[Image.Image, Image.Image, np.ndarray]:
        """Unions optional promptable SAM masks into existing neural FG/MID RGBA layers."""
        if not spec.sam_fg_boxes and not spec.sam_mid_boxes:
            return fg_img, mid_img, union_mask_u8

        rgb_np = np.asarray(rgb_image, dtype=np.uint8)
        fg_alpha = np.asarray(fg_img, dtype=np.uint8)[:, :, 3].copy()
        mid_alpha = np.asarray(mid_img, dtype=np.uint8)[:, :, 3].copy()

        if spec.sam_fg_boxes:
            extra_fg = self._segment_box_group(rgb_image, spec.sam_fg_boxes)
            fg_alpha = np.maximum(fg_alpha, extra_fg)
        if spec.sam_mid_boxes:
            extra_mid = self._segment_box_group(rgb_image, spec.sam_mid_boxes)
            extra_mid = np.where(fg_alpha > 128, 0, extra_mid).astype(np.uint8)
            mid_alpha = np.maximum(mid_alpha, extra_mid)

        new_union = np.where(
            (union_mask_u8 > 0) | (fg_alpha > 8) | (mid_alpha > 8), 255, 0
        ).astype(np.uint8)
        return (
            Image.fromarray(np.dstack([rgb_np, fg_alpha]), mode="RGBA"),
            Image.fromarray(np.dstack([rgb_np, mid_alpha]), mode="RGBA"),
            new_union,
        )


def partition_sculpture_layers(
    rgb_np: np.ndarray,
    alpha_u8: np.ndarray,
    depth_norm: np.ndarray,
    seg_cfg: SegmentationConfig,
    spec: SceneBuildSpec,
) -> tuple[Image.Image, Image.Image, np.ndarray]:
    """
    Partitions IS-Net neural matte into distinct Foreground (Z=0) and Midground (Z=-150)
    RGBA cutouts using connected-component spatial centroids and Depth Anything V2 quantiles.
    """
    height, width = alpha_u8.shape
    binary = (alpha_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    fg_weight = np.zeros((height, width), dtype=np.float32)
    mid_weight = np.zeros((height, width), dtype=np.float32)

    valid_pixels_depth = depth_norm[binary > 0]
    depth_cutoff = (
        float(np.quantile(valid_pixels_depth, spec.fg_min_depth_quantile))
        if valid_pixels_depth.size > 0
        else 0.5
    )
    x_min_norm, x_max_norm = spec.fg_x_range_norm

    kept_components = 0
    for comp_id in range(1, num_labels):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < seg_cfg.min_component_area_px:
            continue
        kept_components += 1
        comp_mask = labels == comp_id
        cx_norm = float(centroids[comp_id, 0]) / float(width)
        comp_median_depth = float(np.median(depth_norm[comp_mask]))

        in_fg_span = x_min_norm <= cx_norm <= x_max_norm
        if in_fg_span or comp_median_depth >= depth_cutoff + 0.18:
            fg_weight[comp_mask] = 1.0
        else:
            mid_weight[comp_mask] = 1.0

    if kept_components == 0:
        raise RuntimeError(
            f"Scene '{spec.scene_id}': zero valid connected sculpture components survived filtering."
        )

    fg_field = ndi.gaussian_filter(fg_weight, sigma=max(1.0, seg_cfg.edge_feather_sigma * 2.5))
    mid_field = ndi.gaussian_filter(mid_weight, sigma=max(1.0, seg_cfg.edge_feather_sigma * 2.5))
    denom = np.maximum(1e-5, fg_field + mid_field)

    kept_union = ((fg_weight > 0) | (mid_weight > 0)).astype(np.uint8)
    dilate_support = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    support_mask = cv2.dilate(kept_union, dilate_support) > 0

    clean_alpha = np.where(support_mask, alpha_u8.astype(np.float32), 0.0)
    if seg_cfg.edge_feather_sigma > 0:
        clean_alpha = ndi.gaussian_filter(clean_alpha, sigma=seg_cfg.edge_feather_sigma * 0.5)

    alpha_fg = np.clip(np.round(clean_alpha * (fg_field / denom)), 0, 255).astype(np.uint8)
    alpha_mid = np.clip(np.round(clean_alpha * (mid_field / denom)), 0, 255).astype(np.uint8)

    fg_rgba = np.dstack([rgb_np, alpha_fg])
    mid_rgba = np.dstack([rgb_np, alpha_mid])
    combined_mask_u8 = np.where((alpha_fg > 8) | (alpha_mid > 8), 255, 0).astype(np.uint8)

    return (
        Image.fromarray(fg_rgba, mode="RGBA"),
        Image.fromarray(mid_rgba, mode="RGBA"),
        combined_mask_u8,
    )


class LamaOcclusionInpainter:
    """Synthesizes occluded wall/floor/ceiling background behind dilated sculpture silhouettes via LaMa."""

    def __init__(self, cfg: InpaintingConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu")
        model_path = download_model(LAMA_MODEL_URL)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def inpaint_background(self, rgb_image: Image.Image, binary_mask_u8: np.ndarray) -> Image.Image:
        orig_w, orig_h = rgb_image.size
        radius = self.cfg.dilation_radius_px
        kernel_size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(binary_mask_u8, kernel, iterations=1)
        if self.cfg.mask_blur_sigma > 0:
            dilated_f = ndi.gaussian_filter(dilated.astype(np.float32), sigma=self.cfg.mask_blur_sigma)
            dilated = np.where(dilated_f > 64.0, 255, 0).astype(np.uint8)

        mask_img = Image.fromarray(dilated, mode="L")
        img_tensor, mask_tensor = prepare_img_and_mask(rgb_image, mask_img, self.device)
        with torch.inference_mode():
            inpainted_tensor = self.model(img_tensor, mask_tensor)
        res_np = inpainted_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        res_np = res_np[:orig_h, :orig_w, :]
        res_u8 = np.clip(np.round(res_np * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(res_u8, mode="RGB")


def compose_outpainted_room(
    inner_rgb: Image.Image,
    raw_outpaint_rgb: Image.Image,
    dims: CanvasDimensionsConfig,
    outpaint_cfg: OutpaintingConfig,
) -> tuple[Image.Image, float]:
    """
    Registers a wide-angle outpainted gallery plate onto the canonical expanded grid
    (e.g. 2700x1800 around 1500x1000) via 9-patch projective homography estimation,
    verifies correlation >= min_registration_corr, harmonizes RGB gain/bias across the
    seam collar, and feather-locks the exact authentic LaMa-inpainted room into the center.
    Returns (expanded_rgb_image, registration_correlation).
    """
    inner_w, inner_h = dims.width, dims.height
    if inner_rgb.size != (inner_w, inner_h):
        raise ValueError(f"Expected inner_rgb size {(inner_w, inner_h)}, got {inner_rgb.size}")

    exp_w, exp_h = outpaint_cfg.expanded_dimensions(dims)
    pad_x = (exp_w - inner_w) // 2
    pad_y = (exp_h - inner_h) // 2

    inner_np = np.asarray(inner_rgb.convert("RGB"), dtype=np.float32)
    inner_gray = cv2.cvtColor(inner_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Upsample raw outpaint to target expanded dimensions
    cand_rgb = np.asarray(
        raw_outpaint_rgb.convert("RGB").resize((exp_w, exp_h), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    cand_gray = cv2.cvtColor(cand_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Match 3x3 grid of interior patches to align aspect/scale/translation to sub-pixel precision
    patch_w, patch_h = 300, 220
    search_margin = 90
    xs_inner = [40, (inner_w - patch_w) // 2, inner_w - patch_w - 40]
    ys_inner = [40, (inner_h - patch_h) // 2, inner_h - patch_h - 40]

    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []
    for iy in ys_inner:
        for ix in xs_inner:
            tpl = inner_gray[iy : iy + patch_h, ix : ix + patch_w]
            nom_x = pad_x + ix
            nom_y = pad_y + iy
            sx0 = max(0, nom_x - search_margin)
            sy0 = max(0, nom_y - search_margin)
            sx1 = min(exp_w, nom_x + patch_w + search_margin)
            sy1 = min(exp_h, nom_y + patch_h + search_margin)
            roi = cand_gray[sy0:sy1, sx0:sx1]
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_loc = cv2.minMaxLoc(res)
            if max_v >= 0.70:
                matched_x = sx0 + max_loc[0] + patch_w * 0.5
                matched_y = sy0 + max_loc[1] + patch_h * 0.5
                target_x = nom_x + patch_w * 0.5
                target_y = nom_y + patch_h * 0.5
                src_pts.append([matched_x, matched_y])
                dst_pts.append([target_x, target_y])

    if len(src_pts) < 4:
        raise RuntimeError(
            f"Outpaint registration failed: only {len(src_pts)}/9 anchor patches matched."
        )

    H_mat, _ = cv2.findHomography(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts, dtype=np.float32),
        cv2.RANSAC,
        ransacReprojThreshold=4.0,
    )
    if H_mat is None:
        raise RuntimeError("Outpaint registration failed: cv2.findHomography returned None.")

    warped_rgb = cv2.warpPerspective(
        cand_rgb,
        H_mat,
        (exp_w, exp_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Verify post-warp normalized cross-correlation over central frame
    warped_center_gray = cv2.cvtColor(
        np.clip(warped_rgb[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w], 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    )
    corr_mat = cv2.matchTemplate(warped_center_gray, inner_gray, cv2.TM_CCOEFF_NORMED)
    corr = float(corr_mat[0, 0])
    if corr < outpaint_cfg.min_registration_corr:
        raise RuntimeError(
            f"Outpaint registration correlation {corr:.4f} is below required "
            f"minimum {outpaint_cfg.min_registration_corr:.4f}."
        )

    # Per-channel linear exposure/white-balance harmonization on the perimeter collar
    band = outpaint_cfg.color_match_band_px
    collar_mask = np.zeros((inner_h, inner_w), dtype=bool)
    collar_mask[:band, :] = True
    collar_mask[-band:, :] = True
    collar_mask[:, :band] = True
    collar_mask[:, -band:] = True

    warped_crop = warped_rgb[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]
    harmonized_rgb = warped_rgb.copy()
    for ch in range(3):
        ref_vals = inner_np[:, :, ch][collar_mask]
        src_vals = warped_crop[:, :, ch][collar_mask]
        src_std = float(np.std(src_vals))
        ref_std = float(np.std(ref_vals))
        gain = np.clip(ref_std / max(1.0, src_std), 0.85, 1.18)
        bias = float(np.mean(ref_vals)) - gain * float(np.mean(src_vals))
        harmonized_rgb[:, :, ch] = harmonized_rgb[:, :, ch] * gain + bias

    # Feather-lock exact authentic inner LaMa room strictly inside [pad_x:pad_x+inner_w, pad_y:pad_y+inner_h]
    collar = outpaint_cfg.feather_collar_px
    rx = np.ones(inner_w, dtype=np.float32)
    ry = np.ones(inner_h, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, collar, dtype=np.float32))
    rx[:collar] = ramp
    rx[-collar:] = ramp[::-1]
    ry[:collar] = ramp
    ry[-collar:] = ramp[::-1]
    inner_alpha = np.outer(ry, rx)[:, :, np.newaxis]

    blended = harmonized_rgb.copy()
    blended[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w] = (
        inner_alpha * inner_np
        + (1.0 - inner_alpha) * harmonized_rgb[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]
    )
    blended_u8 = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    return Image.fromarray(blended_u8, mode="RGB"), corr


def build_aperture_reflection_layer(
    canvas_width: int,
    canvas_height: int,
    reflection_src: Image.Image,
    apertures: Sequence[ApertureBox],
) -> Image.Image:
    """
    Renders subtle architectural mirror reflections strictly bounded to physical
    mirror apertures (never floating unmasked over bare drywall).
    """
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    src_rgba = reflection_src.convert("RGB")

    for ap in apertures:
        box_w = ap.x1 - ap.x0
        box_h = ap.y1 - ap.y0
        if box_w <= 4 or box_h <= 4:
            raise ValueError(f"Invalid aperture dimensions: {ap}")

        crop_rgb = np.asarray(src_rgba.resize((box_w, box_h), Image.Resampling.LANCZOS), dtype=np.uint8)
        mask = np.zeros((box_h, box_w), dtype=np.float32)
        pad_x = max(2, int(round(box_w * 0.08)))
        pad_y = max(2, int(round(box_h * 0.08)))
        mask[pad_y : box_h - pad_y, pad_x : box_w - pad_x] = float(ap.max_alpha)
        mask = ndi.gaussian_filter(mask, sigma=ap.feather_sigma)
        mask_u8 = np.clip(np.round(mask), 0, 255).astype(np.uint8)

        canvas[ap.y0 : ap.y1, ap.x0 : ap.x1, :3] = crop_rgb
        canvas[ap.y0 : ap.y1, ap.x0 : ap.x1, 3] = np.maximum(
            canvas[ap.y0 : ap.y1, ap.x0 : ap.x1, 3], mask_u8
        )

    return Image.fromarray(canvas, mode="RGBA")
