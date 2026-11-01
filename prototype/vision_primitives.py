#!/usr/bin/env python3
"""
prototype/vision_primitives.py
Upstream neural computer-vision primitives for 2.5D Layered Depth Image (LDI) synthesis:
1. Monocular depth estimation via Depth Anything V2 (HuggingFace Transformers + PyTorch MPS/CPU),
   plus sculpted 3D distance-transform dome relief synthesis for archival drawings.
2. Filamentary salient sculpture matting via IS-Net (`rembg`).
3. Promptable architectural sculpture segmentation via Meta Segment Anything Model (`SamModel`).
4. Graphite/ink stroke contrast segmentation on archival paper with pinhole-bounded closing.
5. Cross-scene RGBA object & drawn-island extraction and compositing (`LayerOverlaySpec`).
6. Morphological mask dilation and neural Fast Fourier Convolution inpainting via LaMa (`big-lama.pt`).
7. Multi-candidate homographic & Sobel-salient similarity outpaint registration, seam color
   harmonization, and center-lock blending.

Strict No-Fallback Guarantee: raises explicit RuntimeError/ValueError if required neural packages,
weights, segmentation contracts, or outpainting registration correlations fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    import cv2
    import numpy as np
    import scipy.ndimage as ndi
    import torch
    from PIL import Image, ImageFilter, ImageOps
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
    CeilingMirrorReflectionConfig,
    DepthModelConfig,
    DiffuseApertureFusionSpec,
    DrawingStrokeBox,
    ExtraDepthLayerSpec,
    FloatingElementExtractionConfig,
    GraphiteDrawingConfig,
    InpaintingConfig,
    LayerOverlaySpec,
    OrientedMatteBoxSpec,
    OutpaintingConfig,
    PipeTunnelReliefConfig,
    SamPointGroupPrompt,
    SceneBuildSpec,
    SegmentationConfig,
    default_floating_element_extraction_config,
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

    def build_drawing_dome_relief_depth_uint8(
        self,
        expanded_paper_rgb: Image.Image,
        draw_cfg: GraphiteDrawingConfig,
    ) -> Image.Image:
        """
        Synthesizes an invented 3D sculptural relief depth map for stippled/graphite drawings
        by combining Euclidean distance-transform dome inflation over the drawn vessel silhouette,
        a studio worktable perspective ramp, and neural Depth Anything V2 monocular cues.
        """
        exp_w, exp_h = expanded_paper_rgb.size
        paper_np = np.asarray(expanded_paper_rgb.convert("RGB"), dtype=np.uint8)
        paper_gray = cv2.cvtColor(paper_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        bg_level = float(np.percentile(paper_gray, draw_cfg.dome_bg_percentile))
        vessel_mask = ((bg_level - paper_gray) > draw_cfg.dome_ink_delta_threshold).astype(np.uint8)
        k_close = (
            draw_cfg.dome_morph_close_px
            if draw_cfg.dome_morph_close_px % 2 == 1
            else draw_cfg.dome_morph_close_px + 1
        )
        vessel_closed = cv2.morphologyEx(
            vessel_mask * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)),
        )
        dist = cv2.distanceTransform((vessel_closed > 0).astype(np.uint8), cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        if max_dist > 0.0:
            dome = np.power(dist / max_dist, draw_cfg.dome_power_exponent)
        else:
            dome = np.zeros_like(paper_gray)
        dome_smooth = ndi.gaussian_filter(dome, sigma=draw_cfg.dome_gaussian_sigma)

        yy = np.linspace(0.0, 1.0, exp_h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, exp_w, dtype=np.float32)[None, :]
        stage_ramp = draw_cfg.stage_y_weight * np.power(yy, draw_cfg.stage_y_exponent) + (
            draw_cfg.stage_x_weight * (1.0 - np.clip(xx * xx, 0.0, 1.0))
        )
        neural_depth = self.predict_depth(expanded_paper_rgb)

        combined = (
            draw_cfg.depth_dome_weight * dome_smooth
            + draw_cfg.depth_stage_weight * stage_ramp
            + draw_cfg.depth_neural_weight * neural_depth
        )
        smooth_relief = ndi.gaussian_filter(combined, sigma=draw_cfg.depth_final_sigma)
        r_min = float(smooth_relief.min())
        r_max = float(smooth_relief.max())
        norm_relief = (smooth_relief - r_min) / max(1e-6, r_max - r_min)
        relief_u8 = np.clip(np.round(norm_relief * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(relief_u8, mode="L")

    def build_flat_gallery_wall_depth_uint8(
        self,
        clean_wall_rgb: Image.Image,
        max_u8: int,
    ) -> Image.Image:
        """
        Computes a clean, pipe-free far-wall & floor depth map (`0..max_u8` uint8, zero pipe imprint)
        on the LaMa-inpainted Binyamin Gallery wall plate for `layer0` (`roomDepthAmplitude = 0.0`).
        """
        if max_u8 <= 0:
            w, h = clean_wall_rgb.size
            return Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L")
        raw_depth = self.predict_depth(clean_wall_rgb)
        smooth_wall = ndi.gaussian_filter(raw_depth, sigma=self.cfg.background_gaussian_sigma * 1.5)
        d_min = float(smooth_wall.min())
        d_max = float(smooth_wall.max())
        norm_wall = (smooth_wall - d_min) / max(1e-6, d_max - d_min)
        wall_u8 = np.clip(np.round(norm_wall * float(max_u8)), 0, max_u8).astype(np.uint8)
        return Image.fromarray(wall_u8, mode="L")


_PIPE_TUNNEL_REQUIRED_FIELDS: tuple[str, ...] = (
    "front_center_xy",
    "front_radii_xy",
    "rear_center_xy",
    "rear_radii_xy",
    "scene_bg_z",
    "pipe_depth_amplitude",
    "reference_camera_z",
    "front_rim_z",
    "rim_face_width_px",
    "rear_rim_z",
    "rear_chamber_z",
    "rear_chamber_taper_r",
    "rear_chamber_core_clamp_ratio",
    "mouth_join_blend_px",
    "bell_sam_prompt",
    "conduit_sam_prompt",
    "bell_sam_max_pinhole_px",
    "bell_sam_min_component_px",
    "conduit_sam_max_pinhole_px",
    "conduit_sam_min_component_px",
    "polar_angle_samples",
    "polar_sam_probe_inward_px",
    "polar_sam_probe_outward_px",
    "polar_refine_inward_px",
    "polar_refine_outward_px",
    "polar_da2_edge_weight",
    "wall_bright_lum_threshold",
    "wall_bright_penalty_weight",
    "rim_edge_inward_bias_px",
    "polar_smooth_sigma_samples",
    "left_join_angle_range_deg",
    "left_pipe_search_end_exp_x",
    "left_pipe_join_exp_x",
    "left_pipe_top_clamp_exp_y",
    "left_pipe_bot_clamp_exp_y",
    "left_pipe_snap_window_px",
    "left_pipe_elbow_exp_x",
    "left_pipe_corner_shadow_halfwidth_px",
    "left_pipe_corner_shadow_weight",
    "left_pipe_irls_iters",
    "left_pipe_irls_huber_px",
    "left_pipe_hinge_smooth_sigma_px",
    "left_pipe_left_edge_z",
    "left_pipe_elbow_z",
    "left_pipe_mouth_join_z",
    "left_pipe_left_cyl_radius_z",
    "left_pipe_join_cyl_radius_z",
    "shell_alpha_aa_halfwidth_px",
    "shell_pure_concrete_erode_px",
    "shell_edge_pad_outward_px",
    "neural_relief_amplitude_z",
    "neural_highpass_sigma",
    "interior_smooth_sigma_px",
    "bisection_iters",
    "wall_inpaint_downsample_wh",
    "wall_inpaint_dilation_px",
    "wall_inpaint_extra_boxes_exp",
    "wall_inpaint_feather_px",
    "wall_grain_patch_exp_xyxy",
    "wall_grain_highpass_sigma_px",
    "wall_grain_strength",
    "flat_wall_depth_max_u8",
    "contact_pins",
)


def _require_config_fields(cfg_obj: object, required_fields: tuple[str, ...], cls_name: str) -> None:
    """Enforces strict SSOT schema completeness without silent fallback defaults."""
    missing = [name for name in required_fields if not hasattr(cfg_obj, name)]
    if missing:
        raise ValueError(
            f"{cls_name} is missing required SSOT configuration fields {missing}."
        )


def _fit_corner_hinged_line(
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
    x_hinge: float,
    num_iters: int,
    huber_px: float,
) -> np.ndarray:
    """
    Fits a continuous 2-piece linear cylinder contour y(x) = a + b*(x - x_hinge)_minus + c*(x - x_hinge)_plus
    via Huber iteratively reweighted least squares so room-corner shadows never distort the straight conduit.
    """
    xm = np.minimum(0.0, xs - x_hinge)
    xp = np.maximum(0.0, xs - x_hinge)
    design = np.column_stack([np.ones_like(xs), xm, xp])
    w = weights.astype(np.float64).copy()
    params = np.zeros(3, dtype=np.float64)
    safe_huber = max(0.25, float(huber_px))
    for _ in range(max(1, num_iters)):
        sqrt_w = np.sqrt(np.maximum(1e-8, w))
        params, _, _, _ = np.linalg.lstsq(design * sqrt_w[:, None], ys * sqrt_w, rcond=None)
        resid = np.abs(ys - design @ params)
        w = weights / np.maximum(safe_huber, resid)
    return (design @ params).astype(np.float32)


def _segment_concrete_pipe_signed_distance(
    pipe_plate_rgb: Image.Image,
    expanded_bg_np: np.ndarray,
    neural_depth: np.ndarray,
    fc_exp_xy: tuple[float, float],
    ptr_cfg: PipeTunnelReliefConfig,
    sam_extractor: SamBoxMaskExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Segments the exact outer boundary of the physical concrete pipe (circular bell + horizontal
    left conduit extending to x=0) by combining promptable Meta SAM masks with radial DA2/luminance
    edge snapping and corner-hinged cylindrical edge fitting.
    Returns `(sd_pipe, sd_bell, y_top_full, y_bot_full)` where `sd >= 0` is strictly inside concrete.
    """
    exp_h, exp_w, _ = expanded_bg_np.shape
    lum = (
        0.299 * expanded_bg_np[:, :, 0]
        + 0.587 * expanded_bg_np[:, :, 1]
        + 0.114 * expanded_bg_np[:, :, 2]
    )

    bell_bool = (
        sam_extractor.segment_point_groups(
            rgb_image=pipe_plate_rgb,
            point_groups=(ptr_cfg.bell_sam_prompt,),
            exclude_boxes=(),
            max_pinhole_fill_px=ptr_cfg.bell_sam_max_pinhole_px,
            min_major_component_px=ptr_cfg.bell_sam_min_component_px,
        )
        >= 128
    )
    cond_bool = (
        sam_extractor.segment_point_groups(
            rgb_image=pipe_plate_rgb,
            point_groups=(ptr_cfg.conduit_sam_prompt,),
            exclude_boxes=(),
            max_pinhole_fill_px=ptr_cfg.conduit_sam_max_pinhole_px,
            min_major_component_px=ptr_cfg.conduit_sam_min_component_px,
        )
        >= 128
    )

    fc_x, fc_y = fc_exp_xy
    nom_rx, nom_ry = ptr_cfg.front_radii_xy
    n_angles = ptr_cfg.polar_angle_samples
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False, dtype=np.float32)
    r_bell = np.zeros(n_angles, dtype=np.float32)
    join_lo, join_hi = ptr_cfg.left_join_angle_range_deg
    n_probe = max(
        65,
        int(round((ptr_cfg.polar_sam_probe_inward_px + ptr_cfg.polar_sam_probe_outward_px) * 4.0))
        + 1,
    )
    n_refine = max(
        13,
        int(round((ptr_cfg.polar_refine_inward_px + ptr_cfg.polar_refine_outward_px) * 2.0)) + 1,
    )

    for idx, theta in enumerate(angles):
        cs = float(np.cos(theta))
        sn = float(np.sin(theta))
        r_nom = 1.0 / float(np.hypot(cs / max(1.0, nom_rx), sn / max(1.0, nom_ry)))
        rs_probe = np.linspace(
            r_nom - ptr_cfg.polar_sam_probe_inward_px,
            r_nom + ptr_cfg.polar_sam_probe_outward_px,
            n_probe,
            dtype=np.float32,
        )
        xs_p = np.clip(np.round(fc_x + rs_probe * cs).astype(int), 0, exp_w - 1)
        ys_p = np.clip(np.round(fc_y + rs_probe * sn).astype(int), 0, exp_h - 1)
        in_sam = bell_bool[ys_p, xs_p]
        r_sam = float(rs_probe[np.where(in_sam)[0][-1]]) if np.any(in_sam) else r_nom

        deg = float(np.rad2deg(theta)) % 360.0
        if join_lo <= deg <= join_hi:
            r_bell[idx] = r_sam
            continue

        rs_loc = np.linspace(
            r_sam - ptr_cfg.polar_refine_inward_px,
            r_sam + ptr_cfg.polar_refine_outward_px,
            n_refine,
            dtype=np.float32,
        )
        xs_l = np.clip(np.round(fc_x + rs_loc * cs).astype(int), 0, exp_w - 1)
        ys_l = np.clip(np.round(fc_y + rs_loc * sn).astype(int), 0, exp_h - 1)
        lum_ray = lum[ys_l, xs_l]
        grad_luma = np.gradient(lum_ray)
        grad_da2 = -np.gradient(neural_depth[ys_l, xs_l]) * 255.0
        wall_pen = (
            np.maximum(0.0, lum_ray - ptr_cfg.wall_bright_lum_threshold)
            * ptr_cfg.wall_bright_penalty_weight
        )
        score = grad_luma + ptr_cfg.polar_da2_edge_weight * grad_da2 - wall_pen
        r_bell[idx] = float(rs_loc[int(np.argmax(score))]) - ptr_cfg.rim_edge_inward_bias_px

    pad_n = max(8, int(round(ptr_cfg.polar_smooth_sigma_samples * 6.0)))
    r_pad = np.concatenate([r_bell[-pad_n:], r_bell, r_bell[:pad_n]])
    r_bell_s = ndi.gaussian_filter1d(r_pad, sigma=ptr_cfg.polar_smooth_sigma_samples)[
        pad_n:-pad_n
    ]

    yy_grid, xx_grid = np.mgrid[0:exp_h, 0:exp_w].astype(np.float32)
    dx = xx_grid - fc_x
    dy = yy_grid - fc_y
    th_idx = np.mod(np.arctan2(dy, dx), 2.0 * np.pi) / (2.0 * np.pi) * float(n_angles)
    i0 = np.floor(th_idx).astype(int) % n_angles
    frac = th_idx - np.floor(th_idx)
    r_lim = (1.0 - frac) * r_bell_s[i0] + frac * r_bell_s[(i0 + 1) % n_angles]
    sd_bell = r_lim - np.hypot(dx, dy)

    search_end_x = min(exp_w, max(16, ptr_cfg.left_pipe_search_end_exp_x))
    join_x = min(exp_w, max(search_end_x, ptr_cfg.left_pipe_join_exp_x))
    top_min_y, top_max_y = ptr_cfg.left_pipe_top_clamp_exp_y
    bot_min_y, bot_max_y = ptr_cfg.left_pipe_bot_clamp_exp_y
    snap_in, snap_out = ptr_cfg.left_pipe_snap_window_px

    y_top_raw = np.zeros(search_end_x, dtype=np.float32)
    y_bot_raw = np.zeros(search_end_x, dtype=np.float32)
    w_top = np.ones(search_end_x, dtype=np.float32)
    w_bot = np.ones(search_end_x, dtype=np.float32)

    for x in range(search_end_x):
        ys_in = np.where(cond_bool[:bot_max_y, x])[0]
        yt_s = (
            float(ys_in[0])
            if ys_in.size >= 20
            else (y_top_raw[x - 1] if x > 0 else float(0.5 * (top_min_y + top_max_y)))
        )
        yb_s = (
            float(ys_in[-1])
            if ys_in.size >= 20
            else (y_bot_raw[x - 1] if x > 0 else float(0.5 * (bot_min_y + bot_max_y)))
        )

        yt0 = max(top_min_y, min(top_max_y - 6, int(round(yt_s)) - snap_in))
        yt1 = max(yt0 + 5, min(top_max_y, int(round(yt_s)) + snap_out))
        ys_t = np.arange(yt0, yt1)
        gt = -np.gradient(lum[ys_t, x])
        y_top_raw[x] = float(ys_t[int(np.argmax(gt))]) + ptr_cfg.rim_edge_inward_bias_px
        w_top[x] = max(0.05, float(np.max(gt)))

        yb0 = max(bot_min_y, min(bot_max_y - 6, int(round(yb_s)) - snap_out))
        yb1 = max(yb0 + 5, min(bot_max_y, int(round(yb_s)) + snap_in + 2))
        ys_b = np.arange(yb0, yb1)
        gb = np.gradient(lum[ys_b, x])
        y_bot_raw[x] = float(ys_b[int(np.argmax(gb))]) - ptr_cfg.rim_edge_inward_bias_px
        w_bot[x] = max(0.05, float(np.max(gb)))

    xs_arr = np.arange(search_end_x, dtype=np.float64)
    corner_band = (
        np.abs(xs_arr - ptr_cfg.left_pipe_elbow_exp_x)
        < ptr_cfg.left_pipe_corner_shadow_halfwidth_px
    )
    w_top[corner_band] *= ptr_cfg.left_pipe_corner_shadow_weight
    w_bot[corner_band] *= ptr_cfg.left_pipe_corner_shadow_weight

    y_top_hinged = _fit_corner_hinged_line(
        xs=xs_arr,
        ys=y_top_raw.astype(np.float64),
        weights=w_top,
        x_hinge=ptr_cfg.left_pipe_elbow_exp_x,
        num_iters=ptr_cfg.left_pipe_irls_iters,
        huber_px=ptr_cfg.left_pipe_irls_huber_px,
    )
    y_bot_hinged = _fit_corner_hinged_line(
        xs=xs_arr,
        ys=y_bot_raw.astype(np.float64),
        weights=w_bot,
        x_hinge=ptr_cfg.left_pipe_elbow_exp_x,
        num_iters=ptr_cfg.left_pipe_irls_iters,
        huber_px=ptr_cfg.left_pipe_irls_huber_px,
    )
    y_top_s = ndi.gaussian_filter1d(
        y_top_hinged, sigma=ptr_cfg.left_pipe_hinge_smooth_sigma_px, mode="nearest"
    )
    y_bot_s = ndi.gaussian_filter1d(
        y_bot_hinged, sigma=ptr_cfg.left_pipe_hinge_smooth_sigma_px, mode="nearest"
    )

    y_top_full = np.empty(join_x, dtype=np.float32)
    y_bot_full = np.empty(join_x, dtype=np.float32)
    y_top_full[:search_end_x] = y_top_s
    y_bot_full[:search_end_x] = y_bot_s
    if join_x > search_end_x:
        y_top_full[search_end_x:] = y_top_s[-1]
        y_bot_full[search_end_x:] = y_bot_s[-1]

    yy_col = np.arange(exp_h, dtype=np.float32)[:, None]
    sd_cond = np.full((exp_h, exp_w), -1e4, dtype=np.float32)
    sd_cond_sub = np.minimum(yy_col - y_top_full[None, :], y_bot_full[None, :] - yy_col)
    sd_cond_sub = np.minimum(
        sd_cond_sub, (float(join_x) - np.arange(join_x, dtype=np.float32))[None, :]
    )
    sd_cond[:, :join_x] = sd_cond_sub

    sd_pipe = np.maximum(sd_bell, sd_cond)
    return sd_pipe, sd_bell, y_top_full, y_bot_full


def _build_decontaminated_pipe_shell_rgba(
    expanded_bg_np: np.ndarray,
    sd_pipe: np.ndarray,
    ptr_cfg: PipeTunnelReliefConfig,
) -> Image.Image:
    """
    Builds the crisp `2700x1800` RGBA concrete pipe shell cutout (`<= 1.0px` AA for `uAlphaCutout = 0.5`),
    decontaminating light gallery-wall edge bleed and solidifying pure concrete RGB into the immediate
    `alpha = 0` border so WebGL bilinear filtering at `alpha = 0.5` produces zero white or dark halo.
    """
    aa_span = max(0.05, 2.0 * ptr_cfg.shell_alpha_aa_halfwidth_px)
    alpha_norm = np.clip(0.5 + sd_pipe / aa_span, 0.0, 1.0)
    alpha_u8 = np.clip(np.round(alpha_norm * 255.0), 0, 255).astype(np.uint8)

    rgb_u8 = decontaminate_matte_edge_rgb(
        np.clip(np.round(expanded_bg_np), 0, 255).astype(np.uint8),
        alpha_u8,
    )

    erode_px = max(1, ptr_cfg.shell_pure_concrete_erode_px)
    pure_core = sd_pipe >= float(erode_px)
    if np.any(pure_core):
        nearest_core = ndi.distance_transform_edt(
            ~pure_core,
            return_distances=False,
            return_indices=True,
        )
        core_rgb = rgb_u8[nearest_core[0], nearest_core[1]]
        cur_lum = (
            0.299 * rgb_u8[:, :, 0].astype(np.float32)
            + 0.587 * rgb_u8[:, :, 1].astype(np.float32)
            + 0.114 * rgb_u8[:, :, 2].astype(np.float32)
        )
        core_lum = (
            0.299 * core_rgb[:, :, 0].astype(np.float32)
            + 0.587 * core_rgb[:, :, 1].astype(np.float32)
            + 0.114 * core_rgb[:, :, 2].astype(np.float32)
        )
        rim_zone = (sd_pipe >= 0.0) & (~pure_core) & (cur_lum > core_lum + 8.0)
        rgb_u8 = np.where(rim_zone[:, :, None], core_rgb, rgb_u8)

        outward_pad = (sd_pipe < 0.0) & (sd_pipe >= -float(ptr_cfg.shell_edge_pad_outward_px))
        rgb_u8 = np.where(outward_pad[:, :, None], core_rgb, rgb_u8)
        far_empty = sd_pipe < -float(ptr_cfg.shell_edge_pad_outward_px)
        rgb_u8 = np.where(far_empty[:, :, None], 0, rgb_u8)

    return Image.fromarray(np.dstack([rgb_u8, alpha_u8]), mode="RGBA")


def _build_extrapolated_pipe_shell_depth_uint8(
    neural_depth: np.ndarray,
    sd_pipe: np.ndarray,
    sd_bell: np.ndarray,
    y_top_full: np.ndarray,
    y_bot_full: np.ndarray,
    dims: CanvasDimensionsConfig,
    ptr_cfg: PipeTunnelReliefConfig,
) -> Image.Image:
    """
    Computes calibrated 3D perspective tunnel + cylindrical horizontal conduit depth strictly INSIDE
    `pipe_mask` (`front_rim_z = +8.0` right to the outermost circular concrete pixel with ZERO wall
    bevel ramp; normalized convolution interior smoothing), then extrapolates boundary depth outward
    across `~pipe_mask` via `scipy.ndimage.distance_transform_edt` so straddling WebGL mesh triangles
    remain 100% rigid and undeformed.
    """
    exp_h, exp_w = sd_pipe.shape
    pad_x = (exp_w - dims.width) * 0.5
    pad_y = (exp_h - dims.height) * 0.5

    xx = (np.arange(exp_w, dtype=np.float32) - pad_x)[None, :]
    yy = (np.arange(exp_h, dtype=np.float32) - pad_y)[:, None]

    fc_x, fc_y = ptr_cfg.front_center_xy
    fr_x, fr_y = ptr_cfg.front_radii_xy
    rc_x, rc_y = ptr_cfg.rear_center_xy
    rr_x, rr_y = ptr_cfg.rear_radii_xy

    r_rear = np.hypot((xx - rc_x) / max(1.0, rr_x), (yy - rc_y) / max(1.0, rr_y))
    inner_fr_x = max(64.0, fr_x - ptr_cfg.rim_face_width_px)
    inner_fr_y = max(64.0, fr_y - ptr_cfg.rim_face_width_px)

    lo = np.zeros((exp_h, exp_w), dtype=np.float32)
    hi = np.ones((exp_h, exp_w), dtype=np.float32)
    for _ in range(ptr_cfg.bisection_iters):
        mid = 0.5 * (lo + hi)
        cx = rc_x + mid * (fc_x - rc_x)
        cy = rc_y + mid * (fc_y - rc_y)
        rx = rr_x + mid * (inner_fr_x - rr_x)
        ry = rr_y + mid * (inner_fr_y - rr_y)
        val = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        outside = val > 1.0
        lo = np.where(outside, mid, lo)
        hi = np.where(outside, hi, mid)
    s_barrel = np.clip(0.5 * (lo + hi), 0.0, 1.0)

    z_cam = ptr_cfg.reference_camera_z
    inv_dist = (1.0 - s_barrel) / max(1.0, z_cam - ptr_cfg.rear_rim_z) + s_barrel / max(
        1.0, z_cam - ptr_cfg.front_rim_z
    )
    z_barrel = z_cam - 1.0 / np.maximum(1e-6, inv_dist)

    t_rear = np.clip(
        (r_rear - ptr_cfg.rear_chamber_taper_r) / max(1e-4, 1.0 - ptr_cfg.rear_chamber_taper_r),
        0.0,
        1.0,
    )
    t_rear_s = t_rear * t_rear * (3.0 - 2.0 * t_rear)
    z_chamber = ptr_cfg.rear_chamber_z + t_rear_s * (
        ptr_cfg.rear_rim_z - ptr_cfg.rear_chamber_z
    )
    z_tunnel_base = np.where(r_rear < 1.0, z_chamber, z_barrel)

    neural_hp = neural_depth - ndi.gaussian_filter(
        neural_depth, sigma=ptr_cfg.neural_highpass_sigma
    )
    # Damp high-pass relief on both the deep rear aperture and the flat front concrete rim face
    rim_face_damp = np.clip((1.0 - s_barrel) / 0.08, 0.0, 1.0)
    chamber_damp = np.where(r_rear < 1.0, t_rear_s, rim_face_damp)
    z_tunnel = z_tunnel_base + neural_hp * ptr_cfg.neural_relief_amplitude_z * chamber_damp

    for pin in ptr_cfg.contact_pins:
        dist_pin = np.hypot(xx - pin.inner_xy[0], yy - pin.inner_xy[1])
        u_pin = np.clip((dist_pin - pin.radius_px) / max(1e-4, pin.feather_px), 0.0, 1.0)
        w_pin = 1.0 - u_pin * u_pin * (3.0 - 2.0 * u_pin)
        z_tunnel = z_tunnel + w_pin * (pin.target_z - z_tunnel)

    core_clamp_r = ptr_cfg.rear_chamber_taper_r * ptr_cfg.rear_chamber_core_clamp_ratio
    z_tunnel = np.where(
        r_rear <= core_clamp_r,
        ptr_cfg.rear_chamber_z,
        np.maximum(z_tunnel, ptr_cfg.rear_chamber_z),
    )

    # Cylindrical horizontal left concrete conduit relief (never ramped to -430!)
    join_x = y_top_full.shape[0]
    xx_row = np.arange(join_x, dtype=np.float32)
    elbow_x = ptr_cfg.left_pipe_elbow_exp_x
    t_left_seg = np.clip(xx_row / max(1.0, elbow_x), 0.0, 1.0)
    t_right_seg = np.clip((xx_row - elbow_x) / max(1.0, float(join_x) - elbow_x), 0.0, 1.0)
    t_right_smooth = t_right_seg * t_right_seg * (3.0 - 2.0 * t_right_seg)
    z_axis_1d = np.where(
        xx_row <= elbow_x,
        ptr_cfg.left_pipe_left_edge_z
        + t_left_seg * (ptr_cfg.left_pipe_elbow_z - ptr_cfg.left_pipe_left_edge_z),
        ptr_cfg.left_pipe_elbow_z
        + t_right_smooth * (ptr_cfg.left_pipe_mouth_join_z - ptr_cfg.left_pipe_elbow_z),
    )
    cyl_rad_z_1d = ptr_cfg.left_pipe_left_cyl_radius_z + t_right_smooth * (
        ptr_cfg.left_pipe_join_cyl_radius_z - ptr_cfg.left_pipe_left_cyl_radius_z
    )
    z_edge_1d = z_axis_1d - cyl_rad_z_1d

    yy_col = np.arange(exp_h, dtype=np.float32)[:, None]
    y_mid = 0.5 * (y_top_full + y_bot_full)
    y_half = np.maximum(8.0, 0.5 * (y_bot_full - y_top_full))
    u_cyl = np.clip((yy_col - y_mid[None, :]) / y_half[None, :], -1.0, 1.0)
    cyl_profile = np.sqrt(np.maximum(0.0, 1.0 - u_cyl * u_cyl))

    z_left_sub = z_edge_1d[None, :] + cyl_profile * cyl_rad_z_1d[None, :]

    # Smooth C1 weld where horizontal conduit flares into the circular concrete bell:
    # `w_cond_vert` is 1.0 everywhere inside `[y_top_full, y_bot_full]` and tapers smoothly
    # outside `[y_top_full, y_bot_full]` on the upper/lower circular bell so `w_use_cond`
    # has zero jump across `sd_bell == 0`.
    t_bell_in = np.clip(sd_bell[:, :join_x] / max(1.0, ptr_cfg.mouth_join_blend_px), 0.0, 1.0)
    w_bell = t_bell_in * t_bell_in * (3.0 - 2.0 * t_bell_in)
    cond_vert_margin = np.minimum(yy_col - y_top_full[None, :], y_bot_full[None, :] - yy_col)
    v_taper = max(1.0, ptr_cfg.conduit_bell_vert_taper_px)
    t_cond_vert = np.clip((cond_vert_margin + v_taper) / v_taper, 0.0, 1.0)
    w_cond_vert = t_cond_vert * t_cond_vert * (3.0 - 2.0 * t_cond_vert)
    w_use_cond = np.where(sd_bell[:, :join_x] <= 0.0, 1.0, (1.0 - w_bell) * w_cond_vert)

    z_raw = z_tunnel.copy()
    z_raw[:, :join_x] = z_tunnel[:, :join_x] + w_use_cond * (z_left_sub - z_tunnel[:, :join_x])

    pipe_mask = sd_pipe >= 0.0
    if ptr_cfg.interior_smooth_sigma_px > 0.0:
        mask_f = pipe_mask.astype(np.float32)
        num = ndi.gaussian_filter(z_raw * mask_f, sigma=ptr_cfg.interior_smooth_sigma_px)
        den = np.maximum(
            1e-5, ndi.gaussian_filter(mask_f, sigma=ptr_cfg.interior_smooth_sigma_px)
        )
        z_interior = num / den
    else:
        z_interior = z_raw

    # Keep circular front concrete rim right at `front_rim_z` (+8.0) all the way to the outer edge
    # with smooth C1 angular & radial tapers so zero step occurs at the conduit join.
    deg_grid = np.rad2deg(
        np.mod(np.arctan2(yy - fc_y, xx - fc_x), 2.0 * np.pi)
    )
    join_lo, join_hi = ptr_cfg.left_join_angle_range_deg
    arc_taper = max(1.0, ptr_cfg.rim_arc_taper_deg)
    ang_out_dist = np.where(
        (deg_grid >= join_lo) & (deg_grid <= join_hi),
        0.0,
        np.minimum(np.abs(deg_grid - join_lo), np.abs(deg_grid - join_hi)),
    )
    t_arc = np.clip(ang_out_dist / arc_taper, 0.0, 1.0)
    w_arc = t_arc * t_arc * (3.0 - 2.0 * t_arc)

    rim_flat = ptr_cfg.rim_face_width_px * 0.45
    rim_end = max(rim_flat + 1.0, ptr_cfg.rim_face_width_px * 0.95)
    t_rad = 1.0 - np.clip((sd_bell - rim_flat) / (rim_end - rim_flat), 0.0, 1.0)
    w_rad = t_rad * t_rad * (3.0 - 2.0 * t_rad)

    w_rim = np.where(pipe_mask & (sd_bell >= 0.0), w_arc * w_rad, 0.0)
    z_interior = z_interior + w_rim * np.maximum(0.0, ptr_cfg.front_rim_z - z_interior)
    z_interior = np.where(r_rear <= core_clamp_r, ptr_cfg.rear_chamber_z, z_interior)

    # CRITICAL: Extrapolate nearest interior concrete boundary depth across all `~pipe_mask` pixels
    nearest_idx = ndi.distance_transform_edt(
        ~pipe_mask,
        return_distances=False,
        return_indices=True,
    )
    z_extrapolated = np.where(
        pipe_mask,
        z_interior,
        z_interior[nearest_idx[0], nearest_idx[1]],
    )

    norm_d = np.clip(
        (z_extrapolated - ptr_cfg.scene_bg_z) / max(1e-4, ptr_cfg.pipe_depth_amplitude),
        0.0,
        1.0,
    )
    depth_u8 = np.clip(np.round(norm_d * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(depth_u8, mode="L")


def _inpaint_clean_gallery_wall_and_flat_depth(
    pipe_plate_rgb: Image.Image,
    expanded_bg_np: np.ndarray,
    pipe_mask: np.ndarray,
    ptr_cfg: PipeTunnelReliefConfig,
    depth_estimator: DepthAnythingEstimator,
    lama_inpainter: LamaOcclusionInpainter,
) -> tuple[Image.Image, Image.Image]:
    """
    Synthesizes the clean `2700x1800` Binyamin Gallery wall, vertical corner, and floor baseboard
    behind the concrete pipe silhouette via neural LaMa FFC inpainting on a solid dilated pipe mask
    (preserving 100% native `2700x1800` crispness outside the dilated mask, with zero pipe-bore
    imprint so off-axis strafe/yaw disocclusions show only pure gallery wall), plus the clean
    pipe-free far-wall depth map (`scene4_plan_b*_depth.png`).
    """
    exp_h, exp_w, _ = expanded_bg_np.shape
    dil_px = max(4, ptr_cfg.wall_inpaint_dilation_px)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil_px + 1, 2 * dil_px + 1))
    wall_hole_u8 = cv2.dilate((pipe_mask.astype(np.uint8)) * 255, k_dil)
    for eb in ptr_cfg.wall_inpaint_extra_boxes_exp:
        wall_hole_u8[
            max(0, eb.y0) : min(exp_h, eb.y1),
            max(0, eb.x0) : min(exp_w, eb.x1),
        ] = 255

    ds_w, ds_h = ptr_cfg.wall_inpaint_downsample_wh
    plate_ds = pipe_plate_rgb.resize((ds_w, ds_h), Image.Resampling.LANCZOS)
    mask_ds = cv2.resize(wall_hole_u8, (ds_w, ds_h), interpolation=cv2.INTER_NEAREST)
    mask_ds = np.where(mask_ds > 32, 255, 0).astype(np.uint8)
    inpainted_ds = lama_inpainter.inpaint_background(
        plate_ds, mask_ds, dilation_radius_px=2, mask_blur_sigma=0.0
    )
    inpainted_2700 = np.asarray(
        inpainted_ds.resize((exp_w, exp_h), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )

    gx0, gy0, gx1, gy1 = ptr_cfg.wall_grain_patch_exp_xyxy
    grain_src = expanded_bg_np[gy0:gy1, gx0:gx1]
    if grain_src.size > 0 and ptr_cfg.wall_grain_strength > 0.0:
        grain_hp = grain_src - ndi.gaussian_filter(
            grain_src,
            sigma=(
                ptr_cfg.wall_grain_highpass_sigma_px,
                ptr_cfg.wall_grain_highpass_sigma_px,
                0.0,
            ),
        )
        tiled_grain = np.tile(
            grain_hp,
            (exp_h // grain_hp.shape[0] + 1, exp_w // grain_hp.shape[1] + 1, 1),
        )[:exp_h, :exp_w]
        inpainted_2700 = np.clip(
            inpainted_2700 + tiled_grain * ptr_cfg.wall_grain_strength,
            0.0,
            255.0,
        )

    dist_inside_hole = cv2.distanceTransform(
        (wall_hole_u8 > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    t_inpaint = np.clip(
        dist_inside_hole / max(1.0, ptr_cfg.wall_inpaint_feather_px),
        0.0,
        1.0,
    )
    w_inpaint = (t_inpaint * t_inpaint * (3.0 - 2.0 * t_inpaint))[:, :, None]
    pure_clean_wall_np = expanded_bg_np * (1.0 - w_inpaint) + inpainted_2700 * w_inpaint
    pure_clean_wall_rgb = Image.fromarray(
        np.clip(np.round(pure_clean_wall_np), 0, 255).astype(np.uint8),
        mode="RGB",
    )

    flat_wall_depth_img = depth_estimator.build_flat_gallery_wall_depth_uint8(
        clean_wall_rgb=pure_clean_wall_rgb,
        max_u8=ptr_cfg.flat_wall_depth_max_u8,
    )
    return pure_clean_wall_rgb, flat_wall_depth_img


def synthesize_pipe_discontinuity_assets(
    pipe_plate_expanded_rgb: Image.Image,
    dims: CanvasDimensionsConfig,
    ptr_cfg: PipeTunnelReliefConfig,
    depth_estimator: DepthAnythingEstimator,
    sam_extractor: SamBoxMaskExtractor,
    lama_inpainter: LamaOcclusionInpainter,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """
    Synthesizes all four `2700x1800` textures for the true two-surface 3D concrete pipe split:
    1) `clean_wall_bg_rgb`: Clean LaMa-inpainted Binyamin Gallery wall/corner/floor backdrop (`layer0`).
    2) `flat_wall_depth_img`: Pipe-free far-wall depth map (`layer0`, `roomDepthAmplitude = 0.0`).
    3) `pipe_shell_rgba`: Crisp silhouette RGBA concrete pipe + horizontal conduit cutout (`pipe_shell`).
    4) `pipe_shell_depth_img`: Unblurred interior 3D pipe/tunnel depth extrapolated outward via EDT.
    """
    _require_config_fields(ptr_cfg, _PIPE_TUNNEL_REQUIRED_FIELDS, "PipeTunnelReliefConfig")
    if ptr_cfg.rear_chamber_z <= ptr_cfg.scene_bg_z:
        raise ValueError(
            f"PipeTunnelReliefConfig.rear_chamber_z ({ptr_cfg.rear_chamber_z}) must exceed "
            f"scene_bg_z ({ptr_cfg.scene_bg_z})."
        )

    exp_w, exp_h = pipe_plate_expanded_rgb.size
    expanded_bg_np = np.asarray(pipe_plate_expanded_rgb.convert("RGB"), dtype=np.float32)
    pad_x = (exp_w - dims.width) * 0.5
    pad_y = (exp_h - dims.height) * 0.5
    neural_depth = depth_estimator.predict_depth(pipe_plate_expanded_rgb)

    sd_pipe, sd_bell, y_top_full, y_bot_full = _segment_concrete_pipe_signed_distance(
        pipe_plate_rgb=pipe_plate_expanded_rgb,
        expanded_bg_np=expanded_bg_np,
        neural_depth=neural_depth,
        fc_exp_xy=(ptr_cfg.front_center_xy[0] + pad_x, ptr_cfg.front_center_xy[1] + pad_y),
        ptr_cfg=ptr_cfg,
        sam_extractor=sam_extractor,
    )
    pipe_shell_rgba = _build_decontaminated_pipe_shell_rgba(
        expanded_bg_np=expanded_bg_np,
        sd_pipe=sd_pipe,
        ptr_cfg=ptr_cfg,
    )
    pipe_shell_depth_img = _build_extrapolated_pipe_shell_depth_uint8(
        neural_depth=neural_depth,
        sd_pipe=sd_pipe,
        sd_bell=sd_bell,
        y_top_full=y_top_full,
        y_bot_full=y_bot_full,
        dims=dims,
        ptr_cfg=ptr_cfg,
    )
    clean_wall_bg_rgb, flat_wall_depth_img = _inpaint_clean_gallery_wall_and_flat_depth(
        pipe_plate_rgb=pipe_plate_expanded_rgb,
        expanded_bg_np=expanded_bg_np,
        pipe_mask=sd_pipe >= 0.0,
        ptr_cfg=ptr_cfg,
        depth_estimator=depth_estimator,
        lama_inpainter=lama_inpainter,
    )
    return clean_wall_bg_rgb, flat_wall_depth_img, pipe_shell_rgba, pipe_shell_depth_img


class SalientMatteExtractor:
    """Extracts neural alpha foreground mattes via `rembg` (IS-Net / U2-Net)."""

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
    """Segments multi-part architectural sculptures & home elements via Meta Segment Anything (`SamModel`)."""

    def __init__(self, cfg: SegmentationConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.processor = SamProcessor.from_pretrained(cfg.sam_model_id)
        self.model = SamModel.from_pretrained(cfg.sam_model_id).to(self.device)
        self.model.eval()

    def segment_box_group(
        self, rgb_image: Image.Image, boxes: Sequence[BoundingBoxPrompt]
    ) -> np.ndarray:
        width, height = rgb_image.size
        if not boxes:
            return np.zeros((height, width), dtype=np.uint8)

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
        if not spec.sam_fg_boxes and not spec.sam_mid_boxes:
            raise ValueError(f"Scene '{spec.scene_id}' uses sam_boxes but defines zero SAM boxes.")
        rgb_np = np.asarray(rgb_image, dtype=np.uint8)
        alpha_fg = self.segment_box_group(rgb_image, spec.sam_fg_boxes)
        alpha_mid = self.segment_box_group(rgb_image, spec.sam_mid_boxes)
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
            extra_fg = self.segment_box_group(rgb_image, spec.sam_fg_boxes)
            fg_alpha = np.maximum(fg_alpha, extra_fg)
        if spec.sam_mid_boxes:
            extra_mid = self.segment_box_group(rgb_image, spec.sam_mid_boxes)
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

    def segment_point_groups(
        self,
        rgb_image: Image.Image,
        point_groups: Sequence[SamPointGroupPrompt],
        exclude_boxes: Sequence[BoundingBoxPrompt],
        max_pinhole_fill_px: int,
        min_major_component_px: int,
        right_edge_fade_x_start: int | None = None,
    ) -> np.ndarray:
        """
        Segments multi-person/object foregrounds via promptable SAM box + positive/negative point groups,
        computing the image encoder embedding once across all prompts.
        """
        width, height = rgb_image.size
        if not point_groups:
            raise ValueError("segment_point_groups requires at least one SamPointGroupPrompt.")

        base_inputs = self.processor(rgb_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeddings = self.model.get_image_embeddings(base_inputs["pixel_values"])

        union_bool = np.zeros((height, width), dtype=bool)
        for pg in point_groups:
            box_xyxy = [float(v) for v in pg.box.as_xyxy_list()]
            pts = [[float(px), float(py)] for px, py in pg.points_xy]
            lbls = [int(lbl) for lbl in pg.point_labels]
            inp = self.processor(
                rgb_image,
                input_boxes=[[box_xyxy]],
                input_points=[[pts]],
                input_labels=[[lbls]],
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(
                    image_embeddings=image_embeddings,
                    input_points=inp["input_points"],
                    input_labels=inp["input_labels"],
                    input_boxes=inp["input_boxes"],
                    multimask_output=True,
                )
            masks = self.processor.image_processor.post_process_masks(
                out.pred_masks.cpu(),
                inp["original_sizes"].cpu(),
                inp["reshaped_input_sizes"].cpu(),
            )[0]
            scores = out.iou_scores.cpu()[0, 0]
            best_idx = int(scores.argmax().item())
            union_bool |= masks[0, best_idx].numpy()

        combined = (union_bool.astype(np.uint8)) * 255
        for eb in exclude_boxes:
            combined[eb.y0 : eb.y1, eb.x0 : eb.x1] = 0

        if self.cfg.sam_morph_close_px > 1:
            k = self.cfg.sam_morph_close_px
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
        if right_edge_fade_x_start is not None and 0 < right_edge_fade_x_start < width:
            active_xs = np.where(alpha_f[int(height * 0.72) :, :] > 16)[1]
            fade_end = (
                int(active_xs.max()) + 1
                if active_xs.size > 0 and int(active_xs.max()) > right_edge_fade_x_start
                else width
            )
            span = max(8, fade_end - right_edge_fade_x_start)
            ramp = 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, span, dtype=np.float32))
            y_start = int(round(height * 0.72))
            alpha_f[y_start:, right_edge_fade_x_start : right_edge_fade_x_start + span] *= ramp[
                None, :
            ]
            if right_edge_fade_x_start + span < width:
                alpha_f[y_start:, right_edge_fade_x_start + span :] = 0.0

        if self.cfg.edge_feather_sigma > 0:
            alpha_f = ndi.gaussian_filter(alpha_f, sigma=self.cfg.edge_feather_sigma)
        return np.clip(np.round(alpha_f), 0, 255).astype(np.uint8)

    def extract_point_group_fg_and_inner_mask(
        self,
        inner_rgb: Image.Image,
        raw_fg_expanded_rgb: Image.Image | None,
        spec: SceneBuildSpec,
        dims: CanvasDimensionsConfig,
        outpaint_cfg: OutpaintingConfig,
        matte_extractor: SalientMatteExtractor | None = None,
    ) -> tuple[Image.Image, Image.Image, np.ndarray, Image.Image | None]:
        """
        Extracts either an expanded (e.g. 2700x1800) or inner (1500x1000) foreground RGBA layer
        via promptable SAM point groups (plus optional upright-oriented IS-Net hair/eyeglass matting),
        an empty inner midground canvas, the 1500x1000 inner occlusion mask for LaMa background
        inpainting, and the registered expanded RGB plate (when fg_is_expanded=True) for reuse on
        intermediate person-depth layers.
        """
        inner_w, inner_h = dims.width, dims.height
        mid_empty = Image.fromarray(np.zeros((inner_h, inner_w, 4), dtype=np.uint8), mode="RGBA")

        if spec.fg_is_expanded:
            if raw_fg_expanded_rgb is None:
                raise ValueError(
                    f"Scene '{spec.scene_id}' sets fg_is_expanded=True but raw_fg_expanded_rgb is None."
                )
            exp_plate_rgb, corr = register_expanded_foreground_plate(
                inner_rgb=inner_rgb,
                raw_fg_expanded_rgb=raw_fg_expanded_rgb,
                dims=dims,
                outpaint_cfg=outpaint_cfg,
                min_registration_corr=spec.min_registration_corr,
            )
            print(
                f"  -> Registered expanded foreground plate "
                f"({exp_plate_rgb.size[0]}x{exp_plate_rgb.size[1]}, corr={corr:.4f})"
            )
            alpha_exp = self.segment_point_groups(
                rgb_image=exp_plate_rgb,
                point_groups=spec.sam_fg_point_groups,
                exclude_boxes=spec.sam_fg_exclude_boxes,
                max_pinhole_fill_px=spec.sam_max_pinhole_fill_px,
                min_major_component_px=spec.sam_min_major_component_px,
                right_edge_fade_x_start=spec.sam_fg_right_edge_fade_x_start,
            )
            rgb_exp_u8 = np.asarray(exp_plate_rgb, dtype=np.uint8)
            if spec.fg_oriented_matte_boxes:
                if matte_extractor is None:
                    raise RuntimeError(
                        f"Scene '{spec.scene_id}' defines fg_oriented_matte_boxes but matte_extractor is None."
                    )
                alpha_exp = merge_sam_and_oriented_mattes_u8(
                    rgb_image=exp_plate_rgb,
                    sam_u8=alpha_exp,
                    oriented_boxes=spec.fg_oriented_matte_boxes,
                    support_dilate_px=spec.sam_core_support_dilate_px,
                    right_edge_fade_x_start=spec.sam_fg_right_edge_fade_x_start,
                    rembg_session=matte_extractor.session,
                )
                rgb_exp_u8 = decontaminate_matte_edge_rgb(rgb_exp_u8, alpha_exp)
            exp_w, exp_h = exp_plate_rgb.size
            pad_x = (exp_w - inner_w) // 2
            pad_y = (exp_h - inner_h) // 2
            inner_alpha = alpha_exp[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]
            fg_rgba = Image.fromarray(np.dstack([rgb_exp_u8, alpha_exp]), mode="RGBA")
            union_mask_u8 = np.where(inner_alpha > 8, 255, 0).astype(np.uint8)
            return fg_rgba, mid_empty, union_mask_u8, exp_plate_rgb

        alpha_in = self.segment_point_groups(
            rgb_image=inner_rgb,
            point_groups=spec.sam_fg_point_groups,
            exclude_boxes=spec.sam_fg_exclude_boxes,
            max_pinhole_fill_px=spec.sam_max_pinhole_fill_px,
            min_major_component_px=spec.sam_min_major_component_px,
            right_edge_fade_x_start=spec.sam_fg_right_edge_fade_x_start,
        )
        rgb_in_u8 = np.asarray(inner_rgb, dtype=np.uint8)
        if spec.fg_oriented_matte_boxes:
            if matte_extractor is None:
                raise RuntimeError(
                    f"Scene '{spec.scene_id}' defines fg_oriented_matte_boxes but matte_extractor is None."
                )
            alpha_in = merge_sam_and_oriented_mattes_u8(
                rgb_image=inner_rgb,
                sam_u8=alpha_in,
                oriented_boxes=spec.fg_oriented_matte_boxes,
                support_dilate_px=spec.sam_core_support_dilate_px,
                right_edge_fade_x_start=spec.sam_fg_right_edge_fade_x_start,
                rembg_session=matte_extractor.session,
            )
            rgb_in_u8 = decontaminate_matte_edge_rgb(rgb_in_u8, alpha_in)
        fg_rgba = Image.fromarray(np.dstack([rgb_in_u8, alpha_in]), mode="RGBA")
        union_mask_u8 = np.where(alpha_in > 8, 255, 0).astype(np.uint8)
        return fg_rgba, mid_empty, union_mask_u8, None

    def extract_extra_person_depth_layer(
        self,
        plate_rgb: Image.Image,
        el_spec: ExtraDepthLayerSpec,
        dims: CanvasDimensionsConfig,
        support_dilate_px: int,
        matte_extractor: SalientMatteExtractor | None,
        lama_inpainter: LamaOcclusionInpainter | None,
    ) -> tuple[Image.Image, np.ndarray]:
        """
        Extracts an intermediate 3D person layer (e.g. Guy at Z=-76 behind the floating chest pod)
        via SAM point groups + upright-oriented IS-Net hair matting + optional LaMa disocclusion
        inpainting of shoulder fabric occluded by nearer foreground subjects.
        Returns (person_rgba_img, inner_1500x1000_occlusion_mask_u8).
        """
        alpha_u8 = self.segment_point_groups(
            rgb_image=plate_rgb,
            point_groups=el_spec.sam_point_groups,
            exclude_boxes=el_spec.exclude_boxes,
            max_pinhole_fill_px=el_spec.max_pinhole_fill_px,
            min_major_component_px=el_spec.min_major_component_px,
            right_edge_fade_x_start=None,
        )
        if el_spec.oriented_matte_boxes:
            if matte_extractor is None:
                raise RuntimeError(
                    f"Extra layer '{el_spec.layer_id}' defines oriented_matte_boxes without SalientMatteExtractor."
                )
            alpha_u8 = merge_sam_and_oriented_mattes_u8(
                rgb_image=plate_rgb,
                sam_u8=alpha_u8,
                oriented_boxes=el_spec.oriented_matte_boxes,
                support_dilate_px=support_dilate_px,
                right_edge_fade_x_start=None,
                rembg_session=matte_extractor.session,
            )

        inner_w, inner_h = dims.width, dims.height
        pw, ph = plate_rgb.size
        if el_spec.is_expanded:
            pad_x = (pw - inner_w) // 2
            pad_y = (ph - inner_h) // 2
            inner_alpha = alpha_u8[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]
        else:
            inner_alpha = alpha_u8
        inner_mask_u8 = np.where(inner_alpha > 8, 255, 0).astype(np.uint8)

        working_rgb = plate_rgb
        working_alpha = alpha_u8
        if el_spec.occluder_inpaint_boxes or el_spec.right_shoulder_extend_px > 0:
            if lama_inpainter is None:
                raise RuntimeError(
                    f"Extra layer '{el_spec.layer_id}' requires LamaOcclusionInpainter for shoulder disocclusion."
                )
            working_rgb, working_alpha = extend_occluded_person_plate(
                plate_rgb=working_rgb,
                alpha_u8=working_alpha,
                occluder_inpaint_boxes=el_spec.occluder_inpaint_boxes,
                right_shoulder_extend_box=el_spec.right_shoulder_extend_box,
                right_shoulder_extend_px=el_spec.right_shoulder_extend_px,
                lama_inpainter=lama_inpainter,
            )

        rgb_u8 = decontaminate_matte_edge_rgb(np.asarray(working_rgb, dtype=np.uint8), working_alpha)
        return Image.fromarray(np.dstack([rgb_u8, working_alpha]), mode="RGBA"), inner_mask_u8


def extract_oriented_isnet_roi_alpha_f32(
    rgb_image: Image.Image,
    omb: OrientedMatteBoxSpec,
    rembg_session: object,
) -> np.ndarray:
    """
    Rotates a subject's head/hair/glasses ROI to canonical upright portrait orientation before
    running IS-Net (`isnet-general-use`), then rotates the continuous 8-bit neural alpha matte
    back into full-canvas coordinates with cosine perimeter tapering and optional color-bleed suppression.
    """
    w_full, h_full = rgb_image.size
    x0 = max(0, min(w_full, omb.box.x0))
    y0 = max(0, min(h_full, omb.box.y0))
    x1 = max(0, min(w_full, omb.box.x1))
    y1 = max(0, min(h_full, omb.box.y1))
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError(f"Invalid OrientedMatteBoxSpec dimensions: {omb.box}")

    crop = rgb_image.crop((x0, y0, x1, y1)).convert("RGB")
    cw, ch = crop.size
    if abs(omb.upright_angle_deg) > 0.1:
        rotated = crop.rotate(
            omb.upright_angle_deg, resample=Image.Resampling.BICUBIC, expand=True
        )
        cutout = remove(rotated, session=rembg_session, only_mask=False)
        if not isinstance(cutout, Image.Image):
            raise RuntimeError("rembg failed to return a PIL Image during oriented hair matting.")
        cut_a = cutout.convert("RGBA").split()[3]
        unrotated = cut_a.rotate(
            -omb.upright_angle_deg, resample=Image.Resampling.BICUBIC, expand=False
        )
        bw, bh = unrotated.size
        ox = (bw - cw) // 2
        oy = (bh - ch) // 2
        roi_a = np.asarray(unrotated.crop((ox, oy, ox + cw, oy + ch)), dtype=np.float32)
    else:
        cutout = remove(crop, session=rembg_session, only_mask=False)
        if not isinstance(cutout, Image.Image):
            raise RuntimeError("rembg failed to return a PIL Image during oriented hair matting.")
        roi_a = np.asarray(cutout.convert("RGBA"), dtype=np.float32)[:, :, 3]

    m = omb.taper_margin_px
    if m > 1:
        taper = np.ones((ch, cw), dtype=np.float32)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, m, dtype=np.float32))
        if x0 > 0 and cw > 2 * m:
            taper[:, :m] *= ramp[None, :]
        if x1 < w_full and cw > 2 * m:
            taper[:, -m:] *= ramp[::-1][None, :]
        if y0 > 0 and ch > 2 * m:
            taper[:m, :] *= ramp[:, None]
        if y1 < h_full and ch > 2 * m:
            taper[-m:, :] *= ramp[::-1][:, None]
        roi_a = roi_a * taper

    full_a = np.zeros((h_full, w_full), dtype=np.float32)
    full_a[y0:y1, x0:x1] = roi_a

    for eb in omb.exclude_boxes:
        full_a[eb.y0 : eb.y1, eb.x0 : eb.x1] = 0.0

    if omb.suppress_warm_box is not None:
        sb = omb.suppress_warm_box
        rgb_np = np.asarray(rgb_image, dtype=np.float32)
        rg_diff = rgb_np[:, :, 0] - rgb_np[:, :, 1]
        warm_zone = np.zeros((h_full, w_full), dtype=bool)
        warm_zone[sb.y0 : sb.y1, sb.x0 : sb.x1] = (
            rg_diff[sb.y0 : sb.y1, sb.x0 : sb.x1] > omb.suppress_warm_rg_diff
        )
        warm_smooth = ndi.gaussian_filter(warm_zone.astype(np.float32), sigma=omb.suppress_warm_sigma)
        full_a *= np.clip(1.0 - warm_smooth * omb.suppress_warm_gain, 0.0, 1.0)

    return full_a


def merge_sam_and_oriented_mattes_u8(
    rgb_image: Image.Image,
    sam_u8: np.ndarray,
    oriented_boxes: Sequence[OrientedMatteBoxSpec],
    support_dilate_px: int,
    right_edge_fade_x_start: int | None,
    rembg_session: object,
) -> np.ndarray:
    """
    Combines promptable SAM body segmentation with upright-oriented IS-Net hair/glasses mattes:
    - Erodes binary SAM slightly inside head ROIs so continuous 8-bit hair strands and eyeglass
      rims are driven by IS-Net rather than stair-stepped binary masks.
    - Restricts IS-Net alpha to a dilated neighborhood (`support_dilate_px`) around the SAM core
      so attached wispy hair strands survive intact while distant background objects vanish.
    """
    if not oriented_boxes:
        return sam_u8

    h_full, w_full = sam_u8.shape
    dk = support_dilate_px if support_dilate_px % 2 == 1 else support_dilate_px + 1
    core_support = (
        cv2.dilate(
            (sam_u8 > 128).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk)),
        )
        > 0
    )

    base_sam_f = sam_u8.astype(np.float32).copy()
    max_isnet = np.zeros((h_full, w_full), dtype=np.float32)

    for omb in oriented_boxes:
        roi_isnet = extract_oriented_isnet_roi_alpha_f32(rgb_image, omb, rembg_session)
        max_isnet = np.maximum(max_isnet, roi_isnet)
        ek = omb.sam_core_erode_px
        if ek > 1:
            if ek % 2 == 0:
                ek += 1
            eroded = cv2.erode(
                sam_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
            ).astype(np.float32)
            m = max(4, omb.taper_margin_px)
            bx0 = min(w_full, omb.box.x0 + m)
            by0 = min(h_full, omb.box.y0 + m)
            bx1 = max(bx0, omb.box.x1 - (m if omb.box.x1 < w_full else 0))
            by1 = max(by0, omb.box.y1 - (m if omb.box.y1 < h_full else 0))
            base_sam_f[by0:by1, bx0:bx1] = eroded[by0:by1, bx0:bx1]

    if right_edge_fade_x_start is not None and 0 < right_edge_fade_x_start < w_full:
        y_start = int(round(h_full * 0.72))
        max_isnet[y_start:, right_edge_fade_x_start:] = np.minimum(
            max_isnet[y_start:, right_edge_fade_x_start:],
            sam_u8[y_start:, right_edge_fade_x_start:].astype(np.float32),
        )

    merged = np.where(core_support, np.maximum(base_sam_f, max_isnet), 0.0)
    return np.clip(np.round(merged), 0, 255).astype(np.uint8)


def decontaminate_matte_edge_rgb(rgb_u8: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    """
    Unmixes local background color from semi-transparent hair/rim pixels (`6 <= alpha <= 242`)
    via normalized Gaussian background propagation so wispy strands show zero stone/mirror halo.
    """
    rgb_f = rgb_u8.astype(np.float32)
    a_norm = alpha_u8.astype(np.float32) / 255.0
    bg_weight = (alpha_u8 < 6).astype(np.float32)
    w_blur = ndi.gaussian_filter(bg_weight, sigma=9.0)
    valid_bg = w_blur > 1e-3

    out = rgb_f.copy()
    edge = (alpha_u8 >= 6) & (alpha_u8 <= 242) & valid_bg
    if not np.any(edge):
        return rgb_u8

    a_safe = np.maximum(a_norm, 0.36)
    for ch in range(3):
        bg_ch = ndi.gaussian_filter(rgb_f[:, :, ch] * bg_weight, sigma=9.0) / np.maximum(
            w_blur, 1e-4
        )
        unmixed = np.clip((rgb_f[:, :, ch] - (1.0 - a_safe) * bg_ch) / a_safe, 0.0, 255.0)
        out[:, :, ch] = np.where(edge, unmixed, rgb_f[:, :, ch])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def extend_occluded_person_plate(
    plate_rgb: Image.Image,
    alpha_u8: np.ndarray,
    occluder_inpaint_boxes: Sequence[BoundingBoxPrompt],
    right_shoulder_extend_box: BoundingBoxPrompt | None,
    right_shoulder_extend_px: int,
    lama_inpainter: LamaOcclusionInpainter,
) -> tuple[Image.Image, np.ndarray]:
    """
    Disoccludes clothing/shoulder texture on a midground person where a nearer foreground person
    overlaps in the original photograph, using local LaMa FFC inpainting + directional alpha extension.
    """
    w_full, h_full = plate_rgb.size
    rgb_np = np.asarray(plate_rgb, dtype=np.uint8).copy()

    if occluder_inpaint_boxes:
        ux0 = max(0, min(b.x0 for b in occluder_inpaint_boxes) - 240)
        uy0 = max(0, min(b.y0 for b in occluder_inpaint_boxes) - 180)
        ux1 = min(w_full, max(b.x1 for b in occluder_inpaint_boxes) + 180)
        uy1 = min(h_full, max(b.y1 for b in occluder_inpaint_boxes) + 40)
        sub_rgb = Image.fromarray(rgb_np[uy0:uy1, ux0:ux1], mode="RGB")
        sub_mask = np.zeros((uy1 - uy0, ux1 - ux0), dtype=np.uint8)
        for b in occluder_inpaint_boxes:
            sub_mask[
                max(0, b.y0 - uy0) : min(uy1 - uy0, b.y1 - uy0),
                max(0, b.x0 - ux0) : min(ux1 - ux0, b.x1 - ux0),
            ] = 255
        inpainted_sub = lama_inpainter.inpaint_background(
            sub_rgb, sub_mask, dilation_radius_px=6, mask_blur_sigma=1.5
        )
        rgb_np[uy0:uy1, ux0:ux1] = np.asarray(inpainted_sub, dtype=np.uint8)

    ext_alpha = alpha_u8.copy()
    if right_shoulder_extend_box is not None and right_shoulder_extend_px > 0:
        sb = right_shoulder_extend_box
        ext_px = right_shoulder_extend_px
        # Rightward-only morphological kernel: anchor sits on left half so foreground shifts right into occluded zone
        kern = np.zeros((5, 2 * ext_px + 1), dtype=np.uint8)
        kern[:, ext_px:] = 1
        dil = cv2.dilate(ext_alpha, kern)
        dil_soft = np.clip(ndi.gaussian_filter(dil.astype(np.float32), sigma=3.2), 0.0, 255.0)
        roi_cur = ext_alpha[sb.y0 : sb.y1, sb.x0 : sb.x1].astype(np.float32)
        roi_ext = np.maximum(roi_cur, dil_soft[sb.y0 : sb.y1, sb.x0 : sb.x1])
        ext_alpha[sb.y0 : sb.y1, sb.x0 : sb.x1] = np.round(roi_ext).astype(np.uint8)

    return Image.fromarray(rgb_np, mode="RGB"), ext_alpha


def register_expanded_foreground_plate(
    inner_rgb: Image.Image,
    raw_fg_expanded_rgb: Image.Image,
    dims: CanvasDimensionsConfig,
    outpaint_cfg: OutpaintingConfig,
    min_registration_corr: float,
) -> tuple[Image.Image, float]:
    """
    Registers an expanded foreground outpaint onto the canonical expanded grid (e.g. 2700x1800)
    by matching the central high-detail subject region of inner_rgb, color-harmonizing the outer
    collar, and feather-locking 100% authentic inner_rgb into [pad_x..pad_x+W, pad_y..pad_y+H].
    """
    inner_w, inner_h = dims.width, dims.height
    exp_w, exp_h = outpaint_cfg.expanded_dimensions(dims)
    pad_x = (exp_w - inner_w) // 2
    pad_y = (exp_h - inner_h) // 2

    inner_np = np.asarray(inner_rgb.convert("RGB"), dtype=np.float32)
    exp_raw = np.asarray(raw_fg_expanded_rgb.convert("RGB"), dtype=np.float32)
    eh, ew, _ = exp_raw.shape

    in_g = cv2.cvtColor(inner_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ex_g = cv2.cvtColor(exp_raw.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    tx0 = int(round(inner_w * 0.30))
    tx1 = int(round(inner_w * 0.62))
    ty0 = int(round(inner_h * 0.24))
    ty1 = int(round(inner_h * 0.86))
    tpl = in_g[ty0:ty1, tx0:tx1]

    best_corr = -1.0
    best_s = 1.0
    best_loc = (0, 0)
    for s in np.linspace(
        outpaint_cfg.similarity_coarse_scale_min,
        outpaint_cfg.similarity_coarse_scale_max,
        64,
    ):
        tw = int(round(tpl.shape[1] * s))
        th = int(round(tpl.shape[0] * s))
        if tw < 24 or th < 24 or tw >= ew - 4 or th >= eh - 4:
            continue
        res = cv2.matchTemplate(ex_g, cv2.resize(tpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv > best_corr:
            best_corr = float(mv)
            best_s = float(s)
            best_loc = ml

    if best_corr < min_registration_corr:
        raise RuntimeError(
            f"Expanded foreground registration correlation {best_corr:.4f} < {min_registration_corr:.4f}."
        )

    inv_s = 1.0 / best_s
    shift_x = (pad_x + tx0) - inv_s * best_loc[0]
    shift_y = (pad_y + ty0) - inv_s * best_loc[1]
    M = np.array([[inv_s, 0.0, shift_x], [0.0, inv_s, shift_y]], dtype=np.float32)
    warped = cv2.warpAffine(
        exp_raw, M, (exp_w, exp_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    band = max(16, outpaint_cfg.feather_collar_px)
    collar = np.zeros((inner_h, inner_w), dtype=bool)
    collar[:band, :] = True
    collar[-band:, :] = True
    collar[:, :band] = True
    collar[:, -band:] = True
    w_inner = warped[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]
    for ch in range(3):
        im = float(np.mean(inner_np[:, :, ch][collar]))
        wm = float(np.mean(w_inner[:, :, ch][collar]))
        istd = float(np.std(inner_np[:, :, ch][collar]))
        wstd = max(1.0, float(np.std(w_inner[:, :, ch][collar])))
        gain = np.clip(istd / wstd, outpaint_cfg.gain_clamp_min, outpaint_cfg.gain_clamp_max)
        warped[:, :, ch] = np.clip(warped[:, :, ch] * gain + (im - gain * wm), 0.0, 255.0)

    cf = outpaint_cfg.feather_collar_px
    fx = np.ones(inner_w, dtype=np.float32)
    fy = np.ones(inner_h, dtype=np.float32)
    rf = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, cf, dtype=np.float32))
    fx[:cf], fx[-cf:] = rf, rf[::-1]
    fy[:cf], fy[-cf:] = rf, rf[::-1]
    a_in = np.outer(fy, fx)[:, :, None]
    warped[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w] = (
        a_in * inner_np + (1.0 - a_in) * warped[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]
    )
    return Image.fromarray(np.clip(np.round(warped), 0, 255).astype(np.uint8), mode="RGB"), best_corr


def estimate_paper_background(
    gray_f32: np.ndarray,
    draw_cfg: GraphiteDrawingConfig,
) -> np.ndarray:
    """Estimates smooth local paper illumination via downsampled percentile filtering."""
    h, w = gray_f32.shape
    small_gray = cv2.resize(
        gray_f32,
        (draw_cfg.paper_downsample_width, draw_cfg.paper_downsample_height),
        interpolation=cv2.INTER_AREA,
    )
    small_bg = ndi.percentile_filter(
        small_gray,
        percentile=draw_cfg.paper_bg_percentile,
        size=draw_cfg.paper_filter_size,
    )
    return cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_CUBIC)


def extract_graphite_stroke_alpha(
    gray_f32: np.ndarray,
    paper_bg_f32: np.ndarray,
    boxes: Sequence[DrawingStrokeBox],
    draw_cfg: GraphiteDrawingConfig,
) -> np.ndarray:
    """
    Extracts crisp ink-and-hatching alpha masks for drawn islands on paper while keeping
    open vertical spaces between thin stilts 100% transparent.
    """
    h, w = gray_f32.shape
    ink_delta = (paper_bg_f32 - gray_f32).copy()
    b = draw_cfg.frame_border_ignore_px
    if b > 0:
        ink_delta[:b, :] = 0.0
        ink_delta[-b:, :] = 0.0
        ink_delta[:, :b] = 0.0
        ink_delta[:, -b:] = 0.0

    union_u8 = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        roi_delta = ink_delta[box.y0 : box.y1, box.x0 : box.x1]
        raw = ((roi_delta >= box.min_ink_delta).astype(np.uint8)) * 255
        if box.morph_close_px > 1:
            k = box.morph_close_px if box.morph_close_px % 2 == 1 else box.morph_close_px + 1
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kern)
        if box.max_hole_fill_area_px > 0:
            inv = (raw == 0).astype(np.uint8)
            nh, h_lbls, h_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
            rh, rw = raw.shape
            for hid in range(1, nh):
                hx = int(h_stats[hid, cv2.CC_STAT_LEFT])
                hy = int(h_stats[hid, cv2.CC_STAT_TOP])
                hw = int(h_stats[hid, cv2.CC_STAT_WIDTH])
                hh = int(h_stats[hid, cv2.CC_STAT_HEIGHT])
                harea = int(h_stats[hid, cv2.CC_STAT_AREA])
                touches_edge = hx == 0 or hy == 0 or (hx + hw >= rw) or (hy + hh >= rh)
                if (not touches_edge) and harea <= box.max_hole_fill_area_px:
                    raw[h_lbls == hid] = 255

        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(
            (raw > 0).astype(np.uint8), connectivity=8
        )
        clean_roi = np.zeros_like(raw)
        for cid in range(1, n_lbl):
            if int(stats[cid, cv2.CC_STAT_AREA]) >= draw_cfg.min_component_area_px:
                clean_roi[lbls == cid] = 255
        union_u8[box.y0 : box.y1, box.x0 : box.x1] = np.maximum(
            union_u8[box.y0 : box.y1, box.x0 : box.x1], clean_roi
        )

    if draw_cfg.edge_feather_sigma > 0:
        feathered = ndi.gaussian_filter(union_u8.astype(np.float32), sigma=draw_cfg.edge_feather_sigma)
        return np.clip(np.round(feathered), 0, 255).astype(np.uint8)
    return union_u8


def suppress_paper_scan_border(rgb_image: Image.Image, border_px: int) -> Image.Image:
    """Replaces thin outer scan-border shadow on paper drawings with mirrored interior paper tone."""
    if border_px <= 0:
        return rgb_image
    arr = np.asarray(rgb_image.convert("RGB"), dtype=np.uint8).copy()
    b = border_px
    arr[:b, :, :] = arr[b + 2 : 2 * b + 2, :, :][::-1, :, :]
    arr[-b:, :, :] = arr[-(2 * b + 2) : -(b + 2), :, :][::-1, :, :]
    arr[:, :b, :] = arr[:, b + 2 : 2 * b + 2, :][:, ::-1, :]
    arr[:, -b:, :] = arr[:, -(2 * b + 2) : -(b + 2), :][:, ::-1, :]
    return Image.fromarray(arr, mode="RGB")


def extract_graphite_drawing_layers(
    rgb_image: Image.Image,
    spec: SceneBuildSpec,
    draw_cfg: GraphiteDrawingConfig,
) -> tuple[Image.Image, Image.Image, np.ndarray, Image.Image]:
    """
    Segments a graphite-on-paper drawing into multi-plane FG & MID island cutouts,
    returns the union inpainting mask, and returns a border-cleaned RGB sheet for LaMa.
    """
    rgb_np = np.asarray(rgb_image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    paper_bg = estimate_paper_background(gray, draw_cfg)

    fg_alpha = extract_graphite_stroke_alpha(gray, paper_bg, spec.drawing_fg_boxes, draw_cfg)
    mid_alpha = extract_graphite_stroke_alpha(gray, paper_bg, spec.drawing_mid_boxes, draw_cfg)
    mid_alpha = np.where(fg_alpha > 96, 0, mid_alpha).astype(np.uint8)

    union_mask_u8 = np.where((fg_alpha > 8) | (mid_alpha > 8), 255, 0).astype(np.uint8)
    borderless_rgb = suppress_paper_scan_border(rgb_image, draw_cfg.frame_border_ignore_px)
    return (
        Image.fromarray(np.dstack([rgb_np, fg_alpha]), mode="RGBA"),
        Image.fromarray(np.dstack([rgb_np, mid_alpha]), mode="RGBA"),
        union_mask_u8,
        borderless_rgb,
    )


def _perimeter_taper_mask(height: int, width: int) -> np.ndarray:
    margin_x = max(6, int(round(width * 0.06)))
    margin_y = max(6, int(round(height * 0.10)))
    tx = np.ones(width, dtype=np.float32)
    ty = np.ones(height, dtype=np.float32)
    rx = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, margin_x, dtype=np.float32))
    ry = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, margin_y, dtype=np.float32))
    tx[:margin_x] = rx
    tx[-margin_x:] = rx[::-1]
    ty[:margin_y] = ry
    ty[-margin_y:] = ry[::-1]
    return np.outer(ty, tx)


def extract_oval_board_island_rgba(
    darc_rgb: Image.Image,
    crop_xyxy: tuple[int, int, int, int],
    dest_wh: tuple[int, int],
    min_delta: float,
    draw_cfg: GraphiteDrawingConfig,
) -> Image.Image:
    """
    Segments a drawn/painted island off an oval mixed-media board (e.g., IMG_DARC_0164),
    suppressing the outer oval frame and tapering perimeter alpha to zero so zero rectangular edges appear.
    """
    sw = draw_cfg.oval_board_scaled_width
    sh = draw_cfg.oval_board_scaled_height
    scaled = np.asarray(darc_rgb.convert("RGB").resize((sw, sh), Image.Resampling.LANCZOS), dtype=np.float32)

    cx, cy = draw_cfg.oval_center_xy
    rx, ry = draw_cfg.oval_radii_xy
    yy_full, xx_full = np.ogrid[:sh, :sw]
    oval_inside = (((xx_full - cx) / rx) ** 2 + ((yy_full - cy) / ry) ** 2) <= 1.0

    x0, y0, x1, y1 = crop_xyxy
    crop = scaled[y0:y1, x0:x1].copy()
    oval_crop = oval_inside[y0:y1, x0:x1]
    if not np.any(oval_crop):
        raise ValueError(f"Crop {crop_xyxy} falls outside oval board interior.")

    gray_crop = 0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]
    bright_mask = oval_crop & (gray_crop >= np.percentile(gray_crop[oval_crop], 75))
    bg_rgb = np.median(crop[bright_mask], axis=0)

    color_dist = np.linalg.norm(crop - bg_rgb[None, None, :], axis=2)
    color_dist = np.where(oval_crop, color_dist, 0.0)

    raw_alpha = np.clip((color_dist - min_delta) / 20.0, 0.0, 1.0) * 255.0
    binary = (raw_alpha > 48).astype(np.uint8) * 255
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    nl, lbls, stats, _ = cv2.connectedComponentsWithStats((closed > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(closed)
    for cid in range(1, nl):
        if int(stats[cid, cv2.CC_STAT_AREA]) >= draw_cfg.oval_min_component_area_px:
            keep[lbls == cid] = 255

    alpha_f = np.where(keep > 0, np.clip(raw_alpha * 1.45, 0.0, 255.0), 0.0).astype(np.float32)
    taper = _perimeter_taper_mask(alpha_f.shape[0], alpha_f.shape[1])
    alpha_smooth_f = np.clip(ndi.gaussian_filter(alpha_f * taper, sigma=0.75), 0.0, 255.0)
    a_norm = (alpha_smooth_f / 255.0)[:, :, None]
    dehalo_rgb = np.where(
        a_norm > 0.02,
        np.clip((crop - (1.0 - a_norm) * bg_rgb[None, None, :]) / np.maximum(a_norm, 0.30), 0.0, 255.0),
        crop,
    )
    rgba = np.dstack([dehalo_rgb.astype(np.uint8), alpha_smooth_f.astype(np.uint8)])
    return Image.fromarray(rgba, mode="RGBA").resize(dest_wh, Image.Resampling.LANCZOS)


def extract_graphite_crop_rgba(
    src_rgb: Image.Image,
    crop_xyxy: tuple[int, int, int, int],
    dest_wh: tuple[int, int],
    min_delta: float,
    dims: CanvasDimensionsConfig,
    draw_cfg: GraphiteDrawingConfig,
) -> Image.Image:
    """Segments crisp dark graphite ink strokes on 1500x1000 coordinates without opaque paper fill."""
    resized = np.asarray(
        src_rgb.convert("RGB").resize((dims.width, dims.height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32)
    paper_bg = estimate_paper_background(gray, draw_cfg)
    x0, y0, x1, y1 = crop_xyxy
    box = DrawingStrokeBox(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        min_ink_delta=min_delta,
        morph_close_px=3,
        max_hole_fill_area_px=0,
    )
    full_alpha = extract_graphite_stroke_alpha(gray, paper_bg, (box,), draw_cfg)
    crop_rgb = resized[y0:y1, x0:x1].astype(np.float32)
    crop_bg = paper_bg[y0:y1, x0:x1, None]
    crop_alpha = full_alpha[y0:y1, x0:x1].astype(np.float32)
    taper = _perimeter_taper_mask(crop_alpha.shape[0], crop_alpha.shape[1])
    alpha_f = np.clip(crop_alpha * taper, 0.0, 255.0)
    a_norm = (alpha_f / 255.0)[:, :, None]
    dehalo_rgb = np.where(
        a_norm > 0.02,
        np.clip((crop_rgb - (1.0 - a_norm) * crop_bg) / np.maximum(a_norm, 0.32), 0.0, 255.0),
        crop_rgb,
    )
    alpha_u8 = np.round(alpha_f).astype(np.uint8)
    return Image.fromarray(np.dstack([dehalo_rgb.astype(np.uint8), alpha_u8]), mode="RGBA").resize(
        dest_wh, Image.Resampling.LANCZOS
    )


def extract_clean_graphite_island_rgba(
    drawing_rgb: Image.Image,
    ov: LayerOverlaySpec,
    fe_cfg: FloatingElementExtractionConfig,
) -> Image.Image:
    """
    Histogram-based optical paper unmixing for pencil-sketched floating islands:
    - Estimates smooth local paper RGB via downsampled percentile filter + MAD noise floor.
    - Maps paper background strictly to alpha = 0 (zero morphological closing or hole-filling,
      preventing any white paper bleed between cross-hatched pencil strokes).
    - Optionally suppresses thin tether strings and unmixes paper RGB from semi-transparent stroke edges.
    """
    if ov.src_crop_norm is None:
        raise ValueError("graphite_unmix_island overlay requires src_crop_norm.")

    rgb_np = np.asarray(drawing_rgb.convert("RGB"), dtype=np.float32)
    h_full, w_full, _ = rgb_np.shape
    x0 = max(0, int(round(ov.src_crop_norm[0] * w_full)))
    y0 = max(0, int(round(ov.src_crop_norm[1] * h_full)))
    x1 = min(w_full, int(round(ov.src_crop_norm[2] * w_full)))
    y1 = min(h_full, int(round(ov.src_crop_norm[3] * h_full)))
    crop = rgb_np[y0:y1, x0:x1]
    ch, cw, _ = crop.shape

    gray = 0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]
    ds_w = max(16, cw // fe_cfg.paper_downsample_divisor)
    ds_h = max(16, ch // fe_cfg.paper_downsample_divisor)
    small_rgb = cv2.resize(crop, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
    paper_rgb = np.stack(
        [
            cv2.resize(
                ndi.percentile_filter(
                    small_rgb[:, :, c],
                    percentile=fe_cfg.paper_percentile,
                    size=fe_cfg.paper_filter_size,
                ),
                (cw, ch),
                interpolation=cv2.INTER_CUBIC,
            )
            for c in range(3)
        ],
        axis=2,
    )
    paper_gray = (
        0.299 * paper_rgb[:, :, 0] + 0.587 * paper_rgb[:, :, 1] + 0.114 * paper_rgb[:, :, 2]
    )

    diff = paper_gray - gray
    paper_samples = diff[diff < fe_cfg.paper_sample_max_diff]
    if paper_samples.size > 50:
        mad = float(np.median(np.abs(paper_samples - np.median(paper_samples))))
        sigma_paper = max(fe_cfg.min_paper_sigma, 1.4826 * mad)
    else:
        sigma_paper = fe_cfg.min_paper_sigma * 1.4

    floor_delta = max(fe_cfg.min_delta_floor, fe_cfg.noise_floor_sigma_mult * sigma_paper)
    stroke_deltas = diff[diff > floor_delta]
    full_ink_delta = (
        float(np.percentile(stroke_deltas, fe_cfg.ink_upper_percentile))
        if stroke_deltas.size > 30
        else floor_delta + fe_cfg.min_ink_span * 2.0
    )
    full_ink_delta = max(floor_delta + fe_cfg.min_ink_span, full_ink_delta)

    norm_density = np.clip((diff - floor_delta) / (full_ink_delta - floor_delta), 0.0, 1.0)
    alpha_raw = np.power(norm_density, ov.gamma)

    if ov.suppress_thin_tethers:
        binary = (alpha_raw > fe_cfg.tether_binary_threshold).astype(np.uint8)
        local_density = ndi.uniform_filter(
            binary.astype(np.float32), size=fe_cfg.tether_window_hw
        )
        core = binary & (local_density > fe_cfg.tether_min_density)
        dk = fe_cfg.tether_dilate_px
        core_dilated = (
            cv2.dilate(
                core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
            )
            > 0
        )
        alpha_raw = np.where(core_dilated, alpha_raw, 0.0)

    fg_mask = (alpha_raw > fe_cfg.min_graphite_alpha).astype(np.uint8)
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    keep = np.zeros_like(fg_mask, dtype=bool)
    for cid in range(1, n_lbl):
        if int(stats[cid, cv2.CC_STAT_AREA]) >= fe_cfg.min_graphite_component_px:
            keep |= lbls == cid
    alpha_clean = np.where(keep, alpha_raw, 0.0)
    alpha_boosted = np.clip(alpha_clean * fe_cfg.stroke_presence_boost, 0.0, 1.0)
    taper = _perimeter_taper_mask(ch, cw)
    alpha_final = np.clip(alpha_boosted * taper, 0.0, 1.0)

    a_safe = np.maximum(alpha_final[:, :, None], fe_cfg.unmix_alpha_floor)
    c_unmixed = np.clip((crop - (1.0 - a_safe) * paper_rgb) / a_safe, 0.0, 255.0)
    tint_arr = np.array(fe_cfg.graphite_tint_rgb, dtype=np.float32)[None, None, :]
    c_graphite = (
        fe_cfg.preserve_shading_blend * np.minimum(c_unmixed, fe_cfg.max_shaded_lum)
        + (1.0 - fe_cfg.preserve_shading_blend) * tint_arr
    )

    rgba = np.dstack(
        [
            np.clip(np.round(c_graphite), 0, 255).astype(np.uint8),
            np.clip(np.round(alpha_final * 255.0), 0, 255).astype(np.uint8),
        ]
    )
    nz_y, nz_x = np.where(rgba[:, :, 3] > 6)
    if nz_x.size > 10:
        rgba = rgba[
            max(0, int(nz_y.min()) - 4) : min(ch, int(nz_y.max()) + 5),
            max(0, int(nz_x.min()) - 4) : min(cw, int(nz_x.max()) + 5),
        ]
    return Image.fromarray(rgba, mode="RGBA")


def extract_clean_brown_paper_capsule_rgba(
    capsules_rgb: Image.Image,
    ov: LayerOverlaySpec,
    fe_cfg: FloatingElementExtractionConfig,
    rembg_session: object,
) -> Image.Image:
    """
    Extracts a brown packing-paper floating sculpture at native resolution via IS-Net neural matting,
    neutral gallery-wall color decontamination, optional upper pale-wisp trimming, and directional
    sun/shadow lighting harmony.
    """
    if ov.src_crop_norm is None:
        raise ValueError("isnet_dehalo_capsule overlay requires src_crop_norm.")

    w_full, h_full = capsules_rgb.size
    x0 = max(0, int(round(ov.src_crop_norm[0] * w_full)))
    y0 = max(0, int(round(ov.src_crop_norm[1] * h_full)))
    x1 = min(w_full, int(round(ov.src_crop_norm[2] * w_full)))
    y1 = min(h_full, int(round(ov.src_crop_norm[3] * h_full)))

    pad = fe_cfg.capsule_context_pad_px
    px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
    px1, py1 = min(w_full, x1 + pad), min(h_full, y1 + pad)
    padded_crop = capsules_rgb.crop((px0, py0, px1, py1)).convert("RGB")

    cutout = remove(padded_crop, session=rembg_session, only_mask=False)
    if not isinstance(cutout, Image.Image):
        raise RuntimeError("rembg failed to return a PIL Image during capsule extraction.")
    cut_np = np.asarray(cutout.convert("RGBA"), dtype=np.float32)
    cx0, cy0 = x0 - px0, y0 - py0
    sub = cut_np[cy0 : cy0 + (y1 - y0), cx0 : cx0 + (x1 - x0)].copy()
    rgb = sub[:, :, :3]
    raw_a = sub[:, :, 3] / 255.0

    bg_pixels = rgb[raw_a < fe_cfg.wall_bg_max_alpha]
    wall_rgb = (
        np.percentile(bg_pixels, fe_cfg.wall_bg_percentile, axis=0)
        if bg_pixels.shape[0] > 40
        else np.array(fe_cfg.fallback_wall_rgb, dtype=np.float32)
    )

    warmth = rgb[:, :, 0] - rgb[:, :, 2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    wall_dist = np.linalg.norm(rgb - wall_rgb[None, None, :], axis=2)

    solid_core = (raw_a > fe_cfg.solid_core_min_alpha) & (
        (warmth > fe_cfg.solid_core_min_warmth) | (lum < fe_cfg.solid_core_dark_lum)
    )
    pk = fe_cfg.porthole_close_px
    core_closed = cv2.morphologyEx(
        solid_core.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pk, pk)),
    )
    inv = (core_closed == 0).astype(np.uint8)
    nh, h_lbls, h_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    sh, sw = core_closed.shape
    porthole_interior = np.zeros_like(solid_core)
    for hid in range(1, nh):
        hx = int(h_stats[hid, cv2.CC_STAT_LEFT])
        hy = int(h_stats[hid, cv2.CC_STAT_TOP])
        hw = int(h_stats[hid, cv2.CC_STAT_WIDTH])
        hh = int(h_stats[hid, cv2.CC_STAT_HEIGHT])
        touches = hx == 0 or hy == 0 or (hx + hw >= sw) or (hy + hh >= sh)
        if not touches and int(h_stats[hid, cv2.CC_STAT_AREA]) < fe_cfg.porthole_max_area_px:
            porthole_interior |= h_lbls == hid

    pale_wall_bleed = (
        (~porthole_interior)
        & (warmth < fe_cfg.pale_bleed_max_warmth)
        & (lum > fe_cfg.pale_bleed_min_lum)
        & (wall_dist < fe_cfg.pale_bleed_max_wall_dist)
    )
    raw_a = np.where(pale_wall_bleed, 0.0, raw_a)

    if ov.trim_upper_pale_wisps:
        yy_g, xx_g = np.ogrid[:sh, :sw]
        wx0, wx1 = fe_cfg.wisp_x_range_norm
        wisp_zone = (
            (yy_g < int(round(fe_cfg.wisp_y_max_norm * sh)))
            & (xx_g > int(round(wx0 * sw)))
            & (xx_g < int(round(wx1 * sw)))
        )
        is_wisp = wisp_zone & (warmth < fe_cfg.wisp_max_warmth) & (lum > fe_cfg.wisp_min_lum)
        raw_a = np.where(is_wisp, 0.0, raw_a)
        porthole_interior = np.where(is_wisp, False, porthole_interior)

    a_span = max(0.05, fe_cfg.capsule_alpha_high - fe_cfg.capsule_alpha_low)
    alpha = np.clip((raw_a - fe_cfg.capsule_alpha_low) / a_span, 0.0, 1.0)
    alpha = np.where(porthole_interior, np.maximum(alpha, raw_a), alpha)

    fg_u8 = (alpha > 0.10).astype(np.uint8)
    nl, lbls, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    if ov.keep_largest_component_only and nl > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = lbls == largest
    else:
        keep = np.zeros_like(fg_u8, dtype=bool)
        for cid in range(1, nl):
            if int(stats[cid, cv2.CC_STAT_AREA]) >= fe_cfg.min_capsule_component_px:
                keep |= lbls == cid
    alpha = np.where(keep, alpha, 0.0)
    if ov.trim_vertical_tether_ends:
        end_band = max(4, int(round(0.13 * sh)))
        max_stem_w = max(8, int(round(0.14 * sw)))
        for ry in list(range(0, end_band)) + list(range(max(0, sh - end_band), sh)):
            xs_on = np.where(alpha[ry, :] > 0.08)[0]
            if xs_on.size > 0 and (int(xs_on.max()) - int(xs_on.min()) + 1) <= max_stem_w:
                alpha[ry, :] = 0.0
    alpha = np.clip(ndi.gaussian_filter(alpha, sigma=fe_cfg.capsule_alpha_sigma), 0.0, 1.0)

    a_safe = np.maximum(alpha[:, :, None], fe_cfg.capsule_unmix_alpha_floor)
    dehalo = np.where(
        alpha[:, :, None] > 0.02,
        np.clip((rgb - (1.0 - a_safe) * wall_rgb[None, None, :]) / a_safe, 0.0, 255.0),
        rgb,
    )

    yy = np.linspace(1.0, -0.8, sh, dtype=np.float32)[:, None]
    xx = np.linspace(-0.5, 1.0, sw, dtype=np.float32)[None, :]
    sun_grad = np.clip(0.55 * yy + 0.45 * xx, -1.0, 1.0)[:, :, None]
    warm_gain = 1.0 + ov.warm_sun_harmony * sun_grad * np.array(
        [1.05, 0.96, 0.80], dtype=np.float32
    )[None, None, :]
    harmonized = np.clip(dehalo * warm_gain, 0.0, 255.0)

    if ov.shadow_pocket_lighting:
        sx = np.linspace(0.0, 1.0, sw, dtype=np.float32)[None, :]
        sy = np.linspace(1.0, 0.0, sh, dtype=np.float32)[:, None]
        sun_edge = np.clip(0.65 * sx + 0.35 * sy, 0.0, 1.0)[:, :, None]
        cool_sh = np.array(fe_cfg.shadow_pocket_cool_rgb, dtype=np.float32)[None, None, :]
        warm_rm = np.array(fe_cfg.shadow_pocket_warm_rgb, dtype=np.float32)[None, None, :]
        harmonized = np.clip(
            harmonized * ((1.0 - sun_edge) * cool_sh + sun_edge * warm_rm), 0.0, 255.0
        )

    rgba = np.dstack(
        [np.round(harmonized).astype(np.uint8), np.round(alpha * 255.0).astype(np.uint8)]
    )
    nz_y, nz_x = np.where(rgba[:, :, 3] > 8)
    if nz_x.size > 20:
        rgba = rgba[
            max(0, int(nz_y.min()) - 3) : min(sh, int(nz_y.max()) + 4),
            max(0, int(nz_x.min()) - 3) : min(sw, int(nz_x.max()) + 4),
        ]
    return Image.fromarray(rgba, mode="RGBA")


def _render_overlay_patch_and_dest(
    ov: LayerOverlaySpec,
    project_root: Path,
    dims: CanvasDimensionsConfig,
    draw_cfg: GraphiteDrawingConfig,
    fe_cfg: FloatingElementExtractionConfig,
    sam_extractor: SamBoxMaskExtractor | None,
    matte_extractor: SalientMatteExtractor | None,
) -> tuple[Image.Image, tuple[int, int]]:
    """Extracts, resizes, applies opacity, and resolves top-left placement for a single LayerOverlaySpec."""
    ov_path = project_root / ov.source_image_relpath
    if not ov_path.is_file():
        raise FileNotFoundError(f"Required overlay source image missing: {ov_path}")

    if ov.extraction_mode == "darc_oval_island":
        patch = extract_oval_board_island_rgba(
            darc_rgb=Image.open(ov_path),
            crop_xyxy=ov.src_crop_xyxy,
            dest_wh=ov.dest_wh,
            min_delta=ov.min_delta,
            draw_cfg=draw_cfg,
        )
    elif ov.extraction_mode == "graphite_crop":
        patch = extract_graphite_crop_rgba(
            src_rgb=Image.open(ov_path),
            crop_xyxy=ov.src_crop_xyxy,
            dest_wh=ov.dest_wh,
            min_delta=ov.min_delta,
            dims=dims,
            draw_cfg=draw_cfg,
        )
    elif ov.extraction_mode == "rgba_asset":
        src_rgba = Image.open(ov_path).convert("RGBA")
        x0, y0, x1, y1 = ov.src_crop_xyxy
        patch = src_rgba.crop((x0, y0, x1, y1)).resize(ov.dest_wh, Image.Resampling.LANCZOS)
    elif ov.extraction_mode == "sam_crop":
        if sam_extractor is None:
            raise RuntimeError("SamBoxMaskExtractor is required for sam_crop overlays.")
        src_rgb = Image.open(ov_path).convert("RGB").resize(
            (dims.width, dims.height), Image.Resampling.LANCZOS
        )
        x0, y0, x1, y1 = ov.src_crop_xyxy
        alpha_full = sam_extractor.segment_box_group(
            src_rgb, (BoundingBoxPrompt(x0=x0, y0=y0, x1=x1, y1=y1),)
        )
        src_np = np.asarray(src_rgb, dtype=np.uint8)
        patch = Image.fromarray(
            np.dstack([src_np[y0:y1, x0:x1], alpha_full[y0:y1, x0:x1]]), mode="RGBA"
        ).resize(ov.dest_wh, Image.Resampling.LANCZOS)
    elif ov.extraction_mode == "graphite_unmix_island":
        raw_patch = extract_clean_graphite_island_rgba(Image.open(ov_path), ov, fe_cfg)
        tw = ov.dest_wh[0]
        th = (
            max(16, int(round(tw * (raw_patch.height / max(1, raw_patch.width)))))
            if ov.dest_center_xy is not None
            else ov.dest_wh[1]
        )
        patch = raw_patch.resize((tw, th), Image.Resampling.LANCZOS)
    elif ov.extraction_mode == "isnet_dehalo_capsule":
        if matte_extractor is None:
            raise RuntimeError("SalientMatteExtractor is required for isnet_dehalo_capsule overlays.")
        raw_patch = extract_clean_brown_paper_capsule_rgba(
            Image.open(ov_path), ov, fe_cfg, matte_extractor.session
        )
        tw = ov.dest_wh[0]
        th = (
            max(16, int(round(tw * (raw_patch.height / max(1, raw_patch.width)))))
            if ov.dest_center_xy is not None
            else ov.dest_wh[1]
        )
        patch = raw_patch.resize((tw, th), Image.Resampling.LANCZOS)
    else:
        raise ValueError(f"Unsupported overlay extraction_mode: {ov.extraction_mode}")

    if ov.opacity < 0.999:
        p_np = np.asarray(patch, dtype=np.float32).copy()
        p_np[:, :, 3] = np.clip(np.round(p_np[:, :, 3] * ov.opacity), 0, 255)
        patch = Image.fromarray(p_np.astype(np.uint8), mode="RGBA")

    if ov.dest_center_xy is not None:
        dest_xy = (
            int(round(ov.dest_center_xy[0] - patch.width * 0.5)),
            int(round(ov.dest_center_xy[1] - patch.height * 0.5)),
        )
    else:
        dest_xy = ov.dest_xy
    return patch, dest_xy


def apply_layer_overlays(
    base_layer_rgba: Image.Image,
    overlays: Sequence[LayerOverlaySpec],
    project_root: Path,
    dims: CanvasDimensionsConfig,
    draw_cfg: GraphiteDrawingConfig,
    sam_extractor: SamBoxMaskExtractor | None,
    matte_extractor: SalientMatteExtractor | None = None,
    fe_cfg: FloatingElementExtractionConfig | None = None,
) -> Image.Image:
    """Composites configured cross-scene sculptural or drawn elements onto an RGBA depth layer."""
    if not overlays:
        return base_layer_rgba

    effective_fe = fe_cfg if fe_cfg is not None else default_floating_element_extraction_config()
    canvas = base_layer_rgba.convert("RGBA").copy()
    for ov in overlays:
        patch, dest_xy = _render_overlay_patch_and_dest(
            ov=ov,
            project_root=project_root,
            dims=dims,
            draw_cfg=draw_cfg,
            fe_cfg=effective_fe,
            sam_extractor=sam_extractor,
            matte_extractor=matte_extractor,
        )
        canvas.alpha_composite(patch, dest=dest_xy)

    return canvas


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
    """Synthesizes occluded wall/floor/garden/paper background behind dilated silhouettes via LaMa."""

    def __init__(self, cfg: InpaintingConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu")
        model_path = download_model(LAMA_MODEL_URL)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def inpaint_background(
        self,
        rgb_image: Image.Image,
        binary_mask_u8: np.ndarray,
        dilation_radius_px: int | None = None,
        mask_blur_sigma: float | None = None,
    ) -> Image.Image:
        orig_w, orig_h = rgb_image.size
        radius = (
            dilation_radius_px if dilation_radius_px is not None else self.cfg.dilation_radius_px
        )
        blur_sigma = mask_blur_sigma if mask_blur_sigma is not None else self.cfg.mask_blur_sigma

        kernel_size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(binary_mask_u8, kernel, iterations=1)
        if blur_sigma > 0:
            dilated_f = ndi.gaussian_filter(dilated.astype(np.float32), sigma=blur_sigma)
            dilated = np.where(dilated_f > 64.0, 255, 0).astype(np.uint8)

        mask_img = Image.fromarray(dilated, mode="L")
        img_tensor, mask_tensor = prepare_img_and_mask(rgb_image, mask_img, self.device)
        with torch.inference_mode():
            inpainted_tensor = self.model(img_tensor, mask_tensor)
        res_np = inpainted_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        res_np = res_np[:orig_h, :orig_w, :]
        res_u8 = np.clip(np.round(res_np * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(res_u8, mode="RGB")


def _warp_preserves_canvas_bounds(
    H_mat: np.ndarray, exp_w: int, exp_h: int, max_border_pull_px: float = 6.0
) -> bool:
    """Verifies that warping by H_mat does not pull source borders inward and expose edge replication."""
    try:
        H_inv = np.linalg.inv(H_mat)
    except np.linalg.LinAlgError:
        return False
    corners = np.array(
        [[0.0, 0.0, 1.0], [exp_w, 0.0, 1.0], [exp_w, exp_h, 1.0], [0.0, exp_h, 1.0]],
        dtype=np.float64,
    ).T
    src = H_inv @ corners
    if np.any(np.abs(src[2]) < 1e-6):
        return False
    sx = src[0] / src[2]
    sy = src[1] / src[2]
    return bool(
        np.all(sx >= -max_border_pull_px)
        and np.all(sx <= exp_w + max_border_pull_px)
        and np.all(sy >= -max_border_pull_px)
        and np.all(sy <= exp_h + max_border_pull_px)
    )


def _try_homography_warp(
    inner_gray: np.ndarray,
    cand_rgb: np.ndarray,
    cand_gray: np.ndarray,
    exp_w: int,
    exp_h: int,
    pad_x: int,
    pad_y: int,
    outpaint_cfg: OutpaintingConfig,
) -> tuple[np.ndarray, float] | None:
    inner_h, inner_w = inner_gray.shape
    patch_w = outpaint_cfg.homography_patch_width
    patch_h = outpaint_cfg.homography_patch_height
    search_margin = outpaint_cfg.homography_search_margin_px
    xs_inner = [40, (inner_w - patch_w) // 2, inner_w - patch_w - 40]
    ys_inner = [40, (inner_h - patch_h) // 2, inner_h - patch_h - 40]

    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []
    for iy in ys_inner:
        for ix in xs_inner:
            tpl = inner_gray[iy : iy + patch_h, ix : ix + patch_w]
            if float(np.std(tpl)) < 3.0:
                continue
            nom_x = pad_x + ix
            nom_y = pad_y + iy
            sx0 = max(0, nom_x - search_margin)
            sy0 = max(0, nom_y - search_margin)
            sx1 = min(exp_w, nom_x + patch_w + search_margin)
            sy1 = min(exp_h, nom_y + patch_h + search_margin)
            roi = cand_gray[sy0:sy1, sx0:sx1]
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_loc = cv2.minMaxLoc(res)
            if max_v >= outpaint_cfg.homography_min_patch_corr:
                matched_x = sx0 + max_loc[0] + patch_w * 0.5
                matched_y = sy0 + max_loc[1] + patch_h * 0.5
                target_x = nom_x + patch_w * 0.5
                target_y = nom_y + patch_h * 0.5
                src_pts.append([matched_x, matched_y])
                dst_pts.append([target_x, target_y])

    if len(src_pts) < 4:
        return None

    H_mat, _ = cv2.findHomography(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts, dtype=np.float32),
        cv2.RANSAC,
        ransacReprojThreshold=4.0,
    )
    if H_mat is None or not _warp_preserves_canvas_bounds(
        H_mat, exp_w, exp_h, outpaint_cfg.max_warp_border_pull_px
    ):
        return None

    warped_rgb = cv2.warpPerspective(
        cand_rgb,
        H_mat,
        (exp_w, exp_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_center_gray = cv2.cvtColor(
        np.clip(warped_rgb[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w], 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    )
    corr = float(cv2.matchTemplate(warped_center_gray, inner_gray, cv2.TM_CCOEFF_NORMED)[0, 0])
    return warped_rgb, corr


def _try_salient_similarity_warp(
    inner_gray: np.ndarray,
    cand_rgb: np.ndarray,
    cand_gray: np.ndarray,
    exp_w: int,
    exp_h: int,
    pad_x: int,
    pad_y: int,
    outpaint_cfg: OutpaintingConfig,
) -> tuple[np.ndarray, float]:
    inner_h, inner_w = inner_gray.shape
    gx = cv2.Sobel(inner_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(inner_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    thresh = float(np.percentile(grad, outpaint_cfg.similarity_salient_grad_percentile))
    ys, xs = np.where(grad >= thresh)
    bx0 = max(20, int(np.percentile(xs, 3)) - 30)
    bx1 = min(inner_w - 20, int(np.percentile(xs, 97)) + 30)
    by0 = max(20, int(np.percentile(ys, 3)) - 30)
    by1 = min(inner_h - 20, int(np.percentile(ys, 97)) + 30)
    tpl_salient = inner_gray[by0:by1, bx0:bx1]
    sal_w = bx1 - bx0
    sal_h = by1 - by0
    sal_cx_in_inner = bx0 + sal_w * 0.5
    sal_cy_in_inner = by0 + sal_h * 0.5

    ds = outpaint_cfg.similarity_coarse_downsample
    cand_s = cv2.resize(cand_gray, (0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    best_score = -1.0
    best_scale = 1.0
    best_scx = exp_w * 0.5
    best_scy = exp_h * 0.5

    for scale in np.linspace(
        outpaint_cfg.similarity_coarse_scale_min,
        outpaint_cfg.similarity_coarse_scale_max,
        outpaint_cfg.similarity_coarse_steps,
    ):
        tw = max(24, int(round(sal_w * scale * ds)))
        th = max(24, int(round(sal_h * scale * ds)))
        if tw >= cand_s.shape[1] - 4 or th >= cand_s.shape[0] - 4:
            continue
        tpl_s = cv2.resize(tpl_salient, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(cand_s, tpl_s, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)
        if max_v > best_score:
            best_score = float(max_v)
            best_scale = float(scale)
            best_scx = (max_loc[0] + tw * 0.5) / ds
            best_scy = (max_loc[1] + th * 0.5) / ds

    fine_best_score = -1.0
    fine_scale = best_scale
    fine_scx = best_scx
    fine_scy = best_scy
    span = outpaint_cfg.similarity_fine_half_span
    fmargin = outpaint_cfg.similarity_fine_margin_px
    for scale in np.linspace(
        max(0.35, best_scale - span),
        min(1.25, best_scale + span),
        outpaint_cfg.similarity_fine_steps,
    ):
        tw = max(32, int(round(sal_w * scale)))
        th = max(32, int(round(sal_h * scale)))
        rx0 = max(0, int(round(best_scx - tw * 0.5 - fmargin)))
        ry0 = max(0, int(round(best_scy - th * 0.5 - fmargin)))
        rx1 = min(exp_w, int(round(best_scx + tw * 0.5 + fmargin)))
        ry1 = min(exp_h, int(round(best_scy + th * 0.5 + fmargin)))
        if rx1 - rx0 <= tw or ry1 - ry0 <= th:
            continue
        tpl_f = cv2.resize(
            tpl_salient,
            (tw, th),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        res = cv2.matchTemplate(cand_gray[ry0:ry1, rx0:rx1], tpl_f, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)
        if max_v > fine_best_score:
            fine_best_score = float(max_v)
            fine_scale = float(scale)
            fine_scx = rx0 + max_loc[0] + tw * 0.5
            fine_scy = ry0 + max_loc[1] + th * 0.5

    inv_s = 1.0 / fine_scale
    tx = (pad_x + sal_cx_in_inner) - inv_s * fine_scx
    ty = (pad_y + sal_cy_in_inner) - inv_s * fine_scy
    H_sim = np.array([[inv_s, 0.0, tx], [0.0, inv_s, ty], [0.0, 0.0, 1.0]], dtype=np.float32)
    if not _warp_preserves_canvas_bounds(
        H_sim, exp_w, exp_h, outpaint_cfg.max_warp_border_pull_px
    ):
        return cand_rgb, -1.0

    warped_rgb = cv2.warpPerspective(
        cand_rgb,
        H_sim,
        (exp_w, exp_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_center_gray = cv2.cvtColor(
        np.clip(warped_rgb[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w], 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    )
    corr = float(cv2.matchTemplate(warped_center_gray, inner_gray, cv2.TM_CCOEFF_NORMED)[0, 0])
    return warped_rgb, corr


def _solve_harmonic_seam_offset_field(
    inner_np: np.ndarray,
    harmonized_rgb: np.ndarray,
    pad_x: int,
    pad_y: int,
    downsample: int,
    num_iters: int,
) -> np.ndarray:
    """
    Solves a 2D Dirichlet harmonic RGB offset field on the outer outpaint collar with zero-flux
    Neumann outer boundaries, pinned to the smoothed boundary difference along the 4 edges of
    `inner_np`. Eliminates uneven per-side outpaint brightness steps without altering `inner_np`.
    """
    exp_h, exp_w, _ = harmonized_rgb.shape
    inner_h, inner_w, _ = inner_np.shape
    ds = max(2, downsample)
    gh = max(24, exp_h // ds)
    gw = max(24, exp_w // ds)

    diff_inner = inner_np - harmonized_rgb[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]
    diff_smooth = ndi.gaussian_filter(diff_inner, sigma=(18.0, 18.0, 0.0))

    ipx0 = int(round(pad_x * gw / exp_w))
    ipx1 = max(ipx0 + 4, int(round((pad_x + inner_w) * gw / exp_w)))
    ipy0 = int(round(pad_y * gh / exp_h))
    ipy1 = max(ipy0 + 4, int(round((pad_y + inner_h) * gh / exp_h)))

    diff_ds = cv2.resize(
        diff_smooth,
        (ipx1 - ipx0, ipy1 - ipy0),
        interpolation=cv2.INTER_AREA,
    )
    field = np.zeros((gh, gw, 3), dtype=np.float32)
    fixed = np.zeros((gh, gw), dtype=bool)
    fixed[ipy0:ipy1, ipx0:ipx1] = True
    field[ipy0:ipy1, ipx0:ipx1] = diff_ds

    for _ in range(max(10, num_iters)):
        padded = np.pad(field, ((1, 1), (1, 1), (0, 0)), mode="edge")
        avg = 0.25 * (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        )
        field = np.where(fixed[:, :, None], field, avg)

    return cv2.resize(field, (exp_w, exp_h), interpolation=cv2.INTER_CUBIC)


def compose_outpainted_room(
    inner_rgb: Image.Image,
    raw_outpaint_rgb: Image.Image,
    dims: CanvasDimensionsConfig,
    outpaint_cfg: OutpaintingConfig,
    min_registration_corr: float | None = None,
    feather_collar_px: int | None = None,
    registration_core_margin_xy: tuple[int, int] = (0, 0),
    harmonize_seam_field: bool = False,
    seam_harmonic_downsample: int = 8,
    seam_harmonic_iters: int = 240,
) -> tuple[Image.Image, float]:
    """
    Registers a wide-angle outpainted room/landscape/paper plate onto the canonical expanded grid
    (e.g. 2700x1800 around 1500x1000) by selecting the highest-correlation alignment among
    identity, 9-patch homography, and Sobel-salient multiscale similarity; verifies correlation
    against the configured minimum; harmonizes RGB gain/bias across the seam collar; and
    feather-locks the exact authentic LaMa-inpainted plate into the center.
    """
    inner_w, inner_h = dims.width, dims.height
    if inner_rgb.size != (inner_w, inner_h):
        raise ValueError(f"Expected inner_rgb size {(inner_w, inner_h)}, got {inner_rgb.size}")

    exp_w, exp_h = outpaint_cfg.expanded_dimensions(dims)
    pad_x = (exp_w - inner_w) // 2
    pad_y = (exp_h - inner_h) // 2

    inner_np = np.asarray(inner_rgb.convert("RGB"), dtype=np.float32)
    inner_gray = cv2.cvtColor(inner_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    cand_rgb = np.asarray(
        raw_outpaint_rgb.convert("RGB").resize((exp_w, exp_h), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    cand_gray = cv2.cvtColor(cand_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    core_mx, core_my = registration_core_margin_xy
    cx0 = max(0, core_mx)
    cy0 = max(0, core_my)
    cx1 = min(inner_w, inner_w - core_mx)
    cy1 = min(inner_h, inner_h - core_my)
    if cx1 - cx0 < 120 or cy1 - cy0 < 120:
        raise ValueError(f"Invalid registration_core_margin_xy {registration_core_margin_xy}")

    def _measure_core_corr(warped_arr: np.ndarray) -> float:
        center_u8 = np.clip(
            warped_arr[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w], 0, 255
        ).astype(np.uint8)
        center_gray = cv2.cvtColor(center_u8, cv2.COLOR_RGB2GRAY)
        return float(
            cv2.matchTemplate(
                center_gray[cy0:cy1, cx0:cx1],
                inner_gray[cy0:cy1, cx0:cx1],
                cv2.TM_CCOEFF_NORMED,
            )[0, 0]
        )

    # Candidate 1: Direct identity alignment (optimal for guided pre-centered outpaints)
    direct_corr = _measure_core_corr(cand_rgb)
    best_warped_rgb = cand_rgb
    best_corr = direct_corr

    # Candidate 2: 9-patch projective homography
    hom_candidate = _try_homography_warp(
        inner_gray, cand_rgb, cand_gray, exp_w, exp_h, pad_x, pad_y, outpaint_cfg
    )
    if hom_candidate is not None:
        hom_corr = _measure_core_corr(hom_candidate[0])
        if hom_corr > best_corr:
            best_warped_rgb, best_corr = hom_candidate[0], hom_corr

    # Candidate 3: Sobel-salient multiscale similarity (handles low-variance paper/sky & zoom offsets)
    req_corr = (
        min_registration_corr
        if min_registration_corr is not None
        else outpaint_cfg.min_registration_corr
    )
    if best_corr < req_corr + 0.02:
        sim_rgb, _ = _try_salient_similarity_warp(
            inner_gray, cand_rgb, cand_gray, exp_w, exp_h, pad_x, pad_y, outpaint_cfg
        )
        sim_corr = _measure_core_corr(sim_rgb)
        if sim_corr > best_corr:
            best_warped_rgb, best_corr = sim_rgb, sim_corr

    if best_corr < req_corr:
        raise RuntimeError(
            f"Outpaint registration correlation {best_corr:.4f} is below required "
            f"minimum {req_corr:.4f}."
        )

    band = outpaint_cfg.color_match_band_px
    collar_mask = np.zeros((inner_h, inner_w), dtype=bool)
    collar_mask[cy0 : cy0 + band, cx0:cx1] = True
    collar_mask[cy1 - band : cy1, cx0:cx1] = True
    collar_mask[cy0:cy1, cx0 : cx0 + band] = True
    collar_mask[cy0:cy1, cx1 - band : cx1] = True

    warped_crop = best_warped_rgb[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w]
    harmonized_rgb = best_warped_rgb.copy()
    for ch in range(3):
        ref_vals = inner_np[:, :, ch][collar_mask]
        src_vals = warped_crop[:, :, ch][collar_mask]
        src_std = float(np.std(src_vals))
        ref_std = float(np.std(ref_vals))
        gain = np.clip(
            ref_std / max(1.0, src_std),
            outpaint_cfg.gain_clamp_min,
            outpaint_cfg.gain_clamp_max,
        )
        bias = float(np.mean(ref_vals)) - gain * float(np.mean(src_vals))
        harmonized_rgb[:, :, ch] = harmonized_rgb[:, :, ch] * gain + bias

    if harmonize_seam_field:
        seam_offset = _solve_harmonic_seam_offset_field(
            inner_np=inner_np,
            harmonized_rgb=harmonized_rgb,
            pad_x=pad_x,
            pad_y=pad_y,
            downsample=seam_harmonic_downsample,
            num_iters=seam_harmonic_iters,
        )
        harmonized_rgb = harmonized_rgb + seam_offset

    collar = feather_collar_px if feather_collar_px is not None else outpaint_cfg.feather_collar_px
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
    return Image.fromarray(blended_u8, mode="RGB"), best_corr


def build_aperture_reflection_layer(
    canvas_width: int,
    canvas_height: int,
    reflection_src: Image.Image,
    apertures: Sequence[ApertureBox],
) -> Image.Image:
    """
    Renders subtle architectural mirror/tunnel reflections strictly bounded to physical
    apertures (never floating unmasked over bare drywall).
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


def build_ceiling_mirror_reflection_layer(
    canvas_width: int,
    canvas_height: int,
    mirror_bg_inner_rgb: Image.Image,
    cmr_cfg: CeilingMirrorReflectionConfig,
    project_root: Path,
    dims: CanvasDimensionsConfig,
    draw_cfg: GraphiteDrawingConfig,
    fe_cfg: FloatingElementExtractionConfig,
    sam_extractor: SamBoxMaskExtractor | None,
    matte_extractor: SalientMatteExtractor | None,
    outpaint_expansion_factor: float = 1.80,
) -> Image.Image:
    """
    Synthesizes perspective-convergent overhead mirror reflections (e.g. Norman Foster's
    L'Ombrière du Vieux-Port stainless-steel pavilion):
    - Maps each floating artwork's direct center p_direct toward mirror-normal vanishing point c_vp
      via p_refl = c_vp + rho * (p_direct - c_vp), with rho = h / (2 - h).
    - Vertically flips each scaled element across the horizontal ceiling mirror plane and modulates
      brightness/coolness according to local sunlit vs shaded mirror illumination.
    """
    refl_layer = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    mirror_np = np.asarray(
        mirror_bg_inner_rgb.convert("RGB").resize(
            (canvas_width, canvas_height), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    )
    cvp_x, cvp_y = cmr_cfg.vanishing_point_xy
    steel_tint = np.array(fe_cfg.mirror_steel_tint_rgb, dtype=np.float32)[None, None, :]
    exp_w = int(round(dims.width * outpaint_expansion_factor))
    exp_h = int(round(dims.height * outpaint_expansion_factor))
    pad_x = (exp_w - dims.width) // 2
    pad_y = (exp_h - dims.height) // 2

    for item in cmr_cfg.items:
        direct_patch, dest_xy = _render_overlay_patch_and_dest(
            ov=item.overlay,
            project_root=project_root,
            dims=dims,
            draw_cfg=draw_cfg,
            fe_cfg=fe_cfg,
            sam_extractor=sam_extractor,
            matte_extractor=matte_extractor,
        )
        cx = dest_xy[0] + direct_patch.width * 0.5
        cy = dest_xy[1] + direct_patch.height * 0.5
        if item.overlay_is_expanded:
            cx -= pad_x
            cy -= pad_y
        h_norm = item.height_norm
        rho = h_norm / max(0.05, 2.0 - h_norm)

        rcx = cvp_x + rho * (cx - cvp_x)
        rcy = cvp_y + rho * (cy - cvp_y)
        rw = max(14, int(round(direct_patch.width * rho)))
        rh = max(14, int(round(direct_patch.height * rho)))

        scaled = direct_patch.resize((rw, rh), Image.Resampling.LANCZOS)
        flipped = ImageOps.flip(scaled)
        blur_sigma = fe_cfg.mirror_blur_base_sigma + fe_cfg.mirror_blur_dist_gain * (1.0 - h_norm)
        if blur_sigma > 0.15:
            flipped = flipped.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

        arr = np.asarray(flipped, dtype=np.float32).copy()
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3] / 255.0

        rx0 = max(0, min(canvas_width - rw, int(round(rcx - rw * 0.5))))
        ry0 = max(0, min(canvas_height - rh, int(round(rcy - rh * 0.5))))
        bg_crop = mirror_np[ry0 : ry0 + rh, rx0 : rx0 + rw]
        if bg_crop.shape[0] == rh and bg_crop.shape[1] == rw:
            bg_lum = (
                0.299 * bg_crop[:, :, 0] + 0.587 * bg_crop[:, :, 1] + 0.114 * bg_crop[:, :, 2]
            )
            illum_factor = np.clip(
                fe_cfg.mirror_illum_base
                + fe_cfg.mirror_illum_gain
                * ((bg_lum - fe_cfg.mirror_bg_dark_lum) / max(1.0, fe_cfg.mirror_bg_lum_span)),
                fe_cfg.mirror_illum_min,
                fe_cfg.mirror_illum_max,
            )[:, :, None]
        else:
            illum_factor = 1.0

        refl_rgb = np.clip(rgb * steel_tint * illum_factor, 0.0, 255.0)
        refl_alpha = np.clip(alpha * item.steel_reflectance, 0.0, 1.0)
        refl_patch = Image.fromarray(
            np.dstack(
                [np.round(refl_rgb).astype(np.uint8), np.round(refl_alpha * 255.0).astype(np.uint8)]
            ),
            mode="RGBA",
        )
        rx = int(round(rcx - rw * 0.5))
        ry = int(round(rcy - rh * 0.5))
        refl_layer.alpha_composite(refl_patch, dest=(rx, ry))

    return refl_layer


_DIFFUSE_FUSION_REQUIRED_FIELDS: tuple[str, ...] = (
    "source_image_relpath",
    "src_crop_xyxy",
    "dest_center_inner_xy",
    "dest_inner_wh",
    "canvas_scale_mult",
    "occluder_remove_boxes",
    "front_subject_boxes",
    "front_exclude_boxes",
    "back_subject_boxes",
    "back_hair_matte_box",
    "shadow_rgb",
    "midtone_rgb",
    "highlight_rgb",
    "mist_glow_rgb",
    "chroma_retention",
    "shadow_lift",
    "luma_gamma",
    "mist_veil_core",
    "mist_veil_edge",
    "halo_rx_ratio",
    "halo_ry_ratio",
    "halo_feather_power",
    "halo_max_alpha",
    "halo_patch_scale_wh",
    "halo_offset_inner_xy",
    "halo_luminous_lift",
    "halo_bg_blur_sigma_px",
    "tabletop_source_band_ratios",
    "tabletop_blend_band_px",
    "tabletop_alpha_bridge_sigma_px",
    "tabletop_rgb_diffuse_sigma_yx",
    "arch_center_x_ratio",
    "arch_half_span_x_ratio",
    "bottom_arch_curvature",
    "organic_wave_amp_ratio",
    "depth_contour_amp_ratio",
    "depth_contour_smooth_sigma_px",
    "bottom_fade_start_ratio",
    "bottom_fade_end_ratio",
    "bottom_fade_power",
    "top_fade_end_ratio",
    "left_fade_end_ratio",
    "right_fade_start_ratio",
    "right_fade_end_ratio",
    "lower_bloom_strength",
    "lower_bloom_gamma",
    "edge_light_wrap_sigma_px",
    "edge_light_wrap_strength",
    "aperture_glow_center_inner_xy",
    "aperture_glow_radii_inner_xy",
    "aperture_glow_core_rgb",
    "aperture_glow_max_strength",
    "aperture_glow_feather_power",
    "aperture_grain_preserve_sigma_px",
    "aperture_grain_preserve_weight",
    "lama_dilation_px",
    "warm_spill_rgb",
    "warm_rim_wrap_strength",
    "lower_warm_pool_strength",
    "tabletop_preserve_real_rgb_weight",
)


def harmonize_v3_rear_aperture_luminance(
    lama_inner_rgb: Image.Image,
    fusion_cfg: DiffuseApertureFusionSpec,
) -> Image.Image:
    """
    Restores the luminous diffuse gallery-light field across Plan B v3's LaMa-inpainted rear
    aperture while preserving authentic papier-mâché / cement micro-grain, eliminating dark
    inpainted slate shadows beneath Tami & Ya'ara's dissolving lower boundary.
    """
    _require_config_fields(
        fusion_cfg,
        _DIFFUSE_FUSION_REQUIRED_FIELDS,
        "DiffuseApertureFusionSpec",
    )
    arr = np.asarray(lama_inner_rgb.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    cx, cy = fusion_cfg.aperture_glow_center_inner_xy
    rx, ry = fusion_cfg.aperture_glow_radii_inner_xy

    xx = np.arange(w, dtype=np.float32)[None, :]
    yy = np.arange(h, dtype=np.float32)[:, None]
    r_sq = ((xx - cx) / max(4.0, rx)) ** 2 + ((yy - cy) / max(4.0, ry)) ** 2
    glow_w = fusion_cfg.aperture_glow_max_strength * np.power(
        np.clip(1.0 - r_sq, 0.0, 1.0),
        fusion_cfg.aperture_glow_feather_power,
    )[:, :, None]

    low_freq = ndi.gaussian_filter(
        arr, sigma=(fusion_cfg.aperture_grain_preserve_sigma_px, fusion_cfg.aperture_grain_preserve_sigma_px, 0.0)
    )
    grain = arr - low_freq
    core_rgb = np.array(fusion_cfg.aperture_glow_core_rgb, dtype=np.float32)[None, None, :]

    blended = (
        (1.0 - glow_w) * arr
        + glow_w * (core_rgb + fusion_cfg.aperture_grain_preserve_weight * grain)
    )
    return Image.fromarray(np.clip(np.round(blended), 0.0, 255.0).astype(np.uint8), mode="RGB")


def _relight_crop_to_diffuse_bluish_white(
    rgb_img: Image.Image,
    fusion_cfg: DiffuseApertureFusionSpec,
) -> np.ndarray:
    """
    Relights a twilight indoor photograph into the radiant, frosted diffuse bluish-white
    illumination of Plan B's rear chamber while retaining natural facial/page tonal hierarchy.
    """
    rgb = np.asarray(rgb_img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    lum_diff = np.clip(
        fusion_cfg.shadow_lift
        + (1.0 - fusion_cfg.shadow_lift) * np.power(np.clip(lum, 0.0, 1.0), fusion_cfg.luma_gamma),
        0.0,
        1.0,
    )

    sh = np.array(fusion_cfg.shadow_rgb, dtype=np.float32)[None, None, :]
    md = np.array(fusion_cfg.midtone_rgb, dtype=np.float32)[None, None, :]
    hi = np.array(fusion_cfg.highlight_rgb, dtype=np.float32)[None, None, :]
    mist = np.array(fusion_cfg.mist_glow_rgb, dtype=np.float32)[None, None, :]

    t_low = np.clip(lum_diff * 2.0, 0.0, 1.0)[:, :, None]
    t_high = np.clip((lum_diff - 0.5) * 2.0, 0.0, 1.0)[:, :, None]
    pal = np.where(
        lum_diff[:, :, None] <= 0.5,
        (1.0 - t_low) * sh + t_low * md,
        (1.0 - t_high) * md + t_high * hi,
    )

    chroma_delta = rgb - (lum[:, :, None] * 255.0)
    relit = pal + fusion_cfg.chroma_retention * chroma_delta

    uu = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    vv = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    rad = np.clip(np.hypot(uu, vv), 0.0, 1.25) / 1.25
    veil = (
        fusion_cfg.mist_veil_core
        + (fusion_cfg.mist_veil_edge - fusion_cfg.mist_veil_core) * (rad * rad)
    )[:, :, None]
    relit = (1.0 - veil) * relit + veil * mist
    return np.clip(np.round(relit), 0.0, 255.0).astype(np.uint8)


def _bridge_tabletop_and_diffuse_lower_skirt(
    relit_rgb_u8: np.ndarray,
    raw_alpha_u8: np.ndarray,
    fusion_cfg: DiffuseApertureFusionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eliminates SAM's razor-sharp horizontal dining-table cutoff across the lower crop by smoothly
    continuing per-column subject alpha and vertically diffused luminous RGB into the lower skirt.
    """
    crop_h, crop_w = raw_alpha_u8.shape
    r0, r1 = fusion_cfg.tabletop_source_band_ratios
    y0_b = max(0, min(crop_h - 16, int(round(crop_h * r0))))
    y1_b = max(y0_b + 8, min(crop_h - 4, int(round(crop_h * r1))))
    blend_px = max(8, fusion_cfg.tabletop_blend_band_px)
    y_blend_start = max(0, y1_b - blend_px)

    alpha_f = raw_alpha_u8.astype(np.float32)
    rgb_f = relit_rgb_u8.astype(np.float32)
    mist = np.array(fusion_cfg.mist_glow_rgb, dtype=np.float32)[None, None, :]
    warm_spill = np.array(fusion_cfg.warm_spill_rgb, dtype=np.float32)[None, None, :]
    pool_tint = (
        (1.0 - fusion_cfg.lower_warm_pool_strength) * mist
        + fusion_cfg.lower_warm_pool_strength * warm_spill
    )

    col_alpha = ndi.gaussian_filter1d(
        alpha_f[y0_b:y1_b, :].mean(axis=0),
        sigma=fusion_cfg.tabletop_alpha_bridge_sigma_px,
    )

    yy_skirt = np.arange(crop_h - y_blend_start, dtype=np.float32)[:, None]
    t_blend = np.clip(yy_skirt / float(max(1, y1_b - y_blend_start)), 0.0, 1.0)
    w_bridge = t_blend * t_blend * (3.0 - 2.0 * t_blend)

    alpha_bridged = alpha_f.copy()
    skirt_target_a = np.maximum(alpha_f[y_blend_start:, :], col_alpha[None, :])
    alpha_bridged[y_blend_start:, :] = (
        (1.0 - w_bridge) * alpha_f[y_blend_start:, :] + w_bridge * skirt_target_a
    )

    src_band_rgb = rgb_f[y0_b:y1_b, :, :]
    src_band_w = np.maximum(1.0, alpha_f[y0_b:y1_b, :, None])
    weighted_col_rgb = (src_band_rgb * src_band_w).sum(axis=0) / src_band_w.sum(axis=0)
    sig_y, sig_x = fusion_cfg.tabletop_rgb_diffuse_sigma_yx
    smooth_col_rgb = ndi.gaussian_filter1d(weighted_col_rgb, sigma=sig_x, axis=0)

    skirt_h = crop_h - y_blend_start
    t_mist = np.clip(yy_skirt / max(1.0, float(skirt_h - 1)), 0.0, 1.0)[:, :, None]
    t_mist_s = t_mist * (2.0 - t_mist)
    diffused_skirt_rgb = (1.0 - t_mist_s) * smooth_col_rgb[None, :, :] + t_mist_s * pool_tint
    if sig_y > 0.0:
        diffused_skirt_rgb = ndi.gaussian_filter(diffused_skirt_rgb, sigma=(sig_y, 0.0, 0.0))

    preserve_w = np.clip(fusion_cfg.tabletop_preserve_real_rgb_weight, 0.0, 1.0)
    real_skirt_rgb = (1.0 - 0.55 * t_mist_s) * rgb_f[y_blend_start:, :, :] + (0.55 * t_mist_s) * pool_tint
    skirt_rgb = preserve_w * real_skirt_rgb + (1.0 - preserve_w) * diffused_skirt_rgb

    rgb_bridged = rgb_f.copy()
    w_rgb = w_bridge[:, :, None]
    rgb_bridged[y_blend_start:, :, :] = (
        (1.0 - w_rgb) * rgb_f[y_blend_start:, :, :] + w_rgb * skirt_rgb
    )

    return (
        np.clip(np.round(rgb_bridged), 0.0, 255.0).astype(np.uint8),
        np.clip(np.round(alpha_bridged), 0.0, 255.0).astype(np.uint8),
    )


def _compute_arched_aperture_immersion_envelope(
    crop_h: int,
    crop_w: int,
    neural_crop_depth: np.ndarray,
    fusion_cfg: DiffuseApertureFusionSpec,
) -> np.ndarray:
    """
    Computes a curved, pipe-bore-matched elliptical-arch + Depth Anything V2 + multi-harmonic
    cloud dissolution envelope with zero straight horizontal or vertical cutoff lines.
    """
    u = (np.arange(crop_w, dtype=np.float32) / max(1.0, float(crop_w - 1)))[None, :]
    v = (np.arange(crop_h, dtype=np.float32) / max(1.0, float(crop_h - 1)))[:, None]

    da2_smooth = ndi.gaussian_filter(
        neural_crop_depth.astype(np.float32),
        sigma=fusion_cfg.depth_contour_smooth_sigma_px,
    )
    d_min = float(da2_smooth.min())
    d_max = float(da2_smooth.max())
    da2_norm = (da2_smooth - d_min) / max(1e-5, d_max - d_min)

    dx_arch = (u - fusion_cfg.arch_center_x_ratio) / max(0.08, fusion_cfg.arch_half_span_x_ratio)
    arch_lift = fusion_cfg.bottom_arch_curvature * (dx_arch * dx_arch)

    organic_wave = fusion_cfg.organic_wave_amp_ratio * (
        0.65 * np.cos(2.0 * np.pi * u * 1.35 - 0.45)
        + 0.35 * np.sin(2.0 * np.pi * u * 2.35 + 0.85)
    )
    depth_push = fusion_cfg.depth_contour_amp_ratio * (da2_norm - 0.5)

    v_end = np.clip(
        fusion_cfg.bottom_fade_end_ratio - arch_lift + organic_wave + depth_push,
        0.52,
        0.99,
    )
    v_start = np.clip(
        fusion_cfg.bottom_fade_start_ratio - 0.62 * arch_lift + 0.5 * organic_wave + 0.7 * depth_push,
        0.34,
        v_end - 0.16,
    )
    t_bot = np.clip((v - v_start) / np.maximum(0.12, v_end - v_start), 0.0, 1.0)
    env_bot = np.power(0.5 + 0.5 * np.cos(np.pi * t_bot), fusion_cfg.bottom_fade_power)

    vert_curve = (v - 0.48) ** 2
    u_left_end = fusion_cfg.left_fade_end_ratio + 0.10 * vert_curve
    t_left = np.clip(u / np.maximum(0.04, u_left_end), 0.0, 1.0)
    env_left = 0.5 - 0.5 * np.cos(np.pi * t_left)

    u_right_start = fusion_cfg.right_fade_start_ratio - 0.10 * vert_curve
    u_right_end = np.maximum(u_right_start + 0.06, fusion_cfg.right_fade_end_ratio - 0.08 * vert_curve)
    t_right = np.clip((u - u_right_start) / np.maximum(0.04, u_right_end - u_right_start), 0.0, 1.0)
    env_right = 0.5 + 0.5 * np.cos(np.pi * t_right)

    v_top_end = fusion_cfg.top_fade_end_ratio + 0.08 * (dx_arch * dx_arch)
    t_top = np.clip(v / np.maximum(0.03, v_top_end), 0.0, 1.0)
    env_top = 0.5 - 0.5 * np.cos(np.pi * t_top)

    return np.clip(env_bot * env_left * env_right * env_top, 0.0, 1.0).astype(np.float32)


def _apply_aperture_light_wrap(
    rgb_u8: np.ndarray,
    bridged_alpha_u8: np.ndarray,
    perim_env: np.ndarray,
    fusion_cfg: DiffuseApertureFusionSpec,
) -> np.ndarray:
    """
    Infuses backlit silhouette light-wrap and upward lower-aperture gallery bloom into subject RGB
    so Tami, Ya'ara, and the open picture book appear optically immersed in the glowing pipe air.
    """
    rgb_f = rgb_u8.astype(np.float32)
    crop_h = rgb_f.shape[0]
    mist = np.array(fusion_cfg.mist_glow_rgb, dtype=np.float32)[None, None, :]
    warm_spill = np.array(fusion_cfg.warm_spill_rgb, dtype=np.float32)[None, None, :]
    vv_norm = (np.arange(crop_h, dtype=np.float32) / max(1.0, float(crop_h - 1)))[:, None, None]
    lower_ramp = np.clip((vv_norm - 0.38) / 0.62, 0.0, 1.0)
    rim_tint = (
        (1.0 - fusion_cfg.warm_rim_wrap_strength) * mist
        + fusion_cfg.warm_rim_wrap_strength * warm_spill
    )
    pool_tint = (
        (1.0 - fusion_cfg.lower_warm_pool_strength * lower_ramp) * mist
        + (fusion_cfg.lower_warm_pool_strength * lower_ramp) * warm_spill
    )

    outside_matte = 1.0 - (bridged_alpha_u8.astype(np.float32) / 255.0)
    outside_blur = ndi.gaussian_filter(outside_matte, sigma=fusion_cfg.edge_light_wrap_sigma_px)
    edge_wrap = np.clip(outside_blur * fusion_cfg.edge_light_wrap_strength, 0.0, 1.0)[:, :, None]

    dissolve_bloom = (
        np.power(np.clip(1.0 - perim_env, 0.0, 1.0), fusion_cfg.lower_bloom_gamma)
        * fusion_cfg.lower_bloom_strength
    )[:, :, None]
    wrapped = (1.0 - edge_wrap) * rgb_f + edge_wrap * rim_tint
    immersed = (1.0 - np.clip(dissolve_bloom, 0.0, 0.92)) * wrapped + np.clip(dissolve_bloom, 0.0, 0.92) * pool_tint
    return np.clip(np.round(immersed), 0.0, 255.0).astype(np.uint8)


def build_diffuse_aperture_fusion_layers(
    fusion_cfg: DiffuseApertureFusionSpec,
    project_root: Path,
    dims: CanvasDimensionsConfig,
    sam_extractor: SamBoxMaskExtractor,
    matte_extractor: SalientMatteExtractor,
    lama_inpainter: LamaOcclusionInpainter,
    depth_estimator: DepthAnythingEstimator,
) -> dict[str, Image.Image]:
    """
    Synthesizes the three high-resolution RGBA layers for Plan B Version 3 (`subject_front`,
    `subject_back`, `chamber_halo`) with tabletop discontinuity bridging, Depth Anything V2
    arched organic dissolution, and optical gallery light-wrap immersion.
    """
    _require_config_fields(
        fusion_cfg,
        _DIFFUSE_FUSION_REQUIRED_FIELDS,
        "DiffuseApertureFusionSpec",
    )
    src_path = project_root / fusion_cfg.source_image_relpath
    if not src_path.is_file():
        raise FileNotFoundError(f"Diffuse aperture fusion source missing: {src_path}")

    full_img = Image.open(src_path).convert("RGB")
    cx0, cy0, cx1, cy1 = fusion_cfg.src_crop_xyxy
    crop_rgb = full_img.crop((cx0, cy0, cx1, cy1))
    crop_w, crop_h = crop_rgb.size

    if fusion_cfg.occluder_remove_boxes:
        occ_alpha = sam_extractor.segment_box_group(crop_rgb, fusion_cfg.occluder_remove_boxes)
        occ_mask_u8 = np.where(occ_alpha > 8, 255, 0).astype(np.uint8)
        clean_crop_rgb = lama_inpainter.inpaint_background(
            crop_rgb,
            occ_mask_u8,
            dilation_radius_px=fusion_cfg.lama_dilation_px,
        )
    else:
        clean_crop_rgb = crop_rgb

    neural_crop_depth = depth_estimator.predict_depth(clean_crop_rgb)

    front_sam_u8 = sam_extractor.segment_box_group(clean_crop_rgb, fusion_cfg.front_subject_boxes)
    for eb in fusion_cfg.front_exclude_boxes:
        front_sam_u8[eb.y0 : eb.y1, eb.x0 : eb.x1] = 0

    yaara_head_matte = OrientedMatteBoxSpec(
        box=BoundingBoxPrompt(
            x0=int(round(crop_w * 0.20)),
            y0=int(round(crop_h * 0.36)),
            x1=int(round(crop_w * 0.49)),
            y1=int(round(crop_h * 0.72)),
        ),
        upright_angle_deg=0.0,
        taper_margin_px=24,
        sam_core_erode_px=5,
    )
    front_alpha_u8 = merge_sam_and_oriented_mattes_u8(
        rgb_image=clean_crop_rgb,
        sam_u8=front_sam_u8,
        oriented_boxes=(yaara_head_matte,),
        support_dilate_px=45,
        right_edge_fade_x_start=None,
        rembg_session=matte_extractor.session,
    )
    for eb in fusion_cfg.front_exclude_boxes:
        front_alpha_u8[eb.y0 : eb.y1, eb.x0 : eb.x1] = 0

    front_occ_mask_u8 = np.where(front_alpha_u8 > 8, 255, 0).astype(np.uint8)
    tami_disoccluded_rgb = lama_inpainter.inpaint_background(
        clean_crop_rgb,
        front_occ_mask_u8,
        dilation_radius_px=fusion_cfg.lama_dilation_px,
    )

    back_sam_u8 = sam_extractor.segment_box_group(clean_crop_rgb, fusion_cfg.back_subject_boxes)
    tami_hair_matte = OrientedMatteBoxSpec(
        box=fusion_cfg.back_hair_matte_box,
        upright_angle_deg=0.0,
        taper_margin_px=28,
        sam_core_erode_px=6,
    )
    back_alpha_u8 = merge_sam_and_oriented_mattes_u8(
        rgb_image=clean_crop_rgb,
        sam_u8=back_sam_u8,
        oriented_boxes=(tami_hair_matte,),
        support_dilate_px=55,
        right_edge_fade_x_start=None,
        rembg_session=matte_extractor.session,
    )

    kern_left = np.zeros((7, 73), dtype=np.uint8)
    kern_left[:, :37] = 1
    dil_left = cv2.dilate(back_alpha_u8, kern_left)
    dil_left_soft = np.clip(ndi.gaussian_filter(dil_left.astype(np.float32), sigma=4.0), 0.0, 255.0)
    sy0 = int(round(crop_h * 0.48))
    sy1 = int(round(crop_h * 0.86))
    sx0 = int(round(crop_w * 0.38))
    sx1 = int(round(crop_w * 0.56))
    overlap_zone = (front_alpha_u8[sy0:sy1, sx0:sx1] > 32).astype(np.float32)
    overlap_soft = ndi.gaussian_filter(overlap_zone, sigma=5.0)
    back_roi = back_alpha_u8[sy0:sy1, sx0:sx1].astype(np.float32)
    back_alpha_u8[sy0:sy1, sx0:sx1] = np.clip(
        np.round(
            np.maximum(
                back_roi,
                dil_left_soft[sy0:sy1, sx0:sx1] * np.clip(overlap_soft * 1.4, 0.0, 1.0),
            )
        ),
        0,
        255,
    ).astype(np.uint8)
    n_cc, cc_lbls, cc_stats, _ = cv2.connectedComponentsWithStats(
        (back_alpha_u8 > 8).astype(np.uint8), connectivity=8
    )
    if n_cc > 1:
        keep_back = np.zeros_like(back_alpha_u8, dtype=bool)
        for cid in range(1, n_cc):
            if int(cc_stats[cid, cv2.CC_STAT_AREA]) >= 800:
                keep_back |= cc_lbls == cid
        back_alpha_u8 = np.where(keep_back, back_alpha_u8, 0).astype(np.uint8)

    both_mask_u8 = np.where((front_alpha_u8 > 8) | (back_alpha_u8 > 8), 255, 0).astype(np.uint8)
    halo_bg_rgb = lama_inpainter.inpaint_background(
        clean_crop_rgb,
        both_mask_u8,
        dilation_radius_px=fusion_cfg.lama_dilation_px,
    )

    front_decontam_u8 = decontaminate_matte_edge_rgb(
        _relight_crop_to_diffuse_bluish_white(clean_crop_rgb, fusion_cfg),
        front_alpha_u8,
    )
    back_decontam_u8 = decontaminate_matte_edge_rgb(
        _relight_crop_to_diffuse_bluish_white(tami_disoccluded_rgb, fusion_cfg),
        back_alpha_u8,
    )

    front_bridged_rgb, front_bridged_a = _bridge_tabletop_and_diffuse_lower_skirt(
        front_decontam_u8, front_alpha_u8, fusion_cfg
    )
    back_bridged_rgb, back_bridged_a = _bridge_tabletop_and_diffuse_lower_skirt(
        back_decontam_u8, back_alpha_u8, fusion_cfg
    )

    perim_env = _compute_arched_aperture_immersion_envelope(
        crop_h=crop_h,
        crop_w=crop_w,
        neural_crop_depth=neural_crop_depth,
        fusion_cfg=fusion_cfg,
    )

    front_immersed_u8 = _apply_aperture_light_wrap(
        front_bridged_rgb, front_bridged_a, perim_env, fusion_cfg
    )
    back_immersed_u8 = _apply_aperture_light_wrap(
        back_bridged_rgb, back_bridged_a, perim_env, fusion_cfg
    )

    front_final_a = np.clip(
        np.round(front_bridged_a.astype(np.float32) * perim_env), 0, 255
    ).astype(np.uint8)
    back_final_a = np.clip(
        np.round(back_bridged_a.astype(np.float32) * perim_env), 0, 255
    ).astype(np.uint8)

    halo_base_f = _relight_crop_to_diffuse_bluish_white(halo_bg_rgb, fusion_cfg).astype(np.float32)
    if fusion_cfg.halo_bg_blur_sigma_px > 0.0:
        halo_base_f = ndi.gaussian_filter(
            halo_base_f,
            sigma=(fusion_cfg.halo_bg_blur_sigma_px, fusion_cfg.halo_bg_blur_sigma_px, 0.0),
        )
    mist_arr = np.array(fusion_cfg.mist_glow_rgb, dtype=np.float32)[None, None, :]
    warm_arr = np.array(fusion_cfg.warm_spill_rgb, dtype=np.float32)[None, None, :]
    vv_halo = (np.arange(crop_h, dtype=np.float32) / max(1.0, float(crop_h - 1)))[:, None, None]
    halo_warm_w = np.clip((vv_halo - 0.32) / 0.68, 0.0, 1.0) * (fusion_cfg.lower_warm_pool_strength * 0.55)
    halo_glow_target = (1.0 - halo_warm_w) * mist_arr + halo_warm_w * warm_arr
    halo_relit_u8 = np.clip(
        np.round(
            (1.0 - fusion_cfg.halo_luminous_lift) * halo_base_f
            + fusion_cfg.halo_luminous_lift * halo_glow_target
        ),
        0.0,
        255.0,
    ).astype(np.uint8)

    uu = (np.arange(crop_w, dtype=np.float32) / max(1.0, float(crop_w - 1)) - 0.50)[None, :]
    vv = (np.arange(crop_h, dtype=np.float32) / max(1.0, float(crop_h - 1)) - 0.50)[:, None]
    r_sq = (uu / max(0.05, fusion_cfg.halo_rx_ratio)) ** 2 + (
        vv / max(0.05, fusion_cfg.halo_ry_ratio)
    ) ** 2
    halo_env = fusion_cfg.halo_max_alpha * np.power(
        np.clip(1.0 - r_sq, 0.0, 1.0), fusion_cfg.halo_feather_power
    )
    halo_final_a = np.clip(np.round(halo_env * 255.0), 0, 255).astype(np.uint8)

    mult = fusion_cfg.canvas_scale_mult
    canvas_w = max(dims.width, int(round(dims.width * mult)))
    canvas_h = max(dims.height, int(round(dims.height * mult)))
    patch_w = max(32, int(round(fusion_cfg.dest_inner_wh[0] * mult)))
    patch_h = max(32, int(round(fusion_cfg.dest_inner_wh[1] * mult)))
    dest_x = int(round(fusion_cfg.dest_center_inner_xy[0] * mult - patch_w * 0.5))
    dest_y = int(round(fusion_cfg.dest_center_inner_xy[1] * mult - patch_h * 0.5))

    halo_scale_w, halo_scale_h = fusion_cfg.halo_patch_scale_wh
    halo_off_x, halo_off_y = fusion_cfg.halo_offset_inner_xy
    halo_w = max(32, int(round(fusion_cfg.dest_inner_wh[0] * halo_scale_w * mult)))
    halo_h = max(32, int(round(fusion_cfg.dest_inner_wh[1] * halo_scale_h * mult)))
    halo_x = int(round((fusion_cfg.dest_center_inner_xy[0] + halo_off_x) * mult - halo_w * 0.5))
    halo_y = int(round((fusion_cfg.dest_center_inner_xy[1] + halo_off_y) * mult - halo_h * 0.5))

    def _place_on_canvas(
        rgb_arr: np.ndarray,
        alpha_arr: np.ndarray,
        pw: int,
        ph: int,
        dx: int,
        dy: int,
    ) -> Image.Image:
        patch = Image.fromarray(np.dstack([rgb_arr, alpha_arr]), mode="RGBA").resize(
            (pw, ph), Image.Resampling.LANCZOS
        )
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        canvas.alpha_composite(patch, dest=(dx, dy))
        return canvas

    return {
        "subject_front": _place_on_canvas(
            front_immersed_u8, front_final_a, patch_w, patch_h, dest_x, dest_y
        ),
        "subject_back": _place_on_canvas(
            back_immersed_u8, back_final_a, patch_w, patch_h, dest_x, dest_y
        ),
        "chamber_halo": _place_on_canvas(
            halo_relit_u8, halo_final_a, halo_w, halo_h, halo_x, halo_y
        ),
    }

