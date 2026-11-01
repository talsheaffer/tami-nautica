#!/usr/bin/env python3
r"""
prototype/depth_topology.py — Canonical 2.5D Depth Topology Primitives & Solvers

Implements the mathematical invariants defined in
`.agents/skills/coding-guidelines/references/depth_topology_principles.md`:
1. Law 1 ($C^0$ Contact Continuity): Harmonic boundary-value anchoring so any layer
   touching a supporting room surface matches $Z_{\text{world}}^{\text{host}}$
   ($\Delta Z_{\text{world}} = 0$) along its physical contact locus while carrying
   genuine neural 3D sculptural relief above the anchor.
2. Law 2 ($C^{-1}$ Free-Silhouette Discontinuity): Outward Euclidean Distance Transform
   (EDT) depth extrapolation across all transparent cutout pixels (`pinMarginUv = 0.0`)
   so straddling mesh triangles never rubber-sheet toward zero/wall depth.
3. Law 3 (Ghost-Free Occluded Background Depth): Sparse discrete Laplace healing of
   `layer0` room depth maps beneath dilated foreground/midground cutout masks so no
   foreground ghost depth mounds bulge out of empty floors or walls.
4. Law 4 (Depth Monotonicity) & Programmatic Verification Gate: Quantitative checks for
   contact $|\Delta Z_{\text{world}}|$, exterior rim EDT continuity, and host clearance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import numpy.typing as npt
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

ContactAnchorMode = Literal[
    "bottom_band",
    "top_band",
    "left_band",
    "right_band",
    "perimeter_rim",
    "all_opaque",
]

VALID_CONTACT_ANCHOR_MODES: frozenset[str] = frozenset(
    {
        "bottom_band",
        "top_band",
        "left_band",
        "right_band",
        "perimeter_rim",
        "all_opaque",
    }
)

UINT8_MAX: float = 255.0


@dataclass(frozen=True)
class ContactAnchorBox:
    """Explicit pixel-space bounding region and locus rule for surface contact anchoring."""

    x0: int
    y0: int
    x1: int
    y1: int
    mode: ContactAnchorMode = "bottom_band"
    band_px: int = 6

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(
                f"ContactAnchorBox requires x0 < x1 and y0 < y1, got "
                f"({self.x0}, {self.y0}, {self.x1}, {self.y1})."
            )
        if self.mode not in VALID_CONTACT_ANCHOR_MODES:
            raise ValueError(
                f"Unsupported ContactAnchorBox.mode {self.mode!r}; "
                f"expected one of {sorted(VALID_CONTACT_ANCHOR_MODES)}."
            )
        if self.band_px < 1:
            raise ValueError(f"ContactAnchorBox.band_px must be >= 1, got {self.band_px}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "x0": int(self.x0),
            "y0": int(self.y0),
            "x1": int(self.x1),
            "y1": int(self.y1),
            "mode": self.mode,
            "band_px": int(self.band_px),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContactAnchorBox:
        return cls(
            x0=int(data["x0"]),
            y0=int(data["y0"]),
            x1=int(data["x1"]),
            y1=int(data["y1"]),
            mode=data["mode"],
            band_px=int(data["band_px"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> ContactAnchorBox:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class RoomDepthCleanConfig:
    """SSOT configuration for Laplace healing of foreground ghost mounds in `layer0` depth."""

    alpha_threshold_u8: int = 16
    mask_dilation_px: int = 18
    extra_clean_boxes_expanded: tuple[tuple[int, int, int, int], ...] = ()
    preserve_boxes_expanded: tuple[tuple[int, int, int, int], ...] = ()
    post_smooth_sigma_px: float = 2.5
    interior_blend_band_px: float = 10.0

    def __post_init__(self) -> None:
        if not (1 <= self.alpha_threshold_u8 <= 255):
            raise ValueError(
                f"alpha_threshold_u8 must be in [1, 255], got {self.alpha_threshold_u8}."
            )
        if self.mask_dilation_px < 0:
            raise ValueError(f"mask_dilation_px must be >= 0, got {self.mask_dilation_px}.")
        if self.post_smooth_sigma_px < 0.0:
            raise ValueError(
                f"post_smooth_sigma_px must be >= 0.0, got {self.post_smooth_sigma_px}."
            )
        if self.interior_blend_band_px <= 0.0:
            raise ValueError(
                f"interior_blend_band_px must be > 0.0, got {self.interior_blend_band_px}."
            )
        for box_name, boxes in (
            ("extra_clean_boxes_expanded", self.extra_clean_boxes_expanded),
            ("preserve_boxes_expanded", self.preserve_boxes_expanded),
        ):
            for box in boxes:
                if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"Invalid box in {box_name}: {box!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_threshold_u8": int(self.alpha_threshold_u8),
            "mask_dilation_px": int(self.mask_dilation_px),
            "extra_clean_boxes_expanded": [list(b) for b in self.extra_clean_boxes_expanded],
            "preserve_boxes_expanded": [list(b) for b in self.preserve_boxes_expanded],
            "post_smooth_sigma_px": float(self.post_smooth_sigma_px),
            "interior_blend_band_px": float(self.interior_blend_band_px),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoomDepthCleanConfig:
        return cls(
            alpha_threshold_u8=int(data["alpha_threshold_u8"]),
            mask_dilation_px=int(data["mask_dilation_px"]),
            extra_clean_boxes_expanded=tuple(
                (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                for b in data["extra_clean_boxes_expanded"]
            ),
            preserve_boxes_expanded=tuple(
                (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                for b in data["preserve_boxes_expanded"]
            ),
            post_smooth_sigma_px=float(data["post_smooth_sigma_px"]),
            interior_blend_band_px=float(data["interior_blend_band_px"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> RoomDepthCleanConfig:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class ContactAnchoredLayerDepthConfig:
    """SSOT configuration for solving contact-anchored, EDT-extrapolated cutout layer depth."""

    bg_z: float
    room_depth_amplitude: float
    room_pin_margin_uv: float
    layer_z: float
    layer_depth_amplitude: float
    layer_pin_margin_uv: float = 0.0
    alpha_threshold_u8: int = 20
    component_bridge_px: int = 6
    min_component_area_px: int = 24
    auto_floor_band_px: int = 6
    auto_floor_max_rise_px: int = 48
    contact_boxes: tuple[ContactAnchorBox, ...] = ()
    neural_relief_amplitude: float = 24.0
    contact_relief_taper_px: float = 8.0
    enforce_front_of_host: bool = True
    host_clearance_tolerance_world: float = 0.0
    interior_smooth_sigma_px: float = 1.5
    overflow_tolerance_world: float = 0.5

    def __post_init__(self) -> None:
        if self.layer_depth_amplitude <= 0.0:
            raise ValueError(
                f"layer_depth_amplitude must be > 0.0, got {self.layer_depth_amplitude}."
            )
        if self.room_pin_margin_uv < 0.0 or self.room_pin_margin_uv >= 0.5:
            raise ValueError(
                f"room_pin_margin_uv must be in [0.0, 0.5), got {self.room_pin_margin_uv}."
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
        if self.component_bridge_px < 0:
            raise ValueError(
                f"component_bridge_px must be >= 0, got {self.component_bridge_px}."
            )
        if self.min_component_area_px < 1:
            raise ValueError(
                f"min_component_area_px must be >= 1, got {self.min_component_area_px}."
            )
        if self.auto_floor_band_px < 1:
            raise ValueError(
                f"auto_floor_band_px must be >= 1, got {self.auto_floor_band_px}."
            )
        if self.auto_floor_max_rise_px < 0:
            raise ValueError(
                f"auto_floor_max_rise_px must be >= 0, got {self.auto_floor_max_rise_px}."
            )
        if self.neural_relief_amplitude < 0.0:
            raise ValueError(
                f"neural_relief_amplitude must be >= 0.0, got {self.neural_relief_amplitude}."
            )
        if self.contact_relief_taper_px <= 0.0:
            raise ValueError(
                f"contact_relief_taper_px must be > 0.0, got {self.contact_relief_taper_px}."
            )
        if self.interior_smooth_sigma_px < 0.0:
            raise ValueError(
                f"interior_smooth_sigma_px must be >= 0.0, got {self.interior_smooth_sigma_px}."
            )
        if self.overflow_tolerance_world < 0.0:
            raise ValueError(
                f"overflow_tolerance_world must be >= 0.0, got {self.overflow_tolerance_world}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bg_z": float(self.bg_z),
            "room_depth_amplitude": float(self.room_depth_amplitude),
            "room_pin_margin_uv": float(self.room_pin_margin_uv),
            "layer_z": float(self.layer_z),
            "layer_depth_amplitude": float(self.layer_depth_amplitude),
            "layer_pin_margin_uv": float(self.layer_pin_margin_uv),
            "alpha_threshold_u8": int(self.alpha_threshold_u8),
            "component_bridge_px": int(self.component_bridge_px),
            "min_component_area_px": int(self.min_component_area_px),
            "auto_floor_band_px": int(self.auto_floor_band_px),
            "auto_floor_max_rise_px": int(self.auto_floor_max_rise_px),
            "contact_boxes": [b.to_dict() for b in self.contact_boxes],
            "neural_relief_amplitude": float(self.neural_relief_amplitude),
            "contact_relief_taper_px": float(self.contact_relief_taper_px),
            "enforce_front_of_host": bool(self.enforce_front_of_host),
            "host_clearance_tolerance_world": float(self.host_clearance_tolerance_world),
            "interior_smooth_sigma_px": float(self.interior_smooth_sigma_px),
            "overflow_tolerance_world": float(self.overflow_tolerance_world),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContactAnchoredLayerDepthConfig:
        return cls(
            bg_z=float(data["bg_z"]),
            room_depth_amplitude=float(data["room_depth_amplitude"]),
            room_pin_margin_uv=float(data["room_pin_margin_uv"]),
            layer_z=float(data["layer_z"]),
            layer_depth_amplitude=float(data["layer_depth_amplitude"]),
            layer_pin_margin_uv=float(data["layer_pin_margin_uv"]),
            alpha_threshold_u8=int(data["alpha_threshold_u8"]),
            component_bridge_px=int(data["component_bridge_px"]),
            min_component_area_px=int(data["min_component_area_px"]),
            auto_floor_band_px=int(data["auto_floor_band_px"]),
            auto_floor_max_rise_px=int(data["auto_floor_max_rise_px"]),
            contact_boxes=tuple(ContactAnchorBox.from_dict(b) for b in data["contact_boxes"]),
            neural_relief_amplitude=float(data["neural_relief_amplitude"]),
            contact_relief_taper_px=float(data["contact_relief_taper_px"]),
            enforce_front_of_host=bool(data["enforce_front_of_host"]),
            host_clearance_tolerance_world=float(data["host_clearance_tolerance_world"]),
            interior_smooth_sigma_px=float(data["interior_smooth_sigma_px"]),
            overflow_tolerance_world=float(data["overflow_tolerance_world"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> ContactAnchoredLayerDepthConfig:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class LayerTopologyVerificationConfig:
    """Thresholds for programmatic validation of contact continuity and EDT rim discontinuity."""

    alpha_threshold_u8: int = 20
    rim_erosion_px: int = 2
    rim_exterior_probe_px: int = 8
    max_allowed_contact_dz_world: float = 2.0
    max_allowed_p95_contact_dz_world: float = 1.2
    max_allowed_rim_jump_world: float = 2.5
    host_monotonicity_tolerance_world: float = 2.0

    def __post_init__(self) -> None:
        if not (1 <= self.alpha_threshold_u8 <= 255):
            raise ValueError(
                f"alpha_threshold_u8 must be in [1, 255], got {self.alpha_threshold_u8}."
            )
        if self.rim_erosion_px < 1:
            raise ValueError(f"rim_erosion_px must be >= 1, got {self.rim_erosion_px}.")
        if self.rim_exterior_probe_px < 1:
            raise ValueError(
                f"rim_exterior_probe_px must be >= 1, got {self.rim_exterior_probe_px}."
            )
        if self.max_allowed_contact_dz_world <= 0.0:
            raise ValueError("max_allowed_contact_dz_world must be > 0.0.")
        if self.max_allowed_p95_contact_dz_world <= 0.0:
            raise ValueError("max_allowed_p95_contact_dz_world must be > 0.0.")
        if self.max_allowed_rim_jump_world <= 0.0:
            raise ValueError("max_allowed_rim_jump_world must be > 0.0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_threshold_u8": int(self.alpha_threshold_u8),
            "rim_erosion_px": int(self.rim_erosion_px),
            "rim_exterior_probe_px": int(self.rim_exterior_probe_px),
            "max_allowed_contact_dz_world": float(self.max_allowed_contact_dz_world),
            "max_allowed_p95_contact_dz_world": float(self.max_allowed_p95_contact_dz_world),
            "max_allowed_rim_jump_world": float(self.max_allowed_rim_jump_world),
            "host_monotonicity_tolerance_world": float(self.host_monotonicity_tolerance_world),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerTopologyVerificationConfig:
        return cls(
            alpha_threshold_u8=int(data["alpha_threshold_u8"]),
            rim_erosion_px=int(data["rim_erosion_px"]),
            rim_exterior_probe_px=int(data["rim_exterior_probe_px"]),
            max_allowed_contact_dz_world=float(data["max_allowed_contact_dz_world"]),
            max_allowed_p95_contact_dz_world=float(data["max_allowed_p95_contact_dz_world"]),
            max_allowed_rim_jump_world=float(data["max_allowed_rim_jump_world"]),
            host_monotonicity_tolerance_world=float(data["host_monotonicity_tolerance_world"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> LayerTopologyVerificationConfig:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class LayerTopologyVerificationReport:
    """Quantitative audit report emitted by `verify_layer_contact_and_discontinuity`."""

    contact_pixel_count: int
    opaque_pixel_count: int
    rim_probe_pixel_count: int
    max_abs_contact_dz_world: float
    p95_abs_contact_dz_world: float
    mean_abs_contact_dz_world: float
    max_abs_rim_jump_world: float
    p95_abs_rim_jump_world: float
    min_signed_dz_vs_host_world: float
    contact_passed: bool
    discontinuity_passed: bool
    monotonicity_passed: bool
    all_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_pixel_count": int(self.contact_pixel_count),
            "opaque_pixel_count": int(self.opaque_pixel_count),
            "rim_probe_pixel_count": int(self.rim_probe_pixel_count),
            "max_abs_contact_dz_world": float(self.max_abs_contact_dz_world),
            "p95_abs_contact_dz_world": float(self.p95_abs_contact_dz_world),
            "mean_abs_contact_dz_world": float(self.mean_abs_contact_dz_world),
            "max_abs_rim_jump_world": float(self.max_abs_rim_jump_world),
            "p95_abs_rim_jump_world": float(self.p95_abs_rim_jump_world),
            "min_signed_dz_vs_host_world": float(self.min_signed_dz_vs_host_world),
            "contact_passed": bool(self.contact_passed),
            "discontinuity_passed": bool(self.discontinuity_passed),
            "monotonicity_passed": bool(self.monotonicity_passed),
            "all_passed": bool(self.all_passed),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerTopologyVerificationReport:
        return cls(
            contact_pixel_count=int(data["contact_pixel_count"]),
            opaque_pixel_count=int(data["opaque_pixel_count"]),
            rim_probe_pixel_count=int(data["rim_probe_pixel_count"]),
            max_abs_contact_dz_world=float(data["max_abs_contact_dz_world"]),
            p95_abs_contact_dz_world=float(data["p95_abs_contact_dz_world"]),
            mean_abs_contact_dz_world=float(data["mean_abs_contact_dz_world"]),
            max_abs_rim_jump_world=float(data["max_abs_rim_jump_world"]),
            p95_abs_rim_jump_world=float(data["p95_abs_rim_jump_world"]),
            min_signed_dz_vs_host_world=float(data["min_signed_dz_vs_host_world"]),
            contact_passed=bool(data["contact_passed"]),
            discontinuity_passed=bool(data["discontinuity_passed"]),
            monotonicity_passed=bool(data["monotonicity_passed"]),
            all_passed=bool(data["all_passed"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> LayerTopologyVerificationReport:
        return cls.from_dict(json.loads(raw))


def glsl_smoothstep(
    edge0: float,
    edge1: float,
    values: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float32]:
    """Exact NumPy equivalent of GLSL `smoothstep(edge0, edge1, x)`."""
    if edge1 <= edge0:
        return (values >= edge0).astype(np.float32)
    t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0).astype(np.float32)
    return t * t * (3.0 - 2.0 * t)


def compute_perimeter_pin_weights(
    height: int,
    width: int,
    pin_margin_uv: float,
) -> npt.NDArray[np.float32]:
    """Computes `pinX * pinY` matching `ROOM_DEPTH_VERTEX_SHADER` in `scene_manager.js`."""
    if height < 2 or width < 2:
        raise ValueError(f"Expected height >= 2 and width >= 2, got ({height}, {width}).")
    if pin_margin_uv <= 0.0:
        return np.ones((height, width), dtype=np.float32)
    u = np.linspace(0.0, 1.0, width, dtype=np.float32)
    v = np.linspace(1.0, 0.0, height, dtype=np.float32)
    pin_x = glsl_smoothstep(0.0, pin_margin_uv, u) * glsl_smoothstep(
        0.0, pin_margin_uv, 1.0 - u
    )
    pin_y = glsl_smoothstep(0.0, pin_margin_uv, v) * glsl_smoothstep(
        0.0, pin_margin_uv, 1.0 - v
    )
    return np.outer(pin_y, pin_x).astype(np.float32)


def normalize_depth_array(depth_map: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Converts a uint8 `[0, 255]` or float `[0.0, 1.0]` 2D depth map to float32 `[0.0, 1.0]`."""
    if depth_map.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth_map.shape}.")
    if np.issubdtype(depth_map.dtype, np.integer):
        return (depth_map.astype(np.float32) / UINT8_MAX).clip(0.0, 1.0)
    arr = depth_map.astype(np.float32)
    if float(arr.max()) > 1.001 or float(arr.min()) < -0.001:
        raise ValueError(
            f"Floating-point depth map must lie in [0.0, 1.0], "
            f"got [{float(arr.min()):.4f}, {float(arr.max()):.4f}]."
        )
    return np.clip(arr, 0.0, 1.0)


def compute_world_z_map(
    depth_map: npt.NDArray[Any],
    plane_z: float,
    depth_amplitude: float,
    pin_margin_uv: float = 0.0,
) -> npt.NDArray[np.float32]:
    """Evaluates unified Three.js world-space Z: `plane_z + depth_amplitude * d * pin(u, v)`."""
    d_norm = normalize_depth_array(depth_map)
    h, w = d_norm.shape
    pin = compute_perimeter_pin_weights(h, w, pin_margin_uv)
    return (float(plane_z) + float(depth_amplitude) * d_norm * pin).astype(np.float32)


def align_mask_to_expanded_canvas(
    mask_or_alpha: npt.NDArray[Any],
    expanded_shape: tuple[int, int],
    alpha_threshold_u8: int,
) -> npt.NDArray[np.bool_]:
    """Embeds an inner `(1500x1000)` or expanded `(2700x1800)` mask/alpha into `expanded_shape`."""
    exp_h, exp_w = expanded_shape
    if mask_or_alpha.ndim == 3 and mask_or_alpha.shape[2] == 4:
        alpha = mask_or_alpha[:, :, 3]
        binary = alpha >= alpha_threshold_u8
    elif mask_or_alpha.ndim == 2:
        if mask_or_alpha.dtype == np.bool_:
            binary = mask_or_alpha
        else:
            binary = mask_or_alpha >= alpha_threshold_u8
    else:
        raise ValueError(
            f"Expected 2D mask/alpha or HxWx4 RGBA array, got shape {mask_or_alpha.shape}."
        )

    h, w = binary.shape
    if (h, w) == (exp_h, exp_w):
        return binary.copy()
    if h > exp_h or w > exp_w or (exp_h - h) % 2 != 0 or (exp_w - w) % 2 != 0:
        raise ValueError(
            f"Cannot center-embed layer shape {(h, w)} into expanded shape {(exp_h, exp_w)}."
        )
    pad_y = (exp_h - h) // 2
    pad_x = (exp_w - w) // 2
    out = np.zeros((exp_h, exp_w), dtype=np.bool_)
    out[pad_y : pad_y + h, pad_x : pad_x + w] = binary
    return out


def extract_layer_window_from_expanded(
    expanded_field: npt.NDArray[np.float32],
    layer_shape: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Extracts the centered inner window (or full canvas if expanded) matching `layer_shape`."""
    exp_h, exp_w = expanded_field.shape
    lh, lw = layer_shape
    if (lh, lw) == (exp_h, exp_w):
        return expanded_field.copy()
    if lh > exp_h or lw > exp_w or (exp_h - lh) % 2 != 0 or (exp_w - lw) % 2 != 0:
        raise ValueError(
            f"Cannot extract centered window {(lh, lw)} from expanded shape {(exp_h, exp_w)}."
        )
    pad_y = (exp_h - lh) // 2
    pad_x = (exp_w - lw) // 2
    return expanded_field[pad_y : pad_y + lh, pad_x : pad_x + lw].copy()


def _solve_harmonic_field_on_domain(
    domain_mask: npt.NDArray[np.bool_],
    dirichlet_mask: npt.NDArray[np.bool_],
    dirichlet_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Solves the 4-neighbor discrete Laplace equation $\\nabla^2 u = 0$ on `domain_mask \\ dirichlet_mask`
    with exact Dirichlet values on `dirichlet_mask` and homogeneous Neumann ($\\partial_n u = 0$)
    across any boundary neighbor outside `domain_mask | dirichlet_mask`.
    """
    unknown_mask = domain_mask & (~dirichlet_mask)
    result = np.zeros(domain_mask.shape, dtype=np.float64)
    result[dirichlet_mask] = dirichlet_values[dirichlet_mask]

    n_unknown = int(unknown_mask.sum())
    if n_unknown == 0:
        return result

    # Verify every connected component of `unknown_mask` touches at least one `dirichlet_mask` pixel
    active_union = unknown_mask | dirichlet_mask
    labeled_union, num_union = ndimage.label(active_union)
    for comp_id in range(1, num_union + 1):
        comp = labeled_union == comp_id
        if np.any(comp & unknown_mask) and not np.any(comp & dirichlet_mask):
            ys, xs = np.where(comp & unknown_mask)
            raise ValueError(
                f"Harmonic PDE component (area={len(xs)} px, bbox="
                f"[{int(xs.min())},{int(ys.min())},{int(xs.max())},{int(ys.max())}]) "
                f"has zero Dirichlet anchor pixels."
            )

    h, w = domain_mask.shape
    idx_grid = np.full((h, w), -1, dtype=np.int32)
    uy, ux = np.where(unknown_mask)
    idx_grid[uy, ux] = np.arange(n_unknown, dtype=np.int32)

    diag = np.zeros(n_unknown, dtype=np.float64)
    rhs = np.zeros(n_unknown, dtype=np.float64)
    rows_list: list[npt.NDArray[np.int32]] = []
    cols_list: list[npt.NDArray[np.int32]] = []
    vals_list: list[npt.NDArray[np.float64]] = []

    p_indices = np.arange(n_unknown, dtype=np.int32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny = uy + dy
        nx = ux + dx
        in_bounds = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        valid_p = p_indices[in_bounds]
        v_ny = ny[in_bounds]
        v_nx = nx[in_bounds]

        is_unknown_nb = unknown_mask[v_ny, v_nx]
        is_dirichlet_nb = dirichlet_mask[v_ny, v_nx]

        if np.any(is_unknown_nb):
            src_p = valid_p[is_unknown_nb]
            dst_q = idx_grid[v_ny[is_unknown_nb], v_nx[is_unknown_nb]]
            np.add.at(diag, src_p, 1.0)
            rows_list.append(src_p)
            cols_list.append(dst_q)
            vals_list.append(np.full(len(src_p), -1.0, dtype=np.float64))

        if np.any(is_dirichlet_nb):
            dir_p = valid_p[is_dirichlet_nb]
            dir_vals = dirichlet_values[v_ny[is_dirichlet_nb], v_nx[is_dirichlet_nb]]
            np.add.at(diag, dir_p, 1.0)
            np.add.at(rhs, dir_p, dir_vals)

    rows_list.append(p_indices)
    cols_list.append(p_indices)
    vals_list.append(diag)

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_vals = np.concatenate(vals_list)

    laplacian = sparse.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_unknown, n_unknown))
    solved = spsolve(laplacian, rhs)
    if not np.all(np.isfinite(solved)):
        raise RuntimeError("Harmonic Laplace solver produced non-finite values.")

    result[uy, ux] = solved
    return result


def extrapolate_depth_outward_edt(
    interior_depth: npt.NDArray[np.float32],
    opaque_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float32]:
    """
    Law 2 ($C^{-1}$ Free-Silhouette Discontinuity): Extrapolates `interior_depth` outward
    from `opaque_mask` across all transparent pixels via Euclidean Distance Transform.
    """
    if interior_depth.shape != opaque_mask.shape:
        raise ValueError(
            f"Shape mismatch: interior_depth {interior_depth.shape} vs opaque_mask {opaque_mask.shape}."
        )
    if not np.any(opaque_mask):
        raise ValueError("Cannot extrapolate depth from an empty opaque_mask.")
    if np.all(opaque_mask):
        return interior_depth.astype(np.float32, copy=True)

    _, nearest_indices = ndimage.distance_transform_edt(~opaque_mask, return_indices=True)
    extrapolated = interior_depth[nearest_indices[0], nearest_indices[1]].astype(np.float32)
    extrapolated[opaque_mask] = interior_depth[opaque_mask]
    return extrapolated


def clean_room_depth_under_cutouts(
    room_depth: npt.NDArray[Any],
    cutout_layers: Sequence[npt.NDArray[Any]],
    cfg: RoomDepthCleanConfig,
) -> npt.NDArray[np.uint8]:
    """
    Law 3: Removes foreground ghost depth mounds from a `layer0` room depth map beneath
    dilated foreground/midground cutouts via sparse harmonic interpolation anchored to
    the surrounding unoccluded room surface.
    """
    d_norm = normalize_depth_array(room_depth).astype(np.float64)
    exp_h, exp_w = d_norm.shape

    heal_mask = np.zeros((exp_h, exp_w), dtype=np.bool_)
    for layer in cutout_layers:
        embedded = align_mask_to_expanded_canvas(
            layer,
            expanded_shape=(exp_h, exp_w),
            alpha_threshold_u8=cfg.alpha_threshold_u8,
        )
        if cfg.mask_dilation_px > 0 and np.any(embedded):
            dist_out = ndimage.distance_transform_edt(~embedded)
            embedded = dist_out <= float(cfg.mask_dilation_px)
        heal_mask |= embedded

    for x0, y0, x1, y1 in cfg.extra_clean_boxes_expanded:
        cx0 = max(0, min(exp_w, x0))
        cy0 = max(0, min(exp_h, y0))
        cx1 = max(0, min(exp_w, x1))
        cy1 = max(0, min(exp_h, y1))
        heal_mask[cy0:cy1, cx0:cx1] = True

    for x0, y0, x1, y1 in cfg.preserve_boxes_expanded:
        px0 = max(0, min(exp_w, x0))
        py0 = max(0, min(exp_h, y0))
        px1 = max(0, min(exp_w, x1))
        py1 = max(0, min(exp_h, y1))
        heal_mask[py0:py1, px0:px1] = False

    if not np.any(heal_mask):
        return np.rint(np.clip(d_norm, 0.0, 1.0) * UINT8_MAX).astype(np.uint8)

    known_mask = ~heal_mask
    if not np.any(known_mask):
        raise ValueError("clean_room_depth_under_cutouts: heal_mask covers 100% of the canvas.")

    full_domain = np.ones((exp_h, exp_w), dtype=np.bool_)
    healed = _solve_harmonic_field_on_domain(
        domain_mask=full_domain,
        dirichlet_mask=known_mask,
        dirichlet_values=d_norm,
    )

    if cfg.post_smooth_sigma_px > 0.0:
        smoothed = ndimage.gaussian_filter(healed, sigma=cfg.post_smooth_sigma_px, mode="nearest")
        interior_dist = ndimage.distance_transform_edt(heal_mask)
        blend = np.clip(interior_dist / cfg.interior_blend_band_px, 0.0, 1.0)
        healed = (1.0 - blend) * healed + blend * smoothed

    return np.rint(np.clip(healed, 0.0, 1.0) * UINT8_MAX).astype(np.uint8)


def extract_contact_anchor_mask(
    opaque_mask: npt.NDArray[np.bool_],
    cfg: ContactAnchoredLayerDepthConfig,
) -> npt.NDArray[np.bool_]:
    """
    Determines the physical contact pixel locus $C \\subset \\Omega$ for an RGBA cutout mask.
    Uses explicit `cfg.contact_boxes` when provided; otherwise detects floor footprints
    along the lowest opaque column bands of each bridged physical component.
    """
    h, w = opaque_mask.shape
    contact_mask = np.zeros((h, w), dtype=np.bool_)
    if not np.any(opaque_mask):
        raise ValueError("Cannot extract contact anchors from an empty cutout mask.")

    if cfg.contact_boxes:
        for box in cfg.contact_boxes:
            bx0 = max(0, min(w, box.x0))
            by0 = max(0, min(h, box.y0))
            bx1 = max(0, min(w, box.x1))
            by1 = max(0, min(h, box.y1))
            sub = opaque_mask[by0:by1, bx0:bx1]
            if not np.any(sub):
                raise ValueError(
                    f"ContactAnchorBox ({box.x0}, {box.y0}, {box.x1}, {box.y1}) "
                    f"contains zero opaque pixels."
                )
            sub_contact = np.zeros_like(sub, dtype=np.bool_)
            sh, sw = sub.shape
            if box.mode == "all_opaque":
                sub_contact = sub.copy()
            elif box.mode == "perimeter_rim":
                eroded = ndimage.binary_erosion(sub, iterations=box.band_px, border_value=0)
                sub_contact = sub & (~eroded)
            elif box.mode == "bottom_band":
                for col in range(sw):
                    ys = np.where(sub[:, col])[0]
                    if len(ys) > 0:
                        y_max = int(ys.max())
                        sub_contact[max(0, y_max - box.band_px + 1) : y_max + 1, col] = (
                            sub[max(0, y_max - box.band_px + 1) : y_max + 1, col]
                        )
            elif box.mode == "top_band":
                for col in range(sw):
                    ys = np.where(sub[:, col])[0]
                    if len(ys) > 0:
                        y_min = int(ys.min())
                        sub_contact[y_min : min(sh, y_min + box.band_px), col] = (
                            sub[y_min : min(sh, y_min + box.band_px), col]
                        )
            elif box.mode == "left_band":
                for row in range(sh):
                    xs = np.where(sub[row, :])[0]
                    if len(xs) > 0:
                        x_min = int(xs.min())
                        sub_contact[row, x_min : min(sw, x_min + box.band_px)] = (
                            sub[row, x_min : min(sw, x_min + box.band_px)]
                        )
            elif box.mode == "right_band":
                for row in range(sh):
                    xs = np.where(sub[row, :])[0]
                    if len(xs) > 0:
                        x_max = int(xs.max())
                        sub_contact[row, max(0, x_max - box.band_px + 1) : x_max + 1] = (
                            sub[row, max(0, x_max - box.band_px + 1) : x_max + 1]
                        )
            contact_mask[by0:by1, bx0:bx1] |= sub_contact
        return contact_mask

    # Automatic floor footprint mode per bridged object group
    if cfg.component_bridge_px > 0:
        bridged = ndimage.distance_transform_edt(~opaque_mask) <= float(cfg.component_bridge_px)
    else:
        bridged = opaque_mask.copy()
    labeled, num_groups = ndimage.label(bridged)
    for gid in range(1, num_groups + 1):
        group_opaque = opaque_mask & (labeled == gid)
        if int(group_opaque.sum()) < cfg.min_component_area_px:
            continue
        ys_all, xs_all = np.where(group_opaque)
        y_group_max = int(ys_all.max())
        x_min, x_max = int(xs_all.min()), int(xs_all.max())
        for col in range(x_min, x_max + 1):
            col_ys = np.where(group_opaque[:, col])[0]
            if len(col_ys) == 0:
                continue
            y_bot = int(col_ys.max())
            if y_bot >= y_group_max - cfg.auto_floor_max_rise_px:
                y_start = max(0, y_bot - cfg.auto_floor_band_px + 1)
                contact_mask[y_start : y_bot + 1, col] |= group_opaque[y_start : y_bot + 1, col]

    return contact_mask


@dataclass(frozen=True)
class ContactAnchoredSolveResult:
    """Outputs of `solve_contact_anchored_layer_depth` for asset persistence and diagnostics."""

    depth_u8: npt.NDArray[np.uint8]
    world_z_map: npt.NDArray[np.float32]
    opaque_mask: npt.NDArray[np.bool_]
    contact_mask: npt.NDArray[np.bool_]


def solve_contact_anchored_layer_depth(
    room_depth_expanded: npt.NDArray[Any],
    cutout_rgba: npt.NDArray[Any],
    cfg: ContactAnchoredLayerDepthConfig,
    neural_protrusion_norm: npt.NDArray[Any] | None = None,
) -> ContactAnchoredSolveResult:
    """
    Laws 1, 2, & 4: Solves a contact-anchored, sculpturally relieved, outward-EDT-extrapolated
    depth map for an RGBA cutout layer (`mid`, `fg`, or `extraLayer`).

    Parameters
    ----------
    room_depth_expanded:
        Cleaned `layer0` room depth map on the expanded `(2700x1800)` canvas.
    cutout_rgba:
        `HxWx4` RGBA cutout image (`1500x1000` inner or `2700x1800` expanded).
    cfg:
        Typed SSOT configuration specifying `(bg_z, room_depth_amplitude, room_pin_margin_uv)`
        and `(layer_z, layer_depth_amplitude)`.
    neural_protrusion_norm:
        Optional normalized `[0, 1]` Depth Anything V2 protrusion field matching `cutout_rgba`
        dimensions (e.g. room-detrended neural relief `max(0, d_raw - d_room_interp)` normalized
        over opaque pixels). Zeroed smoothly at `contact_mask` so $\\Delta Z_{\\text{world}} = 0$
        is preserved exactly at every contact anchor.
    """
    if cutout_rgba.ndim != 3 or cutout_rgba.shape[2] != 4:
        raise ValueError(f"Expected HxWx4 RGBA cutout, got shape {cutout_rgba.shape}.")

    lh, lw = cutout_rgba.shape[:2]
    opaque_mask = cutout_rgba[:, :, 3] >= cfg.alpha_threshold_u8
    if not np.any(opaque_mask):
        raise ValueError("Cutout RGBA contains zero opaque pixels above alpha_threshold_u8.")

    room_world_z_exp = compute_world_z_map(
        depth_map=room_depth_expanded,
        plane_z=cfg.bg_z,
        depth_amplitude=cfg.room_depth_amplitude,
        pin_margin_uv=cfg.room_pin_margin_uv,
    )
    host_world_z = extract_layer_window_from_expanded(room_world_z_exp, layer_shape=(lh, lw))

    # Filter out microscopic alpha dust (< min_component_area_px) before solving PDE
    raw_labeled, raw_n = ndimage.label(opaque_mask)
    if raw_n > 0:
        comp_sizes = np.bincount(raw_labeled.ravel())
        keep_labels = comp_sizes >= cfg.min_component_area_px
        keep_labels[0] = False
        clean_opaque = keep_labels[raw_labeled]
        if np.any(clean_opaque):
            opaque_mask = clean_opaque

    contact_mask = extract_contact_anchor_mask(opaque_mask, cfg)
    if not np.any(contact_mask):
        raise ValueError("extract_contact_anchor_mask produced zero contact pixels.")

    # Bridge close finials/wisps into their parent sculpture's harmonic PDE domain
    if cfg.component_bridge_px > 0:
        pde_domain = ndimage.distance_transform_edt(~opaque_mask) <= float(cfg.component_bridge_px)
    else:
        pde_domain = opaque_mask.copy()

    # Ensure any bridged PDE component without its own contact anchor (e.g. an isolated dust speck
    # larger than min_component_area_px) is excluded rather than failing if detached
    labeled_pde, n_pde = ndimage.label(pde_domain)
    anchored_pde_domain = np.zeros_like(pde_domain)
    for pid in range(1, n_pde + 1):
        comp = labeled_pde == pid
        if np.any(comp & contact_mask):
            anchored_pde_domain |= comp

    anchored_opaque = opaque_mask & anchored_pde_domain
    if not np.any(anchored_opaque):
        raise ValueError("No opaque sculpture components intersect contact_mask.")

    # 1. Harmonic support-surface continuation Z_anchor(y, x)
    z_anchor = _solve_harmonic_field_on_domain(
        domain_mask=anchored_pde_domain,
        dirichlet_mask=contact_mask,
        dirichlet_values=host_world_z.astype(np.float64),
    )

    # 2. Contact-tapered neural 3D sculptural relief above the contact anchor
    z_relief = np.zeros((lh, lw), dtype=np.float64)
    if neural_protrusion_norm is not None and cfg.neural_relief_amplitude > 0.0:
        prot = normalize_depth_array(neural_protrusion_norm).astype(np.float64)
        if prot.shape != (lh, lw):
            raise ValueError(
                f"neural_protrusion_norm shape {prot.shape} != cutout shape {(lh, lw)}."
            )
        if cfg.interior_smooth_sigma_px > 0.0:
            prot = ndimage.gaussian_filter(prot, sigma=cfg.interior_smooth_sigma_px, mode="nearest")
        dist_from_contact = ndimage.distance_transform_edt(~contact_mask)
        taper = glsl_smoothstep(
            0.0,
            cfg.contact_relief_taper_px,
            dist_from_contact.astype(np.float32),
        ).astype(np.float64)
        z_relief = float(cfg.neural_relief_amplitude) * prot * taper

    z_obj = z_anchor + z_relief

    # 3. Law 4: Optional host-surface monotonicity (never bury non-contact sculpture behind host)
    if cfg.enforce_front_of_host:
        non_contact_opaque = anchored_opaque & (~contact_mask)
        min_allowed_z = host_world_z.astype(np.float64) - float(cfg.host_clearance_tolerance_world)
        # Taper clearance enforcement smoothly near contact_mask so C^0 continuity is preserved
        dist_from_contact = ndimage.distance_transform_edt(~contact_mask)
        clearance_taper = glsl_smoothstep(
            0.0,
            cfg.contact_relief_taper_px,
            dist_from_contact.astype(np.float32),
        ).astype(np.float64)
        floor_bound = (
            (1.0 - clearance_taper) * z_anchor + clearance_taper * np.maximum(z_obj, min_allowed_z)
        )
        z_obj[non_contact_opaque] = floor_bound[non_contact_opaque]

    # Exact Dirichlet contact guarantee prior to uint8 quantization
    z_obj[contact_mask] = host_world_z[contact_mask].astype(np.float64)

    # 4. Validate dynamic range (Rule 1: zero silent clamping!)
    opaque_z_vals = z_obj[anchored_opaque]
    z_min_solved = float(opaque_z_vals.min())
    z_max_solved = float(opaque_z_vals.max())
    z_lower_cap = float(cfg.layer_z) - float(cfg.overflow_tolerance_world)
    z_upper_cap = (
        float(cfg.layer_z)
        + float(cfg.layer_depth_amplitude)
        + float(cfg.overflow_tolerance_world)
    )
    if z_min_solved < z_lower_cap or z_max_solved > z_upper_cap:
        raise ValueError(
            f"Solved object world-Z range [{z_min_solved:.2f}, {z_max_solved:.2f}] overflows "
            f"configured layer window [{cfg.layer_z:.2f}, "
            f"{cfg.layer_z + cfg.layer_depth_amplitude:.2f}] "
            f"(tolerance={cfg.overflow_tolerance_world:.2f}). "
            f"Adjust layer_z / layer_depth_amplitude in SSOT config instead of clamping."
        )

    d_interior = np.clip(
        (z_obj - float(cfg.layer_z)) / float(cfg.layer_depth_amplitude),
        0.0,
        1.0,
    ).astype(np.float32)

    # 5. Law 2: Outward Euclidean Distance Transform extrapolation across transparent pixels
    d_extrapolated = extrapolate_depth_outward_edt(
        interior_depth=d_interior,
        opaque_mask=anchored_opaque,
    )
    depth_u8 = np.rint(d_extrapolated * UINT8_MAX).astype(np.uint8)
    quantized_world_z = compute_world_z_map(
        depth_map=depth_u8,
        plane_z=cfg.layer_z,
        depth_amplitude=cfg.layer_depth_amplitude,
        pin_margin_uv=cfg.layer_pin_margin_uv,
    )

    return ContactAnchoredSolveResult(
        depth_u8=depth_u8,
        world_z_map=quantized_world_z,
        opaque_mask=anchored_opaque,
        contact_mask=contact_mask,
    )


def compute_room_relative_neural_protrusion(
    raw_neural_depth_expanded: npt.NDArray[Any],
    cutout_rgba: npt.NDArray[Any],
    clean_cfg: RoomDepthCleanConfig,
    alpha_threshold_u8: int = 20,
    upper_percentile: float = 98.0,
) -> npt.NDArray[np.float32]:
    """
    Computes genuine 3D sculptural protrusion from a raw Depth Anything V2 map on the
    original source plate by subtracting the harmonically healed room baseline from the
    same neural prediction (`max(0, d_raw - d_healed_room)`), eliminating global vertical
    perspective bias while preserving multi-surface sculptural relief.
    """
    d_raw_exp = normalize_depth_array(raw_neural_depth_expanded)
    d_room_healed_u8 = clean_room_depth_under_cutouts(
        room_depth=d_raw_exp,
        cutout_layers=[cutout_rgba],
        cfg=clean_cfg,
    )
    d_room_healed_exp = normalize_depth_array(d_room_healed_u8)

    lh, lw = cutout_rgba.shape[:2]
    d_raw_win = extract_layer_window_from_expanded(d_raw_exp, layer_shape=(lh, lw))
    d_healed_win = extract_layer_window_from_expanded(d_room_healed_exp, layer_shape=(lh, lw))

    excess = np.maximum(0.0, d_raw_win - d_healed_win)
    opaque = cutout_rgba[:, :, 3] >= alpha_threshold_u8
    if not np.any(opaque):
        return np.zeros((lh, lw), dtype=np.float32)

    opaque_vals = excess[opaque]
    scale = float(np.percentile(opaque_vals, upper_percentile))
    if scale <= 1e-5:
        return np.zeros((lh, lw), dtype=np.float32)
    return np.clip(excess / scale, 0.0, 1.0).astype(np.float32)


def verify_layer_contact_and_discontinuity(
    room_depth_expanded: npt.NDArray[Any],
    layer_rgba: npt.NDArray[Any],
    layer_depth: npt.NDArray[Any],
    contact_mask: npt.NDArray[np.bool_],
    bg_z: float,
    room_depth_amplitude: float,
    room_pin_margin_uv: float,
    layer_z: float,
    layer_depth_amplitude: float,
    layer_pin_margin_uv: float,
    verify_cfg: LayerTopologyVerificationConfig,
) -> LayerTopologyVerificationReport:
    r"""
    Programmatically verifies Laws 1, 2, & 4 for a displaced cutout layer:
    - Law 1 ($C^0$ Contact Continuity): $|\Delta Z_{\text{world}}|$ on `contact_mask`.
    - Law 2 ($C^{-1}$ EDT Rim Discontinuity): Exterior transparent rim pixels within
      `rim_exterior_probe_px` match nearest interior silhouette pixel without cliff-to-zero drops.
    - Law 4 (Depth Monotonicity): Opaque foreground pixels satisfy
      $Z_{\text{world}}^{\text{layer}} \ge Z_{\text{world}}^{\text{host}} - \text{tol}$.
    """
    if layer_rgba.ndim != 3 or layer_rgba.shape[2] != 4:
        raise ValueError(f"Expected HxWx4 RGBA array, got {layer_rgba.shape}.")
    lh, lw = layer_rgba.shape[:2]
    if layer_depth.shape != (lh, lw) or contact_mask.shape != (lh, lw):
        raise ValueError(
            f"Dimension mismatch: rgba={(lh, lw)}, depth={layer_depth.shape}, "
            f"contact={contact_mask.shape}."
        )
    if not np.any(contact_mask):
        raise ValueError("verify_layer_contact_and_discontinuity requires non-empty contact_mask.")

    room_z_exp = compute_world_z_map(
        depth_map=room_depth_expanded,
        plane_z=bg_z,
        depth_amplitude=room_depth_amplitude,
        pin_margin_uv=room_pin_margin_uv,
    )
    host_z = extract_layer_window_from_expanded(room_z_exp, layer_shape=(lh, lw))
    layer_z_map = compute_world_z_map(
        depth_map=layer_depth,
        plane_z=layer_z,
        depth_amplitude=layer_depth_amplitude,
        pin_margin_uv=layer_pin_margin_uv,
    )

    opaque_mask = layer_rgba[:, :, 3] >= verify_cfg.alpha_threshold_u8
    if not np.any(opaque_mask):
        raise ValueError("verify_layer_contact_and_discontinuity: zero opaque pixels in layer_rgba.")

    # Law 1: Contact error
    contact_dz = np.abs(layer_z_map[contact_mask] - host_z[contact_mask])
    max_contact_dz = float(contact_dz.max())
    p95_contact_dz = float(np.percentile(contact_dz, 95.0))
    mean_contact_dz = float(contact_dz.mean())

    # Law 2: Exterior rim EDT continuity across free silhouette
    eroded_interior = ndimage.binary_erosion(
        opaque_mask, iterations=verify_cfg.rim_erosion_px, border_value=0
    )
    reference_interior = eroded_interior if np.any(eroded_interior) else opaque_mask
    dist_outside, nearest_in = ndimage.distance_transform_edt(~opaque_mask, return_indices=True)
    exterior_rim = (~opaque_mask) & (dist_outside <= float(verify_cfg.rim_exterior_probe_px))
    _, nearest_ref = ndimage.distance_transform_edt(~reference_interior, return_indices=True)

    if np.any(exterior_rim):
        # Compare exterior pixel against the nearest opaque boundary pixel (and eroded interior)
        rim_y, rim_x = np.where(exterior_rim)
        nb_y = nearest_in[0, rim_y, rim_x]
        nb_x = nearest_in[1, rim_y, rim_x]
        rim_jumps = np.abs(layer_z_map[rim_y, rim_x] - layer_z_map[nb_y, nb_x])
        max_rim_jump = float(rim_jumps.max())
        p95_rim_jump = float(np.percentile(rim_jumps, 95.0))
        rim_count = int(len(rim_y))
    else:
        max_rim_jump = 0.0
        p95_rim_jump = 0.0
        rim_count = 0

    # Law 4: Monotonicity vs host room surface
    signed_dz = layer_z_map[opaque_mask] - host_z[opaque_mask]
    min_signed_dz = float(signed_dz.min())

    contact_passed = (
        max_contact_dz <= verify_cfg.max_allowed_contact_dz_world
        and p95_contact_dz <= verify_cfg.max_allowed_p95_contact_dz_world
    )
    discontinuity_passed = (
        layer_pin_margin_uv == 0.0
        and max_rim_jump <= verify_cfg.max_allowed_rim_jump_world
    )
    monotonicity_passed = (
        min_signed_dz >= -verify_cfg.host_monotonicity_tolerance_world
    )

    return LayerTopologyVerificationReport(
        contact_pixel_count=int(contact_mask.sum()),
        opaque_pixel_count=int(opaque_mask.sum()),
        rim_probe_pixel_count=rim_count,
        max_abs_contact_dz_world=max_contact_dz,
        p95_abs_contact_dz_world=p95_contact_dz,
        mean_abs_contact_dz_world=mean_contact_dz,
        max_abs_rim_jump_world=max_rim_jump,
        p95_abs_rim_jump_world=p95_rim_jump,
        min_signed_dz_vs_host_world=min_signed_dz,
        contact_passed=contact_passed,
        discontinuity_passed=discontinuity_passed,
        monotonicity_passed=monotonicity_passed,
        all_passed=bool(contact_passed and discontinuity_passed and monotonicity_passed),
    )
