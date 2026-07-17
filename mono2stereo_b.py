"""Depth-edge-sharpened DIBR variant.

This file deliberately leaves ``mono2stereo_a.py`` unchanged.  It reuses the
existing video/depth/inpaint pipeline, but replaces the DIBR entry point with a
wrapper that collapses wide, interpolated disparity ramps into a foreground /
background step before forward splatting.  The goal is to turn displaced
foreground-colour bands into an explicit disocclusion hole.

Example:
    python mono2stereo_b.py \
        --video-path data/cccc-Trim.mp4 \
        --output output/test/cccc-Trim-b.mp4 \
        --fp16 --profile-time --profile-total

Variant-only arguments are consumed here before the remaining arguments are
passed unchanged to mono2stereo_a.py:
    --depth-edge-kernel 15
    --depth-edge-threshold 3.0
    --depth-edge-iterations 1
    --disable-depth-edge-sharpen
    --strict-bg-max-distance 64
    --strict-bg-safety-margin 2
    --disable-transition-reject
    --disable-strict-bg-inpaint
"""

import argparse
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

import mono2stereo_a as base


def _parse_variant_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--depth-edge-kernel",
        type=int,
        default=15,
        help="视差边缘锐化的水平窗口宽度（DIBR像素，奇数，默认15）",
    )
    parser.add_argument(
        "--depth-edge-threshold",
        type=float,
        default=3.0,
        help="触发锐化所需的窗口内视差跨度（DIBR像素，默认3.0）",
    )
    parser.add_argument(
        "--depth-edge-iterations",
        type=int,
        default=1,
        help="视差边缘锐化次数（默认1；通常不建议超过2）",
    )
    parser.add_argument(
        "--disable-depth-edge-sharpen",
        action="store_true",
        help="禁用B版本的视差边缘锐化，用于消融对照",
    )
    parser.add_argument(
        "--strict-bg-max-distance",
        type=int,
        default=64,
        help="严格背景填充允许搜索的最大水平距离（DIBR像素，默认64）",
    )
    parser.add_argument(
        "--strict-bg-safety-margin",
        type=int,
        default=2,
        help="取色前要求连续背景的安全宽度，避开抗锯齿前景边（默认2）",
    )
    parser.add_argument(
        "--strict-bg-depth-tolerance",
        type=float,
        default=0.025,
        help="局部右背景允许比左边界更近的near容差（默认0.025）",
    )
    parser.add_argument(
        "--transition-reject-margin",
        type=float,
        default=0.10,
        help="从splat中排除深度斜坡中间80%的不可靠RGB（默认0.10）",
    )
    parser.add_argument(
        "--narrow-hole-fallback-width",
        type=int,
        default=10,
        help="窄洞可直接按右向遮挡几何填充的最大宽度（默认10像素）",
    )
    parser.add_argument(
        "--disable-transition-reject",
        action="store_true",
        help="保留被深度锐化改变的过渡RGB像素（默认会将其转为hole）",
    )
    parser.add_argument(
        "--disable-strict-bg-inpaint",
        action="store_true",
        help="禁用B版本严格背景填充，退回A版本的空洞修复",
    )
    args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    if args.depth_edge_kernel < 1 or args.depth_edge_kernel % 2 == 0:
        raise ValueError("depth-edge-kernel 必须是正奇数")
    if args.depth_edge_threshold < 0:
        raise ValueError("depth-edge-threshold 必须 >= 0")
    if args.depth_edge_iterations < 0:
        raise ValueError("depth-edge-iterations 必须 >= 0")
    if args.strict_bg_max_distance < 1:
        raise ValueError("strict-bg-max-distance 必须 >= 1")
    if args.strict_bg_safety_margin < 0:
        raise ValueError("strict-bg-safety-margin 必须 >= 0")
    if args.strict_bg_depth_tolerance < 0:
        raise ValueError("strict-bg-depth-tolerance 必须 >= 0")
    if not 0.0 <= args.transition_reject_margin < 0.5:
        raise ValueError("transition-reject-margin 必须在 [0, 0.5) 内")
    if args.narrow_hole_fallback_width < 0:
        raise ValueError("narrow-hole-fallback-width 必须 >= 0")
    return args


@torch.no_grad()
def sharpen_disparity_edges(
    disparity: torch.Tensor,
    kernel_size: int = 15,
    threshold: float = 3.0,
    iterations: int = 1,
    reject_margin: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collapse broad disparity ramps into a two-level depth discontinuity.

    ``disparity`` is [H, W] in DIBR-pixel units.  For each horizontal window,
    pixels whose local disparity span is below ``threshold`` are kept exactly
    unchanged.  Across a strong depth boundary, values are snapped to the
    nearer of the local foreground/background extrema.  This removes the
    intermediate disparity values that otherwise copy foreground colours into
    the disoccluded side of an object.

    Returns the sharpened disparity and a bool mask for only the ambiguous
    centre of the original depth ramp.  Pixels close to either local surface
    extremum are snapped but remain trusted, avoiding an unnecessarily wide
    rejected band around thin foreground structures.
    """
    if disparity.ndim != 2:
        raise ValueError("disparity 必须是 [H, W] 二维张量")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须是正奇数")
    if threshold < 0:
        raise ValueError("threshold 必须 >= 0")
    if not 0.0 <= reject_margin < 0.5:
        raise ValueError("reject_margin 必须在 [0, 0.5) 内")
    if iterations <= 0 or kernel_size == 1:
        return disparity, torch.zeros_like(disparity, dtype=torch.bool)

    sharpened = disparity
    unreliable_total = torch.zeros_like(disparity, dtype=torch.bool)
    padding = kernel_size // 2

    for _ in range(iterations):
        d = sharpened[None, None]
        local_max = F.max_pool2d(
            d, kernel_size=(1, kernel_size), stride=1, padding=(0, padding)
        )
        local_min = -F.max_pool2d(
            -d, kernel_size=(1, kernel_size), stride=1, padding=(0, padding)
        )
        local_span = local_max - local_min
        strong_edge = local_span >= threshold
        midpoint = 0.5 * (local_max + local_min)
        snapped = torch.where(d >= midpoint, local_max, local_min)
        next_disparity = torch.where(strong_edge, snapped, d)[0, 0]

        relative_depth = (d - local_min) / local_span.clamp_min(1e-6)
        ambiguous = (
            strong_edge
            & (relative_depth > reject_margin)
            & (relative_depth < 1.0 - reject_margin)
        )[0, 0]
        unreliable_total |= ambiguous & ((next_disparity - sharpened).abs() > 1e-6)
        sharpened = next_disparity

    return sharpened, unreliable_total


_VARIANT_ARGS = _parse_variant_args()
_BASE_FORWARD_WARP = base.forward_warp_right_gpu
_BASE_INPAINT = base.fast_inpaint_gpu
_LAST_TARGET_NEAR = None


@torch.no_grad()
def project_source_mask(
    source_mask: torch.Tensor,
    disparity: torch.Tensor,
) -> torch.Tensor:
    """Forward-project a source-space bool mask to both bilinear target taps."""
    h, w = source_mask.shape
    device = source_mask.device
    xs = torch.arange(w, device=device).view(1, w).expand(h, w)
    ys = torch.arange(h, device=device).view(h, 1).expand(h, w)
    x_float = xs.to(disparity.dtype) - disparity
    x_left = torch.floor(x_float).long()
    frac = (x_float - x_left.to(x_float.dtype)).clamp(0.0, 1.0)

    target_mask = torch.zeros((h * w,), device=device, dtype=torch.bool)
    for x_target, weight in ((x_left, 1.0 - frac), (x_left + 1, frac)):
        valid = (
            source_mask
            & (x_target >= 0)
            & (x_target < w)
            & (weight > 1e-6)
        )
        target_index = (ys * w + x_target)[valid]
        target_mask[target_index] = True
    return target_mask.reshape(h, w)


@torch.no_grad()
def forward_target_near(
    near_score: torch.Tensor,
    disparity: torch.Tensor,
    excluded_source: torch.Tensor,
) -> torch.Tensor:
    """Build a target-coordinate near map from trusted source contributions."""
    h, w = near_score.shape
    n = h * w
    device = near_score.device
    xs = torch.arange(w, device=device).view(1, w).expand(h, w)
    ys = torch.arange(h, device=device).view(h, 1).expand(h, w)
    x_float = xs.to(disparity.dtype) - disparity
    x_left = torch.floor(x_float).long()
    frac = (x_float - x_left.to(x_float.dtype)).clamp(0.0, 1.0)
    near_flat = near_score.reshape(-1)
    target_near = torch.full(
        (n,), -1.0, device=device, dtype=near_score.dtype
    )

    for x_target, weight in ((x_left, 1.0 - frac), (x_left + 1, frac)):
        valid = (
            (~excluded_source)
            & (x_target >= 0)
            & (x_target < w)
            & (weight > 1e-6)
        )
        valid_flat = valid.reshape(-1)
        target_index = (ys * w + x_target).reshape(-1)[valid_flat]
        target_near.scatter_reduce_(
            0,
            target_index,
            near_flat[valid_flat],
            reduce="amax",
            include_self=True,
        )
    return target_near.reshape(h, w)


@torch.no_grad()
def forward_warp_excluding_source(
    left_rgb: torch.Tensor,
    disparity: torch.Tensor,
    near_score: torch.Tensor,
    excluded_source: torch.Tensor,
    stage_times: Dict[str, float],
    profile_sync: bool,
    depth_tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward splat while completely excluding unreliable source pixels.

    Unlike clearing their projected target taps after a normal warp, excluding
    them in both Z-buffer passes preserves any trusted foreground/background
    contribution that lands on the same target pixel.
    """
    h, w, _ = left_rgb.shape
    n = h * w
    device = left_rgb.device
    trusted = ~excluded_source

    t0 = time.perf_counter()
    ys = torch.arange(h, device=device).view(h, 1).expand(h, w)
    xs = torch.arange(w, device=device).view(1, w).expand(h, w)
    x_float = xs.to(disparity.dtype) - disparity
    x_left = torch.floor(x_float).long()
    frac = (x_float - x_left.to(x_float.dtype)).clamp(0.0, 1.0)
    splats = ((x_left, 1.0 - frac), (x_left + 1, frac))
    base._maybe_sync(device, profile_sync)
    base._stage_add(stage_times, "warp_grid_gen", time.perf_counter() - t0)

    t0 = time.perf_counter()
    row_offset = ys * w
    near_flat = near_score.reshape(-1)
    trusted_flat = trusted.reshape(-1)
    base._maybe_sync(device, profile_sync)
    base._stage_add(stage_times, "warp_index_prep", time.perf_counter() - t0)

    t0 = time.perf_counter()
    max_near = torch.full((n,), -1.0, device=device, dtype=near_score.dtype)
    for x_target, spatial_weight in splats:
        valid = (
            trusted
            & (x_target >= 0)
            & (x_target < w)
            & (spatial_weight > 1e-6)
        )
        valid_flat = valid.reshape(-1)
        target_index = (row_offset + x_target).reshape(-1)[valid_flat]
        max_near.scatter_reduce_(
            0,
            target_index,
            near_flat[valid_flat],
            reduce="amax",
            include_self=True,
        )
    base._maybe_sync(device, profile_sync)
    base._stage_add(stage_times, "warp_z_buffer", time.perf_counter() - t0)

    t0 = time.perf_counter()
    colours = left_rgb.reshape(-1, 3)
    output = torch.zeros_like(colours)
    weight_sum = torch.zeros((n,), device=device, dtype=left_rgb.dtype)
    for x_target, spatial_weight in splats:
        valid = (
            trusted
            & (x_target >= 0)
            & (x_target < w)
            & (spatial_weight > 1e-6)
        )
        valid_flat = valid.reshape(-1)
        target_index = (row_offset + x_target).reshape(-1)[valid_flat]
        source_near = near_flat[valid_flat]
        visible = source_near >= (max_near[target_index] - depth_tolerance)
        if not torch.any(visible):
            continue
        target_visible = target_index[visible]
        weights = spatial_weight.reshape(-1)[valid_flat][visible].to(left_rgb.dtype)
        source_indices = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)[visible]
        output.scatter_add_(
            0,
            target_visible[:, None].expand(-1, 3),
            colours[source_indices] * weights[:, None],
        )
        weight_sum.scatter_add_(0, target_visible, weights)

    covered = weight_sum > 1e-6
    output[covered] /= weight_sum[covered, None]
    base._maybe_sync(device, profile_sync)
    base._stage_add(stage_times, "warp_scatter_pixels", time.perf_counter() - t0)
    return output.reshape(h, w, 3), (~covered).reshape(h, w)


@torch.no_grad()
def strict_background_inpaint_gpu_b(
    image: torch.Tensor,
    hole_mask: torch.Tensor,
    kernel_size: int,
    max_iter: int,
    stage_times: Dict[str, float],
    profile_sync: bool,
    near: torch.Tensor = None,
    bg_threshold: float = 0.3,
    edge_kernel_size: int = 5,
    non_edge_kernel_size: int = 11,
    edge_fill_mode: int = 0,
) -> torch.Tensor:
    """Fill holes only from target-aligned background pixels.

    For this right-eye warp (x_target = x_source - disparity), newly revealed
    background lies on the right side of a foreground boundary.  Candidate
    validity is decided from local target-space depth ordering rather than a
    single global near threshold: the right side may not be appreciably closer
    than the known surface on the left.  Texture is mirrored progressively
    from farther inside that right background instead of repeating one column.
    There is deliberately no arbitrary-known-pixel or global-average fallback.
    """
    del kernel_size, max_iter, near, edge_kernel_size, non_edge_kernel_size, edge_fill_mode
    global _LAST_TARGET_NEAR

    t0 = time.perf_counter()
    if not torch.any(hole_mask):
        base._stage_add(
            stage_times, "inpaint_b_skip_no_holes", time.perf_counter() - t0
        )
        return image.clone()

    target_near = _LAST_TARGET_NEAR
    if target_near is None or target_near.shape != hole_mask.shape:
        # Refuse a source-coordinate depth fallback: leaving a hole black is
        # safer than silently introducing foreground ghosting.
        result = image.masked_fill(hole_mask.unsqueeze(-1), 0.0)
        base._maybe_sync(image.device, profile_sync)
        base._stage_add(
            stage_times, "inpaint_b_missing_target_depth", time.perf_counter() - t0
        )
        return result

    h, w = hole_mask.shape
    x = torch.arange(w, device=image.device, dtype=torch.long).view(1, w).expand(h, w)
    known = (~hole_mask) & (target_near >= 0.0)

    # Do not sample the first antialiased pixels beside a silhouette.  For a
    # right-side candidate, requiring background pixels immediately to its
    # left moves the first eligible sample ``margin`` pixels into the revealed
    # background.  The rare left fallback applies the symmetric rule.
    right_bg = known.clone()
    left_bg = known.clone()
    for offset in range(1, _VARIANT_ARGS.strict_bg_safety_margin + 1):
        has_bg_left = torch.zeros_like(known)
        has_bg_left[:, offset:] = known[:, :-offset]
        right_bg &= has_bg_left

        has_bg_right = torch.zeros_like(known)
        has_bg_right[:, :-offset] = known[:, offset:]
        left_bg &= has_bg_right

    # Nearest valid background at or to the right of every target pixel.
    right_seed = torch.where(right_bg, x, torch.full_like(x, w))
    right_index = torch.flip(
        torch.cummin(torch.flip(right_seed, dims=(1,)), dim=1).values,
        dims=(1,),
    )
    right_distance = right_index - x
    right_ok_geometry = (
        (right_index < w)
        & (right_distance >= 0)
        & (right_distance <= _VARIANT_ARGS.strict_bg_max_distance)
    )

    # The nearest known pixel on the left represents the foreground-side
    # boundary depth used to validate the proposed right background.
    raw_left_seed = torch.where(known, x, torch.full_like(x, -1))
    raw_left_index = torch.cummax(raw_left_seed, dim=1).values
    raw_right_seed = torch.where(known, x, torch.full_like(x, w))
    raw_right_index = torch.flip(
        torch.cummin(torch.flip(raw_right_seed, dims=(1,)), dim=1).values,
        dims=(1,),
    )
    left_boundary_exists = raw_left_index >= 0
    right_boundary_exists = raw_right_index < w
    hole_run_width = raw_right_index - raw_left_index - 1
    narrow_directional_fallback = (
        hole_mask
        & left_boundary_exists
        & right_boundary_exists
        & (hole_run_width > 0)
        & (hole_run_width <= _VARIANT_ARGS.narrow_hole_fallback_width)
    )
    left_boundary_near = torch.gather(
        target_near, 1, raw_left_index.clamp(0, w - 1)
    )
    right_boundary_near = torch.gather(
        target_near, 1, right_index.clamp(0, w - 1)
    )
    right_depth_ok = (
        (~left_boundary_exists)
        | (
            right_boundary_near
            <= left_boundary_near + _VARIANT_ARGS.strict_bg_depth_tolerance
        )
    )
    right_ok = right_ok_geometry & (
        right_depth_ok | narrow_directional_fallback
    )

    # Mirror progressively farther background texture into wider holes.  The
    # safety margin is already included in right_distance and is subtracted so
    # that the pixel nearest the hole starts at the first safe sample.
    mirror_offset = (
        right_distance - _VARIANT_ARGS.strict_bg_safety_margin - 1
    ).clamp_min(0)
    mirror_budget = (
        _VARIANT_ARGS.strict_bg_max_distance - right_distance
    ).clamp_min(0)
    mirror_offset = torch.minimum(mirror_offset, mirror_budget)
    sample_right_index = (right_index + mirror_offset).clamp(0, w - 1)
    sample_right_known = torch.gather(known, 1, sample_right_index)
    sample_right_near = torch.gather(target_near, 1, sample_right_index)
    sample_right_depth_ok = (
        (~left_boundary_exists)
        | (
            sample_right_near
            <= left_boundary_near + _VARIANT_ARGS.strict_bg_depth_tolerance
        )
    )
    right_ok &= sample_right_known & (
        sample_right_depth_ok | narrow_directional_fallback
    )

    # Strict left fallback is reserved for the image boundary and still uses
    # the conservative global far-background criterion.
    left_seed = torch.where(left_bg, x, torch.full_like(x, -1))
    left_index = torch.cummax(left_seed, dim=1).values
    left_distance = x - left_index
    left_ok = (
        (left_index >= 0)
        & (left_distance >= 0)
        & (left_distance <= _VARIANT_ARGS.strict_bg_max_distance)
    )
    left_candidate_near = torch.gather(
        target_near, 1, left_index.clamp(0, w - 1)
    )
    left_ok &= left_candidate_near < bg_threshold

    use_right = hole_mask & right_ok
    use_left = hole_mask & (~right_ok) & left_ok
    result = image.clone()
    result[hole_mask] = 0.0

    gather_right = sample_right_index.unsqueeze(-1).expand(h, w, 3)
    gather_left = left_index.clamp(0, w - 1).unsqueeze(-1).expand(h, w, 3)
    right_colour = torch.gather(image, 1, gather_right)
    left_colour = torch.gather(image, 1, gather_left)
    result[use_right] = right_colour[use_right]
    result[use_left] = left_colour[use_left]

    base._maybe_sync(image.device, profile_sync)
    base._stage_add(
        stage_times, "inpaint_b_strict_background", time.perf_counter() - t0
    )
    return result


@torch.no_grad()
def forward_warp_right_gpu_b(
    left_rgb: torch.Tensor,
    disparity: torch.Tensor,
    near_score: torch.Tensor,
    stage_times: Dict[str, float],
    profile_sync: bool,
    depth_tolerance: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sharpen disparity boundaries, then call the existing soft-splat DIBR."""
    global _LAST_TARGET_NEAR
    if (
        _VARIANT_ARGS.disable_depth_edge_sharpen
        or _VARIANT_ARGS.depth_edge_iterations == 0
    ):
        right, hole = _BASE_FORWARD_WARP(
            left_rgb,
            disparity,
            near_score,
            stage_times,
            profile_sync,
            depth_tolerance=depth_tolerance,
        )
        _LAST_TARGET_NEAR = forward_target_near(
            near_score, disparity, torch.zeros_like(disparity, dtype=torch.bool)
        )
        return right, hole

    t0 = time.perf_counter()
    disparity_sharp, transition_source = sharpen_disparity_edges(
        disparity,
        kernel_size=_VARIANT_ARGS.depth_edge_kernel,
        threshold=_VARIANT_ARGS.depth_edge_threshold,
        iterations=_VARIANT_ARGS.depth_edge_iterations,
        reject_margin=_VARIANT_ARGS.transition_reject_margin,
    )
    base._maybe_sync(disparity.device, profile_sync)
    base._stage_add(
        stage_times, "depth_edge_sharpen", time.perf_counter() - t0
    )

    if not _VARIANT_ARGS.disable_transition_reject:
        right, hole = forward_warp_excluding_source(
            left_rgb,
            disparity_sharp,
            near_score,
            transition_source,
            stage_times,
            profile_sync,
            depth_tolerance,
        )
        excluded_source = transition_source
    else:
        right, hole = _BASE_FORWARD_WARP(
            left_rgb,
            disparity_sharp,
            near_score,
            stage_times,
            profile_sync,
            depth_tolerance=depth_tolerance,
        )
        excluded_source = torch.zeros_like(transition_source)

    _LAST_TARGET_NEAR = forward_target_near(
        near_score, disparity_sharp, excluded_source
    )
    return right, hole


def main() -> None:
    base.forward_warp_right_gpu = forward_warp_right_gpu_b
    if not _VARIANT_ARGS.disable_strict_bg_inpaint:
        base.fast_inpaint_gpu = strict_background_inpaint_gpu_b
    if _VARIANT_ARGS.disable_depth_edge_sharpen:
        print("[mono2stereo-b] 深度轮廓锐化: OFF")
    else:
        print(
            "[mono2stereo-b] 深度轮廓锐化: ON "
            f"(kernel={_VARIANT_ARGS.depth_edge_kernel}, "
            f"threshold={_VARIANT_ARGS.depth_edge_threshold:g}px, "
            f"iterations={_VARIANT_ARGS.depth_edge_iterations})"
        )
    print(
        "[mono2stereo-b] 过渡RGB拒绝: "
        f"{'OFF' if _VARIANT_ARGS.disable_transition_reject else 'ON'}"
    )
    print(
        "[mono2stereo-b] 严格背景修复: "
        f"{'OFF' if _VARIANT_ARGS.disable_strict_bg_inpaint else 'ON'} "
        f"(max_distance={_VARIANT_ARGS.strict_bg_max_distance}px, "
        f"safety_margin={_VARIANT_ARGS.strict_bg_safety_margin}px, "
        f"depth_tolerance={_VARIANT_ARGS.strict_bg_depth_tolerance:g}, "
        f"narrow_fallback={_VARIANT_ARGS.narrow_hole_fallback_width}px)"
    )
    base.main()


if __name__ == "__main__":
    main()
