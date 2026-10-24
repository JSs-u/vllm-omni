# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolution constraints for the Helios stage-2 pyramid denoiser.

Stage 2 denoises a pyramid: it halves the latent grid ``pyramid_num_stages - 1``
times with ``F.interpolate`` (floor division on the target size), denoises the
smallest level, and upsamples with ``nearest`` on the way back up. Two integer
properties have to hold for that to be well defined:

* every level must be divisible by the transformer patch size (``(1, 2, 2)``),
  because the patch embedding is a stride-2 convolution with no padding, so an
  odd side loses its last column/row at the patch embedding and the scheduler
  then receives latents one pixel narrower than the ones it is updating; and
* the halving has to be exact, otherwise the upsample chain returns a grid that
  is smaller than the one it started from (``50 -> 25 -> 12 -> 24 -> 48``).

Both are implied by the smallest level being even, i.e. by the starting latent
grid being a multiple of ``2 ** pyramid_num_stages``. The helpers below turn
that requirement into a single check so callers fail with an actionable error
instead of a tensor-shape mismatch inside the first scheduler step.
"""

from __future__ import annotations

__all__ = [
    "stage2_latent_alignment",
    "stage2_pixel_alignment",
    "validate_stage2_latent_geometry",
    "validate_stage2_pixel_geometry",
]


def stage2_latent_alignment(num_stages: int) -> int:
    """Multiple the *latent* grid has to satisfy, in latent pixels.

    The smallest pyramid level is ``latent // 2 ** (num_stages - 1)`` and has to
    be even for the stride-2 patch embedding, hence ``2 ** num_stages``.
    """
    _check_num_stages(num_stages)
    return 1 << num_stages


def stage2_pixel_alignment(vae_spatial_scale: int, num_stages: int) -> int:
    """Multiple the *request* height/width has to satisfy, in image pixels."""
    if vae_spatial_scale < 1:
        raise ValueError(f"vae_spatial_scale must be >= 1, got {vae_spatial_scale}")
    return int(vae_spatial_scale) * stage2_latent_alignment(num_stages)


def validate_stage2_pixel_geometry(
    height: int,
    width: int,
    *,
    num_stages: int,
    vae_spatial_scale: int,
) -> None:
    """Reject a request the stage-2 pyramid cannot be built for.

    Called before denoising starts, so an unsupported resolution fails with a
    clear message instead of a ``RuntimeError`` from ``scheduler.step``.
    """
    multiple = stage2_pixel_alignment(vae_spatial_scale, num_stages)
    _validate(
        height,
        width,
        multiple=multiple,
        unit="pixels",
        num_stages=num_stages,
        scale_hint=f"vae_scale_factor_spatial={vae_spatial_scale} * 2**pyramid_num_stages={num_stages}",
    )


def validate_stage2_latent_geometry(height: int, width: int, *, num_stages: int) -> None:
    """Same check for callers that already hold the latent grid."""
    multiple = stage2_latent_alignment(num_stages)
    _validate(
        height,
        width,
        multiple=multiple,
        unit="latent pixels",
        num_stages=num_stages,
        scale_hint=f"2**pyramid_num_stages={num_stages}",
    )


def _check_num_stages(num_stages: int) -> None:
    if num_stages < 1:
        raise ValueError(f"pyramid_num_stages must be >= 1, got {num_stages}")


def _nearest_multiples(value: int, multiple: int) -> tuple[int, int]:
    lower = (value // multiple) * multiple
    return lower, lower + multiple


def _validate(
    height: int,
    width: int,
    *,
    multiple: int,
    unit: str,
    num_stages: int,
    scale_hint: str,
) -> None:
    _check_num_stages(num_stages)
    bad_h = height % multiple != 0
    bad_w = width % multiple != 0
    if not (bad_h or bad_w):
        return

    h_low, h_high = _nearest_multiples(height, multiple)
    w_low, w_high = _nearest_multiples(width, multiple)
    details = []
    if bad_h:
        details.append(f"height={height} (nearest {h_low} or {h_high})")
    if bad_w:
        details.append(f"width={width} (nearest {w_low} or {w_high})")

    raise ValueError(
        f"Helios stage-2 pyramid denoising with pyramid_num_stages={num_stages} "
        f"requires each spatial dimension to be a multiple of {multiple} {unit} "
        f"({scale_hint}); got {height}x{width}: "
        + ", ".join(details)
        + ". The pyramid halves the latent grid once per stage and the "
        "transformer patch embedding is a stride-2 convolution without padding, "
        "so an unaligned size makes the smallest level lose a column/row and the "
        "scheduler fails on mismatched extents. "
        f"Use {h_low}x{w_low} (round down) or {h_high}x{w_high} (round up), "
        "or disable stage 2."
    )
