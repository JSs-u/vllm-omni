# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-level regression tests for the Helios stage-2 resolution contract.

No model, no torch: the failure mode in #8303 is an integer geometry mismatch,
so the contract can be pinned by arithmetic alone.
"""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.models.helios.stage2_geometry import (
    stage2_latent_alignment,
    stage2_pixel_alignment,
    validate_stage2_latent_geometry,
    validate_stage2_pixel_geometry,
)

VAE_SPATIAL_SCALE = 8


def _pyramid(latent_side: int, num_stages: int) -> list[int]:
    """Replicate the floor-halving in ``_stage2_sample`` / ``_prepare_stage2_chunk``."""
    sides = [latent_side]
    for _ in range(num_stages - 1):
        sides.append(sides[-1] // 2)
    return sides


def test_alignment_is_scale_times_pyramid_stages():
    assert stage2_latent_alignment(1) == 2
    assert stage2_latent_alignment(2) == 4
    assert stage2_latent_alignment(3) == 8
    assert stage2_pixel_alignment(VAE_SPATIAL_SCALE, 3) == 64
    assert stage2_pixel_alignment(VAE_SPATIAL_SCALE, 2) == 32
    assert stage2_pixel_alignment(VAE_SPATIAL_SCALE, 1) == 16


@pytest.mark.parametrize("num_stages", [0, -1])
def test_rejects_invalid_stage_count(num_stages):
    with pytest.raises(ValueError, match="pyramid_num_stages must be >= 1"):
        stage2_latent_alignment(num_stages)


@pytest.mark.parametrize(
    "height,width",
    [
        (400, 624),  # the reported request: 50x78 -> 25x39 -> 12x19, 19 is odd
        (400, 640),  # patch-aligned at every level, but restores to 48 instead of 50
        (416, 624),  # width is the only offending axis
        (720, 1280),
    ],
)
def test_rejects_unaligned_request(height, width):
    with pytest.raises(ValueError, match="multiple of 64 pixels"):
        validate_stage2_pixel_geometry(
            height,
            width,
            num_stages=3,
            vae_spatial_scale=VAE_SPATIAL_SCALE,
        )


@pytest.mark.parametrize(
    "height,width",
    [
        (384, 640),
        (512, 512),
        (1024, 1024),
        (256, 1024),
    ],
)
def test_accepts_aligned_request(height, width):
    validate_stage2_pixel_geometry(
        height,
        width,
        num_stages=3,
        vae_spatial_scale=VAE_SPATIAL_SCALE,
    )


def test_error_names_the_offending_axis_and_a_fix():
    with pytest.raises(ValueError) as excinfo:
        validate_stage2_pixel_geometry(
            400,
            624,
            num_stages=3,
            vae_spatial_scale=VAE_SPATIAL_SCALE,
        )
    message = str(excinfo.value)
    # Both axes are unaligned here, so both have to be named.
    assert "height=400 (nearest 384 or 448)" in message
    assert "width=624 (nearest 576 or 640)" in message
    assert "Use 384x576 (round down) or 448x640 (round up)" in message


def test_error_only_names_the_unaligned_axis():
    with pytest.raises(ValueError) as excinfo:
        validate_stage2_pixel_geometry(
            384,
            624,
            num_stages=3,
            vae_spatial_scale=VAE_SPATIAL_SCALE,
        )
    message = str(excinfo.value)
    assert "width=624" in message
    assert "height=384" not in message


def test_stage_count_changes_the_contract():
    # One stage only needs patch alignment: 50x78 latents, no halving.
    validate_stage2_pixel_geometry(400, 624, num_stages=1, vae_spatial_scale=VAE_SPATIAL_SCALE)

    # Two stages halve to 25x40, and 25 is odd: 400x640 is no longer usable
    # even though 400x624 was.
    with pytest.raises(ValueError, match="multiple of 32 pixels"):
        validate_stage2_pixel_geometry(400, 640, num_stages=2, vae_spatial_scale=VAE_SPATIAL_SCALE)
    validate_stage2_pixel_geometry(416, 640, num_stages=2, vae_spatial_scale=VAE_SPATIAL_SCALE)


def test_latent_validator_matches_the_pixel_one():
    # 50x78 latents is exactly the 400x624 request.
    with pytest.raises(ValueError, match="multiple of 8 latent pixels"):
        validate_stage2_latent_geometry(50, 78, num_stages=3)
    validate_stage2_latent_geometry(48, 80, num_stages=3)


def test_documented_failure_is_arithmetic():
    """Pin the mechanism #8303 reports, so the guard cannot be "simplified" away."""
    sides = _pyramid(78, 3)
    assert sides == [78, 39, 19]

    # Patch embedding is stride 2 without padding: 19 -> 9 patches -> 18 pixels
    # while the scheduler still holds a 19-wide sample.
    assert sides[-1] // 2 * 2 == 18
    assert sides[-1] == 19

    # Height 50 is even at the smallest level but the halving is not exact, so
    # the upsample chain cannot restore the requested 50.
    assert _pyramid(50, 3) == [50, 25, 12]
    assert 12 * 2 * 2 == 48
