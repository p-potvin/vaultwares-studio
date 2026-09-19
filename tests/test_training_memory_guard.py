"""The CPU image cache has to fit the flavour, and the gate says so before paying.

On 17 Sep a 2000-frame full-resolution run was OOMKilled after 271 minutes and
$3.62. Nothing was uploaded — no splat, no checkpoint, not even a partial — so
the whole run was lost. The cause was arithmetic that could have been done in
advance: ``--pipeline.datamanager.cache-images cpu`` holds every training image
in host RAM uncompressed, which was 12.4 GB of an l4x1's 30 GB before splatfacto
allocated anything of its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prepare_zerogpu_training import (  # noqa: E402
    CACHE_SHARE_LIMIT,
    FLAVOR_RAM_GB,
    check_host_memory,
    image_cache_gb,
)

HD = (1920, 1080)


def test_the_cache_size_is_frames_times_pixels_times_three():
    assert image_cache_gb(2000, *HD) == pytest.approx(12.44, abs=0.01)
    assert image_cache_gb(1600, *HD) == pytest.approx(9.95, abs=0.01)
    assert image_cache_gb(500, *HD) == pytest.approx(3.11, abs=0.01)


def test_the_runs_that_completed_are_allowed():
    """500 and 1600 frames both finished on l4x1; the gate must not block them."""
    for frames in (500, 1600):
        assert check_host_memory(frames, *HD, ["l4x1"], log=lambda *_: None) == ["l4x1"]


def test_the_run_that_was_oomkilled_is_refused():
    with pytest.raises(SystemExit) as excinfo:
        check_host_memory(2000, *HD, ["l4x1"], log=lambda *_: None)
    message = str(excinfo.value)
    assert "12.4 GB" in message and "l4x1" in message
    # The refusal has to say what to do instead, or it just blocks the user.
    assert "--subsample" in message or "--flavor" in message
    assert "--allow-memory-risk" in message


def test_a_roomier_flavour_accepts_the_same_run():
    """a10g-large has 46 GB, so 12.4 GB is 27% and inside the limit."""
    assert check_host_memory(2000, *HD, ["a10g-large"], log=lambda *_: None) == ["a10g-large"]


def test_an_unsafe_fallback_is_stripped_from_the_chain():
    """The runner falls back when the head will not schedule, so a safe head
    with an unsafe tail is the unsafe configuration on a delay. This is the
    chain that was actually queued on 18 Sep before it was caught."""
    kept = check_host_memory(2000, *HD, ["a10g-large", "l4x1"], log=lambda *_: None)
    assert kept == ["a10g-large"]


def test_a_chain_that_fits_nowhere_is_refused():
    with pytest.raises(SystemExit, match="fits none of"):
        check_host_memory(2000, *HD, ["l4x1", "a10g-small"], log=lambda *_: None)


def test_the_override_is_honoured_and_announced():
    said = []
    kept = check_host_memory(2000, *HD, ["l4x1"], allow_over=True, log=said.append)
    assert kept == ["l4x1"]
    assert any("OVER" in line for line in said)


def test_an_unknown_flavour_warns_rather_than_refusing():
    said = []
    kept = check_host_memory(9000, *HD, ["some-new-flavor"], log=said.append)
    assert kept == ["some-new-flavor"]
    assert any("unknown RAM" in line for line in said)


def test_the_limit_sits_between_what_survived_and_what_died():
    ram = FLAVOR_RAM_GB["l4x1"]
    assert image_cache_gb(1600, *HD) / ram <= CACHE_SHARE_LIMIT
    assert image_cache_gb(2000, *HD) / ram > CACHE_SHARE_LIMIT
