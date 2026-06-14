import json
import math

import numpy as np
import pytest

from vaultwares_studio.camera_paths import sample_path
from vaultwares_studio.walk_patterns import (
    SceneBounds,
    WALK_PATTERNS,
    available_patterns,
    build_pattern,
    crane_up,
    dolly_in,
    dolly_out,
    doorway_reveal,
    figure_8,
    orbit,
    retrace_steps,
)


BOUNDS = SceneBounds(center=(1.0, 0.5, -2.0), radius=3.0)


def _distances_to_center(entity):
    return [
        math.sqrt(
            (keyframe.position[0] - BOUNDS.center[0]) ** 2
            + (keyframe.position[2] - BOUNDS.center[2]) ** 2
        )
        for keyframe in entity.keyframes
    ]


def test_registry_lists_expected_patterns():
    expected = {"orbit", "dolly_in", "dolly_out", "crane_up", "figure_8", "doorway_reveal", "retrace_steps"}
    assert expected.issubset(set(available_patterns()))
    # No spiral — the user prefers retrace_steps over a spiral fly-through.
    assert "spiral" not in WALK_PATTERNS


def test_orbit_is_closed_loop_at_fixed_altitude():
    entity = orbit(BOUNDS, seconds=10.0, stops=9)
    assert entity.is_path
    assert entity.duration == pytest.approx(10.0)
    np.testing.assert_allclose(entity.keyframes[0].position, entity.keyframes[-1].position, atol=1e-6)
    altitudes = {round(keyframe.position[1], 6) for keyframe in entity.keyframes}
    assert len(altitudes) == 1


def test_orbit_direction_flips_signed_area():
    cw = orbit(BOUNDS, direction="cw")
    ccw = orbit(BOUNDS, direction="ccw")
    # Second keyframe sits on opposite side of the X-axis when direction flips.
    assert math.copysign(1.0, cw.keyframes[1].position[2] - BOUNDS.center[2]) == 1.0
    assert math.copysign(1.0, ccw.keyframes[1].position[2] - BOUNDS.center[2]) == -1.0


def test_dolly_in_moves_closer_to_center():
    entity = dolly_in(BOUNDS, seconds=4.0)
    distances = _distances_to_center(entity)
    assert distances[0] > distances[-1]


def test_dolly_out_inverts_dolly_in_defaults():
    entity = dolly_out(BOUNDS)
    distances = _distances_to_center(entity)
    assert distances[0] < distances[-1]
    assert entity.name == "Dolly Out"


def test_crane_up_rises_monotonically():
    entity = crane_up(BOUNDS, seconds=6.0)
    altitudes = [keyframe.position[1] for keyframe in entity.keyframes]
    assert all(later >= earlier - 1e-9 for earlier, later in zip(altitudes, altitudes[1:]))
    assert altitudes[-1] > altitudes[0]


def test_figure_8_crosses_origin_and_closes():
    entity = figure_8(BOUNDS, seconds=8.0, stops=17)
    # Lemniscate hits the centre at t = pi/2 and 3pi/2 (indices 4 and 12 of 17).
    for crossing_index in (4, 12):
        crossing = entity.keyframes[crossing_index]
        assert math.isclose(crossing.position[0], BOUNDS.center[0], abs_tol=1e-6)
        assert math.isclose(crossing.position[2], BOUNDS.center[2], abs_tol=1e-6)
    # Closed loop.
    np.testing.assert_allclose(entity.keyframes[0].position, entity.keyframes[-1].position, atol=1e-6)


def test_doorway_reveal_pulls_in_and_yaws_look_at():
    entity = doorway_reveal(BOUNDS)
    distances = _distances_to_center(entity)
    assert distances[0] > distances[-1]
    # Look-at drifts opposite the camera, so first and last targets differ.
    assert entity.keyframes[0].look_at != entity.keyframes[-1].look_at


def test_retrace_steps_replays_transforms_json(tmp_path):
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "fps": 6,
                "frames": [
                    {"file_path": "frame_000.jpg", "transform_matrix": _identity_at([0.0, 0.5, 4.0]).tolist()},
                    {"file_path": "frame_001.jpg", "transform_matrix": _identity_at([0.0, 0.5, 2.0]).tolist()},
                    {"file_path": "frame_002.jpg", "transform_matrix": _identity_at([0.0, 0.5, 0.0]).tolist()},
                ],
            }
        ),
        encoding="utf-8",
    )
    entity = retrace_steps(BOUNDS, transforms_json=transforms)
    assert entity.is_path
    assert [keyframe.position[2] for keyframe in entity.keyframes] == [4.0, 2.0, 0.0]
    # look_at uses each frame's forward (-Z column = +Z forward after negation).
    assert entity.keyframes[0].look_at[2] < entity.keyframes[0].position[2]


def test_retrace_steps_normalises_total_duration(tmp_path):
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "fps": 30,
                "frames": [
                    {"file_path": "f0", "transform_matrix": _identity_at([0, 0, 0]).tolist()},
                    {"file_path": "f1", "transform_matrix": _identity_at([1, 0, 0]).tolist()},
                    {"file_path": "f2", "transform_matrix": _identity_at([2, 0, 0]).tolist()},
                ],
            }
        ),
        encoding="utf-8",
    )
    entity = retrace_steps(BOUNDS, transforms_json=transforms, seconds=6.0)
    assert entity.duration == pytest.approx(6.0)


def test_retrace_steps_rejects_too_few_frames(tmp_path):
    transforms = tmp_path / "transforms.json"
    transforms.write_text(json.dumps({"frames": [{"file_path": "f", "transform_matrix": np.eye(4).tolist()}]}))
    with pytest.raises(ValueError):
        retrace_steps(BOUNDS, transforms_json=transforms)


def test_build_pattern_routes_through_registry():
    entity = build_pattern("orbit", BOUNDS, seconds=4.0, stops=5)
    assert entity.name == "Orbit"
    assert entity.duration == pytest.approx(4.0)
    with pytest.raises(KeyError):
        build_pattern("spiral", BOUNDS)


def test_patterns_sample_to_smooth_paths():
    for name in ("orbit", "dolly_in", "crane_up", "figure_8", "doorway_reveal"):
        entity = build_pattern(name, BOUNDS)
        frames = sample_path(entity, fps=30)
        assert len(frames) >= 2
        positions = np.array([frame[0] for frame in frames])
        # Catmull-Rom should not blow up: every sample sits within a generous
        # multiple of the scene radius from the centroid.
        center = np.asarray(BOUNDS.center)
        assert np.linalg.norm(positions - center, axis=1).max() < BOUNDS.radius * 6


def _identity_at(position):
    matrix = np.eye(4)
    matrix[:3, 3] = position
    return matrix
