import sys
from pathlib import Path

SPACE = Path(__file__).resolve().parents[1] / "spaces" / "da3-zerogpu"
sys.path.insert(0, str(SPACE))

from core import DEFAULT_KEEP_FRAMES, MAX_KEEP_FRAMES, PRESETS, candidate_fps, get_preset, selected_frame_indices, validate_capture


def test_high_quality_preset_matches_48gb_experiment_contract():
    preset = get_preset("high")
    assert (preset.width, preset.height, preset.chunk_size, preset.overlap) == (672, 378, 90, 45)
    assert preset.tokens_per_window == 116_640
    assert preset.gpu_duration_seconds == 1800


def test_selected_frames_preserve_coverage_endpoints():
    indices = selected_frame_indices(1045, 500)
    assert len(indices) == 500
    assert indices[0] == 0 and indices[-1] == 1044
    assert indices == sorted(indices)


def test_capture_validation_rejects_short_or_invalid_input():
    assert validate_capture(149, 500, PRESETS["high"]) == []
    assert validate_capture(0, 500, PRESETS["high"])
    assert validate_capture(149, 89, PRESETS["high"])


def test_capture_validation_caps_frames_at_the_console_ceiling():
    assert DEFAULT_KEEP_FRAMES == 500
    assert validate_capture(348, 1000, PRESETS["high"]) == []
    assert validate_capture(348, MAX_KEEP_FRAMES, PRESETS["high"]) == []
    assert validate_capture(348, MAX_KEEP_FRAMES + 1, PRESETS["high"])
    assert validate_capture(348, 501, PRESETS["high"], max_frames=500)


def test_candidate_rate_follows_the_request():
    assert candidate_fps(134.28, 500) == 7          # the 13 Sep run, unchanged
    assert candidate_fps(347.5, 1000) == 6
    assert candidate_fps(482.0, 1600) == 7          # two clips back to back
    assert candidate_fps(30.0, 500) == 10
    assert candidate_fps(2000.0, 500) == 2
    assert MAX_KEEP_FRAMES == 2000


def test_selected_frames_keep_everything_when_candidates_fit():
    assert len(selected_frame_indices(1043, 1000)) == 1000
    assert selected_frame_indices(900, 1000) == list(range(900))


def test_space_keeps_gpu_work_bounded_and_cpu_archive_outside_decorator():
    source = (SPACE / "app.py").read_text(encoding="utf-8")
    assert '@spaces.GPU(duration=1800, size="large")' in source
    assert "def package(" in source
    assert "runner.close()" in source
    assert "ZIP_STORED" in source
    assert "resize((preset.width, preset.height)" in source


def test_space_runner_keeps_the_single_event_open_until_gradio_completes():
    source = (SPACE.parents[1] / "tools" / "run_zerogpu_da3.py").read_text(encoding="utf-8")
    assert 'f"{SPACE_URL}/gradio_api/call/run/{event_id}"' in source
    assert "current_event in {\"complete\", \"error\"}" in source
    assert "args.preset, args.loop_closure, args.frames, args.loop_similarity]" in source
    assert "for line in response.iter_lines()" in source
