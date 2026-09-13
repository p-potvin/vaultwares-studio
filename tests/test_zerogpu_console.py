import sys
from pathlib import Path

SPACE = Path(__file__).resolve().parents[1] / "spaces" / "da3-zerogpu"
sys.path.insert(0, str(SPACE))

from core import PRESETS, get_preset, selected_frame_indices, validate_capture


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
    assert "for line in response.iter_lines()" in source
