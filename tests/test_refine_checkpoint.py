"""The --refine-mode checkpoint lookup, pinned by the failure that found it."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _entrypoint():
    spec = importlib.util.spec_from_file_location(
        "da3_entrypoint_under_test", ROOT / "docker" / "worker" / "da3_entrypoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_dir(root: Path, stamp: str, steps: list[int]) -> Path:
    run = root / "processed" / "splatfacto" / stamp
    models = run / "nerfstudio_models"
    models.mkdir(parents=True)
    # The run directory holds the things that broke the old lookup.
    (run / "config.yml").write_text("_target: splatfacto\n")
    (run / "dataparser_transforms.json").write_text("{}")
    for step in steps:
        (models / f"step-{step:09d}.ckpt").write_bytes(b"x")
    return models


def test_finds_the_checkpoint_directory_not_the_run_directory(tmp_path):
    models = _run_dir(tmp_path, "2026-09-13_131416", [18999, 19999])
    found = _entrypoint().find_checkpoint_dir(tmp_path)
    assert found == models
    # This is the contract nerfstudio relies on: every entry parses as a step.
    assert all(int(p.stem.split("-")[1]) for p in found.glob("*.ckpt"))
    assert not any(p.suffix == ".yml" for p in found.iterdir())


def test_picks_the_newest_run_when_the_archive_has_several(tmp_path):
    old = _run_dir(tmp_path, "2026-09-01_000000", [4999])
    new = _run_dir(tmp_path, "2026-09-13_131416", [19999])
    import os
    import time

    os.utime(old / "step-000004999.ckpt", (time.time() - 9000, time.time() - 9000))
    assert _entrypoint().find_checkpoint_dir(tmp_path) == new


def test_returns_none_when_the_archive_carries_no_checkpoint(tmp_path):
    (tmp_path / "processed").mkdir()
    (tmp_path / "processed" / "config.yml").write_text("_target: splatfacto\n")
    assert _entrypoint().find_checkpoint_dir(tmp_path) is None
