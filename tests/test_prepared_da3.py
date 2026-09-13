import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from tools import run_prepared_da3 as runner


ARGS = ["--stream-sfm", "--da3-model", "depth-anything/DA3-LARGE-1.1",
        "--stream-resolution", "504x280", "--stream-chunk-size", "60", "--stream-overlap", "30"]


def test_prepared_runner_rejects_training_or_changed_budget():
    runner.validate_args(ARGS)
    runner.validate_args(ARGS + ["--stream-loop-closure"])
    with pytest.raises(ValueError):
        runner.validate_args(["--train-only"])
    with pytest.raises(ValueError):
        runner.validate_args(ARGS + ["--train-args", "[]"])


def test_prepared_runner_retains_failure_without_time_limit(tmp_path, monkeypatch):
    work = tmp_path / "work"; work.mkdir()
    output = tmp_path / "out"
    monkeypatch.setenv("VW_OUT", str(output))
    monkeypatch.setattr(runner.sys, "argv", ["runner", *ARGS])
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **kwargs: str(work))
    class Process:
        pid = 123
        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"]
            self.stdout = ["test diagnostic\n"]
            (Path(kwargs["env"]["VW_OUT"]) / "partial.bin").write_bytes(b"retained")
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self, timeout=None):
            assert timeout is None
            return 1
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    assert runner.main() == 1
    with zipfile.ZipFile(output / "streaming_artifacts.zip") as archive:
        assert archive.read("partial.bin") == b"retained"
        assert archive.getinfo("partial.bin").compress_type == zipfile.ZIP_STORED
    assert json.loads((output / "run_result.json").read_text())["returncode"] != 0
