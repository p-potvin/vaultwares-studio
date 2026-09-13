import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vaultwares_studio.runners import HfJobsStageRunner, StageContext, CancelToken


def test_downloader_preserves_nested_duplicate_names(tmp_path):
    prefix = "jobs/test/reconstruction_sfm"
    files = {f"{prefix}/out/summary.json": b"root",
             f"{prefix}/out/streaming/summary.json": b"nested",
             f"{prefix}/out/streaming/depth/a.npy": b"depth",
             f"{prefix}/out/streaming/confidence/a.npy": b"confidence"}
    def download(**kwargs):
        path = Path(kwargs["local_dir"]) / "download"
        path.write_bytes(files[kwargs["filename"]])
        return str(path)
    api = SimpleNamespace(list_repo_files=lambda *a, **k: list(files), hf_hub_download=download)
    expected = tmp_path / "summary.json"
    ctx = StageContext(job_dir=tmp_path, job_id="test", stage_key="reconstruction_sfm",
                       params={}, inputs=[], expected_outputs=[expected], log=lambda m: None, cancel=CancelToken())
    outputs = HfJobsStageRunner()._download_outputs(api, "test/repo", prefix, ctx)
    assert len(outputs) == 4
    assert expected.read_bytes() == b"root"
    root = tmp_path / "reconstruction_sfm/remote_out/streaming"
    assert (root / "summary.json").read_bytes() == b"nested"
    assert (root / "depth/a.npy").read_bytes() == b"depth"
    assert (root / "confidence/a.npy").read_bytes() == b"confidence"


@pytest.mark.parametrize("loop_enabled", [False, True])
def test_streaming_preserves_partial_outputs_on_failure(tmp_path, monkeypatch, loop_enabled):
    import sys
    import yaml
    import huggingface_hub
    from PIL import Image
    import vaultwares_studio.streaming_convert as converter
    path = Path(__file__).resolve().parents[1] / "docker/worker/da3_entrypoint.py"
    spec = importlib.util.spec_from_file_location("retention_da3", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "STREAMING_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "streaming_convert", converter)
    source = tmp_path / "input.jpg"; Image.new("RGB", (192, 108)).save(source)
    weights = tmp_path / "cached"; weights.write_bytes(b"test")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a: str(weights))
    staged_salad = []
    def stage_salad(folder):
        staged_salad.append(folder)
        (folder / "dino_salad.ckpt").write_bytes(b"test checkpoint")
    monkeypatch.setattr(module, "stage_salad_checkpoint", stage_salad)
    def fake_run(command, cwd):
        output = Path(command[command.index("--output_dir") + 1])
        config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
        assert config["Model"]["save_depth_conf_result"] is True
        assert config["Model"]["save_debug_info"] is True
        assert config["Model"]["delete_temp_files"] is True
        assert config["Model"]["loop_enable"] is loop_enabled
        assert Path(config["Weights"]["SALAD"]).exists() is loop_enabled
        (output / "partial-depth.npz").write_bytes(b"partial")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(module, "run_da3_streaming", fake_run)
    args = SimpleNamespace(stream_resolution="504x280", stream_chunk_size=80, stream_overlap=40,
                           stream_loop_closure=loop_enabled, da3_model="test/model", max_sfm_frames=80)
    work = tmp_path / "work"; work.mkdir()
    out = tmp_path / "out"; out.mkdir()
    result = module.run_stream_sfm(args, [source], work, tmp_path / "processed", out, {})
    assert result == 1
    assert (out / "streaming/partial-depth.npz").read_bytes() == b"partial"
    assert (out / "streaming/input_frames.json").exists()
    assert (out / "streaming/config.yaml").exists()
    assert not list(out.rglob("*.safetensors"))
    assert len(staged_salad) == int(loop_enabled)
