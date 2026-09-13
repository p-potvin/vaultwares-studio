from pathlib import Path


def test_training_client_chains_cpu_gpu_cpu_without_video_upload():
    source = (Path(__file__).resolve().parents[1] / "tools/run_zerogpu_training.py").read_text(encoding="utf-8")
    assert 'call(client, "prepare_training"' in source
    assert 'call(client, "train_gpu"' in source
    assert 'call(client, "package_training"' in source
    assert "frames.zip" not in source


def test_july_training_contract_uses_supported_current_nerfstudio_vis_mode():
    source = (Path(__file__).resolve().parents[1] / "spaces/da3-zerogpu/app.py").read_text(encoding="utf-8")
    assert '"--vis", "tensorboard"' in source
    assert '"--vis", "none"' not in source


def test_training_space_exposes_cuda_runtime_for_gsplat():
    source = (Path(__file__).resolve().parents[1] / "spaces/da3-zerogpu/app.py").read_text(encoding="utf-8")
    requirements = (Path(__file__).resolve().parents[1] / "spaces/da3-zerogpu/requirements.txt").read_text(encoding="utf-8")
    assert "nvidia-cuda-runtime-cu12==12.8.90" in requirements
    assert "nvidia-cuda-nvcc-cu12==12.8.93" in requirements
    assert "configure_cuda_runtime()" in source
    assert "CUDA_HOME" in source and "CUDA_PATH" in source
    assert "toolkit_path / \"bin\" / \"nvcc\"" in source
