"""Focused Gradio console: one video -> bounded DA3 Streaming -> artifact ZIP."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
import site
from pathlib import Path

import spaces
import gradio as gr
import numpy as np
from core import PRESETS, get_preset, selected_frame_indices, validate_capture

APP_ROOT = Path(__file__).parent
WORK_ROOT = Path("/tmp/da3-console")
SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
SOURCE_ROOT = WORK_ROOT / "Depth-Anything-3"
MODEL_ID = "depth-anything/DA3-LARGE-1.1"
SALAD_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
JULY_REPO = "clopeux/vw-studio-artifacts"
JULY_PREFIX = "jobs/local-run-20260713-022745"


def configure_cuda_runtime() -> None:
    """Expose the matching pip-installed CUDA 12 toolkit to gsplat JIT."""
    library_dirs: list[str] = []
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        candidate = Path(root) / "nvidia" / "cuda_runtime" / "lib"
        if candidate.is_dir():
            library_dirs.append(str(candidate))
        toolkit = Path(root) / "nvidia" / "cuda_nvcc"
        if (toolkit / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = str(toolkit)
            os.environ["PATH"] = str(toolkit / "bin") + os.pathsep + os.environ.get("PATH", "")
    if library_dirs:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs + ([current] if current else []))


def cuda_training_env() -> dict[str, str]:
    """Return a subprocess environment pinned to Torch 2.8's CUDA 12.8."""
    env = dict(os.environ)
    candidates = [Path("/usr/local/cuda-12.8"), Path("/usr/local/cuda-12")]
    candidates += [Path(root) / "nvidia" / "cuda_nvcc" for root in site.getsitepackages() + [site.getusersitepackages()]]
    toolkit = next((path for path in candidates if (path / "bin" / "nvcc").exists()), None)
    if toolkit is None:
        for root in site.getsitepackages() + [site.getusersitepackages()]:
            matches = list((Path(root) / "nvidia").rglob("nvcc")) if (Path(root) / "nvidia").exists() else []
            if matches:
                toolkit = matches[0].parent.parent
                break
    if toolkit:
        env["CUDA_HOME"] = str(toolkit)
        env["CUDA_PATH"] = str(toolkit)
        env["PATH"] = str(toolkit / "bin") + os.pathsep + env.get("PATH", "")
        runtime = next((Path(root) / "nvidia" / "cuda_runtime" / "lib" for root in site.getsitepackages() + [site.getusersitepackages()]
                        if (Path(root) / "nvidia" / "cuda_runtime" / "lib").is_dir()), None)
        if runtime:
            env["LD_LIBRARY_PATH"] = str(runtime) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


configure_cuda_runtime()


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=True, timeout=timeout)


def ensure_streaming_source() -> Path:
    if (SOURCE_ROOT / "da3_streaming" / "da3_streaming.py").exists():
        return SOURCE_ROOT
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", "--recurse-submodules", "https://github.com/ByteDance-Seed/Depth-Anything-3.git", str(SOURCE_ROOT)], timeout=900)
    _run(["git", "fetch", "--depth", "1", "origin", SOURCE_REVISION], cwd=SOURCE_ROOT)
    _run(["git", "checkout", "--detach", SOURCE_REVISION], cwd=SOURCE_ROOT)
    if not (SOURCE_ROOT / "da3_streaming" / "loop_utils" / "salad").is_dir():
        raise RuntimeError("DA3 Streaming SALAD submodule was not available.")
    return SOURCE_ROOT


SOURCE = ensure_streaming_source()
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(SOURCE / "da3_streaming"))


def load_da3_model():
    """Model placement happens at module scope for ZeroGPU CUDA emulation."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from depth_anything_3.api import DepthAnything3

    # The first cold Space start fetches the public files; subsequent requests
    # reuse the Space cache and upload only their video.
    config_path = hf_hub_download(MODEL_ID, "config.json")
    weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = DepthAnything3(**config)
    model.load_state_dict(load_file(weights_path), strict=False)
    return model.eval().to("cuda"), Path(config_path), Path(weights_path)


MODEL, MODEL_CONFIG, MODEL_WEIGHTS = load_da3_model()


def _probe(video: Path) -> dict:
    result = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=width,height:stream_side_data=rotation", "-of", "json", str(video)])
    return json.loads(result.stdout)


def _duration(probe: dict) -> float:
    return float(probe["format"]["duration"])


def prepare_video(video_path: str, preset_key: str) -> tuple[Path, dict, list[str]]:
    preset = get_preset(preset_key)
    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError("Uploaded video is unavailable.")
    work = Path(tempfile.mkdtemp(prefix="da3-request-", dir=WORK_ROOT))
    candidates = work / "candidates"; candidates.mkdir()
    probe = _probe(source)
    duration = _duration(probe)
    fps = max(2, min(10, round(1000 / duration)))
    _run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", str(source), "-vf", f"fps={fps}", "-q:v", "2", str(candidates / "candidate_%05d.jpg")], timeout=900)
    frames = sorted(candidates.glob("*.jpg"))
    indices = selected_frame_indices(len(frames), 500)
    selected = [frames[i] for i in indices]
    errors = validate_capture(duration, len(selected), preset)
    if errors:
        raise ValueError(" ".join(errors))
    input_dir = work / "frames"; input_dir.mkdir()
    for index, frame in enumerate(selected):
        from PIL import Image
        with Image.open(frame) as image:
            image.convert("RGB").resize((preset.width, preset.height), Image.Resampling.LANCZOS).save(
                input_dir / f"frame_{index:05d}.jpg", quality=95
            )
    manifest = {"source_name": source.name, "probe": probe, "candidate_frames": len(frames), "selected_frames": len(selected), "fps": fps, "preset": preset.__dict__, "gpu_input_size": [preset.width, preset.height]}
    (work / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return work, manifest, [str(path) for path in sorted(input_dir.glob("*.jpg"))]


def _stage_salad(weights_dir: Path) -> Path:
    target = weights_dir / "dino_salad.ckpt"
    if not target.exists():
        with urllib.request.urlopen(SALAD_URL, timeout=180) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)
    return target


def _streamer(image_dir: Path, output_dir: Path, config: dict):
    """Instantiate upstream runner while reusing the module-level DA3 model."""
    from da3_streaming import DA3_Streaming
    from loop_utils.sim3loop import Sim3LoopOptimizer
    from loop_utils.loop_detector import LoopDetector
    import torch

    runner = DA3_Streaming.__new__(DA3_Streaming)
    runner.config = config; runner.chunk_size = config["Model"]["chunk_size"]; runner.overlap = config["Model"]["overlap"]
    runner.overlap_s = 0; runner.overlap_e = runner.overlap; runner.conf_threshold = 1.5; runner.seed = 42
    runner.device = "cuda"; runner.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    runner.img_dir = str(image_dir); runner.img_list = None; runner.output_dir = str(output_dir)
    runner.result_unaligned_dir = str(output_dir / "_tmp_results_unaligned"); runner.result_aligned_dir = str(output_dir / "_tmp_results_aligned")
    runner.result_loop_dir = str(output_dir / "_tmp_results_loop"); runner.result_output_dir = str(output_dir / "results_output"); runner.pcd_dir = str(output_dir / "pcd")
    for directory in [runner.result_unaligned_dir, runner.result_aligned_dir, runner.result_loop_dir, runner.pcd_dir]: Path(directory).mkdir(parents=True, exist_ok=True)
    runner.all_camera_poses = []; runner.all_camera_intrinsics = []; runner.delete_temp_files = True
    runner.model = MODEL; runner.skyseg_session = None; runner.chunk_indices = None; runner.loop_list = []; runner.loop_optimizer = Sim3LoopOptimizer(config)
    runner.sim3_list = []; runner.loop_sim3_list = []; runner.loop_predict_list = []; runner.loop_enable = config["Model"]["loop_enable"]
    if runner.loop_enable:
        runner.loop_detector = LoopDetector(image_dir=str(image_dir), output=str(output_dir / "loop_closures.txt"), config=config)
        runner.loop_detector.load_model()
    return runner


def _config(work: Path, preset_key: str, loop_closure: bool) -> dict:
    preset = get_preset(preset_key); weights = work / "weights"; weights.mkdir(exist_ok=True)
    salad = _stage_salad(weights) if loop_closure else weights / "dino_salad.ckpt"
    return {"Weights": {"DA3": str(MODEL_WEIGHTS), "DA3_CONFIG": str(MODEL_CONFIG), "SALAD": str(salad)},
            "Model": {"chunk_size": preset.chunk_size, "overlap": preset.overlap, "loop_chunk_size": 20, "loop_enable": loop_closure, "useDBoW": False, "delete_temp_files": True, "align_lib": "torch", "align_method": "sim3", "scale_compute_method": "auto", "align_type": "dense", "ref_view_strategy": "saddle_balanced", "ref_view_strategy_loop": "saddle_balanced", "depth_threshold": 15.0, "save_depth_conf_result": True, "save_debug_info": True, "Sparse_Align": {"keypoint_select": "orb", "keypoint_num": 5000}, "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"}, "Pointcloud_Save": {"sample_ratio": 0.015, "conf_threshold_coef": 0.75}},
            "Loop": {"SALAD": {"image_size": [336, 336], "batch_size": 32, "similarity_threshold": 0.85, "top_k": 5, "use_nms": True, "nms_threshold": 25}, "SIM3_Optimizer": {"lang_version": "python", "max_iterations": 30, "lambda_init": "1e-6"}}}


@spaces.GPU(duration=1800, size="large")
def run_gpu(work_dir: str, preset_key: str, loop_closure: bool) -> dict:
    work = Path(work_dir); output = work / "streaming"; output.mkdir()
    config = _config(work, preset_key, loop_closure)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    from loop_utils.sim3utils import merge_ply_files
    started = time.monotonic(); runner = _streamer(work / "frames", output, config); runner.run()
    merge_ply_files(str(output / "pcd"), str(output / "pcd" / "combined_pcd.ply")); runner.close()
    return {"gpu_seconds": round(time.monotonic() - started, 3), "output": str(output), "cuda": __import__("torch").cuda.get_device_name()}


def package(work: Path, gpu: dict) -> tuple[str, str, str]:
    output = Path(gpu["output"]); report = {"gpu": gpu, "files": []}
    archive_path = work / "da3_streaming_artifacts.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
        for item in sorted(output.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(work)); report["files"].append({"path": str(item.relative_to(work)), "bytes": item.stat().st_size})
        archive.write(work / "input_manifest.json", "input_manifest.json")
    report_path = work / "run_report.json"; report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(archive_path), str(report_path), json.dumps({"gpu_seconds": gpu["gpu_seconds"], "artifact_bytes": archive_path.stat().st_size, "files": len(report["files"])}, indent=2)


def prepare_training(base_job: str = "local-run-20260713-022745") -> tuple[str, str]:
    """CPU-only staging of the July successful split-job inputs."""
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TRAINING_TOKEN")
    if not token:
        raise RuntimeError("The Space is missing its private artifact access secret.")
    work = Path(tempfile.mkdtemp(prefix="da3-train-", dir=WORK_ROOT)); processed = work / "processed"; processed.mkdir()
    prefix = f"jobs/{base_job}/reconstruction_sfm"
    frames_zip = hf_hub_download(JULY_REPO, f"{prefix}/in/frames.zip", repo_type="dataset", token=token)
    processed_zip = hf_hub_download(JULY_REPO, f"{prefix}/out/processed_min.zip", repo_type="dataset", token=token)
    with zipfile.ZipFile(processed_zip) as archive: archive.extractall(processed)
    images = processed / "images"; images.mkdir()
    with zipfile.ZipFile(frames_zip) as archive: archive.extractall(images)
    (work / "training_manifest.json").write_text(json.dumps({"base_job": base_job, "repo": JULY_REPO, "prefix": prefix, "iterations": 15000, "train_args": JULY_TRAIN_ARGS}, indent=2), encoding="utf-8")
    return str(work), json.dumps({"status": "ready", "base_job": base_job, "frames": len(list(images.glob("*.jpg"))), "iterations": 15000, "gpu_duration_seconds": 1800}, indent=2)


# Current Nerfstudio removed `--vis none`; tensorboard is the closest
# non-viewer equivalent and keeps the July training contract otherwise intact.
JULY_TRAIN_ARGS = ["--max-num-iterations", "15000", "--vis", "tensorboard", "--viewer.quit-on-train-completion", "True", "--pipeline.datamanager.cache-images", "cpu", "--steps-per-save", "1000", "--pipeline.model.cull-alpha-thresh", "0.05"]


@spaces.GPU(duration=1800, size="large")
def train_gpu(work_dir: str) -> tuple[str, str]:
    work = Path(work_dir); train_out = work / "train"; export = work / "export"; train_out.mkdir(); export.mkdir()
    started = time.monotonic(); env = cuda_training_env(); env["TORCHDYNAMO_DISABLE"] = "1"
    toolkit_path = Path(env.get("CUDA_HOME", ""))
    nvcc_path = toolkit_path / "bin" / "nvcc"
    if not nvcc_path.exists():
        raise RuntimeError(f"CUDA 12.8 nvcc was not found at {nvcc_path}.")
    version = subprocess.run([str(nvcc_path), "--version"], env=env, capture_output=True, text=True, check=True).stdout
    if "release 12." not in version:
        raise RuntimeError(f"Unexpected CUDA compiler selected: {version[-500:]}")
    subprocess.run(["ns-train", "splatfacto", "--data", str(work / "processed"), "--output-dir", str(train_out), *JULY_TRAIN_ARGS], env=env, check=True, timeout=1800)
    config_files = sorted(train_out.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not config_files: raise RuntimeError("Splatfacto produced no config.yml.")
    subprocess.run(["ns-export", "gaussian-splat", "--load-config", str(config_files[0]), "--output-dir", str(export)], env=env, check=True, timeout=600)
    plys = sorted(export.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not plys: raise RuntimeError("ns-export produced no Gaussian PLY.")
    shutil.copyfile(plys[0], work / "splat.ply")
    return str(work), json.dumps({"status": "trained", "gpu_seconds": round(time.monotonic() - started, 3), "config": str(config_files[0]), "splat_bytes": (work / "splat.ply").stat().st_size}, indent=2)


def package_training(work_dir: str, gpu_summary: str) -> tuple[str, str, str]:
    work = Path(work_dir); archive_path = work / "training_artifacts.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
        for item in sorted(work.rglob("*")):
            if item.is_file() and item.name not in {archive_path.name} and not str(item).startswith(str(work / "train")):
                archive.write(item, item.relative_to(work))
        model_configs = sorted((work / "train").rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
        if model_configs:
            train_root = model_configs[0].parent
            for item in train_root.rglob("*"):
                if item.is_file() and "events" not in item.parts:
                    archive.write(item, Path("model") / item.relative_to(train_root))
    summary = {"gpu": json.loads(gpu_summary), "artifact_bytes": archive_path.stat().st_size, "cpu_archive": True, "train_args": JULY_TRAIN_ARGS}
    report = work / "training_report.json"; report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(archive_path), str(report), json.dumps(summary, indent=2)


TEXT = {"EN": {"title": "DA3 Reconstruction Console", "run": "Run", "diagnostics": "Diagnostics", "artifacts": "Artifacts", "about": "About", "upload": "Video", "preset": "Preset", "loop": "Enable loop closure", "validate": "Validate capture", "start": "Start GPU pass", "ready": "Ready to request GPU.", "mobile": "This experimental console requires a desktop browser."}, "QC": {"title": "Console de reconstruction DA3", "run": "Exécuter", "diagnostics": "Diagnostics", "artifacts": "Artefacts", "about": "À propos", "upload": "Vidéo", "preset": "Préréglage", "loop": "Activer la fermeture de boucle", "validate": "Valider la capture", "start": "Lancer le passage GPU", "ready": "Prêt à demander le GPU.", "mobile": "Cette console expérimentale nécessite un navigateur de bureau."}}


def run(video, preset_key, loop_closure):
    work, manifest, _ = prepare_video(video, preset_key)
    gpu = run_gpu(str(work), preset_key, loop_closure)
    return package(work, gpu)


def validate(video, preset_key):
    work, manifest, _ = prepare_video(video, preset_key)
    preset = get_preset(preset_key)
    return json.dumps({"status": "ready", "selected_frames": manifest["selected_frames"], "gpu_duration_seconds": preset.gpu_duration_seconds, "tokens_per_window": preset.tokens_per_window, "work_dir": str(work)}, indent=2)


with gr.Blocks(theme=gr.themes.Default(font=[gr.themes.GoogleFont("JetBrains Mono")]), title="DA3 Reconstruction Console") as demo:
    gr.Markdown("# DA3 Reconstruction Console\nExperimental markerless video pose/depth reconstruction. GPU is requested only after validation.")
    with gr.Tabs():
        with gr.Tab("Run"):
            video = gr.Video(label="Video", sources=["upload"])
            preset = gr.Radio(choices=[("Preview", "preview"), ("High quality (experimental)", "high")], value="high", label="Preset")
            loop = gr.Checkbox(label="Enable loop closure", value=True)
            validate_button = gr.Button("Validate capture", variant="secondary", icon="🔎")
            confirmation = gr.Code(label="Run summary", language="json")
            run_button = gr.Button("Start GPU pass", variant="primary", icon="▶")
            artifact = gr.File(label="Artifact bundle")
            report = gr.File(label="Run report")
            summary = gr.Code(label="Completion summary", language="json")
            validate_button.click(validate, [video, preset], confirmation)
            run_button.click(run, [video, preset, loop], [artifact, report, summary])
        with gr.Tab("Diagnostics"):
            gr.Markdown("The run summary reports selected frames, requested GPU duration, visual tokens per window, GPU timing, and output file count.")
        with gr.Tab("Artifacts"):
            gr.Markdown("Download the ZIP for poses, intrinsics, loop report, depth/confidence results, point cloud, configuration, and input manifest.")
        with gr.Tab("Training"):
            gr.Markdown("Train a Gaussian splat from the July successful DA3 baseline. CPU stages fetch and archive; only Splatfacto and export use the 48 GB ZeroGPU allocation.")
            base_job = gr.Textbox(value="local-run-20260713-022745", label="Base artifact job")
            prepare_button = gr.Button("Prepare July baseline", variant="secondary")
            train_ready = gr.Code(label="Training plan", language="json")
            train_work = gr.Textbox(visible=False)
            train_button = gr.Button("Start 15,000-iteration GPU training", variant="primary")
            train_gpu_summary = gr.Code(label="GPU training result", language="json")
            train_archive = gr.File(label="Training artifacts")
            train_report = gr.File(label="Training report")
            train_summary = gr.Code(label="Completion summary", language="json")
            prepare_button.click(prepare_training, [base_job], [train_work, train_ready])
            train_button.click(train_gpu, [train_work], [train_work, train_gpu_summary]).then(package_training, [train_work, train_gpu_summary], [train_archive, train_report, train_summary])
        with gr.Tab("About"):
            gr.Markdown("Desktop-only focused DA3 console. Training, USD export, 3D viewing, camera paths, mesh work, Cosmos, and history are intentionally excluded.")

demo.launch(allowed_paths=[str(WORK_ROOT)])
