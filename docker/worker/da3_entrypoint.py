"""DA3 (Depth Anything 3) reconstruction entrypoint.

Replaces COLMAP/MASt3R for the SfM stage. Runs inside the vw-studio-worker
image with VW_IN / VW_OUT set by vw_stage.py.

Modes:
  --sfm-only       Run DA3 inference, produce processed_min.zip (transforms.json
                   + sparse_pc.ply) for splatfacto --train-only. [split Job A]
  --gs-only        Run DA3-GIANT with infer_gs=True, produce da3_gaussians.ply
                   directly. No splatfacto training. [da3-draft preset]
  --train-only     Skip DA3, consume processed_min.zip from VW_IN and run
                   splatfacto + ns-export. [split Job B, same as recon_entrypoint]
  (no flag)        Full pipeline: DA3 SfM + splatfacto training + export.

Inputs (VW_IN):
  frames.zip         Input images (jpg/png)

Outputs (VW_OUT):
  splat.ply          Full-attribute 3DGS PLY (from ns-export or DA3 direct)
  summary.json       Job metadata
  model.zip          Training checkpoint (when --keep-checkpoint)
  processed_min.zip  transforms.json + sparse_pc.ply (SfM output for split jobs)
  da3_gaussians.ply  Direct 3DGS output (when --gs-only)
  error.json         Structured failure info
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[da3-recon] {msg}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log(f"$ {' '.join(cmd)}")
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    result = subprocess.run(cmd, check=False, **kwargs)
    if result.returncode != 0 and result.stderr:
        log(f"stderr: {result.stderr[-2000:]}")
    return result


def run_streaming(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command with live stdout/stderr streaming to log."""
    log(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in result.stdout.splitlines():
        log(f"  {line}")
    if result.returncode != 0:
        log(f"  exit code: {result.returncode}")
    return result


def fail(out_dir: Path, code: str, detail: str) -> int:
    log(f"FAILED: {code} — {detail}")
    (out_dir / "error.json").write_text(
        json.dumps({"code": code, "detail": detail}, indent=2), encoding="utf-8"
    )
    return 1


def load_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)


def da3_inference(
    images: list[Path],
    model_id: str,
    infer_gs: bool,
    export_dir: Path | None = None,
    export_format: str = "mini_npz",
) -> dict:
    """Run DA3 inference and return the prediction outputs.

    Returns a dict with keys: depth, extrinsics, intrinsics, conf,
    processed_images, and optionally gaussians.
    """
    import torch
    from depth_anything_3.api import DepthAnything3

    # Reduce fragmentation on L4 (22GB) — DA3's multi-view attention is memory-hungry.
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading DA3 model: {model_id} on {device}")
    model = DepthAnything3.from_pretrained(model_id).to(device)

    image_paths = [str(p) for p in images]
    log(f"Running DA3 inference on {len(image_paths)} images")

    inference_kwargs: dict = {}
    if infer_gs:
        inference_kwargs["infer_gs"] = True
    if export_dir is not None:
        inference_kwargs["export_dir"] = str(export_dir)
        inference_kwargs["export_format"] = export_format

    prediction = model.inference(image=image_paths, **inference_kwargs)

    result = {
        "depth": prediction.depth,
        "extrinsics": prediction.extrinsics,
        "intrinsics": prediction.intrinsics,
        "conf": getattr(prediction, "conf", None),
        "processed_images": prediction.processed_images,
    }
    if infer_gs and hasattr(prediction, "aux") and "gaussians" in prediction.aux:
        result["gaussians"] = prediction.aux["gaussians"]
        log("DA3 Gaussian prediction available")
    return result


def load_calibration(path: str | None, log=print) -> dict | None:
    """Read a lens calibration JSON, or None when none was given.

    Deliberately permissive about extra keys — the file is also a
    ``CameraCalibration`` dump from vaultwares_studio, which carries provenance
    fields the container has no use for.
    """
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    terms = {k: float(data.get(k, 0.0)) for k in ("k1", "k2", "p1", "p2")}
    if not any(terms.values()):
        log(f"Calibration {path} has no non-zero distortion terms; ignoring it.")
        return None
    log(f"Loaded lens calibration from {path}: {terms}")
    return data


def da3_to_transforms(
    prediction: dict,
    images: list[Path],
    images_dir: Path,
    output_dir: Path,
    calibration: dict | None = None,
) -> Path:
    """Convert DA3 output to nerfstudio transforms.json format.

    DA3 gives us:
      - extrinsics: (N, 3, 4) world-to-camera in OpenCV convention
      - intrinsics: (N, 3, 3) camera intrinsics, estimated independently per frame

    Nerfstudio transforms.json expects:
      - per-frame transform_matrix: 4x4 camera-to-world in OpenGL/Blender convention
      - fl_x, fl_y, cx, cy, w, h — top level for a shared camera, which is what
        we write, or per frame to override it, which we deliberately do not
      - file_path relative to data root

    ``calibration`` optionally carries k1/k2/p1/p2 for the lens. DA3 does not
    estimate distortion, and without these keys nerfstudio's undistortion never
    runs.
    """
    exts = prediction["extrinsics"]  # (N, 3, 4) w2c
    ixts = prediction["intrinsics"]  # (N, 3, 3)
    depths = prediction["depth"]     # (N, H, W)
    processed_imgs = prediction["processed_images"]  # (N, H, W, 3)

    n_frames = len(images)
    frames = []
    per_frame: list[tuple[float, float, float, float]] = []

    # DA3 resizes images internally; we need to scale intrinsics back to
    # the original image resolution so nerfstudio can load the full-res files.
    from PIL import Image as PILImage

    for i in range(n_frames):
        # Convert w2c (3x4) to c2w (4x4)
        w2c = np.eye(4)
        w2c[:3, :] = exts[i]
        c2w = np.linalg.inv(w2c)

        # Convert OpenCV (x-right, y-down, z-forward) to OpenGL (x-right, y-up, z-backward)
        # by flipping Y and Z axes of the camera coordinate system
        opengl_to_cv = np.diag([1, -1, -1, 1]).astype(np.float64)
        c2w_opengl = c2w @ opengl_to_cv

        # DA3's intrinsics are for the resized image — scale to original resolution
        da_h, da_w = depths[i].shape[:2]
        with PILImage.open(images[i]) as img:
            orig_w, orig_h = img.size
        scale_x = orig_w / da_w
        scale_y = orig_h / da_h

        ixt = ixts[i]  # (3, 3)
        fl_x = float(ixt[0, 0]) * scale_x
        fl_y = float(ixt[1, 1]) * scale_y
        cx = float(ixt[0, 2]) * scale_x
        cy = float(ixt[1, 2]) * scale_y

        per_frame.append((fl_x, fl_y, cx, cy))
        frames.append(
            {
                "file_path": f"images/{images[i].name}",
                "transform_matrix": c2w_opengl.tolist(),
            }
        )

    # ONE camera, not N. DA3 estimates intrinsics independently per frame, and
    # on backyard_134s_sunny.mp4 the focal ranged 881.7..915.1 px for a phone
    # whose lens never moved — a 3.75% spread. splatfacto takes those numbers as
    # ground truth, so the frames disagree about where a world point projects
    # and the gaussians blur until they satisfy none of them. Median because a
    # frame with no focal signal (mostly sky) should not move the consensus.
    # See vaultwares_studio/camera_calibration.py for the full measurement; this
    # file ships alone in the job container and cannot import it.
    per_frame_arr = np.asarray(per_frame, dtype=np.float64)
    fl_x, fl_y, cx, cy = (float(np.median(per_frame_arr[:, i])) for i in range(4))
    fl_spread = float(per_frame_arr[:, 0].max() - per_frame_arr[:, 0].min()) / fl_x
    log(
        f"Shared camera: fl_x={fl_x:.2f} fl_y={fl_y:.2f} cx={cx:.1f} cy={cy:.1f} "
        f"(per-frame focal spread {fl_spread * 100:.2f}% collapsed to the median)"
    )

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": int(orig_w),
        "h": int(orig_h),
        "camera_angle_x": float(2 * np.arctan2(cx, fl_x)),
        "frames": frames,
        "ply_file_path": "sparse_pc.ply",
    }

    # DA3 never estimates lens distortion. nerfstudio's full_images_datamanager
    # WILL undistort every image at load time, but only when these keys exist
    # and are non-zero — on a DA3 bundle that path has never run. The lens is a
    # property of the phone, not the capture, so it is solved once by COLMAP and
    # carried in via --calibration. Coefficients are in normalised image
    # coordinates and so are resolution-independent: never scale them with fl.
    if calibration:
        for key in ("k1", "k2", "p1", "p2"):
            if calibration.get(key):
                transforms[key] = float(calibration[key])
        log(f"Applied lens distortion from calibration: "
            f"{ {k: transforms[k] for k in ('k1', 'k2', 'p1', 'p2') if k in transforms} }")

    transforms_path = output_dir / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    log(f"Wrote transforms.json with {n_frames} frames to {transforms_path}")
    return transforms_path


def da3_to_sparse_pc(
    prediction: dict,
    output_dir: Path,
    max_points: int = 500_000,
) -> Path:
    """Fuse DA3 depth maps into a sparse point cloud (PLY).

    Back-projects each frame's depth map using the predicted intrinsics and
    extrinsics, then concatenates and downsamples to max_points.
    """
    depths = prediction["depth"]       # (N, H, W)
    exts = prediction["extrinsics"]    # (N, 3, 4) w2c
    ixts = prediction["intrinsics"]    # (N, 3, 3)
    confs = prediction.get("conf")     # (N, H, W) or None
    processed = prediction["processed_images"]  # (N, H, W, 3)

    all_points = []
    all_colors = []

    for i in range(len(depths)):
        depth = depths[i]  # (H, W)
        h, w = depth.shape
        ixt = ixts[i]

        # Create pixel coordinate grid
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        # Normalize to camera coordinates
        z = depth
        x = (xs - ixt[0, 2]) * z / ixt[0, 0]
        y = (ys - ixt[1, 2]) * z / ixt[1, 1]
        # Stack into (H*W, 3) camera coordinates
        pts_cam = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

        # Filter by confidence if available
        if confs is not None:
            conf = confs[i].flatten()
            # Keep top 60% confidence points
            thresh = np.percentile(conf, 40)
            mask = conf > thresh
            pts_cam = pts_cam[mask]
            colors = processed[i].reshape(-1, 3)[mask]
        else:
            colors = processed[i].reshape(-1, 3)

        # Filter invalid depth (zero, negative, or extreme)
        valid = pts_cam[:, 2] > 0.01
        pts_cam = pts_cam[valid]
        colors = colors[valid]

        # Transform to world coordinates: inv(w2c) * [x, y, z, 1]
        w2c = np.eye(4)
        w2c[:3, :] = exts[i]
        c2w = np.linalg.inv(w2c)
        pts_homog = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        pts_world = (c2w @ pts_homog.T).T[:, :3]

        all_points.append(pts_world)
        all_colors.append(colors)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Downsample if needed
    if len(all_points) > max_points:
        idx = np.random.choice(len(all_points), max_points, replace=False)
        all_points = all_points[idx]
        all_colors = all_colors[idx]

    log(f"Sparse point cloud: {len(all_points)} points")

    # Write PLY
    ply_path = output_dir / "sparse_pc.ply"
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(all_points, all_colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} ")
            r = int(np.clip(col[0], 0, 1) * 255) if col[0] <= 1.0 else int(col[0])
            g = int(np.clip(col[1], 0, 1) * 255) if col[1] <= 1.0 else int(col[1])
            b = int(np.clip(col[2], 0, 1) * 255) if col[2] <= 1.0 else int(col[2])
            f.write(f"{r} {g} {b}\n")

    log(f"Wrote sparse_pc.ply to {ply_path}")
    return ply_path


def export_da3_gs_ply(prediction: dict, output_path: Path) -> Path:
    """Export DA3's direct 3DGS prediction to a standard 3DGS PLY file.

    The DA3 Gaussian output is in the prediction.aux['gaussians'] dict,
    containing positions, rotations, scales, colors, and opacities.
    This function converts them to the standard 3DGS PLY format that
    our splat_io.py can read.
    """
    gaussians = prediction.get("gaussians")
    if gaussians is None:
        raise RuntimeError("No Gaussian data in DA3 prediction (infer_gs=True required)")

    # DA3's Gaussian output format may vary; we try common keys
    positions = gaussians.get("positions") or gaussians.get("means")
    if positions is None:
        raise RuntimeError(f"DA3 gaussians dict keys: {list(gaussians.keys())}")

    # Convert to numpy if needed
    positions = np.asarray(positions)
    n_gaussians = len(positions)
    log(f"DA3 produced {n_gaussians} gaussians")

    # Try to get other attributes
    scales = gaussians.get("scales") or gaussians.get("scale")
    rotations = gaussians.get("rotations") or gaussians.get("quats")
    colors = gaussians.get("colors") or gaussians.get("sh_dc") or gaussians.get("rgb")
    opacities = gaussians.get("opacities") or gaussians.get("opacity")

    # Defaults if missing
    if scales is None:
        scales = np.ones((n_gaussians, 3)) * 0.01
    else:
        scales = np.asarray(scales)

    if rotations is None:
        rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n_gaussians, 1))
    else:
        rotations = np.asarray(rotations)

    if colors is None:
        colors = np.ones((n_gaussians, 3)) * 128
    else:
        colors = np.asarray(colors)

    if opacities is None:
        opacities = np.ones(n_gaussians) * 0.9
    else:
        opacities = np.asarray(opacities)

    # Write 3DGS PLY
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format binary_little_endian 1.0\n")
        f.write(f"element vertex {n_gaussians}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        for i in range(3):
            f.write(f"property float f_dc_{i}\n")
        for i in range(3):
            f.write(f"property float f_scale_{i}\n")
        for i in range(4):
            f.write(f"property float f_rot_{i}\n")
        f.write("property float opacity\n")
        f.write("end_header\n")

        # SH DC coefficient: color * 0.28209479 (SH C0)
        SH_C0 = 0.28209479177387814
        import struct

        for i in range(n_gaussians):
            pos = positions[i]
            sc = np.exp(scales[i]) if np.any(scales[i] < 0) else scales[i]  # log-space → linear
            rot = rotations[i]
            col = colors[i]
            opa = opacities[i]

            # Sigmoid for opacity if raw logits
            if opa < 0 or opa > 1:
                opa = 1.0 / (1.0 + np.exp(-opa))

            f_dc = [float(col[0] / 255.0 - 0.5) / SH_C0,
                    float(col[1] / 255.0 - 0.5) / SH_C0,
                    float(col[2] / 255.0 - 0.5) / SH_C0]

            data = struct.pack(
                "3f3f3f3f4ff",
                float(pos[0]), float(pos[1]), float(pos[2]),
                0.0, 0.0, 0.0,  # normals (unused)
                f_dc[0], f_dc[1], f_dc[2],
                float(sc[0]), float(sc[1]), float(sc[2]),
                float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]),
                float(opa),
            )
            f.write(data)

    log(f"Wrote DA3 3DGS PLY ({n_gaussians} gaussians) to {output_path}")
    return output_path


def write_depth_maps(prediction: dict, images: list[Path], output_dir: Path) -> None:
    """Write per-frame depth (and confidence) maps as .npy files.

    Named after the source frame stem, not the inference index, so a map can
    always be traced back to the frame it came from. splatfacto does not
    consume these — see make_depth_bundle for why we keep them anyway.
    """
    depths_dir = output_dir / "depths"
    depths_dir.mkdir(parents=True, exist_ok=True)
    depths = prediction["depth"]
    for i, depth in enumerate(depths):
        stem = images[i].stem if i < len(images) else f"frame_{i:05d}"
        np.save(depths_dir / f"{stem}.npy", depth)

    confs = prediction.get("conf")
    if confs is not None:
        conf_dir = output_dir / "confidence"
        conf_dir.mkdir(parents=True, exist_ok=True)
        for i, conf in enumerate(confs):
            stem = images[i].stem if i < len(images) else f"frame_{i:05d}"
            np.save(conf_dir / f"{stem}.npy", conf)
        log(f"Wrote {len(confs)} confidence maps to {conf_dir}")

    log(f"Wrote {len(depths)} depth maps to {depths_dir}")


def make_depth_bundle(processed: Path, output_path: Path) -> Path | None:
    """Bundle depths/ + confidence/ into depths.zip.

    Kept deliberately separate from processed_min.zip: that archive is the
    SfM -> training handoff and stays small (~8 MB) because both legs pay to
    transfer it. The depth and confidence fields are a genuine DA3 output that
    nothing downstream consumes yet, so they ride in their own artifact rather
    than being discarded inside the container. The runner downloads everything
    under out/, so this lands in <stage>/remote_out/depths.zip automatically.
    """
    members: list[tuple[Path, str]] = []
    for folder in ("depths", "confidence"):
        source = processed / folder
        if not source.is_dir():
            continue
        for path in sorted(source.glob("*.npy")):
            members.append((path, f"{folder}/{path.name}"))
    if not members:
        log("No depth maps to bundle")
        return None
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in members:
            archive.write(path, arcname)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    log(f"Wrote {output_path} ({len(members)} maps, {size_mb:.1f} MB)")
    return output_path


STREAMING_DIR = Path("/opt/vw/da3_streaming")
SALAD_CHECKPOINT_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"


def stage_salad_checkpoint(weights: Path) -> Path:
    """Fetch the upstream place-recognition checkpoint once, with no retry loop."""
    import urllib.request

    target = weights / "dino_salad.ckpt"
    if target.exists() and target.stat().st_size:
        return target
    partial = target.with_suffix(".partial")
    try:
        with urllib.request.urlopen(SALAD_CHECKPOINT_URL, timeout=120) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if partial.stat().st_size == 0:
            raise ValueError("SALAD checkpoint download was empty.")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def run_stream_sfm(
    args, image_paths: list[Path], work: Path, processed: Path,
    out_dir: Path, timings: dict,
) -> int:
    """DA3-Streaming over every frame, chunked and SIM3-aligned.

    Unlike --sfm-only, which subsamples to --max-sfm-frames because DA3's
    multi-view attention is quadratic, streaming windows the sequence so all
    frames get posed. Output contract is identical (transforms.json +
    sparse_pc.ply), so the training leg is unchanged.
    """
    import yaml
    from PIL import Image

    sys.path.insert(0, "/opt/vw")
    from streaming_convert import write_processed_bundle
    try:
        from camera_calibration import CameraCalibration
    except ImportError:  # running against the repo rather than the flat image
        from vaultwares_studio.camera_calibration import CameraCalibration

    if not STREAMING_DIR.is_dir():
        return fail(out_dir, "missing_streaming", f"{STREAMING_DIR} not present in image")

    try:
        stream_w, stream_h = (int(v) for v in args.stream_resolution.lower().split("x"))
    except ValueError:
        return fail(out_dir, "bad_args",
                    f"--stream-resolution must be WxH, got {args.stream_resolution!r}")
    if stream_w % 14 or stream_h % 14:
        return fail(out_dir, "bad_args",
                    f"--stream-resolution {stream_w}x{stream_h} is not a multiple of 14 "
                    "(DA3 patch size)")

    with Image.open(image_paths[0]) as probe:
        orig_w, orig_h = probe.size
    log(f"Streaming {len(image_paths)} frames: {orig_w}x{orig_h} -> {stream_w}x{stream_h}")

    resized = work / "stream_frames"
    resized.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for path in image_paths:
        with Image.open(path) as img:
            img.convert("RGB").resize((stream_w, stream_h), Image.LANCZOS).save(
                resized / path.name, quality=95
            )
    timings["stream_resize_s"] = round(time.monotonic() - started, 1)

    # Streaming loads weights from local paths rather than a HF model id.
    weights = work / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    started = time.monotonic()
    for filename in ("config.json", "model.safetensors"):
        cached = hf_hub_download(args.da3_model, filename)
        shutil.copyfile(cached, weights / filename)
    timings["stream_weights_s"] = round(time.monotonic() - started, 1)
    log(f"Weights staged in {weights}")
    if args.stream_loop_closure:
        started = time.monotonic()
        stage_salad_checkpoint(weights)
        timings["salad_weights_s"] = round(time.monotonic() - started, 1)

    config = {
        "Weights": {
            "DA3": str(weights / "model.safetensors"),
            "DA3_CONFIG": str(weights / "config.json"),
            "SALAD": str(weights / "dino_salad.ckpt"),
        },
        "Model": {
            "chunk_size": args.stream_chunk_size,
            "overlap": args.stream_overlap,
            "loop_chunk_size": 20,
            "loop_enable": bool(args.stream_loop_closure),
            "useDBoW": False,
            # Retain stable outputs (poses, PCD, loop report and per-frame
            # depth/confidence) but delete DA3's raw aligned/unaligned chunk
            # scratch. Upstream estimates that scratch alone at ~5 GB for 300
            # frames; retaining it makes packaging dominate a short inference.
            "delete_temp_files": True,
            # triton is present in this image via torch 2.1.2; it is the fastest
            # of the four align backends.
            "align_lib": "triton",
            "align_method": "sim3",
            "scale_compute_method": "auto",
            "align_type": "dense",
            "ref_view_strategy": "saddle_balanced",
            "ref_view_strategy_loop": "saddle_balanced",
            "depth_threshold": 15.0,
            "save_depth_conf_result": True,
            "save_debug_info": True,
            "Sparse_Align": {"keypoint_select": "orb", "keypoint_num": 5000},
            # tol and lambda_init below are STRINGS on purpose — streaming does
            # eval() on both (sim3utils.py:1216+, sim3loop.py:227). That works
            # for its own YAML because PyYAML follows YAML 1.1, where an
            # exponent needs a dot: `1e-9` resolves to the *string* '1e-9'.
            # Dumping a Python float here writes `1.0e-09`, which round-trips
            # back as a real float, and eval(float) raises TypeError.
            "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"},
            "Pointcloud_Save": {"sample_ratio": 0.015, "conf_threshold_coef": 0.75},
        },
        "Loop": {
            "SALAD": {
                "image_size": [336, 336], "batch_size": 32,
                "similarity_threshold": 0.85, "top_k": 5,
                "use_nms": True, "nms_threshold": 25,
            },
            "SIM3_Optimizer": {
                # String for the same eval() reason as IRLS.tol above.
                "lang_version": "cpp", "max_iterations": 30, "lambda_init": "1e-6",
            },
        },
    }
    config_path = work / "stream_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    # Retain all outputs, including partial products on normal child failure.
    # Weights remain outside VW_OUT and are not re-uploaded as job data.
    stream_out = out_dir / "streaming"
    stream_out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, stream_out / "config.yaml")
    (stream_out / "input_frames.json").write_text(json.dumps({
        "frames": [p.name for p in image_paths],
        "source_size": [orig_w, orig_h],
        "stream_size": [stream_w, stream_h],
    }, indent=2), encoding="utf-8")
    started = time.monotonic()
    result = run_da3_streaming(
        [
            sys.executable, "da3_streaming.py",
            "--image_dir", str(resized),
            "--config", str(config_path),
            "--output_dir", str(stream_out),
        ],
        cwd=str(STREAMING_DIR),
    )
    timings["stream_inference_s"] = round(time.monotonic() - started, 1)
    if result.returncode != 0:
        return fail(out_dir, "streaming_failed",
                    f"da3_streaming.py exit {result.returncode}")

    # Streaming sorts its input directory, so sorted order is the contract.
    names = [p.name for p in sorted(resized.glob("*"))]
    try:
        calib = load_calibration(args.calibration, log)
        write_processed_bundle(
            stream_out, names, processed,
            stream_size=(stream_w, stream_h),
            original_size=(orig_w, orig_h),
            calibration=CameraCalibration(**{
                k: v for k, v in calib.items()
                if k in CameraCalibration.__dataclass_fields__
            }) if calib else None,
        )
    except Exception as exc:  # noqa: BLE001
        return fail(out_dir, "convert_failed", str(exc))

    # splatfacto reads the FULL-RES originals; transforms.json intrinsics were
    # scaled up to match. The downscaled copies exist only for streaming.
    target_images = processed / "images"
    target_images.mkdir(parents=True, exist_ok=True)
    for path in image_paths:
        shutil.copyfile(path, target_images / path.name)

    log(f"Streaming posed {len(names)} frames (vs {args.max_sfm_frames} for --sfm-only)")
    return 0


def run_da3_streaming(cmd: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Run da3_streaming.py, echoing output as it arrives.

    Named distinctly from the module-level ``run_streaming`` helper above: an
    earlier revision called this one ``run_streaming`` too, silently shadowing
    it and breaking every --train-only call site with a TypeError.

    Deliberately not subprocess.run(capture_output=True): streaming takes many
    minutes and buffering the whole thing means the HF job log shows nothing
    until it finishes, so a hang is indistinguishable from slow progress.
    """
    log(f"$ (cd {cwd} && {' '.join(cmd)})")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(f"  {line.rstrip()}")
    proc.wait()
    return subprocess.CompletedProcess(cmd, proc.returncode, "", "")


def find_checkpoint_dir(root: Path) -> Path | None:
    """The directory ``ns-train --load-dir`` wants: the one holding the .ckpt files.

    Not the run directory. nerfstudio's ``_load_checkpoint`` does a bare
    ``os.listdir(load_dir)`` and parses a step number out of *every* entry
    (``step-000019999.ckpt`` -> 19999), so handing it the run directory — which
    also contains ``config.yml`` and ``nerfstudio_models/`` — dies with
    ``invalid literal for int() with base 10: 'nerfstudio_model'``. Found the
    first time --refine-mode was ever run, Sun 13 Sep 2026.

    The newest checkpoint wins when an archive carries several runs.
    """
    candidates = {path.parent for path in root.rglob("*.ckpt")}
    if not candidates:
        return None
    return max(candidates, key=lambda p: max(c.stat().st_mtime for c in p.glob("*.ckpt")))


def make_processed_min(processed: Path, output_path: Path) -> None:
    """Bundle transforms.json + sparse_pc.ply into processed_min.zip."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("transforms.json", "sparse_pc.ply"):
            candidate = processed / name
            if candidate.exists():
                archive.write(candidate, name)
        # No colmap_database.db for DA3 — we don't use COLMAP at all.
        # nerfstudio's dataparser only needs transforms.json. Depth maps go in
        # depths.zip instead (make_depth_bundle) so this stays a small handoff.
    log(f"Wrote {output_path}")


def filter_supported_flags(train_args: list[str]) -> list[str]:
    """Drop --flag value pairs that the installed ns-train doesn't know."""
    help_text = ""
    probe = run(["ns-train", "splatfacto", "--help"], capture_output=True, text=True)
    if probe.returncode == 0:
        help_text = probe.stdout + probe.stderr
    if not help_text:
        return train_args
    if "--vis" in train_args:
        import re

        index = train_args.index("--vis") + 1
        window = re.search(r"--vis\b(.{0,300})", help_text, re.DOTALL)
        choices = window.group(1) if window else ""
        if index < len(train_args) and train_args[index] == "none" and "none" not in choices:
            log("--vis none unsupported in this nerfstudio; using tensorboard")
            train_args[index] = "tensorboard"
    kept: list[str] = []
    index = 0
    while index < len(train_args):
        arg = train_args[index]
        if arg.startswith("--") and arg not in help_text:
            log(f"dropping unsupported flag: {arg}")
            index += 2 if index + 1 < len(train_args) and not train_args[index + 1].startswith("--") else 1
            continue
        kept.append(arg)
        index += 1
    return kept


def finish_export(
    args, out_dir: Path, processed: Path, train_out: Path, export: Path,
    timings: dict, frame_count: int, matching_used: str,
) -> int:
    """Shared post-train flow: ns-export, copy outputs, archive bundles, summary."""
    configs = sorted(train_out.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not configs:
        return fail(out_dir, "no_config", "no config.yml produced by training")
    started = time.monotonic()
    exported = run([
        "ns-export", "gaussian-splat",
        "--load-config", str(configs[0]),
        "--output-dir", str(export),
    ])
    timings["export_s"] = round(time.monotonic() - started, 1)
    if exported.returncode != 0:
        return fail(out_dir, "export_failed", f"ns-export exit {exported.returncode}")

    plys = sorted(export.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not plys:
        return fail(out_dir, "no_ply", "ns-export produced no PLY")
    shutil.copyfile(plys[0], out_dir / "splat.ply")

    if args.keep_checkpoint:
        shutil.make_archive(str(out_dir / "model"), "zip", root_dir=str(train_out))
        make_processed_min(processed, out_dir / "processed_min.zip")
        log("checkpoint archived (model.zip + processed_min.zip)")

    # Full mode ran DA3 in this same container, so the depth fields are here
    # to keep. In --train-only they were produced by Job A and already shipped.
    depth_bundle = make_depth_bundle(processed, out_dir / "depths.zip")

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "frames": frame_count,
                "matching_method": matching_used,
                "train_args": json.loads(args.train_args),
                "depth_maps": bool(depth_bundle),
                "timings": timings,
                "sfm_engine": "da3",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log("complete")
    return 0


def main() -> int:  # noqa: PLR0911, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--train-args", default="[]", help="JSON list of ns-train args")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument(
        "--da3-model", default="depth-anything/DA3-LARGE-1.1",
        help="HuggingFace model ID for DA3 (e.g. depth-anything/DA3-GIANT-1.1)",
    )
    parser.add_argument(
        "--gs-only", action="store_true",
        help="Run DA3 with infer_gs=True and output da3_gaussians.ply directly. No splatfacto.",
    )
    parser.add_argument(
        "--sfm-only", action="store_true",
        help="Run DA3 SfM only, produce processed_min.zip. No splatfacto training.",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Skip DA3. Consume processed_min.zip from VW_IN, run splatfacto + ns-export.",
    )
    parser.add_argument(
        "--refine-mode", action="store_true",
        help="Refine an existing splat: VW_IN must contain model.zip + processed_min.zip.",
    )
    parser.add_argument(
        "--merge-splat", action="store_true",
        help="Incremental merge: run DA3 direct 3DGS on new frames, then merge with base splat.ply from VW_IN.",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.05,
        help="Voxel edge length for deduplication during merge (scene units).",
    )
    parser.add_argument(
        "--dynamic-cull-radius", type=float, default=0.10,
        help="Cull base gaussians within this radius of new ones (0 = disable).",
    )
    parser.add_argument(
        "--icp-max-distance", type=float, default=0.5,
        help="ICP max correspondence distance for alignment (0 = skip alignment).",
    )
    parser.add_argument(
        "--calibration",
        help="Path to a lens calibration JSON (k1/k2/p1/p2, optionally fl_x/fl_y/cx/cy). "
             "DA3 never estimates distortion, and nerfstudio only undistorts when these "
             "keys are present and non-zero. Solve it once per phone with COLMAP.",
    )
    parser.add_argument(
        "--max-sfm-frames", type=int, default=80,
        help="Max frames passed to DA3 inference (subsampled evenly). Prevents OOM on 22GB L4.",
    )
    parser.add_argument(
        "--stream-sfm", action="store_true",
        help="Run DA3-Streaming over ALL frames (chunked + SIM3-aligned) instead of "
             "subsampling to --max-sfm-frames. Produces processed_min.zip like --sfm-only.",
    )
    parser.add_argument(
        "--stream-resolution", default="504x280",
        help="WxH to resize frames to before streaming. Must be multiples of 14 "
             "(patch size). 504x280 is 1.80 aspect (~16:9) at 720 patches/frame.",
    )
    parser.add_argument("--stream-chunk-size", type=int, default=80)
    parser.add_argument("--stream-overlap", type=int, default=40)
    parser.add_argument(
        "--stream-loop-closure", action="store_true",
        help="Enable DINO-SALAD loop closure. Needs dino_salad.ckpt; off by default.",
    )
    args = parser.parse_args()

    in_dir = Path(os.environ["VW_IN"])
    out_dir = Path(os.environ["VW_OUT"])
    work = Path("/tmp/da3_recon")
    images = work / "images"
    processed = work / "processed"
    train_out = work / "train"
    export = work / "export"
    for folder in (images, processed, train_out, export):
        folder.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    frames_zip = in_dir / "frames.zip"
    if not frames_zip.exists():
        return fail(out_dir, "missing_input", "frames.zip not found in stage inputs")
    with zipfile.ZipFile(frames_zip) as archive:
        archive.extractall(images)
    image_paths = load_images(images)
    frame_count = len(image_paths)
    log(f"{frame_count} frames extracted")

    # Subsample frames for DA3 inference — the multi-view attention scales
    # quadratically with frame count and OOMs on 22GB L4 above ~100 frames.
    # All frames are still passed to splatfacto training via processed/images/.
    sfm_frames = image_paths
    if not args.train_only and frame_count > args.max_sfm_frames:
        step = frame_count / args.max_sfm_frames
        indices = [int(i * step) for i in range(args.max_sfm_frames)]
        sfm_frames = [image_paths[i] for i in indices]
        log(f"Subsampled {frame_count} → {len(sfm_frames)} frames for DA3 SfM (max-sfm-frames={args.max_sfm_frames})")

    if args.gs_only and (args.sfm_only or args.train_only):
        return fail(out_dir, "bad_args", "--gs-only is mutually exclusive with --sfm-only and --train-only")
    if args.sfm_only and args.train_only:
        return fail(out_dir, "bad_args", "--sfm-only and --train-only are mutually exclusive")
    if args.merge_splat and not args.gs_only:
        return fail(out_dir, "bad_args", "--merge-splat requires --gs-only")
    if args.stream_sfm and (args.gs_only or args.train_only):
        return fail(out_dir, "bad_args",
                    "--stream-sfm is mutually exclusive with --gs-only and --train-only")

    # ---- --stream-sfm: DA3-Streaming over ALL frames ----
    if args.stream_sfm:
        started = time.monotonic()
        rc = run_stream_sfm(args, image_paths, work, processed, out_dir, timings)
        if rc != 0:
            return rc
        timings["stream_total_s"] = round(time.monotonic() - started, 1)
        make_processed_min(processed, out_dir / "processed_min.zip")
        (out_dir / "summary.json").write_text(
            json.dumps({
                "frames": frame_count,
                "frames_to_da3": frame_count,  # streaming poses every frame
                "sfm_engine": "da3-streaming",
                "da3_model": args.da3_model,
                "stream_resolution": args.stream_resolution,
                "stream_chunk_size": args.stream_chunk_size,
                "stream_overlap": args.stream_overlap,
                "loop_closure": bool(args.stream_loop_closure),
                "timings": timings,
            }, indent=2),
            encoding="utf-8",
        )
        log("DA3-Streaming SfM complete")
        return 0

    # ---- --gs-only: DA3 direct 3DGS, no splatfacto ----
    if args.gs_only:
        log(f"DA3 direct 3DGS mode (model: {args.da3_model})")
        started = time.monotonic()
        prediction = None
        try:
            prediction = da3_inference(
                sfm_frames,
                model_id=args.da3_model,
                infer_gs=True,
                export_dir=out_dir,
                export_format="gs_ply",
            )
        except Exception as exc:  # noqa: BLE001
            # DA3's exporter writes the PLY and then renders a preview video.
            # The video step calls moviepy with an fps we never supply and dies
            # on `TypeError: must be real number, not NoneType` — after the PLY
            # is already on disk. Losing a completed 4.5M-gaussian run over a
            # preview clip we don't want would be absurd, so only treat this as
            # fatal if nothing was actually written.
            if not list(out_dir.rglob("*.ply")):
                return fail(out_dir, "da3_inference_failed", str(exc))
            log(f"DA3 export raised after writing the PLY, continuing: {exc}")
        timings["da3_inference_s"] = round(time.monotonic() - started, 1)

        # Try to find the gs_ply exported by DA3's export pipeline
        gs_ply_candidates = sorted(out_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
        if gs_ply_candidates:
            shutil.copyfile(gs_ply_candidates[0], out_dir / "da3_gaussians.ply")
            shutil.copyfile(gs_ply_candidates[0], out_dir / "splat.ply")
            log(f"DA3 gs_ply exported to {out_dir / 'splat.ply'}")
        elif prediction is None:
            # Unreachable in practice — the guard above already failed out when
            # the exporter raised with no PLY written. Explicit so a future edit
            # to that guard can't silently reach export_da3_gs_ply(None).
            return fail(out_dir, "gs_export_failed", "DA3 export produced no PLY")
        else:
            # Fallback: export from prediction.aux['gaussians']
            try:
                export_da3_gs_ply(prediction, out_dir / "da3_gaussians.ply")
                shutil.copyfile(out_dir / "da3_gaussians.ply", out_dir / "splat.ply")
            except Exception as exc:
                return fail(out_dir, "gs_export_failed", str(exc))

        (out_dir / "summary.json").write_text(
            json.dumps({
                "frames": frame_count,
                "sfm_engine": "da3",
                "da3_model": args.da3_model,
                "direct_gs": True,
                "timings": timings,
            }, indent=2),
            encoding="utf-8",
        )
        # ---- --merge-splat: merge new DA3 3DGS with base splat ----
        if args.merge_splat:
            base_ply = in_dir / "splat.ply"
            if not base_ply.exists():
                return fail(out_dir, "missing_input",
                            "--merge-splat requires splat.ply (base splat) in $VW_IN")
            log(f"Merging new DA3 splat with base: {base_ply}")
            from gaussian_merge import MergeConfig, merge_splats
            from splat_io import read_gaussian_ply, write_gaussian_ply

            merge_config = MergeConfig(
                voxel_size=args.voxel_size,
                icp_max_distance=args.icp_max_distance,
                dynamic_cull_radius=args.dynamic_cull_radius,
                prefer_new=True,
            )
            merge_started = time.monotonic()
            base_splat = read_gaussian_ply(base_ply)
            new_splat = read_gaussian_ply(out_dir / "splat.ply")
            log(f"Base: {base_splat.count} gaussians | New: {new_splat.count} gaussians")
            merged = merge_splats(base_splat, new_splat, merge_config)
            timings["merge_s"] = round(time.monotonic() - merge_started, 1)
            log(f"Merged: {merged.count} gaussians (culled {base_splat.count + new_splat.count - merged.count})")
            write_gaussian_ply(merged, out_dir / "splat.ply")

            (out_dir / "summary.json").write_text(
                json.dumps({
                    "frames": frame_count,
                    "sfm_engine": "da3",
                    "da3_model": args.da3_model,
                    "direct_gs": True,
                    "incremental_merge": True,
                    "base_gaussians": base_splat.count,
                    "new_gaussians": new_splat.count,
                    "merged_gaussians": merged.count,
                    "merge_config": {
                        "voxel_size": args.voxel_size,
                        "dynamic_cull_radius": args.dynamic_cull_radius,
                        "icp_max_distance": args.icp_max_distance,
                    },
                    "timings": timings,
                }, indent=2),
                encoding="utf-8",
            )
            log("DA3 incremental merge complete")
            return 0

        log("DA3 direct 3DGS complete")
        return 0

    # ---- --train-only: skip DA3, consume processed_min.zip ----
    if args.train_only:
        processed_zip = in_dir / "processed_min.zip"
        if not processed_zip.exists():
            return fail(out_dir, "missing_input",
                        "--train-only requires processed_min.zip in $VW_IN")
        with zipfile.ZipFile(processed_zip) as archive:
            archive.extractall(processed)
        # Only move extracted frames into processed/images/ if the SfM zip
        # didn't already include them. DA3's processed_min.zip contains only
        # the subsampled SfM frames — moving all 500 would cause a mismatch
        # between transforms.json (80 frames) and images/ (500 files).
        target_images = processed / "images"
        if not target_images.exists() and images.exists():
            shutil.move(str(images), str(target_images))

        train_args = filter_supported_flags(json.loads(args.train_args))
        # Disable torch.compile/dynamo — nerfstudio's splatfacto uses @torch.compile
        # which triggers a broken inductor import chain when torch version is mismatched.
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        started = time.monotonic()
        train_cmd = [
            "ns-train", "splatfacto",
            "--data", str(processed),
            "--output-dir", str(train_out),
            *train_args,
        ]
        if args.refine_mode:
            bundle_model = in_dir / "model.zip"
            if not bundle_model.exists():
                return fail(out_dir, "missing_input", "--refine-mode requires model.zip in $VW_IN")
            base_train = Path("/tmp/da3_recon/base_train")
            base_train.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(bundle_model) as archive:
                archive.extractall(base_train)
            load_dir = find_checkpoint_dir(base_train)
            if load_dir is None:
                return fail(out_dir, "no_checkpoint",
                            f"no *.ckpt inside model.zip ({bundle_model})")
            train_cmd.extend(["--load-dir", str(load_dir)])
            log(f"Refine: resuming from {load_dir} "
                f"({', '.join(sorted(p.name for p in load_dir.glob('*.ckpt'))[-2:])})")
        result = run_streaming(train_cmd)
        timings["train_s"] = round(time.monotonic() - started, 1)
        if result.returncode != 0:
            return fail(out_dir, "train_failed", f"ns-train exit {result.returncode}")
        return finish_export(args, out_dir, processed, train_out, export, timings, frame_count, "da3")

    # ---- --sfm-only or full: run DA3 SfM first ----
    log(f"DA3 SfM mode (model: {args.da3_model})")
    started = time.monotonic()
    prediction = da3_inference(
        sfm_frames,
        model_id=args.da3_model,
        infer_gs=False,
    )
    timings["da3_inference_s"] = round(time.monotonic() - started, 1)

    # Convert DA3 output to nerfstudio format (use sfm_frames — prediction only has these)
    da3_to_transforms(prediction, sfm_frames, images, processed,
                      calibration=load_calibration(args.calibration, log))
    da3_to_sparse_pc(prediction, processed)
    # Depth + confidence fields are kept as their own artifact. They are NOT
    # referenced from transforms.json — pointing splatfacto at them is what
    # broke the first run (depth path mismatch), and splatfacto ignores them
    # regardless. Retained because they're a real DA3 output we may want for
    # occupancy grids / mesh extraction later.
    write_depth_maps(prediction, sfm_frames, processed)

    # Move ALL images into processed/images/ for nerfstudio dataparser
    # (splatfacto can use more frames than DA3 processed for training supervision)
    target_images = processed / "images"
    target_images.mkdir(parents=True, exist_ok=True)
    for img in image_paths:
        shutil.copyfile(img, target_images / img.name)

    if args.sfm_only:
        make_processed_min(processed, out_dir / "processed_min.zip")
        depth_bundle = make_depth_bundle(processed, out_dir / "depths.zip")
        (out_dir / "summary.json").write_text(
            json.dumps({
                "frames": frame_count,
                "frames_sfm": len(sfm_frames),
                "sfm_engine": "da3",
                "da3_model": args.da3_model,
                "depth_maps": bool(depth_bundle),
                "timings": timings,
            }, indent=2),
            encoding="utf-8",
        )
        log("DA3 SfM-only complete")
        return 0

    # Full mode: continue to splatfacto training
    train_args = filter_supported_flags(json.loads(args.train_args))
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    started = time.monotonic()
    train_cmd = [
        "ns-train", "splatfacto",
        "--data", str(processed),
        "--output-dir", str(train_out),
        *train_args,
    ]
    result = run_streaming(train_cmd)
    timings["train_s"] = round(time.monotonic() - started, 1)
    if result.returncode != 0:
        return fail(out_dir, "train_failed", f"ns-train exit {result.returncode}")
    return finish_export(args, out_dir, processed, train_out, export, timings, frame_count, "da3")


if __name__ == "__main__":
    sys.exit(main())
