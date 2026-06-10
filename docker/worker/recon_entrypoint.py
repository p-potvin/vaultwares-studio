"""Remote reconstruction stage: frames.zip -> trained gaussian splat.

Runs inside the vw-studio-worker image with VW_IN / VW_OUT set by
vw_stage.py. Pipeline:

    unzip frames.zip
    ns-process-data images (sequential matching — frames come from video)
    ns-train splatfacto    (preset args, unsupported flags auto-dropped)
    ns-export gaussian-splat

Outputs written to $VW_OUT:
    splat.ply     full-attribute 3DGS PLY
    summary.json  registered image count, durations, args actually used
    model.zip     training config + checkpoint (when --keep-checkpoint),
                  reused by remote ns-render walkthroughs in M2
    error.json    structured failure info (e.g. too_few_registered_images)
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

MIN_REGISTERED_IMAGES = 10


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"[recon] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False, **kwargs)


def filter_supported_flags(train_args: list[str]) -> list[str]:
    """Drop --flag value pairs that the installed ns-train doesn't know."""
    help_text = ""
    probe = run(["ns-train", "splatfacto", "--help"], capture_output=True, text=True)
    if probe.returncode == 0:
        help_text = probe.stdout + probe.stderr
    if not help_text:
        return train_args
    # Value drift, not just flag drift: newer nerfstudio removed the 'none'
    # choice for --vis; tensorboard is the headless-safe choice everywhere.
    if "--vis" in train_args:
        index = train_args.index("--vis") + 1
        import re

        window = re.search(r"--vis\b(.{0,300})", help_text, re.DOTALL)
        choices = window.group(1) if window else ""
        if index < len(train_args) and train_args[index] == "none" and "none" not in choices:
            print("[recon] --vis none unsupported in this nerfstudio; using tensorboard", flush=True)
            train_args[index] = "tensorboard"
    kept: list[str] = []
    index = 0
    while index < len(train_args):
        arg = train_args[index]
        if arg.startswith("--") and arg not in help_text:
            print(f"[recon] dropping unsupported flag: {arg}", flush=True)
            index += 2 if index + 1 < len(train_args) and not train_args[index + 1].startswith("--") else 1
            continue
        kept.append(arg)
        index += 1
    return kept


def colmap_db_stats(processed_dir: Path) -> dict:
    """Keypoint/match telemetry from COLMAP's database — pinpoints WHERE SfM starves."""
    import sqlite3

    db_path = next(processed_dir.rglob("database.db"), None)
    if db_path is None:
        return {}
    try:
        con = sqlite3.connect(db_path)
        kp_count, kp_avg = con.execute("SELECT COUNT(*), AVG(rows) FROM keypoints").fetchone()
        match_pairs, match_avg = con.execute(
            "SELECT COUNT(*), AVG(rows) FROM matches WHERE rows > 0"
        ).fetchone()
        verified_pairs, verified_avg = con.execute(
            "SELECT COUNT(*), AVG(rows) FROM two_view_geometries WHERE rows > 0"
        ).fetchone()
        con.close()
        return {
            "images_with_keypoints": kp_count,
            "avg_keypoints": round(kp_avg or 0, 1),
            "raw_match_pairs": match_pairs,
            "avg_raw_matches": round(match_avg or 0, 1),
            "verified_pairs": verified_pairs,
            "avg_verified_matches": round(verified_avg or 0, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return {"db_stats_error": str(exc)}


def count_registered_images(processed_dir: Path) -> int:
    transforms = processed_dir / "transforms.json"
    if not transforms.exists():
        return 0
    try:
        payload = json.loads(transforms.read_text(encoding="utf-8"))
        return len(payload.get("frames", []))
    except (OSError, json.JSONDecodeError):
        return 0


def fail(out_dir: Path, code: str, detail: str) -> int:
    print(f"[recon] FAILED: {code} — {detail}", flush=True)
    (out_dir / "error.json").write_text(
        json.dumps({"code": code, "detail": detail}, indent=2), encoding="utf-8"
    )
    return 1


def main() -> int:  # noqa: PLR0911, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--train-args", default="[]", help="JSON list of ns-train args")
    parser.add_argument("--keep-checkpoint", action="store_true")
    args = parser.parse_args()

    in_dir = Path(os.environ["VW_IN"])
    out_dir = Path(os.environ["VW_OUT"])
    work = Path("/tmp/recon")
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
    frame_count = len(list(images.rglob("*.png"))) + len(list(images.rglob("*.jpg")))
    print(f"[recon] {frame_count} frames extracted", flush=True)

    registered = 0
    matching_used = ""
    # Sequential first (fast, right for video). If too few register, escalate
    # to exhaustive matching once — more pair candidates rescue footage with
    # skips, blur, or exposure swings; costs nothing when sequential works.
    for matching in ("sequential", "exhaustive"):
        if processed.exists():
            shutil.rmtree(processed)
        processed.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        process = run([
            "ns-process-data", "images",
            "--data", str(images),
            "--output-dir", str(processed),
            "--matching-method", matching,
            "--num-downscales", "3",
            # CPU SIFT: COLMAP's GPU feature path fails silently in the L4
            # job container (3 videos x ~3 registered with thousands of
            # locally-verified inliers). CPU costs minutes, not correctness.
            "--no-gpu",
        ])
        timings[f"process_data_{matching}_s"] = round(time.monotonic() - started, 1)
        if process.returncode != 0:
            return fail(out_dir, "process_data_failed", f"ns-process-data exit {process.returncode}")
        registered = count_registered_images(processed)
        matching_used = matching
        stats = colmap_db_stats(processed)
        print(f"[recon] registered images ({matching}): {registered}/{frame_count}", flush=True)
        print(f"[recon] colmap db stats: {json.dumps(stats)}", flush=True)
        if registered >= MIN_REGISTERED_IMAGES:
            break

    if registered < MIN_REGISTERED_IMAGES:
        return fail(
            out_dir,
            "too_few_registered_images",
            f"COLMAP registered only {registered} images even with exhaustive matching "
            f"(need >= {MIN_REGISTERED_IMAGES}). Capture tips: WALK through the space "
            "(rotation-only pans give no parallax), move slowly, keep good lighting, "
            "avoid blur and large featureless/dark surfaces.",
        )

    train_args = filter_supported_flags(json.loads(args.train_args))
    started = time.monotonic()
    train = run([
        "ns-train", "splatfacto",
        "--data", str(processed),
        "--output-dir", str(train_out),
        *train_args,
        # Dataparser args go after the dataparser name in nerfstudio CLIs.
        "nerfstudio-data",
        "--downscale-factor", str(args.downscale),
    ])
    timings["train_s"] = round(time.monotonic() - started, 1)
    if train.returncode != 0:
        return fail(out_dir, "training_failed", f"ns-train exit {train.returncode}")

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
        # Whole train tree (config.yml references absolute /tmp/recon paths;
        # the render job recreates the same layout) + the processed-dataset
        # essentials the dataparser needs at eval time.
        shutil.make_archive(str(out_dir / "model"), "zip", root_dir=str(train_out))
        with zipfile.ZipFile(out_dir / "processed_min.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("transforms.json", "sparse_pc.ply"):
                candidate = processed / name
                if candidate.exists():
                    archive.write(candidate, name)
        print("[recon] checkpoint archived (model.zip + processed_min.zip)", flush=True)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "frames": frame_count,
                "registered_images": registered,
                "matching_method": matching_used,
                "train_args": train_args,
                "timings": timings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[recon] complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
