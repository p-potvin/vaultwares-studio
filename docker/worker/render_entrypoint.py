"""Remote walkthrough render: trained checkpoint + camera path -> MP4.

Inputs in $VW_IN:
    model.zip          whole training tree from recon (config.yml + ckpt)
    processed_min.zip  transforms.json (+ sparse_pc.ply) for the dataparser
    camera_path.json   nerfstudio camera-path document (from camera_paths.py)

Recreates the exact /tmp/recon layout the checkpoint was trained under so
the absolute paths inside config.yml resolve, then runs ns-render.

Output in $VW_OUT: walkthrough.mp4
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def fail(out_dir: Path, code: str, detail: str) -> int:
    print(f"[render] FAILED: {code} — {detail}", flush=True)
    (out_dir / "error.json").write_text(
        json.dumps({"code": code, "detail": detail}, indent=2), encoding="utf-8"
    )
    return 1


def main() -> int:
    in_dir = Path(os.environ["VW_IN"])
    out_dir = Path(os.environ["VW_OUT"])
    train_dir = Path("/tmp/recon/train")
    processed = Path("/tmp/recon/processed")
    train_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    for name, target in (("model.zip", train_dir), ("processed_min.zip", processed)):
        archive_path = in_dir / name
        if not archive_path.exists():
            return fail(out_dir, "missing_input", f"{name} not found in stage inputs")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target)
    camera_path = in_dir / "camera_path.json"
    if not camera_path.exists():
        return fail(out_dir, "missing_input", "camera_path.json not found in stage inputs")

    configs = sorted(train_dir.rglob("config.yml"))
    if not configs:
        return fail(out_dir, "no_config", "no config.yml inside model.zip")

    output_path = out_dir / "walkthrough.mp4"
    cmd = [
        "ns-render", "camera-path",
        "--load-config", str(configs[0]),
        "--camera-path-filename", str(camera_path),
        "--output-path", str(output_path),
    ]
    print(f"[render] $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0 or not output_path.exists():
        return fail(out_dir, "render_failed", f"ns-render exit {result.returncode}")
    print(f"[render] complete: {output_path.stat().st_size // 1_000_000} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
