"""Fuse a job's DA3-Streaming depth fields into reconstruction/mesh.ply (+ .usda).

Local, CPU, a few minutes. The mesh is placed in the same world as the splat
(trainer normalisation + gravity rotation), so camera staging can reference
it beside cloud.usda.

    .venv\\Scripts\\python.exe tools\\fuse_streaming_mesh.py --job zerogpu-... [--max-frames 250]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _job_dir(job_id: str) -> Path:
    for base in (ROOT / "data" / "jobs", Path("D:/vaultwares-studio-jobs/data/jobs")):
        if (base / job_id / "manifest.json").exists():
            return base / job_id
    raise SystemExit(f"No manifest.json found for {job_id}")


def fuse_job(job_dir: Path, *, max_frames: int | None = None, conf_coef: float = 0.75,
             voxel_size: float | None = None, depth_trunc_factor: float = 1.5, log=print) -> dict:
    from vaultwares_studio.camera_scene import scene_frame_transform
    from vaultwares_studio.depth_fusion import fuse_streaming_mesh, mesh_to_usd, write_fusion_report

    streaming = job_dir / "reconstruction" / "remote_out" / "streaming"
    if not (streaming / "results_output").is_dir():
        raise SystemExit(f"no streaming/results_output under {job_dir}; re-import the artifact")
    started = time.time()
    report = fuse_streaming_mesh(
        streaming, job_dir / "reconstruction" / "mesh.ply",
        max_frames=max_frames, conf_coef=conf_coef, voxel_size=voxel_size,
        depth_trunc_factor=depth_trunc_factor,
        scene_transform=scene_frame_transform(job_dir), log=log,
    )
    usd = mesh_to_usd(job_dir / "reconstruction" / "mesh.ply", job_dir / "reconstruction" / "mesh.usda",
                      source="da3-streaming tsdf")
    report["usd"] = str(usd)
    report["seconds"] = round(time.time() - started, 1)
    write_fusion_report(job_dir, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--conf-coef", type=float, default=0.75)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--depth-trunc-factor", type=float, default=1.5,
                        help="truncate depth at this multiple of the 90th percentile (1.5 = near field only)")
    args = parser.parse_args()
    report = fuse_job(_job_dir(args.job), max_frames=args.max_frames, conf_coef=args.conf_coef,
                      voxel_size=args.voxel_size, depth_trunc_factor=args.depth_trunc_factor)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
