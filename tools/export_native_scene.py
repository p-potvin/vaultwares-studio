"""Export an existing reconstruction for visual review, without model inference.

The destination must be new. Original PLY/USD/checkpoints and job state are
left intact. The result is a portable scene.usda referencing cloud.usdc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vaultwares_studio.camera_scene import compose_scene, load_active_camera, prepare_retrace_transforms
from vaultwares_studio.splat_io import _native_gsplat_schema_available, read_gaussian_ply, splat_to_usd
from vaultwares_studio.walk_patterns import bounds_from_preview_ply, orbit, retrace_steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="Existing local job directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    job = args.job.resolve()
    if not _native_gsplat_schema_available():
        parser.error("The installed OpenUSD does not expose ParticleField3DGaussianSplat.")
    cloud = job / "reconstruction" / "cloud.ply"
    splat = read_gaussian_ply(cloud)
    bounds = bounds_from_preview_ply(job / "reconstruction" / "cloud_preview.ply")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    mode = splat_to_usd(splat, output / "cloud.usdc", source=cloud.name)
    cameras = [orbit(bounds, seconds=12)]
    notes = []
    try:
        transforms = prepare_retrace_transforms(job)
        if transforms:
            cameras.append(retrace_steps(bounds, transforms_json=transforms, seconds=30))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        notes.append(f"Retrace unavailable: {exc}")
    active = load_active_camera(job)
    if active:
        cameras.append(active)
    compose_scene(output / "scene.usda", output / "cloud.usdc", cameras)
    from pxr import Usd, UsdVol
    stage = Usd.Stage.Open(str(output / "scene.usda"))
    prim = stage.GetPrimAtPath("/World/DigitalTwin/GaussianSplats")
    if not prim.IsA(UsdVol.ParticleField3DGaussianSplat):
        raise RuntimeError("Exported scene did not resolve its native Gaussian asset.")
    report = {"usd_version": list(Usd.GetVersion()), "mode": mode, "gaussians": splat.count,
              "scene": "scene.usda", "asset": "cloud.usdc", "metric_scale_known": False,
              "cameras": [c.to_dict() for c in cameras], "notes": notes,
              "validation": "USD composition and schema checked; rendering requires visual review."}
    (output / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"scene": str(output / "scene.usda"), "gaussians": splat.count,
                      "asset_bytes": (output / "cloud.usdc").stat().st_size,
                      "cameras": [c.name for c in cameras], "notes": notes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
