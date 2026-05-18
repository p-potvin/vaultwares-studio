from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_VIDEO = REPO_ROOT / "test_input.mp4"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "test_outputs" / "smoke_scene.usda"


@dataclass(frozen=True)
class SmokeStageResult:
    output_path: Path
    default_prim: str
    reconstruction_points: int


def build_smoke_stage(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    source_video: Path | str | None = DEFAULT_SOURCE_VIDEO,
) -> SmokeStageResult:
    """Create a tiny but valid USD stage for smoke testing."""
    resolved_output = Path(output_path).resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    if resolved_output.exists():
        resolved_output.unlink()

    stage = Usd.Stage.CreateNew(str(resolved_output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    ground = UsdGeom.Cube.Define(stage, "/World/Environment/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3f(20.0, 0.1, 20.0))
    ground.AddTranslateOp().Set(Gf.Vec3f(0.0, -0.05, 0.0))

    light = UsdLux.DistantLight.Define(stage, "/World/Environment/KeyLight")
    light.CreateIntensityAttr(500.0)
    light.CreateAngleAttr(0.53)

    digital_twin = UsdGeom.Xform.Define(stage, "/World/DigitalTwin")
    reconstruction = UsdGeom.Points.Define(stage, "/World/DigitalTwin/Reconstruction")
    points = [
        Gf.Vec3f(-1.0, 0.0, 0.0),
        Gf.Vec3f(0.0, 0.8, 0.2),
        Gf.Vec3f(1.0, 0.0, 0.0),
        Gf.Vec3f(0.0, 0.2, 1.0),
    ]
    reconstruction.GetPointsAttr().Set(points)
    reconstruction.GetWidthsAttr().Set([0.08] * len(points))
    reconstruction.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
        [(0.95, 0.62, 0.20)]
    )

    if source_video is not None:
        digital_twin.GetPrim().CreateAttribute(
            "sourceVideo", Sdf.ValueTypeNames.String, custom=True
        ).Set(str(Path(source_video)))

    stage.GetRootLayer().Save()

    return SmokeStageResult(
        output_path=resolved_output,
        default_prim=stage.GetDefaultPrim().GetPath().pathString,
        reconstruction_points=len(points),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the vaultwares-studio smoke-test USD artifact."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the generated .usda file.",
    )
    parser.add_argument(
        "--source-video",
        default=str(DEFAULT_SOURCE_VIDEO),
        help="Optional source video path recorded into the USD metadata.",
    )
    args = parser.parse_args()

    result = build_smoke_stage(
        output_path=args.output,
        source_video=args.source_video if args.source_video else None,
    )
    print(f"Generated USD artifact: {result.output_path}")
    print(f"Default prim: {result.default_prim}")
    print(f"Reconstruction points: {result.reconstruction_points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
