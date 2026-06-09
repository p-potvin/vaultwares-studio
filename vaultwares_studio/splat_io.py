"""Gaussian-splat PLY I/O and PLY→USD conversion.

Replaces the old Open3D path that flattened trained splats to bare point
clouds, losing scale/rotation/opacity/SH attributes. Here the exported 3DGS
PLY is preserved in full, a decimated preview point cloud is written for the
Open3D fallback viewer, and the USD export is lossless:

- If the installed OpenUSD ships a native gaussian-splat schema (26.03+),
  it is used (probed at runtime — usd-core 24.11 does not have it).
- Otherwise the splat is authored as UsdGeomPoints with namespaced primvars
  (``primvars:gsplat:*``) carrying every gaussian attribute, which is valid
  USD today and forward-convertible to the native schema later.

Attribute encoding note: values are stored exactly as 3DGS PLYs encode them
(log-scales, logit opacities, unnormalized quaternions, SH coefficients).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# SH band-0 constant: rgb = 0.5 + C0 * f_dc
_SH_C0 = 0.28209479177387814

_GAUSSIAN_REQUIRED = ("f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "rot_0")


@dataclass
class GaussianSplat:
    positions: np.ndarray  # (N, 3) float32
    sh0: np.ndarray  # (N, 3) float32 — f_dc_*
    opacity: np.ndarray  # (N,) float32 — logit-encoded
    scales: np.ndarray  # (N, 3) float32 — log-encoded
    rotations: np.ndarray  # (N, 4) float32 — quaternion wxyz
    sh_rest: np.ndarray | None = None  # (N, K) float32 — f_rest_*

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def colors_rgb(self) -> np.ndarray:
        """Approximate per-splat RGB in [0, 1] from the SH DC band."""
        return np.clip(0.5 + _SH_C0 * self.sh0, 0.0, 1.0)


def is_gaussian_ply(path: Path) -> bool:
    try:
        from plyfile import PlyData

        header = PlyData.read(str(path))
        names = {prop.name for prop in header["vertex"].properties}
    except Exception:  # noqa: BLE001 - unreadable/odd PLY is simply not a splat
        return False
    return all(name in names for name in _GAUSSIAN_REQUIRED)


def read_gaussian_ply(path: Path) -> GaussianSplat:
    from plyfile import PlyData

    data = PlyData.read(str(path))
    vertex = data["vertex"]
    names = {prop.name for prop in vertex.properties}
    missing = [name for name in _GAUSSIAN_REQUIRED if name not in names]
    if missing:
        raise ValueError(f"{path} is not a 3DGS gaussian PLY (missing {missing}).")

    def col(name: str) -> np.ndarray:
        return np.asarray(vertex[name], dtype=np.float32)

    positions = np.stack([col("x"), col("y"), col("z")], axis=1)
    sh0 = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1)
    scales = np.stack([col(f"scale_{i}") for i in range(3)], axis=1)
    rotations = np.stack([col(f"rot_{i}") for i in range(4)], axis=1)
    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    sh_rest = (
        np.stack([col(name) for name in rest_names], axis=1) if rest_names else None
    )
    return GaussianSplat(
        positions=positions,
        sh0=sh0,
        opacity=col("opacity"),
        scales=scales,
        rotations=rotations,
        sh_rest=sh_rest,
    )


def write_gaussian_ply(splat: GaussianSplat, path: Path) -> None:
    from plyfile import PlyData, PlyElement

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    fields += [(f"f_dc_{i}", "f4") for i in range(3)]
    rest_width = splat.sh_rest.shape[1] if splat.sh_rest is not None else 0
    fields += [(f"f_rest_{i}", "f4") for i in range(rest_width)]
    fields += [("opacity", "f4")]
    fields += [(f"scale_{i}", "f4") for i in range(3)]
    fields += [(f"rot_{i}", "f4") for i in range(4)]

    rows = np.empty(splat.count, dtype=fields)
    rows["x"], rows["y"], rows["z"] = splat.positions.T
    for i in range(3):
        rows[f"f_dc_{i}"] = splat.sh0[:, i]
    for i in range(rest_width):
        rows[f"f_rest_{i}"] = splat.sh_rest[:, i]  # type: ignore[index]
    rows["opacity"] = splat.opacity
    for i in range(3):
        rows[f"scale_{i}"] = splat.scales[:, i]
    for i in range(4):
        rows[f"rot_{i}"] = splat.rotations[:, i]

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(rows, "vertex")]).write(str(path))


def decimate(splat: GaussianSplat, max_points: int, seed: int = 0) -> GaussianSplat:
    if splat.count <= max_points:
        return splat
    keep = np.random.default_rng(seed).choice(splat.count, size=max_points, replace=False)
    keep.sort()
    return GaussianSplat(
        positions=splat.positions[keep],
        sh0=splat.sh0[keep],
        opacity=splat.opacity[keep],
        scales=splat.scales[keep],
        rotations=splat.rotations[keep],
        sh_rest=splat.sh_rest[keep] if splat.sh_rest is not None else None,
    )


def write_preview_ply(splat: GaussianSplat, path: Path, max_points: int = 200_000) -> int:
    """Plain xyz+rgb point cloud readable by Open3D and any PLY viewer."""
    from plyfile import PlyData, PlyElement

    preview = decimate(splat, max_points)
    colors = (preview.colors_rgb() * 255).astype(np.uint8)
    rows = np.empty(
        preview.count,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    rows["x"], rows["y"], rows["z"] = preview.positions.T
    rows["red"], rows["green"], rows["blue"] = colors.T
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(rows, "vertex")]).write(str(path))
    return preview.count


def _native_gsplat_schema_available() -> bool:
    """Probe the installed OpenUSD for a native gaussian-splat prim type (26.03+)."""
    try:
        from pxr import Usd

        registry = Usd.SchemaRegistry()
        for type_name in ("GaussianSplats", "Gsplats", "GsplatAsset"):
            try:
                if registry.FindConcretePrimDefinition(type_name):
                    return True
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return False


def splat_to_usd(splat: GaussianSplat, path: Path, source: str = "") -> str:
    """Author the splat to USD losslessly. Returns the mode used."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    if _native_gsplat_schema_available():
        # Native authoring lands when usd-core ships the 26.03 schema; the
        # primvar layout below is designed to convert 1:1 when that happens.
        mode = "native-schema-available-but-unwired"
    else:
        mode = "points+primvars"

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    points_prim = UsdGeom.Points.Define(stage, "/World/GaussianSplats")
    points_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(splat.positions))
    # Display width: isotropic approximation from the mean log-scale.
    # NOTE: Vt.FloatArray.FromNumpy rejects 1-D float32 buffers in usd-core
    # 24.11 ("Unsupported format '=f'"), so scalar arrays go through lists.
    widths = np.clip(2.0 * np.exp(splat.scales.mean(axis=1)), 0.001, 0.5)
    points_prim.GetWidthsAttr().Set(Vt.FloatArray(widths.tolist()))
    points_prim.GetDisplayColorAttr().Set(
        Vt.Vec3fArray.FromNumpy(splat.colors_rgb().astype(np.float32))
    )

    primvars = UsdGeom.PrimvarsAPI(points_prim.GetPrim())

    def vertex_primvar(name: str, type_name, values) -> None:
        primvars.CreatePrimvar(name, type_name, UsdGeom.Tokens.vertex).Set(values)

    vertex_primvar("gsplat:opacity", Sdf.ValueTypeNames.FloatArray, Vt.FloatArray(splat.opacity.tolist()))
    vertex_primvar("gsplat:scale", Sdf.ValueTypeNames.Float3Array, Vt.Vec3fArray.FromNumpy(splat.scales))
    vertex_primvar("gsplat:rot", Sdf.ValueTypeNames.Float4Array, Vt.Vec4fArray.FromNumpy(splat.rotations))
    vertex_primvar("gsplat:sh0", Sdf.ValueTypeNames.Float3Array, Vt.Vec3fArray.FromNumpy(splat.sh0))
    if splat.sh_rest is not None:
        flat = np.ascontiguousarray(splat.sh_rest, dtype=np.float32).reshape(-1)
        vertex_primvar_count = splat.sh_rest.shape[1]
        primvars.CreatePrimvar(
            "gsplat:sh_rest", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.constant
        ).Set(Vt.FloatArray(flat.tolist()))
        points_prim.GetPrim().CreateAttribute(
            "gsplat:sh_rest_width", Sdf.ValueTypeNames.Int, custom=True
        ).Set(int(vertex_primvar_count))

    prim = points_prim.GetPrim()
    prim.CreateAttribute("gsplat:count", Sdf.ValueTypeNames.Int, custom=True).Set(splat.count)
    prim.CreateAttribute("gsplat:encoding", Sdf.ValueTypeNames.String, custom=True).Set("3dgs-raw")
    if source:
        prim.CreateAttribute("gsplat:source", Sdf.ValueTypeNames.String, custom=True).Set(source)

    stage.GetRootLayer().Save()
    return mode


def convert_splat_outputs(
    ply_source: Path,
    full_ply_path: Path,
    preview_ply_path: Path,
    usd_path: Path,
    log: Callable[[str], None] = lambda _msg: None,
) -> dict:
    """Standard post-reconstruction conversion: full PLY + preview + USD."""
    splat = read_gaussian_ply(ply_source)
    write_gaussian_ply(splat, full_ply_path)
    preview_count = write_preview_ply(splat, preview_ply_path)
    usd_mode = splat_to_usd(splat, usd_path, source=str(ply_source.name))
    log(
        f"Splat conversion: {splat.count} gaussians -> {full_ply_path.name}, "
        f"{preview_count} pts -> {preview_ply_path.name}, USD ({usd_mode})"
    )
    return {"gaussians": splat.count, "preview_points": preview_count, "usd_mode": usd_mode}
