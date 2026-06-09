import numpy as np
import pytest

from vaultwares_studio.splat_io import (
    GaussianSplat,
    convert_splat_outputs,
    decimate,
    is_gaussian_ply,
    read_gaussian_ply,
    write_gaussian_ply,
    write_preview_ply,
)


def make_splat(count: int = 500, rest_width: int = 45) -> GaussianSplat:
    rng = np.random.default_rng(7)
    return GaussianSplat(
        positions=rng.normal(size=(count, 3)).astype(np.float32),
        sh0=rng.normal(size=(count, 3)).astype(np.float32),
        opacity=rng.normal(size=count).astype(np.float32),
        scales=rng.normal(size=(count, 3)).astype(np.float32),
        rotations=rng.normal(size=(count, 4)).astype(np.float32),
        sh_rest=rng.normal(size=(count, rest_width)).astype(np.float32),
    )


def test_gaussian_ply_round_trip(tmp_path):
    splat = make_splat()
    path = tmp_path / "splat.ply"
    write_gaussian_ply(splat, path)
    assert is_gaussian_ply(path)

    loaded = read_gaussian_ply(path)
    assert loaded.count == splat.count
    np.testing.assert_allclose(loaded.positions, splat.positions, rtol=1e-6)
    np.testing.assert_allclose(loaded.sh0, splat.sh0, rtol=1e-6)
    np.testing.assert_allclose(loaded.opacity, splat.opacity, rtol=1e-6)
    np.testing.assert_allclose(loaded.scales, splat.scales, rtol=1e-6)
    np.testing.assert_allclose(loaded.rotations, splat.rotations, rtol=1e-6)
    assert loaded.sh_rest is not None
    np.testing.assert_allclose(loaded.sh_rest, splat.sh_rest, rtol=1e-6)


def test_plain_point_cloud_is_not_gaussian(tmp_path):
    path = tmp_path / "plain.ply"
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0.0 0.0 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert not is_gaussian_ply(path)
    with pytest.raises(ValueError, match="not a 3DGS gaussian PLY"):
        read_gaussian_ply(path)


def test_decimate_caps_count():
    splat = make_splat(count=1000)
    smaller = decimate(splat, max_points=100)
    assert smaller.count == 100
    assert smaller.sh_rest is not None and smaller.sh_rest.shape == (100, 45)
    untouched = decimate(splat, max_points=5000)
    assert untouched.count == 1000


def test_preview_ply_is_open3d_compatible_shape(tmp_path):
    splat = make_splat(count=300)
    path = tmp_path / "preview.ply"
    count = write_preview_ply(splat, path, max_points=200)
    assert count == 200
    from plyfile import PlyData

    data = PlyData.read(str(path))
    names = {prop.name for prop in data["vertex"].properties}
    assert {"x", "y", "z", "red", "green", "blue"} <= names
    assert not is_gaussian_ply(path)


def test_convert_splat_outputs_writes_full_preview_and_usd(tmp_path):
    splat = make_splat(count=400)
    source = tmp_path / "exported.ply"
    write_gaussian_ply(splat, source)

    full = tmp_path / "cloud.ply"
    preview = tmp_path / "cloud_preview.ply"
    usd = tmp_path / "cloud.usda"
    info = convert_splat_outputs(source, full, preview, usd)

    assert info["gaussians"] == 400
    assert is_gaussian_ply(full)
    assert preview.exists()
    assert usd.exists()

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd))
    prim = stage.GetPrimAtPath("/World/GaussianSplats")
    assert prim.IsValid()
    points = UsdGeom.Points(prim)
    assert len(points.GetPointsAttr().Get()) == 400
    primvars = UsdGeom.PrimvarsAPI(prim)
    opacity = primvars.GetPrimvar("gsplat:opacity").Get()
    assert len(opacity) == 400
    np.testing.assert_allclose(np.asarray(opacity), splat.opacity, rtol=1e-6)
    rot = primvars.GetPrimvar("gsplat:rot").Get()
    assert len(rot) == 400
    assert prim.GetAttribute("gsplat:count").Get() == 400
    assert prim.GetAttribute("gsplat:encoding").Get() == "3dgs-raw"
    assert prim.GetAttribute("gsplat:sh_rest_width").Get() == 45
