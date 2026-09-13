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


def test_plain_point_cloud_can_be_packed_as_opaque_splats(tmp_path):
    from plyfile import PlyData, PlyElement
    from vaultwares_studio.splat_io import read_point_cloud_as_splat
    rows = np.array([(1.0, 2.0, 3.0, 255, 0, 128)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    path = tmp_path / "points.ply"
    PlyData([PlyElement.describe(rows, "vertex")]).write(str(path))
    splat = read_point_cloud_as_splat(path, point_scale=0.02)
    assert splat.count == 1
    np.testing.assert_allclose(splat.positions[0], [1, 2, 3])
    np.testing.assert_allclose(splat.colors_rgb()[0], [1, 0, 128 / 255], atol=1e-6)
    assert splat.opacity[0] == 8
    np.testing.assert_allclose(np.exp(splat.scales[0]), [0.02] * 3, rtol=1e-5)


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


def test_convert_splat_outputs_writes_full_preview_and_usd(tmp_path, monkeypatch):
    monkeypatch.setattr("vaultwares_studio.splat_io._native_gsplat_schema_available", lambda: False)
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


def test_native_splat_encoding_and_sh_order(tmp_path):
    from pxr import Usd, UsdVol
    from vaultwares_studio.splat_io import splat_to_usd

    if not hasattr(UsdVol, "ParticleField3DGaussianSplat"):
        pytest.skip("Native Gaussian schema requires OpenUSD 26.03+")
    splat = make_splat(count=2, rest_width=9)
    splat.scales[:] = np.log([1, 2, 3])
    splat.opacity[:] = [0, np.inf]
    splat.rotations[:] = [2, 0, 0, 0]
    splat.sh_rest[:] = np.arange(9)
    path = tmp_path / "native.usdc"
    assert splat_to_usd(splat, path) == "native-gaussian-splats"
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/World/GaussianSplats")
    assert prim.IsA(UsdVol.ParticleField3DGaussianSplat)
    np.testing.assert_allclose(prim.GetAttribute("positions").Get(), splat.positions)
    np.testing.assert_allclose(prim.GetAttribute("scales").Get(), [[1, 2, 3]] * 2)
    np.testing.assert_allclose(prim.GetAttribute("opacities").Get(), [0.5, 1])
    q = prim.GetAttribute("orientations").Get()[0]
    assert q.GetReal() == pytest.approx(1)
    np.testing.assert_allclose(q.GetImaginary(), [0, 0, 0])
    assert prim.GetAttribute("radiance:sphericalHarmonicsDegree").Get() == 1
    coeff = np.asarray(prim.GetAttribute("radiance:sphericalHarmonicsCoefficients").Get()).reshape(2, 4, 3)
    np.testing.assert_allclose(coeff[:, 0], splat.sh0)
    np.testing.assert_allclose(coeff[0, 1:], [[0, 3, 6], [1, 4, 7], [2, 5, 8]])
    assert len(prim.GetAttribute("extent").Get()) == 2


def test_native_splat_rejects_incomplete_sh_band(tmp_path):
    from vaultwares_studio.splat_io import splat_to_usd, _native_gsplat_schema_available
    if not _native_gsplat_schema_available():
        pytest.skip("Native Gaussian schema unavailable")
    with pytest.raises(ValueError, match="spherical harmonic"):
        splat_to_usd(make_splat(count=2, rest_width=6), tmp_path / "invalid.usda")
