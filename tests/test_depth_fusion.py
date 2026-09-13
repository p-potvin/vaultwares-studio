"""TSDF fusion on a synthetic scene: a flat floor seen from a short walk."""
import numpy as np
import pytest
from pxr import Usd, UsdGeom

from vaultwares_studio.depth_fusion import fuse_streaming_mesh, mesh_to_usd


W, H = 96, 64
FX = FY = 80.0
CX, CY = W / 2, H / 2


def _look_down_c2w(x, z, height=2.0):
    """OpenCV camera (z forward, y down) pointing straight at the floor y=0."""
    c2w = np.eye(4)
    # camera z -> world -y, camera y -> world +z, camera x -> world +x
    c2w[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    c2w[:3, 3] = [x, height, z]
    return c2w


def _write_scene(stream_dir, count=8, height=2.0, scale=1.0):
    results = stream_dir / "results_output"
    results.mkdir(parents=True)
    poses = []
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)
    for i in range(count):
        c2w = _look_down_c2w(0.15 * i, 0.1 * i, height)
        poses.append(c2w.reshape(-1))
        # Planar depth: every pixel at distance `height` along the optical axis.
        depth = np.full((H, W), height / scale, dtype=np.float32)
        conf = np.full((H, W), 5.0, dtype=np.float32)
        conf[:4, :] = 0.0  # a low-confidence band that must be masked out
        image = np.full((H, W, 3), 120, dtype=np.uint8)
        np.savez(results / f"frame_{i}.npz", image=image, depth=depth, conf=conf,
                 intrinsics=K, extrinsics=np.linalg.inv(c2w)[:3].astype(np.float32), s=np.float64(scale))
    np.savetxt(stream_dir / "camera_poses.txt", np.stack(poses))


def test_fusion_recovers_a_flat_floor(tmp_path):
    stream = tmp_path / "streaming"
    _write_scene(stream)
    out = tmp_path / "mesh.ply"
    report = fuse_streaming_mesh(stream, out, voxel_size=0.05, log=lambda _m: None)
    assert out.exists() and report["triangles"] > 100 and report["frames_used"] == 8
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(out))
    vertices = np.asarray(mesh.vertices)
    # The surface is the floor at y = 0, to within a voxel.
    assert abs(float(np.median(vertices[:, 1]))) < 0.05
    assert vertices[:, 1].std() < 0.05


def test_fusion_applies_chunk_scale_and_scene_transform(tmp_path):
    stream = tmp_path / "streaming"
    # Depth stored in chunk units at half scale; s=2 must restore the floor.
    _write_scene(stream, scale=2.0)
    lift = np.eye(4); lift[:3, 3] = [0, 10, 0]
    report = fuse_streaming_mesh(stream, tmp_path / "mesh.ply", voxel_size=0.05,
                                 scene_transform=lift, log=lambda _m: None)
    assert report["scene_transform_applied"]
    import open3d as o3d
    vertices = np.asarray(o3d.io.read_triangle_mesh(str(tmp_path / "mesh.ply")).vertices)
    assert abs(float(np.median(vertices[:, 1])) - 10.0) < 0.05


def test_fusion_rejects_mismatched_counts(tmp_path):
    stream = tmp_path / "streaming"
    _write_scene(stream, count=3)
    (stream / "results_output" / "frame_2.npz").unlink()
    with pytest.raises(ValueError, match="depth frames vs"):
        fuse_streaming_mesh(stream, tmp_path / "mesh.ply", voxel_size=0.05, log=lambda _m: None)


def test_mesh_to_usd_writes_a_mesh_prim(tmp_path):
    stream = tmp_path / "streaming"
    _write_scene(stream, count=4)
    fuse_streaming_mesh(stream, tmp_path / "mesh.ply", voxel_size=0.05, log=lambda _m: None)
    usd = mesh_to_usd(tmp_path / "mesh.ply", tmp_path / "mesh.usda", source="test")
    stage = Usd.Stage.Open(str(usd))
    prim = stage.GetPrimAtPath("/World/FusedSurface")
    assert prim.IsA(UsdGeom.Mesh)
    mesh = UsdGeom.Mesh(prim)
    assert len(mesh.GetPointsAttr().Get()) > 0
    assert set(mesh.GetFaceVertexCountsAttr().Get()) == {3}
    assert prim.GetAttribute("vw:source").Get() == "test"
