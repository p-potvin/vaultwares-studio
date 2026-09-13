from pathlib import Path


def test_gaussian_viewer_plain_ply_fallback_is_opaque_and_identity_rotated():
    source = (Path(__file__).resolve().parents[1] / "vaultwares_studio/webviewer/vendor/gaussian-splats-3d.module.js").read_text(encoding="utf-8")
    assert "newSplat[OFFSET_OPACITY] = 255" in source
    assert "tempRotation.set(0, 0, 0, 1)" in source
