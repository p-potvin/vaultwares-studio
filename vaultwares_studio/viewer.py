from __future__ import annotations

from pathlib import Path


def open_live_viewer(point_cloud_path: Path | str) -> tuple[bool, str]:
    target = Path(point_cloud_path)
    if not target.exists():
        return False, f"Point cloud not found: {target}"

    try:
        import open3d as o3d
    except ImportError:
        return False, "open3d is not installed."

    point_cloud = o3d.io.read_point_cloud(str(target))
    if point_cloud.is_empty():
        return False, "The point cloud is empty and cannot be displayed."

    o3d.visualization.draw_geometries([point_cloud], window_name="Digital Twin Viewer")
    return True, f"Opened viewer for {target}"
