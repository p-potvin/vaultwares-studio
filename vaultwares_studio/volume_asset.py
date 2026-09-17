"""Make the ``.nvdb`` a USD asset, not just a file on disk.

A volume only becomes part of the digital twin when the stage references it.
Until then it is a 16 MB blob nothing knows about.

USD models volumes as a ``UsdVolVolume`` prim holding one or more field prims,
each naming a file and a field inside it, wired to the volume by a relationship
whose *name* is what a renderer asks for. The wiring is the part that fails
quietly: a stage with the field prim present but the relationship missing opens
without complaint and renders nothing.

**On the schema name.** The field prim is ``UsdVolOpenVDBAsset`` and the file we
point it at is NanoVDB. That is not a mistake — Omniverse's volume loader reads
``.nvdb`` through this prim, and NanoVDB is a serialisation of the same tree
OpenVDB holds in memory. It does mean a renderer that takes the schema name
literally and calls a strict OpenVDB reader will reject the file, so this is
worth knowing rather than discovering. ``openvdb`` remains the right extension
to hand a DCC tool; ``.nvdb`` is the right one to hand Omniverse.

**Relative paths.** ``filePath`` is written relative to the USD layer when the
two sit together, so moving the pair keeps the reference intact. An absolute
path works on the machine that wrote it and nowhere else, which is the usual way
a twin arrives broken.
"""

from __future__ import annotations

import os
from pathlib import Path


def volume_to_usd(
    nvdb_path: Path | str,
    usd_path: Path | str,
    *,
    field_name: str = "surface",
    prim_path: str = "/World/Volume",
    grid_class: str = "levelSet",
) -> Path:
    """Author a USD layer referencing ``nvdb_path`` as a volume field.

    ``field_name`` must match the grid's name inside the file — that string is
    the lookup key, and a mismatch yields an empty volume rather than an error.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdVol

    nvdb_path = Path(nvdb_path)
    usd_path = Path(usd_path)
    usd_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        reference = os.path.relpath(nvdb_path.resolve(), usd_path.resolve().parent)
    except ValueError:  # different drives on Windows; absolute is the only option
        reference = str(nvdb_path.resolve())
    reference = reference.replace(os.sep, "/")

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    volume = UsdVol.Volume.Define(stage, prim_path)
    asset = UsdVol.OpenVDBAsset.Define(stage, f"{prim_path}/{field_name}")
    asset.CreateFilePathAttr(Sdf.AssetPath(reference))
    asset.CreateFieldNameAttr(field_name)
    # fieldClass tells a consumer how to interpret the values. A level set is
    # signed distance in world units; labelling it as fog density would render
    # the negative interior as nothing at all.
    asset.CreateFieldClassAttr(grid_class)
    asset.CreateFieldDataTypeAttr("float")
    # The relationship, named for the field, is what actually connects them.
    volume.CreateFieldRelationship(field_name, asset.GetPath())

    stage.GetRootLayer().Save()
    return usd_path
