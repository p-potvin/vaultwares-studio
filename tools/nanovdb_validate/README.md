# Validating a `.nvdb` with NVIDIA's own reader

`nanovdb_write` and `nanovdb_read` are both ours. A file the two agree on proves
only that they agree — if the format were misread, both would be wrong together
and the round trip would say nothing. `pnvalidate.c` closes that gap: it is a
~70-line program built against **`PNanoVDB.h`**, the portable C99/HLSL reference
reader shipped with OpenVDB, written by the format's authors.

It walks the tree through PNanoVDB's read accessor — root → upper → lower →
leaf, following the child masks and relative offsets — so a correct value proves
the whole structure, not just the header.

## Build

`PNanoVDB.h` is header-only with no dependencies, so this needs a C compiler and
nothing else. No OpenVDB build, no vcpkg.

```powershell
curl -sL -o PNanoVDB.h https://raw.githubusercontent.com/AcademySoftwareFoundation/openvdb/master/nanovdb/nanovdb/PNanoVDB.h
$mv  = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231"
$kits = "C:\Program Files (x86)\Windows Kits\10"; $sdk = "10.0.26100.0"
$env:INCLUDE = "$mv\include;$kits\Include\$sdk\ucrt;$kits\Include\$sdk\shared;$kits\Include\$sdk\um"
$env:LIB     = "$mv\lib\x64;$kits\Lib\$sdk\ucrt\x64;$kits\Lib\$sdk\um\x64"
$env:PATH    = "$mv\bin\Hostx64\x64;$env:PATH"
cl /nologo /O2 /I. pnvalidate.c /Fe:pnvalidate.exe
```

Setting `INCLUDE`/`LIB` directly rather than calling `vcvars64.bat` is
deliberate — the batch file takes minutes on this machine and does nothing else
that matters here.

## Run

```powershell
.\pnvalidate.exe sphere.nvdb  -15 0 0  -7 -8 -1  7 8 0
```

## The result that mattered

A sphere's signed distance field, 11,178 voxels, written by `nanovdb_write`:

```text
gridMagic=0x314244566f6e614e
gridType=1 gridClass=1
voxelSize=0.25 0.25 0.25
nodes leaf=60 lower=8 upper=8 voxels=11178
bbox=-15 -15 -15 .. 15 15 15
value -15 0 0 = 3
value -7 -8 -1 = -1.32292175
value  7 8 0  = -1.36985421
```

against the analytic values 3, -1.32292175 and -1.36985419. The probes include
negative coordinates on purpose: `RootData`'s key casts each signed coordinate
to `uint32` before shifting, so anything below zero is where a plausible-looking
implementation goes wrong.

Note the first run of this validator failed with `MAGIC FAIL` and
`gridMagic=0xff314244566f6e61` — the magic shifted one byte. That was a bug in
**this** file, not in the writer: `nameSize` sits at metadata offset 136
(32 + 8 + 48 + 24 + 24), not 148. Worth recording, because a validator that
cannot be wrong is not checking anything.
