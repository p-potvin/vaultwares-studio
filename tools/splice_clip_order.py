"""One sequence out of two clips, spliced where they verifiably overlap.

DA3-Streaming aligns each 90-frame window to the last through the 45 frames
they share, so a sequence is one map only where consecutive frames overlap.
Two clips back to back overlap nowhere at the cut, and the 17 Sep merged run
showed the consequence: a 0.54-unit jump at the seam against 0.02 per step,
with the loop detector finding one cross-clip pair and closing nothing.

This reorders the frames so the cut lands on a pair of frames that a matcher
verified as the same view (``find_cross_clip_overlaps.py``):

    A[0 : a]  +  B[b : ]  +  B[ : b]  +  A[a : ]

Clip A runs to its frame ``a``, the sequence jumps to clip B's frame ``b``
(which looks at the same thing), follows B to its end, wraps to B's start
and continues to ``b`` again, then resumes A. Three seams; two of them are
the verified overlap, and the wrap in B is B's own start-to-end distance,
which a loop-shaped capture keeps short.

The frames are written at 10 fps into one video. The console samples at
10 fps for a request this size, so its candidates are these frames, one to
one and in this order, and the local training replay from the same file
lands on the same frames. Two encodes: full resolution for the replay, a
540p copy for the upload.

    python tools/splice_clip_order.py --candidates <dir> --split 941 \\
        --a 512 --b 1700 --keep-a 560 --keep-b 1040 --out <dir>/../spliced
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def even_subset(items: list, keep: int) -> list:
    if keep >= len(items):
        return list(items)
    idx = [min(len(items) - 1, i * len(items) // keep) for i in range(keep)]
    idx[0], idx[-1] = 0, len(items) - 1
    return [items[i] for i in idx]


def build_order(files: list[Path], split: int, a: int, b: int, keep_a: int, keep_b: int) -> list[Path]:
    """``split``, ``a``, ``b`` are 1-based candidate numbers as the files are named."""
    clip_a, clip_b = files[:split - 1], files[split - 1:]
    a_local, b_local = a - 1, b - split
    # a_local is bounded below exclusively and b_local inclusively, on purpose.
    # A seam at A's first frame leaves A contributing a single frame ahead of
    # the cut, which is not a splice of two clips — it is clip B with a stray
    # frame in front. B has no such degenerate case: b_local == 0 means the cut
    # lands on B's first frame, and all of B still follows it.
    if not (0 < a_local < len(clip_a)) or not (0 <= b_local < len(clip_b)):
        raise SystemExit(f"seam outside the clips: a={a} of {len(clip_a)}, b={b} of {len(clip_b)} (split {split})")
    # Thin each clip evenly to its budget first, then cut; the seam frames
    # themselves are forced in so the cut is exactly the verified pair.
    a_keep = even_subset(clip_a, keep_a)
    b_keep = even_subset(clip_b, keep_b)
    if clip_a[a_local] not in a_keep:
        a_keep.append(clip_a[a_local]); a_keep.sort()
    if clip_b[b_local] not in b_keep:
        b_keep.append(clip_b[b_local]); b_keep.sort()
    ia = a_keep.index(clip_a[a_local])
    ib = b_keep.index(clip_b[b_local])
    return a_keep[:ia + 1] + b_keep[ib:] + b_keep[:ib + 1] + a_keep[ia + 1:]


def encode(frames: list[Path], out: Path, fps: int, scale: str | None) -> None:
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.as_posix()}'\nduration {1 / fps:.6f}\n" for p in frames)
                       + f"file '{frames[-1].as_posix()}'\n", encoding="utf-8")
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
               "-f", "concat", "-safe", "0", "-i", str(listing),
               "-vsync", "cfr", "-r", str(fps)]
    if scale:
        command += ["-vf", f"scale={scale}"]
    command += ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19", "-b:v", "0",
                "-pix_fmt", "yuv420p", "-an", str(out)]
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--a", type=int, required=True, help="clip A candidate number at the seam")
    parser.add_argument("--b", type=int, required=True, help="clip B candidate number at the seam")
    parser.add_argument("--keep-a", type=int, required=True)
    parser.add_argument("--keep-b", type=int, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args(argv)

    files = sorted(args.candidates.glob("candidate_*.jpg"))
    order = build_order(files, args.split, args.a, args.b, args.keep_a, args.keep_b)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "candidates": str(args.candidates), "split": args.split, "seam_a": args.a, "seam_b": args.b,
        "keep_a": args.keep_a, "keep_b": args.keep_b, "fps": args.fps, "frames": len(order),
        "order": [p.name for p in order],
    }
    (args.out / "sequence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    encode(order, args.out / "spliced_1080p.mp4", args.fps, None)
    encode(order, args.out / "spliced_540p.mp4", args.fps, "960:540")
    seams = [i for i in range(1, len(order)) if
             (int(order[i - 1].stem.split("_")[-1]) < args.split) != (int(order[i].stem.split("_")[-1]) < args.split)]
    print(json.dumps({"frames": len(order), "duration_s": len(order) / args.fps,
                      "clip_seams_at": seams, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
