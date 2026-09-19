"""Where do two captures of the same place actually see the same thing?

Before splicing two clips into one sequence for DA3-Streaming, find the frame
pairs across them that a matcher can verify: same surfaces, similar viewing
direction, enough parallax-free correspondence for a fundamental matrix. Those
are the places where a seam can go, because the chunk alignment only joins
frames that overlap — it does not know or care that they came from different
videos.

Two stages, both local and free:

1. A global descriptor (DINOv2-small, CPU is fine at this size) shortlists,
   for every frame of clip A, its nearest frames in clip B by cosine
   similarity. Loop detectors do the same with SALAD; the point of doing it
   here is to see the candidates rather than trust a threshold.
2. SIFT + ratio test + RANSAC fundamental matrix verifies each shortlisted
   pair. The inlier count is the number that matters: a real overlap has
   many, a lookalike has a handful.

    python tools/find_cross_clip_overlaps.py --candidates <dir> --split 940 \\
        --stride 2 --out <dir>/../cross_clip_overlaps.json

``--split`` is the candidate index where clip B starts in a concatenated
candidate folder (clip A's duration times the candidate fps).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def embed(paths: list[Path], model_id: str = "facebook/dinov2-small", batch: int = 16) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(paths), batch):
            images = [Image.open(p).convert("RGB").resize((448, 252)) for p in paths[start:start + batch]]
            inputs = processor(images=images, return_tensors="pt")
            features = model(**inputs).pooler_output
            out.append(torch.nn.functional.normalize(features, dim=1).cpu().numpy())
            if (start // batch) % 20 == 0:
                print(f"[overlap] embedded {min(start + batch, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(out)


def verify(path_a: Path, path_b: Path, sift, matcher, width: int = 960) -> int:
    import cv2

    def load(p: Path):
        # imread reports an unreadable or truncated file by returning None
        # rather than raising, and a candidate directory of 2000 extracted
        # frames is exactly where one short write goes unnoticed.
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        scale = width / image.shape[1]
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    a, b = load(path_a), load(path_b)
    if a is None or b is None:
        return 0
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    good = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
    return int(mask.sum()) if mask is not None else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--split", type=int, required=True, help="first candidate index (1-based file number) of clip B")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-inliers", type=int, default=60)
    parser.add_argument("--min-similarity", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    files = sorted(args.candidates.glob("candidate_*.jpg"))
    a_files = files[:args.split - 1:args.stride]
    b_files = files[args.split - 1::args.stride]
    print(f"[overlap] clip A {len(a_files)} frames, clip B {len(b_files)} frames (stride {args.stride})")

    started = time.perf_counter()
    ea = embed(a_files)
    eb = embed(b_files)
    print(f"[overlap] embeddings in {time.perf_counter() - started:.0f}s")
    similarity = ea @ eb.T
    order = np.argsort(-similarity, axis=1)[:, :args.top_k]

    import cv2
    sift = cv2.SIFT_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    started = time.perf_counter()
    results = []
    checked = 0
    for ia, row in enumerate(order):
        for ib in row:
            sim = float(similarity[ia, ib])
            if sim < args.min_similarity:
                continue
            checked += 1
            inliers = verify(a_files[ia], b_files[ib], sift, matcher)
            if inliers >= args.min_inliers:
                results.append({
                    "a_file": a_files[ia].name, "b_file": b_files[ib].name,
                    "a_candidate": int(a_files[ia].stem.split("_")[-1]),
                    "b_candidate": int(b_files[ib].stem.split("_")[-1]),
                    "similarity": round(sim, 4), "inliers": inliers,
                })
        if ia % 50 == 0:
            print(f"[overlap] verified {ia}/{len(order)} A-frames, {len(results)} overlaps so far", flush=True)
    results.sort(key=lambda r: -r["inliers"])
    report = {
        "candidates": str(args.candidates), "split": args.split, "stride": args.stride,
        "clip_a_frames": len(a_files), "clip_b_frames": len(b_files),
        "pairs_checked": checked, "overlaps": len(results),
        "verify_seconds": round(time.perf_counter() - started, 1),
        "best": results[:40],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "best"}, indent=2))
    for r in results[:10]:
        print(f"  A {r['a_candidate']:5d}  <->  B {r['b_candidate']:5d}   inliers {r['inliers']:4d}  sim {r['similarity']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
