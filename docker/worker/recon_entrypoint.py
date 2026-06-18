"""Remote reconstruction stage: frames.zip -> trained gaussian splat.

Runs inside the vw-studio-worker image with VW_IN / VW_OUT set by
vw_stage.py. Pipeline:

    unzip frames.zip
    ns-process-data images (sequential matching — frames come from video)
    ns-train splatfacto    (preset args, unsupported flags auto-dropped)
    ns-export gaussian-splat

Outputs written to $VW_OUT:
    splat.ply     full-attribute 3DGS PLY
    summary.json  registered image count, durations, args actually used
    model.zip     training config + checkpoint (when --keep-checkpoint),
                  reused by remote ns-render walkthroughs in M2
    error.json    structured failure info (e.g. too_few_registered_images)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# Hard floor for any valid reconstruction; per-job threshold is
# max(MIN_REGISTERED_IMAGES, frame_count * MIN_REGISTERED_RATIO).
MIN_REGISTERED_IMAGES = 10
MIN_REGISTERED_RATIO = 0.5

# Pretrained vocabulary tree for COLMAP's vocab_tree matcher. Downloaded
# once per worker run and cached at the static path below. Source:
# https://demuc.de/colmap/ — same file the official COLMAP wiki points at.
VOCAB_TREE_URL = "https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin"
VOCAB_TREE_PATH = Path("/tmp/vocab_tree_flickr100K_words256K.bin")


def vocab_tree_augment_and_remap(processed: Path, vocab_tree: Path) -> int:
    """Add vocab-tree matches to the existing COLMAP database, then re-map.

    Bypasses ``ns-process-data --matching-method vocab_tree`` because the
    nerfstudio CLI in this worker rejects ``--vocab-tree-path``. We hit
    ``colmap vocab_tree_matcher`` directly on the database sequential left
    behind, then ``retry_mapper`` (which now has more matches to chew on).
    """
    database = processed / "colmap" / "database.db"
    image_dir = processed / "images"
    if not (database.exists() and image_dir.exists()):
        print("[recon] vocab_tree: missing database or image dir, skipping.", flush=True)
        return 0
    print("[recon] augmenting database with vocab_tree matches", flush=True)
    # GPU SIFT matching: COLMAP's GPU path was previously failing silently in
    # the L4 container; defaulting to GPU now and letting the run fail loudly
    # if the install is still broken — cheaper to learn that than to keep
    # burning ~$1.50 per refine on L4-grade dollars for CPU SIFT.
    result = run([
        "colmap", "vocab_tree_matcher",
        "--database_path", str(database),
        "--VocabTreeMatching.vocab_tree_path", str(vocab_tree),
        # 100 nearest-neighbour images per query is the COLMAP default; bumping
        # higher catches more loop closures at proportional CPU cost.
        "--VocabTreeMatching.num_images", "100",
        "--SiftMatching.use_gpu", "1",
    ])
    if result.returncode != 0:
        print(f"[recon] vocab_tree_matcher exit={result.returncode}", flush=True)
        return 0
    # retry_mapper runs mapper 4 times with progressively-relaxed init
    # thresholds. With more matches in the database the init pair candidate
    # pool is wider, so one of those attempts should land a good model.
    return retry_mapper(processed)


def ensure_vocab_tree() -> Path | None:
    """Download the pretrained vocab tree if it isn't already on disk."""
    if VOCAB_TREE_PATH.exists() and VOCAB_TREE_PATH.stat().st_size > 1_000_000:
        return VOCAB_TREE_PATH
    try:
        print(f"[recon] downloading vocab tree from {VOCAB_TREE_URL}", flush=True)
        urllib.request.urlretrieve(VOCAB_TREE_URL, str(VOCAB_TREE_PATH))
        return VOCAB_TREE_PATH if VOCAB_TREE_PATH.exists() else None
    except Exception as exc:  # noqa: BLE001 - log + degrade gracefully
        print(f"[recon] vocab tree download failed: {exc}", flush=True)
        return None


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"[recon] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False, **kwargs)


def filter_supported_flags(train_args: list[str]) -> list[str]:
    """Drop --flag value pairs that the installed ns-train doesn't know."""
    help_text = ""
    probe = run(["ns-train", "splatfacto", "--help"], capture_output=True, text=True)
    if probe.returncode == 0:
        help_text = probe.stdout + probe.stderr
    if not help_text:
        return train_args
    # Value drift, not just flag drift: newer nerfstudio removed the 'none'
    # choice for --vis; tensorboard is the headless-safe choice everywhere.
    if "--vis" in train_args:
        index = train_args.index("--vis") + 1
        import re

        window = re.search(r"--vis\b(.{0,300})", help_text, re.DOTALL)
        choices = window.group(1) if window else ""
        if index < len(train_args) and train_args[index] == "none" and "none" not in choices:
            print("[recon] --vis none unsupported in this nerfstudio; using tensorboard", flush=True)
            train_args[index] = "tensorboard"
    kept: list[str] = []
    index = 0
    while index < len(train_args):
        arg = train_args[index]
        if arg.startswith("--") and arg not in help_text:
            print(f"[recon] dropping unsupported flag: {arg}", flush=True)
            index += 2 if index + 1 < len(train_args) and not train_args[index + 1].startswith("--") else 1
            continue
        kept.append(arg)
        index += 1
    return kept


def colmap_db_stats(processed_dir: Path) -> dict:
    """Keypoint/match telemetry from COLMAP's database — pinpoints WHERE SfM starves."""
    import sqlite3

    db_path = next(processed_dir.rglob("database.db"), None)
    if db_path is None:
        return {}
    try:
        con = sqlite3.connect(db_path)
        kp_count, kp_avg = con.execute("SELECT COUNT(*), AVG(rows) FROM keypoints").fetchone()
        match_pairs, match_avg = con.execute(
            "SELECT COUNT(*), AVG(rows) FROM matches WHERE rows > 0"
        ).fetchone()
        verified_pairs, verified_avg = con.execute(
            "SELECT COUNT(*), AVG(rows) FROM two_view_geometries WHERE rows > 0"
        ).fetchone()
        con.close()
        return {
            "images_with_keypoints": kp_count,
            "avg_keypoints": round(kp_avg or 0, 1),
            "raw_match_pairs": match_pairs,
            "avg_raw_matches": round(match_avg or 0, 1),
            "verified_pairs": verified_pairs,
            "avg_verified_matches": round(verified_avg or 0, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return {"db_stats_error": str(exc)}


def smart_init_pair_candidates(
    database_path: Path, top_n: int = 4, frame_count: int = 0
) -> list[tuple[int, int, int]]:
    """Return top N (image_id1, image_id2, num_matches) init-pair candidates.

    v2 changes over the original heuristic:

    1. **config filter**. COLMAP's ``two_view_geometries.config`` tags each
       pair's geometric model:

       - 2 = CALIBRATED (Essential matrix)
       - 3 = UNCALIBRATED (Fundamental matrix)
       - 4 = PLANAR (homography only — rank-deficient, cannot bootstrap 3D)
       - 5 = PANORAMIC (pure rotation — zero baseline, cannot triangulate)
       - 6 = PLANAR_OR_PANORAMIC (ambiguous, treat as bad)

       v1 ignored config and likely fed mapper planar pairs from backyard
       grass/pavement. v2 takes only ``config IN (2, 3)``.

    2. **Region-spread instead of dedup-by-id**. v1 dedup'd on individual
       image_ids and accidentally walked down a sorted list within a single
       motion-rich region of the timeline (run logs showed picks
       (91,123)(89,121)(92,124)(86,120) — all within 6 seconds). v2 buckets
       picks by (min_id // region_size, max_id // region_size) and takes
       at most one pair per bucket, where region_size = frame_count // 4.
       Guarantees temporal diversity across the capture.

    pair_id encoding: ``pair_id = image_id1 * 2147483647 + image_id2`` per
    COLMAP's database.cc (kMaxNumImages).
    """
    import sqlite3

    if not database_path.exists():
        return []
    MAX_NUM_IMAGES = 2147483647
    try:
        con = sqlite3.connect(str(database_path))
        rows = con.execute(
            "SELECT pair_id, rows, config FROM two_view_geometries WHERE rows > 100"
        ).fetchall()
        con.close()
    except sqlite3.OperationalError as exc:
        print(f"[recon] smart picker DB read failed: {exc}", flush=True)
        return []

    # Diagnostic: which config tags dominate? Tells us at a glance whether
    # the scene is fundamentally planar/panoramic.
    config_counts: dict[int, int] = {}
    for _, _rows, config in rows:
        config_counts[config] = config_counts.get(config, 0) + 1
    print(f"[recon] two_view_geometries config distribution: {config_counts}", flush=True)

    scored = []
    for pair_id, num_matches, config in rows:
        # Filter rank-deficient geometries: planar and panoramic pairs cannot
        # produce a 3D init regardless of match count.
        if config not in (2, 3):
            continue
        id1 = pair_id // MAX_NUM_IMAGES
        id2 = pair_id % MAX_NUM_IMAGES
        if id1 == 0 or id2 == 0 or id1 == id2:
            continue
        temporal_gap = abs(id2 - id1)
        if temporal_gap < 30:
            continue
        scored.append((num_matches * temporal_gap, id1, id2, num_matches, temporal_gap))
    scored.sort(reverse=True)

    # Region-bucket so picks span the timeline. region_size = frame_count/4
    # gives 4 along-the-diagonal zones; pairs land in one of (region_a,
    # region_b) buckets and we keep at most one pair per bucket.
    region_size = max(1, (frame_count or 280) // 4)
    seen_buckets: set[tuple[int, int]] = set()
    out: list[tuple[int, int, int]] = []
    for _, id1, id2, num_matches, _gap in scored:
        bucket = (min(id1, id2) // region_size, max(id1, id2) // region_size)
        if bucket in seen_buckets:
            continue
        seen_buckets.add(bucket)
        out.append((id1, id2, num_matches))
        if len(out) >= top_n:
            break
    return out


def _count_model_images(model_dir: Path) -> int:
    if not (model_dir / "images.txt").exists():
        run([
            "colmap", "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(model_dir),
            "--output_type", "TXT",
        ], capture_output=True)
    images_txt = model_dir / "images.txt"
    if not images_txt.exists():
        return 0
    lines = [l for l in images_txt.read_text(encoding="utf-8", errors="replace").splitlines() if l and not l.startswith("#")]
    return len(lines) // 2


def retry_mapper(processed: Path) -> int:
    """Re-run ONLY the incremental mapper with relaxed initialization.

    COLMAP's mapper is non-deterministic: the same (healthy) match database
    produced 280/280 in one run and 2/280 in another — a degenerate RANSAC
    init pair kills the whole model. Re-mapping is ~2 min vs ~25 for a full
    matching pass, so try it (twice, progressively relaxed) before
    escalating to exhaustive matching.
    """
    database = processed / "colmap" / "database.db"
    image_dir = processed / "images"
    if not (database.exists() and image_dir.exists()):
        return 0
    best_count, best_model = 0, None

    # Smart init-pair candidates from the verified-pair table, ranked by
    # num_matches * temporal_distance. Better than COLMAP's heuristic for
    # walking captures because we explicitly require spatial baseline (via
    # temporal gap) AND filter out rank-deficient planar/panoramic pairs.
    image_count = len(list((processed / "images").glob("*"))) if (processed / "images").exists() else 0
    smart_pairs = smart_init_pair_candidates(database, top_n=4, frame_count=image_count)
    if smart_pairs:
        print(
            f"[recon] smart init pairs ({len(smart_pairs)}): "
            f"{[(a, b, m) for a, b, m in smart_pairs]}",
            flush=True,
        )

    # Threshold ladder applied to each attempt. Index = attempt number; we
    # only relax thresholds if the explicit pair still fails.
    thresholds = [
        ["--Mapper.init_min_tri_angle", "8", "--Mapper.init_min_num_inliers", "50"],
        ["--Mapper.init_min_tri_angle", "4", "--Mapper.init_min_num_inliers", "30",
         "--Mapper.abs_pose_min_num_inliers", "15"],
        ["--Mapper.init_min_tri_angle", "2", "--Mapper.init_min_num_inliers", "20",
         "--Mapper.abs_pose_min_num_inliers", "10"],
        ["--Mapper.init_min_tri_angle", "1.5", "--Mapper.init_min_num_inliers", "15",
         "--Mapper.abs_pose_min_num_inliers", "8"],
    ]

    attempts: list[list[str]] = []
    for idx, (id1, id2, _matches) in enumerate(smart_pairs):
        attempts.append(
            thresholds[min(idx, len(thresholds) - 1)]
            + ["--Mapper.init_image_id1", str(id1), "--Mapper.init_image_id2", str(id2)]
        )
    # If the DB has no good explicit candidates (e.g. all pairs too temporally
    # close), fall back to relaxed thresholds + many trials and let COLMAP pick.
    if not attempts:
        attempts = [t + ["--Mapper.init_num_trials", "2000"] for t in thresholds]
    for attempt, options in enumerate(attempts, start=1):
        out_dir = processed / "colmap" / f"sparse_retry{attempt}"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run([
            "colmap", "mapper",
            "--database_path", str(database),
            "--image_path", str(image_dir),
            "--output_path", str(out_dir),
            "--Mapper.multiple_models", "0",
            *options,
        ])
        if result.returncode != 0:
            continue
        for model_dir in (d for d in out_dir.iterdir() if d.is_dir()):
            count = _count_model_images(model_dir)
            print(f"[recon] mapper retry {attempt}: model {model_dir.name} registered {count}", flush=True)
            if count > best_count:
                best_count, best_model = count, model_dir
        if best_count >= MIN_REGISTERED_IMAGES:
            break
    if best_model is None or best_count < MIN_REGISTERED_IMAGES:
        return 0
    regen = run([
        "ns-process-data", "images",
        "--data", str(image_dir),
        "--output-dir", str(processed),
        "--skip-colmap",
        "--skip-image-processing",
        "--colmap-model-path", str(best_model.relative_to(processed)),
    ])
    if regen.returncode != 0:
        return 0
    return count_registered_images(processed)


def count_registered_images(processed_dir: Path) -> int:
    transforms = processed_dir / "transforms.json"
    if not transforms.exists():
        return 0
    try:
        payload = json.loads(transforms.read_text(encoding="utf-8"))
        return len(payload.get("frames", []))
    except (OSError, json.JSONDecodeError):
        return 0


def fail(out_dir: Path, code: str, detail: str) -> int:
    print(f"[recon] FAILED: {code} — {detail}", flush=True)
    (out_dir / "error.json").write_text(
        json.dumps({"code": code, "detail": detail}, indent=2), encoding="utf-8"
    )
    return 1


def refine_mode_setup(
    in_dir: Path, processed: Path, train_out: Path
) -> tuple[Path, list[str]]:
    """Restore the base reconstruction's database/sparse model + train checkpoint.

    Reads model.zip (the base train output tree) and processed_min.zip (base
    colmap_database + sparse/*.bin + transforms.json + sparse_pc.ply) from
    VW_IN. Lays them out so the regular COLMAP + ns-train commands see them
    in the expected places. Returns the load-dir to pass to ns-train (the
    nerfstudio timestamp dir containing config.yml + nerfstudio_models/) and
    the list of new image filenames (so we know which to feature-extract).
    """
    bundle_processed = in_dir / "processed_min.zip"
    bundle_model = in_dir / "model.zip"
    if not bundle_processed.exists():
        raise RuntimeError(f"--refine-mode but processed_min.zip not in VW_IN ({in_dir})")
    if not bundle_model.exists():
        raise RuntimeError(f"--refine-mode but model.zip not in VW_IN ({in_dir})")

    # Restore the base reconstruction state into the standard processed/ layout
    # so subsequent COLMAP commands and ns-process-data see what they expect.
    base = Path("/tmp/recon/base_processed")
    base.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_processed) as archive:
        archive.extractall(base)
    (processed / "colmap").mkdir(parents=True, exist_ok=True)
    (processed / "colmap" / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / "colmap_database.db", processed / "colmap" / "database.db")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        src = base / "sparse" / name
        if not src.exists():
            raise RuntimeError(
                f"refine bundle is missing sparse/{name} — base job was built "
                "before sparse-model archival; re-run the base in v2-compatible "
                "mode before refining."
            )
        shutil.copyfile(src, processed / "colmap" / "sparse" / "0" / name)

    # Identify NEW images = those in VW_IN/images/ that aren't yet in the
    # base database. Everything else gets skipped on feature_extractor (its
    # features are already cached).
    import sqlite3

    con = sqlite3.connect(str(processed / "colmap" / "database.db"))
    existing = {row[0] for row in con.execute("SELECT name FROM images").fetchall()}
    con.close()
    images_dir = Path("/tmp/recon/images")
    all_imgs = [p.name for p in sorted(images_dir.iterdir()) if p.suffix.lower() in (".jpg", ".png")]
    new_imgs = [name for name in all_imgs if name not in existing]
    print(f"[refine] base images: {len(existing)} | new images: {len(new_imgs)} | total on disk: {len(all_imgs)}", flush=True)

    # Find the experiment timestamp dir inside the unpacked model.zip — it's
    # the dir containing config.yml + nerfstudio_models/. ns-train --load-dir
    # wants this as its argument.
    base_train = Path("/tmp/recon/base_train")
    base_train.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_model) as archive:
        archive.extractall(base_train)
    configs = sorted(base_train.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not configs:
        raise RuntimeError("refine model.zip contains no config.yml")
    load_dir = configs[0].parent
    ckpts = list((load_dir / "nerfstudio_models").glob("*.ckpt")) if (load_dir / "nerfstudio_models").exists() else []
    if not ckpts:
        raise RuntimeError(f"no checkpoint .ckpt files found under {load_dir / 'nerfstudio_models'}")
    print(f"[refine] checkpoint load-dir: {load_dir} ({len(ckpts)} ckpts)", flush=True)
    return load_dir, new_imgs


def refine_register_new_images(
    processed: Path, new_imgs: list[str], vocab_tree: Path | None,
    out_dir: Path | None = None,
) -> int:
    """Add new images to the existing sparse reconstruction via image_registrator.

    Sequence:
      feature_extractor (only new images via --image_list_path)
        → sequential_matcher (within new images)
        → vocab_tree_matcher (new vs everything, finds cross-clip matches)
        → image_registrator (extends the base sparse model with new images)
        → bundle_adjuster (joint refinement)
        → ns-process-data --skip-colmap (regenerate transforms.json from final model)

    Returns the new total registered-image count, or 0 on hard failure.

    Passing out_dir triggers periodic database checkpointing — the augmented
    colmap_database is copied to out_dir/colmap_database.db after each
    expensive matcher pass so a mid-run cancellation preserves the work for
    the next attempt (the standard mode already does this; refine mode used
    to drop the floor on cancellation, which burned an entire run's worth of
    cost when the user had to abort).
    """
    def _checkpoint_db(label: str) -> None:
        if out_dir is None:
            return
        src = processed / "colmap" / "database.db"
        if not src.exists():
            return
        try:
            shutil.copyfile(src, out_dir / "colmap_database.db")
            print(f"[refine] db checkpoint after {label}: {src.stat().st_size // 1_000_000} MB", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[refine] db checkpoint failed after {label}: {exc}", flush=True)
    images_dir = Path("/tmp/recon/images")
    database = processed / "colmap" / "database.db"
    base_sparse = processed / "colmap" / "sparse" / "0"
    list_path = Path("/tmp/recon/new_image_list.txt")
    list_path.write_text("\n".join(new_imgs) + "\n", encoding="utf-8")

    # Feature extractor on new images only. single_camera so we share the
    # base run's intrinsics (the iPhone is the same camera — must match for
    # the registered images to share a camera_id with the base set).
    # GPU SIFT for all refine matchers. The L4 container's COLMAP CUDA path
    # was previously broken (silent degradation), so we hard-coded use_gpu=0.
    # Re-enabling because at L4 prices CPU SIFT cost ~$1.50/refine. Failures
    # here exit non-zero, which is loud + recoverable (worst case: revert).
    fe = run([
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(images_dir),
        "--image_list_path", str(list_path),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1",
    ])
    if fe.returncode != 0:
        print(f"[refine] feature_extractor failed: exit={fe.returncode}", flush=True)
        return 0
    _checkpoint_db("feature_extractor")

    # Sequential matcher within new images (cheap; catches adjacent-frame pairs).
    sm = run([
        "colmap", "sequential_matcher",
        "--database_path", str(database),
        "--SiftMatching.use_gpu", "1",
    ])
    if sm.returncode != 0:
        print(f"[refine] sequential_matcher failed: exit={sm.returncode}", flush=True)
    _checkpoint_db("sequential_matcher")

    # Vocab tree matcher: cross-matches new images against the whole database
    # (including the base set). This is where new + base get linked.
    if vocab_tree is not None:
        vtm = run([
            "colmap", "vocab_tree_matcher",
            "--database_path", str(database),
            "--VocabTreeMatching.vocab_tree_path", str(vocab_tree),
            "--VocabTreeMatching.num_images", "100",
            "--SiftMatching.use_gpu", "1",
        ])
        if vtm.returncode != 0:
            print(f"[refine] vocab_tree_matcher failed: exit={vtm.returncode}", flush=True)
        _checkpoint_db("vocab_tree_matcher")

    # image_registrator extends the base sparse model with the new images.
    # Outputs cameras.bin/images.bin/points3D.bin into a new dir.
    registered = processed / "colmap" / "sparse" / "registered"
    registered.mkdir(parents=True, exist_ok=True)
    ir = run([
        "colmap", "image_registrator",
        "--database_path", str(database),
        "--input_path", str(base_sparse),
        "--output_path", str(registered),
    ])
    if ir.returncode != 0:
        return 0

    # Bundle adjuster refines the joint solution. Skip if it fails — the
    # unrefined model is still usable.
    run([
        "colmap", "bundle_adjuster",
        "--input_path", str(registered),
        "--output_path", str(registered),
    ])

    # Regen transforms.json from the registered model. ns-process-data with
    # --skip-colmap reads the model at --colmap-model-path and rewrites
    # transforms.json + sparse_pc.ply.
    regen = run([
        "ns-process-data", "images",
        "--data", str(images_dir),
        "--output-dir", str(processed),
        "--skip-colmap",
        "--skip-image-processing",
        "--colmap-model-path", str(registered.relative_to(processed)),
    ])
    if regen.returncode != 0:
        return 0
    return count_registered_images(processed)


def finish_export(
    args, out_dir: Path, processed: Path, train_out: Path, export: Path,
    timings: dict, frame_count: int, registered: int, matching_used: str,
) -> int:
    """Shared post-train flow: ns-export, copy outputs, archive bundles, summary."""
    configs = sorted(train_out.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not configs:
        return fail(out_dir, "no_config", "no config.yml produced by training")
    started = time.monotonic()
    exported = run([
        "ns-export", "gaussian-splat",
        "--load-config", str(configs[0]),
        "--output-dir", str(export),
    ])
    timings["export_s"] = round(time.monotonic() - started, 1)
    if exported.returncode != 0:
        return fail(out_dir, "export_failed", f"ns-export exit {exported.returncode}")

    plys = sorted(export.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not plys:
        return fail(out_dir, "no_ply", "ns-export produced no PLY")
    shutil.copyfile(plys[0], out_dir / "splat.ply")

    if args.keep_checkpoint:
        shutil.make_archive(str(out_dir / "model"), "zip", root_dir=str(train_out))
        with zipfile.ZipFile(out_dir / "processed_min.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("transforms.json", "sparse_pc.ply"):
                candidate = processed / name
                if candidate.exists():
                    archive.write(candidate, name)
            db_candidate = processed / "colmap" / "database.db"
            if db_candidate.exists():
                archive.write(db_candidate, "colmap_database.db")
            # See main() for why we archive sparse/*.bin.
            sparse_dirs = sorted(
                {p.parent for p in processed.rglob("cameras.bin")},
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            final_sparse = next(
                (
                    d for d in sparse_dirs
                    if (d / "images.bin").exists() and (d / "points3D.bin").exists()
                ),
                None,
            )
            if final_sparse is not None:
                for name in ("cameras.bin", "images.bin", "points3D.bin"):
                    archive.write(final_sparse / name, f"sparse/{name}")
        print("[recon] checkpoint archived (model.zip + processed_min.zip)", flush=True)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "frames": frame_count,
                "registered_images": registered,
                "matching_method": matching_used,
                "train_args": json.loads(args.train_args),
                "timings": timings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[recon] complete", flush=True)
    return 0


def main() -> int:  # noqa: PLR0911, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--train-args", default="[]", help="JSON list of ns-train args")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument(
        "--refine-mode",
        action="store_true",
        help=(
            "Refine an existing splat: VW_IN must contain model.zip + "
            "processed_min.zip from a base run. Skips full COLMAP, uses "
            "image_registrator + ns-train --load-dir."
        ),
    )
    args = parser.parse_args()

    in_dir = Path(os.environ["VW_IN"])
    out_dir = Path(os.environ["VW_OUT"])
    work = Path("/tmp/recon")
    images = work / "images"
    processed = work / "processed"
    train_out = work / "train"
    export = work / "export"
    for folder in (images, processed, train_out, export):
        folder.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    frames_zip = in_dir / "frames.zip"
    if not frames_zip.exists():
        return fail(out_dir, "missing_input", "frames.zip not found in stage inputs")
    with zipfile.ZipFile(frames_zip) as archive:
        archive.extractall(images)
    frame_count = len(list(images.rglob("*.png"))) + len(list(images.rglob("*.jpg")))
    print(f"[recon] {frame_count} frames extracted", flush=True)

    if args.refine_mode:
        # Restore base state, register new images, regenerate transforms.json,
        # then jump straight to training with --load-dir. Skips the standard
        # ns-process-data + retry_mapper + vocab-tree-fallback dance.
        started = time.monotonic()
        try:
            load_dir, new_imgs = refine_mode_setup(in_dir, processed, train_out)
        except Exception as exc:  # noqa: BLE001
            return fail(out_dir, "refine_setup_failed", str(exc))
        timings["refine_setup_s"] = round(time.monotonic() - started, 1)
        started = time.monotonic()
        vocab_tree_path = ensure_vocab_tree()
        registered = refine_register_new_images(
            processed, new_imgs, vocab_tree_path, out_dir=out_dir,
        )
        timings["refine_register_s"] = round(time.monotonic() - started, 1)
        if registered == 0:
            return fail(
                out_dir,
                "refine_registration_failed",
                "image_registrator could not add the new images to the base "
                "reconstruction. Common causes: lighting drift too large for "
                "SIFT matching, or new captures don't overlap the base scene.",
            )
        matching_used = "refine+image_registrator"
        print(f"[refine] joint registered images: {registered}/{frame_count}", flush=True)

        # Archive the augmented database immediately so a downstream failure
        # still leaves us with usable diagnostics.
        if (processed / "colmap" / "database.db").exists():
            try:
                shutil.copyfile(
                    processed / "colmap" / "database.db",
                    out_dir / "colmap_database.db",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[refine] db archive failed: {exc}", flush=True)

        train_args = filter_supported_flags(json.loads(args.train_args))
        started = time.monotonic()
        train = run([
            "ns-train", "splatfacto",
            "--data", str(processed),
            "--output-dir", str(train_out),
            "--load-dir", str(load_dir / "nerfstudio_models"),
            *train_args,
            "nerfstudio-data",
            "--downscale-factor", str(args.downscale),
        ])
        timings["train_s"] = round(time.monotonic() - started, 1)
        if train.returncode != 0:
            return fail(out_dir, "training_failed", f"ns-train (resume) exit {train.returncode}")
        # Fall through to export/archive logic below — share with standard mode.
        # Done by jumping past the standard flow and into the post-train section.
        return finish_export(
            args, out_dir, processed, train_out, export, timings,
            frame_count=frame_count, registered=registered, matching_used=matching_used,
        )

    registered = 0
    matching_used = ""
    # Per-job minimum: 50% of the input frame count, or MIN_REGISTERED_IMAGES,
    # whichever is larger. Under-registered solves feed garbage geometry into
    # splatfacto and waste training compute; fail fast instead.
    min_registered = max(MIN_REGISTERED_IMAGES, int(round(frame_count * MIN_REGISTERED_RATIO)))
    print(f"[recon] registration target: {min_registered}/{frame_count} images", flush=True)

    # Sequential first (fast, right for video).
    if processed.exists():
        shutil.rmtree(processed)
    processed.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = run([
        "ns-process-data", "images",
        "--data", str(images),
        "--output-dir", str(processed),
        "--matching-method", "sequential",
        "--num-downscales", "3",
        # GPU SIFT: previously hard-coded to --no-gpu because the L4 container's
        # COLMAP CUDA path was failing silently. Re-enabling now to see whether
        # the current image works — CPU SIFT was costing ~5x in dollars at L4
        # prices. Failure here is loud (process exit), unlike the silent
        # degradation we used to see.
    ])
    timings["process_data_sequential_s"] = round(time.monotonic() - started, 1)
    if process.returncode != 0:
        return fail(out_dir, "process_data_failed", f"ns-process-data exit {process.returncode}")
    registered = count_registered_images(processed)
    matching_used = "sequential"
    stats = colmap_db_stats(processed)
    print(f"[recon] registered images (sequential): {registered}/{frame_count}", flush=True)
    print(f"[recon] colmap db stats: {json.dumps(stats)}", flush=True)

    # Ship the COLMAP database to $VW_OUT immediately after matching so even
    # a downstream failure (mapper, training, export) returns the DB for
    # local audit. processed_min.zip later in the success path also includes
    # it, but this copy guarantees we get it on any failure mode.
    db_source = processed / "colmap" / "database.db"
    if db_source.exists():
        try:
            shutil.copyfile(db_source, out_dir / "colmap_database.db")
            print(f"[recon] database.db archived to out_dir ({db_source.stat().st_size // 1_000_000} MB)", flush=True)
        except Exception as exc:  # noqa: BLE001 - audit copy is best-effort
            print(f"[recon] database.db archive failed: {exc}", flush=True)

    # On low registration, retry JUST the mapper with relaxed init thresholds
    # (mapper non-determinism, see retry_mapper) before paying for more matches.
    if registered < min_registered:
        started = time.monotonic()
        retried = retry_mapper(processed)
        timings["mapper_retry_sequential_s"] = round(time.monotonic() - started, 1)
        if retried >= min_registered:
            registered = retried
            matching_used = "sequential+mapper-retry"
            print(f"[recon] registered images ({matching_used}): {registered}/{frame_count}", flush=True)

    # If retry_mapper still couldn't recover, augment the existing database
    # with vocab-tree matches (finds loop closures sequential misses) and try
    # mapper again. We invoke `colmap vocab_tree_matcher` directly because
    # ns-process-data's --vocab-tree-path flag doesn't exist in the worker's
    # nerfstudio version — calling colmap straight is version-stable.
    if registered < min_registered:
        vocab_tree_path = ensure_vocab_tree()
        if vocab_tree_path is not None:
            started = time.monotonic()
            vt_registered = vocab_tree_augment_and_remap(processed, vocab_tree_path)
            timings["vocab_tree_match_s"] = round(time.monotonic() - started, 1)
            if vt_registered >= min_registered:
                registered = vt_registered
                matching_used = "sequential+vocab_tree+mapper-retry"
                print(f"[recon] registered images ({matching_used}): {registered}/{frame_count}", flush=True)
            else:
                print(f"[recon] vocab_tree augmentation reached {vt_registered}, still under {min_registered}", flush=True)
            # Refresh the out_dir copy so the audit DB reflects vocab_tree
            # additions (config tags for new pairs are useful diagnostic data).
            if db_source.exists():
                try:
                    shutil.copyfile(db_source, out_dir / "colmap_database.db")
                except Exception:  # noqa: BLE001 - best-effort
                    pass

    if registered < min_registered:
        return fail(
            out_dir,
            "too_few_registered_images",
            f"COLMAP registered only {registered}/{frame_count} images "
            f"(need >= {min_registered} = {int(MIN_REGISTERED_RATIO * 100)}% of input). "
            "This usually means the scene is feature-poor for SfM. Capture tips: "
            "POINT AT TEXTURED SUBJECTS (plants, structures, brick, mulch — avoid grass, "
            "sky, plain pavement), WALK around them slowly with ~70% frame overlap, keep "
            "lighting consistent (cloudy beats sunny), avoid pure rotation pans and any "
            "shots through glass, water, or mirrors.",
        )

    train_args = filter_supported_flags(json.loads(args.train_args))
    started = time.monotonic()
    train = run([
        "ns-train", "splatfacto",
        "--data", str(processed),
        "--output-dir", str(train_out),
        *train_args,
        # Dataparser args go after the dataparser name in nerfstudio CLIs.
        "nerfstudio-data",
        "--downscale-factor", str(args.downscale),
    ])
    timings["train_s"] = round(time.monotonic() - started, 1)
    if train.returncode != 0:
        return fail(out_dir, "training_failed", f"ns-train exit {train.returncode}")

    return finish_export(
        args, out_dir, processed, train_out, export, timings,
        frame_count=frame_count, registered=registered, matching_used=matching_used,
    )


if __name__ == "__main__":
    sys.exit(main())
