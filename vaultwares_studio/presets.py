"""Reconstruction quality presets.

Each preset picks splatfacto training parameters AND the HF Jobs flavor that
fits them, plus a cost estimate for the consent dialog. LOCAL_DEBUG preserves
the historical 250-iteration quick path used when reconstruction runs on the
local machine (heavy local training stays opt-in; the PC hosts the
VaultWares API).

Flag spelling drifts between nerfstudio releases — the worker entrypoint
probes ``ns-train splatfacto --help`` and drops unknown flags rather than
failing the job.

Split-job presets (split_jobs=True) run COLMAP on a cheap cpu-upgrade
instance first, then hand the processed_min.zip to a GPU job for training
only. This eliminates paying L4 rates ($0.80/hr) for CPU-only COLMAP work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .runners import CostEstimate, RATE_TABLE_SOURCE, estimate_cost


@dataclass(frozen=True)
class QualityPreset:
    key: str
    label: str
    iterations: int
    downscale_factor: int
    gaussian_cap: int
    flavor: str
    est_minutes: float
    extra_train_args: tuple[str, ...] = field(default_factory=tuple)
    # Checkpoint cadence for ns-train --steps-per-save. Production default
    # is 1000; lab presets can widen this (fewer ckpts, smaller model.zip)
    # or shrink it (more snapshots through cull boundaries).
    steps_per_save: int = 1000
    # Split-job optimization: run COLMAP on cpu-upgrade, training on `flavor`.
    # When True, sfm_flavor and sfm_est_minutes define the first job; the
    # existing flavor/est_minutes apply only to the training job.
    split_jobs: bool = False
    sfm_flavor: str = "cpu-upgrade"
    sfm_est_minutes: float = 0.0  # 0 = not split
    # Lab-mode toggles: skip ffmpeg sharpness pruning and hard-cap the post-
    # extract frame count. Used by the cpu-upgrade experiment Space to send
    # raw, unpruned dense frame sets to COLMAP. 0 = no cap (production default).
    unrestricted_frames: bool = False
    frame_cap: int = 0
    # Explicit SfM job timeout in seconds; 0 = default (max(3600, sfm_est*60*4)).
    # Needed when sfm_est_minutes can't be trusted as an upper bound (lab runs).
    sfm_timeout_seconds: int = 0
    # Image used by the SfM (Job A) leg. Empty string = use the runner's default
    # worker_image. Lab presets point this at the experimental Space so prod is
    # never touched.
    sfm_image_override: str = ""

    def train_args(self) -> list[str]:
        """splatfacto arguments shared by local and remote execution."""
        args = [
            "--max-num-iterations", str(self.iterations),
            "--vis", "none",
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.datamanager.cache-images", "cpu",
            # Save a checkpoint every 1000 steps instead of only the final
            # one — lets us extract splat snapshots at any step retroactively
            # via ns-export --load-step N. Stitched-video case at step 2900
            # had 1.1M GSs that disappeared after the cull at 3400; only the
            # final checkpoint was kept, so the high-density snapshot was
            # unrecoverable. Each .ckpt is ~150-300 MB; 15 checkpoints in
            # standard run ≈ ~3 GB extra in model.zip. Worth it.
            "--steps-per-save", str(self.steps_per_save),
        ]
        args.extend(self.extra_train_args)
        return args

    def cost(self) -> CostEstimate:
        """Combined cost estimate. For split presets this sums both jobs."""
        if self.split_jobs and self.sfm_est_minutes > 0:
            sfm = estimate_cost(self.sfm_flavor, self.sfm_est_minutes)
            train = estimate_cost(self.flavor, self.est_minutes)
            return CostEstimate(
                flavor=f"{self.sfm_flavor}+{self.flavor}",
                est_minutes=self.sfm_est_minutes + self.est_minutes,
                rate_usd_per_hour=0.0,  # mixed flavors; use est_usd directly
                est_usd=round(sfm.est_usd + train.est_usd, 2),
                source=RATE_TABLE_SOURCE,
            )
        return estimate_cost(self.flavor, self.est_minutes)

    def sfm_cost(self) -> CostEstimate:
        return estimate_cost(self.sfm_flavor, self.sfm_est_minutes)

    def train_cost(self) -> CostEstimate:
        return estimate_cost(self.flavor, self.est_minutes)


PRESETS: dict[str, QualityPreset] = {
    "draft": QualityPreset(
        key="draft",
        label="Draft (fast, low cost)",
        iterations=7_000,
        downscale_factor=4,
        gaussian_cap=300_000,
        flavor="l4x1",
        est_minutes=15,
        extra_train_args=("--pipeline.model.stop-split-at", "5000"),
    ),
    "standard": QualityPreset(
        key="standard",
        label="Standard",
        iterations=15_000,
        # No training downscale — splatfacto sees the full-res images so the
        # output gaussians keep the detail SfM already extracted at full res.
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=25,  # training-only time (was 60 combined before split)
        extra_train_args=("--pipeline.model.cull-alpha-thresh", "0.05"),
        split_jobs=True,
        sfm_flavor="cpu-upgrade",
        sfm_est_minutes=35,  # ns-process-data sequential + mapper on cpu-upgrade
    ),
    # Refine an existing splat: launcher passes --refine-from, worker resumes
    # from the base model.zip checkpoint via ns-train --load-dir.
    #
    # CPU/GPU breakdown (1500-frame joint set, measured):
    #   ~90 min  feature_extractor + vocab_tree matching + image_registrator  (CPU only)
    #   ~15 min  5k splatfacto resume iters                                  (GPU)
    #   ~15 min  boot / IO / model pull
    # Split path: cpu-upgrade handles SfM (~90 min @ $0.10/hr = $0.15),
    # l4x1 handles training-only (~20 min @ $0.80/hr = $0.27) = ~$0.42 total
    # vs $1.60 for the naive single-job path.
    "refine": QualityPreset(
        key="refine",
        label="Refine (resume from base checkpoint)",
        iterations=5_000,
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=20,  # training-only time (was 120 combined before split)
        extra_train_args=("--pipeline.model.cull-alpha-thresh", "0.05"),
        split_jobs=True,
        sfm_flavor="cpu-upgrade",
        sfm_est_minutes=90,  # refine COLMAP: feature_extractor + vocab_tree + image_registrator
    ),
    "high": QualityPreset(
        key="high",
        label="High (slow, best quality)",
        iterations=30_000,
        downscale_factor=2,
        gaussian_cap=1_500_000,
        flavor="a10g-large",
        est_minutes=75,
        extra_train_args=("--pipeline.model.rasterize-mode", "antialiased"),
    ),
    # Experimental: cpu-upgrade Space, dense unpruned frame set, hard-cap 3000.
    # Pairs with docker/lab/Dockerfile pushed via tools/push_lab_space.py.
    # Job B (GPU training) is intentionally left pointing at l4x1; queue_lab_recon.py
    # fires Job A only — the GPU half is iterated separately in the HF console.
    "lab-cpu-3000": QualityPreset(
        key="lab-cpu-3000",
        label="Lab: cpu-upgrade SfM, 3000 unpruned frames",
        iterations=15_000,
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=25,
        split_jobs=True,
        sfm_flavor="cpu-upgrade",
        sfm_est_minutes=240,  # 4h labelled budget; sfm_timeout_seconds bumps the actual cap to 12h
        unrestricted_frames=True,
        frame_cap=3000,
        sfm_timeout_seconds=21_600,  # 6h ceiling (HF account caps 12h+ as 500)
        sfm_image_override="hf.co/spaces/{owner}/vw-studio-recon-lab",
        # Wider checkpoint cadence than prod (1500 vs 1000) — gets the 3000-step
        # snapshot near the splatfacto cull boundary while keeping model.zip
        # in the 6-9 GB range on a 15k-iter run.
        steps_per_save=1500,
    ),
    # Historical local quick path: a smoke-level run that proves the toolchain
    # without tying up the local GPU. Used when no remote runner is configured.
    "local-debug": QualityPreset(
        key="local-debug",
        label="Local debug (250 iterations)",
        iterations=250,
        downscale_factor=4,
        gaussian_cap=100_000,
        flavor="cpu-basic",  # unused locally; kept for completeness
        est_minutes=0,
    ),
}

DEFAULT_PRESET_KEY = "standard"


def get_preset(key: str | None) -> QualityPreset:
    return PRESETS.get((key or "").lower(), PRESETS[DEFAULT_PRESET_KEY])
