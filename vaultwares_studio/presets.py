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
    # Split-job optimization: run COLMAP on cpu-upgrade, training on `flavor`.
    # When True, sfm_flavor and sfm_est_minutes define the first job; the
    # existing flavor/est_minutes apply only to the training job.
    split_jobs: bool = False
    sfm_flavor: str = "cpu-upgrade"
    sfm_est_minutes: float = 0.0  # 0 = not split

    def train_args(self) -> list[str]:
        """splatfacto arguments shared by local and remote execution."""
        args = [
            "--max-num-iterations", str(self.iterations),
            "--vis", "none",
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.datamanager.cache-images", "cpu",
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
