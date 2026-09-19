"""Ask the worker image what it actually contains, for about a fifth of a cent.

Every expensive mistake in this pipeline so far came from assuming something
about the container instead of asking it. The image was built in July; whether
its nerfstudio knows `--pipeline.model.strategy mcmc` decides whether the MCMC
plan is a config change or an image rebuild, and `filter_supported_flags` would
silently drop the flag and carry on if it does not.

So this runs a throwaway CPU job that prints what is installed and exits. No
inputs, no outputs, nothing to download. On ``cpu-upgrade`` at about $0.04/hr a
two-minute probe costs well under a cent, which is cheaper than being wrong
about it once.

    python tools/probe_worker_env.py --yes
    python tools/probe_worker_env.py --image hf.co/spaces/clopeux/vw-studio-recon-lab --yes

Read the answers with ``tools/tail_job_logs.py --job-id <id>``.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Runs inside the container. Deliberately defensive: a probe that raises on the
# first missing import tells you about that import and nothing else.
PROBE = textwrap.dedent(
    '''
    import json, os, shutil, subprocess, sys

    def line(k, v):
        print(f"[probe] {k}: {v}", flush=True)

    line("python", sys.version.split()[0])
    try:
        import torch
        line("torch", torch.__version__)
        line("cuda_available", torch.cuda.is_available())
        sdpa = getattr(torch.nn.functional, "scaled_dot_product_attention", None)
        line("sdpa_present", sdpa is not None)
        try:
            from torch.backends.cuda import flash_sdp_enabled, mem_efficient_sdp_enabled
            line("flash_sdp_enabled", flash_sdp_enabled())
            line("mem_efficient_sdp_enabled", mem_efficient_sdp_enabled())
        except Exception as exc:
            line("sdp_backend_query", f"unavailable: {exc}")
    except Exception as exc:
        line("torch", f"MISSING: {exc}")

    for mod in ("nerfstudio", "gsplat", "numpy", "xformers"):
        try:
            m = __import__(mod)
            line(mod, getattr(m, "__version__", "present, no __version__"))
        except Exception as exc:
            line(mod, f"MISSING: {exc}")

    # The question this probe exists for.
    try:
        from nerfstudio.models.splatfacto import SplatfactoModelConfig as C
        import dataclasses
        names = {f.name for f in dataclasses.fields(C)}
        for want in ("strategy", "max_gs_num", "noise_lr", "mcmc_opacity_reg", "mcmc_scale_reg"):
            line(f"splatfacto.{want}", want in names)
        line("splatfacto_field_count", len(names))
    except Exception as exc:
        line("splatfacto_config", f"UNREADABLE: {exc}")

    # What the flag filter would actually keep.
    try:
        helptext = subprocess.run(["ns-train", "splatfacto", "--help"],
                                  capture_output=True, text=True, timeout=300)
        text = helptext.stdout + helptext.stderr
        line("ns_train_help_chars", len(text))
        for flag in ("--pipeline.model.strategy", "--pipeline.model.max-gs-num",
                     "--pipeline.model.stop-split-at", "--pipeline.model.cull-alpha-thresh",
                     "--vis"):
            line(f"help_has {flag}", flag in text)
    except Exception as exc:
        line("ns_train_help", f"FAILED: {exc}")

    colmap = shutil.which("colmap")
    line("colmap_path", colmap or "NOT ON PATH")
    if colmap:
        try:
            out = subprocess.run([colmap, "-h"], capture_output=True, text=True, timeout=120)
            first = (out.stdout + out.stderr).strip().splitlines()[:2]
            line("colmap_version", " | ".join(first))
        except Exception as exc:
            line("colmap_version", f"FAILED: {exc}")

    line("cpu_count", os.cpu_count())
    try:
        info = dict((l.split(":")[0], l.split()[1]) for l in open("/proc/meminfo") if ":" in l)
        line("mem_total_gb", round(int(info["MemTotal"]) / 1e6, 1))
        line("mem_available_gb", round(int(info["MemAvailable"]) / 1e6, 1))
    except Exception as exc:
        line("meminfo", f"unavailable: {exc}")
    try:
        cap = open("/sys/fs/cgroup/memory.max").read().strip()
        line("cgroup_memory_max", cap)
    except Exception:
        pass
    line("done", "probe complete")
    '''
).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="hf.co/spaces/clopeux/vw-studio-da3",
                        help="the image to interrogate; default is the training image")
    parser.add_argument("--flavor", default="cpu-upgrade")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--yes", action="store_true", help="approve the (tiny) cost and submit")
    args = parser.parse_args(argv)

    from huggingface_hub import run_job

    from vaultwares_studio.runners import estimate_cost, get_hf_token

    estimate = estimate_cost(args.flavor, 3.0)
    print(f"[probe] image={args.image} flavor={args.flavor} — 3 min would cost ~${estimate.est_usd:.3f}")
    if not args.yes:
        print("[probe] not submitting; pass --yes to approve.")
        return 0

    token = get_hf_token()
    job = run_job(
        image=args.image,
        command=["python", "-c", PROBE],
        flavor=args.flavor,
        timeout=args.timeout,
        token=token,
    )
    print(f"[probe] job started: {job.url}")
    print(f"[probe] read it with: python tools/tail_job_logs.py --job-id {job.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
