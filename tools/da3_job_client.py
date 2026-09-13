"""One approved prepared job at a time; all workstation HF storage is on D:.

No automatic retries, cancellation, second-job submission, or polling loop.
Each status/logs/download invocation is an explicit action on the saved job ID.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STORAGE = Path("D:/vaultwares-studio-jobs/hf-storage")


def configure_storage() -> dict:
    locations = {
        "HF_HOME": STORAGE / "home", "HF_HUB_CACHE": STORAGE / "hub",
        "HUGGINGFACE_HUB_CACHE": STORAGE / "hub", "HF_XET_CACHE": STORAGE / "xet",
        "HF_ASSETS_CACHE": STORAGE / "assets", "HF_DATASETS_CACHE": STORAGE / "datasets",
        "TORCH_HOME": STORAGE / "torch", "XDG_CACHE_HOME": STORAGE / "cache",
        "TMP": STORAGE / "tmp", "TEMP": STORAGE / "tmp", "TMPDIR": STORAGE / "tmp",
    }
    for name, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        if path.resolve().drive.upper() != "D:":
            raise ValueError(f"HF storage escaped D: for {name}")
        os.environ[name] = str(path)
    tempfile.tempdir = str(locations["TMP"])
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return {k: str(v.resolve()) for k, v in locations.items()}


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def authorize_first(plan_path: Path, timeout_minutes: int | None = None) -> dict:
    from tools.prepare_da3_comparison import package_worker, job_command
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan_path.parent / "loop-off" / "submission_intent.json").exists():
        raise RuntimeError("A submission intent already exists; inspect it instead of resubmitting.")
    plan.update(status="APPROVED_FIRST_ONLY", approved_variants=["loop-off"],
                maximum_compute_usd=timeout_minutes / 60 * plan["hourly_usd"] if timeout_minutes else None,
                backend_timeout_seconds_per_job=timeout_minutes * 60 if timeout_minutes else None,
                worker_timeout_seconds=None, queue_timeout_seconds_per_job=None,
                maximum_status_checks_per_job=None, submit_second_only_if_first_succeeds=False,
                second_job_requires_new_user_instruction=True,
                approved_at=datetime.now().astimezone().strftime("%a, %d %b %Y %H:%M"))
    plan["worker_zip_sha256"] = package_worker(plan_path.parent)
    for variant in plan["variants"]:
        variant["command"] = job_command(variant["worker_args"])
    write_json(plan_path, plan)
    return plan


def job_payload(plan: dict, config: dict, token: str, bootstrap: str) -> dict:
    image = plan["image"]
    space_prefix = "hf.co/spaces/"
    image_spec = {"spaceId": image.removeprefix(space_prefix)} if image.startswith(space_prefix) else {"dockerImage": image}
    return {
        **image_spec, "command": ["python", "-c", "import os;exec(os.environ['VW_BOOTSTRAP'])"],
        "arguments": [], "environment": {"VW_BOOTSTRAP": bootstrap, "VW_STAGE_CONFIG": json.dumps(config)},
        "secrets": {"HF_TOKEN": token}, "flavor": plan["flavor"],
        "timeoutSeconds": plan["backend_timeout_seconds_per_job"], "attempts": 1,
        "labels": {"name": "studio-img1274-loop-off", "studioExperiment": config["prefix"].split("/")[1]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["authorize-first", "submit", "submit-alternate", "resubmit-image-fix", "submit-loop-on", "status", "logs", "download", "storage"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--timeout-minutes", type=int, help="User-approved backend timeout for authorize-first")
    parser.add_argument("--run-name", help="Explicit local name for an approved alternate submission")
    parser.add_argument("--flavor", help="Explicit alternate hardware flavor")
    args = parser.parse_args()
    storage = configure_storage()  # Before importing HF, Xet, or project runners.
    plan_path = args.plan.resolve(strict=True)
    if plan_path.drive.upper() != "D:":
        raise ValueError("Prepared plan, downloads and logs must reside on D:.")
    if args.action == "storage":
        from huggingface_hub import constants
        print(json.dumps({"configured": storage, "effective_hub_cache": constants.HF_HUB_CACHE,
                          "effective_xet_cache": constants.HF_XET_CACHE,
                          "effective_temp": tempfile.gettempdir()}, indent=2))
        return
    if args.action == "authorize-first":
        plan = authorize_first(plan_path, args.timeout_minutes)
        print(json.dumps({"status": plan["status"], "approved_variants": plan["approved_variants"],
                          "timeout": plan["backend_timeout_seconds_per_job"], "storage_root": str(STORAGE)}))
        return
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    from vaultwares_studio.runners.hf_jobs import get_hf_token, HfJobsConfig, BOOTSTRAP_SOURCE
    from tools.prepare_da3_comparison import sha256
    import httpx
    token = get_hf_token()
    if not token:
        raise RuntimeError("No HF token available in the project keyring or environment.")
    api = HfApi(token=token)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    alternate = args.action in {"submit-alternate", "resubmit-image-fix", "submit-loop-on"}
    run_name = args.run_name or "loop-off"
    run_dir = plan_path.parent / run_name
    run_dir.mkdir(exist_ok=True)
    record_path = run_dir / "submission.json"
    if args.action in {"submit", "submit-alternate", "resubmit-image-fix", "submit-loop-on"}:
        if not alternate and (plan.get("status") != "APPROVED_FIRST_ONLY" or plan.get("approved_variants") != ["loop-off"]):
            raise RuntimeError("This client requires explicit first-job-only approval.")
        if args.action == "submit-alternate" and (args.flavor != "a10g-small" or run_name != "loop-off-a10g"):
            raise RuntimeError("Only the explicitly approved a10g-small loop-off alternate is permitted.")
        if args.action == "resubmit-image-fix" and (args.flavor != "a10g-small" or run_name != "loop-off-a10g-spaceid"):
            raise RuntimeError("Image-fix replacement is limited to a10g-small loop-off.")
        if args.action == "submit-loop-on" and not (
            (args.flavor == "a10g-small" and run_name == "loop-on-a10g-spaceid")
            or (args.flavor == "l4x1" and run_name == "loop-on-l4")
        ):
            raise RuntimeError("Loop comparison requires the approved a10g-small or l4x1 run name.")
        if record_path.exists() or (run_dir / "submission_intent.json").exists():
            raise RuntimeError("Submission exists or is uncertain; automatic resubmission is forbidden.")
        if sha256(plan_path.parent / "frames.zip") != plan["frames_zip_sha256"]:
            raise RuntimeError("Prepared frame archive changed.")
        from tools.prepare_da3_comparison import package_worker
        worker_hash = package_worker(plan_path.parent)
        space = api.space_info("clopeux/vw-studio-da3-gs")
        if space.sha != plan["observed_space_sha"]:
            raise RuntimeError("The image Space source changed since preparation; inspect before launch.")
        owner = api.whoami()["name"]
        config = HfJobsConfig.load()
        repo = config.artifact_repo or f"{owner}/vw-studio-artifacts"
        if not api.repo_info(repo, repo_type="dataset").private:
            raise RuntimeError("Video inputs require a private artifact dataset.")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment = (
            f"img1274-loop-on-a10g-spaceid-{stamp}" if args.action == "submit-loop-on" else
            f"img1274-loop-ab-a10g-spaceid-{stamp}" if args.action == "resubmit-image-fix" else
            f"img1274-loop-ab-a10g-{stamp}" if alternate else
            f"img1274-loop-ab-{stamp}"
        )
        prefix = f"jobs/{experiment}/reconstruction_sfm"
        api.create_commit(repo_id=repo, repo_type="dataset", commit_message="Stage approved DA3 loop-off inputs",
            operations=[
                CommitOperationAdd(path_in_repo=f"{prefix}/in/frames.zip", path_or_fileobj=str(plan_path.parent / "frames.zip")),
                CommitOperationAdd(path_in_repo=f"{prefix}/in/worker.zip", path_or_fileobj=str(plan_path.parent / "worker.zip")),
            ])
        variant = next(item for item in plan["variants"] if item["name"] == ("loop-on" if args.action == "submit-loop-on" else "loop-off"))
        config_payload = {"repo": repo, "prefix": prefix, "command": variant["command"], "extra_inputs": []}
        job_plan = {**plan, "flavor": args.flavor} if alternate else plan
        payload = job_payload(job_plan, config_payload, token, BOOTSTRAP_SOURCE)
        payload["labels"]["name"] = f"studio-img1274-{variant['name']}-{job_plan['flavor']}"
        intent = {"repo": repo, "prefix": prefix, "owner": owner, "variant": variant["name"],
                  "flavor": job_plan["flavor"], "worker_zip_sha256": worker_hash,
                  "requested_timeout_seconds": plan["backend_timeout_seconds_per_job"], "storage": storage,
                  "submitted_at": datetime.now().astimezone().strftime("%a, %d %b %Y %H:%M")}
        write_json(run_dir / "submission_intent.json", intent)
        # Explicit null is supported by the live Jobs API schema. The SDK omits
        # timeout=None; omitted timeouts can inherit the documented 30m default.
        with httpx.Client(timeout=120) as client:
            response = client.post(f"https://huggingface.co/api/jobs/{owner}",
                                   headers={"Authorization": f"Bearer {token}"}, json=payload)
        response.raise_for_status()
        created = response.json()
        record = {**intent, "job_id": created["id"],
                  "job_url": f"https://huggingface.co/jobs/{owner}/{created['id']}",
                  "timeout_seconds": created.get("timeout", created.get("timeoutSeconds")), "status": created.get("status")}
        write_json(record_path, record)
        plan["status"] = "FIRST_JOB_SUBMITTED"
        write_json(plan_path, plan)
        print(json.dumps(record, indent=2))
        return
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if args.action == "status":
        info = api.inspect_job(job_id=record["job_id"], namespace=record["owner"])
        state = {"job_id": info.id, "status": str(info.status.stage), "message": info.status.message,
                 "checked_at": datetime.now().astimezone().strftime("%a, %d %b %Y %H:%M"),
                 "timeout_seconds": record["timeout_seconds"]}
        write_json(run_dir / "status.json", state)
        print(json.dumps(state, indent=2))
    elif args.action == "logs":
        lines = list(api.fetch_job_logs(job_id=record["job_id"], namespace=record["owner"], tail=80, follow=False))
        safe = "\n".join(lines).replace(token, "[REDACTED]")
        (run_dir / "remote-tail.log").write_text(safe, encoding="utf-8")
        print(safe)
    elif args.action == "download":
        remote_prefix = record["prefix"] + "/out/"
        files = [f for f in api.list_repo_files(record["repo"], repo_type="dataset") if f.startswith(remote_prefix)]
        for remote in files:
            target = hf_hub_download(repo_id=record["repo"], repo_type="dataset", filename=remote,
                                     local_dir=str(run_dir / "download"), cache_dir=str(STORAGE / "hub"), token=token)
            if Path(target).resolve().drive.upper() != "D:":
                raise RuntimeError("Downloaded artifact did not resolve to D:.")
            print(str(Path(target).resolve()))


if __name__ == "__main__":
    main()
