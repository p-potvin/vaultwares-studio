"""Run one July-baseline Splatfacto training call through the ZeroGPU Space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from da3_job_client import configure_storage
from vaultwares_studio.runners.hf_jobs import get_hf_token

SPACE_URL = "https://clopeux-vw-studio-da3-zerogpu.hf.space"


def call(client: httpx.Client, endpoint: str, data: list) -> list:
    headers = {"Authorization": f"Bearer {get_hf_token()}", "Content-Type": "application/json"}
    submitted = client.post(f"{SPACE_URL}/gradio_api/call/{endpoint}", headers=headers, json={"data": data})
    submitted.raise_for_status()
    event_id = submitted.json()["event_id"]
    events: list[str] = []
    event_kind = ""
    with client.stream("GET", f"{SPACE_URL}/gradio_api/call/{endpoint}/{event_id}", headers=headers) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            events.append(line)
            if line.startswith("event:"):
                event_kind = line.split(":", 1)[1].strip()
            if line.startswith("data:") and event_kind in {"complete", "error"}:
                payload = json.loads(line.split(":", 1)[1].strip())
                if event_kind == "error":
                    raise RuntimeError(f"Space {endpoint} failed: {payload}")
                return payload
    raise RuntimeError(f"Space {endpoint} ended without a result: {events[-10:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-job", default="local-run-20260713-022745")
    args = parser.parse_args()
    configure_storage()
    output = args.output.resolve()
    if output.drive.upper() != "D:":
        raise ValueError("Training artifacts must be saved on D:.")
    output.mkdir(parents=True, exist_ok=False)
    with httpx.Client(timeout=httpx.Timeout(connect=30, read=3600, write=3600, pool=30), follow_redirects=True) as client:
        prepared = call(client, "prepare_training", [args.base_job])
        work_dir, prepare_summary = prepared
        (output / "prepare_summary.json").write_text(prepare_summary, encoding="utf-8")
        trained = call(client, "train_gpu", [work_dir])
        remote_work, gpu_summary = trained
        (output / "gpu_summary.json").write_text(gpu_summary, encoding="utf-8")
        packaged = call(client, "package_training", [remote_work, gpu_summary])
        files, report, summary = packaged
        (output / "completion_summary.json").write_text(summary, encoding="utf-8")
        for item in files[:2]:
            url = item.get("url")
            if not url:
                raise RuntimeError(f"Missing returned training artifact URL: {item}")
            target = output / (item.get("orig_name") or Path(item["path"]).name)
            with client.stream("GET", url, headers={"Authorization": f"Bearer {get_hf_token()}"}) as response:
                response.raise_for_status()
                with target.open("wb") as stream:
                    for block in response.iter_bytes(1024 * 1024):
                        stream.write(block)
            if target.resolve().drive.upper() != "D:":
                raise RuntimeError("Returned artifact escaped D:.")
        print(json.dumps({"prepare": prepare_summary, "gpu": gpu_summary, "artifacts": [str(output / (x.get("orig_name") or Path(x["path"]).name)) for x in files[:2]], "completion": summary}, indent=2))


if __name__ == "__main__":
    main()
