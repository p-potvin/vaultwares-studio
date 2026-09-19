"""Submit one DA3 Space request and retain its returned artifacts on D:.

This holds one Gradio event stream open until completion. Do not use it for a
second request while an active event exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from da3_job_client import configure_storage
from vaultwares_studio.runners.hf_jobs import get_hf_token


SPACE_URL = "https://clopeux-vw-studio-da3-zerogpu.hf.space"


def _headers() -> dict[str, str]:
    token = get_hf_token()
    if not token:
        raise RuntimeError("No Hugging Face token available.")
    return {"Authorization": f"Bearer {token}"}


def _download(client: httpx.Client, file_data: dict, destination: Path) -> Path:
    url = file_data.get("url")
    if not url:
        raise RuntimeError(f"Gradio did not return a download URL: {file_data}")
    target = destination / (file_data.get("orig_name") or Path(file_data["path"]).name)
    with client.stream("GET", url, headers=_headers()) as response:
        response.raise_for_status()
        with target.open("wb") as output:
            for block in response.iter_bytes(1024 * 1024):
                output.write(block)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=["preview", "high"], default="high")
    parser.add_argument("--loop-closure", action="store_true")
    parser.add_argument("--frames", type=int, default=500, help="frames the console keeps (console caps at 1200)")
    parser.add_argument("--loop-similarity", type=float, default=0.85, help="SALAD loop-detection threshold")
    args = parser.parse_args()
    configure_storage()
    source = args.video.resolve(strict=True)
    output = args.output.resolve()
    if output.drive.upper() != "D:":
        raise ValueError("Space artifacts must be saved on D:.")
    output.mkdir(parents=True, exist_ok=False)
    headers = _headers()
    timeout = httpx.Timeout(connect=30, read=3600, write=3600, pool=30)
    with source.open("rb") as stream, httpx.Client(timeout=timeout, follow_redirects=True) as client:
        uploaded = client.post(f"{SPACE_URL}/gradio_api/upload", headers=headers, files={"files": (source.name, stream, "video/quicktime")})
        uploaded.raise_for_status()
        server_path = uploaded.json()[0]
        request = {"data": [{"video": {"path": server_path, "meta": {"_type": "gradio.FileData"}}, "subtitles": None}, args.preset, args.loop_closure, args.frames, args.loop_similarity]}
        submitted = client.post(f"{SPACE_URL}/gradio_api/call/run", headers={**headers, "Content-Type": "application/json"}, json=request)
        submitted.raise_for_status()
        event_id = submitted.json()["event_id"]
        (output / "submission.json").write_text(json.dumps({"event_id": event_id, "preset": args.preset, "loop_closure": args.loop_closure, "frames": args.frames, "loop_similarity": args.loop_similarity, "source": str(source)}, indent=2), encoding="utf-8")
        events: list[str] = []; current_event = ""; completed = None
        with client.stream("GET", f"{SPACE_URL}/gradio_api/call/run/{event_id}", headers=headers) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                events.append(line)
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                if line.startswith("data:") and current_event in {"complete", "error"}:
                    completed = (current_event, json.loads(line.split(":", 1)[1].strip()))
                    break
        (output / "events.log").write_text("\n".join(events), encoding="utf-8")
        if completed is None:
            raise RuntimeError("Gradio stream ended before a complete or error event.")
        kind, data = completed
        if kind != "complete":
            raise RuntimeError(f"Space run failed: {data}")
        files = [_download(client, item, output) for item in data[:2]]
        (output / "completion.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps({"event_id": event_id, "artifacts": [str(path) for path in files], "summary": data[2]}, indent=2))


if __name__ == "__main__":
    main()
