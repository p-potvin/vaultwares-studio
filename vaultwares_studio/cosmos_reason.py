"""Cosmos Reason: what is in the reconstruction, and where it is in the scene.

The last stage before a job is useful to anything downstream. A vision-language
model looks at the frames the capture cameras actually saw, names the objects
and surfaces, and each observation is **anchored in 3D** by unprojecting its
image position through that frame's DA3 depth into the scene world. Repeated
sightings of the same thing across views are clustered into one annotation.

    capture_cameras.json + frames.zip + streaming depth
        -> VLM per sampled view (label, kind, static, normalised bbox)
        -> unproject bbox centre through depth -> DA3 world -> scene world
        -> cluster across views
        -> cosmos/cosmos_annotations.json
        -> /World/Annotations/<slug> Xform prims in the USD stage

An anchored label is a navigation goal: the robot lab's ObjectNav needs a
name and a place, which is exactly what comes out of here.

**Providers.** The intended model is NVIDIA's Cosmos Reason, a VLM post-trained
for physical common sense. Three backends, same interface:

- ``nvidia`` — the NVIDIA API catalog (``integrate.api.nvidia.com``). Set
  ``model`` to ``nvidia/cosmos-reason2-8b`` when the account can reach it. As
  of Sun, 13 Sep 2026 that model answers 404 *Function not found for account*
  for this key — an NVCF entitlement, not a code problem — so the default is
  ``nvidia/nemotron-3-nano-omni-30b-a3b-reasoning``, which is reachable and is
  also a reasoning model. Switching is one flag.
- ``ollama`` — local GGUF, free, no network. ``qwen3-vl:2b`` today;
  Cosmos-Reason2-2B is the same architecture and drops in once pulled.
- ``none`` — no model call; the stage still writes the view plan and camera
  hand-off, so the pipeline never fails for want of an endpoint.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

SCHEMA = 1
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
COSMOS_MODEL = "nvidia/cosmos-reason2-8b"
OLLAMA_DEFAULT_MODEL = "qwen3-vl:2b"

KINDS = ("structure", "vegetation", "ground", "furniture", "vehicle", "equipment", "other")

PROMPT = """You are annotating one frame of a walkthrough video that has been reconstructed in 3D.

List the distinct physical objects and surfaces you can actually see. For each one give:
- "label": two or three words, specific ("wooden shed door", not "object")
- "kind": one of [{kinds}]
- "static": true if it is part of the fixed scene, false if it could move
- "bbox": [x0, y0, x1, y1] as fractions of the image width and height, 0 to 1, top-left origin
- "confidence": 0 to 1

Report at most {limit} entries, the most prominent first. Skip sky and skip anything you
are guessing at. Answer with JSON only: {{"objects": [...]}}"""


# ----------------------------------------------------------------------------
# providers


@dataclass
class Observation:
    view_index: int
    frame_index: int
    label: str
    kind: str
    static: bool
    confidence: float
    bbox: list[float]
    position: list[float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class VlmProvider:
    """One image plus one prompt in, raw text out."""

    name = "none"
    model = ""
    # Whether there is a model behind this at all. Checked by name rather than
    # by isinstance: a test double or a wrapper subclassing NullProvider would
    # otherwise be silently skipped.
    enabled = True

    def describe(self, image: bytes, prompt: str) -> str:
        raise NotImplementedError

    def summary(self) -> dict:
        return {"provider": self.name, "model": self.model}


class NullProvider(VlmProvider):
    enabled = False

    def describe(self, image: bytes, prompt: str) -> str:  # noqa: ARG002
        return ""


class NvidiaCatalogProvider(VlmProvider):
    """NVIDIA API catalog, OpenAI-shaped. Free tier is ~1000 calls at 40/min."""

    name = "nvidia"

    def __init__(self, model: str = NVIDIA_DEFAULT_MODEL, api_key: str | None = None,
                 timeout: float = 180.0, max_tokens: int = 700):
        self.model = model
        self.api_key = api_key or os.environ.get("NGC_API_KEY") or os.environ.get("NGC_CLI_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError("No NVIDIA API key (set NGC_API_KEY) for the nvidia provider.")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def describe(self, image: bytes, prompt: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
            "top_p": 0.7,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")}},
            ]}],
        }
        request = urllib.request.Request(
            NVIDIA_ENDPOINT, data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code == 404:
                raise RuntimeError(
                    f"{self.model} is not provisioned for this NVIDIA account "
                    f"(404 from the catalog): {detail}. Pick another --cosmos-model, "
                    "or accept the model's terms on build.nvidia.com."
                ) from None
            raise RuntimeError(f"NVIDIA catalog {exc.code}: {detail}") from None
        message = body["choices"][0]["message"]
        return message.get("content") or message.get("reasoning_content") or ""


class OllamaProvider(VlmProvider):
    """Local GGUF through Ollama. No network, no quota."""

    name = "ollama"

    def __init__(self, model: str = OLLAMA_DEFAULT_MODEL, url: str = "http://127.0.0.1:11434",
                 timeout: float = 600.0, num_ctx: int = 16384):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx

    def describe(self, image: bytes, prompt: str) -> str:
        payload = {
            "model": self.model, "stream": False, "keep_alive": "10m",
            "think": False,
            "options": {"num_ctx": self.num_ctx, "temperature": 0.2, "num_predict": 900},
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(image).decode("ascii")]}],
        }
        request = urllib.request.Request(
            self.url + "/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        message = body.get("message") or {}
        # Measured on qwen3-vl:2b: with think=false the answer can land in
        # `thinking` while `content` comes back empty.
        return message.get("content") or message.get("thinking") or ""


def build_provider(kind: str, model: str = "", **kwargs) -> VlmProvider:
    kind = (kind or "none").lower()
    if kind == "nvidia":
        return NvidiaCatalogProvider(model or NVIDIA_DEFAULT_MODEL, **kwargs)
    if kind == "ollama":
        return OllamaProvider(model or OLLAMA_DEFAULT_MODEL, **kwargs)
    if kind == "none":
        return NullProvider()
    raise ValueError(f"Unknown VLM provider '{kind}' (nvidia, ollama, none)")


# ----------------------------------------------------------------------------
# parsing


_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _strip_think(text: str) -> str:
    text = _THINK.sub("", text or "")
    opened = text.find("<think>")
    return (text[:opened] if opened >= 0 else text).strip()


def extract_json(text: str):
    """The JSON the model meant, out of whatever it wrapped it in."""
    text = _strip_think(text)
    for candidate in [match.group(1) for match in _FENCE.finditer(text)] + [text]:
        candidate = candidate.strip()
        openers = sorted(
            (pair for pair in (("{", "}"), ("[", "]")) if candidate.find(pair[0]) >= 0),
            key=lambda pair: candidate.find(pair[0]),
        )
        for opener, closer in openers:
            start, end = candidate.find(opener), candidate.rfind(closer)
            if end <= start:
                continue
            blob = candidate[start:end + 1]
            blob = re.sub(r",\s*([\]}])", r"\1", blob)
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                # A reply cut off by a token cap: keep the objects that closed.
                last = blob.rfind("}")
                if last < 0:
                    continue
                for tail in ("", "]", "}", "]}", "}]", "}]}"):
                    try:
                        return json.loads(blob[:last + 1] + tail)
                    except json.JSONDecodeError:
                        continue
    return None


def parse_objects(text: str, view_index: int, frame_index: int, limit: int = 12) -> list[Observation]:
    parsed = extract_json(text)
    if parsed is None:
        return []
    items = parsed.get("objects", parsed) if isinstance(parsed, dict) else parsed
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return []
    out: list[Observation] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("name") or "").strip()
        if not label or label.lower() in {"sky", "background"}:
            continue
        kind = str(item.get("kind") or "other").strip().lower()
        if kind not in KINDS:
            kind = next((k for k in KINDS if k in kind), "other")
        bbox = item.get("bbox") or item.get("box") or [0.25, 0.25, 0.75, 0.75]
        try:
            bbox = [float(v) for v in bbox][:4]
        except (TypeError, ValueError):
            bbox = [0.25, 0.25, 0.75, 0.75]
        if len(bbox) != 4:
            bbox = [0.25, 0.25, 0.75, 0.75]
        # Some models answer in pixels rather than fractions.
        if max(bbox) > 1.5:
            scale = max(bbox)
            bbox = [v / scale for v in bbox]
        bbox = [min(max(v, 0.0), 1.0) for v in bbox]
        if bbox[2] < bbox[0]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if bbox[3] < bbox[1]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
        try:
            confidence = min(max(float(item.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            confidence = 0.5
        out.append(Observation(
            view_index=view_index, frame_index=frame_index, label=label, kind=kind,
            static=bool(item.get("static", True)), confidence=confidence, bbox=bbox,
        ))
    return out


# ----------------------------------------------------------------------------
# 3D anchoring


class DepthAnchor:
    """Unproject an image position through DA3's depth into the scene world.

    Same arithmetic as the TSDF fusion: the retained per-frame depth times its
    chunk scale, seen from that frame's global camera-to-world, then the
    scene transform. Returns ``None`` where the depth is missing or the
    confidence at that pixel is too low to trust.
    """

    def __init__(self, streaming_dir: Path, scene_transform: np.ndarray | None = None,
                 conf_coef: float = 0.5):
        self.dir = Path(streaming_dir)
        self.results = self.dir / "results_output"
        self.scene = np.eye(4) if scene_transform is None else np.asarray(scene_transform, dtype=float)
        self.conf_coef = conf_coef
        poses = np.loadtxt(self.dir / "camera_poses.txt", dtype=np.float64)
        self.poses = poses.reshape(-1, 4, 4) if poses.ndim > 1 else poses.reshape(1, 4, 4)
        self._cache: dict[int, tuple] = {}

    @property
    def available(self) -> bool:
        return self.results.is_dir() and any(self.results.glob("frame_*.npz"))

    def _frame(self, index: int):
        if index not in self._cache:
            path = self.results / f"frame_{index}.npz"
            if not path.exists():
                self._cache[index] = None
            else:
                with np.load(path) as data:
                    scale = float(np.asarray(data["s"])) if "s" in data.files else 1.0
                    depth = np.asarray(data["depth"], dtype=np.float32) * np.float32(scale)
                    conf = np.asarray(data["conf"], dtype=np.float32) if "conf" in data.files else None
                    K = np.asarray(data["intrinsics"], dtype=np.float64)
                self._cache[index] = (depth, conf, K)
        return self._cache[index]

    def anchor(self, frame_index: int, bbox: list[float]) -> list[float] | None:
        frame = self._frame(frame_index)
        if frame is None or frame_index >= len(self.poses):
            return None
        depth, conf, K = frame
        height, width = depth.shape
        u = (bbox[0] + bbox[2]) / 2 * width
        v = (bbox[1] + bbox[3]) / 2 * height
        # Median over a small window: a single pixel can land on a depth hole.
        half = max(2, int(0.02 * min(width, height)))
        col = slice(max(0, int(u) - half), min(width, int(u) + half + 1))
        row = slice(max(0, int(v) - half), min(height, int(v) + half + 1))
        patch = depth[row, col]
        valid = np.isfinite(patch) & (patch > 0)
        if conf is not None:
            valid &= conf[row, col] >= self.conf_coef * float(conf.mean())
        if not valid.any():
            return None
        z = float(np.median(patch[valid]))
        point = np.array([
            (u - K[0, 2]) * z / K[0, 0],
            (v - K[1, 2]) * z / K[1, 1],
            z, 1.0,
        ])
        world = self.poses[frame_index] @ point
        scene = self.scene @ world
        return [round(float(v), 5) for v in scene[:3]]


def slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "object"


# Words that carry no identity: two labels differing only by these are the
# same thing described twice.
_FILLER = frozenset({"a", "an", "the", "of", "with", "and", "large", "small",
                     "medium", "big", "tall", "short", "some", "several"})


def _tokens(label: str) -> frozenset[str]:
    words = {w for w in re.split(r"[^a-z0-9]+", label.lower()) if w}
    return frozenset(words - _FILLER) or frozenset(words)


def _same_thing(left: str, right: str) -> bool:
    """Do two labels name the same object?

    Word sets, not substrings. 'Blue door' and 'blue screen door' are one
    door — the first label's words are all in the second — but a substring
    test misses that, and it was missing it on real output. 'large green
    hedge' and 'medium green bush' share only a colour, so they stay apart.
    """
    a, b = _tokens(left), _tokens(right)
    if a == b:
        return True
    return a < b or b < a  # one description is a refinement of the other


@dataclass
class Annotation:
    label: str
    kind: str
    static: bool
    confidence: float
    observations: int
    position: list[float] | None = None
    spread: float | None = None
    frames: list[int] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["id"] = slugify(self.label)
        return payload


def cluster(
    observations: list[Observation],
    radius: float = 0.12,
    scene_extent: float | None = None,
) -> list[Annotation]:
    """One annotation per thing: same label, and anchors that sit together.

    Labels are matched on their slug plus a loose containment test ('shed door'
    and 'wooden shed door' are one thing); anchored observations additionally
    have to fall within ``radius`` of the running centroid, as a fraction of
    ``scene_extent``, so two different windows on opposite walls stay separate.

    ``scene_extent`` must be the size of the *scene* — the caller passes the
    camera-path extent. Falling back to the spread of the anchors themselves is
    only a last resort: in a scene where every sighting is of one object, that
    spread *is* the jitter being merged, and normalising by it would split
    every group.
    """
    groups: list[dict] = []
    if scene_extent and scene_extent > 0:
        extent = float(scene_extent)
    else:
        anchored = [np.asarray(o.position) for o in observations if o.position]
        stack = np.stack(anchored) if anchored else None
        extent = float(np.linalg.norm(stack.max(0) - stack.min(0))) if stack is not None else 1.0
        extent = extent or 1.0
    for observation in sorted(observations, key=lambda o: -o.confidence):
        target = None
        for group in groups:
            if not any(_same_thing(observation.label, name) for name in group["names"]):
                continue
            if observation.position and group["points"]:
                centre = np.mean(np.stack(group["points"]), axis=0)
                if np.linalg.norm(np.asarray(observation.position) - centre) > radius * extent:
                    continue
            target = group
            break
        if target is None:
            target = {"names": [], "obs": [], "points": []}
            groups.append(target)
        target["names"].append(observation.label)
        target["obs"].append(observation)
        if observation.position:
            target["points"].append(np.asarray(observation.position, dtype=float))

    out: list[Annotation] = []
    for group in groups:
        obs = group["obs"]
        best = max(obs, key=lambda o: o.confidence)
        points = group["points"]
        position = spread = None
        if points:
            stack = np.stack(points)
            position = [round(float(v), 5) for v in np.median(stack, axis=0)]
            spread = round(float(np.linalg.norm(stack.std(axis=0))), 5)
        out.append(Annotation(
            label=best.label, kind=best.kind, static=all(o.static for o in obs),
            confidence=round(float(np.mean([o.confidence for o in obs])), 3),
            observations=len(obs), position=position, spread=spread,
            frames=sorted({o.frame_index for o in obs}),
            aliases=sorted({o.label for o in obs} - {best.label}),
        ))
    out.sort(key=lambda a: (-a.observations, -a.confidence))
    return out


# ----------------------------------------------------------------------------
# images


class FrameSource:
    """The frames the capture cameras saw, from frames.zip or a frames/ dir."""

    def __init__(self, job_dir: Path):
        self.zip_path = job_dir / "reconstruction" / "remote_out" / "frames.zip"
        self.dir = job_dir / "frames"
        self._zip: zipfile.ZipFile | None = None
        self._names: list[str] = []
        if self.zip_path.exists():
            self._zip = zipfile.ZipFile(self.zip_path)
            self._names = sorted(n for n in self._zip.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")))
        elif self.dir.is_dir():
            self._names = [p.name for p in sorted(self.dir.glob("*.jpg")) + sorted(self.dir.glob("*.png"))]

    def __len__(self) -> int:
        return len(self._names)

    def read(self, index: int, max_side: int = 896, quality: int = 85) -> bytes | None:
        if not 0 <= index < len(self._names):
            return None
        name = self._names[index]
        raw = self._zip.read(name) if self._zip else (self.dir / name).read_bytes()
        try:
            from PIL import Image

            with Image.open(io.BytesIO(raw)) as image:
                image = image.convert("RGB")
                image.thumbnail((max_side, max_side))
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=quality)
                return buffer.getvalue()
        except Exception:  # noqa: BLE001 - ship the original rather than nothing
            return raw

    def close(self) -> None:
        if self._zip:
            self._zip.close()


def plan_views(frame_count: int, views: int) -> list[int]:
    """Evenly spaced frames, both ends included."""
    views = max(1, min(int(views), frame_count))
    if views == 1:
        return [frame_count // 2]
    return [min(frame_count - 1, round(i * (frame_count - 1) / (views - 1))) for i in range(views)]


# ----------------------------------------------------------------------------
# the pass


def annotate_job(
    job_dir: Path,
    provider: VlmProvider,
    *,
    views: int = 12,
    per_view_limit: int = 8,
    rate_limit_per_minute: float = 40.0,
    log: Callable[[str], None] = print,
) -> dict:
    """Run the VLM over sampled views and return the annotation payload."""
    from .camera_scene import scene_frame_transform
    from .capture_cameras import load_capture_cameras_json

    job_dir = Path(job_dir)
    cameras = load_capture_cameras_json(job_dir)
    frames = FrameSource(job_dir)
    streaming = job_dir / "reconstruction" / "remote_out" / "streaming"
    anchor: DepthAnchor | None = None
    if (streaming / "camera_poses.txt").exists():
        try:
            candidate = DepthAnchor(streaming, scene_frame_transform(job_dir))
            anchor = candidate if candidate.available else None
        except Exception as exc:  # noqa: BLE001 - anchoring is an enhancement
            log(f"[cosmos] depth anchoring unavailable: {exc}")

    count = len(cameras) or len(frames)
    if not count:
        raise RuntimeError(f"{job_dir.name} has no capture cameras and no frames to annotate.")
    plan = plan_views(count, views)
    prompt = PROMPT.format(kinds=", ".join(KINDS), limit=per_view_limit)
    observations: list[Observation] = []
    failures: list[dict] = []
    interval = 60.0 / rate_limit_per_minute if rate_limit_per_minute > 0 else 0.0
    started = time.time()
    last_call = 0.0

    for view_index, frame_index in enumerate(plan):
        image = frames.read(frame_index)
        if image is None:
            failures.append({"frame": frame_index, "error": "no image"})
            continue
        if not provider.enabled:
            continue
        wait = interval - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.time()
        try:
            text = provider.describe(image, prompt)
        except Exception as exc:  # noqa: BLE001 - one bad view must not kill the pass
            failures.append({"frame": frame_index, "error": str(exc)[:300]})
            log(f"[cosmos] view {view_index + 1}/{len(plan)} (frame {frame_index}) failed: {str(exc)[:160]}")
            if len(failures) >= 3 and not observations:
                raise RuntimeError(f"Cosmos provider failing repeatedly: {failures[-1]['error']}") from None
            continue
        found = parse_objects(text, view_index, frame_index, per_view_limit)
        for observation in found:
            if anchor is not None:
                observation.position = anchor.anchor(frame_index, observation.bbox)
        observations.extend(found)
        log(f"[cosmos] view {view_index + 1}/{len(plan)} (frame {frame_index}): "
            f"{len(found)} objects, {sum(1 for o in found if o.position)} anchored")

    # The camera path is the scene's own yardstick; anchors that sit within a
    # small fraction of it are the same thing seen twice.
    scene_extent = None
    if len(cameras) >= 2:
        positions = np.stack([camera.position for camera in cameras])
        scene_extent = float(np.linalg.norm(positions.max(0) - positions.min(0))) or None
    annotations = cluster(observations, scene_extent=scene_extent)
    frames.close()
    payload = {
        "schema": SCHEMA,
        "generated": time.strftime("%a, %d %b %Y %H:%M"),
        "source": {
            "job_dir": str(job_dir),
            "views_requested": views,
            "frames_sampled": plan,
            "capture_cameras": len(cameras),
            "anchoring": "da3-depth" if anchor is not None else "none",
        },
        **provider.summary(),
        "conventions": {
            "position": "scene/viewer world, same frame as the splat and cameras",
            "units": "reconstruction units; metric scale unknown until calibrated",
            "bbox": "fractions of image width/height, top-left origin",
        },
        "stats": {
            "observations": len(observations),
            "anchored": sum(1 for o in observations if o.position),
            "annotations": len(annotations),
            "failed_views": len(failures),
            "seconds": round(time.time() - started, 1),
        },
        "annotations": [a.to_dict() for a in annotations],
        "observations": [o.to_dict() for o in observations],
        "failures": failures,
    }
    return payload


def write_annotations(job_dir: Path, payload: dict) -> Path:
    path = Path(job_dir) / "cosmos" / "cosmos_annotations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def author_annotation_prims(stage, annotations: list[dict]) -> int:
    """/World/Annotations/<slug>: one Xform per anchored label.

    An Xform rather than a marker mesh — the prim is a named place, and a
    consumer decides whether to draw anything there. ObjectNav reads the
    translation; the rest is provenance.
    """
    from pxr import Gf, Sdf, UsdGeom

    anchored = [a for a in annotations if a.get("position")]
    if not anchored:
        return 0
    UsdGeom.Scope.Define(stage, "/World/Annotations")
    used: set[str] = set()
    for annotation in anchored:
        slug = annotation.get("id") or slugify(annotation["label"])
        name = slug
        suffix = 2
        while name in used:
            name = f"{slug}_{suffix}"
            suffix += 1
        used.add(name)
        xform = UsdGeom.Xform.Define(stage, f"/World/Annotations/{name}")
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in annotation["position"]]))
        prim = xform.GetPrim()
        prim.SetDisplayName(annotation["label"])
        prim.CreateAttribute("vw:label", Sdf.ValueTypeNames.String, custom=True).Set(annotation["label"])
        prim.CreateAttribute("vw:kind", Sdf.ValueTypeNames.String, custom=True).Set(annotation.get("kind", "other"))
        prim.CreateAttribute("vw:static", Sdf.ValueTypeNames.Bool, custom=True).Set(bool(annotation.get("static", True)))
        prim.CreateAttribute("vw:confidence", Sdf.ValueTypeNames.Float, custom=True).Set(float(annotation.get("confidence", 0.0)))
        prim.CreateAttribute("vw:observations", Sdf.ValueTypeNames.Int, custom=True).Set(int(annotation.get("observations", 1)))
        if annotation.get("spread") is not None:
            prim.CreateAttribute("vw:spread", Sdf.ValueTypeNames.Float, custom=True).Set(float(annotation["spread"]))
    return len(anchored)
