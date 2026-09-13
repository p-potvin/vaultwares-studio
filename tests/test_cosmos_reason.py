import json

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from vaultwares_studio.cosmos_reason import (
    DepthAnchor,
    VlmProvider,
    Observation,
    annotate_job,
    author_annotation_prims,
    build_provider,
    cluster,
    extract_json,
    parse_objects,
    plan_views,
    slugify,
)


class FakeProvider(VlmProvider):
    """Answers from a canned list, one reply per call."""

    name = "fake"
    model = "fake-vlm"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def describe(self, image, prompt):  # noqa: ARG002
        self.calls += 1
        return self.replies[(self.calls - 1) % len(self.replies)]


# -- parsing -------------------------------------------------------------------


def test_extract_json_handles_fences_thinking_and_truncation():
    assert extract_json('<think>hm</think>```json\n{"objects": [{"label": "x"}]}\n```')["objects"][0]["label"] == "x"
    cut = '{"objects": [{"label": "shed", "bbox": [0,0,1,1]}, {"label": "cut off mid'
    assert [o["label"] for o in extract_json(cut)["objects"]] == ["shed"]
    assert extract_json("no json at all") is None


def test_parse_objects_normalises_kind_bbox_and_confidence():
    text = json.dumps({"objects": [
        {"label": "Wooden Shed", "kind": "a building structure", "static": True,
         "bbox": [0.6, 0.2, 0.2, 0.6], "confidence": 1.4},
        {"label": "sky", "kind": "other", "bbox": [0, 0, 1, 1]},
        {"label": "car", "kind": "vehicle", "static": False, "bbox": [100, 50, 300, 250], "confidence": "x"},
    ]})
    found = parse_objects(text, view_index=2, frame_index=40)
    assert [o.label for o in found] == ["Wooden Shed", "car"]          # sky is dropped
    assert found[0].kind == "structure"                                 # snapped to the vocabulary
    assert found[0].bbox == [0.2, 0.2, 0.6, 0.6]                        # corners reordered
    assert found[0].confidence == 1.0                                   # clamped
    assert found[1].bbox == pytest.approx([1 / 3, 1 / 6, 1.0, 5 / 6])   # pixels rescaled
    assert found[1].confidence == 0.5 and found[1].static is False
    assert found[0].view_index == 2 and found[0].frame_index == 40


def test_parse_objects_survives_rubbish():
    assert parse_objects("the model refused", 0, 0) == []
    assert parse_objects(json.dumps({"objects": [{"label": ""}]}), 0, 0) == []


# -- clustering ----------------------------------------------------------------


def _obs(label, pos, conf=0.8, frame=0, kind="structure"):
    return Observation(view_index=frame, frame_index=frame, label=label, kind=kind,
                       static=True, confidence=conf, bbox=[0.4, 0.4, 0.6, 0.6], position=pos)


def test_cluster_merges_related_labels_that_sit_together():
    merged = cluster([
        _obs("shed door", [0.0, 0.0, 0.0], 0.7, 0),
        _obs("wooden shed door", [0.05, 0.0, 0.0], 0.9, 1),
        _obs("shed door", [0.02, 0.01, 0.0], 0.6, 2),
    ], scene_extent=1.0)
    assert len(merged) == 1
    assert merged[0].label == "wooden shed door"      # the most confident name wins
    assert merged[0].observations == 3
    assert merged[0].frames == [0, 1, 2]
    assert "shed door" in merged[0].aliases
    assert merged[0].position == pytest.approx([0.02, 0.0, 0.0], abs=0.02)


def test_cluster_keeps_same_label_at_different_places_apart():
    merged = cluster([_obs("window", [0.0, 0.0, 0.0]), _obs("window", [10.0, 0.0, 0.0])], scene_extent=1.0)
    assert len(merged) == 2


def test_cluster_falls_back_to_the_anchor_spread_without_a_scene_scale():
    """No scene scale: the anchors' own spread is the yardstick, and two
    sightings at opposite ends of it are still two things."""
    apart = cluster([_obs("window", [0.0, 0.0, 0.0]), _obs("window", [1.0, 0.0, 0.0])])
    assert len(apart) == 2


def test_cluster_without_anchors_still_groups_by_label():
    merged = cluster([_obs("grass lawn", None, 0.9), _obs("grass lawn", None, 0.5)])
    assert len(merged) == 1 and merged[0].observations == 2 and merged[0].position is None


def test_cluster_static_only_when_every_sighting_agrees():
    a, b = _obs("cat", [0, 0, 0]), _obs("cat", [0.01, 0, 0])
    b.static = False
    assert cluster([a, b], scene_extent=1.0)[0].static is False


# -- view planning and slugs ---------------------------------------------------


def test_plan_views_spans_the_capture_including_both_ends():
    assert plan_views(500, 5) == [0, 125, 250, 374, 499]
    assert plan_views(3, 10) == [0, 1, 2]
    assert plan_views(500, 1) == [250]


def test_slugify():
    assert slugify("Wooden Shed Door!") == "wooden_shed_door"
    assert slugify("???") == "object"


# -- anchoring -----------------------------------------------------------------


def _write_depth_scene(tmp_path, depth=2.0, scale=1.0):
    stream = tmp_path / "streaming"
    (stream / "results_output").mkdir(parents=True)
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 35.0], [0, 0, 1]], dtype=np.float32)
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 0.0, 0.0]
    np.savetxt(stream / "camera_poses.txt", c2w.reshape(1, 16))
    np.savez(stream / "results_output" / "frame_0.npz",
             depth=np.full((70, 100), depth / scale, dtype=np.float32),
             conf=np.full((70, 100), 5.0, dtype=np.float32),
             intrinsics=K, s=np.float64(scale))
    return stream


def test_depth_anchor_unprojects_the_bbox_centre(tmp_path):
    anchor = DepthAnchor(_write_depth_scene(tmp_path))
    assert anchor.available
    # A box centred on the principal point sits straight down the optical axis.
    centre = anchor.anchor(0, [0.4, 0.4, 0.6, 0.6])
    assert centre == pytest.approx([1.0, 0.0, 2.0], abs=0.05)
    # Off-centre in x moves the anchor in x by (u - cx) * z / fx.
    right = anchor.anchor(0, [0.9, 0.4, 1.0, 0.6])
    assert right[0] > centre[0] and right[2] == pytest.approx(2.0, abs=0.05)


def test_depth_anchor_applies_chunk_scale_and_scene_transform(tmp_path):
    lift = np.eye(4)
    lift[:3, 3] = [0.0, 5.0, 0.0]
    anchor = DepthAnchor(_write_depth_scene(tmp_path, scale=2.0), scene_transform=lift)
    assert anchor.anchor(0, [0.4, 0.4, 0.6, 0.6]) == pytest.approx([1.0, 5.0, 2.0], abs=0.05)


def test_depth_anchor_returns_none_for_low_confidence_or_missing_frames(tmp_path):
    stream = _write_depth_scene(tmp_path)
    np.savez(stream / "results_output" / "frame_0.npz",
             depth=np.zeros((70, 100), dtype=np.float32),
             conf=np.zeros((70, 100), dtype=np.float32),
             intrinsics=np.eye(3, dtype=np.float32), s=np.float64(1.0))
    anchor = DepthAnchor(stream)
    anchor._cache.clear()
    assert anchor.anchor(0, [0.4, 0.4, 0.6, 0.6]) is None
    assert anchor.anchor(7, [0.4, 0.4, 0.6, 0.6]) is None


# -- providers -----------------------------------------------------------------


def test_build_provider_selects_backends_and_rejects_unknown(monkeypatch):
    assert build_provider("none").name == "none"
    assert build_provider("ollama").model == "qwen3-vl:2b"
    assert build_provider("ollama", "cosmos-reason2-2b").model == "cosmos-reason2-2b"
    monkeypatch.setenv("NGC_API_KEY", "nvapi-test")
    nvidia = build_provider("nvidia")
    assert nvidia.name == "nvidia" and nvidia.model.startswith("nvidia/")
    assert build_provider("nvidia", "nvidia/cosmos-reason2-8b").model == "nvidia/cosmos-reason2-8b"
    with pytest.raises(ValueError):
        build_provider("hal9000")


def test_nvidia_provider_requires_a_key(monkeypatch):
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("NGC_CLI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No NVIDIA API key"):
        build_provider("nvidia")


# -- the pass ------------------------------------------------------------------


def _job_with_frames(tmp_path, frames=4):
    from PIL import Image

    from vaultwares_studio.capture_cameras import frames_from_transforms, write_capture_cameras_json

    (tmp_path / "frames").mkdir(parents=True)
    for index in range(frames):
        Image.new("RGB", (64, 48), (30 * index, 120, 60)).save(tmp_path / "frames" / f"frame_{index:05d}.jpg")
    transforms = {"frames": [
        {"file_path": f"images/frame_{i:05d}.jpg", "transform_matrix": np.eye(4).tolist(),
         "fl_x": 100.0, "fl_y": 100.0, "cx": 32.0, "cy": 24.0, "w": 64, "h": 48}
        for i in range(frames)]}
    write_capture_cameras_json(tmp_path, frames_from_transforms(transforms, duration=1.0))
    return tmp_path


def test_annotate_job_collects_clusters_and_reports(tmp_path):
    job = _job_with_frames(tmp_path)
    reply = json.dumps({"objects": [
        {"label": "wooden shed", "kind": "structure", "static": True, "bbox": [0.3, 0.3, 0.7, 0.7], "confidence": 0.9},
        {"label": "grass lawn", "kind": "ground", "static": True, "bbox": [0.0, 0.8, 1.0, 1.0], "confidence": 0.8},
    ]})
    provider = FakeProvider([reply])
    payload = annotate_job(job, provider, views=3, log=lambda _m: None)
    assert provider.calls == 3
    assert payload["stats"]["observations"] == 6
    assert {a["label"] for a in payload["annotations"]} == {"wooden shed", "grass lawn"}
    assert all(a["observations"] == 3 for a in payload["annotations"])
    assert payload["source"]["anchoring"] == "none"       # no streaming depth in this job
    assert payload["stats"]["failed_views"] == 0
    assert payload["annotations"][0]["id"] == slugify(payload["annotations"][0]["label"])


def test_annotate_job_tolerates_a_failing_view(tmp_path):
    job = _job_with_frames(tmp_path)
    good = json.dumps({"objects": [{"label": "fence", "kind": "structure", "bbox": [0.1, 0.1, 0.2, 0.2]}]})

    class Flaky(FakeProvider):
        def describe(self, image, prompt):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("gateway timeout")
            return good

    payload = annotate_job(job, Flaky([good]), views=3, log=lambda _m: None)
    assert payload["stats"]["failed_views"] == 1
    assert payload["stats"]["observations"] == 2


def test_annotate_job_gives_up_when_the_provider_is_always_down(tmp_path):
    job = _job_with_frames(tmp_path, frames=6)

    class Dead(FakeProvider):
        def describe(self, image, prompt):
            raise RuntimeError("404 not provisioned")

    with pytest.raises(RuntimeError, match="failing repeatedly"):
        annotate_job(job, Dead([""]), views=5, log=lambda _m: None)


def test_null_provider_still_produces_a_view_plan(tmp_path):
    payload = annotate_job(_job_with_frames(tmp_path), build_provider("none"), views=2, log=lambda _m: None)
    assert payload["annotations"] == [] and payload["source"]["frames_sampled"] == [0, 3]


# -- USD -----------------------------------------------------------------------


def test_author_annotation_prims_places_named_goals(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    count = author_annotation_prims(stage, [
        {"id": "wooden_shed", "label": "wooden shed", "kind": "structure", "static": True,
         "confidence": 0.9, "observations": 4, "position": [1.0, 0.5, -2.0], "spread": 0.1},
        {"id": "wooden_shed", "label": "wooden shed", "kind": "structure", "static": True,
         "confidence": 0.7, "observations": 1, "position": [4.0, 0.5, -2.0]},
        {"id": "no_anchor", "label": "no anchor", "kind": "other", "static": True,
         "confidence": 0.5, "observations": 1, "position": None},
    ])
    assert count == 2
    prim = stage.GetPrimAtPath("/World/Annotations/wooden_shed")
    assert prim.GetAttribute("vw:label").Get() == "wooden shed"
    assert prim.GetAttribute("vw:kind").Get() == "structure"
    assert prim.GetAttribute("vw:observations").Get() == 4
    xform = UsdGeom.Xformable(prim).GetLocalTransformation()
    assert [xform[3][i] for i in range(3)] == pytest.approx([1.0, 0.5, -2.0])
    # A duplicate slug gets its own prim rather than overwriting the first.
    assert stage.GetPrimAtPath("/World/Annotations/wooden_shed_2")
    assert not stage.GetPrimAtPath("/World/Annotations/no_anchor")


def test_compose_scene_carries_annotations(tmp_path):
    from vaultwares_studio.camera_paths import CameraEntity, CameraKeyframe
    from vaultwares_studio.camera_scene import compose_scene, load_annotations

    cloud = tmp_path / "reconstruction" / "cloud.usda"
    cloud.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(cloud))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    stage.GetRootLayer().Save()
    annotations = [{"id": "blue_car", "label": "blue car", "kind": "vehicle", "static": False,
                    "confidence": 0.8, "observations": 2, "position": [-1.0, 0.0, 0.5]}]
    (tmp_path / "cosmos").mkdir()
    (tmp_path / "cosmos" / "cosmos_annotations.json").write_text(json.dumps({"annotations": annotations}))
    assert load_annotations(tmp_path) == annotations
    scene = tmp_path / "usd" / "scene.usda"
    compose_scene(scene, cloud, [CameraEntity("A", keyframes=[CameraKeyframe(0, [0, 1, 4], [0, 0, 0])])],
                  annotations=annotations)
    opened = Usd.Stage.Open(str(scene))
    car = opened.GetPrimAtPath("/World/Annotations/blue_car")
    assert car and car.GetAttribute("vw:static").Get() is False


def test_cluster_merges_a_refined_description_of_the_same_thing():
    """Word sets, not substrings: measured on real output where 'Blue door'
    and 'blue screen door' were left as two objects 0.15 apart."""
    merged = cluster([
        _obs("Blue door", [0.81, -0.06, -0.30], 0.95),
        _obs("blue screen door", [0.70, 0.02, -0.31], 0.95),
    ], scene_extent=2.3)
    assert len(merged) == 1 and merged[0].observations == 2


def test_cluster_keeps_different_plants_apart_despite_a_shared_word():
    merged = cluster([
        _obs("large green hedge", [-0.26, 0.02, -0.19], 0.95, kind="vegetation"),
        _obs("medium green bush", [-0.26, -0.02, -0.06], 0.95, kind="vegetation"),
    ], scene_extent=2.3)
    assert len(merged) == 2


def test_cluster_ignores_filler_words():
    merged = cluster([_obs("the large shed", [0, 0, 0]), _obs("shed", [0.01, 0, 0])], scene_extent=1.0)
    assert len(merged) == 1
