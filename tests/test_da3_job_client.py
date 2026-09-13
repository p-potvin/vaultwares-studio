from tools.da3_job_client import job_payload


def test_single_job_payload_honors_approved_ninety_minutes():
    plan = {"image": "hf.co/spaces/test/worker", "flavor": "l4x1",
            "backend_timeout_seconds_per_job": 5400}
    payload = job_payload(plan, {"prefix": "jobs/test/reconstruction_sfm"}, "test-token", "test-bootstrap")
    assert payload["timeoutSeconds"] == 5400
    assert payload["attempts"] == 1
    assert payload["flavor"] == "l4x1"
    assert payload["spaceId"] == "test/worker"
    assert "dockerImage" not in payload


def test_explicit_null_is_not_replaced_by_default_timeout():
    plan = {"image": "hf.co/spaces/test/worker", "flavor": "l4x1",
            "backend_timeout_seconds_per_job": None}
    assert job_payload(plan, {"prefix": "jobs/test/reconstruction_sfm"}, "test-token", "test-bootstrap")["timeoutSeconds"] is None


def test_regular_oci_image_keeps_docker_image_field():
    plan = {"image": "ghcr.io/example/worker:stable", "flavor": "l4x1", "backend_timeout_seconds_per_job": 1}
    payload = job_payload(plan, {"prefix": "jobs/test/reconstruction_sfm"}, "test-token", "test-bootstrap")
    assert payload["dockerImage"] == "ghcr.io/example/worker:stable"
    assert "spaceId" not in payload


def test_prepared_plan_contains_distinct_loop_variants():
    import json
    from pathlib import Path
    plan = json.loads((Path("D:/vaultwares-studio-jobs/data/review/sep08/da3-loop-comparison/comparison_plan.json")).read_text())
    variants = {item["name"]: item["worker_args"] for item in plan["variants"]}
    assert "--stream-loop-closure" not in variants["loop-off"]
    assert variants["loop-on"][-1] == "--stream-loop-closure"
