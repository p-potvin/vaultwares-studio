"""HfJobsStageRunner flavor-fallback: try candidates in order until one schedules.

HF Jobs has no server-side "give me any of these flavors" request, and spot GPU
capacity fluctuates enough that a submitted job can sit in SCHEDULING
indefinitely (observed 2026-07-30: l4x1, t4-small and a10g-small all refused to
schedule within minutes of each other). The runner therefore submits each
candidate in turn and moves on if it doesn't leave SCHEDULING in time.

These tests fake the hub entirely — no network, no token, no HF account.
"""

from __future__ import annotations

import pytest

from vaultwares_studio.runners import (
    HfJobsConfig,
    HfJobsStageRunner,
    StageContext,
)

# Negative timeout => the "still SCHEDULING?" check trips on the first poll,
# so a stuck candidate is abandoned without ever sleeping. Keeps tests instant
# (the real poll floor is MIN_POLL_INTERVAL_SECONDS = 10s).
GIVE_UP_IMMEDIATELY = -1.0


class _Status:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.message = None


class _Job:
    def __init__(self, job_id: str, flavor: str) -> None:
        self.id = job_id
        self.flavor = flavor
        self.url = f"https://fake/{job_id}"


class _FakeHub:
    """Minimal stand-in for the huggingface_hub module surface run() touches.

    ``schedulable`` names the flavors that leave SCHEDULING; everything else
    stays stuck there forever.
    """

    def __init__(self, schedulable: set[str]) -> None:
        self.schedulable = schedulable
        self.submitted: list[str] = []
        self.cancelled: list[str] = []
        self._flavor_by_job: dict[str, str] = {}

    def run_job(self, *, image, command, env, secrets, flavor, timeout, token, ssh=False):
        job_id = f"job-{len(self.submitted)}"
        self.submitted.append(flavor)
        self._flavor_by_job[job_id] = flavor
        self.last_image = image
        self.last_ssh = ssh
        return _Job(job_id, flavor)

    def inspect_job(self, *, job_id, token):
        flavor = self._flavor_by_job[job_id]
        if job_id in self.cancelled:
            return type("Info", (), {"status": _Status("CANCELED")})()
        stage = "COMPLETED" if flavor in self.schedulable else "SCHEDULING"
        return type("Info", (), {"status": _Status(stage)})()

    def cancel_job(self, *, job_id, token):
        self.cancelled.append(job_id)

    def fetch_job_logs(self, *, job_id, token):
        return iter(())


class _FakeApi:
    def __init__(self) -> None:
        self.uploaded: list[str] = []

    def create_repo(self, *args, **kwargs):
        return None

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
        self.uploaded.append(path_in_repo)

    def list_repo_files(self, repo, repo_type):
        return []


class _FakeRunner(HfJobsStageRunner):
    def __init__(self, schedulable: set[str]) -> None:
        super().__init__(
            config=HfJobsConfig(enabled=True, artifact_repo="fake/repo"),
            confirm_cost=lambda _est: True,
        )
        self.hub = _FakeHub(schedulable)
        self.api = _FakeApi()

    def _hub(self):
        return self.hub

    def _api(self):
        return self.api


@pytest.fixture(autouse=True)
def _fake_token(monkeypatch):
    monkeypatch.setattr("vaultwares_studio.runners.hf_jobs.get_hf_token", lambda: "fake-token")


def _ctx(tmp_path, flavor, **params):
    return StageContext(
        job_dir=tmp_path,
        job_id="job",
        stage_key="stage",
        params={
            "command": ["true"],
            "flavor": flavor,
            "est_minutes": 5,
            "flavor_scheduling_timeout_seconds": GIVE_UP_IMMEDIATELY,
            **params,
        },
        inputs=[],
        expected_outputs=[],
    )


def test_first_available_flavor_wins_and_later_ones_are_never_submitted(tmp_path):
    runner = _FakeRunner(schedulable={"t4-small", "a10g-small"})
    result = runner.run(_ctx(tmp_path, ["t4-small", "a10g-small"]))

    assert result.status == "complete"
    assert runner.hub.submitted == ["t4-small"]
    assert runner.hub.cancelled == []
    assert result.metadata["flavor"] == "t4-small"


def test_stuck_flavor_is_cancelled_and_next_candidate_is_tried(tmp_path):
    # t4 never schedules; a10g does.
    runner = _FakeRunner(schedulable={"a10g-small"})
    result = runner.run(_ctx(tmp_path, ["t4-small", "a10g-small"]))

    assert runner.hub.submitted == ["t4-small", "a10g-small"]
    # The abandoned t4 job must be cancelled — otherwise it could schedule
    # later and bill for a run nothing is waiting on.
    assert runner.hub.cancelled == ["job-0"]
    assert result.metadata["flavor"] == "a10g-small"


def test_all_candidates_exhausted_raises_naming_them(tmp_path):
    runner = _FakeRunner(schedulable=set())
    with pytest.raises(RuntimeError, match=r"None of the requested flavors"):
        runner.run(_ctx(tmp_path, ["t4-small", "a10g-small"]))

    assert runner.hub.submitted == ["t4-small", "a10g-small"]
    # Every submitted job is cancelled; none is left dangling in SCHEDULING.
    assert runner.hub.cancelled == ["job-0", "job-1"]


def test_plain_string_flavor_still_works(tmp_path):
    """Back-compat: the vast majority of presets pass a single flavor."""
    runner = _FakeRunner(schedulable={"l4x1"})
    result = runner.run(_ctx(tmp_path, "l4x1"))

    assert runner.hub.submitted == ["l4x1"]
    assert result.metadata["flavor"] == "l4x1"


def test_space_image_is_delegated_to_the_hub_client(tmp_path):
    runner = _FakeRunner(schedulable={"l4x1"})
    runner.run(_ctx(tmp_path, "l4x1", image="hf.co/spaces/clopeux/vw-studio-da3-gs"))
    # HfApi.run_job emits spaceId for this URL; an agent-side dockerImage
    # workaround would turn it into a nonexistent OCI :latest image.
    assert runner.hub.last_image == "hf.co/spaces/clopeux/vw-studio-da3-gs"


def test_cost_confirmation_quotes_the_priciest_candidate(tmp_path):
    """The confirm dialog must never undercount what could actually run.

    a10g-small ($1.00/hr) is 2.5x t4-small ($0.40/hr); quoting the first
    candidate would show half the worst-case spend.
    """
    seen = []
    runner = _FakeRunner(schedulable={"t4-small"})
    runner.confirm_cost = lambda est: (seen.append(est), True)[1]
    runner.run(_ctx(tmp_path, ["t4-small", "a10g-small"]))

    assert len(seen) == 1
    assert seen[0].flavor == "a10g-small"


def test_cost_denial_happens_before_any_job_is_submitted(tmp_path):
    """The consent gate must precede all network traffic, fallback included."""
    runner = _FakeRunner(schedulable={"t4-small"})
    runner.confirm_cost = lambda _est: False

    from vaultwares_studio.runners import CostDeniedError

    with pytest.raises(CostDeniedError):
        runner.run(_ctx(tmp_path, ["t4-small", "a10g-small"]))
    assert runner.hub.submitted == []


def test_ssh_is_off_unless_asked_for(tmp_path):
    """An SSH endpoint on a running job is a deliberate choice, not a default."""
    runner = _FakeRunner(schedulable={"t4-small"})
    runner.run(_ctx(tmp_path, ["t4-small"]))
    assert runner.hub.last_ssh is False


def test_ssh_is_forwarded_when_requested(tmp_path):
    """Jobs have no Spaces-style dev mode; ssh=True is the equivalent, and it is
    the only way to watch a training that Rich will not narrate into a pipe."""
    runner = _FakeRunner(schedulable={"t4-small"})
    runner.run(_ctx(tmp_path, ["t4-small"], ssh=True))
    assert runner.hub.last_ssh is True
