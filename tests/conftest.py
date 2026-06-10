import pytest


@pytest.fixture(autouse=True)
def isolated_jobs_dir(tmp_path, monkeypatch):
    """Redirect job creation to a temp dir.

    Tests used to write real manifests into data/jobs — including fake
    'completed' reconstructions with random gaussians — which the app's
    resume-latest logic then picked up as the active job (the user got a
    250-gaussian test splat in the viewer instead of their reconstruction).
    """
    monkeypatch.setattr("vaultwares_studio.pipeline.JOBS_DIR", tmp_path / "jobs")
    yield
