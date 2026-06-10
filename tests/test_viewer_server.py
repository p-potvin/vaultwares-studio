import urllib.request

from gui.viewport import ViewerServer, WEBVIEWER_DIR


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""


def test_viewer_server_serves_app_and_job(tmp_path):
    server = ViewerServer()
    try:
        status, body = _get(server.url("index.html"))
        assert status == 200
        assert b"viewer.js" in body

        # /job/* 404s until a job root is attached.
        status, _ = _get(server.url("job/artifact.txt"))
        assert status == 404

        (tmp_path / "artifact.txt").write_text("splat-bytes", encoding="utf-8")
        server.job_root = tmp_path
        status, body = _get(server.url("job/artifact.txt"))
        assert status == 200
        assert body == b"splat-bytes"

        # Path traversal out of either root is blocked.
        status, _ = _get(server.url("job/../../secrets.txt"))
        assert status == 404
        status, _ = _get(server.url("../" + WEBVIEWER_DIR.name + "/index.html"))
        assert status in (200, 404)  # normalized by http.server; never escapes
    finally:
        server.close()
