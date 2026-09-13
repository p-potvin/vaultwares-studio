import sys
import os

import gui  # Initializes USD before QtWebEngine (Windows DLL load ordering).
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

# We must ensure vaultwares-themes is in path if not already
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "vaultwares-themes"))

from gui.main_window import MainWindow

def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")

    # --ignore-gpu-blocklist lets Chromium try the real GPU; --enable-unsafe-
    # swiftshader keeps the software WebGL2 path available when it can't. Both
    # are needed: GaussianSplats3D hard-requires WebGL2, and dropping the
    # SwiftShader fallback leaves the viewport with no context at all.
    _chromium_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    if "--enable-unsafe-swiftshader" not in _chromium_flags:
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
            f"{_chromium_flags}"
            " --ignore-gpu-blocklist"
            " --enable-webgl"
            " --enable-unsafe-swiftshader"
        ).strip()


    from gui.viewport import register_viewer_scheme
    register_viewer_scheme()
    app = QApplication(sys.argv)

    # Set app-wide window icon (taskbar + alt-tab)
    from PySide6.QtGui import QIcon
    from pathlib import Path
    _icon_path = Path(__file__).resolve().parent / "vaultwares-themes" / "assets" / "icons" / "vaultwares-favicon-gold-filled-256.png"
    if _icon_path.exists():
        app.setWindowIcon(QIcon(str(_icon_path)))

    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
