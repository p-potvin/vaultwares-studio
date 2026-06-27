import sys
import os

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

    _chromium_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    if "--enable-unsafe-swiftshader" not in _chromium_flags:
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
            f"{_chromium_flags} --enable-unsafe-swiftshader".strip()
        )
        
    from gui.viewport import register_viewer_scheme
    register_viewer_scheme()
    app = QApplication(sys.argv)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
