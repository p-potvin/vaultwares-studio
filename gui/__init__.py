"""GUI package for VaultWares Studio.

New GUI components land here; gui_app.py shrinks toward a thin entrypoint as
M2 progresses (theme/strings/widgets extraction follows the viewport).
"""

# On Windows, importing QtWebEngine before USD can make pxr.Tf's DLL fail to
# initialize. Load the required USD runtime before any GUI submodule loads Qt.
from pxr import Usd as _Usd
