"""Theme detection + persistent ``QSettings``-backed active-theme bootstrap.

``detect_os_theme`` reads the Windows AppsUseLightTheme registry key (returns
``"dark"`` on any other OS or on failure). ``bootstrap_theme`` loads the saved
theme name from ``QSettings`` and resolves it via the ``VaultThemeManager``
shipped with the ``vaultwares-themes`` submodule.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings

# vaultwares-themes lives as a sibling submodule; the import has to happen
# after we mutate sys.path so the package is reachable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "vaultwares-themes"))
try:
    from theme_manager import VaultTheme, VaultThemeManager  # noqa: E402
except ImportError as _exc:  # pragma: no cover - bootstrap error
    raise RuntimeError(
        "vault-themes submodule not found. Run: git submodule update --init vault-themes"
    ) from _exc


def detect_os_theme() -> str:
    """Return ``"light"`` if the OS theme reports light mode, else ``"dark"``."""
    try:
        import winreg  # noqa: PLC0415

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return "light" if value == 1 else "dark"
    except Exception:  # noqa: BLE001 - winreg unavailable / no key
        return "dark"


def bootstrap_theme(
    settings: QSettings, manager: VaultThemeManager
) -> VaultTheme:
    """Resolve the user's saved theme, falling back to OS-mode appropriate defaults."""
    saved_name = settings.value("theme", None)
    if saved_name:
        result = manager.get_theme_by_name(str(saved_name))
        return result if result else manager.get_theme("Golden Slate")
    default_name = (
        "Solarized Light Revisited" if detect_os_theme() == "light" else "Golden Slate"
    )
    result = manager.get_theme_by_name(default_name)
    return result if result else manager.get_theme("Golden Slate")


__all__ = ["VaultTheme", "VaultThemeManager", "bootstrap_theme", "detect_os_theme"]
