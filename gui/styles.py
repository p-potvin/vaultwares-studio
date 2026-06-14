"""Qt stylesheet builders parameterised on a ``VaultTheme``.

Pure functions of theme — no Qt widget construction, no module-level state.
"""

from __future__ import annotations

from vaultwares_studio.pipeline import StageState


def build_stylesheet(theme) -> str:  # noqa: PLR0914
    bg = theme.background
    surface = theme.surface
    surface_alt = theme.surface_alt
    surface_el = theme.surface_elevated
    text = theme.text_primary
    text_sec = theme.text_secondary
    text_mut = theme.text_muted  # noqa: F841 - retained for stylesheet authoring parity
    text_inv = theme.text_inverse
    accent = theme.accent
    accent_m = theme.accent_muted
    border = theme.border
    muted = theme.muted
    return f"""
QWidget {{
    background: {bg};
    color: {text};
    font-family: "Segoe UI Semilight", "Segoe UI", "Inter", system-ui, sans-serif;
    font-size: 10pt;
}}
QMainWindow, QDialog {{ background: {bg}; }}
QFrame {{
    background: {surface};
    border: none;
    border-radius: 0px;
}}
/* ── Scroll areas ──────────────────────────── */
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget, QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
/* ── Scroll bars ───────────────────────────── */
QScrollBar:vertical {{
    background: {surface_alt};
    width: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {muted};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {accent}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none; height: 0; width: 0;
}}
QScrollBar:horizontal {{
    background: {surface_alt};
    height: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {muted};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background: {accent}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none; height: 0; width: 0;
}}
/* ── Push buttons ──────────────────────────── */
QPushButton {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 10pt;
    min-height: 32px;
}}
QPushButton:hover {{
    background: {accent};
    color: {text_inv};
    border-color: {accent};
}}
QPushButton:pressed {{
    background: {accent_m};
    color: {text_inv};
}}
QPushButton:disabled {{
    background: {surface_alt};
    color: {muted};
    border-color: {border};
}}
/* ── Line edits ────────────────────────────── */
QLineEdit {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 10pt;
    min-height: 36px;
    selection-background-color: {accent};
    selection-color: {text_inv};
}}
QLineEdit:focus {{ border: 1px solid {accent}; }}
/* ── Text edits (log view) ─────────────────── */
QTextEdit {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 8px;
    font-family: "Consolas", "Cascadia Code", "Courier New", monospace;
    font-size: 9pt;
    selection-background-color: {accent};
    selection-color: {text_inv};
}}
QTextEdit:focus {{ border: 1px solid {accent}; }}
/* ── List widget ───────────────────────────── */
QListWidget {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 8px 12px;
    border-radius: 4px;
    color: {text};
}}
QListWidget::item:hover {{ background: {surface_alt}; }}
QListWidget::item:selected {{ background: {accent}; color: {text_inv}; }}
/* ── Combo box ─────────────────────────────── */
QComboBox {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 32px;
    font-size: 10pt;
}}
QComboBox:hover, QComboBox:focus {{ border: 1px solid {accent}; }}
QComboBox::drop-down {{
    border: none;
    width: 24px;
    border-left: 1px solid {border};
}}
QComboBox QAbstractItemView {{
    background: {surface_el};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    selection-background-color: {accent};
    selection-color: {text_inv};
    padding: 4px;
    outline: none;
}}
/* ── Progress bar ──────────────────────────── */
QProgressBar {{
    background: {surface_alt};
    border: none;
    border-radius: 4px;
    max-height: 6px;
    min-height: 6px;
}}
QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}
/* ── Labels ────────────────────────────────── */
QLabel {{ background: transparent; color: {text}; }}
/* ── Splitter ──────────────────────────────── */
QSplitter::handle {{ background: {border}; }}
QSplitter::handle:horizontal {{ width: 4px; margin: 2px 0; }}
QSplitter::handle:vertical   {{ height: 4px; margin: 0 2px; }}
QSplitter::handle:hover      {{ background: {accent_m}; }}
/* ── Tab widget (fallback) ─────────────────── */
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 8px;
    background: {surface};
}}
QTabBar::tab {{
    background: {surface_alt};
    color: {text_sec};
    padding: 8px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border: 1px solid {border};
    margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {surface}; color: {text}; border-bottom-color: {surface}; }}
QTabBar::tab:hover    {{ background: {surface_el}; color: {text}; }}
"""


def card_style(theme) -> str:
    return (
        f"QFrame {{ background: {theme.surface}; border: 1px solid {theme.border};"
        " border-radius: 8px; }}"
    )


def accent_card_style(theme) -> str:
    return (
        f"QFrame {{ background: {theme.surface_elevated}; border: 1px solid {theme.accent};"
        " border-radius: 8px; }}"
    )


def preview_style(theme) -> str:
    return (
        f"QLabel {{ background: {theme.surface_elevated}; color: {theme.text_secondary};"
        f" border: 1px dashed {theme.text_muted}; border-radius: 8px;"
        " min-height: 120px; padding: 12px; }}"
    )


def state_card_style(theme, state: str) -> str:
    if state == StageState.COMPLETE.value:
        left_color = theme.success
    elif state == StageState.FAILED.value:
        left_color = theme.error
    elif state == StageState.RUNNING.value:
        left_color = theme.accent_muted
    else:
        left_color = theme.border
    return (
        f"QFrame {{ background: {theme.surface}; border: 1px solid {theme.border};"
        f" border-left: 4px solid {left_color}; border-radius: 8px; }}"
    )
