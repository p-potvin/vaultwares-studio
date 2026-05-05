<!-- VAULT-THEMES-SUBMODULE:START -->
## Vault Themes Submodule Rules

This repository includes `vault-themes`. Before changing UI, branding, design
systems, theme tokens, shared components, authentication UX, encrypted
client-to-client communication UX, Figma-derived implementation, or agent/IDE
instructions, read these stable root files:

- `vault-themes/AGENTS.md`
- `vault-themes/CONTEXT.md`

When the submodule has the cleaned layout, also read:

- `vault-themes/brand/brand-guide.md`
- `vault-themes/brand/tokens/tokens.ts`

Treat `vault-themes` as the shared VaultWares source of truth. Do not copy its
rules into this repo unless a tool-specific file requires a short pointer.
<!-- VAULT-THEMES-SUBMODULE:END -->

<!-- VAULTWARES_AGENTCIATION:MANAGED:START -->
# VaultWares Theming Skill

## Overview

The VaultWares Theming Skill provides agents and developers with a unified, framework-agnostic interface for applying, inspecting, and enforcing VaultWares design tokens and theming rules across any project. It abstracts the theme source of truth (`vault-themes/theme_manager.py`, `Brand/tokens.ts`, `VaultWares.Brand.xaml`) and provides usage patterns for Qt/PySide6, React/Tailwind, and WinUI3/XAML.

## Capabilities
- List all available VaultWares themes (name, accent, surface, style family)
- Retrieve design-token maps for any theme (Python, TypeScript, XAML)
- Enforce token usage: never hardcode colors, fonts, or spacing
- Apply themes at runtime in supported frameworks
- Integrate with both desktop and web UI pipelines

## Entrypoints
| Symbol/Path | Framework | Purpose |
|---|---|---|
| `VaultThemeManager` | Python (Qt/PySide6) | Source of truth for all themes |
| `get_theme_by_name(name)` | Python | Resolve theme by display name |
| `get_themes()` | Python | List all `VaultTheme` objects |
| `Brand/tokens.ts` | React/Tailwind | TypeScript token map |
| `tailwind.config.ts` | React/Tailwind | Tailwind theme config |
| `VaultWares.Brand.xaml` | WinUI3/XAML | Resource dictionary for XAML apps |

## Usage Patterns

### Qt/PySide6
- Use `VaultThemeManager` to select/apply themes
- Always use `card_style(theme)` and `state_card_style(theme, state)` helpers for QSS
- Never hardcode hex values; always source from the theme
- Example:
```python
from vault_themes.theme_manager import VaultThemeManager
_tm = VaultThemeManager()
theme = _tm.get_theme_by_name("Golden Slate")
widget.setStyleSheet(card_style(theme))
```

### React/Tailwind
- Import tokens from `Brand/tokens.ts` and classes from `tailwind.config.ts`
- Use class names, not raw values: `className="bg-vault-base text-vault-gold"`
- Never use raw hex, px, or font values in JSX/TSX
- Example:
```tsx
import tokens from 'Brand/tokens.ts';
<div className="bg-vault-base text-vault-gold p-8">
  ...
</div>
```

### WinUI3/XAML
- Reference theme resources from `VaultWares.Brand.xaml`
- Use `{StaticResource vault-gold}` etc. for all colors, spacing, and fonts
- Never use raw hex or pixel values in XAML
- Example:
```xml
<Border Background="{StaticResource vault-base}" Padding="8">
  <TextBlock Foreground="{StaticResource vault-gold}" ... />
</Border>
```

## Design Token Roles
| Token | Role |
|---|---|
| `vault-base` | Dark background |
| `vault-paper` / `vault-light` | Light background |
| `vault-gold` | Primary brand accent |
| `vault-cyan` | Interactive/links |
| `vault-green` | Success/secured |
| `vault-burgundy` | Error/destructive |
| `vault-slate` | Secondary text |
| `vault-muted` | Captions/tertiary |

## Rules
- **Never** hardcode colors, fonts, or spacing — always use tokens
- Always source tokens from the canonical theme manager/config
- When adding new UI, ensure all surfaces, borders, and text use token-based styles
- For new frameworks, create a token adapter that maps to the above roles
- All user-facing strings must be i18n-ready (see `Brand/brand.i18n.ts`)
- Security: never expose or persist user theme preferences without consent
- Accessibility: all color choices must meet WCAG AA contrast

## Integration
- Python: `from vault_themes.theme_manager import VaultThemeManager`
- TypeScript: `import tokens from 'Brand/tokens.ts'`
- XAML: `<ResourceDictionary Source="VaultWares.Brand.xaml" />`

## Security & Style
- Follows VaultWares privacy, security, and style guidelines
- All code and UI must pass VaultWares design and accessibility review

<!-- VAULTWARES_AGENTCIATION:MANAGED:END -->
