# Simplex Themes

This directory is the project-local theme layer for your lecture site. Simplex
ships built-in themes, palettes, and code styles; files here override or extend
those defaults without editing the `manim-simplex` package.

## Directory Roles

```text
simplex_themes/
|-- README.md
|-- themes/          complete Simplex theme JSON files
|-- palette_styles/  Manim palette JSON or iTerm2 .itermcolors files
`-- code_styles/     Pygments Style subclasses for code rendering
```

Use `uv run simplex theme-studio` to compare palettes and code styles, then
save exports into the matching directory.

## How Theme Resolution Works

The rendered slide pixels come from the active Simplex theme. With true slide
themes enabled, `site.toml` chooses one theme for the dark render and one for
the light render:

```toml
[slide_themes]
enabled = true
dark = "simplex_dark"
light = "simplex_light"
default = "dark"
```

A deck may override the same block in its own `deck.toml`. Deck settings win
over `site.toml`; missing fields inherit from the site setting, then from
Simplex defaults.

The top-level `theme = "..."` field is only for single-render fallback mode.
For new decks, prefer `[slide_themes]`.

## Theme JSON

Create `simplex_themes/themes/my_light.json` and select it with
`light = "my_light"`:

```json
{
  "manim_palette": "simplex_light",
  "code_style": "simplex_solarized_light",
  "palette": {
    "background": "#EEEAD8",
    "font": "#3C313F",
    "accent": "#A15C00"
  },
  "web_palette": {
    "surface": "#F8F2DD",
    "text_muted": "#756E63"
  },
  "typography": {
    "mono_family": "JetBrains Mono",
    "body": 30,
    "h1": 60,
    "h2": 48,
    "caption": 20
  }
}
```

All fields are optional except that a theme must eventually resolve a full
semantic `palette`. Simplex fills missing palette fields from `manim_palette`;
if `manim_palette` is omitted, it fills from Manim's default palette.

## Manim Palettes

`manim_palette` controls Manim color constants such as `BLUE`, `BLUE_A`,
`WHITE`, `GRAY`, and `ORANGE` before your scene classes import and use them.
It can be:

- `manim_default`
- `simplex_light`
- a vendored iTerm2 color-scheme name
- a file stem from `simplex_themes/palette_styles/*.json`
- a file stem from `simplex_themes/palette_styles/*.itermcolors`

Palette JSON exported by Theme Studio has this shape:

```json
{
  "background_color": "#101820",
  "manim_colors": {
    "BLUE_C": "#3A7CA5",
    "WHITE": "#F7F7F7",
    "BLACK": "#101820"
  }
}
```

Simplex derives missing Manim aliases where possible, for example `BLUE` from
`BLUE_C` and `GRAY` from `GRAY_C`.

## Semantic Slide Palette

The theme `palette` controls Simplex helpers and rendered slide defaults:

- `background`: Manim frame background.
- `font`: default text and math color.
- `accent`: highlights, pointers, and emphasis.
- `vertex`, `vertex_stroke`, `edge`, `weight`, `visited`, `label`,
  `distance`: graph and array helper colors.

You may add custom fields. They are preserved on `get_active_theme().palette`,
so scene code can use them directly:

```json
{
  "palette": {
    "background": "#101820",
    "font": "#F7F7F7",
    "warning": "#FFD166",
    "success": "#2A9D8F"
  }
}
```

```python
from simplex import get_active_theme

theme = get_active_theme()
warning = theme.palette.warning
```

## One-Place Dark And Light Values

For fields that should change with the true slide-theme role, store both values
in one theme file:

```json
{
  "palette": {
    "background": { "light": "#FFFFFF", "dark": "#101820" },
    "font": { "light": "#111111", "dark": "#F7F7F7" },
    "warning": { "light": "#775500", "dark": "#FFD166" }
  }
}
```

Simplex resolves these objects from the actual render role (`dark` or `light`),
not from the theme file name. This means `light = "lecture"` and
`dark = "lecture"` can point at the same file when you want one source of
truth.

When a theme is loaded outside a true-theme render context, Simplex chooses the
`dark` value because `simplex_dark` is the package default.

## Code Styles

`code_style` controls Manim slide `Code` objects. It accepts:

- built-in Simplex styles such as `simplex_pycharm` and
  `simplex_solarized_light`
- any installed Pygments style name
- a custom `Style` subclass from `simplex_themes/code_styles/*.py`

Markdown notes code blocks are separate from rendered slide code. Override
them per deck:

```toml
[web]
notes_code_style = "simplex_pycharm"
```

## Web Palette

`web_palette` controls generated HTML shell colors, not rendered slide pixels.
It maps to CSS variables such as `--simplex-bg`, `--simplex-text`, and
`--simplex-link` for the portal, deck page, and timeline player shell.

Supported fields:

- `accent`
- `background`
- `surface`
- `text_primary`
- `text_muted`
- `link`
- `font_family_sans`
- `font_family_mono`
- `font_size_base`

Per-deck `[web]` fields in `deck.toml` can override the same web palette
values for one deck.

## Practical Recipes

Use the built-in light palette with a warmer paper background:

```json
{
  "manim_palette": "simplex_light",
  "palette": {
    "background": "#F7F1DF",
    "font": "#312A25"
  }
}
```

Use one file for both true slide-theme variants:

```json
{
  "manim_palette": "simplex_light",
  "palette": {
    "background": { "light": "#F7F1DF", "dark": "#111827" },
    "font": { "light": "#312A25", "dark": "#F9FAFB" },
    "accent": { "light": "#9A5B00", "dark": "#F4C84A" }
  }
}
```

Use a custom palette export as the base:

```json
{
  "manim_palette": "my_iterm_export",
  "code_style": "my_code_style",
  "palette": {
    "accent": "#F4C84A"
  }
}
```
