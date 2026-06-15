# Theme JSON Files

Save complete Simplex theme JSON files here. The file stem is the theme name:
`simplex_themes/themes/chalkboard.json` is selected with
`dark = "chalkboard"` or `light = "chalkboard"` in `site.toml` or `deck.toml`.

Minimal theme:

```json
{
  "manim_palette": "simplex_light",
  "palette": {
    "background": "#EEEAD8",
    "font": "#3C313F"
  }
}
```

Theme files may set `manim_palette`, `palette`, `web_palette`, `code_style`,
`typography`, `spacing`, `motion`, and `latex`. Missing semantic palette
fields are derived automatically.

Custom palette fields are preserved:

```json
{
  "palette": {
    "warning": "#FFD166",
    "success": "#2A9D8F"
  }
}
```

Dark/light values can live in one file:

```json
{
  "palette": {
    "warning": { "light": "#775500", "dark": "#FFD166" }
  }
}
```

See [../README.md](../README.md) for the full theme system.
