# Palette Style Exports

Save Manim palette sources here. The file stem becomes the `manim_palette`
name used by theme JSON files.

Supported files:

- `.json` files exported by `uv run simplex theme-studio`
- `.itermcolors` files from iTerm2 color schemes

Example:

```text
simplex_themes/palette_styles/ocean.json
```

```json
{
  "manim_palette": "ocean",
  "palette": {
    "accent": "#F4C84A"
  }
}
```

Palette styles define Manim constants. Semantic slide colors still belong in
`simplex_themes/themes/*.json`. See [../README.md](../README.md) for the full
theme system.
