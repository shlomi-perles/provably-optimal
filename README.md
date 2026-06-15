# My Lectures

Starter repository for a Manim lecture site built with
[Simplex](https://github.com/shlomi-perles/simplex). This repo is an
application template, not a Python package to publish: you write decks under
`decks/`, Simplex renders them with Manim, and GitHub Actions publishes the
generated `site/` folder to GitHub Pages.

## Playback Architecture

Simplex's accepted playback direction is a continuous timeline player: render
scene units independently, compose one lecture timeline per theme, package
HLS/CMAF plus MP4 fallback, and navigate by seeking to cue timestamps instead
of swapping per-slide videos.

## First Setup

1. Create a new repository from this template.
2. In GitHub, enable `Settings -> Pages -> Source -> GitHub Actions`.
3. Clone your new repository locally.
4. Install system tools:

   ```bash
   # Ubuntu / Debian
   sudo apt-get install texlive-latex-extra texlive-fonts-recommended \
                        libcairo2-dev libpango1.0-dev
   ```

   ```powershell
   # Windows
   winget install MiKTeX.MiKTeX
   ```

5. Sync Python dependencies and check the environment:

   ```bash
   uv sync --locked
   uv run simplex doctor
   ```

6. Build and preview the starter site:

   ```bash
   uv run simplex build
   uv run simplex serve
   ```

## Daily Workflow

Create a deck:

```bash
uv run simplex new algorithms/hash-tables
```

Render one deck:

```bash
uv run simplex render hash-tables
```

Build and serve the whole portal:

```bash
uv run simplex build
uv run simplex serve --watch
```

Render only one true slide-theme variant during iteration:

```bash
uv run simplex build --slide-theme dark
uv run simplex test --slide-theme dark
```

Push to `main` to publish with GitHub Pages.

## Repository Layout

```text
.
|-- pyproject.toml            application dependencies
|-- uv.lock                   locked environment for local and CI parity
|-- site.toml                 site brand, tagline, nav, section defaults
|-- manim.cfg                 shared Manim render defaults
|-- ruff.toml                 lint config
|-- simplex_themes/           repo-local theme, palette, and code-style docs
|   |-- README.md             full theme-system reference
|   |-- code_styles/          custom Pygments style modules
|   |-- palette_styles/       Theme Studio JSON and .itermcolors palettes
|   `-- themes/               Simplex theme JSON files
|-- .github/workflows/
|   |-- ci.yml                smoke render and static checks
|   `-- deploy.yml            build and deploy to GitHub Pages
`-- decks/example/
    |-- deck.toml             deck metadata and scene entrypoints
    |-- notes.md              rendered into the deck page and notes PDF
    |-- refs.bib              optional BibTeX references
    `-- slides/intro.py       Manim scene classes
```

The starter deck is real on purpose so the first build produces a useful
site. Rename it, edit it, or delete it after you add your own decks.

## Manim Configuration

Keep shared Manim settings in the root `manim.cfg`:

```ini
[CLI]
plugins = simplex
quality = high_quality
```

Simplex renders with `cwd` set to each deck directory, but it also looks for
the root `manim.cfg`. If a deck contains its own `manim.cfg`, Simplex merges
the two files before invoking Manim: deck-local options override matching root
options, and unrelated sections or fields are preserved. Use this only for
deck-specific render needs, for example a lower quality setting while drafting
one heavy deck.

Command-line flags still win for one-off renders:

```bash
uv run simplex render hash-tables --disable_caching --fps 60
```

## Deck Configuration

Each deck has a `deck.toml`:

```toml
slug = "hash-tables"
title = "Hash Tables"
summary = "Open addressing, chaining, and amortized lookup intuition."
tags = ["data-structures"]
order = 10
entrypoints = ["slides.intro:Intro", "slides.intro:KeyIdea"]

# Optional. If omitted, Simplex derives the public deck date from Git history.
date = "2026-05-19"

[slide_themes]
enabled = true
dark = "simplex_dark"
light = "simplex_light"
default = "dark"

[web]
show_slide_number = true
show_clock = false
show_stopwatch = false
show_notes_date = true
notes_code_style = "simplex_solarized_light"
```

`entrypoints` points to scene classes relative to the deck directory. Add
`@opengl` to one entrypoint when a scene needs ManimCE's OpenGL renderer:

```toml
entrypoints = ["slides.intro:Intro", "slides.surface:Surface@opengl"]
```

The `date` field is optional. When it is absent, homepage cards use the first
Git commit that added the deck. If that cannot be found, Simplex falls back to
the last time the deck's slide order/count metadata changed, then to the last
changed Python scene file. Set `date = "YYYY-MM-DD"` when you want an
explicit publication date. Set `[web] show_notes_date = true` to display that
same resolved date under the first notes heading and in the generated notes
PDF.

Deck notes support:

- Slide refs by normalized slide title, such as `[slide:a-second-slide]` for a
  slide titled `A Second Slide`. You do not write anchor names in `deck.toml`.
- Inline and display math with KaTeX syntax.
- Sidenotes with `^[...]`.
- Theorem-style callouts from blockquotes.
- BibTeX citations such as `\cite{KB15}` from `refs.bib`; Simplex appends the
  References section automatically.

## Voiceover

Voiceover is not a core `manim.cfg` field and Simplex does not need a
`voiceover = false` deck option. For narrated videos, use the separate
`manim-voiceover` plugin in your scene code, add its Python dependency to this
project, and configure its speech service in Python. Keep deck metadata about
rendering and the web page in `deck.toml`; keep Manim CLI render options in
`manim.cfg`.

## Themes

This template renders true dark and light slide artifacts by default. Configure
the theme names globally in `site.toml`, or override them in one deck's
`deck.toml`:

```toml
[slide_themes]
enabled = true
dark = "simplex_dark"
light = "my_light"
default = "dark"
```

Theme JSON files live in `simplex_themes/themes/`. Palette exports live in
`simplex_themes/palette_styles/`, and Pygments code-style modules live in
`simplex_themes/code_styles/`. See [simplex_themes/README.md](simplex_themes/README.md)
for the full theme system, including custom palette fields and one-place
dark/light values such as:

```json
{
  "palette": {
    "warning": { "light": "#775500", "dark": "#FFD166" }
  }
}
```

Open Theme Studio with:

```bash
uv run simplex theme-studio
```

## GitHub Pages

The deploy workflow builds `site/` and uploads it with `actions/deploy-pages`.
There is no `gh-pages` branch.

By default, the workflow sets `SIMPLEX_BASE_URL` to `/<repository-name>`,
which matches ordinary GitHub project pages. For a custom domain or a root
`owner.github.io` site, set a repository variable named `SIMPLEX_BASE_URL` to
`/`.

Optional deployment settings:

- `SIMPLEX_BASE_URL`: URL prefix for generated links.
- `SIMPLEX_GA_TAG`: Google Analytics tag. The workflow reads this from a
  repository secret named `GA_TAG`.
- `SIMPLEX_BRAND`: override the site brand without editing `site.toml`.
- `SIMPLEX_PREVIEW`: set to `1` to suppress analytics during previews.

## Updating Simplex

This template tracks `uv.lock` so local builds and GitHub Actions use the same
dependency graph. Simplex release automation opens dependency PRs for this
template. To update manually:

```bash
uv lock --upgrade-package manim-simplex
uv sync --locked
uv run simplex test
```

Commit both `pyproject.toml` and `uv.lock` when dependency constraints change.

## License

Use whatever license suits your lectures. The template itself is MIT.
