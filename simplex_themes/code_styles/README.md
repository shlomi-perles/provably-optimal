# Code Style Modules

Save custom Pygments `Style` subclasses here. Simplex uses these styles for
Manim slide `Code` objects when a theme JSON sets `code_style`.

Example:

```python
from pygments.style import Style
from pygments.token import Keyword, Name, Text


class MyLectureCode(Style):
    background_color = "#101820"
    default_style = ""
    styles = {
        Text: "#F7F7F7",
        Keyword: "bold #F4C84A",
        Name.Function: "#7DD3FC",
    }
```

Then select it from a theme:

```json
{
  "code_style": "MyLectureCode"
}
```

Run `uv run simplex theme-studio` after adding a file to inspect it. Markdown
notes code blocks are configured separately with `[web] notes_code_style` in a
deck's `deck.toml`.

See [../README.md](../README.md) for how code styles fit into the full theme
system.
