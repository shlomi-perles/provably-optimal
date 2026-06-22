"""Shared visual language for the AdaGrad and Adam deck."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from manim import (
    BLACK,
    BLUE,
    DARK_GRAY,
    DR,
    DOWN,
    GRAY,
    GREEN,
    LEFT,
    LIGHT_GRAY,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    SMALL_BUFF,
    TEAL,
    UP,
    WHITE,
    YELLOW,
    Line,
    MathTex,
    Mobject,
    SurroundingRectangle,
    Title,
    VGroup,
    VMobject,
)
from simplex import Slide, color_substrings, get_active_theme
from simplex.engine.region import Region
from simplex.engine.scaling import scale_to_fit_mobject


@dataclass(frozen=True, slots=True)
class Palette:
    text: str = str(WHITE)
    muted: str = str(GRAY)
    grid: str = str(DARK_GRAY)
    frame: str = str(LIGHT_GRAY)
    panel: str = str(BLACK)
    panel_soft: str = str(DARK_GRAY)
    panel_deep: str = str(BLACK)
    contour: str = str(BLUE)
    blue: str = str(BLUE)
    orange: str = str(ORANGE)
    green: str = str(GREEN)
    purple: str = str(PURPLE)
    red: str = str(RED)
    teal: str = str(TEAL)
    yellow: str = str(YELLOW)
    optimum: str = str(WHITE)
    optimum_stroke: str = str(WHITE)


@dataclass(frozen=True, slots=True)
class LayerOrder:
    heatmap: int = 0
    contour: int = 1
    trajectory: int = 2
    frame: float = 2.5
    markers: int = 3


PALETTE = Palette()
LAYERS = LayerOrder()

C_TEXT = PALETTE.text
C_MUTED = PALETTE.muted
C_GRID = PALETTE.grid
C_FRAME = PALETTE.frame
C_PANEL = PALETTE.panel
C_PANEL_SOFT = PALETTE.panel_soft
C_PANEL_DEEP = PALETTE.panel_deep
C_CONTOUR = PALETTE.contour
C_BLUE = PALETTE.blue
C_ORANGE = PALETTE.orange
C_GREEN = PALETTE.green
C_PURPLE = PALETTE.purple
C_RED = PALETTE.red
C_TEAL = PALETTE.teal
C_YELLOW = PALETTE.yellow
C_OPTIMUM = PALETTE.optimum
C_OPTIMUM_STROKE = PALETTE.optimum_stroke

HEATMAP_BLUE = PALETTE.blue
HEATMAP_MAX_ALPHA = 0.42

LAYER_HEATMAP = LAYERS.heatmap
LAYER_CONTOUR = LAYERS.contour
LAYER_TRAJECTORY = LAYERS.trajectory
LAYER_FRAME = LAYERS.frame
LAYER_MARKERS = LAYERS.markers

PANEL_CORNER_RADIUS = SMALL_BUFF
PANEL_BUFF = 2 * SMALL_BUFF
PANEL_STROKE_WIDTH = 0.8
ACCENT_STROKE_WIDTH = 2.4
LABEL_TEX_DOT_SCALE = 2

PANEL_FILL_OPACITY = 0.38
PANEL_STROKE_OPACITY = 0.22
SOFT_PANEL_FILL_OPACITY = 0.26
SOFT_PANEL_STROKE_OPACITY = 0.16
GLASS_PANEL_FILL_OPACITY = 0.36
SEGMENTED_PANEL_FILL_OPACITY = 0.5
SEGMENTED_PANEL_STROKE_OPACITY = 0.28
ACCENT_STROKE_OPACITY = 0.92

_TEX_TEXT_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_tex_text(text: str) -> str:
    return "".join(_TEX_TEXT_REPLACEMENTS.get(char, char) for char in text)


def start_slide(scene: Slide, title: str) -> Title:
    scene.slide(title=title)
    title_mob = Title(title)
    scene.region.place(title_mob, UP)
    scene.region.update(top=title_mob)
    return title_mob


def split_weighted(region: Region, ratios: Sequence[float]) -> list[Region]:
    total = float(sum(ratios))
    cursor = region.left
    pieces: list[Region] = []
    for ratio in ratios:
        width = region.width * ratio / total
        next_cursor = cursor + width
        pieces.append(
            Region(
                left=cursor,
                right=next_cursor,
                top=region.top,
                bottom=region.bottom,
            )
        )
        cursor = next_cursor
    return pieces


def split_rows(region: Region, ratios: Sequence[float]) -> list[Region]:
    total = float(sum(ratios))
    cursor = region.top
    pieces: list[Region] = []
    for ratio in ratios:
        height = region.height * ratio / total
        next_cursor = cursor - height
        pieces.append(
            Region(
                left=region.left,
                right=region.right,
                top=cursor,
                bottom=next_cursor,
            )
        )
        cursor = next_cursor
    return pieces


def themed_box(mobject: Mobject, color: str = C_PANEL) -> VGroup:
    box = SurroundingRectangle(mobject, buff=PANEL_BUFF, corner_radius=PANEL_CORNER_RADIUS)
    box.set_fill(color, opacity=PANEL_FILL_OPACITY)
    box.set_stroke(C_FRAME, width=PANEL_STROKE_WIDTH, opacity=PANEL_STROKE_OPACITY)
    return VGroup(box, mobject)


def panel_shell(mobject: Mobject, *, color: str = C_PANEL_SOFT) -> VGroup:
    shell = SurroundingRectangle(mobject, buff=PANEL_BUFF, corner_radius=PANEL_CORNER_RADIUS)
    shell.set_fill(color, opacity=SOFT_PANEL_FILL_OPACITY)
    shell.set_stroke(C_FRAME, width=PANEL_STROKE_WIDTH, opacity=SOFT_PANEL_STROKE_OPACITY)
    return VGroup(shell, mobject)


def glass_panel(mobject: Mobject) -> VGroup:
    panel = SurroundingRectangle(mobject, buff=PANEL_BUFF, corner_radius=PANEL_CORNER_RADIUS)
    panel.set_fill(C_PANEL, opacity=GLASS_PANEL_FILL_OPACITY)
    panel.set_stroke(C_FRAME, width=PANEL_STROKE_WIDTH, opacity=PANEL_STROKE_OPACITY)
    return VGroup(panel, mobject)


def segmented_panel(mobjects: VGroup) -> VGroup:
    panel = VGroup(*(SurroundingRectangle(mob, buff=PANEL_BUFF) for mob in mobjects))
    panel.set_fill(C_PANEL, opacity=SEGMENTED_PANEL_FILL_OPACITY).set_stroke(
        C_MUTED,
        width=PANEL_STROKE_WIDTH,
        opacity=SEGMENTED_PANEL_STROKE_OPACITY,
    )
    return panel


def accent_rule(mobject: Mobject, color: str) -> Line:
    rule = Line(LEFT * mobject.width / 2, RIGHT * mobject.width / 2)
    rule.set_stroke(color, width=ACCENT_STROKE_WIDTH, opacity=ACCENT_STROKE_OPACITY)
    rule.next_to(mobject, UP, buff=SMALL_BUFF)
    return rule


def theme_math(*tex_strings: str, color: str | None = None, typography: str = "body") -> MathTex:
    font_size = getattr(get_active_theme().typography, typography)
    if color is None:
        return MathTex(*tex_strings, font_size=font_size)
    return MathTex(*tex_strings, color=color, font_size=font_size)


def label_for_dot(
    tex: str,
    dot: Mobject,
    *,
    color: str = C_TEXT,
    direction: Sequence[float] = DR,
    scale: float = LABEL_TEX_DOT_SCALE,
    buff: float = SMALL_BUFF,
) -> MathTex:
    label = MathTex(tex, color=color)
    scale_to_fit_mobject(label, dot, scaleback=scale)
    direction_vector = list(direction)
    label.next_to(dot.get_boundary_point(direction_vector), direction_vector, buff=buff)
    return label


def formula_stack(*items: VMobject, buff: float | None = None) -> VGroup:
    stack = VGroup(*items)
    if buff is None:
        return stack.arrange(DOWN, aligned_edge=LEFT)
    return stack.arrange(DOWN, aligned_edge=LEFT, buff=buff)


def color_text_parts(mobject: VMobject, colors: dict[str, str]) -> None:
    if mobject.__class__ is MathTex:
        color_substrings(mobject, colors)
        return
    for child in mobject.submobjects:
        if isinstance(child, VMobject):
            color_text_parts(child, colors)
