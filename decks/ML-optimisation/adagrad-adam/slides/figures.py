"""Manim-native figure slides for the AdaGrad lecture notes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Callable, Sequence

import contourpy
import numpy as np
import numpy.typing as npt
from manim import (
    DL,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    UL,
    UR,
    Arrow,
    Axes,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    ImageMobject,
    Line,
    MathTex,
    Mobject,
    PI,
    Rectangle,
    SurroundingRectangle,
    Title,
    VGroup,
    VMobject,
    Write,
    always_redraw,
)
from simplex import Caption, DN, Slide, VT, color_substrings, get_active_theme
from simplex.engine.region import Region

FloatArray = npt.NDArray[np.float64]

MU = 0.02
LAMBDA_MAX = 1.0
BASE_THETA_DEG = 28.0
BASE_INITIAL_VECTOR = np.array([5.05, 1.42], dtype=np.float64)
T_STEPS = np.arange(0, 151, dtype=np.float64)

C_TEXT = "#F8FAFC"
C_MUTED = "#94A3B8"
C_GRID = "#3A4553"
C_FRAME = "#BFC9D2"
C_PANEL = "#18212F"
C_PANEL_SOFT = "#101827"
C_PANEL_DEEP = "#0B1220"
C_CONTOUR = "#91BBD0"
C_BLUE = "#3D8FC7"
C_ORANGE = "#FF6600"
C_GREEN = "#34D399"
C_PURPLE = "#A78BFA"
C_RED = "#F87171"
C_TEAL = "#2DD4BF"
C_YELLOW = "#FFD700"
HEATMAP_BLUE = "#3D8FC7"
HEATMAP_MAX_ALPHA = 0.42
LAYER_HEATMAP = 0
LAYER_CONTOUR = 1
LAYER_TRAJECTORY = 2
LAYER_FRAME = 2.5
LAYER_MARKERS = 3


@dataclass(frozen=True, slots=True)
class ResponseSpec:
    title: str
    method: str
    eta: float


@dataclass(frozen=True, slots=True)
class SliderSpec:
    label: str
    minimum: float
    maximum: float
    decimals: int
    color: str


def _rotation(theta_deg: float) -> FloatArray:
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _theta_deg() -> float:
    base = _rotation(BASE_THETA_DEG) @ BASE_INITIAL_VECTOR
    offset = np.rad2deg(np.arctan2(base[0] - base[1], base[0] + base[1]))
    return float(BASE_THETA_DEG + offset)


THETA_DEG = _theta_deg()


def _rotated_quadratic_matrix() -> tuple[FloatArray, FloatArray]:
    eigvecs = _rotation(THETA_DEG)
    matrix = eigvecs @ np.diag([MU, LAMBDA_MAX]) @ eigvecs.T
    return matrix, eigvecs


def _start_slide(scene: Slide, title: str) -> Title:
    scene.slide(title=title)
    title_mob = Title(title)
    scene.region.place(title_mob, UP)
    scene.region.update(top=title_mob)
    return title_mob


def _split_weighted(region: Region, ratios: Sequence[float]) -> list[Region]:
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


def _split_rows(region: Region, ratios: Sequence[float]) -> list[Region]:
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


def _themed_box(mobject: Mobject, color: str = C_PANEL) -> VGroup:
    box = SurroundingRectangle(mobject, buff=0.18, corner_radius=0.08)
    box.set_fill(color, opacity=0.38)
    box.set_stroke(C_FRAME, width=0.9, opacity=0.22)
    return VGroup(box, mobject)


def _panel_shell(mobject: Mobject, *, color: str = C_PANEL_SOFT, buff: float = 0.16) -> VGroup:
    shell = SurroundingRectangle(mobject, buff=buff, corner_radius=0.08)
    shell.set_fill(color, opacity=0.26)
    shell.set_stroke(C_FRAME, width=0.8, opacity=0.16)
    return VGroup(shell, mobject)


def _accent_rule(mobject: Mobject, color: str) -> Line:
    rule = Line(LEFT * mobject.width / 2, RIGHT * mobject.width / 2)
    rule.set_stroke(color, width=2.4, opacity=0.92)
    rule.next_to(mobject, UP, buff=0.08)
    return rule


def _formula_stack(*items: VMobject, buff: float = 0.24) -> VGroup:
    return VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def _color_text_parts(mobject: VMobject, colors: dict[str, str]) -> None:
    if mobject.__class__ is MathTex:
        color_substrings(mobject, colors)
        return
    for child in mobject.submobjects:
        if isinstance(child, VMobject):
            _color_text_parts(child, colors)


def _axis_config() -> dict[str, object]:
    return {
        "include_ticks": False,
        "include_numbers": False,
        "stroke_color": C_MUTED,
        "stroke_width": 1.4,
        "stroke_opacity": 0.82,
    }


def _make_axes(
    x_range: tuple[float, float, float],
    y_range: tuple[float, float, float],
    *,
    x_length: float,
    y_length: float,
    preserve_unit_aspect: bool = False,
) -> Axes:
    if preserve_unit_aspect:
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        unit = min(x_length / x_span, y_length / y_span)
        x_length = unit * x_span
        y_length = unit * y_span
    return Axes(
        x_range=[*x_range],
        y_range=[*y_range],
        x_length=x_length,
        y_length=y_length,
        tips=False,
        axis_config=_axis_config(),
    )


def _axes_limits(axes: Axes) -> tuple[float, float, float, float]:
    return (
        float(axes.x_range[0]),
        float(axes.x_range[1]),
        float(axes.y_range[0]),
        float(axes.y_range[1]),
    )


def _axes_point(axes: Axes, point: FloatArray) -> FloatArray:
    return np.asarray(axes.c2p(float(point[0]), float(point[1])), dtype=np.float64)


def _inside_axes(axes: Axes, point: FloatArray) -> bool:
    x_min, x_max, y_min, y_max = _axes_limits(axes)
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def _plot_frame(axes: Axes) -> Rectangle:
    x_min, x_max, y_min, y_max = _axes_limits(axes)
    lower_left = axes.c2p(x_min, y_min)
    upper_right = axes.c2p(x_max, y_max)
    frame = Rectangle(
        width=float(upper_right[0] - lower_left[0]),
        height=float(upper_right[1] - lower_left[1]),
    )
    frame.set_stroke(C_FRAME, width=1.0, opacity=0.9)
    frame.move_to((lower_left + upper_right) / 2)
    return frame


def _quadratic_values(matrix: FloatArray, x_grid: FloatArray, y_grid: FloatArray) -> FloatArray:
    return 0.5 * (
        matrix[0, 0] * x_grid**2
        + 2.0 * matrix[0, 1] * x_grid * y_grid
        + matrix[1, 1] * y_grid**2
    )


def _normalize_heat(values: FloatArray) -> FloatArray:
    lower, upper = np.quantile(values, [0.03, 0.92])
    normalized = np.clip((values - lower) / (upper - lower), 0, 1)
    return normalized**0.72


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    raw = color.removeprefix("#")
    return tuple(int(raw[index : index + 2], 16) for index in (0, 2, 4))


def _blue_alpha_heatmap(values: FloatArray) -> npt.NDArray[np.uint8]:
    intensity = (1.0 - values) ** 1.35
    rgb = np.array(_hex_to_rgb(HEATMAP_BLUE), dtype=np.float64)
    alpha = 255.0 * HEATMAP_MAX_ALPHA * intensity
    channels = [np.full_like(values, channel) for channel in rgb]
    return np.dstack([*channels, alpha]).astype(np.uint8)


def _quadratic_heatmap(axes: Axes, matrix: FloatArray, *, width: int = 640) -> ImageMobject:
    x_min, x_max, y_min, y_max = _axes_limits(axes)
    frame = _plot_frame(axes)
    height = max(80, int(width * frame.height / frame.width))
    x_values = np.linspace(x_min, x_max, width)
    y_values = np.linspace(y_max, y_min, height)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    values = np.log10(_quadratic_values(matrix, x_grid, y_grid) + 1.0)
    image = ImageMobject(_blue_alpha_heatmap(_normalize_heat(values)))
    image.set_width(frame.width)
    image.move_to(frame)
    return image


def _smooth_curve(
    axes: Axes,
    xs: FloatArray,
    fn: Callable[[FloatArray], FloatArray],
    *,
    color: str,
    width: float = 3.0,
    opacity: float = 1.0,
) -> VMobject:
    curve = VMobject()
    ys = fn(xs)
    _, _, y_min, y_max = _axes_limits(axes)
    mask = np.isfinite(ys) & (ys >= y_min) & (ys <= y_max)
    points = [axes.c2p(float(x), float(y)) for x, y in zip(xs[mask], ys[mask], strict=True)]
    if len(points) < 2:
        points = [axes.c2p(float(xs[0]), float(np.clip(ys[0], y_min, y_max)))]
        points.append(points[0] + RIGHT * 1e-3)
    curve.set_points_smoothly(points)
    curve.set_stroke(color, width=width, opacity=opacity)
    return curve


def _polyline(points: Sequence[FloatArray], *, color: str, width: float = 3.0) -> VMobject:
    line = VMobject()
    line.set_points_as_corners([np.asarray(point, dtype=np.float64) for point in points])
    line.set_stroke(color, width=width)
    return line


def _axes_polyline(
    axes: Axes,
    points: FloatArray,
    *,
    color: str,
    width: float = 3.0,
) -> VMobject:
    return _polyline([axes.c2p(float(x), float(y)) for x, y in points], color=color, width=width)


def _axis_arrow(
    axes: Axes,
    start: FloatArray,
    end: FloatArray,
    *,
    color: str,
    width: float = 3.0,
) -> Arrow:
    arrow = Arrow(
        axes.c2p(float(start[0]), float(start[1])),
        axes.c2p(float(end[0]), float(end[1])),
        buff=0,
        color=color,
        stroke_width=width,
        max_tip_length_to_length_ratio=0.09,
    )
    return arrow


def _quadratic_level_sets(axes: Axes, matrix: FloatArray, count: int = 10) -> VGroup:
    x_min, x_max, y_min, y_max = _axes_limits(axes)
    x_values = np.linspace(x_min, x_max, 260)
    y_values = np.linspace(y_min, y_max, 260)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    log_values = np.log10(_quadratic_values(matrix, x_grid, y_grid) + 1.0)
    levels = np.linspace(
        float(np.quantile(log_values, 0.07)),
        float(np.quantile(log_values, 0.92)),
        count,
    )
    generator = contourpy.contour_generator(x=x_values, y=y_values, z=log_values)

    contours = VGroup()
    for opacity, level in zip(np.linspace(0.32, 0.72, len(levels)), levels, strict=True):
        for segment in generator.lines(float(level)):
            if len(segment) < 2:
                continue
            step = max(1, len(segment) // 150)
            sampled = segment[::step]
            if not np.array_equal(sampled[-1], segment[-1]):
                sampled = np.vstack([sampled, segment[-1]])
            line = VMobject()
            line.set_points_as_corners([axes.c2p(float(x), float(y)) for x, y in sampled])
            line.set_stroke(C_CONTOUR, width=1.05, opacity=float(opacity))
            contours.add(line)
    return contours


def _quadratic_panel(
    matrix: FloatArray,
    title: str,
    *,
    x_range: tuple[float, float, float] = (-3.7, 3.7, 1.0),
    y_range: tuple[float, float, float] = (-3.7, 3.7, 1.0),
    x_length: float = 4.2,
    y_length: float = 3.5,
) -> Group:
    axes = _make_axes(
        x_range,
        y_range,
        x_length=x_length,
        y_length=y_length,
        preserve_unit_aspect=True,
    )
    heatmap = _quadratic_heatmap(axes, matrix, width=520).set_z_index(LAYER_HEATMAP)
    levels = _quadratic_level_sets(axes, matrix)
    levels.set_z_index(LAYER_CONTOUR)
    origin = Dot(axes.c2p(0, 0), color=C_TEXT, radius=0.055)
    origin_label = MathTex(r"x^\star", color=C_TEXT, font_size=24).next_to(origin, DOWN + RIGHT)
    label = Caption(title)
    label.next_to(axes, UP, buff=0.14)
    frame = _plot_frame(axes)
    frame.set_stroke(C_FRAME, width=1.0, opacity=0.72)
    frame.set_z_index(LAYER_FRAME)
    axes.set_z_index(LAYER_FRAME)
    markers = VGroup(origin, origin_label).set_z_index(LAYER_MARKERS)
    field = Group(heatmap, levels)
    return Group(frame, axes, field, markers, label, _accent_rule(frame, C_BLUE))


def _gd_path(matrix: FloatArray, x0: FloatArray, eta: float, steps: int) -> FloatArray:
    x = x0.astype(np.float64).copy()
    points = [x.copy()]
    for _ in range(steps):
        x = x - eta * matrix @ x
        points.append(x.copy())
    return np.asarray(points, dtype=np.float64)


def _linear_preconditioner_path(
    matrix: FloatArray,
    x0: FloatArray,
    preconditioner: FloatArray,
    steps: int,
    damping: float = 1.0,
) -> FloatArray:
    x = x0.astype(np.float64).copy()
    path = [x.copy()]
    for _ in range(steps):
        x = x - damping * preconditioner @ (matrix @ x)
        path.append(x.copy())
    return np.asarray(path, dtype=np.float64)


def _coordinate_adagrad_path(matrix: FloatArray, x0: FloatArray, steps: int) -> FloatArray:
    x = x0.astype(np.float64).copy()
    accumulator = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(steps):
        gradient = matrix @ x
        accumulator += gradient**2
        scale = np.sqrt(accumulator)
        normalized = np.divide(gradient, scale, out=np.zeros_like(gradient), where=scale > 0)
        x = x - normalized
        path.append(x.copy())
    return np.asarray(path, dtype=np.float64)


def _radial_adagrad_path(x0: FloatArray, steps: int, eta: float = 0.32) -> FloatArray:
    normalized_position = 1.0
    accumulated_activity = 0.0
    path = [x0.copy()]
    for _ in range(steps):
        accumulated_activity += normalized_position**2
        normalized_position -= eta * normalized_position / np.sqrt(accumulated_activity)
        path.append(normalized_position * x0)
    return np.asarray(path, dtype=np.float64)


def _path_with_dots(
    axes: Axes,
    path: FloatArray,
    *,
    color: str,
    step: int = 2,
    radius: float = 0.035,
) -> VGroup:
    line = _axes_polyline(axes, path[::step], color=color, width=2.5)
    dots = VGroup(
        *(Dot(axes.c2p(float(x), float(y)), color=color, radius=radius) for x, y in path[::step])
    )
    dots[-1].scale(1.3)
    return VGroup(line, dots)


def _mode_path(axes: Axes, eta: float, steps: int = 50) -> VGroup:
    matrix, eigvecs = _rotated_quadratic_matrix()
    x0 = np.array([6.0, 8.0], dtype=np.float64)
    eigenvalues = np.array([MU, LAMBDA_MAX], dtype=np.float64)
    coefficients = eigvecs.T @ x0
    factors = 1.0 - eta * eigenvalues
    path = np.array([eigvecs @ (coefficients * factors**step) for step in range(steps + 1)])

    connectors = VGroup()
    dots = VGroup()
    previous_screen: FloatArray | None = None
    for index, point in enumerate(path[::2]):
        if not np.all(np.isfinite(point)) or not _inside_axes(axes, point):
            previous_screen = None
            continue
        screen_point = _axes_point(axes, point)
        if previous_screen is not None:
            connector = Line(previous_screen, screen_point)
            connector.set_stroke(C_GREEN, width=2.0, opacity=0.88)
            connectors.add(connector)
        dot = Dot(screen_point, color=C_GREEN, radius=0.027)
        if index == 0:
            dot.scale(1.55)
        dots.add(dot)
        previous_screen = screen_point

    trajectory = VGroup(connectors, dots)

    arrows = VGroup()
    for step in range(0, steps + 1, 10):
        opacity = 0.25 + 0.65 * step / steps
        components = [
            coefficients[0] * factors[0] ** step * eigvecs[:, 0],
            coefficients[1] * factors[1] ** step * eigvecs[:, 1],
        ]
        for vector, color, width in (
            (components[0], C_BLUE, 1.4),
            (components[1], C_ORANGE, 1.4),
        ):
            if _inside_axes(axes, vector):
                arrow = _axis_arrow(axes, np.zeros(2), vector, color=color, width=width)
                arrow.set_opacity(opacity)
                arrows.add(arrow)

    return VGroup(trajectory, arrows)


def _response_bars(
    *,
    values: FloatArray,
    color: str,
    width: float,
    height: float,
    title: str,
) -> VGroup:
    max_value = max(float(np.max(values)), 1e-6)
    bars = VGroup()
    count = len(values)
    for index, value in enumerate(values):
        bar_height = height * float(value) / max_value
        bar = Rectangle(width=width / count * 0.72, height=max(bar_height, 0.015))
        bar.set_fill(color, opacity=0.85)
        bar.set_stroke(color, opacity=0)
        x = -width / 2 + width * (index + 0.5) / count
        bar.move_to([x, -height / 2 + bar.height / 2, 0])
        bars.add(bar)

    frame = Rectangle(width=width, height=height)
    frame.set_fill(C_PANEL_DEEP, opacity=0.18)
    frame.set_stroke(C_FRAME, width=0.8, opacity=0.42)
    baseline = Line(frame.get_left(), frame.get_right())
    baseline.set_stroke(C_MUTED, width=0.7, opacity=0.42)
    baseline.align_to(frame, DOWN)
    title_mob = Caption(title)
    title_mob.next_to(frame, UP, buff=0.08)
    return VGroup(frame, baseline, bars, title_mob)


def _gd_response(lambda_i: float, eta: float) -> FloatArray:
    return (1.0 - eta * lambda_i) ** T_STEPS


def _adagrad_coordinate_response(eta: float) -> FloatArray:
    x = 1.0
    history = []
    sum_sq = 0.0
    for _ in T_STEPS:
        history.append(abs(x))
        sum_sq += x * x
        x -= eta * x / np.sqrt(sum_sq)
    return np.asarray(history, dtype=np.float64)


def _slider_alpha(tracker: VT, spec: SliderSpec) -> float:
    value = np.clip(~tracker, spec.minimum, spec.maximum)
    return float((value - spec.minimum) / (spec.maximum - spec.minimum))


def _eta_slider(tracker: VT, spec: SliderSpec) -> VGroup:
    theme = get_active_theme()
    track = Line(LEFT * 1.28, RIGHT * 1.28)
    track.set_stroke(C_MUTED, width=5, opacity=0.45)

    label = MathTex(spec.label, color=spec.color, font_size=theme.typography.caption)
    label.next_to(track, LEFT, buff=0.22)

    value = DN(tracker, num_decimal_places=spec.decimals, font_size=theme.typography.caption - 2)
    value.next_to(track, RIGHT, buff=0.22)
    value.add_updater(lambda mob: mob.next_to(track, RIGHT, buff=0.22))

    fill = always_redraw(
        lambda: Line(
            track.get_start(),
            track.point_from_proportion(_slider_alpha(tracker, spec)),
        ).set_stroke(spec.color, width=7)
    )
    knob = always_redraw(
        lambda: Dot(
            track.point_from_proportion(_slider_alpha(tracker, spec)),
            color=spec.color,
            radius=0.085,
        )
    )

    ticks = VGroup()
    for value_at_tick in (1.0 / LAMBDA_MAX, 2.0 / (MU + LAMBDA_MAX), 2.0 / LAMBDA_MAX):
        tick_alpha = (value_at_tick - spec.minimum) / (spec.maximum - spec.minimum)
        tick = Line(DOWN * 0.075, UP * 0.075)
        tick.set_stroke(C_TEXT, width=1.0, opacity=0.58)
        tick.move_to(track.point_from_proportion(float(np.clip(tick_alpha, 0, 1))))
        ticks.add(tick)

    return VGroup(label, track, ticks, fill, knob, value)


def _mode_magnitude_chart(
    eta: VT,
    lambda_i: float,
    *,
    coefficient: float,
    color: str,
    mode_index: int,
    mode_tag: str,
    width: float,
    height: float,
    show_x_label: bool,
) -> VGroup:
    y_max = 10.0 * np.ceil(max(15.0, coefficient**2 * 1.05) / 10.0)
    frame = Rectangle(width=width, height=height)
    frame.set_stroke(C_FRAME, width=0.9, opacity=0.75)
    baseline = Line(LEFT * width / 2, RIGHT * width / 2)
    baseline.set_stroke(C_MUTED, width=0.8, opacity=0.76)
    baseline.move_to(frame.get_bottom())

    epsilon_y = frame.get_bottom()[1] + frame.height * 0.1 / y_max
    epsilon_line = Line(frame.get_left(), frame.get_right())
    epsilon_line.set_stroke(C_MUTED, width=0.65, opacity=0.52)
    epsilon_line.move_to([frame.get_center()[0], epsilon_y, frame.get_center()[2]])
    epsilon_label = MathTex(r"\epsilon", color=C_MUTED, font_size=14)
    epsilon_label.next_to(epsilon_line, RIGHT, buff=0.04)

    steps = np.arange(0, 151, 2)
    bars = VGroup()
    for index, step in enumerate(steps):
        bar = Rectangle(width=0.01, height=0.01)

        def update_bar(mob: Rectangle, *, step: int = int(step), index: int = index) -> None:
            response = abs(1.0 - ~eta * lambda_i) ** (2 * (step + 1))
            value = float(coefficient**2 * response)
            bar_width = frame.width / len(steps) * 0.62
            bar_height = max(frame.height * min(value / y_max, 1.0), 0.004)
            x = frame.get_left()[0] + frame.width * (index + 0.5) / len(steps)
            y = frame.get_bottom()[1] + bar_height / 2
            replacement = Rectangle(width=bar_width, height=bar_height)
            replacement.set_fill(color, opacity=0.88)
            replacement.set_stroke(color, opacity=0)
            replacement.move_to([x, y, frame.get_center()[2]])
            mob.become(replacement)

        bar.add_updater(update_bar)
        bars.add(bar)

    y_label = MathTex(
        rf"\|(1-\eta\lambda_{mode_index})^{{t+1}}\alpha_{mode_index}v_{mode_index}\|_2^2",
        color=color,
        font_size=15,
    )
    y_label.rotate(PI / 2)
    y_label.next_to(frame, LEFT, buff=0.10)
    tag = MathTex(mode_tag, color=color, font_size=17)
    tag.move_to(frame.get_corner(UL) + RIGHT * 0.60 + DOWN * 0.15)
    r_readout = VGroup(
        MathTex(r"r_i=", color=C_MUTED, font_size=18),
        DN(lambda: 1.0 - ~eta * lambda_i, num_decimal_places=3, font_size=17),
    ).arrange(RIGHT, buff=0.06)
    r_readout.move_to(frame.get_corner(UR) + LEFT * 0.55 + DOWN * 0.18)
    x_label = Caption(r"even iteration $t$") if show_x_label else VGroup()
    if show_x_label:
        x_label.next_to(frame, DOWN, buff=0.08)
    return VGroup(frame, baseline, epsilon_line, epsilon_label, bars, y_label, tag, r_readout, x_label)


def _mode_response_stack(eta: VT, *, width: float = 3.1, height: float = 1.08) -> VGroup:
    title = Caption("squared mode magnitudes")
    _, eigvecs = _rotated_quadratic_matrix()
    coefficients = eigvecs.T @ np.array([6.0, 8.0], dtype=np.float64)
    flat = _mode_magnitude_chart(
        eta,
        MU,
        coefficient=float(coefficients[0]),
        color=C_BLUE,
        mode_index=1,
        mode_tag=r"\lambda_1=\alpha",
        width=width,
        height=height,
        show_x_label=False,
    )
    steep = _mode_magnitude_chart(
        eta,
        LAMBDA_MAX,
        coefficient=float(coefficients[1]),
        color=C_ORANGE,
        mode_index=2,
        mode_tag=r"\lambda_2=\beta",
        width=width,
        height=height,
        show_x_label=True,
    )
    charts = VGroup(flat, steep).arrange(DOWN, buff=0.33, aligned_edge=RIGHT)
    title.next_to(charts, UP, buff=0.12)
    return VGroup(title, charts)


def _bar_chart_for_response(
    title: str,
    top_values: FloatArray,
    bottom_values: FloatArray,
) -> VGroup:
    title_parts = [Caption(part) for part in title.splitlines()]
    column_title = VGroup(*title_parts).arrange(DOWN, buff=0.02) if len(title_parts) > 1 else title_parts[0]
    top = _response_bars(
        values=top_values[::4],
        color=C_BLUE,
        width=2.25,
        height=0.78,
        title=r"$\alpha$ coordinate",
    )
    bottom = _response_bars(
        values=bottom_values[::4],
        color=C_ORANGE,
        width=2.25,
        height=0.78,
        title=r"$\beta$ coordinate",
    )
    bottom.next_to(top, DOWN, buff=0.30)
    top[3].align_to(top[0], LEFT)
    bottom[3].align_to(bottom[0], LEFT)
    rows = VGroup(top, bottom)
    chart = VGroup(column_title, rows).arrange(DOWN, buff=0.16)
    return _panel_shell(chart, buff=0.1)


class SecondOrderApproximation(Slide):
    """Taylor's local quadratic model and the Newton displacement."""

    def construct(self) -> None:
        title = _start_slide(self, "Second-order approximation")
        left, right = _split_weighted(self.region, [1.72, 1.0])
        theme = get_active_theme()

        alpha = VT(0.70)
        beta = VT(3.00)
        center = 0.35
        x_t = -1.15

        def f(xs: FloatArray) -> FloatArray:
            return 0.04 * (xs - center) ** 4 + 0.42 * (xs - 0.15) ** 2 + 0.18

        def grad(x: float) -> float:
            return 0.16 * (x - center) ** 3 + 0.84 * (x - 0.15)

        def hess(x: float) -> float:
            return 0.48 * (x - center) ** 2 + 0.84

        f_t = float(f(np.array([x_t]))[0])
        g_t = grad(x_t)
        h_t = hess(x_t)
        x_next = x_t - g_t / h_t
        roots = np.roots(
            [
                0.16,
                -0.48 * center,
                0.48 * center**2 + 0.84,
                -0.16 * center**3 - 0.84 * 0.15,
            ]
        )
        x_star = min((root.real for root in roots if abs(root.imag) < 1e-8), key=lambda x: f(np.array([x]))[0])
        xs = np.linspace(-2.2, 1.72, 220)

        axes = _make_axes((-2.2, 1.72, 1), (-0.8, 4.2, 1), x_length=7.7, y_length=4.45)
        true_curve = _smooth_curve(axes, xs, f, color=C_TEXT, width=3.2)
        local_model = _smooth_curve(
            axes,
            xs,
            lambda values: f_t + g_t * (values - x_t) + 0.5 * h_t * (values - x_t) ** 2,
            color=C_PURPLE,
            width=2.5,
        )
        lower_model = always_redraw(
            lambda: _smooth_curve(
                axes,
                xs,
                lambda values: f_t + g_t * (values - x_t) + 0.5 * ~alpha * (values - x_t) ** 2,
                color=C_BLUE,
                width=2.0,
                opacity=0.86,
            )
        )
        upper_model = always_redraw(
            lambda: _smooth_curve(
                axes,
                xs,
                lambda values: f_t + g_t * (values - x_t) + 0.5 * ~beta * (values - x_t) ** 2,
                color=C_ORANGE,
                width=2.0,
                opacity=0.86,
            )
        )
        x_t_dot = Dot(axes.c2p(x_t, f_t), color=C_YELLOW, radius=0.07)
        newton_dot = Dot(
            axes.c2p(x_next, f_t + g_t * (x_next - x_t) + 0.5 * h_t * (x_next - x_t) ** 2),
            color=C_PURPLE,
            radius=0.07,
        )
        star_dot = Dot(axes.c2p(float(x_star), float(f(np.array([x_star]))[0])), color=C_TEXT, radius=0.06)
        x_t_label = MathTex(r"x_t", color=C_YELLOW, font_size=26).next_to(x_t_dot, UP + LEFT)
        newton_label = MathTex(r"x_{t+1}", color=C_PURPLE, font_size=26).next_to(newton_dot, DOWN + RIGHT)
        star_label = MathTex(r"x^\star", color=C_TEXT, font_size=26).next_to(star_dot, DOWN + RIGHT)
        x_line = DashedLine(axes.c2p(x_t, -0.72), axes.c2p(x_t, 4.05), color=C_MUTED)
        next_line = DashedLine(axes.c2p(x_next, -0.72), axes.c2p(x_next, 4.05), color=C_PURPLE)
        frame = _plot_frame(axes)
        frame.set_fill(C_PANEL_DEEP, opacity=0.22)
        frame.set_stroke(C_FRAME, width=1.0, opacity=0.72)
        plot_title = Caption("local quadratic interface")
        plot_title.next_to(frame, UP, buff=0.12)
        accent = _accent_rule(frame, C_PURPLE)
        plot = VGroup(
            frame,
            axes,
            true_curve,
            local_model,
            lower_model,
            upper_model,
            x_line,
            next_line,
            x_t_dot,
            newton_dot,
            star_dot,
            x_t_label,
            newton_label,
            star_label,
            plot_title,
            accent,
        )
        left.scale_and_place(plot, buff=0.16)

        taylor = MathTex(
            r"f(x_t+\delta)",
            r"\approx",
            r"f(x_t)+\nabla f(x_t)^\top\delta",
            r"+\frac12\delta^\top\nabla^2 f(x_t)\delta",
            font_size=theme.typography.caption + 2,
        )
        solve = MathTex(
            r"\nabla f(x_t)+\nabla^2f(x_t)\delta=0",
            font_size=theme.typography.caption + 2,
        )
        update = MathTex(
            r"x_{t+1}=x_t-\nabla^2 f(x_t)^{-1}\nabla f(x_t)",
            font_size=theme.typography.caption + 2,
        )
        envelope = MathTex(
            r"\alpha I\preceq \nabla^2 f(x)\preceq \beta I",
            font_size=theme.typography.caption + 2,
        )
        for mob in (taylor, solve, update, envelope):
            color_substrings(
                mob,
                {
                    r"\delta": C_PURPLE,
                    r"x_t": C_YELLOW,
                    r"x_{t+1}": C_PURPLE,
                    r"\alpha": C_BLUE,
                    r"\beta": C_ORANGE,
                },
            )
        readouts = VGroup(
            VGroup(MathTex(r"\alpha=", color=C_BLUE, font_size=24), DN(alpha, num_decimal_places=2)).arrange(RIGHT),
            VGroup(MathTex(r"\beta=", color=C_ORANGE, font_size=24), DN(beta, num_decimal_places=2)).arrange(RIGHT),
        ).arrange(RIGHT, buff=0.35)
        sidebar = _formula_stack(
            Caption("local model"),
            taylor,
            Caption("minimize the model"),
            solve,
            update,
            Caption("curvature envelope"),
            envelope,
            readouts,
            buff=0.22,
        )
        right.scale_and_place(_themed_box(sidebar), buff=0.22)

        self.play(Write(title), FadeIn(frame), Write(accent), Create(axes), Write(true_curve), FadeIn(x_t_dot, x_t_label, plot_title))
        self.play(Write(taylor))
        self.fragment(title="Newton model")
        self.play(Write(local_model), FadeIn(newton_dot, newton_label, star_dot, star_label), Write(solve), Write(update))
        self.fragment(title="Curvature envelope")
        self.play(Write(lower_model), Write(upper_model), Write(envelope), FadeIn(readouts))
        self.play(alpha @ 0.42, beta @ 4.2, run_time=2.0)
        self.play(alpha @ 0.95, beta @ 2.35, run_time=2.0)


class QuadraticRotation(Slide):
    """Rotate a quadratic into its Hessian eigenbasis."""

    def construct(self) -> None:
        title = _start_slide(self, "The quadratic microscope")
        matrix, eigvecs = _rotated_quadratic_matrix()
        diagonal = np.diag([MU, LAMBDA_MAX])
        body, strip = _split_rows(self.region, [4.3, 1.0])
        left, mid, right = _split_weighted(body, [1.0, 0.42, 1.0])

        original = _quadratic_panel(matrix, "original coordinates", x_length=4.4, y_length=4.4)
        flat_vec = eigvecs[:, 0]
        steep_vec = eigvecs[:, 1]
        original_axes = original[1]
        original.add(
            _axis_arrow(original_axes, np.zeros(2), 3.25 * flat_vec, color=C_BLUE),
            _axis_arrow(original_axes, np.zeros(2), 1.75 * steep_vec, color=C_ORANGE),
            MathTex(r"v_{\min}", color=C_BLUE, font_size=23).move_to(
                original_axes.c2p(*(3.55 * flat_vec))
            ),
            MathTex(r"v_{\max}", color=C_ORANGE, font_size=23).move_to(
                original_axes.c2p(*(2.05 * steep_vec))
            ),
        )
        eigenbasis = _quadratic_panel(diagonal, "eigenbasis coordinates", x_length=4.4, y_length=4.4)
        eigen_axes = eigenbasis[1]
        eigenbasis.add(
            _axis_arrow(eigen_axes, np.zeros(2), np.array([3.25, 0.0]), color=C_BLUE),
            _axis_arrow(eigen_axes, np.zeros(2), np.array([0.0, 1.75]), color=C_ORANGE),
            MathTex(r"v_{\min}", color=C_BLUE, font_size=23).move_to(eigen_axes.c2p(3.55, 0.22)),
            MathTex(r"v_{\max}", color=C_ORANGE, font_size=23).move_to(eigen_axes.c2p(0.34, 2.05)),
        )
        left.scale_and_place(original)
        right.scale_and_place(eigenbasis)

        map_label = VGroup(
            MathTex(r"x\mapsto", color=C_PURPLE, font_size=25),
            MathTex(r"V^\top(x-x^\star)", color=C_PURPLE, font_size=25),
        ).arrange(DOWN, buff=0.04)
        map_arrow = Arrow(LEFT, RIGHT, color=C_PURPLE, buff=0)
        map_group = VGroup(map_label, map_arrow).arrange(DOWN, buff=0.18)
        mid.scale_and_place(map_group, buff=0.08)

        equations = VGroup(
            MathTex(r"f(x)=\frac12(x-x^\star)^\top A(x-x^\star)", font_size=30),
            MathTex(r"V^\top A V=\operatorname{diag}(\alpha,\beta)", font_size=30),
            MathTex(r"A v_i=\lambda_i v_i", font_size=30),
            MathTex(r"x_0-x^\star=\sum_i \alpha_i v_i", font_size=30),
        ).arrange(RIGHT, buff=0.42)
        _color_text_parts(
            equations,
            {
                r"v_i": C_BLUE,
                r"\lambda_i": C_ORANGE,
                r"\alpha_i": C_GREEN,
                r"\alpha": C_BLUE,
                r"\beta": C_ORANGE,
            },
        )
        strip.scale_and_place(_themed_box(equations), buff=0.1)

        self.play(Write(title), FadeIn(original))
        self.fragment(title="Rotate into modes")
        self.play(Write(map_group), FadeIn(eigenbasis))
        self.fragment(title="Separated recurrence")
        self.play(Write(equations))


class GradientDescentModes(Slide):
    """Gradient descent as independent eigenmode multipliers."""

    def construct(self) -> None:
        title = _start_slide(self, "Gradient descent is a scalar compromise")
        eta = VT(1.0 / LAMBDA_MAX)
        matrix, _ = _rotated_quadratic_matrix()
        top, bottom = _split_rows(self.region, [2.0, 1.0])
        top_left, top_right = _split_weighted(top, [3.0, 2.0])

        axes = _make_axes(
            (-2.75, 10.75, 2),
            (-4.35, 9.35, 2),
            x_length=6.6,
            y_length=4.2,
            preserve_unit_aspect=True,
        )
        heatmap = _quadratic_heatmap(axes, matrix).set_z_index(LAYER_HEATMAP)
        contours = _quadratic_level_sets(axes, matrix, count=14).set_z_index(LAYER_CONTOUR)
        frame = _plot_frame(axes).set_z_index(LAYER_FRAME)
        origin = Dot(axes.c2p(0, 0), color=C_TEXT, radius=0.055).set_z_index(LAYER_MARKERS)
        origin_label = MathTex(r"x^\star", color=C_TEXT, font_size=24).next_to(origin, DOWN + RIGHT)
        start = Dot(axes.c2p(6, 8), color=C_GREEN, radius=0.065).set_z_index(LAYER_MARKERS)
        start_label = MathTex(r"x_0", color=C_GREEN, font_size=24).next_to(start, UP + RIGHT)
        markers = VGroup(origin, origin_label, start, start_label)
        plot_shell = Group(heatmap, contours, axes, frame, markers)
        top_left.scale_and_place(plot_shell, buff=0.08)

        slider = _eta_slider(
            eta,
            SliderSpec(r"\eta", 0.0, 2.0 / LAMBDA_MAX, 3, C_ORANGE),
        )
        slider.scale(0.84)
        slider.move_to(frame.get_corner(UL) + RIGHT * 1.55 + DOWN * 0.32)
        trajectory = always_redraw(lambda: _mode_path(axes, ~eta).set_z_index(LAYER_TRAJECTORY))

        responses = _mode_response_stack(eta)
        top_right.scale_and_place(responses, buff=0.1)
        responses.update()

        rule = MathTex(
            r"x_{t+1}-x^\star=\sum_i(1-\eta\lambda_i)^{t+1}\alpha_i v_i",
            font_size=30,
        )
        modes = MathTex(r"r_i=1-\eta\lambda_i", font_size=30)
        energy = MathTex(
            r"f(x_t)-f_\star=\frac12\sum_i\lambda_i\alpha_i^2(1-\eta\lambda_i)^{2t}",
            font_size=30,
        )
        rho = VGroup(
            MathTex(r"\rho_{\mathrm{GD}}(\eta)=\max\{|1-\eta\alpha|,\ |1-\eta\beta|\}=", font_size=28),
            DN(
                lambda: max(abs(1.0 - ~eta * MU), abs(1.0 - ~eta * LAMBDA_MAX)),
                num_decimal_places=3,
                font_size=25,
            ),
        ).arrange(RIGHT, buff=0.08)
        factors = VGroup(
            VGroup(
                MathTex(r"1-\eta\alpha=", color=C_BLUE, font_size=25),
                DN(lambda: 1.0 - ~eta * MU, num_decimal_places=3, font_size=23),
            ).arrange(RIGHT, buff=0.06),
            VGroup(
                MathTex(r"1-\eta\beta=", color=C_ORANGE, font_size=25),
                DN(lambda: 1.0 - ~eta * LAMBDA_MAX, num_decimal_places=3, font_size=23),
            ).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.36)
        equations = VGroup(
            VGroup(rule, modes).arrange(RIGHT, buff=0.55),
            VGroup(energy, rho).arrange(RIGHT, buff=0.55),
            factors,
        ).arrange(DOWN, buff=0.18)
        _color_text_parts(
            equations,
            {
                r"\eta": C_ORANGE,
                r"\lambda_i": C_ORANGE,
                r"\alpha_i": C_GREEN,
                r"\alpha": C_BLUE,
                r"\beta": C_ORANGE,
                r"r_i": C_GREEN,
            },
        )
        bottom.scale_and_place(_themed_box(equations), buff=0.12)

        self.play(
            Write(title),
            FadeIn(heatmap),
            Write(contours),
            Write(frame),
            FadeIn(markers),
        )
        self.play(Write(trajectory), FadeIn(slider), FadeIn(responses))
        self.play(Write(equations))
        self.fragment(title="Balance the endpoints")
        self.play(eta @ (2.0 / (MU + LAMBDA_MAX)), run_time=3.0)
        self.fragment(title="Let the steep mode oscillate")
        self.play(eta @ (0.5 * (2.0 / (MU + LAMBDA_MAX) + 2.0 / LAMBDA_MAX)), run_time=3.0)
        self.fragment(title="Return to the safe step")
        self.play(eta @ (1.0 / LAMBDA_MAX), run_time=2.4)


class AdaGradKnownRuler(Slide):
    """Known diagonal curvature is perfect only when axes agree."""

    def construct(self) -> None:
        title = _start_slide(self, "A diagonal ruler can rescale, not rotate")
        body, legend_region = _split_rows(self.region, [4.4, 0.82])
        left, right = _split_weighted(body, [1.0, 1.0])
        rotated_matrix, _ = _rotated_quadratic_matrix()
        axis_matrix = np.diag([MU, LAMBDA_MAX])
        x0 = np.array([2.0, 4.0], dtype=np.float64)
        eta_gd = 1.0 / (MU + LAMBDA_MAX)

        panels = []
        for region, matrix, label, radial in (
            (left, rotated_matrix, "rotated quadratic", False),
            (right, axis_matrix, "axis-aligned quadratic", True),
        ):
            panel = _quadratic_panel(
                matrix,
                label,
                x_range=(-2.5, 8.8, 1),
                y_range=(-2.5, 8.8, 1),
                x_length=5.0,
                y_length=3.85,
            )
            axes = panel[1]
            gd = _linear_preconditioner_path(matrix, x0, np.eye(2), 102, eta_gd)
            diagonal = np.diag(np.diag(matrix))
            known = _linear_preconditioner_path(matrix, x0, np.linalg.inv(diagonal), 65)
            adagrad = _radial_adagrad_path(x0, 71) if radial else _coordinate_adagrad_path(matrix, x0, 71)
            paths = VGroup(
                _path_with_dots(axes, gd, color=C_GREEN, step=4),
                _path_with_dots(axes, known, color=C_PURPLE, step=3),
                _path_with_dots(axes, adagrad, color=C_TEAL, step=3),
            )
            start = Dot(axes.c2p(float(x0[0]), float(x0[1])), color=C_YELLOW, radius=0.06)
            start_label = MathTex(r"x_0", color=C_YELLOW, font_size=24).next_to(start, UR)
            panel.add(paths, start, start_label)
            region.scale_and_place(panel, buff=0.08)
            panels.append(panel)

        legend = VGroup(
            VGroup(Dot(color=C_GREEN), Caption(r"scalar GD")).arrange(RIGHT, buff=0.14),
            VGroup(Dot(color=C_PURPLE), Caption(r"known diagonal inverse")).arrange(RIGHT, buff=0.14),
            VGroup(Dot(color=C_TEAL), Caption(r"AdaGrad coordinate ruler")).arrange(RIGHT, buff=0.14),
            MathTex(r"x_{t+1}=x_t-D_t^{-1}\nabla f(x_t)", font_size=28),
        ).arrange(RIGHT, buff=0.52)
        _color_text_parts(legend, {r"D_t": C_TEAL})
        legend_region.scale_and_place(_themed_box(legend), buff=0.08)

        self.play(Write(title), *(FadeIn(panel[:6]) for panel in panels))
        self.fragment(title="Compare rulers")
        self.play(*(Write(panel[6][0]) for panel in panels), FadeIn(legend[0]))
        self.play(*(Write(panel[6][1]) for panel in panels), FadeIn(legend[1]))
        self.play(*(Write(panel[6][2]) for panel in panels), FadeIn(legend[2:]))


class AdaGradDiagonalScaling(Slide):
    """Diagonal scaling squeezes level sets toward circles."""

    def construct(self) -> None:
        title = _start_slide(self, "Diagonal scaling as geometry")
        body, note_region = _split_rows(self.region, [4.7, 0.55])
        rows = _split_rows(body, [1.0, 1.0])
        rotated_matrix, _ = _rotated_quadratic_matrix()
        axis_matrix = np.diag([MU, LAMBDA_MAX])

        row_groups = []
        for row_region, matrix, label in (
            (rows[0], axis_matrix, "axis-aligned"),
            (rows[1], rotated_matrix, "rotated"),
        ):
            before_region, arrow_region, after_region = _split_weighted(row_region, [1.0, 0.30, 1.0])
            diagonal = np.diag(np.diag(matrix))
            scaled = np.linalg.inv(np.sqrt(diagonal)) @ matrix @ np.linalg.inv(np.sqrt(diagonal))
            before = _quadratic_panel(matrix, f"{label} quadratic", x_length=4.0, y_length=2.65)
            after = _quadratic_panel(scaled, "after diagonal scaling", x_length=4.0, y_length=2.65)
            before_region.scale_and_place(before, buff=0.08)
            after_region.scale_and_place(after, buff=0.08)
            arrow = VGroup(
                MathTex(r"x\mapsto", color=C_PURPLE, font_size=22),
                MathTex(r"D_A^{-1/2}x", color=C_PURPLE, font_size=22),
                Arrow(LEFT, RIGHT, color=C_PURPLE, buff=0),
            ).arrange(DOWN, buff=0.06)
            arrow_region.scale_and_place(arrow, buff=0.06)
            row_groups.append((before, arrow, after))

        note = MathTex(
            r"D_A=\operatorname{diag}(A_{11},A_{22})",
            r"\qquad",
            r"D_A^{-1/2}AD_A^{-1/2}",
            font_size=28,
        )
        color_substrings(note, {r"D_A": C_PURPLE})
        note_region.scale_and_place(note, buff=0.08)

        self.play(Write(title), *(FadeIn(before) for before, _, _ in row_groups))
        self.fragment(title="Apply diagonal map")
        self.play(*(Write(arrow) for _, arrow, _ in row_groups), Write(note))
        self.play(*(FadeIn(after) for _, _, after in row_groups))


class AdaGradCoordinateResponse(Slide):
    """Compare fixed GD multipliers with AdaGrad's online coordinate gain."""

    def construct(self) -> None:
        title = _start_slide(self, "AdaGrad changes the gain while it moves")
        chart_region, equation_region = _split_weighted(self.region, [2.1, 0.92])
        columns = _split_weighted(chart_region, [1.0, 1.0, 1.0])
        specs = [
            ResponseSpec("GD: safe global step", "GD", 1.0 / LAMBDA_MAX),
            ResponseSpec("GD: balanced\ninverse-curvature step", "GD", 2.0 / (MU + LAMBDA_MAX)),
            ResponseSpec("AdaGrad: activity counter", "AdaGrad", 0.15),
        ]

        charts = []
        adagrad_response = _adagrad_coordinate_response(0.15)
        for region, spec in zip(columns, specs, strict=True):
            if spec.method == "GD":
                top = np.abs(_gd_response(MU, spec.eta))
                bottom = np.abs(_gd_response(LAMBDA_MAX, spec.eta))
            else:
                top = adagrad_response
                bottom = adagrad_response
            chart = _bar_chart_for_response(spec.title, top, bottom)
            region.scale_and_place(chart, buff=0.16)
            charts.append(chart)

        equations = _formula_stack(
            Caption("axis-aligned quadratic"),
            MathTex(r"f(x)=\frac12\sum_i\lambda_i x_i^2", font_size=30),
            Caption("AdaGrad accumulator"),
            MathTex(r"G_{t,i}=\sum_{\tau=1}^{t}g_{\tau,i}^2", font_size=30),
            MathTex(r"x_{t+1,i}=x_{t,i}-\eta\frac{g_{t,i}}{\sqrt{G_{t,i}}}", font_size=29),
            Caption(r"because $g_{t,i}=\lambda_i x_{t,i}$"),
            MathTex(
                r"\frac{\lambda_i x_{t,i}}{\sqrt{\sum_\tau\lambda_i^2x_{\tau,i}^2}}"
                r"=\frac{x_{t,i}}{\sqrt{\sum_\tau x_{\tau,i}^2}}",
                font_size=27,
            ),
        )
        _color_text_parts(equations, {r"\lambda_i": C_ORANGE, r"G_{t,i}": C_TEAL, r"\eta": C_YELLOW})
        equation_region.scale_and_place(_themed_box(equations), buff=0.18)

        self.play(Write(title))
        self.play(Write(charts[0]))
        self.fragment(title="A better scalar")
        self.play(Write(charts[1]))
        self.fragment(title="AdaGrad normalization")
        self.play(Write(charts[2]), Write(equations))


class AdaGradWeightedLedger(Slide):
    """The inner-product lemma as a change of ruler."""

    def construct(self) -> None:
        title = _start_slide(self, "The weighted ledger identity")
        visual_region, proof_region = _split_rows(self.region, [2.0, 1.0])
        left, arrow_region, right = _split_weighted(visual_region, [1.0, 0.30, 1.0])

        d_sqrt = np.diag([2.0, 1.0])
        d_inv_sqrt = np.linalg.inv(d_sqrt)
        weighted_points = {
            "star": np.array([0.0, 0.0], dtype=np.float64),
            "xt": np.array([-2.0, 3.0], dtype=np.float64),
            "xt1": np.array([2.0, 1.5], dtype=np.float64),
        }
        euclidean_points = {name: d_inv_sqrt @ point for name, point in weighted_points.items()}

        weighted = self._triangle_panel(weighted_points, "weighted coordinates", r"D_t")
        euclidean = self._triangle_panel(euclidean_points, "Euclidean coordinates", r"2")
        left.scale_and_place(weighted, buff=0.08)
        right.scale_and_place(euclidean, buff=0.08)

        map_arrow = VGroup(
            MathTex(r"x\mapsto", color=C_PURPLE, font_size=22),
            MathTex(r"D_t^{-1/2}x", color=C_PURPLE, font_size=22),
            Arrow(LEFT, RIGHT, color=C_PURPLE, buff=0),
        ).arrange(DOWN, buff=0.06)
        arrow_region.scale_and_place(map_arrow, buff=0.06)

        ledger_lines = VGroup(
            MathTex(r"2\eta\langle g_t,x_t-x^\star\rangle", font_size=31),
            MathTex(
                r"= \|x_t-x^\star\|_{D_t}^2-\|x_{t+1}-x^\star\|_{D_t}^2",
                font_size=30,
            ),
            MathTex(r"+ \|x_t-x_{t+1}\|_{D_t}^2", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        setup = MathTex(r"x_{t+1}=x_t-\eta D_t^{-1}g_t", font_size=31)
        for line in ledger_lines:
            line.set_color_by_tex(r"\eta", C_YELLOW)
            line.set_color_by_tex(r"D_t", C_PURPLE)
        setup.set_color_by_tex(r"\eta", C_YELLOW)
        setup.set_color_by_tex(r"D_t", C_PURPLE)
        proof = _formula_stack(
            Caption("weighted parallelogram bookkeeping"),
            VGroup(setup, ledger_lines).arrange(RIGHT, buff=0.7),
            Caption("progress = distance drop + movement"),
            buff=0.18,
        )
        proof_region.scale_and_place(_themed_box(proof), buff=0.12)

        self.play(Write(title), FadeIn(weighted))
        self.fragment(title="Change the ruler")
        self.play(Write(map_arrow), FadeIn(euclidean))
        self.fragment(title="Read off the ledger")
        self.play(Write(proof))

    def _triangle_panel(
        self,
        points: dict[str, FloatArray],
        title: str,
        norm_suffix: str,
    ) -> Group:
        axes = _make_axes(
            (-2.8, 3.4, 1),
            (-1.0, 3.9, 1),
            x_length=4.65,
            y_length=3.35,
            preserve_unit_aspect=True,
        )
        matrix = np.diag([0.55, 1.55]) if norm_suffix == "D_t" else np.eye(2)
        heatmap = _quadratic_heatmap(axes, matrix, width=430).set_z_index(LAYER_HEATMAP)
        levels = _quadratic_level_sets(axes, matrix, count=7)
        levels.set_z_index(LAYER_CONTOUR)
        if norm_suffix == "D_t":
            labels = {
                "star": r"D_t^{1/2}x^\star",
                "xt": r"D_t^{1/2}x_t",
                "xt1": r"D_t^{1/2}x_{t+1}",
            }
        else:
            labels = {
                "star": r"x^\star",
                "xt": r"x_t",
                "xt1": r"x_{t+1}",
            }
        p_star = points["star"]
        p_xt = points["xt"]
        p_xt1 = points["xt1"]
        old_distance = DashedLine(
            axes.c2p(float(p_xt[0]), float(p_xt[1])),
            axes.c2p(float(p_star[0]), float(p_star[1])),
            color=C_MUTED,
        )
        new_distance = DashedLine(
            axes.c2p(float(p_xt1[0]), float(p_xt1[1])),
            axes.c2p(float(p_star[0]), float(p_star[1])),
            color=C_MUTED,
        )
        step = _axis_arrow(axes, p_xt, p_xt1, color=C_RED)
        dots = VGroup(
            *(
                Dot(axes.c2p(float(point[0]), float(point[1])), color=C_TEXT, radius=0.055)
                for point in points.values()
            )
        )
        point_labels = VGroup(
            *(
                MathTex(label, font_size=18 if norm_suffix == "D_t" else 22, color=C_TEXT).next_to(
                    Dot(axes.c2p(float(points[key][0]), float(points[key][1])), radius=0),
                    direction,
                    buff=0.08,
                )
                for key, label, direction in (
                    ("star", labels["star"], DL),
                    ("xt", labels["xt"], UP),
                    ("xt1", labels["xt1"], RIGHT),
                )
            )
        )
        side_labels = VGroup(
            MathTex(rf"\|x_t-x^\star\|_{{{norm_suffix}}}^2", color=C_MUTED, font_size=20),
            MathTex(rf"\|x_{{t+1}}-x^\star\|_{{{norm_suffix}}}^2", color=C_MUTED, font_size=20),
            MathTex(rf"\|x_t-x_{{t+1}}\|_{{{norm_suffix}}}^2", color=C_RED, font_size=20),
        )
        side_labels[0].move_to(axes.c2p(-1.0, 1.35))
        side_labels[1].move_to(axes.c2p(1.0, 0.55))
        side_labels[2].move_to(axes.c2p(0.1, 2.45))
        caption = Caption(title).next_to(axes, UP, buff=0.12)
        frame = Rectangle(width=axes.width, height=axes.height)
        frame.move_to(axes)
        frame.set_stroke(C_FRAME, width=0.8, opacity=0.54)
        field = Group(heatmap, levels)
        distances = VGroup(old_distance, new_distance, step, dots, point_labels, side_labels)
        return Group(frame, axes, field, distances, caption, _accent_rule(frame, C_TEAL))
