"""Manim-native figure slides for the AdaGrad lecture notes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import contourpy
import numpy as np
import numpy.typing as npt
from manim import (
    DL,
    DOWN,
    DR,
    LEFT,
    MED_LARGE_BUFF,
    MED_SMALL_BUFF,
    ORIGIN,
    RIGHT,
    SMALL_BUFF,
    UP,
    UL,
    UR,
    WHITE,
    Arrow,
    Axes,
    BraceBetweenPoints,
    Circle,
    Create,
    CurvedArrow,
    DashedLine,
    DashedVMobject,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    ImageMobject,
    Line,
    MathTex,
    PI,
    Rectangle,
    TexTemplate,
    VGroup,
    VMobject,
    Write,
    always_redraw,
)
from simplex import Caption, DN, Slide, TexPage, VT, color_substrings, get_active_theme

from slides.controls import SliderSpec, ValueSlider
from slides.plotting import (
    axes_limits as _axes_limits,
    axes_point as _axes_point,
    blue_alpha_heatmap as _blue_alpha_heatmap,
    inside_axes as _inside_axes,
    make_axes as _make_axes,
    normalize_heat as _normalize_heat,
    plot_frame as _plot_frame,
)
from slides.style import (
    C_BLUE,
    C_CONTOUR,
    C_FRAME,
    C_GRID,
    C_GREEN,
    C_MUTED,
    C_ORANGE,
    C_PANEL_DEEP,
    C_PURPLE,
    C_RED,
    C_TEAL,
    C_TEXT,
    C_YELLOW,
    LAYER_CONTOUR,
    LAYER_FRAME,
    LAYER_HEATMAP,
    LAYER_MARKERS,
    LAYER_TRAJECTORY,
    PANEL_BUFF,
    accent_rule as _accent_rule,
    color_text_parts as _color_text_parts,
    formula_stack as _formula_stack,
    label_for_dot,
    split_rows as _split_rows,
    split_weighted as _split_weighted,
    start_slide as _start_slide,
    themed_box as _themed_box,
    panel_shell as _panel_shell,
    theme_math,
)

FloatArray = npt.NDArray[np.float64]

MU = 0.02
LAMBDA_MAX = 1.0
BASE_THETA_DEG = 28.0
BASE_INITIAL_VECTOR = np.array([5.05, 1.42], dtype=np.float64)
RESPONSE_STEP_COUNT = 150
T_STEPS = np.arange(0, RESPONSE_STEP_COUNT + 1, dtype=np.float64)

ZERO_AXIS_EPSILON = 1e-8
GRID_TICK_STOP_EPSILON = 0.1
POLYLINE_FALLBACK_STEP = 1e-3
GRID_LINE_STROKE_WIDTH = 0.45
GRID_LINE_OPACITY = 0.32
GRID_AXIS_INSET_RATIO = 0.03
GRID_AXIS_STROKE_WIDTH = 0.75
GRID_AXIS_OPACITY = 0.78
GRID_AXIS_TIP_RATIO = 0.018
BOTTOM_TICK_HEIGHT_RATIO = 0.035
BOTTOM_TICK_STROKE_WIDTH = 1.1
TICK_LABEL_TEX_SCALE = 2
X_MARKER_STROKE_WIDTH = 2.6
X_MARKER_OPACITY = 0.95
X_MARKER_FRAME_HEIGHT_RATIO = 0.05
CONTOUR_GRID_SAMPLES = 260
CONTOUR_POLYLINE_MAX_POINTS = 150
CONTOUR_STROKE_WIDTH = 1.05
CONTOUR_OPACITY_RANGE = (0.32, 0.72)
CONTOUR_QUANTILE_RANGE = (0.07, 0.92)
AXIS_ARROW_TIP_RATIO = 0.09
DEFAULT_QUADRATIC_HEATMAP_WIDTH = 640
QUADRATIC_PANEL_HEATMAP_WIDTH = 520
LEDGER_HEATMAP_WIDTH = 430
HEATMAP_MIN_HEIGHT = 80
PANEL_MARKER_FRAME_HEIGHT_RATIO = 1 / 64
PATH_LINE_STROKE_WIDTH = 2.5
PATH_DOT_FRAME_HEIGHT_RATIO = 1 / 110
PATH_END_DOT_SCALE = 1.3
MODE_TRAJECTORY_STROKE_WIDTH = 2.0
MODE_TRAJECTORY_OPACITY = 0.88
MODE_DOT_FRAME_HEIGHT_RATIO = 1 / 155
MODE_START_DOT_SCALE = 1.55
MODE_VECTOR_ARROW_SCALE = 1.4
MODE_ARROW_INTERVAL = 10
MODE_ARROW_OPACITY_RANGE = (0.25, 0.9)
MODE_INITIAL_POINT = np.array([6.0, 8.0], dtype=np.float64)
MODE_TRAJECTORY_STEPS = 50
RESPONSE_BAR_WIDTH_FRACTION = 0.72
RESPONSE_BAR_MIN_HEIGHT = 0.015
RESPONSE_BAR_OPACITY = 0.85
RESPONSE_FRAME_FILL_OPACITY = 0.18
RESPONSE_FRAME_STROKE_OPACITY = 0.42
RESPONSE_BASELINE_STROKE_WIDTH = 0.7
RESPONSE_FRAME_STROKE_WIDTH = 0.8
RESPONSE_CHART_WIDTH = 2.25
RESPONSE_CHART_HEIGHT = 0.78
RESPONSE_SAMPLE_STRIDE = 4
ETA_SLIDER_HALF_LENGTH = 1.28
MODE_CHART_STEP_STRIDE = 2
MODE_CHART_MIN_Y_MAX = 15.0
MODE_CHART_Y_STEP = 10.0
MODE_CHART_HEADROOM = 1.05
MODE_CHART_FRAME_STROKE_WIDTH = 0.9
MODE_CHART_FRAME_OPACITY = 0.75
MODE_CHART_BASELINE_STROKE_WIDTH = 0.8
MODE_CHART_BASELINE_OPACITY = 0.76
MODE_CHART_EPSILON = 0.1
MODE_CHART_EPSILON_STROKE_WIDTH = 0.65
MODE_CHART_EPSILON_OPACITY = 0.52
MODE_CHART_BAR_WIDTH_FRACTION = 0.62
MODE_CHART_BAR_MIN_HEIGHT_RATIO = 1 / 200
MODE_CHART_BAR_PLACEHOLDER_RATIO = 1 / 100
MODE_CHART_BAR_OPACITY = 0.88
MODE_CHART_TAG_OFFSET = RIGHT * 0.60 + DOWN * 0.15
MODE_CHART_READOUT_OFFSET = LEFT * 0.55 + DOWN * 0.18
MULTILINE_TITLE_BUFF = SMALL_BUFF / 5
LEDGER_PROOF_COLUMN_BUFF = 0.7
LEDGER_LABEL_DISTANCE = 0.17
LEDGER_LABEL_OUTER_DISTANCE = 0.23
LEDGER_B_LABEL_POSITION = 0.47
LEDGER_NEW_DISTANCE_LABEL_POSITION = 0.52
LEDGER_STEP_LABEL_POSITION = 0.54
LEDGER_EUCLIDEAN_STEP_LABEL_POSITION = 0.35
LEDGER_EUCLIDEAN_STEP_LABEL_DISTANCE = 0.24
LEDGER_POINT_LABEL_DISTANCE = 0.38
LEDGER_P_PROJECTION_START = 0.16
LEDGER_P_PROJECTION_END = 0.42
LEDGER_P_POINT_POSITION = 0.40
LEDGER_FRAME_STROKE_WIDTH = 0.8
LEDGER_FRAME_STROKE_OPACITY = 0.54
LOCAL_CURVE_SAMPLES = 220
LOCAL_ALPHA_INITIAL = 0.70
LOCAL_BETA_INITIAL = 3.00
LOCAL_MODEL_CENTER = 0.35
LOCAL_CURRENT_X = -1.15
LOCAL_QUADRATIC_CENTER = 0.15
LOCAL_QUARTIC_WEIGHT = 0.04
LOCAL_QUADRATIC_WEIGHT = 0.42
LOCAL_VALUE_OFFSET = 0.18
LOCAL_X_RANGE = (-2.2, 1.72, 1.0)
LOCAL_AXIS_X_LENGTH = 7.7
LOCAL_AXIS_Y_LENGTH = 4.45
LOCAL_Y_PADDING_BELOW = 0.10
LOCAL_Y_PADDING_ABOVE = 0.12
LOCAL_Y_MAX_CAP = 6.3
LOCAL_CURVE_STROKE_WIDTH = 3.2
LOCAL_MODEL_STROKE_WIDTH = 2.5
LOCAL_BOUND_STROKE_WIDTH = 2.0
LOCAL_BOUND_OPACITY = 0.86
LOCAL_MODEL_DASH_COUNT = 56
LOCAL_BOTTOM_AXIS_STROKE_WIDTH = 0.9
LOCAL_BOTTOM_AXIS_OPACITY = 0.82
LOCAL_VERTICAL_GUIDE_STROKE_WIDTH = 0.8
LOCAL_CURRENT_GUIDE_OPACITY = 0.65
LOCAL_NEXT_GUIDE_OPACITY = 0.62
LOCAL_BRACKET_Y_RATIO = 0.03
LOCAL_BRACKET_TICK_HEIGHT_RATIO = 0.03
LOCAL_BRACKET_HALF_TICK_SCALE = 0.45
LOCAL_BRACKET_STROKE_WIDTH = 1.35
LOCAL_BRACKET_LABEL_Y_RATIO = 0.055
LOCAL_LEGEND_FRAME_WIDTH_RATIO = 0.34
LOCAL_LEGEND_FRAME_HEIGHT_RATIO = 0.30
LOCAL_LEGEND_SAMPLE_FRAME_WIDTH_RATIO = LOCAL_LEGEND_FRAME_WIDTH_RATIO / 6
LOCAL_PANEL_FILL_OPACITY = 0.22
LOCAL_PANEL_STROKE_WIDTH = 1.0
LOCAL_PANEL_STROKE_OPACITY = 0.72
LOCAL_F_LABEL_OFFSET = np.array([-0.17, 0.48], dtype=np.float64)
LOCAL_STAR_LABEL_OFFSET = np.array([0.22, -0.23], dtype=np.float64)
SECOND_ORDER_MARKER_FRAME_HEIGHT_RATIO = 1 / 64
SECOND_ORDER_STAR_DOT_SCALE = 0.85
ADAGRAD_ACTIVITY_STEP_FRACTION = 0.15
ADAGRAD_ACTIVITY_STEP_PERCENT = int(100 * ADAGRAD_ACTIVITY_STEP_FRACTION)

@dataclass(frozen=True, slots=True)
class ResponseSpec:
    title: str
    method: str
    eta: float


def _rotation(theta_deg: float) -> FloatArray:
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _quadratic_eigendecomposition(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order].astype(np.float64, copy=False)
    eigenvectors = eigenvectors[:, order].astype(np.float64, copy=True)
    for index in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, index]
        anchor = int(np.argmax(np.abs(vector)))
        if vector[anchor] < 0:
            eigenvectors[:, index] = -vector
    return eigenvalues, eigenvectors


def _quadratic_axis_matrix(matrix: FloatArray) -> FloatArray:
    eigenvalues, _ = _quadratic_eigendecomposition(matrix)
    return np.diag(eigenvalues)


def _theta_deg() -> float:
    base = _rotation(BASE_THETA_DEG) @ BASE_INITIAL_VECTOR
    offset = np.rad2deg(np.arctan2(base[0] - base[1], base[0] + base[1]))
    return float(BASE_THETA_DEG + offset)


THETA_DEG = _theta_deg()


def _rotated_quadratic_matrix() -> tuple[FloatArray, FloatArray]:
    basis = _rotation(THETA_DEG)
    matrix = basis @ np.diag([MU, LAMBDA_MAX]) @ basis.T
    _, eigvecs = _quadratic_eigendecomposition(matrix)
    return matrix, eigvecs


def _coordinate_grid(axes: Axes, *, step: float = 1.0) -> VGroup:
    x_min, x_max, y_min, y_max = _axes_limits(axes)
    axis_x_inset = GRID_AXIS_INSET_RATIO * (x_max - x_min)
    axis_y_inset = GRID_AXIS_INSET_RATIO * (y_max - y_min)
    grid = VGroup()
    x_ticks = np.arange(np.ceil(x_min), np.floor(x_max) + GRID_TICK_STOP_EPSILON, step)
    y_ticks = np.arange(np.ceil(y_min), np.floor(y_max) + GRID_TICK_STOP_EPSILON, step)
    for x in x_ticks:
        if abs(x) < ZERO_AXIS_EPSILON:
            continue
        line = Line(axes.c2p(float(x), y_min), axes.c2p(float(x), y_max))
        line.set_stroke(C_GRID, width=GRID_LINE_STROKE_WIDTH, opacity=GRID_LINE_OPACITY)
        grid.add(line)
    for y in y_ticks:
        if abs(y) < ZERO_AXIS_EPSILON:
            continue
        line = Line(axes.c2p(x_min, float(y)), axes.c2p(x_max, float(y)))
        line.set_stroke(C_GRID, width=GRID_LINE_STROKE_WIDTH, opacity=GRID_LINE_OPACITY)
        grid.add(line)
    if y_min <= 0 <= y_max:
        x_axis = Arrow(
            axes.c2p(x_min + axis_x_inset, 0),
            axes.c2p(x_max - axis_x_inset, 0),
            buff=0,
            color=C_MUTED,
            stroke_width=GRID_AXIS_STROKE_WIDTH,
            max_tip_length_to_length_ratio=GRID_AXIS_TIP_RATIO,
        )
        x_axis.set_opacity(GRID_AXIS_OPACITY)
        grid.add(x_axis)
    if x_min <= 0 <= x_max:
        y_axis = Arrow(
            axes.c2p(0, y_min + axis_y_inset),
            axes.c2p(0, y_max - axis_y_inset),
            buff=0,
            color=C_MUTED,
            stroke_width=GRID_AXIS_STROKE_WIDTH,
            max_tip_length_to_length_ratio=GRID_AXIS_TIP_RATIO,
        )
        y_axis.set_opacity(GRID_AXIS_OPACITY)
        grid.add(y_axis)
    return grid


def _bottom_tick(axes: Axes, x: float, label: str, *, color: str = C_TEXT) -> VGroup:
    x_min, _, y_min, y_max = _axes_limits(axes)
    _ = x_min
    span_y = y_max - y_min
    tick = Line(axes.c2p(x, y_min), axes.c2p(x, y_min + BOTTOM_TICK_HEIGHT_RATIO * span_y))
    tick.set_stroke(color, width=BOTTOM_TICK_STROKE_WIDTH)
    tex = MathTex(label, color=color)
    tex.scale_to_fit_height(TICK_LABEL_TEX_SCALE * tick.height)
    tex.next_to(tick, DOWN, buff=SMALL_BUFF)
    return VGroup(tick, tex)


def _data_label(
    axes: Axes,
    x: float,
    y: float,
    label: str,
    *,
    color: str = C_TEXT,
    typography: str = "caption",
) -> MathTex:
    return theme_math(label, color=color, typography=typography).move_to(axes.c2p(x, y))


def _quadratic_values(matrix: FloatArray, x_grid: FloatArray, y_grid: FloatArray) -> FloatArray:
    return 0.5 * (
        matrix[0, 0] * x_grid**2
        + 2.0 * matrix[0, 1] * x_grid * y_grid
        + matrix[1, 1] * y_grid**2
    )


def _quadratic_heatmap(
    axes: Axes,
    matrix: FloatArray,
    *,
    width: int = DEFAULT_QUADRATIC_HEATMAP_WIDTH,
) -> ImageMobject:
    x_min, x_max, y_min, y_max = _axes_limits(axes)
    frame = _plot_frame(axes)
    height = max(HEATMAP_MIN_HEIGHT, int(width * frame.height / frame.width))
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
        points.append(points[0] + RIGHT * POLYLINE_FALLBACK_STEP)
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
        max_tip_length_to_length_ratio=AXIS_ARROW_TIP_RATIO,
    )
    return arrow


def _eigenmode_annotation(
    axes: Axes,
    vector: FloatArray,
    eigenvalue: float,
    eigenvalues: FloatArray,
    label: str,
    *,
    color: str,
) -> VGroup:
    x_min, x_max, y_min, y_max = _axes_limits(axes)
    base_radius = 0.5 * min(x_max - x_min, y_max - y_min) * (1 - 2 * GRID_AXIS_INSET_RATIO)
    relative_length = min(
        1.0,
        float(np.mean(eigenvalues)) / max(float(eigenvalue), ZERO_AXIS_EPSILON),
    )
    direction = vector / np.linalg.norm(vector)
    end = base_radius * relative_length * direction
    arrow = _axis_arrow(axes, np.zeros(2), end, color=color)
    label_mob = theme_math(label, color=color, typography="caption")
    label_mob.next_to(arrow.get_end(), direction=arrow.get_end() - arrow.get_start(), buff=SMALL_BUFF)
    return VGroup(arrow, label_mob)


def _quadratic_level_sets(axes: Axes, matrix: FloatArray, count: int = 10) -> VGroup:
    x_min, x_max, y_min, y_max = _axes_limits(axes)
    x_values = np.linspace(x_min, x_max, CONTOUR_GRID_SAMPLES)
    y_values = np.linspace(y_min, y_max, CONTOUR_GRID_SAMPLES)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    log_values = np.log10(_quadratic_values(matrix, x_grid, y_grid) + 1.0)
    levels = np.linspace(
        float(np.quantile(log_values, CONTOUR_QUANTILE_RANGE[0])),
        float(np.quantile(log_values, CONTOUR_QUANTILE_RANGE[1])),
        count,
    )
    generator = contourpy.contour_generator(x=x_values, y=y_values, z=log_values)

    contours = VGroup()
    opacities = np.linspace(*CONTOUR_OPACITY_RANGE, len(levels))
    for opacity, level in zip(opacities, levels, strict=True):
        for segment in generator.lines(float(level)):
            if len(segment) < 2:
                continue
            step = max(1, len(segment) // CONTOUR_POLYLINE_MAX_POINTS)
            sampled = segment[::step]
            if not np.array_equal(sampled[-1], segment[-1]):
                sampled = np.vstack([sampled, segment[-1]])
            line = VMobject()
            line.set_points_as_corners([axes.c2p(float(x), float(y)) for x, y in sampled])
            line.set_stroke(C_CONTOUR, width=CONTOUR_STROKE_WIDTH, opacity=float(opacity))
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
    title_inside: bool = False,
) -> Group:
    axes = _make_axes(
        x_range,
        y_range,
        x_length=x_length,
        y_length=y_length,
        preserve_unit_aspect=True,
    )
    heatmap = _quadratic_heatmap(axes, matrix, width=QUADRATIC_PANEL_HEATMAP_WIDTH).set_z_index(
        LAYER_HEATMAP
    )
    frame = _plot_frame(axes)
    grid = _coordinate_grid(axes).set_z_index(LAYER_CONTOUR)
    levels = _quadratic_level_sets(axes, matrix)
    levels.set_z_index(LAYER_CONTOUR)
    label = Caption(title)
    if title_inside:
        label.next_to(frame.get_corner(UL), direction=DR, buff=SMALL_BUFF)
    else:
        label.next_to(axes, UP)
    label.set_z_index(LAYER_MARKERS)
    origin = Dot(
        axes.c2p(0, 0),
        color=C_TEXT,
        radius=frame.height * PANEL_MARKER_FRAME_HEIGHT_RATIO,
    )
    origin_label = label_for_dot(r"x^\star", origin, direction=DR)
    frame.set_stroke(C_FRAME, width=1.0, opacity=0.72)
    frame.set_z_index(LAYER_FRAME)
    axes.set_opacity(0)
    axes.set_z_index(LAYER_FRAME)
    markers = VGroup(origin, origin_label).set_z_index(LAYER_MARKERS)
    field = Group(heatmap, grid, levels)
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
    radius: float | None = None,
) -> VGroup:
    if radius is None:
        radius = _plot_frame(axes).height * PATH_DOT_FRAME_HEIGHT_RATIO
    line = _axes_polyline(axes, path[::step], color=color, width=PATH_LINE_STROKE_WIDTH)
    dots = VGroup(
        *(Dot(axes.c2p(float(x), float(y)), color=color, radius=radius) for x, y in path[::step])
    )
    dots[-1].scale(PATH_END_DOT_SCALE)
    return VGroup(line, dots)


def _mode_path(axes: Axes, eta: float, steps: int = MODE_TRAJECTORY_STEPS) -> VGroup:
    matrix, _ = _rotated_quadratic_matrix()
    eigenvalues, eigvecs = _quadratic_eigendecomposition(matrix)
    coefficients = eigvecs.T @ MODE_INITIAL_POINT
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
            connector.set_stroke(
                C_GREEN,
                width=MODE_TRAJECTORY_STROKE_WIDTH,
                opacity=MODE_TRAJECTORY_OPACITY,
            )
            connectors.add(connector)
        dot = Dot(screen_point, color=C_GREEN, radius=axes.height * MODE_DOT_FRAME_HEIGHT_RATIO)
        if index == 0:
            dot.scale(MODE_START_DOT_SCALE)
        dots.add(dot)
        previous_screen = screen_point

    trajectory = VGroup(connectors, dots)

    arrows = VGroup()
    for step in range(0, steps + 1, MODE_ARROW_INTERVAL):
        opacity = MODE_ARROW_OPACITY_RANGE[0] + (
            MODE_ARROW_OPACITY_RANGE[1] - MODE_ARROW_OPACITY_RANGE[0]
        ) * step / steps
        components = [
            coefficients[0] * factors[0] ** step * eigvecs[:, 0],
            coefficients[1] * factors[1] ** step * eigvecs[:, 1],
        ]
        for vector, color, width in (
            (components[0], C_BLUE, MODE_VECTOR_ARROW_SCALE),
            (components[1], C_ORANGE, MODE_VECTOR_ARROW_SCALE),
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
        bar = Rectangle(
            width=width / count * RESPONSE_BAR_WIDTH_FRACTION,
            height=max(bar_height, RESPONSE_BAR_MIN_HEIGHT),
        )
        bar.set_fill(color, opacity=RESPONSE_BAR_OPACITY)
        bar.set_stroke(color, opacity=0)
        x = -width / 2 + width * (index + 0.5) / count
        bar.move_to([x, -height / 2 + bar.height / 2, 0])
        bars.add(bar)

    frame = Rectangle(width=width, height=height)
    frame.set_fill(C_PANEL_DEEP, opacity=RESPONSE_FRAME_FILL_OPACITY)
    frame.set_stroke(
        C_FRAME,
        width=RESPONSE_FRAME_STROKE_WIDTH,
        opacity=RESPONSE_FRAME_STROKE_OPACITY,
    )
    baseline = Line(frame.get_left(), frame.get_right())
    baseline.set_stroke(
        C_MUTED,
        width=RESPONSE_BASELINE_STROKE_WIDTH,
        opacity=RESPONSE_FRAME_STROKE_OPACITY,
    )
    baseline.align_to(frame, DOWN)
    title_mob = Caption(title)
    title_mob.next_to(frame, UP)
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


def _eta_slider(tracker: VT, spec: SliderSpec, eigenvalues: FloatArray) -> VGroup:
    theme = get_active_theme()
    alpha, beta = float(eigenvalues[0]), float(eigenvalues[-1])
    return ValueSlider(
        tracker,
        spec,
        half_length=ETA_SLIDER_HALF_LENGTH,
        label_font_size=theme.typography.caption,
        value_font_size=theme.typography.caption,
        tick_values=(1.0 / beta, 2.0 / (alpha + beta), 2.0 / beta),
    )


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
    y_max = MODE_CHART_Y_STEP * np.ceil(
        max(MODE_CHART_MIN_Y_MAX, coefficient**2 * MODE_CHART_HEADROOM) / MODE_CHART_Y_STEP
    )
    frame = Rectangle(width=width, height=height)
    frame.set_stroke(
        C_FRAME,
        width=MODE_CHART_FRAME_STROKE_WIDTH,
        opacity=MODE_CHART_FRAME_OPACITY,
    )
    baseline = Line(LEFT * width / 2, RIGHT * width / 2)
    baseline.set_stroke(
        C_MUTED,
        width=MODE_CHART_BASELINE_STROKE_WIDTH,
        opacity=MODE_CHART_BASELINE_OPACITY,
    )
    baseline.move_to(frame.get_bottom())

    epsilon_y = frame.get_bottom()[1] + frame.height * MODE_CHART_EPSILON / y_max
    epsilon_line = Line(frame.get_left(), frame.get_right())
    epsilon_line.set_stroke(
        C_MUTED,
        width=MODE_CHART_EPSILON_STROKE_WIDTH,
        opacity=MODE_CHART_EPSILON_OPACITY,
    )
    epsilon_line.move_to([frame.get_center()[0], epsilon_y, frame.get_center()[2]])
    epsilon_label = theme_math(r"\epsilon", color=C_MUTED, typography="caption")
    epsilon_label.next_to(epsilon_line, RIGHT, buff=SMALL_BUFF)

    steps = np.arange(0, RESPONSE_STEP_COUNT + 1, MODE_CHART_STEP_STRIDE)
    bars = VGroup()
    for index, step in enumerate(steps):
        bar = Rectangle(
            width=frame.width * MODE_CHART_BAR_PLACEHOLDER_RATIO,
            height=frame.height * MODE_CHART_BAR_PLACEHOLDER_RATIO,
        )

        def update_bar(mob: Rectangle, *, step: int = int(step), index: int = index) -> None:
            response = abs(1.0 - ~eta * lambda_i) ** (2 * (step + 1))
            value = float(coefficient**2 * response)
            bar_width = frame.width / len(steps) * MODE_CHART_BAR_WIDTH_FRACTION
            bar_height = max(
                frame.height * min(value / y_max, 1.0),
                frame.height * MODE_CHART_BAR_MIN_HEIGHT_RATIO,
            )
            x = frame.get_left()[0] + frame.width * (index + 0.5) / len(steps)
            y = frame.get_bottom()[1] + bar_height / 2
            replacement = Rectangle(width=bar_width, height=bar_height)
            replacement.set_fill(color, opacity=MODE_CHART_BAR_OPACITY)
            replacement.set_stroke(color, opacity=0)
            replacement.move_to([x, y, frame.get_center()[2]])
            mob.become(replacement)

        bar.add_updater(update_bar)
        bars.add(bar)

    y_label = theme_math(
        rf"\|(1-\eta\lambda_{mode_index})^{{t+1}}\alpha_{mode_index}v_{mode_index}\|_2^2",
        color=color,
        typography="caption",
    )
    y_label.rotate(PI / 2)
    y_label.next_to(frame, LEFT, buff=SMALL_BUFF)
    tag = theme_math(mode_tag, color=color, typography="caption")
    tag.move_to(frame.get_corner(UL) + MODE_CHART_TAG_OFFSET)
    r_readout = VGroup(
        theme_math(r"r_i=", color=C_MUTED, typography="caption"),
        DN(lambda: 1.0 - ~eta * lambda_i, num_decimal_places=3),
    ).arrange(RIGHT, buff=SMALL_BUFF)
    r_readout.move_to(frame.get_corner(UR) + MODE_CHART_READOUT_OFFSET)
    x_label = Caption(r"even iteration $t$") if show_x_label else VGroup()
    if show_x_label:
        x_label.next_to(frame, DOWN)
    return VGroup(frame, baseline, epsilon_line, epsilon_label, bars, y_label, tag, r_readout, x_label)


def _mode_response_stack(eta: VT, *, width: float = 3.1, height: float = 1.08) -> VGroup:
    title = Caption("squared mode magnitudes")
    matrix, _ = _rotated_quadratic_matrix()
    eigenvalues, eigvecs = _quadratic_eigendecomposition(matrix)
    coefficients = eigvecs.T @ MODE_INITIAL_POINT
    flat = _mode_magnitude_chart(
        eta,
        float(eigenvalues[0]),
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
        float(eigenvalues[-1]),
        coefficient=float(coefficients[1]),
        color=C_ORANGE,
        mode_index=2,
        mode_tag=r"\lambda_2=\beta",
        width=width,
        height=height,
        show_x_label=True,
    )
    charts = VGroup(flat, steep).arrange(DOWN, buff=MED_SMALL_BUFF, aligned_edge=RIGHT)
    title.next_to(charts, UP)
    return VGroup(title, charts)


def _bar_chart_for_response(
    title: str,
    top_values: FloatArray,
    bottom_values: FloatArray,
    *,
    top_note: str,
    bottom_note: str,
    top_callout: str | None = None,
    bottom_callout: str | None = None,
) -> VGroup:
    title_parts = [Caption(part) for part in title.splitlines()]
    column_title = (
        VGroup(*title_parts).arrange(DOWN, buff=MULTILINE_TITLE_BUFF)
        if len(title_parts) > 1
        else title_parts[0]
    )
    top = _response_bars(
        values=top_values[::RESPONSE_SAMPLE_STRIDE],
        color=C_BLUE,
        width=RESPONSE_CHART_WIDTH,
        height=RESPONSE_CHART_HEIGHT,
        title=r"$\mu$ coordinate",
    )
    bottom = _response_bars(
        values=bottom_values[::RESPONSE_SAMPLE_STRIDE],
        color=C_ORANGE,
        width=RESPONSE_CHART_WIDTH,
        height=RESPONSE_CHART_HEIGHT,
        title=r"$L$ coordinate",
    )
    bottom.next_to(top, DOWN, buff=MED_SMALL_BUFF)
    top[3].align_to(top[0], LEFT)
    bottom[3].align_to(bottom[0], LEFT)
    rows = VGroup(top, bottom)
    chart = VGroup(column_title, rows).arrange(DOWN)
    notes = VGroup()
    for frame_group, note in ((top, top_note), (bottom, bottom_note)):
        note_mob = theme_math(note, color=C_TEXT, typography="caption")
        note_mob.move_to(frame_group[0].get_corner(UL) + RIGHT * 0.40 + DOWN * 0.20)
        notes.add(note_mob)
    if top_callout:
        callout = Caption(top_callout)
        callout.set_color(C_TEAL)
        callout.move_to(top[0].get_center() + RIGHT * 0.35)
        notes.add(callout)
    if bottom_callout:
        callout = Caption(bottom_callout)
        callout.set_color(C_TEAL)
        callout.move_to(bottom[0].get_center() + RIGHT * 0.35)
        notes.add(callout)
    chart.add(notes)
    return _panel_shell(chart)


class SecondOrderApproximation(Slide):
    """Taylor's local quadratic model and the Newton displacement."""

    def construct(self) -> None:
        title = _start_slide(self, "Second-order approximation")
        left, right = _split_weighted(self.region, [1.72, 1.0])

        alpha = VT(LOCAL_ALPHA_INITIAL)
        beta = VT(LOCAL_ALPHA_INITIAL)
        alpha_anchor = VT(LOCAL_CURRENT_X)
        beta_anchor = VT(LOCAL_CURRENT_X)
        view_scale = VT(1.0)
        center = LOCAL_MODEL_CENTER
        x_t = LOCAL_CURRENT_X

        def f(xs: FloatArray) -> FloatArray:
            return (
                LOCAL_QUARTIC_WEIGHT * (xs - center) ** 4
                + LOCAL_QUADRATIC_WEIGHT * (xs - LOCAL_QUADRATIC_CENTER) ** 2
                + LOCAL_VALUE_OFFSET
            )

        def grad(x: float) -> float:
            return (
                4 * LOCAL_QUARTIC_WEIGHT * (x - center) ** 3
                + 2 * LOCAL_QUADRATIC_WEIGHT * (x - LOCAL_QUADRATIC_CENTER)
            )

        def hess(x: float) -> float:
            return 12 * LOCAL_QUARTIC_WEIGHT * (x - center) ** 2 + 2 * LOCAL_QUADRATIC_WEIGHT

        def f_scalar(x: float) -> float:
            return float(f(np.array([x]))[0])

        def model_values(values: FloatArray, anchor: float, curvature: float) -> FloatArray:
            return f_scalar(anchor) + grad(anchor) * (values - anchor) + 0.5 * curvature * (values - anchor) ** 2

        f_t = float(f(np.array([x_t]))[0])
        g_t = grad(x_t)
        h_t = hess(x_t)
        x_next = x_t - g_t / h_t
        y_next = float(model_values(np.array([x_next]), x_t, h_t)[0])
        roots = np.roots(
            [
                4 * LOCAL_QUARTIC_WEIGHT,
                -12 * LOCAL_QUARTIC_WEIGHT * center,
                12 * LOCAL_QUARTIC_WEIGHT * center**2 + 2 * LOCAL_QUADRATIC_WEIGHT,
                -(4 * LOCAL_QUARTIC_WEIGHT * center**3)
                - 2 * LOCAL_QUADRATIC_WEIGHT * LOCAL_QUADRATIC_CENTER,
            ]
        )
        x_star = min(
            (root.real for root in roots if abs(root.imag) < ZERO_AXIS_EPSILON),
            key=lambda x: f(np.array([x]))[0],
        )
        f_star = float(f(np.array([x_star]))[0])
        x_min, x_max, x_step = LOCAL_X_RANGE
        xs = np.linspace(x_min, x_max, LOCAL_CURVE_SAMPLES)
        dx = xs - x_t
        fixed_lower = f_t + g_t * dx + 0.5 * LOCAL_ALPHA_INITIAL * dx**2
        fixed_upper = f_t + g_t * dx + 0.5 * LOCAL_BETA_INITIAL * dx**2
        fixed_local = f_t + g_t * dx + 0.5 * h_t * dx**2
        y_min = (
            min(float(np.min(fixed_lower)), float(np.min(fixed_local)), float(np.min(f(xs))))
            - LOCAL_Y_PADDING_BELOW
        )
        y_max = min(
            max(float(np.max(f(xs))), float(np.max(fixed_upper))) + LOCAL_Y_PADDING_ABOVE,
            LOCAL_Y_MAX_CAP,
        )

        axes = _make_axes(
            (x_min, x_max, x_step),
            (y_min, y_max, 1),
            x_length=LOCAL_AXIS_X_LENGTH,
            y_length=LOCAL_AXIS_Y_LENGTH,
        )
        axes.set_opacity(0)
        frame = _plot_frame(axes)
        frame.set_fill(C_PANEL_DEEP, opacity=LOCAL_PANEL_FILL_OPACITY)
        frame.set_stroke(
            C_FRAME,
            width=LOCAL_PANEL_STROKE_WIDTH,
            opacity=LOCAL_PANEL_STROKE_OPACITY,
        )
        bottom_axis = Line(frame.get_corner(DL), frame.get_corner(DR))
        bottom_axis.set_stroke(
            C_MUTED,
            width=LOCAL_BOTTOM_AXIS_STROKE_WIDTH,
            opacity=LOCAL_BOTTOM_AXIS_OPACITY,
        )

        base_x_center = 0.5 * (x_min + x_max)
        base_y_center = 0.5 * (y_min + y_max)
        base_x_span = x_max - x_min
        base_y_span = y_max - y_min

        def view_limits() -> tuple[float, float, float, float]:
            scale = max(float(~view_scale), ZERO_AXIS_EPSILON)
            half_x = 0.5 * base_x_span * scale
            half_y = 0.5 * base_y_span * scale
            return (
                base_x_center - half_x,
                base_x_center + half_x,
                base_y_center - half_y,
                base_y_center + half_y,
            )

        def data_point(x: float, y: float) -> FloatArray:
            view_x_min, view_x_max, view_y_min, view_y_max = view_limits()
            x_ratio = (x - view_x_min) / (view_x_max - view_x_min)
            y_ratio = (y - view_y_min) / (view_y_max - view_y_min)
            return frame.get_corner(DL) + RIGHT * (frame.width * x_ratio) + UP * (frame.height * y_ratio)

        def curve_for(
            fn: Callable[[FloatArray], FloatArray],
            *,
            color: str,
            width: float,
            opacity: float = 1.0,
        ) -> VMobject:
            view_x_min, view_x_max, view_y_min, view_y_max = view_limits()
            visible_xs = np.linspace(view_x_min, view_x_max, LOCAL_CURVE_SAMPLES)
            ys = fn(visible_xs)
            mask = np.isfinite(ys) & (ys >= view_y_min) & (ys <= view_y_max)
            points = [
                data_point(float(x), float(y))
                for x, y in zip(visible_xs[mask], ys[mask], strict=True)
            ]
            if len(points) < 2:
                clipped_y = float(np.clip(ys[0], view_y_min, view_y_max))
                points = [data_point(float(visible_xs[0]), clipped_y)]
                points.append(points[0] + RIGHT * POLYLINE_FALLBACK_STEP)
            curve = VMobject()
            curve.set_points_smoothly(points)
            curve.set_stroke(color, width=width, opacity=opacity)
            return curve

        def dashed_curve_for(
            fn: Callable[[FloatArray], FloatArray],
            *,
            color: str,
            width: float,
        ) -> DashedVMobject:
            return DashedVMobject(
                curve_for(fn, color=color, width=width),
                num_dashes=LOCAL_MODEL_DASH_COUNT,
            ).set_z_index(LAYER_TRAJECTORY)

        def track_with_factory(mob: VMobject, factory: Callable[[], VMobject]) -> None:
            mob.add_updater(lambda tracked: tracked.become(factory()))

        def true_curve_factory() -> VMobject:
            return curve_for(f, color=C_TEXT, width=LOCAL_CURVE_STROKE_WIDTH).set_z_index(
                LAYER_TRAJECTORY
            )

        def local_model_factory() -> DashedVMobject:
            return dashed_curve_for(
                lambda values: model_values(values, x_t, h_t),
                color=C_GREEN,
                width=LOCAL_MODEL_STROKE_WIDTH,
            )

        def lower_model_factory() -> VMobject:
            return curve_for(
                lambda values: model_values(values, float(~alpha_anchor), float(~alpha)),
                color=C_BLUE,
                width=LOCAL_BOUND_STROKE_WIDTH,
                opacity=LOCAL_BOUND_OPACITY,
            ).set_z_index(LAYER_TRAJECTORY)

        def upper_model_factory() -> VMobject:
            return curve_for(
                lambda values: model_values(values, float(~beta_anchor), float(~beta)),
                color=C_ORANGE,
                width=LOCAL_BOUND_STROKE_WIDTH,
                opacity=LOCAL_BOUND_OPACITY,
            ).set_z_index(LAYER_TRAJECTORY)

        true_curve = true_curve_factory()
        local_model = local_model_factory()
        lower_model = lower_model_factory()
        upper_model = upper_model_factory()
        local_marker_radius = frame.height * SECOND_ORDER_MARKER_FRAME_HEIGHT_RATIO

        def tracked_dot(
            x_fn: Callable[[], float],
            y_fn: Callable[[], float],
            *,
            color: str,
            radius: float,
        ) -> Dot:
            dot = Dot(data_point(x_fn(), y_fn()), color=color, radius=radius).set_z_index(
                LAYER_MARKERS
            )
            dot.add_updater(lambda mob: mob.move_to(data_point(x_fn(), y_fn())))
            return dot

        def bottom_tick(x_fn: Callable[[], float], label: str, *, color: str) -> VGroup:
            _, _, view_y_min, view_y_max = view_limits()
            tick_height = BOTTOM_TICK_HEIGHT_RATIO * (view_y_max - view_y_min)
            tick = Line(data_point(x_fn(), view_y_min), data_point(x_fn(), view_y_min + tick_height))
            tick.set_stroke(color, width=BOTTOM_TICK_STROKE_WIDTH)
            tex = MathTex(label, color=color)
            tex.scale_to_fit_height(TICK_LABEL_TEX_SCALE * tick.height)
            tex.next_to(tick, DOWN, buff=SMALL_BUFF)
            group = VGroup(tick, tex).set_z_index(LAYER_MARKERS)

            def update_tick(mob: VGroup) -> None:
                _, _, current_y_min, current_y_max = view_limits()
                current_tick_height = BOTTOM_TICK_HEIGHT_RATIO * (current_y_max - current_y_min)
                mob[0].put_start_and_end_on(
                    data_point(x_fn(), current_y_min),
                    data_point(x_fn(), current_y_min + current_tick_height),
                )
                mob[0].set_stroke(color, width=BOTTOM_TICK_STROKE_WIDTH)
                mob[1].next_to(mob[0], DOWN, buff=SMALL_BUFF)

            group.add_updater(update_tick)
            return group

        def label_next_to_dot(
            label: str,
            dot: Dot,
            *,
            direction: FloatArray,
            color: str,
            colors: dict[str, str] | None = None,
        ) -> MathTex:
            mob = theme_math(label, color=color, typography="caption").set_z_index(LAYER_MARKERS)
            if colors is not None:
                color_substrings(mob, colors)
            mob.next_to(dot, direction)
            mob.add_updater(lambda label_mob: label_mob.next_to(dot, direction))
            return mob

        def vertical_guide_factory(
            x_fn: Callable[[], float],
            *,
            color: str,
            opacity: float,
        ) -> DashedLine:
            _, _, current_y_min, current_y_max = view_limits()
            line = DashedLine(data_point(x_fn(), current_y_min), data_point(x_fn(), current_y_max), color=color)
            line.set_stroke(width=LOCAL_VERTICAL_GUIDE_STROKE_WIDTH, opacity=opacity)
            return line.set_z_index(LAYER_TRAJECTORY)

        x_t_dot = tracked_dot(lambda: x_t, lambda: f_t, color=C_YELLOW, radius=local_marker_radius)
        newton_dot = tracked_dot(lambda: x_next, lambda: y_next, color=C_GREEN, radius=local_marker_radius)
        star_dot = tracked_dot(
            lambda: float(x_star),
            lambda: f_star,
            color=C_TEXT,
            radius=local_marker_radius * SECOND_ORDER_STAR_DOT_SCALE,
        )
        x_t_tick = bottom_tick(lambda: x_t, r"x_t", color=C_YELLOW)
        next_tick = bottom_tick(lambda: x_next, r"x_{t+1}", color=C_GREEN)
        x_t_value = label_next_to_dot(
            r"f(x_t)",
            x_t_dot,
            direction=UR,
            color=C_TEXT,
            colors={r"x_t": C_YELLOW},
        )
        star_label = label_next_to_dot(
            r"f(x^\star)",
            star_dot,
            direction=DR,
            color=C_TEXT,
        )
        x_line = vertical_guide_factory(
            lambda: x_t,
            color=C_YELLOW,
            opacity=LOCAL_CURRENT_GUIDE_OPACITY,
        )
        next_line = vertical_guide_factory(
            lambda: x_next,
            color=C_GREEN,
            opacity=LOCAL_NEXT_GUIDE_OPACITY,
        )
        track_with_factory(
            x_line,
            lambda: vertical_guide_factory(
                lambda: x_t,
                color=C_YELLOW,
                opacity=LOCAL_CURRENT_GUIDE_OPACITY,
            ),
        )
        track_with_factory(
            next_line,
            lambda: vertical_guide_factory(
                lambda: x_next,
                color=C_GREEN,
                opacity=LOCAL_NEXT_GUIDE_OPACITY,
            ),
        )

        def bracket_geometry() -> tuple[float, float]:
            _, _, current_y_min, current_y_max = view_limits()
            y_span = current_y_max - current_y_min
            return (
                current_y_min + LOCAL_BRACKET_Y_RATIO * y_span,
                LOCAL_BRACKET_TICK_HEIGHT_RATIO * y_span,
            )

        delta_bracket = VGroup(Line(), Line(), Line())

        def update_delta_bracket(mob: VGroup) -> None:
            bracket_y, tick_height = bracket_geometry()
            segments = (
                (data_point(x_t, bracket_y), data_point(x_next, bracket_y)),
                (
                    data_point(x_t, bracket_y - LOCAL_BRACKET_HALF_TICK_SCALE * tick_height),
                    data_point(x_t, bracket_y + LOCAL_BRACKET_HALF_TICK_SCALE * tick_height),
                ),
                (
                    data_point(x_next, bracket_y - LOCAL_BRACKET_HALF_TICK_SCALE * tick_height),
                    data_point(x_next, bracket_y + LOCAL_BRACKET_HALF_TICK_SCALE * tick_height),
                ),
            )
            for line, (start, end) in zip(mob, segments, strict=True):
                line.put_start_and_end_on(start, end)
            mob.set_stroke(WHITE, width=LOCAL_BRACKET_STROKE_WIDTH)

        delta_bracket.add_updater(update_delta_bracket)
        update_delta_bracket(delta_bracket)
        delta_label = theme_math(
            r"\delta=-\nabla^2 f(x_t)^{-1}\nabla f(x_t)",
            color=WHITE,
            typography="caption",
        ).set_z_index(LAYER_MARKERS)
        color_substrings(delta_label, {r"x_t": C_YELLOW})

        def update_delta_label(mob: MathTex) -> None:
            bracket_y, _ = bracket_geometry()
            _, _, current_y_min, current_y_max = view_limits()
            mob.scale_to_fit_width(max(delta_bracket[0].width - 2 * SMALL_BUFF, SMALL_BUFF))
            mob.move_to(
                data_point(
                    0.5 * (x_t + x_next),
                    bracket_y + LOCAL_BRACKET_LABEL_Y_RATIO * (current_y_max - current_y_min),
                )
            )

        delta_label.add_updater(update_delta_label)
        update_delta_label(delta_label)

        def alpha_minimum() -> tuple[float, float]:
            value = max(~alpha, 1e-3)
            anchor = float(~alpha_anchor)
            x_alpha = anchor - grad(anchor) / value
            y_alpha = float(model_values(np.array([x_alpha]), anchor, value)[0])
            return float(x_alpha), float(y_alpha)

        def beta_minimum() -> tuple[float, float]:
            value = max(~beta, 1e-3)
            anchor = float(~beta_anchor)
            x_beta = anchor - grad(anchor) / value
            y_beta = float(model_values(np.array([x_beta]), anchor, value)[0])
            return float(x_beta), float(y_beta)

        def x_marker(x: float, y: float, *, color: str) -> VGroup:
            center_point = data_point(x, y)
            size = frame.height * X_MARKER_FRAME_HEIGHT_RATIO
            marker = VGroup(
                Line(center_point + (LEFT + DOWN) * size / 2, center_point + (RIGHT + UP) * size / 2),
                Line(center_point + (LEFT + UP) * size / 2, center_point + (RIGHT + DOWN) * size / 2),
            )
            marker.set_stroke(color, width=X_MARKER_STROKE_WIDTH, opacity=X_MARKER_OPACITY)
            return marker.set_z_index(LAYER_MARKERS)

        alpha_marker = x_marker(*alpha_minimum(), color=C_BLUE)
        beta_marker = x_marker(*beta_minimum(), color=C_ORANGE)
        track_with_factory(alpha_marker, lambda: x_marker(*alpha_minimum(), color=C_BLUE))
        track_with_factory(beta_marker, lambda: x_marker(*beta_minimum(), color=C_ORANGE))
        plot = VGroup(
            frame,
            bottom_axis,
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
            x_t_tick,
            next_tick,
            x_t_value,
            star_label,
            delta_bracket,
            delta_label,
            alpha_marker,
            beta_marker,
        )
        left.scale_and_place(plot, buff=SMALL_BUFF)

        def color_formula(mob: VMobject) -> VMobject:
            _color_text_parts(
                mob,
                {
                    r"x_t": C_YELLOW,
                    r"x_{t+1}": C_GREEN,
                    r"\alpha": C_BLUE,
                    r"\beta": C_ORANGE,
                    r"\nabla^2": C_GREEN,
                },
            )
            return mob

        def plot_legend_entry(tex: str, *, color: str, width: float, dashed: bool = False) -> VGroup:
            label = color_formula(theme_math(tex, color=C_TEXT, typography="caption"))
            sample_width = frame.width * LOCAL_LEGEND_SAMPLE_FRAME_WIDTH_RATIO
            sample = DashedLine(ORIGIN, RIGHT * sample_width, color=color) if dashed else Line(ORIGIN, RIGHT * sample_width)
            sample.set_stroke(color, width=width)
            return VGroup(sample, label).arrange(RIGHT, buff=SMALL_BUFF)

        f_legend = plot_legend_entry(r"f(x)", color=C_TEXT, width=LOCAL_CURVE_STROKE_WIDTH)
        hessian_legend = plot_legend_entry(
            r"\begin{aligned}"
            r"f(x_t)&+\left\langle\nabla f(x_t),x-x_t\right\rangle\\"
            r"&+\frac{1}{2}(x-x_t)^\top\nabla^2 f(x_t)(x-x_t)"
            r"\end{aligned}",
            color=C_GREEN,
            width=LOCAL_MODEL_STROKE_WIDTH,
            dashed=True,
        )
        alpha_legend = plot_legend_entry(
            r"\begin{aligned}"
            r"f(x_t)&+\left\langle\nabla f(x_t),x-x_t\right\rangle\\"
            r"&+\frac{\alpha}{2}\left\|x-x_t\right\|_2^2"
            r"\end{aligned}",
            color=C_BLUE,
            width=LOCAL_BOUND_STROKE_WIDTH,
        )
        beta_legend = plot_legend_entry(
            r"\begin{aligned}"
            r"f(x_t)&+\left\langle\nabla f(x_t),x-x_t\right\rangle\\"
            r"&+\frac{\beta}{2}\left\|x-x_t\right\|_2^2"
            r"\end{aligned}",
            color=C_ORANGE,
            width=LOCAL_BOUND_STROKE_WIDTH,
        )
        plot_legend = VGroup(
            f_legend,
            hessian_legend,
            alpha_legend,
            beta_legend,
        ).arrange(DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
        legend_scale = min(
            frame.width * LOCAL_LEGEND_FRAME_WIDTH_RATIO / plot_legend.width,
            frame.height * LOCAL_LEGEND_FRAME_HEIGHT_RATIO / plot_legend.height,
            1.0,
        )
        plot_legend.scale(legend_scale)
        plot_legend.move_to(
            frame.get_corner(UR)
            + LEFT * (plot_legend.width / 2 + PANEL_BUFF)
            + DOWN * (plot_legend.height / 2 + PANEL_BUFF)
        )

        def legend_background_for(entry_count: int) -> VMobject:
            visible_entries = VGroup(*plot_legend.submobjects[:entry_count])
            width_spacer = Line(ORIGIN, RIGHT * plot_legend.width).set_opacity(0)
            width_spacer.move_to(visible_entries)
            width_spacer.align_to(plot_legend, RIGHT)
            background_target = VGroup(visible_entries, width_spacer)
            return _themed_box(background_target, color=C_PANEL_DEEP)[0].set_z_index(
                LAYER_MARKERS
            )

        plot_legend_background = legend_background_for(1)
        plot_legend.set_z_index(LAYER_MARKERS + 1)
        plot.add(plot_legend_background, plot_legend)

        definition_template = TexTemplate()
        definition_template.add_to_preamble(r"\usepackage{amsthm}")
        definition_template.add_to_preamble(
            r"\theoremstyle{definition}\newtheorem*{definition}{Definition}"
        )
        definition_font_size = get_active_theme().typography.caption
        alpha_definition = TexPage(
            r"\begin{definition}[$\alpha$-strong convexity]"
            r"\["
            r"f(x)\ge f(x_t)+\left\langle\nabla f(x_t),x-x_t\right\rangle"
            r"+\frac{\alpha}{2}\left\|x-x_t\right\|_2^2"
            r"\]"
            r"\end{definition}",
            page_width=right.width,
            buff=SMALL_BUFF,
            tex_template=definition_template,
            font_size=definition_font_size,
        )
        beta_definition = TexPage(
            r"\begin{definition}[$\beta$-smoothness]"
            r"\["
            r"f(x)\le f(x_t)+\left\langle\nabla f(x_t),x-x_t\right\rangle"
            r"+\frac{\beta}{2}\left\|x-x_t\right\|_2^2"
            r"\]"
            r"\end{definition}",
            page_width=right.width,
            buff=SMALL_BUFF,
            tex_template=definition_template,
            font_size=definition_font_size,
        )
        color_substrings(
            alpha_definition,
            {r"x_t": C_YELLOW, r"\alpha": C_BLUE},
            probe_class=MathTex,
        )
        color_substrings(
            beta_definition,
            {r"x_t": C_YELLOW, r"\beta": C_ORANGE},
            probe_class=MathTex,
        )
        definition_stack = VGroup(alpha_definition, beta_definition).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=MED_LARGE_BUFF,
        )
        right_panel = _themed_box(definition_stack)
        right.scale_and_place(right_panel, buff=SMALL_BUFF)
        sidebar_background = right_panel[0]

        zoom_out_scale = 2.0

        self.play(
            Write(title),
            Write(frame),
            Write(bottom_axis),
            Write(sidebar_background),
            Write(plot_legend_background),
            Write(true_curve),
            Write(x_t_dot),
            Write(x_t_tick),
            Write(x_t_value),
            Write(f_legend),
        )
        track_with_factory(true_curve, true_curve_factory)
        self.next_slide()

        self.play(
            Write(local_model),
            Write(newton_dot),
            Write(next_tick),
            Write(star_dot),
            Write(star_label),
            Write(x_line),
            Write(next_line),
            Write(delta_bracket),
            Write(delta_label),
            plot_legend_background.animate.become(legend_background_for(2)),
            Write(hessian_legend),
        )
        track_with_factory(local_model, local_model_factory)
        self.next_slide()

        self.play(
            Write(lower_model),
            Write(alpha_marker),
            Write(alpha_definition),
            plot_legend_background.animate.become(legend_background_for(3)),
            Write(alpha_legend),
        )
        track_with_factory(lower_model, lower_model_factory)
        self.next_slide()

        self.play(Write(upper_model), Write(beta_marker))
        track_with_factory(upper_model, upper_model_factory)
        self.next_slide()

        self.play(beta @ LOCAL_BETA_INITIAL)
        self.play(
            Write(beta_definition),
            plot_legend_background.animate.become(legend_background_for(4)),
            Write(beta_legend),
        )
        self.next_slide()

        self.play(view_scale @ zoom_out_scale)
        self.next_slide()

        self.play(beta_anchor @ float(x_star))
        self.next_slide()

        self.play(beta_anchor @ x_t)
        self.next_slide()

        self.play(alpha_anchor @ float(x_star))
        self.next_slide()

        self.play(alpha_anchor @ x_t)
        self.next_slide()


class QuadraticRotation(Slide):
    """Rotate a quadratic into its Hessian eigenbasis."""

    def construct(self) -> None:
        title = _start_slide(self, "The quadratic microscope")
        matrix, eigvecs = _rotated_quadratic_matrix()
        eigenvalues, eigvecs = _quadratic_eigendecomposition(matrix)
        diagonal = np.diag(eigenvalues)
        body, strip = _split_rows(self.region, [4.3, 1.0])
        left, mid, right = _split_weighted(body, [1.0, 0.42, 1.0])

        original = _quadratic_panel(
            matrix,
            "original coordinates",
            x_length=4.4,
            y_length=4.4,
            title_inside=True,
        )
        original_axes = original[1]
        original.add(
            _eigenmode_annotation(
                original_axes,
                eigvecs[:, 0],
                float(eigenvalues[0]),
                eigenvalues,
                r"v_{\min}",
                color=C_BLUE,
            ),
            _eigenmode_annotation(
                original_axes,
                eigvecs[:, 1],
                float(eigenvalues[-1]),
                eigenvalues,
                r"v_{\max}",
                color=C_ORANGE,
            ),
        )
        eigenbasis = _quadratic_panel(
            diagonal,
            "eigenbasis coordinates",
            x_length=4.4,
            y_length=4.4,
            title_inside=True,
        )
        diagonal_values, diagonal_basis = _quadratic_eigendecomposition(diagonal)
        eigen_axes = eigenbasis[1]
        eigenbasis.add(
            _eigenmode_annotation(
                eigen_axes,
                diagonal_basis[:, 0],
                float(diagonal_values[0]),
                diagonal_values,
                r"v_{\min}",
                color=C_BLUE,
            ),
            _eigenmode_annotation(
                eigen_axes,
                diagonal_basis[:, 1],
                float(diagonal_values[-1]),
                diagonal_values,
                r"v_{\max}",
                color=C_ORANGE,
            ),
        )
        left.scale_and_place(original)
        right.scale_and_place(eigenbasis)

        map_label = VGroup(
            theme_math(r"x\mapsto", color=C_YELLOW, typography="caption"),
            theme_math(r"V^\top(x-x^\star)", color=C_YELLOW, typography="caption"),
        ).arrange(DOWN, buff=SMALL_BUFF)
        map_arrow = CurvedArrow(LEFT, RIGHT, color=C_YELLOW)
        map_group = VGroup(map_label, map_arrow).arrange(DOWN)
        mid.scale_and_place(map_group, buff=SMALL_BUFF)

        equations = VGroup(
            theme_math(r"f(x)=\frac12(x-x^\star)^\top A(x-x^\star)"),
            theme_math(r"V^\top A V=\operatorname{diag}(\alpha,\beta)"),
            theme_math(r"A v_i=\lambda_i v_i"),
            theme_math(r"x_0-x^\star=\sum_i \alpha_i v_i"),
        ).arrange(RIGHT, buff=MED_LARGE_BUFF)
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
        strip.scale_and_place(_themed_box(equations), buff=SMALL_BUFF)

        self.play(Write(title), FadeIn(original))
        self.fragment(title="Rotate into modes")
        self.play(Write(map_group), FadeIn(eigenbasis))
        self.fragment(title="Separated recurrence")
        self.play(Write(equations))


class GradientDescentModes(Slide):
    """Gradient descent as independent eigenmode multipliers."""

    def construct(self) -> None:
        title = _start_slide(self, "Gradient descent is a scalar compromise")
        matrix, _ = _rotated_quadratic_matrix()
        eigenvalues, _ = _quadratic_eigendecomposition(matrix)
        alpha, beta = float(eigenvalues[0]), float(eigenvalues[-1])
        eta = VT(1.0 / beta)
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
        marker_radius = frame.height * PANEL_MARKER_FRAME_HEIGHT_RATIO
        origin = Dot(axes.c2p(0, 0), color=C_TEXT, radius=marker_radius).set_z_index(LAYER_MARKERS)
        origin_label = label_for_dot(r"x^\star", origin, direction=DR)
        start = Dot(axes.c2p(6, 8), color=C_GREEN, radius=marker_radius).set_z_index(
            LAYER_MARKERS
        )
        start_label = label_for_dot(r"x_0", start, color=C_GREEN, direction=UR)
        markers = VGroup(origin, origin_label, start, start_label)
        plot_shell = Group(heatmap, contours, axes, frame, markers)
        top_left.scale_and_place(plot_shell, buff=SMALL_BUFF)

        slider = _eta_slider(
            eta,
            SliderSpec(r"\eta", 0.0, 2.0 / beta, 3, C_ORANGE),
            eigenvalues,
        )
        slider.scale(0.84)
        slider.move_to(frame.get_corner(UL) + RIGHT * 1.55 + DOWN * 0.32)
        trajectory = always_redraw(lambda: _mode_path(axes, ~eta).set_z_index(LAYER_TRAJECTORY))

        responses = _mode_response_stack(eta)
        top_right.scale_and_place(responses, buff=SMALL_BUFF)
        responses.update()

        rule = theme_math(
            r"x_{t+1}-x^\star=\sum_i(1-\eta\lambda_i)^{t+1}\alpha_i v_i",
        )
        modes = theme_math(r"r_i=1-\eta\lambda_i")
        energy = theme_math(
            r"f(x_t)-f_\star=\frac12\sum_i\lambda_i\alpha_i^2(1-\eta\lambda_i)^{2t}",
        )
        rho = VGroup(
            theme_math(r"\rho_{\mathrm{GD}}(\eta)=\max\{|1-\eta\alpha|,\ |1-\eta\beta|\}="),
            DN(
                lambda: max(abs(1.0 - ~eta * alpha), abs(1.0 - ~eta * beta)),
                num_decimal_places=3,
            ),
        ).arrange(RIGHT, buff=SMALL_BUFF)
        factors = VGroup(
            VGroup(
                theme_math(r"1-\eta\alpha=", color=C_BLUE, typography="caption"),
                DN(lambda: 1.0 - ~eta * alpha, num_decimal_places=3),
            ).arrange(RIGHT, buff=SMALL_BUFF),
            VGroup(
                theme_math(r"1-\eta\beta=", color=C_ORANGE, typography="caption"),
                DN(lambda: 1.0 - ~eta * beta, num_decimal_places=3),
            ).arrange(RIGHT, buff=SMALL_BUFF),
        ).arrange(RIGHT, buff=MED_SMALL_BUFF)
        equations = VGroup(
            VGroup(rule, modes).arrange(RIGHT, buff=MED_LARGE_BUFF),
            VGroup(energy, rho).arrange(RIGHT, buff=MED_LARGE_BUFF),
            factors,
        ).arrange(DOWN)
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
        bottom.scale_and_place(_themed_box(equations), buff=SMALL_BUFF)

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
        self.play(eta @ (2.0 / (alpha + beta)), run_time=3.0)
        self.fragment(title="Let the steep mode oscillate")
        self.play(eta @ (0.5 * (2.0 / (alpha + beta) + 2.0 / beta)), run_time=3.0)
        self.fragment(title="Return to the safe step")
        self.play(eta @ (1.0 / beta), run_time=2.4)


class AdaGradKnownRuler(Slide):
    """Known diagonal curvature is perfect only when axes agree."""

    def construct(self) -> None:
        title = _start_slide(self, "A diagonal ruler can rescale, not rotate")
        body, legend_region = _split_rows(self.region, [4.4, 0.82])
        left, right = _split_weighted(body, [1.0, 1.0])
        rotated_matrix, _ = _rotated_quadratic_matrix()
        eigenvalues, _ = _quadratic_eigendecomposition(rotated_matrix)
        axis_matrix = np.diag(eigenvalues)
        x0 = np.array([2.0, 4.0], dtype=np.float64)
        eta_gd = 1.0 / float(np.sum(eigenvalues))

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
            frame = panel[0]
            gd = _linear_preconditioner_path(matrix, x0, np.eye(2), 102, eta_gd)
            diagonal = np.diag(np.diag(matrix))
            known = _linear_preconditioner_path(matrix, x0, np.linalg.inv(diagonal), 65)
            adagrad = _radial_adagrad_path(x0, 71) if radial else _coordinate_adagrad_path(matrix, x0, 71)
            paths = VGroup(
                _path_with_dots(axes, gd, color=C_GREEN, step=4),
                _path_with_dots(axes, known, color=C_PURPLE, step=3),
                _path_with_dots(axes, adagrad, color=C_TEAL, step=3),
            )
            start = Dot(
                axes.c2p(float(x0[0]), float(x0[1])),
                color=C_GREEN,
                radius=frame.height * PANEL_MARKER_FRAME_HEIGHT_RATIO,
            )
            start_label = label_for_dot(r"x_0", start, color=C_GREEN, direction=UR)
            method_key = VGroup(
                VGroup(
                    Dot(color=C_GREEN),
                    theme_math(
                        r"x_{t+1}=x_t-\eta\nabla f(x_t),\ \eta=\frac{1}{\alpha+\beta}",
                        color=C_TEXT,
                        typography="caption",
                    ),
                ).arrange(RIGHT, buff=SMALL_BUFF),
                VGroup(
                    Dot(color=C_PURPLE),
                    theme_math(
                        r"x_{t+1}=x_t-\Lambda^{-1}\nabla f(x_t)",
                        color=C_TEXT,
                        typography="caption",
                    ),
                ).arrange(RIGHT, buff=SMALL_BUFF),
                VGroup(
                    Dot(color=C_TEAL),
                    theme_math(
                        r"x_{t+1}=x_t-D_t^{-1}\nabla f(x_t)",
                        color=C_TEXT,
                        typography="caption",
                    ),
                ).arrange(RIGHT, buff=SMALL_BUFF),
            ).arrange(DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
            method_key.move_to(axes.c2p(-2.10, 8.35), aligned_edge=UL)
            panel.add(paths, VGroup(start, start_label, method_key))
            region.scale_and_place(panel, buff=SMALL_BUFF)
            panels.append(panel)

        legend = VGroup(
            VGroup(Dot(color=C_GREEN), Caption(r"scalar GD")).arrange(RIGHT),
            VGroup(Dot(color=C_PURPLE), Caption(r"known diagonal inverse")).arrange(RIGHT),
            VGroup(Dot(color=C_TEAL), Caption(r"AdaGrad coordinate ruler")).arrange(RIGHT),
            theme_math(r"x_{t+1}=x_t-D_t^{-1}\nabla f(x_t)"),
        ).arrange(RIGHT, buff=MED_LARGE_BUFF)
        _color_text_parts(legend, {r"D_t": C_TEAL})
        legend_region.scale_and_place(_themed_box(legend), buff=SMALL_BUFF)

        self.play(Write(title), *(FadeIn(panel[:6], panel[7]) for panel in panels))
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
        axis_matrix = _quadratic_axis_matrix(rotated_matrix)

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
            before_region.scale_and_place(before, buff=SMALL_BUFF)
            after_region.scale_and_place(after, buff=SMALL_BUFF)
            arrow = VGroup(
                theme_math(r"x\mapsto", color=C_PURPLE, typography="caption"),
                theme_math(r"D_A^{-1/2}x", color=C_PURPLE, typography="caption"),
                Arrow(LEFT, RIGHT, color=C_PURPLE, buff=0),
            ).arrange(DOWN, buff=SMALL_BUFF)
            arrow_region.scale_and_place(arrow, buff=SMALL_BUFF)
            row_groups.append((before, arrow, after))

        note = theme_math(
            r"D_A=\operatorname{diag}(A_{11},A_{22})",
            r"\qquad",
            r"D_A^{-1/2}AD_A^{-1/2}",
        )
        color_substrings(note, {r"D_A": C_PURPLE})
        note_region.scale_and_place(note, buff=SMALL_BUFF)

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
        matrix, _ = _rotated_quadratic_matrix()
        eigenvalues, _ = _quadratic_eigendecomposition(matrix)
        alpha, beta = float(eigenvalues[0]), float(eigenvalues[-1])
        specs = [
            ResponseSpec("GD: safe global step", "GD", 1.0 / beta),
            ResponseSpec("GD: balanced\ninverse-curvature step", "GD", 2.0 / (alpha + beta)),
            ResponseSpec("AdaGrad: activity counter", "AdaGrad", ADAGRAD_ACTIVITY_STEP_FRACTION),
        ]

        charts = []
        adagrad_response = _adagrad_coordinate_response(ADAGRAD_ACTIVITY_STEP_FRACTION)
        for region, spec in zip(columns, specs, strict=True):
            if spec.method == "GD":
                top = np.abs(_gd_response(alpha, spec.eta))
                bottom = np.abs(_gd_response(beta, spec.eta))
                top_note = (
                    rf"\begin{{aligned}}\eta&={spec.eta:.3f}\\"
                    rf"|g_0|&={alpha:.2f}\\"
                    rf"|1-\eta\lambda_i|&={abs(1.0 - spec.eta * alpha):.3f}\end{{aligned}}"
                )
                bottom_note = (
                    rf"\begin{{aligned}}\eta&={spec.eta:.3f}\\"
                    rf"|g_0|&={beta:.2f}\\"
                    rf"|1-\eta\lambda_i|&={abs(1.0 - spec.eta * beta):.3f}\end{{aligned}}"
                )
                top_callout = None
                bottom_callout = None
            else:
                top = adagrad_response
                bottom = adagrad_response
                top_note = (
                    rf"\begin{{aligned}}\eta_A&={spec.eta:.2f}\\"
                    rf"\text{{first move}}&={ADAGRAD_ACTIVITY_STEP_PERCENT}\%\text{{ of }}|x_0|\\"
                    rf"|g_0|&={alpha:.2f}\end{{aligned}}"
                )
                bottom_note = (
                    rf"\begin{{aligned}}\eta_A&={spec.eta:.2f}\\"
                    rf"\text{{first move}}&={ADAGRAD_ACTIVITY_STEP_PERCENT}\%\text{{ of }}|x_0|\\"
                    rf"|g_0|&={beta:.2f}\end{{aligned}}"
                )
                top_callout = "same trace"
                bottom_callout = "curvature scale cancels"
            chart = _bar_chart_for_response(
                spec.title,
                top,
                bottom,
                top_note=top_note,
                bottom_note=bottom_note,
                top_callout=top_callout,
                bottom_callout=bottom_callout,
            )
            region.scale_and_place(chart, buff=SMALL_BUFF)
            charts.append(chart)

        equations = _formula_stack(
            Caption("axis-aligned quadratic"),
            theme_math(r"f(x)=\frac12\sum_i\lambda_i x_i^2"),
            Caption("AdaGrad accumulator"),
            theme_math(r"G_{t,i}=\sum_{\tau=1}^{t}g_{\tau,i}^2"),
            theme_math(r"x_{t+1,i}=x_{t,i}-\eta\frac{g_{t,i}}{\sqrt{G_{t,i}}}"),
            Caption(r"because $g_{t,i}=\lambda_i x_{t,i}$"),
            theme_math(
                r"\frac{\lambda_i x_{t,i}}{\sqrt{\sum_\tau\lambda_i^2x_{\tau,i}^2}}"
                r"=\frac{x_{t,i}}{\sqrt{\sum_\tau x_{\tau,i}^2}}",
            ),
        )
        _color_text_parts(equations, {r"\lambda_i": C_ORANGE, r"G_{t,i}": C_TEAL, r"\eta": C_YELLOW})
        equation_region.scale_and_place(_themed_box(equations))

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
        left.scale_and_place(weighted, buff=SMALL_BUFF)
        right.scale_and_place(euclidean, buff=SMALL_BUFF)

        map_arrow = VGroup(
            theme_math(r"x\mapsto", color=C_PURPLE, typography="caption"),
            theme_math(r"D_t^{-1/2}x", color=C_PURPLE, typography="caption"),
            Arrow(LEFT, RIGHT, color=C_PURPLE, buff=0),
        ).arrange(DOWN, buff=SMALL_BUFF)
        arrow_region.scale_and_place(map_arrow, buff=SMALL_BUFF)

        ledger_lines = VGroup(
            theme_math(r"2\eta\langle g_t,x_t-x^\star\rangle"),
            theme_math(
                r"= \|x_t-x^\star\|_{D_t}^2-\|x_{t+1}-x^\star\|_{D_t}^2",
            ),
            theme_math(r"+ \|x_t-x_{t+1}\|_{D_t}^2"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
        setup = theme_math(r"x_{t+1}=x_t-\eta D_t^{-1}g_t")
        for line in ledger_lines:
            line.set_color_by_tex(r"\eta", C_YELLOW)
            line.set_color_by_tex(r"D_t", C_PURPLE)
        setup.set_color_by_tex(r"\eta", C_YELLOW)
        setup.set_color_by_tex(r"D_t", C_PURPLE)
        proof = _formula_stack(
            Caption("weighted parallelogram bookkeeping"),
            VGroup(setup, ledger_lines).arrange(RIGHT, buff=LEDGER_PROOF_COLUMN_BUFF),
            Caption("progress = distance drop + movement"),
        )
        proof_region.scale_and_place(_themed_box(proof), buff=SMALL_BUFF)

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
        grid = _coordinate_grid(axes).set_z_index(LAYER_CONTOUR)
        levels = _quadratic_level_sets(axes, matrix, count=7)
        levels.set_z_index(LAYER_CONTOUR)
        axes.set_opacity(0)
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

        def triangle_center() -> FloatArray:
            return (p_star + p_xt + p_xt1) / 3.0

        def readable_angle(pa: FloatArray, pb: FloatArray) -> float:
            angle = float(np.arctan2(pb[1] - pa[1], pb[0] - pa[0]))
            if angle > np.pi / 2:
                angle -= np.pi
            if angle < -np.pi / 2:
                angle += np.pi
            return angle

        def side_normal(pa: FloatArray, pb: FloatArray, center: FloatArray) -> FloatArray:
            segment = pb - pa
            normal = np.array([-segment[1], segment[0]], dtype=np.float64)
            norm = np.linalg.norm(normal)
            if norm == 0:
                return np.array([0.0, 1.0], dtype=np.float64)
            normal = normal / norm
            midpoint = 0.5 * (pa + pb)
            if np.dot(center - midpoint, normal) < 0:
                normal = -normal
            return normal

        def line_label(
            pa: FloatArray,
            pb: FloatArray,
            text: str,
            *,
            side: str,
            color: str,
            pos: float = 0.5,
            distance: float = LEDGER_LABEL_DISTANCE,
            typography: str = "caption",
        ) -> MathTex:
            center = triangle_center()
            normal = side_normal(pa, pb, center)
            if side == "outer":
                normal = -normal
            location = pa + pos * (pb - pa) + distance * normal
            label = theme_math(text, color=color, typography=typography)
            label.move_to(axes.c2p(float(location[0]), float(location[1])))
            label.rotate(readable_angle(pa, pb))
            return label

        def point_label(key: str, typography: str) -> MathTex:
            center = triangle_center()
            point = points[key]
            direction = point - center
            norm = np.linalg.norm(direction)
            direction = (
                np.array([0.0, -1.0], dtype=np.float64) if norm == 0 else direction / norm
            )
            location = point + LEDGER_POINT_LABEL_DISTANCE * direction
            return theme_math(labels[key], color=C_TEXT, typography=typography).move_to(
                axes.c2p(float(location[0]), float(location[1]))
            )

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
        frame = Rectangle(width=axes.width, height=axes.height)
        frame.move_to(axes)
        vertex_radius = frame.height * PANEL_MARKER_FRAME_HEIGHT_RATIO
        dots = VGroup(
            *(Dot(axes.c2p(float(point[0]), float(point[1])), color=C_TEXT, radius=vertex_radius) for point in points.values())
        )
        label_typography = "caption"
        point_labels = VGroup(
            point_label("star", label_typography),
            point_label("xt", label_typography),
            point_label("xt1", label_typography),
        )
        side_labels = VGroup(
            line_label(
                p_xt,
                p_star,
                rf"\|x_t-x^\star\|_{{{norm_suffix}}}^2",
                side="inner",
                color=C_MUTED,
            ),
            line_label(
                p_xt1,
                p_star,
                rf"\|x_{{t+1}}-x^\star\|_{{{norm_suffix}}}^2",
                side="inner",
                color=C_MUTED,
                pos=LEDGER_NEW_DISTANCE_LABEL_POSITION,
            ),
            line_label(
                p_xt,
                p_xt1,
                rf"\|x_{{t+1}}-x_t\|_{{{norm_suffix}}}^2",
                side="inner",
                color=C_RED,
                pos=LEDGER_STEP_LABEL_POSITION,
            ),
        )
        extras = VGroup()
        if norm_suffix == "D_t":
            extras.add(
                line_label(
                    p_xt,
                    p_star,
                    r"A",
                    side="outer",
                    color=C_TEXT,
                    distance=LEDGER_LABEL_OUTER_DISTANCE,
                ),
                line_label(
                    p_xt,
                    p_xt1,
                    r"B",
                    side="outer",
                    color=C_TEXT,
                    pos=LEDGER_B_LABEL_POSITION,
                    distance=LEDGER_LABEL_OUTER_DISTANCE,
                ),
                line_label(
                    p_xt1,
                    p_star,
                    r"A-B",
                    side="outer",
                    color=C_TEXT,
                    distance=LEDGER_LABEL_OUTER_DISTANCE,
                ),
            )
            center = triangle_center()
            outer = -side_normal(p_xt, p_star, center)
            brace_start = p_xt + LEDGER_P_PROJECTION_START * (p_star - p_xt)
            brace_end = p_xt + LEDGER_P_PROJECTION_END * (p_star - p_xt)
            brace = BraceBetweenPoints(
                axes.c2p(float(brace_start[0]), float(brace_start[1])),
                axes.c2p(float(brace_end[0]), float(brace_end[1])),
                direction=np.array([outer[0], outer[1], 0.0]),
                color=C_PURPLE,
            )
            p_point = p_xt + LEDGER_P_POINT_POSITION * (p_star - p_xt)
            open_point = Circle(radius=vertex_radius, color=C_PURPLE).move_to(
                axes.c2p(float(p_point[0]), float(p_point[1]))
            )
            open_point.set_fill(C_PANEL_DEEP, opacity=1.0)
            p_label = theme_math(r"P", color=C_PURPLE, typography="caption").next_to(
                brace,
                LEFT,
                buff=SMALL_BUFF,
            )
            extras.add(brace, p_label, open_point)
        else:
            extras.add(
                line_label(
                    p_xt,
                    p_xt1,
                    r"\eta\nabla f(x_t)",
                    side="outer",
                    color=C_RED,
                    pos=LEDGER_EUCLIDEAN_STEP_LABEL_POSITION,
                    distance=LEDGER_EUCLIDEAN_STEP_LABEL_DISTANCE,
                )
            )
        caption = Caption(title).next_to(axes, UP)
        frame.set_stroke(
            C_FRAME,
            width=LEDGER_FRAME_STROKE_WIDTH,
            opacity=LEDGER_FRAME_STROKE_OPACITY,
        )
        field = Group(heatmap, grid, levels)
        distances = VGroup(old_distance, new_distance, step, dots, point_labels, side_labels, extras)
        return Group(frame, axes, field, distances, caption, _accent_rule(frame, C_TEAL))
