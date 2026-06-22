"""Shared Manim figure helpers for the AdaGrad lecture slides."""

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
    StealthTip,
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
    Integer,
    Line,
    MathTex,
    PI,
    Rectangle,
    TexTemplate,
    VGroup,
    VMobject,
    Write,
    rush_into,
)
from simplex import Caption, DN, Slide, TexPage, VT, color_substrings, get_active_theme

from slides.helpers.controls import SliderSpec, ValueSlider
from slides.helpers.optimization_paths import OptimizationPath, OptimizationPathStyle
from slides.helpers.plotting import (
    axes_limits as _axes_limits,
    axes_point as _axes_point,
    blue_alpha_heatmap as _blue_alpha_heatmap,
    inside_axes as _inside_axes,
    make_axes as _make_axes,
    normalize_heat as _normalize_heat,
    plot_frame as _plot_frame,
)
from slides.helpers.style import (
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
MODE_CHART_RESPONSE_Y_MAX = 1.0
MODE_CHART_FRAME_STROKE_WIDTH = 0.9
MODE_CHART_FRAME_OPACITY = 0.75
MODE_CHART_BASELINE_STROKE_WIDTH = 0.8
MODE_CHART_BASELINE_OPACITY = 0.76
MODE_CHART_EPSILON = 0.1
MODE_CHART_EPSILON_STROKE_WIDTH = 0.65
MODE_CHART_EPSILON_OPACITY = 0.52
MODE_CHART_EPSILON_CROSSING_STROKE_WIDTH = 0.82
MODE_CHART_EPSILON_CROSSING_OPACITY = 0.82
MODE_CHART_BAR_WIDTH_FRACTION = 0.62
MODE_CHART_BAR_MIN_HEIGHT_RATIO = 1 / 200
MODE_CHART_BAR_PLACEHOLDER_RATIO = 1 / 100
MODE_CHART_BAR_OPACITY = 0.88
MODE_CHART_TAG_OFFSET = RIGHT * 0.60 + DOWN * 0.15
MODE_INITIAL_ETA_SUM_FACTOR = 0.54
MODE_OVERSHOOT_ETA_NUMERATOR = 2.01
MODE_FACTOR_INTERIOR_DOT_COUNT = 3
MODE_FACTOR_DOT_INITIAL_STEP = 1
MODE_FACTOR_DOT_FINAL_STEP = 80
MODE_FACTOR_DOT_RANDOM_SEED = 24
MODE_FACTOR_AXIS_HEIGHT_RATIO = 1 / 5
MODE_FACTOR_DOT_RADIUS_RATIO = 1 / 26
MODE_FACTOR_ZERO_TICK_DOT_HEIGHT_RATIO = 1.5
MODE_FACTOR_DOT_COLORS = (C_BLUE, C_GREEN, C_PURPLE, C_YELLOW, C_ORANGE)
C_ETA = C_GREEN
MODE_TRAJECTORY_STYLE = OptimizationPathStyle(color=C_ETA)
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
            tip_shape=StealthTip,
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
            tip_shape=StealthTip,
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
        tip_shape=StealthTip,
    )
    return arrow


def _perpendicular_label_direction(arrow: Arrow) -> FloatArray:
    direction = arrow.get_end() - arrow.get_start()
    perpendicular = np.array([-direction[1], direction[0], 0.0], dtype=np.float64)
    if np.linalg.norm(perpendicular) < ZERO_AXIS_EPSILON:
        return UP
    if np.dot(perpendicular, UP) < 0:
        perpendicular = -perpendicular
    return perpendicular


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
    label_mob.next_to(
        arrow.get_end(),
        direction=_perpendicular_label_direction(arrow),
        buff=SMALL_BUFF,
    )
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
    return Group(frame, axes, field, markers, label)


def _scale_and_place_matching_frame_heights(regions: Sequence[object], panels: Sequence[Group]) -> None:
    for region, panel in zip(regions, panels, strict=True):
        region.scale_and_place(panel)
    target_height = min(panel[0].height for panel in panels)
    for region, panel in zip(regions, panels, strict=True):
        panel.scale(target_height / panel[0].height)
        region.place(panel)


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


def _mode_system() -> tuple[FloatArray, FloatArray, FloatArray]:
    matrix, _ = _rotated_quadratic_matrix()
    eigenvalues, eigvecs = _quadratic_eigendecomposition(matrix)
    coefficients = eigvecs.T @ MODE_INITIAL_POINT
    return eigenvalues, eigvecs, coefficients


def _mode_point(
    *,
    eigenvalues: FloatArray,
    eigvecs: FloatArray,
    coefficients: FloatArray,
    eta: float,
    step: int,
) -> FloatArray:
    factors = 1.0 - eta * eigenvalues
    return eigvecs @ (coefficients * factors**step)


def _mode_component(
    *,
    eigenvalues: FloatArray,
    eigvecs: FloatArray,
    coefficients: FloatArray,
    eta: float,
    step: int,
    mode_index: int,
) -> FloatArray:
    factor = 1.0 - eta * eigenvalues[mode_index]
    return coefficients[mode_index] * factor**step * eigvecs[:, mode_index]


def _mode_path(axes: Axes, eta: float, steps: int = MODE_TRAJECTORY_STEPS) -> VGroup:
    eigenvalues, eigvecs, coefficients = _mode_system()
    factors = 1.0 - eta * eigenvalues
    path = np.array([eigvecs @ (coefficients * factors**step) for step in range(steps + 1)])
    trajectory = OptimizationPath(axes, path, style=MODE_TRAJECTORY_STYLE)

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


class ModeTrajectory(VGroup):
    """A GD eigenmode trajectory whose pieces update in place."""

    def __init__(self, axes: Axes, eta: VT, steps: int = MODE_TRAJECTORY_STEPS) -> None:
        super().__init__()
        self.axes = axes
        self.eta = eta
        self.steps = steps
        self.eigenvalues, self.eigvecs, self.coefficients = _mode_system()
        self.origin = axes.c2p(0, 0)
        self.frame = _plot_frame(axes)
        self.style = MODE_TRAJECTORY_STYLE

        dots = self._make_dots()
        connectors = self._make_connectors(dots)
        head_ring = self._make_head_ring(dots)
        arrows = self._make_arrows()
        self.add(dots, connectors, head_ring, arrows)

    def _point(self, step: int) -> FloatArray:
        return _mode_point(
            eigenvalues=self.eigenvalues,
            eigvecs=self.eigvecs,
            coefficients=self.coefficients,
            eta=~self.eta,
            step=step,
        )

    def _component(self, step: int, mode_index: int) -> FloatArray:
        return _mode_component(
            eigenvalues=self.eigenvalues,
            eigvecs=self.eigvecs,
            coefficients=self.coefficients,
            eta=~self.eta,
            step=step,
            mode_index=mode_index,
        )

    def _make_dots(self) -> VGroup:
        dots = VGroup()
        radius = self.frame.height * self.style.step_dot_frame_height_ratio
        for step in range(self.steps + 1):
            dot = Dot(radius=radius, color=self.style.color)
            if step == 0:
                dot.set_color(self.style.start_color)
                dot.scale(self.style.start_dot_scale)
            elif step == self.steps:
                dot.scale(self.style.head_dot_scale)
            dot.mode_visible = True
            self._update_dot(dot, step)
            dot.add_updater(lambda mob, step=step: self._update_dot(mob, step))
            dots.add(dot)
        return dots

    def _update_dot(self, dot: Dot, step: int) -> None:
        point = self._point(step)
        visible = bool(np.all(np.isfinite(point)) and _inside_axes(self.axes, point))
        if visible:
            dot.move_to(_axes_point(self.axes, point))
            if not dot.mode_visible:
                dot.set_opacity(1)
                dot.mode_visible = True
            return

        if dot.mode_visible:
            dot.set_opacity(0)
            dot.mode_visible = False

    def _make_connectors(self, dots: VGroup) -> VGroup:
        connectors = VGroup()
        for start, end in pairwise(dots):
            connector = Line(start.get_center(), end.get_center())
            connector.set_stroke(
                self.style.color,
                width=self.style.stroke_width,
                opacity=self.style.stroke_opacity,
            )
            connector.mode_visible = True
            self._update_connector(connector, start, end)
            connector.add_updater(
                lambda mob, start=start, end=end: self._update_connector(mob, start, end)
            )
            connectors.add(connector)
        return connectors

    def _update_connector(self, connector: Line, start: Dot, end: Dot) -> None:
        visible = start.mode_visible and end.mode_visible
        if visible:
            connector.put_start_and_end_on(start.get_center(), end.get_center())
            if not connector.mode_visible:
                connector.set_opacity(self.style.stroke_opacity)
                connector.mode_visible = True
            return

        if connector.mode_visible:
            connector.set_opacity(0)
            connector.mode_visible = False

    def _make_head_ring(self, dots: VGroup) -> Circle:
        radius = self.frame.height * self.style.head_dot_frame_height_ratio
        head_ring = Circle(radius=radius * self.style.head_ring_dot_scale, color=self.style.color)
        head_ring.set_stroke(self.style.color, width=self.style.head_ring_stroke_width)
        head_ring.mode_visible = True
        self._update_head_ring(head_ring, dots[-1])
        head_ring.add_updater(lambda mob: self._update_head_ring(mob, dots[-1]))
        return head_ring

    def _update_head_ring(self, ring: Circle, head: Dot) -> None:
        if head.mode_visible:
            ring.move_to(head)
            if not ring.mode_visible:
                ring.set_opacity(1)
                ring.mode_visible = True
            return

        if ring.mode_visible:
            ring.set_opacity(0)
            ring.mode_visible = False

    def _make_arrows(self) -> VGroup:
        arrows = VGroup()
        for step in range(0, self.steps + 1, MODE_ARROW_INTERVAL):
            opacity = MODE_ARROW_OPACITY_RANGE[0] + (
                MODE_ARROW_OPACITY_RANGE[1] - MODE_ARROW_OPACITY_RANGE[0]
            ) * step / self.steps
            for mode_index, color in ((0, C_BLUE), (1, C_ORANGE)):
                arrow = _axis_arrow(
                    self.axes,
                    np.zeros(2),
                    self._initial_arrow_vector(step, mode_index),
                    color=color,
                    width=MODE_VECTOR_ARROW_SCALE,
                )
                arrow.mode_visible = True
                arrow.mode_opacity = opacity
                arrow.set_opacity(opacity)
                self._update_arrow(arrow, step, mode_index)
                arrow.add_updater(
                    lambda mob, step=step, mode_index=mode_index: self._update_arrow(
                        mob,
                        step,
                        mode_index,
                    )
                )
                arrows.add(arrow)
        return arrows

    def _initial_arrow_vector(self, step: int, mode_index: int) -> FloatArray:
        vector = self._component(step, mode_index)
        if self._arrow_visible(vector):
            return vector
        return np.array([POLYLINE_FALLBACK_STEP, 0.0], dtype=np.float64)

    def _arrow_visible(self, vector: FloatArray) -> bool:
        return bool(
            np.all(np.isfinite(vector))
            and np.linalg.norm(vector) > ZERO_AXIS_EPSILON
            and _inside_axes(self.axes, vector)
        )

    def _update_arrow(self, arrow: Arrow, step: int, mode_index: int) -> None:
        vector = self._component(step, mode_index)
        visible = self._arrow_visible(vector)
        if visible:
            arrow.put_start_and_end_on(self.origin, _axes_point(self.axes, vector))
            if not arrow.mode_visible:
                arrow.set_opacity(arrow.mode_opacity)
                arrow.mode_visible = True
            return

        if arrow.mode_visible:
            arrow.set_opacity(0)
            arrow.mode_visible = False


def _mode_trajectory(axes: Axes, eta: VT, steps: int = MODE_TRAJECTORY_STEPS) -> ModeTrajectory:
    return ModeTrajectory(axes, eta, steps)


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
        tick_values=(
            1.0 / beta,
            2.0 / (alpha + beta),
            2.0 / beta,
            spec.maximum,
        ),
        show_endpoint_labels=True,
        endpoint_label_texts=(f"{spec.minimum:g}", f"{spec.maximum:.1f}"),
    )


def _mode_response_values(
    *,
    lambda_i: float,
    eta: float,
    response_steps: FloatArray,
) -> FloatArray:
    factor = 1.0 - eta * lambda_i
    return factor**response_steps


def _mode_response_value(
    *,
    lambda_i: float,
    eta: float,
    response_step: float,
) -> float:
    factor = 1.0 - eta * lambda_i
    return float(factor**response_step)


def _mode_epsilon_crossing_step(
    *,
    lambda_i: float,
    eta: float,
    max_step: float = RESPONSE_STEP_COUNT,
) -> float | None:
    factor = abs(1.0 - eta * lambda_i)
    if factor >= 1.0:
        return None
    if factor <= ZERO_AXIS_EPSILON:
        return 0.0

    step = np.log(MODE_CHART_EPSILON) / np.log(factor)
    if not np.isfinite(step) or step > max_step:
        return None
    return max(float(step), 0.0)


def _mode_epsilon_linear_rate_func(
    *, lambda_i: float, eta_start: float, eta_end: float
) -> Callable[[float], float]:
    start_step = _mode_epsilon_crossing_step(lambda_i=lambda_i, eta=eta_start)
    end_step = _mode_epsilon_crossing_step(lambda_i=lambda_i, eta=eta_end)
    eta_delta = eta_end - eta_start
    start_factor = 1.0 - eta_start * lambda_i
    end_factor = 1.0 - eta_end * lambda_i

    if (
        start_step is None
        or end_step is None
        or abs(eta_delta) <= ZERO_AXIS_EPSILON
        or start_factor * end_factor < 0
    ):
        return lambda alpha: float(alpha)

    branch_factor = start_factor if abs(start_factor) > ZERO_AXIS_EPSILON else end_factor
    branch_sign = 1.0 if branch_factor >= 0.0 else -1.0

    def rate_func(alpha: float) -> float:
        crossing_step = (1.0 - alpha) * start_step + alpha * end_step
        if crossing_step <= ZERO_AXIS_EPSILON:
            eta = 1.0 / lambda_i
        else:
            factor = MODE_CHART_EPSILON ** (1.0 / crossing_step)
            eta = (1.0 - branch_sign * factor) / lambda_i
        return float(np.clip((eta - eta_start) / eta_delta, 0.0, 1.0))

    return rate_func


def _mode_chart_sample_x(frame: Rectangle, sample_index: int, sample_count: int) -> float:
    return float(frame.get_left()[0] + frame.width * (sample_index + 0.5) / sample_count)


def _mode_chart_step_x(frame: Rectangle, step: float, response_steps: FloatArray) -> float:
    sample_index = step / MODE_CHART_STEP_STRIDE
    return _mode_chart_sample_x(frame, sample_index, len(response_steps))


def _mode_bar_height(frame: Rectangle, value: float, y_max: float) -> float:
    return max(
        frame.height * min(value / y_max, 1.0),
        frame.height * MODE_CHART_BAR_MIN_HEIGHT_RATIO,
    )


def _mode_multiplier_chart(
    eta: VT,
    lambda_i: float,
    *,
    color: str,
    mode_label: str,
    mode_tag: str,
    width: float,
    height: float,
    show_x_label: bool,
) -> VGroup:
    y_max = MODE_CHART_RESPONSE_Y_MAX
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

    response_steps = np.arange(0, RESPONSE_STEP_COUNT + 1, MODE_CHART_STEP_STRIDE)
    bars = VGroup()
    for index, response_step in enumerate(response_steps):
        bar_width = frame.width / len(response_steps) * MODE_CHART_BAR_WIDTH_FRACTION
        x = _mode_chart_sample_x(frame, index, len(response_steps))
        initial_value = _mode_response_value(
            lambda_i=lambda_i,
            eta=~eta,
            response_step=float(response_step),
        )
        initial_height = _mode_bar_height(frame, initial_value, y_max)
        bar = Rectangle(
            width=bar_width,
            height=initial_height,
        )
        bar.set_fill(color, opacity=MODE_CHART_BAR_OPACITY)
        bar.set_stroke(color, opacity=0)
        bar.move_to([x, frame.get_bottom()[1] + initial_height / 2, frame.get_center()[2]])

        def update_bar(
            mob: Rectangle,
            *,
            response_step: int = int(response_step),
            index: int = index,
        ) -> None:
            value = _mode_response_value(
                lambda_i=lambda_i,
                eta=~eta,
                response_step=response_step,
            )
            bar_height = _mode_bar_height(frame, value, y_max)
            mob.stretch_to_fit_height(bar_height, about_edge=DOWN)

        bar.add_updater(update_bar)
        bars.add(bar)

    epsilon_crossing = Line(frame.get_bottom(), frame.get_top())
    epsilon_crossing.set_stroke(
        C_MUTED,
        width=MODE_CHART_EPSILON_CROSSING_STROKE_WIDTH,
        opacity=MODE_CHART_EPSILON_CROSSING_OPACITY,
    )
    epsilon_crossing.mode_crossing_visible = True

    def update_epsilon_crossing(mob: Line) -> None:
        crossing_step = _mode_epsilon_crossing_step(
            lambda_i=lambda_i,
            eta=~eta,
            max_step=float(response_steps[-1]),
        )
        if crossing_step is None:
            if mob.mode_crossing_visible:
                mob.set_opacity(0)
                mob.mode_crossing_visible = False
            return

        x = _mode_chart_step_x(frame, crossing_step, response_steps)
        if not mob.mode_crossing_visible:
            mob.set_opacity(MODE_CHART_EPSILON_CROSSING_OPACITY)
            mob.mode_crossing_visible = True
        mob.set_x(x)

    update_epsilon_crossing(epsilon_crossing)
    epsilon_crossing.add_updater(update_epsilon_crossing)

    y_label = theme_math(
        r"(1-",
        r"\eta",
        mode_label,
        r")^t",
        color=C_TEXT,
        typography="caption",
    )
    y_label[1].set_color(C_ETA)
    y_label[2].set_color(color)
    y_label.rotate(PI / 2)
    y_label.scale_to_fit_height(frame.height / 2)
    y_label.next_to(frame, LEFT, buff=SMALL_BUFF)
    tag = theme_math(mode_tag, color=color, typography="caption")
    tag.move_to(frame.get_corner(UL) + MODE_CHART_TAG_OFFSET)
    x_label = Caption(r"Iteration $t$") if show_x_label else VGroup()
    if show_x_label:
        x_label.next_to(frame, DOWN, buff=SMALL_BUFF)
    return VGroup(
        frame,
        baseline,
        epsilon_line,
        epsilon_crossing,
        epsilon_label,
        bars,
        y_label,
        tag,
        x_label,
    )


def _mode_response_stack(eta: VT, *, width: float = 3.1, height: float = 1.08) -> VGroup:
    eigenvalues, _, _ = _mode_system()
    flat = _mode_multiplier_chart(
        eta,
        float(eigenvalues[0]),
        color=C_BLUE,
        mode_label=r"\lambda_{\min}",
        mode_tag=r"\lambda_{\min}=\alpha",
        width=width,
        height=height,
        show_x_label=False,
    )
    steep = _mode_multiplier_chart(
        eta,
        float(eigenvalues[-1]),
        color=C_ORANGE,
        mode_label=r"\lambda_{\max}",
        mode_tag=r"\lambda_{\max}=\beta",
        width=width,
        height=height,
        show_x_label=True,
    )
    return VGroup(flat, steep).arrange(DOWN, buff=SMALL_BUFF, aligned_edge=RIGHT)


def _mode_factor_lambdas(lambda_min: float, lambda_max: float) -> FloatArray:
    rng = np.random.default_rng(MODE_FACTOR_DOT_RANDOM_SEED)
    interior = np.sort(
        rng.uniform(lambda_min, lambda_max, MODE_FACTOR_INTERIOR_DOT_COUNT)
    )
    return np.concatenate(([lambda_min], interior, [lambda_max]))


def _mode_factor_axis_bound(
    *,
    lambda_min: float,
    lambda_max: float,
    eta_values: Sequence[float],
) -> float:
    lambdas = np.array([lambda_min, lambda_max], dtype=np.float64)
    factors = 1.0 - np.outer(np.asarray(eta_values, dtype=np.float64), lambdas)
    powers = factors**MODE_FACTOR_DOT_INITIAL_STEP
    return float(np.max(np.abs(powers)))


def _mode_factor_power(iteration: VT) -> float:
    return float(
        np.clip(
            ~iteration,
            MODE_FACTOR_DOT_INITIAL_STEP,
            MODE_FACTOR_DOT_FINAL_STEP,
        )
    )


def _mode_factor_display_iteration(iteration: VT) -> int:
    return round(_mode_factor_power(iteration))


def _mode_factor_value(*, lambda_i: float, eta: float, iteration: float) -> float:
    factor = 1.0 - eta * lambda_i
    if factor >= 0:
        return float(factor**iteration)

    nearest_integer = round(iteration)
    if abs(iteration - nearest_integer) <= ZERO_AXIS_EPSILON:
        return float(factor**nearest_integer)

    return float((abs(factor) ** iteration) * np.cos(PI * iteration))


def _mode_factor_iteration(iteration: VT) -> int:
    return _mode_factor_display_iteration(iteration)


def _mode_factor_axis(
    eta: VT,
    iteration: VT,
    *,
    lambda_min: float,
    lambda_max: float,
    eta_values: Sequence[float],
    width: float,
    height: float,
) -> VGroup:
    x_bound = _mode_factor_axis_bound(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        eta_values=eta_values,
    )
    axes = Axes(
        x_range=[-x_bound, x_bound, x_bound],
        y_range=[-1.0, 1.0, 1.0],
        x_length=width - 2 * SMALL_BUFF,
        y_length=height * MODE_FACTOR_AXIS_HEIGHT_RATIO,
        tips=False,
        axis_config={
            "include_ticks": False,
            "include_numbers": False,
            "stroke_color": WHITE,
            "stroke_width": MODE_CHART_BASELINE_STROKE_WIDTH,
            "stroke_opacity": 1.0,
        },
    )
    axes.y_axis.set_opacity(0)

    dot_radius = height * MODE_FACTOR_DOT_RADIUS_RATIO
    zero_tick_half_height = (
        dot_radius * MODE_FACTOR_ZERO_TICK_DOT_HEIGHT_RATIO
    )
    zero_tick = Line(DOWN * zero_tick_half_height, UP * zero_tick_half_height)
    zero_tick.set_stroke(WHITE, width=MODE_CHART_BASELINE_STROKE_WIDTH)
    zero_tick.move_to(axes.c2p(0, 0))
    zero_label = theme_math("0", color=C_TEXT, typography="caption")
    zero_label.next_to(zero_tick, DOWN, buff=SMALL_BUFF)

    title = theme_math(
        r"(1-",
        r"\eta",
        r"\lambda_i",
        r")^t",
        color=C_TEXT,
        typography="caption",
    )
    title[1].set_color(C_ETA)
    title[2].set_color(C_ORANGE)

    iteration_label = theme_math("t=", color=C_YELLOW, typography="caption")
    iteration_value = Integer(
        _mode_factor_display_iteration(iteration),
        color=C_TEXT,
        font_size=get_active_theme().typography.caption,
    )

    def update_iteration_value(mob: Integer) -> None:
        mob.set_value(_mode_factor_display_iteration(iteration))

    iteration_value.add_updater(update_iteration_value)
    iteration_readout = VGroup(iteration_label, iteration_value).arrange(
        RIGHT,
        buff=0,
    )

    dots = VGroup()
    for lambda_i, color in zip(
        _mode_factor_lambdas(lambda_min, lambda_max),
        MODE_FACTOR_DOT_COLORS,
        strict=True,
    ):
        dot = Dot(radius=dot_radius, color=color)

        def update_dot(mob: Dot, *, lambda_i: float = float(lambda_i)) -> None:
            value = _mode_factor_value(
                lambda_i=lambda_i,
                eta=~eta,
                iteration=_mode_factor_power(iteration),
            )
            value = float(np.clip(value, -x_bound, x_bound))
            mob.move_to(axes.c2p(value, 0))

        update_dot(dot)
        dot.add_updater(update_dot)
        dots.add(dot)

    axis_group = VGroup(axes, zero_tick, zero_label, dots)
    title.next_to(axis_group, UP, buff=SMALL_BUFF)
    iteration_readout.align_to(axes, LEFT)
    iteration_readout.set_y(title.get_y())
    return VGroup(title, iteration_readout, axis_group)


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


__all__ = [name for name in globals() if not name.startswith("__")]
