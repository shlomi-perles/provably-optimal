"""Axes, frame, and heatmap helpers shared by deck figures."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from manim import Axes, Rectangle

from slides.helpers.style import C_FRAME, C_MUTED, HEATMAP_BLUE, HEATMAP_MAX_ALPHA

FloatArray = npt.NDArray[np.float64]

AXIS_STROKE_WIDTH = 1.4
AXIS_STROKE_OPACITY = 0.82
FRAME_STROKE_WIDTH = 1.0
FRAME_STROKE_OPACITY = 0.9
HEATMAP_QUANTILE_RANGE = (0.03, 0.92)
HEATMAP_GAMMA = 0.72
HEATMAP_INTENSITY_GAMMA = 1.35
MAX_RGB_CHANNEL = 255.0


def axis_config() -> dict[str, object]:
    return {
        "include_ticks": False,
        "include_numbers": False,
        "stroke_color": C_MUTED,
        "stroke_width": AXIS_STROKE_WIDTH,
        "stroke_opacity": AXIS_STROKE_OPACITY,
    }


def make_axes(
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
        axis_config=axis_config(),
    )


def axes_limits(axes: Axes) -> tuple[float, float, float, float]:
    return (
        float(axes.x_range[0]),
        float(axes.x_range[1]),
        float(axes.y_range[0]),
        float(axes.y_range[1]),
    )


def axes_point(axes: Axes, point: FloatArray) -> FloatArray:
    return np.asarray(axes.c2p(float(point[0]), float(point[1])), dtype=np.float64)


def inside_axes(axes: Axes, point: FloatArray) -> bool:
    x_min, x_max, y_min, y_max = axes_limits(axes)
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def plot_frame(axes: Axes) -> Rectangle:
    x_min, x_max, y_min, y_max = axes_limits(axes)
    lower_left = axes.c2p(x_min, y_min)
    upper_right = axes.c2p(x_max, y_max)
    frame = Rectangle(
        width=float(upper_right[0] - lower_left[0]),
        height=float(upper_right[1] - lower_left[1]),
    )
    frame.set_stroke(C_FRAME, width=FRAME_STROKE_WIDTH, opacity=FRAME_STROKE_OPACITY)
    frame.move_to((lower_left + upper_right) / 2)
    return frame


def sample_contour_points(axes: Axes, segment: FloatArray, *, max_points: int) -> list[FloatArray]:
    step = max(1, len(segment) // max_points)
    sampled = segment[::step]
    if not np.array_equal(sampled[-1], segment[-1]):
        sampled = np.vstack([sampled, segment[-1]])
    return [axes.c2p(float(x), float(y)) for x, y in sampled]


def normalize_heat(values: FloatArray) -> FloatArray:
    lower, upper = np.quantile(values, HEATMAP_QUANTILE_RANGE)
    normalized = np.clip((values - lower) / (upper - lower), 0, 1)
    return normalized**HEATMAP_GAMMA


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    raw = color.removeprefix("#")
    return tuple(int(raw[index : index + 2], 16) for index in (0, 2, 4))


def blue_alpha_heatmap(values: FloatArray) -> npt.NDArray[np.uint8]:
    intensity = (1.0 - values) ** HEATMAP_INTENSITY_GAMMA
    rgb = np.array(hex_to_rgb(HEATMAP_BLUE), dtype=np.float64)
    alpha = MAX_RGB_CHANNEL * HEATMAP_MAX_ALPHA * intensity
    channels = [np.full_like(values, channel) for channel in rgb]
    return np.dstack([*channels, alpha]).astype(np.uint8)
