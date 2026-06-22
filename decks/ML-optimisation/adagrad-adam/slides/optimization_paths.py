"""Reusable optimization trajectory mobjects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from manim import Axes, Circle, Dot, Line, VGroup

from slides.plotting import axes_point, inside_axes, plot_frame
from slides.style import C_YELLOW

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OptimizationPathStyle:
    """Visual proportions shared with the Rosenbrock momentum trajectory."""

    color: str
    start_color: str = C_YELLOW
    stroke_width: float = 2.1
    stroke_opacity: float = 0.86
    step_dot_frame_height_ratio: float = 1 / 186
    start_dot_scale: float = 1.35
    head_dot_frame_height_ratio: float = 1 / 48
    head_dot_scale: float = 2.0
    head_ring_dot_scale: float = 1.6
    head_ring_stroke_width: float = 2.0


class OptimizationPath(VGroup):
    """Polyline, step dots, and live-end ring for a discrete optimizer path."""

    def __init__(
        self,
        axes: Axes,
        points: FloatArray,
        *,
        style: OptimizationPathStyle,
    ) -> None:
        frame = plot_frame(axes)
        step_dot_radius = frame.height * style.step_dot_frame_height_ratio
        head_dot_radius = frame.height * style.head_dot_frame_height_ratio
        dots = VGroup()
        connectors = VGroup()
        previous_screen: FloatArray | None = None
        previous_index: int | None = None

        for index, point in enumerate(points):
            if not np.all(np.isfinite(point)) or not inside_axes(axes, point):
                previous_screen = None
                previous_index = None
                continue

            screen_point = axes_point(axes, point)
            if previous_screen is not None and previous_index == index - 1:
                connector = Line(previous_screen, screen_point)
                connector.set_stroke(
                    style.color,
                    width=style.stroke_width,
                    opacity=style.stroke_opacity,
                )
                connectors.add(connector)

            dot = Dot(screen_point, color=style.color, radius=step_dot_radius)
            if index == 0:
                dot.set_color(style.start_color)
                dot.scale(style.start_dot_scale)
            dots.add(dot)
            previous_screen = screen_point
            previous_index = index

        if len(dots) == 0:
            super().__init__(dots, connectors)
            return

        dots[-1].scale(style.head_dot_scale)
        head_ring = Circle(radius=head_dot_radius * style.head_ring_dot_scale, color=style.color)
        head_ring.set_stroke(style.color, width=style.head_ring_stroke_width)
        head_ring.move_to(dots[-1])

        super().__init__(dots, connectors, head_ring)
