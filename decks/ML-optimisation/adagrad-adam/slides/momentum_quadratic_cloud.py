"""Momentum memory as samples from a quadratic bowl."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from manim import (
    DEGREES,
    Dot3D,
    FadeIn,
    Group,
    LaggedStart,
    Line3D,
    MathTex,
    SMALL_BUFF,
    ThreeDAxes,
    Title,
    UP,
    Write,
    config,
)
from manim.utils.color import color_gradient
from scipy.stats import qmc

config.renderer = "opengl"
config.write_to_movie = True

from simplex import ThreeDSlide, get_active_theme

from slides.helpers.style import (
    C_BLUE,
    C_FRAME,
    C_GREEN,
    C_GRID,
    C_MUTED,
    C_ORANGE,
    C_TEAL,
)

type FloatArray = npt.NDArray[np.float64]

TITLE_TEXT = "Memory reveals curvature"
DOMAIN_RADIUS = 2.0
AXIS_TICK_STEP = DOMAIN_RADIUS / 2
VALUE_MAX = 2 * DOMAIN_RADIUS**2
AXES_X_RANGE = (-DOMAIN_RADIUS, DOMAIN_RADIUS, AXIS_TICK_STEP)
AXES_Y_RANGE = (-DOMAIN_RADIUS, DOMAIN_RADIUS, AXIS_TICK_STEP)
AXES_Z_RANGE = (0.0, VALUE_MAX, DOMAIN_RADIUS)
AXES_X_LENGTH = 5.8
AXES_Y_LENGTH = AXES_X_LENGTH
AXES_Z_LENGTH = AXES_X_LENGTH / 2
SAMPLES_PER_AXIS_TICK = 3
SAMPLE_AXIS_COUNT = round(2 * DOMAIN_RADIUS / AXIS_TICK_STEP * SAMPLES_PER_AXIS_TICK + 1)
SAMPLE_COUNT = SAMPLE_AXIS_COUNT**2
SAMPLE_EXCLUSION_RADIUS = AXIS_TICK_STEP / SAMPLES_PER_AXIS_TICK
DOT_RADIUS_TO_SAMPLE_CELL = 1 / 4
FIRST_SAMPLE = np.zeros(2, dtype=np.float64)
COLOR_LEVEL_COUNT = SAMPLE_AXIS_COUNT
FRAME_THICKNESS = AXES_X_LENGTH / 360
GRID_THICKNESS = FRAME_THICKNESS / 2
GRID_OPACITY = 0.26
FRAME_OPACITY = 2 * GRID_OPACITY
AXIS_LABEL_OFFSET = AXIS_TICK_STEP / 2
CAMERA_ELEVATION = np.arctan(1 / np.sqrt(2))
CAMERA_PHI = 90 * DEGREES - CAMERA_ELEVATION
CAMERA_THETA = -45 * DEGREES


def _quadratic_values(points: FloatArray) -> FloatArray:
    return np.sum(points**2, axis=-1)


def _sample_points() -> FloatArray:
    sampler = qmc.Halton(d=2, scramble=False)
    candidate_count = 2 * SAMPLE_COUNT
    unit_points = sampler.random(candidate_count + 1)[1:]
    candidates = (unit_points - 0.5) * (2 * DOMAIN_RADIUS)
    rest = candidates[np.linalg.norm(candidates - FIRST_SAMPLE, axis=1) >= SAMPLE_EXCLUSION_RADIUS]
    rest = rest[: SAMPLE_COUNT - 1]
    ordering = np.lexsort((np.arctan2(rest[:, 1], rest[:, 0]), _quadratic_values(rest)))
    return np.vstack((FIRST_SAMPLE, rest[ordering]))


def _color_for_value(value: float):
    palette = color_gradient((C_BLUE, C_TEAL, C_GREEN, C_ORANGE), COLOR_LEVEL_COUNT)
    index = round(np.clip(value / VALUE_MAX, 0.0, 1.0) * (COLOR_LEVEL_COUNT - 1))
    return palette[index]


class MomentumQuadraticCloud(ThreeDSlide):
    """Show that many remembered samples reveal a quadratic bowl."""

    def setup(self) -> None:
        super().setup()
        self.region = self.region.fix_in_frame()

    def construct(self) -> None:
        self.slide(title=TITLE_TEXT)
        self.set_camera_orientation(phi=CAMERA_PHI, theta=CAMERA_THETA)

        title = Title(TITLE_TEXT)
        self.region.place(title, UP)
        self.region.update(top=title)
        title.fix_in_frame()

        axes = self._make_axes()
        frame = self._make_frame(axes)
        axis_labels = self._make_axis_labels(axes)
        dots = self._make_dots(axes, _sample_points())
        world = Group(axes, frame, axis_labels, dots)
        self.region.scale_and_place(world, buff=SMALL_BUFF)
        for label in axis_labels:
            label.fix_orientation()

        self.play(
            Write(title),
            FadeIn(frame),
            FadeIn(axis_labels),
            FadeIn(dots[0]),
        )
        self.next_slide()

        rest_dots = Group(*dots[1:])
        self.play(
            LaggedStart(
                *(FadeIn(dot) for dot in rest_dots),
                lag_ratio=1 / len(rest_dots),
            )
        )
        self.next_slide()

        self.clear_scene()
        self.next_slide()

    def _make_axes(self) -> ThreeDAxes:
        axes = ThreeDAxes(
            x_range=[*AXES_X_RANGE],
            y_range=[*AXES_Y_RANGE],
            z_range=[*AXES_Z_RANGE],
            x_length=AXES_X_LENGTH,
            y_length=AXES_Y_LENGTH,
            z_length=AXES_Z_LENGTH,
            tips=False,
            axis_config={
                "include_ticks": False,
                "include_numbers": False,
                "stroke_opacity": 0,
            },
        )
        axes.set_opacity(0)
        return axes

    def _make_frame(self, axes: ThreeDAxes) -> Group:
        x_min, x_max, _ = AXES_X_RANGE
        y_min, y_max, _ = AXES_Y_RANGE
        z_min, z_max, z_step = AXES_Z_RANGE

        def point(x: float, y: float, z: float) -> FloatArray:
            return axes.c2p(x, y, z)

        corners = {
            (x, y, z): point(x, y, z)
            for x in (x_min, x_max)
            for y in (y_min, y_max)
            for z in (z_min, z_max)
        }
        edge_specs = (
            ((x_min, y_min, z_min), (x_max, y_min, z_min)),
            ((x_min, y_max, z_min), (x_max, y_max, z_min)),
            ((x_min, y_min, z_max), (x_max, y_min, z_max)),
            ((x_min, y_max, z_max), (x_max, y_max, z_max)),
            ((x_min, y_min, z_min), (x_min, y_max, z_min)),
            ((x_max, y_min, z_min), (x_max, y_max, z_min)),
            ((x_min, y_min, z_max), (x_min, y_max, z_max)),
            ((x_max, y_min, z_max), (x_max, y_max, z_max)),
            ((x_min, y_min, z_min), (x_min, y_min, z_max)),
            ((x_max, y_min, z_min), (x_max, y_min, z_max)),
            ((x_min, y_max, z_min), (x_min, y_max, z_max)),
            ((x_max, y_max, z_min), (x_max, y_max, z_max)),
        )
        edges = Group(
            *(
                self._line(corners[start], corners[end], color=C_FRAME, thickness=FRAME_THICKNESS)
                for start, end in edge_specs
            )
        )
        edges.set_opacity(FRAME_OPACITY)

        ticks_xy = np.arange(x_min, x_max + AXIS_TICK_STEP, AXIS_TICK_STEP)
        ticks_z = np.arange(z_min + z_step, z_max, z_step)
        grid_lines = []
        for tick in ticks_xy:
            grid_lines.append(self._grid_line(point(tick, y_min, z_min), point(tick, y_max, z_min)))
            grid_lines.append(self._grid_line(point(x_min, tick, z_min), point(x_max, tick, z_min)))
            grid_lines.append(self._grid_line(point(tick, y_max, z_min), point(tick, y_max, z_max)))
            grid_lines.append(self._grid_line(point(x_max, tick, z_min), point(x_max, tick, z_max)))
        for tick in ticks_z:
            grid_lines.append(self._grid_line(point(x_min, y_max, tick), point(x_max, y_max, tick)))
            grid_lines.append(self._grid_line(point(x_max, y_min, tick), point(x_max, y_max, tick)))

        grid = Group(*grid_lines)
        grid.set_opacity(GRID_OPACITY)
        return Group(grid, edges)

    def _grid_line(self, start: FloatArray, end: FloatArray) -> Line3D:
        return self._line(start, end, color=C_GRID, thickness=GRID_THICKNESS)

    @staticmethod
    def _line(start: FloatArray, end: FloatArray, *, color: str, thickness: float) -> Line3D:
        return Line3D(start, end, color=color, thickness=thickness)

    def _make_dots(self, axes: ThreeDAxes, points: FloatArray) -> Group:
        sample_cell_width = axes.x_axis.get_length() / SAMPLE_AXIS_COUNT
        dot_radius = sample_cell_width * DOT_RADIUS_TO_SAMPLE_CELL
        return Group(
            *(
                self._dot(axes, point, dot_radius)
                for point in points
            )
        )

    def _dot(self, axes: ThreeDAxes, point: FloatArray, radius: float):
        value = float(_quadratic_values(point))
        return Dot3D(
            axes.c2p(float(point[0]), float(point[1]), value),
            color=_color_for_value(value),
            radius=radius,
        )

    def _make_axis_labels(self, axes: ThreeDAxes) -> Group:
        theme = get_active_theme()
        font_size = theme.typography.caption
        x_min, x_max, _ = AXES_X_RANGE
        y_min, y_max, _ = AXES_Y_RANGE
        z_min, z_max, _ = AXES_Z_RANGE
        return Group(
            MathTex(r"x_1", color=C_MUTED, font_size=font_size).move_to(
                axes.c2p(x_max, y_min - AXIS_LABEL_OFFSET, z_min)
            ),
            MathTex(r"x_2", color=C_MUTED, font_size=font_size).move_to(
                axes.c2p(x_max + AXIS_LABEL_OFFSET, y_max, z_min)
            ),
            MathTex(r"f(x)", color=C_MUTED, font_size=font_size).move_to(
                axes.c2p(x_min, y_min, z_max)
            ),
        )
