"""Momentum gradient descent on Rosenbrock, with live alpha/beta controls."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import contourpy
import numpy as np
import numpy.typing as npt
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Axes,
    Circle,
    Create,
    Dot,
    FadeIn,
    Group,
    ImageMobject,
    Line,
    MathTex,
    Rectangle,
    SurroundingRectangle,
    Title,
    VGroup,
    VMobject,
    Write,
    YELLOW,
    always_redraw,
)
from simplex import Caption, DN, Slide, VT, color_substrings, get_active_theme

type FloatArray = npt.NDArray[np.float64]


X_RANGE = (-2.0, 2.0)
Y_RANGE = (-2.0 / 3.0 + 0.4, 2.0 / 3.0 + 0.4)
START = np.array([-1.21, 0.853], dtype=np.float64)
OPTIMUM = np.array([1.0, 1.0 / 3.0], dtype=np.float64)
N_STEPS = 80
ROSEN_Y_SCALE = 3.0
ROSEN_CURVATURE = 20.0

C_CONTOUR = "#91BBD0"
C_FRAME = "#BFC9D2"
C_OPTIMUM = "#4A6578"
C_PATH = "#FF6600"
C_ALPHA = C_PATH
C_BETA = "#2F65C8"
C_MUTED = "#94A3B8"
C_PANEL = "#18212F"
X_0_COLOR = YELLOW
HEATMAP_BLUE = "#3D8FC7"
HEATMAP_MAX_ALPHA = 0.42
STEP_DOT_RADIUS = 0.018
HEAD_DOT_RADIUS = 0.07
MAX_POINT_NORM = 1e3


@dataclass(frozen=True, slots=True)
class SliderSpec:
    label: str
    minimum: float
    maximum: float
    decimals: int
    color: str


class MomentumRosenbrock(Slide):
    """Illustrate how alpha and beta reshape momentum GD on Rosenbrock contours."""

    def construct(self) -> None:
        self.next_slide(name="Momentum on Rosenbrock")
        theme = get_active_theme()

        title = Title(
            r"Momentum gradient descent on the Rosenbrock function",
            font_size=theme.typography.h2,
            include_underline=False,
        )
        self.region.place(title, UP)
        self.region.update(top=title)

        alpha = VT(0.0016)
        beta = VT(0.74)
        axes = self._make_axes()
        heatmap = self._make_heatmap(axes)
        contours = self._make_contours(axes)
        trajectory = self._make_trajectory(axes, alpha, beta)
        markers = self._make_static_markers(axes)
        frame = self._plot_frame(axes)
        plot = Group(axes, heatmap, contours, trajectory, markers, frame)

        controls = self._make_controls(alpha, beta)
        plot_region, control_region = self.region.split_regions(DOWN, 2)
        plot_region.scale_and_place(plot, buff=0.08)
        control_region.scale_and_place(controls, buff=0.16)

        self.play(Write(title), FadeIn(heatmap), Write(contours), Write(markers), Write(frame), Write(trajectory))
        self.play(FadeIn(controls))

        self.next_slide()
        self.play(alpha @ 0.004, run_time=2.4)

        self.next_slide()
        self.play(alpha @ 0.0016, run_time=2.4)

        self.next_slide()
        self.play(beta @ 0.9, run_time=2.4)

        self.next_slide()
        self.play(beta @ 0.93, run_time=2.4)

        self.next_slide()
        self.play(beta @ 0.2, run_time=2.4)

        self.clear_scene()

    def _make_axes(self) -> Axes:
        return Axes(
            x_range=[X_RANGE[0], X_RANGE[1], 1],
            y_range=[Y_RANGE[0], Y_RANGE[1], 1],
            x_length=12.4,
            y_length=3.35,
            tips=False,
            axis_config={
                "include_ticks": False,
                "stroke_opacity": 0,
            },
        )

    def _make_heatmap(self, axes: Axes) -> ImageMobject:
        width, height = 840, 240
        x_values = np.linspace(*X_RANGE, width)
        y_values = np.linspace(Y_RANGE[1], Y_RANGE[0], height)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        values = np.log10(self._banana_value(np.stack([x_grid, y_grid])) + 1.0)
        image = ImageMobject(self._blue_alpha_heatmap(self._normalize_heat(values)))
        frame = self._plot_frame(axes)
        image.stretch_to_fit_width(frame.width)
        image.stretch_to_fit_height(frame.height)
        image.move_to(frame)
        return image

    def _make_contours(self, axes: Axes) -> VGroup:
        x_values = np.linspace(*X_RANGE, 220)
        y_values = np.linspace(*Y_RANGE, 220)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        log_values = np.log10(self._banana_value(np.stack([x_grid, y_grid])) + 1.0)
        generator = contourpy.contour_generator(x=x_values, y=y_values, z=log_values)
        levels = np.linspace(0.05, 2.05, 18)

        contours = VGroup()
        for opacity, level in zip(np.linspace(0.34, 0.72, len(levels)), levels, strict=True):
            for segment in generator.lines(float(level)):
                if len(segment) < 2:
                    continue
                line = VMobject()
                line.set_points_as_corners(self._contour_points(axes, segment))
                line.set_stroke(C_CONTOUR, width=1.05, opacity=float(opacity))
                contours.add(line)
        return contours

    def _make_static_markers(self, axes: Axes) -> VGroup:
        start = Dot(self._axes_point(axes, START), color=X_0_COLOR, radius=0.08)
        optimum = Dot(self._axes_point(axes, OPTIMUM), color=C_OPTIMUM, radius=0.08)
        optimum_halo = Circle(radius=0.2, color=C_OPTIMUM)
        optimum_halo.set_fill(C_OPTIMUM, opacity=0.14).set_stroke(width=0)
        optimum_halo.move_to(optimum)

        start_label = MathTex(r"x_0", color=X_0_COLOR, font_size=28)
        start_label.next_to(start, UP + RIGHT, buff=0.12)
        optimum_label = MathTex(r"x^\star", color=C_OPTIMUM, font_size=28)
        optimum_label.next_to(optimum, DOWN + RIGHT, buff=0.12)

        return VGroup(start, optimum_halo, optimum, start_label, optimum_label)

    def _make_controls(self, alpha: VT, beta: VT) -> VGroup:
        theme = get_active_theme()
        equation = MathTex(
            r"v_{t+1}=\beta v_t+\nabla f(x_t)\qquad x_{t+1}=x_t-\alpha v_{t+1}",
            font_size=theme.typography.body,
        )
        color_substrings(
            equation,
            {
                r"x_{t+1}": X_0_COLOR,
                r"x_t": X_0_COLOR,
                r"\alpha": C_ALPHA,
                r"\beta": C_BETA,
            },
        )

        alpha_slider = self._slider(
            alpha,
            SliderSpec(r"\alpha", 0.0, 0.006, 4, C_ALPHA),
        )
        beta_slider = self._slider(
            beta,
            SliderSpec(r"\beta", 0.0, 0.99, 2, C_BETA),
        )

        readout = self._make_path_readout(alpha, beta)
        sliders = VGroup(alpha_slider, beta_slider).arrange(RIGHT, buff=0.45)
        controls = VGroup(equation, sliders, readout).arrange(RIGHT, buff=0.52)
        panel = self._panel(controls)
        return VGroup(panel, controls)

    def _slider(self, tracker: VT, spec: SliderSpec) -> VGroup:
        theme = get_active_theme()
        track = Line(LEFT * 1.05, RIGHT * 1.05)
        track.set_stroke(C_MUTED, width=5, opacity=0.45)

        label = MathTex(spec.label, color=spec.color, font_size=theme.typography.body)
        label.next_to(track, LEFT, buff=0.28)

        value = DN(tracker, num_decimal_places=spec.decimals, font_size=theme.typography.caption)
        value.next_to(track, RIGHT, buff=0.25)
        value.add_updater(lambda mob: mob.next_to(track, RIGHT, buff=0.25))

        fill = always_redraw(
            lambda: Line(
                track.get_start(),
                track.point_from_proportion(self._slider_alpha(tracker, spec)),
            ).set_stroke(spec.color, width=7)
        )
        knob = always_redraw(
            lambda: Dot(
                track.point_from_proportion(self._slider_alpha(tracker, spec)),
                color=spec.color,
                radius=0.095,
            )
        )

        return VGroup(label, track, fill, knob, value)

    def _make_trajectory(self, axes: Axes, alpha: VT, beta: VT) -> VGroup:
        points = self._momentum_points(~alpha, ~beta)
        dots = VGroup(
            *(
                Dot(
                    self._axes_point(axes, point),
                    color=C_PATH,
                    radius=STEP_DOT_RADIUS,
                )
                for point in points
            )
        )
        dots[0].set_color(X_0_COLOR).scale(1.35)
        dots[-1].scale(2.0)

        self._attach_start_updater(dots[0], axes)
        for previous, dot in pairwise(dots):
            self._attach_step_updater(dot, previous, axes, alpha, beta)

        connectors = VGroup()
        for start, end in pairwise(dots):
            connector = Line(start.get_center(), end.get_center())
            connector.set_stroke(C_PATH, width=2.1, opacity=0.86)
            connector.add_updater(
                lambda mob, start=start, end=end: self._update_connector(mob, start, end)
            )
            connectors.add(connector)

        head_ring = Circle(radius=0.11, color=C_PATH)
        head_ring.set_stroke(C_PATH, width=2.0)
        head_ring.add_updater(lambda mob: mob.move_to(dots[-1]))

        return VGroup(dots, connectors, head_ring)

    def _make_path_readout(self, alpha: VT, beta: VT) -> VGroup:
        theme = get_active_theme()
        steps = DN(
            lambda: N_STEPS,
            num_decimal_places=0,
            font_size=theme.typography.caption,
        )
        distance = DN(
            lambda: self._path_distance(~alpha, ~beta),
            num_decimal_places=2,
            font_size=theme.typography.caption,
        )

        step_row = VGroup(Caption("steps"), steps).arrange(RIGHT, buff=0.18)
        distance_row = VGroup(Caption(r"$\|x_t-x^\star\|$"), distance).arrange(
            RIGHT,
            buff=0.18,
        )
        readout = VGroup(step_row, distance_row).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        readout.set_color(C_MUTED)
        return readout

    def _path_distance(self, alpha: float, beta: float) -> float:
        points = self._momentum_points(alpha, beta)
        return float(np.linalg.norm(points[-1] - OPTIMUM))

    def _momentum_points(self, alpha: float, beta: float) -> FloatArray:
        point = START.copy()
        velocity = np.zeros(2, dtype=np.float64)
        points = [point.copy()]
        stopped = False

        for _ in range(N_STEPS):
            if stopped:
                points.append(point.copy())
                continue
            candidate, velocity = self._next_momentum_point(point, velocity, alpha, beta)
            if not np.all(np.isfinite(candidate)) or np.linalg.norm(candidate) > MAX_POINT_NORM:
                stopped = True
            else:
                point = candidate
            points.append(point.copy())

        return np.asarray(points, dtype=np.float64)

    def _panel(self, mobject: VGroup) -> VGroup:
        panel = VGroup(*(SurroundingRectangle(mob, buff=0.16) for mob in mobject))
        panel.set_fill(C_PANEL, opacity=0.5).set_stroke(C_MUTED, width=0.8, opacity=0.28)
        return panel

    def _plot_frame(self, axes: Axes) -> Rectangle:
        lower_left = axes.c2p(X_RANGE[0], Y_RANGE[0])
        upper_right = axes.c2p(X_RANGE[1], Y_RANGE[1])
        frame = Rectangle(
            width=float(upper_right[0] - lower_left[0]),
            height=float(upper_right[1] - lower_left[1]),
        )
        frame.set_stroke(C_FRAME, width=1.0, opacity=0.9)
        frame.move_to((lower_left + upper_right) / 2)
        return frame

    def _contour_points(self, axes: Axes, segment: FloatArray) -> list[FloatArray]:
        step = max(1, len(segment) // 130)
        sampled = segment[::step]
        if not np.array_equal(sampled[-1], segment[-1]):
            sampled = np.vstack([sampled, segment[-1]])
        return [self._axes_point(axes, point) for point in sampled]

    def _axes_point(self, axes: Axes, point: FloatArray) -> FloatArray:
        return np.asarray(axes.c2p(float(point[0]), float(point[1])), dtype=np.float64)

    def _banana_value(self, points: FloatArray) -> FloatArray:
        x = points[0]
        y = ROSEN_Y_SCALE * points[1]
        return (1 - x) ** 2 + ROSEN_CURVATURE * (y - x * x) ** 2

    def _banana_gradient(self, point: FloatArray) -> FloatArray:
        x = float(point[0])
        y = ROSEN_Y_SCALE * float(point[1])
        offset = y - x * x
        return np.array(
            [
                -2 * (1 - x) - 4 * ROSEN_CURVATURE * x * offset,
                2 * ROSEN_CURVATURE * ROSEN_Y_SCALE * offset,
            ],
            dtype=np.float64,
        )

    def _next_momentum_point(
        self,
        point: FloatArray,
        velocity: FloatArray,
        alpha: float,
        beta: float,
    ) -> tuple[FloatArray, FloatArray]:
        next_velocity = beta * velocity + self._banana_gradient(point)
        return point - alpha * next_velocity, next_velocity

    def _slider_alpha(self, tracker: VT, spec: SliderSpec) -> float:
        value = np.clip(~tracker, spec.minimum, spec.maximum)
        return float((value - spec.minimum) / (spec.maximum - spec.minimum))

    def _attach_start_updater(self, dot: Dot, axes: Axes) -> None:
        dot.momentum_point = START.copy()
        dot.momentum_velocity = np.zeros(2, dtype=np.float64)
        dot.momentum_visible = True
        dot.add_updater(lambda mob: self._update_start_dot(mob, axes))

    def _attach_step_updater(
        self,
        dot: Dot,
        previous: Dot,
        axes: Axes,
        alpha: VT,
        beta: VT,
    ) -> None:
        dot.momentum_point = START.copy()
        dot.momentum_velocity = np.zeros(2, dtype=np.float64)
        dot.momentum_visible = True
        dot.add_updater(
            lambda mob: self._update_step_dot(mob, previous, axes, ~alpha, ~beta)
        )

    def _update_start_dot(self, dot: Dot, axes: Axes) -> None:
        dot.momentum_point = START.copy()
        dot.momentum_velocity = np.zeros(2, dtype=np.float64)
        dot.momentum_visible = True
        dot.move_to(self._axes_point(axes, START))
        dot.set_opacity(1)

    def _update_step_dot(
        self,
        dot: Dot,
        previous: Dot,
        axes: Axes,
        alpha: float,
        beta: float,
    ) -> None:
        point, velocity = self._next_momentum_point(
            previous.momentum_point,
            previous.momentum_velocity,
            alpha,
            beta,
        )
        if not np.all(np.isfinite(point)) or np.linalg.norm(point) > MAX_POINT_NORM:
            point = previous.momentum_point.copy()
            velocity = previous.momentum_velocity.copy()

        dot.momentum_point = point
        dot.momentum_velocity = velocity
        dot.momentum_visible = self._inside_plot(point)
        dot.move_to(self._axes_point(axes, point) if dot.momentum_visible else previous)
        dot.set_opacity(1 if dot.momentum_visible else 0)

    def _update_connector(self, connector: Line, start: Dot, end: Dot) -> None:
        visible = start.momentum_visible and end.momentum_visible
        if not visible:
            connector.set_opacity(0)
            return

        connector.set_points_as_corners([start.get_center(), end.get_center()])
        connector.set_stroke(C_PATH, width=2.1, opacity=0.86)

    def _inside_plot(self, point: FloatArray) -> bool:
        return (
            X_RANGE[0] <= point[0] <= X_RANGE[1]
            and Y_RANGE[0] <= point[1] <= Y_RANGE[1]
        )

    def _normalize_heat(self, values: FloatArray) -> FloatArray:
        lower, upper = np.quantile(values, [0.03, 0.92])
        normalized = np.clip((values - lower) / (upper - lower), 0, 1)
        return normalized**0.72

    def _blue_alpha_heatmap(self, values: FloatArray) -> npt.NDArray[np.uint8]:
        intensity = (1.0 - values) ** 1.35
        rgb = np.array(self._hex_to_rgb(HEATMAP_BLUE), dtype=np.float64)
        alpha = 255.0 * HEATMAP_MAX_ALPHA * intensity
        channels = [np.full_like(values, channel) for channel in rgb]
        return np.dstack([*channels, alpha]).astype(np.uint8)

    def _hex_to_rgb(self, color: str) -> tuple[int, int, int]:
        raw = color.removeprefix("#")
        return tuple(int(raw[index : index + 2], 16) for index in (0, 2, 4))
