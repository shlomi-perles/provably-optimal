"""Momentum gradient descent on Rosenbrock, with live alpha/beta controls."""

from __future__ import annotations

from dataclasses import dataclass

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
    Line,
    MathTex,
    SurroundingRectangle,
    Tex,
    VGroup,
    VMobject,
    Write,
    always_redraw,
)
from scipy.optimize import rosen, rosen_der
from simplex import Caption, DN, Slide, VT, get_active_theme

type FloatArray = npt.NDArray[np.float64]


X_RANGE = (-2.0, 2.0)
Y_RANGE = (-0.5, 3.0)
START = np.array([-1.25, 1.8], dtype=np.float64)
OPTIMUM = np.array([1.0, 1.0], dtype=np.float64)
N_STEPS = 80

C_CONTOUR = "#7DD3FC"
C_START = "#FF6B6B"
C_OPTIMUM = "#FFD166"
C_PATH = "#35D07F"
C_ALPHA = "#F59E0B"
C_BETA = "#60A5FA"
C_MUTED = "#94A3B8"
C_PANEL = "#18212F"


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
        self.slide(title="Momentum on Rosenbrock")
        theme = get_active_theme()

        title = Tex(
            r"Momentum gradient descent on the Rosenbrock function",
            font_size=theme.typography.h2,
        )
        self.region.place(title, UP)
        self.region.update(top=title)

        alpha = VT(0.0012)
        beta = VT(0.0)
        axes = self._make_axes()
        contours = self._make_contours(axes)
        trajectory = always_redraw(lambda: self._trajectory(axes, ~alpha, ~beta))
        markers = self._make_static_markers(axes)
        plot = VGroup(axes, contours, trajectory, markers)

        controls = self._make_controls(alpha, beta)
        plot_region, control_region = self.region.split_regions(RIGHT, 2)
        plot_region.scale_and_place(plot)
        control_region.scale_and_place(controls)

        self.play(Write(title), Create(axes), FadeIn(contours), FadeIn(markers))
        self.play(FadeIn(controls), Write(trajectory))

        self.fragment(title="Increase momentum")
        self.play(beta @ 0.85, run_time=2.4)

        self.fragment(title="Increase step size")
        self.play(alpha @ 0.0018, run_time=2.4)

        self.fragment(title="High momentum")
        self.play(beta @ 0.93, run_time=2.4)

        self.fragment(title="Too much alpha")
        self.play(alpha @ 0.0024, run_time=2.4)

        self.fragment(title="Stable setting")
        self.play(alpha @ 0.0018, beta @ 0.93, run_time=2.4)

    def _make_axes(self) -> Axes:
        return Axes(
            x_range=[X_RANGE[0], X_RANGE[1], 1],
            y_range=[Y_RANGE[0], Y_RANGE[1], 1],
            x_length=7.4,
            y_length=5.4,
            tips=False,
            axis_config={"stroke_color": C_MUTED, "include_numbers": False},
        )

    def _make_contours(self, axes: Axes) -> VGroup:
        x_values = np.linspace(*X_RANGE, 220)
        y_values = np.linspace(*Y_RANGE, 220)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        log_values = np.log10(rosen(np.stack([x_grid, y_grid])) + 1.0)
        generator = contourpy.contour_generator(x=x_values, y=y_values, z=log_values)
        levels = np.linspace(0.12, 3.15, 15)

        contours = VGroup()
        for opacity, level in zip(np.linspace(0.28, 0.82, len(levels)), levels, strict=True):
            for segment in generator.lines(float(level)):
                if len(segment) < 2:
                    continue
                line = VMobject()
                line.set_points_as_corners(self._contour_points(axes, segment))
                line.set_stroke(C_CONTOUR, width=1.4, opacity=float(opacity))
                contours.add(line)
        return contours

    def _make_static_markers(self, axes: Axes) -> VGroup:
        start = Dot(self._axes_point(axes, START), color=C_START, radius=0.08)
        optimum = Dot(self._axes_point(axes, OPTIMUM), color=C_OPTIMUM, radius=0.08)
        optimum_halo = Circle(radius=0.2, color=C_OPTIMUM)
        optimum_halo.set_fill(C_OPTIMUM, opacity=0.14).set_stroke(width=0)
        optimum_halo.move_to(optimum)

        start_label = MathTex(r"x_0", color=C_START, font_size=28)
        start_label.next_to(start, UP + LEFT, buff=0.12)
        optimum_label = MathTex(r"x^\star", color=C_OPTIMUM, font_size=28)
        optimum_label.next_to(optimum, DOWN + RIGHT, buff=0.12)

        caption = Caption("Rosenbrock contours")
        caption.next_to(axes, DOWN, buff=0.22)
        return VGroup(start, optimum_halo, optimum, start_label, optimum_label, caption)

    def _make_controls(self, alpha: VT, beta: VT) -> VGroup:
        theme = get_active_theme()
        equation = MathTex(
            r"v_{t+1}",
            "=",
            r"\beta v_t",
            "+",
            r"\nabla f(x_t)",
            r"\qquad",
            r"x_{t+1}",
            "=",
            r"x_t",
            "-",
            r"\alpha v_{t+1}",
            font_size=theme.typography.body,
        )
        equation[2].set_color(C_BETA)
        equation[10].set_color(C_ALPHA)

        alpha_slider = self._slider(
            alpha,
            SliderSpec(r"\alpha", 0.0005, 0.0024, 4, C_ALPHA),
        )
        beta_slider = self._slider(
            beta,
            SliderSpec(r"\beta", 0.0, 0.95, 2, C_BETA),
        )

        readout = self._make_path_readout(alpha, beta)
        sliders = VGroup(alpha_slider, beta_slider).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        controls = VGroup(equation, sliders, readout).arrange(
            DOWN,
            buff=0.55,
            aligned_edge=LEFT,
        )
        panel = self._panel(controls)
        return VGroup(panel, controls)

    def _slider(self, tracker: VT, spec: SliderSpec) -> VGroup:
        theme = get_active_theme()
        track = Line(LEFT * 1.35, RIGHT * 1.35)
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

    def _trajectory(self, axes: Axes, alpha: float, beta: float) -> VGroup:
        points = self._momentum_points(alpha, beta)
        coords = [self._axes_point(axes, point) for point in points]
        trail = VMobject()
        trail.set_points_as_corners(coords)
        trail.set_stroke(C_PATH, width=4.5, opacity=0.95)

        step_dots = VGroup(
            *(Dot(coord, color=C_PATH, radius=0.035) for coord in coords[4:-1:4])
        )
        head = Dot(coords[-1], color=C_PATH, radius=0.09)
        head_halo = Circle(radius=0.18, color=C_PATH)
        head_halo.set_fill(C_PATH, opacity=0.16).set_stroke(width=0)
        head_halo.move_to(head)

        return VGroup(trail, step_dots, head_halo, head)

    def _make_path_readout(self, alpha: VT, beta: VT) -> VGroup:
        theme = get_active_theme()
        steps = DN(
            lambda: len(self._momentum_points(~alpha, ~beta)) - 1,
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

        for _ in range(N_STEPS):
            gradient = np.asarray(rosen_der(point), dtype=np.float64)
            velocity = beta * velocity + gradient
            candidate = point - alpha * velocity
            if not np.all(np.isfinite(candidate)):
                break
            if self._outside_plot(candidate):
                points.append(self._clip_to_plot(candidate))
                break
            point = candidate
            points.append(point.copy())
            if np.linalg.norm(point - OPTIMUM) < 0.025:
                break

        return np.asarray(points, dtype=np.float64)

    def _panel(self, mobject: VGroup) -> VGroup:
        panel = VGroup(*(SurroundingRectangle(mob, buff=0.16) for mob in mobject))
        panel.set_fill(C_PANEL, opacity=0.5).set_stroke(C_MUTED, width=0.8, opacity=0.28)
        return panel

    def _contour_points(self, axes: Axes, segment: FloatArray) -> list[FloatArray]:
        step = max(1, len(segment) // 130)
        sampled = segment[::step]
        if not np.array_equal(sampled[-1], segment[-1]):
            sampled = np.vstack([sampled, segment[-1]])
        return [self._axes_point(axes, point) for point in sampled]

    def _axes_point(self, axes: Axes, point: FloatArray) -> FloatArray:
        return np.asarray(axes.c2p(float(point[0]), float(point[1])), dtype=np.float64)

    def _slider_alpha(self, tracker: VT, spec: SliderSpec) -> float:
        value = np.clip(~tracker, spec.minimum, spec.maximum)
        return float((value - spec.minimum) / (spec.maximum - spec.minimum))

    def _outside_plot(self, point: FloatArray) -> bool:
        return (
            point[0] < X_RANGE[0] - 0.08
            or point[0] > X_RANGE[1] + 0.08
            or point[1] < Y_RANGE[0] - 0.08
            or point[1] > Y_RANGE[1] + 0.08
        )

    def _clip_to_plot(self, point: FloatArray) -> FloatArray:
        return np.array(
            [
                np.clip(point[0], *X_RANGE),
                np.clip(point[1], *Y_RANGE),
            ],
            dtype=np.float64,
        )
