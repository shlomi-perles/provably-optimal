"""Momentum gradient descent on Rosenbrock, with live alpha/beta controls."""

from __future__ import annotations

from itertools import pairwise

import contourpy
import numpy as np
import numpy.typing as npt
from manim import (
    DOWN,
    DR,
    LEFT,
    MED_LARGE_BUFF,
    SMALL_BUFF,
    RIGHT,
    UP,
    UR,
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
    always_redraw,
)
from simplex import Caption, DN, Slide, VT, color_substrings, get_active_theme

from slides.helpers.controls import SliderSpec, ValueSlider
from slides.helpers.plotting import (
    axes_point as _axes_point,
    blue_alpha_heatmap as _blue_alpha_heatmap,
    normalize_heat as _normalize_heat,
    plot_frame as _plot_frame,
    sample_contour_points as _sample_contour_points,
)
from slides.helpers.style import (
    C_CONTOUR,
    C_FRAME,
    C_MUTED,
    C_OPTIMUM,
    C_OPTIMUM_STROKE,
    C_ORANGE,
    C_PANEL,
    C_YELLOW,
    LAYER_CONTOUR,
    LAYER_FRAME,
    LAYER_HEATMAP,
    LAYER_MARKERS,
    LAYER_TRAJECTORY,
    label_for_dot,
    segmented_panel,
)

type FloatArray = npt.NDArray[np.float64]


X_RANGE = (-2.0, 2.0)
Y_RANGE = (-2.0 / 3.0 + 0.4, 2.0 / 3.0 + 0.4)
START = np.array([-1.21, 0.853], dtype=np.float64)
OPTIMUM = np.array([1.0, 1.0 / 3.0], dtype=np.float64)
N_STEPS = 80
ROSEN_Y_SCALE = 3.0
ROSEN_CURVATURE = 20.0
INITIAL_ALPHA = 0.0016
INITIAL_BETA = 0.74
ALPHA_SLIDER_RANGE = (0.0, 0.006)
BETA_SLIDER_RANGE = (0.0, 0.99)
ALPHA_DECIMALS = 4
BETA_DECIMALS = 2
ALPHA_SWEEP_VALUES = (0.004, INITIAL_ALPHA)
BETA_SWEEP_VALUES = (0.9, 0.93, 0.2)
PARAMETER_SWEEP_RUN_TIME = 2.4

C_PATH = C_ORANGE
C_ALPHA = C_PATH
C_BETA = "#2F65C8"
X_0_COLOR = C_YELLOW
MAX_POINT_NORM = 1e3
PLOT_X_LENGTH = 12.4
PLOT_Y_LENGTH = 3.35
HEATMAP_WIDTH = 840
HEATMAP_HEIGHT = 240
ROSEN_CONTOUR_SAMPLES = 220
ROSEN_LEVEL_COUNT = 18
ROSEN_LEVEL_RANGE = (0.05, 2.05)
ROSEN_CONTOUR_OPACITY_RANGE = (0.34, 0.72)
ROSEN_CONTOUR_MAX_POINTS = 130
ROSEN_CONTOUR_STROKE_WIDTH = 1.05
ROSEN_MARKER_FRAME_HEIGHT_RATIO = 1 / 42
OPTIMUM_HALO_DOT_SCALE = 2.5
OPTIMUM_STROKE_WIDTH = 2.0
OPTIMUM_STROKE_OPACITY = 0.95
OPTIMUM_HALO_FILL_OPACITY = 0.16
OPTIMUM_HALO_STROKE_WIDTH = 1.0
OPTIMUM_HALO_STROKE_OPACITY = 0.35
STEP_DOT_FRAME_HEIGHT_RATIO = 1 / 186
START_DOT_SCALE = 1.35
HEAD_DOT_FRAME_HEIGHT_RATIO = 1 / 48
HEAD_DOT_SCALE = 2.0
HEAD_RING_DOT_SCALE = 1.6
TRAJECTORY_STROKE_WIDTH = 2.1
TRAJECTORY_STROKE_OPACITY = 0.86
HEAD_RING_STROKE_WIDTH = 2.0
MOMENTUM_SLIDER_HALF_LENGTH = 1.05


class MomentumRosenbrock(Slide):
    """Illustrate how alpha and beta reshape momentum GD on Rosenbrock contours."""

    def construct(self) -> None:
        self.next_slide(name="Momentum on Rosenbrock")

        title = Title(
            r"Momentum gradient descent on the Rosenbrock function",
        )
        self.region.place(title, UP)
        self.region.update(top=title)

        alpha = VT(INITIAL_ALPHA)
        beta = VT(INITIAL_BETA)
        axes = self._make_axes()
        heatmap = self._make_heatmap(axes).set_z_index(LAYER_HEATMAP)
        contours = self._make_contours(axes).set_z_index(LAYER_CONTOUR)
        trajectory = self._make_trajectory(axes, alpha, beta).set_z_index(LAYER_TRAJECTORY)
        markers = self._make_static_markers(axes).set_z_index(LAYER_MARKERS)
        frame = _plot_frame(axes).set_z_index(LAYER_FRAME)
        plot = Group(axes, heatmap, contours, trajectory, frame, markers)

        controls = self._make_controls(alpha, beta)
        plot_region, control_region = self.region.split_regions(DOWN, 2)
        plot_region.scale_and_place(plot, buff=SMALL_BUFF)
        control_region.scale_and_place(controls)

        self.play(
            Write(title),
            FadeIn(heatmap),
            Write(contours),
            Write(trajectory),
            Write(frame),
            Write(markers),
        )
        self.add_foreground_mobjects(markers)
        self.play(FadeIn(controls))

        for alpha_value in ALPHA_SWEEP_VALUES:
            self.next_slide()
            self.play(alpha @ alpha_value, run_time=PARAMETER_SWEEP_RUN_TIME)

        for beta_value in BETA_SWEEP_VALUES:
            self.next_slide()
            self.play(beta @ beta_value, run_time=PARAMETER_SWEEP_RUN_TIME)

        self.next_slide()
        self.clear_scene()

    def _make_axes(self) -> Axes:
        return Axes(
            x_range=[X_RANGE[0], X_RANGE[1], 1],
            y_range=[Y_RANGE[0], Y_RANGE[1], 1],
            x_length=PLOT_X_LENGTH,
            y_length=PLOT_Y_LENGTH,
            tips=False,
            axis_config={
                "include_ticks": False,
                "stroke_opacity": 0,
            },
        )

    def _make_heatmap(self, axes: Axes) -> ImageMobject:
        width, height = HEATMAP_WIDTH, HEATMAP_HEIGHT
        x_values = np.linspace(*X_RANGE, width)
        y_values = np.linspace(Y_RANGE[1], Y_RANGE[0], height)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        values = np.log10(self._banana_value(np.stack([x_grid, y_grid])) + 1.0)
        image = ImageMobject(_blue_alpha_heatmap(_normalize_heat(values)))
        frame = _plot_frame(axes)
        image.stretch_to_fit_width(frame.width)
        image.stretch_to_fit_height(frame.height)
        image.move_to(frame)
        return image

    def _make_contours(self, axes: Axes) -> VGroup:
        x_values = np.linspace(*X_RANGE, ROSEN_CONTOUR_SAMPLES)
        y_values = np.linspace(*Y_RANGE, ROSEN_CONTOUR_SAMPLES)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        log_values = np.log10(self._banana_value(np.stack([x_grid, y_grid])) + 1.0)
        generator = contourpy.contour_generator(x=x_values, y=y_values, z=log_values)
        levels = np.linspace(*ROSEN_LEVEL_RANGE, ROSEN_LEVEL_COUNT)

        contours = VGroup()
        opacities = np.linspace(*ROSEN_CONTOUR_OPACITY_RANGE, len(levels))
        for opacity, level in zip(opacities, levels, strict=True):
            for segment in generator.lines(float(level)):
                if len(segment) < 2:
                    continue
                line = VMobject()
                line.set_points_as_corners(
                    _sample_contour_points(axes, segment, max_points=ROSEN_CONTOUR_MAX_POINTS)
                )
                line.set_stroke(
                    C_CONTOUR,
                    width=ROSEN_CONTOUR_STROKE_WIDTH,
                    opacity=float(opacity),
                )
                contours.add(line)
        return contours

    def _make_static_markers(self, axes: Axes) -> VGroup:
        marker_radius = _plot_frame(axes).height * ROSEN_MARKER_FRAME_HEIGHT_RATIO
        start = Dot(_axes_point(axes, START), color=X_0_COLOR, radius=marker_radius)
        optimum = Dot(_axes_point(axes, OPTIMUM), color=C_OPTIMUM, radius=marker_radius)
        optimum.set_stroke(
            C_OPTIMUM_STROKE,
            width=OPTIMUM_STROKE_WIDTH,
            opacity=OPTIMUM_STROKE_OPACITY,
        )
        optimum_halo = Circle(radius=marker_radius * OPTIMUM_HALO_DOT_SCALE, color=C_OPTIMUM)
        optimum_halo.set_fill(C_OPTIMUM, opacity=OPTIMUM_HALO_FILL_OPACITY).set_stroke(
            C_OPTIMUM_STROKE,
            width=OPTIMUM_HALO_STROKE_WIDTH,
            opacity=OPTIMUM_HALO_STROKE_OPACITY,
        )
        optimum_halo.move_to(optimum)

        start_label = label_for_dot(r"x_0", start, color=X_0_COLOR, direction=UR)
        optimum_label = label_for_dot(r"x^\star", optimum, color=C_OPTIMUM, direction=DR)

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
            SliderSpec(r"\alpha", *ALPHA_SLIDER_RANGE, ALPHA_DECIMALS, C_ALPHA),
        )
        beta_slider = self._slider(
            beta,
            SliderSpec(r"\beta", *BETA_SLIDER_RANGE, BETA_DECIMALS, C_BETA),
        )

        readout = self._make_path_readout(alpha, beta)
        sliders = VGroup(alpha_slider, beta_slider).arrange(RIGHT, buff=MED_LARGE_BUFF)
        controls = VGroup(equation, sliders, readout).arrange(RIGHT, buff=MED_LARGE_BUFF)
        panel = self._panel(controls)
        return VGroup(panel, controls)

    def _slider(self, tracker: VT, spec: SliderSpec) -> VGroup:
        theme = get_active_theme()
        return ValueSlider(
            tracker,
            spec,
            half_length=MOMENTUM_SLIDER_HALF_LENGTH,
            label_font_size=theme.typography.body,
            value_font_size=theme.typography.caption,
        )

    def _make_trajectory(self, axes: Axes, alpha: VT, beta: VT) -> VGroup:
        points = self._momentum_points(~alpha, ~beta)
        frame = _plot_frame(axes)
        step_dot_radius = frame.height * STEP_DOT_FRAME_HEIGHT_RATIO
        head_dot_radius = frame.height * HEAD_DOT_FRAME_HEIGHT_RATIO
        dots = VGroup(
            *(
                Dot(
                    _axes_point(axes, point),
                    color=C_PATH,
                    radius=step_dot_radius,
                )
                for point in points
            )
        )
        dots[0].set_color(X_0_COLOR).scale(START_DOT_SCALE)
        dots[-1].scale(HEAD_DOT_SCALE)

        self._attach_start_updater(dots[0], axes)
        for previous, dot in pairwise(dots):
            self._attach_step_updater(dot, previous, axes, alpha, beta)

        connectors = VGroup()
        for start, end in pairwise(dots):
            connector = Line(start.get_center(), end.get_center())
            connector.set_stroke(
                C_PATH,
                width=TRAJECTORY_STROKE_WIDTH,
                opacity=TRAJECTORY_STROKE_OPACITY,
            )
            connector.add_updater(
                lambda mob, start=start, end=end: self._update_connector(mob, start, end)
            )
            connectors.add(connector)

        head_ring = Circle(radius=head_dot_radius * HEAD_RING_DOT_SCALE, color=C_PATH)
        head_ring.set_stroke(C_PATH, width=HEAD_RING_STROKE_WIDTH)
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

        step_row = VGroup(Caption("steps"), steps).arrange(RIGHT, buff=SMALL_BUFF)
        distance_row = VGroup(Caption(r"$\|x_t-x^\star\|$"), distance).arrange(
            RIGHT,
            buff=SMALL_BUFF,
        )
        readout = VGroup(step_row, distance_row).arrange(DOWN, buff=SMALL_BUFF, aligned_edge=LEFT)
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
        return segmented_panel(mobject)

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
        dot.move_to(_axes_point(axes, START))
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
        dot.move_to(_axes_point(axes, point) if dot.momentum_visible else previous)
        dot.set_opacity(1 if dot.momentum_visible else 0)

    def _update_connector(self, connector: Line, start: Dot, end: Dot) -> None:
        visible = start.momentum_visible and end.momentum_visible
        if not visible:
            connector.set_opacity(0)
            return

        connector.set_points_as_corners([start.get_center(), end.get_center()])
        connector.set_stroke(
            C_PATH,
            width=TRAJECTORY_STROKE_WIDTH,
            opacity=TRAJECTORY_STROKE_OPACITY,
        )

    def _inside_plot(self, point: FloatArray) -> bool:
        return (
            X_RANGE[0] <= point[0] <= X_RANGE[1]
            and Y_RANGE[0] <= point[1] <= Y_RANGE[1]
        )
