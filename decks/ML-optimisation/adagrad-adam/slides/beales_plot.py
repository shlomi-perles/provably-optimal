"""OpenGL Beale surface matching the source Plotly figure numerically."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from manim import (
    BLACK,
    DEGREES,
    SMALL_BUFF,
    TAU,
    UP,
    Create,
    FadeIn,
    Group,
    Line3D,
    MathTex,
    Sphere,
    ThreeDAxes,
    Title,
    VGroup,
    Write,
    config,
)

config.renderer = "opengl"
config.write_to_movie = True

from simplex import ScalarFieldSurface, ThreeDSlide, get_active_theme

from slides.helpers.style import (
    C_GREEN,
    C_MUTED,
    C_ORANGE,
)

FloatArray = npt.NDArray[np.float64]
Scalar = float | FloatArray

BEALE_LIMIT = 4.5
BEALE_PAD_THICKNESS = 2.5
BEALE_SCALE = 0.2
BEALE_PUSH_STRENGTH = 0.25
BEALE_Z_MAX = 20.0
BEALE_GRID_POINTS = 250
BEALE_PATCH_POINTS = 96
BEALE_START_POINT = np.array([1.1, 2.2], dtype=np.float64)
BEALE_LEARNING_RATE = 0.7
GRADIENT_DESCENT_LEARNING_RATE = BEALE_LEARNING_RATE / 2
BEALE_MOMENTUM = 0.65
BEALE_STEPS = 3
FINITE_DIFFERENCE_STEP = 1e-2
ADAGRAD_START_POINT = np.array([0.36, -4.7], dtype=np.float64)
ADAGRAD_LEARNING_RATE = 0.9
ADAGRAD_INITIAL_ACCUMULATOR = 0.1
ADAGRAD_WARMUP_STEPS = 2

DOMAIN_LIMIT = BEALE_LIMIT + BEALE_PAD_THICKNESS
PAD_HEIGHT = BEALE_Z_MAX * BEALE_SCALE
RAW_COLOR_FLOOR = 0.1
RAW_COLOR_CEILING_FACTOR = 0.8
RAW_COLOR_CEILING = BEALE_Z_MAX * RAW_COLOR_CEILING_FACTOR
LOG_COLOR_RANGE = (float(np.log10(RAW_COLOR_FLOOR)), float(np.log10(BEALE_Z_MAX)))

SURFACE_RESOLUTION = (BEALE_GRID_POINTS, BEALE_GRID_POINTS)
PATCH_RESOLUTION = (BEALE_PATCH_POINTS, 4 * BEALE_PATCH_POINTS)
GRADIENT_DESCENT_RESOLUTION_SCALE = 1 / 2
GRADIENT_DESCENT_PATCH_RESOLUTION = tuple(
    round(points * GRADIENT_DESCENT_RESOLUTION_SCALE) for points in PATCH_RESOLUTION
)
SURFACE_COLORMAP = "Viridis_r"
PATCH_COLORMAP = "colorbrewer:YlOrBr"

MODEL_OPACITIES: tuple[float | None, ...] = (0.55, 0.75, 1.0)
MARKER_Z_LIFT = 0.24
MODEL_CAP_LIFT = 0.3
GRADIENT_DESCENT_MODEL_OPACITIES: tuple[float | None, ...] = (None,) * BEALE_STEPS
ADAGRAD_MODEL_CAP_LIFT = 2.4
ADAGRAD_MODEL_OPACITIES: tuple[float | None, ...] = (None,)
LABEL_XY_OFFSETS = np.array(
    [
        [-0.55, 0.55],
        [0.65, 0.25],
        [0.55, -0.45],
    ],
    dtype=np.float64,
)
LABEL_Z_LIFT_TO_PAD_HEIGHT = 0.26
LABEL_Z_OFFSETS = np.full(3, PAD_HEIGHT * LABEL_Z_LIFT_TO_PAD_HEIGHT, dtype=np.float64)

AXES_X_LENGTH = 6.4
AXES_Y_LENGTH = AXES_X_LENGTH
AXES_Z_LENGTH = 2.35
AXIS_TICK_STEP = 2.0
AXIS_Z_TICK_STEP = 1.0
AXIS_STROKE_WIDTH = 2.0
AXIS_LABEL_INSET_TO_DOMAIN = 0.16
AXIS_LABEL_SIDE_TO_DOMAIN = 0.42
AXIS_Z_LABEL_HEIGHT_TO_RANGE = 0.68
PLOTLY_TO_MANIM_THETA_OFFSET = 90 * DEGREES
CAMERA_AZIMUTH_FLIP = 180 * DEGREES
CAMERA_Z_AXIS_ROTATION = 180 * DEGREES
CAMERA_INITIAL_PLOTLY_AZIMUTH = -67.4 * DEGREES
CAMERA_INITIAL_PLOTLY_ELEVATION = 37.5 * DEGREES
CAMERA_FINAL_PLOTLY_AZIMUTH = -28.5 * DEGREES
CAMERA_FINAL_PLOTLY_ELEVATION = 27.7 * DEGREES
CAMERA_INITIAL_PHI = 90 * DEGREES - CAMERA_INITIAL_PLOTLY_ELEVATION
CAMERA_INITIAL_THETA = (
    CAMERA_INITIAL_PLOTLY_AZIMUTH
    + PLOTLY_TO_MANIM_THETA_OFFSET
    + CAMERA_AZIMUTH_FLIP
    + CAMERA_Z_AXIS_ROTATION
)
CAMERA_FINAL_PHI = 90 * DEGREES - CAMERA_FINAL_PLOTLY_ELEVATION
CAMERA_FINAL_THETA = (
    CAMERA_FINAL_PLOTLY_AZIMUTH
    + PLOTLY_TO_MANIM_THETA_OFFSET
    + CAMERA_AZIMUTH_FLIP
    + CAMERA_Z_AXIS_ROTATION
)
ADAGRAD_CAMERA_INITIAL_PHI = CAMERA_INITIAL_PHI
ADAGRAD_CAMERA_INITIAL_THETA = CAMERA_INITIAL_THETA
ADAGRAD_CAMERA_FINAL_PHI = CAMERA_FINAL_PHI
ADAGRAD_CAMERA_FINAL_THETA = CAMERA_FINAL_THETA
CAMERA_MOVE_RUN_TIME = 2.0

PATH_DOT_RADIUS_TO_Z_AXIS = 1 / (4 * 34)
PATH_DOT_RESOLUTION = (16, 16)
PATH_LINE_THICKNESS_TO_DOT_RADIUS = 0.34
PATH_LINE_SAMPLES = round(BEALE_GRID_POINTS / 10)


@dataclass(frozen=True, slots=True)
class ModelPatchSpec:
    center: FloatArray
    radius: float
    min_height: float
    cap_height: float
    learning_rate: float
    opacity: float | None

    @property
    def cap_delta(self) -> float:
        return self.cap_height - self.min_height


@dataclass(frozen=True, slots=True)
class BealePlotData:
    visible_points: FloatArray
    marker_heights: FloatArray
    label_xy: FloatArray
    label_heights: FloatArray
    patches: tuple[ModelPatchSpec, ...]

    @property
    def z_axis_max(self) -> float:
        patch_tops = tuple(patch.cap_height for patch in self.patches)
        return float(
            max(PAD_HEIGHT, self.label_heights.max(), self.marker_heights.max(), *patch_tops)
        )


def beale_value(x: Scalar, y: Scalar) -> Scalar:
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y**2) ** 2 + (2.625 - x + x * y**3) ** 2


def _radially_pushed_coordinates(x: Scalar, y: Scalar) -> tuple[Scalar, Scalar]:
    radius = np.sqrt(x**2 + y**2)
    factor = 1.0 + BEALE_PUSH_STRENGTH * radius
    return x / factor, y / factor


def _smoothstep(alpha: Scalar) -> Scalar:
    return 3.0 * alpha**2 - 2.0 * alpha**3


def _square_pad_alpha(x: Scalar, y: Scalar) -> Scalar:
    dx = np.maximum(np.abs(x) - BEALE_LIMIT, 0.0)
    dy = np.maximum(np.abs(y) - BEALE_LIMIT, 0.0)
    distance_from_core = np.maximum(dx, dy)
    return _smoothstep(np.clip(distance_from_core / BEALE_PAD_THICKNESS, 0.0, 1.0))


def _displayed_beale_surface(x: Scalar, y: Scalar) -> tuple[Scalar, Scalar]:
    x_eval, y_eval = _radially_pushed_coordinates(x, y)
    raw = beale_value(x_eval, y_eval)
    soft_cut = BEALE_Z_MAX * np.tanh(raw / BEALE_Z_MAX) * BEALE_SCALE
    alpha = _square_pad_alpha(x, y)

    height = (1.0 - alpha) * soft_cut + alpha * PAD_HEIGHT
    color = np.log10(np.clip(raw, a_min=RAW_COLOR_FLOOR, a_max=RAW_COLOR_CEILING))
    return height, (1.0 - alpha) * color + alpha * np.log10(BEALE_Z_MAX)


def _displayed_beale_height(point: FloatArray) -> float:
    height, _ = _displayed_beale_surface(point[0], point[1])
    return float(height)


def _displayed_beale_color(x: Scalar, y: Scalar) -> Scalar:
    _, color = _displayed_beale_surface(x, y)
    return color


def _gradient_at(point: FloatArray) -> FloatArray:
    x, y = point

    def height(offset_x: float, offset_y: float) -> float:
        return _displayed_beale_height(np.array([x + offset_x, y + offset_y], dtype=np.float64))

    return np.array(
        [
            (height(FINITE_DIFFERENCE_STEP, 0.0) - height(-FINITE_DIFFERENCE_STEP, 0.0))
            / (2.0 * FINITE_DIFFERENCE_STEP),
            (height(0.0, FINITE_DIFFERENCE_STEP) - height(0.0, -FINITE_DIFFERENCE_STEP))
            / (2.0 * FINITE_DIFFERENCE_STEP),
        ],
        dtype=np.float64,
    )


def _simulate_beale_steps() -> FloatArray:
    point = BEALE_START_POINT.copy()
    velocity = np.zeros(2, dtype=np.float64)
    points = [point.copy()]
    for _ in range(BEALE_STEPS):
        velocity = BEALE_MOMENTUM * velocity + _gradient_at(point)
        point = point - BEALE_LEARNING_RATE * velocity
        points.append(point.copy())
    return np.asarray(points, dtype=np.float64)


def _simulate_gradient_descent_steps() -> FloatArray:
    point = BEALE_START_POINT.copy()
    points = [point.copy()]
    for _ in range(BEALE_STEPS):
        point = point - GRADIENT_DESCENT_LEARNING_RATE * _gradient_at(point)
        points.append(point.copy())
    return np.asarray(points, dtype=np.float64)


def _simulate_adagrad_steps() -> tuple[FloatArray, list[FloatArray]]:
    point = ADAGRAD_START_POINT.copy()
    accumulator = np.full(2, ADAGRAD_INITIAL_ACCUMULATOR, dtype=np.float64)
    points = [point.copy()]
    accumulators = [accumulator.copy()]
    for _ in range(ADAGRAD_WARMUP_STEPS + 1):
        gradient = _gradient_at(point)
        accumulator = accumulator + gradient**2
        point = point - ADAGRAD_LEARNING_RATE * gradient / np.sqrt(accumulator)
        points.append(point.copy())
        accumulators.append(accumulator.copy())
    return np.asarray(points, dtype=np.float64), accumulators


@dataclass(frozen=True, slots=True)
class AdaGradModel:
    step_points: FloatArray
    base_height: float
    gradient: FloatArray
    preconditioner: FloatArray
    minimum_height: float
    cap_height: float


def _make_adagrad_model() -> AdaGradModel:
    points, accumulators = _simulate_adagrad_steps()
    base_index = ADAGRAD_WARMUP_STEPS
    base_point = points[base_index]
    base_height = _displayed_beale_height(base_point)
    gradient = _gradient_at(base_point)
    preconditioner = np.sqrt(accumulators[base_index + 1])
    minimizer = base_point - ADAGRAD_LEARNING_RATE * gradient / preconditioner
    minimum_height = float(
        base_height - 0.5 * ADAGRAD_LEARNING_RATE * np.sum(gradient**2 / preconditioner)
    )
    return AdaGradModel(
        step_points=np.asarray([base_point, minimizer], dtype=np.float64),
        base_height=base_height,
        gradient=gradient,
        preconditioner=preconditioner,
        minimum_height=minimum_height,
        cap_height=base_height + ADAGRAD_MODEL_CAP_LIFT,
    )


def _make_plot_data(
    points: FloatArray,
    *,
    model_cap_lift: float,
    model_opacities: Sequence[float | None],
    model_learning_rate: float,
    visible_start: int = 0,
    constant_cap_delta: float | None = None,
    patch_cap_deltas: Sequence[float] | None = None,
) -> BealePlotData:
    point_heights = np.asarray([_displayed_beale_height(point) for point in points])
    visible_stop = min(len(points), visible_start + len(LABEL_XY_OFFSETS))
    visible_points = points[visible_start:visible_stop]
    visible_heights = point_heights[visible_start:visible_stop]
    marker_heights = visible_heights + MARKER_Z_LIFT
    label_xy = visible_points + LABEL_XY_OFFSETS[: len(visible_points)]
    label_heights = visible_heights + LABEL_Z_OFFSETS[: len(visible_points)]

    patches: list[ModelPatchSpec] = []
    if patch_cap_deltas is not None and len(patch_cap_deltas) != len(model_opacities):
        raise ValueError("patch_cap_deltas must match model_opacities")

    for patch_index, (opacity, (base_point, next_point), base_height, next_height) in enumerate(
        zip(
            model_opacities,
            pairwise(points),
            point_heights[:-1],
            point_heights[1:],
            strict=True,
        )
    ):
        if constant_cap_delta is None:
            cap_height = float(base_height + model_cap_lift)
            if patch_cap_deltas is None:
                radius = float(np.linalg.norm(next_point - base_point))
                cap_delta = 0.5 * radius**2 / model_learning_rate
            else:
                cap_delta = float(patch_cap_deltas[patch_index])
                radius = float(np.sqrt(2.0 * model_learning_rate * cap_delta))
            min_height = cap_height - cap_delta
        else:
            cap_delta = float(constant_cap_delta)
            radius = float(np.sqrt(2.0 * model_learning_rate * cap_delta))
            min_height = float(next_height)
            cap_height = min_height + cap_delta

        patches.append(
            ModelPatchSpec(
                center=next_point,
                radius=radius,
                min_height=min_height,
                cap_height=cap_height,
                learning_rate=model_learning_rate,
                opacity=opacity,
            )
        )

    return BealePlotData(
        visible_points=visible_points,
        marker_heights=marker_heights,
        label_xy=label_xy,
        label_heights=label_heights,
        patches=tuple(patches),
    )


class BealesPlot(ThreeDSlide):
    """Beale surface, momentum samples, and capped quadratic models."""

    slide_title = "Beale's Function"
    sample_fragment_title = "Momentum samples"
    model_fragment_title = "Capped quadratic models"
    label_texts = (r"x_{t-1}", r"x_t", r"x_{t+1}")
    model_cap_lift = MODEL_CAP_LIFT
    model_opacities = MODEL_OPACITIES
    model_learning_rate = BEALE_LEARNING_RATE
    visible_start = 0
    surface_resolution = SURFACE_RESOLUTION
    patch_resolution = PATCH_RESOLUTION
    path_color = C_GREEN
    camera_initial_phi = CAMERA_INITIAL_PHI
    camera_initial_theta = CAMERA_INITIAL_THETA
    camera_final_phi = CAMERA_FINAL_PHI
    camera_final_theta = CAMERA_FINAL_THETA

    def setup(self) -> None:
        super().setup()
        self.region = self.region.fix_in_frame()

    def construct(self) -> None:
        self.set_camera_orientation(phi=self.camera_initial_phi, theta=self.camera_initial_theta)

        data = self._make_plot_data()
        title = Title(self.slide_title)
        self.region.place(title, UP)
        self.region.update(top=title)
        title.fix_in_frame()

        axes = self._make_axes(data)
        surface = self._make_surface(axes)
        patches = Group(*(self._make_patch(axes, spec) for spec in data.patches))
        path_lines, path_dots = self._make_path(axes, data)
        labels = self._make_labels(axes, data)
        axis_labels = self._make_axis_labels(axes, data)

        world = Group(axes, surface, patches, path_lines, path_dots, labels, axis_labels)
        self.region.scale_and_place(world, buff=SMALL_BUFF)
        for label in (*labels, *axis_labels):
            label.fix_orientation()

        self.play(
            FadeIn(title),
            FadeIn(axes, axis_labels),
            Create(surface),
        )
        self.wait(0.4)
        self.play(Create(path_lines), FadeIn(path_dots), Write(labels))
        self.wait(0.4)
        self.next_slide(title=self.model_fragment_title)
        self.play(Create(patches))
        self._rotate_camera()
        self.wait(self.cue_boundary_wait_time)
        self.wait(0.4)
        self.next_slide(title="Rotated view")
        self.wait(0.4)
        self.next_slide()
        self.clear_scene()

    def _trajectory_points(self) -> FloatArray:
        return _simulate_beale_steps()

    def _rotate_camera(self) -> None:
        theta_rate = (self.camera_final_theta - self.camera_initial_theta) / CAMERA_MOVE_RUN_TIME
        self.begin_ambient_camera_rotation(rate=theta_rate)
        self.wait(CAMERA_MOVE_RUN_TIME)
        self.stop_ambient_camera_rotation()

    def _make_plot_data(self) -> BealePlotData:
        return _make_plot_data(
            self._trajectory_points(),
            model_cap_lift=self.model_cap_lift,
            model_opacities=self.model_opacities,
            model_learning_rate=self.model_learning_rate,
            visible_start=self.visible_start,
        )

    def _make_axes(self, data: BealePlotData) -> ThreeDAxes:
        return ThreeDAxes(
            x_range=[-DOMAIN_LIMIT, DOMAIN_LIMIT, AXIS_TICK_STEP],
            y_range=[-DOMAIN_LIMIT, DOMAIN_LIMIT, AXIS_TICK_STEP],
            z_range=[0.0, data.z_axis_max, AXIS_Z_TICK_STEP],
            x_length=AXES_X_LENGTH,
            y_length=AXES_Y_LENGTH,
            z_length=AXES_Z_LENGTH,
            tips=False,
            axis_config={"stroke_color": C_MUTED, "stroke_width": AXIS_STROKE_WIDTH},
        )

    def _make_surface(self, axes: ThreeDAxes) -> ScalarFieldSurface:
        return ScalarFieldSurface(
            lambda u, v: axes.c2p(u, v, _displayed_beale_height(np.array([u, v]))),
            u_range=[-DOMAIN_LIMIT, DOMAIN_LIMIT],
            v_range=[-DOMAIN_LIMIT, DOMAIN_LIMIT],
            resolution=self.surface_resolution,
            color_func=_displayed_beale_color,
            colormap=SURFACE_COLORMAP,
            color_range=LOG_COLOR_RANGE,
        )

    def _make_patch(self, axes: ThreeDAxes, spec: ModelPatchSpec) -> ScalarFieldSurface:
        def patch_point(radius: float, angle: float) -> FloatArray:
            xy = spec.center + radius * np.array([np.cos(angle), np.sin(angle)])
            height = spec.min_height + 0.5 * radius**2 / spec.learning_rate
            return axes.c2p(float(xy[0]), float(xy[1]), float(height))

        surface_kwargs = {}
        if spec.opacity is not None:
            surface_kwargs["opacity"] = spec.opacity

        return ScalarFieldSurface(
            patch_point,
            u_range=[0.0, spec.radius],
            v_range=[0.0, TAU],
            resolution=self.patch_resolution,
            color_func="height",
            colormap=PATCH_COLORMAP,
            **surface_kwargs,
        )

    def _make_path(self, axes: ThreeDAxes, data: BealePlotData) -> tuple[Group, Group]:
        def to_axes_point(point: FloatArray, height: float) -> FloatArray:
            return axes.c2p(float(point[0]), float(point[1]), float(height))

        path_points = [
            to_axes_point(point, height)
            for point, height in zip(data.visible_points, data.marker_heights, strict=True)
        ]
        path_line_points: list[FloatArray] = []
        for start, end in pairwise(data.visible_points):
            for alpha in np.linspace(0.0, 1.0, PATH_LINE_SAMPLES):
                point = (1.0 - alpha) * start + alpha * end
                path_line_points.append(
                    to_axes_point(point, _displayed_beale_height(point) + MARKER_Z_LIFT)
                )

        dot_radius = axes.z_axis.get_length() * PATH_DOT_RADIUS_TO_Z_AXIS
        line_thickness = dot_radius * PATH_LINE_THICKNESS_TO_DOT_RADIUS
        dots = Group(
            *(
                Sphere(
                    center=point,
                    color=self.path_color,
                    radius=dot_radius,
                    resolution=PATH_DOT_RESOLUTION,
                )
                for point in path_points
            )
        )
        lines = Group(
            *(
                Line3D(start, end, color=self.path_color, thickness=line_thickness)
                for start, end in pairwise(path_line_points)
            )
        )
        return lines, dots

    def _make_labels(self, axes: ThreeDAxes, data: BealePlotData) -> VGroup:
        theme = get_active_theme()
        return VGroup(
            *(
                MathTex(label, color=BLACK, font_size=theme.typography.caption).move_to(
                    axes.c2p(float(point[0]), float(point[1]), float(height))
                )
                for label, point, height in zip(
                    self.label_texts,
                    data.label_xy,
                    data.label_heights,
                    strict=True,
                )
            )
        )

    def _make_axis_labels(self, axes: ThreeDAxes, data: BealePlotData) -> VGroup:
        label_font_size = get_active_theme().typography.caption * 0.82
        inset = DOMAIN_LIMIT * AXIS_LABEL_INSET_TO_DOMAIN
        side = DOMAIN_LIMIT * AXIS_LABEL_SIDE_TO_DOMAIN
        return VGroup(
            MathTex("X", color=C_MUTED, font_size=label_font_size).move_to(
                axes.c2p(side, -DOMAIN_LIMIT + inset, 0.0)
            ),
            MathTex("Y", color=C_MUTED, font_size=label_font_size).move_to(
                axes.c2p(-DOMAIN_LIMIT + inset, side, 0.0)
            ),
            MathTex(r"f(X,Y)", color=C_MUTED, font_size=label_font_size).move_to(
                axes.c2p(
                    -DOMAIN_LIMIT + inset,
                    -DOMAIN_LIMIT + inset,
                    data.z_axis_max * AXIS_Z_LABEL_HEIGHT_TO_RANGE,
                )
            ),
        )


class BealesGradientDescentPlot(BealesPlot):
    """Beale surface, gradient-descent samples, and capped quadratic models."""

    slide_title = "Gradient Descent on Beale's Function"
    sample_fragment_title = "Gradient descent samples"
    label_texts = (r"x_t", r"x_{t+1}", r"x_{t+2}")
    model_cap_lift = MODEL_CAP_LIFT
    model_opacities = GRADIENT_DESCENT_MODEL_OPACITIES
    model_learning_rate = GRADIENT_DESCENT_LEARNING_RATE
    visible_start = 1
    patch_resolution = GRADIENT_DESCENT_PATCH_RESOLUTION

    def _trajectory_points(self) -> FloatArray:
        return _simulate_gradient_descent_steps()

    def _make_plot_data(self) -> BealePlotData:
        points = self._trajectory_points()
        point_heights = np.asarray([_displayed_beale_height(point) for point in points])
        cap_delta = float(point_heights[0] - point_heights[1])
        return _make_plot_data(
            points,
            model_cap_lift=self.model_cap_lift,
            model_opacities=self.model_opacities,
            model_learning_rate=self.model_learning_rate,
            visible_start=self.visible_start,
            constant_cap_delta=cap_delta,
        )


class BealesAdaGradPlot(BealesPlot):
    """One AdaGrad step on Beale, with a diagonal capped quadratic model."""

    slide_title = "AdaGrad Step on Beale's Function"
    sample_fragment_title = "One AdaGrad step"
    model_fragment_title = "Diagonal capped model"
    label_texts = (r"x_t", r"x_{t+1}")
    model_cap_lift = ADAGRAD_MODEL_CAP_LIFT
    model_opacities = ADAGRAD_MODEL_OPACITIES
    model_learning_rate = ADAGRAD_LEARNING_RATE
    patch_resolution = GRADIENT_DESCENT_PATCH_RESOLUTION
    path_color = C_ORANGE
    camera_initial_phi = ADAGRAD_CAMERA_INITIAL_PHI
    camera_initial_theta = ADAGRAD_CAMERA_INITIAL_THETA
    camera_final_phi = ADAGRAD_CAMERA_FINAL_PHI
    camera_final_theta = ADAGRAD_CAMERA_FINAL_THETA

    def _trajectory_points(self) -> FloatArray:
        return _make_adagrad_model().step_points

    def _make_plot_data(self) -> BealePlotData:
        model = _make_adagrad_model()
        cap_delta = model.cap_height - model.minimum_height
        return _make_plot_data(
            model.step_points,
            model_cap_lift=self.model_cap_lift,
            model_opacities=self.model_opacities,
            model_learning_rate=self.model_learning_rate,
            patch_cap_deltas=(cap_delta,),
        )

    def _make_patch(self, axes: ThreeDAxes, spec: ModelPatchSpec) -> ScalarFieldSurface:
        model = _make_adagrad_model()
        inverse_sqrt_preconditioner = np.divide(1.0, np.sqrt(model.preconditioner))

        def patch_point(local_radius: float, angle: float) -> FloatArray:
            local = local_radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            xy = spec.center + inverse_sqrt_preconditioner * local
            height = spec.min_height + 0.5 * local_radius**2 / spec.learning_rate
            return axes.c2p(float(xy[0]), float(xy[1]), float(height))

        return ScalarFieldSurface(
            patch_point,
            u_range=[0.0, spec.radius],
            v_range=[0.0, TAU],
            resolution=self.patch_resolution,
            color_func="height",
            colormap=PATCH_COLORMAP,
        )
