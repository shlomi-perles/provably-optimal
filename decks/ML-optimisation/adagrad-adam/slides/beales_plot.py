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
    DOWN,
    LEFT,
    MED_SMALL_BUFF,
    RIGHT,
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

from simplex import Caption, ColorBar, ScalarFieldSurface, ThreeDSlide, get_active_theme

from slides.helpers.style import (
    C_BLUE,
    C_GREEN,
    C_MUTED,
    C_ORANGE,
    C_TEAL,
    C_TEXT,
    C_YELLOW,
    glass_panel,
    split_weighted,
    theme_math,
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
BEALE_START_POINT = np.array([1.6, 2.7], dtype=np.float64)
BEALE_LEARNING_RATE = 0.7
BEALE_MOMENTUM = 0.65
BEALE_STEPS = 3
FINITE_DIFFERENCE_STEP = 1e-2

DOMAIN_LIMIT = BEALE_LIMIT + BEALE_PAD_THICKNESS
PAD_HEIGHT = BEALE_Z_MAX * BEALE_SCALE
RAW_COLOR_FLOOR = 0.1
RAW_COLOR_CEILING_FACTOR = 0.8
RAW_COLOR_CEILING = BEALE_Z_MAX * RAW_COLOR_CEILING_FACTOR
LOG_COLOR_RANGE = (float(np.log10(RAW_COLOR_FLOOR)), float(np.log10(BEALE_Z_MAX)))

SURFACE_RESOLUTION = (BEALE_GRID_POINTS, BEALE_GRID_POINTS)
PATCH_RESOLUTION = (BEALE_PATCH_POINTS, 4 * BEALE_PATCH_POINTS)
SURFACE_COLORMAP = "Viridis_r"
PATCH_COLORMAP = "colorbrewer:YlOrBr"

MODEL_OPACITIES: tuple[float | None, ...] = (0.55, 0.75, 1.0)
MARKER_Z_LIFT = 0.24
MODEL_CAP_LIFT = 0.3
GRADIENT_DESCENT_MODEL_CAP_LIFT = 2.0
GRADIENT_DESCENT_MODEL_OPACITIES: tuple[float | None, ...] = (None,) * BEALE_STEPS
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
CAMERA_INITIAL_PHI = 54 * DEGREES
CAMERA_INITIAL_THETA = -46 * DEGREES
CAMERA_FINAL_PHI = 62 * DEGREES
CAMERA_FINAL_THETA = -38 * DEGREES
CAMERA_MOVE_RUN_TIME = 2.0
PLOT_HUD_WIDTH_RATIOS = (3.2, 1.0)
WORLD_CENTER_LIFT = UP * 0.08 + np.array([0.0, 0.0, -0.28])
COLOR_BAR_HEIGHT_TO_Z_AXIS = 0.9

PATH_DOT_RADIUS_TO_Z_AXIS = 1 / 34
PATH_DOT_RESOLUTION = (16, 16)
PATH_LINE_THICKNESS_TO_DOT_RADIUS = 0.34
HUD_Z_INDEX = 20


@dataclass(frozen=True, slots=True)
class ModelPatchSpec:
    center: FloatArray
    radius: float
    cap_height: float
    opacity: float | None

    @property
    def cap_delta(self) -> float:
        return 0.5 * self.radius**2 / BEALE_LEARNING_RATE


@dataclass(frozen=True, slots=True)
class BealePlotData:
    points: FloatArray
    visible_points: FloatArray
    point_heights: FloatArray
    visible_heights: FloatArray
    marker_heights: FloatArray
    label_xy: FloatArray
    label_heights: FloatArray
    patches: tuple[ModelPatchSpec, ...]

    @property
    def z_axis_max(self) -> float:
        patch_tops = tuple(patch.cap_height for patch in self.patches)
        return float(max(PAD_HEIGHT, self.label_heights.max(), self.marker_heights.max(), *patch_tops))


def beale_value(x: Scalar, y: Scalar) -> Scalar:
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


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
    height, _ = _displayed_beale_surface(np.asarray(point[0]), np.asarray(point[1]))
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
        point = point - BEALE_LEARNING_RATE * _gradient_at(point)
        points.append(point.copy())
    return np.asarray(points, dtype=np.float64)


def _make_plot_data(
    points: FloatArray,
    *,
    model_cap_lift: float,
    model_opacities: Sequence[float | None],
) -> BealePlotData:
    visible_points = points[:3]
    point_heights = np.asarray([_displayed_beale_height(point) for point in points])
    visible_heights = point_heights[:3]
    marker_heights = visible_heights + MARKER_Z_LIFT
    label_xy = visible_points + LABEL_XY_OFFSETS
    label_heights = visible_heights + LABEL_Z_OFFSETS

    patches: list[ModelPatchSpec] = []
    for opacity, (base_point, next_point), base_height in zip(
        model_opacities,
        pairwise(points),
        point_heights[:-1],
        strict=True,
    ):
        patches.append(
            ModelPatchSpec(
                center=next_point,
                radius=float(np.linalg.norm(next_point - base_point)),
                cap_height=float(base_height + model_cap_lift),
                opacity=opacity,
            )
        )

    return BealePlotData(
        points=points,
        visible_points=visible_points,
        point_heights=point_heights,
        visible_heights=visible_heights,
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
    update_equation = r"x_{t+1}=x_t-\eta\sum_{j=0}^{t}\gamma^j\nabla f(x_{t-j})"
    model_caption = "momentum models"
    model_bound = rf"Q_t(x)\le f(x_t)+{MODEL_CAP_LIFT:g}"
    label_texts = (r"x_{t-1}", r"x_t", r"x_{t+1}")
    model_cap_lift = MODEL_CAP_LIFT
    model_opacities = MODEL_OPACITIES

    def setup(self) -> None:
        super().setup()
        self.region = self.region.fix_in_frame()

    def construct(self) -> None:
        self.slide(title=self.slide_title)
        self.set_camera_orientation(phi=CAMERA_INITIAL_PHI, theta=CAMERA_INITIAL_THETA)

        data = self._make_plot_data()
        title = Title(self.slide_title)
        self.region.place(title, UP)
        self.region.update(top=title)
        plot_region, hud_region = split_weighted(self.region, PLOT_HUD_WIDTH_RATIOS)

        axes = self._make_axes(data)
        surface = self._make_surface(axes)
        patches = Group(*(self._make_patch(axes, spec) for spec in data.patches))
        path_lines, path_dots = self._make_path(axes, data)
        labels = self._make_labels(axes, data)
        axis_labels = self._make_axis_labels(axes, data)

        world = Group(axes, surface, patches, path_lines, path_dots, labels, axis_labels)
        plot_region.scale_and_place(world, buff=SMALL_BUFF)
        world.shift(WORLD_CENTER_LIFT)
        for label in (*labels, *axis_labels):
            fix_orientation = getattr(label, "fix_orientation", None)
            if callable(fix_orientation):
                fix_orientation()

        hud_panel, hud_mobjects = self._make_hud(axes, hud_region)
        for mob in (title, hud_panel, *hud_mobjects):
            set_z_index = getattr(mob, "set_z_index", None)
            if callable(set_z_index):
                set_z_index(HUD_Z_INDEX)
            fix_in_frame = getattr(mob, "fix_in_frame", None)
            if callable(fix_in_frame):
                fix_in_frame()

        self.play(
            Write(title),
            FadeIn(axes, axis_labels),
            Create(surface),
            FadeIn(hud_panel[0]),
            Write(hud_mobjects[0]),
        )
        self.fragment(title=self.sample_fragment_title)
        self.play(Create(path_lines), FadeIn(path_dots), Write(labels))
        self.fragment(title=self.model_fragment_title)
        self.play(Create(patches), Write(hud_mobjects[1]), Write(hud_mobjects[2]))
        self.move_camera(
            phi=CAMERA_FINAL_PHI,
            theta=CAMERA_FINAL_THETA,
            run_time=CAMERA_MOVE_RUN_TIME,
        )

    def _trajectory_points(self) -> FloatArray:
        return _simulate_beale_steps()

    def _make_plot_data(self) -> BealePlotData:
        return _make_plot_data(
            self._trajectory_points(),
            model_cap_lift=self.model_cap_lift,
            model_opacities=self.model_opacities,
        )

    def _make_axes(self, data: BealePlotData) -> ThreeDAxes:
        axes = ThreeDAxes(
            x_range=[-DOMAIN_LIMIT, DOMAIN_LIMIT, AXIS_TICK_STEP],
            y_range=[-DOMAIN_LIMIT, DOMAIN_LIMIT, AXIS_TICK_STEP],
            z_range=[0.0, data.z_axis_max, AXIS_Z_TICK_STEP],
            x_length=AXES_X_LENGTH,
            y_length=AXES_Y_LENGTH,
            z_length=AXES_Z_LENGTH,
            tips=False,
            axis_config={"stroke_color": C_MUTED, "stroke_width": AXIS_STROKE_WIDTH},
        )
        return axes

    def _make_surface(self, axes: ThreeDAxes) -> ScalarFieldSurface:
        return ScalarFieldSurface(
            lambda u, v: axes.c2p(u, v, _displayed_beale_height(np.array([u, v]))),
            u_range=[-DOMAIN_LIMIT, DOMAIN_LIMIT],
            v_range=[-DOMAIN_LIMIT, DOMAIN_LIMIT],
            resolution=SURFACE_RESOLUTION,
            color_func=_displayed_beale_color,
            colormap=SURFACE_COLORMAP,
            color_range=LOG_COLOR_RANGE,
        )

    def _make_patch(self, axes: ThreeDAxes, spec: ModelPatchSpec) -> ScalarFieldSurface:
        def patch_point(radius: float, angle: float) -> FloatArray:
            xy = spec.center + radius * np.array([np.cos(angle), np.sin(angle)])
            height = spec.cap_height - spec.cap_delta + 0.5 * radius**2 / BEALE_LEARNING_RATE
            return axes.c2p(float(xy[0]), float(xy[1]), float(height))

        surface_kwargs = {}
        if spec.opacity is not None:
            surface_kwargs["opacity"] = spec.opacity

        return ScalarFieldSurface(
            patch_point,
            u_range=[0.0, spec.radius],
            v_range=[0.0, TAU],
            resolution=PATCH_RESOLUTION,
            color_func="height",
            colormap=PATCH_COLORMAP,
            **surface_kwargs,
        )

    def _make_path(self, axes: ThreeDAxes, data: BealePlotData) -> tuple[Group, Group]:
        path_points = [
            axes.c2p(float(point[0]), float(point[1]), float(height))
            for point, height in zip(data.visible_points, data.marker_heights, strict=True)
        ]
        dot_radius = axes.z_axis.get_length() * PATH_DOT_RADIUS_TO_Z_AXIS
        line_thickness = dot_radius * PATH_LINE_THICKNESS_TO_DOT_RADIUS
        dots = Group(
            *(
                Sphere(
                    center=point,
                    color=C_GREEN,
                    radius=dot_radius,
                    resolution=PATH_DOT_RESOLUTION,
                )
                for point in path_points
            )
        )
        lines = Group(
            *(
                Line3D(start, end, color=C_GREEN, thickness=line_thickness)
                for start, end in pairwise(path_points)
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

    def _make_hud(self, axes: ThreeDAxes, hud_region: object) -> tuple[VGroup, tuple[VGroup, ...]]:
        theme = get_active_theme()
        equation = VGroup(
            Caption("soft-clipped Beale landscape"),
            theme_math(
                r"f(x,y)=\sum_{k=1}^{3}\left(a_k-x+xy^k\right)^2",
                color=C_TEXT,
                typography="caption",
            ),
            theme_math(
                self.update_equation,
                color=C_TEXT,
                typography="caption",
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
        for mob in equation[1:]:
            mob.set_color_by_tex(r"\eta", C_YELLOW)
            mob.set_color_by_tex(r"\gamma", C_TEAL)
            mob.set_color_by_tex(r"\nabla", C_ORANGE)

        bar = ColorBar(
            colormap=SURFACE_COLORMAP,
            min_value=LOG_COLOR_RANGE[0],
            max_value=LOG_COLOR_RANGE[1],
            n_labels=4,
            height=axes.z_axis.get_length() * COLOR_BAR_HEIGHT_TO_Z_AXIS,
            font_size=theme.typography.caption,
        )
        bar_label = Caption(r"$\log_{10}(f)$")
        color_legend = VGroup(bar, bar_label).arrange(DOWN, buff=SMALL_BUFF)
        color_legend.align_to(equation, LEFT)

        patch_legend = VGroup(
            Caption(self.model_caption),
            theme_math(self.model_bound, color=C_TEXT, typography="caption"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
        patch_legend[1].set_color_by_tex(r"Q_t", C_ORANGE)

        hud = VGroup(equation, color_legend, patch_legend).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=MED_SMALL_BUFF,
        )
        panel = glass_panel(hud)
        hud_region.scale_and_place(panel, buff=SMALL_BUFF)
        return panel, (equation, color_legend, patch_legend)


class BealesGradientDescentPlot(BealesPlot):
    """Beale surface, gradient-descent samples, and capped quadratic models."""

    slide_title = "Gradient Descent on Beale's Function"
    sample_fragment_title = "Gradient descent samples"
    update_equation = r"x_{t+1}=x_t-\eta\nabla f(x_t)"
    model_caption = "gradient descent models"
    model_bound = rf"Q_t(x)\le f(x_t)+{GRADIENT_DESCENT_MODEL_CAP_LIFT:g}"
    label_texts = (r"x_t", r"x_{t+1}", r"x_{t+2}")
    model_cap_lift = GRADIENT_DESCENT_MODEL_CAP_LIFT
    model_opacities = GRADIENT_DESCENT_MODEL_OPACITIES

    def _trajectory_points(self) -> FloatArray:
        return _simulate_gradient_descent_steps()
