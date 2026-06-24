"""OpenGL slide for AdaGrad's local diagonal quadratic model."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from manim import (
    DEGREES,
    DOWN,
    LEFT,
    MED_SMALL_BUFF,
    RIGHT,
    SMALL_BUFF,
    TAU,
    UP,
    Create,
    Dot3D,
    FadeIn,
    Group,
    MathTex,
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
    C_ORANGE,
    C_TEAL,
    C_TEXT,
    C_YELLOW,
    glass_panel as _glass_panel,
    theme_math,
)

FloatArray = npt.NDArray[np.float64]

SURFACE_QUADRATIC_CENTER = np.array([-0.70, -0.60], dtype=np.float64)
SURFACE_QUADRATIC_WEIGHTS = np.array([0.05, 0.10], dtype=np.float64)
SURFACE_EXPONENTIAL_WEIGHTS = np.array([0.05, 0.05, 0.025], dtype=np.float64)
SURFACE_EXPONENTIAL_RATES = np.array(
    [
        [2.20, 0.25],
        [-0.90, 2.20],
        [1.20, 1.65],
    ],
    dtype=np.float64,
)
INITIAL_CAMERA_PHI = 63 * DEGREES
INITIAL_CAMERA_THETA = -52 * DEGREES
FINAL_CAMERA_PHI = 69 * DEGREES
FINAL_CAMERA_THETA = -35 * DEGREES
CAMERA_MOVE_RUN_TIME = 2.0
AXES_X_RANGE = (-1.3, 1.2, 1.0)
AXES_Y_RANGE = (-1.0, 1.35, 1.0)
AXES_Z_RANGE = (0.0, 2.6, 1.0)
AXES_X_LENGTH = 3.4
AXES_Y_LENGTH = AXES_X_LENGTH
AXES_Z_LENGTH = 1.8
SURFACE_U_RANGE = (-1.20, 1.10)
SURFACE_V_RANGE = (-0.95, 1.25)
SURFACE_RESOLUTION = (46, 46)
MODEL_PATCH_HALF_SIDE = 0.72
MODEL_PATCH_RADIUS = MODEL_PATCH_HALF_SIDE * np.sqrt(2)
MODEL_PATCH_RADIAL_RANGE = (0.0, MODEL_PATCH_RADIUS)
MODEL_PATCH_ANGLE_RANGE = (0.0, TAU)
MODEL_PATCH_BASE_RESOLUTION = 32
MODEL_PATCH_RESOLUTION = (MODEL_PATCH_BASE_RESOLUTION, 2 * MODEL_PATCH_BASE_RESOLUTION)
CURRENT_POINT = np.array([-0.35, 1.10], dtype=np.float64)
CURRENT_DOT_RADIUS_TO_Z_AXIS = 1 / 32
CURRENT_DOT_Z_OFFSET_RATIO = 0.03
CURRENT_LABEL_XY_OFFSET_RATIO = 0.035
CURRENT_LABEL_Z_OFFSET_RATIO = 0.12
MODEL_PATCH_OPACITY = 0.95
SURFACE_OPACITY = 0.82
COLOR_BAR_HEIGHT_TO_AXES_Z = 0.92
COLOR_BAR_LABEL_COUNT = 3
WORLD_SCALE = 0.82
WORLD_SHIFT = LEFT * 2.55 + UP * 0.12 + np.array([0.0, 0.0, -0.75])
HUD_EQUATION_LIFT = 0.55
HUD_Z_INDEX = 20


def _surface_value(x: float, y: float) -> float:
    point = np.array([x, y], dtype=np.float64)
    offset = point - SURFACE_QUADRATIC_CENTER
    quadratic = float(np.sum(SURFACE_QUADRATIC_WEIGHTS * offset**2))
    exponential = float(
        np.sum(SURFACE_EXPONENTIAL_WEIGHTS * np.exp(SURFACE_EXPONENTIAL_RATES @ point))
    )
    return quadratic + exponential


def _gradient_at(point: FloatArray) -> FloatArray:
    offset = point - SURFACE_QUADRATIC_CENTER
    quadratic_gradient = 2 * SURFACE_QUADRATIC_WEIGHTS * offset
    weighted_exponentials = SURFACE_EXPONENTIAL_WEIGHTS * np.exp(
        SURFACE_EXPONENTIAL_RATES @ point
    )
    return quadratic_gradient + weighted_exponentials @ SURFACE_EXPONENTIAL_RATES


def _hessian_at(point: FloatArray) -> FloatArray:
    hessian = np.diag(2 * SURFACE_QUADRATIC_WEIGHTS)
    weighted_exponentials = SURFACE_EXPONENTIAL_WEIGHTS * np.exp(
        SURFACE_EXPONENTIAL_RATES @ point
    )
    for weight, rate in zip(weighted_exponentials, SURFACE_EXPONENTIAL_RATES, strict=True):
        hessian += weight * np.outer(rate, rate)
    return hessian


class AdaGradLocalSqueeze(ThreeDSlide):
    """A local diagonal Hessian patch over a nonlinear loss surface."""

    def setup(self) -> None:
        super().setup()
        self.region = self.region.fix_in_frame()

    def construct(self) -> None:
        theme = get_active_theme()
        self.set_camera_orientation(phi=INITIAL_CAMERA_PHI, theta=INITIAL_CAMERA_THETA)

        title = Title("AdaGrad's local diagonal squeeze")
        self.region.place(title, UP)
        self.region.update(top=title)

        axes = ThreeDAxes(
            x_range=[*AXES_X_RANGE],
            y_range=[*AXES_Y_RANGE],
            z_range=[*AXES_Z_RANGE],
            x_length=AXES_X_LENGTH,
            y_length=AXES_Y_LENGTH,
            z_length=AXES_Z_LENGTH,
        )

        surface = ScalarFieldSurface(
            lambda u, v: axes.c2p(u, v, _surface_value(u, v)),
            u_range=[*SURFACE_U_RANGE],
            v_range=[*SURFACE_V_RANGE],
            resolution=SURFACE_RESOLUTION,
            color_func="height",
            colormap=[C_BLUE, C_TEAL, C_GREEN, C_ORANGE],
            opacity=SURFACE_OPACITY,
        )

        x_t = CURRENT_POINT
        f_t = _surface_value(float(x_t[0]), float(x_t[1]))
        gradient = _gradient_at(x_t)
        diagonal_hessian = np.diag(np.diag(_hessian_at(x_t)))
        inverse_diagonal = np.linalg.inv(diagonal_hessian)
        model_center = x_t - inverse_diagonal @ gradient
        model_minimum = f_t - 0.5 * gradient @ inverse_diagonal @ gradient
        inverse_sqrt = np.diag(np.sqrt(np.diag(inverse_diagonal)))

        def patch_point(radius: float, angle: float) -> FloatArray:
            local = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            xy = model_center + inverse_sqrt @ local
            z = model_minimum + 0.5 * radius**2
            return axes.c2p(float(xy[0]), float(xy[1]), float(z))

        patch = ScalarFieldSurface(
            patch_point,
            u_range=[*MODEL_PATCH_RADIAL_RANGE],
            v_range=[*MODEL_PATCH_ANGLE_RANGE],
            resolution=MODEL_PATCH_RESOLUTION,
            color_func="height",
            colormap=[C_ORANGE, C_YELLOW],
            opacity=MODEL_PATCH_OPACITY,
        )
        current_z_offset = (axes.z_range[1] - axes.z_range[0]) * CURRENT_DOT_Z_OFFSET_RATIO
        current = Dot3D(
            axes.c2p(float(x_t[0]), float(x_t[1]), f_t + current_z_offset),
            color=C_YELLOW,
            radius=axes.z_axis.get_length() * CURRENT_DOT_RADIUS_TO_Z_AXIS,
        )
        current_label_xy_offset = np.array(
            [
                (AXES_X_RANGE[1] - AXES_X_RANGE[0]) * CURRENT_LABEL_XY_OFFSET_RATIO,
                (AXES_Y_RANGE[1] - AXES_Y_RANGE[0]) * CURRENT_LABEL_XY_OFFSET_RATIO,
            ],
            dtype=np.float64,
        )
        current_label_z_offset = (AXES_Z_RANGE[1] - AXES_Z_RANGE[0]) * CURRENT_LABEL_Z_OFFSET_RATIO
        current_label = theme_math(r"x_t", color=C_TEXT, typography="caption")
        current_label.move_to(
            axes.c2p(
                float(x_t[0] + current_label_xy_offset[0]),
                float(x_t[1] + current_label_xy_offset[1]),
                f_t + current_label_z_offset,
            )
        )
        axis_labels = VGroup(
            theme_math(r"x_1", color=C_TEXT, typography="caption").move_to(
                axes.c2p(AXES_X_RANGE[1], AXES_Y_RANGE[0], AXES_Z_RANGE[0])
            ),
            theme_math(r"x_2", color=C_TEXT, typography="caption").move_to(
                axes.c2p(AXES_X_RANGE[0], AXES_Y_RANGE[1], AXES_Z_RANGE[0])
            ),
            theme_math(r"f(x)", color=C_TEXT, typography="caption").move_to(
                axes.c2p(AXES_X_RANGE[0], AXES_Y_RANGE[0], AXES_Z_RANGE[1])
            ),
        )
        world = Group(axes, surface, patch, current, current_label, axis_labels)
        world.scale(WORLD_SCALE).shift(WORLD_SHIFT)
        self.add_fixed_orientation_mobjects(current_label, *axis_labels)

        equation = VGroup(
            Caption("diagonal local model"),
            theme_math(
                r"f(x)\approx f(x_t)+g_t^\top(x-x_t)"
                r"+\frac12(x-x_t)^\top\Lambda_t(x-x_t)",
            ),
            theme_math(
                r"x_{t+1}=x_t-\eta\Lambda_t^{-1}g_t",
            ),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=SMALL_BUFF)
        for mob in equation[1:]:
            mob.set_color_by_tex(r"\Lambda_t", C_ORANGE)
            mob.set_color_by_tex(r"\eta", C_YELLOW)
            mob.set_color_by_tex(r"g_t", C_TEAL)
        self.region.place(equation, RIGHT)
        equation.shift(UP * HUD_EQUATION_LIFT)

        bar = ColorBar(
            colormap=[C_BLUE, C_TEAL, C_GREEN, C_ORANGE],
            min_value=0,
            max_value=AXES_Z_RANGE[1],
            n_labels=COLOR_BAR_LABEL_COUNT,
            height=axes.z_axis.get_length() * COLOR_BAR_HEIGHT_TO_AXES_Z,
            font_size=theme.typography.caption,
        )
        bar.next_to(equation, DOWN, buff=MED_SMALL_BUFF)
        bar.align_to(equation, LEFT)
        hud = VGroup(equation, bar)
        hud_panel = _glass_panel(hud)

        for mob in (title, hud_panel, equation, bar):
            set_z_index = getattr(mob, "set_z_index", None)
            if callable(set_z_index):
                set_z_index(HUD_Z_INDEX)
            fix_in_frame = getattr(mob, "fix_in_frame", None)
            if callable(fix_in_frame):
                fix_in_frame()

        self.play(Write(title), FadeIn(axes, axis_labels), Create(surface), FadeIn(hud_panel[0]), Write(bar))
        self.wait(0.4)
        self.next_slide(title="Choose a point")
        self.play(FadeIn(current, current_label), Write(equation[0]))
        self.wait(0.4)
        self.next_slide(title="Diagonal quadratic patch")
        self.play(Create(patch), Write(equation[1:]))
        self.move_camera(
            phi=FINAL_CAMERA_PHI,
            theta=FINAL_CAMERA_THETA,
            run_time=CAMERA_MOVE_RUN_TIME,
        )
        self.wait(0.4)
        self.next_slide()
        self.clear_scene()
