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

from simplex import Caption, ColorBar, ScalarFieldSurface, ThreeDSlide

from slides.style import (
    C_BLUE,
    C_GREEN,
    C_ORANGE,
    C_TEAL,
    C_TEXT,
    C_YELLOW,
    glass_panel as _glass_panel,
)

FloatArray = npt.NDArray[np.float64]


def _surface_value(x: float, y: float) -> float:
    return float(
        0.05 * (x + 0.70) ** 2
        + 0.10 * (y + 0.60) ** 2
        + 0.05 * np.exp(2.20 * x + 0.25 * y)
        + 0.05 * np.exp(-0.90 * x + 2.20 * y)
        + 0.025 * np.exp(1.20 * x + 1.65 * y)
    )


def _gradient_at(point: FloatArray) -> FloatArray:
    x, y = point
    exp_1 = np.exp(2.20 * x + 0.25 * y)
    exp_2 = np.exp(-0.90 * x + 2.20 * y)
    exp_3 = np.exp(1.20 * x + 1.65 * y)
    return np.array(
        [
            0.10 * (x + 0.70)
            + 0.05 * 2.20 * exp_1
            - 0.05 * 0.90 * exp_2
            + 0.025 * 1.20 * exp_3,
            0.20 * (y + 0.60)
            + 0.05 * 0.25 * exp_1
            + 0.05 * 2.20 * exp_2
            + 0.025 * 1.65 * exp_3,
        ],
        dtype=np.float64,
    )


def _hessian_at(point: FloatArray) -> FloatArray:
    x, y = point
    exp_1 = np.exp(2.20 * x + 0.25 * y)
    exp_2 = np.exp(-0.90 * x + 2.20 * y)
    exp_3 = np.exp(1.20 * x + 1.65 * y)
    h_xx = (
        0.10
        + 0.05 * 2.20**2 * exp_1
        + 0.05 * 0.90**2 * exp_2
        + 0.025 * 1.20**2 * exp_3
    )
    h_yy = (
        0.20
        + 0.05 * 0.25**2 * exp_1
        + 0.05 * 2.20**2 * exp_2
        + 0.025 * 1.65**2 * exp_3
    )
    h_xy = (
        0.05 * 2.20 * 0.25 * exp_1
        + 0.05 * (-0.90) * 2.20 * exp_2
        + 0.025 * 1.20 * 1.65 * exp_3
    )
    return np.array([[h_xx, h_xy], [h_xy, h_yy]], dtype=np.float64)


class AdaGradLocalSqueeze(ThreeDSlide):
    """A local diagonal Hessian patch over a nonlinear loss surface."""

    def setup(self) -> None:
        super().setup()
        self.region = self.region.fix_in_frame()

    def construct(self) -> None:
        self.slide(title="AdaGrad's local diagonal squeeze")
        self.set_camera_orientation(phi=63 * DEGREES, theta=-52 * DEGREES)

        title = Title("AdaGrad's local diagonal squeeze")
        self.region.place(title, UP)
        self.region.update(top=title)

        axes = ThreeDAxes(
            x_range=[-1.3, 1.2, 1],
            y_range=[-1.0, 1.35, 1],
            z_range=[0, 2.6, 1],
            x_length=3.4,
            y_length=3.4,
            z_length=1.8,
        )

        surface = ScalarFieldSurface(
            lambda u, v: axes.c2p(u, v, _surface_value(u, v)),
            u_range=[-1.20, 1.10],
            v_range=[-0.95, 1.25],
            resolution=(46, 46),
            color_func="height",
            colormap=[C_BLUE, C_TEAL, C_GREEN, C_ORANGE],
            opacity=0.82,
        )

        x_t = np.array([-0.35, 1.10], dtype=np.float64)
        f_t = _surface_value(float(x_t[0]), float(x_t[1]))
        gradient = _gradient_at(x_t)
        diagonal_hessian = np.diag(np.diag(_hessian_at(x_t)))
        inverse_diagonal = np.linalg.inv(diagonal_hessian)
        model_center = x_t - inverse_diagonal @ gradient
        model_minimum = f_t - 0.5 * gradient @ inverse_diagonal @ gradient
        inverse_sqrt = np.diag(np.sqrt(np.diag(inverse_diagonal)))

        def patch_point(u: float, v: float) -> FloatArray:
            xy = model_center + inverse_sqrt @ np.array([u, v], dtype=np.float64)
            z = model_minimum + 0.5 * (u**2 + v**2)
            return axes.c2p(float(xy[0]), float(xy[1]), float(z))

        patch = ScalarFieldSurface(
            patch_point,
            u_range=[-0.72, 0.72],
            v_range=[-0.72, 0.72],
            resolution=(32, 32),
            color_func="height",
            colormap=[C_ORANGE, C_YELLOW],
            opacity=0.95,
        )
        current = Dot3D(axes.c2p(float(x_t[0]), float(x_t[1]), f_t + 0.08), color=C_YELLOW, radius=0.055)
        current_label = MathTex(r"x_t", color=C_TEXT, font_size=22)
        current_label.move_to(axes.c2p(float(x_t[0]) + 0.08, float(x_t[1]) + 0.08, f_t + 0.30))
        axis_labels = VGroup(
            MathTex(r"x_1", color=C_TEXT, font_size=18).move_to(axes.c2p(1.18, -1.02, 0.0)),
            MathTex(r"x_2", color=C_TEXT, font_size=18).move_to(axes.c2p(-1.34, 1.35, 0.0)),
            MathTex(r"f(x)", color=C_TEXT, font_size=18).move_to(axes.c2p(-1.30, -1.0, 2.65)),
        )
        world = Group(axes, surface, patch, current, current_label, axis_labels)
        world.scale(0.82).shift(LEFT * 2.55 + UP * 0.12 + np.array([0.0, 0.0, -0.75]))
        self.add_fixed_orientation_mobjects(current_label, *axis_labels)

        equation = VGroup(
            Caption("diagonal local model"),
            MathTex(
                r"f(x)\approx f(x_t)+g_t^\top(x-x_t)"
                r"+\frac12(x-x_t)^\top\Lambda_t(x-x_t)",
                font_size=25,
            ),
            MathTex(
                r"x_{t+1}=x_t-\eta\Lambda_t^{-1}g_t",
                font_size=27,
            ),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=SMALL_BUFF)
        for mob in equation[1:]:
            mob.set_color_by_tex(r"\Lambda_t", C_ORANGE)
            mob.set_color_by_tex(r"\eta", C_YELLOW)
            mob.set_color_by_tex(r"g_t", C_TEAL)
        self.region.place(equation, RIGHT)
        equation.shift(UP * 0.55)

        bar = ColorBar(
            colormap=[C_BLUE, C_TEAL, C_GREEN, C_ORANGE],
            min_value=0,
            max_value=2.6,
            n_labels=3,
            height=1.65,
            font_size=16,
        )
        bar.next_to(equation, DOWN, buff=MED_SMALL_BUFF)
        bar.align_to(equation, LEFT)
        hud = VGroup(equation, bar)
        hud_panel = _glass_panel(hud)

        for mob in (title, hud_panel, equation, bar):
            set_z_index = getattr(mob, "set_z_index", None)
            if callable(set_z_index):
                set_z_index(20)
            fix_in_frame = getattr(mob, "fix_in_frame", None)
            if callable(fix_in_frame):
                fix_in_frame()

        self.play(Write(title), FadeIn(axes, axis_labels), Create(surface), FadeIn(hud_panel[0]), Write(bar))
        self.fragment(title="Choose a point")
        self.play(FadeIn(current, current_label), Write(equation[0]))
        self.fragment(title="Diagonal quadratic patch")
        self.play(Create(patch), Write(equation[1:]))
        self.move_camera(phi=69 * DEGREES, theta=-35 * DEGREES, run_time=2.0)
