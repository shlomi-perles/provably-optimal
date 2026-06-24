"""Momentum memory as a quadratic point cloud."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from manim import (
    Create,
    DEGREES,
    FadeIn,
    FadeOut,
    Group,
    ManimColor,
    MathTex,
    OpenGLPMobject,
    ThreeDAxes,
    Title,
    VGroup,
    Write,
    config,
)
from manim.mobject.three_d.three_dimensions import Dot3D
from manim.utils.color import color_gradient

config.renderer = "opengl"
config.write_to_movie = True

from simplex import ThreeDSlide

from slides.helpers.style import (
    C_BLUE,
    C_GREEN,
    C_MUTED,
    C_ORANGE,
    C_TEAL,
    C_TEXT,
    C_YELLOW,
    theme_math,
)

FloatArray = npt.NDArray[np.float64]

SLIDE_TITLE = "Memory turns samples into shape"
SOURCE_RANDOM_SEED = 17
SOURCE_SAMPLE_COUNT = 520
SAMPLE_LIMIT = 2.2
SAMPLE_AREA = (2 * SAMPLE_LIMIT) ** 2
AVERAGE_SAMPLE_CELL_SIDE = np.sqrt(SAMPLE_AREA / SOURCE_SAMPLE_COUNT)
AXIS_LIMIT = SAMPLE_LIMIT + AVERAGE_SAMPLE_CELL_SIDE
Z_AXIS_LIMIT = 2 * SAMPLE_LIMIT**2
XY_TICK_STEP = round(SAMPLE_LIMIT / 2)
Z_TICK_STEP = round(Z_AXIS_LIMIT / 5)
MATPLOTLIB_BOX_ASPECT = np.array([4.0, 4.0, 3.0], dtype=np.float64)
AXES_X_LENGTH = float(MATPLOTLIB_BOX_ASPECT[0])
AXES_Y_LENGTH = float(MATPLOTLIB_BOX_ASPECT[1])
AXES_Z_LENGTH = float(MATPLOTLIB_BOX_ASPECT[2])
SOURCE_ELEVATION = 24 * DEGREES
SOURCE_AZIMUTH = -48 * DEGREES
CAMERA_PHI = 90 * DEGREES - SOURCE_ELEVATION
CAMERA_THETA = SOURCE_AZIMUTH
DOT_RADIUS_TO_AVERAGE_CELL = 1 / 2


class MomentumQuadraticCloud(ThreeDSlide):
    """Reveal sampled function values as momentum's stored evidence."""

    def setup(self) -> None:
        super().setup()
        self.region = self.region.fix_in_frame()

    def construct(self) -> None:
        self.set_camera_orientation(phi=CAMERA_PHI, theta=CAMERA_THETA)

        title = Title(SLIDE_TITLE)
        self.region.update(top=title)
        title.fix_in_frame()

        axes = self._make_axes()
        axis_labels = self._make_axis_labels(axes)
        samples = self._sample_points()
        heights = self._quadratic_heights(samples)
        order = np.argsort(heights)
        samples = samples[order]
        heights = heights[order]

        dot_radius = self._dot_radius(axes)
        cloud_colors = color_gradient((C_BLUE, C_TEAL, C_GREEN, C_ORANGE), len(samples))
        seed_dot = self._dot(axes, samples[0], heights[0], color=C_YELLOW, radius=dot_radius)
        world = Group(axes, axis_labels, seed_dot)
        self.region.scale_and_place(world)
        self.add_fixed_orientation_mobjects(*axis_labels)

        self.play(Write(title), FadeIn(axes, axis_labels), FadeIn(seed_dot))
        self.wait(0.4)
        self.next_slide()

        cloud = self._cloud(axes, samples[1:], heights[1:], cloud_colors[1:])
        self.play(seed_dot.animate.set_color(cloud_colors[0]), Create(cloud))
        self.wait(0.4)
        self.next_slide()

        scene_mobjects = Group(world, cloud)
        self.play(FadeOut(title), FadeOut(scene_mobjects))
        self.wait(0.4)
        self.next_slide()
        self.clear_scene()

    def _make_axes(self) -> ThreeDAxes:
        return ThreeDAxes(
            x_range=[-AXIS_LIMIT, AXIS_LIMIT, XY_TICK_STEP],
            y_range=[-AXIS_LIMIT, AXIS_LIMIT, XY_TICK_STEP],
            z_range=[0, Z_AXIS_LIMIT, Z_TICK_STEP],
            x_length=AXES_X_LENGTH,
            y_length=AXES_Y_LENGTH,
            z_length=AXES_Z_LENGTH,
            axis_config={"stroke_color": C_MUTED},
        )

    def _make_axis_labels(self, axes: ThreeDAxes) -> VGroup:
        label_scale = axes.z_axis.get_length() / AXES_Z_LENGTH
        labels = VGroup(
            self._axis_label(r"x_1", axes.c2p(AXIS_LIMIT, 0, 0), label_scale),
            self._axis_label(r"x_2", axes.c2p(-AXIS_LIMIT, 0, 0), label_scale),
            self._axis_label(r"f(x)", axes.c2p(0, 0, Z_AXIS_LIMIT), label_scale),
        )
        labels.set_color(C_TEXT)
        return labels

    def _axis_label(self, tex: str, point: FloatArray, label_scale: float) -> MathTex:
        label = theme_math(tex, color=C_TEXT, typography="caption")
        label.scale(label_scale)
        label.move_to(point)
        return label

    def _dot(
        self,
        axes: ThreeDAxes,
        point: FloatArray,
        height: float,
        *,
        color: str,
        radius: float,
    ) -> Dot3D:
        return Dot3D(
            point=axes.c2p(float(point[0]), float(point[1]), height),
            color=color,
            radius=radius,
        )

    def _cloud(
        self,
        axes: ThreeDAxes,
        points: FloatArray,
        heights: FloatArray,
        colors: Sequence[ManimColor],
    ) -> OpenGLPMobject:
        cloud = OpenGLPMobject()
        cloud.add_points(
            np.asarray(
                [
                    axes.c2p(float(point[0]), float(point[1]), float(height))
                    for point, height in zip(points, heights, strict=True)
                ],
                dtype=np.float64,
            ),
            rgbas=np.asarray([ManimColor(color).to_rgba() for color in colors]),
        )
        return cloud

    @staticmethod
    def _sample_points() -> FloatArray:
        rng = np.random.default_rng(SOURCE_RANDOM_SEED)
        return rng.uniform(-SAMPLE_LIMIT, SAMPLE_LIMIT, size=(SOURCE_SAMPLE_COUNT, 2))

    @staticmethod
    def _quadratic_heights(points: FloatArray) -> FloatArray:
        return points[:, 0] ** 2 + points[:, 1] ** 2

    @staticmethod
    def _dot_radius(axes: ThreeDAxes) -> float:
        data_to_scene_scale = axes.x_axis.get_length() / (2 * AXIS_LIMIT)
        average_cell_side = AVERAGE_SAMPLE_CELL_SIDE * data_to_scene_scale
        return average_cell_side * DOT_RADIUS_TO_AVERAGE_CELL
