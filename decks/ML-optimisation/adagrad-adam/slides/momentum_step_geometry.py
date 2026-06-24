"""Heavy-ball and Nesterov step geometry."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from manim import DashedVMobject, MoveToTarget, VMobject
from manim.utils.color import color_gradient

from slides.helpers.figure_helpers import *

type FloatArray = np.ndarray


SLIDE_TITLE = "Momentum step geometry"

SOURCE_ETA = 0.66
SOURCE_GAMMA = 0.60
SOURCE_X_PREV = np.array([20.0, 6.0], dtype=np.float64)
SOURCE_X_T = np.array([1.0, 12.0], dtype=np.float64)
SOURCE_X_LIMITS = (-15.0, 22.0)
SOURCE_Y_LIMITS = (2.0, 19.0)
SOURCE_LEVEL_COUNT = len(np.arange(12))
SOURCE_EXTRA_LEVEL_FRACTIONS = (0.0025,)
SOURCE_CONTOUR_OPACITY = 0.56
SOURCE_CONTOUR_STROKE_WIDTH = 0.58
SOURCE_FIGURE_WIDTH = 8.9
SOURCE_PATH_STROKE_WIDTH = 1.25
SOURCE_PREVIOUS_ARROW_STROKE_WIDTH = 1.55
SOURCE_CORRECTION_STROKE_WIDTH = 1.75
SOURCE_DASHED_STROKE_WIDTH = 1.25
SOURCE_SCATTER_AREA = 46.0
SOURCE_BRACE_BULGE_RATIO = 0.06
SOURCE_BRACE_LABEL_OFFSET_RATIO = 0.04
SOURCE_MEMORY_BRACE_STROKE_WIDTH = 2.6
ELLIPSE_SAMPLE_COUNT = 320
CLIP_EPSILON = 1e-9

PLOT_X_RANGE = (*SOURCE_X_LIMITS, 1.0)
PLOT_Y_RANGE = (*SOURCE_Y_LIMITS, 1.0)
PLOT_Y_TO_X_RATIO = (SOURCE_Y_LIMITS[1] - SOURCE_Y_LIMITS[0]) / (
    SOURCE_X_LIMITS[1] - SOURCE_X_LIMITS[0]
)
PLOT_X_LENGTH = SOURCE_FIGURE_WIDTH
PLOT_Y_LENGTH = PLOT_X_LENGTH * PLOT_Y_TO_X_RATIO
SOURCE_AXIS_HEIGHT_POINTS = SOURCE_FIGURE_WIDTH * PLOT_Y_TO_X_RATIO * 72.0
SOURCE_DOT_FRAME_HEIGHT_RATIO = np.sqrt(SOURCE_SCATTER_AREA / np.pi) / SOURCE_AXIS_HEIGHT_POINTS
PLOT_FRAME_FILL_OPACITY = 0.18
PANEL_TITLE_GAP = SMALL_BUFF
FORMULA_INSET_FRACTION = np.array([0.035, 0.075], dtype=np.float64)
COMPARISON_PANEL_BUFF = MED_SMALL_BUFF

FORMULA_COLORS = {
    r"x_{t-1}": C_GREEN,
    r"x_t": C_GREEN,
    r"x_{t+1}": C_GREEN,
    r"y_t": C_ORANGE,
    r"\gamma": C_PURPLE,
    r"\eta": C_ORANGE,
    r"\nabla f(x_t)": C_ORANGE,
    r"\nabla f(y_t)": C_BLUE,
}


@dataclass(frozen=True, slots=True)
class StepData:
    x_prev: FloatArray
    x_t: FloatArray
    lookahead: FloatArray
    correction_anchor: FloatArray
    gradient_only_point: FloatArray
    x_next: FloatArray
    correction_label: str
    formula: str


class MomentumStepGeometry(Slide):
    """Compare the point where heavy ball and Nesterov evaluate the gradient."""

    def construct(self) -> None:
        title = _start_slide(self, SLIDE_TITLE)
        body_region = self.region.copy()

        matrix, _ = _rotated_quadratic_matrix()
        heavy_ball = self._step_panel(
            "Heavy ball",
            self._heavy_ball_step(matrix),
            show_current_gradient=True,
        )
        body_region.scale_and_place(
            heavy_ball,
            UP,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )

        self.play(Write(title), Write(heavy_ball))
        self.fragment(title="Nesterov look-ahead")

        left_region, right_region = _split_weighted(body_region, [1, 1])
        heavy_ball.generate_target()
        left_region.scale_and_place(
            heavy_ball.target,
            UP,
            buff=COMPARISON_PANEL_BUFF,
            scale_kwargs={"max_scale": 1},
        )

        nesterov = self._step_panel(
            "Nesterov",
            self._nesterov_step(matrix),
            show_lookahead_arrow=True,
            show_net_step=True,
        )
        right_region.scale_and_place(
            nesterov,
            UP,
            buff=COMPARISON_PANEL_BUFF,
            scale_kwargs={"max_scale": 1},
        )

        self.play(MoveToTarget(heavy_ball), FadeIn(nesterov, shift=RIGHT * SMALL_BUFF))
        self.fragment(title="Heavy ball vs. Nesterov")
        self.clear_scene()

    def _heavy_ball_step(self, matrix: FloatArray) -> StepData:
        gradient_correction = -SOURCE_ETA * (matrix @ SOURCE_X_T)
        carried_displacement = SOURCE_GAMMA * (SOURCE_X_T - SOURCE_X_PREV)
        lookahead = SOURCE_X_T + carried_displacement
        return StepData(
            x_prev=SOURCE_X_PREV,
            x_t=SOURCE_X_T,
            lookahead=lookahead,
            correction_anchor=lookahead,
            gradient_only_point=SOURCE_X_T + gradient_correction,
            x_next=lookahead + gradient_correction,
            correction_label=r"-\eta\nabla f(x_t)",
            formula=r"x_{t+1}=x_t+\gamma(x_t-x_{t-1})-\eta\nabla f(x_t)",
        )

    def _nesterov_step(self, matrix: FloatArray) -> StepData:
        carried_displacement = SOURCE_GAMMA * (SOURCE_X_T - SOURCE_X_PREV)
        lookahead = SOURCE_X_T + carried_displacement
        correction = -SOURCE_ETA * (matrix @ lookahead)
        return StepData(
            x_prev=SOURCE_X_PREV,
            x_t=SOURCE_X_T,
            lookahead=lookahead,
            correction_anchor=lookahead,
            gradient_only_point=SOURCE_X_T - SOURCE_ETA * (matrix @ SOURCE_X_T),
            x_next=lookahead + correction,
            correction_label=r"-\eta\nabla f(y_t)",
            formula=r"y_t=x_t+\gamma(x_t-x_{t-1}),\quad x_{t+1}=y_t-\eta\nabla f(y_t)",
        )

    def _step_panel(
        self,
        title: str,
        data: StepData,
        *,
        show_current_gradient: bool = False,
        show_lookahead_arrow: bool = False,
        show_net_step: bool = False,
    ) -> VGroup:
        matrix, _ = _rotated_quadratic_matrix()
        axes = _make_axes(
            PLOT_X_RANGE,
            PLOT_Y_RANGE,
            x_length=PLOT_X_LENGTH,
            y_length=PLOT_Y_LENGTH,
        )
        axes.set_opacity(0)
        frame = _plot_frame(axes)
        frame.set_fill(C_PANEL_DEEP, opacity=PLOT_FRAME_FILL_OPACITY)
        frame.set_z_index(LAYER_FRAME)

        grid = _coordinate_grid(axes).set_z_index(LAYER_CONTOUR)
        contours = self._quadratic_ellipses(axes, matrix).set_z_index(LAYER_CONTOUR)
        path = _axes_polyline(
            axes,
            np.vstack([data.x_prev, data.x_t, data.x_next]),
            color=C_GREEN,
            width=SOURCE_PATH_STROKE_WIDTH,
        ).set_z_index(LAYER_TRAJECTORY)

        previous_step = _axis_arrow(
            axes,
            data.x_prev,
            data.x_t,
            color=C_GREEN,
            width=SOURCE_PREVIOUS_ARROW_STROKE_WIDTH,
        ).set_z_index(LAYER_MARKERS)
        correction = _axis_arrow(
            axes,
            data.correction_anchor,
            data.x_next,
            color=C_ORANGE if show_current_gradient else C_BLUE,
            width=SOURCE_CORRECTION_STROKE_WIDTH,
        ).set_z_index(LAYER_MARKERS)
        memory_brace, memory_label = self._memory_brace(axes, data.x_t, data.lookahead)
        markers = self._markers_and_labels(axes, data)
        correction_label = self._label_between(
            axes,
            data.correction_anchor,
            data.x_next,
            data.correction_label,
            color=C_ORANGE if show_current_gradient else C_BLUE,
            direction=LEFT if show_current_gradient else RIGHT,
        )

        extras = VGroup()
        if show_current_gradient:
            extras.add(
                self._dashed_arrow(
                    axes,
                    data.x_t,
                    data.gradient_only_point,
                    color=C_MUTED,
                    width=SOURCE_DASHED_STROKE_WIDTH,
                ).set_z_index(LAYER_TRAJECTORY),
                self._label_between(
                    axes,
                    data.x_t,
                    data.gradient_only_point,
                    r"-\eta\nabla f(x_t)",
                    color=C_MUTED,
                    direction=RIGHT,
                ),
            )

        if show_lookahead_arrow:
            extras.add(
                _axis_arrow(
                    axes,
                    data.x_t,
                    data.lookahead,
                    color=C_ORANGE,
                    width=SOURCE_CORRECTION_STROKE_WIDTH,
                ).set_z_index(LAYER_MARKERS),
                self._label_between(
                    axes,
                    data.x_t,
                    data.lookahead,
                    r"y_t=x_t+\gamma(x_t-x_{t-1})",
                    color=C_ORANGE,
                    direction=UP,
                ),
            )

        if show_net_step:
            extras.add(
                self._dashed_arrow(
                    axes,
                    data.x_t,
                    data.x_next,
                    color=C_PURPLE,
                    width=SOURCE_PREVIOUS_ARROW_STROKE_WIDTH,
                ).set_z_index(LAYER_TRAJECTORY),
                self._label_between(
                    axes,
                    data.x_t,
                    data.x_next,
                    r"\text{net step}",
                    color=C_PURPLE,
                    direction=LEFT,
                ),
            )

        formula = theme_math(data.formula, color=C_TEXT, typography="caption")
        _color_text_parts(formula, FORMULA_COLORS)
        self._place_formula(frame, formula)

        panel_title = Caption(title)
        panel_title.set_color(C_TEXT)
        panel_title.next_to(frame, UP, buff=PANEL_TITLE_GAP)

        return VGroup(
            frame,
            grid,
            contours,
            axes,
            path,
            previous_step,
            correction,
            memory_brace,
            memory_label,
            markers,
            correction_label,
            extras,
            formula,
            panel_title,
        )

    def _quadratic_ellipses(self, axes: Axes, matrix: FloatArray) -> VGroup:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        x_min, x_max, y_min, y_max = _axes_limits(axes)
        corners = np.array(
            [
                [x_min, y_min],
                [x_min, y_max],
                [x_max, y_min],
                [x_max, y_max],
            ],
            dtype=np.float64,
        )
        corner_heights = 0.5 * np.einsum("bi,ij,bj->b", corners, matrix, corners)
        max_visible_level = float(np.max(corner_heights))
        levels = np.linspace(
            0.035 * max_visible_level,
            0.98 * max_visible_level,
            SOURCE_LEVEL_COUNT,
        )
        extra_levels = np.array(SOURCE_EXTRA_LEVEL_FRACTIONS, dtype=np.float64) * max_visible_level
        levels = np.sort(np.concatenate([levels, extra_levels]))
        contour_colors = color_gradient((C_BLUE, C_TEAL, C_GREEN, C_ORANGE), len(levels))

        contours = VGroup()
        angles = np.linspace(0, 2 * PI, ELLIPSE_SAMPLE_COUNT, endpoint=True)
        for level, color in zip(levels, contour_colors, strict=True):
            radii = np.sqrt(2.0 * float(level) / eigenvalues)
            ellipse_points = eigenvectors @ np.vstack(
                [radii[0] * np.cos(angles), radii[1] * np.sin(angles)]
            )
            contours.add(
                self._clipped_polyline_group(
                    axes,
                    ellipse_points.T,
                    color=color,
                    width=SOURCE_CONTOUR_STROKE_WIDTH,
                    opacity=SOURCE_CONTOUR_OPACITY,
                )
            )
        return contours

    def _clipped_polyline_group(
        self,
        axes: Axes,
        points: FloatArray,
        *,
        color: str,
        width: float,
        opacity: float,
    ) -> VGroup:
        chunks: list[list[FloatArray]] = []
        active: list[FloatArray] = []
        x_min, x_max, y_min, y_max = _axes_limits(axes)

        for start, end in pairwise(points):
            clipped = self._clip_segment_to_rect(start, end, x_min, x_max, y_min, y_max)
            if clipped is None:
                if len(active) > 1:
                    chunks.append(active)
                active = []
                continue

            clipped_start, clipped_end = clipped
            if not active or np.linalg.norm(active[-1] - clipped_start) > CLIP_EPSILON:
                if len(active) > 1:
                    chunks.append(active)
                active = [clipped_start, clipped_end]
            else:
                active.append(clipped_end)

        if len(active) > 1:
            chunks.append(active)

        group = VGroup()
        for chunk in chunks:
            line = VMobject()
            line.set_points_as_corners([axes.c2p(float(point[0]), float(point[1])) for point in chunk])
            line.set_stroke(color, width=width, opacity=opacity)
            group.add(line)
        return group

    @staticmethod
    def _clip_segment_to_rect(
        start: FloatArray,
        end: FloatArray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> tuple[FloatArray, FloatArray] | None:
        delta = end - start
        lower, upper = 0.0, 1.0
        checks = (
            (-delta[0], start[0] - x_min),
            (delta[0], x_max - start[0]),
            (-delta[1], start[1] - y_min),
            (delta[1], y_max - start[1]),
        )
        for edge_delta, distance in checks:
            if abs(edge_delta) < CLIP_EPSILON:
                if distance < 0:
                    return None
                continue

            ratio = distance / edge_delta
            if edge_delta < 0:
                lower = max(lower, float(ratio))
            else:
                upper = min(upper, float(ratio))
            if lower > upper:
                return None

        return start + lower * delta, start + upper * delta

    def _memory_brace(self, axes: Axes, start: FloatArray, end: FloatArray) -> tuple[VMobject, MathTex]:
        segment = end - start
        length = float(np.linalg.norm(segment))
        tangent = segment / max(length, ZERO_AXIS_EPSILON)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        if normal[0] < 0:
            normal = -normal
        bulge = SOURCE_BRACE_BULGE_RATIO * length

        def point(along: float, out: float) -> FloatArray:
            return start + along * segment + out * bulge * normal

        data_points = [
            point(0, 0),
            point(0, 1),
            point(0.5, 1),
            point(0.5, 2),
            point(0.5, 1),
            point(1, 1),
            point(1, 0),
        ]
        scene_points = [axes.c2p(float(p[0]), float(p[1])) for p in data_points]
        brace = VMobject()
        brace.start_new_path(scene_points[0])
        brace.add_cubic_bezier_curve_to(scene_points[1], scene_points[2], scene_points[3])
        brace.add_cubic_bezier_curve_to(scene_points[4], scene_points[5], scene_points[6])
        brace.set_stroke(C_PURPLE, width=SOURCE_MEMORY_BRACE_STROKE_WIDTH)
        brace.set_z_index(LAYER_MARKERS)

        label_point = point(0.5, 2) + SOURCE_BRACE_LABEL_OFFSET_RATIO * length * normal
        label = theme_math(r"\gamma(x_t-x_{t-1})", color=C_PURPLE, typography="caption")
        label.move_to(axes.c2p(float(label_point[0]), float(label_point[1])))
        label.set_z_index(LAYER_MARKERS)
        return brace, label

    def _markers_and_labels(self, axes: Axes, data: StepData) -> VGroup:
        frame = _plot_frame(axes)
        dot_radius = frame.height * SOURCE_DOT_FRAME_HEIGHT_RATIO
        dots = VGroup(
            Dot(axes.c2p(*data.x_prev), color=C_GREEN, radius=dot_radius),
            Dot(axes.c2p(*data.x_t), color=C_GREEN, radius=dot_radius),
            Dot(axes.c2p(*data.lookahead), color=C_ORANGE, radius=dot_radius),
            Dot(axes.c2p(*data.x_next), color=C_GREEN, radius=dot_radius),
        )
        for dot in dots:
            dot.set_stroke(C_TEXT, width=SOURCE_CONTOUR_STROKE_WIDTH)

        labels = VGroup(
            self._point_label(axes, data.x_prev + np.array([-0.5, 0.1]), r"x_{t-1}", C_GREEN),
            self._point_label(axes, data.x_t + np.array([0.35, 0.35]), r"x_t", C_GREEN),
            self._point_label(
                axes,
                data.lookahead + np.array([0.0, 0.7]),
                r"y_t" if not np.allclose(data.lookahead, data.x_t) else r"x_t",
                C_ORANGE,
            ),
            self._point_label(
                axes,
                data.x_next + np.array([-0.3, -0.5]),
                r"x_{t+1}",
                C_GREEN,
            ),
        )
        return VGroup(dots, labels).set_z_index(LAYER_MARKERS)

    def _point_label(self, axes: Axes, point: FloatArray, tex: str, color: str) -> MathTex:
        label = theme_math(tex, color=color, typography="caption")
        label.move_to(axes.c2p(float(point[0]), float(point[1])))
        return label

    def _label_between(
        self,
        axes: Axes,
        start: FloatArray,
        end: FloatArray,
        tex: str,
        *,
        color: str,
        direction: FloatArray,
    ) -> MathTex:
        label = theme_math(tex, color=color, typography="caption")
        midpoint = 0.5 * (start + end)
        data_span = SOURCE_Y_LIMITS[1] - SOURCE_Y_LIMITS[0]
        offset = np.asarray(direction[:2], dtype=np.float64)
        if np.linalg.norm(offset) > ZERO_AXIS_EPSILON:
            offset = offset / np.linalg.norm(offset) * (data_span / 38)
        label.move_to(axes.c2p(float(midpoint[0] + offset[0]), float(midpoint[1] + offset[1])))
        _color_text_parts(label, FORMULA_COLORS)
        label.set_z_index(LAYER_MARKERS)
        return label

    def _dashed_arrow(
        self,
        axes: Axes,
        start: FloatArray,
        end: FloatArray,
        *,
        color: str,
        width: float,
    ) -> DashedVMobject:
        arrow = _axis_arrow(axes, start, end, color=color, width=width)
        dashed = DashedVMobject(arrow, num_dashes=16)
        dashed.set_stroke(color, width=width, opacity=0.82)
        return dashed

    @staticmethod
    def _place_formula(frame: Rectangle, formula: MathTex) -> None:
        formula.move_to(
            frame.get_corner(DL)
            + RIGHT * frame.width * FORMULA_INSET_FRACTION[0]
            + UP * frame.height * FORMULA_INSET_FRACTION[1],
            aligned_edge=DL,
        )
        formula.set_z_index(LAYER_MARKERS)
