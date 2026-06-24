"""Momentum damping regimes controlled by gamma."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from manim import (
    DL,
    DOWN,
    LEFT,
    MED_LARGE_BUFF,
    MED_SMALL_BUFF,
    RIGHT,
    SMALL_BUFF,
    UP,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Indicate,
    Line,
    MathTex,
    RoundedRectangle,
    SurroundingRectangle,
    VGroup,
    Write,
    always_redraw,
)
from manim.utils.color import color_gradient
from simplex import DN, Slide, TexPage, VT, color_substrings, get_active_theme
from simplex.engine.region import Region

from slides.helpers.controls import (
    SLIDER_TICK_OPACITY,
    SLIDER_TICK_STROKE_WIDTH,
    SLIDER_TRACK_OPACITY,
    SLIDER_TRACK_STROKE_WIDTH,
)
from slides.helpers.figure_helpers import _split_rows, _split_weighted, _start_slide, theme_math
from slides.helpers.style import (
    C_FRAME,
    C_GREEN,
    C_MUTED,
    C_ORANGE,
    C_PANEL,
    C_PANEL_DEEP,
    C_PURPLE,
    C_TEXT,
    C_YELLOW,
    LAYER_MARKERS,
    LAYER_TRAJECTORY,
    PANEL_CORNER_RADIUS,
    PANEL_STROKE_WIDTH,
    SOFT_PANEL_FILL_OPACITY,
    SOFT_PANEL_STROKE_OPACITY,
)

type FloatArray = npt.NDArray[np.float64]


C_ETA = C_ORANGE
C_GAMMA = C_PURPLE
C_CRITICAL = C_GREEN
C_UNDER = C_ORANGE
C_OVER = C_MUTED

BETA_RANGE = (0.0, 1.0)
CRITICAL_BETA = 0.8
UNDERDAMPED_BETA = 0.96
UNDERDAMPED_SWEEP_BETA = 0.86
OVERDAMPED_BETA = 0.02
OVERDAMPED_SWEEP_BETA = 0.55
MOMENTUM_STEP_PRODUCT = (1.0 - np.sqrt(CRITICAL_BETA)) ** 2
INITIAL_POSITION = 1.0
INITIAL_VELOCITY = 0.0
PHASE_STEP_COUNT = 160
PHASE_DOT_TARGET_COUNT = 26
PHASE_FRAME_INSET_FRACTION = 1 / 8
PHASE_FILL_RATIO = 1 - 2 * PHASE_FRAME_INSET_FRACTION
PHASE_DOT_HEIGHT_RATIO = 1 / 60
PHASE_ENDPOINT_DOT_HEIGHT_RATIO = 1 / 38
PHASE_STROKE_WIDTH = PANEL_STROKE_WIDTH * 3
AXIS_STROKE_WIDTH = SLIDER_TICK_STROKE_WIDTH
AXIS_OPACITY = SLIDER_TICK_OPACITY
BETA_TICK_COUNT = 6
BETA_SWEEP_RUN_TIME = 2.0
PANEL_INSET = MED_SMALL_BUFF
PANEL_ROW_RATIOS = (1, 4)

FORMULA_COLORS = {
    r"\eta": C_ETA,
    r"\gamma": C_GAMMA,
    r"\gamma_\star": C_CRITICAL,
    r"\lambda": C_GREEN,
    r"\nabla f(x_t)": C_ETA,
    r"x_{t-1}": C_YELLOW,
    r"x_t": C_YELLOW,
    r"x_{t+1}": C_YELLOW,
    r"x^\star": C_TEXT,
}


@dataclass(frozen=True, slots=True)
class DampingSpec:
    title: str
    beta: float
    color: str


class MomentumDampingRegimes(Slide):
    """Show momentum as under-, critical-, and over-damped eigenmode motion."""

    def construct(self) -> None:
        title = _start_slide(self, "Momentum damping regimes")
        equation_region, track_region, panels_region = _split_rows(
            self.region,
            [0.78, 0.62, 2.65],
        )
        panel_regions = _split_weighted(panels_region, [1, 1, 1])

        under_beta = VT(UNDERDAMPED_BETA)
        over_beta = VT(OVERDAMPED_BETA)

        equation = self._make_equation(equation_region)
        beta_track, track_line = self._make_beta_track(track_region, under_beta, over_beta)

        under_panel = self._make_panel(
            panel_regions[0],
            DampingSpec(
                "Underdamping",
                UNDERDAMPED_BETA,
                C_UNDER,
            ),
            under_beta,
        )
        critical_panel = self._make_panel(
            panel_regions[1],
            DampingSpec(
                "Critical damping",
                CRITICAL_BETA,
                C_CRITICAL,
            ),
            CRITICAL_BETA,
        )
        over_panel = self._make_panel(
            panel_regions[2],
            DampingSpec(
                "Overdamping",
                OVERDAMPED_BETA,
                C_OVER,
            ),
            over_beta,
        )
        panels = VGroup(under_panel, critical_panel, over_panel)
        connectors = self._make_connectors(track_line, panels, under_beta, over_beta)
        critical_rule = self._make_critical_rule(equation_region)
        critical_highlight = SurroundingRectangle(
            critical_panel,
            color=C_CRITICAL,
            buff=SMALL_BUFF,
            corner_radius=PANEL_CORNER_RADIUS,
        )
        critical_highlight.set_stroke(C_CRITICAL, width=PHASE_STROKE_WIDTH)

        self.play(Write(title), Write(equation), Write(beta_track))
        self.wait(2)
        self.next_slide()
        self.wait(0.5)

        self.play(Write(connectors[0]), FadeIn(under_panel, shift=DOWN * SMALL_BUFF))
        self.wait(2)
        self.next_slide()
        self.wait(0.5)

        self.play(
            Write(connectors[1]),
            FadeIn(critical_panel, shift=DOWN * SMALL_BUFF),
            Write(critical_rule),
        )
        self.wait(2)
        self.next_slide()
        self.wait(0.5)

        self.play(Write(connectors[2]), FadeIn(over_panel, shift=DOWN * SMALL_BUFF))
        self.wait(2)
        self.next_slide()
        self.wait(0.5)

        self.play(under_beta @ UNDERDAMPED_SWEEP_BETA, run_time=BETA_SWEEP_RUN_TIME)
        self.play(under_beta @ UNDERDAMPED_BETA, run_time=BETA_SWEEP_RUN_TIME)
        self.wait(2)
        self.next_slide()
        self.wait(0.5)

        self.play(over_beta @ OVERDAMPED_SWEEP_BETA, run_time=BETA_SWEEP_RUN_TIME)
        self.play(over_beta @ OVERDAMPED_BETA, run_time=BETA_SWEEP_RUN_TIME)
        self.wait(2)
        self.next_slide()
        self.wait(0.5)

        self.play(Create(critical_highlight), Indicate(connectors[1]))
        self.wait(2)
        self.next_slide()
        self.wait(0.5)

        self.play(
            FadeOut(
                title,
                equation,
                beta_track,
                connectors,
                panels,
                critical_rule,
                critical_highlight,
            )
        )
        self.wait(2)
        self.next_slide()
        self.wait(0.5)
        self.clear_scene()

    def _make_equation(self, region: Region) -> VGroup:
        update = theme_math(
            r"x_{t+1}=x_t+\gamma(x_t-x_{t-1})-\eta\nabla f(x_t)",
        )
        color_substrings(update, FORMULA_COLORS)
        note = TexPage(
            r"For one Hessian eigenmode, this is a damped spring.",
            page_width=region,
            buff=SMALL_BUFF,
            font_size=get_active_theme().typography.caption,
        )
        group = VGroup(update, note).arrange(DOWN, buff=SMALL_BUFF)
        region.scale_and_place(group, buff=SMALL_BUFF, scale_kwargs={"max_scale": 1})
        return group

    def _make_critical_rule(self, region: Region) -> MathTex:
        rule = theme_math(
            r"\gamma_\star=(1-\sqrt{\eta\lambda})^2",
            color=C_TEXT,
            typography="caption",
        )
        color_substrings(rule, FORMULA_COLORS)
        region.place(rule, RIGHT, buff=MED_SMALL_BUFF)
        return rule

    def _make_beta_track(
        self,
        region: Region,
        under_beta: VT,
        over_beta: VT,
    ) -> tuple[VGroup, Line]:
        label = theme_math(r"\gamma", color=C_GAMMA)
        half_length = max(
            SMALL_BUFF,
            (region.width - label.width - 3 * MED_LARGE_BUFF) / 2,
        )
        track = Line(LEFT * half_length, RIGHT * half_length)
        track.set_stroke(
            C_MUTED,
            width=SLIDER_TRACK_STROKE_WIDTH,
            opacity=SLIDER_TRACK_OPACITY,
        )

        ticks = VGroup()
        tick_height = label.height / 3
        for beta in np.linspace(BETA_RANGE[1], BETA_RANGE[0], BETA_TICK_COUNT):
            point = self._beta_point(track, float(beta))
            tick = Line(point + DOWN * tick_height / 2, point + UP * tick_height / 2)
            tick.set_stroke(
                C_FRAME,
                width=SLIDER_TICK_STROKE_WIDTH,
                opacity=SLIDER_TICK_OPACITY,
            )
            tick_label = theme_math(f"{beta:.1f}", color=C_MUTED, typography="caption")
            tick_label.scale_to_fit_height(label.height / 2)
            tick_label.next_to(tick, DOWN, buff=SMALL_BUFF)
            ticks.add(VGroup(tick, tick_label))

        under_dot = always_redraw(
            lambda: Dot(
                self._beta_point(track, float(~under_beta)),
                color=C_UNDER,
                radius=SMALL_BUFF,
            )
        )
        critical_dot = Dot(
            self._beta_point(track, CRITICAL_BETA),
            color=C_CRITICAL,
            radius=SMALL_BUFF,
        )
        over_dot = always_redraw(
            lambda: Dot(
                self._beta_point(track, float(~over_beta)),
                color=C_OVER,
                radius=SMALL_BUFF,
            )
        )
        ruler = VGroup(track, ticks, under_dot, critical_dot, over_dot)
        group = VGroup(label, ruler).arrange(RIGHT, buff=MED_LARGE_BUFF)
        region.scale_and_place(group, buff=SMALL_BUFF, scale_kwargs={"max_scale": 1})
        return group, track

    def _make_panel(self, region: Region, spec: DampingSpec, beta: VT | float) -> VGroup:
        shell = RoundedRectangle(
            width=region.width - 2 * SMALL_BUFF,
            height=region.height - 2 * SMALL_BUFF,
            corner_radius=PANEL_CORNER_RADIUS,
        )
        shell.set_fill(C_PANEL, opacity=SOFT_PANEL_FILL_OPACITY)
        shell.set_stroke(C_FRAME, width=PANEL_STROKE_WIDTH, opacity=SOFT_PANEL_STROKE_OPACITY)
        region.place(shell)

        inner = self._inset(region, PANEL_INSET)
        header_region, plot_region = _split_rows(inner, PANEL_ROW_RATIOS)

        header = VGroup(
            self._panel_title(spec),
            self._beta_readout(beta, spec.color),
        ).arrange(RIGHT, buff=MED_SMALL_BUFF)
        header_region.scale_and_place(header, buff=SMALL_BUFF, scale_kwargs={"max_scale": 1})

        plot = self._phase_plot(plot_region, beta, spec.color)
        return VGroup(shell, header, plot)

    def _panel_title(self, spec: DampingSpec) -> MathTex:
        title = MathTex(
            rf"\textbf{{{spec.title}}}",
            color=spec.color,
            font_size=get_active_theme().typography.body,
        )
        return title

    def _beta_readout(self, beta: VT | float, color: str) -> VGroup:
        label = theme_math(r"\gamma=", color=C_GAMMA, typography="caption")

        def value() -> float:
            return float(~beta) if isinstance(beta, VT) else float(beta)

        number = DN(value, num_decimal_places=2, font_size=get_active_theme().typography.caption)
        number.set_color(color)
        return VGroup(label, number).arrange(RIGHT, buff=SMALL_BUFF)

    def _phase_plot(self, region: Region, beta: VT | float, color: str) -> VGroup:
        frame_region = self._square_region(self._inset(region, SMALL_BUFF))
        frame = RoundedRectangle(
            width=frame_region.width,
            height=frame_region.height,
            corner_radius=PANEL_CORNER_RADIUS,
        )
        frame.set_fill(C_PANEL_DEEP, opacity=SOFT_PANEL_FILL_OPACITY)
        frame.set_stroke(C_FRAME, width=PANEL_STROKE_WIDTH, opacity=SOFT_PANEL_STROKE_OPACITY)
        frame_region.place(frame)

        horizontal = Line(frame.get_left(), frame.get_right())
        vertical = Line(frame.get_bottom(), frame.get_top())
        axes = VGroup(horizontal, vertical)
        axes.set_stroke(C_MUTED, width=AXIS_STROKE_WIDTH, opacity=AXIS_OPACITY)

        path = always_redraw(lambda: self._phase_path(frame, self._beta_value(beta), color))
        optimum = Dot(
            frame.get_center(),
            color=C_TEXT,
            radius=frame.height * PHASE_ENDPOINT_DOT_HEIGHT_RATIO,
        )
        optimum.set_stroke(C_FRAME, width=PANEL_STROKE_WIDTH)
        label = theme_math(r"x^\star", color=C_TEXT, typography="caption")
        label.scale_to_fit_height(optimum.height)
        label.next_to(optimum, DL, buff=SMALL_BUFF)
        return VGroup(frame, axes, path, optimum, label)

    def _phase_path(self, frame: RoundedRectangle, beta: float, color: str) -> VGroup:
        raw_points = self._phase_points(beta)
        max_x = max(float(np.max(np.abs(raw_points[:, 0]))), np.finfo(float).eps)
        max_y = max(float(np.max(np.abs(raw_points[:, 1]))), np.finfo(float).eps)
        scale = min(frame.width / (2 * max_x), frame.height / (2 * max_y)) * PHASE_FILL_RATIO
        points = [
            frame.get_center() + RIGHT * point[0] * scale + UP * point[1] * scale
            for point in raw_points
        ]

        segments = VGroup()
        for start, end in pairwise(points):
            segment = Line(start, end)
            segment.set_stroke(width=PHASE_STROKE_WIDTH)
            segments.add(segment)
        segment_colors = color_gradient((C_YELLOW, color, C_TEXT), len(segments))
        for segment, segment_color in zip(segments, segment_colors, strict=True):
            segment.set_color(segment_color)

        dot_stride = max(1, len(points) // PHASE_DOT_TARGET_COUNT)
        dot_radius = frame.height * PHASE_DOT_HEIGHT_RATIO
        dots = VGroup(*(Dot(point, radius=dot_radius) for point in points[::dot_stride]))
        dot_colors = color_gradient((C_YELLOW, color, C_TEXT), len(dots))
        for dot, dot_color in zip(dots, dot_colors, strict=True):
            dot.set_color(dot_color)

        start = Dot(
            points[0],
            color=C_YELLOW,
            radius=frame.height * PHASE_ENDPOINT_DOT_HEIGHT_RATIO,
        )
        end = Dot(
            points[-1],
            color=color,
            radius=frame.height * PHASE_ENDPOINT_DOT_HEIGHT_RATIO,
        )
        return VGroup(segments, dots, start, end).set_z_index(LAYER_TRAJECTORY)

    def _phase_points(self, beta: float) -> FloatArray:
        clipped_beta = float(np.clip(beta, *BETA_RANGE))
        position = INITIAL_POSITION
        displacement = INITIAL_VELOCITY
        points: list[tuple[float, float]] = []
        for _ in range(PHASE_STEP_COUNT):
            points.append((displacement, position))
            displacement = clipped_beta * displacement - MOMENTUM_STEP_PRODUCT * position
            position += displacement
        return np.asarray(points, dtype=np.float64)

    def _make_connectors(
        self,
        track: Line,
        panels: VGroup,
        under_beta: VT,
        over_beta: VT,
    ) -> VGroup:
        targets = [panel.get_top() for panel in panels]
        return VGroup(
            always_redraw(
                lambda: self._connector(
                    self._beta_point(track, float(~under_beta)),
                    targets[0],
                    C_UNDER,
                )
            ),
            self._connector(self._beta_point(track, CRITICAL_BETA), targets[1], C_CRITICAL),
            always_redraw(
                lambda: self._connector(
                    self._beta_point(track, float(~over_beta)),
                    targets[2],
                    C_OVER,
                )
            ),
        ).set_z_index(LAYER_MARKERS)

    def _connector(self, start: FloatArray, end: FloatArray, color: str) -> DashedLine:
        connector = DashedLine(start, end)
        connector.set_stroke(color, width=AXIS_STROKE_WIDTH, opacity=AXIS_OPACITY)
        return connector

    @staticmethod
    def _beta_value(beta: VT | float) -> float:
        return float(~beta) if isinstance(beta, VT) else float(beta)

    @staticmethod
    def _beta_point(track: Line, beta: float) -> FloatArray:
        beta_alpha = (np.clip(beta, *BETA_RANGE) - BETA_RANGE[1]) / (
            BETA_RANGE[0] - BETA_RANGE[1]
        )
        return np.asarray(track.point_from_proportion(float(beta_alpha)), dtype=np.float64)

    @staticmethod
    def _inset(region: Region, buff: float) -> Region:
        return Region(
            top=region.top - buff,
            bottom=region.bottom + buff,
            left=region.left + buff,
            right=region.right - buff,
        )

    @staticmethod
    def _square_region(region: Region) -> Region:
        side = min(region.width, region.height)
        center_x = (region.left + region.right) / 2
        center_y = (region.top + region.bottom) / 2
        half_side = side / 2
        return Region(
            top=center_y + half_side,
            bottom=center_y - half_side,
            left=center_x - half_side,
            right=center_x + half_side,
        )
