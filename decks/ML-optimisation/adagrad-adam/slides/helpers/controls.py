"""Reusable control mobjects for deck figures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from manim import DOWN, LEFT, RIGHT, SMALL_BUFF, UP, Dot, Line, MathTex, VGroup, always_redraw
from simplex import DN, VT, get_active_theme

from slides.helpers.style import C_TEXT, C_MUTED

SLIDER_TRACK_STROKE_WIDTH = 5
SLIDER_FILL_STROKE_WIDTH = 7
SLIDER_TRACK_OPACITY = 0.45
SLIDER_TICK_OPACITY = 0.58
SLIDER_TICK_STROKE_WIDTH = 1.0


@dataclass(frozen=True, slots=True)
class SliderSpec:
    label: str
    minimum: float
    maximum: float
    decimals: int
    color: str


class ValueSlider(VGroup):
    """A labelled live slider backed by a Simplex ``VT`` tracker."""

    def __init__(
        self,
        tracker: VT,
        spec: SliderSpec,
        *,
        half_length: float,
        label_font_size: float | None = None,
        value_font_size: float | None = None,
        tick_values: Iterable[float] = (),
        show_endpoint_labels: bool = False,
    ) -> None:
        theme = get_active_theme()
        label_size = label_font_size if label_font_size is not None else theme.typography.caption
        value_size = value_font_size if value_font_size is not None else theme.typography.caption

        track = Line(LEFT * half_length, RIGHT * half_length)
        track.set_stroke(C_MUTED, width=SLIDER_TRACK_STROKE_WIDTH, opacity=SLIDER_TRACK_OPACITY)

        label = MathTex(spec.label, color=spec.color, font_size=label_size)
        value = DN(tracker, num_decimal_places=spec.decimals, font_size=value_size)
        if show_endpoint_labels:
            label.next_to(track, UP, buff=SMALL_BUFF)
            label.align_to(track, LEFT)
            value.next_to(track, UP, buff=SMALL_BUFF)
            value.align_to(track, RIGHT)
            value.add_updater(
                lambda mob: mob.next_to(track, UP, buff=SMALL_BUFF).align_to(track, RIGHT)
            )
        else:
            label.next_to(track, LEFT)
            value.next_to(track, RIGHT)
            value.add_updater(lambda mob: mob.next_to(track, RIGHT))

        fill = always_redraw(
            lambda: Line(
                track.get_start(),
                track.point_from_proportion(slider_alpha(tracker, spec)),
            ).set_stroke(spec.color, width=SLIDER_FILL_STROKE_WIDTH)
        )
        knob = always_redraw(
            lambda: Dot(
                track.point_from_proportion(slider_alpha(tracker, spec)),
                color=spec.color,
                radius=SMALL_BUFF,
            )
        )
        ticks = VGroup(*(self._tick(track, tracker_value, spec) for tracker_value in tick_values))
        mobjects = [label, track, ticks, fill, knob, value]
        if show_endpoint_labels:
            endpoint_labels = VGroup(
                self._endpoint_label(spec.minimum, spec.decimals, value_size),
                self._endpoint_label(spec.maximum, spec.decimals, value_size),
            )
            endpoint_labels[0].next_to(track.get_start(), DOWN, buff=SMALL_BUFF)
            endpoint_labels[0].align_to(track, LEFT)
            endpoint_labels[1].next_to(track.get_end(), DOWN, buff=SMALL_BUFF)
            endpoint_labels[1].align_to(track, RIGHT)
            mobjects.append(endpoint_labels)
        super().__init__(*mobjects)

    @staticmethod
    def _endpoint_label(value: float, decimals: int, font_size: float) -> MathTex:
        return MathTex(f"{value:.{decimals}f}", color=C_MUTED, font_size=font_size)

    @staticmethod
    def _tick(track: Line, tracker_value: float, spec: SliderSpec) -> Line:
        tick_alpha = (tracker_value - spec.minimum) / (spec.maximum - spec.minimum)
        tick = Line(DOWN * SMALL_BUFF, UP * SMALL_BUFF)
        tick.set_stroke(C_TEXT, width=SLIDER_TICK_STROKE_WIDTH, opacity=SLIDER_TICK_OPACITY)
        tick.move_to(track.point_from_proportion(float(np.clip(tick_alpha, 0, 1))))
        return tick


def slider_alpha(tracker: VT, spec: SliderSpec) -> float:
    value = np.clip(~tracker, spec.minimum, spec.maximum)
    return float((value - spec.minimum) / (spec.maximum - spec.minimum))
