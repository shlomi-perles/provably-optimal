"""Reusable control mobjects for deck figures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from manim import DOWN, LEFT, RIGHT, SMALL_BUFF, UP, Dot, Line, MathTex, VGroup
from simplex import DN, VT, get_active_theme

from slides.helpers.style import C_TEXT, C_MUTED

SLIDER_TRACK_STROKE_WIDTH = 5
SLIDER_FILL_STROKE_WIDTH = 7
SLIDER_TRACK_OPACITY = 0.45
SLIDER_TICK_OPACITY = 0.58
SLIDER_TICK_STROKE_WIDTH = 1.0
type EndpointLabels = tuple[str, str]


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
        endpoint_label_texts: EndpointLabels | None = None,
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

        fill = Line(track.get_start(), self._slider_point(track, tracker, spec))
        fill.set_stroke(spec.color, width=SLIDER_FILL_STROKE_WIDTH)
        fill.add_updater(
            lambda mob: mob.put_start_and_end_on(
                track.get_start(),
                self._slider_point(track, tracker, spec),
            )
        )
        knob = Dot(
            self._slider_point(track, tracker, spec),
            color=spec.color,
            radius=SMALL_BUFF,
        )
        knob.add_updater(lambda mob: mob.move_to(self._slider_point(track, tracker, spec)))
        ticks = VGroup(*(self._tick(track, tracker_value, spec) for tracker_value in tick_values))
        mobjects = [label, track, ticks, fill, knob, value]
        if show_endpoint_labels:
            endpoint_label_texts = endpoint_label_texts or (
                f"{spec.minimum:.{spec.decimals}f}",
                f"{spec.maximum:.{spec.decimals}f}",
            )
            endpoint_labels = VGroup(
                self._endpoint_label(endpoint_label_texts[0], value_size),
                self._endpoint_label(endpoint_label_texts[1], value_size),
            )
            endpoint_labels[0].next_to(track.get_start(), DOWN, buff=SMALL_BUFF)
            endpoint_labels[0].align_to(track, LEFT)
            endpoint_labels[1].next_to(track.get_end(), DOWN, buff=SMALL_BUFF)
            endpoint_labels[1].align_to(track, RIGHT)
            mobjects.append(endpoint_labels)
        super().__init__(*mobjects)

    @staticmethod
    def _endpoint_label(text: str, font_size: float) -> MathTex:
        return MathTex(text, color=C_MUTED, font_size=font_size)

    @staticmethod
    def _tick(track: Line, tracker_value: float, spec: SliderSpec) -> Line:
        tick_alpha = (tracker_value - spec.minimum) / (spec.maximum - spec.minimum)
        tick = Line(DOWN * SMALL_BUFF, UP * SMALL_BUFF)
        tick.set_stroke(C_TEXT, width=SLIDER_TICK_STROKE_WIDTH, opacity=SLIDER_TICK_OPACITY)
        tick.move_to(track.point_from_proportion(float(np.clip(tick_alpha, 0, 1))))
        return tick

    @staticmethod
    def _slider_point(track: Line, tracker: VT, spec: SliderSpec) -> np.ndarray:
        return track.point_from_proportion(slider_alpha(tracker, spec))


def slider_alpha(tracker: VT, spec: SliderSpec) -> float:
    value = np.clip(~tracker, spec.minimum, spec.maximum)
    return float((value - spec.minimum) / (spec.maximum - spec.minimum))
