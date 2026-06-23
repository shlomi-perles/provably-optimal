"""Reusable reminder stack mobjects for lecture slides."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import numpy as np
from manim import (
    DL,
    DOWN,
    RIGHT,
    SMALL_BUFF,
    UP,
    Animation,
    AnimationGroup,
    ApplyMethod,
    FadeIn,
    FadeOut,
    Group,
    Line,
    Mobject,
    MoveToTarget,
    Rectangle,
    VGroup,
    config,
)
from simplex import get_active_theme
from simplex.engine.region import Region

from slides.helpers.style import (
    ACCENT_STROKE_OPACITY,
    C_MUTED,
    LAYER_MARKERS,
    PANEL_STROKE_WIDTH,
)

REMINDER_WIDTH_FRACTION = 1 / 2
REMINDER_MAX_HEIGHT_FRACTION = 1/3
REMINDER_INNER_BUFF = 2 * SMALL_BUFF
REMINDER_ENTRY_SHIFT = UP * SMALL_BUFF
REMINDER_CORNER_BUFF = SMALL_BUFF
REMINDER_EPSILON = float(np.finfo(float).eps)
ReminderOrientation = Literal["vertical", "horizontal"]


class _ReminderStackAnimation(AnimationGroup):
    """AnimationGroup with a tiny post-finish layout hook."""

    def __init__(
        self,
        *animations: Animation,
        cleanup: Callable[[], None],
    ) -> None:
        self._cleanup = cleanup
        super().__init__(*animations)

    def finish(self) -> None:
        super().finish()
        self._cleanup()


class ReminderStack(Group):
    """A bottom-left stack of compact reminders with add/remove transitions."""

    def __init__(
        self,
        reminders: Iterable[Mobject] = (),
        *,
        width: float | None = None,
        max_height: float | None = None,
        corner: Sequence[float] | None = DL,
        orientation: ReminderOrientation = "vertical",
        adaptive: bool = False,
        inner_buff: float = REMINDER_INNER_BUFF,
        cell_buff: float = SMALL_BUFF,
        fill_color: str | None = None,
        frame_color: str | None = None,
        divider_color: str | None = None,
    ) -> None:
        theme = get_active_theme()
        self.reminder_width = width if width is not None else config.frame_width * REMINDER_WIDTH_FRACTION
        self.max_height = (
            max_height if max_height is not None else config.frame_height * REMINDER_MAX_HEIGHT_FRACTION
        )
        self.corner = np.asarray(corner, dtype=float) if corner is not None else None
        self.orientation = orientation
        if self.orientation not in ("vertical", "horizontal"):
            raise ValueError("orientation must be 'vertical' or 'horizontal'.")
        self.adaptive = adaptive
        self.inner_buff = inner_buff
        self.cell_buff = cell_buff
        self.fill_color = fill_color if fill_color is not None else theme.web_palette.surface
        self.frame_color = frame_color if frame_color is not None else theme.palette.accent
        self.divider_color = divider_color if divider_color is not None else C_MUTED
        self._natural_sizes: dict[int, tuple[float, float]] = {}
        self._fixed_height: float | None = None

        self.frame = Rectangle(
            width=self.reminder_width,
            height=2 * self.cell_buff,
        )
        self.frame.set_fill(self.fill_color, opacity=1)
        self.frame.set_stroke(
            self.frame_color,
            width=PANEL_STROKE_WIDTH,
            opacity=ACCENT_STROKE_OPACITY,
        )
        self.frame.set_z_index(LAYER_MARKERS)

        self.dividers = VGroup()
        self.entries = Group()
        super().__init__(self.frame, self.dividers, self.entries)

        for reminder in reminders:
            self._remember_natural_size(reminder)
            self.entries.add(reminder)

        self._apply_layout()
        if self.corner is not None:
            self.to_corner(self.corner, buff=REMINDER_CORNER_BUFF)

    def add_reminder(self, reminder: Mobject) -> None:
        """Add a reminder immediately, without animation."""
        self._remember_natural_size(reminder)
        self.entries.add(reminder)
        self._apply_layout()

    def animate_add(self, reminder: Mobject, *, from_existing: bool = False) -> Animation:
        """Animate adding ``reminder`` while growing the frame if needed."""
        return self.animate_add_many([reminder], from_existing=from_existing)

    def animate_add_many(
        self,
        reminders: Iterable[Mobject],
        *,
        from_existing: bool = False,
    ) -> Animation:
        """Animate adding reminders, optionally moving visible mobjects into the stack."""
        sources = tuple(reminders)
        if len(sources) == 0:
            return AnimationGroup()

        new_entries = self._entries_to_add(sources, from_existing=from_existing)
        old_entries = tuple(self.entries)
        future_entries = [*old_entries, *new_entries]

        target_frame = self._target_frame_for(future_entries)
        entry_targets = self._entry_targets(future_entries, target_frame)
        new_dividers = self._make_dividers(future_entries, target_frame)
        old_dividers = self.dividers

        self.frame.target = target_frame
        self.entries.add(*new_entries)
        self.add(new_dividers)

        animations: list[Animation] = [MoveToTarget(self.frame)]
        for entry, target in zip(old_entries, entry_targets[: len(old_entries)], strict=True):
            entry.target = target
            animations.append(MoveToTarget(entry))

        for source, entry, target in zip(sources, new_entries, entry_targets[len(old_entries) :], strict=True):
            if from_existing:
                entry.target = target
                animations.append(MoveToTarget(entry))
                animations.append(ApplyMethod(source.set_opacity, 0))
            else:
                self._apply_entry_target(entry, target)
                animations.append(FadeIn(entry, shift=self._entry_shift()))

        animations.extend(FadeOut(divider) for divider in old_dividers)
        animations.extend(FadeIn(divider) for divider in new_dividers)

        def cleanup() -> None:
            if from_existing:
                for source in sources:
                    source.set_opacity(0)
            self._replace_dividers(new_dividers)
            self._style_entries_and_dividers()

        return _ReminderStackAnimation(*animations, cleanup=cleanup)

    def _entries_to_add(
        self,
        reminders: Sequence[Mobject],
        *,
        from_existing: bool,
    ) -> tuple[Mobject, ...]:
        entries = []
        for reminder in reminders:
            self._remember_natural_size(reminder)
            if from_existing:
                entry = reminder.copy()
                self._natural_sizes[id(entry)] = self._natural_sizes[id(reminder)]
                entry.move_to(reminder)
                entry.set_z_index(LAYER_MARKERS + 2)
            else:
                entry = reminder
            entries.append(entry)
        return tuple(entries)

    def animate_remove(self, reminder: int | Mobject = -1) -> Animation:
        """Animate removing a reminder, then compact the remaining stack."""
        if len(self.entries) == 0:
            return AnimationGroup()

        removed = self._entry_from_index_or_mobject(reminder)
        future_entries = [entry for entry in self.entries if entry is not removed]
        target_frame = self._target_frame_for(future_entries)
        entry_targets = self._entry_targets(future_entries, target_frame)
        new_dividers = self._make_dividers(future_entries, target_frame)
        old_dividers = self.dividers

        self.frame.target = target_frame
        self.add(new_dividers)

        animations: list[Animation] = [
            MoveToTarget(self.frame),
            FadeOut(removed, shift=-self._entry_shift()),
        ]
        for entry, target in zip(future_entries, entry_targets, strict=True):
            entry.target = target
            animations.append(MoveToTarget(entry))

        animations.extend(FadeOut(divider) for divider in old_dividers)
        animations.extend(FadeIn(divider) for divider in new_dividers)

        def cleanup() -> None:
            self.entries.remove(removed)
            self._replace_dividers(new_dividers)
            self._style_entries_and_dividers()

        return _ReminderStackAnimation(*animations, cleanup=cleanup)

    def _apply_layout(self) -> None:
        target_frame = self._target_frame_for(self.entries)
        self.frame.stretch_to_fit_width(target_frame.width)
        self.frame.stretch_to_fit_height(target_frame.height)
        self.frame.move_to(target_frame)
        self.frame.match_style(target_frame)
        self.dividers = self._make_dividers(self.entries, self.frame)
        self.submobjects = [self.frame, self.dividers, self.entries]
        for entry, target in zip(self.entries, self._entry_targets(self.entries, self.frame), strict=True):
            self._apply_entry_target(entry, target)
        self._style_entries_and_dividers()

    def _replace_dividers(self, dividers: VGroup) -> None:
        self.dividers = dividers
        self.submobjects = [self.frame, self.dividers, self.entries]

    def _target_frame_for(self, entries: Iterable[Mobject]) -> Rectangle:
        entry_tuple = tuple(entries)
        if self.orientation == "horizontal":
            height = self._frame_height_for(entry_tuple)
            width = self._frame_width_for(entry_tuple)
        else:
            width = self._frame_width_for(entry_tuple)
            height = self._vertical_frame_height_for(entry_tuple, width)
        target = self.frame.copy()
        anchor = self._anchor_point()
        target.stretch_to_fit_width(width)
        target.stretch_to_fit_height(height)
        if self.corner is None:
            target.move_to(self.frame)
        else:
            target.move_to(anchor, aligned_edge=self.corner)
        target.set_fill(self.fill_color, opacity=1)
        target.set_stroke(
            self.frame_color,
            width=PANEL_STROKE_WIDTH,
            opacity=ACCENT_STROKE_OPACITY,
        )
        target.set_z_index(LAYER_MARKERS)
        return target

    def _anchor_point(self) -> np.ndarray:
        if self.corner is None:
            return self.frame.get_center()
        return self.frame.get_corner(self.corner)

    def _frame_width_for(self, entries: Sequence[Mobject]) -> float:
        if self.orientation == "horizontal":
            return self._horizontal_frame_width_for(entries)
        if self.adaptive:
            return self._vertical_frame_width_for(entries)
        return self.reminder_width

    def _horizontal_frame_width_for(self, entries: Sequence[Mobject]) -> float:
        if len(entries) == 0:
            return 2 * self.cell_buff
        return sum(self._horizontal_cell_widths_for(entries))

    def _frame_height_for(self, entries: Sequence[Mobject]) -> float:
        if self.orientation == "horizontal" and self.adaptive:
            return self._horizontal_frame_height_for(entries)
        if self.orientation == "horizontal":
            return self._fixed_horizontal_frame_height_for(entries)
        return self._vertical_frame_height_for(entries)

    def _fixed_horizontal_frame_height_for(self, entries: Sequence[Mobject]) -> float:
        if self._fixed_height is None and len(entries) > 0:
            self._fixed_height = self._horizontal_frame_height_for(entries)
        if self._fixed_height is not None:
            return self._fixed_height
        return 2 * self.cell_buff

    def _vertical_frame_width_for(self, entries: Sequence[Mobject]) -> float:
        if len(entries) == 0:
            return 2 * self.cell_buff
        content_widths, _content_heights = self._vertical_scaled_content_sizes_for(
            entries,
            self._content_width(),
        )
        return max(max(content_widths) + 2 * self.inner_buff, 2 * self.cell_buff)

    def _vertical_frame_height_for(
        self,
        entries: Sequence[Mobject],
        width: float | None = None,
    ) -> float:
        if len(entries) == 0:
            return 2 * self.cell_buff
        return sum(self._vertical_cell_heights_for(entries, width))

    def _horizontal_frame_height_for(self, entries: Sequence[Mobject]) -> float:
        if len(entries) == 0:
            return 2 * self.cell_buff
        scales = self._horizontal_entry_scales_for(entries)
        content_height = max(
            self._natural_height(entry) * scale for entry, scale in zip(entries, scales, strict=True)
        )
        return max(content_height + 2 * self.cell_buff, 2 * self.cell_buff)

    def _entry_targets(self, entries: Iterable[Mobject], frame: Mobject) -> list[Mobject]:
        entry_tuple = tuple(entries)
        targets = []
        for entry, cell in zip(entry_tuple, self._cell_regions_for(entry_tuple, frame), strict=True):
            target = entry.copy()
            self._restore_natural_size(target, entry)
            cell.scale_and_place(
                target,
                buff=self.cell_buff,
                scale_kwargs={"max_scale": 1},
            )
            target.set_z_index(LAYER_MARKERS + 2)
            targets.append(target)
        return targets

    @staticmethod
    def _apply_entry_target(entry: Mobject, target: Mobject) -> None:
        if entry.height > REMINDER_EPSILON and target.height > REMINDER_EPSILON:
            entry.scale(target.height / entry.height)
        if entry.width > REMINDER_EPSILON and target.width > REMINDER_EPSILON:
            entry.scale(target.width / entry.width)
        entry.move_to(target)
        entry.set_z_index(LAYER_MARKERS + 2)

    def _make_dividers(self, entries: Iterable[Mobject], frame: Mobject) -> VGroup:
        entry_tuple = tuple(entries)
        cells = self._cell_regions_for(entry_tuple, frame)
        if len(cells) < 2:
            return VGroup()

        divider_region = self._divider_region_for(frame)
        dividers = VGroup()
        for cell in cells[:-1]:
            if self.orientation == "horizontal":
                divider_x = cell.right
                divider = Line(
                    np.array([divider_x, divider_region.bottom, 0.0]),
                    np.array([divider_x, divider_region.top, 0.0]),
                )
            else:
                divider_y = cell.bottom
                divider = Line(
                    np.array([divider_region.left, divider_y, 0.0]),
                    np.array([divider_region.right, divider_y, 0.0]),
                )
            divider.set_stroke(
                self.divider_color,
                width=PANEL_STROKE_WIDTH,
                opacity=ACCENT_STROKE_OPACITY / 2,
            )
            divider.set_z_index(LAYER_MARKERS + 1)
            dividers.add(divider)
        return dividers

    def _cell_regions_for(self, entries: Sequence[Mobject], frame: Mobject) -> list[Region]:
        if len(entries) == 0:
            return []
        if self.orientation == "horizontal":
            return self._horizontal_cell_regions_for(entries, frame)
        return self._vertical_cell_regions_for(entries, frame)

    def _horizontal_cell_regions_for(self, entries: Sequence[Mobject], frame: Mobject) -> list[Region]:
        region = self._content_region_for(frame)
        widths = self._horizontal_cell_widths_for(entries)
        left = region.left
        cells = []
        for index, width in enumerate(widths):
            right = region.right if index == len(widths) - 1 else left + width
            cells.append(Region(top=region.top, bottom=region.bottom, left=left, right=right))
            left = right
        return cells

    def _vertical_cell_regions_for(self, entries: Sequence[Mobject], frame: Mobject) -> list[Region]:
        region = self._content_region_for(frame)
        heights = self._vertical_cell_heights_for(entries, frame.width)
        top = region.top
        cells = []
        for index, height in enumerate(heights):
            bottom = region.bottom if index == len(heights) - 1 else top - height
            cells.append(Region(top=top, bottom=bottom, left=region.left, right=region.right))
            top = bottom
        return cells

    def _content_region_for(self, frame: Mobject) -> Region:
        if self.orientation == "horizontal":
            return Region(
                top=frame.get_top()[1],
                bottom=frame.get_bottom()[1],
                left=frame.get_left()[0],
                right=frame.get_right()[0],
            )
        return self._horizontal_inset_region_for(frame)

    def _divider_region_for(self, frame: Mobject) -> Region:
        if self.orientation == "horizontal":
            inset = self._divider_vertical_inset_for(frame)
            return Region(
                top=frame.get_top()[1] - inset,
                bottom=frame.get_bottom()[1] + inset,
                left=frame.get_left()[0],
                right=frame.get_right()[0],
            )
        return self._horizontal_inset_region_for(frame)

    def _horizontal_inset_region_for(self, frame: Mobject) -> Region:
        return Region(
            top=frame.get_top()[1],
            bottom=frame.get_bottom()[1],
            left=frame.get_left()[0] + self.inner_buff,
            right=frame.get_right()[0] - self.inner_buff,
        )

    def _divider_vertical_inset_for(self, frame: Mobject) -> float:
        return max(min(self.inner_buff, frame.height / 2 - REMINDER_EPSILON), 0)

    def _entry_shift(self) -> np.ndarray:
        if self.orientation == "horizontal":
            return RIGHT * SMALL_BUFF
        return REMINDER_ENTRY_SHIFT

    def _content_width(self, width: float | None = None) -> float:
        frame_width = self.reminder_width if width is None else width
        return max(frame_width - 2 * self.inner_buff, REMINDER_EPSILON)

    def _horizontal_cell_widths_for(self, entries: Sequence[Mobject]) -> list[float]:
        return [
            self._natural_width(entry) * scale + 2 * self.cell_buff
            for entry, scale in zip(entries, self._horizontal_entry_scales_for(entries), strict=True)
        ]

    def _vertical_cell_heights_for(
        self,
        entries: Sequence[Mobject],
        width: float | None = None,
    ) -> list[float]:
        _content_widths, content_heights = self._vertical_scaled_content_sizes_for(
            entries,
            self._content_width(width),
        )
        return [content_height + 2 * self.cell_buff for content_height in content_heights]

    def _vertical_scaled_content_sizes_for(
        self,
        entries: Sequence[Mobject],
        content_width: float,
    ) -> tuple[list[float], list[float]]:
        widths = []
        heights = []
        for entry in entries:
            natural_width, natural_height = self._natural_sizes[id(entry)]
            width_scale = min(content_width / natural_width, 1)
            widths.append(natural_width * width_scale)
            heights.append(natural_height * width_scale)
        height_scale = self._vertical_height_scale_for(entries, heights)
        return (
            [width * height_scale for width in widths],
            [height * height_scale for height in heights],
        )

    def _horizontal_entry_scales_for(self, entries: Sequence[Mobject]) -> list[float]:
        available_height = self._available_horizontal_content_height()
        height_scales = [
            min(available_height / self._natural_height(entry), 1)
            for entry in entries
        ]
        content_widths = [
            self._natural_width(entry) * scale
            for entry, scale in zip(entries, height_scales, strict=True)
        ]
        width_scale = self._available_horizontal_content_width(len(entries)) / sum(content_widths)
        shared_width_scale = max(min(width_scale, 1), REMINDER_EPSILON)
        return [
            max(height_scale * shared_width_scale, REMINDER_EPSILON)
            for height_scale in height_scales
        ]

    def _vertical_height_scale_for(
        self,
        entries: Sequence[Mobject],
        content_heights: Sequence[float],
    ) -> float:
        total_content_height = sum(content_heights)
        available_height = self.max_height - 2 * self.cell_buff * len(entries)
        height_scale = max(available_height, REMINDER_EPSILON) / total_content_height
        return max(min(height_scale, 1), REMINDER_EPSILON)

    def _available_horizontal_content_width(self, entry_count: int) -> float:
        return max(self.reminder_width - 2 * self.cell_buff * entry_count, REMINDER_EPSILON)

    def _available_horizontal_content_height(self) -> float:
        if self.orientation == "horizontal" and not self.adaptive and self._fixed_height is not None:
            return max(self._fixed_height - 2 * self.cell_buff, REMINDER_EPSILON)
        return max(self.max_height - 2 * self.cell_buff, REMINDER_EPSILON)

    def _natural_width(self, entry: Mobject) -> float:
        natural_width, _natural_height = self._natural_sizes[id(entry)]
        return natural_width

    def _natural_height(self, entry: Mobject) -> float:
        _natural_width, natural_height = self._natural_sizes[id(entry)]
        return natural_height

    def _remember_natural_size(self, reminder: Mobject) -> None:
        self._natural_sizes[id(reminder)] = (
            max(reminder.width, REMINDER_EPSILON),
            max(reminder.height, REMINDER_EPSILON),
        )

    def _restore_natural_size(self, target: Mobject, entry: Mobject) -> None:
        _natural_width, natural_height = self._natural_sizes[id(entry)]
        if target.height > REMINDER_EPSILON:
            target.scale(natural_height / target.height)

    def _entry_from_index_or_mobject(self, reminder: int | Mobject) -> Mobject:
        if isinstance(reminder, int):
            return self.entries[reminder]
        if reminder not in self.entries:
            raise ValueError("Reminder is not in this stack.")
        return reminder

    def _style_entries_and_dividers(self) -> None:
        self.entries.set_z_index(LAYER_MARKERS + 2)
        self.dividers.set_z_index(LAYER_MARKERS + 1)
