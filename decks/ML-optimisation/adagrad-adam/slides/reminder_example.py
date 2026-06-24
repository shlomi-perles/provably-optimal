"""Example slide for the reusable reminder stack."""

from __future__ import annotations

from manim import DL, DOWN, LEFT, MED_LARGE_BUFF, RIGHT, SMALL_BUFF, FadeIn, VGroup, Write
from simplex import Caption, Slide

from slides.helpers.reminders import ReminderStack
from slides.helpers.style import (
    C_BLUE,
    C_GREEN,
    C_ORANGE,
    C_TEAL,
    C_TEXT,
    C_YELLOW,
    color_text_parts,
    split_weighted,
    start_slide,
    theme_math,
    themed_box,
)


class ReminderExample(Slide):
    """Demonstrate adding and removing stacked reminders."""

    def construct(self) -> None:
        title = start_slide(self, "Reminder stacks")
        memory_region, work_region = split_weighted(self.region, [0.86, 1.7])
        row_region, column_region = memory_region.split_regions(DOWN, 2)

        problem = VGroup(
            Caption("local quadratic model"),
            theme_math(r"f(x)=\frac12 x^\top Hx"),
            theme_math(r"g_t=Hx_t"),
            theme_math(r"x_{t+1}=x_t-\eta g_t"),
        ).arrange(DOWN, aligned_edge=LEFT)
        color_text_parts(
            problem,
            {
                "H": C_ORANGE,
                r"g_t": C_YELLOW,
                r"\eta": C_GREEN,
                r"x_t": C_BLUE,
            },
        )
        work_region.scale_and_place(themed_box(problem), buff=MED_LARGE_BUFF)

        first = theme_math(r"g_t=\nabla f(x_t)", color=C_TEXT)
        reminders = ReminderStack(
            [first],
        )
        column_region.place(reminders, DL, buff=SMALL_BUFF)

        rate = theme_math(r"\eta_t", color=C_TEXT)
        horizontal_reminders = ReminderStack(
            [rate],
            orientation="horizontal",
        )
        row_region.place(horizontal_reminders, DL, buff=SMALL_BUFF)

        curvature = theme_math(
            r"\alpha I\preceq H\preceq \beta I",
            color=C_TEXT,
        )
        color_text_parts(curvature, {r"\alpha": C_BLUE, r"\beta": C_ORANGE})
        diagonal = theme_math(
            r"D_t=\operatorname{diag}\sqrt{G_t}",
            color=C_TEXT,
        )
        color_text_parts(diagonal, {r"D_t": C_TEAL, r"G_t": C_YELLOW})
        accumulated = theme_math(r"G_t=\sum_{i\le t}g_i^2", color=C_YELLOW)
        diagonal_chip = theme_math(r"D_t", color=C_TEAL)

        self.play(
            Write(title),
            FadeIn(problem, shift=RIGHT * SMALL_BUFF),
            FadeIn(horizontal_reminders),
            FadeIn(reminders),
        )
        self.fragment(title="Pull equations into reminders")
        self.play(
            reminders.animate_add(problem[1], from_existing=True),
            horizontal_reminders.animate_add(problem[2], from_existing=True),
        )
        self.fragment(title="Add curvature memory")
        self.play(reminders.animate_add(curvature), horizontal_reminders.animate_add(accumulated))
        self.fragment(title="Add AdaGrad memory")
        self.play(
            reminders.animate_add(diagonal),
            horizontal_reminders.animate_add(diagonal_chip),
        )
        self.fragment(title="Remove stale gradient")
        self.play(
            reminders.animate_remove(first),
            horizontal_reminders.animate_remove(rate),
            problem.animate.shift(LEFT * SMALL_BUFF),
        )
        self.clear_scene()
