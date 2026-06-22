"""Ada Grad Diagonal Scaling slide."""

from __future__ import annotations

from slides.helpers.figure_helpers import *


class AdaGradDiagonalScaling(Slide):
    """Diagonal scaling squeezes level sets toward circles."""

    def construct(self) -> None:
        title = _start_slide(self, "Diagonal scaling as geometry")
        body, note_region = _split_rows(self.region, [4.7, 0.55])
        rows = _split_rows(body, [1.0, 1.0])
        rotated_matrix, _ = _rotated_quadratic_matrix()
        axis_matrix = _quadratic_axis_matrix(rotated_matrix)

        row_groups = []
        for row_region, matrix, label in (
            (rows[0], axis_matrix, "axis-aligned"),
            (rows[1], rotated_matrix, "rotated"),
        ):
            before_region, arrow_region, after_region = _split_weighted(row_region, [1.0, 0.30, 1.0])
            diagonal = np.diag(np.diag(matrix))
            scaled = np.linalg.inv(np.sqrt(diagonal)) @ matrix @ np.linalg.inv(np.sqrt(diagonal))
            before = _quadratic_panel(matrix, f"{label} quadratic", x_length=4.0, y_length=2.65)
            after = _quadratic_panel(scaled, "after diagonal scaling", x_length=4.0, y_length=2.65)
            before_region.scale_and_place(before, buff=SMALL_BUFF)
            after_region.scale_and_place(after, buff=SMALL_BUFF)
            arrow = VGroup(
                theme_math(r"x\mapsto", color=C_PURPLE, typography="caption"),
                theme_math(r"D_A^{-1/2}x", color=C_PURPLE, typography="caption"),
                Arrow(LEFT, RIGHT, color=C_PURPLE, buff=0, tip_shape=StealthTip),
            ).arrange(DOWN, buff=SMALL_BUFF)
            arrow_region.scale_and_place(arrow, buff=SMALL_BUFF)
            row_groups.append((before, arrow, after))

        note = theme_math(
            r"D_A=\operatorname{diag}(A_{11},A_{22})",
            r"\qquad",
            r"D_A^{-1/2}AD_A^{-1/2}",
        )
        color_substrings(note, {r"D_A": C_PURPLE})
        note_region.scale_and_place(note, buff=SMALL_BUFF)

        self.play(Write(title), *(FadeIn(before) for before, _, _ in row_groups))
        self.fragment(title="Apply diagonal map")
        self.play(*(Write(arrow) for _, arrow, _ in row_groups), Write(note))
        self.play(*(FadeIn(after) for _, _, after in row_groups))
