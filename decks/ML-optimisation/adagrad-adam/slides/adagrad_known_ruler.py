"""Ada Grad Known Ruler slide."""

from __future__ import annotations

from slides.helpers.figure_helpers import *


class AdaGradKnownRuler(Slide):
    """Known diagonal curvature is perfect only when axes agree."""

    def construct(self) -> None:
        title = _start_slide(self, "A diagonal ruler can rescale, not rotate")
        body, legend_region = _split_rows(self.region, [4.4, 0.82])
        left, right = _split_weighted(body, [1.0, 1.0])
        rotated_matrix, _ = _rotated_quadratic_matrix()
        eigenvalues, _ = _quadratic_eigendecomposition(rotated_matrix)
        axis_matrix = np.diag(eigenvalues)
        x0 = np.array([2.0, 4.0], dtype=np.float64)
        eta_gd = 1.0 / float(np.sum(eigenvalues))

        panels = []
        for region, matrix, label, radial in (
            (left, rotated_matrix, "rotated quadratic", False),
            (right, axis_matrix, "axis-aligned quadratic", True),
        ):
            panel = _quadratic_panel(
                matrix,
                label,
                x_range=(-2.5, 8.8, 1),
                y_range=(-2.5, 8.8, 1),
                x_length=5.0,
                y_length=3.85,
            )
            axes = panel[1]
            frame = panel[0]
            gd = _linear_preconditioner_path(matrix, x0, np.eye(2), 102, eta_gd)
            diagonal = np.diag(np.diag(matrix))
            known = _linear_preconditioner_path(matrix, x0, np.linalg.inv(diagonal), 65)
            adagrad = _radial_adagrad_path(x0, 71) if radial else _coordinate_adagrad_path(matrix, x0, 71)
            paths = VGroup(
                _path_with_dots(axes, gd, color=C_GREEN, step=4),
                _path_with_dots(axes, known, color=C_PURPLE, step=3),
                _path_with_dots(axes, adagrad, color=C_TEAL, step=3),
            )
            start = Dot(
                axes.c2p(float(x0[0]), float(x0[1])),
                color=C_GREEN,
                radius=frame.height * PANEL_MARKER_FRAME_HEIGHT_RATIO,
            )
            start_label = label_for_dot(r"x_0", start, color=C_GREEN, direction=UR)
            method_key = VGroup(
                VGroup(
                    Dot(color=C_GREEN),
                    theme_math(
                        r"x_{t+1}=x_t-\eta\nabla f(x_t),\ \eta=\frac{1}{\alpha+\beta}",
                        color=C_TEXT,
                        typography="caption",
                    ),
                ).arrange(RIGHT, buff=SMALL_BUFF),
                VGroup(
                    Dot(color=C_PURPLE),
                    theme_math(
                        r"x_{t+1}=x_t-\Lambda^{-1}\nabla f(x_t)",
                        color=C_TEXT,
                        typography="caption",
                    ),
                ).arrange(RIGHT, buff=SMALL_BUFF),
                VGroup(
                    Dot(color=C_TEAL),
                    theme_math(
                        r"x_{t+1}=x_t-D_t^{-1}\nabla f(x_t)",
                        color=C_TEXT,
                        typography="caption",
                    ),
                ).arrange(RIGHT, buff=SMALL_BUFF),
            ).arrange(DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
            method_key.move_to(axes.c2p(-2.10, 8.35), aligned_edge=UL)
            start_group = VGroup(start, start_label, method_key)
            panel.add(paths, start_group)
            region.scale_and_place(panel, buff=SMALL_BUFF)
            panels.append((panel, paths, start_group))

        legend = VGroup(
            VGroup(Dot(color=C_GREEN), Caption(r"scalar GD")).arrange(RIGHT),
            VGroup(Dot(color=C_PURPLE), Caption(r"known diagonal inverse")).arrange(RIGHT),
            VGroup(Dot(color=C_TEAL), Caption(r"AdaGrad coordinate ruler")).arrange(RIGHT),
            theme_math(r"x_{t+1}=x_t-D_t^{-1}\nabla f(x_t)"),
        ).arrange(RIGHT, buff=MED_LARGE_BUFF)
        _color_text_parts(legend, {r"D_t": C_TEAL})
        legend_region.scale_and_place(_themed_box(legend), buff=SMALL_BUFF)

        self.play(Write(title), *(FadeIn(panel[:5], start_group) for panel, _, start_group in panels))
        self.wait(0.4)
        self.next_slide(title="Compare rulers")
        self.play(*(Write(paths[0]) for _, paths, _ in panels), FadeIn(legend[0]))
        self.play(*(Write(paths[1]) for _, paths, _ in panels), FadeIn(legend[1]))
        self.play(*(Write(paths[2]) for _, paths, _ in panels), FadeIn(legend[2:]))
        self.wait(0.4)
        self.next_slide()
        self.clear_scene()
