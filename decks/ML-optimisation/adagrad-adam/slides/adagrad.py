"""AdaGrad slide."""

from __future__ import annotations

from manim import TransformMatchingTex

from slides.helpers.figure_helpers import *
from slides.helpers.reminders import ReminderStack
from slides.helpers.second_order import FORMULA_COLORS


class AdaGrad(Slide):
    """Diagonal curvature as AdaGrad's first useful approximation."""

    def construct(self) -> None:
        title = _start_slide(self, "Diagonal Approximation: AdaGrad")
        full_region = self.region.copy()
        intro_region = self.region.copy()
        intro_region.update(left=title.get_left(), right=title.get_right())

        reminder_region, body_region = _split_rows(full_region, [1, 4])
        subtitle_region, figure_region = _split_rows(body_region, [1, 5])

        color_map = {
            r"\lambda": C_ORANGE,
            r"\lambda_i": C_ORANGE,
            r"\Lambda": C_ORANGE,
            r"D_t": C_TEAL,
            r"D_A": C_PURPLE,
            r"V": C_BLUE,
            r"x_t": FORMULA_COLORS[r"x_t"],
            r"x_{t+1}": FORMULA_COLORS[r"x_{t+1}"],
        }

        quadratic_model = theme_math(
            r"f(x)=\frac{1}{2}(x-x^\star)^\top A(x-x^\star)"
        )
        gradient_formula = theme_math(r"\nabla f(x_t)=A(x_t-x^\star)")
        hessian_formula = theme_math(r"\nabla^2 f(x)", r"=", r"A")
        eigendecomposition = theme_math(r"A=V\Lambda V^\top")
        identity_equations = VGroup(
            quadratic_model,
            gradient_formula,
            hessian_formula,
            eigendecomposition,
        ).arrange(DOWN)
        _color_text_parts(identity_equations, color_map)

        hessian_diagonal = theme_math(
            r"\nabla^2 f(x_t)",
            r"\approx",
            r"D_t",
        )
        _color_text_parts(hessian_diagonal, color_map)

        intro_region.scale_and_place(
            identity_equations,
            scale_kwargs={"scaleback": 1 / 2},
        )
        hessian_diagonal.move_to(hessian_formula)

        adagrad_update = theme_math(
            r"x_{t+1}=x_t-D_t\nabla f(x_t)",
            typography="caption",
        )
        _color_text_parts(adagrad_update, color_map)
        reminders = ReminderStack(
            [adagrad_update],
            width=reminder_region.width,
            max_height=reminder_region.height,
            corner=UR,
            orientation="horizontal",
        )
        reminder_region.place(reminders, UR, buff=SMALL_BUFF)
        body_region.update(top=reminders)

        first_guess = theme_math(
            r"\text{First guess: }\nabla^2 f(x_t)\approx \Lambda"
        )
        _color_text_parts(first_guess, color_map)
        subtitle_region.update(top=body_region.get_top())
        subtitle_region.scale_and_place(
            first_guess,
            UL,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )
        figure_region.update(top=first_guess)

        known_ruler, known_paths = self._known_ruler_figures(figure_region, color_map)
        diagonal_scaling = self._diagonal_scaling_figures(figure_region, color_map)

        self.play(Write(title), Write(identity_equations))
        self.next_slide()

        self.play(TransformMatchingTex(hessian_formula, hessian_diagonal))
        self.next_slide()

        self.play(
            FadeIn(reminders.frame, reminders.dividers, *reminders.entries),
            reminders.animate_add_many(
                [
                    quadratic_model,
                    gradient_formula,
                    hessian_diagonal,
                    eigendecomposition,
                ],
                from_existing=True,
            ),
        )
        self.next_slide()

        self.play(Write(first_guess))
        self.play(
            FadeIn(known_ruler),
            *(Write(paths) for paths in known_paths),
        )
        self.next_slide()

        self.play(FadeOut(known_ruler, *known_paths))
        self.play(FadeIn(diagonal_scaling[0]))
        self.play(
            *(Write(arrow) for arrow in diagonal_scaling[1]),
            FadeIn(diagonal_scaling[2]),
        )
        self.play(Write(diagonal_scaling[3]))
        self.next_slide()

        self.clear_scene()
        self.next_slide()

    def _known_ruler_figures(
        self,
        region,
        color_map: dict[str, str],
    ) -> tuple[Group, tuple[VGroup, ...]]:
        body, legend_region = _split_rows(region, [4, 1])
        left, right = _split_weighted(body, [1, 1])
        rotated_matrix, _ = _rotated_quadratic_matrix()
        eigenvalues, _ = _quadratic_eigendecomposition(rotated_matrix)
        axis_matrix = np.diag(eigenvalues)
        x0 = np.array([2.0, 4.0], dtype=np.float64)
        eta_gd = 1.0 / float(np.sum(eigenvalues))

        panel_groups: list[Group] = []
        path_groups: list[VGroup] = []
        for panel_region, matrix, radial in (
            (left, rotated_matrix, False),
            (right, axis_matrix, True),
        ):
            panel = _quadratic_panel(
                matrix,
                "",
                x_range=(-2.5, 8.8, 1),
                y_range=(-2.5, 8.8, 1),
                x_length=5.0,
                y_length=3.85,
            )
            axes = panel[1]
            frame = panel[0]
            diagonal = np.diag(np.diag(matrix))
            gd = _linear_preconditioner_path(matrix, x0, np.eye(2), 102, eta_gd)
            known = _linear_preconditioner_path(matrix, x0, np.linalg.inv(diagonal), 65)
            adagrad = (
                _radial_adagrad_path(x0, 71)
                if radial
                else _coordinate_adagrad_path(matrix, x0, 71)
            )
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
            panel_base = Group(*panel[:4], start, start_label)
            panel_region.scale_and_place(Group(panel_base, paths), buff=SMALL_BUFF)
            panel_groups.append(panel_base)
            path_groups.append(paths)

        legend = VGroup(
            VGroup(Dot(color=C_GREEN), Caption(r"scalar GD")).arrange(RIGHT),
            VGroup(Dot(color=C_PURPLE), Caption(r"known diagonal inverse")).arrange(RIGHT),
            VGroup(Dot(color=C_TEAL), Caption(r"AdaGrad coordinate ruler")).arrange(RIGHT),
            theme_math(r"x_{t+1}=x_t-D_t\nabla f(x_t)", typography="caption"),
        ).arrange(RIGHT, buff=MED_LARGE_BUFF)
        _color_text_parts(legend, color_map)
        legend_shell = _themed_box(legend)
        legend_region.scale_and_place(
            legend_shell,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )
        return Group(*panel_groups, legend_shell), tuple(path_groups)

    def _diagonal_scaling_figures(
        self,
        region,
        color_map: dict[str, str],
    ) -> tuple[VGroup, VGroup, VGroup, MathTex]:
        body, note_region = _split_rows(region, [5, 1])
        rows = _split_rows(body, [1, 1])
        rotated_matrix, _ = _rotated_quadratic_matrix()
        axis_matrix = _quadratic_axis_matrix(rotated_matrix)

        before_panels = Group()
        arrows = Group()
        after_panels = Group()
        for row_region, matrix, label in (
            (rows[0], axis_matrix, "axis-aligned"),
            (rows[1], rotated_matrix, "rotated"),
        ):
            before_region, arrow_region, after_region = _split_weighted(
                row_region,
                [1, 0.3, 1],
            )
            diagonal = np.diag(np.diag(matrix))
            inverse_sqrt = np.linalg.inv(np.sqrt(diagonal))
            scaled = inverse_sqrt @ matrix @ inverse_sqrt
            before = _quadratic_panel(
                matrix,
                f"{label} quadratic",
                x_length=4.0,
                y_length=2.65,
                title_inside=True,
            )
            after = _quadratic_panel(
                scaled,
                "after diagonal scaling",
                x_length=4.0,
                y_length=2.65,
                title_inside=True,
            )
            before_region.scale_and_place(before, buff=SMALL_BUFF)
            after_region.scale_and_place(after, buff=SMALL_BUFF)

            map_label = VGroup(
                theme_math(r"x\mapsto", color=C_PURPLE, typography="caption"),
                theme_math(r"D_A^{-1/2}x", color=C_PURPLE, typography="caption"),
            ).arrange(DOWN, buff=SMALL_BUFF)
            map_arrow = CurvedArrow(
                LEFT,
                RIGHT,
                angle=-PI / 2,
                color=C_PURPLE,
                tip_shape=StealthTip,
            )
            arrow = VGroup(map_label, map_arrow).arrange(DOWN)
            arrow_region.scale_and_place(arrow, buff=SMALL_BUFF)

            before_panels.add(before)
            arrows.add(arrow)
            after_panels.add(after)

        note = theme_math(
            r"D_A=\operatorname{diag}(A_{11},A_{22})",
            r"\qquad",
            r"D_A^{-1/2}AD_A^{-1/2}",
        )
        _color_text_parts(note, color_map)
        note_region.scale_and_place(
            note,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )
        return before_panels, arrows, after_panels, note
