"""AdaGrad slide."""

from __future__ import annotations

from manim import Mobject, TransformMatchingTex, Unwrite

from slides.helpers.figure_helpers import *
from slides.helpers.reminders import ReminderStack
from slides.helpers.second_order import FORMULA_COLORS


REMINDER_EQUATION_SCALE = 2


def _reminder_equation(mobject: Mobject) -> Mobject:
    equation = mobject.copy()
    if hasattr(equation, "font_size"):
        equation.font_size = float(equation.font_size) * REMINDER_EQUATION_SCALE
        return equation
    equation.scale(REMINDER_EQUATION_SCALE)
    return equation


class AdaGrad(Slide):
    """Diagonal curvature as AdaGrad's first useful approximation."""

    def construct(self) -> None:
        title = _start_slide(self, "Diagonal Approximation: AdaGrad")
        screen_region = self.region.copy()
        intro_region = self.region.copy()
        intro_region.update(left=title.get_left(), right=title.get_right())
        main_region = self.region.copy()

        color_map = {
            r"\lambda": C_ORANGE,
            r"\lambda_i": C_ORANGE,
            r"\lambda_i^2": C_ORANGE,
            r"\alpha_i": C_GREEN,
            r"\alpha": C_BLUE,
            r"\beta": C_ORANGE,
            r"\Lambda": C_ORANGE,
            r"D_t": C_TEAL,
            r"D_A": C_PURPLE,
            r"V": C_BLUE,
            r"\eta": C_YELLOW,
            r"g_t": C_GREEN,
            r"s_{t,i}": C_TEAL,
            r"x_t": FORMULA_COLORS[r"x_t"],
            r"x_{t+1}": FORMULA_COLORS[r"x_{t+1}"],
            r"x_i": C_YELLOW,
            r"x_j": C_YELLOW,
            r"x_{t,i}": C_YELLOW,
            r"x_t^2": FORMULA_COLORS[r"x_t"],
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
        hessian_diagonal.scale(hessian_formula[0].height / hessian_diagonal[0].height)
        hessian_diagonal.move_to(hessian_formula)

        adagrad_update = theme_math(
            r"x_{t+1}=x_t-\eta D_t^{-1}\nabla f(x_t)",
            typography="caption",
        )
        _color_text_parts(adagrad_update, color_map)
        adagrad_update_reminder = _reminder_equation(adagrad_update)
        reminders = ReminderStack(
            [adagrad_update_reminder],
            width=screen_region.width - 2 * SMALL_BUFF,
            orientation="horizontal",
        )
        screen_region.place(reminders, DL, buff=SMALL_BUFF)
        main_region.update(bottom=reminders)
        subtitle_region, figure_region = _split_rows(main_region, [1, 5])

        first_guess = theme_math(
            r"\text{Since }f(x_t)-f_\star= \frac12\sum_{i=1}^n \lambda_i\alpha_i^2(1-\eta\lambda_i)^{2t}\text{, first guess: } \nabla^2 f(x_t)\approx \Lambda"
        )
        _color_text_parts(first_guess, color_map)
        subtitle_region.scale_and_place(
            first_guess,
            UL,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )
        figure_region.update(top=first_guess)

        known_ruler, known_paths, method_keys = self._known_ruler_figures(
            figure_region,
            color_map,
        )
        known_method_labels = VGroup(*(row for key in method_keys for row in key[:2]))
        adagrad_method_labels = VGroup(*(key[2] for key in method_keys))

        self.play(Write(title), Write(identity_equations))
        self.next_slide()

        self.play(TransformMatchingTex(hessian_formula, hessian_diagonal))
        self.next_slide()

        self.play(
            FadeIn(reminders.frame, reminders.dividers, *reminders.entries),
            reminders.animate_add_many(
                [
                    _reminder_equation(quadratic_model).move_to(quadratic_model),
                    _reminder_equation(gradient_formula).move_to(gradient_formula),
                    _reminder_equation(hessian_diagonal).move_to(hessian_diagonal),
                    _reminder_equation(eigendecomposition).move_to(eigendecomposition),
                ],
                from_existing=True,
            ),
            FadeOut(quadratic_model, gradient_formula, hessian_diagonal, eigendecomposition),
        )
        self.next_slide()

        self.play(Write(first_guess))
        self.play(
            FadeIn(known_ruler),
            *(Write(paths[0]) for paths in known_paths),
            *(Write(paths[1]) for paths in known_paths),
            FadeIn(known_method_labels),
        )
        self.next_slide()

        self.play(
            *(Write(paths[2]) for paths in known_paths),
            FadeIn(adagrad_method_labels),
        )
        self.next_slide()

        diagonal_region = main_region.copy()
        diagonal_region.update(top=title)
        diagonal_scaling = self._diagonal_scaling_figures(diagonal_region)
        self.play(
            Unwrite(first_guess),
            FadeOut(
                known_ruler,
                *known_paths,
                known_method_labels,
                adagrad_method_labels,
            ),
        )
        self.play(FadeIn(diagonal_scaling[0]))
        self.play(
            *(Write(arrow) for arrow in diagonal_scaling[1]),
            FadeIn(diagonal_scaling[2]),
        )
        self.next_slide()

        second_guess_region = main_region.copy()
        second_guess_region.update(top=title)
        second_guess_title_region, second_guess_page_region = _split_rows(
            second_guess_region,
            [1, 5],
        )
        second_guess = theme_math(
            r"\text{Second Guess: Gradient Sums}",
            color=C_YELLOW,
            typography="h2",
        )
        second_guess_title_region.place(second_guess)
        second_guess_page_region.update(top=second_guess)
        gradients_page = TexPage(
            r"\["
            r"x_{t+1,i}=x_{t,i}-"
            r"\frac{\nabla f(x_t)_i}"
            r"{\sqrt{\sum_{j=0}^t \nabla f(x_j)_i^2}}"
            r"\]"
            r"Equivalently, with"
            r"\["
            r"D_t=\operatorname{diag}(s_{t,1},\ldots,s_{t,n}),"
            r"\qquad "
            r"s_{t,i}=\sqrt{\sum_{j=0}^t \nabla f(x_j)_i^2},"
            r"\]"
            r"the vector update is"
            r"\["
            r"\boxed{x_{t+1}=x_t-\eta D_t^{-1} \nabla f(x_t)}"
            r"\]",
            page_width=second_guess_page_region,
        )
        color_substrings(gradients_page, color_map, probe_class=MathTex)
        second_guess_page_region.scale_and_place(
            gradients_page,
        )
        first_gradient_equation = gradients_page.equation(0)
        gradients_page_rest = VGroup(
            *gradients_page.lines,
            *gradients_page.equations[1:],
        )
        normalization_region = second_guess_page_region.copy()
        normalization_region.update(top=first_gradient_equation)
        normalization_page = TexPage(
            r"For $f(x)$, the gradient is "
            r"$\nabla f(x)_i = \lambda_i x_i$. Hence"
            r"\["
            r"\frac{\nabla f(x_t)_i}{\sqrt{\sum_{t=1}^{T} \nabla f(x_t)_i^2}}"
            r"="
            r"\frac{\lambda_i x_{t,i}}{\sqrt{\sum_{t=1}^{T} \lambda_i^2 x_{t,i}^2}}"
            r"="
            r"\frac{x_{t,i}}{\sqrt{\sum_{t=1}^{T} x_{t,i}^2}}"
            r"\]",
            page_width=normalization_region,
        )
        color_substrings(normalization_page, color_map, probe_class=MathTex)
        normalization_region.scale_and_place(normalization_page)
        self.play(FadeOut(*diagonal_scaling))
        self.play(Write(second_guess))
        self.play(Write(gradients_page))
        self.next_slide()
        self.play(Unwrite(gradients_page_rest))
        self.play(Write(normalization_page))
        self.next_slide()
        self.clear_scene()

    def _known_ruler_figures(
        self,
        region,
        color_map: dict[str, str],
    ) -> tuple[Group, tuple[VGroup, ...], tuple[VGroup, ...]]:
        left, right = _split_weighted(region, [1, 1])
        rotated_matrix, _ = _rotated_quadratic_matrix()
        eigenvalues, _ = _quadratic_eigendecomposition(rotated_matrix)
        axis_matrix = np.diag(eigenvalues)
        x0 = np.array([2.0, 4.0], dtype=np.float64)
        eta_gd = 1.0 / float(np.sum(eigenvalues))

        panel_groups: list[Group] = []
        path_groups: list[VGroup] = []
        method_keys: list[VGroup] = []
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
            panel_base = Group(*panel[:4], start, start_label)
            panel_region.scale_and_place(Group(panel_base, paths, method_key), buff=SMALL_BUFF)
            panel_groups.append(panel_base)
            path_groups.append(paths)
            method_keys.append(method_key)

        color_substrings(VGroup(*method_keys), color_map, probe_class=MathTex)
        return Group(*panel_groups), tuple(path_groups), tuple(method_keys)

    def _diagonal_scaling_figures(
        self,
        region,
    ) -> tuple[Group, Group, Group]:
        rows = _split_rows(region, [1, 1])
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
                [1, 0.42, 1],
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
                theme_math(r"x\mapsto", color=C_YELLOW, typography="caption"),
                theme_math(r"D_A^{-1/2}x", color=C_YELLOW, typography="caption"),
            ).arrange(DOWN, buff=SMALL_BUFF)
            map_arrow = CurvedArrow(
                LEFT,
                RIGHT,
                angle=-PI / 2,
                color=C_YELLOW,
                tip_shape=StealthTip,
            )
            arrow = VGroup(map_label, map_arrow).arrange(DOWN)
            arrow_region.scale_and_place(arrow, buff=SMALL_BUFF)

            before_panels.add(before)
            arrows.add(arrow)
            after_panels.add(after)

        return before_panels, arrows, after_panels
