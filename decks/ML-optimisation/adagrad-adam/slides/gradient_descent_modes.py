"""Gradient Descent Modes slide."""

from __future__ import annotations

from manim import (
    DL,
    DR,
    LARGE_BUFF,
    MoveToTarget,
    ReplacementTransform,
    SurroundingRectangle,
    TransformMatchingTex,
    UL,
    UR,
    Unwrite,
    VGroup,
)
from manim.mobject.text.text_mobject import remove_invisible_chars

from slides.helpers.figure_helpers import *
from slides.helpers.reminders import ReminderStack
from slides.helpers.second_order import FORMULA_COLORS
from slides.helpers.style import PANEL_CORNER_RADIUS, PANEL_STROKE_WIDTH


class GradientDescentModes(Slide):
    """Gradient descent as independent eigenmode multipliers."""

    def construct(self) -> None:
        title = _start_slide(self, "Scalar Approximation: Gradient Descent")
        screen_region = self.region.copy()
        self.region.update(left=title.get_left(), right=title.get_right())
        top, bottom = _split_rows(self.region, [1, 2])
        equations_region, mode_factor_region = _split_weighted(top, [2, 3])
        mode_note_region, mode_factor_region = _split_rows(mode_factor_region, [1, 1])
        figure_region, response_region = _split_weighted(bottom, [2, 3])
        matrix, _ = _rotated_quadratic_matrix()
        eigenvalues, _ = _quadratic_eigendecomposition(matrix)
        alpha, beta = float(eigenvalues[0]), float(eigenvalues[-1])
        eta_initial = 1.0 / (MODE_INITIAL_ETA_SUM_FACTOR * (alpha + beta))
        eta_safe = 1.0 / beta
        eta_balanced = 2.0 / (alpha + beta)
        eta_overshoot = MODE_OVERSHOOT_ETA_NUMERATOR / (alpha + beta)
        eta = VT(eta_initial)
        mode_factor_iteration = VT(MODE_FACTOR_DOT_INITIAL_STEP)

        color_map = {
            r"\lambda": C_ORANGE,
            r"\lambda_i": C_ORANGE,
            r"\lambda_{\min}": C_BLUE,
            r"\lambda_{\max}": C_ORANGE,
            r"\eta": C_ETA,
            r"\alpha_i": C_GREEN,
            r"\alpha": C_BLUE,
            r"\beta": C_ORANGE,
            r"\rho": C_GREEN,
            r"\kappa": C_PURPLE,
            r"\epsilon": C_YELLOW,
            r"x_t": FORMULA_COLORS[r"x_t"],
            r"x_{t+1}": FORMULA_COLORS[r"x_{t+1}"],
            r"x_0": C_YELLOW,
        }

        quadratic_model = theme_math(
            r"f(x)=\frac{1}{2}(x-x^\star)^\top A(x-x^\star)"
        )
        gradient_formula = theme_math(r"\nabla f(x_t)=A(x_t-x^\star)")
        hessian_formula = theme_math(r"\nabla^2 f(x)", r"=", r"A")
        hessian_scalar = theme_math(r"\nabla^2 f(x)", r"\approx", r"\eta")
        identity_equations = VGroup(
            quadratic_model,
            gradient_formula,
            hessian_formula,
        ).arrange(DOWN)
        _color_text_parts(identity_equations, color_map)
        _color_text_parts(hessian_scalar, color_map)

        gd_update = theme_math(
            r"x_{t+1}=x_t-\eta\nabla f(x_t)",
            typography="caption",
        )
        _color_text_parts(gd_update, color_map)
        reminders = ReminderStack(
            [gd_update],
            width=mode_note_region.width,
            orientation="horizontal",
        )
        reminder_parts = VGroup(reminders.frame, reminders.dividers, *reminders.entries)
        screen_region.place(reminders, DL, buff=SMALL_BUFF)

        proof_region = self.region.copy()
        proof_region.update(bottom=reminders)
        proof_region.scale_and_place(
            identity_equations,
            scale_kwargs={"scaleback": 1 / 2},
        )
        hessian_scalar.scale(hessian_formula[0].height / hessian_scalar[0].height)
        hessian_scalar.move_to(hessian_formula)

        derivation_page = TexPage(
            r"\["
            r"\begin{aligned}"
            r"x_{t+1}-x^\star"
            r"&=x_t-x^\star-\eta\nabla f(x_t),\\"
            r"&=x_t-x^\star-\eta A(x_t-x^\star),\\"
            r"&=(I-\eta A)(x_t-x^\star)."
            r"\end{aligned}"
            r"\]"
            r"Here $I$ is the identity matrix. Recursing gives"
            r"\["
            r"x_{t+1}-x^\star=(I-\eta A)^{t+1}(x_0-x^\star)."
            r"\]"
            r"Now use the eigenbasis. Since $A v_i=\lambda_i v_i$ and "
            r"$x_0-x^\star=\sum_i\alpha_i v_i$, ",
            r"\["
            r"x_{t+1}-x^\star"
            r"=\sum_{i=1}^n(1-\eta\lambda_i)^{t+1}\alpha_i v_i"
            r"\]",
            r"\["
            r"f(x_t)-f_\star"
            r"=\frac{1}{2}\sum_{i=1}^n"
            r"\lambda_i\alpha_i^2(1-\eta\lambda_i)^{2t}"
            r"\]",
            page_width=proof_region,
        )
        color_substrings(derivation_page, color_map, probe_class=MathTex)
        derivation_page.equations[-1].next_to(derivation_page.equations[-2], DOWN, buff=SMALL_BUFF)
        proof_region.place(derivation_page)
        derivation_lines = tuple(remove_invisible_chars(line) for line in derivation_page.lines)
        derivation_equations = tuple(
            remove_invisible_chars(equation) for equation in derivation_page.equations
        )
        mode_position_equation = derivation_page.equations[-2]
        mode_value_equation = derivation_page.equations[-1]
        mode_equations = VGroup(mode_position_equation, mode_value_equation)
        derivation_to_unwrite = VGroup(*derivation_lines, *derivation_equations[:-2])
        gd_mode_sum_frame = SurroundingRectangle(
            mode_equations,
            color=C_YELLOW,
            buff=SMALL_BUFF,
            corner_radius=PANEL_CORNER_RADIUS,
        )
        gd_mode_sum_frame.set_stroke(C_YELLOW, width=PANEL_STROKE_WIDTH)

        axes = _make_axes(
            (-2.75, 10.75, 2),
            (-4.35, 9.35, 2),
            x_length=6.6,
            y_length=4.2,
            preserve_unit_aspect=True,
        )
        heatmap = _quadratic_heatmap(axes, matrix).set_z_index(LAYER_HEATMAP)
        contours = _quadratic_level_sets(axes, matrix, count=14).set_z_index(LAYER_CONTOUR)
        frame = _plot_frame(axes).set_z_index(LAYER_FRAME)
        marker_radius = frame.height * PANEL_MARKER_FRAME_HEIGHT_RATIO
        origin = Dot(axes.c2p(0, 0), color=C_TEXT, radius=marker_radius).set_z_index(LAYER_MARKERS)
        origin_label = label_for_dot(r"x^\star", origin, direction=DR)
        start = Dot(axes.c2p(6, 8), color=C_YELLOW, radius=marker_radius).set_z_index(
            LAYER_MARKERS
        )
        start_label = label_for_dot(r"x_0", start, color=C_YELLOW, direction=UR)
        markers = VGroup(origin, origin_label, start, start_label)
        plot_shell = Group(heatmap, contours, axes, frame, markers)
        figure_region.scale_and_place(plot_shell, buff=SMALL_BUFF)
        plot_shell.shift(RIGHT * (title.get_left()[0] - frame.get_left()[0]))

        slider = _eta_slider(
            eta,
            SliderSpec(r"\eta", 0.0, max(2.0 / beta, eta_overshoot), 3, C_ETA),
            eigenvalues,
        )
        slider.scale(0.84)
        slider.move_to(frame.get_corner(DL), aligned_edge=DL)
        slider.shift(RIGHT * SMALL_BUFF + UP * SMALL_BUFF)
        trajectory = _mode_trajectory(axes, eta).set_z_index(LAYER_TRAJECTORY)
        left_figure = Group(plot_shell, trajectory, slider)

        responses = _mode_response_stack(
            eta,
            width=response_region.width - 2 * SMALL_BUFF,
            height=(response_region.height - SMALL_BUFF) / 2,
        )
        response_region.scale_and_place(responses, buff=SMALL_BUFF)
        responses.update()

        response_chart_frame = responses[0][0]
        responses.shift(RIGHT * (title.get_right()[0] - response_chart_frame.get_right()[0]))
        responses.update()
        mode_factor_region.update(
            left=response_chart_frame.get_left(),
            right=response_chart_frame.get_right(),
        )
        mode_factor_axis = _mode_factor_axis(
            eta,
            mode_factor_iteration,
            lambda_min=alpha,
            lambda_max=beta,
            eta_values=(eta_initial, eta_overshoot, eta_balanced, eta_safe),
            width=mode_factor_region.width,
            height=mode_factor_region.height,
        )
        mode_factor_region.scale_and_place(mode_factor_axis, buff=0)
        mode_factor_axis.update()
        rate_axis_title = mode_factor_axis[0]
        rate_axis_axes = mode_factor_axis[2][0]
        mode_factor_axis.shift(RIGHT * (title.get_right()[0] - rate_axis_axes.x_axis.get_right()[0]))
        mode_factor_axis.update()

        dynamic_mobjects = VGroup(slider, trajectory, responses, mode_factor_axis)
        dynamic_mobjects.suspend_updating()

        self.play(Write(title), Write(identity_equations))
        self.next_slide()

        self.play(TransformMatchingTex(hessian_formula, hessian_scalar))
        self.next_slide()

        self.play(
            FadeIn(reminder_parts),
            reminders.animate_add_many(
                [quadratic_model, gradient_formula],
                from_existing=True,
            ),
            Unwrite(hessian_scalar),
            Write(derivation_equations[0]),
        )
        self.next_slide()

        self.play(
            Write(derivation_lines[0]),
            Write(derivation_equations[1]),
        )
        self.next_slide()

        self.play(
            Write(derivation_lines[-1]),
            Write(mode_position_equation),
            Write(mode_value_equation),
            Create(gd_mode_sum_frame),
        )
        self.next_slide()

        reminders.generate_target()
        if reminders.target.width > ZERO_AXIS_EPSILON:
            reminders.target.scale(rate_axis_axes.x_axis.width / reminders.target.width)
        mode_note_region.update(
            left=response_chart_frame.get_left(),
            right=response_chart_frame.get_right(),
        )
        mode_note_region.place(reminders.target, UR, buff=SMALL_BUFF)
        reminders.target.align_to(title, RIGHT)
        compact_equations_region = equations_region.copy()
        compact_equations_region.update(
            right=reminders.target.get_left() + LEFT * SMALL_BUFF
        )
        compact_mode_sum = mode_equations.copy().arrange(DOWN, buff=SMALL_BUFF)
        compact_equations_region.scale_and_place(
            compact_mode_sum,
        )
        self.play(
            Unwrite(derivation_to_unwrite),
            Unwrite(gd_mode_sum_frame),
        )
        self.play(
            ReplacementTransform(mode_equations, compact_mode_sum),
            MoveToTarget(reminders),
            FadeIn(heatmap),
            Write(contours),
            Write(frame),
            Write(markers),
            Write(trajectory),
            FadeIn(slider),
            FadeIn(responses),
            FadeIn(mode_factor_axis),
        )
        dynamic_mobjects.resume_updating()
        dynamic_mobjects.update(0)
        self.next_slide(title="Overshoot the balanced step")
        self.play(
            eta @ eta_overshoot,
            run_time=3.0,
            rate_func=_mode_epsilon_linear_rate_func(
                lambda_i=alpha,
                eta_start=eta_initial,
                eta_end=eta_overshoot,
            ),
        )
        self.next_slide(title="Balance the endpoints")
        self.play(
            eta @ eta_balanced,
            run_time=1.8,
            rate_func=_mode_epsilon_linear_rate_func(
                lambda_i=alpha,
                eta_start=eta_overshoot,
                eta_end=eta_balanced,
            ),
        )
        self.next_slide(title="Return to the steep safe step")
        self.play(
            eta @ eta_safe,
            run_time=3.0,
            rate_func=_mode_epsilon_linear_rate_func(
                lambda_i=beta,
                eta_start=eta_balanced,
                eta_end=eta_safe,
            ),
        )
        self.play(
            mode_factor_iteration @ MODE_FACTOR_DOT_FINAL_STEP,
            run_time=3.0,
            rate_func=rush_into,
        )
        dynamic_mobjects.suspend_updating()
        self.wait(2)

        self.next_slide()

        rate_summary_region = equations_region.copy()
        rate_summary_region.update(top=compact_mode_sum, bottom=left_figure.get_bottom())

        balance_condition = MathTex(
            r"\begin{gathered}"
            r"|1-\eta\alpha|=|1-\eta\beta|\\"
            r"\eta^\star=\frac{2}{\alpha+\beta}"
            r"\end{gathered}",
        )
        balance_condition = remove_invisible_chars(balance_condition)
        color_substrings(balance_condition, color_map)

        rho_star = MathTex(
            r"* \rho_\star^{\mathrm{GD}}"
            r"=1-\frac{2}{\kappa+1}",
        )
        rho_star = remove_invisible_chars(rho_star)
        color_substrings(rho_star, color_map)

        convergence_bound = MathTex(
            r"f(x_t)-f_\star\le"
            r"(\rho_\star^{\mathrm{GD}})^{2t}"
            r"\left(f(x_0)-f_\star\right)"
        )
        convergence_bound = remove_invisible_chars(convergence_bound)
        color_substrings(convergence_bound, color_map)

        iteration_bound = MathTex(
            r"t=O\left(\kappa\log(1/\epsilon)\right)",
        )
        iteration_bound = remove_invisible_chars(iteration_bound)
        color_substrings(iteration_bound, color_map)
        rate_summary = VGroup(
            balance_condition,
            convergence_bound,
            iteration_bound,
            rho_star,
        ).arrange(DOWN, buff=MED_LARGE_BUFF)
        rho_star.scale(0.6)
        rate_summary_region.place(rho_star, DL, buff=0)
        rate_summary_region.scale_and_place(
            rate_summary[:-1],
        )
        self.play(FadeOut(left_figure))
        self.play(Write(balance_condition))
        self.next_slide()
        self.play(Write(convergence_bound), Write(iteration_bound), Write(rho_star))
        self.next_slide()
        self.clear_scene()
s