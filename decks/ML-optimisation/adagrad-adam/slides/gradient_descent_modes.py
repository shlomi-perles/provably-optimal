"""Gradient Descent Modes slide."""

from __future__ import annotations

from manim import SurroundingRectangle, Transform

from slides.helpers.figure_helpers import *
from slides.helpers.reminders import ReminderStack
from slides.helpers.style import C_PANEL_SOFT, PANEL_CORNER_RADIUS, PANEL_STROKE_WIDTH


class GradientDescentModes(Slide):
    """Gradient descent as independent eigenmode multipliers."""

    def construct(self) -> None:
        title = _start_slide(self, "Gradient descent is a scalar compromise")
        self.region.update(left=title.get_left(), right=title.get_right())
        intro_region, proof_region = _split_rows(self.region, [1, 5])
        function_region, reminder_region = _split_rows(intro_region, [1, 1])
        top, bottom = _split_rows(self.region, [3, 4])
        equations_region, mode_factor_region = _split_weighted(top, [3, 2])
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
            r"\lambda_{\min}": C_BLUE,
            r"\lambda_{\max}": C_ORANGE,
            r"\lambda_i": C_ORANGE,
            r"\eta": C_ETA,
            r"\alpha_i": C_GREEN,
            r"\alpha": C_BLUE,
            r"\beta": C_ORANGE,
            r"\rho": C_GREEN,
            r"\kappa": C_PURPLE,
            r"\epsilon": C_YELLOW,
            r"x_t": C_YELLOW,
            r"x_0": C_YELLOW,
            r"A": C_ORANGE,
        }

        function_definition = theme_math(
            r"f(x)=\frac{1}{2}(x-x^\star)^\top A(x-x^\star)",
        )
        _color_text_parts(function_definition, color_map)
        function_region.scale_and_place(
            function_definition,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )

        gd_update = theme_math(
            r"x_{t+1}=x_t-\eta\nabla f(x_t)",
            typography="caption",
        )
        inverse_hessian_note = theme_math(
            r"(\nabla^2 f(x_t))^{-1}\approx\eta",
            typography="caption",
        )
        _color_text_parts(gd_update, color_map)
        _color_text_parts(inverse_hessian_note, color_map)
        reminders = ReminderStack(
            [gd_update, inverse_hessian_note],
            width=reminder_region.width - 2 * SMALL_BUFF,
            max_height=reminder_region.height,
            corner=None,
            orientation="horizontal",
            fill_color=C_PANEL_SOFT,
        )
        reminder_region.scale_and_place(
            reminders,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )

        derivation_page = TexPage(
            r"\["
            r"\nabla f(x_t)=A(x_t-x^\star)"
            r"\]"
            r"Therefore"
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
            r"$x_0-x^\star=\sum_i\alpha_i v_i$,"
            r"\["
            r"\begin{aligned}"
            r"x_{t+1}-x^\star"
            r"&=\sum_{i=1}^n(1-\eta\lambda_i)^{t+1}\alpha_i v_i\\"
            r"f(x_t)-f_\star"
            r"&=\frac{1}{2}\sum_{i=1}^n"
            r"\lambda_i\alpha_i^2(1-\eta\lambda_i)^{2t}"
            r"\end{aligned}"
            r"\]",
            page_width=proof_region,
            buff=SMALL_BUFF,
            font_size=get_active_theme().typography.caption,
        )
        color_substrings(derivation_page, color_map, probe_class=MathTex)
        proof_region.scale_and_place(
            derivation_page,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )
        eq_gd_mode_sum = derivation_page.equation(3)
        gd_mode_sum_frame = SurroundingRectangle(
            eq_gd_mode_sum,
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

        slider = _eta_slider(
            eta,
            SliderSpec(r"\eta", 0.0, max(2.0 / beta, eta_overshoot), 3, C_ETA),
            eigenvalues,
        )
        slider.scale(0.84)
        slider.move_to(frame.get_corner(DL), aligned_edge=DL)
        slider.shift(RIGHT * SMALL_BUFF + UP * SMALL_BUFF)
        trajectory = _mode_trajectory(axes, eta).set_z_index(LAYER_TRAJECTORY)

        responses = _mode_response_stack(
            eta,
            width=response_region.width - 2 * SMALL_BUFF,
            height=(response_region.height - SMALL_BUFF) / 2,
        )
        response_region.scale_and_place(responses, buff=SMALL_BUFF)
        responses.update()

        mode_factor_axis = _mode_factor_axis(
            eta,
            mode_factor_iteration,
            lambda_min=alpha,
            lambda_max=beta,
            eta_values=(eta_initial, eta_overshoot, eta_balanced, eta_safe),
            width=mode_factor_region.width,
            height=mode_factor_region.height,
        )
        mode_factor_region.scale_and_place(mode_factor_axis, buff=SMALL_BUFF)
        mode_factor_axis.update()

        compact_mode_sum = eq_gd_mode_sum.copy()
        equations_region.scale_and_place(
            compact_mode_sum,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )

        self.play(Write(title), Write(function_definition), FadeIn(reminders))
        self.next_slide()

        self.play(Write(derivation_page.equation(0)))
        self.next_slide()

        self.play(Write(derivation_page.line(0)), Write(derivation_page.equation(1)))
        self.next_slide()

        self.play(Write(derivation_page.line(1)), Write(derivation_page.equation(2)))
        self.next_slide()

        self.play(Write(derivation_page.line(2)), Write(eq_gd_mode_sum))
        self.play(Create(gd_mode_sum_frame))
        self.next_slide()

        self.play(
            FadeOut(function_definition),
            FadeOut(reminders),
            FadeOut(derivation_page.equation(0)),
            FadeOut(derivation_page.line(0)),
            FadeOut(derivation_page.equation(1)),
            FadeOut(derivation_page.line(1)),
            FadeOut(derivation_page.equation(2)),
            FadeOut(derivation_page.line(2)),
            FadeOut(gd_mode_sum_frame),
            Transform(eq_gd_mode_sum, compact_mode_sum),
            FadeIn(heatmap),
            Write(contours),
            Write(frame),
            FadeIn(markers),
            Write(trajectory),
            FadeIn(slider),
            FadeIn(responses),
            FadeIn(mode_factor_axis),
        )
        self.fragment(title="Overshoot the balanced step")
        self.play(
            eta @ eta_overshoot,
            run_time=3.0,
            rate_func=_mode_epsilon_linear_rate_func(
                lambda_i=alpha,
                eta_start=eta_initial,
                eta_end=eta_overshoot,
            ),
        )
        self.fragment(title="Balance the endpoints")
        self.play(
            eta @ eta_balanced,
            run_time=1.8,
            rate_func=_mode_epsilon_linear_rate_func(
                lambda_i=alpha,
                eta_start=eta_overshoot,
                eta_end=eta_balanced,
            ),
        )
        self.fragment(title="Return to the steep safe step")
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
        self.wait(2)

        self.next_slide()

        prior_equation_region, rate_summary_region = _split_rows(equations_region, [1, 2])
        compact_mode_sum_top = eq_gd_mode_sum.copy()
        prior_equation_region.scale_and_place(
            compact_mode_sum_top,
            UP,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )

        balance_condition = theme_math(
            r"1-\eta\lambda_{\min}=1-\eta\lambda_{\max}",
            r"\quad\Rightarrow\quad",
            r"\eta^\star=\frac{2}{\alpha+\beta}",
            typography="caption",
        )
        rho_star = theme_math(
            r"\rho_\star^{\mathrm{GD}}="
            r"\max_{\lambda\in\{\alpha,\beta\}}"
            r"|1-\eta^{\star}\lambda|"
            r"=1-\frac{2}{\kappa+1}",
            typography="caption",
        )
        convergence_bound = theme_math(
            r"f(x_t)-f_\star\le"
            r"(\rho_\star^{\mathrm{GD}})^{2t}"
            r"\bigl(f(x_0)-f_\star\bigr)",
            typography="caption",
        )
        iteration_bound = theme_math(
            r"t=O\left(\kappa\log\frac{1}{\epsilon}\right)",
            typography="caption",
        )
        rate_summary = _formula_stack(
            balance_condition,
            rho_star,
            convergence_bound,
            iteration_bound,
            buff=SMALL_BUFF,
        )
        _color_text_parts(rate_summary, color_map)
        rate_summary_region.scale_and_place(
            rate_summary,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )
        self.play(Transform(eq_gd_mode_sum, compact_mode_sum_top), Write(rate_summary))
        self.next_slide()

        self.clear_scene()
        self.next_slide()
