"""Gradient Descent Modes slide."""

from __future__ import annotations

from slides.helpers.figure_helpers import *


class GradientDescentModes(Slide):
    """Gradient descent as independent eigenmode multipliers."""

    def construct(self) -> None:
        title = _start_slide(self, "Gradient descent is a scalar compromise")
        matrix, _ = _rotated_quadratic_matrix()
        eigenvalues, _ = _quadratic_eigendecomposition(matrix)
        alpha, beta = float(eigenvalues[0]), float(eigenvalues[-1])
        eta_initial = 1.0 / (MODE_INITIAL_ETA_SUM_FACTOR * (alpha + beta))
        eta_safe = 1.0 / beta
        eta_balanced = 2.0 / (alpha + beta)
        eta_overshoot = MODE_OVERSHOOT_ETA_NUMERATOR / (alpha + beta)
        eta = VT(eta_initial)
        mode_factor_iteration = VT(MODE_FACTOR_DOT_INITIAL_STEP)
        top, bottom = _split_rows(self.region, [2.0, 1.0])
        top_left, top_right = _split_weighted(top, [2.0, 3.0])
        bottom_left, bottom_right = _split_weighted(bottom, [2.0, 3.0])

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
        top_left.scale_and_place(plot_shell, buff=SMALL_BUFF)

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
            width=top_right.width - 2 * SMALL_BUFF,
            height=(top_right.height - SMALL_BUFF) / 2,
        )
        top_right.scale_and_place(responses, buff=SMALL_BUFF)
        responses.update()

        rule = theme_math(
            r"x_{t+1}-x^\star=\sum_i(1-\eta\lambda_i)^{t+1}\alpha_i v_i",
        )
        modes = theme_math(r"r_i=1-\eta\lambda_i")
        energy = theme_math(
            r"f(x_t)-f_\star=\frac12\sum_i\lambda_i\alpha_i^2(1-\eta\lambda_i)^{2t}",
        )
        rho = VGroup(
            theme_math(r"\rho_{\mathrm{GD}}(\eta)=\max\{|1-\eta\alpha|,\ |1-\eta\beta|\}="),
            DN(
                lambda: max(abs(1.0 - ~eta * alpha), abs(1.0 - ~eta * beta)),
                num_decimal_places=3,
            ),
        ).arrange(RIGHT, buff=SMALL_BUFF)
        factors = VGroup(
            VGroup(
                theme_math(r"1-\eta\alpha=", color=C_BLUE, typography="caption"),
                DN(lambda: 1.0 - ~eta * alpha, num_decimal_places=3),
            ).arrange(RIGHT, buff=SMALL_BUFF),
            VGroup(
                theme_math(r"1-\eta\beta=", color=C_ORANGE, typography="caption"),
                DN(lambda: 1.0 - ~eta * beta, num_decimal_places=3),
            ).arrange(RIGHT, buff=SMALL_BUFF),
        ).arrange(RIGHT, buff=MED_SMALL_BUFF)
        equations = _formula_stack(rule, modes, energy, rho, factors, buff=SMALL_BUFF)
        _color_text_parts(
            equations,
            {
                r"\eta": C_ETA,
                r"\lambda_i": C_ORANGE,
                r"\alpha_i": C_GREEN,
                r"\alpha": C_BLUE,
                r"\beta": C_ORANGE,
                r"r_i": C_GREEN,
            },
        )
        bottom_left.scale_and_place(_themed_box(equations), buff=SMALL_BUFF)

        mode_factor_axis = _mode_factor_axis(
            eta,
            mode_factor_iteration,
            lambda_min=alpha,
            lambda_max=beta,
            eta_values=(eta_initial, eta_overshoot, eta_balanced, eta_safe),
            width=bottom_right.width,
            height=bottom_right.height,
        )
        bottom_right.scale_and_place(mode_factor_axis, buff=SMALL_BUFF)
        mode_factor_axis.update()

        self.play(
            Write(title),
            FadeIn(heatmap),
            Write(contours),
            Write(frame),
            FadeIn(markers),
        )
        self.play(Write(trajectory), FadeIn(slider), FadeIn(responses))
        self.play(Write(equations), FadeIn(mode_factor_axis))
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
