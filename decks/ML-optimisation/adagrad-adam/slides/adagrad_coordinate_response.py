"""Ada Grad Coordinate Response slide."""

from __future__ import annotations

from slides.helpers.figure_helpers import *


class AdaGradCoordinateResponse(Slide):
    """Compare fixed GD multipliers with AdaGrad's online coordinate gain."""

    def construct(self) -> None:
        title = _start_slide(self, "AdaGrad changes the gain while it moves")
        chart_region, equation_region = _split_weighted(self.region, [2.1, 0.92])
        columns = _split_weighted(chart_region, [1.0, 1.0, 1.0])
        matrix, _ = _rotated_quadratic_matrix()
        eigenvalues, _ = _quadratic_eigendecomposition(matrix)
        alpha, beta = float(eigenvalues[0]), float(eigenvalues[-1])
        specs = [
            ResponseSpec("GD: safe global step", "GD", 1.0 / beta),
            ResponseSpec("GD: balanced\ninverse-curvature step", "GD", 2.0 / (alpha + beta)),
            ResponseSpec("AdaGrad: activity counter", "AdaGrad", ADAGRAD_ACTIVITY_STEP_FRACTION),
        ]

        charts = []
        adagrad_response = _adagrad_coordinate_response(ADAGRAD_ACTIVITY_STEP_FRACTION)
        for region, spec in zip(columns, specs, strict=True):
            if spec.method == "GD":
                top = np.abs(_gd_response(alpha, spec.eta))
                bottom = np.abs(_gd_response(beta, spec.eta))
                top_note = (
                    rf"\begin{{aligned}}\eta&={spec.eta:.3f}\\"
                    rf"|g_0|&={alpha:.2f}\\"
                    rf"|1-\eta\lambda_i|&={abs(1.0 - spec.eta * alpha):.3f}\end{{aligned}}"
                )
                bottom_note = (
                    rf"\begin{{aligned}}\eta&={spec.eta:.3f}\\"
                    rf"|g_0|&={beta:.2f}\\"
                    rf"|1-\eta\lambda_i|&={abs(1.0 - spec.eta * beta):.3f}\end{{aligned}}"
                )
                top_callout = None
                bottom_callout = None
            else:
                top = adagrad_response
                bottom = adagrad_response
                top_note = (
                    rf"\begin{{aligned}}\eta_A&={spec.eta:.2f}\\"
                    rf"\text{{first move}}&={ADAGRAD_ACTIVITY_STEP_PERCENT}\%\text{{ of }}|x_0|\\"
                    rf"|g_0|&={alpha:.2f}\end{{aligned}}"
                )
                bottom_note = (
                    rf"\begin{{aligned}}\eta_A&={spec.eta:.2f}\\"
                    rf"\text{{first move}}&={ADAGRAD_ACTIVITY_STEP_PERCENT}\%\text{{ of }}|x_0|\\"
                    rf"|g_0|&={beta:.2f}\end{{aligned}}"
                )
                top_callout = "same trace"
                bottom_callout = "curvature scale cancels"
            chart = _bar_chart_for_response(
                spec.title,
                top,
                bottom,
                top_note=top_note,
                bottom_note=bottom_note,
                top_callout=top_callout,
                bottom_callout=bottom_callout,
            )
            region.scale_and_place(chart, buff=SMALL_BUFF)
            charts.append(chart)

        equations = _formula_stack(
            Caption("axis-aligned quadratic"),
            theme_math(r"f(x)=\frac12\sum_i\lambda_i x_i^2"),
            Caption("AdaGrad accumulator"),
            theme_math(r"G_{t,i}=\sum_{\tau=1}^{t}g_{\tau,i}^2"),
            theme_math(r"x_{t+1,i}=x_{t,i}-\eta\frac{g_{t,i}}{\sqrt{G_{t,i}}}"),
            Caption(r"because $g_{t,i}=\lambda_i x_{t,i}$"),
            theme_math(
                r"\frac{\lambda_i x_{t,i}}{\sqrt{\sum_\tau\lambda_i^2x_{\tau,i}^2}}"
                r"=\frac{x_{t,i}}{\sqrt{\sum_\tau x_{\tau,i}^2}}",
            ),
        )
        _color_text_parts(equations, {r"\lambda_i": C_ORANGE, r"G_{t,i}": C_TEAL, r"\eta": C_YELLOW})
        equation_region.scale_and_place(_themed_box(equations))

        self.play(Write(title))
        self.play(Write(charts[0]))
        self.fragment(title="A better scalar")
        self.play(Write(charts[1]))
        self.fragment(title="AdaGrad normalization")
        self.play(Write(charts[2]), Write(equations))
        self.next_slide()
        self.clear_scene()
