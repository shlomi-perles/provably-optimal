"""Quadratic Rotation slide."""

from __future__ import annotations

from slides.helpers.figure_helpers import *


class QuadraticRotation(Slide):
    """Rotate a quadratic into its Hessian eigenbasis."""

    def construct(self) -> None:
        title = _start_slide(self, "The quadratic microscope")
        matrix, eigvecs = _rotated_quadratic_matrix()
        eigenvalues, eigvecs = _quadratic_eigendecomposition(matrix)
        diagonal = np.diag(eigenvalues)
        body, strip = _split_rows(self.region, [4.3, 1.0])
        left, mid, right = _split_weighted(body, [1.0, 0.42, 1.0])

        original = _quadratic_panel(
            matrix,
            "original coordinates",
            x_length=4.4,
            y_length=4.4,
            title_inside=True,
        )
        original_axes = original[1]
        original.add(
            _eigenmode_annotation(
                original_axes,
                eigvecs[:, 0],
                float(eigenvalues[0]),
                eigenvalues,
                r"v_{\min}",
                color=C_BLUE,
            ),
            _eigenmode_annotation(
                original_axes,
                eigvecs[:, 1],
                float(eigenvalues[-1]),
                eigenvalues,
                r"v_{\max}",
                color=C_ORANGE,
            ),
        )
        eigenbasis = _quadratic_panel(
            diagonal,
            "eigenbasis coordinates",
            x_length=4.4,
            y_length=4.4,
            title_inside=True,
        )
        diagonal_values, diagonal_basis = _quadratic_eigendecomposition(diagonal)
        eigen_axes = eigenbasis[1]
        eigenbasis.add(
            _eigenmode_annotation(
                eigen_axes,
                diagonal_basis[:, 0],
                float(diagonal_values[0]),
                diagonal_values,
                r"v_{\min}",
                color=C_BLUE,
            ),
            _eigenmode_annotation(
                eigen_axes,
                diagonal_basis[:, 1],
                float(diagonal_values[-1]),
                diagonal_values,
                r"v_{\max}",
                color=C_ORANGE,
            ),
        )
        _scale_and_place_matching_frame_heights((left, right), (original, eigenbasis))

        map_label = VGroup(
            theme_math(r"x\mapsto", color=C_YELLOW, typography="caption"),
            theme_math(r"V^\top(x-x^\star)", color=C_YELLOW, typography="caption"),
        ).arrange(DOWN, buff=SMALL_BUFF)
        map_arrow = CurvedArrow(LEFT, RIGHT, angle=-PI / 2, color=C_YELLOW, tip_shape=StealthTip)
        map_group = VGroup(map_label, map_arrow).arrange(DOWN)
        mid.scale_and_place(map_group, buff=SMALL_BUFF)

        equations = VGroup(
            theme_math(r"f(x)=\frac12(x-x^\star)^\top A(x-x^\star)"),
            theme_math(r"V^\top A V=\operatorname{diag}(\alpha,\beta)"),
            theme_math(r"A v_i=\lambda_i v_i"),
            theme_math(r"x_0-x^\star=\sum_i \alpha_i v_i"),
        ).arrange(RIGHT, buff=MED_LARGE_BUFF)
        _color_text_parts(
            equations,
            {
                r"v_i": C_BLUE,
                r"\lambda_i": C_ORANGE,
                r"\alpha_i": C_GREEN,
                r"\alpha": C_BLUE,
                r"\beta": C_ORANGE,
            },
        )
        strip.scale_and_place(_themed_box(equations), buff=SMALL_BUFF)

        self.play(Write(title), FadeIn(original))
        self.fragment(title="Rotate into modes")
        self.play(Write(map_group), FadeIn(eigenbasis))
        self.fragment(title="Separated recurrence")
        self.play(Write(equations))
        self.clear_scene()
