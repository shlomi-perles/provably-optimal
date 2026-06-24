"""Helpers for the second-order approximation slide."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import SMALL_BUFF, SurroundingRectangle, TexTemplate, VGroup, VMobject
from simplex import TexPage, get_active_theme
from simplex.engine.region import Region

from slides.helpers.figure_helpers import (
    C_BLUE,
    C_FRAME,
    C_GREEN,
    C_ORANGE,
    C_PURPLE,
    C_YELLOW,
    LOCAL_CURRENT_X,
    LOCAL_CURVE_SAMPLES,
    LOCAL_MODEL_CENTER,
    LOCAL_MODEL_DASH_COUNT,
    LOCAL_QUADRATIC_CENTER,
    LOCAL_QUADRATIC_WEIGHT,
    LOCAL_QUARTIC_WEIGHT,
    LOCAL_VALUE_OFFSET,
    LOCAL_X_RANGE,
    LOCAL_Y_MAX_CAP,
    LOCAL_Y_PADDING_ABOVE,
    LOCAL_Y_PADDING_BELOW,
    ZERO_AXIS_EPSILON,
    FloatArray,
    _color_text_parts,
)
from slides.helpers.style import (
    C_PANEL_SOFT,
    LAYER_MARKERS,
    PANEL_CORNER_RADIUS,
    PANEL_FILL_OPACITY,
    PANEL_STROKE_OPACITY,
    PANEL_STROKE_WIDTH,
)

FORMULA_COLORS = {
    r"x_t": C_YELLOW,
    r"x_{t+1}": C_YELLOW,
    r"\alpha": C_BLUE,
    r"\beta": C_ORANGE,
    r"\delta": C_GREEN,
    r"\lambda_{\min}": C_BLUE,
    r"\lambda_{\max}": C_ORANGE,
    r"\kappa": C_PURPLE,
}


@dataclass(frozen=True, slots=True)
class SecondOrderData:
    x_t: float
    f_t: float
    g_t: float
    h_t: float
    x_next: float
    y_next: float
    x_star: float
    f_star: float
    x_min: float
    x_max: float
    x_step: float
    y_min: float
    y_max: float
    alpha_target: float
    beta_target: float
    alpha_sweep_x: float
    beta_sweep_x: float
    beta_left_sweep_x: float
    zoom_out_scale: float

    @property
    def x_span(self) -> float:
        return self.x_max - self.x_min

    @property
    def zoom_anchor_x_ratio(self) -> float:
        return (self.x_star - self.x_min) / self.x_span

    @property
    def zoom_anchor_y_ratio(self) -> float:
        return (self.f_star - self.y_min) / self.y_span


def objective(xs: FloatArray) -> FloatArray:
    return (
        LOCAL_QUARTIC_WEIGHT * (xs - LOCAL_MODEL_CENTER) ** 4
        + LOCAL_QUADRATIC_WEIGHT * (xs - LOCAL_QUADRATIC_CENTER) ** 2
        + LOCAL_VALUE_OFFSET
    )


def gradient(x: float) -> float:
    return 4 * LOCAL_QUARTIC_WEIGHT * (x - LOCAL_MODEL_CENTER) ** 3 + 2 * (
        LOCAL_QUADRATIC_WEIGHT * (x - LOCAL_QUADRATIC_CENTER)
    )


def hessian(x: float) -> float:
    return 12 * LOCAL_QUARTIC_WEIGHT * (x - LOCAL_MODEL_CENTER) ** 2 + 2 * (LOCAL_QUADRATIC_WEIGHT)


def objective_value(x: float) -> float:
    return float(objective(np.array([x]))[0])


def quadratic_model(values: FloatArray, anchor: float, curvature: float) -> FloatArray:
    return (
        objective_value(anchor)
        + gradient(anchor) * (values - anchor)
        + curvature * (values - anchor) ** 2 / 2
    )


def minimum_x() -> float:
    roots = np.roots(
        [
            4 * LOCAL_QUARTIC_WEIGHT,
            -12 * LOCAL_QUARTIC_WEIGHT * LOCAL_MODEL_CENTER,
            12 * LOCAL_QUARTIC_WEIGHT * LOCAL_MODEL_CENTER**2 + 2 * LOCAL_QUADRATIC_WEIGHT,
            -(4 * LOCAL_QUARTIC_WEIGHT * LOCAL_MODEL_CENTER**3)
            - 2 * LOCAL_QUADRATIC_WEIGHT * LOCAL_QUADRATIC_CENTER,
        ]
    )
    return float(
        min(
            (root.real for root in roots if abs(root.imag) < ZERO_AXIS_EPSILON),
            key=objective_value,
        )
    )


def zoom_scale_for(x_star: float, x_t: float, sweep_x: float) -> float:
    x_min, x_max, x_step = LOCAL_X_RANGE
    padding = x_step / 2
    return max(
        1.0,
        (sweep_x + padding - x_star) / max(x_max - x_star, ZERO_AXIS_EPSILON),
        (x_star - min(x_min, x_t) + padding) / max(x_star - x_min, ZERO_AXIS_EPSILON),
    )


def calibration_domain(x_star: float, x_t: float) -> tuple[FloatArray, tuple[float, ...], float]:
    x_min, x_max, _ = LOCAL_X_RANGE
    calibration_alpha_x = float(x_star + 2 * abs(x_star - x_t))
    calibration_beta_x = float(x_star + abs(x_star - x_t))
    zoom_out_scale = zoom_scale_for(x_star, x_t, max(calibration_alpha_x, calibration_beta_x))
    zoom_x_span = (x_max - x_min) * zoom_out_scale
    zoom_x_min = x_star - (x_star - x_min) / (x_max - x_min) * zoom_x_span
    zoom_x_max = zoom_x_min + zoom_x_span
    domain = np.linspace(
        min(x_min, zoom_x_min, x_t, calibration_alpha_x, calibration_beta_x),
        max(x_max, zoom_x_max, x_t, calibration_alpha_x, calibration_beta_x),
        LOCAL_CURVE_SAMPLES,
    )
    return domain, (x_t, calibration_alpha_x, calibration_beta_x), zoom_out_scale


def curvature_targets(x_star: float, x_t: float) -> tuple[float, float, float]:
    curvature_domain, sweep_endpoints, zoom_out_scale = calibration_domain(x_star, x_t)
    anchor_count = max(
        len(sweep_endpoints),
        LOCAL_CURVE_SAMPLES // LOCAL_MODEL_DASH_COUNT,
    )
    anchor_values = np.linspace(min(sweep_endpoints), max(sweep_endpoints), anchor_count)

    def bounds_for(anchor: float) -> tuple[float, float]:
        deltas = curvature_domain - anchor
        mask = np.abs(deltas) > ZERO_AXIS_EPSILON
        required_curvatures = (
            2
            * (
                objective(curvature_domain[mask])
                - objective_value(anchor)
                - gradient(anchor) * deltas[mask]
            )
            / deltas[mask] ** 2
        )
        return (
            min(hessian(anchor), float(np.min(required_curvatures))),
            max(hessian(anchor), float(np.max(required_curvatures))),
        )

    domain_min = min(float(curvature_domain[0]), float(curvature_domain[-1]))
    domain_max = max(float(curvature_domain[0]), float(curvature_domain[-1]))
    hessian_candidates = [domain_min, domain_max]
    if domain_min <= LOCAL_MODEL_CENTER <= domain_max:
        hessian_candidates.append(LOCAL_MODEL_CENTER)

    bounds = [bounds_for(anchor) for anchor in anchor_values]
    alpha_target = min(
        min(lower for lower, _ in bounds),
        min(hessian(candidate) for candidate in hessian_candidates),
    )
    beta_target = max(upper for _, upper in bounds)
    return float(alpha_target), float(beta_target), zoom_out_scale


def axis_y_range(
    x_t: float, f_t: float, g_t: float, h_t: float, alpha: float, beta: float
) -> tuple[float, float]:
    x_min, x_max, _ = LOCAL_X_RANGE
    xs = np.linspace(x_min, x_max, LOCAL_CURVE_SAMPLES)
    dx = xs - x_t
    fixed_lower = f_t + g_t * dx + alpha * dx**2 / 2
    fixed_upper = f_t + g_t * dx + beta * dx**2 / 2
    fixed_local = f_t + g_t * dx + h_t * dx**2 / 2
    y_min = (
        min(float(np.min(fixed_lower)), float(np.min(fixed_local)), float(np.min(objective(xs))))
        - LOCAL_Y_PADDING_BELOW
    )
    y_max = min(
        max(float(np.max(objective(xs))), float(np.max(fixed_upper))) + LOCAL_Y_PADDING_ABOVE,
        LOCAL_Y_MAX_CAP,
    )
    return y_min, y_max


def second_order_data() -> SecondOrderData:
    x_t = LOCAL_CURRENT_X
    f_t = objective_value(x_t)
    g_t = gradient(x_t)
    h_t = hessian(x_t)
    x_next = x_t - g_t / h_t
    y_next = float(quadratic_model(np.array([x_next]), x_t, h_t)[0])
    x_star = minimum_x()
    f_star = objective_value(x_star)
    sweep_offset = 0.2 * abs(x_star - x_t)
    alpha_sweep_x = float(x_star + sweep_offset)
    beta_sweep_x = float(x_star + sweep_offset)
    beta_left_sweep_x = x_t - 0.2
    alpha_target, beta_target, calibration_zoom = curvature_targets(x_star, x_t)
    y_min, y_max = axis_y_range(x_t, f_t, g_t, h_t, alpha_target, beta_target)
    x_min, x_max, x_step = LOCAL_X_RANGE
    return SecondOrderData(
        x_t=x_t,
        f_t=f_t,
        g_t=g_t,
        h_t=h_t,
        x_next=x_next,
        y_next=y_next,
        x_star=x_star,
        f_star=f_star,
        x_min=x_min,
        x_max=x_max,
        x_step=x_step,
        y_min=y_min,
        y_max=y_max,
        alpha_target=alpha_target,
        beta_target=beta_target,
        alpha_sweep_x=alpha_sweep_x,
        beta_sweep_x=beta_sweep_x,
        beta_left_sweep_x=beta_left_sweep_x,
        zoom_out_scale=2 * calibration_zoom,
    )


def color_formula(mob: VMobject) -> VMobject:
    _color_text_parts(mob, FORMULA_COLORS)
    return mob


def style_panel(mob: VMobject, *, z_index: int = LAYER_MARKERS) -> VMobject:
    mob.set_fill(C_PANEL_SOFT, opacity=PANEL_FILL_OPACITY)
    mob.set_stroke(C_FRAME, width=PANEL_STROKE_WIDTH, opacity=PANEL_STROKE_OPACITY)
    mob.set_z_index(z_index)
    return mob


def definition_template() -> TexTemplate:
    template = TexTemplate()
    template.add_to_preamble(r"\usepackage{amsthm}")
    template.add_to_preamble(r"\theoremstyle{definition}\newtheorem*{definition}{Definition}")
    return template


def definition_page(
    title: str,
    relation: str,
    region: Region,
    *,
    tex_template: TexTemplate,
) -> TexPage:
    return TexPage(
        rf"\begin{{definition}}[{title}]"
        r"\["
        rf"{relation}"
        r"\]"
        r"\end{definition}",
        page_width=region,
        buff=SMALL_BUFF,
        tex_template=tex_template,
        font_size=get_active_theme().typography.caption,
    )


def definition_panel(definitions: VGroup) -> SurroundingRectangle:
    return style_panel(
        SurroundingRectangle(
            definitions,
            buff=SMALL_BUFF,
            corner_radius=PANEL_CORNER_RADIUS,
        )
    )
