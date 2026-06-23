"""Second Order Approximation slide."""

from __future__ import annotations

from manim import RoundedRectangle, SurroundingRectangle, Transform, Unwrite
from simplex.engine.region import Region

from slides.helpers.figure_helpers import *
from slides.helpers.reminders import ReminderStack
from slides.helpers.style import (
    C_PANEL_SOFT,
    PANEL_CORNER_RADIUS,
    PANEL_FILL_OPACITY,
    PANEL_STROKE_OPACITY,
    PANEL_STROKE_WIDTH,
)


class SecondOrderApproximation(Slide):
    """Taylor's local quadratic model and the Newton displacement."""

    def construct(self) -> None:
        title = _start_slide(self, "Second-order approximation")
        left, right = _split_weighted(self.region, [1.72, 1.0])

        alpha_anchor = VT(LOCAL_CURRENT_X)
        beta_anchor = VT(LOCAL_CURRENT_X)
        view_scale = VT(1.0)
        center = LOCAL_MODEL_CENTER
        x_t = LOCAL_CURRENT_X

        def f(xs: FloatArray) -> FloatArray:
            return (
                LOCAL_QUARTIC_WEIGHT * (xs - center) ** 4
                + LOCAL_QUADRATIC_WEIGHT * (xs - LOCAL_QUADRATIC_CENTER) ** 2
                + LOCAL_VALUE_OFFSET
            )

        def grad(x: float) -> float:
            return 4 * LOCAL_QUARTIC_WEIGHT * (x - center) ** 3 + 2 * LOCAL_QUADRATIC_WEIGHT * (
                x - LOCAL_QUADRATIC_CENTER
            )

        def hess(x: float) -> float:
            return 12 * LOCAL_QUARTIC_WEIGHT * (x - center) ** 2 + 2 * LOCAL_QUADRATIC_WEIGHT

        def f_scalar(x: float) -> float:
            return float(f(np.array([x]))[0])

        def model_values(values: FloatArray, anchor: float, curvature: float) -> FloatArray:
            return (
                f_scalar(anchor)
                + grad(anchor) * (values - anchor)
                + 0.5 * curvature * (values - anchor) ** 2
            )

        f_t = float(f(np.array([x_t]))[0])
        g_t = grad(x_t)
        h_t = hess(x_t)
        x_next = x_t - g_t / h_t
        y_next = float(model_values(np.array([x_next]), x_t, h_t)[0])
        roots = np.roots(
            [
                4 * LOCAL_QUARTIC_WEIGHT,
                -12 * LOCAL_QUARTIC_WEIGHT * center,
                12 * LOCAL_QUARTIC_WEIGHT * center**2 + 2 * LOCAL_QUADRATIC_WEIGHT,
                -(4 * LOCAL_QUARTIC_WEIGHT * center**3)
                - 2 * LOCAL_QUADRATIC_WEIGHT * LOCAL_QUADRATIC_CENTER,
            ]
        )
        x_star = min(
            (root.real for root in roots if abs(root.imag) < ZERO_AXIS_EPSILON),
            key=lambda x: f(np.array([x]))[0],
        )
        f_star = float(f(np.array([x_star]))[0])
        x_min, x_max, x_step = LOCAL_X_RANGE
        base_x_span = x_max - x_min
        calibration_alpha_sweep_x = float(x_star + 2 * abs(x_star - x_t))
        calibration_beta_sweep_x = float(x_star + abs(x_star - x_t))
        calibration_view_sweep_x = max(calibration_alpha_sweep_x, calibration_beta_sweep_x)
        alpha_sweep_x = float(x_star + 5)
        beta_sweep_x = float(x_star + 8)
        beta_left_sweep_x = x_t - 3
        zoom_anchor_x = float(x_star)
        zoom_anchor_x_ratio = (zoom_anchor_x - x_min) / base_x_span
        zoom_padding = x_step / 2
        calibration_zoom_out_scale = max(
            1.0,
            (calibration_view_sweep_x + zoom_padding - zoom_anchor_x)
            / max(x_max - zoom_anchor_x, ZERO_AXIS_EPSILON),
            (zoom_anchor_x - min(x_min, x_t) + zoom_padding)
            / max(zoom_anchor_x - x_min, ZERO_AXIS_EPSILON),
        )
        zoom_out_scale = 2 * calibration_zoom_out_scale
        calibration_zoom_x_span = base_x_span * calibration_zoom_out_scale
        calibration_zoom_x_min = zoom_anchor_x - zoom_anchor_x_ratio * calibration_zoom_x_span
        calibration_zoom_x_max = calibration_zoom_x_min + calibration_zoom_x_span
        curvature_domain = np.linspace(
            min(
                x_min,
                calibration_zoom_x_min,
                x_t,
                calibration_alpha_sweep_x,
                calibration_beta_sweep_x,
            ),
            max(
                x_max,
                calibration_zoom_x_max,
                x_t,
                calibration_alpha_sweep_x,
                calibration_beta_sweep_x,
            ),
            LOCAL_CURVE_SAMPLES,
        )
        sweep_endpoints = (x_t, calibration_alpha_sweep_x, calibration_beta_sweep_x)
        curvature_anchor_count = max(
            len(sweep_endpoints),
            LOCAL_CURVE_SAMPLES // LOCAL_MODEL_DASH_COUNT,
        )
        curvature_anchor_values = np.linspace(
            min(sweep_endpoints),
            max(sweep_endpoints),
            curvature_anchor_count,
        )

        def curvature_bounds_for_anchor(anchor: float) -> tuple[float, float]:
            deltas = curvature_domain - anchor
            mask = np.abs(deltas) > ZERO_AXIS_EPSILON
            required_curvatures = (
                2
                * (f(curvature_domain[mask]) - f_scalar(anchor) - grad(anchor) * deltas[mask])
                / deltas[mask] ** 2
            )
            return (
                min(hess(anchor), float(np.min(required_curvatures))),
                max(hess(anchor), float(np.max(required_curvatures))),
            )

        hessian_domain_min = min(
            float(curvature_domain[0]),
            float(curvature_domain[-1]),
        )
        hessian_domain_max = max(
            float(curvature_domain[0]),
            float(curvature_domain[-1]),
        )
        hessian_candidates = [hessian_domain_min, hessian_domain_max]
        if hessian_domain_min <= center <= hessian_domain_max:
            hessian_candidates.append(center)
        curvature_bounds = [
            curvature_bounds_for_anchor(anchor) for anchor in curvature_anchor_values
        ]
        alpha_target = float(
            min(
                min(lower_bound for lower_bound, _ in curvature_bounds),
                min(hess(candidate) for candidate in hessian_candidates),
            )
        )
        beta_target = float(max(upper_bound for _, upper_bound in curvature_bounds))
        alpha = VT(alpha_target)
        beta = VT(beta_target)
        xs = np.linspace(x_min, x_max, LOCAL_CURVE_SAMPLES)
        dx = xs - x_t
        fixed_lower = f_t + g_t * dx + 0.5 * alpha_target * dx**2
        fixed_upper = f_t + g_t * dx + 0.5 * beta_target * dx**2
        fixed_local = f_t + g_t * dx + 0.5 * h_t * dx**2
        y_min = (
            min(float(np.min(fixed_lower)), float(np.min(fixed_local)), float(np.min(f(xs))))
            - LOCAL_Y_PADDING_BELOW
        )
        y_max = min(
            max(float(np.max(f(xs))), float(np.max(fixed_upper))) + LOCAL_Y_PADDING_ABOVE,
            LOCAL_Y_MAX_CAP,
        )

        grid_row_count = max(1, round(LOCAL_AXIS_Y_LENGTH / MED_LARGE_BUFF))
        grid_col_count = max(1, round(grid_row_count * LOCAL_AXIS_X_LENGTH / LOCAL_AXIS_Y_LENGTH))
        grid_cell_size = LOCAL_AXIS_Y_LENGTH / grid_row_count
        axes = _make_axes(
            (x_min, x_max, x_step),
            (y_min, y_max, 1),
            x_length=grid_col_count * grid_cell_size,
            y_length=grid_row_count * grid_cell_size,
        )
        axes.set_opacity(0)
        frame = _plot_frame(axes)
        frame.set_fill(C_PANEL_DEEP, opacity=LOCAL_PANEL_FILL_OPACITY)
        frame.set_stroke(
            C_FRAME,
            width=LOCAL_PANEL_STROKE_WIDTH,
            opacity=LOCAL_PANEL_STROKE_OPACITY,
        )
        bottom_axis = Line(frame.get_corner(DL), frame.get_corner(DR))
        bottom_axis.set_stroke(
            C_MUTED,
            width=LOCAL_BOTTOM_AXIS_STROKE_WIDTH,
            opacity=LOCAL_BOTTOM_AXIS_OPACITY,
        )
        base_y_span = y_max - y_min
        zoom_anchor_y = f_star
        zoom_anchor_y_ratio = (zoom_anchor_y - y_min) / base_y_span

        def view_scale_factor() -> float:
            return max(float(~view_scale), ZERO_AXIS_EPSILON)

        def view_limits() -> tuple[float, float, float, float]:
            scale = view_scale_factor()
            x_span = base_x_span * scale
            y_span = base_y_span * scale
            view_x_min = zoom_anchor_x - zoom_anchor_x_ratio * x_span
            view_y_min = zoom_anchor_y - zoom_anchor_y_ratio * y_span
            return (
                view_x_min,
                view_x_min + x_span,
                view_y_min,
                view_y_min + y_span,
            )

        def data_point(x: float, y: float) -> FloatArray:
            view_x_min, view_x_max, view_y_min, view_y_max = view_limits()
            x_ratio = (x - view_x_min) / (view_x_max - view_x_min)
            y_ratio = (y - view_y_min) / (view_y_max - view_y_min)
            return (
                frame.get_corner(DL)
                + RIGHT * (frame.width * x_ratio)
                + UP * (frame.height * y_ratio)
            )

        grid_x_step = base_x_span / grid_col_count
        grid_y_step = base_y_span / grid_row_count

        def grid_values(anchor: float, step: float, lower: float, upper: float) -> FloatArray:
            start = np.floor((lower - anchor) / step)
            stop = np.ceil((upper - anchor) / step)
            return anchor + np.arange(start, stop + 1) * step

        def update_graph_grid(mob: VMobject) -> None:
            view_x_min, view_x_max, view_y_min, view_y_max = view_limits()
            grid = VGroup()
            for x_value in grid_values(x_min, grid_x_step, view_x_min, view_x_max):
                line = Line(
                    data_point(float(x_value), view_y_min),
                    data_point(float(x_value), view_y_max),
                )
                line.set_stroke(
                    C_GRID,
                    width=GRID_LINE_STROKE_WIDTH,
                    opacity=GRID_LINE_OPACITY,
                )
                grid.add(line)
            for y_value in grid_values(y_min, grid_y_step, view_y_min, view_y_max):
                line = Line(
                    data_point(view_x_min, float(y_value)),
                    data_point(view_x_max, float(y_value)),
                )
                line.set_stroke(
                    C_GRID,
                    width=GRID_LINE_STROKE_WIDTH,
                    opacity=GRID_LINE_OPACITY,
                )
                grid.add(line)
            grid.set_z_index(LAYER_CONTOUR)
            mob.become(grid)

        graph_grid = VGroup().set_z_index(LAYER_CONTOUR)
        update_graph_grid(graph_grid)

        def clipped_curve_points(
            xs: FloatArray,
            ys: FloatArray,
            *,
            y_min: float,
            y_max: float,
        ) -> list[FloatArray]:
            points: list[FloatArray] = []
            previous: tuple[float, float] | None = None
            for x_raw, y_raw in zip(xs, ys, strict=True):
                if not np.isfinite(y_raw):
                    previous = None
                    continue

                x_value = float(x_raw)
                y_value = float(y_raw)
                if previous is not None:
                    previous_x, previous_y = previous
                    y_delta = y_value - previous_y
                    if abs(y_delta) > ZERO_AXIS_EPSILON:
                        crossings: list[tuple[float, float, float]] = []
                        for boundary_y in (y_min, y_max):
                            ratio = (boundary_y - previous_y) / y_delta
                            if ZERO_AXIS_EPSILON < ratio < 1 - ZERO_AXIS_EPSILON:
                                x_boundary = previous_x + ratio * (x_value - previous_x)
                                crossings.append((ratio, x_boundary, boundary_y))
                        for _, x_boundary, boundary_y in sorted(crossings):
                            points.append(data_point(x_boundary, boundary_y))
                if y_min <= y_value <= y_max:
                    points.append(data_point(x_value, y_value))
                previous = (x_value, y_value)
            return points

        def curve_points(fn: Callable[[FloatArray], FloatArray]) -> list[FloatArray]:
            view_x_min, view_x_max, view_y_min, view_y_max = view_limits()
            visible_xs = np.linspace(view_x_min, view_x_max, LOCAL_CURVE_SAMPLES)
            ys = fn(visible_xs)
            points = clipped_curve_points(
                visible_xs,
                ys,
                y_min=view_y_min,
                y_max=view_y_max,
            )
            if len(points) < 2:
                clipped_y = float(np.clip(ys[0], view_y_min, view_y_max))
                points = [data_point(float(visible_xs[0]), clipped_y)]
                points.append(points[0] + RIGHT * POLYLINE_FALLBACK_STEP)
            return points

        local_model_dash_count = LOCAL_MODEL_DASH_COUNT
        vertical_guide_dash_count = max(2, LOCAL_MODEL_DASH_COUNT // 4)

        def update_curve(
            mob: VMobject,
            fn: Callable[[FloatArray], FloatArray],
            *,
            color: str,
            width: float,
            opacity: float = 1.0,
        ) -> None:
            mob.set_points_smoothly(curve_points(fn))
            mob.set_stroke(color, width=width, opacity=opacity)
            mob.set_z_index(LAYER_TRAJECTORY)

        def curve_for(
            fn: Callable[[FloatArray], FloatArray],
            *,
            color: str,
            width: float,
            opacity: float = 1.0,
        ) -> VMobject:
            curve = VMobject()
            update_curve(curve, fn, color=color, width=width, opacity=opacity)
            return curve

        def track_curve(
            mob: VMobject,
            fn: Callable[[FloatArray], FloatArray],
            *,
            color: str,
            width: float,
            opacity: float = 1.0,
        ) -> None:
            mob.add_updater(
                lambda tracked: update_curve(
                    tracked,
                    fn,
                    color=color,
                    width=width,
                    opacity=opacity,
                )
            )

        def dashed_curve_for(
            fn: Callable[[FloatArray], FloatArray],
            *,
            color: str,
            width: float,
        ) -> DashedVMobject:
            base_curve = curve_for(fn, color=color, width=width)
            curve = DashedVMobject(
                base_curve,
                num_dashes=local_model_dash_count,
                color=color,
            )
            curve.set_stroke(color, width=width)
            curve.set_z_index(LAYER_TRAJECTORY)
            return curve

        def update_dashed_curve(
            mob: VMobject,
            fn: Callable[[FloatArray], FloatArray],
            *,
            color: str,
            width: float,
        ) -> None:
            mob.become(dashed_curve_for(fn, color=color, width=width))

        def track_dashed_curve(
            mob: VMobject,
            fn: Callable[[FloatArray], FloatArray],
            *,
            color: str,
            width: float,
        ) -> None:
            mob.add_updater(
                lambda tracked: update_dashed_curve(
                    tracked,
                    fn,
                    color=color,
                    width=width,
                )
            )

        def local_model_values(values: FloatArray) -> FloatArray:
            return model_values(values, x_t, h_t)

        def lower_model_values(values: FloatArray) -> FloatArray:
            return model_values(values, float(~alpha_anchor), float(~alpha))

        def upper_model_values(values: FloatArray) -> FloatArray:
            return model_values(values, float(~beta_anchor), float(~beta))

        true_curve = curve_for(f, color=C_TEXT, width=LOCAL_CURVE_STROKE_WIDTH)
        local_model = dashed_curve_for(
            local_model_values,
            color=C_GREEN,
            width=LOCAL_MODEL_STROKE_WIDTH,
        )
        lower_model = curve_for(
            lower_model_values,
            color=C_BLUE,
            width=LOCAL_BOUND_STROKE_WIDTH,
            opacity=LOCAL_BOUND_OPACITY,
        )
        upper_model = curve_for(
            upper_model_values,
            color=C_ORANGE,
            width=LOCAL_BOUND_STROKE_WIDTH,
            opacity=LOCAL_BOUND_OPACITY,
        )
        local_marker_radius = frame.height * SECOND_ORDER_MARKER_FRAME_HEIGHT_RATIO

        def tracked_dot(
            x_fn: Callable[[], float],
            y_fn: Callable[[], float],
            *,
            color: str,
            radius: float,
        ) -> Dot:
            dot = Dot(data_point(x_fn(), y_fn()), color=color, radius=radius).set_z_index(
                LAYER_MARKERS
            )
            return dot

        def track_dot(dot: Dot, x_fn: Callable[[], float], y_fn: Callable[[], float]) -> None:
            dot.add_updater(lambda mob: mob.move_to(data_point(x_fn(), y_fn())))

        def update_bottom_tick(
            mob: VGroup,
            x_fn: Callable[[], float],
            *,
            color: str,
            tick_height: float,
        ) -> None:
            _, _, current_y_min, _ = view_limits()
            mob[0].put_start_and_end_on(
                data_point(x_fn(), current_y_min),
                data_point(x_fn(), current_y_min + tick_height),
            )
            mob[0].set_stroke(color, width=BOTTOM_TICK_STROKE_WIDTH)
            mob[1].scale_to_fit_height(TICK_LABEL_TEX_SCALE * mob[0].height)
            mob[1].next_to(mob[0], DOWN, buff=SMALL_BUFF / view_scale_factor())

        def bottom_tick(x_fn: Callable[[], float], label: str, *, color: str) -> VGroup:
            tick_height = BOTTOM_TICK_HEIGHT_RATIO * base_y_span
            group = VGroup(Line(), MathTex(label, color=color)).set_z_index(LAYER_MARKERS)
            update_bottom_tick(group, x_fn, color=color, tick_height=tick_height)
            return group

        def track_bottom_tick(mob: VGroup, x_fn: Callable[[], float], *, color: str) -> None:
            tick_height = BOTTOM_TICK_HEIGHT_RATIO * base_y_span
            mob.add_updater(
                lambda tracked: update_bottom_tick(
                    tracked,
                    x_fn,
                    color=color,
                    tick_height=tick_height,
                )
            )

        def label_next_to_dot(
            label: str,
            dot: Dot,
            *,
            direction: FloatArray,
            color: str,
            colors: dict[str, str] | None = None,
            buff: float | None = None,
        ) -> MathTex:
            mob = theme_math(label, color=color, typography="caption").set_z_index(LAYER_MARKERS)
            if colors is not None:
                color_substrings(mob, colors)
            if buff is None:
                mob.next_to(dot, direction)
            else:
                mob.next_to(dot, direction, buff=buff)
            return mob

        def track_label_next_to_dot(
            mob: MathTex,
            dot: Dot,
            direction: FloatArray,
            buff: float | None = None,
        ) -> None:
            def update_label(label_mob: MathTex) -> None:
                if buff is None:
                    label_mob.next_to(dot, direction)
                else:
                    label_mob.next_to(dot, direction, buff=buff)

            mob.add_updater(update_label)

        def update_vertical_guide(
            mob: VGroup,
            x_fn: Callable[[], float],
            *,
            color: str,
            opacity: float,
        ) -> None:
            _, _, current_y_min, current_y_max = view_limits()
            dash_span = (current_y_max - current_y_min) / (2 * len(mob) - 1)
            x_value = x_fn()
            for index, dash in enumerate(mob):
                y_start = current_y_min + 2 * index * dash_span
                dash.put_start_and_end_on(
                    data_point(x_value, y_start),
                    data_point(x_value, min(y_start + dash_span, current_y_max)),
                )
                dash.set_stroke(color, width=LOCAL_VERTICAL_GUIDE_STROKE_WIDTH, opacity=opacity)

        def vertical_guide(
            x_fn: Callable[[], float],
            *,
            color: str,
            opacity: float,
        ) -> VGroup:
            guide = VGroup(*(Line() for _ in range(vertical_guide_dash_count))).set_z_index(
                LAYER_TRAJECTORY
            )
            update_vertical_guide(guide, x_fn, color=color, opacity=opacity)
            return guide

        def track_vertical_guide(
            mob: VGroup,
            x_fn: Callable[[], float],
            *,
            color: str,
            opacity: float,
        ) -> None:
            mob.add_updater(
                lambda tracked: update_vertical_guide(
                    tracked,
                    x_fn,
                    color=color,
                    opacity=opacity,
                )
            )

        x_t_dot = tracked_dot(lambda: x_t, lambda: f_t, color=C_YELLOW, radius=local_marker_radius)
        newton_dot = tracked_dot(
            lambda: x_next, lambda: y_next, color=C_GREEN, radius=local_marker_radius
        )
        star_dot = tracked_dot(
            lambda: float(x_star),
            lambda: f_star,
            color=C_TEXT,
            radius=local_marker_radius * SECOND_ORDER_STAR_DOT_SCALE,
        )
        x_t_tick = bottom_tick(lambda: x_t, r"x_t", color=C_YELLOW)
        next_tick = bottom_tick(lambda: x_next, r"x_{t+1}", color=C_YELLOW)
        x_t_value = label_next_to_dot(
            r"f(x_t)",
            x_t_dot,
            direction=DL,
            color=C_TEXT,
            colors={r"x_t": C_YELLOW},
            buff=0,
        )
        star_label = label_next_to_dot(
            r"f(x^\star)",
            star_dot,
            direction=DR,
            color=C_TEXT,
            buff=0,
        )
        x_line = vertical_guide(
            lambda: x_t,
            color=C_YELLOW,
            opacity=LOCAL_CURRENT_GUIDE_OPACITY,
        )
        next_line = vertical_guide(
            lambda: x_next,
            color=C_GREEN,
            opacity=LOCAL_NEXT_GUIDE_OPACITY,
        )

        def bracket_geometry() -> tuple[float, float]:
            _, _, current_y_min, current_y_max = view_limits()
            y_span = current_y_max - current_y_min
            return (
                current_y_min + LOCAL_BRACKET_Y_RATIO * y_span,
                LOCAL_BRACKET_TICK_HEIGHT_RATIO * y_span,
            )

        delta_bracket = VGroup(Line(), Line(), Line())

        def update_delta_bracket(mob: VGroup) -> None:
            bracket_y, tick_height = bracket_geometry()
            segments = (
                (data_point(x_t, bracket_y), data_point(x_next, bracket_y)),
                (
                    data_point(x_t, bracket_y - LOCAL_BRACKET_HALF_TICK_SCALE * tick_height),
                    data_point(x_t, bracket_y + LOCAL_BRACKET_HALF_TICK_SCALE * tick_height),
                ),
                (
                    data_point(x_next, bracket_y - LOCAL_BRACKET_HALF_TICK_SCALE * tick_height),
                    data_point(x_next, bracket_y + LOCAL_BRACKET_HALF_TICK_SCALE * tick_height),
                ),
            )
            for line, (start, end) in zip(mob, segments, strict=True):
                line.put_start_and_end_on(start, end)
            mob.set_stroke(WHITE, width=LOCAL_BRACKET_STROKE_WIDTH)

        update_delta_bracket(delta_bracket)

        def track_delta_bracket(mob: VGroup) -> None:
            mob.add_updater(update_delta_bracket)

        delta_label = theme_math(
            r"\delta=-\nabla^2 f(x_t)^{-1}\nabla f(x_t)",
            color=WHITE,
            typography="caption",
        ).set_z_index(LAYER_MARKERS)
        color_substrings(delta_label, {r"x_t": C_YELLOW})

        def update_delta_label(mob: MathTex) -> None:
            bracket_y, _ = bracket_geometry()
            _, _, current_y_min, current_y_max = view_limits()
            mob.scale_to_fit_width(max(delta_bracket[0].width - 2 * SMALL_BUFF, SMALL_BUFF))
            mob.move_to(
                data_point(
                    0.5 * (x_t + x_next),
                    bracket_y + LOCAL_BRACKET_LABEL_Y_RATIO * (current_y_max - current_y_min),
                )
            )

        update_delta_label(delta_label)

        def track_delta_label(mob: MathTex) -> None:
            mob.add_updater(update_delta_label)

        def alpha_minimum() -> tuple[float, float]:
            value = max(float(~alpha), ZERO_AXIS_EPSILON)
            anchor = float(~alpha_anchor)
            x_alpha = anchor - grad(anchor) / value
            y_alpha = float(model_values(np.array([x_alpha]), anchor, value)[0])
            return float(x_alpha), float(y_alpha)

        def beta_minimum() -> tuple[float, float]:
            value = max(float(~beta), ZERO_AXIS_EPSILON)
            anchor = float(~beta_anchor)
            x_beta = anchor - grad(anchor) / value
            y_beta = float(model_values(np.array([x_beta]), anchor, value)[0])
            return float(x_beta), float(y_beta)

        def update_x_marker(
            mob: VGroup,
            point_fn: Callable[[], tuple[float, float]],
            *,
            color: str,
        ) -> None:
            x, y = point_fn()
            center_point = data_point(x, y)
            size = frame.height * X_MARKER_FRAME_HEIGHT_RATIO / view_scale_factor()
            endpoints = (
                (
                    center_point + (LEFT + DOWN) * size / 2,
                    center_point + (RIGHT + UP) * size / 2,
                ),
                (
                    center_point + (LEFT + UP) * size / 2,
                    center_point + (RIGHT + DOWN) * size / 2,
                ),
            )
            for line, (start, end) in zip(mob, endpoints, strict=True):
                line.put_start_and_end_on(start, end)
            mob.set_stroke(color, width=X_MARKER_STROKE_WIDTH, opacity=X_MARKER_OPACITY)

        def x_marker(point_fn: Callable[[], tuple[float, float]], *, color: str) -> VGroup:
            marker = VGroup(Line(), Line()).set_z_index(LAYER_MARKERS)
            update_x_marker(marker, point_fn, color=color)
            return marker

        def track_x_marker(
            mob: VGroup,
            point_fn: Callable[[], tuple[float, float]],
            *,
            color: str,
        ) -> None:
            mob.add_updater(lambda tracked: update_x_marker(tracked, point_fn, color=color))

        def color_formula(mob: VMobject) -> VMobject:
            _color_text_parts(
                mob,
                {
                    r"x_t": C_YELLOW,
                    r"x_{t+1}": C_YELLOW,
                    r"\alpha": C_BLUE,
                    r"\beta": C_ORANGE,
                    r"\delta": C_GREEN,
                    r"\lambda_{\min}": C_BLUE,
                    r"\lambda_{\max}": C_ORANGE,
                    r"\kappa": C_PURPLE,
                },
            )
            return mob

        panel_background_color = C_PANEL_SOFT
        alpha_eigen_note = color_formula(
            theme_math(
                r"\lambda_{\min}(\nabla^2 f(x))=\alpha",
                color=C_TEXT,
                typography="caption",
            )
        )
        beta_eigen_note = color_formula(
            theme_math(
                r"\lambda_{\max}(\nabla^2 f(x))=\beta",
                color=C_TEXT,
                typography="caption",
            )
        )
        kappa_note = color_formula(
            theme_math(
                r"\kappa=\frac{\beta}{\alpha}",
                color=C_TEXT,
                typography="caption",
            )
        )
        curvature_reminders = ReminderStack(
            [alpha_eigen_note],
            width=left.width - 2 * SMALL_BUFF,
            max_height=frame.height * LOCAL_LEGEND_FRAME_HEIGHT_RATIO,
            orientation="horizontal",
            adaptive=True,
            fill_color=panel_background_color,
        )
        left.place(curvature_reminders, DL, buff=SMALL_BUFF)
        left.update(bottom=curvature_reminders)

        alpha_marker = x_marker(alpha_minimum, color=C_BLUE)
        beta_marker = x_marker(beta_minimum, color=C_ORANGE)
        plot = VGroup(
            frame,
            graph_grid,
            bottom_axis,
            axes,
            true_curve,
            local_model,
            lower_model,
            upper_model,
            x_line,
            next_line,
            x_t_dot,
            newton_dot,
            star_dot,
            x_t_tick,
            next_tick,
            x_t_value,
            star_label,
            delta_bracket,
            delta_label,
            alpha_marker,
            beta_marker,
        )
        left.scale_and_place(plot, buff=MED_SMALL_BUFF)
        right.update(left=plot)

        def plot_legend_entry(
            tex: str, *, color: str, width: float, dashed: bool = False
        ) -> VGroup:
            label = color_formula(theme_math(tex, color=C_TEXT, typography="caption"))
            sample_width = frame.width * LOCAL_LEGEND_SAMPLE_FRAME_WIDTH_RATIO
            sample = (
                DashedLine(ORIGIN, RIGHT * sample_width, color=color)
                if dashed
                else Line(ORIGIN, RIGHT * sample_width)
            )
            sample.set_stroke(color, width=width)
            return VGroup(sample, label).arrange(RIGHT, buff=SMALL_BUFF)

        f_legend = plot_legend_entry(r"f(x)", color=C_TEXT, width=LOCAL_CURVE_STROKE_WIDTH)
        hessian_legend = plot_legend_entry(
            r"f(x_t)"
            r"+\left\langle\nabla f(x_t),x-x_t\right\rangle"
            r"+\frac{1}{2}(x-x_t)^\top\nabla^2 f(x_t)(x-x_t)",
            color=C_GREEN,
            width=LOCAL_MODEL_STROKE_WIDTH,
            dashed=True,
        )
        alpha_legend = plot_legend_entry(
            r"f(x_t)"
            r"+\left\langle\nabla f(x_t),x-x_t\right\rangle"
            r"+\frac{\alpha}{2}\left\|x-x_t\right\|_2^2",
            color=C_BLUE,
            width=LOCAL_BOUND_STROKE_WIDTH,
        )
        beta_legend = plot_legend_entry(
            r"f(x_t)"
            r"+\left\langle\nabla f(x_t),x-x_t\right\rangle"
            r"+\frac{\beta}{2}\left\|x-x_t\right\|_2^2",
            color=C_ORANGE,
            width=LOCAL_BOUND_STROKE_WIDTH,
        )
        plot_legend = VGroup(
            f_legend,
            hessian_legend,
            alpha_legend,
            beta_legend,
        ).arrange(DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
        legend_box_buff = SMALL_BUFF
        legend_frame_buff = SMALL_BUFF
        legend_available_width = frame.width - 2 * (legend_box_buff + legend_frame_buff)
        legend_scale = min(
            legend_available_width / plot_legend.width,
            frame.height * LOCAL_LEGEND_FRAME_HEIGHT_RATIO / plot_legend.height,
            1.0,
        )
        plot_legend.scale(legend_scale)
        plot_legend.move_to(
            frame.get_corner(UR)
            + LEFT * (plot_legend.width / 2 + legend_box_buff + legend_frame_buff)
            + DOWN * (plot_legend.height / 2 + legend_box_buff + legend_frame_buff)
        )

        def legend_background_height(entry_count: int) -> float:
            visible_entries = plot_legend.submobjects[:entry_count]
            top = max(entry.get_top()[1] for entry in visible_entries)
            bottom = min(entry.get_bottom()[1] for entry in visible_entries)
            return float(top - bottom + 2 * legend_box_buff)

        legend_background_heights = tuple(
            legend_background_height(entry_count)
            for entry_count in range(1, len(plot_legend.submobjects) + 1)
        )
        legend_background_top = plot_legend.get_top() + UP * legend_box_buff
        plot_legend_background = RoundedRectangle(
            corner_radius=PANEL_CORNER_RADIUS,
            width=plot_legend.width + 2 * legend_box_buff,
            height=legend_background_heights[0],
        )
        plot_legend_background.set_fill(panel_background_color, opacity=PANEL_FILL_OPACITY)
        plot_legend_background.set_stroke(
            C_FRAME,
            width=PANEL_STROKE_WIDTH,
            opacity=PANEL_STROKE_OPACITY,
        )
        plot_legend_background.move_to(legend_background_top, aligned_edge=UP)
        plot_legend_background.set_z_index(LAYER_MARKERS)
        plot_legend.set_z_index(LAYER_MARKERS + 1)
        plot.add(plot_legend_background, plot_legend)

        def grow_legend_background(entry_count: int):
            return plot_legend_background.animate.stretch_to_fit_height(
                legend_background_heights[entry_count - 1]
            ).move_to(legend_background_top, aligned_edge=UP)

        hessian_equation = color_formula(
            theme_math(
                r"f(x_t+\delta)\approx f(x_t)"
                r"+\nabla f(x_t)^\top\delta"
                r"+\frac{1}{2}\delta^\top\nabla^2 f(x_t)\delta",
            )
        ).set_z_index(LAYER_MARKERS)
        right.scale_and_place(hessian_equation)
        hessian_region, definition_region = _split_rows(right, [1, 5])
        hessian_equation_top = hessian_equation.copy()
        hessian_region.scale_and_place(hessian_equation_top, buff=MED_SMALL_BUFF)

        definition_template = TexTemplate()
        definition_template.add_to_preamble(r"\usepackage{amsthm}")
        definition_template.add_to_preamble(
            r"\theoremstyle{definition}\newtheorem*{definition}{Definition}"
        )
        definition_font_size = get_active_theme().typography.caption
        alpha_definition = TexPage(
            r"\begin{definition}[$\alpha$-strong convexity]"
            r"\["
            r"f(x)\ge f(x_t)+\left\langle\nabla f(x_t),x-x_t\right\rangle"
            r"+\frac{\alpha}{2}\left\|x-x_t\right\|_2^2"
            r"\]"
            r"\end{definition}",
            page_width=definition_region,
            buff=SMALL_BUFF,
            tex_template=definition_template,
            font_size=definition_font_size,
        )
        beta_definition = TexPage(
            r"\begin{definition}[$\beta$-smoothness]"
            r"\["
            r"f(x)\le f(x_t)+\left\langle\nabla f(x_t),x-x_t\right\rangle"
            r"+\frac{\beta}{2}\left\|x-x_t\right\|_2^2"
            r"\]"
            r"\end{definition}",
            page_width=definition_region,
            buff=SMALL_BUFF,
            tex_template=definition_template,
            font_size=definition_font_size,
        )
        color_substrings(
            alpha_definition,
            {r"x_t": C_YELLOW, r"\alpha": C_BLUE},
            probe_class=MathTex,
        )
        color_substrings(
            beta_definition,
            {r"x_t": C_YELLOW, r"\beta": C_ORANGE},
            probe_class=MathTex,
        )
        definition_stack = VGroup(alpha_definition, beta_definition).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=MED_LARGE_BUFF,
        )
        definition_background = SurroundingRectangle(
            definition_stack,
            buff=SMALL_BUFF,
            corner_radius=PANEL_CORNER_RADIUS,
        )
        definition_background.set_fill(panel_background_color, opacity=PANEL_FILL_OPACITY)
        definition_background.set_stroke(
            C_FRAME,
            width=PANEL_STROKE_WIDTH,
            opacity=PANEL_STROKE_OPACITY,
        )
        definition_background.set_z_index(LAYER_MARKERS)
        definition_stack.set_z_index(LAYER_MARKERS + 1)
        definition_panel = VGroup(definition_background, definition_stack)
        definition_region.scale_and_place(
            definition_panel,
            buff=MED_SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )

        proof_separator = Line(
            hessian_region.get_corner(DL) + RIGHT * SMALL_BUFF,
            hessian_region.get_corner(DR) + LEFT * SMALL_BUFF,
        )
        proof_separator.set_stroke(
            C_FRAME,
            width=LOCAL_PANEL_STROKE_WIDTH,
            opacity=LOCAL_PANEL_STROKE_OPACITY,
        )
        proof_separator.set_z_index(LAYER_MARKERS + 1)

        derivative_equation = color_formula(
            theme_math(
                r"\nabla_{",
                r"\delta",
                r"} f_{\text{apx}}(\delta)="
                r"\nabla f(x_t)+\nabla^2 f(x_t)\delta",
            )
        )
        derivative_equation[1].set_color(C_GREEN)
        newton_delta_equation = color_formula(
            theme_math(
                r"\delta=-\frac{1}{\nabla^2 f(x_t)}\nabla f(x_t)",
            )
        )
        delta_relation_note = color_formula(theme_math(r"*\ \delta=x_{t+1}-x_t"))
        first_down_arrow = theme_math(r"\Downarrow", color=C_TEXT)
        second_down_arrow = theme_math(r"\Downarrow", color=C_TEXT)
        newton_update = color_formula(
            theme_math(
                r"x_{t+1}=x_t-\frac{1}{\nabla^2 f(x_t)}\nabla f(x_t)",
            )
        )
        derivation_stack = VGroup(
            derivative_equation,
            first_down_arrow,
            newton_delta_equation,
            second_down_arrow,
            newton_update,
        ).arrange(DOWN, buff=MED_SMALL_BUFF)
        definition_region.scale_and_place(
            derivation_stack,
            buff=MED_SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )
        derivation_stack.set_z_index(LAYER_MARKERS + 1)
        delta_relation_note.scale(derivative_equation.height / delta_relation_note.height * 2 / 3)
        definition_region.place(delta_relation_note, DR, buff=SMALL_BUFF)
        delta_relation_note.set_z_index(LAYER_MARKERS + 1)

        newton_frame = SurroundingRectangle(
            newton_update,
            color=C_YELLOW,
            buff=SMALL_BUFF,
            corner_radius=PANEL_CORNER_RADIUS,
        )
        newton_frame.set_stroke(C_YELLOW, width=PANEL_STROKE_WIDTH)
        newton_frame.set_z_index(LAYER_MARKERS)
        highlighted_newton = VGroup(newton_update, newton_frame)
        highlighted_newton_top = highlighted_newton.copy()
        top_proof_region, middle_proof_region, bottom_proof_region = (
            definition_region.split_regions(
                DOWN,
                3,
            )
        )
        convergence_region = Region(
            top=middle_proof_region.top,
            bottom=bottom_proof_region.bottom,
            left=definition_region.left,
            right=definition_region.right,
        )
        top_proof_region.scale_and_place(
            highlighted_newton_top,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )

        convergence_note = TexPage(
            r"Under a few assumptions, Newton's method converges after:"
            r"\[{\small O\left(\log\log\frac{1}{\epsilon}\right)}\]",
            page_width=convergence_region,
            buff=SMALL_BUFF,
            font_size=get_active_theme().typography.caption * 3 / 2,
        )
        color_substrings(convergence_note, {r"\epsilon": C_YELLOW}, probe_class=MathTex)
        convergence_region.scale_and_place(
            convergence_note,
            buff=MED_SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )
        convergence_note.set_z_index(LAYER_MARKERS + 1)

        computer_fire = ImageMobject("assets/figures/computer_fire.jpg")
        computer_fire.rotate(-PI / 18)
        computer_fire.scale_to_fit_width(right.width / 2)
        right.place(computer_fire, DR, buff=SMALL_BUFF)
        computer_fire.set_z_index(LAYER_MARKERS + 3)

        plot_view_mobjects = (
            true_curve,
            local_model,
            lower_model,
            upper_model,
            x_t_dot,
            newton_dot,
            star_dot,
            x_t_tick,
            next_tick,
            x_t_value,
            star_label,
            x_line,
            next_line,
            delta_bracket,
            delta_label,
            alpha_marker,
            beta_marker,
            graph_grid,
        )

        def track_plot_view() -> None:
            track_curve(true_curve, f, color=C_TEXT, width=LOCAL_CURVE_STROKE_WIDTH)
            track_dashed_curve(
                local_model,
                local_model_values,
                color=C_GREEN,
                width=LOCAL_MODEL_STROKE_WIDTH,
            )
            track_curve(
                lower_model,
                lower_model_values,
                color=C_BLUE,
                width=LOCAL_BOUND_STROKE_WIDTH,
                opacity=LOCAL_BOUND_OPACITY,
            )
            track_curve(
                upper_model,
                upper_model_values,
                color=C_ORANGE,
                width=LOCAL_BOUND_STROKE_WIDTH,
                opacity=LOCAL_BOUND_OPACITY,
            )
            track_dot(x_t_dot, lambda: x_t, lambda: f_t)
            track_dot(newton_dot, lambda: x_next, lambda: y_next)
            track_dot(star_dot, lambda: float(x_star), lambda: f_star)
            track_bottom_tick(x_t_tick, lambda: x_t, color=C_YELLOW)
            track_bottom_tick(next_tick, lambda: x_next, color=C_YELLOW)
            track_label_next_to_dot(x_t_value, x_t_dot, DL, 0)
            track_label_next_to_dot(star_label, star_dot, DR, 0)
            track_vertical_guide(
                x_line,
                lambda: x_t,
                color=C_YELLOW,
                opacity=LOCAL_CURRENT_GUIDE_OPACITY,
            )
            track_vertical_guide(
                next_line,
                lambda: x_next,
                color=C_GREEN,
                opacity=LOCAL_NEXT_GUIDE_OPACITY,
            )
            track_delta_bracket(delta_bracket)
            track_delta_label(delta_label)
            track_x_marker(alpha_marker, alpha_minimum, color=C_BLUE)
            track_x_marker(beta_marker, beta_minimum, color=C_ORANGE)
            graph_grid.add_updater(update_graph_grid)

        def clear_plot_view_updaters() -> None:
            for mob in plot_view_mobjects:
                mob.clear_updaters()

        def refresh_plot_view() -> None:
            for mob in plot_view_mobjects:
                mob.update(0)

        self.play(
            Write(title),
            Write(frame),
            Write(graph_grid),
            Write(bottom_axis),
            Write(plot_legend_background),
            Write(true_curve),
            Write(local_model),
            Write(newton_dot),
            Write(star_dot),
            Write(star_label),
            Write(x_line),
            Write(x_t_dot),
            Write(x_t_tick),
            Write(x_t_value),
            Write(f_legend),
            grow_legend_background(2),
            Write(hessian_legend),
            Write(hessian_equation),
        )
        self.next_slide()

        self.play(
            Transform(hessian_equation, hessian_equation_top),
            Create(proof_separator),
            Write(lower_model),
            Write(alpha_marker),
            Write(definition_background),
            Write(alpha_definition),
            Write(curvature_reminders.frame),
            Write(VGroup(*curvature_reminders.entries)),
            grow_legend_background(3),
            Write(alpha_legend),
        )
        self.next_slide()

        self.play(
            Write(upper_model),
            Write(beta_marker),
            Write(beta_definition),
            curvature_reminders.animate_add_many([beta_eigen_note, kappa_note]),
            grow_legend_background(4),
            Write(beta_legend),
        )
        self.next_slide()

        track_plot_view()
        self.play(view_scale @ zoom_out_scale)
        refresh_plot_view()
        clear_plot_view_updaters()
        self.next_slide()

        track_curve(
            lower_model,
            lower_model_values,
            color=C_BLUE,
            width=LOCAL_BOUND_STROKE_WIDTH,
            opacity=LOCAL_BOUND_OPACITY,
        )
        track_x_marker(alpha_marker, alpha_minimum, color=C_BLUE)
        self.play(alpha_anchor @ alpha_sweep_x, run_time=2)
        self.play(alpha_anchor @ x_t, run_time=2)
        self.next_slide()

        for mob in (lower_model, alpha_marker):
            mob.clear_updaters()
        track_curve(
            upper_model,
            upper_model_values,
            color=C_ORANGE,
            width=LOCAL_BOUND_STROKE_WIDTH,
            opacity=LOCAL_BOUND_OPACITY,
        )
        track_x_marker(beta_marker, beta_minimum, color=C_ORANGE)
        self.play(beta_anchor @ beta_sweep_x, run_time=2)
        self.play(beta_anchor @ beta_left_sweep_x, run_time=2)
        self.play(beta_anchor @ x_t, run_time=2)
        self.next_slide()

        for mob in (upper_model, beta_marker):
            mob.clear_updaters()

        self.play(
            Unwrite(definition_background),
            Unwrite(alpha_definition),
            Unwrite(beta_definition),
        )
        self.next_slide()

        track_plot_view()
        self.play(view_scale @ 1)
        refresh_plot_view()
        clear_plot_view_updaters()
        self.play(Write(derivative_equation))
        self.next_slide()

        self.play(Write(first_down_arrow), Write(newton_delta_equation))
        self.next_slide()

        self.play(Write(second_down_arrow), Write(newton_update), Write(delta_relation_note))
        self.next_slide()

        self.play(Write(next_line), Write(next_tick), Write(delta_bracket), Write(delta_label))
        self.play(Create(newton_frame))
        self.next_slide()

        self.play(
            Unwrite(derivative_equation),
            Unwrite(first_down_arrow),
            Unwrite(newton_delta_equation),
            Unwrite(second_down_arrow),
            Unwrite(delta_relation_note),
        )
        self.play(
            Transform(highlighted_newton, highlighted_newton_top),
        )
        self.play(Write(convergence_note))
        self.play(FadeIn(computer_fire))
        self.next_slide()

        self.clear_scene()
        self.next_slide()
