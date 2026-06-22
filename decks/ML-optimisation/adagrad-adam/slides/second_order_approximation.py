"""Second Order Approximation slide."""

from __future__ import annotations

from slides.helpers.figure_helpers import *


class SecondOrderApproximation(Slide):
    """Taylor's local quadratic model and the Newton displacement."""

    def construct(self) -> None:
        title = _start_slide(self, "Second-order approximation")
        left, right = _split_weighted(self.region, [1.72, 1.0])

        alpha = VT(LOCAL_ALPHA_INITIAL)
        beta = VT(LOCAL_ALPHA_INITIAL)
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
            return (
                4 * LOCAL_QUARTIC_WEIGHT * (x - center) ** 3
                + 2 * LOCAL_QUADRATIC_WEIGHT * (x - LOCAL_QUADRATIC_CENTER)
            )

        def hess(x: float) -> float:
            return 12 * LOCAL_QUARTIC_WEIGHT * (x - center) ** 2 + 2 * LOCAL_QUADRATIC_WEIGHT

        def f_scalar(x: float) -> float:
            return float(f(np.array([x]))[0])

        def model_values(values: FloatArray, anchor: float, curvature: float) -> FloatArray:
            return f_scalar(anchor) + grad(anchor) * (values - anchor) + 0.5 * curvature * (values - anchor) ** 2

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
        xs = np.linspace(x_min, x_max, LOCAL_CURVE_SAMPLES)
        dx = xs - x_t
        fixed_lower = f_t + g_t * dx + 0.5 * LOCAL_ALPHA_INITIAL * dx**2
        fixed_upper = f_t + g_t * dx + 0.5 * LOCAL_BETA_INITIAL * dx**2
        fixed_local = f_t + g_t * dx + 0.5 * h_t * dx**2
        y_min = (
            min(float(np.min(fixed_lower)), float(np.min(fixed_local)), float(np.min(f(xs))))
            - LOCAL_Y_PADDING_BELOW
        )
        y_max = min(
            max(float(np.max(f(xs))), float(np.max(fixed_upper))) + LOCAL_Y_PADDING_ABOVE,
            LOCAL_Y_MAX_CAP,
        )

        axes = _make_axes(
            (x_min, x_max, x_step),
            (y_min, y_max, 1),
            x_length=LOCAL_AXIS_X_LENGTH,
            y_length=LOCAL_AXIS_Y_LENGTH,
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

        base_x_center = 0.5 * (x_min + x_max)
        base_y_center = 0.5 * (y_min + y_max)
        base_x_span = x_max - x_min
        base_y_span = y_max - y_min

        def view_limits() -> tuple[float, float, float, float]:
            scale = max(float(~view_scale), ZERO_AXIS_EPSILON)
            half_x = 0.5 * base_x_span * scale
            half_y = 0.5 * base_y_span * scale
            return (
                base_x_center - half_x,
                base_x_center + half_x,
                base_y_center - half_y,
                base_y_center + half_y,
            )

        def data_point(x: float, y: float) -> FloatArray:
            view_x_min, view_x_max, view_y_min, view_y_max = view_limits()
            x_ratio = (x - view_x_min) / (view_x_max - view_x_min)
            y_ratio = (y - view_y_min) / (view_y_max - view_y_min)
            return frame.get_corner(DL) + RIGHT * (frame.width * x_ratio) + UP * (frame.height * y_ratio)

        def curve_for(
            fn: Callable[[FloatArray], FloatArray],
            *,
            color: str,
            width: float,
            opacity: float = 1.0,
        ) -> VMobject:
            view_x_min, view_x_max, view_y_min, view_y_max = view_limits()
            visible_xs = np.linspace(view_x_min, view_x_max, LOCAL_CURVE_SAMPLES)
            ys = fn(visible_xs)
            mask = np.isfinite(ys) & (ys >= view_y_min) & (ys <= view_y_max)
            points = [
                data_point(float(x), float(y))
                for x, y in zip(visible_xs[mask], ys[mask], strict=True)
            ]
            if len(points) < 2:
                clipped_y = float(np.clip(ys[0], view_y_min, view_y_max))
                points = [data_point(float(visible_xs[0]), clipped_y)]
                points.append(points[0] + RIGHT * POLYLINE_FALLBACK_STEP)
            curve = VMobject()
            curve.set_points_smoothly(points)
            curve.set_stroke(color, width=width, opacity=opacity)
            return curve

        def dashed_curve_for(
            fn: Callable[[FloatArray], FloatArray],
            *,
            color: str,
            width: float,
        ) -> DashedVMobject:
            return DashedVMobject(
                curve_for(fn, color=color, width=width),
                num_dashes=LOCAL_MODEL_DASH_COUNT,
            ).set_z_index(LAYER_TRAJECTORY)

        def track_with_factory(mob: VMobject, factory: Callable[[], VMobject]) -> None:
            mob.add_updater(lambda tracked: tracked.become(factory()))

        def true_curve_factory() -> VMobject:
            return curve_for(f, color=C_TEXT, width=LOCAL_CURVE_STROKE_WIDTH).set_z_index(
                LAYER_TRAJECTORY
            )

        def local_model_factory() -> DashedVMobject:
            return dashed_curve_for(
                lambda values: model_values(values, x_t, h_t),
                color=C_GREEN,
                width=LOCAL_MODEL_STROKE_WIDTH,
            )

        def lower_model_factory() -> VMobject:
            return curve_for(
                lambda values: model_values(values, float(~alpha_anchor), float(~alpha)),
                color=C_BLUE,
                width=LOCAL_BOUND_STROKE_WIDTH,
                opacity=LOCAL_BOUND_OPACITY,
            ).set_z_index(LAYER_TRAJECTORY)

        def upper_model_factory() -> VMobject:
            return curve_for(
                lambda values: model_values(values, float(~beta_anchor), float(~beta)),
                color=C_ORANGE,
                width=LOCAL_BOUND_STROKE_WIDTH,
                opacity=LOCAL_BOUND_OPACITY,
            ).set_z_index(LAYER_TRAJECTORY)

        true_curve = true_curve_factory()
        local_model = local_model_factory()
        lower_model = lower_model_factory()
        upper_model = upper_model_factory()
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
            dot.add_updater(lambda mob: mob.move_to(data_point(x_fn(), y_fn())))
            return dot

        def bottom_tick(x_fn: Callable[[], float], label: str, *, color: str) -> VGroup:
            _, _, view_y_min, view_y_max = view_limits()
            tick_height = BOTTOM_TICK_HEIGHT_RATIO * (view_y_max - view_y_min)
            tick = Line(data_point(x_fn(), view_y_min), data_point(x_fn(), view_y_min + tick_height))
            tick.set_stroke(color, width=BOTTOM_TICK_STROKE_WIDTH)
            tex = MathTex(label, color=color)
            tex.scale_to_fit_height(TICK_LABEL_TEX_SCALE * tick.height)
            tex.next_to(tick, DOWN, buff=SMALL_BUFF)
            group = VGroup(tick, tex).set_z_index(LAYER_MARKERS)

            def update_tick(mob: VGroup) -> None:
                _, _, current_y_min, current_y_max = view_limits()
                current_tick_height = BOTTOM_TICK_HEIGHT_RATIO * (current_y_max - current_y_min)
                mob[0].put_start_and_end_on(
                    data_point(x_fn(), current_y_min),
                    data_point(x_fn(), current_y_min + current_tick_height),
                )
                mob[0].set_stroke(color, width=BOTTOM_TICK_STROKE_WIDTH)
                mob[1].next_to(mob[0], DOWN, buff=SMALL_BUFF)

            group.add_updater(update_tick)
            return group

        def label_next_to_dot(
            label: str,
            dot: Dot,
            *,
            direction: FloatArray,
            color: str,
            colors: dict[str, str] | None = None,
        ) -> MathTex:
            mob = theme_math(label, color=color, typography="caption").set_z_index(LAYER_MARKERS)
            if colors is not None:
                color_substrings(mob, colors)
            mob.next_to(dot, direction)
            mob.add_updater(lambda label_mob: label_mob.next_to(dot, direction))
            return mob

        def vertical_guide_factory(
            x_fn: Callable[[], float],
            *,
            color: str,
            opacity: float,
        ) -> DashedLine:
            _, _, current_y_min, current_y_max = view_limits()
            line = DashedLine(data_point(x_fn(), current_y_min), data_point(x_fn(), current_y_max), color=color)
            line.set_stroke(width=LOCAL_VERTICAL_GUIDE_STROKE_WIDTH, opacity=opacity)
            return line.set_z_index(LAYER_TRAJECTORY)

        x_t_dot = tracked_dot(lambda: x_t, lambda: f_t, color=C_YELLOW, radius=local_marker_radius)
        newton_dot = tracked_dot(lambda: x_next, lambda: y_next, color=C_GREEN, radius=local_marker_radius)
        star_dot = tracked_dot(
            lambda: float(x_star),
            lambda: f_star,
            color=C_TEXT,
            radius=local_marker_radius * SECOND_ORDER_STAR_DOT_SCALE,
        )
        x_t_tick = bottom_tick(lambda: x_t, r"x_t", color=C_YELLOW)
        next_tick = bottom_tick(lambda: x_next, r"x_{t+1}", color=C_GREEN)
        x_t_value = label_next_to_dot(
            r"f(x_t)",
            x_t_dot,
            direction=UR,
            color=C_TEXT,
            colors={r"x_t": C_YELLOW},
        )
        star_label = label_next_to_dot(
            r"f(x^\star)",
            star_dot,
            direction=DR,
            color=C_TEXT,
        )
        x_line = vertical_guide_factory(
            lambda: x_t,
            color=C_YELLOW,
            opacity=LOCAL_CURRENT_GUIDE_OPACITY,
        )
        next_line = vertical_guide_factory(
            lambda: x_next,
            color=C_GREEN,
            opacity=LOCAL_NEXT_GUIDE_OPACITY,
        )
        track_with_factory(
            x_line,
            lambda: vertical_guide_factory(
                lambda: x_t,
                color=C_YELLOW,
                opacity=LOCAL_CURRENT_GUIDE_OPACITY,
            ),
        )
        track_with_factory(
            next_line,
            lambda: vertical_guide_factory(
                lambda: x_next,
                color=C_GREEN,
                opacity=LOCAL_NEXT_GUIDE_OPACITY,
            ),
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

        delta_bracket.add_updater(update_delta_bracket)
        update_delta_bracket(delta_bracket)
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

        delta_label.add_updater(update_delta_label)
        update_delta_label(delta_label)

        def alpha_minimum() -> tuple[float, float]:
            value = max(~alpha, 1e-3)
            anchor = float(~alpha_anchor)
            x_alpha = anchor - grad(anchor) / value
            y_alpha = float(model_values(np.array([x_alpha]), anchor, value)[0])
            return float(x_alpha), float(y_alpha)

        def beta_minimum() -> tuple[float, float]:
            value = max(~beta, 1e-3)
            anchor = float(~beta_anchor)
            x_beta = anchor - grad(anchor) / value
            y_beta = float(model_values(np.array([x_beta]), anchor, value)[0])
            return float(x_beta), float(y_beta)

        def x_marker(x: float, y: float, *, color: str) -> VGroup:
            center_point = data_point(x, y)
            size = frame.height * X_MARKER_FRAME_HEIGHT_RATIO
            marker = VGroup(
                Line(center_point + (LEFT + DOWN) * size / 2, center_point + (RIGHT + UP) * size / 2),
                Line(center_point + (LEFT + UP) * size / 2, center_point + (RIGHT + DOWN) * size / 2),
            )
            marker.set_stroke(color, width=X_MARKER_STROKE_WIDTH, opacity=X_MARKER_OPACITY)
            return marker.set_z_index(LAYER_MARKERS)

        alpha_marker = x_marker(*alpha_minimum(), color=C_BLUE)
        beta_marker = x_marker(*beta_minimum(), color=C_ORANGE)
        track_with_factory(alpha_marker, lambda: x_marker(*alpha_minimum(), color=C_BLUE))
        track_with_factory(beta_marker, lambda: x_marker(*beta_minimum(), color=C_ORANGE))
        plot = VGroup(
            frame,
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
        left.scale_and_place(plot, buff=SMALL_BUFF)

        def color_formula(mob: VMobject) -> VMobject:
            _color_text_parts(
                mob,
                {
                    r"x_t": C_YELLOW,
                    r"x_{t+1}": C_GREEN,
                    r"\alpha": C_BLUE,
                    r"\beta": C_ORANGE,
                    r"\nabla^2": C_GREEN,
                },
            )
            return mob

        def plot_legend_entry(tex: str, *, color: str, width: float, dashed: bool = False) -> VGroup:
            label = color_formula(theme_math(tex, color=C_TEXT, typography="caption"))
            sample_width = frame.width * LOCAL_LEGEND_SAMPLE_FRAME_WIDTH_RATIO
            sample = DashedLine(ORIGIN, RIGHT * sample_width, color=color) if dashed else Line(ORIGIN, RIGHT * sample_width)
            sample.set_stroke(color, width=width)
            return VGroup(sample, label).arrange(RIGHT, buff=SMALL_BUFF)

        f_legend = plot_legend_entry(r"f(x)", color=C_TEXT, width=LOCAL_CURVE_STROKE_WIDTH)
        hessian_legend = plot_legend_entry(
            r"\begin{aligned}"
            r"f(x_t)&+\left\langle\nabla f(x_t),x-x_t\right\rangle\\"
            r"&+\frac{1}{2}(x-x_t)^\top\nabla^2 f(x_t)(x-x_t)"
            r"\end{aligned}",
            color=C_GREEN,
            width=LOCAL_MODEL_STROKE_WIDTH,
            dashed=True,
        )
        alpha_legend = plot_legend_entry(
            r"\begin{aligned}"
            r"f(x_t)&+\left\langle\nabla f(x_t),x-x_t\right\rangle\\"
            r"&+\frac{\alpha}{2}\left\|x-x_t\right\|_2^2"
            r"\end{aligned}",
            color=C_BLUE,
            width=LOCAL_BOUND_STROKE_WIDTH,
        )
        beta_legend = plot_legend_entry(
            r"\begin{aligned}"
            r"f(x_t)&+\left\langle\nabla f(x_t),x-x_t\right\rangle\\"
            r"&+\frac{\beta}{2}\left\|x-x_t\right\|_2^2"
            r"\end{aligned}",
            color=C_ORANGE,
            width=LOCAL_BOUND_STROKE_WIDTH,
        )
        plot_legend = VGroup(
            f_legend,
            hessian_legend,
            alpha_legend,
            beta_legend,
        ).arrange(DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
        legend_scale = min(
            frame.width * LOCAL_LEGEND_FRAME_WIDTH_RATIO / plot_legend.width,
            frame.height * LOCAL_LEGEND_FRAME_HEIGHT_RATIO / plot_legend.height,
            1.0,
        )
        plot_legend.scale(legend_scale)
        plot_legend.move_to(
            frame.get_corner(UR)
            + LEFT * (plot_legend.width / 2 + PANEL_BUFF)
            + DOWN * (plot_legend.height / 2 + PANEL_BUFF)
        )

        def legend_background_for(entry_count: int) -> VMobject:
            visible_entries = VGroup(*plot_legend.submobjects[:entry_count])
            width_spacer = Line(ORIGIN, RIGHT * plot_legend.width).set_opacity(0)
            width_spacer.move_to(visible_entries)
            width_spacer.align_to(plot_legend, RIGHT)
            background_target = VGroup(visible_entries, width_spacer)
            return _themed_box(background_target, color=C_PANEL_DEEP)[0].set_z_index(
                LAYER_MARKERS
            )

        plot_legend_background = legend_background_for(1)
        plot_legend.set_z_index(LAYER_MARKERS + 1)
        plot.add(plot_legend_background, plot_legend)

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
            page_width=right.width,
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
            page_width=right.width,
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
        right_panel = _themed_box(definition_stack)
        right.scale_and_place(right_panel, buff=SMALL_BUFF)
        sidebar_background = right_panel[0]

        zoom_out_scale = 2.0

        self.play(
            Write(title),
            Write(frame),
            Write(bottom_axis),
            Write(sidebar_background),
            Write(plot_legend_background),
            Write(true_curve),
            Write(x_t_dot),
            Write(x_t_tick),
            Write(x_t_value),
            Write(f_legend),
        )
        track_with_factory(true_curve, true_curve_factory)
        self.next_slide()

        self.play(
            Write(local_model),
            Write(newton_dot),
            Write(next_tick),
            Write(star_dot),
            Write(star_label),
            Write(x_line),
            Write(next_line),
            Write(delta_bracket),
            Write(delta_label),
            plot_legend_background.animate.become(legend_background_for(2)),
            Write(hessian_legend),
        )
        track_with_factory(local_model, local_model_factory)
        self.next_slide()

        self.play(
            Write(lower_model),
            Write(alpha_marker),
            Write(alpha_definition),
            plot_legend_background.animate.become(legend_background_for(3)),
            Write(alpha_legend),
        )
        track_with_factory(lower_model, lower_model_factory)
        self.next_slide()

        self.play(Write(upper_model), Write(beta_marker))
        track_with_factory(upper_model, upper_model_factory)
        self.next_slide()

        self.play(beta @ LOCAL_BETA_INITIAL)
        self.play(
            Write(beta_definition),
            plot_legend_background.animate.become(legend_background_for(4)),
            Write(beta_legend),
        )
        self.next_slide()

        self.play(view_scale @ zoom_out_scale)
        self.next_slide()

        self.play(beta_anchor @ float(x_star))
        self.next_slide()

        self.play(beta_anchor @ x_t)
        self.next_slide()

        self.play(alpha_anchor @ float(x_star))
        self.next_slide()

        self.play(alpha_anchor @ x_t)
        self.next_slide()
