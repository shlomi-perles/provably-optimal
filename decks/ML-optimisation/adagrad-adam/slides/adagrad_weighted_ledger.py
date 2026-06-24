"""Ada Grad Weighted Ledger slide."""

from __future__ import annotations

from slides.helpers.figure_helpers import *


class AdaGradWeightedLedger(Slide):
    """The inner-product lemma as a change of ruler."""

    def construct(self) -> None:
        title = _start_slide(self, "The weighted ledger identity")
        visual_region, proof_region = _split_rows(self.region, [2.0, 1.0])
        left, arrow_region, right = _split_weighted(visual_region, [1.0, 0.30, 1.0])

        d_sqrt = np.diag([2.0, 1.0])
        d_inv_sqrt = np.linalg.inv(d_sqrt)
        weighted_points = {
            "star": np.array([0.0, 0.0], dtype=np.float64),
            "xt": np.array([-2.0, 3.0], dtype=np.float64),
            "xt1": np.array([2.0, 1.5], dtype=np.float64),
        }
        euclidean_points = {name: d_inv_sqrt @ point for name, point in weighted_points.items()}

        weighted = self._triangle_panel(weighted_points, "weighted coordinates", r"D_t")
        euclidean = self._triangle_panel(euclidean_points, "Euclidean coordinates", r"2")
        left.scale_and_place(weighted, buff=SMALL_BUFF)
        right.scale_and_place(euclidean, buff=SMALL_BUFF)

        map_arrow = VGroup(
            theme_math(r"x\mapsto", color=C_PURPLE, typography="caption"),
            theme_math(r"D_t^{-1/2}x", color=C_PURPLE, typography="caption"),
            Arrow(LEFT, RIGHT, color=C_PURPLE, buff=0, tip_shape=StealthTip),
        ).arrange(DOWN, buff=SMALL_BUFF)
        arrow_region.scale_and_place(map_arrow, buff=SMALL_BUFF)

        ledger_lines = VGroup(
            theme_math(r"2\eta\langle g_t,x_t-x^\star\rangle"),
            theme_math(
                r"= \|x_t-x^\star\|_{D_t}^2-\|x_{t+1}-x^\star\|_{D_t}^2",
            ),
            theme_math(r"+ \|x_t-x_{t+1}\|_{D_t}^2"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
        setup = theme_math(r"x_{t+1}=x_t-\eta D_t^{-1}g_t")
        for line in ledger_lines:
            line.set_color_by_tex(r"\eta", C_YELLOW)
            line.set_color_by_tex(r"D_t", C_PURPLE)
        setup.set_color_by_tex(r"\eta", C_YELLOW)
        setup.set_color_by_tex(r"D_t", C_PURPLE)
        proof = _formula_stack(
            Caption("weighted parallelogram bookkeeping"),
            VGroup(setup, ledger_lines).arrange(RIGHT, buff=LEDGER_PROOF_COLUMN_BUFF),
            Caption("progress = distance drop + movement"),
        )
        proof_region.scale_and_place(_themed_box(proof), buff=SMALL_BUFF)

        self.play(Write(title), FadeIn(weighted))
        self.next_slide(title="Change the ruler")
        self.play(Write(map_arrow), FadeIn(euclidean))
        self.next_slide(title="Read off the ledger")
        self.play(Write(proof))
        self.next_slide()
        self.clear_scene()

    def _triangle_panel(
        self,
        points: dict[str, FloatArray],
        title: str,
        norm_suffix: str,
    ) -> Group:
        axes = _make_axes(
            (-2.8, 3.4, 1),
            (-1.0, 3.9, 1),
            x_length=4.65,
            y_length=3.35,
            preserve_unit_aspect=True,
        )
        matrix = np.diag([0.55, 1.55]) if norm_suffix == "D_t" else np.eye(2)
        heatmap = _quadratic_heatmap(axes, matrix, width=430).set_z_index(LAYER_HEATMAP)
        grid = _coordinate_grid(axes).set_z_index(LAYER_CONTOUR)
        levels = _quadratic_level_sets(axes, matrix, count=7)
        levels.set_z_index(LAYER_CONTOUR)
        axes.set_opacity(0)
        if norm_suffix == "D_t":
            labels = {
                "star": r"D_t^{1/2}x^\star",
                "xt": r"D_t^{1/2}x_t",
                "xt1": r"D_t^{1/2}x_{t+1}",
            }
        else:
            labels = {
                "star": r"x^\star",
                "xt": r"x_t",
                "xt1": r"x_{t+1}",
            }
        p_star = points["star"]
        p_xt = points["xt"]
        p_xt1 = points["xt1"]

        def triangle_center() -> FloatArray:
            return (p_star + p_xt + p_xt1) / 3.0

        def readable_angle(pa: FloatArray, pb: FloatArray) -> float:
            angle = float(np.arctan2(pb[1] - pa[1], pb[0] - pa[0]))
            if angle > np.pi / 2:
                angle -= np.pi
            if angle < -np.pi / 2:
                angle += np.pi
            return angle

        def side_normal(pa: FloatArray, pb: FloatArray, center: FloatArray) -> FloatArray:
            segment = pb - pa
            normal = np.array([-segment[1], segment[0]], dtype=np.float64)
            norm = np.linalg.norm(normal)
            if norm == 0:
                return np.array([0.0, 1.0], dtype=np.float64)
            normal = normal / norm
            midpoint = 0.5 * (pa + pb)
            if np.dot(center - midpoint, normal) < 0:
                normal = -normal
            return normal

        def line_label(
            pa: FloatArray,
            pb: FloatArray,
            text: str,
            *,
            side: str,
            color: str,
            pos: float = 0.5,
            distance: float = LEDGER_LABEL_DISTANCE,
            typography: str = "caption",
        ) -> MathTex:
            center = triangle_center()
            normal = side_normal(pa, pb, center)
            if side == "outer":
                normal = -normal
            location = pa + pos * (pb - pa) + distance * normal
            label = theme_math(text, color=color, typography=typography)
            label.move_to(axes.c2p(float(location[0]), float(location[1])))
            label.rotate(readable_angle(pa, pb))
            return label

        def point_label(key: str, typography: str) -> MathTex:
            center = triangle_center()
            point = points[key]
            direction = point - center
            norm = np.linalg.norm(direction)
            direction = (
                np.array([0.0, -1.0], dtype=np.float64) if norm == 0 else direction / norm
            )
            location = point + LEDGER_POINT_LABEL_DISTANCE * direction
            return theme_math(labels[key], color=C_TEXT, typography=typography).move_to(
                axes.c2p(float(location[0]), float(location[1]))
            )

        old_distance = DashedLine(
            axes.c2p(float(p_xt[0]), float(p_xt[1])),
            axes.c2p(float(p_star[0]), float(p_star[1])),
            color=C_MUTED,
        )
        new_distance = DashedLine(
            axes.c2p(float(p_xt1[0]), float(p_xt1[1])),
            axes.c2p(float(p_star[0]), float(p_star[1])),
            color=C_MUTED,
        )
        step = _axis_arrow(axes, p_xt, p_xt1, color=C_RED)
        frame = Rectangle(width=axes.width, height=axes.height)
        frame.move_to(axes)
        vertex_radius = frame.height * PANEL_MARKER_FRAME_HEIGHT_RATIO
        dots = VGroup(
            *(Dot(axes.c2p(float(point[0]), float(point[1])), color=C_TEXT, radius=vertex_radius) for point in points.values())
        )
        label_typography = "caption"
        point_labels = VGroup(
            point_label("star", label_typography),
            point_label("xt", label_typography),
            point_label("xt1", label_typography),
        )
        side_labels = VGroup(
            line_label(
                p_xt,
                p_star,
                rf"\|x_t-x^\star\|_{{{norm_suffix}}}^2",
                side="inner",
                color=C_MUTED,
            ),
            line_label(
                p_xt1,
                p_star,
                rf"\|x_{{t+1}}-x^\star\|_{{{norm_suffix}}}^2",
                side="inner",
                color=C_MUTED,
                pos=LEDGER_NEW_DISTANCE_LABEL_POSITION,
            ),
            line_label(
                p_xt,
                p_xt1,
                rf"\|x_{{t+1}}-x_t\|_{{{norm_suffix}}}^2",
                side="inner",
                color=C_RED,
                pos=LEDGER_STEP_LABEL_POSITION,
            ),
        )
        extras = VGroup()
        if norm_suffix == "D_t":
            extras.add(
                line_label(
                    p_xt,
                    p_star,
                    r"A",
                    side="outer",
                    color=C_TEXT,
                    distance=LEDGER_LABEL_OUTER_DISTANCE,
                ),
                line_label(
                    p_xt,
                    p_xt1,
                    r"B",
                    side="outer",
                    color=C_TEXT,
                    pos=LEDGER_B_LABEL_POSITION,
                    distance=LEDGER_LABEL_OUTER_DISTANCE,
                ),
                line_label(
                    p_xt1,
                    p_star,
                    r"A-B",
                    side="outer",
                    color=C_TEXT,
                    distance=LEDGER_LABEL_OUTER_DISTANCE,
                ),
            )
            center = triangle_center()
            outer = -side_normal(p_xt, p_star, center)
            brace_start = p_xt + LEDGER_P_PROJECTION_START * (p_star - p_xt)
            brace_end = p_xt + LEDGER_P_PROJECTION_END * (p_star - p_xt)
            brace = BraceBetweenPoints(
                axes.c2p(float(brace_start[0]), float(brace_start[1])),
                axes.c2p(float(brace_end[0]), float(brace_end[1])),
                direction=np.array([outer[0], outer[1], 0.0]),
                color=C_PURPLE,
            )
            p_point = p_xt + LEDGER_P_POINT_POSITION * (p_star - p_xt)
            open_point = Circle(radius=vertex_radius, color=C_PURPLE).move_to(
                axes.c2p(float(p_point[0]), float(p_point[1]))
            )
            open_point.set_fill(C_PANEL_DEEP, opacity=1.0)
            p_label = theme_math(r"P", color=C_PURPLE, typography="caption").next_to(
                brace,
                LEFT,
                buff=SMALL_BUFF,
            )
            extras.add(brace, p_label, open_point)
        else:
            extras.add(
                line_label(
                    p_xt,
                    p_xt1,
                    r"\eta\nabla f(x_t)",
                    side="outer",
                    color=C_RED,
                    pos=LEDGER_EUCLIDEAN_STEP_LABEL_POSITION,
                    distance=LEDGER_EUCLIDEAN_STEP_LABEL_DISTANCE,
                )
            )
        caption = Caption(title).next_to(axes, UP)
        frame.set_stroke(
            C_FRAME,
            width=LEDGER_FRAME_STROKE_WIDTH,
            opacity=LEDGER_FRAME_STROKE_OPACITY,
        )
        field = Group(heatmap, grid, levels)
        distances = VGroup(old_distance, new_distance, step, dots, point_labels, side_labels, extras)
        return Group(frame, axes, field, distances, caption)
