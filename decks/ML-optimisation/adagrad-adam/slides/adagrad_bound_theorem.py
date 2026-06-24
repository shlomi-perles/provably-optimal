"""AdaGrad coordinate-wise bound theorem slide."""

from __future__ import annotations

from manim import TexTemplate, Unwrite

from slides.adagrad_weighted_ledger import AdaGradWeightedLedger
from slides.helpers.figure_helpers import *
from slides.helpers.reminders import ReminderStack
from slides.helpers.second_order import FORMULA_COLORS


BOUND_NORM_LABEL_SCALE = 0.9

ADAGRAD_BOUND_COLORS = {
    r"x_{t+1}": FORMULA_COLORS[r"x_{t+1}"],
    r"x_t": FORMULA_COLORS[r"x_t"],
    r"x_{t,i}": FORMULA_COLORS[r"x_t"],
    r"x_{t+1,i}": FORMULA_COLORS[r"x_{t+1}"],
    r"\nabla f(x_t)": C_GREEN,
    r"\nabla f(x_t)_i": C_GREEN,
    r"g_t": C_GREEN,
    r"g_{t,i}": C_GREEN,
    r"D_t": C_PURPLE,
    r"\eta": C_YELLOW,
    r"R_\infty": C_YELLOW,
    r"R_i": C_BLUE,
    r"\alpha_i": C_GREEN,
}


def _adagrad_bound_template() -> TexTemplate:
    template = TexTemplate()
    template.add_to_preamble(r"\usepackage{amsmath,amssymb,amsthm,mathtools}")
    template.add_to_preamble(r"\newcommand{\R}{\mathbb{R}}")
    template.add_to_preamble(r"\newcommand{\ip}[2]{\left\langle #1,#2\right\rangle}")
    template.add_to_preamble(r"\newcommand{\norm}[2][]{\left\lVert #2\right\rVert_{#1}}")
    template.add_to_preamble(r"\newtheorem{thm}{Theorem}")
    return template


def _color_adagrad_math(mobject: VMobject) -> VMobject:
    color_substrings(mobject, ADAGRAD_BOUND_COLORS, probe_class=MathTex)
    return mobject


def _bound_math(*parts: str, tex_template: TexTemplate, typography: str = "body") -> MathTex:
    math = MathTex(
        *parts,
        tex_template=tex_template,
        font_size=getattr(get_active_theme().typography, typography),
    )
    _color_adagrad_math(math)
    return math


def _side_normal(pa: FloatArray, pb: FloatArray, center: FloatArray) -> FloatArray:
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


class AdaGradBoundTheorem(Slide):
    """State AdaGrad's coordinate-wise regret bound and open the proof ledger."""

    def construct(self) -> None:
        tex_template = _adagrad_bound_template()
        title = _start_slide(self, "AdaGrad coordinate-wise bound")

        screen_region = self.region.copy()
        theorem_region = self.region.copy()
        theorem_region.update(top=title)

        theorem_page = self._theorem_page(theorem_region, tex_template)
        theorem_region.place(theorem_page)

        self.play(Write(title), Write(theorem_page))
        self.wait(2)
        self.next_slide(title="Keep the useful facts")
        self.wait(0.5)

        reminders = self._reminder_stack(screen_region, theorem_page, tex_template)
        work_region = self.region.copy()
        work_region.update(top=title, bottom=reminders)

        theorem_rest = VGroup(*theorem_page.lines, *theorem_page.equations[1:])
        self.play(Unwrite(theorem_rest))
        self.play(
            FadeIn(reminders.frame),
            reminders.animate_add_many(
                self._reminder_sources(theorem_page, tex_template),
                from_existing=True,
            ),
        )
        self.remove(theorem_page)
        self.wait(2)
        self.next_slide(title="Convexity opens the ledger")
        self.wait(0.5)

        proof_page, definitions = self._proof_header(work_region, tex_template)
        self.play(Write(proof_page))
        self.play(Write(definitions))

        ledger_region = work_region.copy()
        ledger_region.update(top=definitions, bottom=reminders)
        weighted, map_arrow, euclidean, weighted_points = self._ledger_figures(ledger_region)
        self.play(FadeIn(weighted), Write(map_arrow), FadeIn(euclidean))
        self.wait(2)
        self.next_slide(title="Progress is the bracket")
        self.wait(0.5)

        progress_marker = self._progress_marker(weighted, weighted_points)
        self.play(
            Write(progress_marker[2]),
            Write(progress_marker[0]),
            Write(progress_marker[1]),
        )
        self.wait(2)
        self.next_slide(title="Bound the progress account")
        self.wait(0.5)

        p_bound = self._progress_bound(ledger_region, tex_template)
        self.play(FadeOut(weighted, map_arrow, euclidean, progress_marker))
        self.play(Write(p_bound))
        self.wait(2)
        self.next_slide(title="Use the root-sum bound")
        self.wait(0.5)

        root_sum = self._root_sum_bound(ledger_region, p_bound, tex_template)
        self.play(Write(root_sum))
        self.wait(2)
        self.next_slide()
        self.wait(0.5)
        self.clear_scene()

    def _theorem_page(self, region, tex_template: TexTemplate) -> TexPage:
        page = TexPage(
            r"Assume:"
            r"\begin{enumerate}"
            r"\item $f:\R^n\to\R$ is convex and differentiable"
            r"\item $g_t=\nabla f(x_t)$"
            r"\item for every coordinate there is a radius $R_i$ such that "
            r"$|x_{t,i}-x_i^\star|\le R_i$"
            r"\item let $R_\infty=\max_i R_i$"
            r"\end{enumerate}"
            r"\begin{thm}[A coordinate-wise AdaGrad bound]"
            r"Under the assumptions above,"
            r"\["
            r"\frac{1}{T}\sum_{t=1}^{T}\bigl(f(x_t)-f(x^\star)\bigr)"
            r"\le"
            r"\frac{\sqrt2 R_\infty}{T}"
            r"\sum_{i=1}^{n}"
            r"\sqrt{\sum_{t=1}^{T}g_{t,i}^2}"
            r"\]"
            r"Let $g_{t,i}$ be constants, and consider the update"
            r"\["
            r"x_{t+1,i} = x_{t,i} - \alpha_i g_{t,i}."
            r"\]"
            r"Then"
            r"\["
            r"\alpha_i^\star = \frac{R_i}{\sqrt{\sum_{t=1}^{T} g_{t,i}^2}}."
            r"\]"
            r"\end{thm}",
            page_width=region,
            tex_template=tex_template,
            font_size=get_active_theme().typography.caption,
        )
        return _color_adagrad_math(page)

    def _reminder_stack(
        self,
        screen_region,
        theorem_page: TexPage,
        tex_template: TexTemplate,
    ) -> ReminderStack:
        reminders = ReminderStack(
            [],
            width=screen_region.width - 2 * SMALL_BUFF,
            orientation="horizontal",
        )
        screen_region.place(reminders, DL, buff=SMALL_BUFF)
        return reminders

    def _reminder_sources(self, theorem_page: TexPage, tex_template: TexTemplate) -> list[VMobject]:
        adagrad_coordinate_update = _bound_math(
            r"x_{t+1,i}=x_{t,i}-"
            r"\left(\sqrt{\sum_{j=0}^t \nabla f(x_j)_i^2}\right)^{-1}"
            r"\nabla f(x_t)_i",
            tex_template=tex_template,
            typography="caption",
        ).move_to(theorem_page.equations[1])
        radius_definition = _bound_math(
            r"R_\infty=\max_i |x_{t,i}-x_i^\star|",
            tex_template=tex_template,
            typography="caption",
        ).move_to(theorem_page.equations[-1])
        return [adagrad_coordinate_update, theorem_page.equations[0], radius_definition]

    def _proof_header(self, region, tex_template: TexTemplate) -> tuple[TexPage, VGroup]:
        proof_region, definitions_region, _ = _split_rows(region, [0.95, 0.8, 2.75])
        proof_page =  MathTex(r"$f(x_t)-f(x^\star)\le \ip{\nabla f(x_t)}{x_t-x^\star}$ \quad \rightarrow \quad \eta\sum_{t=1}^{T}\bigl(f(x_t)-f(x^\star)\bigr)")
        _color_adagrad_math(proof_page)
        proof_region.scale_and_place(proof_page, buff=SMALL_BUFF)

        progress_account = _bound_math(
            r"P:="
            r"\frac12\sum_{t=1}^{T}"
            r"\left("
            r"\norm[D_t]{x_t-x^\star}^2"
            r"-"
            r"\norm[D_t]{x_{t+1}-x^\star}^2"
            r"\right)"
            r"\label{eq:adagrad-progress-account}",
            tex_template=tex_template,
            typography="caption",
        )
        movement_account = _bound_math(
            r"M="
            r"\frac12\sum_{t=1}^{T}\norm[D_t]{x_t-x_{t+1}}^2"
            r"="
            r"\frac{\eta^2}{2}\sum_{i=1}^{n}\sum_{t=1}^{T}"
            r"\frac{g_{t,i}^2}{\eta_{t,i}}.",
            tex_template=tex_template,
            typography="caption",
        )
        progress_account.set_color_by_tex("P", C_YELLOW)
        movement_account.set_color_by_tex("M", C_RED)
        definitions = VGroup(progress_account, movement_account).arrange(
            RIGHT,
            buff=MED_LARGE_BUFF,
        )
        definitions_region.scale_and_place(
            definitions,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )
        return proof_page, definitions

    def _ledger_figures(self, region) -> tuple[Group, VGroup, Group, dict[str, FloatArray]]:
        left, arrow_region, right = _split_weighted(region, [1.0, 0.30, 1.0])
        d_sqrt = np.diag([2.0, 1.0])
        d_inv_sqrt = np.linalg.inv(d_sqrt)
        weighted_points = {
            "star": np.array([0.0, 0.0], dtype=np.float64),
            "xt": np.array([-2.0, 3.0], dtype=np.float64),
            "xt1": np.array([2.0, 1.5], dtype=np.float64),
        }
        euclidean_points = {name: d_inv_sqrt @ point for name, point in weighted_points.items()}

        weighted = AdaGradWeightedLedger._triangle_panel(
            self,
            weighted_points,
            "",
            r"D_t",
            show_caption=False,
            show_progress_marker=False,
            norm_label_color=C_TEXT,
            norm_label_scale=BOUND_NORM_LABEL_SCALE,
            norm_label_color_map=ADAGRAD_BOUND_COLORS,
            nudge_xt1_label=True,
        )
        euclidean = AdaGradWeightedLedger._triangle_panel(
            self,
            euclidean_points,
            "",
            r"2",
            show_caption=False,
            norm_label_color=C_TEXT,
            norm_label_scale=BOUND_NORM_LABEL_SCALE,
            norm_label_color_map=ADAGRAD_BOUND_COLORS,
            nudge_xt1_label=True,
        )
        left.scale_and_place(weighted, buff=SMALL_BUFF)
        right.scale_and_place(euclidean, buff=SMALL_BUFF)

        map_label = VGroup(
            theme_math(r"x\mapsto", color=C_YELLOW, typography="caption"),
            theme_math(r"D_t^{-1/2}x", color=C_YELLOW, typography="caption"),
        ).arrange(DOWN, buff=SMALL_BUFF)
        map_arrow = CurvedArrow(
            LEFT,
            RIGHT,
            angle=-PI / 2,
            color=C_YELLOW,
            tip_shape=StealthTip,
        )
        map_group = VGroup(map_label, map_arrow).arrange(DOWN)
        arrow_region.scale_and_place(map_group, buff=SMALL_BUFF)
        return weighted, map_group, euclidean, weighted_points

    def _progress_marker(self, weighted: Group, points: dict[str, FloatArray]) -> VGroup:
        frame = weighted[0]
        axes = weighted[1]
        p_xt = points["xt"]
        p_star = points["star"]
        p_xt1 = points["xt1"]
        center = (p_star + p_xt + p_xt1) / 3.0
        outer = -_side_normal(p_xt, p_star, center)
        p_point = p_xt + LEDGER_P_POINT_POSITION * (p_star - p_xt)
        brace = BraceBetweenPoints(
            axes.c2p(float(p_xt[0]), float(p_xt[1])),
            axes.c2p(float(p_point[0]), float(p_point[1])),
            direction=np.array([outer[0], outer[1], 0.0]),
            color=C_YELLOW,
        )
        open_point = Circle(
            radius=frame.height * PANEL_MARKER_FRAME_HEIGHT_RATIO,
            color=C_PURPLE,
        ).move_to(axes.c2p(float(p_point[0]), float(p_point[1])))
        open_point.set_fill(C_PANEL_DEEP, opacity=1.0)
        p_label = theme_math(r"P", color=C_YELLOW, typography="caption")
        brace.put_at_tip(p_label)
        return VGroup(brace, p_label, open_point)

    def _progress_bound(self, region, tex_template: TexTemplate) -> MathTex:
        p_bound = _bound_math(
            r"P \leq \frac12\sum_{t=1}^{T} \norm[D_t]{x_t-x^*}^2"
            r"\leq \frac12 \sum_{i=1}^{n} R_i \eta_{t,i}",
            tex_template=tex_template,
            typography="caption",
        )
        p_bound.set_color_by_tex("P", C_YELLOW)
        region.scale_and_place(
            p_bound,
            UP,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )
        return p_bound

    def _root_sum_bound(
        self,
        region,
        p_bound: MathTex,
        tex_template: TexTemplate,
    ) -> TexPage:
        root_region = region.copy()
        root_region.update(top=p_bound)
        root_sum = TexPage(
            r"If $A_t=A_{t-1}+a_t^2$ with $A_0>0$, then"
            r"\["
            r"\sum_{t=1}^{T}\frac{a_t^2}{\sqrt{A_t}}"
            r"\le"
            r"2(\sqrt{A_T}-\sqrt{A_0})."
            r"\label{eq:root-sum-bound}"
            r"\]",
            page_width=root_region,
            tex_template=tex_template,
            font_size=get_active_theme().typography.caption,
        )
        _color_adagrad_math(root_sum)
        root_region.scale_and_place(
            root_sum,
            buff=SMALL_BUFF,
            scale_kwargs={"max_scale": 1},
        )
        return root_sum
