"""Intro slide for the AdaGrad and Adam deck."""

from manim import DOWN, MED_LARGE_BUFF, ORIGIN, Tex, Write

from simplex import SimplexScene, get_active_theme

from slides.helpers.style import escape_tex_text


class Intro(SimplexScene):
    title: str = "AdaGrad & Adam"
    subtitle: str = r"Curvature, coordinates, and adaptive learning rates"

    def construct(self) -> None:
        self.slide(title=self.title)
        theme = get_active_theme()
        title_mob = Tex(escape_tex_text(self.title), font_size=theme.typography.h1)
        self.region.place(title_mob, ORIGIN)

        sub = Tex(escape_tex_text(self.subtitle), font_size=theme.typography.h2)
        sub.next_to(title_mob, DOWN, buff=MED_LARGE_BUFF)
        self.play(Write(title_mob), Write(sub))
        self.wait(0.4)
        self.next_slide()
        self.clear_scene()
