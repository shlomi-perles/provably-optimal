"""Unused scaffold key-idea slide."""

from manim import DOWN, MED_LARGE_BUFF, MED_SMALL_BUFF, ORIGIN, FadeIn, Tex, VGroup, Write

from simplex import SimplexScene, get_active_theme

from slides.helpers.style import escape_tex_text


class KeyIdea(SimplexScene):
    title: str = "A Second Slide"
    body: str = "Use notes for citations, slide refs, and supporting detail."

    def construct(self) -> None:
        self.slide(title=self.title)
        theme = get_active_theme()
        title_mob = Tex(
            escape_tex_text(self.title),
            font_size=theme.typography.h2,
            color=theme.palette.accent,
        )
        body_mob = Tex(escape_tex_text(self.body))
        group = VGroup(title_mob, body_mob).arrange(DOWN, buff=MED_LARGE_BUFF)
        self.region.place(group, ORIGIN)

        self.play(Write(title_mob), FadeIn(body_mob, shift=DOWN * MED_SMALL_BUFF))
        self.wait(2)
        self.next_slide()
        self.wait(0.5)
        self.clear_scene()
