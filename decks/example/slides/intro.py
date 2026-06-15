"""Example scenes -- a title slide plus a second content slide."""

from manim import DOWN, ORIGIN, VGroup, Tex, Write, FadeIn

from simplex import get_active_theme

try:
    from simplex import SimplexScene as SimplexBase
except ImportError:
    from simplex import Slide as SimplexBase


def _start_slide(scene: SimplexBase, title: str) -> bool:
    marker = getattr(scene, "slide", None)
    if callable(marker):
        marker(title=title)
        return True
    return False


def _finish_legacy_slide(scene: SimplexBase, title: str, started: bool) -> None:
    if not started:
        scene.next_slide(name=title)


class Intro(SimplexBase):
    title: str = "Hello, Simplex"
    subtitle: str = r"$f(x) = e^{i\pi} + 1 = 0$"

    def construct(self) -> None:
        started = _start_slide(self, self.title)
        theme = get_active_theme()
        title_mob = Tex(self.title, font_size=theme.typography.h1)
        self.region.place(title_mob, ORIGIN)

        sub = Tex(self.subtitle, font_size=theme.typography.h2)
        sub.next_to(title_mob, DOWN, buff=0.4)
        self.play(Write(title_mob), Write(sub))
        _finish_legacy_slide(self, self.title, started)


class KeyIdea(SimplexBase):
    title: str = "A Second Slide"
    body: str = "Use notes for citations, slide refs, and supporting detail."

    def construct(self) -> None:
        started = _start_slide(self, self.title)
        theme = get_active_theme()
        title_mob = Tex(self.title, font_size=theme.typography.h2, color=theme.palette.accent)
        body_mob = Tex(self.body)
        group = VGroup(title_mob, body_mob).arrange(DOWN, buff=0.45)
        self.region.place(group, ORIGIN)

        self.play(Write(title_mob), FadeIn(body_mob, shift=DOWN * 0.2))
        _finish_legacy_slide(self, self.title, started)
