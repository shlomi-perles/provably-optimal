"""Slide entrypoint modules for the AdaGrad and Adam deck.

Each rendered slide lives in its own module and shared building blocks live
under ``slides.helpers``. Reference scene classes from ``deck.toml`` by their
defining modules; do not re-export them here because Manim filters scene
classes by ``__module__`` during discovery.
"""
