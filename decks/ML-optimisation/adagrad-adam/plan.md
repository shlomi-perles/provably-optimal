# Manim-Native AdaGrad Figure Slides

## Summary

- Create 8 polished Simplex scene classes, one for each `\pythonfigure` embedded in `main_intro_gd.tex`.
- Do not display static PDFs. Recreate each figure as Manim-native geometry, equations, trackers, and animations; use the note Python only for math logic, constants, and parameter choices.
- Keep the existing `MomentumRosenbrock` scene as a live companion after the GD-mode slide, and remove the placeholder `KeyIdea` entrypoint.

## Key Changes

- Add one `Slide` subclass per note figure: `SecondOrderApproximation`, `QuadraticRotation`, `GradientDescentModes`, `AdaGradKnownRuler`, `AdaGradDiagonalScaling`, `AdaGradLocalSqueeze`, `AdaGradCoordinateResponse`, and `AdaGradWeightedLedger`.
- Use `ThreeDSlide` with `@opengl` only for `AdaGradLocalSqueeze`; all other scenes use Cairo `Slide`.
- Add small shared helpers for themed titles, non-equal `Region` column splits, quadratic level sets, trajectory/path drawing, response bars, captions, and color-coded equations.
- Update `deck.toml` entrypoints in lecture-note order: intro, the 8 new figure scenes, existing `MomentumRosenbrock` after `GradientDescentModes`, no template `KeyIdea`.

## Slide Layouts And Beats

| Class | Layout | Animation Plan |
|---|---|---|
| `SecondOrderApproximation` | Wide left curve panel; right proof column for Taylor model and Newton step. | Build `f`, tangent quadratic, then sweep `\alpha`/`\beta` envelopes with `VT`; highlight `x_t`, local minimizer, true minimizer. |
| `QuadraticRotation` | Dual panel: original coordinates left, eigenbasis right; center rotation arrow. | Draw rotated ellipses/eigenvectors, morph to axis-aligned ellipses, reveal `V^\top A V=\operatorname{diag}(\alpha,\beta)`. |
| `GradientDescentModes` | Top large trajectory panel; bottom two compact mode-response strips; right formula column. | Sweep `\eta` through safe, near-minimax, minimax values; update path, eigenmode arrows, and `1-\eta\lambda_i` readouts. |
| `AdaGradKnownRuler` | Dual panel: rotated quadratic vs axis-aligned quadratic; small legend below. | Animate GD, known diagonal inverse, and AdaGrad paths; pause on "one step only when axes agree." |
| `AdaGradDiagonalScaling` | 2x2 transformation grid with arrows between before/after panels. | Show diagonal scaling squeezing ellipses toward circles, then contrast axis-aligned success with rotated limitation. |
| `AdaGradLocalSqueeze` | Full 3D surface in main region; fixed-frame equation/legend strip on right. | Use Simplex `ScalarFieldSurface`/`ColorBar`; show local point, diagonal quadratic patch, camera move, and formula reveal. |
| `AdaGradCoordinateResponse` | 2x3 response chart grid with a narrow right equation card. | Animate bars over time; compare fixed GD multipliers with AdaGrad's curvature-canceling normalized traces. |
| `AdaGradWeightedLedger` | Dual panel weighted coordinates vs Euclidean coordinates; proof ledger on right/bottom. | Build triangle identity, transform via `D_t^{-1/2}`, then reveal the inner-product lemma term by term. |

## Implementation Rules

- Prefer Simplex helpers: `self.slide`, `self.fragment`, `Region`, `scale_and_place`, `Caption`, `TexPage`, `color_substrings`, `VT`, `DN`, `clear_scene`, `ScalarFieldSurface`, and `ColorBar`.
- Use Manim defaults unless intentionally changing behavior; avoid explicit default `buff`, timing, or style arguments.
- Keep colors semantic across scenes: flat mode blue, steep mode orange, iterates green/teal, diagonal ruler purple, warnings/red instability, current focus yellow.
- Use progressive reveal: geometry first, equations second, proof terms last.

## Test Plan

- Run `mcp__simplex_mcp.inspect_simplex_file` on each new scene file.
- Render low-quality videos with `mcp__simplex_mcp.render_simplex_file`; use video/cue checks rather than final-frame PNGs because scenes may call `clear_scene()`.
- Run `mcp__simplex_mcp.run_simplex_cli` for the full `adagrad-adam` deck in dark theme.
- Verify cues, no clipped text, no empty frames, all trackers update, OpenGL local-squeeze scene writes video, and deck entrypoint order matches the lecture narrative.

## Assumptions

- Scope is exactly the 8 figures embedded in `main_intro_gd.tex`; the 3 extra generator-catalog figures remain out of scope.
- The new slides should be self-contained Manim code, not runtime imports from the external notes directory.
- Existing `MomentumRosenbrock` stays as an additional live demo because it already matches the requested animated-parameter style.
