# AdaGrad and Adam: curvature, coordinates, and adaptive learning rates

The cheerful way to read this lecture is as a sequence of increasingly cheap
answers to one very expensive question:

> **Remark.** What would we do if we could cheaply use the inverse Hessian?

Newton's method says: use it directly. Gradient descent says: replace it by one
number. AdaGrad says: keep only a diagonal, and learn that diagonal from the
gradients. Momentum says: keep the cheap ruler, but make the direction smarter
by remembering the past. Adam then mixes the two happy tricks: momentum in the
numerator, adaptive coordinate scaling in the denominator \cite{KB15}.

The deck starts this story at [slide:adagrad-adam].

## The second-order local model

Suppose we are at $x_t$ and move by $\delta$. The second-order Taylor model is

$$
f(x_t+\delta)
\approx
f(x_t)+\nabla f(x_t)^\top \delta
+\frac12 \delta^\top \nabla^2 f(x_t)\delta .
$$

The constant term tells us where we are, the linear term tells us which
directions initially decrease the function, and the quadratic term tells us
how costly those directions become because of curvature. The animation in
[slide:second-order-approximation] is the visual version of this sentence: a
local quadratic touches the function at the current point, and the curvature
envelopes say how flat or steep the world is allowed to be.

Minimizing the local quadratic in $\delta$ gives

$$
\nabla f(x_t)+\nabla^2 f(x_t)\delta = 0,
$$

so, when the Hessian is invertible,

$$
\delta
=
-\nabla^2 f(x_t)^{-1}\nabla f(x_t).
$$

With $\delta=x_{t+1}-x_t$, this is Newton's method:

$$
x_{t+1}
=
x_t-\nabla^2 f(x_t)^{-1}\nabla f(x_t).
$$

Newton is the dream step: it asks the local quadratic, "where is your minimum?"
and walks there.

> **Definition.** \label{def:strong-smooth} Let $\alpha,\beta>0$. A
> differentiable function $f$ is $\alpha$-strongly convex if, for every $x,y$,
>
> $$
> f(y)\ge
> f(x)+\nabla f(x)^\top (y-x)+\frac{\alpha}{2}\|y-x\|_2^2.
> $$
>
> It is $\beta$-smooth if, for every $x,y$,
>
> $$
> f(y)\le
> f(x)+\nabla f(x)^\top (y-x)+\frac{\beta}{2}\|y-x\|_2^2.
> $$

Strong convexity is a lower quadratic support: the function cannot be flatter
than curvature $\alpha$. Smoothness is an upper quadratic support: the function
cannot bend upward faster than curvature $\beta$. The condition number is

$$
\kappa=\frac{\beta}{\alpha}.
$$

When $f$ is twice differentiable, this corresponds to Hessian eigenvalues in
the interval

$$
\lambda_{\min}(\nabla^2 f(x))\ge \alpha,
\qquad
\lambda_{\max}(\nabla^2 f(x))\le \beta.
$$

A large $\kappa$ means the problem has both gentle directions and steep
directions. One learning rate must somehow survive all of them. That is where
the fun begins.

Newton can have a very fast local rate under stronger assumptions such as a
Lipschitz Hessian and an initial point close enough to the optimum.^[A common
headline is $O(\log\log(1/\epsilon))$ local iteration complexity. This is a
local statement, not a global promise from a random starting point.] But it is
expensive: computing and solving with a dense $n\times n$ Hessian is costly,
and the local model can be misleading far away from the optimum.

So the guiding question becomes:

$$
\text{How much of } \nabla^2 f(x_t)^{-1} \text{ can we approximate cheaply?}
$$

## The quadratic microscope

Quadratics are not the whole world, but they are the smallest world where
curvature, conditioning, and zigzagging are all visible. Let

$$
f(x)
=
\frac12 (x-x^\star)^\top A(x-x^\star),
\qquad
A=A^\top\succ 0.
$$

Write the eigendecomposition

$$
A=V\Lambda V^\top,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n),
\qquad
Av_i=\lambda_i v_i.
$$

The eigenvectors $v_i$ are the natural directions of the quadratic. If we
rotate by $V^\top$, the loss separates into independent one-dimensional
problems. This is the microscope:

1. rotate into the Hessian eigenbasis;
2. write a scalar recurrence for each eigenvalue $\lambda_i$;
3. rotate back by summing the surviving modes.

If

$$
x_0-x^\star=\sum_{i=1}^n a_i v_i,
$$

then a method that preserves eigendirections can be understood by asking what
happens to each scalar coefficient $a_i$. Ill-conditioning means the same
algorithm must behave well for every $\lambda_i\in[\alpha,\beta]$.

## Gradient descent: one scalar for every direction

Gradient descent is Newton with a wonderfully rude approximation:

$$
\nabla^2 f(x_t)^{-1}\approx \eta I.
$$

Plugging this into the Newton-style step gives

$$
x_{t+1}=x_t-\eta \nabla f(x_t).
$$

The slide [slide:gradient-descent-modes] is the main companion here. It shows
the path, the eigendirection components, and the scalar response of each mode.
The Beale surface demo at [slide:beales-gradient-descent-plot] shows the same
story on a less polite function: one global step size can be brave in one
direction and timid in another.

On the quadratic above,

$$
\nabla f(x_t)=A(x_t-x^\star),
$$

so the error $e_t=x_t-x^\star$ satisfies

$$
e_{t+1}
=
(I-\eta A)e_t.
$$

Recursing gives

$$
e_t=(I-\eta A)^t e_0.
$$

Now use the eigenbasis. Since $Av_i=\lambda_i v_i$ and
$e_0=\sum_i a_i v_i$,

$$
x_t-x^\star
=
\sum_{i=1}^n
(1-\eta\lambda_i)^t a_i v_i.
$$

This is the whole spectral story of gradient descent. Each mode has multiplier

$$
r_i=1-\eta\lambda_i.
$$

If $|r_i|<1$, that component shrinks. If $r_i<0$, it shrinks while crossing the
minimum back and forth. If $|r_i|>1$, that mode diverges. The loss separates as

$$
f(x_t)-f_\star
=
\frac12
\sum_{i=1}^n
\lambda_i a_i^2(1-\eta\lambda_i)^{2t}.
$$

The square appears because the objective does not care which side of the
minimum a mode is on; it only cares how far away it is.

### The best scalar is a compromise

Assume all Hessian eigenvalues lie in $[\alpha,\beta]$. For a fixed $\eta$, the
worst contraction factor is

$$
\rho_{\mathrm{GD}}(\eta)
=
\max_{\lambda\in[\alpha,\beta]} |1-\eta\lambda|
=
\max\{|1-\eta\alpha|,\ |1-\eta\beta|\}.
$$

The best scalar step makes the flat endpoint and steep endpoint equally bad:

$$
1-\eta\alpha=-(1-\eta\beta).
$$

Solving gives

$$
\eta_\star^{\mathrm{GD}}
=
\frac{2}{\alpha+\beta},
\qquad
\rho_\star^{\mathrm{GD}}
=
\frac{\beta-\alpha}{\alpha+\beta}
=
\frac{\kappa-1}{\kappa+1}.
$$

This $\eta_\star$ is a truce. It is not trying to make the steep direction
happy; it is trying to make the worst flat and steep directions equally
unhappy. The slide [slide:gradient-descent-modes] is especially nice here:
the balanced step is where the endpoint errors match in magnitude.

For the exact quadratic,

$$
f(x_t)-f_\star
\le
(\rho_\star^{\mathrm{GD}})^{2t}
\bigl(f(x_0)-f_\star\bigr).
$$

Since

$$
\rho_\star^{\mathrm{GD}}
=
1-\frac{2}{\kappa+1},
$$

we get the familiar scale

$$
t
=
O\!\left(\kappa\log\frac1\epsilon\right).
$$

> **Remark.** Newton divides each eigenmode by its own curvature $\lambda_i$.
> Gradient descent divides every mode by the same scalar. If the spectrum is
> narrow, one scalar is good enough. If the spectrum is wide, the scalar must
> be small enough for the steep direction, and that makes it painfully
> conservative in the flat direction.

## AdaGrad: a diagonal ruler learned from gradients

The next approximation keeps more structure:

$$
\nabla^2 f(x_t)^{-1}
\approx
D_t^{-1},
$$

where $D_t$ is diagonal. The slide [slide:ada-grad] is the visual home for
this step: diagonal scaling can rescale coordinate axes, and it can be
brilliant when the axes match the curvature. But it still cannot rotate.

The first fantasy is to use the inverse diagonal curvature directly. If the
Hessian eigenvectors are already the coordinate axes, then

$$
\nabla^2 f(x_t)=\Lambda_t
$$

is diagonal, and the natural step is

$$
x_{t+1}
=
x_t-\Lambda_t^{-1}\nabla f(x_t).
$$

On an axis-aligned quadratic this reaches the optimum in one step. Lovely.
But in a rotated coordinate system, a diagonal matrix can only stretch and
shrink the displayed axes. It cannot learn the missing rotation. That is the
important limitation to keep in mind while enjoying the method.

Of course, in real optimization we do not usually know the curvature scales.
Computing them would put us right back near Newton's expense. AdaGrad replaces
the unavailable diagonal curvature vector by something it can observe:
coordinate-wise gradient activity.

Define

$$
G_{t,i}
=
\sum_{\tau=1}^t g_{\tau,i}^2,
\qquad
s_{t,i}=\sqrt{G_{t,i}},
$$

and

$$
D_t=\operatorname{diag}(s_{t,1},\ldots,s_{t,n}).
$$

Coordinate-wise AdaGrad uses

$$
x_{t+1,i}
=
x_{t,i}
-
\eta\frac{g_{t,i}}{s_{t,i}},
$$

or, in vector form,

$$
x_{t+1}
=
x_t-\eta D_t^{-1}g_t.
$$

Implementations usually add a small stabilizer in the denominator,
$\sqrt{\delta^2+\sum_{\tau=1}^t g_{\tau,i}^2}$, so that a coordinate with no
signal does not divide by zero.

The denominator measures activity, not signed progress. If gradients alternate
sign, they do not cancel inside $G_{t,i}$. A coordinate that keeps lighting up
gets a more conservative effective step. A coordinate that has barely appeared
keeps a larger gain. This is the sparse-feature magic.

### Why the diagonal toy model is so revealing

Consider the axis-aligned quadratic

$$
f(x)
=
\frac12\sum_{i=1}^n \lambda_i x_i^2,
\qquad
\lambda_i>0.
$$

Then $g_{t,i}=\lambda_i x_{t,i}$. Ignoring the stabilizer,

$$
\frac{g_{t,i}}{\sqrt{\sum_{\tau=1}^t g_{\tau,i}^2}}
=
\frac{\lambda_i x_{t,i}}
       {\sqrt{\sum_{\tau=1}^t \lambda_i^2 x_{\tau,i}^2}}
=
\frac{x_{t,i}}
       {\sqrt{\sum_{\tau=1}^t x_{\tau,i}^2}}.
$$

The curvature scale $\lambda_i$ cancels. Hooray, a real little cancellation!

> **Lemma.** \label{lem:adagrad-cancels-diagonal-curvature} On an
> axis-aligned quadratic, coordinate-wise AdaGrad with exact gradients and no
> stabilizer has the recurrence
>
> $$
> x_{t+1,i}
> =
> x_{t,i}
> -
> \eta
> \frac{x_{t,i}}
>      {\sqrt{\sum_{\tau=1}^t x_{\tau,i}^2}},
> $$
>
> so the coordinate recurrence does not explicitly contain $\lambda_i$.

This explains the core intuition. If a coordinate produces gradients $100$
times larger just because its axis curvature is $100$ times larger, AdaGrad
puts the same factor in the denominator. In the diagonal toy model, the scale
is divided out.

But for a rotated quadratic the coordinate ruler and the Hessian eigenvectors
need not agree. In eigen-coordinates,

$$
y_{t+1}
=
\left(I-\eta V^\top D_t^{-1}V\Lambda\right)y_t,
$$

and $V^\top D_t^{-1}V$ is generally not diagonal. The eigenmodes now interact.
Diagonal AdaGrad can learn axis scales; it cannot learn rotations.

The Beale comparison at [slide:beales-ada-grad-plot] is a good place to keep
this intuition honest: adaptive diagonal steps are powerful, but not magic.

### Rare features and activation counts

Define the activation count

$$
N_{t,i}
=
\left|\{\tau\le t:\ g_{\tau,i}\ne 0\}\right|.
$$

If every nonzero gradient in coordinate $i$ has magnitude $c_i$, then

$$
s_{t,i}=|c_i|\sqrt{N_{t,i}},
\qquad
\left|\frac{g_{t,i}}{s_{t,i}}\right|
=
\frac1{\sqrt{N_{t,i}}}.
$$

So the effective learning-rate decay depends on how often that coordinate has
actually been active, not on the global iteration number.

| Round | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Common coordinate step | $\eta$ | $\eta/\sqrt2$ | $\eta/\sqrt3$ | $\eta/2$ | $\eta/\sqrt5$ |
| Rare coordinate step | $\eta$ | $0$ | $0$ | $0$ | $\eta/\sqrt2$ |

At round $5$, a global schedule treats both coordinates as late in training.
AdaGrad distinguishes them: the common coordinate has five activations, while
the rare coordinate has only two. This is why AdaGrad feels so natural for
sparse data.

## The AdaGrad proof as accounting

The proof scenes are [slide:ada-grad-weighted-ledger] and
[slide:ada-grad-bound-theorem]. The first builds the weighted geometric
identity; the second turns it into the regret-style bound. The mood of the
proof is not mysterious: convexity charges each round to an inner product,
then a weighted three-point identity splits that charge into progress and
movement.

Assume:

1. $f:\mathbb{R}^n\to\mathbb{R}$ is convex and differentiable;
2. $x^\star$ minimizes $f$;
3. $g_t=\nabla f(x_t)$;
4. for each coordinate there is a radius $R_i$ with
   $|x_{t,i}-x_i^\star|\le R_i$ for the iterates in the proof;
5. AdaGrad uses $D_t=\operatorname{diag}(s_{t,1},\ldots,s_{t,n})$ and
   $x_{t+1}=x_t-\eta D_t^{-1}g_t$.

> **Theorem.** \label{thm:adagrad-coordinate-bound} Under the assumptions
> above,
>
> $$
> \sum_{t=1}^T \bigl(f(x_t)-f(x^\star)\bigr)
> \le
> \sum_{i=1}^n
> \left(\frac{R_i^2}{2\eta}+\eta\right)s_{T,i}.
> $$
>
> Consequently, for the average iterate
> $\bar x_T=T^{-1}\sum_{t=1}^T x_t$,
>
> $$
> f(\bar x_T)-f(x^\star)
> \le
> \frac1T
> \sum_{i=1}^n
> \left(\frac{R_i^2}{2\eta}+\eta\right)
> \sqrt{\sum_{t=1}^T g_{t,i}^2}.
> $$
>
> If $R_i\le R_\infty$ for all $i$ and
> $\eta=R_\infty/\sqrt2$, then
>
> $$
> f(\bar x_T)-f(x^\star)
> \le
> \frac{\sqrt2 R_\infty}{T}
> \sum_{i=1}^n
> \sqrt{\sum_{t=1}^T g_{t,i}^2}.
> $$

The weighted identity is the little engine of the proof.

> **Lemma.** \label{lem:weighted-ledger} For the AdaGrad step
> $x_{t+1}=x_t-\eta D_t^{-1}g_t$,
>
> $$
> 2\eta\langle g_t,x_t-x^\star\rangle
> =
> \|x_t-x^\star\|_{D_t}^2
> -
> \|x_{t+1}-x^\star\|_{D_t}^2
> +
> \|x_t-x_{t+1}\|_{D_t}^2.
> $$

Here $\|z\|_D^2=z^\top D z$. Geometrically, $D_t$ is a changing ruler.
Applying $x\mapsto D_t^{-1/2}x$ turns the weighted picture into an ordinary
Euclidean one, which is exactly what [slide:ada-grad-weighted-ledger] shows.

Now convexity gives

$$
f(x_t)-f(x^\star)
\le
\langle g_t,x_t-x^\star\rangle.
$$

Combining this with \autoref{lem:weighted-ledger} and summing over time gives

$$
\eta
\sum_{t=1}^T \bigl(f(x_t)-f(x^\star)\bigr)
\le
P+M,
$$

where

$$
P
=
\frac12
\sum_{t=1}^T
\left(
\|x_t-x^\star\|_{D_t}^2
-
\|x_{t+1}-x^\star\|_{D_t}^2
\right)
$$

is the progress account, and

$$
M
=
\frac12
\sum_{t=1}^T
\|x_t-x_{t+1}\|_{D_t}^2
=
\frac{\eta^2}{2}
\sum_{i=1}^n
\sum_{t=1}^T
\frac{g_{t,i}^2}{s_{t,i}}
$$

is the movement account.

For progress, write $z_{t,i}=x_{t,i}-x_i^\star$. Coordinate by coordinate,

$$
\sum_{t=1}^T s_{t,i}(z_{t,i}^2-z_{t+1,i}^2)
\le
R_i^2 s_{T,i},
$$

because $s_{t,i}$ is nondecreasing and $|z_{t,i}|\le R_i$. Therefore

$$
P
\le
\frac12\sum_{i=1}^n R_i^2 s_{T,i}.
$$

For movement, use the root-sum bound:

$$
\sum_{t=1}^T
\frac{a_t^2}{\sqrt{A_t}}
\le
2(\sqrt{A_T}-\sqrt{A_0}),
\qquad
A_t=A_{t-1}+a_t^2.
$$

This is the telescoping sweetness in the proof. Applying it to each coordinate
with $A_t=\sum_{\tau=1}^t g_{\tau,i}^2$ gives

$$
M
\le
\eta^2\sum_{i=1}^n s_{T,i}.
$$

Substitute the progress and movement bounds, divide by $\eta$, and
\autoref{thm:adagrad-coordinate-bound} follows.

### Reading the data-dependent bound

Define the final activity summary

$$
\mathcal A_T
=
\sum_{i=1}^n
\sqrt{\sum_{t=1}^T g_{t,i}^2}.
$$

The uniform-radius version of the theorem is essentially

$$
f(\bar x_T)-f(x^\star)
\lesssim
\frac{R_\infty \mathcal A_T}{T}.
$$

The important point is that $\mathcal A_T$ is observed coordinate activity,
not a worst-case global gradient bound. If only $k$ coordinates ever receive
nonzero gradients, then

$$
\mathcal A_T
\le
\sqrt{k}
\sqrt{\sum_{t=1}^T \|g_t\|_2^2}.
$$

The ambient dimension $n$ is replaced by the number of active coordinates. If
coordinate $i$ has typical nonzero gradient magnitude $c_i$ and activation
count $N_{T,i}$, then

$$
\mathcal A_T
\approx
\sum_{i=1}^n |c_i|\sqrt{N_{T,i}}.
$$

That is the practical reading: the guarantee sees different histories in
different coordinates. Nice.

## Momentum: a scalar ruler with memory

Now we go back to a scalar approximation of the inverse Hessian, but change the
direction signal. Gradient descent uses only the current gradient. Momentum
uses a discounted sum of past gradients. The Rosenbrock demo at
[slide:momentum-on-rosenbrock] and the damping regimes at
[slide:momentum-damping-regimes] are the visual companions for this section.

Let $g_t=\nabla f(x_t)$. Heavy-ball momentum is

$$
m_t=\gamma m_{t-1}+g_t,
\qquad
m_{-1}=0,
$$

and

$$
x_{t+1}=x_t-\eta m_t.
$$

Since $x_t-x_{t-1}=-\eta m_{t-1}$, this can also be written as

$$
x_{t+1}
=
x_t+\gamma(x_t-x_{t-1})-\eta g_t.
$$

The new step contains two pieces: continue some of the previous displacement,
then correct using the current gradient. The step geometry is shown in
[slide:momentum-step-geometry].

Unrolling the recursion gives

$$
m_t
=
g_t+\gamma g_{t-1}+\gamma^2 g_{t-2}+\cdots+\gamma^t g_0.
$$

Recent gradients receive large weight; old gradients fade geometrically.
Gradients that point in a consistent direction accumulate. Gradients that
alternate tend to cancel. This is why momentum can glide along long valleys
instead of nervously responding to every zig and zag.

### Momentum as discounted local models

There is a nice Hessian-lens interpretation. At iteration $t$, let every past
point vote with a discounted second-order model:

$$
Q_t(\delta)
=
\sum_{j=0}^t
\gamma^j
\left[
f(x_{t-j})
+\nabla f(x_{t-j})^\top\delta
+\frac12\delta^\top\nabla^2 f(x_{t-j})\delta
\right].
$$

The cloud in [slide:momentum-quadratic-cloud] is the intuition: enough samples
of a bowl tell us something about the shape, even before we draw the whole
surface.

Collecting terms,

$$
Q_t(\delta)
=
C_t+\bar g_t^\top\delta+\frac12\delta^\top\bar H_t\delta,
$$

where

$$
\bar g_t=\sum_{j=0}^t \gamma^j\nabla f(x_{t-j}),
\qquad
\bar H_t=\sum_{j=0}^t \gamma^j\nabla^2 f(x_{t-j}).
$$

The ideal discounted Newton step would be

$$
\delta_t^\star
=
-
\left(
\sum_{j=0}^t
\gamma^j\nabla^2 f(x_{t-j})
\right)^{-1}
\left(
\sum_{j=0}^t
\gamma^j\nabla f(x_{t-j})
\right).
$$

Now make the cheap scalar approximation again:

$$
\bar H_t^{-1}\approx \eta I.
$$

Then

$$
x_{t+1}
=
x_t
-
\eta
\sum_{j=0}^t
\gamma^j\nabla f(x_{t-j})
=
x_t-\eta m_t.
$$

So heavy-ball momentum can be read as gradient descent where the curvature
inverse is still one scalar, but the direction is learned from a discounted
stack of local models.

### Heavy-ball rate on quadratics

For the quadratic $f(x)=\frac12(x-x^\star)^\top A(x-x^\star)$, let
$e_t=x_t-x^\star$. Since $g_t=Ae_t$,

$$
e_{t+1}
=
\bigl((1+\gamma)I-\eta A\bigr)e_t-\gamma e_{t-1}.
$$

In an eigenmode with curvature $\lambda$,

$$
a_{t+1}
=
(1+\gamma-\eta\lambda)a_t-\gamma a_{t-1}.
$$

The characteristic roots solve

$$
r^2-(1+\gamma-\eta\lambda)r+\gamma=0.
$$

> **Theorem.** \label{thm:heavy-ball-quadratic-rate} Let $f$ be a strongly
> convex quadratic with $\alpha I\preceq A\preceq \beta I$. Choose
>
> $$
> \eta_{\mathrm{HB}}
> =
> \frac{4}{(\sqrt\beta+\sqrt\alpha)^2},
> \qquad
> \gamma_{\mathrm{HB}}
> =
> \left(
> \frac{\sqrt\beta-\sqrt\alpha}
>      {\sqrt\beta+\sqrt\alpha}
> \right)^2.
> $$
>
> Then every eigenmode has characteristic-root magnitude at most
>
> $$
> \rho_{\mathrm{HB}}
> =
> \frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
> $$
>
> Up to endpoint polynomial factors, the asymptotic iteration scale is
>
> $$
> O\!\left(\sqrt\kappa\log\frac1\epsilon\right).
> $$

This clean theorem is for quadratics. Heavy ball can be more delicate on
general smooth strongly convex functions, because the stored displacement may
remain tuned to curvature that has changed. Still, the quadratic microscope
explains the speedup: momentum changes the scalar recurrence from first order
to second order, and that lets the optimal dependence improve from $\kappa$ to
$\sqrt\kappa$ in the quadratic story.

## Nesterov acceleration

Heavy ball measures the gradient at the current point and then carries the
previous displacement. Nesterov acceleration changes the order: first predict
where the previous displacement would take us, and only then ask the gradient
how to correct the prediction. The look-ahead comparison is in
[slide:momentum-step-geometry].

A common smooth convex form is

$$
y_t=x_t+\theta_t(x_t-x_{t-1}),
\qquad
x_{t+1}=y_t-\frac1\beta\nabla f(y_t),
$$

with, for example,

$$
\theta_t=\frac{t-1}{t+2}.
$$

The important difference is not simply "there is momentum." It is where the
gradient is evaluated. Heavy ball corrects at $x_t$. Nesterov corrects at the
look-ahead point $y_t$.

> **Theorem.** \label{thm:nesterov-convex-rate} If $f$ is convex and
> $\beta$-smooth, an accelerated-gradient method of the form above can satisfy
>
> $$
> f(x_T)-f_\star
> \le
> \frac{2\beta\|x_0-x^\star\|_2^2}{(T+1)^2}.
> $$
>
> Thus, for fixed initial distance,
>
> $$
> f(x_T)-f_\star=O(1/T^2).
> $$
>
> In the $\alpha$-strongly convex case, fixed-parameter acceleration achieves
> the scale
>
> $$
> O\!\left(\sqrt\kappa\log\frac1\epsilon\right).
> $$

## Where Adam fits

At this point Adam should feel much less mysterious. We have two ingredients:

1. AdaGrad-style coordinate-wise scaling: divide by a measure of recent
   squared gradient activity.
2. Momentum-style direction smoothing: use a moving average of gradients
   instead of the raw current gradient.

Adam uses exponential moving averages

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,
$$

where $g_t^2$ is coordinate-wise. After bias correction,

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
$$

the update is

$$
x_{t+1}
=
x_t
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Read this through the whole note:

$$
\text{direction}
\quad
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\quad
\text{coordinate-wise ruler}.
$$

The numerator is momentum. The denominator is adaptive coordinate scaling. The
key difference from AdaGrad is memory: AdaGrad accumulates all past squared
gradients, while Adam uses an exponential moving average, so old activity can
fade.

This is useful and cheerful in practice, but it is not a free theorem. Adam's
behavior depends on the moving-average parameters, bias correction, stabilizer,
and problem geometry. The point of this note is not that Adam is automatically
optimal; it is that the formula has a clean ancestry.

## The map of the lecture

Gradient descent:

$$
\nabla^2 f(x_t)^{-1}\approx \eta I.
$$

AdaGrad:

$$
\nabla^2 f(x_t)^{-1}\approx D_t^{-1},
\qquad
D_t=\operatorname{diag}
\left(
\sqrt{\sum_{\tau\le t}g_{\tau,1}^2},
\ldots,
\sqrt{\sum_{\tau\le t}g_{\tau,n}^2}
\right).
$$

Momentum:

$$
g_t
\quad\leadsto\quad
m_t=g_t+\gamma g_{t-1}+\gamma^2 g_{t-2}+\cdots.
$$

Adam:

$$
g_t
\quad\leadsto\quad
\frac{\text{smoothed gradient}}
     {\sqrt{\text{smoothed squared gradient}}+\epsilon}.
$$

The remaining limitation is rotation. Scalar and diagonal approximations are
cheap, friendly, and often excellent. But they do not learn the full Hessian
eigenbasis. That is the big geometric caveat, and also the reason optimization
keeps being such a lively subject.
