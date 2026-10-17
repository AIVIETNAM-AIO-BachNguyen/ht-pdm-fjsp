# Post-hoc optimistic scalar-mixing ceiling, before computing this analysis

No new data/seed/model learning. Original FULL costFAIL and frozen gradientFAIL
remain unchanged. Use ALL15independently verified stored gradient matrices, not a
chosen window. This is an exploratory in-sample quadratic ceiling, NOT a held-out
effect, population bound, scientific confirmation or trained policy result.

Write h=gV-gB. The mixed baseline V+lambda(B-V) gives g(lambda)=gV-lambda*h.
On each full128episode sample, v=traceCov(gV),d=traceCov(h),c=traceCov(gV,h).
If d>1e-20, lambda*=c/d minimizes the sample quadratic; optimistic gain=c²/(d*v).
If d<=1e-20, use lambda=0,gain0. Save all15coefficients/v/d/c/gains, verify the
quadratic value against direct centered covariance of mixed gradient vectors.
No coefficient grid/search/clamping/selected seeds. This is classical control
variate coefficient fitting; not an algorithmic novelty claim. A coefficient fit
on the evaluation sample may overfit and must NOT be used as scientific benefit.

Decision screen only: compute mean optimistic gain across ALL5nodes per replica;
if ANYmean<5%, do not launch a scalar-coefficient tuning experiment to rescue this
configuration. Even if allmeans>=5%, a separately registered fresh fit/holdout
diagnostic would be needed before new learning, with no source verdict change.
No automatic experiment/promotion/CF/test/scaling from this analysis.

Artifacts freshUTC, source/audit bindings and copies of code/plan, all15records,
three means and VI findings. OneCPU,5minutehardstop,tests/commit/push before
analysis, preserve failures. Tests check analytic minimizer, zero control and
derivative/direct quadratic equality. No additional smoke/new physical state.
