# Frozen removal-credit actor-gradient variance diagnostic v1

Prospective, before implementation/new diagnostic states. Source: verified c6d72dc
FULL removal-PPO DEV, main gain -0.161234%, blind gain -0.220084%, FAIL17/19.
Preserve that negative result, its original protocol and all sealed CF/TEST/N>4.
Late return-residual variance ratio .972829 and nonzero context weights do not
establish useful actor-gradient noise reduction. No LR/encoder/budget/entropy
sweep, no new training, teacher, optimizer or attempt to rescue the source verdict.

Specific question: at the SAME frozen actor/critics and paired NEW stochastic
trajectories, does replacing state V by conditional removal B materially reduce
the variance of actual shared-actor parameter gradients? Scalar residual variance
need not equal gradient variance. Also compare proposal-score/actual-removal
gradient to its physical-factor/conditional-removal Rao-Blackwell estimator.

Use ONLY candidate removal_rb_ppo, ALL3original random replicas and ALL5fixed
checkpoints0/122880/245760/368640/491520. No best checkpoint/window/replica choice.
128 independent H48+common12tail episodes per group, same128fresh physical tapes
across all15groups, distinct action stream per group. Total1920episodes,
92160policy decisions,115200physical transitions,0optimizer updates. N4/K2 and
original physical/reward/health semantics unchanged. OneCPU/BLASthread,tqdm,
60minute hardstop,newUTCdirectory,no overwrite/resume/extra samples. No separate
smoke; correctness/integration tests then commit/push before this FULL diagnostic.

Promote stored float32 tensor VALUES exactly into float64 for both collection
and gradient evaluation, avoiding mixed precision in this algebra diagnostic.
This is a frozen F64 re-evaluation, not bitwise original F32 behavior. Actor
sampling uses explicit generator and independent proposals. No greedy evaluation
or action-dependent filtering. Critic parameters frozen; actor autograd enabled
only to measure gradients, no parameter changes. Cache V, actual b and B at fixed
weights; compare to independent NumPy dense25proposal baselines on every state.

For episode e, G_t is exact reverse sum of own -cost/20 rewards including tail
in the final reward. Save THREE full11141parameter vectors, each /48/2:

    gV_e = sum_(t,i) grad log f_i(S_t) * (G_t - V(s_t))
    gB_e = sum_(t,i) grad log f_i(S_t) * (G_t - B_i(s_t,S_t))
    gb_e = sum_(t,i) grad log p_i(U_ti) * (G_t - b_i(s_t,M_minus_i))

No centering, scale fitting, clipping, entropy/critic gradient or optimizer. These
are unclipped reward-gradient estimators, not an emulation of full finite PPO.
Full episodes are independent sampling units; do not treat correlated timesteps
as independent. Trace covariance = sum_e ||g_e-mean(g)||²/(128-1).
Primary per-checkpoint gain=1-trCov(gB)/trCov(gV). Primary mechanism screen:
EACH replica's unweighted mean gain over ALL5checkpoints >=5%, and >=3/5
checkpoints have strictly positive gain; all denominators finite and >1e-20.
5% is the prospectively locked material diagnostic threshold; point diagnostic
only, not scientific CI/confirmation or proof of better cost. Report all15points,
sample gradient mean norms/differences, gB vs gb variance and scalar residual
ratios; no filtering zero/low-entropy/poor points. Step0 equality is a correctness
check, not evidence of learned benefit. Any failure means this frozen diagnostic
does not establish the hypothesized material actor-gradient benefit; no automatic
retuning. Even ALLPASS does NOT override source cost failure or open CF/scaling.

Independently reconstruct every primitive transition/event/population/return,
every stochastic draw from NumPy actor probabilities and explicit generator,
and every gradient through a closed NumPy MLP VJP (no production autograd).
F64 factor/proposal logs, V/b/B absolute tolerance1e-9+1e-10|reference|;
full gradient vectors1e-9+1e-9|reference|. No tolerance relaxation to rescue a run.
Tests include finite-difference VJP, covariance math, step0 invariance and learned
nonconstant critic with masked states. Source bindings checked before/after,
all weights sealed unchanged; do not claim independent historical optimizer replay.

Artifacts: protocol/registry/resolved settings/source bindings/sealed code,
manifest and JSON summaries, per-group observation/action/return/baseline arrays,
three full gradient matrices (npz), per-episode metrics CSV and complete physical
and policy journals. Independent checks must PASS before interpreting metrics.
Fresh raw physical340100000..127, group action bases340000000..14,
analysis340200000; engineering physical340700000/action340700001. All derived
roles THIS diagnostic version. Raw/derived separation/history audit before data;
source model train seeds/checkpoints are intentional frozen references, not newly
claimed fresh seeds. No scientific confirmation/test states are constructed.
