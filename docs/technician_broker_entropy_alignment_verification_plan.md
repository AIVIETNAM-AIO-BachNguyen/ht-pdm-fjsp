# Independent entropy alignment verification

Read-only audit after source diagnostic is complete; no new learning or change
to its failed screen. Hash all inputs/source checkpoints before/after. Replay all
92160physical transitions with independent event tapes/primitive physics, all
73728policy-state observation hashes/supports/recorded likelihoods/values and
1536episode cost/health metrics. Use independent manual float32 MLP and same
private generator hashes to replay ALL stochastic draws; source policy class is
not used in verifier. Recompute MC by explicit future sums and normalized source
float32 advantages. Independent manual float64 MLP plus CLOSED product-of-sums
matching masses (versus source25preimage matrix) recomputes Hp/Hq/all11141actor
gradients including four reward blocks on all24checkpoints. Compare with source
float32 (entropy/probabilities/logs atol5e-6, gradients atol5e-6+rtol2e-4), record
max errors; original behavior value tolerance retained. Recompute all metrics
and12primaryreplica checks from stored gradients using ordinary scalar math.

Tests synthetic/ENG only, commit/push before execution, tqdm,oneCPU/BLASthread,
15minute hardstop,new timestamp/refuse overwrite,preserve failure. No CI/Adam
step/population-variance/training-effect/novelty/scaling claim from audit PASS.
