## Material Passport

- Origin: ARS experiment-agent, inline local engineering.
- Date: 2026-10-12, Asia/Taipei.
- Verification: engineering VERIFIED; full pilot test outcomes unopened.

# Engineering gate PASS

The confirmed fixed joint setting and user-authorized Mac execution are unchanged. The learner budget, fresh train/development/test registry, checkpoints, masks, objective and exploratory statistics were written prospectively. No learner performance tuning from smoke. This pilot does not replace the independently confirmed setting verdict or require a MARL win.

Complete suite:880 passed, one existing SB3 rich/tqdm warning,258.44s. Subsequent concrete audit revisions received final targeted12 tests: masked environment, independent60-tick scalar return, checkpoint/RNG preservation, training-worker provenance and step reconciliation, tail economics/checkpoint identity tamper, crossed bootstrap, timeout and no-overwrite. Final real Mac smoke on revised runner: COMPLETED / audit PASS,512 learned steps,12 test episodes/720 replayed physical decisions,10 completed training episodes plus two partial16-tick trajectories; five saved/reloaded checkpoints and development-only selection. All36 SHA256/size artifact bindings independently checked. Scientific ENGINEERING_ONLY, full test unopened. Smoke path /Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_joint_learning_mac_v1_smoke_cpu_20261012T064425Z,2.156s.

Checkpoint validation preserves Python/NumPy/Torch RNG state; SB3 load cannot restart training streams. Checkpoints occur at rollout start after the previous gradient update and at training end after the last update; metadata counts optimization epochs and completed rollout updates correctly. Full return includes the12-tick common tail in the terminal learned reward. Every masked model prediction is reconstructed in scalar replay; all reported episode fields and checkpoint identities checked. Training accounting uses contiguous worker episode IDs and domain-separated physical seeds, with explicit incomplete trajectories at fixed step budget.

Runtime includes exact installed Torch/NumPy/SB3/sb3-contrib/Gymnasium versions. No new dependency. Models live under per-replica directories; overwrite forbidden. Source/checkpoints and selections hashed. Full:3 independent training replicas ×65,536 steps =196,608 learned steps;32 development seeds ×4 checkpoints ×3 replicas =384 development episodes;128 new paired test seeds ×10 controllers =1,280 test episodes/76,800 physical ticks. All losses retained; exploratory four95% bootstrap intervals only, no hidden algorithm success gate. Commit/push these verified seven files, then execute full on this Mac per explicit user authorization. No SSH/run/transfer on lab.
