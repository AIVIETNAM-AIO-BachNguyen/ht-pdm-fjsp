# Engineering gate — maintenance policy continuation v1

Material Passport: academic-research-suite / experiment-agent; inline plan/run;
2026-10-11; maintenance_policy_continuation_v1; VERIFIED engineering gate PASS.
Scientific full forecast outcomes remain unopened on Mac. The locked plan's
initial UNVERIFIED label is preserved as a prospective record; this report
records the completed engineering gate without changing the protocol.

Thí nghiệm đã sẵn sàng cho lab. Dùng nguyên 10 checkpoint đã chọn trước và
2.560 trạng thái đã có, không train lại. Tại cùng một trạng thái, can thiệp vào
quyết định hiện tại rồi để RL deterministic tự quyết định trên trạng thái riêng
của từng nhánh. Bốn nhãn: action gốc của RL; chờ một bước; bỏ riêng PM nhưng giữ
nguyên mọi cặp sửa chữa CM/kỹ thuật viên; action tham chiếu đã đóng băng. Gộp các
action trùng nhau. Cùng kịch bản sự cố cho mọi nhánh và cả hai continuation.

Horizon đúng training: RL chạy trong các interval đến47, quan sát48-global_time;
đoạn rule tail deadline_matching48..59 dài đúng12 bước. Công việc đang làm không
bị reset ở48. Tính toàn bộ chi phí từ tick lấy mẫu đến hết59, tách base/tail và
lưu trạng thái vật lý ở48/60. Nhánh deadline chạy tiếp trên cùng tổng horizon
là kiểm soát phụ. Đây là deployed deterministic policy, không phải hành vi
sampling stochastic của quá trình training.

Bốn contrast chính: PM-deferral và WAIT-all, mỗi contrast ở dense/skill_mask.
WAIT-all bỏ cả PM và CM trong bước hiện tại; không diễn giải thành PM-only.
Dương nghĩa là can thiệp chờ/bỏ PM có chi phí thấp hơn, âm ủng hộ action gốc.
CI98.75% dùng20.000 crossed bootstrap model/profile; các nhóm phụ descriptive.
Input là panel cũ đã phân tích, nên đây là exploratory mechanism follow-up,
không phải kiểm định tổng quát hóa trên trạng thái vật lý mới.

## Verification

- Focused suite:19 tests passed in12.35s. Whole repository:644 tests passed
  in213.29s, one existing TqdmExperimentalWarning from stable_baselines3.
- Feature/mask equality to scalar feature_batch is exact on dense/skill_mask,
  with learned time48/6/1; batch inference actions agree. Scalar fallback for
  near-tied logits is tested. Model weights remain fixed.
- Tick6/42/47 fixtures replay every arm/root/scenario action, component, base/
  tail return, cutoff/final state. Six-tick repairs begun at47 remain assigned
  with5 ticks outstanding at48. Forced WAIT and keep-CM interventions tested.
- Source hashes/provenance, complete grid/roots/role seeds, fresh seed audit,
  conditional bootstrap, missing-source/dirty/thread/nonempty guards and
  interruption with retained partial states tested. Synthetic fixture results
  provide engineering evidence only.
- Actual-checkpoint smoke: maintenance_policy_continuation_v1_smoke_20261011T080617Z; COMPLETED/PASS,6.011s overall,
  4 historical states,11 unique roots,22 root-value rows,1.584 forecast samples,
  220 continuation RL decisions,1.056 tail transition samples. Every action,
  cost component and boundary/final state matches independent scalar replay.
- All85 historical source artifacts and all10 checkpoint roots/payloads verified
  against locked contract; one model active in smoke. All79 new smoke output
  files verified independently by hashes/bytes (2.740MB excluding manifest).
  Reloaded checkpoint weights/root decisions agree, no weight changes.
- Full forecast and analysis raw seeds14100000/14200000 remain unopened; smoke
  uses14300000/14300010. Historical source states deliberately reused. Registry
  audit PASS:1.284 distinct new derived roles, no raw/derived collision.

## Runtime and handoff

Full fixed budget:2.560 states ×32 future scenarios, up to4 roots ×2 arms;
max23.592.960 physical transition samples and7.536.640 batched continuation RL
rows. No training, no new physical trajectories, no action oracle search.

Engineering benchmark uses ONLY the four locked smoke states at42, with32
scenarios, identical smoke forecast roles, repeated3 times. Per-state fork time
0.0139–0.0179s on Mac. Inference-clock linear extrapolations are195.8s for mean
landmark depth and375.1s for the longest depth applied to every state: roughly
3–7min for the simulation portion on this Mac. These are not timing bounds.
Earlier-state action distributions, near ties, source preflight, bootstrap,
serialization and Ubuntu hardware may change runtime. No full forecast outcome
was executed to obtain this estimate. Raw numeric cost/boundary cubes total
~147MB before compression; additional CSV/snapshot/model data is modest.
Reserve250MB for artifacts; compressed size depends on outcomes.

Full run is authorized for the user to execute in the Ubuntu lab terminal.
Only dependency directory needed is
artifacts/maintenance_timing_diagnostic_v1_full_cpu_20261011T072307Z, including
its ten frozen_models. Original joint run is retained as pinned provenance in
the source contract but need not be downloaded again. Full requires clean code,
CPU/thread1 and a new timestamp directory. Failed runs keep partial records;
no resume contract. User executes the exact lab and rsync commands at handoff.
Assistant has not launched a lab run, used SSH or started transfer.
