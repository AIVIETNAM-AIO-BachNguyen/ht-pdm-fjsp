# Kết luận goal nghiên cứu: RL điều phối bảo trì nguồn lực dùng chung

**Kết luận:** đã xây dựng và chạy được framework RL tập trung có simulator,
feasibility, baselines và evaluation tách bạch; kết quả khoa học hiện tại **chưa
chứng minh PPO vượt heuristic/planner mạnh một cách nhất quán**. Đây là nền
nghiên cứu có thể mở rộng, chưa phải phương pháp sẵn sàng triển khai nhà máy
hoặc claim một bài báo hoàn chỉnh. Mọi kết quả dương và âm của hai vòng được
giữ trong `research_result/`; không điều chỉnh physics/objective để cứu PPO.

## Hợp đồng ra quyết định

Một controller của plant quyết định ở mỗi tick. Input được phép: tuổi/hỏng và
request PM/CM hiện tại của bốn máy, technician rảnh/bận và thời gian còn lại,
ma trận khả năng/thời gian service, clock; không có realized future events.
Requester threshold age≥3 tạo PM cố định (hỏng tự tạo CM). Controller chọn
partial matching request–technician, có thể chọn subset, ưu tiên máy, phân công
worker hoặc WAIT. Một technician/một job, một máy/tối đa một technician;
service không ngắt, phục hồi tại completion. Chọn một job chiếm shared worker
nhiều tick và ảnh hưởng thời gian chờ, hỏng và khả dụng của các máy khác. Reward
là âm cost PM/CM start + unavailable machine-ticks + new failures, tính cả
H48+common tail16. Không áp waiting penalty nhân tạo. Đây là vấn đề theo thời
gian: PM chờ vẫn suy giảm, CM queue tồn tại, service nhiều tick và chọn worker
nhanh hôm nay ảnh hưởng cơ hội của request khác ngày sau.

## Tính đúng, đánh giá, bằng chứng

Kernel cũ đã có unit tests về conservation, lifecycle, completion, feasibility,
random tape và terminal effects; diagnostic trước goal replay độc lập 221.184
transitions. Framework mới có Gym adapter, controller interface cho rule,
rollout và RL, mask legal matching, reward H+tail, checkpoint/seed provenance.
48 tests v1 và 50 tests v2 targeted PASS; hai smoke Ubuntu PASS. Hai full run
COMPLETED, hai verifier độc lập PASS, zero invalid assignment. Future tape nằm
bên trong env và không đi vào observation/controller; mask bảo đảm feasibility.

| Vòng | Protocol, source | TEST và verdict |
|---|---|---|
| V1 PPO chỉ train anchor | `docs/maintenance_framework_v1_protocol_vi.md`, `2fc6840` | 1.024 episode; anchor PPO 817,45 vs risk 839,91, CI paired PPO−risk [−60,94;12,92], nên H1 chưa xác nhận. PPO thua rollout và hai case phụ; thắng risk theo interval mô tả ở reciprocal scale3. |
| V2 PPO train/chọn bốn case | `research_result/framework_v2_protocol_vi.md`, `526272a` | 1.280 episode; primary mixture−v1 pooled +8,64 [−0,68;+18,91], H1 không đạt. Specialist cải thiện −41,30 nhưng anchor +28,02 và reciprocal scale3 +55,97. |

V1 evidence: `framework_v1_findings_vi.md`, raw
`framework_v1_full_20261019T153014Z/`, audit
`framework_v1_audit_20261019T154240Z/`. V2 evidence:
`framework_v2_findings_vi.md`, raw
`framework_v2_full_20261019T154912Z/`, audit
`framework_v2_audit_20261019T160153Z/`. Raw/checkpoints/logs nằm ngoài Git;
Git chứa code, protocols, verifier và báo cáo nhỏ. Các untracked documents
cũ ngoài phạm vi goal được giữ nguyên.

## Giới hạn và chương trình nghiên cứu tiếp

N4/K2 synthetic hazard, bốn config và finite tail chưa đại diện toàn bộ
sản xuất; không có operation routing/precedence, dữ liệu nhà máy hay quyết
định gửi request bằng RL. PPO MLP output là 21 matching cố định, chưa scale
N/K. Rollout là bounded forecast, không phải exact optimum; fairness về
information/physics/seeds/objective được giữ nhưng compute khác rõ. Chọn model
bằng DEV và chỉ một replica được đem vào TEST; CI trên physical seeds không
bao phủ đầy đủ uncertainty do training-seed selection. V2 đổi đồng thời
training distribution và checkpoint selection, không xác định riêng effect.

Hướng hợp lý tiếp theo vẫn lấy RL làm controller: dùng representation có cấu
trúc request–worker và đánh giá action timing/assignment trên config mới;
hoặc huấn luyện từ train-only rollout teacher rồi PPO fine-tune, so warm-only,
scratch và planner với đầy đủ teacher compute. Khóa protocol, fresh seeds,
health và hardstop trước mỗi run. Cần đánh giá riêng variable N/K, skill graph,
busy worker và production scheduling nếu mở rộng. Không tuyên bố novelty,
optimality hoặc khả năng ứng dụng thực tế từ dữ liệu hiện tại.

## Tái lập trên lab hiện tại

```bash
cd /home/bachng/ht-pdm-fjsp
uv sync --frozen
uv run pytest -q tests/test_maintenance_framework_v1.py tests/test_maintenance_framework_v2.py tests/test_maintenance_dispatch.py tests/test_maintenance_requests.py tests/test_maintenance_request_experiment.py
RUN_ID="framework_v1_replica_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python -m ht_pdm_fjsp.maintenance_framework_v1_campaign --profile full --device cpu --output-dir "research_result/$RUN_ID"
```

Exact v2 reproduction uses the retained original v1 selected checkpoint at
`research_result/framework_v1_full_20261019T153014Z/checkpoints/62000002/199680.zip`
with SHA256 `6f1400b2e11990d86dbd14d137b85c2f2ac73ccf1c1c380bee8d03a779727cb9`:

```bash
RUN_ID="framework_v2_replica_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python -m ht_pdm_fjsp.maintenance_framework_v2_campaign --profile full --device cpu --output-dir "research_result/$RUN_ID"
```

Mỗi lệnh tạo thư mục mới, không ghi đè. Exact campaign provenance còn trong
manifest/source hashes của run gốc. Nếu checkpoint v1 gốc bị mất, v2 runner sẽ
từ chối thay comparator âm thầm; chạy lại v1 có thể tái lập quy trình nhưng
không tự được coi là cùng frozen comparator nếu checkpoint hash khác.
