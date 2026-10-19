# Phân tích RA-QMIX artifact diagnostic — 02/10/2026

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: phân tích exploratory từ artifacts đã hoàn tất.
- Version: artifact_diagnostic_analysis_v1.
- Verification: đọc CSV gốc và kiểm tra hashes local, không training hoặc replay thêm.
- Source run: ra_qmix_artifact_diagnostic_full_cpu_20261002T015308Z; diagnostic commit 81840a91bf8bee6ccd03c6cb0a434412c35133fc; source confirmation commit 33ab6196f0dd4d4e12038b74b65fd3c5782096f7.

## Kết luận

Ở endpoint, nominal regression là chênh lệch maintenance/queue efficiency, không phải failure avoidance. Queue waiting chiếm 0,90/1,10 = 81,82% cost delta; maintenance chiếm 0,20 = 18,18%. Curriculum policy vẫn tốt lên so với chính checkpoint boundary, nhưng cải thiện nominal chậm hơn control. Dữ liệu phù hợp với việc timing/coordination thay đổi khi adaptation; chưa xác định component gây ra thay đổi.

## Kiểm tra độ đầy đủ

Manifest COMPLETED, full profile, clean Git state; 10 training seeds 84600–84609, 50 common evaluation seeds 84700–84749, hai regimes, bốn scenarios, ba checkpoints. CSV có 12000/12000 unique episode keys và 594000 decision records. Invalid requests = 0; maximum absolute cost reconciliation error = 0. Tất cả 67 SHA256 của files nguồn trên Mac khớp source_hashes.json, gồm 60 checkpoints và 7 input files. Runner báo mọi replayed numeric metric khớp source within 1e-6 và shared boundary state_dict giống nhau. Analysis không tái chạy checkpoint; local hash và CSV audit được kiểm tra độc lập. Sealed panel 201–300 không được đánh giá bởi run này.

## Nominal theo checkpoints

| Steps | Control cost | Curriculum cost | Delta | Failure cost delta | Downtime delta | Maintenance delta | Queue delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 240192 | 14,134 | 14,134 | 0 | 0 | 0 | 0 | 0 |
| 360288 | 9,650 | 13,186 | +3,536 | +1,320 | +0,416 | +0,500 | +1,300 |
| 480384 | 8,350 | 9,450 | +1,100 | 0 | 0 | +0,200 | +0,900 |

Ở midpoint có temporary reliability gap: failure events mean 0 vs 0,088. Đến endpoint cả hai zero failures trên toàn nominal evaluation panel. Không gọi endpoint là catastrophic forgetting về reliability. Policy nominal của curriculum cải thiện 14,134 → 9,450, trong khi control cải thiện 14,134 → 8,350. Adaptation gap thu hẹp từ +3,536 xuống +1,100; không có bằng chứng từ ba snapshots này rằng cứ train lâu hơn thì nominal càng xấu.

Final paired absolute delta = +1,10, SD giữa 10 training seeds 2,413; 7/10 seeds xấu hơn, range [−3,5; +4,5]. Mean paired relative degradation từ confirmation = 16,14%, khác ratio grand means 13,17%. Giữ gate confirmation thất bại: chỉ 3/10 seeds trong nominal tolerance 10%, mean vượt 5%. Replay đúng các checkpoints cũ là xác minh và diagnosis, không phải replication trên models độc lập mới.

## Queue và request timing

Các số dưới đây là trung bình của từng training-seed metric, sau khi tính trên cùng 50 eval episodes. Metrics conditional requests không có denominator giống whole-episode metrics; không suy causal effect từ conditional comparison.

| Nominal endpoint metric | Control | Curriculum |
|---|---:|---:|
| Queue waiting steps/episode | 1,70 | 3,50 |
| Service starts/episode (tất cả preventive) | 7,50 | 7,70 |
| Requests/episode | 7,60 | 7,90 |
| Mean age tại request | 2,815 | 2,328 |
| Fraction requests gửi tới technician đang bận | 16,59% | 32,18% |
| Fraction requests chọn service dài hơn minimum feasible | 0% | 4,84% |
| Chosen service time mean | 2,289 | 2,365 |

Curriculum yêu cầu sớm hơn khoảng 0,49 age units và busy-request fraction tăng khoảng 15,60 percentage points. Gửi request khi technician bận là hành động hợp lệ của FIFO, không phải invalid action. Chọn technician có service dài hơn chưa chắc tệ nếu bù được queue/availability; phải đọc cùng request context.

Action traces ghép theo seed/eval seed/time/machine: boundary có 0/18000 differences, midpoint 6450/18000, endpoint 5450/18000 (30,28%). Trong endpoint differences, 5350/5450 (98,17%) là request/defer và 100/5450 là hai technician IDs khác nhau. Đây là mô tả rollout sau khi states đã có thể phân kỳ, không phải controlled same-state intervention. Đồng thời defer có thể bị ép bởi action mask khi queued/assigned; không quy mọi request/defer difference cho quyết định timing tự do.

Seed 84606 có delta +4,5: queue +3,5, maintenance +1; request count 8 → 11, request age mean 3,0 → 1,636. Seed 84601 tốt hơn −3,5: maintenance −3, queue −0,5. Các seed xấu không bị loại khỏi summary. Mẫu hình không đồng nhất là lý do giữ phân tích paired-seed, không chỉ grand mean.

## Stress tradeoff

| Scenario | Control cost | Curriculum cost | Cost delta | Failure + downtime delta | Maintenance + queue delta |
|---|---:|---:|---:|---:|---:|
| Early failure | 137,870 | 86,598 | −51,272 | −60,540 | +9,268 |
| Slow service | 98,506 | 84,588 | −13,918 | −18,624 | +4,706 |
| Combined pressure | 187,404 | 156,043 | −31,361 | −36,042 | +4,681 |

Relative raw cost reductions theo từng scenario là 37,19%, 14,13%, 16,73%. Không gộp raw costs giữa horizon khác nhau. Curriculum mua giảm failure/downtime bằng nhiều maintenance và queue cost hơn trong stress; ở nominal khi cả hai đã tránh hết failure, cùng kiểu request pattern không tạo thêm reliability benefit quan sát được. Đây là lời giải thích phù hợp decomposition, chưa chứng minh causal mechanism của CF hoặc mixer.

## Training logs và Q margins

Mỗi seed/regime có tổng 120033 optimizer updates: 59985 ở phase one, 30024 ở mỗi khoảng sau. Epsilon range khớp: phase one 1 → 0,40625; đoạn tiếp tới 0,10938; đoạn cuối tới 0,05. Vì vậy không có chênh lệch budget/update/exploration schedule để giải thích outcome trong panel này. Phase-two curriculum có 59970–60045 stratified updates và 3–78 warmup updates trước khi stress buffer đủ; sau warmup batch split đúng 112 nominal + 16 stress theo protocol/source audits.

Cột first_updates/last_updates trong diagnostic training_log_summary.csv ghi update_count của episode đầu/cuối, không phải cumulative counts. Không lấy hiệu hai cột đó để suy số updates. Derived training_update_totals.csv ở báo cáo này cộng update_count từ log nguồn để sửa cách diễn giải; raw artifacts được giữ nguyên. Trong derived CSV td/cf/weighted_cf/total là tổng episode mean nhân episode update count; chia cho updates để lấy update-weighted mean.

Đoạn cuối, update-weighted TD loss mean control 5,9122, curriculum 12,2625; weighted CF 0,7119 vs 0,6871. Raw CF 14,2389 vs 13,7415. Khác replay distribution làm scalar TD loss không thể so trực tiếp như cùng test loss. CF scalar không tăng đủ để gán nominal regression cho CF; scalar magnitude không đại diện gradient magnitude. Historical gradient norms/replay sample states chưa được log nên vẫn thiếu.

Average per-seed median feasible-action Q margin cuối: control 0,5615, curriculum 0,3453. Q scales chưa được calibrate giữa networks; margin nhỏ không phải xác suất sai. Seed 84603 vừa có degradation +2,5 vừa có median margin 0,9224, trong khi seed 84601 cải thiện −3,5 có median margin 0,0358. Không coi low Q margin là nguyên nhân chung.

## Quyết định cho vòng tiếp theo

1. Giữ nguyên verdict: curriculum + replay giúp stress nhưng chưa đạt nominal-preservation gate. H1 decomposition được hỗ trợ ở endpoint; H2 divergence sau boundary được quan sát, chưa kết luận causal.
2. Ưu tiên kiểm tra request timing/queue coordination. Chưa đủ evidence để bỏ technician-edge encoder, thêm graph model hoặc chỉ tune CF weight. Không hard-mask busy requests vì điều đó đổi action semantics của FIFO.
3. Vòng can thiệp nhỏ vẫn hợp lý: Full và no-CF × nominal-only và curriculum, matched steps/updates, fresh seeds, cùng gate. Capacity-matched QMIX cần giữ cho architecture attribution. Nếu muốn tách timing và technician choice trực tiếp, phải thiết kế/test một controlled same-state policy intervention riêng, không suy từ rollout differences hiện tại.
4. Sửa nghĩa logging update_count trong lần phát triển runner tiếp theo; bổ sung gradient norms TD/CF, per-scenario replay loss và timing/queue metrics. Mọi sửa code/full run tiếp theo theo workflow tests → Mac smoke → commit/push → user lab command.
5. Không sửa simulator trong vòng attribution. Recovery-at-start không charge toàn bộ service downtime và vẫn cho age increment trong service; kết luận chỉ về cost abstraction đã khóa. Recovery-at-completion/physical availability là experiment riêng.

## Kiểm tra giới hạn diễn giải — 11/11 loại

Simpson/ecological: không gộp scenarios/horizons, suy luận ở training seed. Berkson/collider: request-conditioned metrics chỉ mô tả, không control để claim causality. Base rate: nominal endpoint failure base rate zero được ghi rõ. Regression-to-mean: có paired control và shared boundary; không chọn worst seed để infer chung. Survivorship: đủ toàn bộ 10 seeds/12000 episodes, không bỏ seeds xấu. Look-elsewhere/forking paths: kiểm tra nhiều mechanism metrics là exploratory, không p-values hoặc gọi là fresh confirmation. Correlation/causality và reverse causality: rollout states phân kỳ, Q margins/queue patterns chưa xác định nguyên nhân component. N/A áp dụng cho các suy luận dân số/người tham gia không có trong synthetic benchmark; kiểm tra này không chứng nhận mọi bias đã bị loại.
