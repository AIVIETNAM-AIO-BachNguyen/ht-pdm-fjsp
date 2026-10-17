# Trạng thái nghiên cứu technician N4/K2 — 2026-10-17

**Mục tiêu tổng thể chưa đạt.** Typed-Gibbs đã qua DEV12/12 nhưng vòng xác nhận
5trainingseeds mới/256TESTseeds **không đạt:4/12 CI gates**. Đã kiểm chứng độc lập
đủ dữ liệu và phép bootstrap; không đổi ngưỡng hay chọn replica/checkpoint khác.
N lớn vẫn đóng. Không teacher/distillation/PPO warm-start; giữ mọi kết quả âm.

## Mới nhất: xác nhận typed-Gibbs N4 — không đạt độ ổn định và failure bounds

35mô hình mới×491.520decisions=17.203.200decisions,358.400trainEP,268.800Adamupdates.
Freeze đủ35model hashes sau11.200CFDEVnodes rồi mới mở256TESTseeds. Kết thúc
28phút55giây;9.216TESTepisodes,552.960physical transitions,442.368policy decisions.
Không test-time optimizer, không best-checkpoint hoặc seed filtering.

| Nhánh | TEST H60cost/máy | H48 failure/máy |
|---|---:|---:|
| Typed Gibbs |264,295898|0,225000|
| Myopic đối chứng chính |282,877930|0,687891|
| Myopic reverse |274,108594|0,422852|
| PPO gốc |263,754883|0,146875|
| PPO normalized |263,310938|0,121875|
| MAPPO local |268,924609|0,432422|
| MAPPO full |265,751758|0,358398|

Điểm gain vsmyopic6,569%, nhưng lowerbound **−1,032%**, chưa đạt+.5%; cả6cost
CI gates đều không đạt. Gain mỗi replica10,423%/10,084%/0,844%/10,915%/**−0,731%**,
extra every-replica-positive cũng không đạt. Mean gần PPO không chứng minh N.I.:
lowerbound vsPPO−1,897%,vsnorm−2,377%, dưới biên−1% đã khóa. Failure-delta upper
bounds vsPPO0,279343 vàvsnorm0,260742, vượt mức+.25. Availability/pending4bounds
đạt,zero invalid,p95 N4 actor+DP+decoder3,101ms/720samples. Không thắng PPO.

Điểm12/12 và DEV12/12 **không thay thế** CI4/12. Cơ chế learnedfuture chưa được
xác nhận ổn định theo protocol; không đổ lỗi mặc định cho thiếu budget/encoder/LR.
Không mở N lớn từ kết quả này. Tiếp theo chỉ được chẩn đoán nguyên nhân cụ thể
trên train/CFDEV và đăng ký follow-up có giới hạn trước dữ liệu mới; giữ test cũ.

Independent audit PASS102,59s:35manualinitializers,51.200pairedtrain tapes, đủ
CFDEV/freeze/TEST,NumPyNN/explicit21matching, toàn bộ200.000bootstrap indexrows và
2.400.000endpointvalues bằng explicit Cartesian indexing. Hai actualPNG đã nhìn
và kiểm tra layout. Audit PASS xác nhận execution/verdict âm, không phải đạt mục tiêu.

[Kết quả xác nhận và tất cả12bounds](technician_typed_gibbs_confirmation_v1_findings_vi.md)
· [Protocol xác nhận](technician_typed_gibbs_confirmation_v1_plan.md).

## Chẩn đoán sau xác nhận: chưa cho phép tăng budget

Đã audit18.000frozen policy-state evaluations trên240CFDEV states có sẵn,0training/
physicalstates mới. Candidate pooledlateCFDEVgain0,6783%,4/5replicas dương, nhưng
meanfixed-stateTV chỉ0,001773 dưới ngưỡng.01; không mở2x-budget study từ readout này.
Đây chỉ là một pool của1CFDEVphysicalseed, không chứng minh hội tụ hoặc nguyên nhân.
Không thêm encoder/LR/attention để cứu verdict. [Chẩn đoán và giới hạn](technician_typed_gibbs_optimization_audit_v1_findings_vi.md).

## Vòng trước: typed-capacity Gibbs PPO — đạt DEV, chưa xác nhận khoa học

Mỗi máy phát utility cho hai loại technician; broker tính xác suất joint matching
bằng weighted count DP, bao gồm utility của những máy phía sau. Giữ PPO chuẩn.
Đối chứng chính cùng mạng/thông tin/dung lượng/ngân sách chỉ đếm feasible future
continuations, bỏ learned future utilities. Cả hai có exact physical entropy/MAP,
khởi tạo cùng phân phối uniform matching. Đây không phải fine-tuning PPO gốc.

21fits×491.520decisions,3trainingseeds mới/32DEVseeds,10.321.920decisions và
215.040trainEP; kết thúc15phút19giây. Fixed final, không chọn checkpoint tốt nhất.

| Nhánh | Cost/máy H60 |
|---|---:|
| Typed Gibbs candidate |263,575521|
| Myopic, đối chứng chính |281,971354|
| Myopic reverse |297,578125|
| PPO gốc |263,171875|
| PPO normalized |263,380208|
| MAPPO local |266,583333|
| MAPPO full information |263,591146|

Giảm cost6,524% so myopic, ba replica đều dương1,790%/8,499%/8,929%. Đạt12/12
tiêu chí điểm và các extra guards; cả sáu health bounds đạt. Cost cao hơn PPO
gốc0,153% và normalized0,074%; **chưa thắng PPO**. So full MAPPO chỉ thấp hơn
0,006%, thực tế gần như ngang điểm; chưa có bằng chứng ưu thế thống kê.
P95 actor+DP+decoder3,076ms tạiN4, chưa có bằng chứng scalability.

Audit độc lập PASS10,72s: dựng21fresh actors/critics, đủ215.040trainEP/3.360DEVnodes,
704finalEP/42.240primitive transitions/33.792policy decisions, NumPy utilities và
explicit matching densities/entropy/MAP,12gates và432rawtimings.56unique correctness
tests đã đạt qua các đợt (55combined trước khi thêm một seed-contract test; latest
verifier14PASS). Hai PNG report đã kiểm tra trực quan. Knockout giữ trọng số chỉ
đổi MAP ở38/4.608trạng thái với myopic,147/4.608với reverse (loại near ties); đó
là mô tả, không thay retraining causal contrast. Không đổi protocol từ kết quả.

[Phương pháp/kết quả đầy đủ](technician_typed_gibbs_ppo_v1_findings_vi.md) ·
[Protocol FULL đã khóa](technician_typed_gibbs_ppo_v1_plan.md) ·
[Ranh giới prior art](technician_typed_gibbs_prior_art_update.md).
Global normalization/structured policy/partition DP không phải ý tưởng mới chung;
đóng góp cụ thể và test/scaling vẫn cần chứng minh. Vòng xác nhận riêng đã thực hiện như phần mới nhất ở trên và không đạt; không dùng phần DEV này để cứu verdict. N4/K2 vẫn là trọng tâm.

## Vòng trước: capacity-priority machine-agent PPO — không đạt

Một giả thuyết policy riêng đã được khóa, triển khai và chạy FULL từ đầu: cho máy ít lựa chọn phục vụ nhanh chọn technician trước, rồi broadcast reservation. Dùng raw MLP104→64→64→3, PPO scalar joint chuẩn và prior đếm matching để bốn nhánh phối hợp có cùng phân phối matching lúc khởi tạo. Đây là đổi chiều/thứ tự phân công, không thêm removal critic hay cứu scalar calibration.

33 fits,16.220.160 quyết định mới, kết thúc21phút28giây. Kết quả12/16, không đạt; cả ba seed đều kém fixed ID order.

| Nhánh | Cost/máy H60 |
|---|---:|
| Capacity order candidate |265,716146|
| Fixed ID order |263,898438|
| Independent machine physical-q |267,682292|
| PPO gốc |262,755208|
| PPO normalized |262,773438|
| MAPPO full information |265,385417|

Candidate tốt hơn independent0,735%, nhưng kém fixed ID0,689% và kém PPO khoảng1,12%; cả sáu health gates đạt. P95 N4 actor+decoder1,929ms. Independent verification PASS14,76s bằng matching-completion counts/NumPy actors/primitive physics, gồm all81latent proposals cho physical-q. Report/PNG đã kiểm tra trực quan; đây vẫn là DEV, không CI/scaling.

[Phương pháp và đủ16tiêu chí mới](technician_capacity_order_ppo_v1_findings_vi.md). Không mở confirmation/test hay N lớn; không tự động sửa order formula, LR hoặc budget. Không promote fixed-order control thành phương pháp đã xác nhận.

## Các vòng removal-credit trước (giữ nguyên kết quả âm)

| Nhánh học từ đầu | Cost/máy H60, fixed final |
|---|---:|
| Removal RB candidate | 262.075521 |
| Factor + state V, đối chứng chính cùng thông tin/capacity | 261.653646 |
| Blind removal | 261.500000 |
| PPO gốc | 261.898438 |
| PPO normalized / capacity PPO | 261.023438 |
| MAPPO local | 271.023438 |
| MAPPO full information | 263.263021 |

33 mô hình ×491.520 quyết định, tổng16.220.160 quyết định và253.440 Adam
update. Ba training seeds mới,32 DEV seeds, fixed final491.520; không đổi sang
checkpoint đẹp hơn. Candidate kém đối chứng chính0,161% và blind0,220%; chỉ
17/19 tiêu chí điểm đạt, điều kiện thắng mỗi training seed cũng không đạt.
Chín giới hạn sức khỏe đạt, không assignment sai; p95 actor+broker0,365ms ở N4.
Các số DEV chưa có CI nên không tuyên bố thắng hay tương đương PPO về mặt
thống kê, dù chênh lệch điểm cost nhỏ. Audit độc lập PASS16,30s.

[Phương pháp, kiểm chứng và đầy đủ19tiêu chí](technician_broker_removal_ppo_v1_findings_vi.md).

## Đã kiểm tra nguyên nhân, thay vì thêm encoder/LR/attention

Critic context có học, baseline thay đổi; return-residual variance cuối kỳ giảm
khoảng2,7%. Nhưng gradient actor có thật sự bớt nhiễu hay không là câu hỏi khác.
Chẩn đoán1.920 episode mới với policy đóng băng đo đầy đủ11.141 tọa độ actor:
mean variance gain trên tất cả năm checkpoint là1,59%/6,79%/3,47%; chỉ một seed
đạt mức5% đã khóa. Một số checkpoint giảm scalar residual mà tăng gradient
variance. Bộ kiểm chứng riêng replay5.760 gradient vectors và toàn bộ115.200
chuyển trạng thái, PASS30,06s.

[Chẩn đoán phương sai gradient và ranh giới prior art](technician_broker_removal_gradient_diagnostic_v1_findings_vi.md).

Đã tính thêm giới hạn giải tích cho scalar mixing trên chính dữ liệu chẩn đoán.
Mức tối ưu trên mẫu còn tiềm năng, nhưng đó là fit trên dữ liệu đánh giá. Vì vậy
đã khóa một phép kiểm tra mới:128fit rồi256heldout độc lập cho mỗi checkpoint;
hash hệ số trước khi thu heldout, không clip/regularize/quét hệ số hay refit.

FULL kiểm tra này gồm5.760 episode mới,276.480 quyết định,345.600 chuyển trạng
thái và0optimizer update. Heldout mean gains0,96%/20,87%/−0,38%; chỉ4/9điều kiện
đạt. Nhánh scalar calibration bị loại. Kiểm chứng độc lập17.280 gradient vectors
(192.516.480tọa độ), physics, fit-only coefficient và heldout verdict PASS88,88s.

[Kết quả fit/heldout và toàn bộ15checkpoint](technician_broker_removal_coefficient_calibration_v1_findings_vi.md).

## Quyết định đối với các nhánh removal-credit đã loại

Không chạy thêm biến thể hệ số, clip, regularization, seed hay budget để cứu các
nhánh vừa bị loại. Không dùng seed thứ hai tốt làm kết quả đại diện. Tất cả
protocol, mã đã commit/push, artifact mới và kết quả âm được giữ nguyên.

Kết quả hiện hỗ trợ tính đúng của broker-marginal/removal identities và phép đo
variance; chưa hỗ trợ câu chuyện “thêm removal critic thì quyết định tốt hơn”.
Lợi ích Rao-Blackwell so với proposal-score estimator không thay cho lợi ích
critic mới so với đối chứng đã dùng physical factors. Hệ số control-variate hay
score-weighted baseline cũng đã có prior art, không thể tự nhận là đóng góp mới.

Hướng nghiên cứu tiếp phải tác động được lên phân công và có giả thuyết riêng
được đo trước khi chạy học mới. Cần đối chiếu các thử nghiệm ordered-prefix,
remaining-capacity và broker trước đây để tránh lặp lại một cơ chế đã bị bác bỏ.
Các chẩn đoán này không tự mở nhánh học lại removal/scalar. FULL typed-Gibbs ở trên là giả thuyết policy riêng với protocol mới và kết quả DEV riêng. Vẫn giữ N4/K2, random initialization, cùng thông tin/ngân sách và giới hạn sức khỏe trước mở rộng. Goal tiếp tục active.
