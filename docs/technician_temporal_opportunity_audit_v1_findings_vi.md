# Cơ hội phân công theo thời gian: audit H4 chưa đủ cơ sở cho learner mới

**Không mở conditional-Q learning trial từ audit này.** Cả5điều kiện cần không
đồng thời đạt: candidate greedy hiện tại đã gần nghiệm H4 tốt nhất ở các states
được chọn. Goal thuật toán học từ đầu vẫn chưa đạt; CF/test/larger-N chưa mở.

Protocol đăng ký b291846; code a9b7c5a,12testsPASS1,44s rồi commit/push trước chạy.
FULL diagnostic COMPLETED/auditPASS2,923890s:
`artifacts/technician_temporal_opportunity_audit_v1_20261016T163425641762Z`.
96root records từ candidate/PPO greedy trajectories của frozen mode diagnostic,
3replicas×4physical seeds300100000..3×ticks0/12/24/36×2origins. Có12state duy nhất;
giữ nguyên record frequency, không coi96records là96quan sát độc lập.

Không training, Q teacher labels, forecast RNG hoặc state test mới. Tại mỗi root,
Bellman reference tối ưu expected cost4physical ticks, zero terminal value, giữ
urgent support và physics gốc, tích phân đúng Bernoulli(.45). Nó liệt kê matching
N4, nên không phải scalable actor hoặc policy được đề xuất.

“Isolated” chấm riêng từng technician khi đồng nghiệp WAIT rồi áp quality broker.
“Conditional” giữ lựa chọn isolated của T0, tối ưu lựa chọn T1 ở prefix thực tế.
“Joint” tối ưu toàn matching; mọi root đều dùng cùng optimal V3 continuation.
“Source” giữ nguyên root matching đã ghi. Q4 không phải fullH60episode cost.

| Origin | Isolated meanQ4 | Conditional | Joint | Source |
|---|---:|---:|---:|---:|
|Candidate|62,345833|61,877083|61,877083|61,895833|
|PPO|60,740625|60,290625|60,290625|60,928646|

Candidate population48records: joint và conditional đều cải thiện isolated
**0,751855%**, đạt>=0,5%. Conditional gain ở3replicas lần lượt
0,728376% /0,763435% /0,763435%, đều>0. Tuy nhiên source gap tới joint chỉ
**0,030293%**, dưới>=0,5%; conditional tốt hơn isolated ở**5records**, cần>=6.
Screen3/5PASS, `conditional_q_mechanism_further_design_motivated=false`.
PPO-origin source gap1,04716% chỉ mô tả, không dùng làm population khác để cứu gate.
Không chọn lại landmarks/order/horizon hoặc đổi mask sau khi xem kết quả.

Source candidate khác joint ở2records, cùng physical seed300100000,tick12,
replicas290000001/2; mỗi root Q gap0,45. Năm conditional improvements cũng tập
trung ở seed300100000 (ba tại tick0,hai tại12). Vì vậy pooled0,75% không phải
bằng chứng cơ hội phân công phổ biến hoặc improvement trên fulltrajectory.

Independent verifier1cc7cb8,15testsPASS trước chạy, PASS31,062857s:
`artifacts/technician_temporal_opportunity_verification_v1_20261016T163911730203Z`.
Nó truy ngược cả96contexts về physical journal gốc, dùng repository scalar
transition với **đủ16tổ hợp sự kiện** mỗi state/action và Bellman implementation
khác, kiểm lại cả384root Q4, minima, isolated/conditional argmins, broker,
means/replicas và screen.14.635Bellman expansions,80.990state-action transition
groups; max Q error**4,2633e−14**. Audit production dùng sparse relevant-event
integration/pure dictionary dynamics; hai cách khớp. Source/output hashes PASS.

Kết quả này không loại mọi mô hình temporal coordination. Nó loại động cơ đã khóa
cho conditional-Q ở population/horizon này; policy hiện tại đã tránh được phần
lớn lỗi của isolated scorer. H4zero-terminal không chứng minh tối ưu H60 hoặc
sức khỏe hệ thống: không có health claim hoặc goal completion từ audit này.
Không dùng oracle values huấn luyện, distill hay warm-start.

Một quan sát exploratory riêng trên learning curves FULL cũ: tại122.880steps,
candidate có cost270,211 so với proposal-control277,146 (+2,50%gain); cả3replicas
gain dương (3,697% /3,582% /0,050%). Tại245.760steps control lại thấp hơn candidate
0,173%. Điều này gợi ý một câu hỏi KHÁC về tốc độ học ban đầu, không sửa negative
verdict ở983.040steps và chưa đủ chứng minh sample efficiency. Một phép kiểm tốc
độ học cần integral/cadence/budget khóa trước, fresh random-init training/seeds,
đối chứng cùng ngân sách và final quality/health safeguards; không chọn một
checkpoint thuận lợi trên seed cũ rồi tuyên bố đóng góp.

Protocol: [audit H4](technician_temporal_opportunity_audit_v1_plan.md).

Lấy artifacts thủ công trên Mac khi cần; agent không chạy transfer:

```sh
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
"/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```
