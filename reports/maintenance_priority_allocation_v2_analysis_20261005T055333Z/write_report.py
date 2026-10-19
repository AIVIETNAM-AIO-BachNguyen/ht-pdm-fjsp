import csv,json,hashlib,statistics
from pathlib import Path
report=Path("/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_priority_allocation_v2_analysis_20261005T055333Z");run=Path("/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_priority_allocation_v2_full_cpu_20261005T041429Z")
s=json.loads((report/'statistics.json').read_text());audit=json.loads((report/'audit.json').read_text())
with (run/'exact_small_metrics.csv').open() as f: exact=list(csv.DictReader(f))
with (report/'family_metrics.csv').open() as f: metrics=list(csv.DictReader(f))
def num(x):return f'{x:.3f}'
def percent(x):return f'{100*x:.2f}%'
def ci(x):return '['+', '.join(f'{v:.3f}' for v in x)+']'
names={'learned_both':'Học cả hai','fixed_priority':'Ưu tiên cố định, học STOP/phân bổ','fixed_allocation':'Học ưu tiên/STOP, phân bổ cố định','risk_skill_rule':'Heuristic thuần'}
primary_table='\n'.join(f"| {names[r['control']]} | {percent(r['relative_reduction'])} | {ci(r['ci'])} | {r['wins']}/10 | FAIL |" for r in s['primary'])
costtable='\n'.join(f"| {names[mode]} | {statistics.fmean(float(r['objective']) for r in metrics if r['algorithm']==mode and r['family'] in ['n3_nominal','n3_pressure','n4_nominal','n4_pressure']):.4f} |" for mode in names)
familytable=[]
for family in ['n3_nominal','n3_pressure','n4_nominal','n4_pressure','n5_pressure','n5_k3_sparse','small']:
    values=[next(float(r['objective']) for r in metrics if r['algorithm']==mode and r['family']==family) for mode in names]
    relative=next(r['relative_reduction'] for r in s['secondary_descriptive'] if r['control']=='risk_skill_rule' and r['families']==[family])
    familytable.append('| '+family+' | '+' | '.join(f'{x:.3f}' for x in values)+' | '+percent(relative)+' |')
exacttable=[]
for mode in names:
    rr=[r for r in exact if r['algorithm']==mode]
    values=[statistics.fmean(float(r[f]) for r in rr) for f in ['exact_policy_cost','optimality_gap','exact_selection_regret','exact_allocation_regret']]
    exacttable.append('| '+names[mode]+' | '+' | '.join(f'{x:.4f}' for x in values)+' |')
text=f"""## Material Passport

- Type: analysis of completed local artifacts, protocol maintenance_priority_allocation_v2.
- Verification Status: **ANALYZED**; full training was not reproduced.
- Raw run: {run.name}.
- Commit: cb1a655063843befeff0da0667eb6d190f2f5f07, clean; frozen source/plan/registry match the verified Mac smoke bundle.
- Raw artifacts preserved. This report and its diagnostic outputs are separate.
- Locked protocol: pinned_repository/docs/maintenance_priority_allocation_v2_plan.md.
- Skill: ARS experiment-agent validate, handled inline; statistical fallacy scan 11/11 below.
- No new training, simulator change, threshold change, Git push, remote execution or transfer performed.

## Kết luận

**Policy mới học được; có bằng chứng lợi ích từ học ưu tiên, nhưng chưa đạt mục tiêu giảm ít nhất 5% so với cả hai đối chứng và chưa chứng minh cần MARL.** Kết quả cost tốt hơn đi cùng thời gian máy hỏng chờ lâu hơn. Ngoài ra, head phân bổ thiếu context riêng của technician về những máy khác, và các lựa chọn gần bằng điểm nhạy với runtime/tie-break. Không nên kết luận quyết định phân bổ không quan trọng từ lần chạy này.

## Độ đầy đủ và tính hợp lệ

Run COMPLETED, 30 model, 3.6m physical training steps, 300000 training episodes, 7500 rollout logs, 120000 optimizer steps, 10500 learned test +600 development episodes, 350 test +20 development rule references. Đủ 31 exact/headroom rows và 30 exact contrasts. Parameter counts bằng nhau, seed panels đúng protocol/disjoint, nguồn đóng băng đúng Git và artifact snapshot, 26 engineering audits PASS, không mở panel 201–300/601–700.

Manifest ghi **device=cpu**, dù máy thuê có GPU; torch là 2.14.0+cu126, không phải bằng chứng đã dùng CUDA. Runtime khoảng **77 phút 53 giây**, từ 04:19:04 đến 05:36:57 UTC.

Audit riêng kiểm tra toàn bộ training cost/budget và population pairing; replay **10850 test episodes /119350 physical ticks** gồm rule, đối chiếu 42100 machine records (learned records có cả development), kiểm tra từng chi phí occupancy/start và mọi state/action/cost trong trace. Sai lệch số lớn nhất {audit['max_numeric_error']:.3g}. DP N2 có 1475 states/5363 action values, cùng optimum 30.5; toàn bộ Bellman values khớp lại. CI và mean tự tính từ raw CSV, không chỉ đọc summary.

**Giới hạn tái lập:** replay động lực từ action đã log khớp; việc chạy lại greedy checkpoint trên Mac không hoàn toàn giống Linux ở tie. Trên 115500 state kiểm tra, 646 construction sequences khác (khoảng 0.56%). Mọi lựa chọn đã log đều nằm cách argmax local không quá {audit['maximum_recorded_choice_logit_margin']:.3g} logit; đây là chênh lệch float32 nhỏ, nhưng downstream allocation có thể khác. 28/620 development rollouts có khác ít nhất một metric. Exact của **7/10 fixed_allocation model** không tái lập cùng giá trị trên local batched evaluator; chênh lệch cost cá thể lớn nhất 1.3333. Các exact cost của learned_both, fixed_priority và rule khớp. Không thay raw kết quả bằng số local; xem policy_tie_sensitivity.json và local_exact_differences.json. First strict-replay failure được giữ trong analysis_attempt1.log; tiêu chuẩn kiểm tra checkpoint local được đổi sang tolerance và báo sai khác, KHÔNG đổi protocol/gate khoa học.

## Primary: hai contrast khóa trước

Cost trung bình lấy trọng số bằng nhau cho N3/N4 × nominal/pressure, mỗi model đánh giá 50 common episode seeds/family. Đơn vị CI là **10 training seeds**, không phải 10500 episode độc lập. CI 97.5% two-sided paired t, Bonferroni cho hai contrast, df9.

| Policy | Primary mean cost, thấp hơn tốt hơn |
|---|---:|
{costtable}

| Đối chứng so với học cả hai | Giảm cost | CI 97.5% của delta cost | Seed thắng | Gate tổng |
|---|---:|---:|---:|---|
{primary_table}

- **Ưu tiên cố định:** delta −4.484; CI [−5.453, −3.515], p paired-t {s['primary'][0]['paired_t_p']:.6g}, p Bonferroni {s['primary'][0]['bonferroni_p']:.6g}. Thắng 10/10; nominal mean −0.146%. Có lợi ích rõ theo CI đã khóa, nhưng point effect 3.04% <5%; chỉ fail practical-effect condition.
- **Phân bổ cố định:** delta −1.5055; CI [−3.272, +0.261], p unadjusted {s['primary'][1]['paired_t_p']:.6g}, p Bonferroni {s['primary'][1]['bonferroni_p']:.6g}. Thắng 9/10; nominal mean −2.033%. Fail effect≥5% và CI upper<0. Không gọi lợi ích thêm của allocation là đã được xác lập, cũng không gọi là bằng zero.
- Overall H1 **FAIL**, đúng ngưỡng khóa trước. Nominal guard PASS cho cả hai contrast; chỉ là cost trung bình, không đảm bảo từng seed/máy hay failure rate nominal.
- Conditional-on-panel paired t assumption: n=10, phân phối paired differences chưa được chứng minh normal. Exact sign-flip exploratory p={s['primary'][0]['sign_flip_p_exploratory']} và {s['primary'][1]['sign_flip_p_exploratory']} được lưu như sensitivity, giả định symmetry; không thay thế locked CI hay chuyển FAIL thành PASS. Paired dz lớn không biến mức giảm cost nhỏ thành lợi ích thực tiễn ≥5%.

## Heuristic và phân nhóm: descriptive

Học cả hai so với rule: giảm **3.071%**, thắng 10/10 training seeds; CI95 delta [−5.066, −3.997], nominal mean −0.433%. Fixed_priority gần ngang rule (+/− uncertainty); fixed_allocation giảm khoảng2.05% so với rule. Đây là secondary descriptive contrasts; rule chỉ có một fixed episode panel, không có mười lần train độc lập.

| Family | Học cả hai | Ưu tiên cố định | Phân bổ cố định | Rule | Học cả hai giảm vs rule |
|---|---:|---:|---:|---:|---:|
{chr(10).join(familytable)}

Lợi ích tập trung ở thiếu nguồn lực: N4 pressure −5.92% vs rule, −6.21% vs fixed_priority; N5K2 pressure −8.11% vs rule. Các con số phân nhóm này **không thay aggregate H1 thành PASS**. N5/K3 là transfer probe ngoài train size/resource counts, không phải claim tổng quát về mọi size. N5K2 và N5K3, học cả hai còn hơi kém fixed_allocation ở point mean; allocation không giúp đồng đều mọi family.

## Cost tốt hơn nhưng repair waiting xấu hơn

Ở N4 pressure:

| Metric /episode | Học cả hai | Ưu tiên cố định | Phân bổ cố định | Rule |
|---|---:|---:|---:|---:|
| Total cost | 231.414 | 246.730 | 232.588 | 245.980 |
| Failures | 2.842 | 4.992 | 2.820 | 4.940 |
| Failed waiting ticks | 8.574 | 3.172 | 8.920 | 3.520 |
| Service occupied ticks | 21.412 | 23.452 | 21.338 | 23.140 |
| PM starts | 5.812 | 3.798 | 5.796 | 3.600 |
| CM starts | 1.528 | 4.154 | 1.472 | 4.160 |
| Failed pending demands at terminal | 1.314 | 0.838 | 1.348 | 0.780 |
| P95 max waiting /machine | 10 | 2 | 10 | 2 |

So với rule, giảm failure-event expenditure **31.470**, giảm maintenance-start expenditure **3.052**, nhưng tăng unavailability cost **19.956**, tổng giảm **14.566**. Số liệu khớp chính xác decomposition; nó chưa phải ablation chứng minh một cơ chế là nguyên nhân duy nhất.

Early failed-defer fraction N4 pressure: học cả hai **67.77%** vs rule **15.97%**, trong các quyết định máy hỏng có edge rảnh hợp lệ và còn nhiều hơn max service duration trước terminal. Denominator là decision exposure do policy tạo ra; không coi đây là causal rate comparison độc lập. P95 waiting tính trên tất cả machine-episode records, không chỉ các demands.

**12.4% tất cả machine-episode records** của học cả hai trong N4 pressure kết thúc với máy hỏng và chưa từng có service start trong episode; rule 0%. Đây là tỷ lệ trên toàn bộ máy, KHÔNG phải 12.4% requests. N5K2 tỷ lệ này 22.84% vs rule0%, P95 max wait10 vs3. Có dấu hiệu bỏ mặc một phần máy, không chỉ dời repair tới tick cuối.

Witness post-hoc 116000/118004/N4 pressure: máy3 failed từ state tick2, chờ tới hết tick12, **10 unavailable waiting ticks**, zero service. Nó có 7 decision times với technician tương thích đang rảnh, 3 lần còn dư >max duration. Policy vẫn PM các máy khỏe khác. Xem waiting_witness.json; minh họa đã mở, không dùng làm independent inferential evidence.

Reward không bị tính sai downtime: máy chờ bị charge đúng. Với failure event15, unavailability6/tick, horizon12 và không có terminal liability, policy có thể đánh đổi sửa một máy đang hỏng để ngăn failure ở máy khác. Đây có thể là hành vi kinh tế hợp lý theo objective đã định nghĩa; nó vẫn mâu thuẫn với tiêu chí ứng dụng “tránh bỏ mặc máy” nếu tiêu chí đó chưa được ràng buộc rõ. Không tự thêm penalty sau khi xem kết quả, không gọi chiến lược này là tốt về SLA/fairness. Maintenance-only unavailable cost chưa phải production makespan/throughput thực.

## Exact N2: learning đạt gần tối ưu, không suy rộng headroom

| Policy | Exact expected cost gốc Linux | Gap tới30.5 | Selection regret | Allocation regret |
|---|---:|---:|---:|---:|
{chr(10).join(exacttable)}

**9/10 learned_both seed cost30.6667, gap0.1667**; seed116001 cost32.6667, có allocation regret1.8333. Mean30.8667 chỉ khoảng1.20% cao hơn optimum. Giá trị này được local evaluator tái lập. N2 cho thấy PPO đã học được, không còn STOP collapse như smoke rất ngắn.

Rule32.4444 có maximum relative headroom **5.993%**; learned_both mean giảm khoảng4.86% vs rule, tận dụng phần lớn dư địa nhỏ trong cell này. 7/10 fixed_priority model có remaining headroom<5%; điều này không chứng minh aggregate N3/N4 không còn dư địa. Selection regret chỉ là chọn MACHINE SET/STOP, không phải regret riêng của thứ tự latent decoder. Regret decomposition dùng optimal future continuation và occupancy của từng policy, không phải một causal experiment tách hai head.

Sample mean small30.260 của learned_both trong50 held-out episodes thấp hơn exact optimum30.5 do finite Monte Carlo panel; nó không chứng minh vượt optimal policy. Exact và sampled mean khác estimand/evaluation noise. Fixed_allocation exact gốc được kiểm tra phép cộng regret/gap nhưng có local tie sensitivity đã báo ở trên; không tuyên bố toàn bộ31 exact deployment values được reproduce cùng một policy trên mọi platform.

## Thiếu context riêng của technician trong allocation head

Code frozen encode() tính edge embedding từ local machine, raw technician state và own edge trước peer aggregation. technician_distribution() cho head nhận selected edge e[m,k], global g chung, prefix chung; updated per-technician neighbor embedding t[k] không được đưa riêng vào head. Do đó khi hai technician đều idle và cùng service/restoration cho máy hiện tại, head không phân biệt được họ qua năng lực cho các máy còn lại, dù toàn observation có thông tin đó.

Witness thật: N3 nominal/118009/tick1, chọn máy0. Service matrix ((2,2),(1,2),(1,2)); restoration ((.75,.75),(1,.75),(1,.5)); hai thợ idle. Với máy0, hai edge giống nhau; với máy1/2, worker0 tốt hơn. Checkpoint cho allocation probabilities **[.5,.5]**. Xem allocation_context_witness.json.

Đây là verified representational limitation của head chọn thợ. Whole autoregressive policy vẫn có thể bù bằng chọn máy khác trước/reservation, nên không kết luận toàn policy không thể đạt tối ưu. Cũng chưa có ablation chứng minh thiếu context giải thích toàn bộ khoảng cách1.04% hoặc mọi tie. Nó hạn chế khả năng dùng experiment hiện tại để phủ định giá trị học allocation/opportunity cost.

## Hướng phát triển được kết quả hỗ trợ

Chưa cần chuyển mặc định sang QMIX/MARL. Phương pháp hiện tại là centralized coordinated autoregressive PPO; nó đã tạo improvement dưới objective mới. Tách ba câu hỏi:

1. Problem ưu tiên PM/CM khi capacity thiếu có impact: thấy cost improvement dưới stress và tradeoff waiting rõ; không đồng nhất với real-factory impact.
2. Học allocation thêm có đáng làm: primary evidence hiện chưa đủ; representation head còn thiếu per-worker competing-demand context và tie resolution cần ổn định trước khi phép so sánh có thể kiểm chứng đầy đủ ý tưởng này.
3. Chất lượng phục vụ cần gì: chốt yêu cầu repaired-machine waiting/service coverage và horizon/terminal contract; hiện có thể giảm cost bằng để một số máy hỏng lâu. Thử horizon dài/terminal consequence hoặc explicit service constraint là hypothesis của follow-up, không sửa retrospective gate.

Chỉ sau khi baseline/coordinated policy có contract rõ và đạt mục tiêu ứng dụng mới hợp lý so sánh execution tập trung với MARL. Experiment này không chứng minh MARL cần thiết, cũng không chứng minh mixer QMIX cũ gây failure.

## Statistical fallacy scan: 11/11 checked

| Loại | Kiểm tra và giới hạn |
|---|---|
| 1 Simpson | Báo cả aggregate và family; N4 nominal vs fixed_priority và N5 vs fixed_allocation có thể đi ngược aggregate. Không suy “mọi cell đều tốt hơn”. |
| 2 Ecological | CI unit training seed, average episode/family; không đếm10500episode thành10500 policy replications. |
| 3 Berkson | Không lọc episode/policy theo success/cost; toàn bộ30 model và preset panels được giữ. Distribution synthetic/selected envelope, không suy rộngfactory. |
| 4 Collider | Không conditioning primary cost vào failure/dispatch success; early-defer là policy-dependent descriptive exposure, không causal controlled rate. |
| 5 Base rate | Waiting/terminal proportions ghi rõ all-machine denominator; raw failure means/costs cùng báo, không diễn giải rate thiếubase. |
| 6 Regression mean | Không chọn seed xấu để claim improvement; witness sau khi mở chỉ minh họa, không confirmatory. |
| 7 Survivorship | Đủ tất cả models/seeds/budgets, không cắt episode cap hay bỏ run thất bại giữa chừng trong run này. |
| 8 Look elsewhere | Giữ2 primary contrasts cóBonferroni; rule/family/transfer/sign-flip làdescriptive/exploratory, không dùngpressure5.92% để đổiH1. |
| 9 Forking paths | Plan/source/seed/budget khóa trước, no retuning; đổi audit tolerance sau strict replay failure được công khai, rawCI/gate không đổi. |
| 10 Correlation/causation | Controlled algorithm ablations hỗ trợ total policy contrast trongsimulator; decomposition/witness/context không chứng minh solecause hay MARLnecessity. |
| 11 Reverse causality | Actions trước shock grid, khôngfutureinputs; observed repair-priority behavior response to failure không được diễn giải thànhcausal riskmodel. |

## Artifacts và reproduce

- audit.json: counts, hashes, seed/engineering/inference scope, tie flags.
- statistics.json: independently recomputed primary/reference/family contrasts, paired t and sign-flip sensitivity.
- family_metrics.csv, learning_curves.csv: operational/learning diagnostics.
- exact_recomputed.csv: local-policy exact values, separate from raw original Linux exact CSV.
- policy_tie_sensitivity.json, local_exact_differences.json: all reproduction differences.
- waiting_witness.json, allocation_context_witness.json: post-hoc diagnostic states/traces.
- analyze.py + pinned_repository: audit/analysis implementation from verified commit. Not an independent simulator.
- plot.py, cost_and_waiting.png/svg: scientific plot of locked CI and descriptive cost components.

![Locked contrast and waiting tradeoff](<{report}/cost_and_waiting.png>)
"""
(report/'analysis_vi.md').write_text(text)
hashes=[]
for file in sorted(run.rglob('*')):
    if not file.is_file():continue
    h=hashlib.sha256()
    with file.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    hashes.append({'file':str(file.relative_to(run)),'sha256':h.hexdigest()})
with (report/'raw_file_sha256.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=['file','sha256']);w.writeheader();w.writerows(hashes)
print(report/'analysis_vi.md')

