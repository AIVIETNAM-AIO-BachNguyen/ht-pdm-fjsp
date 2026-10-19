import json,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
OUT=ROOT/'reports/maintenance_assignment_value_v1_analysis_20261011T155312Z'
RUN=ROOT/'lab_results/maintenance_assignment_value_v1_full_cpu_20261011T153135Z'
s=json.loads((OUT/'summary_verified.json').read_text());d=json.loads((OUT/'diagnostic_summary_verified.json').read_text());x=json.loads((OUT/'supplementary_diagnostics.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,ax=plt.subplots(figsize=(8.6,4.0),layout='constrained')
keys=list(s['primary_contrasts']);labels=[]
for i,key in enumerate(keys):
 r=s['primary_contrasts'][key];v=r['relative_gain']*100;lo,hi=np.array(r['confidence_interval'])*100
 labels.append(key.split(':')[0]+' / vs '+r['baseline'].replace('_',' '))
 color='#0b7292' if r['baseline']=='frozen_ppo' else '#81519d'
 ax.errorbar(v,3-i,xerr=[[v-lo],[hi-v]],fmt='o',color=color,capsize=4,lw=2)
 ax.annotate(f'{v:+.2f}%  [{lo:+.2f}, {hi:+.2f}]',(hi+.10,3-i),va='center',fontsize=9)
ax.set_yticks(range(4),labels[::-1]);ax.axvline(0,color='black',lw=1);ax.axvline(1,color='#bd7200',ls='--',label='1% material margin')
ax.set_xlim(-1.7,8.9);ax.set_xlabel('Episode cost reduction (%) — positive favors true-label value head')
ax.set_title('Primary comparisons: 98.75% crossed replica / profile intervals')
ax.legend(loc='lower right',frameon=False);ax.grid(axis='x',alpha=.15)
fig.savefig(OUT/'primary_cost_contrasts.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12.2,3.8),layout='constrained')
conditions=['dense','skill_mask'];pos=np.arange(2)
for name,label,color,offset in [('true_value','True labels','#0b7292',-.19),('shuffled_value','Shuffled labels','#b7bbc4',.19)]:
 axes[0].bar(pos+offset,[x[c]['true_mse' if name=='true_value' else 'shuffle_mse'] for c in conditions],width=.36,label=label,color=color)
axes[0].set_ylim(0,8.2);axes[0].set_xticks(pos,conditions);axes[0].set_ylabel('MSE of normalized paired cost difference');axes[0].set_title('Held-out ranking: modest learning');axes[0].legend(frameon=False,fontsize=9)
for i,cond in enumerate(conditions):
 r=d['exploratory_contrasts']['assignment:'+cond+':true_value'];v=r['estimate'];lo,hi=r['confidence_interval']
 axes[1].errorbar(v,1-i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color='#0b7292')
 axes[1].annotate(f'n={r["exposures"]}',(hi+.8,1-i),va='center',fontsize=9)
axes[1].set_yticks([0,1],conditions[::-1]);axes[1].axvline(0,color='black',lw=1);axes[1].set_xlim(-21,10)
axes[1].set_xlabel('Mean one-root gain vs PPO (cost units)');axes[1].set_title('Original PM set: gain still uncertain\nExploratory 95% intervals; PPO continuation')
coverage=pd.read_csv(OUT/'label_coverage_by_clock.csv');train=coverage[coverage.split=='train'].groupby('tick').labels.sum()
axes[2].bar([str(t) for t in train.index],train.values,color='#0b7292');axes[2].set_title('Training labels concentrated at t=0');axes[2].set_xlabel('Original episode tick');axes[2].set_ylabel('Paired training labels')
for i,n in enumerate(train.values):axes[2].text(i,n+25,str(n),ha='center',fontsize=9)
axes[2].set_ylim(0,2800)
fig.savefig(OUT/'assignment_learning_diagnostics.png');plt.close(fig)
# Figures and tables support an accessible Vietnamese report; all secondary findings labeled.
primary=[]
for key,r in s['primary_contrasts'].items():
 primary.append(f"| {key.split(':')[0]} | {r['baseline']} | {r['relative_gain']*100:+.3f}% | [{r['confidence_interval'][0]*100:+.3f}; {r['confidence_interval'][1]*100:+.3f}]% | {r['status']} |")
ranking=[]
for cond in conditions:
 for name in ('true_value','shuffled_value','simulation_reference'):
  r=d['exploratory_contrasts'][f'assignment:{cond}:{name}']
  ranking.append(f"| {cond} | {name} | {r['exposures']} | {r['estimate']:+.3f} | [{r['confidence_interval'][0]:+.3f}; {r['confidence_interval'][1]:+.3f}] |")
timing=[]
for cond in conditions:
 for name in ('frozen_ppo','true_value'):
  r=d['exploratory_contrasts'][f'timing:{cond}:{name}:correction_gain']
  timing.append(f"| {cond} | {name} | {r['exposures']} | {r['estimate']:+.3f} | [{r['confidence_interval'][0]:+.3f}; {r['confidence_interval'][1]:+.3f}] |")
text=f'''## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent.
- Origin Mode: validate, inline; user-authorized local artifact analysis.
- Origin Date: 2026-10-11.
- Verification Status: ANALYZED; source/hashes/statistical aggregates VERIFIED;
  logged decisions and a bounded sample of saved future tapes replayed exactly.
  Full training and all future policy paths have not been rerun.
- Version Label: maintenance_assignment_value_v1_analysis.
- Overall Confidence: CAUTION — small policy-improvement pilot, three source replicas.

## Kết quả nghiên cứu hiện tại

**Head đánh giá phân công đã học được một phần tín hiệu từ rollout RL,
nhưng chưa chứng minh được rằng nó chọn technician tốt hơn PPO gốc khi chạy
trọn episode. Experiment này vẫn giữ PPO quyết định PM máy nào/thời điểm nào;
chưa huấn luyện đồng thời timing và phân công.**

Điểm mới có bằng chứng: nhãn đúng làm dự đoán chênh lệch chi phí tốt hơn đối
chứng xáo trộn, và giảm chi phí so với đối chứng ở dense. Điểm chưa đạt:
lợi ích so với PPO gốc còn nhỏ/không chắc; skill-mask chưa đạt ngưỡng giảm1%.
Không suy ra “RL không học được”, “hai policy tương đương”, hay đã có thuật toán
mới vượt baseline.

## Run và kiểm tra dữ liệu

Run: `{RUN.name}`, commit `e821e7f9ccfd9a5cdd06e233dbc39ae39c98bb5c`, clean,
COMPLETED/PASS, CPU một thread;892.14giây =14phút52giây. Dataset/run mới,
độc lập cấu hình vật lý với các panel trước;6 head /3 PPO nguồn.

-3.036 artifact hash/size khớp;82.35MB.189 source snapshot files khớp commit và
 mã đang dùng để phân tích; nguồn PPO có957 artifact được kiểm tra.
-1.536 training states /192 dev states;2.644 paired training labels /348 dev labels;
 3.360 optimizer updates. Mỗi head có7.745 tham số thêm, PPO đóng băng có19.650.
-1.728 test episodes,24 profiles x4 shocks x2 conditions x3 replicas x3 controllers;
 48 bước quyết định +12 bước deadline tail. Cohort/shock được ghép giữa controllers.
-288 held-out assignment states,972 paired classes;120 common timing states,
 240 controller timing records.27.46triệu forecast transitions /501.280 future paths
 là chi phí tạo nhãn/diagnostic, không phải PPO training steps.
-Đã tính lại toàn bộ primary/diagnostic summaries và bootstrap từ CSV;2.784 NPZ
 khớp shocks/RNG/chi phí/components/tail/counts. Tính lại2.992 train/dev labels và
 972 ranking deltas từ raw costs. Dataset/control permutations và mọi checkpoint
 candidate được kiểm tra với dev labels, không chọn lại bằng test.
-27.648 source transitions và4.320 logged test transitions khớp scalar replay;
 inference của mọi logged test decision khớp action, không có lỗi thay CM/máy PM.
 Dự đoán của checkpoint được xác nhận trên288 ranking states. Sample72 saved
 forecast files /246 future paths /11.256 transitions khớp costs/components/boundaries.
 Đây là kiểm tra các tape đã lưu, không phải full experiment mới hay retraining.

`full_test_outcomes_opened=true` là đúng ở thời điểm run đã mở panel sau khi
khóa checkpoint selection, không phải cờ train/test leakage. Nguồn cũ đã từng được
phân tích; run này dùng nguồn đó làm baseline cố định và dùng panel mới.

## Chi phí khi policy thực sự chạy tiếp: primary

Tất cả số dưới là giảm chi phí H48+tail12 của true-label head; dương = tốt hơn.
Bốn so sánh chính dùng CI98.75% crossed paired source-replica/profile bootstrap,
20.000 draws, approximate Bonferroni cho4 contrasts. Ngưỡng thực dụng đã khóa1%.
CI dựa trên3 replicas có giới hạn hiệu chỉnh; shocks là panel cố định.

| Điều kiện | Baseline | Giảm chi phí | CI98.75% | Trạng thái theo ngưỡng1% |
|---|---|---:|---|---|
{chr(10).join(primary)}

Dense: mean cost PPO1173.528, true1167.903, shuffled1197.903. True giảm5.625cost
units so với PPO (+0.479%); hai replica tốt hơn, một replica kém hơn
(-0.562%,+1.145%,+0.818%). CI còn chứa0 nên chưa có lợi ích rõ so với PPO.
So với shuffled, cả3 replica đều tốt hơn; CI[+0.360%,+5.416%] không chứa0,
ủng hộ lợi ích của nhãn đúng ở dense. Cận dưới chưa vượt1%, nên status protocol
vẫn UNRESOLVED về lợi ích thực dụng, không có nghĩa “không có tín hiệu”.

Skill-mask: PPO1210.622, true1211.635, shuffled1221.941. True đắt hơn PPO1.014
cost units (+0.084%cost), cả3 replica có point estimate kém hơn. CI vẫn chứa0,
không chứng minh tác hại; cận trên lợi ích0.282% dưới margin1% trong pilot này.
True tốt hơn shuffled ở point estimate nhưng CI chứa0. Không gộp hai điều kiện
để che kết quả khác nhau.

Dense: giảm base cost6.792, tăng tail cost1.167, net giảm5.625. Thành phần
failure giảm4.740, maintenance giảm0.135, unavailability giảm0.333, waiting giảm0.417.
Skill-mask: base tăng0.378, tail tăng0.635, net tăng1.014. Không có tail mới hay
kết thúc ngắn hơn làm lợi thế giả.

![Primary cost contrasts](primary_cost_contrasts.png)

## Head có học được quan hệ trạng thái–phân công không?

Có tín hiệu học, nhưng chất lượng còn hạn chế. Trên tất cả assignment classes
held-out (n750 dense/n222 skill), MSE normalized của true vs shuffled là
5.745vs6.245 (giảm8.01%) và5.863vs6.553 (giảm10.54%). RMSE vẫn xấp xỉ2.40–2.42,
nhân reward_scale20 tương ứng khoảng48cost units. Dự đoán chọn chiều lợi hơn
đúng63.1%/58.1% so với50.1%/50.9% của shuffled; các tỷ lệ này chỉ mô tả và
không coi972 classes là972 independent physical profiles.

Phân tích bổ sung sau khi mở test (EXPLORATORY, không primary): paired MSE
improvement true vs shuffled, cluster theo replica/profile, dense0.500
CI95%[0.163,0.986], skill0.691[0.019,1.858]. Không chỉnh multiplicity cho các
kiểm tra bổ sung. So với predictor luôn đoán Delta=0, true giảm MSE7.57%/10.03%;
paired CI dense[0.150,0.929] dương, skill[-0.009,1.880] chứa0. Không thể diễn giải
đơn thuần rằng head đã đánh giá chính xác chênh lệch dài hạn.

Dấu hiệu dự đoán co về0: standard deviation của predicted Delta chỉ0.423/0.428,
của measured Delta2.492/2.558 (normalized units). Correlation0.306/0.386. Đây là
mô tả phân phối, chưa đủ xác định nguyên nhân kiến trúc hay training budget.
MC noise MSE chỉ0.171/0.151 trong64 evaluation futures; không giải thích toàn bộ
residual5.745/5.863, nhưng bootstrap không mô hình hóa lại mọi shock tương lai.

Subset có |measured Delta|>1.96*paired MCSE cho sign accuracy68.2%/68.7%.
Subset này được lọc theo outcome để chẩn đoán, không dùng làm headline/pass gate
và không tạo test-tuned policy mới.

## Khi sửa đúng phân công PPO đang dùng, head có tốt không?

Held-out diagnostic giữ nguyên các máy PM và CM; continuation sau root là PPO
đóng băng (Q^pi), không phải head mới. Có72/144 dense states và23/144 skill states
thực sự có lựa chọn technician; no-choice states vẫn được lưu.

| Điều kiện | Root selector | States có lựa chọn | Gain trung bình | CI95% exploratory |
|---|---|---:|---:|---|
{chr(10).join(ranking)}

True ở dense đổi25 lần:13 tốt hơn,12 kém hơn theo64-future measured means;
6/5 lần có per-state MC95% khác0 theo hướng tốt/xấu. Skill-mask đổi5 lần:1 tốt,
4 xấu;0 lần tốt và2 lần xấu có MC95% khác0. Những per-state intervals chưa được
chỉnh nhiều so sánh, không phải independent confirmations.

PPO vốn chọn phía có measured cost thấp hơn45/72 dense và19/23 skill states.
True chọn đúng46/72 và16/23. Vì vậy học tín hiệu tốt hơn đối chứng chưa đồng
nghĩa sửa baseline tốt hơn. Trường hợp skill profile20600014/source18100002:
head đổi technician và làm cost tăng71.828 với MCSE6.939, trong khi chỉ có
5 overrides ở condition đó. Một số lỗi lớn có thể triệt tiêu nhiều cải thiện nhỏ.
Không bỏ outlier này khỏi primary analysis và không chọn seed “tốt nhất”.

Simulation reference chọn trên32 selection futures, đo trên64 independent
futures, giữ cùng PM machines. Dense residual point gain4.081
CI[0.033,9.232] cho thấy vẫn có room sửa assignment trong panel này;
skill0.404CI[-2.587,4.172] chưa rõ. Đây là noisy reference, không phải exact oracle,
không so sánh trực tiếp nó với full episode gain vì khác sampling/continuation.

## Timing vẫn giữ riêng

Counterfactual chỉ giữ/hoãn các PM đã chọn, nguyên technician và CM; không thêm
máy PM. Sau root, chính controller đang được đánh giá chạy đến bước47, sau đó
common deadline tail12. Chọn root bằng32 futures rồi đo trên64 independent futures.
Chỉ first4 test profiles, mỗi condition có60 common states; không phải wholetest.

| Điều kiện | Policy tiếp tục chạy | States có timing choice | Timing correction gain | CI95% exploratory |
|---|---|---:|---:|---|
{chr(10).join(timing)}

Dense PPO còn timing correction gain dương trong exploratory CI; head mới
còn point gain7.458 nhưng CI[-0.007,17.964] chứa0. Skill head4.163
CI[-0.382,8.859] chưa rõ. Không kết luận head đã tự sửa timing: timing chưa được
training mới. Không kết luận head cải thiện timing từ việc hai điểm trung bình
12.80 và7.46 khác nhau; đó cần paired contrast và thiết kế riêng.

Blanket “hoãn toàn bộ PM” khác selective delay: dense head gain-4.723
CI[-25.762,16.217], skill-5.281CI[-11.551,0.540]. Không có căn cứ đổi thành rule
“cứ chờ”. Trong closed-loop test, dense head WAIT khi có PM feasible12/6611
(0.182%), skill14/6561(0.213%). PM có thể dừng sớm trong partial matching, nên
WAIT chỉ là một phần hành vi timing; không lấy WAIT rate làm thước đo tối ưu.

## Vì sao cần sửa bước học phân công trước?

Nhãn training:2.496/2.644 =94.40% ởtick0; dev312/348 =89.66% ởtick0. Dù đã giữ
1536/192 source states và không lọc outcome, các trạng thái muộn thường không
có2 feasible PM assignments do technician bận/CM fixed. Điều này tạo tập nhãn
rất lệch về đầu episode; không có nghĩa các trạng thái muộn bị bỏ khỏi coverage.
Đây là giới hạn dữ liệu quan sát được, chưa chứng minh là nguyên nhân duy nhất.

Head học Delta Q dưới continuation PPO, còn full evaluation áp dụng head lặp
lại (policy mới). Có thể có distribution/continuation mismatch; experiment chưa
cô lập nó như một nguyên nhân. Quan trọng là ngay root diagnostic dưới PPO,
head vẫn chưa đem lại gain rõ, nên không thể chỉ đổ lỗi cho mismatch.

![Assignment learning diagnostics](assignment_learning_diagnostics.png)

## Bước tiếp theo đề xuất (chưa chạy)

Một follow-up nhỏ cho **độ tin cậy của learned assignment advantage so với
phân công PPO**, tiếp tục giữ timing riêng:

1. Thu nhãn trên tập trạng thái mới có so sánh technician ở nhiều giai đoạn;
   chủ động có states technician bận/sắp rảnh/CM tranh nguồn lực. Báo coverage
   cả visited panel tự nhiên và panel diagnostic có chủ đích, không giấu sampling.
2. Học và kiểm tra trực tiếp Delta của alternative so với original PPO trên cùng
   machine set/CM, kiểm tra calibration và MC uncertainty; đối chứng matched
   true/shuffled và luôn giữ PPO. Tập all-PM-subsets vẫn có thể làm phụ nhưng
   không thay metric của quyết định deployment thực tế.
3. So decoder đổi khi predicted advantage vừa dương vs decoder chỉ đổi khi
   lợi thế được mô hình đánh giá đủ chắc. Cách đo uncertainty/margin phải khóa
   trước, chọn bằng dev mới; không lấy threshold từ23/72 test states đã xem.
4. Dùng test configurations/shocks mới, H48+tail12; một primary test vs PPO và
   paired assignment-root check. Sau khi có gain ổn định mới kiểm tra tích hợp
   với timing. Đây là đề xuất nghiên cứu, không khẳng định gating sẽ thành công.

## Fallacy scan:11/11 đã kiểm tra

| Pattern | Kiểm tra / giới hạn |
|---|---|
| Simpson | Báo riêng dense/skill và3 replica; không gộp để đảo chiều skill. |
| Ecological | Classes/futures không được tính như profiles; cost episode và root gains khác đơn vị. |
| Berkson | Chỉ conditional assignment/timing choices khi cóalternative; báo mẫu số, allstates/no-choice. |
| Collider | Không kiểm soát thêm theo improvement hoặc selected checkpoint test; outcome-resolved subset chỉ exploratory. |
| Base rate | Báo exposure72/144 và23/144, sign accuracy/PPO baseline và số overrides; không giấu no-choice. |
| Regression to mean | Reference chọn32, đo64 independent futures; không chọn lại theo eval means. |
| Survivorship | Đủ grid1728 episodes, toàn bộ3 replicas/classes; không drop harmful profile20600014. |
| Look elsewhere |4 primary intervals98.75%; tất cảdiagnostic/MSE bổ sung/sign subsets exploratory. |
| Forking paths | Protocol khóa trước; dev chọncheckpoint; không retune threshold/test hoặc chọn seed sau phân tích. |
| Causation | Paired simulator interventions chỉ support trong generator/H48; không transfer/deployment/novelty claim. |
| Reverse causality | Weights được fit trước test; predictions ghi trước futures và independently checked; test không quay lại training. |

Không cóp-values được báo trong protocol; không suy diễnp-value từ percentile
bootstrap. Các intervals không chứng minh global optimality hay uncertainty
coverage tốt chỉ vì hash/replay PASS. Report không sửa raw artifacts và không
SSH/rsync/full-launch thay user.

## Artifacts phân tích

-`audit.json`: phạm vi xác minh và bounded replay.
-`summary_verified.json` /`diagnostic_summary_verified.json`: numerical recomputation.
-`supplementary_diagnostics.json`: MSE/sign/override/coverage, hậu nghiệm exploratory.
-`verified_profile_differences.csv`: primary clustered cells.
-`checkpoint_selection_verified.csv`:3 truth epochs80/20/40; shuffled60/80/20,
  tất cả chọn theo true dev MSE.
-`assignment_state_gains.csv` /`largest_harmful_overrides.csv`: không bỏ zeros/lỗ.
-`label_coverage_by_clock.csv`, `logged_assignment_decode_audit.csv`, figures,
  và script phân tích cùng hash ledger.
'''
# Normalize typography for readable paragraphs while keeping source numbers exact.
import re
lines=[]
for line in text.splitlines():
 if '`' not in line and not line.startswith('!['):
  line=re.sub(r'(?<=[A-Za-zÀ-ỹ])(?=\d)', ' ', line)
  line=re.sub(r'(?<=\d)(?=[A-Za-zÀ-ỹ])', ' ', line)
  line=re.sub(r'([,;])(?=\S)', r'\1 ',line)
  line=re.sub(r'(?<=\d)\.(?=[A-ZÀ-Ỹ])', '. ',line)
  line=line.replace('H 48','H48').replace('n 750','n=750').replace('n 222','n=222')
 line=re.sub(r'^-(?=\S)', '- ',line)
 lines.append(line)
text='\n'.join(lines)+'\n'
(OUT/'analysis_vi.md').write_text(text)
(OUT/'report_script.py').write_bytes(Path(__file__).read_bytes())
print(OUT/'analysis_vi.md')
