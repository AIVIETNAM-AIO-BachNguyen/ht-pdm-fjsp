import json
from pathlib import Path
from statistics import mean
from datetime import datetime
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/marl-learning-analysis-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
a=json.loads((OUT/'analysis.json').read_text());behavior=json.loads((OUT/'behavior.json').read_text());numerics=json.loads((OUT/'numerical_episode_sensitivity.json').read_text())
g=a['groups'];contrasts=a['contrasts'];primary=contrasts['middle_H12'];L='central_subset_ppo'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
ref=g['middle_H12_independent_dp_matching'];ppo=g[f'middle_H12_{L}'];planner=g['middle_H12_joint_rollout']
price_capture=(ref['objective']-ppo['objective'])/(ref['objective']-planner['objective'])
base_capture=(ref['base_cost']-ppo['base_cost'])/(ref['base_cost']-planner['base_cost'])
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,metric,label in zip(axes,['objective','base_cost'],['Priced cost','Base cost']):
 values=[ref[metric],ppo[metric],planner[metric]]
 ax.bar(range(3),values,color=['#8596a5','#197f86','#986cad'])
 for i,v in enumerate(values):ax.text(i,v+.8,f'{v:.3f}',ha='center')
 ax.set(xticks=range(3),xticklabels=['Reference','Subset PPO','Joint planner'],ylabel='Mean cost per episode',ylim=(0,225),title=label)
 ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('PPO learns part of the shared-capacity benefit\nN=4, K=2, H=12; 10 training seeds × fixed held-out panel')
fig.savefig(OUT/'primary_cost.png',dpi=180);fig.savefig(OUT/'primary_cost.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
for ax,cap in zip(axes,['middle','low']):
 curves=a['development_curves'];steps=sorted(set(x['step'] for x in curves))
 for seed in sorted(set(x['train_seed'] for x in curves)):
  values=[next(x['objective'] for x in curves if x['step']==s and x['train_seed']==seed and x['capacity']==cap) for s in steps]
  ax.plot(np.array(steps)/1000,values,color='#197f86',alpha=.25,lw=1)
 avg=[mean(x['objective'] for x in curves if x['step']==s and x['capacity']==cap) for s in steps]
 ax.plot(np.array(steps)/1000,avg,color='#197f86',lw=2.5,label='Mean PPO; thin lines = seeds')
 devref=191.38 if cap=='middle' else 162.685
 ax.axhline(devref,color='#8596a5',ls='--',label='Selected reference')
 ax.set(xlabel='Physical training steps (thousands)',ylabel='Development priced cost',title='K=2 primary' if cap=='middle' else 'K=4 nominal')
 ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
fig.suptitle('Learning is clear, but later checkpoints still improve\nDevelopment only; initialization is excluded from checkpoint selection')
fig.savefig(OUT/'learning_curve.png',dpi=180);fig.savefig(OUT/'learning_curve.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
components=['maintenance_cost','failure_cost','unavailability_cost','waiting_cost'];colors=['#8a66a1','#dc9753','#409c96','#b54b58']
for ax,cap in zip(axes,['middle','high']):
 b=g[f'{cap}_H12_independent_dp_matching'];p=g[f'{cap}_H12_{L}'];delta=[p[m]-b[m] for m in components]
 bars=ax.bar(range(4),delta,color=colors)
 for bar,d in zip(bars,delta):ax.text(bar.get_x()+bar.get_width()/2,d+(.8 if d>=0 else -.8),f'{d:+.3f}',ha='center',va='bottom' if d>=0 else 'top',fontsize=9)
 ax.axhline(0,color='#333',lw=.8);ax.set(xticks=range(4),xticklabels=['Maintenance','Failure','Downtime','Overdue wait'],ylabel='PPO minus reference cost',ylim=(-19,37),title=f"K={2 if cap=='middle' else 1}: total {sum(delta):+.3f}"+(' (untrained capacity)' if cap=='high' else ''))
 ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('K2 benefit is modest; K1 penalty exposes waiting trade-offs\nNegative values are savings; N=4, H=12')
fig.savefig(OUT/'cost_mechanism.png',dpi=180);fig.savefig(OUT/'cost_mechanism.pdf');plt.close(fig)

summary_rows=[]
for cap,k in [('low',4),('middle',2),('high',1)]:
 c=contrasts[f'{cap}_H12'];p=g[f'{cap}_H12_{L}'];b=g[f'{cap}_H12_{c["reference"]}']
 summary_rows.append(f"| K{k} | {b['objective']:.3f} | {p['objective']:.3f} | {c['planner_cost']:.3f} | {100*c['priced_reduction']:+.2f}% | {100*c['base_reduction']:+.2f}% | {c['training_seed_wins']}/10 | {c['status']} |")
seed_rows=[]
for r in a['paired_seeds']:
 if r['capacity']=='middle' and r['horizon']==12:
  seed_rows.append(f"| {r['train_seed']} | {a['selected_steps'][str(r['train_seed'])]} | {r['learner_cost']:.3f} | {100*(1-r['learner_cost']/r['reference_cost']):.2f}% | {100*r['priced_gap_capture']:.2f}% |")
cohort_rows=[];cohort_values=[]
for cs in sorted(set(r['cohort'] for r in a['cohorts'])):
 panel=[r for r in a['cohorts'] if r['cohort']==cs and r['capacity']=='middle' and r['horizon']==12]
 diff=mean(r['difference'] for r in panel);cohort_values.append(diff)
 cohort_rows.append(f"| {cs} | {mean(r['learner_cost'] for r in panel):.3f} | {mean(r['reference_cost'] for r in panel):.3f} | {diff:+.3f} | {sum(r['difference']<0 for r in panel)}/10 |")
devmeans={str(step):{cap:mean(r['objective'] for r in a['development_curves'] if r['step']==step and r['capacity']==cap) for cap in ['middle','low']} for step in sorted(set(r['step'] for r in a['development_curves']))}
rows_all=[]
for cap in ['low','middle','high']:
 for name in ['risk_skill_rule','deadline_matching','lookahead_matching','independent_dp_matching','joint_rollout',L]:
  r=g[f'{cap}_H12_{name}'];rows_all.append(f"| {cap} | {name} | {r['objective']:.3f} | {r['base_cost']:.3f} | {100*r['waiting_violation']:.2f}% |")
N1=g[f'high_H12_{L}'];R1=g['high_H12_independent_dp_matching']
mechanism='\n'.join(f"| {label} | {ref[m]:.4f} | {ppo[m]:.4f} | {planner[m]:.4f} |" for label,m in [('Maintenance cost','maintenance_cost'),('Failure cost','failure_cost'),('Downtime cost','unavailability_cost'),('Overdue cost','waiting_cost'),('Failures','failures'),('Waiting violation','waiting_violation'),('Utilization','utilization')])
numerical_mean_changes={}
for r in numerics:
 cap=r['key'][2];h=r['key'][3];key=f'{cap}_H{h}';numerical_mean_changes[key]=numerical_mean_changes.get(key,0)+r['difference']/5000
plot='![Primary cost]('+str(OUT/'primary_cost.png')+')'
learningplot='![Learning curve]('+str(OUT/'learning_curve.png')+')'
mechplot='![Cost mechanism]('+str(OUT/'cost_mechanism.png')+')'
report=f'''# Phân tích maintenance contention learning v1

## Material Passport

- Mode: ARS experiment-agent / validate; language: Vietnamese.
- Raw input: `{a['raw']}`; commit `{a['commit']}`.
- Verification Status: ANALYZED. Reproduced full physical trace and training
  provenance; replayed all selected policy actions in batches and a planner
  forecast subset. Cross-platform deterministic action reproduction has6
  secondary near-tie mismatches, investigated below. No full training rerun.
- Output: analysis.json, analyze.py, behavior.json, numerical replay/sensitivity
  JSONs, report này, scientific PNG/PDF figures. Raw artifacts remain unchanged.

**PPO học được một phần lợi ích điều phối ở K2, nhưng FAIL protocol về độ lớn
cải thiện.** Effect1,92% priced cost và1,42% base cost là nhỏ hơn ngưỡng5%/3%
đã khóa. 10/10 training seeds thắng và95% CI cost difference dưới0; nominal và
waiting guards đều đạt. Do đó không nên đồng nhất FAIL protocol với không học.
Chưa có MARL trong thí nghiệm này.

## Dữ liệu, ngân sách và audit

COMPLETED, Git clean, đúng commit. Run `{a['started']}` → `{a['finished']}`,
khoảng43 phút16 giây. Đủ10models,4.915.200 physical steps,409.600 training
episodes,102.400 optimizer steps;22.400 development episodes,45.000 test
episodes,810.000 test intervals,180.000 machine rows. Cả10 seed đủ budget và
80%K2/20%K4 training mix. Teacher labels0, không seed dev/test trực tiếp vào train.

Kiểm tra90 raw file hashes/sizes,60checkpoint files (initial+4fixed budgets mỗi
seed và10selected copies), parameter changes thực sự, source snapshot, seed
splits,3200 progress records/counters/finite losses, tất cả409.600 configuration
và hazard counter roles/hash. Recompute development reference và checkpoint
selection khớp; mọi seed có checkpoint dev-feasible, không drop seed nào.

Replayed all810.000 scalar physical transitions from reconstructed actual
hazard seeds: pre/post state, action feasibility, cost và machine/episode
reconciliation khớp. Reference actions216.000 bước khớp; planner candidate
argmin54.000 bước khớp; forecasts1080 bước thuộc60 episode khớp hoàn toàn.
Replayed540.000 learned action logits/log-probs/values theo batch; sai khác số
log-prob tối đa3,82e-6 và value2,87e-6. Primary K2/H12 có0 action mismatch trong
60.000 bước; phần near-tie secondary xem mục riêng. Không sửa raw sau phân tích.

Các seed cấu hình test mới không đảm bảo physical profile mới: audit cho thấy
cả10 primary profiles đã xuất hiện trong training của từng seed. Generator
hữu hạn, đúng scope protocol. Vì vậy đây là generalization sang seed/draw mới,
không phải đảm bảo chưa thấy các tổ hợp duration/restoration.

## Primary: N4, K2, H12

Reference `independent_dp_matching`, chọn theo priced cost development, cost
199,892. PPO196,0508; joint planner189,870. **PPO giảm1,92%**; planner giảm
**5,01%** trên panel mới. PPO captures **{100*price_capture:.2f}%** observed priced
planner savings và **{100*base_capture:.2f}%** observed base-cost savings. Đây là
fraction của một achievable planner, không phải fraction của optimal headroom.

95% paired training-seed CI của Δcost là
[{primary['ci95'][0]:.4f}, {primary['ci95'][1]:.4f}], estimate−3,8412.
Unit10training seeds, từng seed mean trên10cohorts×50paired shocks. Reference
và planner chạy1lần mỗi panel; không nhân bản chúng thành10independent replicas.
CI conditional trên finite common test panel, không phải environment-population CI.

| Gate đã khóa | Quan sát | Đạt? |
|---|---|---|
| Priced-cost reduction≥5% | 1,92% | Không |
| Wins≥8/10 seeds | 10/10 | Có |
| Upper95% CI<0 | −3,3295 | Có |
| Base-cost reduction≥3% | 1,42% | Không |
| K4/H12 base ratio≤1,1 | 1,00132 | Có |
| K2 waiting violation≤reference+2pp | 0,34% so với5,60% | Có |

{plot}

Ngưỡng5% gần mức5,01% mà planner đạt trên panel mới; để đạt gate này PPO cần
xấp xỉ99,7% mức savings quan sát của planner đó. Đây là mô tả độ khó của target,
không đổi protocol sau khi xem kết quả và không coi planner là ceiling.

| Training seed | Selected physical steps | Test cost | Reduction | Priced planner gap capture |
|---|---:|---:|---:|---:|
{chr(10).join(seed_rows)}

Capture giữa25,98% và47,24%; consistency qua seed tốt nhưng magnitude chưa đủ.
Không thể suy rằng tăng budget chắc chắn đóng hết gap, hoặc RL nói chung không
thể làm tốt hơn từ một PPO architecture/configuration hiện tại.

## Learning curve và cơ chế

Mean development K2 initial323,195 →191,0525 ở122.880steps →188,3400 ở245.760
→184,3325 ở368.640 →183,1405 ở491.520. K4 initial306,3935 →162,899 ởcuối.
8/10seed chọn checkpoint cuối;138002/138007 chọn368.640. PPO đã học rõ; curve
còn cải thiện ở đoạn cuối nên chưa có bằng chứng chắc về convergence hoặc
mức budget đủ. Development cohort khác test; không so trực tiếp dev/test costs
để suy overfitting chỉ từ khác mean workload.

Mixed stochastic training cost (80%K2/20%K4) mean first20rollouts237,791 →
last20rollouts185,113. Entropy1,403→0,176, value loss10,416→0,321. Đây là
learning diagnostics, không thay thế deterministic held-out policy comparison.

{learningplot}

| Metric K2/H12 | Reference | PPO | Planner |
|---|---:|---:|---:|
{mechanism}

PPO so reference: Δmaintenance+1,8408, Δfailure−3,7920, Δdowntime−0,8676,
Δoverdue−1,0224, tổng−3,8412. Failure2,690→2,4372 giảm9,40%, so planner1,960
(giảm27,14% so reference). Waiting violation5,60%→0,34%; failed-wait ticks
4,570→1,3074. PPO tốt ở xử lý waiting nhưng còn cách planner về ngăn failure.
Utilization tăng82,30%→95,29%, so planner89,88%: dùng hết technician không tự
đồng nghĩa với lịch tối ưu.

Preventive starts ởage<2 chiếm23,36% PPO,16,66% planner,0% independent DP.
Early PM có thể hợp lý khi dự đoán capacity sẽ bận; rate cao hơn tự nó không
chứng minh lỗi. Dữ liệu cho thấy PPO maintenance cost cao hơn nhưng phòng
failure kém hơn planner; chưa tách causal contribution của early PM, chọn máy,
encoder, critic và policy optimization.

## Heterogeneity qua workload cohorts

| Cohort | PPO mean across seeds | Reference | Δcost | Seed thắng |
|---|---:|---:|---:|---:|
{chr(10).join(cohort_rows)}

8/10cohort có mean improvement;141001 và141008 worse. Cohort141001 thua ởcả
10seed. Khoảng83,2% tổng net savings đến từ141004 và141007. Vì thế10/10seed
wins biểu thị độ ổn định training trên cùng panel, không bảo đảm thắng mọi
workload. Đây là lý do cần giữ scope conditional của CI.

## Nominal và sensitivities

| Capacity/H12 | Reference cost | PPO | Planner | PPO priced reduction | PPO base reduction | Seed wins | Status |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(summary_rows)}

K4/H12 nominal: cost tăng0,13%, giữ guard≤10%; không có improvement rõ.

K1/H12 là capacity chưa huấn luyện. Priced cost tăng7,64% dù base cost giảm
4,13%. Maintenance+0,043, failure−13,365, downtime+3,084, overdue+31,1688;
waiting cost26,040→57,2088. Violation76,4%→93,22%, max-wait trung bình
5,644→7,4042. Utilization gần100% không giải quyết backlog. Fewer failures
không tự đồng nghĩa reliability tốt hơn khi máy failed chịu downtime/wait
lâu hơn và không thể failure lại trong lúc vẫn failed.

Behavior K1 khi idle và có failed demand: PPO chọn PM thay CM ở60,52% decision
opportunities, so reference33,55%, planner31,48%. Per-machine mean service
starts duration1:1,849→2,690; duration3:1,205→0,977. Duration3 max-wait
3,833→5,272 và violation30,35%→59,93%. Pattern phù hợp bias toward jobs ngắn
và trì hoãn jobs dài. Nó là chẩn đoán OOD, không chứng minh PPO được train
trên K1 cũng sẽ thất bại hoặc việc ưu tiên PM thay CM luôn sai.

H24 exploratory: K2 giảmpriced3,00% vàbase2,49%,10/10seed wins; K4cost tăng
5,28%; K1cost tăng9,62%. HorizonH24 chưa train. K4/H24 có33,56% PM ởage<2,
so reference0%, phù hợp vấn đề over-maintenance trong horizon shift; vẫn cần
intervention để quy nguyên nhân. Không dùng H24 để thay primary FAIL.

{mechplot}

## Numerical reproducibility limitation

6/540.000 learned action batch-replay mismatches trên Mac so logged Ubuntu:
5ởK1 và1ởK2/H24, chỉseed138008. Các cặp máy đổi có compact state và duration/
restoration giống nhau. Logits bằng nhau trên Mac hoặc rất gần nhau;1e-6
near-tie rule chưa bảo đảm deterministic tie behavior bất biến hardware/batch.
Scalar replay6states khớp3logged actions,3còn đổi máy. Xem
`numerical_replay_detail.json`. Không có primary K2/H12 mismatch được phát hiện
trong full batched replay; cost/physical transitions logged đều hợp lệ.

Tái chạy6affected episodes với checkpoint nguyên vẹn trên Mac cho costΔ
+7,0,0,−37,+22,0. Nếu thay sáu episode này trong panel, mean PPO cost K1/H12
chỉ+0,0014 vàK1/H24−0,003;K2/H24 không đổi. Các kết luận tăng cost ởK1 và
primary FAIL không đổi. Đây là bounded numerical sensitivity, không phải
chứng nhận mọi future trajectory đều reproducible. Không sửa checkpoint/code/
raw data để ép match; cần numerical tie test trước một experiment tiếp.

## Kết luận và bước kiểm tra phù hợp

Bài toán có economic scheduling headroom; centralized PPO đã khai thác được
một phần trên K2. Hạn chế hiện tại là học đủ giá trị dài hạn của việc chiếm
technician và chọn preventive/corrective schedule, cùng sensitivity khi capacity
và horizon đổi. Dữ liệu không chỉ ra thiếu conflict là nguyên nhân chính;
planner vẫn giảmcost5,01% trong primary panel. Chưa đủ để xác định PPO optimizer,
encoder hoặc budget là nguyên nhân duy nhất của gap còn lại.

Một diagnostic phù hợp tiếp theo là cùng encoder/action space, so direct PPO,
planner imitation và imitation→PPO trên new sealed panels. Nếu imitation tái
hiện được planner mà PPO yếu, đó là evidence hướng optimization/credit
assignment; nếu cả imitation yếu, cần kiểm tra representation/data coverage/
teacher noise trước. Teacher budget phải báo riêng; không gọi đây là fair
compute comparison khi supervision khác. Đây mới là đề xuất, chưa chạy hoặc
sửa thí nghiệm tiếp.

Kết quả không hỗ trợ MARL necessity: policy hiện tại centralized; không có
MARL comparator hoặc ràng buộc distributed execution thực tế. Cần learned
centralized baseline đủ mạnh và information/communication contract trước claim đó.

## All references H12

| Capacity | Controller | Priced cost | Base cost | Waiting violation |
|---|---|---:|---:|---:|
{chr(10).join(rows_all)}

## Statistical fallacy scan:11/11

- Simpson: split capacity/horizon/cohort; no claim of universal benefit from pooled mean.
- Ecological: conditional seed-level mean; not every episode/machine improves.
- Berkson: full fixed panels retained; finite known-model generator scope.
- Collider: no post-treatment filtering by failure, utilization or dev-feasible seed.
- Base rate: absolute costs, failure counts and violation rates accompany percentage changes.
- Regression-to-mean: selection on dev; fresh held-out seed panels; no test-based seed choice.
- Survivorship: all10models, fixed budgets and complete grids;0 dropped seeds.
- Look-elsewhere: one primary; K1/K4/H24 and cohort/behavior diagnostics exploratory.
- Forking paths: frozen thresholds/checkpoints; preserve FAIL; no post-hoc gate replacement.
- Causal overreach: controlled simulator policy effects, no attribution to a single mechanism or MARL.
- Reverse causality: temporal simulator intervention, not cross-sectional regression.

Ten training seeds give an approximate t-CI; no population-of-workloads uncertainty
is claimed. Training seeds independently sample initialization/actions/configs,
while every model shares a fixed evaluation panel. Small CI does not imply
small uncertainty over unseen real plants, laws, skills or workload distributions.
'''
(OUT/'analysis_vi.md').write_text(report)
(OUT/'report_metrics.json').write_text(json.dumps(dict(priced_gap_capture=price_capture,base_gap_capture=base_capture,development_mean_costs=devmeans,numerical_mean_cost_sensitivity=numerical_mean_changes),indent=2)+'\n')
print('Report and scientific PNG/PDFs written:',OUT)
