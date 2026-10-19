"""Generate standalone scientific plots and Vietnamese analysis report."""
from collections import defaultdict
from datetime import datetime
import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
A=json.loads((OUT/'analysis.json').read_text())
T=json.loads((OUT/'teacher_analysis.json').read_text())
N=json.loads((OUT/'numerical_sensitivity.json').read_text())
S=json.loads((OUT/'statistics.json').read_text())
RAW=Path(A['raw'])
ARMS=('direct_ppo','planner_imitation','imitation_ppo')
LABELS=('Direct PPO','Planner imitation','Imitation + PPO')
COLORS=('#4385be','#cf9147','#378b67')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})

def save(fig,name):
 fig.savefig(OUT/f'{name}.png',bbox_inches='tight');fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def ci(values):
 v=np.asarray(values);avg=float(v.mean());half=2.2621571628540993*float(v.std(ddof=1))/math.sqrt(10)
 return avg,(avg-half,avg+half)

g=A['groups'];primary=A['results']['primary'];pair=primary['paired_replica_costs']
ref=g['middle_H12_independent_dp_matching']['objective'];planner=g['middle_H12_joint_rollout']['objective']
fig,ax=plt.subplots(figsize=(7.8,4.4))
for i,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS)):
 values=list(pair[arm].values());avg,bounds=ci(values)
 ax.plot([i-.23,i+.23],[avg,avg],color=color,lw=4)
 ax.errorbar(i,avg,yerr=avg-bounds[0],color='#222222',capsize=5,fmt='none',lw=1.3)
 ax.scatter(i+np.linspace(-.18,.18,10),values,s=16,color='#222222',alpha=.7,zorder=3)
 ax.text(i,avg+2.1,f'{avg:.2f}\n{100*(1-avg/ref):.2f}% vs reference',ha='center',fontsize=9)
ax.axhline(ref,color='#555555',ls='--',label=f'Reference {ref:.2f}')
ax.axhline(planner,color='#8057a3',ls=':',label=f'Joint rollout {planner:.2f}')
ax.set(xticks=range(3),xticklabels=LABELS,ylim=(175,201),ylabel='Mean priced cost (lower is better)',title='N4 / K2 / H12: new held-out panel')
ax.legend(loc='upper right',fontsize=9);fig.text(.1,-.01,'Dots: 10 replicas; colored line: mean; error bars: t 95% CI conditional on fixed test panel.',fontsize=8)
save(fig,'primary_cost')

fig,axes=plt.subplots(1,2,figsize=(10,4.1))
for arm,label,color in [(ARMS[0],LABELS[0],COLORS[0]),(ARMS[2],LABELS[2],COLORS[2])]:
 rows=[r for r in A['development_curves'] if r['arm']==arm and r['capacity']=='middle']
 steps=sorted({r['step'] for r in rows});means=[np.mean([r['objective'] for r in rows if r['step']==s]) for s in steps]
 axes[0].plot(np.array(steps)/1000,means,'o-',label=label,color=color)
axes[0].set(xlabel='PPO physical steps per replica (thousands)',ylabel='Development priced cost',title='PPO before/after imitation initialization')
axes[0].legend()
rows=[r for r in T['bc_diagnostics'] if r['split']=='validation']
for field,label in [('choice_teacher_agreement','Choice states'),('teacher_agreement','All states')]:
 epochs=sorted({r['epoch'] for r in rows});values=[100*np.mean([r[field] for r in rows if r['epoch']==e]) for e in epochs]
 axes[1].plot(epochs,values,'o-',label=label)
axes[1].set(xlabel='Imitation epochs',ylabel='Validation teacher agreement (%)',ylim=(0,100),title='Teacher imitation learning')
axes[1].legend();fig.tight_layout();save(fig,'learning_curve')

fig,ax=plt.subplots(figsize=(8,4.3))
components=('maintenance_cost','failure_cost','unavailability_cost','waiting_cost')
names=('Maintenance','Failure','Downtime','Overdue waiting')
base=g['middle_H12_independent_dp_matching']
for i,(component,label) in enumerate(zip(components,names)):
 values=[g['middle_H12_'+arm][component]-base[component] for arm in (*ARMS,'joint_rollout')]
 ax.bar(np.arange(4)+(i-1.5)*.18,values,width=.18,label=label)
ax.axhline(0,color='#222222',lw=.8);ax.set(xticks=range(4),xticklabels=(*LABELS,'Joint rollout'),ylabel='Cost difference vs reference',title='K2 / H12: savings mainly come from fewer failures')
ax.legend(ncol=2,fontsize=9);fig.tight_layout();save(fig,'cost_mechanism')

# Exploratory matched contrasts, never promoted to additional primary tests.
exploratory={}
for a,b in [(ARMS[1],ARMS[0]),(ARMS[2],ARMS[1])]:
 delta=[pair[a][str(s)]-pair[b][str(s)] for s in range(144000,144010)]
 avg,bounds=ci(delta)
 exploratory[a+'_vs_'+b]=dict(difference=avg,ci95=bounds,wins=sum(d<0 for d in delta),
                             relative_reduction=1-np.mean(list(pair[a].values()))/np.mean(list(pair[b].values())))
cohort_episodes=defaultdict(list)
with (RAW/'episodes.csv').open() as stream:
 for r in csv.DictReader(stream):
  if r['capacity']=='middle' and r['horizon']=='12':cohort_episodes[int(r['cohort']),r['controller']].append(float(r['objective']))
cohort_cost={str(cs):{name:float(np.mean(cohort_episodes[cs,name])) for name in (*ARMS,'independent_dp_matching','joint_rollout')} for cs in range(147000,147010)}
selected_bc=[]
for seed in range(144000,144010):
 epoch=S['selected'][f'planner_imitation:{seed}']['selected_update']
 selected_bc.append(next(r for r in T['bc_diagnostics'] if r['train_seed']==seed and r['epoch']==epoch and r['split']=='validation'))
metrics=dict(primary=primary,exploratory=exploratory,cohort_costs=cohort_cost,
             selected_bc_validation={k:float(np.mean([r[k] for r in selected_bc])) for k in ('teacher_agreement','choice_teacher_agreement','mean_teacher_score_regret','choice_mean_teacher_score_regret')},
             primary_sign_test_two_sided=2/1024,
             audit=A['audit'],teacher_audit=T['audit'],numerical_sensitivity={k:v for k,v in N.items() if k not in ('details','episode_sensitivity')})
(OUT/'report_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')

lines=['# Phân tích maintenance contention imitation v1','',
 '## Material Passport','',
 'Schema: ARS 9. Material: validation report. Mode: validate. Origin: dữ liệu lab đã chuyển về Mac.',
 'Verification Status: **ANALYZED** — audit artifact và replay; không chạy lại toàn bộ training.',
 f'Run: `{RAW.name}`. Commit: `{A["commit"]}`.',
 f'Thời gian lab: {A["started"]} → {A["finished"]} (khoảng 107 phút 40 giây).','',
 '## Kết luận chính','',
 '**Cả ba policy đạt gate kinh tế ở K2/H12; imitation → PPO tốt nhất.** Hybrid giảm cost 7,25% so với baseline và thêm 1,62% so với PPO trực tiếp, thắng 10/10 replica. Tuy nhiên, primary hypothesis yêu cầu ít nhất 2% cải thiện thêm, nên verdict confirmatory vẫn là **FAIL**. Đây là benefit có hướng ổn định nhưng chưa đạt effect size đã khóa; không phải bằng chứng “imitation không có tác dụng”.','',
 'Policy tập trung đã học được quyết định điều phối có ích. Kết quả chưa chứng minh MARL cần thiết, và chưa đảm bảo tốt trên mọi workload/capacity.','',
 '## Primary và gate kinh tế','',
 '| Phương pháp | Priced cost | Base cost | Giảm priced vs ref | Giảm base vs ref | Capture savings của planner | Gate kinh tế |',
 '|---|---:|---:|---:|---:|---:|---|']
for arm,label in zip(ARMS,LABELS):
 r=A['results']['arms'][arm]['middle_H12'];capture=(ref-r['learner_cost'])/(ref-planner)
 lines.append(f'| {label} | {r["learner_cost"]:.4f} | {g["middle_H12_"+arm]["base_cost"]:.4f} | {100*r["priced_reduction"]:.2f}% | {100*r["base_reduction"]:.2f}% | {100*capture:.2f}% | {r["status"]} |')
lines += [f'| Reference independent DP | {ref:.4f} | {base["base_cost"]:.4f} | — | — | — | — |',
 f'| Joint rollout | {planner:.4f} | {g["middle_H12_joint_rollout"]["base_cost"]:.4f} | {100*(1-planner/ref):.2f}% | {100*(1-g["middle_H12_joint_rollout"]["base_cost"]/base["base_cost"]):.2f}% | 100% | diagnostic |','',
 'Planner là model-based controller khả thi, không phải optimum N4. Capture savings là tỷ lệ so với lợi ích quan sát của controller này, không phải tỷ lệ gần tối ưu.','',
 '![Primary cost](primary_cost.png)','',
 f'Primary hybrid − direct: Δcost **{primary["mean_difference"]:.3f}**, t95% CI **[{primary["ci95"][0]:.3f}, {primary["ci95"][1]:.3f}]**, **10/10 wins**. Reduction **{100*primary["priced_reduction"]:.2f}%**, dưới ngưỡng 2%. Nominal và waiting guards đạt. Tất cả seed được giữ lại; không có development-infeasible model.','',
 'Mỗi CI dùng 10 paired training-replica means trên cùng 500 episode primary. CI có điều kiện trên panel này; không coi 5.000 episode là 5.000 replica độc lập. Với n=10, t-CI giả định phân phối replica differences đủ phù hợp; sign-only sensitivity hai phía cho 10/10 cùng hướng có p=0,001953, chỉ hỗ trợ hướng benefit, không chứng minh đạt ngưỡng 2%.','',
 'Các contrast sau là exploratory, không hiệu chỉnh nhiều so sánh:','',
 '| Contrast | Δpriced cost | t95% CI | Wins | Giảm tương đối |',
 '|---|---:|---:|---:|---:|']
for key,r in exploratory.items():lines.append(f'| {key} | {r["difference"]:.4f} | [{r["ci95"][0]:.4f}, {r["ci95"][1]:.4f}] | {r["wins"]}/10 | {100*r["relative_reduction"]:.2f}% |')
lines += ['',
 'Imitation đơn thuần chưa hơn PPO trực tiếp rõ ràng: CI chứa 0 và chỉ thắng 6/10. Fine-tuning từ imitation thắng imitation trên 10/10 seed, nhưng có thêm thông tin/compute; không diễn giải thành so sánh hiệu quả compute công bằng.','',
 '## Policy đã học gì?','',
 'Direct development K2: mean cost khởi tạo 363,89 → checkpoint cuối 180,71. Imitation: cùng khởi tạo → cuối 180,12. Hybrid khởi tạo từ selected BC với cost 179,93 → cuối 178,28. Không lấy các giá trị development để thay thế test result. Learning curve vẫn cải thiện muộn; chưa chứng minh hội tụ hay ngân sách đủ.','',
 'BC validation trên các trạng thái có hơn một subset khả thi: agreement **22,11% → 77,85%**, teacher root-score regret **9,684 → 1,325**. Train agreement cuối 78,34%, gần validation. Mức học này cho thấy encoder/action space có thể biểu diễn các quyết định hữu ích; chưa chứng minh biểu diễn hoàn hảo hay loại trừ lỗi optimization/coverage. Overall agreement cuối 80,08% gồm cả forced STOP; luôn báo riêng choice states.','',
 f'Teacher noise: independent 128-forecast probe đổi quyết định ở **{100*T["teacher_noise"]["changed_rate"]:.2f}%** trong 320 probe; trong 280 choice states là **{100*T["teacher_noise"]["choice_changed_rate"]:.2f}%**. Root regret của nhãn 32-forecast dưới probe mean {T["teacher_noise"]["mean_regret"]:.3f}, p95 {T["teacher_noise"]["p95_regret"]:.3f}, max {T["teacher_noise"]["max_regret"]:.3f}. 128 forecast vẫn là ước lượng Monte Carlo, không phải ground truth; 320 probe cố định chỉ mô tả phần validation được kiểm tra.','',
 '![Learning diagnostics](learning_curve.png)','',
 '## Cơ chế cost ở K2/H12','',
 '| Controller | Maintenance | Failure | Downtime | Overdue | Failures/episode | Waiting violation | Utilization |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for name,label in [('independent_dp_matching','Reference'),*zip(ARMS,LABELS),('joint_rollout','Planner')]:
 v=g['middle_H12_'+name]
 lines.append(f'| {label} | {v["maintenance_cost"]:.4f} | {v["failure_cost"]:.4f} | {v["unavailability_cost"]:.4f} | {v["waiting_cost"]:.4f} | {v["failures"]:.4f} | {100*v["waiting_violation"]:.2f}% | {100*v["utilization"]:.2f}% |')
lines += ['',
 'Hybrid giảm failures từ 2,814 xuống 1,666 (khoảng 40,8%). Tiết kiệm chủ yếu đến từ failure cost: 42,21 → 24,99. Maintenance và downtime tăng nhẹ; vì vậy kết luận đúng là giảm **tổng cost bằng phòng tránh failure**, không phải giảm mọi thành phần hoặc downtime. Hybrid so với direct cũng đánh đổi waiting violation 0% → 0,54%, vẫn dưới guard reference+2pp. Utilization cao không tự chứng minh lịch tối ưu.','',
 '![Cost components](cost_mechanism.png)','',
 '## Heterogeneity và ngoài phân phối training','',
 'Mean hybrid tốt hơn direct trên cả 10 cohort, nhưng cả ba learner đều thua reference ở cohort147002. Profile này có durations [2,1,3,3], restoration [0,75;1;0,75;1]. PPO đã thấy profile vật lý này trong training của mọi replica; không thể chỉ quy lỗi cho một parameter combination chưa từng thấy. Trong teacher training data, chỉ seed144007 có đúng1 trajectory của profile này (các seed còn lại không có); đó là hạn chế coverage của BC, chưa đủ để giải thích duy nhất lỗi của cả ba policy.','',
 '| Cohort | Reference | Planner | Direct PPO | Imitation | Hybrid |',
 '|---|---:|---:|---:|---:|---:|']
for cs,v in cohort_cost.items():lines.append(f'| {cs} | {v["independent_dp_matching"]:.3f} | {v["joint_rollout"]:.3f} | {v[ARMS[0]]:.3f} | {v[ARMS[1]]:.3f} | {v[ARMS[2]]:.3f} |')
lines += ['',
 'Cohort147002: hybrid191,56 > reference189,28, trong khi planner178,30. Vẫn còn lợi ích chưa được học trên profile này. Để xác định vì sao cần kiểm tra state/action errors theo profile; dữ liệu aggregate chưa tách duy nhất approximation, coverage, optimization hoặc teacher visitation shift.','',
 '| Test condition | Direct gain vs ref | Imitation gain vs ref | Hybrid gain vs ref |',
 '|---|---:|---:|---:|']
for cap,h in [('low',12),('middle',12),('high',12),('low',24),('middle',24),('high',24)]:
 values=[100*A['results']['arms'][arm][f'{cap}_H{h}']['priced_reduction'] for arm in ARMS]
 lines.append(f'| K{dict(low=4,middle=2,high=1)[cap]} / H{h} | {values[0]:+.2f}% | {values[1]:+.2f}% | {values[2]:+.2f}% |')
lines += ['',
 'Dấu + là cost giảm; dấu − là cost tăng. K4/H12 nominal guard đạt cho cả ba. K1 và H24 không nằm trong training, nên là sensitivity OOD. K1/H12 hybrid chỉ giảm priced0,92% và CI so với reference còn chứa0; waiting violation88,08% so với reference64,2% và planner21,2%. Base cost giảm4,98% nhưng overdue cost tăng24,792 →34,541. K1/H24 cả ba learner tăng tổng cost. Chưa có bằng chứng policy học tốt tradeoff phục vụ/chờ khi capacity này bị bóp thêm.','',
 '## So với experiment trước và giới hạn kết luận','',
 'Direct PPO giữ pipeline/architecture cũ nhưng đổi training replicas và development/test panels. Gain mới5,72% so với gain cũ1,92% không phải controlled evidence rằng thuật toán đã được cải tiến. Planner gain trên panel mới8,71%, so với5,01% của panel trước; lượng lợi ích có thể khai thác thay đổi theo workload. Cần giữ cả kết quả trước và mới; không chọn riêng panel đạt gate.','',
 'Bằng chứng hiện tại hỗ trợ: (1) resource scheduling có economic impact ở operating regime K2 này; (2) một policy tập trung học được phần lớn lợi ích planner; (3) supervision/initialization có benefit vừa phải theo cùng test panel. Nó không chứng minh credit assignment là nguyên nhân duy nhất của gap, không chứng minh transformer/mixer khác sẽ giải quyết gap, và không chứng minh MARL cần thiết.','',
 'Nếu tiếp tục, hướng có ích là chẩn đoán state/action regret trên cohort147002 và K1 trước khi đổi thuật toán, hoặc khóa operating regime K2 rồi xác nhận trên workload panel rộng hơn. Đây là đề xuất tiếp theo; chưa triển khai một full experiment mới.','',
 '## Audit và chi phí thực thi','',
 f'- Manifest COMPLETED, clean commit; **{A["audit"]["raw_files"]} artifact hashes** và source/protocol/lock snapshots khớp. Raw artifacts không chỉnh sửa.',
 '- Đủ30 model,180 checkpoint files,9830400 PPO steps/819200 training episodes,256000 optimizer updates;6400 PPO progress rows+800 BC rows.',
 '- Đủ62400 development episodes,105000 test episodes,1890000 intervals và420000 machine rows; test grids/cost/machine totals đúng. Tất cả physical transitions và4 deterministic reference actions được replay.',
 '- Teacher:138240 CSV/tensor rows,11520 trajectories, no reserved seed use; tất cả actual trajectory/behavior/target checks đạt. Full forecast replay720 teacher calls,1080 test planner calls;320 independent noise probes được replay. Không gọi đây là full replay mọi simulated forecast trong dataset.',
 '- BC diagnostic recomputation100 checkpoint/split records khớp trong numerical tolerance; initial direct/BC parameters giống nhau, hybrid initial parameters và parent hash khớp selected BC; BC critic head không đổi.',
 f'- Learned batched replay1620000 actions phát hiện **{N["mismatches"]}** tie differences, **{N["primary_mismatches"]}** thuộc hybrid K2/H12. Mọi pair có trạng thái/profile hiện tại đối xứng; scalar replay tại27 logged states khớp27/27 original choices (logit pairs hòa điểm chính xác trên Mac), và27 affected episodes giữ nguyên total cost (mean changes0 ở mọi cell bị ảnh hưởng). Probability/value sai khác tối đa {A["audit"]["max_log_probability_error"]:.3g}/{A["audit"]["max_value_error"]:.3g}. Verdict primary/economic không đổi. Audit status vẫn NUMERICAL_ACTION_MISMATCH, không ghi zero-mismatch PASS.',
 '- No full-training rerun: report status ANALYZED. No claim of exact portability of batched argmax ties.',
 '- Tất cả10 primary physical profiles có trong PPO training của mọi replica; seed disjointness không đồng nghĩa unseen-profile generalization.',
 '- Lab training+development wall time: direct tổng1626s, BC548s, hybrid fine-tuning1624s; teacher generation624s dùng chung BC/hybrid. Hybrid còn thêm teacher/BC compute. Mean K2/H12 inference/episode: learner khoảng8,76ms, rollout56,52ms, reference4,03ms trên runtime lab đã ghi. Không coi đây là benchmark tốc độ portable hoặc compute-matched superiority.','',
 '## Fallacy scan — 11/11 checked','',
 '| Pattern | Disposition |',
 '|---|---|',
 '| Simpson/group reversal | Report từng cohort/capacity: cohort147002 reversal vs reference; không tuyên bố mọi workload tốt. |',
 '| Ecological inference | Training-seed means không suy ra mỗi episode/machine tốt hơn. |',
 '| Berkson/selection bias | Dev selection công khai; toàn bộ30 models giữ lại, panel hữu hạn. |',
 '| Collider/control bias | Không điều kiện kết quả primary trên post-treatment success/utilization. |',
 '| Base-rate neglect | Agreement báo riêng choice states, forced STOP được đếm; waiting base rate báo cùng. |',
 '| Regression to mean | Có fresh direct comparator cùng replicas; không dùng dev pre/post làm test effect. |',
 '| Survivorship | Run đầy đủ, không loại seed; no dev-infeasible omission. |',
 '| Look-elsewhere | Một primary contrast; contrasts còn lại exploratory, không corrected-confirmatory claims. |',
 '| Forking paths | Giữ2%/5%/3% và các guards đã khóa, không đổi ngưỡng sau kết quả. |',
 '| Correlation/causal mechanism | Benefit của gói teacher+BC init+compute; không kết luận một nguyên nhân learning gap hoặc MARL necessity. |',
 '| Reverse causality | Training/init trước sealed test; không có test-feedback selection. |','',
 '## Files','',
 '- analysis.json/statistics.json: metric, CI, dev curves, cohort, numerical replay audit.',
 '- teacher_analysis.json: teacher diagnostics/seed/physics/noise audit.',
 '- numerical_sensitivity.json: 27 cases và full affected-episode cost replay.',
 '- analyze.py,teacher_audit.py,numerical_sensitivity.py,generate_report.py: reproducible read-only analysis.',
 '- primary_cost,learning_curve,cost_mechanism: PNG và PDF xuất độc lập.','']
report='\n'.join(lines)
for old,new in [('cohort147002','cohort 147002'),('Cohort147002','Cohort 147002'),('seed144007','seed 144007'),('đúng1 trajectory','đúng 1 trajectory')]:
 report=report.replace(old,new)
for figure in ('primary_cost','learning_curve','cost_mechanism'):
 report=report.replace(f']({figure}.png)',f'](<{OUT / (figure+".png")}>)')
(OUT/'analysis_vi.md').write_text(report)
print(json.dumps(dict(report=str(OUT/'analysis_vi.md'),primary=primary['status'],teacher_noise=T['teacher_noise'])))
