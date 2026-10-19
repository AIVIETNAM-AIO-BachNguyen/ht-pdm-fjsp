"""Compile the Vietnamese report from audited numbers; never edit raw data."""
import csv
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN=ROOT/'lab_results/maintenance_boundary_sensitivity_v1_full_cpu_20261010T101822Z'
s=json.loads((OUT/'independent_statistics.json').read_text())
integrity=json.loads((OUT/'integrity_audit.json').read_text())
physical=json.loads((OUT/'physical_replay_audit.json').read_text())
lat=json.loads((OUT/'latency_analysis.json').read_text())
assert physical['status']==lat['status']==integrity['status']=='PASS' and s['physical_replay']=='PASS'
m=json.loads((RUN/'manifest.json').read_text())
chosen=json.loads((RUN/'development_selection.json').read_text())['selected']
rows=list(csv.DictReader((RUN/'episodes.csv').open()))
names={'best_rule':'Rule tốt nhất trên dev','fixed_zero':'Fixed, terminal=0','flexible_zero':'Flexible, terminal=0',
       'fixed_tail':'Fixed, tail=12','flexible_tail':'Flexible, tail=12'}


def mean(condition,h,name,metric):
    return sum(float(r[metric]) for r in rows if r['condition']==condition and int(r['horizon'])==h and r['controller']==name)/512


def percentage_ci(c):
    return f"{100*c['relative_improvement']:.3f}% [{100*c['relative_confidence_interval'][0]:.3f}%, {100*c['relative_confidence_interval'][1]:.3f}%]"


lines=['# Phân tích maintenance boundary sensitivity v1 — 2026-10-11',
'', '## Material Passport','',
'Origin: academic-research-suite / experiment-agent / validate; inline analysis. Verification Status: **ANALYZED**.',
'Integrity, physical replay và thống kê tái tính: **PASS**. Không rerun toàn bộ planner inference hoặc full experiment; không coi audit này là xác nhận tối ưu của planner.',
'',f"Run: `{RUN.name}`. Commit: `{m['commit']}`. Raw data read-only; mọi đầu ra phân tích nằm trong thư mục report này.",
'','## Kết quả chính','',
'Cách xử lý cuối episode ảnh hưởng rõ tới backlog. Khi tính chi phí sau cuối kỳ, số máy hỏng còn chờ bắt đầu sửa giảm; tuy nhiên lợi ích chi phí ở giữa episode dài trong điều kiện primary skill_mask vẫn chưa chắc chắn. Phân công linh hoạt chưa đạt ngưỡng đầu tư 1% đã chốt.',
'', '## Dữ liệu và kiểm chứng','',
f"Run hoàn tất trong **{integrity['wall_seconds']/3600:.3f} giờ** (5 giờ 1 phút 32 giây). Linux x86_64, Python {m['python']}, NumPy {m['numpy']}; CPU, 16 logical CPUs, OMP/OPENBLAS/MKL=1. Không có training hoặc checkpoint.",
'',f"- {integrity['artifact_files']} artifacts, toàn bộ checksum và kích thước đúng manifest; source/config/plan trùng commit, dirty=false.",
'- Đủ 4.032 development episodes và 23.040 test episodes; toàn bộ grid 3 conditions × 3 horizons × 64 profiles × 8 shocks × 5 controllers.',
'- 921.600 physical decisions được replay; mọi action hợp lệ, state/cost và các chi phí thành phần đúng.',
f"- Tái dựng {physical['requests']:,} request records ở hai cutoff và {physical['carryover_requests']:,} carryover records. Không reset state hoặc công việc đang sửa tại H.",
f"- {physical['rule_actions_reproduced']:,} rule actions được tính lại; {physical['time_profile_rows']:,} time-profile rows khớp raw decisions. Planner actions được kiểm tra feasibility và restriction của fixed, không rerun lựa chọn Monte Carlo.",
'- Best rule được chọn lại từ development và khớp file selection. Selection được code khóa trước test; không chọn lại trên test.',
'- Seed panels mới tách biệt; 16 dev / 64 test physical profiles mỗi condition không trùng nhau hoặc 96 profiles của v1 trước, kể cả sau hoán vị máy/worker và bỏ horizon.',
'- Cả 27 contrasts, profile differences và bootstrap CIs đều được tính lại độc lập. 3 primary dùng 98,333333% CI; 24 exploratory dùng 95%. Đơn vị bootstrap là 64 profiles, conditional trên panel 8 shocks, không phải 23.040 episodes hoặc 921.600 ticks.',
'', '## Ba hypothesis primary (skill_mask)','',
'| Hypothesis | Effect | Khoảng tin cậy 98,33% | Gate đã chốt | Kết quả |',
'|---|---:|---:|---|---|']
p1=s['contrasts']['skill_mask:12:pending_terminal'];p2=s['contrasts']['skill_mask:48:interior_headroom'];p3=s['contrasts']['skill_mask:48:interior_assignment']
lines += [f"| P1: fixed_zero − fixed_tail, cutoff pending tại H12 | {p1['mean_difference']:.4f} máy | [{p1['confidence_interval'][0]:.4f}, {p1['confidence_interval'][1]:.4f}] | Lower CI > 0,1 máy | PASS |",
f"| P2: rule − fixed_tail, interior cost tại H48 | {100*p2['relative_improvement']:.3f}% | [{100*p2['relative_confidence_interval'][0]:.3f}%, {100*p2['relative_confidence_interval'][1]:.3f}%] | Lower CI > 1% | INCONCLUSIVE |",
f"| P3: fixed_tail − flexible_tail, interior cost tại H48 | {100*p3['relative_improvement']:.3f}% | [{100*p3['relative_confidence_interval'][0]:.3f}%, {100*p3['relative_confidence_interval'][1]:.3f}%] | Lower CI > 1% | BELOW RESEARCH MARGIN |",
'', 'Interior H48 là ticks [12,36), tức 12–35. Relative effect là chênh lệch trung bình chia trung bình của controller bên trái; CI bootstrap cùng numerator và denominator. Không báo p-value vì protocol dùng CI.',
'',f"Profile wins/ties/losses: P1 {p1['profile_wins']}/{p1['profile_ties']}/{p1['profile_losses']}; P2 {p2['profile_wins']}/{p2['profile_ties']}/{p2['profile_losses']}; P3 {p3['profile_wins']}/{p3['profile_ties']}/{p3['profile_losses']}.",
'', 'P2 chưa đủ bằng chứng planner tốt hơn rule trên 1%, nhưng cũng chưa chứng minh không có lợi ích: CI chứa 0 và vượt 1%. P3 có upper CI 0,527%, dưới margin 1%; đây là đánh giá theo ngưỡng nghiên cứu ở N4/K2, không phải chứng minh hai allocator tương đương hoặc ngưỡng kinh tế nhà máy.',
'','## Vì sao kết quả ngắn hạn dễ gây hiểu nhầm','',
'Tại skill_mask/H12, mọi controller chạy tiếp 12 ticks bằng cùng deadline_matching, giữ nguyên state và công việc đang sửa. Tổng chi phí gồm cả phần sau cutoff:',
'','| Controller | Chi phí trước H12 | 12 ticks sau H12 | Tổng 24 ticks | Pending tại H12 | Tổng máy hỏng tại H12 |',
'|---|---:|---:|---:|---:|---:|']
r=s['means']['skill_mask:12']
for name in ('best_rule','fixed_zero','fixed_tail','flexible_tail'):
    v=r[name];lines.append(f"| {names[name]} | {v['base_objective']:.3f} | {v['tail_objective']:.3f} | {v['extended_objective']:.3f} | {v['cutoff_pending']:.3f} | {v['cutoff_failed']:.3f} |")
lines += ['','Fixed_zero có vẻ rẻ hơn rule **2,70%** khi chỉ nhìn 12 ticks đầu; khi tính cả phần chạy tiếp, nó lại tốn hơn rule **2,85%**. Fixed_tail tăng chi phí trước cutoff để giảm phần sau; tổng chi phí thấp hơn fixed_zero **3,31%**. Các tỷ lệ tổng chi phí này là secondary descriptive means, không có confirmatory CI bổ sung.',
'','Pending giảm từ **0,988 xuống 0,600 máy**, tức **39,33%**. Nhưng service-start không phải repair completion: CM đang sửa tăng từ 0,096 lên 0,334 máy. Về kế toán trạng thái, khoảng 61,3% chênh lệch pending tương ứng với tăng số máy đang sửa; tổng máy hỏng chỉ giảm từ 1,084 xuống 0,934, tức 13,87%. Không diễn giải mức giảm pending là tất cả đã sửa xong.',
'','Fixed_tail/H12 vẫn còn pending cao hơn rule: 0,600 so với 0,439. Gate service đã khóa kiểm tra interior H48; không chuyển gate sang H12 sau khi xem số liệu, nhưng giữ chênh lệch này trong báo cáo.',
'', '## Interior H48 và service guard','',
'| Controller | Cost / machine-tick | Mean pending | Mean total failed | Overdue ticks / physical tick |',
'|---|---:|---:|---:|---:|']
r48=s['means']['skill_mask:48']
for name in ('best_rule','fixed_zero','fixed_tail','flexible_tail'):
    v=r48[name];lines.append(f"| {names[name]} | {v['interior_cost_per_machine_tick']:.6f} | {v['interior_pending_mean']:.6f} | {v['interior_failed_mean']:.6f} | {v['interior_overdue_per_tick']:.6f} |")
lines += ['','Service guard descriptive đều PASS. Tuy nhiên pending của fixed_tail cao hơn rule **0,098389 máy**, chỉ cách margin +0,1 khoảng **0,001611 máy**. Guard pass sát ngưỡng, không chứng minh dịch vụ tương đương. Tổng failed tăng 0,058024 máy; overdue tăng khoảng 0,002116/tick, dưới margin 0,25/12.',
'','**Timing-learning screen: NOT_ESTABLISHED**, vì P2 không vượt lower-CI margin 1%. P1 pass và service guard pass không thay thế yêu cầu P2.',
'','Ở H48, mean extended cost của fixed_tail = 983,475 và flexible_tail = 985,100. Vì vậy mức lợi rất nhỏ ở interior của flexible không đồng nghĩa nó thắng ở mọi objective. Tổng H48+L12 cũng là finite-horizon outcome, không phải infinite-horizon optimum.',
'','## Heterogeneity và toàn bộ mean costs','',
'Không gộp conditions để che khác biệt. Ở H48, headroom fixed_tail so với rule:',
'','| Condition | Relative interior improvement | CI | Nhãn |',
'|---|---|---|---|']
for co in ('homogeneous','specialized','skill_mask'):
    c=s['contrasts'][f'{co}:48:interior_headroom'];lines.append(f"| {co} | {100*c['relative_improvement']:.3f}% | [{100*c['relative_confidence_interval'][0]:.3f}%, {100*c['relative_confidence_interval'][1]:.3f}%] | {'Primary, 98,33%' if co=='skill_mask' else 'Exploratory, 95%'} |")
lines += ['','Homogeneous còn lợi ích khoảng 2,85%; specialized khoảng 1,51%, nhưng lower CI dưới 1%. Do đó không kết luận planner hết giá trị trong mọi condition. Conditions thay đổi cả topology lẫn effective capacity; khác biệt giữa conditions không phải pure causal effect của skill mask.',
'','Negative control homogeneous: fixed_tail và flexible_tail có mean bằng nhau ở cả ba horizons, phù hợp worker giống nhau. Primary assignment conclusion chỉ áp dụng setting hiện tại, chưa áp dụng N/K lớn hơn.',
'','| Condition/H | Best rule | Fixed zero: extended cost | Fixed tail: extended cost | Flexible tail: extended cost |',
'|---|---|---:|---:|---:|']
for co in ('homogeneous','specialized','skill_mask'):
    for h in (12,24,48):
        cell=f'{co}:{h}';v=s['means'][cell];lines.append(f"| {cell} | {chosen[cell]['best_rule']} | {v['fixed_zero']['extended_objective']:.3f} | {v['fixed_tail']['extended_objective']:.3f} | {v['flexible_tail']['extended_objective']:.3f} |")
lines += ['','Không coi chênh lệch với experiment v1 trước là một causal comparison: panel profiles/shocks mới và objective mới khác nhau. Controlled comparison trong run hiện tại là bằng chứng trực tiếp cho boundary treatment.',
'','## Thời gian chạy','',
f"Thực tế có **{lat['forecast_samples']/1e9:.3f} tỷ scenario-ticks** trong forecast. Thời gian đo bên trong action selection là {lat['measured_action_selection_seconds']/3600:.3f} giờ, tương đương **{100*lat['measured_action_selection_seconds']/lat['wall_seconds']:.2f}%** wall time. Phần nặng chủ yếu nằm ở planner, không phải 0,813 GB CSV đầu ra.",
'','| skill_mask/H48 | Mean action latency (ms) | P95 (ms) |',
'|---|---:|---:|']
for name in ('best_rule','fixed_zero','flexible_zero','fixed_tail','flexible_tail'):
    v=lat['cells'][f'skill_mask:48:{name}:base'];lines.append(f"| {names[name]} | {v['mean_ms']:.3f} | {v['p95_ms']:.3f} |")
lines += ['','Latency là đo trong closed-loop run với lịch cache/code hiện tại; không phải benchmark GPU, không equal-compute và không chứng minh khả năng real-time ở nhà máy.',
f"Có {int(lat['forecast_forced_decisions']):,} / 516.096 planner decisions (**{100*lat['forecast_forced_decisions']/516096:.2f}%**) chỉ có một feasible candidate nhưng vẫn mô phỏng tương lai. Đây là cơ hội tối ưu runtime cần thiết kế/kiểm tra riêng; không tự sửa code hoặc hứa mức speedup từ số đếm này.",
'','## Censoring và carryover','',
'Observed waits có censoring tại H và H+L. Served-only mean/p95 là conditional metrics; không dùng chúng thay cho toàn bộ request history. Pending requests tại H được nối bằng machine/request ID qua tail, kể cả yêu cầu mới đến đúng boundary có wait=0.',
'','| skill_mask/H12 | Cutoff requests / episode | Censored tại H | Carryover bắt đầu sửa trong tail | Còn censored sau tail |',
'|---|---:|---:|---:|---:|']
for name in ('best_rule','fixed_zero','fixed_tail','flexible_tail'):
    v=s['means']['skill_mask:12'][name];lines.append(f"| {names[name]} | {mean('skill_mask',12,name,'cutoff_requests'):.3f} | {v['cutoff_censored_requests']:.3f} | {v['carryover_started_in_tail']:.3f} | {v['carryover_censored_after_tail']:.3f} |")
lines += ['','## Fallacy scan — 11/11 checked','',
'| Loại | Đánh giá trong phạm vi dữ liệu |',
'|---|---|',
'| Simpson | Giữ riêng condition/H; headroom khác nhau đáng kể, không suy từ pooled mean. |',
'| Ecological | Bootstrap ở profile; không suy sang từng máy, nhà máy hoặc mọi N/K. |',
'| Berkson / sampling selection | Profiles bị lọc để không trùng run cũ; inference giới hạn distribution và selection protocol này. |',
'| Collider | Không lọc episode theo failure/service outcome để tính effects. Served-only wait chỉ dùng secondary conditional metric. |',
'| Base-rate neglect | Censoring/served metrics đi kèm request counts và state counts; không bỏ denominators. |',
'| Regression to mean | Best rule chọn trên dev mới, test không chọn từ extremes; không dùng before/after cross-run để quy causal gain. |',
'| Survivorship | Đủ toàn bộ grid, không dropout episodes; giữ cả censored requests và carryover. |',
'| Look-elsewhere | Báo đủ 3 primary với correction; 24 còn lại exploratory, không nâng secondary positive result thành primary. |',
'| Forking paths | Plan trùng commit trước full run, seeds/budget/windows/margins không thay; explanatory secondary ratios ghi descriptive. |',
'| Correlation vs causation | Policy treatment kiểm tra trong simulator với paired tapes; không quy ra causal effect nhà máy hoặc topology. |',
'| Reverse causality | Controller và terminal treatment đặt trước simulation; không suy reverse cause từ post-action backlog. |',
'','## Diễn giải và giới hạn','',
'Theo gate đã khóa, kết quả chưa đủ để mở timing-learning follow-up. Assignment phức tạp cũng chưa vượt investment margin ở primary N4/K2. Điều được hỗ trợ rõ hơn là xử lý đúng boundary và dùng rule baseline mạnh.',
'','Không đồng nhất failure-to-start waiting với thời gian máy hỏng tới khi sửa xong. Tail=12 dưới một rule chung là finite policy-evaluation terminal value; không phải terminal value tối ưu. Interior window vẫn có thể chịu ảnh hưởng của quyết định trước đó, không phải stationary performance. Bootstrap uncertainty conditional trên 8 shocks và approximate planner S128; chưa tích phân mọi nguồn Monte Carlo noise.',
'','Giữ riêng evidence, inference và research screen. Không có deployment claim, không có kết luận mọi RL đều vô ích, và không có thay đổi code/launch/transfer trong lượt phân tích này.',
'','## Artifacts phân tích','',
'- `analyze.py`: checksum/source audit, independent bootstrap, full state/cost/request replay và latency audit.',
'- `integrity_audit.json`, `physical_replay_audit.json`, `independent_statistics.json`, `latency_analysis.json`.',
'- `plot.py`, `boundary_summary.png`, `boundary_summary.pdf`: explanatory scientific figure.',
'- `report.py`: tạo báo cáo này từ số đã audit.',
'',f'![Boundary summary](<{OUT}/boundary_summary.png>)','']
(OUT/'analysis_vi.md').write_text('\n'.join(lines))
print(OUT/'analysis_vi.md')
