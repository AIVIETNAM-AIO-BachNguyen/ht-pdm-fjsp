"""Independent root bootstrap, commit/seed binding, and Vietnamese report."""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'analysis.json').read_text());RUN=Path(d['run'])
sys.path.insert(0,'/private/tmp/marl-maintenance-assignment-confidence-v1/src')
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp import maintenance_assignment_opportunity_benchmark as b
cfg=json.loads((RUN/'resolved_config.json').read_text())['settings']
rows=list(csv.DictReader((RUN/'root_states.csv').open()))
rng=np.random.default_rng(role_seed(cfg['analysis_seed'],f'{b.VERSION}:crossed_bootstrap'))
rd=rng.integers(3,size=(20000,3));pd=rng.integers(8,size=(20000,8))
wr=np.stack([np.bincount(x,minlength=3) for x in rd])/3
wp=np.stack([np.bincount(x,minlength=8) for x in pd])/8
for key,r in d['root_diagnostics']['root_contrasts'].items():
    cond,origin,name=key.split(':');selected=[x for x in rows if x['condition']==cond and x['origin']==origin]
    num=np.zeros((3,8));den=np.zeros((3,8))
    for x in selected:
        if not int(x['exposure']):continue
        i=cfg['model_seeds'].index(int(x['model_seed']));j=cfg['test_cohorts'][:8].index(int(x['cohort']))
        num[i,j]+=float(x[f'{name}_gain']);den[i,j]+=1
    sn=np.einsum('bi,ij,bj->b',wr,num,wp);sd=np.einsum('bi,ij,bj->b',wr,den,wp);valid=sd>0
    ci=np.quantile(sn[valid]/sd[valid],[.025,.975])
    assert np.allclose(ci,r['confidence_interval'],atol=1e-10)
    assert int(den.sum())==r['exposures']
    changes=[x for x in selected if int(x[f'{name}_changed'])]
    assert len(changes)==r['overrides'] and sum(float(x[f'{name}_gain'])<0 for x in changes)==r['harmful_overrides']
d['audit']['independent_root_cluster_bootstrap']=12
bound=[]
for p in sorted((RUN/'source_snapshot').glob('*.py')):
    if 'assignment_opportunity' in p.name or p.name=='maintenance_assignment_value_forks.py':
        raw=subprocess.check_output(['git','show',f"{d['commit']}:src/ht_pdm_fjsp/{p.name}"],cwd=RUN.parents[1])
        assert raw==p.read_bytes();bound.append(p.name)
d['audit']['git_commit_source_binding']=bound
manifest=json.loads((RUN/'manifest.json').read_text())
assert b.settings('full')==cfg
assert b.seed_audit(cfg)==manifest['seed_audit']
_, records, audit=b.metadata()
assert json.loads(json.dumps(records['full']))==json.loads((RUN/'resolved_config.json').read_text())['physical_profiles']
assert audit==manifest['physical_split_audit']
d['audit']['independent_seed_and_physical_split_reconstruction']='PASS'

# Residual decomposition at each exact calibrated higher quantile rank.
calrows=list(csv.DictReader((RUN/'calibration_predictions.csv').open()));components=[]
for source in cfg['model_seeds']:
    for arm in b.ARMS:
        for cond in cfg['conditions']:
            rs=[r for r in calrows if int(r['source_seed'])==source and r['arm']==arm and r['condition']==cond]
            errors=np.array([float(r['empirical_overoptimism']) for r in rs])
            ix=np.argsort(errors,kind='stable')[int(np.ceil(.9*(len(rs)-1)))];r=rs[ix]
            components.append(dict(source=source,arm=arm,condition=cond,samples=len(rs),
                margin=float(max(0,errors[ix])),predicted_minus_observed=float(r['predicted_gain'])-float(r['observed_gain']),
                minus_sd=-float(r['ensemble_sd']),mc_penalty=1.96*float(r['mc_se'])))
            margin=d['calibration']['parameters'][str(source)][arm][cond]['margin']
            assert np.isclose(max(0,errors[ix]),margin,atol=2e-6)
d['calibration_quantile_components']=components
(OUT/'analysis.json').write_text(json.dumps(d,indent=2)+'\n')

def percent(x):return f'{100*x:+.3f}%'
def interval(ci):return f'[{percent(ci[0])}; {percent(ci[1])}]'
def weighted(cond,split,arm,key):
    rs=[r for r in d['prediction_metrics'] if (r['condition'],r['split'],r['arm'])==(cond,split,arm)]
    return sum(r[key]*r['labels'] for r in rs)/sum(r['labels'] for r in rs)
means=d['means']
lines=['## Material Passport','', '- Origin Skill: experiment-agent', '- Origin Mode: validate',
    '- Origin Date: 2026-10-12T02:32:33Z','- Verification Status: ANALYZED','- Version Label: validation_v1','',
    '# Phân tích maintenance_assignment_opportunity_v1','',
    '**Kết luận:** calibration đã đủ mẫu và 128 futures làm nhãn ít nhiễu hơn, nhưng chưa có bằng chứng cải thiện chi phí đáng tin cậy. Opportunity-low ungated có trung bình tốt nhất ở dense; opportunity-high không tốt hơn low. High-gated vẫn giữ PPO trên toàn bộ test.','',
    f"Run `{RUN}`; commit `{d['commit']}`; code sạch. COMPLETED/PASS, {d['elapsed_seconds']/60:.2f} phút CPU, 5.190 artifact (300,17 MB). 3 source ×24 profile ×4 shock ×2 điều kiện ×7 controller =4.032 episode. Mỗi condition/controller có288 episode. Chi phí là H48 +tail12, thấp hơn tốt hơn.",'',
    '## Chi phí closed-loop','',
    '| Controller | Dense: chi phí | Giảm so PPO | Skill mask: chi phí | Giảm so PPO | Đổi assignment dense / skill |',
    '|---|---:|---:|---:|---:|---:|']
for name in b.NAMES:
    a,c=means['dense'][name],means['skill_mask'][name]
    ga=(means['dense']['frozen_ppo']['extended_objective']-a['extended_objective'])/means['dense']['frozen_ppo']['extended_objective']
    gc=(means['skill_mask']['frozen_ppo']['extended_objective']-c['extended_objective'])/means['skill_mask']['frozen_ppo']['extended_objective']
    lines.append(f"| {name} | {a['extended_objective']:.3f} | {percent(ga)} | {c['extended_objective']:.3f} | {percent(gc)} | {round(a['assignment_override_ticks']*288)} / {round(c['assignment_override_ticks']*288)} |")
lines+=['','`landmark_low`: cơ hội tự nhiên ở tick0/12/24/36,16 futures. `opportunity_low`: mọi tick có assignment thay thế,16 futures, cân bằng clock bin. `opportunity_high`: cùng trạng thái và cùng draws như low,128 futures. Mọi bản giữ đúng tập máy PM và cạnh CM của PPO tại can thiệp; trajectory tiếp theo có thể đổi lịch. Đây là ablation mới trên panel mới, không phải rerun chính xác natural/broad của confidence-v1.','',
    '## Các tiêu chí chính đã khóa','',
    '20.000 crossed source/profile bootstrap, trung bình4 shock trong mỗi ô;4 CI98,75% với Bonferroni xấp xỉ; mốc thực dụng1%.','',
    '| So sánh | Giảm chi phí | CI98,75% |','|---|---:|---:|']
for cond in cfg['conditions']:
    for baseline in ['frozen_ppo','opportunity_high_ungated']:
        key=f'{cond}:opportunity_high_gated_vs_{baseline}';r=d['contrasts'][key]
        lines.append(f"| {key} | {percent(r['mean_gain'])} | {interval(r['ci'])} |")
lines+=['','Cả4 tiêu chí đều BELOW_MARGIN theo quy tắc đã khóa (cận trên dưới1%); không có PILOT_GAIN_SIGNAL. High-gated/PPO có CI[0;0] vì cùng hành vi trong mẫu, không phải bảo đảm tương đương mọi môi trường. High-gated so high-ungated chứa0 ở cả hai điều kiện; chưa chứng minh guard có lợi hay gây hại.','',
    '## So sánh phụ và bổ sung','',
    '| So sánh | Giảm chi phí | CI95% |','|---|---:|---:|']
for key,r in d['contrasts'].items():
    if r['confidence']==.95:
        lines.append(f"| {key} | {percent(r['mean_gain'])} | {interval(r['ci'])} |")
lines+=['','Các khóa `_posthoc` là so sánh ungated/PPO bổ sung sau khi mở test; không thay tiêu chí chính. Dense opportunity-low/PPO:+0,699%,CI[-0,159%;+1,650%]; high:+0,627%,CI[-0,232%;+1,566%]. Low so landmark:+0,266%,CI[-0,095%;+0,774%]; high so low:−0,072%,CI[-0,277%;+0,127%]. Chưa xác lập H1/H2 về chi phí. Skill ungated có trung bình kém PPO khoảng0,19–0,20%, nhưng CI chứa0 nên chưa chứng minh hại. Các chênh lệch rất nhỏ với CI chạm0 là mô tả thăm dò, không gọi là significant.','',
    'Dense ungated giảm trung bình chi phí ở cả3 PPO nguồn; skill có1 nguồn tốt hơn và2 nguồn kém hơn. Không gộp dense/skill thành lợi ích chung. Chỉ3 source nên dấu cùng chiều ở dense vẫn không đủ làm CI hẹp; không so trực tiếp chi phí tuyệt đối với run cũ có panel khác.','',
    '## H1: lấy mẫu theo cơ hội thay đổi dữ liệu thế nào?','',
    '| Điều kiện / arm | Nhãn train | Tỷ lệ draws tick>=12 |','|---|---:|---:|']
for r in d['coverage_metrics']:
    lines.append(f"| {r['condition']} / {r['arm']} | {r['labels']} | {100*r['late_draw_fraction']:.2f}% |")
lines+=['','2251 cơ hội train (1779 dense,472 skill); landmark giữ1106 trong số này. Dữ liệu đều từ trajectory PPO thật, không có probe tổng hợp. Tỷ lệ draws muộn tăng từ10,04% dense/5,79% skill lên khoảng75%. Thiết kế đồng thời đổi tick coverage và cân bằng clock; không tách nhân quả riêng hai thành phần.','',
    '## H2: nhãn chính xác hơn có giúp policy không?','',
    '| Điều kiện | MCSE16: mean chuẩn hóa | MCSE128 | Mức giảm MCSE | Số nhãn đổi dấu |','|---|---:|---:|---:|---:|']
for r in d['noise_metrics']:
    lines.append(f"| {r['condition']} | {r['low_mc_se_mean']:.4f} | {r['high_mc_se_mean']:.4f} | {100*(1-r['mc_se_ratio']):.2f}% | {r['sign_disagreements']}/{r['labels']} ({100*r['sign_disagreements']/r['labels']:.2f}%) |")
lines+=['','128 so16 futures làm MCSE giảm khoảng64%, gần tỷ lệ1/sqrt(8). Low dùng đúng16 scenario đầu của cùng full tape ở mọi tick. High không phải ground truth: vẫn có MCSE. Các nhãn tương quan theo trajectory/profile, nên không coi2251 nhãn là2251 mẫu độc lập. Đây là kiểm tra precision của estimator, không phải chứng minh lợi ích policy.','',
    '| MSE ensemble trên cùng nhãn128 | Landmark low | Opportunity low | Opportunity high |','|---|---:|---:|---:|']
for cond in cfg['conditions']:
    for split in ['development','calibration']:
        vals=[weighted(cond,split,a,'mse') for a in b.ARMS]
        lines.append(f'| {cond} / {split} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |')
lines+=['','High có MSE hơi thấp hơn opportunity-low trên dev/calibration, rõ hơn ở skill. Tuy nhiên giảm sai số value không chuyển thành giảm chi phí test; trên48 root test, dense MSE low2,225/high2,476,skill low2,006/high1,971. Mẫu root nhỏ và chỉ64 futures nên không kết luận thứ hạng tổng quát. Không có bằng chứng từ ablation này để tiếp tục tăng futures chỉ nhằm cải thiện policy.','',
    '## H3: gate còn bị thiếu calibration không?','',
    '| Source | Dense samples / profiles / late | Skill samples / profiles / late | High margin dense / skill (cost units) |','|---|---:|---:|---:|']
for source in cfg['model_seeds']:
    rs=d['calibration']['parameters'][str(source)]['opportunity_high'];a,c=rs['dense'],rs['skill_mask']
    lines.append(f"| {source} | {a['samples']} / {a['physical_profiles']} / {a['late_samples']} | {c['samples']} / {c['physical_profiles']} / {c['late_samples']} | {20*a['margin']:.2f} / {20*c['margin']:.2f} |")
lines+=['','Tất cả18 gate (3 source ×3 arm ×2 condition) đều đủ40 samples/16 profiles/10 late và được bật. Skill có126/150/151 mẫu từ30–35 profile, không còn lý do insufficient_calibration của run trước. Nhiều mẫu cùng profile/shock vẫn tương quan.','',
    '| Source / condition | High: gain đã trừ SD lớn nhất trên test | Margin |','|---|---:|---:|']
for r in d['gate_diagnostics']:
    if r['controller']=='opportunity_high_gated':
        lines.append(f"| {r['source']} / {r['condition']} | {20*r['max_adjusted_gain']:.2f} | {20*r['margin']:.2f} |")
lines+=['','High-gated vẫn từ chối toàn bộ487 cơ hội dense và148 skill: adjusted gain không vượt margin. Landmark-gated cũng zero; opportunity-low-gated chỉ5 overrides dense (0 skill),giảm0,022% trung bình. Mọi diagnostic development guard đều zero overrides, nên tình trạng guard hầu như bất động đã hiện trên dev trước khi mở test; protocol vẫn chạy test cố định.','',
    'Margin là quantile overoptimism gồm sai số mô hình và1,96 MCSE sau trừ SD ensemble, không phải bằng chứng tin cậy/safety90%. Tại6 hàng calibration quyết định quantile của high, phần predicted-minus-observed là0,987–1,779 chuẩn hóa, lớn hơn phần MC penalty0,308–0,589 (mỗi đơn vị tương đương20 cost). Đây là mô tả tại các hàng quantile, không phải phân rã nhân quả toàn bộ model. Nó cho thấy margin không chỉ do Monte Carlo noise; giảm noise thêm vẫn để lại sai số dự đoán lợi ích. Đủ calibration không đồng nghĩa guard đủ khả năng chọn lọc. High-gated không giữ được lợi ích trung bình của ungated dense vì không thực hiện can thiệp nào. Không retune margin/quantile trên test này.','',
    '## Single-root diagnostics','',
    '| Điều kiện / ungated arm | Cơ hội | Overrides | Overrides có mean gain âm | Mean gain/cơ hội | CI95% |','|---|---:|---:|---:|---:|---:|']
for cond in cfg['conditions']:
    for arm in b.ARMS:
        r=d['root_diagnostics']['root_contrasts'][f'{cond}:natural:{arm}_ungated']
        lines.append(f"| {cond} / {arm} | {r['exposures']} | {r['overrides']} | {r['harmful_overrides']} | {r['mean_gain']:.3f} | [{r['confidence_interval'][0]:.3f};{r['confidence_interval'][1]:.3f}] |")
lines+=['','Root panel có33 cơ hội dense,15 skill từ first8 profiles/firstshock. Dense low10 overrides,2 mean harmful; high9,2. Harm lớn nhất28,859 cost. Skill cả3 ungated4 overrides,1 mean harmful (37,234 cost); net gain/cơ hội chỉ0,690 với CI rộng. Đây là một lần đổi assignment rồi PPO continuation, không phải phần trăm episode. Nhãn “harmful” dùng mean của64 futures và vẫn có MC uncertainty. Reference độc lập selection/evaluation không phải oracle.','',
    '## Tốc độ và compute','',
    'Full run dùng58.800.192 forecast transitions; shared high tape được tính một lần, low chỉ lấy prefix, nên đây là ablation độ chính xác nhãn, không phải ablation ngang chi phí simulator. Mọi learned controller có batch-amortized inference khoảng0,460ms so với PPO0,204–0,205ms (~2,25 lần); batch32, CPU, không phải latency realtime từng quyết định. Near-tie fallback của candidate gốc đã được loại trong code mới; không lấy khác biệt runtime giữa hai run để suy causal effect.','',
    '## Đề xuất từ kết quả này','',
    'Giữ opportunity-low ungated làm đối chứng nghiên cứu ở dense: nó có trung bình tốt nhất nhưng CI vẫn chứa0, chưa đủ để thay PPO. Chưa nên tăng futures nữa chỉ để kỳ vọng tăng reward. Hướng có ích hơn là kiểm tra lựa chọn quyết định và guard: dùng loss/selection phản ánh gain và downside khi định override, cùng hiệu chỉnh trên nhóm quyết định ứng viên. Đây là giả thuyết mới, không phải cải thiện đã chứng minh; lựa chọn trên dev/calibration mới, khóa trước một test panel mới. Skill cần đánh giá riêng vì hiện không có gain closed-loop dù MSE có cải thiện. Không triển khai experiment mới trong yêu cầu phân tích này.','',
    '## Verification và phạm vi','',
    'Checksum5190 artifacts khớp; đủ4032 episode/27 heads/17280 updates, config/seed split dựng lại khớp,8 module liên quan khớp commit. Khôi phục27 head, kiểm tra dev MSE/checkpoint selection và dựng lại18 margin/coverage criteria. Kiểm tra2160 draw rows;720 low/high member-epoch pairs có cùng draws và state table. Tính lại4737 high/low paired labels và48 root gains từ stored forecast. Tính độc lập18 registered intervals (bao gồm2 contrast được báo ở cả98,75% và95%) cùng6 posthoc intervals,12 root crossed bootstraps.','',
    'Replay vật lý168192 source transitions và10080 logged test transitions. Hai zero-override gated controller có1152 episode với mọi base/tail/extended physical metric trùng PPO;2880 logged gated state/action/following trùng PPO. Bounded scalar forecast replay30 files/60 paths/3194 transitions khớp costs, components và boundary states; không tái chạy58,8 triệu forecast transitions hoặc training. Passport giữ ANALYZED, không tuyên bố full training VERIFIED. Raw artifacts không sửa.','',
    'Derived files: `analysis.json`, `prediction_metrics.csv`, `gate_diagnostics.csv`, `validation.log`, `analyze.py`, `report.py`. Scripts dùng worktree venv đã pin và chỉ ghi report directory.','',
    '## Fallacy scan — 11/11','', '| Mẫu suy luận | Kiểm tra / giới hạn |','|---|---|',
    '| Simpson’s paradox | Dense/skill trái dấu ungated/PPO; tách điều kiện và3 nguồn. |',
    '| Ecological fallacy | Không suy mean episode thành mọi assignment; root/episode metrics tách riêng. |',
    '| Berkson’s paradox | Root/labels có điều kiện trên source opportunity; báo denominator và không dùng làm gain toàn episode. |',
    '| Collider bias | Primary cost dùng full grid; không điều kiện hóa theo override/harm để tính hiệu quả. |',
    '| Base rate neglect | Báo487/148 cơ hội test, sample/profile/late calibration,5 và0 overrides. |',
    '| Regression to mean | Không chọn test theo điểm cực đoan; checkpoint chọn dev; mọi state budget cố định. |',
    '| Survivorship bias | Grid4032 đủ, không loại kết quả xấu;27 heads có training/checkpoint. |',
    '| Look-elsewhere effect | Đủ4 primary,14 secondary,6 posthoc; không nâng mean tốt nhất thành significant. |',
    '| Garden of forking paths | Protocol commit trước run; không hạ gate, đổi seeds hoặc kéo dài collection sau test. |',
    '| Correlation != causation | Simulator intervention ghép tapes; H1 thay cả coverage và sampling, không quy riêng một thành phần/ngoại suy production. |',
    '| Reverse causality | Train/dev/calibration trước sealed test; test không chọn checkpoint/gate. |','']
(OUT/'report_vi.md').write_text('\n'.join(lines))
print('PASS: report saved',OUT/'report_vi.md')
