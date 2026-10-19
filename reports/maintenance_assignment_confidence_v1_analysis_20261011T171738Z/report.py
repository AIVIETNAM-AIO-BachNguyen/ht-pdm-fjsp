"""Finish clustered diagnostic checks and build the Vietnamese report."""
import csv,json,subprocess,sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'analysis.json').read_text());RUN=Path(d['run'])
sys.path.insert(0,'/private/tmp/marl-maintenance-assignment-confidence-v1/src')
from ht_pdm_fjsp.maintenance_dispatch import role_seed
cfg=json.loads((RUN/'resolved_config.json').read_text())['settings']
rows=list(csv.DictReader((RUN/'root_states.csv').open()))
rng=np.random.default_rng(role_seed(cfg['analysis_seed'],'maintenance_assignment_confidence_v1:crossed_bootstrap'))
rd=rng.integers(3,size=(20000,3));pd=rng.integers(8,size=(20000,8))
wr=np.stack([np.bincount(x,minlength=3) for x in rd])/3
wp=np.stack([np.bincount(x,minlength=8) for x in pd])/8
for key,r in d['root_diagnostics']['root_contrasts'].items():
    cond,origin,name=key.split(':'); selected=[x for x in rows if x['condition']==cond and x['origin']==origin]
    num=np.zeros((3,8));den=np.zeros((3,8))
    for x in selected:
        if not int(x['exposure']):continue
        i=cfg['model_seeds'].index(int(x['model_seed']));j=cfg['test_cohorts'][:8].index(int(x['cohort']))
        num[i,j]+=float(x[f'{name}_gain']);den[i,j]+=1
    sn=np.einsum('bi,ij,bj->b',wr,num,wp);sd=np.einsum('bi,ij,bj->b',wr,den,wp)
    finite=sd>0;ci=np.quantile(sn[finite]/sd[finite],[.025,.975])
    assert np.allclose(ci,r['confidence_interval'],atol=1e-10)
    assert int(den.sum())==r['exposures']
    changes=[x for x in selected if int(x[f'{name}_changed'])]
    assert len(changes)==r['overrides'] and sum(float(x[f'{name}_gain'])<0 for x in changes)==r['harmful_overrides']
d['audit']['independent_root_cluster_bootstrap']=16
bound=[]
for p in sorted((RUN/'source_snapshot').glob('*.py')):
    if 'assignment_confidence' in p.name or p.name=='maintenance_assignment_value_forks.py':
        b=subprocess.check_output(['git','show',f"{d['commit']}:src/ht_pdm_fjsp/{p.name}"],cwd=RUN.parents[1])
        assert b==p.read_bytes();bound.append(p.name)
d['audit']['git_commit_source_binding']=bound
(OUT/'analysis.json').write_text(json.dumps(d,indent=2)+'\n')

lines=['## Material Passport','', '- Origin Skill: experiment-agent', '- Origin Mode: validate', '- Origin Date: 2026-10-11T17:17:38Z',
       '- Verification Status: ANALYZED','- Version Label: validation_v1','', '# Phân tích maintenance_assignment_confidence_v1','',
       '**Kết luận:** chưa có bằng chứng thuật toán mới giảm chi phí so với PPO. Hai bản gated trở thành PPO vì không thực hiện lần đổi phân công nào. Dữ liệu probe tăng độ phủ nhưng chưa chuyển thành cải thiện trên trajectory thực.','',
       f"Run: `{RUN}`. Commit `{d['commit']}`. Hoàn thành trong {d['elapsed_seconds']/60:.2f} phút; CPU. 3 PPO nguồn × 24 profile test × 4 shock × 2 điều kiện × 5 controller = 2.880 episode. Mỗi điều kiện/controller có 288 episode. Objective là tổng chi phí 48 bước + 12 bước tail; thấp hơn tốt hơn.",'',
       '## Kết quả closed-loop','', '| Controller | Dense: chi phí | Dense: giảm so PPO | Skill mask: chi phí | Skill mask: giảm so PPO | Đổi assignment dense / skill |','|---|---:|---:|---:|---:|---:|']
overrides={'frozen_ppo':'0 / 0','natural_ungated':'131 / 48','natural_gated':'0 / 0','broad_ungated':'123 / 41','broad_gated':'0 / 0'}
for name in d['means']['dense']:
    a=d['means']['dense'][name]['extended_objective'];b=d['means']['skill_mask'][name]['extended_objective']
    gains=[100*(d['means'][c]['frozen_ppo']['extended_objective']-v)/d['means'][c]['frozen_ppo']['extended_objective'] for c,v in [('dense',a),('skill_mask',b)]]
    lines.append(f"| {name} | {a:.3f} | {gains[0]:+.3f}% | {b:.3f} | {gains[1]:+.3f}% | {overrides[name]} |")
lines+=['','`natural`: chỉ dữ liệu trạng thái PPO gặp tự nhiên. `broad`: một nửa lượt lấy mẫu từ natural, một nửa từ probe tổng hợp ở thời điểm muộn. `gated`: chỉ đổi assignment khi lợi ích dự đoán trừ một SD ensemble vượt margin hiệu chỉnh. PPO chọn tập máy PM và mọi cạnh CM tại từng can thiệp; head chỉ đổi technician. Trajectory sau đó có thể đổi cả lịch PM/CM.','',
        '## Bốn so sánh chính','', '20.000 crossed bootstrap, lấy mẫu lại source và profile; 4 shock được lấy trung bình trong mỗi ô source/profile. CI 98,75% cho bốn so sánh chính, Bonferroni xấp xỉ; ngưỡng lợi ích thực dụng 1%. Chỉ 3 source nên bootstrap không khắc phục được hạn chế số replica.','',
        '| So sánh | Giảm chi phí | CI 98,75% |','|---|---:|---:|']
for key in ['dense:broad_gated_vs_frozen_ppo','dense:broad_gated_vs_broad_ungated','skill_mask:broad_gated_vs_frozen_ppo','skill_mask:broad_gated_vs_broad_ungated']:
    r=d['contrasts'][key];lines.append(f"| {key} | {r['mean_gain']*100:+.3f}% | [{r['ci'][0]*100:+.3f}%; {r['ci'][1]*100:+.3f}%] |")
lines+=['','Không so sánh nào chứng minh lợi ích vượt 1%. CI [0;0] của gated/PPO xuất phát từ cùng hành vi trong mẫu này, không phải bằng chứng mọi môi trường đều tương đương hoặc mô hình đã học cách tránh riêng quyết định xấu. CI broad_gated/broad_ungated vẫn chứa 0 ở cả hai điều kiện; dấu âm ở skill không chứng minh bị hại.','',
        '## So sánh bổ sung','', '| So sánh | Giảm chi phí | CI 95% |','|---|---:|---:|']
for key,r in d['contrasts'].items():
    if key.endswith('posthoc') or 'broad_ungated_vs_natural_ungated' in key:
        lines.append(f"| {key} | {r['mean_gain']*100:+.3f}% | [{r['ci'][0]*100:+.3f}%; {r['ci'][1]*100:+.3f}%] |")
lines+=['','Các so sánh với PPO được tính bổ sung sau khi mở test, không phải tiêu chí chính đã khóa. Mọi CI trên đều chứa 0. Broad/natural cũng chưa chứng minh lợi ích; dense broad kém natural ở trung bình của cả ba source. Không so trực tiếp chi phí tuyệt đối với experiment cũ vì panel test mới.','',
        '## Vì sao gate không hoạt động?','',
        'Dense: mỗi source/arm có 34 so sánh hiệu chỉnh. Margin tương đương khoảng 52,49–76,53 đơn vị chi phí (giá trị lưu được chia 20). Lợi ích dự đoán sau trừ SD lớn nhất trên các PM-choice test chỉ khoảng 8,74–16,15 với broad, hoặc 11,17–13,79 với natural. Không có lựa chọn nào vượt ngưỡng. Mỗi gated controller từ chối toàn bộ 566 cơ hội dense.','',
        'Skill mask: chỉ có 8, 7, 3 so sánh hiệu chỉnh theo ba source; thấp hơn mức tối thiểu 10. Gate bị vô hiệu hóa theo thiết kế, nên từ chối toàn bộ 149 cơ hội. Đây là thiếu dữ liệu hiệu chỉnh cho assignment mà PPO thực sự chọn, dù dữ liệu có nhiều nhãn cho các tập máy khác.','',
        'Margin cộng độ bi quan từ cả SD ensemble và 1,96 MCSE của paired forecast. Đây là guard thực nghiệm, không phải bảo đảm an toàn hay xác suất tin cậy 90%. Zero override đồng thời loại bỏ lựa chọn có lợi và có hại; không đánh giá được chất lượng chọn lọc của guard.','',
        '## Dữ liệu đa giai đoạn đã làm được gì?','',
        'Natural train có 2.886 nhãn pair, 2.496 (86,49%) ở tick 0. Probe thêm 8.091 nhãn ở tick 6/18/30/42. Tỷ lệ lượt huấn luyện ở tick >0 tăng từ 12,07% (natural) lên 87,49% (broad), với cùng 80 epoch × 1.024 draws/head. Tuy nhiên probe là trạng thái tổng hợp sau prefix thật, không phải phân phối PPO gặp tự nhiên.','',
        '| MSE ensemble trên nhãn pair | Natural head | Broad head |','|---|---:|---:|']
for cond in ['dense','skill_mask']:
    for split in ['development','calibration']:
        for origin in ['natural','probe']:
            vals=[]
            for arm in ['natural','broad']:
                rs=[r for r in d['prediction_metrics'] if (r['condition'],r['split'],r['origin'],r['arm'])==(cond,split,origin,arm)]
                vals.append(sum(r['mse']*r['labels'] for r in rs)/sum(r['labels'] for r in rs))
            lines.append(f'| {cond}, {split}, {origin} | {vals[0]:.4f} | {vals[1]:.4f} |')
lines+=['','MSE theo đơn vị target chuẩn hóa, tính từ ensemble trung bình; mô tả theo số nhãn, không coi các nhãn cùng state là mẫu độc lập. Broad chỉ cải thiện nhẹ dev probe; kém hơn trên dev natural và mọi ô calibration. Độ phủ tăng chưa đủ giúp dự đoán tốt trên trạng thái tự nhiên. Đây là dấu hiệu cần xem lại phân phối dữ liệu/độ nhiễu target, chưa xác định nguyên nhân duy nhất.','',
        '## Chẩn đoán một can thiệp','', '| Điều kiện / origin | Cơ hội | Natural: override / gây hại | Broad: override / gây hại | Natural / Broad: gain trung bình mỗi cơ hội |','|---|---:|---:|---:|---:|']
for cond in ['dense','skill_mask']:
    for origin in ['natural','probe']:
        a=d['root_diagnostics']['root_contrasts'][f'{cond}:{origin}:natural_ungated'];b=d['root_diagnostics']['root_contrasts'][f'{cond}:{origin}:broad_ungated']
        lines.append(f"| {cond} / {origin} | {a['exposures']} | {a['overrides']} / {a['harmful_overrides']} | {b['overrides']} / {b['harmful_overrides']} | {a['mean_gain']:+.3f} / {b['mean_gain']:+.3f} |")
lines+=['','384 root states; mỗi condition/origin 96 states, nhưng chỉ một phần có assignment thay thế. Tất cả CI 95% gain root đều chứa 0. Dense probe: broad gây hại lớn nhất 52,25 đơn vị chi phí; dense natural: cả hai có một override gây hại 39,64. Skill natural chỉ 8 cơ hội; không đủ để kết luận ổn định. Root gain là một lần đổi assignment rồi tiếp tục PPO với 64 paired futures, không phải phần trăm cải thiện cả episode; reference độc lập select/evaluate không phải oracle.','',
        '## Lưu ý đo tốc độ','',
        'Phát hiện inefficiency tại `source_snapshot/maintenance_assignment_confidence_policy.py:68`: near-tie kiểm tra cả candidate gốc (gain luôn 0). Với ungated, threshold=1e-5 và tolerance near-tie=2e-5 nên luôn kích hoạt scalar recompute. Gated skill với margin null cũng vậy. Latency hiện tại khoảng 1,61–1,63 ms so với PPO 0,217–0,219 ms; dense gated khoảng 0,482–0,484 ms. Đây là latency amortized batch 32 và bị ảnh hưởng bởi nhánh fallback thừa; không dùng để kết luận chi phí inference bản tối ưu. Không thấy lỗi này giải thích zero override: gate logic vẫn đúng. Chưa sửa code trong yêu cầu phân tích này.','',
        '## Hướng thử tiếp theo (đề xuất, chưa triển khai)','',
        'Ưu tiên thu thập trạng thái tự nhiên thực sự có ít nhất hai assignment cho đúng tập PM mà PPO chọn, ở từng condition/source và nhiều profile. Bổ sung calibration đủ mẫu ở skill; đánh giá độ phủ, MCSE và tỷ lệ override trên development trước khi mở test mới. Với cùng budget, dành thêm paired futures cho nhãn gần ranh giới quyết định để kiểm tra xem MC noise có làm margin quá lớn không. Nếu đổi gate/quantile, chỉ lựa chọn trên development và calibration mới, khóa trước một panel test mới; không hạ ngưỡng để khớp test vừa mở. Chưa có cơ sở tăng độ phức tạp kiến trúc từ kết quả này.','',
        '## Kiểm tra và giới hạn tái lập','',
        'Raw run giữ nguyên. Kiểm tra checksum toàn bộ 2.640 artifact; source contract 957 file và 3 PPO; code experiment liên quan khớp commit Git. Tính lại đủ thống kê episode, 16 bootstrap chẩn đoán root, 13.685 target/MCSE từ forecast, lựa chọn 18 head theo dev MSE và margin calibration. Kiểm tra 1.440 dòng sampling theo budget; không chạy lại optimizer.','',
        'Replay vật lý: 25.344 transition nguồn + 4.224 prefix probe + 7.200 transition test đã log. Xác nhận 1.152 gated episode có mọi base/tail/extended physical metric trùng PPO, và 2.880 quyết định gated đã log trùng action/state/following của PPO. Replay scalar forecast phân tầng 60 file, 120 future paths, 6.408 transition khớp cost, components, cutoff/terminal state; không tái chạy toàn bộ 30.378.720 forecast transition hay training. Vì vậy passport vẫn ANALYZED, không tuyên bố full run VERIFIED.','',
        'Script: `analyze.py`, `report.py`; dữ liệu dẫn xuất: `analysis.json`, `prediction_metrics.csv`, `gate_diagnostics.csv`; log: `validation.log`. Python cần numpy/torch/tqdm trong worktree venv.','',
        '## Fallacy scan — 11/11 đã kiểm tra','',
        '| Mẫu sai suy luận | Kiểm tra / giới hạn |','|---|---|',
        "| Simpson’s paradox | Tách condition/source; dense và skill broad/PPO trái dấu nên không gộp thành claim chung. |",
        '| Ecological fallacy | Không suy từ mean episode sang mọi state/assignment. |',
        '| Berkson’s paradox | Root diagnostic có điều kiện trên cơ hội; báo denominator và tách probe/natural. |',
        '| Collider bias | Kết quả chính dùng toàn bộ grid; không điều kiện hóa vào override để ước lượng gain episode. |',
        '| Base rate neglect | Báo 566/149 cơ hội, zero overrides và 34/8/7/3 calibration samples. |',
        '| Regression to the mean | Không chọn test profile theo kết quả cực đoan; checkpoint chọn dev. |',
        '| Survivorship bias | Grid đủ 2.880 episode, không loại thất bại/xấu; head đủ 18. |',
        '| Look-elsewhere effect | Báo đủ 4 primary; so sánh bổ sung là exploratory/posthoc, không claim significant. |',
        '| Garden of forking paths | Code/protocol đã khóa trước run; không retune ngưỡng sau test; đề xuất cần test mới. |',
        '| Correlation / causation | Can thiệp ghép physical tapes trong simulator; không suy rộng ra hệ thống thật hoặc quy nguyên nhân duy nhất cho dữ liệu probe. |',
        '| Reverse causality | Train/dev/calibration trước test; không dùng test gain chọn checkpoint/gate; định hướng dữ liệu không đảo thời gian. |','']
(OUT/'report_vi.md').write_text('\n'.join(lines))
print('PASS: 16 root bootstraps; source binding:',len(bound),'files; report saved',OUT/'report_vi.md')
