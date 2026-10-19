from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
P=Path(__file__).resolve().parent
D=json.loads((P/'independent_analysis.json').read_text());Q=json.loads((P/'request_audit.json').read_text())['groups'];T=json.loads((P/'training_audit.json').read_text());S=json.loads((P/'supplemental_audit.json').read_text());REPLAY=json.loads((P/'checkpoint_replay.json').read_text());LAT=json.loads((P/'latency_audit.json').read_text())
C='central_matching_ppo';L='cooperative_local_ppo';F='cooperative_full_ppo'
N={C:'Centralized RL',L:'MARL cục bộ',F:'MARL toàn cục','selected_rule':'Rule đã chọn','joint_rollout':'Joint rollout planner'}
def g(reg,a,c='skill_mask',h=24):return D['aggregate'][f'{reg}:{a}:{c}_H{h}']
def pct(x):return f'{100*x:.2f}%'
def ci(x):return f'[{x[0]:.4f}, {x[1]:.4f}]'
m=D['manifest'];start=datetime.fromisoformat(m['started']);end=datetime.fromisoformat(m['finished']);tz=timezone(timedelta(hours=8))
b=['# Phân tích training coverage v1\n', '## Material Passport\n', '- Origin Skill: academic-research-suite / experiment-agent\n- Origin Mode: validate\n- Origin Date: 2026-10-10 Asia/Taipei\n- Verification Status: ANALYZED\n- Version Label: maintenance_training_coverage_v1_analysis\n- Overall Confidence: CAUTION — n=10 replica; t-interval assumptions và horizon transfer có giới hạn\n',
'''## 1. Kết luận chính

**Coverage có ích cho skill-mask, nhưng chỉ một trong bốn phép kiểm định primary đạt toàn bộ tiêu chí.** MARL cục bộ giảm cost3,52% ở skill-mask H24 và đạt protocol; MARL toàn cục giảm mean3,81% nhưng CI98,75% chứa0. Cả hai chưa đạt cổng gap closure25% so với centralized RL. Sau coverage, cost vẫn cao hơn central khoảng12,16%/12,81%.

Đây là một can thiệp mixture ở fixed budget: thay40 điểm phần trăm specialized exposure bằng skill-mask. Nó cho bằng chứng về giá trị mixture cụ thể dưới thuật toán hiện tại, không chứng minh coverage là nguyên nhân duy nhất của gap, cũng không chứng minh MARL cần thiết. H24 còn nằm ngoài horizonH12 được huấn luyện. Không phát triển kiến trúc mới hoặc chạy experiment tiếp theo trong lượt phân tích này.
''', '## 2. Tình trạng run và kiểm chứng\n',
f"Source: `lab_results/maintenance_training_coverage_v1_full_cpu_20261009T044236Z`. Commit `{m['commit']}`; clean Git; Linux CPU một torch thread. Run từ **{start.astimezone(tz):%Y-%m-%d %H:%M:%S}** đến **{end.astimezone(tz):%Y-%m-%d %H:%M:%S}** Asia/Taipei, khoảng **20 giờ09 phút**. Raw artifacts1.391GB decimal (~1,295GiB),411 file có hash trong manifest.\n",
'''| Kiểm tra độc lập | Kết quả |
|---|---|
| SHA256 và kích thước411 artifact | PASS |
| Source snapshots khớp Git commit đã push | PASS |
| 60 model /29.491.200 training ticks /614.400 optimizer steps | Đúng protocol |
| Raw training episodes /progress rows | 2.457.600 /19.200, đủ320 update/model |
| Test episode grid | 317.440 tổ hợp đúng; không thiếu, trùng hoặc thừa |
| Evaluation physical intervals | Tổng horizon5.713.920; full tick traces không được ghi |
| Raw machine rows | 1.269.760, đối soát từng episode |
| Raw request rows | 858.570, đối soát từng episode và censoring |
| Raw development grid | 40.960, rule selection tính lại khớp |
| Common-state latency rows | 8.928 |
| Checkpoint files | 360, tất cả load được, finite; final copy khớp snapshot cuối |
| Khởi tạo và training configuration/shock sequence | Khớp cả sáu cell trong từng seed |
| Greedy replay trên common development states | 2.880 model-state pairs,0 action mismatch |

Training legacy mỗi model:32.768specialized+8.192nominal; covered:16.384specialized+8.192skill-mask+8.192nominal. Nominal configuration hashes và positions giữ nguyên giữa hai mixture; trong từng mixture các thuật toán có cùng configuration sequence. Complete-episode reward reconciliation error tối đa2,29e−6 trên log. Episode objective, base cost, failure/down/wait prices, per-machine sums và served/censored overdue cost đều tái tính khớp. Logged invalid assignments bằng0.

Tất cả dùng checkpoint cuối491.520ticks; không loại seed, chọn checkpoint tốt hơn hoặc thay stopping rule sau test. Khởi tạo đã tái sinh trên Mac khớp tensors. Replay chỉ trên48 common DEVELOPMENT states; không chạy lại training, toàn bộ test dynamics hoặc planner forecast trajectories. Do đó verification status ANALYZED; reproducibility verdict PARTIALLY_REPRODUCIBLE. Feasibility full test dựa vào committed runtime checks và logged audits, không gọi đây là independent full trajectory replay.

Fresh shocks185000..185019; replica seeds180000..180009; development shocks184000..184009. Familiar test profile labels163000..163031 đã mở ở các nghiên cứu trước. Không gọi chúng là unseen profiles hoặc sealed profile combinations. Chỉ shock panel mới được khóa trước full run. H12/H24 dùng common shock prefixes, nhưng đây không phải paired physical trajectories khi policy thay đổi theo remaining horizon.
''', '## 3. Bốn phép kiểm định primary — skill-mask H24\n',
'''Đơn vị suy luận là10 paired TRAIN REPLICA means trên cùng32profile×20shock, không phải6.400 episode/arm. Bốn test dùng two-sided98,75% paired-t CI (Bonferroni familywise0,05). Hiệu quả cần >=2% với within-policy coverage, hoặc>=25% với gap closure;>=8/10 negative differences; CI upper<0; waiting/pending guards và preservation cost specialized/nominal H12 cùng pass.

| Coverage contrast | Legacy cost | Covered cost | Giảm cost | Δcost CI98,75% | Seed cải thiện | Verdict |
|---|---:|---:|---:|---|---:|---|
''']
for a in (L,F):
 v=D['contrasts'][f'coverage:{a}:skill_mask_H24'];b.append(f"| {N[a]} | {v['legacy_cost']:.4f} | {v['covered_cost']:.4f} | {pct(v['priced_reduction'])} | {ci(v['ci'])} | {v['wins']}/10 | **{v['status']}** |\n")
b.append('''
| Gap closure contrast | Legacy gap | Covered gap | Phần gap thu hẹp | Difference-in-differences CI98,75% | Seed gap giảm | Verdict |
|---|---:|---:|---:|---|---:|---|
''')
for a in (L,F):
 v=D['contrasts'][f'gap_closure:{a}:skill_mask_H24'];b.append(f"| {N[a]} | {v['legacy_gap_to_central']:.4f} | {v['covered_gap_to_central']:.4f} | {pct(v['gap_fraction_closed'])} | {ci(v['ci'])} | {v['wins']}/10 | **{v['status']}** |\n")
b.append('''
MARL cục bộ gap-closure CI upper≈−0,00128 nằm sát0, nhưng mức thu hẹp22,95%<25% và chỉ7/10 seed, nên FAIL. Không nâng một numerical crossing sát0 thành kết luận mạnh. MARL toàn cục FAIL cả threshold25% lẫn CI. Các primary waiting/pending và specialized/nominal cost-preservation guards đều pass; điều đó không đảm bảo mọi condition hoặc mọi fairness metric được bảo toàn.

Centralized RL cũng giảm mean skill-mask H24 từ329,2613 xuống327,7205 (0,47%; exploratory95% CI delta[−2,8351,−0,2465],8/10seed). Gap closure không được tạo ra bởi central bị làm yếu ở primary này. Residual cooperative-minus-central cost CI95%: local[33,7332,45,9590], full[32,3945,51,5983], đều còn dương.

### Độ nhạy với seed và phương pháp suy luận

Full seed180005: skill-mask H24 covered393,5828 so với legacy365,7766, Δ+27,8063. Chín seed khác cải thiện. Cùng seed đó ở H12 Δ−1,2844; không loại seed này vì nó gây verdict thất bại. Đây là biểu hiện policy nhạy với seed/horizon, chưa xác định lỗi optimizer hoặc credit assignment. Tensors/loss/logs finite; finite logs không đảm bảo policy tốt.

Posthoc leave-one-replica-out mean ranges: local coverage[−15,0913,−11,6988], full[−19,3583,−12,4906]. Mean lợi ích không phụ thuộc hoàn toàn vào một seed, nhưng uncertainty/gates vẫn dùng đầy đủ10seed.

Supplemental exact sign-flip (symmetry under null required), Bonferroni4 adjustedp: local coverage0,05469; full coverage0,14063; local gap closure0,09375; full gap closure0,21094. Đây là kiểm tra độ nhạy exploratory, không thay phương pháp đã khóa để đổi PASS/FAIL. Local PASS dưới paired-t preregistered, nhưng không vượt0,05 trong kiểm tra sign-flip bổ sung; vì vậy mức chắc chắn cần diễn đạt thận trọng và cần replication trước claim rộng. n10 không đủ để chứng minh robustness với mọi seed distribution; không áp equal-variance requirement của unpaired test cho paired design này.
''')
b.append('## 4. Coverage đang sửa phần nào?\n\n| Skill-mask H24 | Cost | Waiting cost | Episode waiting violation | Pending cuối/episode |\n|---|---:|---:|---:|---:|\n')
for reg in ('legacy','covered'):
 for a in (C,L,F):
  v=g(reg,a);b.append(f"| {reg} — {N[a]} | {v['objective']:.4f} | {v['waiting_cost']:.4f} | {pct(v['waiting_violation'])} | {v['terminal_pending']:.4f} |\n")
b.append('''
Local waiting cost giảm43,3988→31,3144 (**27,85%**); full47,2238→31,9538 (**32,34%**). Waiting component chiếm90,1% tổng cost improvement của local; với full chiếm104,3%, tức net các component khác tăng nhẹ. Đây là accounting decomposition, không phải causal mediation hoặc chứng minh critic/representation là nguyên nhân.

Pending cuối không giảm: local0,5170→0,5266 và full0,5122→0,5202. Coverage chủ yếu giảm độ dài/độ nặng của phần chờ quan sát được, chưa cho thấy xử lý sạch backlog. Số served/censored request không có common fixed denominator giữa policy, do policy thay đổi failure/service trajectories.

| Skill-mask H24 request dữ liệu | Request tổng | Censored | Censored/tổng | Waiting cost từ censored requests | Mean wait của served requests |
|---|---:|---:|---:|---:|---:|
''')
for reg in ('legacy','covered'):
 for a in (C,L,F):
  q=Q[f'{reg}:{a}:skill_mask_H24'];b.append(f"| {reg} — {N[a]} | {q['requests']} | {q['censored']} | {pct(q['censored_fraction'])} | {pct(q['censored_waiting_cost_fraction'])} | {q['mean_served_wait']:.4f} |\n")
b.append('''
Sau coverage, **97,65%/98,60% waiting cost MARL vẫn từ request chưa bắt đầu sửa khi episode kết thúc**. Mean observed wait của censored requests giảm9,5376→7,1519 (local),10,2273→7,3442 (full), nhưng chúng là horizon-censored lower bounds. Mean served wait lại tăng; không dùng chỉ một nhóm này để khẳng định fairness tốt/xấu chung.

Waiting vẫn giải thích78,00%/75,53% residual gap với central; phần còn lại cũng đáng chú ý. Local/full covered failure cost thấp hơn central khoảng9,28/8,08 nhưng downtime cost cao hơn20,04/19,69. Máy hỏng chưa sửa có thể không phát sinh thêm failure, nên “ít failure hơn” không tự động là tốt hơn. Request ending at horizon with wait0 vẫn được tính censored, không tự gọi tất cả pending là starvation. Chưa chứng minh permanent starvation.
''')
b.append('## 5. H12 và các điều kiện khác — exploratory\n\n| Condition/horizon | Central legacy→covered | Local legacy→covered | Full legacy→covered |\n|---|---:|---:|---:|\n')
for c in ('skill_mask','specialized','nominal','homogeneous'):
 for h in (12,24):
  b.append(f'| {c} H{h} | '+' | '.join(f"{g('legacy',a,c,h)['objective']:.4f}→{g('covered',a,c,h)['objective']:.4f}" for a in (C,L,F))+' |\n')
b.append('''
Ở skill-mask H12 (horizon đã train), local giảm2,22%, full3,38%; exploratory95% CI delta lần lượt[−6,1622,−1,8487] và[−8,6735,−3,5606],9/10 và10/10seed cải thiện. H12 hỗ trợ coverage có ích nhưng không thay primary H24. H24 mới còn thể hiện horizon transfer; dữ liệu này chưa isolate nó khỏi condition transfer.

Specialized H12 cost local giảm0,94%, full2,03%, dù exposure specialized giảm một nửa. Tuy nhiên pending tăng0,3889→0,4516 (local),0,3622→0,4164 (full), và waiting violation tăng nhẹ. Protocol preservation guards ở đây chỉ giữ cost<=+5%; không được viết “mọi metric đều được bảo toàn”. Nominal local covered có cost trùng rule165,8375 ở H12/339,5625 H24 trên panel, nhưng không suy policy tương đương trên mọi state hoặc tối ưu.

**Đánh đổi lớn ở homogeneous, vẫn ngoài cả hai training mixture.** H24 local cost tăng411,4094→417,7897 (+1,55%), full405,8325→411,2506 (+1,34%). CI95% cost H24 cả hai vẫn chứa0, nên chưa khẳng định thống kê cost worsening; full H12 cost delta CI95%[1,4566,4,4509] dương (exploratory, không sửa multiple comparisons).

Episode waiting violation homogeneous H24 tăng local15,72%→50,52%, full5,77%→38,11%; waiting cost tăng2,3906→8,9663 và0,7856→6,7013. Pending tăng0,5494→0,6455 và0,4708→0,6231. Đây là descriptive tradeoff đủ lớn để không triển khai mixture như “cải thiện toàn diện”. Không có primary homogeneous guard trong protocol; công khai limitation này thay vì thêm guard hậu nghiệm để đổi verdict.
''')
b.append('## 6. Rule/planner và chi phí thực thi\n\n')
b.append('''Skill-mask H24 rule344,5750; planner331,4859; coveredcentral327,7205; coveredlocal367,5666; coveredfull369,7169. Sau coverage cả hai MARL vẫn kém rule/planner về mean objective trên primary panel. Planner waiting violation0,15625%, central1,76563%, local25,28125%, full24,39063%; waiting/SLA và objective không phải cùng ranking. Planner không phải oracle hay optimum.

| Common DEVELOPMENT state mean latency, pooled4conditions | Legacy | Covered |
|---|---:|---:|
''')
for a in (C,L,F):b.append(f"| {N[a]} | {LAT['legacy:'+a]['mean_ms']:.3f}ms | {LAT['covered:'+a]['mean_ms']:.3f}ms |\n")
b.append(f"\nRule≈{LAT['reference:selected_rule']['mean_ms']:.3f}ms, planner≈{LAT['reference:joint_rollout']['mean_ms']:.3f}ms. Coverage gần như không đổi mean latency. Timing phụ thuộc4 rule trajectories48states và thứ tự/cache benchmark; không có latencyCI hoặc mọi production state. Central includes diagnostic critic; serial actors shared full-tensor frontend. Không suy intrinsic decentralized speed advantage. Communication là inherited logical bytes/rounds, chưa đo distributed network, loss hoặc synchronization.\n")
b.append('''
## 7. Hàm ý và việc cần làm tiếp

1. Coverage hypothesis được hỗ trợ một phần: mixture mới có ích cho local skill-mask theo cổng đã chốt, và mean cả hai cooperative giảm. **Không chọn kiến trúc MARL mới chỉ từ kết quả này.**
2. Nút thắt vẫn nằm ở requests chờ kéo dài, dù có exposure skill-mask; full information chưa giải quyết nó. Chưa isolate critic, credit assignment, entropy hoặc policy factorization. Tự quy lỗi cho một module sẽ đi quá dữ liệu.
3. Nếu làm experiment tiếp, ưu tiên một intervention duy nhất về **training horizonH12 versusH24**, giữ covered mixture, decoder, architecture và objective; bao gồm central đối chứng. Dùng equal physical ticks/optimizer budgets và công khai số complete episodes khác nhau, thay vì âm thầm cấp gấp đôi compute. Đây là đề xuất, chưa phải protocol đã khóa hay code đã chạy. Cùng lúc theo dõi homogeneous như stress/safety panel; một ablation coverage rộng hơn là can thiệp riêng, không trộn vào horizon test nếu muốn isolate.
4. Không gọi covered mixture là lời giải robust: cần replication mới và xác nhận tradeoff request waiting. Full seed180005, homogeneous và pending deterioration đều phải ở lại báo cáo.
5. Lý do triển khai MARL vẫn cần constraint thực: local information, communication budget hoặc scale. Dữ liệu này chứng minh training distribution đáng kiểm soát, không chứng minh MARL tốt hơn centralized RL hoặc planner. Kết luận central/planner phù hợp hơn vẫn được phép.

Panel mới đã mở. Không tuning bằng panel này rồi tái dùng nó như sealed confirmatory test. Raw artifacts giữ nguyên; report và scripts nằm trong timestamped analysis directory.

## 8. Fallacy scan — 11/11 checked

| Loại | Đánh giá và biện pháp |
|---|---|
| Simpson's paradox | Tách4conditions×2horizons; không dùng mean gộp che homogeneous deterioration |
| Ecological fallacy | Không suy từ mean cost sang fairness từng máy/request; có per-request censoring |
| Berkson's paradox | Không filter checkpoint theo tốt/xấu; fixed final cho mọi model. Labeled profile population vẫn hạn chế |
| Collider bias | Không chọn chỉ served request hoặc seed thành công; mọi60model/317440episode vào phân tích |
| Base-rate neglect | Episode violation khác overdue-request fraction; báo counts và mẫu số policy-dependent |
| Regression to mean | Effect dựa fresh six-cell paired retraining, không so pre/post old selected checkpoints |
| Survivorship bias | Bao gồm858570served/censored requests; served-only wait không dùng để kết luận chung |
| Look-elsewhere effect | Bốn primary Bonferroni98,75%; toàn bộ52contrast khác exploratory95%, không chọn H12 thay H24 |
| Garden of forking paths | Budget/final checkpoint/mixture/gates đã commit trước run. Sign-flip/leave-one-out là posthoc supplemental, không thay verdict |
| Correlation versus causation | Can thiệp hỗ trợ effect mixture cụ thể ở fixedbudget; component decomposition không isolate causal mechanism/module |
| Reverse causality | Ít failure có thể đi cùng máy hỏng chưa sửa; không suy ngược ítfailure chứng minh chính sách tốt |

## 9. Tệp tái lập

- `analyze.py`: hash/source/grid/cost/statistics/request/machine/training audit độc lập, read-only.
- `independent_analysis.json`:56contrasts,64aggregate panels, paired effect vectors và episode change counts.
- `training_audit.json`: raw2.457.600training rows,19.200progress rows, seed/config pairing, mixture và diagnostics.
- `request_audit.json`: request counts, observed wait histograms và censoring decomposition.
- `latency_audit.json`: pooled same-state mean timing, không network benchmark.
- `checkpoint_replay.py` / `checkpoint_replay.json`:360savedcheckpoint audit, initial tensor regeneration,2880developmentaction replay.
- `followup.py` / `supplemental_audit.json`: development selection, raw seedmeans và explicitly posthoc sensitivity.
- `write_report.py`: render report từ independent outputs; không đổi raw results.

Verification Status ANALYZED. Independent arithmetic và development action replay thành công; toàn bộ test dynamics/training chưa rerun. Không báo VERIFIED scientific replication hoặc bit-identical Linux/Mac inference cho mọi state.
''')
(P/'analysis_vi.md').write_text('\n'.join(b))
print(P/'analysis_vi.md')
