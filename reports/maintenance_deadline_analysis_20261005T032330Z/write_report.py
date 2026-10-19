import csv,json,statistics,collections
from pathlib import Path
ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl');OUT=ROOT/'reports/maintenance_deadline_analysis_20261005T032330Z';RAW=ROOT/'lab_results/maintenance_deadline_full_cpu_20261005T023359Z'
s=json.loads((RAW/'summary.json').read_text());st=json.loads((OUT/'statistics.json').read_text());a=json.loads((OUT/'audit.json').read_text())
with (RAW/'regression_decisions.csv').open() as f:traces=list(csv.DictReader(f))
with (RAW/'regression_job_completions.csv').open() as f:jobs=list(csv.DictReader(f))
work=(260,262,249,242,237,280);service=((2,6),(2,6),(2,6),(6,2),(2,6),(2,6));counts=collections.Counter()
for r in traces:
 if r['instance_seed']!='920211':continue
 before=json.loads(r['state']);after=json.loads(r['next_state'])
 for t,m in enumerate(after['assigned']):
  if m>=0 and before['remaining'][t]==0:counts[r['policy'],m,service[m][t],before['failed'][m]]+=1
witness=dict(scope='opened920211/firstshock921200; observed actual dispatch and completion only',required_work=work,service=service,machine1={pol:dict(pm2=counts[pol,1,2,False],pm6=counts[pol,1,6,False],finished=max(int(r['finished']) for r in jobs if r['instance_seed']=='920211' and r['policy']==pol and r['machine']=='1')) for pol in ('H_health_risk_dispatch','S_capacity_reservation','D_deadline_packing')})
(OUT/'witness_920211.json').write_text(json.dumps(witness,indent=2))
text='''# Phân tích deadline-packing v1

## Material Passport
ARS experiment-agent/validate;2026-10-05 Asia/Taipei. Verification Status: ANALYZED. Run maintenance_deadline_full_cpu_20261005T023359Z; clean commita7089acfd9074a4c1e7c9023673f0f39446562b4. Independent statistics/slot checks plus shared-implementation replay; not independent simulator reproduction. Không chỉnh raw, không triển khai experiment mới.

## Kết luận chính
Có bằng chứng rõ hơn về impact của quyết định bảo trì/phân công thợ trong benchmark, đặc biệt high-pressure/uniform-service. Một scheduler đơn giản D đã thu hồi phần lớn mất mát trong điều kiện này, nên kết quả chưa tạo lý do bắt buộc dùng MARL.
D chưa đủ thay S trong heterogeneous-service: primary adequacy và heterogeneous-tail guard không đạt. Điều đáng chú ý là D có thể không hỏng mà vẫn hoàn thành sản xuất chậm hơn: objective tránh trễ service chưa bao quát chi phí ngừng sản xuất và việc chọn thợ chậm. Đây là residual gap của heuristic hiện tại, không phải bằng chứng mọi heuristic/optimization đều thất bại.

## Audit trước performance
COMPLETED;02:34:02–02:58:21UTC,24phút19giây. Source/protocol/four dependency hashes đúng commit bàn giao; source_dirty=false. Đủ34,560evaluation,10,368development,576configuration+coordination records và180opened-case regressions riêng. Cartesian keys không thiếu/trùng; feasible requests, exact required productive work/completed jobs, PM/CM partitions, costs finite vàdecomposition đúng; partial/finalepisodes byte-identical. Resolved configs khớp factory, checkpoint selection đúng global-dev minima, panel/splits đúng locked protocol.
Bootstrap độc lập10,000draws/seed942100,24instance aggregates conditional20fixedshocks, tái tính6contrasts×24cells khớp summary. Không coi480shock là480instance. Replay tất cả1,728first-shockepisodes đãghi:372,042decision/414,720completion; slot audit riêng120,021Dplans. Mọi180regressionepisode replay;3,335decision/3,456completion đượcghi khớp; H/S của case cũ bằng các transferred run cũ. Max numerical mismatch0, runtime excluded. Không tái chạy full34,560evaluation trên Mac. Các lần simulator replay dùngchung implementation, không tuyên bố independent scientific reproduction.

## Những kết quả quan trọng
N6,64jobs/máy,high pressure. Makespan làmean24instance means, mỗiinstance20shock.

| Skill | CM/PM | H | S | D |
|---|---:|---:|---:|---:|
'''
for skill,ratio in [('heterogeneous',2),('heterogeneous',4),('uniform',2),('uniform',4)]:
 c=s['cells'][f'high_{skill}_jobs64_ratio{ratio}'];v=[c['metrics'][pol]['makespan'] for pol in ('H_health_risk_dispatch','S_capacity_reservation','D_deadline_packing')]
 text+=f'|{skill}|{ratio}|{v[0]:.2f}|{v[1]:.2f}|{v[2]:.2f}|\n'
text+='''
Heterogeneous ratio2 primary: D321.63 vsS308.36; paired +13.27ticks,CI95%[3.11,24.63]. Mean instance-relative (D-S)/S=+4.19%,CI[0.96,7.77]%; upper7.77%>margin2% =>mean_adequacy=false. CI chứa2%: chưa chứng minh noninferiority, không khẳng định gap chắc chắn lớn hơn2%. Dthắng4/tie13/thua7instances vsS: tổn thất không đồng đều trên toàn panel. DvsHmean+0.91%,CI[-2.77,4.52]%, nên khôngkhẳng định DchậmH nói chung.
Heterogeneous ratio4: D326.88 vsS309.20; +17.67ticks,CI[7.10,29.38]; meanrelative+5.65%,CI[2.30,9.34]%. P95relative D-S+5.63%,CI[2.27,9.34]%; upper9.34%>5% =>heterogeneous_tail_guard=false. Mean Dfails=0 trên480shock nhưngPMservice234.50ticks vsS211.06; có ít hỏng hơn không đồng nghĩa makespan tốt hơn. Dthắng3/tie11/thua10instances vsS.
Uniform ratio4: D362.37 vsH528.70/S631.93. Paired relative reduction vsH22.66%,CI[14.16,31.70]%; vsS33.79%,CI[23.91,43.28]%. Đây là mean của per-instance percentages, không phải tỷ lệ từ pooledmeans (vsH khoảng31.46%). DthắngH24/24,thắngS20/tie4/thua0. P95relative D-H=-28.36%,CI[-38.75,-18.08]% =>uniform_tail_guard=true. Uniform mean contrasts làsecondarydescriptive, nhưnguniform p95 guard đãkhóa; không gắn nhiều cell với discovery đã multiplicity-adjust.
Uniform ratio2: D350.05 vsH373.24/S368.29; meanpaired improvement vsH5.27%,CI[2.77,8.22]% vàvsS3.99%,CI[1.42,7.17]%. Không cần chỉ dựa vào extreme ratio4 để thấy benefit của service scheduling; vẫn là giả lập chưacalibrated.
Overall adequacy=false: chỉ1/3gate đạt. Không có claimDthayS mọisetting, cũng không dùng failuregate làm chứng minh cầnlearning. Lowpressure: D vàSmeanmakespantrùng ở cả12cells; sự khác biệt chủ yếu lộ ra khicrewscarce.

## Cơ chế: tránh hỏng và giảm mất sản lượng là hai mục tiêu khác nhau
Dev chọn Hoffset0/2/4; Slead6 mọiratio; Dlead2/4/6. Full đãchọnDlead6 ởratio4, khácengineering smokelead2; không retune trêntest/oldcases. CảH/S/D đều4candidates,global8cells mỗiratio, dùngthôngtincurrentstate chung, khôngfuturehazard.
Trong high/heterogeneous/ratio4, DPMjobs105.42 soS103.95 (gần nhau) nhưngPMserviceticks234.50so211.06. Tracefirstshock24instances: D có142PMdispatch duration6 soS19; duration2 D2388/S2477. Chỉ áp dụng cho first logged shock, không suy từpartial service ticks thànhfull job counts. Các assignmentto slow workers làm rõ nguồn durationburden; chưa decompositioncausal toàn bộpanel.
Case920211/shock921200: máy1 có262ticks productive work. S cho máy này24PM×2ticks =>finish310. H cho32PM×2 =>finish326. D cho9PM×2 và23PM×6 =>finish262+18+138=418. D tạo thêm108ticks mấtservice trênmáy1 soS; toànrun Dmakespan418 vìmáy1 thànhcritical. S trongshock nàymakespan330 do máy5, H348 do máy5. TấtcảH/S/D ởshock này khônghỏng. D cứ đápứng safe-start deadlines vẫn làm mộtmáy mất nhiềuproductive time: greedy earliest-finish/earliest-start worker score không optimizefinalproductionclock.
Đây là lỗi/tradeoff objective của D hiện tại, không phải chứng minh scheduling objective không thể giải bằng toán/quy tắc. Healthy máy chờ thợ vẫnproductive, trong khiPMdài làmngừngmáy; startservice sớm/finishservice sớm riênglẻ không luôn minimizeproductionmakespan.

## Opened regressions: khônggộp scientificpanel
| Instance | H mean | S mean | D mean | D failures |
|---|---:|---:|---:|---:|
'''
for i in ('900215','920216','920211'):
 c=s['regressions']['cases'][i];pols=('H_health_risk_dispatch','S_capacity_reservation','D_deadline_packing')
 text+=f"|{i}|{c[pols[0]]['makespan']:.2f}|{c[pols[1]]['makespan']:.2f}|{c[pols[2]]['makespan']:.2f}|{c[pols[2]]['failures']:.2f}|\n"
text+='''
Mỗirow20shock/policy; Hoffset4,Slead6,Dnew-devlead6. D900215315/0failure trêncả20shock, tựtái hiện benefit của trướcđâyforcedintervention màkhôngbiếtfuturehazards. D920216312/0failure, bằngS vàtránhHcascade. D920211418/0failure trênmọishock, làmmeanxấu hơnH348 vàS362.3 dù tránhSfailure. Ba caseđược chọn sau kếtquảcũ nênchỉ mechanism/regression, khôngpopulation CI.

## Rare tail và phạm vi impact
Uniform ratio4 failure-positive episodes H278/480,S377/480,D13/480; Dgiảmcascade mạnh nhưngkhôngtriệttiêu. Maxepisode makespanD1027>H992 trongcell này. Gateaveraged per-instancep95 đãđạt khôngphải worst-caseguarantee. Heterogeneous ratio2 H230/480,S29/480,D13/480; D ítfailure nhưng vẫnchậmS. Heterogeneous ratio4 H33/480,S5/480,D0/480. S/Dwaitingqueue0 khôngphải proof rằngfailedwaitingoutsidequeue0.
Impact tổng từ chọnpolicy rõ hơn trước: ởscarceuniform có cảratio2 (~5%) vàratio4 (>20%) benefit quaheuristicđơn giản. Nhưngtargetcellheterogeneous vẫnchỉgapvàtradeoff khoảngvài%, khôngtựnhiên chứngminh các yêu cầu kiếntrúcMARL cũ. Ratio1/2/4 làsensitivity, chưaphysicalcalibration, fixedroutes/resumablework,alljobst0, khôngrevenue/scrap/realFJSP. Khôngclaimnovelty hoặcimpactnhàmáy.
Problemgiờ có thể phátbiểu rõ hơn: **phối hợp thời điểm PM và lựa chọn thợ để tránhfailure mà không trả quá nhiềuproduction downtime, đặcbiệt khi tốcđộthợ phụthuộc máy**. Dứngsafe-startconstraint nhưngassignment/maintenanceearliness tạoPMdowntime; S giữproductioncost tốt trongheterogeneous nhưngfragileởuniform. Cần đối chứngproduction-aware mạnh hơn/nhỏ exactoracle để đo residualgap trướckhi soMARL; khôngpatch/tune hay khởi chạyexperimentmới trong lượt này. BaselineD thấtbại khôngđồngnghĩa mọiheuristic thấtbại.

## Fallacy scan11/11
1Simpson: giữskill/capacity/ratio, khônggộpđảo chiều;2Ecological: inferenceinstance vàfixedshocks, khôngclaimmọiepisode;3Berkson:openedcasesselectedhindsight,táchpanel;4Collider:primarykhônglọc theo realizedfailure;5Base-rate:frequency480 vàseverity/maxcùngbáo;6Regression-to-mean:knowncasespaired baselines cũ, khônggeneralizecaseimprovement;7Survivorship:đủCartesian/capfailurekhôngcensor,giữoutliers;8Look-elsewhere:3gateslocked,cellssecondaryexploratory;9Forkingpaths:hash/seed/dev/stoppingfixed,khôngtestretune;10Correlation/causation:policy effects trongsimfixedphysics, chưaplantclaim/chưaisolatedcomponentablation;11Reversecausality:state/action trướchazard kiểmtrace, khôngcoifailureítlàmnguyênnhânmakespantốt. Uncertainty còn lại:24instances/20conditionalshocks, raretail, devn6globaltradeoffs, chưacalibration vàcounterfactualoracle chưađo.
'''
(OUT/'analysis_vi.md').write_text(text);(OUT/'write_report.py').write_text(Path(__file__).read_text());print(OUT/'analysis_vi.md')
