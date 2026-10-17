"""Fixed-final sealed N4 confirmation report after independent complete audit."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS=('typed_gibbs_ppo','typed_myopic_ppo','reverse_typed_myopic_ppo','ppo','normalized_ppo','mappo','full_mappo')
LABELS=('Gibbs learned future','Myopic future counts','Myopic reverse','PPO','PPO normalized','MAPPO local','MAPPO full')
NAMES=dict(zip(ARMS,LABELS))
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);sm=read(source/'manifest.json');vm=read(verification/'manifest.json');v=read(verification/'verification.json');s=read(source/'summary.json')
    assert sm['status']=='COMPLETED' and sm['audit_status']=='PASS' and vm['status']=='COMPLETED' and v['status']=='PASS' and v['source']==str(source) and v['source_manifest_sha256']==sha(source/'manifest.json') and v['independent_checks']==s['checks'] and v['original_confirmation_pass']==s['confirmation_screen_pass'] and s['test_opened']
    for root,entries in ((source,sm['output_bindings']),(verification,vm['outputs'])):
        for name,meta in entries.items():assert sha(root/name)==meta['sha256'] and (root/name).stat().st_size==meta['bytes'],name
    selected=rows(source/'selected_test_episodes.csv');dev=rows(source/'development_episodes.csv');trains=sorted({int(r['train_seed']) for r in dev});steps=sorted({int(r['checkpoint_steps']) for r in dev});curves=[];replicas={};components={};conflicts={};extras={}
    for a in ARMS:
        rr=[r for r in selected if r['method']==a];replicas[a]={str(t):float(np.mean([float(r['normalized_cost']) for r in rr if int(r['train_seed'])==t])) for t in trains};components[a]={k:float(np.mean([float(r[k])/4 for r in rr])) for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost')};conflicts[a]={k:float(np.mean([float(r[k]) for r in rr])) for k in ('broker_conflicts','broker_rejected_proposals')};extras[a]={k:float(np.mean([float(r[k]) for r in rr])) for k in ('technician_utilization','terminal_failed','terminal_pending','jobs')}
        for step in steps:curves.append(dict(method=a,steps=step,normalized_cost=float(np.mean([float(r['normalized_cost']) for r in dev if r['method']==a and int(r['checkpoint_steps'])==step]))))
    data=dict(classification='SEALED_N4_CONFIRMATION_REPORT',source=str(source),verification=str(verification),summary=s,replica_costs=replicas,cost_components_per_machine=components,policy_conflicts=conflicts,additional_metrics=extras,cfdev_curves=curves,goal_pass=False,test_opened=True,limitation='Five fresh training replicas and empirical crossed bootstrap with family adjustment. No distribution-free coverage, PPO-superiority or scaling claim unless its original explicit gates pass.')
    (out/'analysis.json').write_text(json.dumps(data,indent=2)+'\n');bounds=s['confidence_endpoints'];fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained');y=np.arange(7);axes[0].barh(y,[s['means'][a]['normalized_cost'] for a in ARMS],color=['#1565C0']+['#B0BEC5']*6)
    for j,t in enumerate(trains):axes[0].scatter([replicas[a][str(t)] for a in ARMS],y+(j-2)*.065,s=14,color=plt.cm.tab10(j),label=f'Replica{j+1}',zorder=3)
    axes[0].set_yticks(y,LABELS);axes[0].invert_yaxis();axes[0].set_xlabel('Mean TEST H60 cost / machine');axes[0].set_title('Fixed final; dots = 5 replica means');axes[0].legend(fontsize=6.5)
    for i,a in enumerate(ARMS[1:]):
        r=bounds['gain_vs_'+a];point=100*r['point'];bound=100*r['bound'];threshold=100*r['threshold'];color='#2E7D32' if r['passed'] else '#C62828';axes[1].plot([bound,point],[i,i],color=color,lw=2);axes[1].scatter([point],[i],color=color,marker='o',s=27,label='Point' if i==0 else None);axes[1].scatter([bound],[i],color=color,marker='<',s=37,label='Lower bound' if i==0 else None);axes[1].scatter([threshold],[i],color='black',marker='|',s=140,label='Locked threshold' if i==0 else None)
    axes[1].set_yticks(np.arange(6),LABELS[1:]);axes[1].invert_yaxis();axes[1].axvline(0,color='gray',lw=.7);axes[1].set_xlabel('Candidate TEST cost gain (%)');axes[1].set_title('Family-adjusted bootstrap lower bounds');axes[1].legend(fontsize=6.5)
    labels=[]
    for i,(a,metric) in enumerate((a,m) for a in ('ppo','normalized_ppo') for m in ('availability','normalized_pending','normalized_failures')):
        r=bounds[metric+'_delta_vs_'+a];point=r['point']/r['threshold'];bound=r['bound']/r['threshold'];color='#2E7D32' if r['passed'] else '#C62828';axes[2].plot([point,bound],[i,i],color=color,lw=2);axes[2].scatter([point],[i],color=color,marker='o',s=25,label='Point harm' if i==0 else None);axes[2].scatter([bound],[i],color=color,marker='>',s=37,label='Upper harm bound' if i==0 else None);labels.append({'availability':'Avail','normalized_pending':'Pending','normalized_failures':'Failures'}[metric]+' / '+NAMES[a])
    axes[2].set_yticks(np.arange(6),labels);axes[2].invert_yaxis();axes[2].axvline(1,color='black',ls='--',label='Maximum harm');axes[2].axvline(0,color='gray',lw=.7);axes[2].set_xlabel('Health harm / locked allowance');axes[2].set_title('6 health bounds; lower is better');axes[2].legend(fontsize=6.5);fig.suptitle('Typed-Gibbs N4/K2 sealed confirmation — crossed bootstrap, 5 replicas × 256 seeds');fig.savefig(out/'test_quality_health.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for a,color in zip(ARMS,plt.cm.tab10(np.arange(7))):axes[0].plot(np.array(steps)/1000,[r['normalized_cost'] for r in curves if r['method']==a],marker='.',lw=1.2,color=color,label=NAMES[a])
    axes[0].set_xlabel('Training decisions (thousands)');axes[0].set_ylabel('Mean CFDEV H60 cost / machine');axes[0].set_title('All registered checkpoints; no selection');axes[0].legend(fontsize=6,ncol=2)
    per=[100*s['per_replica_main_gain'][str(t)] for t in trains];axes[1].bar(np.arange(5),per,color=['#2E7D32' if x>0 else '#C62828' for x in per]);axes[1].axhline(0,color='black',lw=.7);axes[1].axhline(.5,color='gray',ls='--',label='Pooled practical threshold');axes[1].set_xticks(np.arange(5),[f'R{i+1}' for i in range(5)]);axes[1].set_ylabel('TEST cost gain vs trained myopic (%)');axes[1].set_title('Stability guard: every replica >0');axes[1].legend(fontsize=6.5)
    keys=('maintenance_cost','unavailability_cost','failure_cost','waiting_cost');delta=[components[ARMS[0]][key]-components[ARMS[1]][key] for key in keys];axes[2].barh(np.arange(4),delta,color=['#2E7D32' if x<=0 else '#C62828' for x in delta]);axes[2].set_yticks(np.arange(4),['Maintenance','Unavailability','Failure penalty','Waiting penalty']);axes[2].invert_yaxis();axes[2].axvline(0,color='black',lw=.7);axes[2].set_xlabel('Gibbs − myopic H60 cost / machine');axes[2].set_title('Cost components; negative is lower');fig.suptitle('Learning and matched mechanism comparison — no teacher or test-time updates');fig.savefig(out/'learning_mechanism.png',dpi=170);plt.close(fig)
    passed=sum(s['checks'].values());failed=[key for key,ok in s['checks'].items() if not ok];bad_extra=[key for key,ok in s['extra_checks'].items() if not ok];verdict='ĐẠT xác nhận N4 theo protocol; scalability vẫn chưa kiểm chứng' if s['confirmation_screen_pass'] else 'KHÔNG ĐẠT toàn bộ protocol xác nhận N4; không mở N lớn'
    lines=['# Typed-capacity Gibbs PPO: kết quả xác nhận N4/K2','',f'**{verdict}.** {passed}/12 CI gates đạt; independent audit PASS. Mục tiêu nghiên cứu tổng thể chưa hoàn tất.','',
    '## Thiết kế được giữ cố định','',
    'Giữ nguyên thuật toán đã qua DEV: máy gửi utility cho hai loại technician, broker tính joint labeled matching bằng weighted count DP. Đối chứng chính cùng thông tin/mạng/dung lượng/budget/entropy/MAP chỉ dùng unweighted future completion counts. Giữ PPO scalar joint chuẩn, học từ khởi tạo mới, không teacher/distillation/warm-start. Fullstate broadcast và hai service profiles là giả định hiện tại, chưa phải fully decentralized hoặc arbitrary heterogeneous profiles.','',
    '35fits từ5trainingseeds mới×7methods,491.520decisions/fit; tổng17.203.200decisions/358.400trainEP/268.800Adamupdates.64CFDEVseeds×5checkpoints=11.200nodes. Không chọn checkpoint/seed từ CFDEV: cố định cuối491.520 cho tất cả. Freeze đủ35fileSHA/parameterdigests trước mở256TESTseeds mới.8.960policyEP+256deadline=9.216TESTepisodes,552.960physical transitions/442.368policy decisions; không optimizer trên test.','',
    '200.000paired crossed bootstrap resamples: lấy lại5trainingreplicas và256physicalseeds, cùng indices cho mọi phương pháp/metric, trung bình tích Cartesian rồi mới tính costratio. Alpha theo familyround7=.05/(7×8), Bonferroni12directionalbounds; mỗi endpoint alpha≈0.00007440476, quantilelinear. Đây là empirical uncertainty với5trainingreplicas và tail khoảng14,88samples, không phải guarantee coverage chính xác. Strict-zero MAPPO ties trong1e−12 bị loại; không dùng precision để cứu verdict.','',
    '## TEST fixed final','', '| Nhánh | Cost/máy H60 | Availability H48 | Pending/máy H48 | Failure/máy H48 |','|---|---:|---:|---:|---:|']
    for a in ARMS:
        m=s['means'][a];lines.append(f"| {NAMES[a]} | {m['normalized_cost']:.6f} | {m['availability']:.6f} | {m['normalized_pending']:.6f} | {m['normalized_failures']:.6f} |")
    lines+=['','## Tất cả khoảng tin cậy và giới hạn đã khóa','', '| Endpoint | Point | Bound định hướng | Giới hạn | Đạt |','|---|---:|---:|---:|---|']
    for name,r in bounds.items():lines.append(f"| {name} | {r['point']:.8f} | {r['orientation']} {r['bound']:.8f} | {'>' if r['strict'] else '>=' if r['orientation']=='lower' else '<='} {r['threshold']:.6f} | {'Có' if r['passed'] else 'Không'} |")
    lines+=['','Costgains và availability dùng lower bound; pending/failure dùng upper bound. Các số gain ở bảng là fraction (nhân100 ra%). Mọi12bound cùng các extra guard mới quyết định full protocol pass; không lấy point đẹp hoặc một metric tốt thay verdict.','',f"Main TEST gain mỗi seed: {', '.join(f'{t}: {x:.4f}%' for t,x in zip(trains,per))}.",f"PPO superiority theo rule≥1% lowergainbound vsBOTH PPO/norm và overallpass: {s['ppo_superiority_supported']}.",f"Không đạt CI: {', '.join(failed) or 'không'}. Không đạt extra: {', '.join(bad_extra) or 'không'}.",f"N4 isolated actor+DP+decoder p95={s['latency']['p95_ms']:.6f}ms,720rawsamples; không đo network/observation/physics và chưa chứng minh scaling.",'',
    '## Cơ chế, tranh chấp và sức khỏe','',
    'Retraining myopic có cùng thông tin, tham số và ngân sách là bằng chứng chính cho learned continuation. So local MAPPO ít thông tin hơn không tự chứng minh cơ chế. Zero conflict ở các lõi được đảm bảo bởi hardmatching; không suy ra hiệu năng từ zero conflict. Không thêm knockout test-time để chọn thiết kế hoặc diễn giải lại các CI gates.','',
    '| Nhánh | Maintenance/máy | Unavailability/máy | Failure penalty/máy | Waiting penalty/máy | Conflicts/EP | Rejected/EP |','|---|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        p=components[a];q=conflicts[a];lines.append(f"| {NAMES[a]} | {p['maintenance_cost']:.6f} | {p['unavailability_cost']:.6f} | {p['failure_cost']:.6f} | {p['waiting_cost']:.6f} | {q['broker_conflicts']:.6f} | {q['broker_rejected_proposals']:.6f} |")
    lines+=['','## Kiểm chứng và ranh giới kết luận','',
    'Bộ kiểm chứng độc lập kiểm tra toàn bộ35initializers/51200pairedtrainingtapes/budgets/CFDEVgrid/freeze/testgate, replay toàn bộ9.216TESTepisodes bằng primitive dictionary physics, NumPy actors và explicit21matching likelihood/entropy/MAP. Tái tạo đủ200.000bootstrap index rows từ registered RNG; tính lại toàn bộ2.400.000endpoint values bằng Cartesian indexing, không gọi hàm bootstrap/count-weight inference của production. Tính lại12bounds,5replica signs,720rawtimings và nguyên verdict. Historical optimizer minibatch gradients không lưu đủ nên không tuyên bố replay từng optimizer update.','',
    'Các ý tưởng joint global normalization, structured policy và partition DP đã có prior art. Đóng góp cụ thể về maintenance/two-type matching PPO và cơ chế vẫn phải được đánh giá theo bằng chứng, không chỉ tên thuật toán. Vòng này không có N>4/generalization. Nếu bất kỳ gate thất bại, giữ nguyên kết quả và protocol; không đổi threshold, chọn control/checkpoint/seed/testsubset để cứu. Goal active; chỉ independent ALLPASS mới cho phép protocol scaling riêng.','',f'Nguồn: `{source}`.',f'Kiểm chứng: `{verification}`.','', '![Chất lượng test và sức khỏe](test_quality_health.png)','![Learning và cơ chế](learning_mechanism.png)','']
    (out/'findings_vi.md').write_text('\n'.join(lines));shutil.copy2(__file__,out/'reporter.py');manifest=dict(status='COMPLETED',source=str(source),verification=str(verification),source_manifest_sha256=sha(source/'manifest.json'),verification_manifest_sha256=sha(verification/'manifest.json'),original_confirmation_pass=s['confirmation_screen_pass'],goal_pass=False,test_opened=True,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()});(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--verification',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();run(a.source,a.verification,a.output)
