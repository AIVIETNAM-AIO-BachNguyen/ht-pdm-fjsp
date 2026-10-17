"""Descriptive fixed-final report bound to independent typed-Gibbs verification."""
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
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    sm=read(source/'manifest.json');vm=read(verification/'manifest.json');v=read(verification/'verification.json');s=read(source/'summary.json')
    assert sm['status']=='COMPLETED' and sm['audit_status']=='PASS' and vm['status']=='COMPLETED' and v['status']=='PASS' and v['source']==str(source) and v['source_manifest_sha256']==sha(source/'manifest.json') and v['independent_checks']==s['checks'] and v['original_development_pass']==s['development_screen_pass'] and not s['test_opened']
    for root,entries in ((source,sm['output_bindings']),(verification,vm['outputs'])):
        for name,meta in entries.items():assert sha(root/name)==meta['sha256'],name
    dev=rows(source/'development_episodes.csv');selected=rows(source/'selected_episodes.csv');trains=sorted({int(r['train_seed']) for r in dev});steps=sorted({int(r['checkpoint_steps']) for r in dev});curves=[];replicas={};conflicts={};components={};health_extra={}
    for a in ARMS:
        rr=[r for r in selected if r['method']==a];replicas[a]={str(t):float(np.mean([float(r['normalized_cost']) for r in rr if int(r['train_seed'])==t])) for t in trains}
        conflicts[a]={k:float(np.mean([float(r[k]) for r in rr])) for k in ('broker_conflicts','broker_rejected_proposals')}
        components[a]={k:float(np.mean([float(r[k])/4 for r in rr])) for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost')}
        health_extra[a]={k:float(np.mean([float(r[k]) for r in rr])) for k in ('technician_utilization','jobs','terminal_pending','terminal_failed')}
        for step in steps:curves.append(dict(method=a,steps=step,normalized_cost=float(np.mean([float(r['normalized_cost']) for r in dev if r['method']==a and int(r['checkpoint_steps'])==step]))))
    opportunity=dict(decisions=0,at_least_two_active_with_free_worker=0,active_demand_exceeds_free_workers=0,active_machine_histogram=[0]*5,free_worker_histogram=[0]*3)
    for row in rows(source/'decisions.csv'):
        if row['method']!=ARMS[0] or int(row['tick'])>=48:continue
        state=json.loads(row['state']);free=sum(r==0 for r in state['remaining']);serving={m for m in state['assigned'] if m>=0};active=sum(m not in serving and (state['failed'][m] or state['ages'][m]>=3) for m in range(4))
        opportunity['decisions']+=1;opportunity['at_least_two_active_with_free_worker']+=int(active>=2 and free>0);opportunity['active_demand_exceeds_free_workers']+=int(active>free and free>0);opportunity['active_machine_histogram'][active]+=1;opportunity['free_worker_histogram'][free]+=1
    assert opportunity['decisions']==4608
    mechanism={str(t):v['selected']['neural_checks'][f'{ARMS[0]}:{t}']['mechanism'] for t in trains};data=dict(classification='DESCRIPTIVE_TYPED_GIBBS_DEV_ONLY',summary=s,curves=curves,replica_costs=replicas,policy_conflicts=conflicts,cost_components_per_machine=components,additional_health=health_extra,candidate_opportunities=opportunity,same_weight_mechanism=mechanism,source=str(source),verification=str(verification),goal_pass=False,test_opened=False,limitation='Fixed final DEV point results, no CI. Knockout on candidate trajectories is descriptive; matched independently trained myopic control is primary.')
    (out/'analysis.json').write_text(json.dumps(data,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained');y=np.arange(len(ARMS));axes[0].barh(y,[s['means'][a]['normalized_cost'] for a in ARMS],color=['#1565C0']+['#B0BEC5']*6)
    for j,t in enumerate(trains):axes[0].scatter([replicas[a][str(t)] for a in ARMS],y+(j-1)*.12,color=['#212121','#E65100','#6A1B9A'][j],s=15,label=f'Replica {j+1}',zorder=3)
    axes[0].set_yticks(y,LABELS);axes[0].invert_yaxis();axes[0].set_xlabel('Mean H60 cost / machine');axes[0].set_title('Fixed final 491520; dots = replica means');axes[0].legend(fontsize=7)
    gains=[100*s['endpoints']['gain_vs_'+a] for a in ARMS[1:]];bounds=[.5 if a==ARMS[1] else 0 if a in ('mappo','full_mappo') else -1 for a in ARMS[1:]]
    axes[1].barh(np.arange(6),gains,color=['#2E7D32' if s['checks']['gain_vs_'+a] else '#C62828' for a in ARMS[1:]]);axes[1].scatter(bounds,np.arange(6),color='black',marker='|',s=150,label='Locked bound');axes[1].set_yticks(np.arange(6),LABELS[1:]);axes[1].invert_yaxis();axes[1].axvline(0,color='gray',lw=.7);axes[1].set_xlabel('Candidate cost gain vs control (%)');axes[1].set_title('6 cost gates: green pass, red fail');axes[1].legend(fontsize=7)
    health=[];labels=[];colors=[]
    for a in ('ppo','normalized_ppo'):
        for metric,limit in (('availability',-.01),('normalized_pending',.025),('normalized_failures',.25)):
            key=metric+'_delta_vs_'+a;health.append(s['endpoints'][key]/limit);labels.append({'availability':'Avail','normalized_pending':'Pending','normalized_failures':'Failures'}[metric]+' / '+NAMES[a]);colors.append('#2E7D32' if s['checks'][key] else '#C62828')
    axes[2].barh(np.arange(6),health,color=colors);axes[2].set_yticks(np.arange(6),labels);axes[2].invert_yaxis();axes[2].axvline(1,color='black',ls='--',label='Maximum harm');axes[2].axvline(0,color='gray',lw=.7);axes[2].set_xlabel('Health harm / locked allowance');axes[2].set_title('6 health gates; lower is better');axes[2].legend(fontsize=7);fig.suptitle('Typed-capacity Gibbs PPO — N4/K2 DEV, descriptive, no confidence intervals');fig.savefig(out/'quality_health.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for a,color in zip(ARMS,plt.cm.tab10(np.arange(7))):axes[0].plot(np.array(steps)/1000,[r['normalized_cost'] for r in curves if r['method']==a],label=NAMES[a],color=color,marker='.',lw=1.2)
    axes[0].set_xlabel('Training decisions (thousands)');axes[0].set_ylabel('Mean DEV H60 cost / machine');axes[0].set_title('All registered checkpoints');axes[0].legend(fontsize=6.5,ncol=2)
    per=[100*s['per_replica_main_gain'][str(t)] for t in trains];axes[1].bar(np.arange(3),per,color=['#2E7D32' if g>0 else '#C62828' for g in per]);axes[1].axhline(0,color='black',lw=.7);axes[1].axhline(.5,color='gray',ls='--',label='Pooled bound 0.5%');axes[1].set_xticks(np.arange(3),['Replica1','Replica2','Replica3']);axes[1].set_ylabel('Cost gain vs trained myopic (%)');axes[1].set_title('Additional guard: each replica >0');axes[1].legend(fontsize=7)
    for offset,key,label,color in ((-.17,'candidate_action_outside_myopic_map','Myopic future','#1565C0'),(.17,'candidate_action_outside_reverse_map','Myopic reverse','#E65100')):
        axes[2].bar(np.arange(3)+offset,[100*mechanism[str(t)][key]/mechanism[str(t)]['states'] for t in trains],width=.32,label=label,color=color)
    axes[2].set_xticks(np.arange(3),['Replica1','Replica2','Replica3']);axes[2].set_ylabel('Candidate actions outside knockout MAP (%)');axes[2].set_title('Same weights: descriptive, excludes near ties');axes[2].legend(fontsize=7);fig.suptitle('Learned continuation mechanism — retrained control is the primary test');fig.savefig(out/'learning_mechanism.png',dpi=170);plt.close(fig)
    passed=sum(s['checks'].values());failed=[k for k,ok in s['checks'].items() if not ok];extra=[k for k,ok in s['extra_checks'].items() if not ok];verdict='ĐẠT DEV; cần confirmation riêng trước kết luận khoa học' if s['development_screen_pass'] else 'KHÔNG ĐẠT DEV; giữ kết quả âm, chưa mở confirmation/test'
    lines=['# Typed-capacity Gibbs PPO v1: phương pháp và kết quả N4/K2','',f'**{verdict}.** {passed}/12 tiêu chí điểm đạt. Mục tiêu nghiên cứu tổng thể chưa đạt.','',
    '## Giả thuyết và kiến trúc','',
    'Các utility phân công của những máy phía sau có thể giúp phối hợp việc dùng technician tốt hơn việc chỉ đếm số matching còn hợp lệ. Mỗi máy phát hai điểm u(s,m,type), được học từ reward của chính policy. Broker tính xác suất của matching M theo exp(Σu)/Z bằng quy hoạch động trên số technician còn rảnh của hai loại. Đây là điểm policy, không phải future-cost oracle hay Q-label từ PPO gốc.','',
    'Giữ PPO scalar joint chuẩn và khởi tạo ngẫu nhiên. Không teacher, distillation, warm-start, nhãn PPO hay objective RL mới. Mạng raw89→64→64→2 dùng state71/own-machine6/type-profile6/machineID4/freecounts2; critic state71→64→64→1. Mỗi nhánh lõi có18.883 tham số, cùng tensor ban đầu nhưng bộ nhớ/optimizer riêng. Critic không chạy khi ra quyết định. Fullstate broadcast nên chưa phải local-only hay fully decentralized.','',
    'DP dùng F_i(a,b)=F_next(a,b)+a exp(u_i0)F_next(a−1,b)+b exp(u_i1)F_next(a,b−1), bỏ cạnh không hợp lệ. a/b đếm worker có nhãn; sau chọn type thì chọn đều một worker còn rảnh thuộc type đó. Exact log density có cả gradient qua learned suffix/logZ; entropy và physical MAP cũng tính chính xác. MAP dùng xác suất worker cụ thể, không chọn group lớn nhất.','',
    'Đối chứng chính dùng cùng mạng/utility/đầu vào/budget/decoder/entropy, nhưng future utilities được đặt0 trong thông điệp continuation; utility của lựa chọn hiện tại vẫn học. Đối chứng thêm đảo thứ tự máy của myopic. Cả ba bắt đầu uniform trên mọi matching hợp lệ,1/21 khi đủ N4/K2. Candidate có phân phối không phụ thuộc traversal khi giữ utility cố định; đây là thuộc tính toán học, chưa chứng minh cost tốt hơn.','',
    'Khác PPO tập trung gốc: PPO gốc có21 logits cho catalogue matching N4/K2. Candidate phát utility theo máy/type rồi tính joint policy bằng DP; vẫn học PPO từ đầu. DP có độ phức tạp O(N(K0+1)(K1+1)) khi cố định hai service profiles giống benchmark. Không tuyên bố đa thức khi mỗi technician có profile riêng; không generalization/scaling từ các kiểm tra tensor tổng hợp. Gibbs/count-DP/RL matching là prior art, xem tài liệu thiết kế.','',
    '## Protocol đã khóa','',
    '21fits=7nhánh×3trainingseeds mới,491.520 quyết định/fit; tổng10.321.920 quyết định,215.040 trainEP,161.280 Adamupdates.32DEVseeds,5checkpoints gồm0. PRIMARY cố định bước491.520, không chọn checkpoint hoặc replica tốt nhất. H48policy+12deadline tail chung; health tại cutoff48. FULL trực tiếp trên lab sau42tests/commit/push.12pointgates cùng dấu gain dương ở cả3replica,zeroinvalid và p95≤50ms. Không scientific test/CF/Nlớn ở vòng này.','',
    '## Kết quả fixed final','', '| Nhánh | Cost/máy H60 | Availability H48 | Pending/máy H48 | Failure/máy H48 |','|---|---:|---:|---:|---:|']
    for a in ARMS:
        m=s['means'][a];lines.append(f"| {NAMES[a]} | {m['normalized_cost']:.6f} | {m['availability']:.6f} | {m['normalized_pending']:.6f} | {m['normalized_failures']:.6f} |")
    lines+=['','| Tiêu chí | Giá trị điểm | Đạt |','|---|---:|---|']
    for k,value in s['endpoints'].items():lines.append(f"| {k} | {value:.8f} | {'Có' if s['checks'][k] else 'Không'} |")
    lines+=['',f"Gain vs myopic từng seed: {', '.join(f'{t}: {g:.4f}%' for t,g in zip(trains,per))}.",f"P95 actor+DP+decoder: {s['latency']['p95_ms']:.6f}ms,432mẫu sau khi pool đóng; chưa phải bằng chứng scale.",f"Không đạt: {', '.join(failed) or 'không'}. Extra không đạt: {', '.join(extra) or 'không'}.",'','## Cơ chế và sức khỏe','',
    f"Trên4.608quyết định candidate, có {opportunity['at_least_two_active_with_free_worker']} trạng thái≥2máy cần phục vụ và có worker rảnh; {opportunity['active_demand_exceeds_free_workers']} trạng thái có nhu cầu vượt số worker rảnh. Zero conflict của cả ba lõi là hệ quả hardmatching; không lấy đó làm bằng chứng learned continuation hữu ích.",'',
    '| Seed | Số trạng thái | Outside myopic MAP | Outside reverse MAP | TV điều kiện trung bình | Span future message trung bình |','|---|---:|---:|---:|---:|---:|']
    for t,m in mechanism.items():lines.append(f"| {t} | {m['states']} | {m['candidate_action_outside_myopic_map']} | {m['candidate_action_outside_reverse_map']} | {m['mean_conditional_tv']:.8f} | {m['mean_future_message_span']:.8f} |")
    lines+=['','Knockout giữ nguyên trọng số trên trajectory candidate và loại near ties; chỉ mô tả thông điệp ảnh hưởng phân phối/MAP như thế nào. Nó không ước lượng causal cost, không thay thế đối chứng huấn luyện độc lập, không thêm seed/update hay mở test.','',
    '| Nhánh | Maintenance/máy | Unavailability/máy | Failure-cost/máy | Waiting-cost/máy | Conflicts/EP | Rejected/EP |','|---|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        c=components[a];q=conflicts[a];lines.append(f"| {NAMES[a]} | {c['maintenance_cost']:.6f} | {c['unavailability_cost']:.6f} | {c['failure_cost']:.6f} | {c['waiting_cost']:.6f} | {q['broker_conflicts']:.6f} | {q['broker_rejected_proposals']:.6f} |")
    lines+=['','## Kiểm chứng và giới hạn','',
    'Audit độc lập dựng21bộ actor/critic ban đầu và các RNGroles; kiểm tra đủ215.040trainEP/3.360DEVnodes/fixedfinal. Replay704episode/42.240primitive transitions/33.792policy decisions. NumPy dựng utility từ raw features; enumeration21matching kiểm tra jointdensity/entropy/physicalMAP, tách khỏi productionDP. Tính lại12gates,3replica signs và432raw timings. Không lưu đủ historical optimizer minibatches/gradients nên không tuyên bố replay từng optimizer update.','',
    'Đây là kết quả DEV mô tả, không CI xác nhận hay bằng chứng ưu thế/scaling. Nếu không đủ mọi gate thì giữ nguyên verdict; không đổi control/checkpoint/seed thành thuật toán thắng hoặc thêm component để cứu kết quả. Các kết quả âm trước đây vẫn nguyên vẹn. Goal còn active.','',f'Nguồn: `{source}`.',f'Audit: `{verification}`.','', '![Chất lượng và sức khỏe](quality_health.png)','![Learning và cơ chế](learning_mechanism.png)','']
    (out/'findings_vi.md').write_text('\n'.join(lines));shutil.copy2(__file__,out/'reporter.py');manifest=dict(status='COMPLETED',source_manifest_sha256=sha(source/'manifest.json'),verification_manifest_sha256=sha(verification/'manifest.json'),goal_pass=False,test_opened=False,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.verification,a.output)
