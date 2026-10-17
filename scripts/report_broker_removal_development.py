"""Descriptive removal PPO DEV report; requires independent completed verification."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CORE=('removal_rb_ppo','factor_state_ppo','blind_removal_ppo','proposal_removal_ppo','proposal_state_ppo','joint_state_ppo')
ARMS=(*CORE,'ppo','normalized_ppo','capacity_ppo','mappo','full_mappo')
LABELS=('Removal RB','Factor + state V','Blind removal','Proposal + removal','Proposal + state V','Joint q + state V','PPO','PPO normalized','PPO + shadow critic','MAPPO local','MAPPO full')
NAME=dict(zip(ARMS,LABELS));REFS=('ppo','normalized_ppo','capacity_ppo')
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def check(ok,label):
    if not ok:raise AssertionError(label)


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    sm=read(source/'manifest.json');vm=read(verification/'manifest.json');v=read(verification/'verification.json');check(sm['status']=='COMPLETED' and sm['audit_status']=='PASS' and vm['status']=='COMPLETED' and v['status']=='PASS' and v['source']==str(source) and v['source_manifest_sha256']==sha(source/'manifest.json') and not sm['test_opened'],'independently verified closed-test DEV')
    for name,meta in sm['output_bindings'].items():check(sha(source/name)==meta['sha256'],'source binding '+name)
    for name,meta in vm['outputs'].items():check(sha(verification/name)==meta['sha256'],'verifier binding '+name)
    s=read(source/'summary.json');check(s['checks']==v['independent_checks'] and s['development_screen_pass']==v['original_development_pass'],'unmodified independently recomputed verdict')
    dev=rows(source/'development_episodes.csv');selected=rows(source/'selected_episodes.csv');progress=rows(source/'training_progress.csv');trains=sorted({int(r['train_seed']) for r in dev});steps=sorted({int(r['checkpoint_steps']) for r in dev});curves=[];final={};diagnostics={}
    for a in ARMS:
        final[a]={str(t):float(np.mean([float(r['normalized_cost']) for r in selected if r['method']==a and int(r['train_seed'])==t])) for t in trains}
        for step in steps:
            curves.append(dict(method=a,steps=step,cost=float(np.mean([float(r['normalized_cost']) for r in dev if r['method']==a and int(r['checkpoint_steps'])==step]))))
        if a in CORE or a=='capacity_ppo':
            late=[r for r in progress if r['algorithm']==a and int(r['learned_steps'])>368640];diagnostics[a]={k:float(np.mean([float(r[k]) for r in late])) for k in ('baseline_context_sensitivity_over_sd','baseline_delta_over_sd','fallback_mass','actor_update_norm','critic_update_norm')};diagnostics[a]['mean_residual_variance_ratio']=float(np.mean([float(r['removal_residual_variance'])/float(r['state_residual_variance']) for r in late]))
    components={a:{k:float(np.mean([float(r[k])/4 for r in selected if r['method']==a])) for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost')} for a in ARMS}
    data=dict(classification='DESCRIPTIVE_DEV_ONLY',source=str(source),verification=str(verification),summary=s,curves=curves,final_cost_by_replica=final,late_rollout_diagnostics=diagnostics,final_cost_components_per_machine=components,limitation='Three DEV replicas only; plotted dots/ranges are not confidence intervals. Fixed final491520 primary. No optimizer replay/confirmation/test/scaling claim.')
    (out/'analysis.json').write_text(json.dumps(data,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained');y=np.arange(len(ARMS));means=[s['means'][a]['normalized_cost'] for a in ARMS]
    axes[0].barh(y,means,color=['#1565C0']+['#B0BEC5']*10)
    for j,t in enumerate(trains):axes[0].scatter([final[a][str(t)] for a in ARMS],y+(j-1)*.12,s=12,color=['#212121','#E65100','#6A1B9A'][j],label=f'Replica {j+1}',zorder=3)
    axes[0].set_yticks(y,LABELS);axes[0].invert_yaxis();axes[0].set_xlabel('Mean H60 cost / machine');axes[0].set_title('Fixed final491520; dots = replica means');axes[0].legend(fontsize=7)
    gains=[100*s['endpoints']['gain_vs_'+a] for a in ARMS[1:]];threshold=[.5 if a==CORE[1] else 0 if a in ('blind_removal_ppo','mappo','full_mappo') else -1 for a in ARMS[1:]]
    axes[1].barh(np.arange(10),gains,color=['#2E7D32' if s['checks']['gain_vs_'+a] else '#C62828' for a in ARMS[1:]]);axes[1].scatter(threshold,np.arange(10),marker='|',s=160,c='black',label='Locked bound');axes[1].set_yticks(np.arange(10),LABELS[1:]);axes[1].invert_yaxis();axes[1].axvline(0,color='gray',lw=.6);axes[1].set_xlabel('Candidate cost gain vs control (%)');axes[1].set_title('10 cost gates: green passes, red fails');axes[1].legend(fontsize=8)
    health=[];labels=[];colors=[]
    for a in REFS:
        for metric,limit in (('availability',-.01),('normalized_pending',.025),('normalized_failures',.25)):
            key=metric+'_delta_vs_'+a;health.append(s['endpoints'][key]/limit);labels.append({'availability':'Avail','normalized_pending':'Pending','normalized_failures':'Failures'}[metric]+' / '+NAME[a]);colors.append('#2E7D32' if s['checks'][key] else '#C62828')
    axes[2].barh(np.arange(9),health,color=colors);axes[2].set_yticks(np.arange(9),labels);axes[2].invert_yaxis();axes[2].axvline(1,color='black',ls='--',label='Maximum harm');axes[2].axvline(0,color='gray',lw=.6);axes[2].set_xlabel('Health harm / locked allowance');axes[2].set_title('9 health gates; lower is better');axes[2].legend(fontsize=8)
    fig.suptitle('Removal-credit DEV — descriptive results, no confidence intervals',fontsize=13);fig.savefig(out/'fixed_final_quality_health.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=dict(zip(ARMS,plt.cm.tab20(np.arange(11))));focus=(*CORE[:3],'normalized_ppo','capacity_ppo','full_mappo')
    for a in focus:axes[0,0].plot(np.array(steps)/1000,[r['cost'] for r in curves if r['method']==a],label=NAME[a],color=colors[a],marker='.',lw=1.3)
    axes[0,0].set_title('Complete registered DEV learning interval');axes[0,0].set_ylabel('Mean H60 cost / machine');axes[0,0].set_xlabel('Training decisions (thousands)');axes[0,0].legend(fontsize=7)
    per=[100*s['per_replica_main_gain'][str(t)] for t in trains];axes[0,1].bar(range(3),per,color=['#2E7D32' if g>0 else '#C62828' for g in per]);axes[0,1].axhline(0,color='black',lw=.7);axes[0,1].axhline(.5,color='gray',ls='--',label='Pooled practical bound0.5%');axes[0,1].set_xticks(range(3),['Replica1','Replica2','Replica3']);axes[0,1].set_ylabel('Candidate gain vs factor-state (%)');axes[0,1].set_title('Additional guard: each replica strictly positive');axes[0,1].legend(fontsize=7)
    for a in CORE[:3]:
        rr=[r for r in progress if r['algorithm']==a];grid=sorted({int(r['learned_steps']) for r in rr})
        for ax,key in ((axes[1,0],'baseline_context_sensitivity_over_sd'),(axes[1,1],'residual_ratio')):
            values=[np.mean([float(r['removal_residual_variance'])/float(r['state_residual_variance']) if key=='residual_ratio' else float(r[key]) for r in rr if int(r['learned_steps'])==step]) for step in grid];ax.plot(np.array(grid)/1000,values,label=NAME[a],color=colors[a],lw=1)
    axes[1,0].set_title('Context sensitivity: mean |b(context)-b(empty)| / SD');axes[1,0].set_ylabel('Training diagnostic');axes[1,1].set_title('Var(G-actor baseline) / Var(G-state V)');axes[1,1].axhline(1,color='gray',ls='--');axes[1,1].set_ylabel('Residual variance ratio')
    for ax in axes[1]:ax.set_xlabel('Training decisions (thousands)');ax.legend(fontsize=7)
    fig.suptitle('Mechanism diagnostics — observational traces; ablation cost is the outcome',fontsize=12);fig.savefig(out/'learning_and_mechanism.png',dpi=170);plt.close(fig)
    passed=sum(s['checks'].values());failed=[k for k,ok in s['checks'].items() if not ok];extra_failed=[k for k,ok in s['extra_checks'].items() if not ok];verdict='ĐẠT vòng sàng lọc DEV; cần confirmation mới trước mọi kết luận khoa học' if s['development_screen_pass'] else 'KHÔNG ĐẠT vòng sàng lọc DEV; giữ nguyên kết quả âm và chưa mở confirmation/test'
    lines=['# Removal-credit PPO v1: phương pháp và kết quả DEV','',f'**{verdict}.** {passed}/19 tiêu chí điểm đạt. Đây là kết quả mô tả trên DEV, chưa có khoảng tin cậy và chưa đạt mục tiêu nghiên cứu tổng thể.','',
    '## Giả thuyết và thuật toán','',
    'Giả thuyết: thông tin về đề xuất của các technician còn lại giúp baseline phân bổ tín hiệu học tốt hơn, qua đó giảm cost so với cùng policy chỉ dùng V(s). Tất cả actor và critic học từ khởi tạo ngẫu nhiên; không teacher, distillation hay trọng số PPO đã huấn luyện.','',
    'Mỗi technician đề xuất WAIT hoặc một máy từ cùng trạng thái thô đầy đủ. Broker cố định chọn technician có restoration/duration tốt hơn khi tranh chấp, hòa thì chọn ID nhỏ hơn; người thua chờ và không thử lại. Policy lõi không liệt kê toàn bộ matching.','',
    'Critic ψ dự đoán return thực của policy từ trạng thái và matching khi thay đề xuất technician i bằng WAIT. Đây là baseline dùng return thực để học; không mô phỏng một thế giới thiếu technician và không dùng nhãn cost tương lai. Lấy kỳ vọng baseline trên những đề xuất ẩn cùng tạo ra matching đã quan sát để được B_i. Tính và detach B_i trước optimizer của batch hiện tại, giữ nguyên qua các epoch PPO.','',
    'Actor dùng ratio theo từng thừa số của xác suất matching vật lý và advantage riêng (G−B_i), với clip PPO và chuẩn hóa chung dựa trên G−V(s). Đây là surrogate suy ra từ PPO; không tuyên bố giữ nguyên scalar joint PPO hay đảm bảo cải thiện đơn điệu. Critic phụ chỉ dùng trong huấn luyện, suy luận dùng actor và broker.','',
    'Sáu nhánh lõi có cùng actor, hai critic, thông tin, tham số29383, khởi tạo và ngân sách. Đối chứng chính factor-state vẫn huấn luyện critic phụ nhưng actor chỉ dùng V(s); blind-removal xóa kênh matching khi học và truy vấn. Các nhánh proposal và joint kiểm tra cách gán log-probability. Capacity PPO giữ actor21matching của PPO normalized và thêm critic phụ; vẫn có khác biệt số tham số28375 so với29383. MAPPO local khác thông tin; MAPPO full nhận đầy đủ trạng thái.','',
    '## Thiết kế đã khóa','',
    'N4/K2, H48 quyết định và H12 tail chung; ba seed huấn luyện mới ×11 nhánh, mỗi mô hình491520 quyết định. Tổng16220160 quyết định,337920 episode và253440 Adam update. Đánh giá32 DEV seeds ở năm checkpoint gồm bước0. Kết luận chính dùng cố định checkpoint491520, không dùng checkpoint tốt nhất. 19 tiêu chí gồm cải thiện≥0.5% so với factor-state, cải thiện dương so với blind/local/full MAPPO, không kém hơn1% so với các đối chứng cost còn lại, cùng9 giới hạn sức khỏe. Thêm điều kiện cả ba seed thắng đối chứng chính, không assignment sai và p95≤50ms.','',
    '## Kết quả cố định cuối kỳ','',
    '| Nhánh | Cost/máy H60 | Availability H48 | Pending/máy H48 | Failure/máy H48 |','|---|---:|---:|---:|---:|']
    for a in ARMS:
        m=s['means'][a];lines.append(f"| {NAME[a]} | {m['normalized_cost']:.6f} | {m['availability']:.6f} | {m['normalized_pending']:.6f} | {m['normalized_failures']:.6f} |")
    lines+=['','| Tiêu chí | Giá trị điểm | Đạt |','|---|---:|---|']
    for k,value in s['endpoints'].items():lines.append(f"| {k} | {value:.8f} | {'Có' if s['checks'][k] else 'Không'} |")
    lines+=['',f"Gain chính từng seed: {', '.join(f'{t}: {g:.6f}%' for t,g in zip(trains,per))}.",f"p95 actor+broker: {s['latency']['p95_ms']:.6f}ms, đo432 lượt sau khi worker huấn luyện đóng. Đây là độ trễ trên máy lab N4, chưa chứng minh scaling.",'',f"Tiêu chí điểm không đạt: {', '.join(failed) or 'không'}. Điều kiện bổ sung không đạt: {', '.join(extra_failed) or 'không'}.",'',
    '## Bằng chứng và giới hạn','',
    'Bộ kiểm chứng độc lập tái dựng33 bộ tensor khởi tạo, ngân sách và seed; replay1088 episode/65280 chuyển trạng thái vật lý và52224 quyết định policy; kiểm tra actor bằng NumPy, baseline conditional bằng tổng đủ25 cặp đề xuất, cùng19 phép tính tiêu chí và432 mẫu timing. Audit PASS xác nhận tính đúng của báo cáo, không biến một thí nghiệm không đạt thành thành công. Các minibatch và tensor cache lịch sử không được lưu đầy đủ nên chưa replay độc lập từng optimizer update.','',
    'Đồ thị residual variance, context sensitivity và fallback mass là chẩn đoán mô tả. Chúng không tự chứng minh cơ chế có ích; bằng chứng chính phải đến từ ablation cùng thông tin/ngân sách và kết quả cost. Trọng số matching khác0 cũng chưa đủ kết luận phối hợp thành công.','',
    'Không có CI trên test mới, không tuyên bố thắng PPO về hiệu năng, không đánh giá generalization hay mở rộng N/K ở vòng này. Nếu DEV không đạt, không tự động thử lại learning rate/entropy/seed/budget; vòng tiếp theo cần nguyên nhân cụ thể và protocol mới trước khi chạy. Các kết quả âm trước đây vẫn giữ nguyên.','',
    f'Artifact nguồn: `{source}`.',f'Audit độc lập: `{verification}`.','',
    '![Chất lượng và sức khỏe](fixed_final_quality_health.png)','![Learning và cơ chế](learning_and_mechanism.png)','']
    (out/'findings_vi.md').write_text('\n'.join(lines));shutil.copy2(__file__,out/'reporter.py');manifest=dict(status='COMPLETED',classification='DESCRIPTIVE_DEV_REPORT',source_manifest_sha256=sha(source/'manifest.json'),verification_manifest_sha256=sha(verification/'manifest.json'),goal_pass=False,test_opened=False,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.verification,a.output)
