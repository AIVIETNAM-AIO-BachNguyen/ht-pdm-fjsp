"""Descriptive fresh fit/heldout coefficient report, requires independent audit."""
import argparse,hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);sm=read(source/'manifest.json');vm=read(verification/'manifest.json');v=read(verification/'verification.json');s=read(source/'summary.json');assert sm['status']=='COMPLETED' and vm['status']=='COMPLETED' and v['status']=='PASS' and v['source_manifest_sha256']==sha(source/'manifest.json') and v['source']==str(source) and s['checks']==v['checks'] and not s['source_cost_verdict_overridden']
    for root,entries in ((source,sm['output_bindings']),(verification,vm['outputs'])):
        for name,meta in entries.items():assert sha(root/name)==meta['sha256'],name
    trains=sorted({r['train_seed'] for r in s['records']});fig,axes=plt.subplots(1,3,figsize=(15,4.9),layout='constrained');colors=['#1565C0','#EF6C00','#6A1B9A'];x=np.arange(3);fit=[100*np.mean([r['coefficient']['fit_optimistic_gain'] for r in s['records'] if r['train_seed']==t]) for t in trains];heldout=[100*s['replicas'][str(t)]['mean_gain'] for t in trains];fixed=[100*s['replicas'][str(t)]['mean_fixed_one_gain'] for t in trains]
    for shift,values,color,label in ((-.25,fit,'#90A4AE','Fit optimum'),(0,heldout,'#1565C0','Heldout calibrated'),(.25,fixed,'#EF6C00','Heldout fixed1')):axes[0].bar(x+shift,values,width=.24,color=color,label=label)
    axes[0].set_xticks(x,['Replica1','Replica2','Replica3']);axes[0].axhline(5,color='black',ls='--',label='Material bound5%');axes[0].axhline(0,color='gray',lw=.6);axes[0].set_title('Mean of ALL five frozen checkpoints');axes[0].set_ylabel('Actor-gradient variance reduction (%)');axes[0].legend(fontsize=7)
    for t,color in zip(trains,colors):
        rr=sorted([r for r in s['records'] if r['train_seed']==t],key=lambda r:r['steps']);steps=np.array([r['steps'] for r in rr])/1000;axes[1].plot(steps,[100*r['evaluation']['gain'] for r in rr],color=color,marker='o',label=str(t));axes[1].plot(steps,[100*r['evaluation']['fixed_one_gain'] for r in rr],color=color,ls='--',alpha=.5);axes[2].plot(steps,[r['coefficient']['coefficient'] for r in rr],color=color,marker='o',label=str(t))
    axes[1].axhline(0,color='gray',lw=.7);axes[1].set_title('Heldout: solid calibrated, dashed fixed1');axes[1].set_ylabel('Variance reduction (%)');axes[2].axhline(1,color='gray',ls='--');axes[2].axhline(0,color='gray',lw=.7);axes[2].set_title('Coefficient fitted before heldout collection');axes[2].set_ylabel('Fit-only scalar coefficient')
    for ax in axes[1:]:ax.set_xlabel('Frozen checkpoint (thousand decisions)');ax.legend(fontsize=7)
    fig.suptitle('Fresh calibration: 128 fit + 256 heldout episodes/node; frozen F64, no CI');fig.savefig(out/'fit_heldout_calibration.png',dpi=170);plt.close(fig)
    count=sum(ok for row in s['checks'].values() for ok in row.values());passed=s['mechanism_screen_pass'];lines=['# Hiệu chỉnh hệ số baseline: fit và heldout độc lập','',f"**{'ĐẠT' if passed else 'KHÔNG ĐẠT'} vòng chẩn đoán: {count}/9 điều kiện.** Kết quả FULL cost âm trước đó được giữ nguyên; goal nghiên cứu chưa đạt.",'','## Giả thuyết và thiết kế','',
    'Giả thuyết: hệ số1 chưa phù hợp với gradient actor dùng chung; một hệ số control-variate ước lượng từ dữ liệu riêng có thể giảm nhiễu trên quỹ đạo mới. Công thức hệ số control-variate là prior art, không phải phát minh mới.','',
    'Dùng candidate đóng băng ở tất cả ba seed và năm checkpoint. Mỗi nhóm lấy128 episode fit, tính lambda=traceCov(gV,h)/traceVar(h), với h=gV−gB; đóng băng và hash lambda trước khi thu256 episode heldout khác. Hệ số không bị clip, regularize, quét hay chọn lại trên heldout. Nếu control variance≤1e−20 thì lambda0. Tất cả trọng số actor/critic giữ nguyên.','',
    'Tổng5.760 episode,276.480 quyết định,345.600 chuyển trạng thái và17.280 vector gradient đầy đủ; 0 optimizer update. Physical/action seeds mới và hai tập tách rời. Vẫn N4/K2, H48+tail12 và reward/health ban đầu. F64 giữ nguyên giá trị tensor checkpoint.','',
    'Mỗi seed phải có mean gain trên đủ năm checkpoint≥5%, ít nhất3/5 checkpoint cải thiện, và mean gain tốt hơn baseline hệ số1 trên cùng heldout trajectories. Step0 được kiểm tra V=B, hệ số0 và gain chính0. Đây là chẩn đoán điểm, chưa có CI hay bằng chứng về cập nhật PPO/cost.','',
    '## Kết quả heldout','', '| Seed | Gain hiệu chỉnh | Gain hệ số1 | Checkpoint cải thiện | Đạt cả3 |','|---|---:|---:|---:|---|']
    for t in trains:
        r=s['replicas'][str(t)];ok=all(s['checks'][str(t)].values());lines.append(f"| {t} | {100*r['mean_gain']:.4f}% | {100*r['mean_fixed_one_gain']:.4f}% | {r['positive_checkpoints']}/5 | {'Có' if ok else 'Không'} |")
    lines+=['','| Seed | Checkpoint | Lambda từ fit | Gain trên fit | Gain heldout | Gain fixed1 heldout |','|---|---:|---:|---:|---:|---:|']
    for r in s['records']:lines.append(f"| {r['train_seed']} | {r['steps']} | {r['coefficient']['coefficient']:.6f} | {100*r['coefficient']['fit_optimistic_gain']:.4f}% | {100*r['evaluation']['gain']:.4f}% | {100*r['evaluation']['fixed_one_gain']:.4f}% |")
    lines+=['','## Kết luận có giới hạn','',('Dữ liệu heldout ủng hộ việc nghiên cứu tiếp cơ chế hiệu chỉnh trong một protocol học từ đầu riêng. Chưa có bằng chứng nó giúp cost, PPO với clipping/normalization hay generalization. Mọi mô hình học tiếp phải khởi tạo ngẫu nhiên và có đối chứng cùng thông tin/critic/compute; checkpoint đóng băng ở đây chỉ dùng chẩn đoán.' if passed else 'Hướng scalar calibration không vượt đủ điều kiện trên dữ liệu mới. Theo protocol, loại cấu hình này; không thử lại bằng clip/regularization, thêm episode, đổi seed hay chọn checkpoint tốt. Một mức giảm trên tập fit không đủ để cứu kết quả heldout.'),'',
    'Bộ kiểm chứng độc lập replay toàn bộ345.600 chuyển trạng thái, action draws, returns/baselines và17.280 vector gradient (192.516.480 tọa độ); tính lại hệ số chỉ từ fit và áp dụng đúng hệ số ấy trên heldout. Audit PASS xác nhận các số liệu, không thay cho thành công khoa học. Scientific confirmation/test và N lớn vẫn đóng.','',f'Nguồn: `{source}`.',f'Kiểm chứng: `{verification}`.','', '![Fit và heldout](fit_heldout_calibration.png)','']
    (out/'findings_vi.md').write_text('\n'.join(lines));(out/'analysis.json').write_text(json.dumps(dict(summary=s,verification=v,goal_pass=False,test_opened=False),indent=2));shutil.copy2(__file__,out/'reporter.py');manifest=dict(status='COMPLETED',source_manifest_sha256=sha(source/'manifest.json'),verification_manifest_sha256=sha(verification/'manifest.json'),goal_pass=False,test_opened=False,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.verification,a.output)
