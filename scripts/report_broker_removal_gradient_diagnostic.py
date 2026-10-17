"""Descriptive report of independently verified frozen actor-gradient diagnostics."""
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
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);sm=read(source/'manifest.json');vm=read(verification/'manifest.json');v=read(verification/'verification.json');s=read(source/'summary.json');assert sm['status']=='COMPLETED' and vm['status']=='COMPLETED' and v['status']=='PASS' and v['source_manifest_sha256']==sha(source/'manifest.json') and v['source']==str(source) and v['checks']==s['checks'] and not s['goal_pass']
    for root,bindings in ((source,sm['output_bindings']),(verification,vm['outputs'])):
        for name,meta in bindings.items():assert sha(root/name)==meta['sha256'],name
    records=s['records'];trains=sorted({r['train_seed'] for r in records});fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained');colors=['#1565C0','#EF6C00','#6A1B9A']
    for t,color in zip(trains,colors):
        rr=sorted([r for r in records if r['train_seed']==t],key=lambda r:r['steps']);x=[r['steps']/1000 for r in rr];label=str(t);axes[0].plot(x,[100*r['gain_vs_state'] for r in rr],color=color,marker='o',label=label);axes[1].plot(x,[100*r['gain_vs_proposal'] for r in rr],color=color,marker='o',label=label);axes[2].scatter([r['scalar_residual_ratio'] for r in rr],[1-r['gain_vs_state'] for r in rr],color=color,s=35,label=label)
    axes[0].axhline(0,color='gray',lw=.7);axes[0].axhline(5,color='black',ls='--',label='5% material reference');axes[0].set_title('Conditional removal vs same actor + state V');axes[0].set_ylabel('Actor-gradient trace covariance reduction (%)');axes[1].set_title('Conditional removal vs proposal + actual b');axes[1].set_ylabel('Rao-Blackwell variance reduction (%)');axes[1].axhline(0,color='gray',lw=.7)
    for ax in axes[:2]:ax.set_xlabel('Frozen checkpoint (thousand training decisions)');ax.legend(fontsize=7)
    axes[2].axhline(1,color='gray',lw=.7);axes[2].axvline(1,color='gray',lw=.7);axes[2].set_xlabel('Scalar return-residual variance ratio');axes[2].set_ylabel('Actor-gradient trace covariance ratio');axes[2].set_title('Lower scalar residual is not sufficient');axes[2].legend(fontsize=7);fig.suptitle('Frozen F64 diagnostic — 128 independent episodes/node; point estimates, no CI');fig.savefig(out/'gradient_variance.png',dpi=170);plt.close(fig)
    lines=['# Removal-credit: chẩn đoán phương sai gradient actor','',f"**{'ĐẠT' if s['mechanism_screen_pass'] else 'KHÔNG ĐẠT'} ngưỡng chẩn đoán đã khóa.** Kết quả cost âm của FULL trước đó được giữ nguyên. Đây là phân tích mô tả, chưa có CI và không chứng minh một cập nhật PPO hay cost tốt hơn.",'','## Câu hỏi và thiết kế','',
    'Trên cùng policy đóng băng, baseline removal có giảm phương sai gradient của toàn bộ actor dùng chung, thay vì chỉ giảm phần dư return? Dùng tất cả ba seed và năm checkpoint của candidate, mỗi nhóm128 episode mới. Tổng1.920 episode/92.160 quyết định/115.200 chuyển trạng thái; 0 cập nhật trọng số. Tensor checkpoint float32 được giữ nguyên giá trị khi chuyển float64 để lấy mẫu và đo gradient. Không teacher hay fine-tuning actor.','',
    'Trên từng episode tính ba vector gradient11.141 chiều: physical factors với V(s), physical factors với conditional removal B_i, và proposal scores với actual removal b_i. Mọi vector dùng cùng quỹ đạo và return thật, chia chung48×2; không clipping, entropy hay chuẩn hóa advantage. Đơn vị thống kê là episode, không coi48 tick tương quan là mẫu độc lập.','',
    'Tiêu chí chính: mỗi seed giảm trung bình ít nhất5% trace covariance trên đủ năm checkpoint và cải thiện ở ít nhất3/5 checkpoint. Step0 có V=B theo thiết kế, được kiểm tra và đặt gain chính bằng0 để sai số làm tròn không được tính là học thành công.','',
    '## Kết quả','', '| Seed | Giảm phương sai trung bình | Checkpoint cải thiện | Đạt cả hai |','|---|---:|---:|---|']
    for t in trains:
        r=s['replicas'][str(t)];ok=all(s['checks'][str(t)].values());lines.append(f"| {t} | {100*r['mean_gain']:.4f}% | {r['positive_checkpoints']}/5 | {'Có' if ok else 'Không'} |")
    lines+=['','| Seed | Checkpoint | Gain gradient B/V | Gain B/proposal-b | Ratio phần dư return |','|---|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['train_seed']} | {r['steps']} | {100*r['gain_vs_state']:.4f}% | {100*r['gain_vs_proposal']:.4f}% | {r['scalar_residual_ratio']:.6f} |")
    lines+=['','## Diễn giải có giới hạn','',
    'Chỉ một trong ba seed đạt mức giảm phương sai trung bình5%. Ở seed330000000, hai checkpoint cuối có phần dư return nhỏ hơn nhưng phương sai gradient lớn hơn khoảng1,30% và1,78%. Vì vậy việc giảm MSE/variance của return không tự động đem lại tín hiệu actor tốt hơn. Đây là bằng chứng mô tả trên panel này, không phải chứng minh tổng quát hay kết luận mọi baseline removal đều vô ích.','',
    'So sánh B với proposal-b có mức giảm lớn hơn, khoảng12,8–79,0% tùy checkpoint. Đó là hiệu ứng của marginalization/Rao-Blackwell so với proposal estimator; đối chứng chính của nghiên cứu đã dùng physical factors. Không được lấy lợi ích này thay cho đóng góp thêm của critic mới so với factor-state.','',
    'Không thấy lỗi toán học trong phép đo: kiểm chứng độc lập tái dựng toàn bộ115.200 chuyển trạng thái và return, mọi action draw/baseline, cùng5.760 vector gradient (64.172.160 tọa độ). Reference dùng MLP thủ công và tổng25 latent proposals, chỉ cho gradient đi qua xác suất của technician đang xét; khác với VJP NumPy đóng của lần chạy nguồn. Audit PASS xác nhận phép đo, không thay đổi verdict âm.','',
    'Việc học baseline theo gradient variance đã có prior art: [Greensmith et al., JMLR 2004](https://www.jmlr.org/papers/v5/greensmith04a.html) phân tích baseline/value function tối ưu cho variance; [Kuba et al., NeurIPS 2021](https://papers.nips.cc/paper_files/paper/2021/hash/6fe6a8a6e6cb710584efc4af0c34ce50-Abstract.html) nghiên cứu optimal baseline trong MARL. Do đó không coi việc thêm score weight hay đổi critic loss đơn thuần là phát minh mới.','',
    'Hướng cần kiểm tra trước một vòng học mới: trong cấu trúc broker và actor dùng chung này, baseline nào giảm được trace covariance trên quỹ đạo chưa dùng để fit? Phải tách fit/đánh giá và dùng đối chứng cùng thông tin/capacity; không chỉ giảm variance trên dữ liệu fit. Chưa có thí nghiệm mới cho hướng đó ở báo cáo này. Không tự động mở CF/test/N lớn, tăng budget, chọn seed hay checkpoint khác.','',f'Nguồn: `{source}`.',f'Kiểm chứng: `{verification}`.','', '![Phương sai gradient](gradient_variance.png)','']
    (out/'findings_vi.md').write_text('\n'.join(lines));(out/'analysis.json').write_text(json.dumps(dict(summary=s,verification=v,classification='DESCRIPTIVE_GRADIENT_DIAGNOSTIC_ONLY',goal_pass=False,test_opened=False),indent=2));shutil.copy2(__file__,out/'reporter.py');manifest=dict(status='COMPLETED',source_manifest_sha256=sha(source/'manifest.json'),verification_manifest_sha256=sha(verification/'manifest.json'),goal_pass=False,test_opened=False,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.verification,a.output)
