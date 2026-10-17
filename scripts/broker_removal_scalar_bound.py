"""Post-hoc in-sample analytic control-variate ceiling, no policy optimization."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def scalar_bound(state,conditional):
    x=np.asarray(state,float);h=x-np.asarray(conditional,float);assert x.ndim==2 and x.shape==h.shape and len(x)>1 and np.isfinite(x).all() and np.isfinite(h).all();xc=x-x.mean(0);hc=h-h.mean(0);n=len(x)-1;v=float(np.square(xc).sum()/n);d=float(np.square(hc).sum()/n);c=float((xc*hc).sum()/n);assert v>1e-20
    coefficient=c/d if d>1e-20 else 0.;gain=c*c/(d*v) if d>1e-20 else 0.;mixed=xc-coefficient*hc;direct=float(np.square(mixed).sum()/n);assert np.isclose(direct,v*(1-gain),atol=1e-9,rtol=1e-10) and -.0000000001<=gain<=1.0000000001
    return dict(state_variance=v,control_variance=d,covariance=c,coefficient=coefficient,optimistic_in_sample_gain=gain,direct_mixed_variance=direct,zero_control=d<=1e-20)


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();sm=read(source/'manifest.json');vm=read(verification/'manifest.json');v=read(verification/'verification.json');summary=read(source/'summary.json');assert sm['status']=='COMPLETED' and vm['status']=='COMPLETED' and v['status']=='PASS' and v['source']==str(source) and v['source_manifest_sha256']==sha(source/'manifest.json') and not summary['mechanism_screen_pass'];bindings=(sha(source/'manifest.json'),sha(verification/'manifest.json'))
    def check_bindings():
        assert bindings==(sha(source/'manifest.json'),sha(verification/'manifest.json'))
        for root,entries in ((source,sm['output_bindings']),(verification,vm['outputs'])):
            for name,meta in entries.items():assert time.monotonic()-start<300 and sha(root/name)==meta['sha256'],name
    check_bindings();records=[]
    for record in tqdm(summary['records'],desc='15analytic scalar ceilings'):
        t,step=record['train_seed'],record['steps']
        with np.load(source/f'{t}_{step}'/'gradients.npz',allow_pickle=False) as f:bound=scalar_bound(f['state'],f['conditional'])
        records.append(dict(train_seed=t,steps=step,**bound))
    assert len(records)==15 and {(r['train_seed'],r['steps']) for r in records}=={(t,step) for t in range(330000000,330000003) for step in (0,122880,245760,368640,491520)};means={str(t):float(np.mean([r['optimistic_in_sample_gain'] for r in records if r['train_seed']==t])) for t in range(330000000,330000003)};eligible=all(g>=.05 for g in means.values());result=dict(classification='POST_HOC_OPTIMISTIC_IN_SAMPLE_CEILING_ONLY',source=str(source),verification=str(verification),records=records,replica_mean_optimistic_gain=means,eligible_for_separate_fresh_calibration_study=eligible,source_negative_verdict_overridden=False,goal_pass=False,test_opened=False,optimizer_updates=0,limitation='Evaluation-sample-fitted coefficient; optimistic sample ceiling only, not held-out benefit or population variance bound.')
    (out/'analysis.json').write_text(json.dumps(result,indent=2));lines=['# Giới hạn trộn scalar baseline: phân tích hậu kiểm','', '**Chỉ là giới hạn tối ưu trên chính mẫu đã có; không phải kết quả held-out hay cải thiện policy.** Công thức giải tích chọn một hệ số cho mỗi checkpoint, không quét hyperparameter. Các kết quả âm trước đây giữ nguyên.','', '| Seed | Mức giảm phương sai tối ưu trung bình đủ5 checkpoint |','|---|---:|']
    for t,g in means.items():lines.append(f'| {t} | {100*g:.4f}% |')
    lines+=['',('Đủ tiềm năng trên mẫu để cân nhắc protocol fit/holdout mới; chưa được học hay triển khai hệ số này.' if eligible else 'Ít nhất một seed không đạt5% ngay cả với hệ số tối ưu trên dữ liệu đánh giá. Không chạy thêm một sweep hệ số để cứu cấu hình này.'),'','| Seed | Checkpoint | Hệ số tối ưu | Giảm phương sai trên mẫu |','|---|---:|---:|---:|']
    for r in records:lines.append(f"| {r['train_seed']} | {r['steps']} | {r['coefficient']:.6f} | {100*r['optimistic_in_sample_gain']:.4f}% |")
    lines+=['','Đây là control-variate cổ điển, không phải đóng góp thuật toán mới. Fit và đánh giá dùng cùng dữ liệu nên hệ số có thể overfit; không dùng làm bằng chứng xác nhận cơ chế. Không mở scientific test, confirmation hay N lớn.','',f'Nguồn: `{source}`.',''];(out/'findings_vi.md').write_text('\n'.join(lines));shutil.copy2(__file__,out/'analysis.py');shutil.copy2(ROOT/'docs/technician_broker_removal_scalar_bound_v1_plan.md',out/'protocol.md');check_bindings();manifest=dict(status='COMPLETED',source_manifest_sha256=bindings[0],verification_manifest_sha256=bindings[1],elapsed_seconds=time.monotonic()-start,goal_pass=False,test_opened=False,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(output=str(out),means=means,eligible_for_fresh_calibration=eligible)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.verification,a.output)
