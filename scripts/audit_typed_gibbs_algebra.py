"""Fresh sealed engineering-only typed-Gibbs algebra audit, no physical states."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
FILES = ('docs/technician_typed_gibbs_design.md',
         'src/ht_pdm_fjsp/technician_typed_gibbs.py',
         'tests/test_typed_gibbs_algebra.py',
         'scripts/audit_typed_gibbs_algebra.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = dict(status='RUNNING', training_steps=0, test_states=0,
                    physical_states=0, scientific_seeds_constructed=False,
                    goal_pass=False, algebra_protocol='typed_gibbs_v1')
    def write():
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    write()
    try:
        git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
        assert not git('status', '--porcelain'), 'clean source gate'
        manifest['commit'] = git('rev-parse', 'HEAD')
        assert manifest['commit'] == git('rev-parse', '@{upstream}'), 'pushed source gate'
        for name in FILES:
            target = out/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        manifest['source_seal'] = {name: sha(ROOT/name) for name in FILES}
        write()
        import sys
        command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_typed_gibbs_algebra.py']
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
        (out/'pytest.stdout.txt').write_text(result.stdout)
        (out/'pytest.stderr.txt').write_text(result.stderr)
        assert result.returncode == 0, 'exact algebra suite failed; preserve failure'
        assert all(sha(ROOT/name) == digest for name,digest in manifest['source_seal'].items()), 'source changed'
        manifest.update(status='COMPLETED', audit_status='PASS', test_command=command,
                        test_exit_code=result.returncode, elapsed_seconds=time.monotonic()-start)
        write()
        print(json.dumps(dict(output=str(out),status='PASS',training_steps=0,goal_pass=False)))
    except Exception as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic()-start)
        write()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    run(parser.parse_args().output_dir)
