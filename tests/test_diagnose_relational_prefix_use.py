"""Safety guards for the DEV-only same-weight B2 diagnostic."""

import json
from pathlib import Path

import pytest

from scripts import diagnose_relational_prefix_use as diagnosis


def write_fixture(tmp_path, *, test_opened=False, audit_hash=None):
    source=tmp_path/'source';source.mkdir()
    source_manifest=dict(status='COMPLETED',audit_status='PASS',
        profile='development',test_opened=test_opened)
    (source/'manifest.json').write_text(json.dumps(source_manifest))
    audit=tmp_path/'audit';audit.mkdir()
    report=dict(status='PASS',verdict='DEV_GATE_FAIL',test_opened=False,
        source_artifact=str(source.resolve()),
        source_manifest_sha256=audit_hash or diagnosis.sha(source/'manifest.json'))
    (audit/'audit.json').write_text(json.dumps(report))
    return source,audit,tmp_path/'output'


def test_rejects_test_opened_source(tmp_path):
    source,audit,output=write_fixture(tmp_path,test_opened=True)
    with pytest.raises(AssertionError):
        diagnosis.run(source,audit,output,'smoke')
    assert json.loads((output/'manifest.json').read_text())['status']=='FAILED'


def test_rejects_independent_audit_hash_mismatch(tmp_path):
    source,audit,output=write_fixture(tmp_path,audit_hash='wrong')
    with pytest.raises(AssertionError):
        diagnosis.run(source,audit,output,'smoke')
    assert json.loads((output/'manifest.json').read_text())['status']=='FAILED'
