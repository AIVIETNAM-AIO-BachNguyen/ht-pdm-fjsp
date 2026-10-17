import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('scalar',Path(__file__).resolve().parents[1]/'scripts/broker_removal_scalar_bound.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def test_exact_quadratic_minimizer_beats_fixed_coefficient_without_sweep():
    x=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]);h=x*np.array([.5,0]);r=s.scalar_bound(x,x-h);assert r['coefficient']==pytest.approx(2.) and r['optimistic_in_sample_gain']==pytest.approx(.5)
    direct=lambda coefficient:np.square((x-coefficient*h)-(x-coefficient*h).mean(0)).sum()/3
    assert direct(r['coefficient'])<direct(1) and direct(2-1e-5)==pytest.approx(direct(2+1e-5))


def test_zero_control_and_negative_optimal_coefficient():
    x=np.array([[1.,2.],[3.,4.],[5.,6.]]);r=s.scalar_bound(x,x);assert r['zero_control'] and r['coefficient']==0 and r['optimistic_in_sample_gain']==0
    r=s.scalar_bound(x,x*2);assert r['coefficient']==pytest.approx(-1.) and r['optimistic_in_sample_gain']==pytest.approx(1.)


def test_invalid_samples_rejected():
    with pytest.raises(AssertionError):s.scalar_bound([[1.,2.]],[[1.,2.]])
    with pytest.raises(AssertionError):s.scalar_bound([[1.],[np.nan]],[[0.],[0.]])
