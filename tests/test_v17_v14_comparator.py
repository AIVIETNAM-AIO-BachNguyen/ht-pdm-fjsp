import numpy as np
from research_result.run_v17_v14_comparator import bootstrap


def test_independent_training_seed_bootstrap_preserves_constant_cost_gap():
    candidate=np.full((5,16,4),10.0)
    comparator=np.full((5,16,4),15.0)
    np.testing.assert_allclose(bootstrap(candidate,comparator),[-5.0,-5.0])
