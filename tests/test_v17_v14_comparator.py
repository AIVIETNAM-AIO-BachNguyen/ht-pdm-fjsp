import numpy as np
from research_result.run_v17_v14_comparator import bootstrap,cost_tensor


def test_independent_training_seed_bootstrap_preserves_constant_cost_gap():
    candidate=np.full((5,16,4),10.0)
    comparator=np.full((5,16,4),15.0)
    np.testing.assert_allclose(bootstrap(candidate,comparator),[-5.0,-5.0])


def test_cost_tensor_accepts_in_memory_integer_and_csv_string_seeds():
    rows=[dict(controller='competition',train_seed=72600000,case='reciprocal_scale2',seed=73011000,objective=7),
          dict(controller='competition',train_seed='72600000',case='reciprocal_scale2',seed='73011001',objective='9')]
    np.testing.assert_array_equal(cost_tensor('competition',[72600000],rows,['reciprocal_scale2'],[73011000,73011001]),
                                  np.array([[[7.],[9.]]]))
