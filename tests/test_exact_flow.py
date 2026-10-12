from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/verl/utils/bert_emd.py"
SPEC = spec_from_file_location("bert_emd", MODULE_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
exact_flow = MODULE.exact_flow
quantize_marginal = MODULE.quantize_marginal

SOLVER = {
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
    "absolute_residual_tolerance": 1e-8,
    "relative_residual_tolerance": 1e-5,
    "negative_flow_tolerance": 1e-12,
    "objective_scale_floor": 1e-12,
    "minimum_scaled_mass": 1e-4,
    "maximum_solver_scale": 1e6,
}


def test_exact_flow_is_invariant_to_tiny_positive_cost_scale():
    rng = np.random.default_rng(42)
    base = rng.random((28, 36))
    a = rng.random(28) + 1e-8
    b = rng.random(36) + 1e-8
    a /= a.sum()
    b /= b.sum()

    tiny = exact_flow(torch.as_tensor(base * 1e-8), a, b, 10**12, SOLVER).numpy()
    ordinary = exact_flow(torch.as_tensor(base), a, b, 10**12, SOLVER).numpy()

    np.testing.assert_allclose(tiny, ordinary, atol=1e-12, rtol=0)
    np.testing.assert_allclose(tiny.sum(1), a, atol=1e-10, rtol=1e-7)
    np.testing.assert_allclose(tiny.sum(0), b, atol=1e-10, rtol=1e-7)


def test_exact_flow_preserves_floor_sized_masses():
    cost = torch.arange(28 * 36, dtype=torch.float64).reshape(28, 36) * 1e-12
    a = np.full(28, (1.0 - 1e-10) / 27)
    b = np.full(36, (1.0 - 1e-10) / 35)
    a[0] = 1e-10
    b[0] = 1e-10

    flow = exact_flow(cost, a, b, 10**12, SOLVER).numpy()

    np.testing.assert_allclose(flow.sum(1), a, atol=1e-12, rtol=1e-5)
    np.testing.assert_allclose(flow.sum(0), b, atol=1e-12, rtol=1e-5)


def test_documented_mass_quantization_is_exact():
    masses = quantize_marginal(np.full(28, 1 / 28))

    assert masses.dtype == np.int64
    assert masses.sum() == 10**12
    assert masses.max() - masses.min() <= 1
