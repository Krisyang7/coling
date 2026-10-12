"""Exact hidden-state EMD with fixed uniform marginals (not Sinkhorn)."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


MASS_QUANTIZATION_SCALE = 10**12
REPRESENTATION_EPSILON = 1e-12


def layer_cost(student, teacher, mask, projector, epsilon=REPRESENTATION_EPSILON):
    """[b,Ns,R,ds], [b,Nt,R,dt], [b,R] -> [Ns,Nt], sequence mean."""
    if student.ndim != 4 or teacher.ndim != 4 or student.shape[0] != teacher.shape[0]:
        raise ValueError("Expected batched response hidden states")
    if student.shape[2] != teacher.shape[2] or mask.shape != student.shape[::2]:
        raise ValueError("Mismatched response positions")
    costs = []
    for hs, ht, valid in zip(student, teacher.detach(), mask.bool()):
        if not valid.any():
            raise ValueError("Empty response in EMD")
        z = F.normalize(
            projector(hs[:, valid].float()), p=2, dim=-1, eps=epsilon
        ).flatten(1)
        t = F.normalize(
            ht[:, valid].float(), p=2, dim=-1, eps=epsilon
        ).flatten(1)
        d = (z.square().sum(1)[:, None] + t.square().sum(1)[None, :] - 2 * z @ t.T) / z.shape[1]
        if not torch.isfinite(d).all():
            raise FloatingPointError("Invalid EMD distance")
        costs.append(d.clamp_min(0))
    return torch.stack(costs).mean(0)


def quantize_marginal(marginal, scale=MASS_QUANTIZATION_SCALE):
    """Quantize a probability vector to integer masses summing to ``scale``."""
    values = np.asarray(marginal, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("transport marginals must be finite positive vectors")
    values = values / values.sum()
    scaled = values * scale
    masses = np.floor(scaled).astype(np.int64)
    remainder = int(scale - masses.sum())
    order = np.argsort(-(scaled - masses), kind="stable")
    masses[order[:remainder]] += 1
    if np.any(masses <= 0) or masses.sum() != scale:
        raise ValueError("mass quantization produced an invalid marginal")
    return masses


def exact_flow(cost, a, b, mass_scale, solver):
    d = cost.detach().double().cpu().numpy()
    if not (np.isfinite(d).all() and (d >= 0).all()):
        raise ValueError("Invalid transport input")
    if mass_scale != MASS_QUANTIZATION_SCALE:
        raise ValueError(f"mass_scale must be {MASS_QUANTIZATION_SCALE}")
    a_mass = quantize_marginal(a, mass_scale)
    b_mass = quantize_marginal(b, mass_scale)
    a = a_mass.astype(np.float64) / mass_scale
    b = b_mass.astype(np.float64) / mass_scale
    ns, nt = d.shape
    constraints = lil_matrix((ns + nt, ns * nt))
    for i in range(ns): constraints[i, i * nt:(i + 1) * nt] = 1
    for j in range(nt): constraints[ns + j, j::nt] = 1
    # The integer masses define the quantized problem. Dividing both sides by
    # the common scale is equivalent and keeps HiGHS in a stable numeric range.
    keep = np.ones(ns + nt, dtype=bool)
    keep[ns + int(np.argmax(b))] = False
    required = {
        "primal_feasibility_tolerance",
        "dual_feasibility_tolerance",
        "absolute_residual_tolerance",
        "relative_residual_tolerance",
        "negative_flow_tolerance",
        "objective_scale_floor",
        "minimum_scaled_mass",
        "maximum_solver_scale",
    }
    if set(solver) != required or any(solver[key] is None for key in required):
        raise ValueError("all solver numerical settings must be explicitly supplied")
    values = {key: float(value) for key, value in solver.items()}
    if any(value <= 0 or not np.isfinite(value) for value in values.values()):
        raise ValueError("solver numerical settings must be finite and positive")
    objective = d.ravel() / max(float(d.max()), values["objective_scale_floor"])
    attempts = []
    min_mass = min(float(a.min()), float(b.min()))
    solver_scale = min(
        values["maximum_solver_scale"],
        max(1.0, values["minimum_scaled_mass"] / min_mass),
    )
    quantized_marginals = (np.r_[a, b] * solver_scale)[keep]
    for method, presolve in (("highs-ds", True),
                             ("highs-ds", False),
                             ("highs-ipm", True)):
        result = linprog(
            objective,
            A_eq=constraints.tocsr()[keep],
            b_eq=quantized_marginals,
            bounds=(0, None),
            method=method,
            options={"presolve": presolve,
                     "primal_feasibility_tolerance": values["primal_feasibility_tolerance"],
                     "dual_feasibility_tolerance": values["dual_feasibility_tolerance"]},
        )
        if not result.success:
            attempts.append(f"{method}/presolve={presolve}: {result.message}")
            continue
        flow = result.x.reshape(ns, nt) / solver_scale
        residual = max(np.abs(flow.sum(1) - a).max(),
                       np.abs(flow.sum(0) - b).max())
        relative = max((np.abs(flow.sum(1) - a) / a).max(),
                       (np.abs(flow.sum(0) - b) / b).max())
        if (residual <= values["absolute_residual_tolerance"]
                and relative <= values["relative_residual_tolerance"]
                and flow.min() >= -values["negative_flow_tolerance"]):
            return torch.from_numpy(flow).to(device=cost.device).detach()
        attempts.append(
            f"{method}/presolve={presolve}: invalid plan "
            f"residual={residual}, relative={relative}, min={flow.min()}"
        )
    raise RuntimeError(
        "Exact EMD failed after well-scaled solver retries: "
        f"{' | '.join(attempts)}; cost_range=({d.min()}, {d.max()}); "
        f"mass_range=({min(a.min(), b.min())}, {max(a.max(), b.max())}); "
        f"mass_scale={mass_scale}"
    )
