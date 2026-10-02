"""Exact hidden-state EMD with paper cost-attention marginals (not Sinkhorn)."""
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


def layer_cost(student, teacher, mask, projector):
    """[b,Ns,R,ds], [b,Nt,R,dt], [b,R] -> [Ns,Nt], sequence mean."""
    if student.ndim != 4 or teacher.ndim != 4 or student.shape[0] != teacher.shape[0]:
        raise ValueError("Expected batched response hidden states")
    if student.shape[2] != teacher.shape[2] or mask.shape != student.shape[::2]:
        raise ValueError("Mismatched response positions")
    costs = []
    for hs, ht, valid in zip(student, teacher.detach(), mask.bool()):
        if not valid.any():
            raise ValueError("Empty response in EMD")
        z = projector(hs[:, valid].float()).flatten(1)
        t = ht[:, valid].float().flatten(1)
        d = (z.square().sum(1)[:, None] + t.square().sum(1)[None, :] - 2 * z @ t.T) / z.shape[1]
        if not torch.isfinite(d).all() or torch.any(d < -1e-4):
            raise FloatingPointError("Invalid EMD distance")
        costs.append(d.clamp_min(0))
    return torch.stack(costs).mean(0)


def exact_flow(cost, a, b):
    d = cost.detach().double().cpu().numpy()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not (np.isfinite(d).all() and (d >= 0).all() and (a > 0).all() and (b > 0).all()):
        raise ValueError("Invalid transport input")
    ns, nt = d.shape
    constraints = lil_matrix((ns + nt, ns * nt))
    for i in range(ns): constraints[i, i * nt:(i + 1) * nt] = 1
    for j in range(nt): constraints[ns + j, j::nt] = 1
    # Keep floor-sized masses above the solver's absolute tolerance.
    # Scaling mass does not change the minimizing normalized transport plan.
    scale = 1e12
    # One marginal equation is redundant. Drop the largest column equation so
    # rounding of the total mass cannot make the scaled system inconsistent;
    # all tiny-mass equations remain explicit and all marginals are checked below.
    keep = np.ones(ns + nt, dtype=bool)
    keep[ns + int(np.argmax(b))] = False
    result = linprog(d.ravel(), A_eq=constraints.tocsr()[keep], b_eq=(np.r_[a, b] * scale)[keep],
                     bounds=(0, None), method="highs",
                     options={"primal_feasibility_tolerance": 1e-10,
                              "dual_feasibility_tolerance": 1e-10})
    if not result.success:
        raise RuntimeError(f"Exact EMD failed: {result.message}")
    flow = result.x.reshape(ns, nt) / scale
    residual = max(np.abs(flow.sum(1) - a).max(), np.abs(flow.sum(0) - b).max())
    relative = max((np.abs(flow.sum(1) - a) / a).max(), (np.abs(flow.sum(0) - b) / b).max())
    if residual > 1e-8 or relative > 1e-5 or flow.min() < -1e-15:
        raise RuntimeError(f"Invalid transport marginal residual {residual}")
    return torch.from_numpy(flow).to(device=cost.device).detach()


def update_marginals(cost, flow, a, b, tau):
    if tau <= 0:
        raise ValueError("EMD temperature must be positive")
    d, f = cost.detach().double().cpu(), flow.detach().double().cpu()
    if d.max() <= 1e-12:
        return np.array(a, copy=True), np.array(b, copy=True)
    def attention(unit_cost):
        c = unit_cost.clamp_min(1e-12)
        logits = c.sum() / c / tau
        mass = torch.softmax(logits - logits.max(), dim=0).clamp_min(1e-10)
        return (mass / mass.sum()).numpy()
    return attention((f * d).sum(1) / torch.as_tensor(a)), attention((f * d).sum(0) / torch.as_tensor(b))
