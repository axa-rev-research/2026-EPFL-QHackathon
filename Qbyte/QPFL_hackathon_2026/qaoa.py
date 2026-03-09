#!/usr/bin/env python3
"""
Greedy (sequential) QAOA optimization for a gapless Cat XoL tower using event-level losses.

CSV format:
year,event_id,loss
- loss is in MILLIONS in your data -> we scale by 1e6 to base currency units.

We optimize N layers sequentially:
- A0 is fixed (retention)
- At layer k, attachment A_k is known from previous chosen limits:
    A_k = A0 + sum_{i<k} L_i
- We choose (L_k, r_k) from discrete grids by solving a small one-hot QUBO with QAOA.

This uses only ~|L_grid|*|r_grid| qubits per layer (e.g., 9), avoiding exponential memory.

NOTE: Greedy != globally optimal tower, but it's practical and stable for QAOA demos.
"""

import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


# -------------------------
# Data loading
# -------------------------

def read_event_losses_csv(path: str, loss_unit_scale: float = 1e6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read CSV with columns: year,event_id,loss.
    loss_unit_scale converts CSV loss units to base currency units.
    For your data where loss is in millions, use 1e6.
    """
    years, eids, losses = [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"year", "event_id", "loss"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must have columns: year,event_id,loss")
        for row in reader:
            years.append(int(row["year"]))
            eids.append(int(row["event_id"]))
            losses.append(float(row["loss"]) * loss_unit_scale)

    if not losses:
        raise ValueError("CSV has no data rows.")

    # event_id global uniqueness sanity check
    if len(set(eids)) != len(eids):
        raise ValueError("event_id must be globally unique across all years.")

    return np.array(years, dtype=int), np.array(eids, dtype=int), np.array(losses, dtype=float)


def group_events_by_year(years: np.ndarray, losses: np.ndarray) -> Dict[int, np.ndarray]:
    """Map year -> array of event losses in that year."""
    by_year: Dict[int, List[float]] = {}
    for y, L in zip(years, losses):
        by_year.setdefault(int(y), []).append(float(L))
    return {y: np.array(vals, dtype=float) for y, vals in by_year.items()}


# -------------------------
# Contract / economics
# -------------------------

@dataclass(frozen=True)
class Option:
    """Discrete option for a layer: (L, r)."""
    L: float
    r: int


def layer_year_recovery(losses_in_year: np.ndarray, A: float, L: float, r: int) -> float:
    """
    Annual recovery for one layer given event losses within a year.
    Per-event payoff: min(max(loss - A, 0), L)
    Annual cap with reinstatements: (1+r)*L
    """
    pay_events = np.minimum(np.maximum(losses_in_year - A, 0.0), L)
    raw = float(np.sum(pay_events))
    cap = (1 + r) * L
    return min(cap, raw)


def option_cost_for_attachment(
    events_by_year: Dict[int, np.ndarray],
    A: float,
    opt: Option,
    rho: float
) -> float:
    """
    Compute the layer cost coefficient c_u for a fixed attachment A and option (L,r).

    We use the same surrogate as before (constants dropped):
      minimize  Premium + mean(R_t) - mean(ReinstPrem_t)

    Pricing without market curve:
      Premium = (1+rho)*mean(R_t)

    Reinstatement premium (1@100 style):
      ReinstPrem_t = Premium * min(r*L, R_t) / L
    """
    years = sorted(events_by_year.keys())
    R_t = np.array([layer_year_recovery(events_by_year[y], A, opt.L, opt.r) for y in years], dtype=float)
    R_bar = float(np.mean(R_t))

    premium = (1.0 + rho) * R_bar
    reinst_t = premium * np.minimum(opt.r * opt.L, R_t) / opt.L
    reinst_bar = float(np.mean(reinst_t))

    return premium + R_bar - reinst_bar


# -------------------------
# Small one-hot QUBO per layer
# -------------------------

def build_onehot_qubo(costs: List[float], P: float) -> QuadraticProgram:
    """
    QUBO:
      minimize  sum_u costs[u]*y_u + P*(sum_u y_u - 1)^2

    Expanded penalty yields:
      linear: (costs[u] - P) * y_u
      quadratic: 2P * y_u*y_v for u<v
    """
    n = len(costs)
    qp = QuadraticProgram()
    for u in range(n):
        qp.binary_var(name=f"y_{u}")

    linear = {f"y_{u}": (costs[u] - P) for u in range(n)}
    quadratic = {(f"y_{u}", f"y_{v}"): 2.0 * P for u in range(n) for v in range(u + 1, n)}

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def solve_layer_qaoa(qp: QuadraticProgram, reps: int = 1, maxiter: int = 200):
    sampler = StatevectorSampler()

    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)

    meo = MinimumEigenOptimizer(qaoa)
    return meo.solve(qp)


def select_onehot_index(result) -> int:
    x = result.x
    chosen = [i for i, v in enumerate(x) if v > 0.5]
    if len(chosen) == 1:
        return chosen[0]
    return int(np.argmax(x))


# -------------------------
# Greedy tower optimization
# -------------------------

def greedy_optimize_tower(
    events_by_year: Dict[int, np.ndarray],
    n_layers: int,
    A0: float,
    options: List[Option],
    rho: float,
    P_onehot: float,
    qaoa_reps: int,
    qaoa_maxiter: int
) -> List[Tuple[float, float, int]]:
    """
    Sequentially choose each layer option using QAOA on a small one-hot QUBO.
    Returns list of (A_k, L_k, r_k).
    """
    tower: List[Tuple[float, float, int]] = []
    cumulative = 0.0

    for k in range(n_layers):
        A_k = A0 + cumulative

        # Compute costs for each option at this attachment
        costs = [option_cost_for_attachment(events_by_year, A=A_k, opt=opt, rho=rho) for opt in options]

        # Build and solve QUBO for this layer
        qp = build_onehot_qubo(costs, P=P_onehot)
        result = solve_layer_qaoa(qp, reps=qaoa_reps, maxiter=qaoa_maxiter)
        idx = select_onehot_index(result)

        chosen = options[idx]
        tower.append((A_k, chosen.L, chosen.r))

        # Update cumulative limit for next attachment (gapless tower)
        cumulative += chosen.L

    return tower


# -------------------------
# Main
# -------------------------

def main():
    csv_path = "event_losses.csv"
    LOSS_SCALE = 1e6              # loss in CSV is millions

    # Tower / discretization
    n_layers = 3
    A0 = 10e6  # fixed retention in base currency units (10M)

    # Given your losses can be very large (hundreds of millions to >1B),
    # you will often want larger limits / higher A0 for a meaningful trade-off.
    L_grid = [5e6, 10e6, 15e6]  # start with bigger limits than 5-20M
    r_grid = [0, 1, 2]

    options = [Option(L=L, r=r) for L in L_grid for r in r_grid]  # |U| = len(L_grid)*len(r_grid)

    # Pricing surrogate
    rho = 0.30

    # QUBO penalty (scale-dependent). Needs to dominate typical cost differences.
    # With costs ~ 1e8..1e9, use 1e10 as a safe starting point.
    P_onehot = 1e10

    # QAOA settings
    reps = 1
    maxiter = 50

    years, eids, losses = read_event_losses_csv(csv_path, loss_unit_scale=LOSS_SCALE)
    events_by_year = group_events_by_year(years, losses)

    # Quick scale diagnostics
    all_losses_m = losses / 1e6
    print(f"Events: {len(losses)}, Years: {len(events_by_year)}")
    print(f"Loss scale (millions): min={all_losses_m.min():.3g}, median={np.median(all_losses_m):.3g}, max={all_losses_m.max():.3g}")
    print(f"Config: A0={A0/1e6:.0f}M, L_grid={[int(L/1e6) for L in L_grid]}M, r_grid={r_grid}, layers={n_layers}")
    print(f"Qubits per layer QAOA (one-hot) = |U| = {len(options)}")

    tower = greedy_optimize_tower(
        events_by_year=events_by_year,
        n_layers=n_layers,
        A0=A0,
        options=options,
        rho=rho,
        P_onehot=P_onehot,
        qaoa_reps=reps,
        qaoa_maxiter=maxiter,
    )

    print("\nGreedy QAOA tower:")
    top = A0
    for k, (A_k, L_k, r_k) in enumerate(tower):
        top = A_k + L_k
        print(f"  Layer {k}: A={A_k/1e6:.0f}M, L={L_k/1e6:.0f}M, r={r_k}   (top after layer={top/1e6:.0f}M)")

    print("\nNOTE:")
    print("  This sequential approach keeps qubits small but is not a global optimum over all layers.")
    print("  If you later need global optimization with ~O(layers*options) qubits, we can add auxiliary variables,")
    print("  but the QUBO becomes more involved.")


if __name__ == "__main__":
    main()