#!/usr/bin/env python3
"""
DP (inside each group) + QAOA (portfolio-level) for Cat XoL towers.

Input CSV (loss in MILLIONS):
year,event_id,group,loss

Goal:
- 3 groups
- thousands of years
- 3-layer gapless tower per group
- common premium budget
- common portfolio tail-risk proxy (TVaR/CVaR-like) via fixed tail-years

Pipeline:
1) For each group:
   - build a small set of candidate towers using a DP/beam-search over layers
   - evaluate each candidate: premium, expected cost, and yearly net-loss vector

2) Portfolio QUBO:
   - choose exactly 1 candidate per group
   - penalize budget violation
   - penalize tail-loss proxy on a fixed tail set of years
   - solve with QAOA

Notes:
- This avoids exponential memory from enumerating all global portfolios.
- The "TVaR" proxy here is average net loss over the worst tail-years chosen by gross loss.
  (Scales to thousands of years without adding thousands of binary variables.)
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
# Config (edit these)
# -------------------------

CSV_PATH = "catastrophe_scene.csv"
LOSS_SCALE = 1e6  # loss in CSV is in millions

GROUPS = ["0", "1", "2"]

N_LAYERS = 4
A0_BY_GROUP = {g: 50e6 for g in GROUPS}  # fixed retention per group

# Discrete grids
L_GRID = [25e6, 50e6, 100e6, 200e6]   # limits
R_GRID = [0, 1, 2] # reinstatements

RHO = 0.30  # pricing loading: Premium = (1+rho)*E[Recovery]

# Candidate generation
BEAM_WIDTH = 30          # DP keeps at most this many partial towers per state
CANDIDATES_PER_GROUP = 4 # how many final towers per group to keep for portfolio optimization

# Portfolio constraints / penalties
BUDGET = 20e6            # total premium budget across all groups
ALPHA = 0.99             # "TVaR/CVaR" tail level (we use worst (1-alpha) fraction of years)
W_TAIL = 1.0             # weight for tail-loss proxy in objective
P_ONEHOT = 1e10          # one-hot penalty
P_BUDGET = 1e-6          # budget penalty scaling (see note below)

# QAOA
QAOA_REPS = 1
QAOA_MAXITER = 30


# -------------------------
# Data loading
# -------------------------

def read_events_csv(path: str, loss_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read CSV with columns: year,event_id,group,loss
    loss is scaled by loss_scale into base currency units.
    """
    years, eids, groups, losses = [], [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        req = {"year", "event_id", "group", "loss"}
        if not req.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must have columns: year,event_id,group,loss")
        for row in reader:
            years.append(int(row["year"]))
            eids.append(int(row["event_id"]))
            groups.append(str(row["group"]))
            losses.append(float(row["loss"]) * loss_scale)

    if len(losses) == 0:
        raise ValueError("CSV has no rows.")

    if len(set(eids)) != len(eids):
        raise ValueError("event_id must be globally unique across all years.")

    return (np.array(years, dtype=int),
            np.array(eids, dtype=int),
            np.array(groups, dtype=object),
            np.array(losses, dtype=float))


def build_events_by_group_and_year(
    years: np.ndarray,
    groups: np.ndarray,
    losses: np.ndarray,
    target_groups: List[str]
) -> Tuple[List[int], Dict[str, Dict[int, np.ndarray]]]:
    """
    Returns:
      all_years_sorted: list of all years observed in the dataset (sorted)
      events[g][year] = np.array(event_losses) for group g and year
      Missing years are filled with empty arrays.
    """
    all_years = sorted(set(int(y) for y in years))
    events: Dict[str, Dict[int, List[float]]] = {g: {y: [] for y in all_years} for g in target_groups}

    for y, g, L in zip(years, groups, losses):
        if g in events:
            events[g][int(y)].append(float(L))

    # convert lists to arrays
    events_np: Dict[str, Dict[int, np.ndarray]] = {}
    for g in target_groups:
        events_np[g] = {}
        for y in all_years:
            events_np[g][y] = np.array(events[g][y], dtype=float)

    return all_years, events_np


# -------------------------
# XoL tower evaluation
# -------------------------

@dataclass(frozen=True)
class LayerChoice:
    L: float
    r: int


@dataclass(frozen=True)
class Tower:
    layers: Tuple[LayerChoice, ...]  # length N_LAYERS


def layer_year_recovery(losses_in_year: np.ndarray, A: float, L: float, r: int) -> float:
    """
    Annual recovery for one layer given event losses within a year.
      pay_event = min(max(loss - A, 0), L)
      R_raw = sum(pay_event)
      R_year = min((1+r)*L, R_raw)
    """
    if L <= 0:
        raise ValueError("L must be > 0.")
    pay = np.minimum(np.maximum(losses_in_year - A, 0.0), L)
    raw = float(np.sum(pay))
    cap = (1 + r) * L
    return min(cap, raw)


def evaluate_tower_for_group(
    all_years: List[int],
    events_by_year: Dict[int, np.ndarray],
    A0: float,
    tower: Tower,
    rho: float
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate one tower for one group.

    Returns:
      premium_total: sum of layer premiums
      cost_total: surrogate cost (lower is better)
      netloss_by_year: array length T, net loss per year for this group under this tower

    Definitions (simplified, consistent with earlier code and paper structure):
    - gross_loss_year[t] = sum of event losses in that year
    - per layer:
        R_{k,t} = annual recovery
        Premium_k = (1+rho)*mean_t(R_{k,t})
        ReinstPrem_{k,t} = Premium_k * min(r_k*L_k, R_{k,t}) / L_k
      group net loss per year:
        NL_t = gross_loss_t - sum_k (R_{k,t} - ReinstPrem_{k,t})
      objective surrogate cost:
        cost = sum_k [ Premium_k + mean(R_{k,t}) - mean(ReinstPrem_{k,t}) ]
      (constants independent of tower are omitted)
    """
    T = len(all_years)
    gross = np.array([float(np.sum(events_by_year[y])) for y in all_years], dtype=float)

    # attachments implied by gapless tower
    Ls = [lc.L for lc in tower.layers]
    As = []
    cum = A0
    for k in range(len(Ls)):
        As.append(cum)
        cum += Ls[k]

    # accumulate per layer
    premium_total = 0.0
    cost_total = 0.0
    netloss = gross.copy()

    for k, lc in enumerate(tower.layers):
        A_k = As[k]
        L_k = lc.L
        r_k = lc.r

        R_t = np.array([layer_year_recovery(events_by_year[y], A_k, L_k, r_k) for y in all_years], dtype=float)
        R_bar = float(np.mean(R_t))

        premium_k = (1.0 + rho) * R_bar
        reinst_t = premium_k * np.minimum(r_k * L_k, R_t) / L_k
        reinst_bar = float(np.mean(reinst_t))

        premium_total += premium_k
        cost_total += premium_k + R_bar - reinst_bar

        # netloss subtracts (recovery - reinstatement premium)
        netloss -= (R_t - reinst_t)

    return float(premium_total), float(cost_total), netloss


# -------------------------
# DP/Beam-search to generate candidate towers per group
# -------------------------

def generate_candidates_dp_beam(
    all_years: List[int],
    events_by_year: Dict[int, np.ndarray],
    A0: float,
    L_grid: List[float],
    r_grid: List[int],
    rho: float,
    n_layers: int,
    beam_width: int,
    n_candidates: int
) -> List[Tuple[Tower, float, float, np.ndarray]]:
    """
    Beam-search DP:
    - Build towers layer by layer.
    - Keep only a limited number of partial towers (beam_width) per layer-depth.
    - Rank partial towers by a cheap proxy: current cumulative premium + current mean netloss.
      (Simple and fast; final selection uses full evaluation.)

    Returns list of candidates:
      [(tower, premium_total, cost_total, netloss_by_year), ...] length <= n_candidates
    """
    options = [LayerChoice(L=L, r=r) for L in L_grid for r in r_grid]

    partial: List[Tuple[List[LayerChoice], float]] = [([], 0.0)]  # (layers_so_far, cumulative_limit_sum)

    for depth in range(n_layers):
        new_partial: List[Tuple[List[LayerChoice], float]] = []
        for layers_so_far, cumL in partial:
            for opt in options:
                new_layers = layers_so_far + [opt]
                new_cumL = cumL + opt.L
                new_partial.append((new_layers, new_cumL))

        # Beam pruning using quick proxy evaluation on partial towers:
        # Evaluate partial tower as if it's the full tower truncated (cheap enough for 3 layers, still ok)
        scored = []
        for layers_so_far, _ in new_partial:
            # Quick proxy: evaluate the truncated tower (depth+1 layers)
            t = Tower(layers=tuple(layers_so_far))
            prem, cost, netloss = evaluate_tower_for_group(all_years, events_by_year, A0, t, rho)
            # proxy score: cost + small premium weight (already in cost) + mean netloss
            proxy = cost + 0.1 * float(np.mean(netloss))
            scored.append((proxy, layers_so_far))

        scored.sort(key=lambda x: x[0])
        partial = [(layers, 0.0) for _, layers in scored[:beam_width]]
        
        print(f"DP: depth={depth+1}/{n_layers}, partial towers kept = {len(partial)}")

    # Fully evaluate final towers and keep best diverse set
    evaluated = []
    for layers_so_far, _ in partial:
        t = Tower(layers=tuple(layers_so_far))
        prem, cost, netloss = evaluate_tower_for_group(all_years, events_by_year, A0, t, rho)
        evaluated.append((t, prem, cost, netloss))

    # Sort by cost primarily
    evaluated.sort(key=lambda x: x[2])

    # Keep n_candidates but try to diversify by premium bins
    picked = []
    seen_bins = set()
    for t, prem, cost, netloss in evaluated:
        b = int(prem / 5e6)  # 5M premium bins (tune as needed)
        if b not in seen_bins or len(picked) < n_candidates // 2:
            picked.append((t, prem, cost, netloss))
            seen_bins.add(b)
        if len(picked) >= n_candidates:
            break

    return picked


# -------------------------
# Portfolio QUBO + QAOA
# -------------------------

def select_tail_years_by_gross_loss(
    all_years: List[int],
    events_by_group_and_year: Dict[str, Dict[int, np.ndarray]],
    groups: List[str],
    alpha: float
) -> List[int]:
    """
    Pick tail years based on portfolio gross loss (sum over groups).
    Tail size K = ceil((1-alpha)*T), at least 1.
    """
    T = len(all_years)
    K = max(1, int(np.ceil((1.0 - alpha) * T)))

    gross_port = []
    for y in all_years:
        s = 0.0
        for g in groups:
            s += float(np.sum(events_by_group_and_year[g][y]))
        gross_port.append((s, y))

    gross_port.sort(reverse=True, key=lambda x: x[0])
    return [y for _, y in gross_port[:K]]


def build_portfolio_qubo(
    candidates_by_group: Dict[str, List[Tuple[Tower, float, float, np.ndarray]]],
    groups: List[str],
    all_years: List[int],
    tail_years: List[int],
    budget: float,
    w_tail: float,
    p_onehot: float,
    p_budget: float
) -> Tuple[QuadraticProgram, List[Tuple[str, int]]]:
    """
    Variables: x_{g,m} in {0,1} select candidate m for group g.
    Constraints via penalties:
      - one-hot per group
      - budget: (sum premium - budget)^2 penalty (soft constraint)

    Objective (minimize):
      sum cost[g,m] * x_{g,m}
      + w_tail * mean_{y in tail_years}( portfolio_netloss_y )
      + penalties

    portfolio_netloss_y = sum_g netloss[g,sel,m][y]

    This tail-risk proxy is linear in x (tail years fixed).
    """
    # variable indexing list -> to decode later
    var_index: List[Tuple[str, int]] = []  # (group, m)

    qp = QuadraticProgram()

    # Create variables
    for g in groups:
        M = len(candidates_by_group[g])
        for m in range(M):
            name = f"x_{g}_{m}"
            qp.binary_var(name=name)
            var_index.append((g, m))

    linear = {}

    # Base objective: expected cost + tail proxy
    # tail proxy uses average over tail years
    tail_idx = [all_years.index(y) for y in tail_years]
    tail_norm = 1.0 / len(tail_years)

    for g in groups:
        for m, (_t, prem, cost, netloss) in enumerate(candidates_by_group[g]):
            name = f"x_{g}_{m}"
            tail_mean = float(np.mean(netloss[tail_idx]))  # average net loss on tail set
            linear[name] = cost + w_tail * tail_mean

    quadratic = {}

    # One-hot penalties per group: p_onehot * (sum_m x_{g,m} - 1)^2
    for g in groups:
        names = [f"x_{g}_{m}" for m in range(len(candidates_by_group[g]))]
        # expand as: -p*sum x + 2p*sum_{i<j} x_i x_j (constant ignored)
        for v in names:
            linear[v] = linear.get(v, 0.0) - p_onehot
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                quadratic[(names[i], names[j])] = quadratic.get((names[i], names[j]), 0.0) + 2.0 * p_onehot

    # Budget penalty: p_budget * (sum prem*x - budget)^2
    # Expand: p*(sum w_i x_i)^2 - 2 p*budget*(sum w_i x_i) + const
    # Binary: x_i^2 = x_i -> diagonal quadratic folds into linear
    # We'll implement full expansion with linear+quadratic.
    all_vars = []
    all_w = []
    for g in groups:
        for m, (_t, prem, _cost, _netloss) in enumerate(candidates_by_group[g]):
            all_vars.append(f"x_{g}_{m}")
            all_w.append(float(prem))

    # linear part from w_i^2*x_i and -2*budget*w_i*x_i
    for v, w in zip(all_vars, all_w):
        linear[v] = linear.get(v, 0.0) + p_budget * (w * w - 2.0 * budget * w)

    # quadratic part 2*w_i*w_j*x_i*x_j
    for i in range(len(all_vars)):
        for j in range(i + 1, len(all_vars)):
            key = (all_vars[i], all_vars[j])
            quadratic[key] = quadratic.get(key, 0.0) + 2.0 * p_budget * all_w[i] * all_w[j]

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp, var_index


# def cb(nfev, x, f):
#     print(f"QAOA iter {nfev}: f={f}")

def solve_qubo_qaoa(qp: QuadraticProgram, reps: int, maxiter: int):
    sampler = StatevectorSampler()
    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
    return MinimumEigenOptimizer(qaoa).solve(qp)


def decode_selection(result, groups: List[str], candidates_by_group: Dict[str, List], var_index: List[Tuple[str, int]]):
    x = result.x
    chosen = {g: None for g in groups}
    # pick max for each group (in case penalties not perfect)
    scores = {g: (-1.0, None) for g in groups}
    for idx, val in enumerate(x):
        g, m = var_index[idx]
        if val > scores[g][0]:
            scores[g] = (val, m)
    for g in groups:
        chosen[g] = scores[g][1]
    return chosen



import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def portfolio_eval_for_choice(GROUPS, choice_idx_by_group, candidates_by_group, all_years, tail_years, events):
    """
    Evaluate portfolio metrics for a particular candidate index selection per group.
    Returns: total_premium, pure_cost_sum, tail_mean_netloss, netloss_port_vector, gross_port_vector
    """
    T = len(all_years)
    netloss_port = np.zeros(T, dtype=float)
    total_premium = 0.0
    pure_cost_sum = 0.0

    for g in GROUPS:
        m = choice_idx_by_group[g]
        _tower, prem, cost, netloss = candidates_by_group[g][m]
        total_premium += prem
        pure_cost_sum += cost
        netloss_port += netloss

    # baseline gross portfolio loss for comparison
    gross_port = np.array(
        [sum(float(np.sum(events[g][y])) for g in GROUPS) for y in all_years],
        dtype=float
    )

    tail_idx = [all_years.index(y) for y in tail_years]
    tail_mean = float(np.mean(netloss_port[tail_idx]))

    return total_premium, pure_cost_sum, tail_mean, netloss_port, gross_port


def enumerate_all_portfolios(GROUPS, candidates_by_group):
    """
    Enumerate all combinations of candidate indices for all groups.
    Returns a list of dict choices: [{g: idx, ...}, ...]
    """
    idx_lists = [list(range(len(candidates_by_group[g]))) for g in GROUPS]
    all_choices = []
    for combo in product(*idx_lists):
        choice = {g: combo[i] for i, g in enumerate(GROUPS)}
        all_choices.append(choice)
    return all_choices


def solve_portfolio_exact_under_budget(GROUPS, candidates_by_group, all_years, tail_years, events, B, W_tail=1.0):
    """
    Exact classical solver for the outer selection problem:
      minimize sum(cost) + W_tail*tail_mean(netloss)  subject to total_premium <= B
    Returns best choice dict + metrics.
    """
    all_choices = enumerate_all_portfolios(GROUPS, candidates_by_group)

    best = None
    best_obj = float("inf")
    best_metrics = None

    for choice in all_choices:
        prem, cost_sum, tail_mean, netloss_port, gross_port = portfolio_eval_for_choice(
            GROUPS, choice, candidates_by_group, all_years, tail_years, events
        )
        if prem > B:
            continue
        obj = cost_sum + W_tail * tail_mean
        if obj < best_obj:
            best_obj = obj
            best = choice
            best_metrics = (prem, cost_sum, tail_mean, netloss_port, gross_port)

    return best, best_obj, best_metrics


def compute_budget_risk_curve(GROUPS, candidates_by_group, all_years, tail_years, events, budgets, W_tail=1.0):
    """
    For each budget, solve exact outer problem and store metrics.
    """
    xs_B = []
    ys_tail = []
    ys_prem = []
    ys_obj = []
    choices = []

    for B in budgets:
        choice, obj, metrics = solve_portfolio_exact_under_budget(
            GROUPS, candidates_by_group, all_years, tail_years, events, B, W_tail=W_tail
        )
        if choice is None:
            # infeasible budget
            xs_B.append(B)
            ys_tail.append(np.nan)
            ys_prem.append(np.nan)
            ys_obj.append(np.nan)
            choices.append(None)
            continue

        prem, cost_sum, tail_mean, *_ = metrics
        xs_B.append(B)
        ys_tail.append(tail_mean)
        ys_prem.append(prem)
        ys_obj.append(obj)
        choices.append(choice)

    return np.array(xs_B), np.array(ys_tail), np.array(ys_prem), np.array(ys_obj), choices


def plot_budget_risk(xs_B, ys_tail, baseline_tail=None):
    plt.figure()
    plt.plot(xs_B / 1e6, ys_tail / 1e6, marker="o")
    plt.xlabel("Budget B (M)")
    plt.ylabel("Tail-risk proxy: mean net loss on tail years (M)")
    if baseline_tail is not None:
        plt.axhline(baseline_tail / 1e6)
    plt.title("Budget vs Tail Risk")
    plt.tight_layout()


def plot_budget_premium(xs_B, ys_prem):
    plt.figure()
    plt.plot(xs_B / 1e6, ys_prem / 1e6, marker="o")
    plt.xlabel("Budget B (M)")
    plt.ylabel("Selected total premium (M)")
    plt.title("Budget vs Premium Used")
    plt.tight_layout()


def plot_pareto_scatter(GROUPS, candidates_by_group, all_years, tail_years, events, highlight_choice=None):
    """
    Scatter all combinations: x=premium, y=tail-risk proxy
    """
    all_choices = enumerate_all_portfolios(GROUPS, candidates_by_group)
    premiums = []
    tails = []

    for choice in all_choices:
        prem, _cost_sum, tail_mean, *_ = portfolio_eval_for_choice(
            GROUPS, choice, candidates_by_group, all_years, tail_years, events
        )
        premiums.append(prem)
        tails.append(tail_mean)

    premiums = np.array(premiums)
    tails = np.array(tails)

    plt.figure()
    plt.scatter(premiums / 1e6, tails / 1e6, s=10)
    plt.xlabel("Total premium (M)")
    plt.ylabel("Tail-risk proxy (M)")
    plt.title("Portfolio Pareto Cloud (all candidate combinations)")

    if highlight_choice is not None:
        prem_h, _c, tail_h, *_ = portfolio_eval_for_choice(
            GROUPS, highlight_choice, candidates_by_group, all_years, tail_years, events
        )
        plt.scatter([prem_h / 1e6], [tail_h / 1e6], s=80)
    plt.tight_layout()


def plot_tail_histogram(netloss_port, gross_port, tail_years, all_years, bins=30):
    """
    Histogram / distribution comparison.
    """
    plt.figure()
    plt.hist(gross_port / 1e6, bins=bins, alpha=0.5, label="No reinsurance (gross)")
    plt.hist(netloss_port / 1e6, bins=bins, alpha=0.5, label="Selected (net)")
    plt.xlabel("Portfolio loss (M)")
    plt.ylabel("Count of years")
    plt.title("Distribution of annual portfolio loss")
    plt.legend()
    plt.tight_layout()


def make_plots_fast(GROUPS, candidates_by_group, all_years, tail_years, events, W_tail=1.0):
    # Baseline tail (no reinsurance)
    gross_port = np.array(
        [sum(float(np.sum(events[g][y])) for g in GROUPS) for y in all_years],
        dtype=float
    )
    tail_idx = [all_years.index(y) for y in tail_years]
    baseline_tail = float(np.mean(gross_port[tail_idx]))

    # Choose a small set of budgets to sweep (fast)
    # Example: from min feasible premium to something higher
    # Compute min-premium portfolio (pick min premium per group)
    min_prem = 0.0
    for g in GROUPS:
        min_prem += min(c[1] for c in candidates_by_group[g])
    budgets = np.linspace(min_prem, min_prem * 1.8, 8)  # 8 points

    xs_B, ys_tail, ys_prem, ys_obj, choices = compute_budget_risk_curve(
        GROUPS, candidates_by_group, all_years, tail_years, events, budgets, W_tail=W_tail
    )

    plot_budget_risk(xs_B, ys_tail, baseline_tail=baseline_tail)
    plot_budget_premium(xs_B, ys_prem)

    # Highlight best at highest budget point
    best_choice = choices[-1] if choices[-1] is not None else None
    plot_pareto_scatter(GROUPS, candidates_by_group, all_years, tail_years, events, highlight_choice=best_choice)

    # Plot histogram for highlighted solution (if exists)
    if best_choice is not None:
        prem, cost_sum, tail_mean, netloss_port, gross_port = portfolio_eval_for_choice(
            GROUPS, best_choice, candidates_by_group, all_years, tail_years, events
        )
        plot_tail_histogram(netloss_port, gross_port, tail_years, all_years)

    plt.show()




# -------------------------
# Main
# -------------------------

def main():
    years, eids, groups_arr, losses = read_events_csv(CSV_PATH, LOSS_SCALE)

    # sanity: check we have at least the required groups
    present_groups = set(groups_arr.tolist())
    for g in GROUPS:
        if g not in present_groups:
            raise ValueError(f"Group '{g}' not found in CSV. Present groups: {sorted(present_groups)}")

    all_years, events = build_events_by_group_and_year(years, groups_arr, losses, GROUPS)

    print(f"Loaded events: {len(losses)}, years: {len(all_years)}, groups: {GROUPS}")
    print(f"Loss scale in millions: min={losses.min()/1e6:.3g}, median={np.median(losses)/1e6:.3g}, max={losses.max()/1e6:.3g}")

    # Tail years selection (fixed set, portfolio gross-based)
    tail_years = select_tail_years_by_gross_loss(all_years, events, GROUPS, ALPHA)
    print(f"Tail years count = {len(tail_years)} (alpha={ALPHA}); example tail years (first 10): {tail_years[:10]}")

    # Generate candidates per group (DP/beam)
    candidates_by_group: Dict[str, List[Tuple[Tower, float, float, np.ndarray]]] = {}
    for g in GROUPS:
        cand = generate_candidates_dp_beam(
            all_years=all_years,
            events_by_year=events[g],
            A0=A0_BY_GROUP[g],
            L_grid=L_GRID,
            r_grid=R_GRID,
            rho=RHO,
            n_layers=N_LAYERS,
            beam_width=BEAM_WIDTH,
            n_candidates=CANDIDATES_PER_GROUP,
        )
        candidates_by_group[g] = cand

        print(f"\nGroup {g}: candidates kept = {len(cand)}")
        for i, (t, prem, cost, _nl) in enumerate(cand[:5]):
            desc = ", ".join([f"(L={lc.L/1e6:.0f}M,r={lc.r})" for lc in t.layers])
            print(f"  cand {i}: premium={prem/1e6:.2f}M, cost={cost/1e6:.2f}M, layers={desc}")

    make_plots_fast(GROUPS, candidates_by_group, all_years, tail_years, events, W_tail=W_TAIL)

    # Build portfolio QUBO
    qp, var_index = build_portfolio_qubo(
        candidates_by_group=candidates_by_group,
        groups=GROUPS,
        all_years=all_years,
        tail_years=tail_years,
        budget=BUDGET,
        w_tail=W_TAIL,
        p_onehot=P_ONEHOT,
        p_budget=P_BUDGET,
    )

    print("\nPortfolio QUBO (conceptual):")
    print("  minimize  Σ cost[g,m]*x[g,m] + W_tail*mean_tail_years(NetLoss_portfolio)")
    print("            + P_onehot*Σ_g (Σ_m x[g,m] - 1)^2")
    print("            + P_budget*(Σ premium[g,m]*x[g,m] - Budget)^2")
    print(f"  Budget={BUDGET/1e6:.2f}M, alpha={ALPHA}, tail_years={len(tail_years)}, vars={len(var_index)}")
    print("  NOTE: tail-risk is approximated using a fixed tail-year set (scales to thousands of years).")

    # Solve with QAOA
    result = solve_qubo_qaoa(qp, reps=QAOA_REPS, maxiter=QAOA_MAXITER)
    chosen = decode_selection(result, GROUPS, candidates_by_group, var_index)

    # Report selection + check budget and tail proxy
    total_prem = 0.0
    netloss_port = np.zeros(len(all_years), dtype=float)

    print("\nSelected portfolio towers:")
    for g in GROUPS:
        m = chosen[g]
        t, prem, cost, netloss = candidates_by_group[g][m]
        total_prem += prem
        netloss_port += netloss

        # print tower with implied attachments
        A0 = A0_BY_GROUP[g]
        cum = 0.0
        print(f"\nGroup {g} (chosen cand {m}): premium={prem/1e6:.2f}M, cost={cost/1e6:.2f}M")
        for k, lc in enumerate(t.layers):
            A_k = A0 + cum
            cum += lc.L
            print(f"  layer {k}: A={A_k/1e6:.0f}M, L={lc.L/1e6:.0f}M, r={lc.r}")

    tail_idx = [all_years.index(y) for y in tail_years]
    tail_mean = float(np.mean(netloss_port[tail_idx]))

    print(f"Tail-loss proxy (mean net loss on tail years) = {tail_mean/1e6:.2f}M")
    print(f"QUBO objective value = {float(result.fval):.6g}")
    
    # Baseline: no reinsurance -> portfolio netloss = portfolio gross loss
    gross_port = np.array(
        [sum(float(np.sum(events[g][y])) for g in GROUPS) for y in all_years],
        dtype=float
    )

    tail_no_re = float(np.mean(gross_port[tail_idx]))
    tail_sel = float(np.mean(netloss_port[tail_idx]))

    delta = tail_no_re - tail_sel
    pct = 100.0 * delta / tail_no_re if tail_no_re > 0 else np.nan
    eff = delta / total_prem if total_prem > 0 else np.nan

    print(f"\nBaseline tail (no reinsurance) = {tail_no_re/1e6:.2f}M")
    print(f"Selected tail proxy            = {tail_sel/1e6:.2f}M")
    print(f"Tail reduction                 = {delta/1e6:.2f}M ({pct:.2f}%)")
    print(f"Total premium                  = {total_prem/1e6:.2f}M (budget {BUDGET/1e6:.2f}M)")
    print(f"Tail reduction per premium     = {eff:.2f} (M tail reduced per M premium)")

    print("\nTips:")
    print("  - If budget is violated often, increase P_BUDGET (or rescale premiums).")
    print("  - If QAOA is slow, reduce CANDIDATES_PER_GROUP (e.g., 6) or QAOA_MAXITER.")
    print("  - For better risk fidelity, compute tail_years on portfolio netloss iteratively (outer loop),")
    print("    or cluster years to reduce scenarios.")


if __name__ == "__main__":
    main()
    
    
    
    
    
