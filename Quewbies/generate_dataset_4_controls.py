"""
Synthetic dataset generator for ransomware insurance QUBO/QAOA optimization.
Generates company profiles with security controls and premium decisions.
"""

import numpy as np
import pandas as pd
from itertools import product as iproduct
import time
import os
# ─────────────────────────────────────────────────────────────────────────────
# Global Parameters
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_PARAMS = {
    # Proxy coefficients
    "a0": 50, "a1": 2,
    "e0": 1.0, "e1": 0.4, "e2": 0.3, "e3": 0.2, "e4": 0.1,
    "Imax": 5,

    # Security posture weights (per-control and pairwise) - Truncated to 4
    "z0": -1.0,
    "w":  np.array([0.4, 0.5, 0.3, 0.4]),
    "wij": {(1, 3): 0.10},   # EDR+Patch synergy kept; NetSeg removed

    # Frequency model
    "lambda0": 0.05, "lambda_C": 0.3, "lambda_R": 0.2,

    # Severity model
    "c1": 500, "d0": 3, "d1": 0.02, "gamma": 0.02,

    # Policy terms
    "deductible": 10_000,
    "limit":      1_000_000,
    "k":          1.0,

    # Acceptance model
    "eta0": 1.5,
    "eta_p": 0.00008,
    "eta_c": 2.0,
    "eta_u": 0.0000005,
    "eta_d": 0.00002,

    # Penalties 
    "mu_B":   1e-4,   
    "mu_dep": 5.0,    
}

# ─────────────────────────────────────────────────────────────────────────────
# Control Definitions
# ─────────────────────────────────────────────────────────────────────────────
CONTROL_NAMES = ["MFA", "EDR", "Offline_Backup", "Patch_Mgmt"]
N_CONTROLS = 4

# Truncated control attributes
KAPPA = np.array([2_500, 4_000, 3_000, 2_000])
ALPHA = np.array([0.30,  0.25,  0.05,  0.20])
BETA  = np.array([0.10,  0.20,  0.40,  0.15])
FRIC  = np.array([0.15,  0.20,  0.10,  0.25])

# Pairwise interactions: (alpha_ij, beta_ij)
INTERACTIONS = {(1, 3): (0.05, 0.03)} # Dropped (0, 5)
DEPENDENCIES = []                     # Dropped IR_Plan -> Backup dependency

P_MIN, P_MAX = 5_000, 60_000


# # ─────────────────────────────────────────────────────────────────────────────
# # Global Parameters
# # ─────────────────────────────────────────────────────────────────────────────
# GLOBAL_PARAMS = {
#     # Proxy coefficients
#     "a0": 50, "a1": 2,
#     "e0": 1.0, "e1": 0.4, "e2": 0.3, "e3": 0.2, "e4": 0.1,
#     "Imax": 5,

#     # Security posture weights (per-control and pairwise)
#     "z0": -1.0,
#     "w":  np.array([0.4, 0.5, 0.3, 0.4, 0.3, 0.5]),
#     "wij": {(1, 3): 0.10, (0, 5): 0.08},   # EDR+Patch, MFA+NetSeg synergies

#     # Frequency model
#     "lambda0": 0.05, "lambda_C": 0.3, "lambda_R": 0.2,

#     # Severity model
#     "c1": 500, "d0": 3, "d1": 0.02, "gamma": 0.02,

#     # Policy terms (fixed in this version)
#     "deductible": 10_000,
#     "limit":      1_000_000,
#     "k":          1.0,

#     # Acceptance model
#     "eta0": 1.5,
#     "eta_p": 0.00008,
#     "eta_c": 2.0,
#     "eta_u": 0.0000005,
#     "eta_d": 0.00002,

#     # Penalties (these are "soft constraints" in the analytical objective)
#     "mu_B":   1e-4,   # budget penalty weight
#     "mu_dep": 5.0,    # dependency penalty weight
# }


# # ─────────────────────────────────────────────────────────────────────────────
# # Control Definitions
# # ─────────────────────────────────────────────────────────────────────────────
# CONTROL_NAMES = ["MFA", "EDR", "Offline_Backup", "Patch_Mgmt", "IR_Plan", "Net_Segmentation"]
# N_CONTROLS = 6
# KAPPA = np.array([2_500, 4_000, 3_000, 2_000, 1_500, 5_000])
# ALPHA = np.array([0.30,  0.25,  0.05,  0.20,  0.10,  0.15])
# BETA  = np.array([0.10,  0.20,  0.40,  0.15,  0.15,  0.20])
# FRIC  = np.array([0.15,  0.20,  0.10,  0.25,  0.10,  0.30])

# # Pairwise interactions: (alpha_ij, beta_ij)
# INTERACTIONS = {(1, 3): (0.05, 0.03), (0, 5): (0.03, 0.02)}
# DEPENDENCIES = [(4, 2)]   # IR_Plan requires Offline_Backup

# # Global premium hard bounds (still used for clipping)
# P_MIN, P_MAX = 5_000, 60_000


# ─────────────────────────────────────────────────────────────────────────────
# Company Generator
# ─────────────────────────────────────────────────────────────────────────────

INDUSTRY_PROFILES = {
    # name: (g_mean, g_std, rev_range, n_range, cloud_mean, incident_mean)
    "healthcare":    (1.8, 0.2, (2e6,  50e6),  (50,  500),  0.55, 2.5),
    "finance":       (1.6, 0.2, (5e6, 200e6),  (100, 2000), 0.60, 1.8),
    "tech":          (1.3, 0.2, (1e6,  80e6),  (30,  800),  0.80, 1.5),
    "manufacturing": (1.1, 0.2, (3e6, 100e6),  (100, 3000), 0.35, 1.0),
    "retail":        (1.2, 0.2, (2e6,  60e6),  (50,  2000), 0.50, 1.2),
    "education":     (1.0, 0.1, (1e6,  20e6),  (50,  1000), 0.60, 0.8),
    "government":    (1.4, 0.2, (5e6, 500e6),  (200, 5000), 0.40, 2.0),
    "energy":        (1.5, 0.2, (10e6,300e6),  (200, 4000), 0.30, 1.3),
}

SIZE_TIERS = {
    "micro":       (0.01, 0.03),
    "small":       (0.05, 0.10),
    "medium":      (0.20, 0.25),
    "large":       (0.60, 0.70),
    "enterprise":  (1.00, 1.00),
}


def sample_company(rng, industry=None, size_tier=None):
    if industry is None:
        industry = rng.choice(list(INDUSTRY_PROFILES.keys()))
    if size_tier is None:
        size_tier = rng.choice(list(SIZE_TIERS.keys()),
                               p=[0.15, 0.30, 0.30, 0.20, 0.05])

    prof = INDUSTRY_PROFILES[industry]
    g_mean, g_std, (rev_min, rev_max), (n_min, n_max), cloud_mean, inc_mean = prof
    size_rev_f, size_n_f = SIZE_TIERS[size_tier]

    # Revenue: uniform within a size band inside the industry range
    rev_band_max = rev_min + size_rev_f * (rev_max - rev_min)
    rev_band_min = rev_min + max(0, size_rev_f - 0.15) * (rev_max - rev_min)
    rev = rng.uniform(max(rev_band_min, rev_min * 0.5), rev_band_max)

    # Employees: correlated with revenue but with noise
    n_band_max = int(n_min + size_n_f * (n_max - n_min))
    n_band_min = int(n_min + max(0, size_n_f - 0.15) * (n_max - n_min))
    N = rng.integers(max(n_band_min, n_min), max(n_band_max + 1, n_min + 1))

    # Cloud dependency
    C = np.clip(rng.normal(cloud_mean, 0.15), 0.05, 0.98)

    # Remote ratio
    R = np.clip(rng.normal(0.3 + 0.3 * C, 0.12), 0.0, 0.95)

    # Attack surface
    As_mean = 20 + 0.08 * min(N, 500) + 30 * C
    As = np.clip(rng.normal(As_mean, 10), 5, 99)

    # Past incidents
    I = min(int(rng.poisson(inc_mean)), 10)

    # Industry risk factor
    g = max(0.5, rng.normal(g_mean, g_std))

    # Competitor price: function of Rev and risk, with noise
    pcomp_base = 0.005 * rev * g + 5000
    pcomp = max(3000, rng.normal(pcomp_base, pcomp_base * 0.15))

    # Budget: fraction of revenue
    budget_pct = rng.uniform(0.003, 0.012)   # 0.3–1.2% of revenue
    budget = rev * budget_pct

    return {
        "industry":   industry,
        "size_tier":  size_tier,
        "Rev":        float(round(rev, 0)),
        "N":          int(N),
        "C":          float(round(C, 4)),
        "R":          float(round(R, 4)),
        "As":         float(round(As, 2)),
        "I":          int(I),
        "g":          float(round(g, 4)),
        "pcomp":      float(round(pcomp, 0)),
        "budget":     float(round(budget, 0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Premium Grid Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_premium_grid(company, K=16,
                       low_mult=0.70, high_mult=1.50,
                       p_min=P_MIN, p_max=P_MAX):
    pcomp = float(company["pcomp"])
    lo = max(p_min, low_mult * pcomp)
    hi = min(p_max, high_mult * pcomp)
    if hi <= lo + 1:
        lo = p_min
        hi = p_max
    grid = np.linspace(lo, hi, K)
    grid = np.clip(grid, p_min, p_max)
    return grid.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Risk and Profit Models
# ─────────────────────────────────────────────────────────────────────────────

def compute_proxies(company):
    gp = GLOBAL_PARAMS
    N  = company["N"]
    C  = company["C"]
    R  = company["R"]
    As = company["As"]
    I  = company["I"]
    A_hat     = gp["a0"] + gp["a1"] * N
    E_hat     = (gp["e0"]
                 + gp["e1"] * As/100
                 + gp["e2"] * C
                 + gp["e3"] * R
                 + gp["e4"] * min(I, gp["Imax"]))
    Rev_daily = company["Rev"] / 365.0
    return float(A_hat), float(E_hat), float(Rev_daily)


def security_score(x):
    gp = GLOBAL_PARAMS
    z  = gp["z0"] + float(np.dot(gp["w"], x))
    for (i, j), wij in gp["wij"].items():
        z += float(wij) * float(x[i]) * float(x[j])
    return float(1.0 / (1.0 + np.exp(-z)))


def ransomware_prob(x, company, A_hat, E_hat):
    gp = GLOBAL_PARAMS
    g  = company["g"]
    C  = company["C"]
    R  = company["R"]

    lam_Z    = gp["lambda0"] * g * (1.0 + gp["lambda_C"] * C) * (1.0 + gp["lambda_R"] * R)

    # multiplicative frequency reductions
    phi_freq = np.prod((1.0 - ALPHA) ** x)
    for (i, j), (a_ij, _) in INTERACTIONS.items():
        phi_freq *= (1.0 - a_ij) ** (x[i] * x[j])

    S       = security_score(x)
    lam_eff = lam_Z * phi_freq

    return float(1.0 - np.exp(-lam_eff * A_hat * E_hat * (1.0 - S)))


def effective_impact(x, company, A_hat, Rev_daily):
    gp       = GLOBAL_PARAMS
    S        = security_score(x)

    recovery = gp["c1"] * A_hat
    downtime = gp["d0"] + gp["d1"] * A_hat * (1.0 - S)
    bi       = Rev_daily * downtime
    legal    = gp["gamma"] * company["Rev"]

    impact0  = recovery + bi + legal

    # multiplicative severity reductions
    phi_sev  = np.prod((1.0 - BETA) ** x)
    for (i, j), (_, b_ij) in INTERACTIONS.items():
        phi_sev *= (1.0 - b_ij) ** (x[i] * x[j])

    return float(impact0 * phi_sev)


def insurance_payout(impact):
    gp = GLOBAL_PARAMS
    return float(gp["k"] * min(max(impact - gp["deductible"], 0.0), gp["limit"]))


def acceptance_prob(x, p, company):
    gp    = GLOBAL_PARAMS
    F     = float(np.dot(FRIC, x))
    logit = (gp["eta0"]
             - gp["eta_p"] * (p - company["pcomp"])
             - gp["eta_c"] * F
             + gp["eta_u"] * gp["limit"]
             - gp["eta_d"] * gp["deductible"])
    return float(1.0 / (1.0 + np.exp(-logit)))


def penalty_terms(x, company):
    gp = GLOBAL_PARAMS
    C_ctrl = float(np.dot(KAPPA, x))

    # budget excess (hinge squared)
    budget_excess = max(C_ctrl - float(company["budget"]), 0.0)
    P_B   = gp["mu_B"] * budget_excess ** 2

    # dependency: IR Plan requires Offline Backup
    # DEPENDENCIES: (i, j) meaning if x[i]=1 then x[j]=1
    P_dep = 0.0
    for i, j in DEPENDENCIES:
        P_dep += gp["mu_dep"] * float(x[i]) * (1.0 - float(x[j]))

    return float(P_B + P_dep), float(P_B), float(P_dep)


def expected_profit(x, p, company):
    A_hat, E_hat, Rev_daily = compute_proxies(company)
    A   = acceptance_prob(x, p, company)
    C   = float(np.dot(KAPPA, x))
    pr  = ransomware_prob(x, company, A_hat, E_hat)
    imp = effective_impact(x, company, A_hat, Rev_daily)
    EP  = pr * insurance_payout(imp)
    return float(A * (p - C - EP))


def joint_loss(x, p, company):
    prof = expected_profit(x, p, company)
    pnl, _, _ = penalty_terms(x, company)
    return float(-prof + pnl)


# ─────────────────────────────────────────────────────────────────────────────
# Optimal Decision Finder
# ─────────────────────────────────────────────────────────────────────────────

def best_premium_idx_for_x(company, x, p_grid):
    """For a fixed x, return k* = argmin_k joint_loss(x, p_grid[k])."""
    losses = [joint_loss(x, float(p_grid[k]), company) for k in range(len(p_grid))]
    k_star = int(np.argmin(losses))
    return k_star, float(losses[k_star])


def find_optimal_decision_discrete(company, p_grid):
    best_x = None
    best_k = None
    best_loss = np.inf

    for bits in iproduct([0, 1], repeat=N_CONTROLS):
        x = np.array(bits, dtype=float)
        k_star, loss_star = best_premium_idx_for_x(company, x, p_grid)

        if loss_star < best_loss:
            best_loss = float(loss_star)
            best_x = x.copy()
            best_k = int(k_star)

    opt_p = float(p_grid[best_k])
    opt_profit = expected_profit(best_x, opt_p, company)
    return best_x, best_k, opt_p, float(best_loss), float(opt_profit)


# ─────────────────────────────────────────────────────────────────────────────
# Row Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_row(sample_id, company_id, company, x, premium_idx, p_grid,
              opt_x, opt_k, opt_p, is_optimal):
    p = float(p_grid[int(premium_idx)])

    A_hat, E_hat, Rev_daily = compute_proxies(company)
    S   = security_score(x)
    pr  = ransomware_prob(x, company, A_hat, E_hat)
    imp = effective_impact(x, company, A_hat, Rev_daily)
    pay = insurance_payout(imp)
    EP  = pr * pay
    acc = acceptance_prob(x, p, company)
    C   = float(np.dot(KAPPA, x))
    F   = float(np.dot(FRIC, x))
    pnl, P_B, P_dep = penalty_terms(x, company)
    prof = acc * (p - C - EP)
    loss = -prof + pnl

    row = {
        # Metadata
        "sample_id":        int(sample_id),
        "company_id":       int(company_id),
        "industry":         company["industry"],
        "size_tier":        company["size_tier"],
        "is_optimal":       int(is_optimal),

        # Company features
        "rev":              company["Rev"],
        "n_employees":      company["N"],
        "cloud_dep":        company["C"],
        "remote_ratio":     company["R"],
        "attack_surface":   company["As"],
        "past_incidents":   company["I"],
        "industry_risk":    company["g"],
        "competitor_price": company["pcomp"],
        "budget":           company["budget"],

        # Derived proxies
        "A_hat":            round(A_hat, 2),
        "E_hat":            round(E_hat, 4),
        "rev_daily":        round(Rev_daily, 2),

        # Controls
        "x_MFA":            int(x[0]),
        "x_EDR":            int(x[1]),
        "x_Offline_Backup": int(x[2]),
        "x_Patch_Mgmt":     int(x[3]),
        # "x_IR_Plan":        int(x[4]),
        # "x_Net_Seg":        int(x[5]),
        "n_controls_req":   int(x.sum()),
        "control_cost":     round(C, 2),
        "friction_score":   round(F, 4),

        # Premium (one-hot index)
        "premium_idx":      int(premium_idx),
        "premium":          round(p, 2),
        "premium_delta":    round(p - company["pcomp"], 2),  # vs competitor

        # Risk outcomes
        "security_score":   round(S, 6),
        "pr_ransomware":    round(pr, 6),
        "impact":           round(imp, 2),
        "payout":           round(pay, 2),
        "expected_payout":  round(EP, 2),
        "acceptance_prob":  round(acc, 6),

        # Financial outcomes
        "penalty_total":    round(pnl, 6),
        "penalty_budget":   round(P_B, 6),
        "penalty_dep":      round(P_dep, 6),
        "expected_profit":  round(prof, 2),
        "joint_loss":       round(loss, 6),

        # Optimal labels
        "opt_x_MFA":            int(opt_x[0]),
        "opt_x_EDR":            int(opt_x[1]),
        "opt_x_Offline_Backup": int(opt_x[2]),
        "opt_x_Patch_Mgmt":     int(opt_x[3]),
        # "opt_x_IR_Plan":        int(opt_x[4]),
        # "opt_x_Net_Seg":        int(opt_x[5]),
        "opt_premium_idx":      int(opt_k),
        "opt_premium":          round(float(opt_p), 2),
        "opt_n_controls":       int(opt_x.sum()),
        "opt_control_cost":     round(float(np.dot(KAPPA, opt_x)), 2),
    }
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    n_companies: int = 500,
    K_premiums: int = 16,
    exploration_rows_per_company: int = 32,
    include_all_x_configs: bool = True,
    include_all_premiums: bool = True,
    include_optimal_row: bool = True,
    seed: int = 42,
    verbose: bool = True,
):
    rng = np.random.default_rng(seed)
    rows = []
    sample_id = 0
    t0 = time.time()

    industries = list(INDUSTRY_PROFILES.keys())
    size_tiers = list(SIZE_TIERS.keys())

    industry_cycle = [industries[i % len(industries)] for i in range(n_companies)]
    size_cycle     = [size_tiers[i % len(size_tiers)]  for i in range(n_companies)]
    rng.shuffle(industry_cycle)
    rng.shuffle(size_cycle)

    for company_id in range(n_companies):
        if verbose and company_id % 50 == 0:
            elapsed = time.time() - t0
            rate = (company_id + 1) / max(elapsed, 0.01)
            eta = (n_companies - company_id) / max(rate, 0.01)
            print(f"  Company {company_id+1:>4}/{n_companies}  [{elapsed:5.1f}s elapsed, ETA {eta:5.1f}s]  Rows: {len(rows):,}")

        # Sample company
        company = sample_company(
            rng,
            industry=industry_cycle[company_id],
            size_tier=size_cycle[company_id],
        )

        # Build premium grid and find optimal decision
        p_grid = build_premium_grid(company, K=K_premiums)
        opt_x, opt_k, opt_p, opt_loss, opt_profit = find_optimal_decision_discrete(company, p_grid)

        # Add optimal row
        if include_optimal_row:
            rows.append(build_row(
                sample_id, company_id, company,
                x=opt_x, premium_idx=opt_k, p_grid=p_grid,
                opt_x=opt_x, opt_k=opt_k, opt_p=opt_p,
                is_optimal=True
            ))
            sample_id += 1

        # Enumerate all control configs
        if include_all_x_configs:
            for bits in iproduct([0, 1], repeat=N_CONTROLS):
                x = np.array(bits, dtype=float)

                if include_all_premiums:
                    for k in range(K_premiums):
                        # avoid duplicating the explicit optimal row if we already added it
                        if include_optimal_row and np.array_equal(x, opt_x) and k == opt_k:
                            continue
                        rows.append(build_row(
                            sample_id, company_id, company,
                            x=x, premium_idx=k, p_grid=p_grid,
                            opt_x=opt_x, opt_k=opt_k, opt_p=opt_p,
                            is_optimal=(np.array_equal(x, opt_x) and k == opt_k)
                        ))
                        sample_id += 1
                else:
                    k_star, _ = best_premium_idx_for_x(company, x, p_grid)
                    if include_optimal_row and np.array_equal(x, opt_x) and k_star == opt_k:
                        continue
                    rows.append(build_row(
                        sample_id, company_id, company,
                        x=x, premium_idx=k_star, p_grid=p_grid,
                        opt_x=opt_x, opt_k=opt_k, opt_p=opt_p,
                        is_optimal=(np.array_equal(x, opt_x) and k_star == opt_k)
                    ))
                    sample_id += 1

        # Add random exploration rows
        for _ in range(exploration_rows_per_company):
            x = rng.integers(0, 2, size=N_CONTROLS).astype(float)
            pcomp = company["pcomp"]
            p_vals = p_grid

            regime = rng.choice(["below_comp", "near_comp", "above_comp"], p=[0.25, 0.35, 0.40])

            if regime == "below_comp":
                candidates = np.where(p_vals <= pcomp)[0]
            elif regime == "near_comp":
                candidates = np.where((p_vals >= 0.8 * pcomp) & (p_vals <= 1.3 * pcomp))[0]
            else:
                candidates = np.where(p_vals >= 1.1 * pcomp)[0]

            if len(candidates) == 0:
                premium_idx = int(rng.integers(0, K_premiums))
            else:
                premium_idx = int(rng.choice(candidates))

            if include_optimal_row and np.array_equal(x, opt_x) and premium_idx == opt_k:
                continue

            rows.append(build_row(
                sample_id, company_id, company,
                x=x, premium_idx=premium_idx, p_grid=p_grid,
                opt_x=opt_x, opt_k=opt_k, opt_p=opt_p,
                is_optimal=False
            ))
            sample_id += 1

    df = pd.DataFrame(rows)
    elapsed = time.time() - t0

    summary = {
        "n_rows":              int(len(df)),
        "n_companies":         int(n_companies),
        "K_premiums":          int(K_premiums),
        "n_optimal_rows":      int(df["is_optimal"].sum()),
        "elapsed_sec":         round(elapsed, 1),
        "rows_per_sec":        round(len(df) / max(elapsed, 1e-6), 1),
        "industries":          df["industry"].value_counts().to_dict(),
        "size_tiers":          df["size_tier"].value_counts().to_dict(),
        "mean_opt_premium":    float(df[df["is_optimal"]==1]["opt_premium"].mean()) if int(df["is_optimal"].sum()) else None,
        "mean_opt_n_controls": float(df[df["is_optimal"]==1]["opt_n_controls"].mean()) if int(df["is_optimal"].sum()) else None,
        "mean_opt_profit":     float(df[df["is_optimal"]==1]["expected_profit"].mean()) if int(df["is_optimal"].sum()) else None,
        "pct_positive_profit": float((df["expected_profit"] > 0).mean() * 100.0),
    }

    return df, summary


# ─────────────────────────────────────────────────────────────────────────────
# Train/Val/Test Split
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(df, train_frac=0.70, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    company_ids = df["company_id"].unique()
    rng.shuffle(company_ids)

    n = len(company_ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_ids = set(company_ids[:n_train])
    val_ids   = set(company_ids[n_train:n_train + n_val])
    test_ids  = set(company_ids[n_train + n_val:])

    train_df = df[df["company_id"].isin(train_ids)].copy()
    val_df   = df[df["company_id"].isin(val_ids)].copy()
    test_df  = df[df["company_id"].isin(test_ids)].copy()

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()

    df["log_rev"]         = np.log1p(df["rev"])
    df["log_n_employees"] = np.log1p(df["n_employees"])
    df["rev_per_employee"]= df["rev"] / (df["n_employees"] + 1)

    df["exposure_composite"] = (
        0.4 * df["attack_surface"] / 100.0 +
        0.3 * df["cloud_dep"] +
        0.2 * df["remote_ratio"] +
        0.1 * df["past_incidents"] / 10.0
    )

    df["risk_revenue_ratio"] = df["industry_risk"] * df["exposure_composite"]

    total_control_cost = float(KAPPA.sum())
    df["budget_tightness"] = df["budget"] / max(total_control_cost, 1.0)

    df["pcomp_as_pct_rev"] = df["competitor_price"] / (df["rev"] + 1.0)

    df["cloud_x_remote"] = df["cloud_dep"] * df["remote_ratio"]

    x_cols = ["x_MFA","x_EDR","x_Offline_Backup","x_Patch_Mgmt","x_IR_Plan","x_Net_Seg"]
    df["has_mfa_and_edr"] = (df["x_MFA"] * df["x_EDR"]).astype(int)
    df["has_backup_and_irplan"] = (df["x_Offline_Backup"] * df["x_IR_Plan"]).astype(int)
    df["pct_controls_req"] = df["n_controls_req"] / float(N_CONTROLS)

    df["premium_pct_rev"]  = df["premium"] / (df["rev"] + 1.0)
    df["premium_vs_comp"]  = df["premium"] / (df["competitor_price"] + 1.0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(df, K_premiums=16):
    issues = []

    nan_counts = df.isnull().sum()
    if nan_counts.any():
        issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")

    # Probabilities in [0, 1]
    for col in ["pr_ransomware", "acceptance_prob", "security_score",
                "cloud_dep", "remote_ratio"]:
        out = df[(df[col] < 0) | (df[col] > 1)]
        if len(out):
            issues.append(f"{col}: {len(out)} values outside [0,1]")

    # Binary columns
    for col in [c for c in df.columns if c.startswith("x_") or c.startswith("opt_x_") or c == "is_optimal"]:
        bad = df[~df[col].isin([0, 1])]
        if len(bad):
            issues.append(f"{col}: {len(bad)} non-binary values")

    # premium_idx in range
    bad_idx = df[(df["premium_idx"] < 0) | (df["premium_idx"] >= K_premiums)]
    if len(bad_idx):
        issues.append(f"premium_idx: {len(bad_idx)} values outside [0, {K_premiums-1}]")

    # Premium bounds (soft)
    out_range = df[(df["premium"] < P_MIN * 0.95) | (df["premium"] > P_MAX * 1.05)]
    if len(out_range):
        issues.append(f"{len(out_range)} premiums outside expected range")

    # Dependency check in optimal solutions: IR_Plan -> Offline_Backup
    opt = df[df["is_optimal"] == 1]
    dep_violated = opt[(opt["opt_x_IR_Plan"] == 1) & (opt["opt_x_Offline_Backup"] == 0)]
    if len(dep_violated):
        issues.append(f"{len(dep_violated)} optimal rows violate IR_Plan → Backup dependency")

    if not issues:
        print("  ✅ All validation checks passed.")
    else:
        for issue in issues:
            print(f"  ⚠️  {issue}")

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df, summary):
    opt = df[df["is_optimal"] == 1]

    print(f"\n{'='*70}")
    print(f"  DATASET SUMMARY (ONE-HOT PREMIUM)")
    print(f"{'='*70}")
    print(f"  Total rows:              {len(df):>10,}")
    print(f"  Unique companies:        {df['company_id'].nunique():>10,}")
    print(f"  Premium options K:       {summary['K_premiums']:>10d}")
    print(f"  Optimal rows:            {summary['n_optimal_rows']:>10,}")
    print(f"  Generation time:         {summary['elapsed_sec']:>9.1f}s")
    print(f"  Throughput:              {summary['rows_per_sec']:>9.1f} rows/s")

    print(f"\n  ── Company Feature Ranges ───────────────────────────")
    for col, label in [("rev","Revenue $"), ("n_employees","Employees"),
                       ("attack_surface","Attack surface"),
                       ("industry_risk","Industry risk g"),
                       ("competitor_price","Comp. price $")]:
        print(f"  {label:<22}  "
              f"min={df[col].min():>10,.1f}  "
              f"mean={df[col].mean():>10,.1f}  "
              f"max={df[col].max():>10,.1f}")

    if len(opt):
        print(f"\n  ── Optimal Decision Summary ─────────────────────────")
        print(f"  Mean optimal premium:    ${opt['opt_premium'].mean():>10,.0f}")
        print(f"  Mean optimal premium idx: {opt['opt_premium_idx'].mean():>9.2f}")
        print(f"  Mean controls required:   {opt['opt_n_controls'].mean():>9.2f} / {N_CONTROLS}")
        print(f"  Mean expected profit:    ${opt['expected_profit'].mean():>10,.0f}")

    print(f"\n  ── Column Groups (high level) ───────────────────────")
    groups = {
        "Metadata": ["sample_id","company_id","industry","size_tier","is_optimal"],
        "Company features": ["rev","n_employees","cloud_dep","remote_ratio","attack_surface","past_incidents","industry_risk","competitor_price","budget"],
        "Derived proxies": ["A_hat","E_hat","rev_daily"],
        "Controls": ["x_MFA","x_EDR","x_Offline_Backup","x_Patch_Mgmt","x_IR_Plan","x_Net_Seg","n_controls_req","control_cost","friction_score"],
        "Premium (one-hot index)": ["premium_idx","premium","premium_delta"],
        "Risk outcomes": ["security_score","pr_ransomware","impact","payout","expected_payout","acceptance_prob"],
        "Financial outcomes": ["penalty_total","expected_profit","joint_loss"],
        "Optimal labels": [c for c in df.columns if c.startswith("opt_")],
    }
    for group, cols in groups.items():
        print(f"  {group:<24} ({len(cols)} cols): {', '.join(cols[:6])}{'...' if len(cols) > 6 else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR = "dataset_output_onehot"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  RANSOMWARE INSURANCE — INPUTS-ONLY DATASET GENERATOR")
    print("=" * 70)

    print(f"\n[1/4] Generating inputs-only dataset ...")
    df, summary = generate_dataset(
        n_companies=200,
        K_premiums=16,
        exploration_rows_per_company=32,
        seed=42,
        verbose=True,
    )

    print(f"\n[2/4] Engineering features ...")
    df_feat = engineer_features(df)
    print(f"  Added {len(df_feat.columns) - len(df.columns)} engineered features.")

    print(f"\n[3/4] Splitting dataset (70/15/15)...")
    train_df, val_df, test_df = split_dataset(df_feat, train_frac=0.70, val_frac=0.15, seed=42)
    print(f"  Train:      {len(train_df):>7,} rows ({train_df['company_id'].nunique()} companies)")
    print(f"  Validation: {len(val_df):>7,} rows ({val_df['company_id'].nunique()} companies)")
    print(f"  Test:       {len(test_df):>7,} rows ({test_df['company_id'].nunique()} companies)")

    print(f"\n[4/4] Saving files ...")
    full_path  = os.path.join(OUTPUT_DIR, "ransomware_inputs_only_full.csv")
    train_path = os.path.join(OUTPUT_DIR, "ransomware_inputs_only_train.csv")
    val_path   = os.path.join(OUTPUT_DIR, "ransomware_inputs_only_val.csv")
    test_path  = os.path.join(OUTPUT_DIR, "ransomware_inputs_only_test.csv")

    df_feat.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"  ✅ {full_path:<50} ({os.path.getsize(full_path)/1024:.0f} KB)")
    print(f"  ✅ {train_path:<50} ({os.path.getsize(train_path)/1024:.0f} KB)")
    print(f"  ✅ {val_path:<50} ({os.path.getsize(val_path)/1024:.0f} KB)")
    print(f"  ✅ {test_path:<50} ({os.path.getsize(test_path)/1024:.0f} KB)")

    print(f"\n{'='*70}")
    print(f"  Inputs-only dataset ready in {OUTPUT_DIR}/")
    print(f"{'='*70}")