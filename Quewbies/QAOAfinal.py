import numpy as np
import pandas as pd
from itertools import product as iproduct
import time
import os
from itertools import combinations

import numpy as np
from qiskit_optimization import QuadraticProgram
import math
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import QAOAAnsatz

GLOBAL_PARAMS = {
    # ─────────────────────────────────────────
    # 1. Asset Scaling (calibrated)
    # ─────────────────────────────────────────
    "a0": 20,
    "a1": 1.2,

    # ─────────────────────────────────────────
    # 2. Exposure Scaling (calibrated)
    # ─────────────────────────────────────────
    "e0": 0.5,
    "e1": 1.0,
    "e2": 0.5,
    "e3": 0.5,
    "e4": 0.3,
    "Imax": 5,

    # ─────────────────────────────────────────
    # 3. Security Posture
    # ─────────────────────────────────────────
    "z0": -1.0,
    "w":  np.array([0.4, 0.5, 0.3, 0.4, 0.3, 0.5]),
    "wij": {(1, 3): 0.10, (0, 5): 0.08},

    # ─────────────────────────────────────────
    # 4. Frequency Model (calibrated)
    # ─────────────────────────────────────────
    "lambda0": 0.002,
    "lambda_C": 0.3,
    "lambda_R": 0.2,

    # ─────────────────────────────────────────
    # 5. Impact Model (calibrated)
    # ─────────────────────────────────────────
    "c1": 2500,
    "d0": 3,
    "d1": 0.02,
    "gamma": 0.05,

    # ─────────────────────────────────────────
    # 6. Policy Terms
    # ─────────────────────────────────────────
    "deductible": 10_000,
    "limit":      1_000_000,
    "k":          1.0,

    # ─────────────────────────────────────────
    # 7. Acceptance Model (calibrated)
    # ─────────────────────────────────────────
    "eta0": 0.4,
    "eta_p": 2e-5,
    "eta_c": 0.5,
    "eta_u": 1e-6,
    "eta_d": 1e-5,

    # ─────────────────────────────────────────
    # 8. Penalties (unchanged)
    # ─────────────────────────────────────────
    "mu_B":   1e-4,
    "mu_dep": 5.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Control Definitions
# ─────────────────────────────────────────────────────────────────────────────
CONTROL_NAMES = ["MFA", "EDR", "Offline_Backup"]
N_CONTROLS = 3
KAPPA = np.array([2_500, 4_000, 3_000])
ALPHA = np.array([0.30,  0.25,  0.05])
BETA  = np.array([0.10,  0.20,  0.40])
FRIC  = np.array([0.15,  0.20,  0.10])

# Pairwise interactions: (alpha_ij, beta_ij)
INTERACTIONS = {(0, 1): (0.05, 0.03)}
DEPENDENCIES = [(4, 2)]   # IR_Plan requires Offline_Backup

# Global premium hard bounds (still used for clipping)
P_MIN, P_MAX = 5_000, 60_000


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


def compute_lambda_Z(lambda0, g, lambda_C, C, lambda_R, R):
    return lambda0 * g * (1 + lambda_C * C) * (1 + lambda_R * R)

def compute_A_hat(a0, a1, N):
    return a0 + a1 * N

def compute_E_hat(e0, e1, As, e2, C, e3, R, e4, I, Imax):
    return (
        e0
        + e1 * As / 100
        + e2 * C
        + e3 * R
        + e4 * min(I, Imax)
    )

def compute_Lambda(lambda_Z, A_hat, E_hat):
    return lambda_Z * A_hat * E_hat

def compute_S0(z0):
    return 1 / (1 + np.exp(-z0))

def compute_s_i(S0, w_i):
    return S0 * (1 - S0) * w_i

def compute_s_ij(S0, w_ij):
    return S0 * (1 - S0) * w_ij

def compute_P0(Lambda, S0):
    return Lambda * (1 - S0)

def compute_Pi(Lambda, S0, alpha_i, w_i):
    return Lambda * (
        -(1 - S0) * alpha_i
        - S0 * (1 - S0) * w_i * (1 - alpha_i)
    )

def compute_Pij(Lambda, S0, alpha_i, alpha_j, alpha_ij, w_i, w_j, w_ij):
    return Lambda * (
        - S0 * (1 - S0) * w_ij
        - (1 - S0) * alpha_ij
        + alpha_i * S0 * (1 - S0) * w_j
        + alpha_j * S0 * (1 - S0) * w_i
    )

def compute_I0(c1, a0, a1, N, Rev, d0, d1, S0, gamma):
    A = a0 + a1 * N
    return (
        c1 * A
        + Rev/365 * (d0 + d1 * A * (1 - S0))
        + gamma * Rev
    )

def compute_Ii_raw(Rev, d1, a0, a1, N, s_i):
    A = a0 + a1 * N
    return - Rev/365 * d1 * A * s_i

def compute_Iij_raw(Rev, d1, a0, a1, N, s_ij):
    A = a0 + a1 * N
    return - Rev/365 * d1 * A * s_ij

def compute_Ii(Ii_raw, I0, beta_i):
    return Ii_raw - I0 * beta_i + Ii_raw * beta_i

def compute_Iij(Iij_raw, I0, Ii_raw, Ij_raw, beta_i, beta_j, beta_ij):
    return (
        Iij_raw
        - I0 * beta_ij
        + Ii_raw * beta_j
        + Ij_raw * beta_i
    )

def compute_E0(P0, I0, d):
    return P0 * (I0 - d)

def compute_Ei(P0, Ii, I0, d, Pi):
    I0_tilde = I0 - d
    return P0 * Ii + I0_tilde * Pi + Pi * Ii

def compute_Eij(P0, Iij, I0, d, Pij, Pi, Pj, Ii, Ij):
    I0_tilde = I0 - d
    return (
        P0 * Iij
        + I0_tilde * Pij
        + Pi * Ij
        + Pj * Ii
    )

def compute_A0(L0):
    return 1 / (1 + np.exp(-L0))

def compute_alpha0(A0, eta_tilde0, L0):
    return A0 + A0 * (1 - A0) * (eta_tilde0 - L0)

def compute_alpha_p(A0, eta_p_Z):
    return - A0 * (1 - A0) * eta_p_Z

def compute_alpha_i(A0, eta_c, f_i):
    return - A0 * (1 - A0) * eta_c * f_i

def compute_C(alpha0, E0):
    return - alpha0 * E0

def compute_Cp(alpha0, alpha_p, E0):
    return alpha0 - alpha_p * E0

def compute_Cpp(alpha_p):
    return alpha_p

def compute_Ci(alpha0, alpha_i, kappa_i, Ei, E0):
    return (
        - alpha0 * (kappa_i + Ei)
        - E0 * alpha_i
        - alpha_i * (kappa_i + Ei)
    )

def compute_Cip(alpha_i, alpha_p, kappa_i, Ei):
    return alpha_i - alpha_p * (kappa_i + Ei)

def compute_Cij(alpha0, alpha_i, alpha_j, kappa_i, kappa_j, Ei, Ej, Eij):
    return (
        - alpha0 * Eij
        - alpha_i * (kappa_j + Ej)
        - alpha_j * (kappa_i + Ei)
    )

def build_qubo_matrix(company, M=6, Delta=None, normalize=False):

    if Delta is None:
        Delta = (P_MAX - P_MIN) / (2**M - 1)
    
    gp = GLOBAL_PARAMS
    n_controls = N_CONTROLS
    n_bits = M
    n = n_controls + n_bits

    Q = np.zeros((n, n))

    # -------------------------------------------------
    # 1. PRE-COMPUTATIONS
    # -------------------------------------------------

    lambda_Z = compute_lambda_Z(
        gp["lambda0"], company["g"],
        gp["lambda_C"], company["C"],
        gp["lambda_R"], company["R"]
    )

    A_hat = compute_A_hat(gp["a0"], gp["a1"], company["N"])

    E_hat = compute_E_hat(
        gp["e0"], gp["e1"], company["As"],
        gp["e2"], company["C"],
        gp["e3"], company["R"],
        gp["e4"], company["I"],
        gp["Imax"]
    )

    Lambda = compute_Lambda(lambda_Z, A_hat, E_hat)
    S0 = compute_S0(gp["z0"])

    s_i = np.array([compute_s_i(S0, w) for w in gp["w"]])

    s_ij = {
        (i, j): compute_s_ij(S0, w_ij)
        for (i, j), w_ij in gp["wij"].items()
    }

    # -------------------------------------------------
    # 2. PROBABILITY COEFFICIENTS
    # -------------------------------------------------

    P0 = compute_P0(Lambda, S0)

    Pi = np.array([
        compute_Pi(Lambda, S0, ALPHA[i], gp["w"][i])
        for i in range(n_controls)
    ])

    Pij = {}
    for i, j in combinations(range(n_controls), 2):

        alpha_ij, _ = INTERACTIONS.get((i, j), (0.0, 0.0))
        w_ij = gp["wij"].get((i, j), 0.0)

        Pij[(i, j)] = compute_Pij(
            Lambda, S0,
            ALPHA[i], ALPHA[j], alpha_ij,
            gp["w"][i], gp["w"][j],
            w_ij
        )

    # -------------------------------------------------
    # 3. IMPACT COEFFICIENTS
    # -------------------------------------------------

    I0 = compute_I0(
        gp["c1"], gp["a0"], gp["a1"],
        company["N"], company["Rev"],
        gp["d0"], gp["d1"], S0,
        gp["gamma"]
    )

    Ii_raw = np.array([
        compute_Ii_raw(
            company["Rev"], gp["d1"],
            gp["a0"], gp["a1"],
            company["N"], s_i[i]
        )
        for i in range(n_controls)
    ])

    Ii = np.array([
        compute_Ii(Ii_raw[i], I0, BETA[i])
        for i in range(n_controls)
    ])

    Iij = {}
    for i, j in combinations(range(n_controls), 2):

        _, beta_ij = INTERACTIONS.get((i, j), (0.0, 0.0))
        w_ij = gp["wij"].get((i, j), 0.0)

        s_ij_val = compute_s_ij(S0, w_ij)

        Iij_raw_val = compute_Iij_raw(
            company["Rev"], gp["d1"],
            gp["a0"], gp["a1"],
            company["N"],
            s_ij_val
        )

        Iij[(i, j)] = compute_Iij(
            Iij_raw_val,
            I0,
            Ii_raw[i], Ii_raw[j],
            BETA[i], BETA[j],
            beta_ij
        )

    # -------------------------------------------------
    # 4. EXPECTED PAYOUT
    # -------------------------------------------------

    E0 = compute_E0(P0, I0, gp["deductible"])

    Ei = np.array([
        compute_Ei(P0, Ii[i], I0,
                   gp["deductible"], Pi[i])
        for i in range(n_controls)
    ])

    Eij = {}
    for i, j in combinations(range(n_controls), 2):

        Eij[(i, j)] = compute_Eij(
            P0,
            Iij[(i, j)],
            I0,
            gp["deductible"],
            Pij[(i, j)],
            Pi[i], Pi[j],
            Ii[i], Ii[j]
        )

    # -------------------------------------------------
    # 5. ACCEPTANCE COEFFICIENTS
    # -------------------------------------------------

    L0 = gp["eta0"] + gp["eta_u"] * gp["limit"] - gp["eta_d"] * gp["deductible"]
    A0 = compute_A0(L0)

    eta_tilde0 = (
        gp["eta0"]
        + gp["eta_p"] * company["pcomp"]
        + gp["eta_u"] * gp["limit"]
        - gp["eta_d"] * gp["deductible"]
    )

    alpha0 = compute_alpha0(A0, eta_tilde0, L0)
    alpha_p = compute_alpha_p(A0, gp["eta_p"])

    alpha_i = np.array([
        compute_alpha_i(A0, gp["eta_c"], FRIC[i])
        for i in range(n_controls)
    ])

    # -------------------------------------------------
    # 6. PROFIT COEFFICIENTS
    # -------------------------------------------------

    Ci = np.array([
        compute_Ci(alpha0, alpha_i[i],
                   KAPPA[i], Ei[i], E0)
        for i in range(n_controls)
    ])

    Cip = np.array([
        compute_Cip(alpha_i[i], alpha_p,
                    KAPPA[i], Ei[i])
        for i in range(n_controls)
    ])

    Cij = {}
    for i, j in combinations(range(n_controls), 2):

        Cij[(i, j)] = compute_Cij(
            alpha0,
            alpha_i[i], alpha_i[j],
            KAPPA[i], KAPPA[j],
            Ei[i], Ei[j],
            Eij[(i, j)]
        )

    Cp = compute_Cp(alpha0, alpha_p, E0)
    Cpp = compute_Cpp(alpha_p)

    # -------------------------------------------------
    # 7. BUILD Q MATRIX
    # -------------------------------------------------

    p_min = P_MIN

    # --- Linear control terms ---
    for i in range(n_controls):
        Q[i, i] += Ci[i]

        # ADD MISSING TERM: Cip * p_min * x_i
        Q[i, i] += Cip[i] * p_min

    # --- Control-control quadratic ---
    for (i, j), val in Cij.items():
        Q[i, j] += val
        Q[j, i] += val

    # --- Premium bits ---
    for m in range(n_bits):
        idx_m = n_controls + m

        # Linear bit term
        Q[idx_m, idx_m] += (
            Cp * Delta * 2**m
            + Cpp * (
                2 * p_min * Delta * 2**m
                + Delta**2 * 2**(2*m)
            )
        )

        # Bit-bit quadratic
        for n2 in range(m + 1, n_bits):
            idx_n = n_controls + n2
            val = Cpp * Delta**2 * 2**(m + n2)
            Q[idx_m, idx_n] += val
            Q[idx_n, idx_m] += val

        # Control-bit interactions
        for i in range(n_controls):
            val = Cip[i] * Delta * 2**m
            Q[i, idx_m] += val
            Q[idx_m, i] += val

    # -------------------------------------------------
    # Optional normalization
    # -------------------------------------------------

    if normalize:
        max_abs = np.max(np.abs(Q))
        if max_abs > 0:
            Q = Q / max_abs

    return Q

def compute_profit_from_bitstring(bitstring, company, M=3, Delta=None):

    if Delta is None:
        Delta = (P_MAX - P_MIN) / (2**M - 1)

    bitstring = np.array(bitstring)

    x_controls = bitstring[:N_CONTROLS]
    x_bits = bitstring[N_CONTROLS:N_CONTROLS+M]

    # Premium reconstruction
    p = P_MIN + Delta * sum(
        x_bits[m] * 2**m for m in range(M)
    )

    # Reconstruct Q without normalization to get the correct energy
    Q = build_qubo_matrix(
        company,
        M=M,
        Delta=Delta,
        normalize=False
    )

    # Energy QUBO 
    energy = bitstring @ Q @ bitstring

    # Profit = -energy
    profit = -energy

    return {
        "premium": p,
        "controls": x_controls,
        "profit": profit
    }

def build_qp_from_Q(Q, n_controls, min_active=None):
    n = Q.shape[0]
    qp = QuadraticProgram()

    # Binary variables
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    # Linear terms
    linear = {f"x{i}": Q[i, i] for i in range(n)}

    # Quadratic terms
    quadratic = {}
    for i in range(n):
        for j in range(i+1, n):
            if Q[i, j] != 0:
                quadratic[(f"x{i}", f"x{j}")] = Q[i, j]

    qp.minimize(linear=linear, quadratic=quadratic)

    if min_active is not None:
        coeffs = {f"x{i}": 1 for i in range(n_controls)}
        qp.linear_constraint(
            linear=coeffs,
            sense=">=",
            rhs=min_active,
            name="min_controls"
        )
    bit_coeffs = {
        f"x{i}": 1 for i in range(n_controls, n)
    }
    qp.linear_constraint(
        linear=bit_coeffs,
        sense=">=",
        rhs=1,
        name="min_premium_bits"
    )

    return qp


company = sample_company(rng=np.random.default_rng(),industry="tech", size_tier="medium")

Q = build_qubo_matrix(
    company,
    M=3,
    Delta=2000,
    normalize=False
)

n_controls = N_CONTROLS
min_active = math.ceil(n_controls / 2)

qp = build_qp_from_Q(Q, n_controls, min_active=min_active)

converter = QuadraticProgramToQubo(penalty=1e6)
qubo = converter.convert(qp)

print("Variables after conversion:", qubo.get_num_vars())
print(qubo.prettyprint())

op, offset = qubo.to_ising()
print("Total qubits number:", op.num_qubits)

sampler = StatevectorSampler()

qaoa = QAOA(
    sampler=sampler,
    optimizer=COBYLA(maxiter=200),
    reps=2
)

solver = MinimumEigenOptimizer(qaoa)
result = solver.solve(qp)

print("Solution:", result.x)
print("Value:", result.fval)
res = compute_profit_from_bitstring(
    result.x,
    company,
    M=3,
    Delta=2000
)

print(res)
