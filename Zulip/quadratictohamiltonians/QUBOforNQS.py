import numpy as np
import pandas as pd
import json
import math


# ============================================================
# Core formulas  (aligné sur vpennylane.py)
# ============================================================

def compute_R(P, v, M_param, C, p):
    """Reward for investigating claim i. Uses SCALED M and C."""
    return P * (v * M_param * p + C + M_param) - C


def compute_B_lin(clusters, C, alpha, k):
    """B_c = (sum_{j in I_c} C_j) * alpha * tanh(|I_c| / k)"""
    K = len(clusters)
    B_lin = np.zeros(K)
    for c in range(K):
        sum_cj = sum(C[j] for j in clusters[c])
        card = len(clusters[c])
        B_lin[c] = sum_cj * alpha * np.tanh(card / k)
    return B_lin


def build_qubo(n, K, clusters, R, B_lin, C, B_budget, lambda1, lambda2):
    M = math.floor(math.log2(B_budget)) + 1
    dim = n + K + M
    Q = np.zeros((dim, dim))
    x_off, y_off, z_off = 0, n, n + K

    # --- Cluster consistency: lambda1 * sum_c sum_{j in I_c} (1 - x_j) * y_c ---
    # = lambda1 * N_c * y_c  -  lambda1 * x_j * y_c
    for c in range(K):
        N_c = len(clusters[c])
        Q[y_off + c, y_off + c] += lambda1 * N_c          # diagonal: +lambda1 * N_c * y_c
        for j in clusters[c]:
            Q[x_off + j, y_off + c] -= lambda1 / 2        # cross: -lambda1 * x_j * y_c
            Q[y_off + c, x_off + j] -= lambda1 / 2

    # --- Budget constraint: lambda2 * (sum C_i x_i + sum 2^k z_k - B)^2 ---
    for i in range(n):
        Q[x_off + i, x_off + i] += lambda2 * (C[i] ** 2 - 2 * B_budget * C[i])
        for j in range(i + 1, n):
            val = lambda2 * C[i] * C[j]
            Q[x_off + i, x_off + j] += val
            Q[x_off + j, x_off + i] += val

    for k_bit in range(M):
        pk = 2 ** k_bit
        Q[z_off + k_bit, z_off + k_bit] += lambda2 * (pk ** 2 - 2 * B_budget * pk)
        for l in range(k_bit + 1, M):
            pl = 2 ** l
            val = lambda2 * pk * pl
            Q[z_off + k_bit, z_off + l] += val
            Q[z_off + l, z_off + k_bit] += val

    for i in range(n):
        for k_bit in range(M):
            val = lambda2 * C[i] * (2 ** k_bit)
            Q[x_off + i, z_off + k_bit] += val
            Q[z_off + k_bit, x_off + i] += val

    # --- Objective: -R_i * x_i  -  B_lin_c * y_c ---
    for i in range(n):
        Q[x_off + i, x_off + i] -= R[i]
    for c in range(K):
        Q[y_off + c, y_off + c] -= B_lin[c]

    info = {
        "x_indices": list(range(x_off, x_off + n)),
        "y_indices": list(range(y_off, y_off + K)),
        "z_indices": list(range(z_off, z_off + M)),
        "dim": dim, "M_bits": M, "n": n, "K": K,
    }
    return Q, info


# ============================================================
# Data loading
# ============================================================

def load_data(csv_path, clusters_json_path):
    df = pd.read_csv(csv_path).sort_values("claim_id").reset_index(drop=True)
    claim_to_idx = {cid: i for i, cid in enumerate(df["claim_id"])}

    with open(clusters_json_path) as f:
        raw_clusters = json.load(f)

    clusters, skipped = {}, []
    for c_str, claim_ids in raw_clusters.items():
        indices = [claim_to_idx[cid] for cid in claim_ids if cid in claim_to_idx]
        [skipped.append(cid) for cid in claim_ids if cid not in claim_to_idx]
        if indices:
            clusters[len(clusters)] = indices

    if skipped:
        print(f"[WARNING] {len(skipped)} claims not in CSV (ignored)")

    return df, clusters


# ============================================================
# All-in-one: files -> QUBO -> Ising Hamiltonian (SparsePauliOp)
# Now with cost scaling (same as vpennylane.py)
# ============================================================

def get_hamiltonian(csv_path, clusters_json_path,
                    p=0.5, alpha=0.3, k=3.0, B_budget=20000.0,
                    lambda1=3.5, lambda2=4.0, max_slack_bits=5):
    """
    Builds the QUBO with cost scaling and converts to Ising Hamiltonian.

    Parameters
    ----------
    max_slack_bits : int
        Maximum number of slack bits. Costs are scaled down to fit.

    Returns
    -------
    hamiltonian : SparsePauliOp
    offset      : float
    info        : dict
    """
    from qiskit_optimization import QuadraticProgram

    df, clusters = load_data(csv_path, clusters_json_path)
    n, K = len(df), len(clusters)

    # --- Cost scaling (same logic as vpennylane.py) ---
    M_bits_raw = math.floor(math.log2(B_budget)) + 1
    if M_bits_raw > max_slack_bits:
        scale = math.ceil(B_budget / (2**max_slack_bits - 1))
        print(f"[SCALE] Dividing costs by {scale} to fit {max_slack_bits} slack bits "
              f"(raw={M_bits_raw} bits)")
    else:
        scale = 1

    C_s = np.round(df["C_i"].values / scale).astype(int)
    M_s = np.round(df["M_i"].values / scale).astype(int)
    B_s = int(round(B_budget / scale))

    n_slack = math.floor(math.log2(B_s)) + 1
    total_qubits = n + K + n_slack
    print(f"[QUBO] n={n}, K={K}, slack={n_slack}, total_qubits={total_qubits}, scale={scale}")

    R = compute_R(df["P_i"].values, df["v_i"].values, M_s, C_s, p)
    B_lin = compute_B_lin(clusters, C_s, alpha, k)
    Q, info = build_qubo(n, K, clusters, R, B_lin, C_s, B_s, lambda1, lambda2)

    # Variable names
    var_names = ([f"x_{i}" for i in range(n)]
                 + [f"y_{c}" for c in range(K)]
                 + [f"z_{k}" for k in range(info["M_bits"])])

    # QuadraticProgram -> Ising
    qp = QuadraticProgram("qubo_claims")
    for name in var_names:
        qp.binary_var(name)

    dim = info["dim"]
    linear = {var_names[i]: float(Q[i, i]) for i in range(dim) if Q[i, i] != 0}
    quadratic = {}
    for i in range(dim):
        for j in range(i + 1, dim):
            coeff = Q[i, j] + Q[j, i]
            if coeff != 0:
                quadratic[(var_names[i], var_names[j])] = float(coeff)

    qp.minimize(linear=linear, quadratic=quadratic)

    hamiltonian, offset = qp.to_ising()

    info.update({
        "clusters": clusters,
        "claim_ids": list(df["claim_id"]),
        "var_names": var_names,
        "Q": Q,
        "cost_scale": scale,
    })

    print(f"[Hamiltonian] {hamiltonian.num_qubits} qubits, "
          f"{len(hamiltonian)} Pauli terms, offset={offset:.2f}")

    return hamiltonian, offset, info


# ============================================================
# Usage
# ============================================================

if __name__ == "__main__":
    df, clusters = load_data("datasets/small_12/claims.csv",
                             "datasets/small_12/groupes_points.json")
    n, K = len(df), len(clusters)

    # With scaling
    scale = math.ceil(20000 / (2**5 - 1))  # = 646
    C_s = np.round(df["C_i"].values / scale).astype(int)
    M_s = np.round(df["M_i"].values / scale).astype(int)
    B_s = int(round(20000 / scale))

    R = compute_R(df["P_i"].values, df["v_i"].values, M_s, C_s, 0.5)
    B_lin = compute_B_lin(clusters, C_s, 0.3, 3.0)
    Q, info = build_qubo(n, K, clusters, R, B_lin, C_s, B_s, 3.5, 4.0)
    print(f"Q: {Q.shape}, symmetric={np.allclose(Q, Q.T)}, "
          f"dim={info['dim']}, slack_bits={info['M_bits']}")
    print(f"Q range: [{Q.min():.1f}, {Q.max():.1f}]")
