import numpy as np
import pandas as pd
import json
import math


# ============================================================
# Core formulas
# ============================================================

def compute_R(P, v, M_param, C, p):
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

    # Term 1: +lambda1 * sum_c sum_{j in I_c} (1 - x_j) * y_c
    #       = +lambda1 * sum_c N_c * y_c  -  lambda1 * sum_c sum_j x_j * y_c
    for c in range(K):
        N_c = len(clusters[c])
        Q[y_off + c, y_off + c] += lambda1 * N_c          # linear: +lambda1 * N_c * y_c
        for j in clusters[c]:
            Q[x_off + j, y_off + c] -= lambda1 / 2        # bilinear: -lambda1 * x_j * y_c
            Q[y_off + c, x_off + j] -= lambda1 / 2

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


def build_qubo_from_files(csv_path, clusters_json_path,
                          p=0.5, alpha=0.3, k=3.0, B_budget=20000.0,
                          lambda1=100, lambda2=100, max_slack_bits=8):
    df, clusters = load_data(csv_path, clusters_json_path)
    n, K = len(df), len(clusters)

    # Auto-scale monetary values to keep slack bits <= max_slack_bits
    M_bits_raw = math.floor(math.log2(B_budget)) + 1
    if M_bits_raw > max_slack_bits:
        scale = math.ceil(B_budget / (2**max_slack_bits - 1))
        print(f"[SCALE] Dividing costs by {scale} to fit {max_slack_bits} slack bits")
    else:
        scale = 1

    C_s = np.round(df["C_i"].values / scale).astype(int)
    M_s = np.round(df["M_i"].values / scale).astype(int)
    B_s = int(round(B_budget / scale))

    print(f"[QUBO] n={n}, K={K}, slack={math.floor(math.log2(B_s))+1}, "
          f"total qubits={n + K + math.floor(math.log2(B_s)) + 1}")

    R = compute_R(df["P_i"].values, df["v_i"].values, M_s, C_s, p)
    B_lin = compute_B_lin(clusters, C_s, alpha, k)
    Q, info = build_qubo(n, K, clusters, R, B_lin, C_s, B_s, lambda1, lambda2)

    info.update({"clusters": clusters, "claim_ids": list(df["claim_id"]),
                 "cost_scale": scale})
    return Q, info


# ============================================================
# Q -> PennyLane Hamiltonian (Ising)
# ============================================================

def qubo_to_pennylane(Q, info):
    """
    x_i = (1 - Z_i) / 2  ->  Ising Hamiltonian + offset
    """
    import pennylane as qml

    dim = info["dim"]
    coeffs = []
    obs = []
    offset = 0.0

    # Diagonal: Q[i,i] * x_i = Q[i,i] * (1 - Z_i)/2
    for i in range(dim):
        if Q[i, i] != 0:
            offset += Q[i, i] / 2
            coeffs.append(-Q[i, i] / 2)
            obs.append(qml.Z(i))

    # Off-diagonal: (Q[i,j]+Q[j,i]) * x_i*x_j = coeff*(1 - Z_i - Z_j + Z_iZ_j)/4
    for i in range(dim):
        for j in range(i + 1, dim):
            coeff = Q[i, j] + Q[j, i]
            if coeff != 0:
                offset += coeff / 4
                coeffs.append(-coeff / 4)
                obs.append(qml.Z(i))
                coeffs.append(-coeff / 4)
                obs.append(qml.Z(j))
                coeffs.append(coeff / 4)
                obs.append(qml.Z(i) @ qml.Z(j))

    H = qml.Hamiltonian(coeffs, obs, grouping_type="qwc")
    print(f"[Hamiltonian] {dim} qubits, {len(coeffs)} Pauli terms, offset={offset:.2f}")
    return H, offset, dim


# ============================================================
# Hardware-efficient ansatz
# ============================================================

def make_ansatz(n_qubits, n_layers=1):
    """RY rotations + circular CNOT entanglement."""
    import pennylane as qml

    def ansatz(params):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(params[layer * n_qubits + i], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

    return ansatz, n_layers * n_qubits


# ============================================================
# VQE on GPU
# ============================================================

def run_vqe(Q, info,
            n_layers=1,
            optimizer_name="adam",
            stepsize=0.01,
            max_steps=200,
            print_every=20,
            seed=42,
            draw=True,
            gpu=True):
    """
    Full VQE pipeline on GPU via lightning.gpu (cuQuantum).

    Install:
        pip install pennylane pennylane-lightning[gpu]
        # requires CUDA toolkit + cuQuantum

    Falls back to lightning.qubit (CPU) if GPU unavailable.

    Parameters
    ----------
    Q, info        : from build_qubo
    n_layers       : int    ansatz depth
    optimizer_name : str    "adam", "gd", "nesterov", "spsa"
    stepsize       : float
    max_steps      : int
    print_every    : int
    seed           : int
    draw           : bool   draw circuit before optimizing
    gpu            : bool   use lightning.gpu (True) or lightning.qubit (False)
    """
    import pennylane as qml

    np.random.seed(seed)

    # 1. Hamiltonian
    H, offset, n_qubits = qubo_to_pennylane(Q, info)

    # 2. Ansatz
    ansatz, n_params = make_ansatz(n_qubits, n_layers)
    print(f"[VQE] {n_qubits} qubits, {n_layers} layers, {n_params} params")

    # 3. Device: GPU or CPU
    if gpu:
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits)
            print("[VQE] Using lightning.gpu (CUDA/cuQuantum)")
        except Exception:
            dev = qml.device("lightning.qubit", wires=n_qubits)
            print("[VQE] GPU unavailable, falling back to lightning.qubit (CPU)")
    else:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        print("[VQE] Using lightning.qubit (CPU)")

    # 4. Cost function with adjoint diff (fastest for lightning)
    @qml.qnode(dev, diff_method="adjoint")
    def cost_fn(params):
        ansatz(params)
        return qml.expval(H)

    # 5. Draw circuit
    if draw:
        init_params = np.zeros(n_params)
        print("\n=== Circuit (first 15 wires) ===")
        print(qml.draw(cost_fn, max_length=120, show_all_wires=False)(init_params))
        print("================================\n")
        try:
            fig, ax = qml.draw_mpl(cost_fn, show_all_wires=False)(init_params)
            fig.savefig("vqe_circuit.png", dpi=150, bbox_inches="tight")
            print("[VQE] Circuit saved to vqe_circuit.png")
        except Exception:
            pass

    # 6. Optimizer
    opt_map = {
        "adam": qml.AdamOptimizer(stepsize=stepsize),
        "gd": qml.GradientDescentOptimizer(stepsize=stepsize),
        "nesterov": qml.NesterovMomentumOptimizer(stepsize=stepsize),
        "spsa": qml.SPSAOptimizer(maxiter=max_steps),
    }
    opt = opt_map.get(optimizer_name, qml.AdamOptimizer(stepsize=stepsize))

    # 7. Optimization loop
    import pennylane.numpy as pnp; params = pnp.array(np.random.randn(n_params) * 0.1, requires_grad=True)
    history = []

    print(f"[VQE] Optimizing with {optimizer_name}...")
    for step in range(max_steps):
        params, energy = opt.step_and_cost(cost_fn, params)
        total_energy = float(energy) + offset
        history.append(total_energy)
        if step % print_every == 0:
            print(f"  Step {step:4d}: L = {total_energy:.4f}")

    final_energy = float(cost_fn(params)) + offset
    print(f"  Final:      L = {final_energy:.4f}")

    # 8. Extract best bitstring
    @qml.qnode(dev)
    def probs_fn(params):
        ansatz(params)
        return qml.probs(wires=range(n_qubits))

    probs = probs_fn(params)
    best_state = np.argmax(probs)
    bitstring = np.array([int(b) for b in format(best_state, f"0{n_qubits}b")])

    n, K, M = info["n"], info["K"], info["M_bits"]
    x_vals = bitstring[:n]
    y_vals = bitstring[n:n + K]
    z_vals = bitstring[n + K:n + K + M]

    selected = [info["claim_ids"][i] for i in range(n) if x_vals[i] == 1] if "claim_ids" in info else []
    active = [c for c in range(K) if y_vals[c] == 1]

    return {
        "params": params,
        "energy": final_energy,
        "bitstring": bitstring,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "z_vals": z_vals,
        "selected_claims": selected,
        "active_clusters": active,
        "history": history,
    }





  