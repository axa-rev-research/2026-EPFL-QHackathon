import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from quboforpennylane import build_qubo_from_files, load_data, compute_R, compute_B_lin
import time
import json


def qubo_to_ising(Q):
    """QUBO -> Ising (h_i, J_ij, offset) avec x_i = (1 - Z_i)/2."""
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0
    for i in range(n):
        h[i] = -Q[i, i] / 2
        offset += Q[i, i] / 2
    for i in range(n):
        for j in range(i + 1, n):
            coeff = Q[i, j] + Q[j, i]
            if coeff != 0:
                J[i, j] = coeff / 4
                h[i] -= coeff / 4
                h[j] -= coeff / 4
                offset += coeff / 4
    return h, J, offset


def substitute_fixed(Q, fixed_vars):
    """Substitute all fixed variables into QUBO, return reduced Q and active indices."""
    n = Q.shape[0]
    active = sorted([i for i in range(n) if i not in fixed_vars])
    Q_red = Q[np.ix_(active, active)].copy()
    for j_new, j_active in enumerate(active):
        for fix_idx, fix_val in fixed_vars.items():
            Q_red[j_new, j_new] += (Q[fix_idx, j_active] + Q[j_active, fix_idx]) * fix_val
    return Q_red, active


def greedy_solve(df, clusters_dict, B_budget, scale, p=0.5):
    """Glouton naif : trier par R_i decroissant, prendre tant que budget le permet."""
    C_s = np.round(df["C_i"].values / scale).astype(int)
    M_s = np.round(df["M_i"].values / scale).astype(int)
    R = compute_R(df["P_i"].values, df["v_i"].values, M_s, C_s, p=p)
    B_s = int(round(B_budget / scale))
    order = np.argsort(-R)
    selected = []
    cost = 0
    for i in order:
        if cost + C_s[i] <= B_s:
            selected.append(i)
            cost += C_s[i]
    sel_ids = [df["claim_id"].values[i] for i in selected]
    R_total = sum(R[i] for i in selected)
    cost_real = sum(df["C_i"].values[i] for i in selected)
    print(f"\n=== Resultat Greedy ===")
    print(f"Claims selectionnes : {sel_ids}")
    print(f"Cout reel           : {cost_real} / {B_budget}")
    print(f"Sum R_i             : {R_total:.2f}")
    return {"selected": selected, "selected_ids": sel_ids,
            "cost_real": cost_real, "R_total": R_total, "R_all": R}


def brute_force_solve(df, clusters_json_path, B_budget, scale, p=0.5, alpha=0.3, k=3.0):
    """Enumere les 2^n combinaisons de x_i, optimise y_c analytiquement."""
    C_s = np.round(df["C_i"].values / scale).astype(int)
    M_s = np.round(df["M_i"].values / scale).astype(int)
    R = compute_R(df["P_i"].values, df["v_i"].values, M_s, C_s, p=p)
    B_s = int(round(B_budget / scale))
    n = len(df)
    with open(clusters_json_path) as f:
        raw_clusters = json.load(f)
    clusters_idx = [[int(s[1:])-1 for s in raw_clusters[str(c)]] for c in range(len(raw_clusters))]
    B_lin = compute_B_lin(clusters_idx, C_s, alpha, k)
    best_obj = -float("inf")
    best_x = None
    n_feasible = 0
    for mask in range(2**n):
        x = np.array([(mask >> i) & 1 for i in range(n)])
        cost = sum(C_s[i] * x[i] for i in range(n))
        if cost > B_s:
            continue
        n_feasible += 1
        obj = sum(R[i] * x[i] for i in range(n))
        for c, members in enumerate(clusters_idx):
            if all(x[m] == 1 for m in members):
                obj += B_lin[c]
        if obj > best_obj:
            best_obj = obj
            best_x = x.copy()
    sel_idx = [i for i in range(n) if best_x[i] == 1]
    sel_ids = [df["claim_id"].values[i] for i in sel_idx]
    cost_real = sum(df["C_i"].values[i] for i in sel_idx)
    cost_scaled = sum(C_s[i] for i in sel_idx)
    r_total = sum(R[i] for i in sel_idx)
    bonus = 0.0
    active_cl = []
    for c, members in enumerate(clusters_idx):
        if all(best_x[m] == 1 for m in members):
            bonus += B_lin[c]
            active_cl.append(c)
    print(f"\n=== Resultat Brute-Force (optimal) ===")
    print(f"Solutions feasibles : {n_feasible} / {2**n}")
    print(f"Claims selectionnes : {sel_ids}")
    print(f"Cout scaled         : {cost_scaled} / {B_s}")
    print(f"Cout reel           : {cost_real} / {B_budget}")
    print(f"Sum R_i             : {r_total:.2f}")
    print(f"Bonus clusters      : {bonus:.2f}  (clusters {active_cl})")
    print(f"Objectif total      : {best_obj:.2f}")
    return {"selected_idx": sel_idx, "selected_ids": sel_ids,
            "cost_real": cost_real, "cost_scaled": cost_scaled,
            "R_total": r_total, "bonus": bonus, "obj_total": best_obj,
            "active_clusters": active_cl, "R_all": R}


def rqaoa_solve(csv_path, clusters_path, B_budget,
                lambda1=3.5, lambda2_start=0.5, lambda2_end=4.0,
                max_slack_bits=5,
                p=8, stepsize=0.1, steps_per_round=100,
                n_starts=5, min_qubits=10, gpu=True):
    """RQAOA with lambda scheduling + multi-start per round."""

    # Initial build to get dimensions
    Q0, info = build_qubo_from_files(csv_path, clusters_path, B_budget=B_budget,
                                      lambda1=lambda1, lambda2=lambda2_start,
                                      max_slack_bits=max_slack_bits)
    n_total = info["dim"]
    n_rounds = n_total - min_qubits
    lambda_schedule = np.linspace(lambda2_start, lambda2_end, n_rounds)

    print(f"\n{'='*60}")
    print(f"RQAOA: lambda scheduling + multi-start")
    print(f"  {n_total} -> {min_qubits} qubits, {n_rounds} rounds")
    print(f"  p={p}, {n_starts} starts x {steps_per_round} steps/round, lr={stepsize}")
    print(f"  lambda2: {lambda2_start} -> {lambda2_end}")
    print(f"  schedule: {['%.2f' % l for l in lambda_schedule]}")
    print(f"{'='*60}")

    fixed = {}
    fix_order = []
    t0 = time.time()
    all_histories = []

    # Build warm-start bias: y_i -> 0, z_k -> 1, x_i -> 0.5 (no bias)
    n_claims = info["n"]
    K_cl = info["K"]
    M_sl = info["M_bits"]
    warm_bias = {}  # qubit_orig_index -> target value (0 or 1)
    for c in range(K_cl):
        warm_bias[n_claims + c] = 0  # y_i = 0
    for k in range(M_sl):
        warm_bias[n_claims + K_cl + k] = 1  # z_k = 1
    print(f"  [Warm init] y_i -> 0, z_k -> 1")

    for round_num in range(n_rounds):
        current_lambda2 = lambda_schedule[round_num]

        # Rebuild QUBO with current lambda2
        Q_full, info = build_qubo_from_files(csv_path, clusters_path, B_budget=B_budget,
                                              lambda1=lambda1, lambda2=current_lambda2,
                                              max_slack_bits=max_slack_bits)

        # Substitute all previously fixed variables
        Q_reduced, active = substitute_fixed(Q_full, fixed)
        n_cur = len(active)

        print(f"\n--- RQAOA Round {round_num+1}/{n_rounds}: {n_cur} qubits, lambda2={current_lambda2:.2f} ---")

        # Ising conversion
        h, J, offset = qubo_to_ising(Q_reduced)
        zz_terms = [(i, j, J[i, j]) for i in range(n_cur)
                     for j in range(i + 1, n_cur) if abs(J[i, j]) > 1e-12]
        z_terms = [(i, h[i]) for i in range(n_cur) if abs(h[i]) > 1e-12]

        coeffs, obs = [], []
        for i, hi in z_terms:
            coeffs.append(hi)
            obs.append(qml.Z(i))
        for i, j, Jij in zz_terms:
            coeffs.append(Jij)
            obs.append(qml.Z(i) @ qml.Z(j))
        H_round = qml.Hamiltonian(coeffs, obs)

        if gpu:
            try:
                dev = qml.device("lightning.gpu", wires=n_cur)
            except Exception:
                dev = qml.device("lightning.qubit", wires=n_cur)
        else:
            dev = qml.device("lightning.qubit", wires=n_cur)

        # Build warm-start init for active qubits
        warm_epsilon = 0.5
        active_warm = {}  # local_idx -> target (0 or 1)
        for local_idx, orig_idx in enumerate(active):
            if orig_idx in warm_bias and orig_idx not in fixed:
                active_warm[local_idx] = warm_bias[orig_idx]

        @qml.qnode(dev, diff_method="best")
        def qaoa_cost(params):
            gammas = params[:p]
            betas = params[p:]
            for w in range(n_cur):
                if w in active_warm:
                    if active_warm[w] == 1:
                        qml.RY(np.pi - warm_epsilon, wires=w)
                    else:
                        qml.RY(warm_epsilon, wires=w)
                else:
                    qml.Hadamard(wires=w)
            for layer in range(p):
                for ii, jj, Jij in zz_terms:
                    qml.IsingZZ(2 * gammas[layer] * Jij, wires=[ii, jj])
                for ii, hi in z_terms:
                    qml.RZ(2 * gammas[layer] * hi, wires=ii)
                for w in range(n_cur):
                    qml.RX(2 * betas[layer], wires=w)
            return qml.expval(H_round)

        @qml.qnode(dev)
        def probs_fn(params):
            gammas = params[:p]
            betas = params[p:]
            for w in range(n_cur):
                if w in active_warm:
                    if active_warm[w] == 1:
                        qml.RY(np.pi - warm_epsilon, wires=w)
                    else:
                        qml.RY(warm_epsilon, wires=w)
                else:
                    qml.Hadamard(wires=w)
            for layer in range(p):
                for ii, jj, Jij in zz_terms:
                    qml.IsingZZ(2 * gammas[layer] * Jij, wires=[ii, jj])
                for ii, hi in z_terms:
                    qml.RZ(2 * gammas[layer] * hi, wires=ii)
                for w in range(n_cur):
                    qml.RX(2 * betas[layer], wires=w)
            return qml.probs(wires=range(n_cur))

        # Multi-start: run n_starts optimizations, keep the best
        global_best_cost = float("inf")
        global_best_params = None

        for start in range(n_starts):
            run_seed = 42 + round_num * 137 + start * 31
            np.random.seed(run_seed)
            params = pnp.array(np.random.uniform(0, np.pi, 2 * p), requires_grad=True)
            opt = qml.AdamOptimizer(stepsize=stepsize)
            best_cost = float("inf")
            best_params = params.copy()

            for step in range(steps_per_round):
                params, cost_val = opt.step_and_cost(qaoa_cost, params)
                total = float(cost_val) + offset
                if total < best_cost:
                    best_cost = total
                    best_params = params.copy()

            print(f"  start {start+1}/{n_starts}: best L={best_cost:.2f}")
            if best_cost < global_best_cost:
                global_best_cost = best_cost
                global_best_params = best_params.copy()

        print(f"  >> Best across {n_starts} starts: L={global_best_cost:.2f}")

        # Get expectations <x_i> from best params
        probs = probs_fn(global_best_params)
        states = np.arange(2**n_cur)
        x_exp = np.zeros(n_cur)
        for i in range(n_cur):
            bit_mask = 1 << (n_cur - 1 - i)
            x_exp[i] = probs[(states & bit_mask) > 0].sum()

        # Find most biased variable
        bias = np.abs(x_exp - 0.5)
        fix_local = int(np.argmax(bias))
        fix_val = int(round(x_exp[fix_local]))
        fix_orig = active[fix_local]

        n_claims, K_cl, M_sl = info["n"], info["K"], info["M_bits"]
        if fix_orig < n_claims:
            var_name = f"x_{info['claim_ids'][fix_orig]}"
        elif fix_orig < n_claims + K_cl:
            var_name = f"y_{fix_orig - n_claims}"
        else:
            var_name = f"z_{fix_orig - n_claims - K_cl}"

        exp_str = " ".join([f"{e:.2f}" for e in x_exp])
        print(f"  <x_i>: [{exp_str}]")
        print(f"  >> Fix {var_name} (orig={fix_orig}) = {fix_val}  (<x>={x_exp[fix_local]:.3f}, bias={bias[fix_local]:.3f})")

        fixed[fix_orig] = fix_val
        fix_order.append((fix_orig, var_name, fix_val, x_exp[fix_local], current_lambda2))

    # Brute-force remaining variables (use final lambda2 QUBO)
    Q_final, info = build_qubo_from_files(csv_path, clusters_path, B_budget=B_budget,
                                           lambda1=lambda1, lambda2=lambda2_end,
                                           max_slack_bits=max_slack_bits)
    Q_bf, active_bf = substitute_fixed(Q_final, fixed)
    n_remain = len(active_bf)
    print(f"\n--- RQAOA: Brute-force {n_remain} qubits restants ({2**n_remain} combinaisons) ---")
    print(f"  (lambda2={lambda2_end:.2f} pour le brute-force final)")

    best_obj = float("inf")
    best_x = None
    for mask in range(2**n_remain):
        x = np.array([(mask >> i) & 1 for i in range(n_remain)], dtype=float)
        obj = float(x @ Q_bf @ x)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()

    # Reconstruct full bitstring
    bitstring = np.zeros(n_total, dtype=int)
    for orig_idx, val in fixed.items():
        bitstring[orig_idx] = val
    for local_idx, orig_idx in enumerate(active_bf):
        bitstring[orig_idx] = int(best_x[local_idx])

    elapsed = time.time() - t0

    # Extract x, y, z
    x_vals = bitstring[:info["n"]]
    y_vals = bitstring[info["n"]:info["n"] + info["K"]]
    z_vals = bitstring[info["n"] + info["K"]:info["n"] + info["K"] + info["M_bits"]]

    selected = [info["claim_ids"][i] for i in range(info["n"]) if x_vals[i] == 1]
    active_clusters = [c for c in range(info["K"]) if y_vals[c] == 1]
    budget_used = sum(z_vals[k] * 2**k for k in range(info["M_bits"]))

    # QUBO energy (with final lambda2)
    energy = float(bitstring.astype(float) @ Q_final @ bitstring.astype(float))

    print(f"\n=== Resultat RQAOA (lambda scheduling) ===")
    print(f"Rounds              : {n_rounds} (fixed {len(fixed)} vars)")
    print(f"lambda2             : {lambda2_start} -> {lambda2_end}")
    print(f"Claims selectionnes : {selected}")
    print(f"Clusters actifs     : {active_clusters}")
    print(f"Budget slack        : {budget_used}")
    print(f"Energie QUBO        : {energy:.2f}")
    print(f"Temps total         : {elapsed:.1f}s")

    print(f"\nVariables fixees (dans l'ordre) :")
    for orig_idx, var_name, val, exp, lam in fix_order:
        print(f"  {var_name:>10} = {val}  (<x>={exp:.3f}, lambda2={lam:.2f})")

    return {
        "energy": energy, "bitstring": bitstring,
        "x_vals": x_vals, "y_vals": y_vals, "z_vals": z_vals,
        "selected_claims": selected, "active_clusters": active_clusters,
        "time_total": elapsed, "fixed_vars": fixed, "fix_order": fix_order,
        "all_histories": all_histories, "n_rounds": n_rounds,
        "info": info,
    }


if __name__ == "__main__":
    BASE = "/users/eleves-a/2024/max.anglade/QuantumInsuranceResourceAllocations"
    csv_path = f"{BASE}/datasets/small_12/claims.csv"
    clusters_path = f"{BASE}/datasets/small_12/groupes_points.json"
    B_budget = 20000

    # --- Greedy ---
    Q_ref, info_ref = build_qubo_from_files(csv_path, clusters_path, B_budget=B_budget,
                                             lambda1=3.5, lambda2=4, max_slack_bits=5)
    df, _ = load_data(csv_path, clusters_path)
    scale = info_ref.get("cost_scale", 1)
    greedy_result = greedy_solve(df, clusters_path, B_budget, scale)

    # --- RQAOA with lambda scheduling ---
    rqaoa_result = rqaoa_solve(csv_path, clusters_path, B_budget,
                                lambda1=3.5, lambda2_start=0.0, lambda2_end=4.0,
                                max_slack_bits=5,
                                p=15, stepsize=0.1, steps_per_round=100,
                                n_starts=5, min_qubits=10, gpu=True)

    # --- Brute-Force ---
    bf_result = brute_force_solve(df, clusters_path, B_budget, scale)

    # --- Analyse ---
    C_s = np.round(df["C_i"].values / scale).astype(int)
    R = greedy_result["R_all"]

    with open(clusters_path) as f:
        raw_clusters = json.load(f)
    clusters_list = [raw_clusters[str(c)] for c in range(len(raw_clusters))]
    clusters_idx = [[int(s[1:])-1 for s in cl] for cl in clusters_list]
    B_lin = compute_B_lin(clusters_idx, C_s, alpha=0.3, k=3.0)

    rqaoa_sel_idx = [int(s[1:])-1 for s in rqaoa_result["selected_claims"]]
    greedy_sel_idx = [int(s[1:])-1 for s in greedy_result["selected_ids"]]
    rqaoa_cost_real = sum(df["C_i"].values[i] for i in rqaoa_sel_idx)
    rqaoa_R = sum(R[i] for i in rqaoa_sel_idx)
    greedy_R = greedy_result["R_total"]

    def cluster_bonus(sel_idx):
        bonus = 0.0
        for c, members in enumerate(clusters_idx):
            if all(m in sel_idx for m in members):
                bonus += B_lin[c]
        return bonus

    rqaoa_bonus = cluster_bonus(rqaoa_sel_idx)
    greedy_bonus = cluster_bonus(greedy_sel_idx)
    rqaoa_obj = rqaoa_R + rqaoa_bonus
    greedy_obj = greedy_R + greedy_bonus
    bf_obj = bf_result["obj_total"]

    print("\n" + "=" * 70)
    print("COMPARAISON RQAOA vs GREEDY vs BRUTE-FORCE (OPTIMAL)")
    print("=" * 70)

    print(f"\n{'':>20} | {'RQAOA':>12} | {'Greedy':>12} | {'Optimal':>12}")
    print(f"{'-'*20} | {'-'*12} | {'-'*12} | {'-'*12}")
    print(f"{'Claims':>20} | {len(rqaoa_result['selected_claims']):>12} | {len(greedy_result['selected_ids']):>12} | {len(bf_result['selected_ids']):>12}")
    print(f"{'Cout (euros)':>20} | {rqaoa_cost_real:>12} | {greedy_result['cost_real']:>12} | {bf_result['cost_real']:>12}")
    print(f"{'Budget':>20} | {B_budget:>12} | {B_budget:>12} | {B_budget:>12}")
    print(f"{'Respect budget':>20} | {'OUI' if rqaoa_cost_real <= B_budget else 'NON':>12} | {'OUI' if greedy_result['cost_real'] <= B_budget else 'NON':>12} | {'OUI' if bf_result['cost_real'] <= B_budget else 'NON':>12}")
    print(f"{'Sum R_i':>20} | {rqaoa_R:>12.2f} | {greedy_R:>12.2f} | {bf_result['R_total']:>12.2f}")
    print(f"{'Bonus clusters':>20} | {rqaoa_bonus:>12.2f} | {greedy_bonus:>12.2f} | {bf_result['bonus']:>12.2f}")
    print(f"{'Objectif (R+B)':>20} | {rqaoa_obj:>12.2f} | {greedy_obj:>12.2f} | {bf_obj:>12.2f}")
    print(f"{'Temps':>20} | {rqaoa_result['time_total']:>10.1f}s | {'<0.01s':>12} | {'<0.01s':>12}")

    if bf_obj != 0:
        rqaoa_gap = (1 - rqaoa_obj / bf_obj) * 100
        greedy_gap = (1 - greedy_obj / bf_obj) * 100
    else:
        rqaoa_gap = greedy_gap = 0
    print(f"\n--- GAPS vs OPTIMAL ---")
    print(f"  RQAOA  : {rqaoa_obj:.2f} / {bf_obj:.2f} = {rqaoa_obj/bf_obj*100:.1f}%  (gap = {rqaoa_gap:.1f}%)")
    print(f"  Greedy : {greedy_obj:.2f} / {bf_obj:.2f} = {greedy_obj/bf_obj*100:.1f}%  (gap = {greedy_gap:.1f}%)")

    print(f"\nClusters:")
    for c_str, members in raw_clusters.items():
        c = int(c_str)
        rq_in = [m for m in members if m in rqaoa_result["selected_claims"]]
        greedy_in = [m for m in members if m in greedy_result["selected_ids"]]
        bf_in = [m for m in members if m in bf_result["selected_ids"]]
        print(f"  Cluster {c_str} {members}: RQAOA {len(rq_in)}/3  Greedy {len(greedy_in)}/3  Optimal {len(bf_in)}/3  (bonus={B_lin[c]:.2f})")

    print(f"\nDetail R_i par claim (scaled) :")
    for i in range(len(df)):
        cid = df["claim_id"].values[i]
        in_r = "R" if i in rqaoa_sel_idx else " "
        in_g = "G" if i in greedy_sel_idx else " "
        in_b = "*" if i in bf_result["selected_idx"] else " "
        print(f"  [{in_r}{in_g}{in_b}] {cid}: R={R[i]:7.2f}  C_s={C_s[i]}")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Convergence per round + lambda schedule
    ax = axes[0]
    colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(rqaoa_result["all_histories"])))
    offset_step = 0
    lambda_sched = np.linspace(0.5, 4.0, len(rqaoa_result["all_histories"]))
    for r, hist in enumerate(rqaoa_result["all_histories"]):
        steps_x = range(offset_step, offset_step + len(hist))
        ax.plot(steps_x, hist, color=colors[r], linewidth=1.5, alpha=0.8,
                label=f"R{r+1} l2={lambda_sched[r]:.1f}" if r % 3 == 0 else None)
        if r > 0:
            ax.axvline(x=offset_step, color="gray", linestyle=":", alpha=0.3)
        offset_step += len(hist)
    ax.set_xlabel("Step (cumule)", fontsize=11)
    ax.set_ylabel("Energie QUBO", fontsize=11)
    ax.set_title(f"RQAOA lambda scheduling ({rqaoa_result['n_rounds']} rounds)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Comparaison barres
    ax = axes[1]
    categories = ["Claims", "Cout (euros)", "Objectif\n(R+B)"]
    rqaoa_vals = [len(rqaoa_result["selected_claims"]), rqaoa_cost_real, rqaoa_obj]
    greedy_vals = [len(greedy_result["selected_ids"]), greedy_result["cost_real"], greedy_obj]
    bf_vals = [len(bf_result["selected_ids"]), bf_result["cost_real"], bf_obj]

    x_pos = np.arange(len(categories))
    width = 0.25
    ax.bar(x_pos - width, rqaoa_vals, width, label="RQAOA", color="#E91E63", alpha=0.8)
    ax.bar(x_pos, greedy_vals, width, label="Greedy", color="#4CAF50", alpha=0.8)
    ax.bar(x_pos + width, bf_vals, width, label="Optimal", color="#2196F3", alpha=0.8)
    ax.axhline(y=B_budget, color="red", linestyle="--", alpha=0.5, label=f"Budget={B_budget}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_title("RQAOA vs Greedy vs Optimal", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("rqaoa_analysis.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved: rqaoa_analysis.png")
