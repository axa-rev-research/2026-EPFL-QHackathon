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


def run_qaoa(Q, info,
             p=1,
             stepsize=0.05,
             max_steps=150,
             seed=42,
             gpu=True,
             warm_start=None,
             warm_epsilon=0.25,
             n_starts=1):
    n_qubits = info["dim"]
    h, J, offset = qubo_to_ising(Q)

    zz_terms = [(i, j, J[i, j]) for i in range(n_qubits)
                for j in range(i + 1, n_qubits) if abs(J[i, j]) > 1e-12]
    z_terms = [(i, h[i]) for i in range(n_qubits) if abs(h[i]) > 1e-12]

    print(f"[QAOA] {n_qubits} qubits, p={p}, {len(zz_terms)} ZZ + {len(z_terms)} Z termes")
    print(f"[QAOA] {2*p} parametres a optimiser")
    if warm_start is not None:
        print(f"[QAOA] Warm-start depuis solution classique (epsilon={warm_epsilon})")
    print(f"[QAOA] Multi-start: {n_starts} runs x {max_steps} steps")

    coeffs, obs = [], []
    for i, hi in z_terms:
        coeffs.append(hi)
        obs.append(qml.Z(i))
    for i, j, Jij in zz_terms:
        coeffs.append(Jij)
        obs.append(qml.Z(i) @ qml.Z(j))
    H = qml.Hamiltonian(coeffs, obs)

    if gpu:
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits)
            print("[QAOA] lightning.gpu (CUDA)")
        except Exception:
            dev = qml.device("lightning.qubit", wires=n_qubits)
            print("[QAOA] fallback lightning.qubit (CPU)")
    else:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        print("[QAOA] lightning.qubit (CPU)")

    @qml.qnode(dev, diff_method="best")
    def qaoa_cost(params):
        gammas = params[:p]
        betas = params[p:]
        if warm_start is not None:
            for i in range(n_qubits):
                if warm_start[i] == 1:
                    qml.RY(np.pi - warm_epsilon, wires=i)
                else:
                    qml.RY(warm_epsilon, wires=i)
        else:
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
        for layer in range(p):
            for i, j, Jij in zz_terms:
                qml.IsingZZ(2 * gammas[layer] * Jij, wires=[i, j])
            for i, hi in z_terms:
                qml.RZ(2 * gammas[layer] * hi, wires=i)
            for i in range(n_qubits):
                qml.RX(2 * betas[layer], wires=i)
        return qml.expval(H)

    @qml.qnode(dev)
    def probs_fn(params):
        gammas = params[:p]
        betas = params[p:]
        if warm_start is not None:
            for i in range(n_qubits):
                if warm_start[i] == 1:
                    qml.RY(np.pi - warm_epsilon, wires=i)
                else:
                    qml.RY(warm_epsilon, wires=i)
        else:
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
        for layer in range(p):
            for i, j, Jij in zz_terms:
                qml.IsingZZ(2 * gammas[layer] * Jij, wires=[i, j])
            for i, hi in z_terms:
                qml.RZ(2 * gammas[layer] * hi, wires=i)
            for i in range(n_qubits):
                qml.RX(2 * betas[layer], wires=i)
        return qml.probs(wires=range(n_qubits))

    # Multi-start optimization
    global_best_cost = float("inf")
    global_best_params = None
    all_histories = []
    t0_global = time.time()

    for run in range(n_starts):
        run_seed = seed + run * 137  # seeds bien espacees
        np.random.seed(run_seed)
        params = pnp.array(np.random.uniform(0, np.pi, 2 * p), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=stepsize)

        print(f"\n--- Run {run+1}/{n_starts} (seed={run_seed}) ---")
        best_params = params.copy()
        best_cost = float("inf")
        history = []
        t0 = time.time()

        for step in range(max_steps):
            params, cost_val = opt.step_and_cost(qaoa_cost, params)
            total = float(cost_val) + offset
            history.append(total)
            if total < best_cost:
                best_cost = total
                best_params = params.copy()
            if step % 50 == 0 or step == max_steps - 1:
                print(f"  step {step:4d}: L = {total:12.2f}  (best={best_cost:.2f})")

        elapsed_run = time.time() - t0
        print(f"  Run {run+1} done: best L = {best_cost:.2f}  ({elapsed_run:.1f}s)")
        all_histories.append(history)

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_params = best_params.copy()
            best_run = run

    elapsed_total = time.time() - t0_global
    print(f"\n[QAOA] Meilleur run: #{best_run+1}, L = {global_best_cost:.2f}  (total {elapsed_total:.1f}s)")

    # Extraire le meilleur bitstring du meilleur run
    probs = probs_fn(global_best_params)
    best_state = np.argmax(probs)
    bitstring = np.array([int(b) for b in format(best_state, f"0{n_qubits}b")])

    n, K, M = info["n"], info["K"], info["M_bits"]
    x_vals = bitstring[:n]
    y_vals = bitstring[n:n + K]
    z_vals = bitstring[n + K:n + K + M]

    selected = [info["claim_ids"][i] for i in range(n) if x_vals[i] == 1]
    active = [c for c in range(K) if y_vals[c] == 1]
    budget_used = sum(z_vals[k] * 2**k for k in range(M))

    print(f"\n=== Resultat QAOA (meilleur des {n_starts} runs) ===")
    print(f"Claims selectionnes : {selected}")
    print(f"Clusters actifs     : {active}")
    print(f"Budget slack        : {budget_used}")
    print(f"Energie             : {global_best_cost:.2f}")

    # Combiner les histories pour le plot (bout a bout)
    combined_history = []
    for h in all_histories:
        combined_history.extend(h)

    return {
        "params": global_best_params, "energy": global_best_cost, "bitstring": bitstring,
        "x_vals": x_vals, "y_vals": y_vals, "z_vals": z_vals,
        "selected_claims": selected, "active_clusters": active,
        "history": combined_history, "times": [],
        "time_total": elapsed_total, "nfev": max_steps * n_starts,
        "all_histories": all_histories, "n_starts": n_starts,
        "best_run": best_run,
    }


def greedy_solve(df, clusters_dict, B_budget, scale, p=0.5):
    """Glouton naif : trier par R_i decroissant, prendre tant que budget le permet."""
    C_s = np.round(df["C_i"].values / scale).astype(int)
    M_s = np.round(df["M_i"].values / scale).astype(int)
    R = compute_R(df["P_i"].values, df["v_i"].values, M_s, C_s, p=p)
    B_s = int(round(B_budget / scale))

    order = np.argsort(-R)  # meilleur R d'abord
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
        # Cluster bonus: y_c = 1 ssi tous les membres sont selectionnes
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

    # Cluster bonus
    bonus = 0.0
    active = []
    for c, members in enumerate(clusters_idx):
        if all(best_x[m] == 1 for m in members):
            bonus += B_lin[c]
            active.append(c)

    print(f"\n=== Resultat Brute-Force (optimal) ===")
    print(f"Solutions feasibles : {n_feasible} / {2**n}")
    print(f"Claims selectionnes : {sel_ids}")
    print(f"Cout scaled         : {cost_scaled} / {B_s}")
    print(f"Cout reel           : {cost_real} / {B_budget}")
    print(f"Sum R_i             : {r_total:.2f}")
    print(f"Bonus clusters      : {bonus:.2f}  (clusters {active})")
    print(f"Objectif total      : {best_obj:.2f}")

    return {"selected_idx": sel_idx, "selected_ids": sel_ids,
            "cost_real": cost_real, "cost_scaled": cost_scaled,
            "R_total": r_total, "bonus": bonus, "obj_total": best_obj,
            "active_clusters": active, "R_all": R}


def plot_results(df, clusters_json_path, qaoa_result, greedy_result, scale, B_budget):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    with open(clusters_json_path) as f:
        raw_clusters = json.load(f)

    C_s = np.round(df["C_i"].values / scale).astype(int)
    M_s = np.round(df["M_i"].values / scale).astype(int)
    R = greedy_result["R_all"]

    qaoa_sel = set(qaoa_result["selected_claims"])
    greedy_sel = set(greedy_result["selected_ids"])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Plot 1: Réseau ---
    ax = axes[0]
    G = nx.Graph()
    labels_map = {}
    for i, row in df.iterrows():
        cid = row["claim_id"]
        G.add_node(cid)
        labels_map[cid] = f"{cid}\nR={R[i]:.1f}\nP={row['P_i']:.2f}"

    cluster_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    node_cluster = {}
    for c_str, members in raw_clusters.items():
        for m in members:
            node_cluster[m] = int(c_str)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                G.add_edge(members[i], members[j], weight=2)

    pos = nx.spring_layout(G, seed=42, k=2)

    # Couleurs des noeuds
    node_colors = []
    node_edge_colors = []
    for node in G.nodes():
        if node in node_cluster:
            node_colors.append(cluster_colors[node_cluster[node]])
        else:
            node_colors.append("#CCCCCC")

        in_qaoa = node in qaoa_sel
        in_greedy = node in greedy_sel
        if in_qaoa and in_greedy:
            node_edge_colors.append("black")
        elif in_qaoa:
            node_edge_colors.append("blue")
        elif in_greedy:
            node_edge_colors.append("green")
        else:
            node_edge_colors.append("gray")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           edgecolors=node_edge_colors, linewidths=3, node_size=800)
    nx.draw_networkx_labels(G, pos, labels_map, ax=ax, font_size=7)

    legend_handles = [
        mpatches.Patch(facecolor=cluster_colors[0], label="Cluster 0"),
        mpatches.Patch(facecolor=cluster_colors[1], label="Cluster 1"),
        mpatches.Patch(facecolor=cluster_colors[2], label="Cluster 2"),
        mpatches.Patch(facecolor="#CCCCCC", label="Isole"),
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='black',
                   markeredgewidth=3, markersize=10, label="QAOA + Greedy"),
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='blue',
                   markeredgewidth=3, markersize=10, label="QAOA seul"),
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='green',
                   markeredgewidth=3, markersize=10, label="Greedy seul"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7)
    ax.set_title("Reseau de sinistres et selections", fontsize=12)
    ax.axis("off")

    # --- Plot 2: Convergence (multi-start) ---
    ax = axes[1]
    if "all_histories" in qaoa_result and qaoa_result.get("n_starts", 1) > 1:
        colors_run = plt.cm.Blues(np.linspace(0.3, 1.0, len(qaoa_result["all_histories"])))
        for r, hist in enumerate(qaoa_result["all_histories"]):
            lw = 2.5 if r == qaoa_result.get("best_run", 0) else 1.0
            alpha = 1.0 if r == qaoa_result.get("best_run", 0) else 0.5
            ax.plot(hist, color=colors_run[r], linewidth=lw, alpha=alpha,
                    label=f"Run {r+1}" if r == qaoa_result.get("best_run", 0) else None)
    else:
        ax.plot(qaoa_result["history"], color="blue", linewidth=1.5, alpha=0.8)
    ax.axhline(y=greedy_result["R_total"], color="green", linestyle="--",
               linewidth=2, label=f"Greedy: {greedy_result['R_total']:.1f}")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Energie QUBO", fontsize=11)
    ax.set_title(f"Convergence QAOA ({qaoa_result.get('n_starts',1)} runs)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Comparaison barres ---
    ax = axes[2]
    qaoa_cost_real = sum(df["C_i"].values[int(s[1:])-1] for s in qaoa_result["selected_claims"])
    qaoa_R = sum(R[int(s[1:])-1] for s in qaoa_result["selected_claims"])

    categories = ["Claims\nselectionnes", "Cout total\n(euros)", "Sum R_i"]
    qaoa_vals = [len(qaoa_result["selected_claims"]), qaoa_cost_real, qaoa_R]
    greedy_vals = [len(greedy_result["selected_ids"]), greedy_result["cost_real"], greedy_result["R_total"]]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, qaoa_vals, width, label="QAOA", color="blue", alpha=0.7)
    bars2 = ax.bar(x + width/2, greedy_vals, width, label="Greedy", color="green", alpha=0.7)
    ax.axhline(y=B_budget, color="red", linestyle="--", alpha=0.5, label=f"Budget={B_budget}")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_title("QAOA vs Greedy", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotations
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("qaoa_analysis.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved: qaoa_analysis.png")

    # --- Plot temps : best per run ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    if "all_histories" in qaoa_result and qaoa_result.get("n_starts", 1) > 1:
        best_per_run = [min(h) for h in qaoa_result["all_histories"]]
        runs = range(1, len(best_per_run) + 1)
        bars = ax2.bar(runs, best_per_run, color=["#2196F3" if r == qaoa_result.get("best_run", 0) else "#90CAF9"
                                                   for r in range(len(best_per_run))], edgecolor="black")
        ax2.axhline(y=greedy_result["R_total"], color="green", linestyle="--", linewidth=2, label="Greedy")
        for bar, val in zip(bars, best_per_run):
            ax2.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
        ax2.set_xlabel("Run #", fontsize=11)
        ax2.set_ylabel("Meilleure energie QUBO", fontsize=11)
        ax2.set_title(f"Multi-start: {len(best_per_run)} runs  (total: {qaoa_result['time_total']:.1f}s)", fontsize=12)
        ax2.legend(fontsize=10)
    else:
        ax2.plot(qaoa_result["history"], "b.-", alpha=0.7)
        ax2.set_xlabel("Step", fontsize=11)
        ax2.set_ylabel("Energie QUBO", fontsize=11)
        ax2.set_title(f"Convergence  (total: {qaoa_result['time_total']:.1f}s)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("qaoa_timing.png", dpi=150, bbox_inches="tight")
    print("Plot saved: qaoa_timing.png")


if __name__ == "__main__":
    BASE = "/users/eleves-a/2024/max.anglade/QuantumInsuranceResourceAllocations"
    csv_path = f"{BASE}/datasets/small_12/claims.csv"
    clusters_path = f"{BASE}/datasets/small_12/groupes_points.json"
    B_budget = 20000

    # --- Build QUBO ---
    Q, info = build_qubo_from_files(csv_path, clusters_path, B_budget=B_budget,
                                    lambda1=3.5, lambda2=4, max_slack_bits=5)

    # --- Greedy (pour warm-start) ---
    df, _ = load_data(csv_path, clusters_path)
    scale = info.get("cost_scale", 1)
    greedy_result = greedy_solve(df, clusters_path, B_budget, scale)

    # --- Construire le bitstring warm-start ---
    n, K, M = info["n"], info["K"], info["M_bits"]
    C_s_ws = np.round(df["C_i"].values / scale).astype(int)
    B_s = int(round(B_budget / scale))

    # x: claims selectionnes par le greedy
    x_ws = np.zeros(n, dtype=int)
    for i in greedy_result["selected"]:
        x_ws[i] = 1

    # y: clusters complets
    with open(clusters_path) as f:
        raw_cl = json.load(f)
    clusters_idx_ws = [[int(s[1:])-1 for s in raw_cl[str(c)]] for c in range(K)]
    y_ws = np.zeros(K, dtype=int)
    for c, members in enumerate(clusters_idx_ws):
        if all(x_ws[m] == 1 for m in members):
            y_ws[c] = 1

    # z: slack bits = B_s - sum(C_s * x)
    slack_val = B_s - sum(C_s_ws[i] * x_ws[i] for i in range(n))
    slack_val = max(0, slack_val)
    z_ws = np.array([(slack_val >> k) & 1 for k in range(M)], dtype=int)

    warm_bits = np.concatenate([x_ws, y_ws, z_ws])
    print(f"\n[Warm-start] x={x_ws.tolist()}, y={y_ws.tolist()}, z={z_ws.tolist()}")
    print(f"[Warm-start] slack={slack_val}, cost_s={sum(C_s_ws*x_ws)}/{B_s}")

    # --- QAOA warm-start multi-start ---
    qaoa_result = run_qaoa(Q, info, p=8, stepsize=0.05, max_steps=100,
                           gpu=True, warm_start=warm_bits, warm_epsilon=0.5,
                           n_starts=5)

    # --- Brute-Force ---
    bf_result = brute_force_solve(df, clusters_path, B_budget, scale)
    # Optimal connu: obj=48.78

    # --- Analyse ---
    C_s = np.round(df["C_i"].values / scale).astype(int)
    M_s = np.round(df["M_i"].values / scale).astype(int)
    R = greedy_result["R_all"]

    with open(clusters_path) as f:
        raw_clusters = json.load(f)
    clusters_list = [raw_clusters[str(c)] for c in range(len(raw_clusters))]
    clusters_idx = [[int(s[1:])-1 for s in cl] for cl in clusters_list]
    B_lin = compute_B_lin(clusters_idx, C_s, alpha=0.3, k=3.0)

    qaoa_sel_idx = [int(s[1:])-1 for s in qaoa_result["selected_claims"]]
    greedy_sel_idx = [int(s[1:])-1 for s in greedy_result["selected_ids"]]
    qaoa_cost = sum(df["C_i"].values[i] for i in qaoa_sel_idx)
    qaoa_R = sum(R[i] for i in qaoa_sel_idx)
    greedy_R = greedy_result["R_total"]

    def cluster_bonus(sel_idx):
        bonus = 0.0
        for c, members in enumerate(clusters_idx):
            if all(m in sel_idx for m in members):
                bonus += B_lin[c]
        return bonus

    qaoa_bonus = cluster_bonus(qaoa_sel_idx)
    greedy_bonus = cluster_bonus(greedy_sel_idx)
    qaoa_obj = qaoa_R + qaoa_bonus
    greedy_obj = greedy_R + greedy_bonus
    bf_obj = bf_result["obj_total"]

    print("\n" + "=" * 70)
    print("COMPARAISON QAOA vs GREEDY vs BRUTE-FORCE (OPTIMAL)")
    print("=" * 70)

    print(f"\n{'':>20} | {'QAOA':>12} | {'Greedy':>12} | {'Optimal':>12}")
    print(f"{'-'*20} | {'-'*12} | {'-'*12} | {'-'*12}")
    print(f"{'Claims':>20} | {len(qaoa_result['selected_claims']):>12} | {len(greedy_result['selected_ids']):>12} | {len(bf_result['selected_ids']):>12}")
    print(f"{'Cout (euros)':>20} | {qaoa_cost:>12} | {greedy_result['cost_real']:>12} | {bf_result['cost_real']:>12}")
    print(f"{'Budget':>20} | {B_budget:>12} | {B_budget:>12} | {B_budget:>12}")
    print(f"{'Respect budget':>20} | {'OUI' if qaoa_cost <= B_budget else 'NON':>12} | {'OUI' if greedy_result['cost_real'] <= B_budget else 'NON':>12} | {'OUI' if bf_result['cost_real'] <= B_budget else 'NON':>12}")
    print(f"{'Sum R_i':>20} | {qaoa_R:>12.2f} | {greedy_R:>12.2f} | {bf_result['R_total']:>12.2f}")
    print(f"{'Bonus clusters':>20} | {qaoa_bonus:>12.2f} | {greedy_bonus:>12.2f} | {bf_result['bonus']:>12.2f}")
    print(f"{'Objectif (R+B)':>20} | {qaoa_obj:>12.2f} | {greedy_obj:>12.2f} | {bf_obj:>12.2f}")
    print(f"{'Temps':>20} | {qaoa_result['time_total']:>10.1f}s | {'<0.01s':>12} | {'<0.01s':>12}")

    # Gaps
    if bf_obj != 0:
        qaoa_gap = (1 - qaoa_obj / bf_obj) * 100
        greedy_gap = (1 - greedy_obj / bf_obj) * 100
    else:
        qaoa_gap = greedy_gap = 0
    print(f"\n--- GAPS vs OPTIMAL ---")
    print(f"  QAOA   : {qaoa_obj:.2f} / {bf_obj:.2f} = {qaoa_obj/bf_obj*100:.1f}%  (gap = {qaoa_gap:.1f}%)")
    print(f"  Greedy : {greedy_obj:.2f} / {bf_obj:.2f} = {greedy_obj/bf_obj*100:.1f}%  (gap = {greedy_gap:.1f}%)")

    print(f"\nBonus clusters (scaled) : {['%.2f' % b for b in B_lin]}")
    print(f"\nClusters:")
    for c_str, members in raw_clusters.items():
        c = int(c_str)
        qaoa_in = [m for m in members if m in qaoa_result["selected_claims"]]
        greedy_in = [m for m in members if m in greedy_result["selected_ids"]]
        bf_in = [m for m in members if m in bf_result["selected_ids"]]
        print(f"  Cluster {c_str} {members}: QAOA {len(qaoa_in)}/3  Greedy {len(greedy_in)}/3  Optimal {len(bf_in)}/3  (bonus={B_lin[c]:.2f})")

    print(f"\nDetail R_i par claim (scaled) :")
    for i in range(len(df)):
        cid = df["claim_id"].values[i]
        in_q = "Q" if i in qaoa_sel_idx else " "
        in_g = "G" if i in greedy_sel_idx else " "
        in_b = "*" if i in bf_result["selected_idx"] else " "
        print(f"  [{in_q}{in_g}{in_b}] {cid}: R={R[i]:7.2f}  C_s={C_s[i]}")

    # --- Plots ---
    plot_results(df, clusters_path, qaoa_result, greedy_result, scale, B_budget)
    print("\nResultats sauvegardes.")
    print("\nResultats sauvegardes.")
