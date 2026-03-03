import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import time
import os
import json
import math
import pandas as pd

from QUBOforNQS import get_hamiltonian, compute_R, compute_B_lin, load_data

# ============================================================
# Parametres
# ============================================================
DATASET = "medium_35"
B_BUDGET = 20000.0
LAMBDA1 = 10.0
LAMBDA2 = 10.0
MAX_SLACK_BITS = 5
P_PARAM = 0.5
ALPHA = 0.3
K_PARAM = 3.0

lr = 0.01
diag_shift = 0.2
n_chains = 4096
n_samples = 8192
n_iter = 200
n_starts = 3
n_sweeps = 10       # sweeps entre echantillons pour meilleur mixing
chunk_size = 256

# ============================================================
# Greedy solver (baseline)
# ============================================================
def greedy_solve(csv_path, clusters_path, B_budget, cost_scale,
                 p=0.5, alpha=0.3, k=3.0):
    """Greedy: pick claims by R_i/C_i ratio, activate clusters when all members selected."""
    df, clusters = load_data(csv_path, clusters_path)
    n, K = len(df), len(clusters)

    C_s = np.round(df["C_i"].values / cost_scale).astype(int)
    M_s = np.round(df["M_i"].values / cost_scale).astype(int)
    B_s = int(round(B_budget / cost_scale))

    R = compute_R(df["P_i"].values, df["v_i"].values, M_s, C_s, p)
    B_lin = compute_B_lin(clusters, C_s, alpha, k)

    # Ratio R_i / C_i (scaled)
    ratio = np.where(C_s > 0, R / C_s, R)
    order = np.argsort(ratio)[::-1]

    x = np.zeros(n, dtype=int)
    budget_used = 0

    for i in order:
        if R[i] <= 0:
            continue
        if budget_used + C_s[i] <= B_s:
            x[i] = 1
            budget_used += C_s[i]

    # Activate clusters where all members are selected
    y = np.zeros(K, dtype=int)
    for c, members in clusters.items():
        if all(x[j] == 1 for j in members):
            y[c] = 1

    selected = np.where(x == 1)[0]
    total_R = sum(R[i] for i in selected)
    total_B = sum(B_lin[c] for c in range(K) if y[c] == 1)
    total_cost_raw = sum(df["C_i"].values[i] for i in selected)

    return {
        "x": x, "y": y,
        "selected": selected,
        "obj": total_R + total_B,
        "reward_R": total_R,
        "bonus_B": total_B,
        "cost_raw": total_cost_raw,
        "cost_scaled": budget_used,
        "B_s": B_s,
        "claim_ids": [df["claim_id"].iloc[i] for i in selected],
    }


# ============================================================
# 1. Chargement dataset
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
dossier_donnees = os.path.join(base_dir, "datasets", DATASET)
chemin_claims = os.path.join(dossier_donnees, "claims.csv")
chemin_groupes = os.path.join(dossier_donnees, "groupes_points.json")

print(f"Dataset: {DATASET}")
print(f"Budget: {B_BUDGET}")

qubit_op, offset, info = get_hamiltonian(
    chemin_claims, chemin_groupes,
    B_budget=B_BUDGET,
    lambda1=LAMBDA1,
    lambda2=LAMBDA2,
    max_slack_bits=MAX_SLACK_BITS,
    p=P_PARAM,
    alpha=ALPHA,
    k=K_PARAM,
)

cost_scale = info.get("cost_scale", 1)

# ============================================================
# Greedy baseline
# ============================================================
print("\n--- Greedy baseline ---")
greedy = greedy_solve(chemin_claims, chemin_groupes, B_BUDGET, cost_scale,
                      p=P_PARAM, alpha=ALPHA, k=K_PARAM)
print(f"Greedy obj (R+B):  {greedy['obj']:.2f}")
print(f"Greedy R:          {greedy['reward_R']:.2f}")
print(f"Greedy B:          {greedy['bonus_B']:.2f}")
print(f"Greedy cost (raw): {greedy['cost_raw']:.0f} / {B_BUDGET}")
print(f"Greedy claims:     {len(greedy['selected'])} selected")
print(f"Greedy claim_ids:  {greedy['claim_ids']}")

# ============================================================
# 2. Conversion Qiskit SparsePauliOp -> NetKet PauliStrings
# ============================================================
n_qubits = qubit_op.num_qubits
chaines_qiskit = qubit_op.paulis.to_labels()
coefficients = qubit_op.coeffs.real

# Qiskit : qubit 0 a droite, NetKet : qubit 0 a gauche
chaines_netket = [chaine[::-1] for chaine in chaines_qiskit]

# Ajouter l offset comme terme Identity
chaines_netket.append("I" * n_qubits)
coefficients = np.append(coefficients, offset)

print(f"\n[NetKet] {n_qubits} qubits, {len(chaines_netket)} Pauli terms")

# ============================================================
# 3. Systeme quantique
# ============================================================
hi = nk.hilbert.Spin(s=1/2, N=n_qubits)
H_nk = nk.operator.PauliStrings(hi, chaines_netket, coefficients)

print(f"[Model] RBM alpha=2 (real params), {n_qubits} qubits")
print(f"GPU: {jax.devices()}")

# ============================================================
# 4. Multi-start VMC
# ============================================================
print(f"\nMulti-start VMC: {n_starts} runs x {n_iter} iter, lr={lr}, "
      f"diag_shift={diag_shift}, n_samples={n_samples}")

all_samples = []
best_energy_global = float("inf")
best_run = -1
start_time = time.time()

for run_idx in range(n_starts):
    seed = 42 + run_idx * 137
    print(f"\n--- Run {run_idx+1}/{n_starts} (seed={seed}) ---", flush=True)

    model = nk.models.RBM(alpha=2, param_dtype=complex)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains, sweep_size=n_sweeps)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)
    vstate.chunk_size = chunk_size

    print(f"  Init: aleatoire", flush=True)

    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    sr = nk.optimizer.SR(diag_shift=diag_shift, solver=jax.scipy.sparse.linalg.cg)

    gs = nk.VMC(
        hamiltonian=H_nk,
        optimizer=optimizer,
        preconditioner=sr,
        variational_state=vstate,
    )

    log = nk.logging.RuntimeLog()

    def make_callback(run_id):
        def cb(step, logdata, driver):
            if step % 100 == 0 or step == n_iter - 1:
                try:
                    e = driver.energy
                    print(f"  [{run_id+1}] Step {step:4d}: E = {e.mean.real:.2f}, "
                          f"Var = {e.variance.real:.2f}", flush=True)
                except Exception:
                    pass
            return True
        return cb

    gs.run(n_iter=n_iter, out=log, callback=make_callback(run_idx))

    # Energie finale
    final_e = float(vstate.expect(H_nk).mean.real)
    print(f"  [{run_idx+1}] Final energy: {final_e:.2f}", flush=True)

    if final_e < best_energy_global:
        best_energy_global = final_e
        best_run = run_idx

    # Echantillonnage
    run_samples = vstate.sample(n_samples=10000)
    run_spins = np.array(run_samples).reshape(-1, n_qubits)
    all_samples.append(run_spins)

execution_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"Multi-start termine en {execution_time:.1f}s")
print(f"Meilleur run: {best_run+1} (E={best_energy_global:.2f})")

# Combiner tous les echantillons
spins = np.vstack(all_samples)

# Projection { -1, 1 } -> { 0, 1 }
bin_samples = ((1 - spins) // 2).astype(int)

Q = info["Q"]
n_claims = info["n"]
K_cl = info["K"]
M_bits = info["M_bits"]

# Evaluate all unique samples
echantillons_uniques = np.unique(bin_samples, axis=0)
print(f"  {len(echantillons_uniques)} etats uniques sur {len(bin_samples)} echantillons")

# Load data for analysis
df = pd.read_csv(chemin_claims).sort_values("claim_id").reset_index(drop=True)
C_raw = df["C_i"].values
P = df["P_i"].values
v = df["v_i"].values
M_raw = df["M_i"].values
claim_ids = list(df["claim_id"])

C_s = np.round(C_raw / cost_scale).astype(int)
M_s = np.round(M_raw / cost_scale).astype(int)
B_s = int(round(B_BUDGET / cost_scale))
R_scaled = compute_R(P, v, M_s, C_s, P_PARAM)
B_lin = compute_B_lin(info["clusters"], C_s, ALPHA, K_PARAM)

# Find best solution (lowest QUBO energy) AND best feasible solution
meilleur_x = None
energie_min = float("inf")
best_feasible_x = None
best_feasible_obj = -float("inf")

for x in echantillons_uniques:
    E_classique = float(x.T @ Q @ x)
    if E_classique < energie_min:
        energie_min = E_classique
        meilleur_x = x

    # Check feasibility (budget constraint)
    x_claims = x[info["x_indices"]]
    z_slack = x[info["z_indices"]]
    cost_used = sum(C_s[i] * x_claims[i] for i in range(n_claims))
    slack = sum(z_slack[k] * (2 ** k) for k in range(M_bits))

    if cost_used <= B_s:
        # Compute objective
        y_cl = x[info["y_indices"]]
        sel = np.where(x_claims == 1)[0]
        obj_R = sum(R_scaled[i] for i in sel)
        obj_B = sum(B_lin[c] for c in range(K_cl) if y_cl[c] == 1)
        obj = obj_R + obj_B
        if obj > best_feasible_obj:
            best_feasible_obj = obj
            best_feasible_x = x

# ============================================================
# 10. Analyse - meilleur etat (pas forcement faisable)
# ============================================================
x_sol = meilleur_x[info["x_indices"]]
y_sol = meilleur_x[info["y_indices"]]
z_sol = meilleur_x[info["z_indices"]]

selected = np.where(x_sol == 1)[0]
total_reward = sum(R_scaled[i] for i in selected)
total_cost_raw = sum(C_raw[i] for i in selected)
total_cost_scaled = sum(C_s[i] for i in selected)
slack_val = sum(z_sol[k] * (2 ** k) for k in range(M_bits))

B_lin_total = 0
for c_idx in range(K_cl):
    if y_sol[c_idx] == 1:
        cluster_claims = list(info["clusters"][c_idx])
        C_cluster = sum(C_s[j] for j in cluster_claims)
        B_c = C_cluster * ALPHA * np.tanh(len(cluster_claims) / K_PARAM)
        B_lin_total += B_c

total_obj = total_reward + B_lin_total

print(f"\n{'='*60}")
print(f"RESULTATS NQS (Jastrow, multi-start) - Meilleure energie QUBO")
print(f"{'='*60}")
print(f"Energie QUBO:        {energie_min:.4f}")
print(f"Claims selectionnees: {len(selected)}")
print(f"Reward (R, scaled):  {total_reward:.2f}")
print(f"Bonus cluster (B):   {B_lin_total:.2f}")
print(f"Objectif (R+B):      {total_obj:.2f}")
print(f"Cout (raw):          {total_cost_raw:.0f} / {B_BUDGET:.0f}")
print(f"Cout (scaled):       {total_cost_scaled} / {B_s}")
print(f"Budget respecte:     {'OUI' if total_cost_scaled <= B_s else 'NON'}")
print(f"Clusters actifs:     {np.where(y_sol == 1)[0].tolist()}")
print(f"Temps:               {execution_time:.1f}s")

# Analyse - meilleur etat faisable
if best_feasible_x is not None:
    xf = best_feasible_x[info["x_indices"]]
    yf = best_feasible_x[info["y_indices"]]
    zf = best_feasible_x[info["z_indices"]]
    sel_f = np.where(xf == 1)[0]
    cost_f_raw = sum(C_raw[i] for i in sel_f)
    cost_f_scaled = sum(C_s[i] for i in sel_f)
    R_f = sum(R_scaled[i] for i in sel_f)
    B_f = sum(B_lin[c] for c in range(K_cl) if yf[c] == 1)
    obj_f = R_f + B_f

    print(f"\n{'='*60}")
    print(f"RESULTATS NQS (Jastrow, multi-start) - Meilleur etat FAISABLE")
    print(f"{'='*60}")
    print(f"Claims selectionnees: {len(sel_f)}")
    print(f"Reward (R):          {R_f:.2f}")
    print(f"Bonus (B):           {B_f:.2f}")
    print(f"Objectif (R+B):      {obj_f:.2f}")
    print(f"Cout (raw):          {cost_f_raw:.0f} / {B_BUDGET:.0f}")
    print(f"Cout (scaled):       {cost_f_scaled} / {B_s}")
    print(f"Clusters actifs:     {np.where(yf == 1)[0].tolist()}")
else:
    obj_f = 0
    print("\nAucun etat faisable trouve parmi les echantillons!")

# ============================================================
# Comparaison
# ============================================================
print(f"\n{'='*60}")
print(f"COMPARAISON")
print(f"{'='*60}")
print(f"Greedy:                  {greedy['obj']:.2f} (baseline)")
print(f"NQS best energy:         {total_obj:.2f} "
      f"({'budget OK' if total_cost_scaled <= B_s else 'BUDGET DEPASSE'})")
if best_feasible_x is not None:
    pct = obj_f / greedy['obj'] * 100 if greedy['obj'] > 0 else 0
    print(f"NQS best feasible:       {obj_f:.2f} ({pct:.1f}% du greedy)")
print(f"Temps NQS:               {execution_time:.1f}s")

# ============================================================
# 11. Export resultats
# ============================================================
resultats = {
    "dataset": DATASET,
    "budget": B_BUDGET,
    "n_qubits": n_qubits,
    "cost_scale": cost_scale,
    "greedy": {
        "obj": float(greedy["obj"]),
        "R": float(greedy["reward_R"]),
        "B": float(greedy["bonus_B"]),
        "cost_raw": float(greedy["cost_raw"]),
        "n_claims": len(greedy["selected"]),
    },
    "nqs_best_energy": {
        "obj": float(total_obj),
        "R": float(total_reward),
        "B": float(B_lin_total),
        "cost_raw": float(total_cost_raw),
        "budget_ok": bool(total_cost_scaled <= B_s),
        "n_claims": int(len(selected)),
    },
    "nqs_best_feasible": {
        "obj": float(obj_f) if best_feasible_x is not None else None,
    },
    "temps_s": execution_time,
    "n_starts": n_starts,
}

results_dir = os.path.join(base_dir, "results_nqs")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, f"resultats_{DATASET}.json")
with open(results_path, "w") as f:
    json.dump(resultats, f, indent=4)

print(f"\nResultats exportes: {results_path}")
