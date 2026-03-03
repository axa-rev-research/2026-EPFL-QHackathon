"""
NQS Constrained - VERSION SANS SCALING (couts bruts)
Plus de slack bits, plus de cost_scale. Tout en valeurs reelles.
"""
import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import flax.linen as nn
import time
import os
import json
import math
import pandas as pd

from quadraticproblem import compute_R, compute_B_lin, load_data

# ============================================================
# Parametres
# ============================================================
DATASET = "medium_35_clusters"
B_BUDGET = 82000.0
LAMBDA1 = 10.0
P_PARAM = 0.5
ALPHA = 1.0
K_PARAM = 3.0

lr = 0.01
diag_shift = 0.5
n_chains = 4096
n_samples = 8192
n_iter = 300
n_starts = 1
chunk_size = 256

# ============================================================
# Greedy solver (baseline) - COUTS BRUTS
# ============================================================
def greedy_solve(csv_path, clusters_path, B_budget,
                 p=0.5, alpha=0.3, k=3.0):
    df, clusters = load_data(csv_path, clusters_path)
    n, K = len(df), len(clusters)
    C = df["C_i"].values
    M = df["M_i"].values
    R = compute_R(df["P_i"].values, df["v_i"].values, M, C, p)
    B_lin = compute_B_lin(clusters, C, alpha, k)
    ratio = np.where(C > 0, R / C, R)
    order = np.argsort(ratio)[::-1]
    x = np.zeros(n, dtype=int)
    budget_used = 0
    for i in order:
        if R[i] <= 0:
            continue
        if budget_used + C[i] <= B_budget:
            x[i] = 1
            budget_used += C[i]
    y = np.zeros(K, dtype=int)
    for c, members in clusters.items():
        if all(x[j] == 1 for j in members):
            y[c] = 1
    selected = np.where(x == 1)[0]
    total_R = sum(R[i] for i in selected)
    total_B = sum(B_lin[c] for c in range(K) if y[c] == 1)
    return {
        "obj": total_R + total_B, "reward_R": total_R, "bonus_B": total_B,
        "cost": budget_used, "B_budget": B_budget,
        "selected": selected, "x": x, "y": y,
        "claim_ids": [df["claim_id"].iloc[i] for i in selected],
    }

# ============================================================
# 1. Chargement - COUTS BRUTS, pas de scaling
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
dossier_donnees = os.path.join(base_dir, "datasets", DATASET)
chemin_claims = os.path.join(dossier_donnees, "claims.csv")
chemin_groupes = os.path.join(dossier_donnees, "groupes_points.json")

df, clusters = load_data(chemin_claims, chemin_groupes)
n_claims, K_cl = len(df), len(clusters)

# Couts BRUTS - pas de scaling!
C = df["C_i"].values.astype(float)
M = df["M_i"].values.astype(float)

R = compute_R(df["P_i"].values, df["v_i"].values, M, C, P_PARAM)
B_lin = compute_B_lin(clusters, C, ALPHA, K_PARAM)

n_qubits = n_claims + K_cl
dim = n_qubits

print(f"Dataset: {DATASET}")
print(f"Budget: {B_BUDGET} (BRUT, pas de scaling)")
print(f"Qubits: {n_claims} claims + {K_cl} clusters = {n_qubits}")
print(f"C range: [{C.min():.0f}, {C.max():.0f}], total={C.sum():.0f}")
print(f"R range: [{R.min():.1f}, {R.max():.1f}]")
print(f"B_lin range: [{B_lin.min():.1f}, {B_lin.max():.1f}]")

# ============================================================
# 2. Construction QUBO en couts bruts
# ============================================================
Q = np.zeros((dim, dim))
x_off, y_off = 0, n_claims

# Objectif: -R_i * x_i - B_lin_c * y_c
for i in range(n_claims):
    Q[x_off + i, x_off + i] -= R[i]
for c in range(K_cl):
    Q[y_off + c, y_off + c] -= B_lin[c]

# Cluster consistency: lambda1 * sum_c sum_{j in I_c} (1 - x_j) * y_c
for c in range(K_cl):
    N_c = len(clusters[c])
    Q[y_off + c, y_off + c] += LAMBDA1 * N_c
    for j in clusters[c]:
        Q[x_off + j, y_off + c] -= LAMBDA1 / 2
        Q[y_off + c, x_off + j] -= LAMBDA1 / 2

# Penalite budget LEGERE (meme approche que la version scalee)
# Avec le scaling (scale=2646), lambda2=2.0 et C_s~2, B_s=31 marchait bien.
# Equivalence: lambda2_raw = lambda2_scaled / scale^2 = 2.0 / 2646^2 = 2.86e-7
# On prend un peu plus fort pour compenser l'absence de slack bits
LAMBDA2_SOFT = 1e-3
print(f"\nPenalite budget legere: lambda2={LAMBDA2_SOFT:.1e}")

Q_budget = np.zeros((dim, dim))
for i in range(n_claims):
    Q_budget[x_off + i, x_off + i] += LAMBDA2_SOFT * (C[i] ** 2 - 2 * B_BUDGET * C[i])
    for j in range(i + 1, n_claims):
        val = LAMBDA2_SOFT * C[i] * C[j]
        Q_budget[x_off + i, x_off + j] += val
        Q_budget[x_off + j, x_off + i] += val

Q_total = Q + Q_budget

# Verif echelle des coefficients
diag_vals = np.diag(Q_total)
print(f"Q diag range: [{diag_vals.min():.1f}, {diag_vals.max():.1f}]")

# ============================================================
# 3. Conversion QUBO -> Ising
# ============================================================
h = np.zeros(dim)
J = np.zeros((dim, dim))
offset = 0.0

for i in range(dim):
    offset += Q_total[i, i] / 4.0
    h[i] -= Q_total[i, i] / 4.0
    for j in range(i + 1, dim):
        coeff = (Q_total[i, j] + Q_total[j, i]) / 2.0
        if coeff != 0:
            offset += coeff / 4.0
            h[i] -= coeff / 4.0
            h[j] -= coeff / 4.0
            J[i, j] += coeff / 4.0

pauli_strings = []
pauli_coeffs = []

for i in range(dim):
    if h[i] != 0:
        s = ["I"] * dim
        s[i] = "Z"
        pauli_strings.append("".join(s))
        pauli_coeffs.append(h[i])

for i in range(dim):
    for j in range(i + 1, dim):
        if J[i, j] != 0:
            s = ["I"] * dim
            s[i] = "Z"
            s[j] = "Z"
            pauli_strings.append("".join(s))
            pauli_coeffs.append(J[i, j])

pauli_strings.append("I" * dim)
pauli_coeffs.append(offset)

print(f"[Hamiltonian] {len(pauli_strings)} Pauli terms, offset={offset:.2f}")

# ============================================================
# Greedy baseline (couts bruts)
# ============================================================
greedy = greedy_solve(chemin_claims, chemin_groupes, B_BUDGET,
                      p=P_PARAM, alpha=ALPHA, k=K_PARAM)
print(f"\n--- Greedy baseline (couts bruts) ---")
print(f"Greedy obj:    {greedy['obj']:.2f}")
print(f"Greedy R:      {greedy['reward_R']:.2f}")
print(f"Greedy B:      {greedy['bonus_B']:.2f}")
print(f"Greedy cout:   {greedy['cost']:.0f} / {B_BUDGET:.0f}")
print(f"Greedy claims: {len(greedy['selected'])} -> {greedy['claim_ids']}")

# ============================================================
# 4. Multi-start VMC
# ============================================================
hi = nk.hilbert.Spin(s=1/2, N=n_qubits)
H_nk = nk.operator.PauliStrings(hi, pauli_strings, pauli_coeffs)

print(f"\nMulti-start VMC: {n_starts} runs x {n_iter} iter, lr={lr}")
print(f"GPU: {jax.devices()}")

all_samples = []
best_energy_global = float("inf")
best_run = -1
start_time = time.time()

for run_idx in range(n_starts):
    seed = 42 + run_idx * 137
    print(f"\n--- Run {run_idx+1}/{n_starts} (seed={seed}) ---", flush=True)

    model = nk.models.RBM(alpha=2, param_dtype=complex)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains, sweep_size=10)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)
    vstate.chunk_size = chunk_size

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
            if step % 50 == 0 or step == n_iter - 1:
                try:
                    e = driver.energy
                    print(f"  [{run_id+1}] Step {step:4d}: E = {e.mean.real:.2f}, "
                          f"Var = {e.variance.real:.2f}", flush=True)
                except Exception:
                    pass
            return True
        return cb

    gs.run(n_iter=n_iter, out=log, callback=make_callback(run_idx))

    final_e = float(vstate.expect(H_nk).mean.real)
    print(f"  [{run_idx+1}] Final energy: {final_e:.2f}", flush=True)

    if final_e < best_energy_global:
        best_energy_global = final_e
        best_run = run_idx

    run_samples = vstate.sample(n_samples=20000)
    run_spins = np.array(run_samples).reshape(-1, n_qubits)
    all_samples.append(run_spins)

execution_time = time.time() - start_time
print(f"\nMulti-start termine en {execution_time:.1f}s")
print(f"Meilleur run: {best_run+1} (E={best_energy_global:.2f})")

# ============================================================
# 5. Analyse: filtrer les etats faisables (BUDGET BRUT)
# ============================================================
spins = np.vstack(all_samples)
bin_samples = ((1 - spins) // 2).astype(int)

echantillons_uniques = np.unique(bin_samples, axis=0)
print(f"\n{len(echantillons_uniques)} etats uniques sur {len(bin_samples)} echantillons")

claim_ids = list(df["claim_id"])

best_obj = -float("inf")
best_x = None
best_feasible_obj = -float("inf")
best_feasible_x = None
n_feasible = 0

for x_full in echantillons_uniques:
    x_claims = x_full[:n_claims]
    y_cl = x_full[n_claims:]

    # Budget check en couts BRUTS
    cost_raw = sum(C[i] * x_claims[i] for i in range(n_claims))
    sel = np.where(x_claims == 1)[0]
    obj_R = sum(R[i] for i in sel)
    obj_B = sum(B_lin[c] for c in range(K_cl) if y_cl[c] == 1)
    obj = obj_R + obj_B

    if obj > best_obj:
        best_obj = obj
        best_x = x_full

    if cost_raw <= B_BUDGET:
        n_feasible += 1
        if obj > best_feasible_obj:
            best_feasible_obj = obj
            best_feasible_x = x_full

print(f"Etats faisables (budget brut): {n_feasible}/{len(echantillons_uniques)}")

# ============================================================
# 6. Resultats
# ============================================================
print(f"\n{'='*60}")
print(f"RESULTATS NQS (couts bruts, sans scaling)")
print(f"{'='*60}")

if best_x is not None:
    x_sol = best_x[:n_claims]
    y_sol = best_x[n_claims:]
    sel = np.where(x_sol == 1)[0]
    cost_best = sum(C[i] for i in sel)
    print(f"Meilleur etat (tout):")
    print(f"  Objectif: {best_obj:.2f}, Claims: {len(sel)}, "
          f"Cout: {cost_best:.0f}/{B_BUDGET:.0f}, "
          f"Budget OK: {cost_best <= B_BUDGET}")

if best_feasible_x is not None:
    xf = best_feasible_x[:n_claims]
    yf = best_feasible_x[n_claims:]
    sel_f = np.where(xf == 1)[0]
    cost_f = sum(C[i] for i in sel_f)
    R_f = sum(R[i] for i in sel_f)
    B_f = sum(B_lin[c] for c in range(K_cl) if yf[c] == 1)

    print(f"\nMeilleur etat FAISABLE:")
    print(f"  Claims: {[claim_ids[i] for i in sel_f]}")
    print(f"  Reward (R): {R_f:.2f}")
    print(f"  Bonus (B):  {B_f:.2f}")
    print(f"  Objectif:   {best_feasible_obj:.2f}")
    print(f"  Cout:       {cost_f:.0f} / {B_BUDGET:.0f}")
    print(f"  Clusters:   {np.where(yf == 1)[0].tolist()}")
else:
    best_feasible_obj = 0
    print("\nAucun etat faisable!")

print(f"\n{'='*60}")
print(f"COMPARAISON (couts bruts)")
print(f"{'='*60}")
print(f"Greedy:    {greedy['obj']:.2f}  (cout {greedy['cost']:.0f})")
if best_feasible_x is not None:
    pct = best_feasible_obj / greedy['obj'] * 100 if greedy['obj'] > 0 else 0
    print(f"NQS:       {best_feasible_obj:.2f}  (cout {cost_f:.0f})  "
          f"= {pct:.1f}% du greedy")
    if best_feasible_obj > greedy['obj']:
        print(f">>> NQS BAT GREEDY de {best_feasible_obj - greedy['obj']:.2f} "
              f"(+{(best_feasible_obj/greedy['obj']-1)*100:.1f}%)")
print(f"Temps: {execution_time:.1f}s")

# ============================================================
# 7. Export JSON
# ============================================================
resultats = {
    "dataset": DATASET, "budget": B_BUDGET, "alpha": ALPHA,
    "n_qubits": n_qubits, "scaling": "NONE (raw costs)",
    "greedy": {
        "obj": float(greedy["obj"]),
        "R": float(greedy["reward_R"]),
        "B": float(greedy["bonus_B"]),
        "cost": float(greedy["cost"]),
        "n_claims": len(greedy["selected"]),
        "claim_ids": greedy["claim_ids"],
    },
    "nqs_feasible": {
        "obj": float(best_feasible_obj) if best_feasible_x is not None else None,
        "cost": float(cost_f) if best_feasible_x is not None else None,
        "claim_ids": [claim_ids[i] for i in sel_f] if best_feasible_x is not None else [],
        "clusters": np.where(yf == 1)[0].tolist() if best_feasible_x is not None else [],
    },
    "n_feasible": n_feasible,
    "n_unique": len(echantillons_uniques),
    "temps_s": execution_time,
}

results_dir = os.path.join(base_dir, "results_nqs")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, f"resultats_raw_{DATASET}.json")
with open(results_path, "w") as f:
    json.dump(resultats, f, indent=4)
print(f"\nResultats: {results_path}")
