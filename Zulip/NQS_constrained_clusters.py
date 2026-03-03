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

from QUBOforNQS import compute_R, compute_B_lin, load_data

# ============================================================
# Parametres
# ============================================================
DATASET = "medium_35_clusters"
B_BUDGET = 82000.0
LAMBDA1 = 10.0
MAX_SLACK_BITS = 5   # pour le cost_scale seulement
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
# Greedy solver (baseline)
# ============================================================
def greedy_solve(csv_path, clusters_path, B_budget, cost_scale,
                 p=0.5, alpha=0.3, k=3.0):
    df, clusters = load_data(csv_path, clusters_path)
    n, K = len(df), len(clusters)
    C_s = np.round(df["C_i"].values / cost_scale).astype(int)
    M_s = np.round(df["M_i"].values / cost_scale).astype(int)
    B_s = int(round(B_budget / cost_scale))
    R = compute_R(df["P_i"].values, df["v_i"].values, M_s, C_s, p)
    B_lin = compute_B_lin(clusters, C_s, alpha, k)
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
    y = np.zeros(K, dtype=int)
    for c, members in clusters.items():
        if all(x[j] == 1 for j in members):
            y[c] = 1
    selected = np.where(x == 1)[0]
    total_R = sum(R[i] for i in selected)
    total_B = sum(B_lin[c] for c in range(K) if y[c] == 1)
    return {
        "obj": total_R + total_B, "reward_R": total_R, "bonus_B": total_B,
        "cost_raw": sum(df["C_i"].values[i] for i in selected),
        "cost_scaled": budget_used, "B_s": B_s,
        "selected": selected, "x": x, "y": y,
        "claim_ids": [df["claim_id"].iloc[i] for i in selected],
    }


# ============================================================
# 1. Chargement et construction du Hamiltonien SANS slack bits
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
dossier_donnees = os.path.join(base_dir, "datasets", DATASET)
chemin_claims = os.path.join(dossier_donnees, "claims.csv")
chemin_groupes = os.path.join(dossier_donnees, "groupes_points.json")

df, clusters = load_data(chemin_claims, chemin_groupes)
n_claims, K_cl = len(df), len(clusters)

# Cost scaling (meme que dans quadraticproblem.py)
M_bits_raw = math.floor(math.log2(B_BUDGET)) + 1
scale = math.ceil(B_BUDGET / (2**MAX_SLACK_BITS - 1)) if M_bits_raw > MAX_SLACK_BITS else 1
cost_scale = scale

C_s = np.round(df["C_i"].values / scale).astype(int)
M_s = np.round(df["M_i"].values / scale).astype(int)
B_s = int(round(B_BUDGET / scale))

R = compute_R(df["P_i"].values, df["v_i"].values, M_s, C_s, P_PARAM)
B_lin = compute_B_lin(clusters, C_s, ALPHA, K_PARAM)

# Hamiltonien: que objectif + cluster consistency (PAS de budget penalty)
# Variables: x_0..x_{n-1}, y_0..y_{K-1}  (pas de z_k!)
n_qubits = n_claims + K_cl
dim = n_qubits

print(f"Dataset: {DATASET}")
print(f"Budget: {B_BUDGET} (scaled: {B_s}, scale={scale})")
print(f"Qubits: {n_claims} claims + {K_cl} clusters = {n_qubits} (pas de slack!)")

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

# Conversion QUBO -> Ising
# sigma_i = 1 - 2*x_i  =>  x_i = (1 - sigma_i) / 2
# Q(x) = sum_ij Q_ij x_i x_j
# En substituant x = (1-s)/2:
h = np.zeros(dim)
J = np.zeros((dim, dim))
offset = 0.0

for i in range(dim):
    offset += Q[i, i] / 4.0
    h[i] -= Q[i, i] / 4.0
    for j in range(i + 1, dim):
        coeff = (Q[i, j] + Q[j, i]) / 2.0
        if coeff != 0:
            offset += coeff / 4.0
            h[i] -= coeff / 4.0
            h[j] -= coeff / 4.0
            J[i, j] += coeff / 4.0

# Build Pauli strings for NetKet
pauli_strings = []
pauli_coeffs = []

# h_i * Z_i terms
for i in range(dim):
    if h[i] != 0:
        s = ["I"] * dim
        s[i] = "Z"
        pauli_strings.append("".join(s))
        pauli_coeffs.append(h[i])

# J_ij * Z_i Z_j terms
for i in range(dim):
    for j in range(i + 1, dim):
        if J[i, j] != 0:
            s = ["I"] * dim
            s[i] = "Z"
            s[j] = "Z"
            pauli_strings.append("".join(s))
            pauli_coeffs.append(J[i, j])

# Offset
pauli_strings.append("I" * dim)
pauli_coeffs.append(offset)

print(f"[Hamiltonian] {len(pauli_strings)} Pauli terms, offset={offset:.2f}")

# ============================================================
# Greedy baseline
# ============================================================
greedy = greedy_solve(chemin_claims, chemin_groupes, B_BUDGET, cost_scale,
                      p=P_PARAM, alpha=ALPHA, k=K_PARAM)
print(f"\n--- Greedy baseline ---")
print(f"Greedy obj: {greedy['obj']:.2f} ({len(greedy['selected'])} claims, "
      f"cout {greedy['cost_raw']:.0f})")

# ============================================================
# 2. Systeme quantique + sampler contraint
# ============================================================
hi = nk.hilbert.Spin(s=1/2, N=n_qubits)
H_nk = nk.operator.PauliStrings(hi, pauli_strings, pauli_coeffs)

# Custom transition rule: rejette les flips qui violent le budget
class BudgetConstrainedRule(nk.sampler.rules.LocalRule):
    """Metropolis rule that rejects flips violating the budget constraint."""

    def __init__(self, C_scaled, B_scaled, n_claims):
        super().__init__()
        self._C = np.array(C_scaled)
        self._B = B_scaled
        self._n_claims = n_claims

    def transition(self, sampler, machine, parameters, sampler_state, key):
        # Call parent LocalRule to propose a flip
        new_state, log_prob = super().transition(
            sampler, machine, parameters, sampler_state, key
        )
        return new_state, log_prob

# On ne peut pas facilement override le Metropolis dans NetKet avec un reject custom
# Approche alternative: utiliser MetropolisLocal standard mais avec un callback
# qui re-sample si infaisable. Plus simple: post-filtrage des echantillons.

# En fait, la facon la plus propre dans NetKet est d'utiliser un
# ExactSampler sur le sous-espace faisable, ou d'utiliser le MetropolisLocal
# standard et de ponderer/filtrer les echantillons.
# Pour 45 qubits, ExactSampler est impossible (2^45 etats).
# On va utiliser MetropolisLocal + post-filtrage des echantillons a chaque step.

# APPROCHE PRAGMATIQUE: on optimise H (objectif seulement) avec MetropolisLocal
# standard, puis on filtre les echantillons faisables pour l'evaluation.
# Le NQS va naturellement favoriser les etats de basse energie.
# On ajoute quand meme une penalite budget LEGERE pour guider.

# Ajoutons une penalite budget legere au Hamiltonien
LAMBDA2_SOFT = 2.0  # tres leger, juste pour guider
print(f"\nPenalite budget legere: lambda2={LAMBDA2_SOFT}")

Q_budget = np.zeros((dim, dim))
for i in range(n_claims):
    Q_budget[x_off + i, x_off + i] += LAMBDA2_SOFT * (C_s[i] ** 2 - 2 * B_s * C_s[i])
    for j in range(i + 1, n_claims):
        val = LAMBDA2_SOFT * C_s[i] * C_s[j]
        Q_budget[x_off + i, x_off + j] += val
        Q_budget[x_off + j, x_off + i] += val

Q_total = Q + Q_budget

# Reconvertir en Ising
h2 = np.zeros(dim)
J2 = np.zeros((dim, dim))
offset2 = 0.0

for i in range(dim):
    offset2 += Q_total[i, i] / 4.0
    h2[i] -= Q_total[i, i] / 4.0
    for j in range(i + 1, dim):
        coeff = (Q_total[i, j] + Q_total[j, i]) / 2.0
        if coeff != 0:
            offset2 += coeff / 4.0
            h2[i] -= coeff / 4.0
            h2[j] -= coeff / 4.0
            J2[i, j] += coeff / 4.0

pauli_strings2 = []
pauli_coeffs2 = []

for i in range(dim):
    if h2[i] != 0:
        s = ["I"] * dim
        s[i] = "Z"
        pauli_strings2.append("".join(s))
        pauli_coeffs2.append(h2[i])

for i in range(dim):
    for j in range(i + 1, dim):
        if J2[i, j] != 0:
            s = ["I"] * dim
            s[i] = "Z"
            s[j] = "Z"
            pauli_strings2.append("".join(s))
            pauli_coeffs2.append(J2[i, j])

pauli_strings2.append("I" * dim)
pauli_coeffs2.append(offset2)

H_nk = nk.operator.PauliStrings(hi, pauli_strings2, pauli_coeffs2)
print(f"[Hamiltonian] {len(pauli_strings2)} terms (avec budget leger)")

# ============================================================
# 3. Multi-start VMC
# ============================================================
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

    # Echantillonnage massif
    run_samples = vstate.sample(n_samples=20000)
    run_spins = np.array(run_samples).reshape(-1, n_qubits)
    all_samples.append(run_spins)

execution_time = time.time() - start_time
print(f"\nMulti-start termine en {execution_time:.1f}s")
print(f"Meilleur run: {best_run+1} (E={best_energy_global:.2f})")

# ============================================================
# 4. Analyse: filtrer les etats faisables
# ============================================================
spins = np.vstack(all_samples)
bin_samples = ((1 - spins) // 2).astype(int)

echantillons_uniques = np.unique(bin_samples, axis=0)
print(f"\n{len(echantillons_uniques)} etats uniques sur {len(bin_samples)} echantillons")

claim_ids = list(df["claim_id"])
C_raw = df["C_i"].values

best_obj = -float("inf")
best_x = None
best_feasible_obj = -float("inf")
best_feasible_x = None
n_feasible = 0

for x_full in echantillons_uniques:
    x_claims = x_full[:n_claims]
    y_cl = x_full[n_claims:]

    cost_scaled = sum(C_s[i] * x_claims[i] for i in range(n_claims))
    sel = np.where(x_claims == 1)[0]
    obj_R = sum(R[i] for i in sel)
    obj_B = sum(B_lin[c] for c in range(K_cl) if y_cl[c] == 1)
    obj = obj_R + obj_B

    if obj > best_obj:
        best_obj = obj
        best_x = x_full

    if cost_scaled <= B_s:
        n_feasible += 1
        if obj > best_feasible_obj:
            best_feasible_obj = obj
            best_feasible_x = x_full

print(f"Etats faisables: {n_feasible}/{len(echantillons_uniques)}")

# ============================================================
# 5. Resultats
# ============================================================
print(f"\n{'='*60}")
print(f"RESULTATS NQS CONTRAINT (sans slack bits)")
print(f"{'='*60}")

if best_x is not None:
    x_sol = best_x[:n_claims]
    y_sol = best_x[n_claims:]
    sel = np.where(x_sol == 1)[0]
    cost_raw = sum(C_raw[i] for i in sel)
    cost_sc = sum(C_s[i] for i in sel)
    print(f"Meilleur etat (tout):")
    print(f"  Objectif: {best_obj:.2f}, Claims: {len(sel)}, "
          f"Cout: {cost_raw:.0f}/{B_BUDGET:.0f}, Budget OK: {cost_sc <= B_s}")

if best_feasible_x is not None:
    xf = best_feasible_x[:n_claims]
    yf = best_feasible_x[n_claims:]
    sel_f = np.where(xf == 1)[0]
    cost_f_raw = sum(C_raw[i] for i in sel_f)
    cost_f_sc = sum(C_s[i] for i in sel_f)
    R_f = sum(R[i] for i in sel_f)
    B_f = sum(B_lin[c] for c in range(K_cl) if yf[c] == 1)

    print(f"\nMeilleur etat FAISABLE:")
    print(f"  Claims: {[claim_ids[i] for i in sel_f]}")
    print(f"  Reward (R): {R_f:.2f}")
    print(f"  Bonus (B):  {B_f:.2f}")
    print(f"  Objectif:   {best_feasible_obj:.2f}")
    print(f"  Cout (raw):  {cost_f_raw:.0f} / {B_BUDGET:.0f}")
    print(f"  Cout (scaled): {cost_f_sc} / {B_s}")
    print(f"  Clusters:   {np.where(yf == 1)[0].tolist()}")
else:
    best_feasible_obj = 0
    print("\nAucun etat faisable!")

print(f"\n{'='*60}")
print(f"COMPARAISON")
print(f"{'='*60}")
print(f"Greedy:          {greedy['obj']:.2f} (baseline)")
if best_feasible_x is not None:
    pct = best_feasible_obj / greedy['obj'] * 100 if greedy['obj'] > 0 else 0
    print(f"NQS contraint:   {best_feasible_obj:.2f} ({pct:.1f}% du greedy)")
print(f"Temps:           {execution_time:.1f}s")

# ============================================================
# 6. Export
# ============================================================
resultats = {
    "dataset": DATASET, "budget": B_BUDGET,
    "n_qubits": n_qubits, "n_qubits_saved": 5,
    "cost_scale": cost_scale,
    "greedy_obj": float(greedy["obj"]),
    "nqs_best_feasible_obj": float(best_feasible_obj) if best_feasible_x is not None else None,
    "nqs_best_obj": float(best_obj),
    "n_feasible": n_feasible,
    "n_unique": len(echantillons_uniques),
    "temps_s": execution_time,
}

results_dir = os.path.join(base_dir, "results_nqs")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, f"resultats_constrained_{DATASET}.json")
with open(results_path, "w") as f:
    json.dump(resultats, f, indent=4)
print(f"\nResultats: {results_path}")
