"""
Visualisation des resultats NQS - Insurance Resource Allocation Hackathon.
4 figures sauvees en PNG sur le serveur.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
import json
import os
import pandas as pd

# ============================================================
# Style global
# ============================================================
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafbfc',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'font.family': 'DejaVu Sans',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# ============================================================
# Chargement donnees
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "datasets", "medium_35")
output_dir = os.path.join(base_dir, "figures")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(os.path.join(dataset_dir, "claims.csv"))
with open(os.path.join(dataset_dir, "groupes_points.json")) as f:
    clusters_json = json.load(f)

claim_to_idx = {cid: i for i, cid in enumerate(df["claim_id"])}
clusters = {}
for c_str, members in clusters_json.items():
    clusters[int(c_str)] = [claim_to_idx[m] for m in members]

n_claims = len(df)
K_cl = len(clusters)

# Solution NQS (egale greedy)
nqs_selected_ids = ['S001', 'S002', 'S010', 'S015', 'S019', 'S022', 'S023', 'S029', 'S033', 'S034']
nqs_selected = set(nqs_selected_ids)
greedy_obj = 68.72

# ============================================================
# FIGURE 1 : Structure des clusters (epure)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(14, 11))
ax1.set_facecolor('white')

G = nx.Graph()
for i in range(n_claims):
    G.add_node(df["claim_id"].iloc[i])

node_cluster = {}
for i in range(n_claims):
    cid = df["claim_id"].iloc[i]
    node_cluster[cid] = -1
    for c_idx, members in clusters.items():
        if i in members:
            node_cluster[cid] = c_idx
            break

# Intra-cluster edges
for c_idx, members in clusters.items():
    member_ids = [df["claim_id"].iloc[m] for m in members]
    for ii in range(len(member_ids)):
        for jj in range(ii + 1, len(member_ids)):
            G.add_edge(member_ids[ii], member_ids[jj], cluster=c_idx)

pos = nx.spring_layout(G, k=3.0, iterations=150, seed=42)

# -- Draw cluster background hulls (convex hulls) --
from scipy.spatial import ConvexHull
for c_idx, members in clusters.items():
    member_ids = [df["claim_id"].iloc[m] for m in members]
    pts = np.array([pos[cid] for cid in member_ids])
    if len(pts) >= 3:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        # Expand hull slightly
        center = pts.mean(axis=0)
        expanded = center + 1.3 * (hull_pts - center)
        ax1.fill(expanded[:, 0], expanded[:, 1], alpha=0.07, color='#9b9b9b',
                 edgecolor='#bbbbbb', linewidth=1.5, linestyle='--')
        cx, cy = center
        ax1.text(cx, cy - 0.18, f'C{c_idx}', fontsize=8, ha='center', va='top',
                 color='#999999', fontweight='bold')
    elif len(pts) == 2:
        mid = pts.mean(axis=0)
        dx, dy = pts[1] - pts[0]
        norm = np.sqrt(dx**2 + dy**2)
        perp = np.array([-dy, dx]) / norm * 0.12
        rect = np.array([pts[0] - perp - 0.05*(pts[1]-pts[0]),
                          pts[1] - perp + 0.05*(pts[1]-pts[0]),
                          pts[1] + perp + 0.05*(pts[1]-pts[0]),
                          pts[0] + perp - 0.05*(pts[1]-pts[0]),
                          pts[0] - perp - 0.05*(pts[1]-pts[0])])
        ax1.fill(rect[:, 0], rect[:, 1], alpha=0.07, color='#9b9b9b',
                 edgecolor='#bbbbbb', linewidth=1.5, linestyle='--')
        ax1.text(mid[0], mid[1] - 0.15, f'C{c_idx}', fontsize=8, ha='center',
                 va='top', color='#999999', fontweight='bold')

# -- Edges: light grey --
nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='#cccccc', alpha=0.6,
                       width=1.5, style='solid', )

# -- Nodes --
COL_SELECTED = '#1a73e8'    # blue for selected
COL_NOT_SELECTED = '#e0e0e0' # light grey for not selected
COL_SELECTED_EDGE = '#0d47a1'
COL_NOT_EDGE = '#aaaaaa'

node_list_sel = []
node_list_not = []
for i in range(n_claims):
    cid = df["claim_id"].iloc[i]
    if cid in nqs_selected:
        node_list_sel.append(cid)
    else:
        node_list_not.append(cid)

# Draw non-selected (background)
sizes_not = [350 + 600 * df.loc[df["claim_id"]==c, "P_i"].values[0] for c in node_list_not]
nx.draw_networkx_nodes(G, pos, nodelist=node_list_not, ax=ax1,
                       node_color=COL_NOT_SELECTED, node_size=sizes_not,
                       edgecolors=COL_NOT_EDGE, linewidths=1.2, alpha=0.7)

# Draw selected (foreground, bigger)
sizes_sel = [600 + 1000 * df.loc[df["claim_id"]==c, "P_i"].values[0] for c in node_list_sel]
nx.draw_networkx_nodes(G, pos, nodelist=node_list_sel, ax=ax1,
                       node_color=COL_SELECTED, node_size=sizes_sel,
                       edgecolors=COL_SELECTED_EDGE, linewidths=3.0, alpha=0.95)

# -- Labels --
labels_sel = {c: c for c in node_list_sel}
labels_not = {c: c for c in node_list_not}
nx.draw_networkx_labels(G, pos, labels=labels_sel, ax=ax1, font_size=10,
                        font_weight='bold', font_color='white')
nx.draw_networkx_labels(G, pos, labels=labels_not, ax=ax1, font_size=8,
                        font_weight='normal', font_color='#666666')

# -- Legend --
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_SELECTED,
               markeredgecolor=COL_SELECTED_EDGE, markeredgewidth=2.5,
               markersize=16, label='Selected by NQS (10 claims)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_NOT_SELECTED,
               markeredgecolor=COL_NOT_EDGE, markeredgewidth=1,
               markersize=12, label='Not selected (25 claims)'),
    plt.Line2D([0], [0], color='#cccccc', linewidth=2,
               label='Intra-cluster correlation'),
    mpatches.Patch(facecolor='#9b9b9b', alpha=0.15, edgecolor='#bbbbbb',
                   linestyle='--', label='Cluster boundary'),
]
ax1.legend(handles=handles, loc='upper left', fontsize=12,
           framealpha=0.95, fancybox=True, shadow=True,
           borderpad=1.0, handletextpad=1.0)
ax1.set_title('Insurance Claims Network\n'
              '10 selected claims (blue) out of 35 | 10 clusters | Budget: 20,663 / 20,000')
ax1.axis('off')

fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, "fig1_clusters.png"), dpi=200, bbox_inches='tight')
print("Saved fig1_clusters.png")
plt.close(fig1)

# ============================================================
# FIGURE 2 : NQS vs Greedy
# ============================================================
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2a: Objective comparison bar
methods = ['Greedy', 'NQS Constrained']
R_vals = [67.67, 67.67]
B_vals = [1.05, 1.05]
x = np.arange(len(methods))

bars_r = axes[0].bar(x, R_vals, 0.55, label='Reward R', color='#3498db',
                     edgecolor='white', linewidth=1.5)
bars_b = axes[0].bar(x, B_vals, 0.55, bottom=R_vals, label='Cluster Bonus B',
                     color='#e67e22', edgecolor='white', linewidth=1.5)

for i in range(len(methods)):
    total = R_vals[i] + B_vals[i]
    axes[0].text(i, total + 1.5, f'{total:.2f}', ha='center', va='bottom',
                 fontsize=18, fontweight='bold', color='#2c3e50')

axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, fontsize=14)
axes[0].set_ylabel('Objective Value')
axes[0].set_ylim(0, 85)
axes[0].set_title('Objective Decomposition (R + B)')
axes[0].legend(fontsize=12, loc='upper right')

# Add "=" sign between bars
axes[0].annotate('=', xy=(0.5, 40), fontsize=40, fontweight='bold',
                 color='#27ae60', ha='center', va='center')

# 2b: Selected claims detail
sorted_idx = sorted(range(len(nqs_selected_ids)),
                    key=lambda i: df.loc[df["claim_id"]==nqs_selected_ids[i], "P_i"].values[0],
                    reverse=True)
sorted_ids = [nqs_selected_ids[i] for i in sorted_idx]
sorted_P = [df.loc[df["claim_id"]==c, "P_i"].values[0] for c in sorted_ids]
sorted_C = [df.loc[df["claim_id"]==c, "C_i"].values[0] for c in sorted_ids]
sorted_M = [df.loc[df["claim_id"]==c, "M_i"].values[0] for c in sorted_ids]

colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(sorted_ids)))
bars = axes[1].barh(range(len(sorted_ids)), sorted_P, color=colors_bar,
                    edgecolor='white', linewidth=1, height=0.7)
axes[1].set_yticks(range(len(sorted_ids)))
axes[1].set_yticklabels(sorted_ids, fontsize=12, fontweight='bold')
axes[1].set_xlabel('Fraud Probability ($P_i$)')
axes[1].set_title('10 Selected Claims (sorted by $P_i$)')
axes[1].set_xlim(0, 1.05)
axes[1].invert_yaxis()

for i, (p, c, m) in enumerate(zip(sorted_P, sorted_C, sorted_M)):
    axes[1].text(p + 0.02, i, f'C={c:,}  M={m:,}', va='center', fontsize=9, color='#555')

fig2.suptitle('NQS Constrained = Greedy: 68.72  (100%)',
              fontsize=18, fontweight='bold', color='#27ae60', y=1.02)
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, "fig2_nqs_vs_greedy.png"), dpi=200, bbox_inches='tight')
print("Saved fig2_nqs_vs_greedy.png")
plt.close(fig2)

# ============================================================
# FIGURE 3 : Convergence
# ============================================================
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(13, 8),
                                   gridspec_kw={'height_ratios': [2.5, 1]})

steps = [0, 50, 100, 150, 200, 250, 299]
energies = [618.64, -988.63, -991.99, -993.31, -993.66, -994.28, -994.57]
variances = [1392214, 759.55, 171.73, 7.10, 27.14, 4.35, 2.0]

# Energy
ax3a.plot(steps, energies, 'o-', color='#2980b9', linewidth=3, markersize=10,
          markerfacecolor='white', markeredgewidth=2.5)
ax3a.fill_between(steps, energies, alpha=0.08, color='#2980b9')
ax3a.axhline(y=-994.57, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7,
             label='Final: E = -994.57')
ax3a.set_ylabel('QUBO Energy', fontsize=14)
ax3a.set_title('Energy Convergence (RBM + SR, lr=0.01, diag_shift=0.5)')
ax3a.legend(fontsize=13, loc='center right')

ax3a.annotate('Random init', xy=(0, 618.64), xytext=(50, 300),
              arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2),
              fontsize=12, color='#e74c3c', fontweight='bold')
ax3a.annotate('E = -994.6\n(converged)', xy=(299, -994.57), xytext=(200, -970),
              arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
              fontsize=12, color='#27ae60', fontweight='bold')

# Variance
ax3b.semilogy(steps, variances, 's-', color='#e74c3c', linewidth=2.5, markersize=9,
              markerfacecolor='white', markeredgewidth=2)
ax3b.fill_between(steps, variances, alpha=0.1, color='#e74c3c')
ax3b.set_xlabel('VMC Iteration', fontsize=14)
ax3b.set_ylabel('Variance (log)', fontsize=14)
ax3b.set_title('Variance Reduction (no mode collapse)')

ax3b.annotate('Var = 1.4M', xy=(0, 1392214), xytext=(50, 5e5),
              arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.5),
              fontsize=11, color='#8e44ad')
ax3b.annotate('Var = 2.0', xy=(299, 2.0), xytext=(220, 100),
              arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
              fontsize=11, color='#27ae60', fontweight='bold')

fig3.tight_layout(h_pad=2)
fig3.savefig(os.path.join(output_dir, "fig3_convergence.png"), dpi=200, bbox_inches='tight')
print("Saved fig3_convergence.png")
plt.close(fig3)

# ============================================================
# FIGURE 4 : Benchmark toutes approches
# ============================================================
fig4, ax4 = plt.subplots(figsize=(16, 8))

benchmark = [
    ("Greedy\n(baseline)",         68.72, True,  "Ratio R/C"),
    ("Jastrow\nlambda2=10",        40.44, False, "Budget violated"),
    ("Jastrow\nlambda2=50",        52.69, False, "Budget violated"),
    ("RBM\nlambda2=100",           16.64, True,  "Mode collapse\nVar=0"),
    ("Jastrow\nlambda1=15",        87.82, False, "Best obj\nbut infeasible"),
    ("RBM diag=1.0\n(1st feasible)", 47.31, True, "First feasible\nNQS result"),
    ("RBM diag=0.5\n3x200 multi",  53.95, True,  "78.5% greedy"),
    ("Constrained\nlr=0.05",       59.68, True,  "86.8% greedy"),
    ("Constrained\nlr=0.01",       68.72, True,  "= GREEDY"),
]

labels = [b[0] for b in benchmark]
values = [b[1] for b in benchmark]
feasible = [b[2] for b in benchmark]
notes = [b[3] for b in benchmark]

bar_colors = []
for i, (_, val, feas, _) in enumerate(benchmark):
    if i == 0:
        bar_colors.append('#7f8c8d')
    elif not feas:
        bar_colors.append('#e74c3c')
    elif val >= greedy_obj:
        bar_colors.append('#27ae60')
    else:
        bar_colors.append('#3498db')

x_pos = np.arange(len(benchmark))
bars = ax4.bar(x_pos, values, color=bar_colors, edgecolor='white',
               linewidth=2, width=0.7)

# Greedy line
ax4.axhline(y=greedy_obj, color='#2c3e50', linestyle='--', linewidth=2.5, alpha=0.6, )
ax4.text(len(benchmark) - 0.3, greedy_obj + 2, f'Greedy = {greedy_obj}',
         fontsize=13, color='#2c3e50', fontweight='bold', ha='right')

# Value labels
for i, (val, feas) in enumerate(zip(values, feasible)):
    color = '#2c3e50'
    suffix = ''
    if not feas:
        color = '#c0392b'
        suffix = '\nINFEASIBLE'
    ax4.text(i, val + 1.5, f'{val:.1f}{suffix}', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color=color)

# Config notes below bars
for i, note in enumerate(notes):
    ax4.text(i, -6, note, ha='center', va='top', fontsize=8,
             style='italic', color='#7f8c8d')

ax4.set_xticks(x_pos)
ax4.set_xticklabels(labels, fontsize=10)
ax4.set_ylabel('Objective Value (R + B)', fontsize=14)
ax4.set_ylim(-15, 100)
ax4.set_title('NQS Optimization Benchmark: medium_35 (45-50 qubits)\n'
              'From mode collapse to matching Greedy')

handles = [
    mpatches.Patch(color='#7f8c8d', label='Greedy baseline'),
    mpatches.Patch(color='#e74c3c', label='Infeasible (budget violated)'),
    mpatches.Patch(color='#3498db', label='Feasible (below greedy)'),
    mpatches.Patch(color='#27ae60', label='Feasible (= greedy)'),
]
ax4.legend(handles=handles, loc='upper left', fontsize=12, framealpha=0.95,
           fancybox=True, shadow=True)

# Progress arrow
ax4.annotate('', xy=(8.2, 72), xytext=(5.5, 56),
             arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3.5))
ax4.text(7.0, 63, 'Iterative\nimprovement', fontsize=13, color='#27ae60',
         fontweight='bold', ha='center', rotation=15)

fig4.tight_layout()
fig4.savefig(os.path.join(output_dir, "fig4_benchmark.png"), dpi=200, bbox_inches='tight')
print("Saved fig4_benchmark.png")
plt.close(fig4)

print("\n" + "=" * 60)
print("4 figures saved in:", output_dir)
print("=" * 60)
