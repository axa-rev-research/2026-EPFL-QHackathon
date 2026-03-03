"""
Visualisation des resultats NQS - medium_35_clusters dataset.
NQS beats Greedy: 42.22 vs 40.55 (104.1%).
Figures 5 and 6 saved as PNG.
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
dataset_dir = os.path.join(base_dir, "datasets", "medium_35_clusters")
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

# ============================================================
# NQS and Greedy solutions
# ============================================================
nqs_selected_ids = [
    'S001', 'S002', 'S003', 'S005', 'S010', 'S011', 'S015', 'S016',
    'S017', 'S020', 'S021', 'S022', 'S023', 'S024', 'S026', 'S027',
    'S028', 'S029', 'S030', 'S032', 'S035'
]
nqs_selected = set(nqs_selected_ids)

# Greedy selected (the 8 traps + some others to fill budget, completing only cluster 5)
# Greedy picks by R_i/C_i ratio -- the isolated high-P claims (traps)
# For the figure we focus on NQS selected vs not; greedy info is in the bar chart

nqs_obj = 42.22
greedy_obj = 40.55
nqs_R = 36.98
nqs_B = 5.25
greedy_R = 40.55
greedy_B = 0.0  # Greedy completed cluster 5 but context says obj=40.55 is all R

# Determine which clusters are completed by NQS
nqs_completed_clusters = []
for c_idx, members in clusters.items():
    member_ids = [df["claim_id"].iloc[m] for m in members]
    if all(mid in nqs_selected for mid in member_ids):
        nqs_completed_clusters.append(c_idx)
print(f"NQS completed clusters: {nqs_completed_clusters}")

# ============================================================
# FIGURE 5 : Cluster network graph
# ============================================================
fig5, ax5 = plt.subplots(figsize=(16, 12))
ax5.set_facecolor('white')

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

pos = nx.spring_layout(G, k=2.5, iterations=200, seed=42)

# -- Draw cluster background hulls (convex hulls) --
from scipy.spatial import ConvexHull

for c_idx, members in clusters.items():
    member_ids = [df["claim_id"].iloc[m] for m in members]
    pts = np.array([pos[cid] for cid in member_ids])

    # Determine if this cluster is completed by NQS
    is_completed = c_idx in nqs_completed_clusters
    hull_color = '#27ae60' if is_completed else '#9b9b9b'
    hull_alpha = 0.12 if is_completed else 0.07
    edge_color = '#27ae60' if is_completed else '#bbbbbb'

    if len(pts) >= 3:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        center = pts.mean(axis=0)
        expanded = center + 1.35 * (hull_pts - center)
        ax5.fill(expanded[:, 0], expanded[:, 1], alpha=hull_alpha, color=hull_color,
                 edgecolor=edge_color, linewidth=2.0, linestyle='--')
        cx, cy = center
        label_text = f'C{c_idx}'
        if is_completed:
            label_text += ' (complete)'
        ax5.text(cx, cy - 0.20, label_text, fontsize=9, ha='center', va='top',
                 color='#27ae60' if is_completed else '#999999', fontweight='bold')
    elif len(pts) == 2:
        mid = pts.mean(axis=0)
        dx, dy = pts[1] - pts[0]
        norm = np.sqrt(dx**2 + dy**2)
        if norm < 1e-8:
            norm = 1.0
        perp = np.array([-dy, dx]) / norm * 0.12
        rect = np.array([
            pts[0] - perp - 0.08 * (pts[1] - pts[0]),
            pts[1] - perp + 0.08 * (pts[1] - pts[0]),
            pts[1] + perp + 0.08 * (pts[1] - pts[0]),
            pts[0] + perp - 0.08 * (pts[1] - pts[0]),
            pts[0] - perp - 0.08 * (pts[1] - pts[0])
        ])
        ax5.fill(rect[:, 0], rect[:, 1], alpha=hull_alpha, color=hull_color,
                 edgecolor=edge_color, linewidth=2.0, linestyle='--')
        label_text = f'C{c_idx}'
        if is_completed:
            label_text += ' (complete)'
        ax5.text(mid[0], mid[1] - 0.15, label_text, fontsize=9, ha='center',
                 va='top', color='#27ae60' if is_completed else '#999999',
                 fontweight='bold')
    elif len(pts) == 1:
        # Single-member cluster: draw a circle around it
        cx, cy = pts[0]
        circle = plt.Circle((cx, cy), 0.12, fill=True, alpha=hull_alpha,
                             facecolor=hull_color, edgecolor=edge_color,
                             linewidth=2.0, linestyle='--')
        ax5.add_patch(circle)
        label_text = f'C{c_idx}'
        ax5.text(cx, cy - 0.17, label_text, fontsize=9, ha='center',
                 va='top', color='#999999', fontweight='bold')

# -- Edges: light grey --
nx.draw_networkx_edges(G, pos, ax=ax5, edge_color='#cccccc', alpha=0.5,
                       width=1.2, style='solid')

# -- Nodes --
COL_SELECTED = '#1a73e8'      # blue for NQS-selected
COL_NOT_SELECTED = '#e0e0e0'  # light grey for not selected
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

# Draw non-selected (smaller)
sizes_not = [250 for _ in node_list_not]
nx.draw_networkx_nodes(G, pos, nodelist=node_list_not, ax=ax5,
                       node_color=COL_NOT_SELECTED, node_size=sizes_not,
                       edgecolors=COL_NOT_EDGE, linewidths=1.0, alpha=0.7)

# Draw selected (larger)
sizes_sel = [550 for _ in node_list_sel]
nx.draw_networkx_nodes(G, pos, nodelist=node_list_sel, ax=ax5,
                       node_color=COL_SELECTED, node_size=sizes_sel,
                       edgecolors=COL_SELECTED_EDGE, linewidths=2.5, alpha=0.95)

# -- Labels --
labels_sel = {c: c for c in node_list_sel}
labels_not = {c: c for c in node_list_not}
nx.draw_networkx_labels(G, pos, labels=labels_sel, ax=ax5, font_size=8,
                        font_weight='bold', font_color='white')
nx.draw_networkx_labels(G, pos, labels=labels_not, ax=ax5, font_size=7,
                        font_weight='normal', font_color='#666666')

# -- Legend --
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_SELECTED,
               markeredgecolor=COL_SELECTED_EDGE, markeredgewidth=2.5,
               markersize=16, label=f'Selected by NQS ({len(node_list_sel)} claims)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_NOT_SELECTED,
               markeredgecolor=COL_NOT_EDGE, markeredgewidth=1,
               markersize=11, label=f'Not selected ({len(node_list_not)} claims)'),
    plt.Line2D([0], [0], color='#cccccc', linewidth=2,
               label='Intra-cluster correlation'),
    mpatches.Patch(facecolor='#27ae60', alpha=0.2, edgecolor='#27ae60',
                   linestyle='--', label='Completed cluster (NQS)'),
    mpatches.Patch(facecolor='#9b9b9b', alpha=0.15, edgecolor='#bbbbbb',
                   linestyle='--', label='Incomplete cluster'),
]
ax5.legend(handles=handles, loc='upper left', fontsize=11,
           framealpha=0.95, fancybox=True, shadow=True,
           borderpad=1.0, handletextpad=1.0)
ax5.set_title(
    'medium_35_clusters: Insurance Claims Network (35 claims, 10 clusters, 45 qubits)\n'
    'NQS selects 21 claims, completes clusters 5 & 7 -- beats Greedy by 4.1%',
    fontsize=15
)
ax5.axis('off')

fig5.tight_layout()
fig5.savefig(os.path.join(output_dir, "fig5_clusters_network_v2.png"),
             dpi=200, bbox_inches='tight')
print("Saved fig5_clusters_network_v2.png")
plt.close(fig5)

# ============================================================
# FIGURE 6 : NQS vs Greedy comparison (2 subplots)
# ============================================================
fig6, axes = plt.subplots(1, 2, figsize=(16, 8),
                          gridspec_kw={'width_ratios': [1, 1.4]})

# ----------------------------------------------------------
# 6a: Stacked bar chart -- R + B decomposition
# ----------------------------------------------------------
methods = ['Greedy', 'NQS Constrained']
R_vals = [greedy_R, nqs_R]
B_vals = [greedy_B, nqs_B]
x = np.arange(len(methods))

bars_r = axes[0].bar(x, R_vals, 0.50, label='Reward R', color='#3498db',
                     edgecolor='white', linewidth=1.5)
bars_b = axes[0].bar(x, B_vals, 0.50, bottom=R_vals, label='Cluster Bonus B',
                     color='#e67e22', edgecolor='white', linewidth=1.5)

# Value annotations on the bars
for i in range(len(methods)):
    total = R_vals[i] + B_vals[i]
    # Show total on top
    axes[0].text(i, total + 0.8, f'{total:.2f}', ha='center', va='bottom',
                 fontsize=18, fontweight='bold', color='#2c3e50')
    # Show R value inside the R bar
    axes[0].text(i, R_vals[i] / 2, f'R={R_vals[i]:.2f}', ha='center', va='center',
                 fontsize=12, fontweight='bold', color='white')
    # Show B value on top of B bar (if B > 0)
    if B_vals[i] > 0.5:
        axes[0].text(i, R_vals[i] + B_vals[i] / 2, f'B={B_vals[i]:.2f}',
                     ha='center', va='center', fontsize=11, fontweight='bold',
                     color='white')

# Add arrow showing NQS > Greedy
axes[0].annotate('', xy=(1, nqs_obj + 3.5), xytext=(0, greedy_obj + 3.5),
                 arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
axes[0].text(0.5, max(nqs_obj, greedy_obj) + 5.5, '+4.1%',
             ha='center', va='bottom', fontsize=16, fontweight='bold',
             color='#27ae60')

# Cluster info annotations
axes[0].text(0, -3.5, 'Cluster 5 only', ha='center', va='top',
             fontsize=10, color='#7f8c8d', style='italic')
axes[0].text(1, -3.5, 'Clusters 5 & 7', ha='center', va='top',
             fontsize=10, color='#27ae60', fontweight='bold', style='italic')

axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, fontsize=13, fontweight='bold')
axes[0].set_ylabel('Objective Value', fontsize=13)
axes[0].set_ylim(-6, 55)
axes[0].set_title('Objective Decomposition (R + B)', fontsize=14)
axes[0].legend(fontsize=12, loc='upper right')

# ----------------------------------------------------------
# 6b: Horizontal bar chart of NQS-selected claims by P_i
# ----------------------------------------------------------
sorted_idx = sorted(range(len(nqs_selected_ids)),
                    key=lambda i: df.loc[df["claim_id"] == nqs_selected_ids[i], "P_i"].values[0],
                    reverse=True)
sorted_ids = [nqs_selected_ids[i] for i in sorted_idx]
sorted_P = [df.loc[df["claim_id"] == c, "P_i"].values[0] for c in sorted_ids]
sorted_C = [df.loc[df["claim_id"] == c, "C_i"].values[0] for c in sorted_ids]
sorted_M = [df.loc[df["claim_id"] == c, "M_i"].values[0] for c in sorted_ids]

# Color by whether the claim is a "trap" (isolated high-P) or cluster-member
# Identify which cluster each selected claim belongs to
claim_cluster_map = {}
for c_idx, members in clusters.items():
    member_ids = [df["claim_id"].iloc[m] for m in members]
    for mid in member_ids:
        claim_cluster_map[mid] = c_idx

# Color gradient based on P_i
colors_bar = []
for cid in sorted_ids:
    cl = claim_cluster_map.get(cid, -1)
    if cl in nqs_completed_clusters:
        colors_bar.append('#27ae60')  # green for completed cluster members
    else:
        colors_bar.append('#3498db')  # blue for others

bars = axes[1].barh(range(len(sorted_ids)), sorted_P, color=colors_bar,
                    edgecolor='white', linewidth=0.8, height=0.7)
axes[1].set_yticks(range(len(sorted_ids)))
axes[1].set_yticklabels(sorted_ids, fontsize=10, fontweight='bold')
axes[1].set_xlabel('Fraud Probability ($P_i$)', fontsize=12)
axes[1].set_title(f'21 Selected Claims (sorted by $P_i$)', fontsize=14)
axes[1].set_xlim(0, 1.15)
axes[1].invert_yaxis()

for i, (p, c, m) in enumerate(zip(sorted_P, sorted_C, sorted_M)):
    cl = claim_cluster_map.get(sorted_ids[i], -1)
    cl_label = f'C{cl}' if cl >= 0 else ''
    axes[1].text(p + 0.02, i, f'C={c:,}  M={m:,}  [{cl_label}]',
                 va='center', fontsize=8, color='#555')

# Legend for colors in right panel
handles_right = [
    mpatches.Patch(color='#27ae60', label='In completed cluster'),
    mpatches.Patch(color='#3498db', label='In incomplete cluster'),
]
axes[1].legend(handles=handles_right, loc='lower right', fontsize=10,
               framealpha=0.95)

fig6.suptitle(
    'NQS Constrained = 42.22 > Greedy = 40.55 (104.1%)\n'
    'medium_35_clusters: 35 claims, 10 clusters, Budget=82,000, alpha=1.0, k=3.0',
    fontsize=16, fontweight='bold', color='#2c3e50', y=1.02
)
fig6.tight_layout()
fig6.savefig(os.path.join(output_dir, "fig6_nqs_vs_greedy_v2.png"),
             dpi=200, bbox_inches='tight')
print("Saved fig6_nqs_vs_greedy_v2.png")
plt.close(fig6)

print("\n" + "=" * 60)
print("2 figures saved in:", output_dir)
print("=" * 60)
