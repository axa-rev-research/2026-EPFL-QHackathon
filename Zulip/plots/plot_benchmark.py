import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# === Resultats compiles ===
optimal = 48.78

methods = [
    # (label, objectif R+B, temps_s, couleur_groupe, budget_ok)
    ("Optimal\n(brute-force)", 48.78, 0.01, "gold", True),
    ("Greedy", 45.64, 0.01, "#4CAF50", False),
    ("QAOA warm-start\n(p=8, \u03b5=0.3, 300 steps)", 45.64, 231, "#2196F3", False),
    ("QAOA warm-start\nmulti-start (5\u00d7100, \u03b5=0.5)", 45.64, 385, "#2196F3", False),
    ("RQAOA\n(p=15, \u03bb\u2082: 0\u21924, 5 starts\nwarm-init y=0,z=1)", 41.36, 1365, "#E91E63", True),
    ("QAOA multi-start\n(5\u00d760, p=8)", 37.29, 231, "#64B5F6", False),
    ("RQAOA\n(p=8, 200 steps, \u03bb\u2082=4)", 37.52, 445, "#F48FB1", False),
    ("RQAOA\n(p=3, 50 steps, \u03bb\u2082=4)", 32.75, 48, "#F48FB1", False),
    ("RQAOA\n(p=8, 200 steps, \u03bb\u2082: 0.5\u21924)", 29.81, 419, "#F48FB1", False),
    ("QAOA Adam\n(p=8, 300 steps)", 23.09, 231, "#90CAF9", False),
]

# Sort by objective descending
methods.sort(key=lambda x: x[1], reverse=True)

labels = [m[0] for m in methods]
objectives = [m[1] for m in methods]
times = [m[2] for m in methods]
colors = [m[3] for m in methods]
budget_ok = [m[4] for m in methods]
pct_optimal = [o / optimal * 100 for o in objectives]

fig, ax = plt.subplots(figsize=(14, 8))

bars = ax.barh(range(len(methods)), pct_optimal, color=colors, edgecolor="black",
               linewidth=0.8, height=0.7, alpha=0.9)

# Annotations: % + objectif + temps
for i, (bar, pct, obj, t, bok) in enumerate(zip(bars, pct_optimal, objectives, times, budget_ok)):
    # % inside bar
    x_text = min(pct - 2, 92)
    ax.text(x_text, i, f"{pct:.1f}%", va="center", ha="right",
            fontsize=11, fontweight="bold", color="white")
    # Objectif + temps outside
    time_str = f"{t:.0f}s" if t >= 1 else "<0.01s"
    budget_str = " [OK]" if bok else " [~]"
    ax.text(pct + 1, i, f"obj={obj:.1f}  |  {time_str}{budget_str}",
            va="center", ha="left", fontsize=9, color="#333333")

# Reference lines
ax.axvline(x=100, color="gold", linestyle="-", linewidth=2, alpha=0.7, label="Optimal (100%)")
ax.axvline(x=93.6, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.7, label="Greedy (93.6%)")

ax.set_yticks(range(len(methods)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("% de l'optimal (brute-force = 48.78)", fontsize=12)
ax.set_title("Benchmark : QAOA vs RQAOA vs Greedy\nDataset small_12, Budget=20000, 20 qubits",
             fontsize=14, fontweight="bold")
ax.set_xlim(0, 115)
ax.grid(True, alpha=0.2, axis="x")
ax.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="gold", edgecolor="black", label="Optimal (brute-force)"),
    Patch(facecolor="#4CAF50", edgecolor="black", label="Classique (Greedy)"),
    Patch(facecolor="#2196F3", edgecolor="black", label="QAOA warm-start"),
    Patch(facecolor="#64B5F6", edgecolor="black", label="QAOA multi-start"),
    Patch(facecolor="#90CAF9", edgecolor="black", label="QAOA basique"),
    Patch(facecolor="#E91E63", edgecolor="black", label="RQAOA (meilleur)"),
    Patch(facecolor="#F48FB1", edgecolor="black", label="RQAOA (autres)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
          framealpha=0.9, edgecolor="gray")

# Budget annotation
ax.text(113, len(methods) - 0.5,
        "[OK] = budget respecte\n[~]  = legere violation\n       (scaling)",
        fontsize=8, va="top", ha="right", color="#666666",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#cccccc"))

plt.tight_layout()
plt.savefig("benchmark_complet.png", dpi=200, bbox_inches="tight")
print("Plot saved: benchmark_complet.png")
