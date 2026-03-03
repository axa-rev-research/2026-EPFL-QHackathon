from quboforpennylane import build_qubo_from_files, run_vqe
import matplotlib.pyplot as plt


Q, info = build_qubo_from_files('/users/eleves-a/2024/max.anglade/QuantumInsuranceResourceAllocations/small_15/claims.csv','/users/eleves-a/2024/max.anglade/QuantumInsuranceResourceAllocations/small_15/groupes_points.json' )

result = run_vqe(Q, info,
                 n_layers=1,
                 optimizer_name="adam",
                 stepsize=0.01,
                 max_steps=300,
                 draw=True,
                 gpu=True)

print(f"Optimal L = {result['energy']:.4f}")
print(f"Selected claims: {result['selected_claims']}")
print(f"Active clusters: {result['active_clusters']}")

import matplotlib.pyplot as plt
plt.plot(result["history"])
plt.xlabel("Step"); plt.ylabel("L"); plt.title("VQE Convergence")
plt.savefig("vqe_convergence.png"); plt.show()