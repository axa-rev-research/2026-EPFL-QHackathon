import numpy as np
import matplotlib.pyplot as plt
 
from scipy.optimize import minimize
from typing import Sequence
 
from qiskit.primitives import StatevectorEstimator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.circuit.library import XGate
from qiskit.circuit.library import efficient_su2
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
)
 
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, Estimator
 
from qiskit_ibm_catalog import QiskitServerless, QiskitFunction


def visualize_results(results):
    plt.plot(results["cost_history"], lw=2)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.show()
 
 
def build_callback(
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    estimator: BaseEstimatorV2,
    callback_dict: dict,
):
    def callback(current_vector):
        # Keep track of the number of iterations
        callback_dict["iters"] += 1
        # Set the prev_vector to the latest one
        callback_dict["prev_vector"] = current_vector
        # Compute the value of the cost function at the current vector
        current_cost = (
            estimator.run([(ansatz, hamiltonian, [current_vector])])
            .result()[0]
            .data.evs[0]
        )
        callback_dict["cost_history"].append(current_cost)
        # Print to screen on single line
        print(
            "Iters. done: {} [Current cost: {}]".format(
                callback_dict["iters"], current_cost
            ),
            end="\r",
            flush=True,
        )
 
    return callback
from QUBOforNQS import get_hamiltonian

hamiltonian, offset, info = get_hamiltonian(
    csv_path="/users/eleves-a/2024/adel.mana/.anaconda/QuantumInsuranceResourceAllocations/datasets/dataset_02/claims.csv",
    clusters_json_path="/users/eleves-a/2024/adel.mana/.anaconda/QuantumInsuranceResourceAllocations/clusters.json"
)
num_spins = hamiltonian.num_qubits  # ← on prend le nombre de qubits depuis votre hamiltonien

ansatz = efficient_su2(num_qubits=num_spins, reps=3)

service = QiskitRuntimeService()
backend = service.least_busy(
    operational=True, min_num_qubits=num_spins, simulator=False
)

ansatz.draw("mpl", style="iqp")


target = backend.target
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
pm.scheduling = PassManager(
    [
        ALAPScheduleAnalysis(durations=target.durations()),
        PadDynamicalDecoupling(
            durations=target.durations(),
            dd_sequence=[XGate(), XGate()],
            pulse_alignment=target.pulse_alignment,
        ),
    ]
)
ansatz_ibm = pm.run(ansatz)
observable_ibm = hamiltonian.apply_layout(ansatz_ibm.layout)
ansatz_ibm.draw("mpl", scale=0.6, style="iqp", fold=-1, idle_wires=False)

def cost_func(
    params: Sequence,
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    estimator: BaseEstimatorV2,
) -> float:
    """Ground state energy evaluation."""
    return (
        estimator.run([(ansatz, hamiltonian, [params])])
        .result()[0]
        .data.evs[0]
    )
 
 
num_params = ansatz_ibm.num_parameters
params = 2 * np.pi * np.random.random(num_params)
 
callback_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

with Session(backend=backend) as session:
    estimator = Estimator()
    callback = build_callback(
        ansatz_ibm, observable_ibm, estimator, callback_dict
    )
    res = minimize(
        cost_func,
        x0=params,
        args=(ansatz_ibm, observable_ibm, estimator),
        callback=callback,
        method="cobyla",
        options={"maxiter": 100},
    )
visualize_results(callback_dict)
print(f'Estimated ground state energy: {res["fun"]}')