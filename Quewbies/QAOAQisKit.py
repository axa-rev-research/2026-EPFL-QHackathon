import numpy as np
from qiskit_optimization import QuadraticProgram
import math
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import QAOAAnsatz

def build_qubo_from_lambda_P(lambda_matrix, P, n, add_majority_constraint=True):
    """
    Construct the problem:
        - sum_i p_i x_i + sum_{i<j} lambda_ij x_i x_j
    optional constraint:
        sum_i x_i >= ceil(n/2)

    Parameters
    ----------
    lambda_matrix : (n x n) array-like
    P : array-like length n
    n : int
    add_majority_constraint : bool

    Returns
    -------
    qp : QuadraticProgram
    """
    
    lambda_matrix = np.array(lambda_matrix)
    P = np.array(P)

    if lambda_matrix.shape != (n, n):
        raise ValueError("lambda_matrix must be n x n")
    if len(P) != n:
        raise ValueError("P has to be lenght n")

    qp = QuadraticProgram()

    # Variabili binarie
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    # Termine lineare
    linear = {f"x{i}": -P[i] for i in range(n)}

    # Termini quadratici (i<j)
    quadratic = {}
    for i in range(n):
        for j in range(i+1, n):
            if lambda_matrix[i, j] != 0:
                quadratic[(f"x{i}", f"x{j}")] = lambda_matrix[i, j]

    qp.minimize(linear=linear, quadratic=quadratic)

    # Vincolo: somma x_i >= ceil(n/2)
    if add_majority_constraint:
        K = math.ceil(n / 2)
        qp.linear_constraint(
            linear={f"x{i}": 1 for i in range(n)},
            sense=">=",
            rhs=K,
            name="majority_constraint"
        )

    return qp

def compute_penalty(lambda_matrix, P):
    n = len(P)
    B_linear = np.sum(np.abs(P))
    B_quad = 0.0
    for i in range(n):
        for j in range(i+1, n):
            B_quad += abs(lambda_matrix[i, j])
    B = B_linear + B_quad
    return 2 * B

def generate_random_instance(
    n,
    p_range=(1.0, 5.0),
    lambda_range=(0.1, 1.0),
    density=1.0,
    seed=None
):
    """
    Generates a random instance:
        - sum_i p_i x_i + sum_{i<j} lambda_ij x_i x_j

    Parameters
    ----------
    n : int
        Variables number
    p_range : tuple (min, max)
        Range for P
    lambda_range : tuple (min, max)
        Range values for Lambda (off-diagonal)
    density : float in (0,1]
        Percentage of off diagonal terms
    seed : int or None
        Replica

    Returns
    -------
    P : np.array shape (n,)
    Lambda : np.array shape (n,n)
    """

    rng = np.random.default_rng(seed)

    P = rng.uniform(p_range[0], p_range[1], size=n)

    Lambda = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            if rng.random() <= density:
                value = rng.uniform(lambda_range[0], lambda_range[1])
                Lambda[i, j] = value
                Lambda[j, i] = value 

    return P, Lambda

if __name__ == "__main__":
    n = 8

    P, Lambda = generate_random_instance(
        n,
        p_range=(1.0, 3.0),
        lambda_range=(0.2, 1.5),
        density=0.7,
        seed=None
    )

    print("P =", P)
    print("Lambda =\n", Lambda)

    qp = build_qubo_from_lambda_P(Lambda, P, n)

    print(qp.prettyprint())
    M = compute_penalty(Lambda, P)

    converter = QuadraticProgramToQubo(penalty=M)
    qubo = converter.convert(qp)
    print("Variables after conversion:", qubo.get_num_vars())
    print(qubo.prettyprint())
    op, offset = qubo.to_ising()

    print("Total qubits number:", op.num_qubits)
    max_coeff = max(abs(c) for c in op.coeffs)
    op = op / max_coeff
    sampler = StatevectorSampler()

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=200),
        reps=2
    )

    ### Solver on QisKit

    solver = MinimumEigenOptimizer(qaoa)
    result = solver.solve(qp)

    print("Solution:", result.x)
    print("Target value:", result.fval)
    print("Sum of xi:", sum(result.x))

