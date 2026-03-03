from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import WarmStartQAOAOptimizer, SlsqpOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.converters import IntegerToBinary



def minimize_binary_loss(qp: QuadraticProgram):
    """
    Minimizes a loss function defined over a binary vector x using QAOA.
    
    Args:
        qp (QuadraticProgram): The objective function and constraints.
        
    Returns:
        OptimizationResult: Contains the optimal binary vector x and the minimum cost.
    """
    # 1. Define a classical optimizer to tune the continuous QAOA parameters
    optimizer = COBYLA(maxiter=100)
    print('1')
    # 2. Define the quantum execution primitive 
    sampler = StatevectorSampler()
    print('2')
    # 3. Instantiate the QAOA solver
    # reps defines the circuit depth (p-layers of Cost and Mixer Hamiltonians)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=2)
    print('3')
    
    # 4. Wrap QAOA in the MinimumEigenOptimizer to handle the QUBO-to-Hamiltonian mapping
    qaoa_optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)

    conv = IntegerToBinary()
    
    qubo_qp = conv.convert(qp)
    print(qp.variables)
    print(qubo_qp.variables)
    # 5. Solve the binary optimization problem
    result = qaoa_optimizer.solve(qubo_qp)
    
    return result


def minimize_binary_loss_warm_start(qp: QuadraticProgram):
    """
    Minimizes a loss function defined over a binary vector x using Warm-Start QAOA.
    
    Args:
        qp (QuadraticProgram): The objective function and constraints.
        
    Returns:
        OptimizationResult: Contains the optimal binary vector x and the minimum cost.
    """
    # 1. Define a classical continuous optimizer as the pre-solver
    # This solves the relaxed (continuous) version of the QUBO
    pre_solver = SlsqpOptimizer()
    
    # 2. Define the quantum execution primitive and the classical outer-loop optimizer
    sampler = StatevectorSampler()
    optimizer = COBYLA(maxiter=100)
    
    # 3. Instantiate the standard QAOA solver
    qaoa_mes = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
    
    # 4. Wrap QAOA in the WarmStartQAOAOptimizer
    # relax_for_pre_solver=True automatically converts binary variables to continuous 
    # epsilon bounds the relaxed solutions away from exactly 0 or 1 to allow quantum exploration
    ws_qaoa = WarmStartQAOAOptimizer(
        pre_solver=pre_solver,
        relax_for_pre_solver=True,
        qaoa=qaoa_mes,
        epsilon=0.25 
    )
    
    # 5. Solve the binary optimization problem
    result = ws_qaoa.solve(qp)
    
    return result

# --- Example Usage ---
if __name__ == "__main__":
    # Define a simple binary optimization problem
    problem = QuadraticProgram()
    
    # Add binary variables x_0, x_1, x_2
    problem.binary_var(name="x_0")
    problem.binary_var(name="x_1")
    problem.binary_var(name="x_2")
    
    # Define the loss function to minimize: f(x) = -2x_0*x_1 + x_1*x_2 - x_0 + 2x_2
    problem.minimize(
        linear={"x_0": -1, "x_2": 2},
        quadratic={("x_0", "x_1"): -2, ("x_1", "x_2"): 1}
    )
    
    # Run the warm-started QAOA function
    optimal_solution = minimize_binary_loss_warm_start(problem)
    
    print(f"Optimal binary vector x: {optimal_solution.x}")
    print(f"Minimum loss value: {optimal_solution.fval}")

    