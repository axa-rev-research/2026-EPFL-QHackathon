from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import GurobiOptimizer

def solve_with_gurobi(qp: QuadraticProgram):
    """
    Solves a given QuadraticProgram using the classical Gurobi exact solver.
    This serves as an absolute benchmark to evaluate the approximation 
    ratio of quantum algorithms like QAOA or VQE.
    
    Args:
        qp (QuadraticProgram): The optimization problem to solve.
        
    Returns:
        OptimizationResult: Contains the exact optimal binary vector x 
                            and the absolute minimum cost.
    """
    # 1. Verify that the gurobipy solver backend is installed in your environment
    if not GurobiOptimizer.is_gurobi_installed():
        raise RuntimeError(
            "Gurobi is not installed or the license is not found. "
            "Please install it using: pip install 'qiskit-optimization[gurobi]'"
        )

    # 2. Instantiate the classical Gurobi optimizer
    # disp=False suppresses the extensive Gurobi console logging
    optimizer = GurobiOptimizer(disp=False)
    
    # 3. Solve the quadratic program exactly
    result = optimizer.solve(qp)
    
    return result

# --- Example Usage ---
if __name__ == "__main__":
    # Define the exact same binary optimization problem used in the QAOA example
    problem = QuadraticProgram()
    
    problem.binary_var(name="x_0")
    problem.binary_var(name="x_1")
    problem.binary_var(name="x_2")
    
    # f(x) = -2x_0*x_1 + x_1*x_2 - x_0 + 2x_2
    problem.minimize(
        linear={"x_0": -1, "x_2": 2},
        quadratic={("x_0", "x_1"): -2, ("x_1", "x_2"): 1}
    )
    
    # Execute the classical exact benchmark
    try:
        benchmark_solution = solve_with_gurobi(problem)
        
        print("--- GUROBI EXACT BENCHMARK ---")
        print(f"Optimal binary vector x: {benchmark_solution.x}")
        print(f"Minimum loss value: {benchmark_solution.fval}")
        print(f"Solver Status: {benchmark_solution.status}")
        
    except RuntimeError as e:
        print(e)