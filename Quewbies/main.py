from cost_function import get_profit_coefficients_vectorized, generate_qps_for_dataset
from optimizer import minimize_binary_loss_warm_start
from generate_dataset import (
    generate_dataset,
    GLOBAL_PARAMS,
    CONTROL_NAMES,
    N_CONTROLS,
    KAPPA,
    ALPHA,
    BETA,
    FRIC,
    INTERACTIONS,
    DEPENDENCIES,
    P_MIN,
    P_MAX
)
print('a')

row, _ = generate_dataset(n_companies=1)
print('b')
coefficients = get_profit_coefficients_vectorized(
    row,
    GLOBAL_PARAMS,
    GLOBAL_PARAMS["w"],
    GLOBAL_PARAMS["wij"],
    ALPHA,
    BETA,
    KAPPA,
    FRIC,
    INTERACTIONS
    )

qps = generate_qps_for_dataset(coefficients, num_companies=1, p_min=1, p_max=100)
optimal_solutions = []
for qp in qps:
    print(qp.get_num_vars)
    optimal_solution = minimize_binary_loss_warm_start(qp)
    optimal_solutions.append(optimal_solutions)
    
print(f"Optimal binary vector x: {optimal_solution.x}")
print(f"Minimum loss value: {optimal_solution.fval}")