import numpy as np
import pandas as pd
import random
import math

# ------------------------------
# 1. Data loading
# ------------------------------


L_CHOICES = [10,20,50,100,200,500,1000,2000]
N_L = len(L_CHOICES)

def load_data(file_path):
    """
    Read event data from CSV, group by year, preserve original order.
    Expected columns: 'year', 'insured_loss_million' (event_id ignored).
    Returns a list yearly_events, where each element is a list of loss amounts for that year.
    """
    df = pd.read_csv(file_path)
    # Keep only the columns we need
    df = df[['year', 'insured_loss_million']]
    # Group by year and collect losses into lists, preserving original row order
    grouped = df.groupby('year')['insured_loss_million'].apply(list)
    # Sort by year to ensure chronological order (years 0,1,2,...)
    yearly_events = [grouped[year] for year in sorted(grouped.index)]
    return yearly_events

# ------------------------------
# 2. Layer recovery function (provided by teammate)
# ------------------------------

def layer_year_recovery(losses_in_year: np.ndarray, A: float, L: float, r: int) -> float:
    """
    Compute annual recovery for one layer given event losses within a year.
      R_event = min(max(loss - A, 0), L)
      R_year_raw = sum_events R_event
      R_year = min((1+r)*L, R_year_raw)
    """
    if L <= 0:
        raise ValueError("Layer limit L must be > 0.")
    R_events = np.minimum(np.maximum(losses_in_year - A, 0.0), L)
    R_raw = float(np.sum(R_events))
    cap = (1 + r) * L
    return min(cap, R_raw)

# ------------------------------
# 3. Objective function (without penalty terms)
# ------------------------------

def objective(x, yearly_events, A1, rho, Pi_gross, C, gamma):
    """
    Compute average annual net profit given decision variables.

    Parameters:
        x: list of length 2K, format [L1, r1, L2, r2, ..., LK, rK]
        yearly_events: list of yearly event loss lists (original order)
        A1: first layer attachment point (constant)
        rho: safety loading factor for premium (e.g., 0.2)
        Pi_gross: gross premium income (constant)

    Returns:
        average annual net profit (float)
    """
    K = len(x) // 2
    L = [L_CHOICES[int(x[2*i])] for i in range(K)]
    # Round r to nearest integer (reinstatement count must be integer)
    r = [x[2*i+1] for i in range(K)]

    # Compute attachment points: A[0]=A1, A[1]=A1+L0, A[2]=A1+L0+L1, ...
    A = [A1]
    for i in range(1, K):
        A.append(A[i-1] + L[i-1])

    T = len(yearly_events)

    # Pre-allocate array for yearly payouts per layer
    yearly_payouts = np.zeros((K, T))

    # Compute payouts for each layer and each year
    for ell in range(K):
        for t in range(T):
            losses = np.array(yearly_events[t])
            yearly_payouts[ell, t] = layer_year_recovery(losses, A[ell], L[ell], r[ell])

    # Compute premiums for each layer based on average payout
    premiums = np.zeros(K)
    for ell in range(K):
        R_bar = np.mean(yearly_payouts[ell, :])
        premiums[ell] = (1 + rho) * R_bar
    total_premium = np.sum(premiums)

    # Accumulate total profit and total penalty (none here)
    total_profit = 0.0
    total_penalty = 0.0

    for t in range(T):
        year_loss = np.sum(yearly_events[t])
        year_recovery = np.sum(yearly_payouts[:, t])

        # Reinstatement cost: only for amount exceeding original layer limit L[ell]
        rstm_cost = 0.0
        for ell in range(K):
            excess = max(0, yearly_payouts[ell, t] - L[ell])
            rstm_cost += premiums[ell] * (excess / L[ell])

        profit_t = Pi_gross - year_loss - total_premium + year_recovery - rstm_cost
        total_profit += profit_t
        
        retained = year_loss - year_recovery
        penalty_t = gamma * max(0, retained - C) 
        total_penalty += penalty_t

    avg_profit = total_profit / T
    avg_penalty = total_penalty / T
    objective_value = avg_profit - avg_penalty
    return objective_value

# ------------------------------
# 4. Simulated Annealing
# ------------------------------

def simulated_annealing(objective_func, bounds,discrete_indices, yearly_events, A1, rho, Pi_gross, C, gamma,
                        T0=100, alpha=0.95, T_min=1e-3, max_iter_per_temp=100, seed=42):
    """
    Simulated annealing for continuous variables.
    """
    random.seed(seed)
    np.random.seed(seed)

    n_vars = len(bounds)
    # Random initial solution within bounds
    x = []
    for i in range(n_vars):
        if i in discrete_indices:
            low, high = int(bounds[i][0]), int(bounds[i][1])
            x.append(random.randint(low, high)) 
        else:  
            x.append(random.uniform(bounds[i][0], bounds[i][1]))
    # Wrapper to pass fixed parameters
    def obj(x):
        return objective_func(x, yearly_events, A1, rho, Pi_gross,C, gamma)

    f = obj(x)
    best_x = x.copy()
    best_f = f

    T = T0
    while T > T_min:
        for _ in range(max_iter_per_temp):
            idx = random.randint(0, n_vars-1)
            x_new = x.copy()
            if idx in discrete_indices:
                low, high = int(bounds[idx][0]), int(bounds[idx][1])
                current = int(x[idx])
                candidates = [c for c in [current-1, current+1] if low <= c <= high]
                if not candidates:
                    candidates = [current]
                x_new[idx] = random.choice(candidates)
            else:
                step = 0.1 * (bounds[idx][1] - bounds[idx][0])
                delta = random.uniform(-step, step)
                x_new[idx] += delta
                x_new[idx] = max(bounds[idx][0], min(bounds[idx][1], x_new[idx]))
            f_new = obj(x_new)
            delta_f = f_new - f   # we are maximizing

            # Metropolis acceptance criterion
            if delta_f > 0 or random.random() < math.exp(delta_f / T):
                x = x_new
                f = f_new
                if f > best_f:
                    best_f = f
                    best_x = x_new.copy()
        T *= alpha

    return best_x, best_f

# ------------------------------
# 5. Main execution
# ------------------------------

if __name__ == "__main__":
    # ====== SET YOUR FILE PATH HERE ======
    data_file = "/Users/green/Downloads/传输2/研究生/Quantum hackathon/catastrophe_scenario.csv"

    # Load data
    yearly_events = load_data(data_file)
    print(f"Data loaded: {len(yearly_events)} years.")

    # Fixed parameters
    A1 = 10.0               # first layer attachment point
    rho = 0.2                # safety loading for premium
    # Gross premium income (set as 1.2 times average yearly loss)
    avg_yearly_loss = np.mean([sum(events) for events in yearly_events])
    Pi_gross = 1.2 * avg_yearly_loss
    print(f"Average yearly loss: {avg_yearly_loss:.2f}, Gross premium income: {Pi_gross:.2f}")
    
    annual_losses = [sum(events) for events in yearly_events]   
    C = np.percentile(annual_losses, 99)                         
    gamma = 1.0                                                  
    print(f"Capital threshold C (90th percentile of annual losses): {C:.2f}")
    print(f"Penalty coefficient gamma: {gamma}")


    # Number of layers
    K = 3
    # Variable bounds: L1, r1, L2, r2, L3, r3
    bounds = []
    for _ in range(K):
        bounds.append((0, N_L-1))   # L bounds
        bounds.append((0, 3))    # r bounds (will be rounded to integer in objective)
    discrete_indices = list(range(2*K)) 
    # Run simulated annealing
    best_x, best_f = simulated_annealing(
        objective_func=objective,
        bounds=bounds,
        discrete_indices = discrete_indices,
        yearly_events=yearly_events,
        A1=A1,
        rho=rho,
        Pi_gross=Pi_gross,
        C = C,
        gamma = gamma,
        T0=100,
        alpha=0.95,
        T_min=1e-3,
        max_iter_per_temp=100,
        seed=42
    )

    # Output results
    print("\nOptimal solution:")
    for i in range(K):
        L_idx = int(best_x[2*i])
        L_val = L_CHOICES[L_idx]
        r_val = int(best_x[2*i+1])
        print(f"Layer {i+1}: L = {L_val} (index {L_idx}), r = {r_val}")
    print(f"Optimal average annual net profit: {best_f:.2f}")
        
    # Compare with no reinsurance
    total_profit_no_re = 0.0
    total_penalty_no_re = 0.0
    for loss in annual_losses:
        profit_no = Pi_gross - loss
        total_profit_no_re += profit_no
        retained_no = loss
        penalty_no = gamma * max(0, retained_no - C)
        total_penalty_no_re += penalty_no
    avg_profit_no_re = total_profit_no_re / len(annual_losses)
    avg_penalty_no_re = total_penalty_no_re / len(annual_losses)
    obj_no_re = avg_profit_no_re - avg_penalty_no_re
    print(f"No reinsurance: avg profit = {avg_profit_no_re:.2f}, avg penalty = {avg_penalty_no_re:.2f}, objective = {obj_no_re:.2f}")