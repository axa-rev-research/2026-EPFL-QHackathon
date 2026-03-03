# 2026-EPFL-QBeasts

# Quantum-Driven Resource Allocation for Insurance Fraud Detection

---

## The Challenge!

Insurance fraud represents a major financial burden for insurers worldwide.

- 5–10% of total claims costs are attributed to fraud  
- Billions are lost globally each year  
- Thousands of suspicious cases are flagged automatically  

However, only a fixed number **k** can be manually investigated per cycle.

Which subset of cases should be investigated to maximize total avoided loss?

Fraud rarely occurs in isolation. It often emerges in clusters:
- Shared brokers  
- Geographic overlap  
- Behavioral similarity  
- Organized fraud rings  

Selecting cases independently is suboptimal.

This is a combinatorial optimization problem.

---

## Our Approach

Instead of ranking individuals, we optimize the entire portfolio of k investigations.

We model the problem as a Quadratic Unconstrained Binary Optimization (QUBO) problem and solve it using:

**QAOA (Quantum Approximate Optimization Algorithm)**  
with a constraint-preserving XY mixer.

---

## Mathematical Formulation (QUBO)

We define a binary decision variable:

- x_i = 1 if case i is investigated  
- x_i = 0 otherwise  

We minimize the following QUBO function:

$$
Q(x) = -\sum_{i=1}^{N} v_i x_i
       -\sum_{i=1}^{N}\sum_{j=i+1}^{N} \omega_{ij} x_i x_j
       +\lambda (\sum_{i=1}^{N} x_i - k)^2
$$

---

### 1. Individual Avoided Loss (v_i)

Represents the expected recovered value if fraud is confirmed.

- Derived from Risk Profile and Claim History  
- Higher v_i increases incentive to investigate  

The negative sign ensures energy minimization corresponds to profit maximization.

---

### 2. Correlation Synergy (omega_ij)

Captures operational or investigative synergy between cases.

Examples:
- Same geographic cluster  
- Same segmentation group  
- Shared professional networks  

If two cases are related:
- Investigating both may reduce marginal cost  
- Increase probability of uncovering organized fraud  

This creates quadratic interaction terms.

---

### 3. Resource Constraint (k)

The fraud department can investigate exactly k cases.

This constraint is embedded directly into the Hamiltonian using a quadratic penalty term weighted by lambda.

---

## Why This Is A Classically Hard Problem ?

With interaction terms omega_ij, the problem becomes equivalent to:

A Quadratic Knapsack / Ising Optimization problem.

The number of possible selections is:

$$
\binom{N}{k}
$$

For realistic N, brute force is infeasible.

Greedy methods:
- Ignore correlations  
- Miss globally optimal portfolios  
- Fail when fraud networks are strongly interdependent  

---

## Why Quantum?

QAOA is designed for Ising-type Hamiltonians with quadratic interactions.

Instead of exploring combinations sequentially, QAOA manipulates a superposition of all possible selections.

It leverages:
- Superposition  
- Quantum interference  
- Variational parameter optimization  

---

## From QUBO to Quantum Circuit

Pipeline:

1. QUBO formulation  
2. QUBO → Ising Hamiltonian transformation  
3. Automatic circuit generation  
4. Parameter optimization (gamma, beta)  
5. Measurement → optimal portfolio

---

## Constraint-Preserving XY Mixer

Instead of the standard X-mixer, we use an XY mixer:

$$H_M = \sum_{i=1}^{N}\sum_{j=i+1}^{N} (X_i X_j + Y_i Y_j)$$

The XY mixer conserves Hamming weight:

$$
\sum_i x_i = k
$$

This means QAOA explores only feasible investigation portfolios.

Benefits:
- Reduced search space  
- Faster convergence  
- No need for large penalty coefficients  

---

## Classical Benchmark

We benchmark QAOA against classical methods:

 - Use of a Brute Force method used as an 'oracle' to check QAOA optimization results
 - Test of a heuristic Tupo Method

---

## Repository Structure


```
2026-EPFL-QBeasts/
│
├── fraud_detection.ipynb
│   (contains all QAOA implementations on QBraid cloud servers)
│
├── Benchmark.ipynb
│   (contains classical implementations to benchmark with quantum solution)
│
├── underwriting_dataset.csv
│   (our LLM generated dataset, manually pre-processed)
│
└── README.md
```

---

## Summary

We propose a quantum-enhanced framework for:

Optimal fraud investigation allocation under resource constraints.

Key contributions:

- Rigorous QUBO modeling  
- Interaction-aware optimization  
- Constraint-preserving XY mixer  
- Full classical benchmarking  
- End-to-end implementation  

This project demonstrates how quantum optimization can address real-world portfolio selection problems in structured, correlated environments.
