<p align="center">

</p>


# EPFL Quantum Hackathon 2026 – AXA x Quobly

Welcome to the **AXA × Quobly Optimization** challenge!  
In this hackathon, your task is to **identify**, **model**, and **quantize** a real insurance optimization problem — and demonstrate how **quantum optimization methods** (QAOA, VQE, annealing etc.) could provide advantages over classical solvers.

---

## Contents

### 1. Choose an Insurance Optimization Problem  
Pick one of the problems below **or invent your own**:

- **Auto Insurance Pricing Optimization**  
  Compute the optimal premium for each customer balancing acceptance probability vs. profitability.

- **Life Insurance Capital Reserving Optimization**  
  Minimize reserve capital while satisfying regulatory solvency constraints.

- **Reinsurance Allocation Optimization**  
  Choose coverage levels across perils to minimize expected cost of large losses.

- **Any novel insurance‑related optimization problem**  
  (underwriting triage, fraud detection resource allocation, claims-routing optimisation, etc.)

---

## A. Use Case Formulation

Your submission should include:
- A clear business description of the problem  
- Objective function (profit, capital, acceptance rate, risk cost, etc.)  
- Practical constraints (regulatory, business, portfolio, limits)  
- Explanation of why this is a **hard optimization problem**

---

## B. Mathematical Modeling

Formulate your use case as a small but complete mathematical optimization model.

Define:
- Decision variables  
- Objective (to minimize/maximize)  
- Constraints with penalties  
- Discussion of modeling assumptions, tradeoffs and classical state-of-art


---

## C. Toy Dataset

Construct/find a **synthetic dataset** tailored to your use case.  
Examples:
- Customer features + claim probabilities  
- Peril frequency/severity tables  
- Regulatory capital factors  
- Simplified underwriting or portfolio metrics  

The dataset should be intentionally small so teams can run quantum algorithms end-to-end.

---

## D. Quantum Algorithm + Embedding

Implement a quantum optimization pipeline using:
- QAOA  
- Quantum annealing  
- VQE‑based cost minimization  

Your work should include:
- a
- Running on a **quantum simulator**

---

## E. Classical Baseline

Provide at least one classical solver:
- Simulated annealing  
- Greedy heuristics  
- MILP for tiny instances  
- Any simple classical optimizer of your choice  

Then compare the classical and quantum results:
- Solution quality  
- Runtime and scaling  
- Parameter sensitivity

---

## F. Analysis & Discussion

Write a short final analysis including:
- Comparison between classical and quantum solutions  
- Resource estimates (qubits, depth, embedding overhead)  
- Where quantum methods might provide advantage  
- Where they currently fall short  
- Effect of problem scaling  
- How the model could be extended  

Creativity is encouraged — tell a story with your findings.

---

## Working Environment

You may use:
- **Any quantum SDK** (Qiskit, Cirq, PennyLane, pyQuil, etc.)  
- **Python notebooks**  
- **Classical optimization libraries** (dimod, scipy, gurobi, networkx, cplex)

Only synthetic/public data is allowed — no private customer data.

---

## Submission

1. Create a folder named after your team.  
2. Include:
   - All classical + quantum code  
   - Toy datasets
   - graphs/plots  
   - Notebooks  
3. Add a `README.md` summarizing your solution or include a PDF.  
4. Ensure the project is fully reproducible.

---

## Evaluation Criteria

Your project will be evaluated on:

### 1. Correctness
Soundness of your optimization model and implementation.

### 2. Technical Depth
Quality of quantum modeling, embedding, and algorithm choices.

### 3. Comparison & Insight
How well you analyze classical vs quantum approaches.

### 4. Creativity
Novel modeling ideas, unique insurance use cases, clever embeddings.

### 5. Writeup Quality
Clear explanations, plots, and discussion of findings.

---

## Key Insight

This challenge is **not** about proving real-world quantum advantage today.  
It *is* about learning:

- How to encode messy insurance problems into clean optimization models  
- When quantum heuristics might outperform classical ones  
- How quantum hardware constraints shape algorithm design  

Good luck and happy hacking! 🚀
