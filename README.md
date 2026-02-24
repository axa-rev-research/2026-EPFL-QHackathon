<p align="center">

</p>


# EPFL Quantum Hackathon 2026 – AXA x Quobly - DRAFT

Welcome to the **AXA × Quobly Optimization** challenge!  
In this hackathon, your task is to **identify**, **model**, and **quantize** a real insurance optimization problem — and demonstrate how **quantum optimization methods** could be applied, with a thought on how this might outperfom classical solvers.


## Contents

### 1. Choose an Insurance Optimization Problem  
Pick one of the problems below **or invent your own**:

- **Auto Insurance Pricing Optimization**  
  Compute the optimal premium for each customer balancing acceptance probability vs. profitability.

- **Life Insurance Capital Reserving Optimization**  
  Minimize reserve capital required while satisfying regulatory solvency constraints.

- **Reinsurance Allocation Optimization**  
  Choose coverage levels across natural catastrophes to minimize expected risk of large losses.

- **Any novel insurance‑related optimization problem**  
  (underwriting triage, fraud detection resource allocation, claims-routing optimisation, etc.)

The accuracy or realism of the insurance scenario is not the key point, what matters is having a concrete use case on which to develop and demonstrate your optimization algorithm.


### 2. Use Case Formulation  & Mathematical Model

Your formulation should include:
- A clear business description of the problem and assumptions
- Decision variables  
- Objective function (profit, capital, acceptance rate, risk cost, etc.)  
- Practical constraints (regulatory, business, portfolio, limits)  
- Synthesised toy datasets to run, these can be found in the public domain or currated through an LLM
- Complexity analysis of the problem (how difficult is it to solve and how do the variables scale with problem size)


### 3. Quantum Algorithm

Implement a gate-based quantum optimization pipeline using an algorithm of your choosing (QAOA, VQE, etc)

Your work should include:
- A well commented code detailing your circuit that implements your use case
- Run this on a **quantum simulator**
- Transform the result back into a solution that makes sense from an insurance perspective



### Extra

If you have time, provide a classical benchmark to your problem using:
- Simulated annealing  
- Greedy heuristics  
- Any simple classical optimizer of your choice  

Then compare the classical and quantum results:
- Solution quality  
- Runtime
- Scalability
- Parameter sensitivity



## Working Environment

You may use:
- **Any quantum SDK** (Qiskit, Cirq, PennyLane, pyQuil, etc.)  
- **Python notebooks**  
- **Classical optimization libraries** (dimod, scipy, gurobi, networkx, cplex)


## Documentation Requirement

Creativity is encouraged — tell a story with your findings.

Your final writeup should include:
- A clear description of the insurance problem you have chosen.
- A mathematical model formulation including (objective function, decision variables & constraints)
- A decision on why you chose your quantum algorithm for your problem
- A description of how your quantum optimization algorithm is applied
- Plots showing the effectiveness of your algorithm
- A discussion of:
  - Resource Estimation (qubits/depth)
  - Where quantum methods might provide advantage
  - Where quantum methods currently fall short
  - Effects on problem scaling
  - How the model could be extended


## Submission

1. Create a folder named after your team.  
2. Include:
   - All classical + quantum code  
   - Any dataset that was used
   - graphs/plots  
   - Notebooks  
3. Add a `README.md` summarizing your solution or include a PDF.  
4. Submit your project the EPFL Quantum Hackathon event instructions and deadlines. Ensure the project is self contained and reproducible.


## Evaluation Criteria

Your project will be evaluated on:

### 1. Correctness
Soundness of your optimization model and implementation.

### 2. Technical Depth
Quality of quantum modeling, and algorithm choices.

### 3. Comparison & Insight
How well you analyze your quantum (and classical) approach.

### 4. Creativity
Novel modeling ideas, unique insurance use cases.

### 5. Writeup Quality
Clear explanations, plots, and discussion of findings.


## Key Insight

This challenge is **not** about proving real-world quantum advantage in insurance today.  
It *is* about learning:

- How to encode messy insurance problems into clean optimization models  
- When quantum algorithms might outperform classical ones    

Good luck and happy hacking! 🚀


## Useful Links

- [Insurance Pricing Article](https://www.theactuary.com/2024/02/01/price-it-right-how-optimise-portfolios-pricing-strategy) <br>
Background on Insurance pricing and how it is generally optimized by insurers.
- [Loss Reserving - Wikipedia](https://en.wikipedia.org/wiki/Loss_reserving) <br>
Background on loss reserving definitions and how it is modelled.
- [Reinsurance Intro](https://www.investopedia.com/terms/t/treaty-reinsurance.asp) <br>
Brief introduction into natcat reinsurance.
- [Quantum in Finance Review](https://arxiv.org/pdf/2307.11230) <br>
This excellent review comprehensively studies the state-of-art of quantum computing for financial applications.
- [Challenges and Oppourtunities in Quantum Optimization](https://arxiv.org/pdf/2312.02279) <br>
Extremely thorough review of the quantum optimziation space covering all aspects of its research including problem classes, execution and benchmarking.


