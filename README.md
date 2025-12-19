# LogRegpy

## Introduction
**LogRegpy** is a Python solver framework for logistic regression, specifically to manage ℓ₀ constraints using a branch and bound algorithm.

## Development
The framework uses 2 interchangeable pieces: a brancher, and an optional initial upper bound heuristic. Brancher's contain both variable selection heuristics and core solvers. We also have capabilities for parallel branchers.

We currently support both MOSEK and sci-kit learn core solvers, as well as an experimental GPU core solver.

We have implemented the following categories of initial upper bound heuristics:
    1. Direct submission of a feasible subset
    2. Coefficient Flooring
    3. Stepwise Regression

We have implemented the following variable choice heuristics:
    1. Coefficient fractionality
    2. Greedy
    3. Random (ordered and pseudorandom)

## Testing
All test datasets are from the UC Irvine Machine Learning Repository.
Results are available in RESULTS.ipynb
To run a sample test, run the following command from the main project directory:
```bash
python -m LogRegpy.tests.sample_test
```