# LogRegpy

## Introduction
**LogRegpy** is a Python solver framework for logistic regression, specifically to manage l0 constraints using a branch and bound algorithm.

## Development
The framework uses 3 interchangeable pieces: a core solver, a variable choice function, and an optional initial upper bound heuristic. Recently, we also added capability for a parallel solver.

We currently support both MOSEK and sklearn core solvers, with plans to expand as we work on GPU methods.

We have implemented the following categories of initial upper bound heuristics:
    1. Direct submission of a feasible subset
    2. Coefficient Flooring
    3. Stepwise Regression

We have implemented the following variable choice heuristics:
    1. Coefficient fractionality
    2. Greedy

## Testing
All test datasets are from the UC Irvine Machine Learning Repository.
Results are available in RESULTS.ipynb
To run a sample test, run the following command from the main project directory:
```bash
python -m LogRegpy.tests.sample_test
```