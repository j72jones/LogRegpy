from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.upper_bounding_func import UpperBounder
from typing import List, Tuple
from numpy import setdiff1d
import numpy as np
from sklearn.linear_model import LogisticRegression
from LogRegpy.bound_algorithms.mosek_logistic_regression import logisticRegression
import time
import math



class VariableFlooring(UpperBounder):
    """
    Finds a feasible bound with variable flooring.
    """

    def __init__(self, data: ProblemData, penalty="l2", solver="saga") -> None:
        self.data = data
        acceptable_penalties = {"l2", "l1", "elasticnet", None}
        acceptable_solvers = {"MOSEK", "lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"}
        if penalty not in acceptable_penalties:
            raise ValueError(f"Invalid method: {penalty}. Must be one of {acceptable_penalties}.")
        if solver not in acceptable_solvers:
            raise ValueError(f"Invalid solver: {solver}. Must be one of {acceptable_solvers}.")
        acceptable_penalties = {
            "lbfgs": ("l2", None),
            "liblinear": ("l1", "l2"),
            "newton-cg": ("l2", None),
            "newton-cholesky": ("l2", None),
            "sag": ("l2", None),
            "saga": ("l1", "l2", None),
            "MOSEK": ("l2", None)
        }
        if penalty not in acceptable_penalties[solver]:
            raise ValueError(f"Invalid penalty: {penalty}. Solver {solver} only supports the following penalties: {acceptable_penalties[solver]}.")
        self.penalty = penalty
        self.solver = solver

    def __call__(self) -> Tuple[float, float, list[int]]:
        start_time = time.time()
        print(f"\033[KRunning variable flooring heuristic | Total Running Time = {time.time()-start_time} seconds", end="\r")
        if self.solver == "MOSEK":
            coefs, obj = logisticRegression(self.data.X, self.data.y, lamb=1 if self.penalty=="l2" else 0)
            variances = np.var(self.data.X, axis=0)
            variances[variances == 0] = 1
            kept_vars = sorted(range(self.data.n), key=lambda k: coefs[k]/variances[k])[:self.data.k]
            x = self.data.X[:, kept_vars]
            _, objective = logisticRegression(x, self.data.y)
        else:
            model = LogisticRegression(penalty=self.penalty, fit_intercept=False, solver=self.solver)
            # Train the logistic regression model
            model.fit(self.data.X, self.data.y)
            # Select vars
            theta = model.coef_.flatten()
            variances = np.var(self.data.X, axis=0)
            variances[variances == 0] = 1
            kept_vars = sorted(range(self.data.n), key=lambda k: theta[k]/variances[k])[:self.data.k]
            # Train the model again
            model = LogisticRegression(penalty=None, fit_intercept=False, solver=self.solver if self.solver != "liblinear" else "saga")
            x = self.data.X[:, kept_vars]
            model.fit(x, self.data.y)
            # Calculate scores
            theta = model.coef_.flatten()
            margin = (1 - 2 * self.data.y) * (x @ theta)  # (n,)
            logistic_terms = np.log1p(np.exp(margin))  # log(1 + exp(margin))
            objective = np.sum(logistic_terms)

        print(f"\033[Kvariable flooring heuristic complete, selected {len(kept_vars)} variables | Total Running Time = {time.time()-start_time} seconds")
        return objective, time.time() - start_time, kept_vars