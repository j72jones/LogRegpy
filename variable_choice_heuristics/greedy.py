from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.variable_chooser import VariableChooser
from LogRegpy.tree.node import Node
from typing import List, Tuple
from numpy import setdiff1d
from sklearn.linear_model import LogisticRegression
from LogRegpy.bound_algorithms.mosek_logistic_regression import logisticRegression
import math
import time
import numpy as np

class Greedy(VariableChooser):
    def __init__(self, data: ProblemData, method="min", penalty="l2", solver="saga", writer=None) -> None:
        self.data = data
        acceptable_methods = {"min_in", "max_in", "min_out", "max_out"}
        if method not in acceptable_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {acceptable_methods}.")
        self.method = method
        acceptable_solvers = {"MOSEK", "lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"}
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

        self.objectives: dict[frozenset, int] = {}
        self.writer=writer
        self.num_iter = 0

    def __call__(self, node: Node) -> Tuple[int, float]:
        start_time = time.time()
        best_var = None
        if self.method == "min_out":
            min_objective = math.inf
            for i in setdiff1d(list(range(self.data.n)), node.fixed_in + node.fixed_out):
                objective = self.find_obj(setdiff1d(list(range(self.data.n)), node.fixed_out + [i]), penalty = self.penalty)
                if objective < min_objective:
                    min_objective = objective
                    best_var = i
        elif self.method == "max_out":
            max_objective = -math.inf
            for i in setdiff1d(list(range(self.data.n)), node.fixed_in + node.fixed_out):
                objective = self.find_obj(setdiff1d(list(range(self.data.n)), node.fixed_out + [i]), penalty = self.penalty)
                if objective > max_objective:
                    max_objective = objective
                    best_var = i
        if self.method == "min_in":
            min_objective = math.inf
            for i in setdiff1d(list(range(self.data.n)), node.fixed_in + node.fixed_out):
                current_set = frozenset(node.fixed_in + [i])
                if current_set not in self.objectives.keys():
                    self.objectives[current_set] = self.find_obj(node.fixed_in  + [i], penalty = self.penalty)
                if self.objectives[current_set] < min_objective:
                    min_objective = self.objectives[current_set]
                    best_var = i
            del self.objectives[frozenset(node.fixed_in + [i])]
        elif self.method == "max_in":
            max_objective = -math.inf
            for i in setdiff1d(list(range(self.data.n)), node.fixed_in + node.fixed_out):
                objective = self.find_obj(node.fixed_in  + [i], penalty = self.penalty)
                if objective > max_objective:
                    max_objective = objective
                    best_var = i
        if self.writer is not None:
            with open(self.writer, mode='a', newline='') as f:
                f.write(f"{self.num_iter}\n")
        return (best_var, start_time - time.time())

    def find_obj(self, fixed_in, penalty):
        self.num_iter += 1
        if self.solver == "MOSEK":
            x = self.data.X[:, fixed_in]
            _, obj = logisticRegression(x, self.data.y, lamb=1 if penalty == "l2" else 0)
            return obj
        else:
            model = LogisticRegression(penalty=penalty, fit_intercept=False, solver=self.solver)
            x = self.data.X[:, fixed_in]
            model.fit(x, self.data.y)
            # Calculate scores
            theta = model.coef_.flatten()
            margin = (1 - 2 * self.data.y) * (x @ theta)  # (n,)
            logistic_terms = np.log1p(np.exp(margin))  # log(1 + exp(margin))
            return np.sum(logistic_terms)