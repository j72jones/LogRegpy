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

class CleverGreedy(VariableChooser):
    def __init__(self, data: ProblemData, method="min", penalty="l2", solver="saga") -> None:
        self.data = data
        acceptable_methods = {"min", "max"}
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

    def __call__(self, node: Node) -> Tuple[int, float]:
        start_time = time.time()
        infeasible_node_list = []
        feasible_node_list = []
        if Node.n - Node.k - len(node.fixed_out) > 1:
            for i in setdiff1d(list(range(self.data.n)), node.fixed_in + node.fixed_out):
                new_node = Node(node.fixed_in, node.fixed_out + [i])
                new_node.lb = self.find_obj(setdiff1d(list(range(self.data.n)), node.fixed_out + [i]), penalty = self.penalty)
                infeasible_node_list.append((new_node,i))
            additional_fixed_in = []
            infeasible_node_list = sorted(infeasible_node_list, key=lambda tup: tup[0], reverse=True)[:Node.k - len(node.fixed_in)]
            for new_tup in infeasible_node_list:
                new_tup[0].fixed_out.extend(additional_fixed_in)
                additional_fixed_in.extend(new_tup[1])
            feasible_node = Node(additional_fixed_in, [])
            if not feasible_node.is_terminal_leaf():
                raise ValueError(f"feasible node with fixed in length: {len(feasible_node.fixed_in)} is being called infeasible")
            feasible_node.lb = self.find_obj(additional_fixed_in, penalty = self.penalty)
            feasible_node_list.append((feasible_node, additional_fixed_in))
            iter = self.data.n - len(node.fixed_in) - len(node.fixed_out) + 1
        elif Node.k - len(node.fixed_in) == 1:
            for i in setdiff1d(list(range(self.data.n)), node.fixed_in + node.fixed_out):
                new_node = Node(node.fixed_in, node.fixed_out + [i])
                new_node.lb = self.find_obj(setdiff1d(list(range(self.data.n)), node.fixed_out + [i]), penalty = self.penalty)
                infeasible_node_list.append(new_node)
            iter = self.data.n - len(node.fixed_in) - len(node.fixed_out)
        else:
            best_obj = math.inf
            best_var = None
            for i in setdiff1d(list(range(self.data.n)), node.fixed_in + node.fixed_out):
                obj = self.find_obj(setdiff1d(list(range(self.data.n)), node.fixed_out + [i]), penalty = self.penalty)
                if obj < best_obj:
                    best_var = i
            feasible_node = Node(node.fixed_in, node.fixed_out + [best_var])
            feasible_node.lb = best_obj
            feasible_node_list.append(feasible_node)
            infeasible_node = Node(node.fixed_in +[best_var], node.fixed_out)
            infeasible_node.lb = self.find_obj(infeasible_node.fixed_in, penalty=self.penalty)
            infeasible_node_list.append(infeasible_node)

            iter = self.data.n - len(node.fixed_in) - len(node.fixed_out) + 1
        return (infeasible_node_list, feasible_node_list, iter, start_time - time.time())

    def find_obj(self, fixed_in, penalty):
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