from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.upper_bounding_func import UpperBounder
from typing import List, Tuple
from numpy import setdiff1d
import numpy as np
from sklearn.linear_model import LogisticRegression
from LogRegpy.bound_algorithms.mosek_logistic_regression import logisticRegression
import time
import math



class StepwiseRegression(UpperBounder):
    """
    Finds a feasible bound with stepwise regression.
    """

    def __init__(self, data: ProblemData, method="forward", solver="saga", penalty=None) -> None:
        self.data = data
        if method not in {"forward", "backward"}:
            raise ValueError(f"Invalid method: {method}. Must be one of {("forward", "backward")}.")
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
        self.solver = solver
        self.method = method
        self.penalty = penalty

    def __call__(self) -> Tuple[float, float, list[int]]:
        start_time = time.time()
        
        if self.method == "forward":
            var_list = []
            for j in range(self.data.k):
                print(f"\033[KRunning forward stepwise regression, selected {len(var_list)} variables | Total Running Time = {time.time()-start_time} seconds", end="\r")
                best_var = None
                min_objective = math.inf
                for i in setdiff1d(list(range(self.data.n)), var_list):
                    # Train the logistic regression model
                    objective = self.find_obj(var_list + [i], self.penalty)
                    # Check if this is the best score at this step
                    if objective < min_objective:
                        best_var = i
                        min_objective = objective
                var_list.append(best_var)
            print(f"\033[KForward stepwise regression complete, selected {len(var_list)} variables | Total Running Time = {time.time()-start_time} seconds")
            print(f"Chose variables: {sorted(var_list)}")
        elif self.method == "backward":
            var_list = list(range(self.data.n))
            for j in range(self.data.n - self.data.k):
                print(f"\033[KRunning backward stepwise regression, selected {len(var_list)} variables | Total Running Time = {time.time()-start_time} seconds", end="\r")
                best_var = None
                min_objective = math.inf
                for i in var_list:
                    # Train the logistic regression model
                    objective = self.find_obj(setdiff1d(var_list, [i]), self.penalty)
                    # Check if this is the best score at this step
                    if objective < min_objective:
                        best_var = i
                        min_objective = objective
                var_list.remove(best_var)
            print(f"\033[KBackward stepwise regression complete, selected {len(var_list)} variables | Total Running Time = {time.time()-start_time} seconds")
        if self.penalty is not None:
            min_objective = self.find_obj(var_list, None)
        return min_objective, time.time() - start_time, var_list
    
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