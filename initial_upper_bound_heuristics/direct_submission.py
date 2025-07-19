from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.upper_bounding_func import UpperBounder
from typing import List, Tuple
from numpy import setdiff1d
import numpy as np
from sklearn.linear_model import LogisticRegression
from LogRegpy.bound_algorithms.mosek_logistic_regression import logisticRegression
import time
import math



class DirectSubmission(UpperBounder):
    """
    Finds a feasible bound with stepwise regression.
    """

    def __init__(self, data: ProblemData, fixed_in=[], fixed_out=[], solver="MOSEK") -> None:
        self.data = data
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
        self.solver = solver
        if len(fixed_in) == 0 and len(fixed_out) == 0:
            raise ValueError("One of fixed in and fixed out must be nonempty")
        self.fixed_in = fixed_in
        self.fixed_out = fixed_out

    def __call__(self) -> Tuple[float, float, list[int]]:
        start_time = time.time()
        if len(self.fixed_in) != 0:
            var_list = self.fixed_in
        else:
            var_list = setdiff1d(list(range(self.data.n)), self.fixed_out)
        print(f"\033[KTesting provided feasible node | Total Running Time = {time.time()-start_time} seconds", end="\r")
        objective = self.find_obj(var_list)
        print(f"\033[KTesting provided feasible node complete | Total Running Time = {time.time()-start_time} seconds")
        return objective, time.time() - start_time, var_list
    
    def find_obj(self, fixed_in, penalty=None):
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