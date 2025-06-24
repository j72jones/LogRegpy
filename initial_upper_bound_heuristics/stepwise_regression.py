from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.upper_bounding_func import UpperBounder
from LogRegpy.tree.node import Node
from typing import List, Tuple
from numpy import setdiff1d
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import time
import math
from copy import deepcopy



class StepwiseRegression(UpperBounder):
    """
    Finds a feasible bound with stepwise regression.
    """

    def __init__(self, data: ProblemData, method="forward") -> None:
        self.data = data
        if method not in {"forward", "backward"}:
            raise ValueError(f"Invalid method: {method}. Must be one of {("forward", "backward")}.")
        self.method = method

    def __call__(self) -> Tuple[float, float, list[int]]:
        start_time = time.time()
        model = LogisticRegression(penalty=None, fit_intercept=False)
        if self.method == "forward":
            var_list = []
            for j in range(self.data.k):
                print(f"\033[KRunning forward stepwise regression, selected {len(var_list)} variables | Total Running Time = {time.time()-start_time} seconds", end="\r")
                best_var = None
                min_objective = math.inf
                for i in setdiff1d(list(range(self.data.n)), var_list):
                    # Train the logistic regression model
                    x = self.data.X[:, var_list + [i]]
                    model.fit(x, self.data.y)
                    # Calculate scores
                    theta = model.coef_.flatten()
                    margin = (1 - 2 * self.data.y) * (x @ theta)  # (n,)
                    logistic_terms = np.log1p(np.exp(margin))  # log(1 + exp(margin))
                    objective = np.sum(logistic_terms)
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
                    x = self.data.X[:, setdiff1d(var_list, [i])]
                    model.fit(x, self.data.y)
                    # Calculate scores
                    theta = model.coef_.flatten()
                    margin = (1 - 2 * self.data.y) * (x @ theta)  # (n,)
                    logistic_terms = np.log1p(np.exp(margin))  # log(1 + exp(margin))
                    objective = np.sum(logistic_terms)
                    # Check if this is the best score at this step
                    if objective < min_objective:
                        best_var = i
                        min_objective = objective
                var_list.remove(best_var)
            print(f"\033[KBackward stepwise regression complete, selected {len(var_list)} variables | Total Running Time = {time.time()-start_time} seconds")
        return min_objective, time.time() - start_time, var_list