from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.upper_bounding_func import UpperBounder
from LogRegpy.tree.node import Node
from typing import Optional
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
import math

class MOSEKUpperBounder(UpperBounder):
    """
    Finds a feasible bound with MOSEK as the subsolver
    """

    def __init__(self, data: ProblemData, method="forward_selection", lamb: Optional[float]=0) -> None:
        self.data = data
        acceptable_methods = ("forward_selection", "backward_elimination", "variable_flooring", "direct_submission")
        if method not in acceptable_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {acceptable_methods}.")
        self.method = method
        self.lamb = lamb
        self.model = LogisticRegression()

    def __call__(self, prefixed_in = 0, prefixed_out = 0) -> Node:
        start_time = time.time()
        
        if self.method == "forward_selection":
            prev_coefs = None
            for _ in range(self.data.k - prefixed_in.bit_count()):
                print(f"\033[KRunning forward stepwise regression, selected {prefixed_in.bit_count()}/{self.data.k} variables | Total Running Time = {time.time()-start_time} seconds", end="\r")
                best_var = None
                best_coefs = None
                min_objective = math.inf
                for i in Node.varbitset_to_list(Node.universal_varbitset & ~(prefixed_in | prefixed_out)):
                    # Train the logistic regression model
                    coefs, objective = self.find_obj(Node.universal_varbitset & ~(prefixed_in | i), prev_coefs)
                    # Check if this is the best score at this step
                    if objective < min_objective:
                        best_var = i
                        best_coefs = coefs
                        min_objective = objective
                prefixed_in |= best_var
                prev_coefs = best_coefs
            print(f"\033[KForward stepwise regression complete, selected {prefixed_in.bit_count()}/{self.data.k} variables | Total Running Time = {time.time()-start_time} seconds")
            print(f"Chose variables: {Node.varbitset_to_list(prefixed_in)}")
        
        elif self.method == "backward_elimination":
            for j in range(self.data.n - self.data.k):
                print(f"\033[KRunning backward stepwise regression, deselected {prefixed_out.bit_count()}/{self.data.n - self.data.k} variables | Total Running Time = {time.time()-start_time} seconds", end="\r")
                best_var = None
                min_objective = math.inf
                for i in Node.varbitset_to_list(Node.universal_varbitset & ~(prefixed_out | prefixed_in)):
                    # Train the logistic regression model
                    objective = self.find_obj(prefixed_out | i, self.penalty)
                    # Check if this is the best score at this step
                    if objective < min_objective:
                        best_var = i
                        min_objective = objective
                prefixed_out |= best_var
            print(f"\033[KBackward stepwise regression complete, deselected {prefixed_out.bit_count()}/{self.data.n - self.data.k} variables | Total Running Time = {time.time()-start_time} seconds")
            prefixed_in = Node.universal_varbitset & ~prefixed_out
            print(f"Chose variables: {Node.varbitset_to_list(prefixed_in)}")

        elif self.method == "variable_flooring":
            coefs, _ = self.find_obj(0)
            kept_vars = sorted(Node.varbitset_to_list(Node.universal_varbitset & ~(prefixed_out | prefixed_in)), key=lambda k: coefs[k])[:self.data.k - prefixed_in.bit_count()]
            prefixed_in |= Node.iter_to_varbitset(kept_vars)
            best_coefs, min_objective = self.find_obj(Node.universal_varbitset & ~prefixed_in, prev_coef=coefs)

        elif self.method == "direct_submission":
            start_time = time.time()
            prefixed_out |= Node.universal_varbitset & ~prefixed_in
            print(f"\033[KTesting provided feasible node | Total Running Time = {time.time()-start_time} seconds", end="\r")
            objective = self.find_obj(prefixed_out)
            print(f"\033[KTesting provided feasible node complete | Total Running Time = {time.time()-start_time} seconds")

        new_node = Node(prefixed_in, prefixed_out, lb=min_objective, coefs=best_coefs)
        if not new_node.is_terminal_leaf():
            raise ValueError(f"Node produced is not terminal, fixed in: {Node.varbitset_to_list(new_node.fixed_in)}")
        return new_node
    
    def find_obj(self, fixed_out_varbitset: int, prev_coef:list[float]=None) -> tuple[list[float], int]:
        attributes = Node.varbitset_to_list(Node.universal_varbitset & ~fixed_out_varbitset)
        # Train the logistic regression model
        x = self.data.X[:, attributes]
        self.model.fit(x, self.data.y)
        coefs = self.model.coef_.flatten()

        margin = (1 - 2 * self.data.y) * (x @ coefs)  # (n,)
        logistic_terms = np.log1p(np.exp(margin))  # log(1 + exp(margin))
        logistic_loss = np.sum(logistic_terms)

        # Calculate score
        return coefs, logistic_loss
