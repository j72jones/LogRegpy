from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.upper_bounding_func import UpperBounder
from LogRegpy.tree.node import Node
from typing import Optional, Literal
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
import math

class sklearnUpperBounder(UpperBounder):
    """
    Finds a feasible bound with Sklearn as the subsolver
    """

    def __init__(self,
                 data: ProblemData,
                 method: Literal["forward_selection", "backward_elimination", "variable_flooring"]="forward_selection",
                 solver_params: dict = {"penalty": None, "fit_intercept": False}) -> None:
        self.data = data
        self.method = method
        self.model = LogisticRegression(**solver_params)

    def __call__(self, prefixed_in = 0, prefixed_out = 0) -> Node:
        start_time = time.time()
        fixed_in = 0
        fixed_out = 0
        if self.method == "forward_selection":
            prev_coefs = None
            for _ in range(self.data.k):
                print(f"\033[KRunning forward stepwise regression, selected {fixed_in.bit_count()}/{self.data.k} variables | Total Running Time = {time.time()-start_time} seconds", end="\r")
                best_var = -1
                best_coefs = None
                min_objective = math.inf
                for i in Node.varbitset_to_list(Node.universal_varbitset & ~fixed_in):
                    # Train the logistic regression model
                    coefs, objective = self.find_obj(Node.universal_varbitset & ~(fixed_in | i), prev_coefs)
                    # Check if this is the best score at this step
                    if objective < min_objective:
                        best_var = i
                        best_coefs = coefs
                        min_objective = objective
                fixed_in |= Node.var_to_varbitset(best_var)
                prev_coefs = best_coefs
            print(f"\033[KForward stepwise regression complete, selected {fixed_in.bit_count()}/{self.data.k} variables | Total Running Time = {time.time()-start_time} seconds")
            print(f"Chose variables: {Node.varbitset_to_list(fixed_in)}")
        
        elif self.method == "backward_elimination":
            prev_coefs = None
            for j in range(self.data.n - self.data.k):
                print(f"\033[KRunning backward stepwise regression, deselected {fixed_out.bit_count()}/{self.data.n - self.data.k} variables | Total Running Time = {time.time()-start_time} seconds", end="\r")
                best_var = -1
                best_coefs = None
                min_objective = math.inf
                for i in Node.varbitset_to_list(Node.universal_varbitset & ~fixed_out):
                    # Train the logistic regression model
                    coefs, objective = self.find_obj(fixed_out | i, prev_coef=prev_coefs)
                    # Check if this is the best score at this step
                    if objective < min_objective:
                        best_var = i
                        best_coefs = coefs
                        min_objective = objective
                fixed_out |= best_var
                prev_coefs = best_coefs
            print(f"\033[KBackward stepwise regression complete, deselected {fixed_out.bit_count()}/{self.data.n - self.data.k} variables | Total Running Time = {time.time()-start_time} seconds")
            fixed_in = Node.universal_varbitset & ~fixed_out
            print(f"Chose variables: {Node.varbitset_to_list(fixed_in)}")

        elif self.method == "variable_flooring":
            coefs, _ = self.find_obj(0)
            kept_vars = sorted(Node.varbitset_to_list(Node.universal_varbitset), key=lambda k: coefs[k])[:self.data.k]
            fixed_in |= Node.iter_to_varbitset(kept_vars)
            best_coefs, min_objective = self.find_obj(Node.universal_varbitset & ~fixed_in, prev_coef=coefs)

        new_node = Node(fixed_in, fixed_out, lb=min_objective, coefs=best_coefs)
        if not new_node.is_terminal_leaf():
            raise ValueError(f"Node produced is not terminal, fixed in: {Node.varbitset_to_list(new_node.fixed_in)}")
        return new_node
    
    def find_obj(self, fixed_out_varbitset: int, prev_coef=None):
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
