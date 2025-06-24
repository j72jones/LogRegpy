from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.variable_chooser import VariableChooser
from LogRegpy.tree.node import Node
from typing import List, Tuple
from numpy import setdiff1d
import numpy as np
import math
import time

class Fractional(VariableChooser):
    def __init__(self, data: ProblemData, method: str = "most") -> None:
        self.data = data
        self.variances = np.var(self.data.X, axis=0)
        self.variances[self.variances == 0] = 1
        if method not in {"most", "least"}:
            raise ValueError(f"Invalid method: {method}. Must be one of {("forward", "backward")}.")
        self.method = method

    def __call__(self, node: Node) -> Tuple[int, float]:
        start_time = time.time()
        if type(node.coefs) == dict:
            coefs = {i: np.abs(v) for i,v in enumerate(node.coefs.values())}
        else:
            coefs = np.abs(node.coefs)
        j = 0
        if self.method == "most":
            min_coef = math.inf
            min_coef_index = None
            for i in range(Node.n):
                if i not in node.fixed_out:
                    if i not in node.fixed_in:
                        if coefs[j] / self.variances[i] < min_coef:
                            min_coef = coefs[j] / self.variances[i]
                            min_coef_index = i
                    j += 1
            return (min_coef_index, start_time - time.time())
        elif self.method == "least":
            max_coef = -math.inf
            max_coef_index = None
            for i in range(Node.n):
                if i not in node.fixed_out:
                    if i not in node.fixed_in:
                        if coefs[j] / self.variances[i]  > max_coef:
                            max_coef = coefs[j] / self.variances[i]
                            max_coef_index = i
                    j += 1
            return (max_coef_index, start_time - time.time())

    def _test_func(proposed_func):
        pass