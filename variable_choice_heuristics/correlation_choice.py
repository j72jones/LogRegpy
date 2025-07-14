from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.variable_chooser import VariableChooser
from LogRegpy.tree.node import Node
from typing import List, Tuple
from numpy import setdiff1d
import numpy as np
import math
import time

class CorrelationChoice(VariableChooser):
    def __init__(self, data: ProblemData, method: str = "most") -> None:
        self.data = data
        if method not in {"most", "least"}:
            raise ValueError(f"Invalid method: {method}. Must be one of {("most", "least")}.")
        self.method = method
        self.corr_matrix =  np.nan_to_num(np.abs(np.corrcoef(self.data.X, rowvar=False)), nan=0)

    def __call__(self, node: Node) -> Tuple[int, float]:
        start_time = time.time()
        considered_nodes = setdiff1d(range(Node.n), node.fixed_out)
        if self.method == "most":
            max_corr = -math.inf
            max_corr_index = None
            for i in setdiff1d(range(Node.n), node.fixed_in + node.fixed_out):
                mean_corr = self.corr_matrix[considered_nodes, i].mean()
                if mean_corr > max_corr:
                    max_corr = mean_corr
                    max_corr_index = i
            return (max_corr_index, start_time - time.time())
        elif self.method == "least":
            min_corr = math.inf
            min_corr_index = None
            for i in setdiff1d(range(Node.n), node.fixed_in + node.fixed_out):
                mean_corr = self.corr_matrix[considered_nodes, i].mean()
                if mean_corr < min_corr:
                    min_corr = mean_corr
                    min_corr_index = i
            return (min_corr_index, start_time - time.time())
        

    def _test_func(proposed_func):
        pass