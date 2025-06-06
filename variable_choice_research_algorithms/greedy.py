from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.variable_chooser import VariableChooser
from LogRegpy.tree.node import Node
from typing import List, Tuple
from numpy import setdiff1d
import math
import time

class Greedy(VariableChooser):
    def __init__(self, data: ProblemData) -> None:
        self.data = data

    def __call__(self, node: Node) -> Tuple[int, float]:
        start_time = time.time()
        coefs = node.model.coef_.flatten()
        j = 0
        max_coef = -math.inf
        max_coef_index = None
        for i in range(Node.n):
            if i not in node.fixed_out:
                if i not in node.fixed_in:
                    if coefs[j] > max_coef:
                        max_coef = coefs[j]
                        max_coef_index = i
                j += 1
        return (max_coef_index, start_time - time.time())

    def _test_func(proposed_func):
        pass