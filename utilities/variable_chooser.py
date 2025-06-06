from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.tree.node import Node
from typing import List, Tuple
from numpy import setdiff1d
import time

class VariableChooser:
    """
    Template for selecting the next variable.
    """

    def __init__(self, data: ProblemData) -> None:
        self.data = data

    def __call__(self, node: Node) -> Tuple[int, float]:
        start_time = time.time()
        for i in range(Node.n):
            if i not in node.fixed_in and i not in node.fixed_out:
                return (i, time.time() - start_time)
    
    def _test_func(proposed_func):
        pass