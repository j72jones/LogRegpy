from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.tree.node import Node
from typing import Callable, List, Tuple


class Bounder:
    """
    Template for evaluating the problem's objective.
    """

    def __init__(self, data: ProblemData, proposed_func: Callable[[ProblemData, Node], Tuple[float, float]]) -> None:
        self.data = data
        self.bounding_func = proposed_func

    def __call__(self, node: Node) -> Tuple[float, float]:
        return self.bounding_func(self.data, node)
    
    def _test_func(proposed_func):
        pass