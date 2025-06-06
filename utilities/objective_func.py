from LogRegpy.utilities.problem_data import ProblemData
from typing import Callable, List, Tuple


class Objective:
    """
    Template for evaluating the problem's objective.
    """

    def __init__(self, data: ProblemData, proposed_func: Callable[[ProblemData, List[int]], Tuple[float, float]]) -> None:
        self.data = data
        self.obj_func = proposed_func

    def __call__(self, fixed_out: List[int]) -> Tuple[float, float]:
        return self.obj_func(self.data, fixed_out)
    
    def _test_func(proposed_func):
        pass