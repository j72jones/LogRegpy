from LogRegpy.utilities.problem_data import ProblemData
from typing import Callable, List, Tuple


class UpperBounder:
    """
    Template for retrieving an initial UB.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, data: ProblemData) -> Tuple[float, float, list[int]]:
        pass
    
    def _test_func(proposed_func):
        pass