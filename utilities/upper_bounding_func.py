from LogRegpy.utilities.problem_data import ProblemData
from typing import Callable, List, Tuple
from abc import ABC, abstractmethod
from LogRegpy.tree.node import Node

class UpperBounder(ABC):
    """
    Template for retrieving an initial UB.
    """
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self) -> Node:
        pass