from LogRegpy.tree.node import Node
from copy import deepcopy
from abc import ABC, abstractmethod


class Brancher:
    def __init__(self):
        pass

    @abstractmethod
    def branch_node(self, node: Node) -> tuple[list[Node], int, int]:
        """
        Args:
            node (Node): Node to be branched on
        Returns:
            list[Node]: Exterior nodes of the branch
            int: Number of iterations
            int: Number of internal nodes skipped
        """
        pass

    @abstractmethod
    def evaluate_single_node(self, node: Node) -> float:
        """
        Args:
            node (Node): Node to be evaluated
        """
        pass