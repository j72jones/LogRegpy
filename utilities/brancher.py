from LogRegpy.tree.node import Node
from copy import deepcopy



class Brancher:
    def __init__(self):
        pass

    def branch_node(self, node: Node) -> tuple[list[Node], int, int]:
        pass

    def evaluate_single_node(self, node: Node) -> tuple[float, int]:
        pass