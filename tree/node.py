from typing import List
import math
from functools import cached_property
from numpy import setdiff1d, ndarray
from typing import Optional

class Node:
    num_instances: int = 0

    k: int
    n: int

    def __init__(self, fixed_in: List[int], fixed_out: List[int]) -> None:
        Node.num_instances += 1
        self.num_node = Node.num_instances # make sure this does what you think it does. TODO:unittest

        self.fixed_in: List[int] = fixed_in
        self.fixed_out: List[int] = fixed_out

        self._lb: float = math.inf # also is obj_val if this is a terminal leaf node
        self.coefs = None

    def __eq__(self, other):
        return self._lb == other._lb
    
    def __lt__(self, other):
        return self._lb < other._lb
    
    @property
    def lb(self):
        return self._lb
    
    @lb.setter
    def lb(self, value):
        self._lb = value

    # from here on I could use all cached properties...not of critical importance
    
    @property
    def fixed_in_full(self) -> bool:
        return len(self.fixed_in) >= Node.k
    
    @property
    def fixed_out_full(self) -> bool:
        return len(self.fixed_in) == Node.n - Node.k

    # @property
    # def is_feasible(self) -> bool:
    #     if len(self.fixed_out) >= Node.n - Node.k:
    #         return True
    #     else:
    #         return False
        
    @property
    def is_terminal_leaf(self) -> bool:
        if len(self.fixed_out) == Node.n - Node.k:
            self.fixed_in = setdiff1d(range(Node.n), self.fixed_out)
            return True
        elif len(self.fixed_in) == Node.k:
            self.fixed_out = setdiff1d(range(Node.n), self.fixed_in)
            return True
        else:
            return False