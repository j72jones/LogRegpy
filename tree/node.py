from typing import List
import math
from typing import Optional, Iterable

class Node:
    k: int
    n: int
    universal_varbitset: int

    @classmethod
    def configure(cls, *, n: int, k: int):
        if not (0 <= k <= n):
            raise ValueError("Require 0 ≤ k ≤ n")

        cls.n = n
        cls.k = k
        cls.universal_varbitset = (1 << n) - 1

    def __init__(self, fixed_in: int, fixed_out: int, lb=math.inf, coefs=None) -> None:
        self.fixed_in: int = fixed_in
        self.fixed_out: int = fixed_out

        self.lb: float = lb
        self.coefs = coefs
    
    def __lt__(self, other):
        return self.lb < other.lb
    
    @property
    def len_fixed_in(self) -> int:
        return self.fixed_in.bit_count()
    @property
    def len_fixed_out(self) -> int:
        return self.fixed_out.bit_count()

    @property
    def fixed_in_full(self) -> bool:
        return self.len_fixed_in >= Node.k
    @property
    def fixed_out_full(self) -> bool:
        return self.len_fixed_out >= Node.n - Node.k
        
    def is_terminal_leaf(self) -> bool:
        if self.fixed_out_full:
            self.fixed_in = Node.universal_varbitset & ~self.fixed_out
            if not self.fixed_in_full:
                raise ValueError(f"Fixed out is overfilled: {self.fixed_out}")
            return True
        elif self.fixed_in_full:
            self.fixed_out = Node.universal_varbitset & ~self.fixed_in
            if not self.fixed_out_full:
                raise ValueError(f"Fixed in is overfilled: {self.fixed_in}")
            return True
        else:
            return False

    def free_varbitset(self) -> int:
        return Node.universal_varbitset & ~ (self.fixed_in | self.fixed_out)

    def __repr__(self):
        return f"Node({repr(self.fixed_in)}, {repr(self.fixed_out)}, lb={repr(self.lb)})"
    
    def to_dict(self):
        if self.lb != math.inf:
            return {"fixed_in": self.fixed_in, "fixed_out": self.fixed_out, "lb": float(self.lb), "coefs": self.coefs}
        else:
            return {"fixed_in": self.fixed_in, "fixed_out": self.fixed_out}
        
    @staticmethod
    def from_dict(passed_dict: dict):
        return Node(passed_dict["fixed_in"], passed_dict["fixed_out"], lb = passed_dict.get("lb", math.inf), coefs = passed_dict.get("coefs", None))
    
    @staticmethod
    def var_to_varbitset(var: int) -> int:
        return (1 << var)

    @staticmethod
    def varbitset_to_list(varbitset: int) -> List:
        out = []
        while varbitset:
            lsb = varbitset & -varbitset
            out.append(lsb.bit_length() - 1)
            varbitset &= varbitset - 1
        return out
    
    @staticmethod
    def iter_to_varbitset(iter: Iterable) -> List:
        out = 0
        for i in iter:
            out += Node.var_to_varbitset(i)
        return out