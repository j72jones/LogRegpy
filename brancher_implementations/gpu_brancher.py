from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.brancher import Brancher
from LogRegpy.tree.node import Node
from LogRegpy.bound_algorithms.logregpy_cupy_logistic_solvers import *
import time
import heapq
import numpy as np

class GPUBrancher(Brancher):
    def __init__(self, data: ProblemData, method="fractional_least", penalty=None) -> None:
        self.data = data
        Node.k = self.data.k
        Node.n = self.data.n
        acceptable_methods = {"fractional_least", "fractional_most", "random"}
        if method not in acceptable_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {acceptable_methods}.")
        self.method = method
        self.penalty = penalty

    def branch_node(self, node: Node) -> tuple[list[Node], int, int]:
        branch_vars: list[Node] = [] #starts as tuple[Node, int]
        if type(node.coefs) == dict:
            coefs = {i: np.abs(v) for i,v in enumerate(node.coefs.values())}
        else:
            coefs = np.abs(node.coefs)
        if self.method == "fractional_least":
            for j in Node.varbitset_to_list(Node.universal_varbitset & ~(node.fixed_in | node.fixed_out)):
                varbitset = Node.var_to_varbitset(j)
                if len(branch_vars) < Node.k - node.len_fixed_in:
                    heapq.heappush(branch_vars, (-np.abs(coefs[j]), varbitset))
                elif -np.abs(coefs[j]) < branch_vars[0][0]:
                    heapq.heapreplace(branch_vars, (-np.abs(coefs[j]), varbitset))
            branch = [None] * (len(branch_vars) + 1)
            union_varbitset = 0
            for i in range(len(branch_vars)):
                branch[i] = Node(node.fixed_in, node.fixed_out | branch_vars[i][1])
                branch[i].coefs = node.coefs
                branch[i].coefs[Node.varbitset_to_list(branch_vars[i][1])[0]] = 0
                union_varbitset |= branch[i][1]
            branch.append(Node(node.fixed_in | union_varbitset, 0))
            if not branch[-1].is_terminal_leaf():
                raise ValueError(f"Bottom of branch not terminal. Fixed in {Node.varbitset_to_list(branch[-1].fixed_in)}, fixed out: {Node.varbitset_to_list(branch[-1].fixed_out)}")
            branch = parallel_gd(self.X, self.y, branch)
            union_varbitset = 0
            for i in sorted(range(len(branch)-1), key=branch.__getitem__, reverse=True):
                branch[i].fixed_in |= union_varbitset
                union_varbitset |= branch_vars[i][1]
            return branch, len(branch) - 1, len(branch) - 2
        
        elif self.method == "fractional_most":
            for j in Node.varbitset_to_list(Node.universal_varbitset & ~(node.fixed_in | node.fixed_out)):
                varbitset = Node.var_to_varbitset(j)
                if len(branch_vars) < Node.k - node.len_fixed_in:
                    heapq.heappush(branch_vars, (np.abs(coefs[j]), varbitset))
                elif np.abs(coefs[j]) < branch_vars[0][0]:
                    heapq.heapreplace(branch_vars, (np.abs(coefs[j]), varbitset))
            branch = [None] * (len(branch_vars) + 1)
            union_varbitset = 0
            for i in range(len(branch_vars)):
                branch[i] = Node(node.fixed_in, node.fixed_out | branch_vars[i][1])
                branch[i].coefs = node.coefs
                branch[i].coefs[Node.varbitset_to_list(branch_vars[i][1])[0]] = 0
                union_varbitset |= branch[i][1]
            branch.append(Node(node.fixed_in | union_varbitset, 0))
            if not branch[-1].is_terminal_leaf():
                raise ValueError(f"Bottom of branch not terminal. Fixed in {Node.varbitset_to_list(branch[-1].fixed_in)}, fixed out: {Node.varbitset_to_list(branch[-1].fixed_out)}")
            branch = parallel_gd(self.X, self.y, branch)
            union_varbitset = 0
            for i in sorted(range(len(branch)-1), key=branch.__getitem__, reverse=True):
                branch[i].fixed_in |= union_varbitset
                union_varbitset |= branch_vars[i][1]
            return branch, len(branch) - 1, len(branch) - 2
        
        elif self.method == "random":
            for i in Node.varbitset_to_list(Node.universal_varbitset & ~(node.fixed_in | node.fixed_out)):
                varbitset = Node.var_to_varbitset(i)
                if len(branch_vars) < Node.k - node.len_fixed_in:
                    branch_vars.append(varbitset)
                else: 
                    break
            branch = [None] * (len(branch_vars) + 1)
            union_varbitset = 0
            for i in range(len(branch_vars)):
                branch[i] = Node(node.fixed_in, node.fixed_out | branch_vars[i])
                branch[i].coefs = node.coefs
                branch[i].coefs[Node.varbitset_to_list(branch_vars[i])[0]] = 0
                union_varbitset |= branch[i][1]
            branch.append(Node(node.fixed_in | union_varbitset, 0))
            if not branch[-1].is_terminal_leaf():
                raise ValueError(f"Bottom of branch not terminal. Fixed in {Node.varbitset_to_list(branch[-1].fixed_in)}, fixed out: {Node.varbitset_to_list(branch[-1].fixed_out)}")
            branch = parallel_gd(self.X, self.y, branch)
            union_varbitset = 0
            for i in sorted(range(len(branch)-1), key=branch.__getitem__, reverse=True):
                branch[i].fixed_in |= union_varbitset
                union_varbitset |= branch_vars[i]
            return branch, len(branch) - 1, len(branch) - 2
                 
    
    def evaluate_single_node(self, node, prev_coefs = None):
        node.coefs, obj = single_gd(self.data.X[:, Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out)], self.data.y, warm_start_coefs=prev_coefs)
        return obj
    
    