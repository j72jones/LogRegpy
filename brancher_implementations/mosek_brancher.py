from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.brancher import Brancher
from LogRegpy.tree.node import Node
from LogRegpy.bound_algorithms.mosek_logistic_model import MosekLogisticModel
from LogRegpy.bound_algorithms.mosek_logistic_regression import logisticRegression
import heapq

class MosekBrancher(Brancher):
    def __init__(self, data: ProblemData, method="fractional_least", penalty=None) -> None:
        self.data = data
        Node.k = self.data.k
        Node.n = self.data.n
        acceptable_methods = {"fractional_least", "fractional_most", "random"}
        if method not in acceptable_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {acceptable_methods}.")
        self.method = method
        self.penalty = penalty
        self.first_run = True
        self.model = None

    def branch_node(self, node: Node) -> tuple[list[Node], int, int]:
        branch: list[Node] = [] #starts as tuple[Node, int]
        if type(node.coefs) == dict:
            coefs = {i: abs(v) for i,v in enumerate(node.coefs.values())}
        else:
            coefs = abs(node.coefs)
        branch = []
        if self.method == "fractional_least":
            j = 0
            for i in range(Node.n):
                varbitset = Node.var_to_varbitset(i)
                if not (varbitset & node.fixed_out):
                    if not (varbitset & node.fixed_in):
                        if len(branch) < Node.k - node.len_fixed_in:
                            heapq.heappush(branch, (-abs(coefs[j]), varbitset))
                        elif -abs(coefs[j]) < branch[0][0]:
                            heapq.heapreplace(branch, (-abs(coefs[j]), varbitset))
                    j += 1
            for i in range(len(branch)):
                branch[i] = (Node(node.fixed_in, node.fixed_out | branch[i][1]), branch[i][1])
                branch[i][0].lb = self.find_obj(branch[i][0], prev_coefs=node.coefs)
            branch.sort(reverse=True)
            union_varbitset = 0
            for i in range(len(branch)):
                branch[i][0].fixed_in |= union_varbitset
                union_varbitset |= branch[i][1]
                branch[i] = branch[i][0]
            branch.append(Node(node.fixed_in | union_varbitset, 0))
            if not branch[-1].is_terminal_leaf():
                raise ValueError(f"Bottom of branch not terminal. Fixed in {Node.varbitset_to_list(branch[-1].fixed_in)}, fixed out: {Node.varbitset_to_list(branch[-1].fixed_out)}")
            branch[-1].lb = self.find_obj(branch[-1], prev_coefs=node.coefs)
            return branch, len(branch) - 1, len(branch) - 2
        
        elif self.method == "fractional_most":
            j = 0
            for i in range(Node.n):
                varbitset = Node.var_to_varbitset(i)
                if not (varbitset & node.fixed_out):
                    if not (varbitset & node.fixed_in):
                        if len(branch) < Node.k - node.len_fixed_in:
                            heapq.heappush(branch, (abs(coefs[j]), varbitset))
                        elif abs(coefs[j]) < branch[0][0]:
                            heapq.heapreplace(branch, (abs(coefs[j]), varbitset))
                    j += 1
            for i in range(len(branch)):
                branch[i] = (Node(node.fixed_in, node.fixed_out | branch[i][1]), branch[i][1])
                branch[i][0].lb = self.find_obj(branch[i][0], prev_coefs=node.coefs)
            branch.sort(reverse=True)
            union_varbitset = 0
            for i in range(len(branch)):
                branch[i][0].fixed_in |= union_varbitset
                union_varbitset |= branch[i][1]
                branch[i] = branch[i][0]
            branch.append(Node(node.fixed_in | union_varbitset, 0))
            if not branch[-1].is_terminal_leaf():
                raise ValueError(f"Bottom of branch not terminal. Fixed in {Node.varbitset_to_list(branch[-1].fixed_in)}, fixed out: {Node.varbitset_to_list(branch[-1].fixed_out)}")
            branch[-1].lb = self.find_obj(branch[-1], prev_coefs=node.coefs)
            return branch, len(branch) - 1, len(branch) - 2
        
        elif self.method == "random":
            for i in range(Node.n):
                varbitset = Node.var_to_varbitset(i)
                if not (varbitset & node.fixed_out) and not (varbitset & node.fixed_in):
                    if len(branch) < Node.k - node.len_fixed_in:
                        branch.append(varbitset)
                    else:
                        break
            for i in range(len(branch)):
                branch[i] = (Node(node.fixed_in, node.fixed_out | branch[i]), branch[i])
                branch[i][0].lb = self.find_obj(branch[i][0], prev_coefs=node.coefs)
            branch.sort(reverse=True)
            union_varbitset = 0
            for i in range(len(branch)):
                branch[i][0].fixed_in |= union_varbitset
                union_varbitset |= branch[i][1]
                branch[i] = branch[i][0]
            branch.append(Node(node.fixed_in | union_varbitset, 0))
            if not branch[-1].is_terminal_leaf():
                raise ValueError(f"Bottom of branch not terminal. Fixed in {Node.varbitset_to_list(branch[-1].fixed_in)}, fixed out: {Node.varbitset_to_list(branch[-1].fixed_out)}")
            branch[-1].lb = self.find_obj(branch[-1], prev_coefs=node.coefs)
            return branch, len(branch) - 1, len(branch) - 2
                 

    def find_obj(self, node: Node, prev_coefs = None):
        if self.first_run:
            self.first_run = False
            self.model = MosekLogisticModel(self.data.X, self.data.y, lamb=0 if self.penalty is None else self.penalty)
        coefs, obj = self.model.solve(Node.varbitset_to_list(node.fixed_out), prev_coef=prev_coefs)
        node.coefs = {i: coefs[i] for i in Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out)}
        return obj
    
    def evaluate_single_node(self, node, prev_coefs = None):
        coefs, obj = logisticRegression(self.data.X[:, Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out)], self.data.y)
        node.coefs = {i: coefs[i] for i in Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out)}
        return obj