from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.brancher import Brancher
from LogRegpy.tree.node import Node
from sklearn.linear_model import LogisticRegression
import time
import heapq
import numpy as np

class SklearnBrancher(Brancher):
    def __init__(self, data: ProblemData, method="fractional_least", penalty=None) -> None:
        self.data = data
        Node.k = self.data.k
        Node.n = self.data.n
        acceptable_methods = {"fractional_least", "fractional_most", "random"}
        if method not in acceptable_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {acceptable_methods}.")
        self.method = method
        self.penalty = penalty
        self.model = LogisticRegression(penalty=None, fit_intercept=False, **self.params)

    def __call__(self, node: Node) -> tuple[list[Node], int, int]:
        start_time = time.time()
        branch: list[Node] = [] #starts as tuple[Node, int]
        if type(node.coefs) == dict:
            coefs = {i: np.abs(v) for i,v in enumerate(node.coefs.values())}
        else:
            coefs = np.abs(node.coefs)
        branch = []
        if self.method == "fractional_least":
            j = 0
            for i in range(Node.n):
                varbitset = Node.var_to_varbitset(i)
                if varbitset & node.fixed_out:
                    if varbitset & node.fixed_in:
                        if len(branch) < Node.k - len(node.fixed_in):
                            heapq.heappush(branch, (-np.abs(coefs[j]), varbitset))
                        elif -np.abs(coefs[j]) < branch[0][0]:
                            heapq.heapreplace(branch, (-np.abs(coefs[j]), varbitset))
                    j += 1
            for i in range(len(branch)):
                branch[i][0] = Node(node.fixed_in, node.fixed_out | branch[i][1])
                branch[i][0].lb = self.evaluate_single_node(branch[i][0], prev_coefs=node.coefs)
            branch.sort()
            union_varbitset = 0
            for i in range(len(branch)):
                branch[i][0].fixed_in |= union_varbitset
                union_varbitset |= branch[i][1]
                branch[i] = branch[i][0]
            branch.append(Node(node.fixed_in | union_varbitset, []))
            if not branch[-1].is_terminal_leaf:
                raise ValueError(f"Bottom of branch not terminal. Fixed in {Node.varbitset_to_list(branch[-1].fixed_in)}, fixed out: {Node.varbitset_to_list(branch[-1].fixed_out)}")
            branch[-1].lb = self.evaluate_single_node(branch[-1], prev_coefs=node.coefs)
            return branch, len(branch) - 1, len(branch) - 2
        
        elif self.method == "fractional_most":
            j = 0
            for i in range(Node.n):
                varbitset = Node.var_to_varbitset(i)
                if varbitset & node.fixed_out:
                    if varbitset & node.fixed_in:
                        if len(branch) < Node.k - len(node.fixed_in):
                            heapq.heappush(branch, (np.abs(coefs[j]), varbitset))
                        elif np.abs(coefs[j]) < branch[0][0]:
                            heapq.heapreplace(branch, (np.abs(coefs[j]), varbitset))
                    j += 1
            for i in range(len(branch)):
                branch[i][0] = Node(node.fixed_in, node.fixed_out | branch[i][1])
                branch[i][0].lb = self.evaluate_single_node(branch[i][0], prev_coefs=node.coefs)
            branch.sort()
            union_varbitset = 0
            for i in range(len(branch)):
                branch[i][0].fixed_in |= union_varbitset
                union_varbitset |= branch[i][1]
                branch[i] = branch[i][0]
            branch.append(Node(node.fixed_in | union_varbitset, []))
            if not branch[-1].is_terminal_leaf:
                raise ValueError(f"Bottom of branch not terminal. Fixed in {Node.varbitset_to_list(branch[-1].fixed_in)}, fixed out: {Node.varbitset_to_list(branch[-1].fixed_out)}")
            branch[-1].lb = self.evaluate_single_node(branch[-1], prev_coefs=node.coefs)
            return branch, len(branch) - 1, len(branch) - 2
        
        elif self.method == "random":
            for i in range(Node.n):
                varbitset = Node.var_to_varbitset(i)
                if varbitset & node.fixed_out and varbitset & node.fixed_in:
                    if len(branch) < Node.k - len(node.fixed_in):
                        branch.append(varbitset)
                    else:
                        break
            for i in range(len(branch)):
                branch[i] = (Node(node.fixed_in, node.fixed_out | branch[i]), branch[i])
                branch[i][0].lb = self.evaluate_single_node(branch[i][0], prev_coefs=node.coefs)
            branch.sort()
            union_varbitset = 0
            for i in range(len(branch)):
                branch[i][0].fixed_in |= union_varbitset
                union_varbitset |= branch[i][1]
                branch[i] = branch[i][0]
            branch.append(Node(node.fixed_in | union_varbitset, []))
            if not branch[-1].is_terminal_leaf:
                raise ValueError(f"Bottom of branch not terminal. Fixed in {Node.varbitset_to_list(branch[-1].fixed_in)}, fixed out: {Node.varbitset_to_list(branch[-1].fixed_out)}")
            branch[-1].lb = self.evaluate_single_node(branch[-1], prev_coefs=node.coefs)
            return branch, len(branch) - 1, len(branch) - 2
                 
  
    def evaluate_single_node(self, node, prev_coefs = None):
        attributes = Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out)
        # Train the logistic regression model
        x = self.data.X[:, attributes]
        self.model.fit(x, self.data.y)
        node.coefs = self.model.coef_.flatten()

        margin = (1 - 2 * self.data.y) * (x @ node.coefs)  # (n,)
        logistic_terms = np.log1p(np.exp(margin))  # log(1 + exp(margin))
        logistic_loss = np.sum(logistic_terms)

        # Calculate score
        return logistic_loss