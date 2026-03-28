from LogRegpy.utilities.problem_data import ProblemData
from LogRegpy.utilities.brancher import Brancher
from LogRegpy.tree.node import Node
from LogRegpy.bound_algorithms.logregpy_cupy_logistic_solvers import *
from typing import Literal
import heapq
import cupy as cp

class GPUBrancher(Brancher):
    def __init__(self,
                 data: ProblemData,
                 method: Literal["largest_coefficient", "smallest_coefficient", "random", "strong_branching"] ="strong_branching",
                 lamb: float =0) -> None:
        self.data = data
        Node.k = self.data.k
        Node.n = self.data.n
        self.method = method
        self.lamb = lamb

    def branch_node(self, node: Node) -> tuple[list[Node], int, int]:
        branch = [] #starts as tuple[Node, int]
        if self.method == "smallest_coefficient":
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
            
        elif self.method == "largest_coefficient":
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
            
        elif self.method == "random":
            for i in Node.varbitset_to_list(Node.universal_varbitset & ~ (node.fixed_out | node.fixed_in)):
                if len(branch) < Node.k - node.len_fixed_in:
                    varbitset = Node.var_to_varbitset(i)
                    new_node = Node(node.fixed_in, node.fixed_out | varbitset)
                    new_node.lb = self.find_obj(new_node, prev_coefs=node.coefs)
                    branch.append((new_node, varbitset))
                else:
                    break
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

        elif self.method == "strong_branching":
            temp_branch = []
            for i in Node.varbitset_to_list(Node.universal_varbitset & ~ (node.fixed_out | node.fixed_in)):
                varbitset = Node.var_to_varbitset(i)
                new_node = Node(node.fixed_in, node.fixed_out | varbitset, coefs=node.coefs)
                new_node.is_terminal_leaf()
                branch.append((new_node, varbitset))
                temp_branch.append(new_node)
            num_converged = parallel_gd2_gpu_kernel(self.data.X, self.data.y, temp_branch, self.lamb, epochs=5000, verbose=False)
            if num_converged < len(branch):
                print(f"Convergence 1 flag: {num_converged}/{len(branch)}")
            branch = sorted(branch, reverse=True)[:Node.k - node.len_fixed_in]
            union_varbitset = 0
            for i in range(len(branch)):
                branch[i][0].fixed_in |= union_varbitset
                union_varbitset |= branch[i][1]
                branch[i] = branch[i][0]
            branch.append(Node(node.fixed_in | union_varbitset, 0))
            if not branch[-1].is_terminal_leaf():
                raise ValueError(f"Bottom of branch not terminal. Fixed in {Node.varbitset_to_list(branch[-1].fixed_in)}, fixed out: {Node.varbitset_to_list(branch[-1].fixed_out)}")
            self.evaluate_single_node(branch[-1], prev_coefs=node.coefs[Node.varbitset_to_list(Node.universal_varbitset & ~branch[-1].fixed_out)])
            # branch[-1] = self.local_search(branch[-1])
        
        return branch, len(branch) - 1, len(branch) - 2
                                  
    
    def evaluate_single_node(self, node, prev_coefs = None):

        node.coefs, node.lb = single_gd(
            self.data.X[:, Node.varbitset_to_list(Node.universal_varbitset & ~node.fixed_out)],
            self.data.y,
            lamb=self.lamb,
            warm_start_coefs=prev_coefs,
            epochs = 5000
            )
   
    def local_search(self, feasible_node):
        local_nodes = [feasible_node]
        for i in Node.varbitset_to_list(feasible_node.fixed_in):
            for j in Node.varbitset_to_list(Node.universal_varbitset & ~ feasible_node.fixed_in):
                new_node = Node((feasible_node.fixed_in & ~ (1 << i)) | (1 << j), 0, coefs=feasible_node.coefs)
                if not new_node.is_terminal_leaf():
                    print("INFEASIBLE NODE IN LOCAL SEARCH")
                local_nodes.append(new_node)
        num_converged = parallel_gd2_gpu_kernel(self.data.X, self.data.y, local_nodes, self.lamb, epochs=5000, verbose=False)
        if num_converged < len(local_nodes):
                print(f"Convergence 2 flag: {num_converged}/{len(local_nodes)}")
        return min(local_nodes)