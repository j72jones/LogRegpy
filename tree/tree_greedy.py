from enum import Enum
from typing import List, Optional
from numbers import Number
from LogRegpy.utilities.bounding_func import Bounder
from LogRegpy.utilities.objective_func import Objective
from LogRegpy.utilities.variable_chooser import VariableChooser
from LogRegpy.utilities.upper_bounding_func import UpperBounder
from LogRegpy.tests.test_logger import TestLogger
from LogRegpy.tree.node import Node
from copy import deepcopy
from numpy import argmax, argmin
import math
import time
from numpy import (setdiff1d)

class BranchStrategy(Enum):
    SHRINK = 1
    DFS = 2
    
class GreedyBranchStrategy(Enum):
    MAX_FIXED_OUT = 1
    MIN_FIXED_IN = 2


class TreeGreedy:

    def __init__(
            self,
            n: int,
            k: int,
            obj: Objective,
            lower_bounder: Bounder,
            lower_bounder_fixed_in_agnostic = False,
            branch_strategy: BranchStrategy = BranchStrategy.SHRINK,
            greedy_branch_strategy: GreedyBranchStrategy = GreedyBranchStrategy.MAX_FIXED_OUT,
            initial_upper_bound_strategy: UpperBounder = None,
            test_logger: TestLogger = None
            ) -> None:
        self.k: int = k
        self.n: int = n
        Node.k = k
        Node.n = n
        
        self.f0: Objective = obj
        self.phi: Bounder = lower_bounder
        self.phi_fixed_in_agnostic = lower_bounder_fixed_in_agnostic
        self.initial_upper_bounder = initial_upper_bound_strategy
        
        self.LB: float = -math.inf
        self.UB: float = math.inf
        self.unexplored_internal_nodes: List[Node] = []
        self.number_infeasible_nodes_explored: int = 0
        self.best_feasible_node: Node = None
        self.number_feasible_nodes_explored: int = 0

        self.branch_strategy: BranchStrategy = branch_strategy
        self.greedy_branch_strategy: GreedyBranchStrategy = greedy_branch_strategy
        
        ### Research Specific Objects ###
        self.known_fixed_in: Optional[List[int]] = None
        self.variable_scores: Optional[List[float]] = None

        self.test_logger = test_logger
        
        ### Framework Metrics ###
        self._status: Optional[str] = None
        self._value = None
        self._solution = None
        self.num_iter: int = 0
        self.UB_update_iterations: List[int] = []
        self.LB_update_iterations: List[int] = []
        self.solve_time: float = 0 # total enumeration time
        self.lb_bound_time: float = 0
        self.ub_bound_time: float = 0
        self.var_selection_time: float = 0
        self.initial_gap: float = math.inf
        self.initial_LB: float = -math.inf
        self.initial_UB: float = math.inf

        self.initial_tree_size = self._subtree_size(0,0)
        self.remaining_tree_size = self._subtree_size(0,0)

    @property
    def gap(self):
        if self.LB == -math.inf:
            return math.inf
        elif self.LB == 0:
            return (self.UB - self.LB)
        else:
            return (self.UB - self.LB) / self.LB
    
    @property
    def bound_time(self) -> float:
        return self.lb_bound_time + self.ub_bound_time
    
    def solve(self,
              eps: Number = 0.0001,
              timeout: Number = 60,
              fixed_in_vars: List[int] = None,
              fixed_out_vars: List[int] = None,
              #var_scores: List[float] = None,
              branch_strategy: str = "shrink",
              greedy_branch_strategy: str = "max_out"
              ) -> bool:
        """Enumerate a branch and bound tree to solve the logistic regression problem to global
        optimality using the bounding and objective functions passed into the tree upon its
        construction.

        Populates the :code:'status' and :code:'value' attributes on the
        tree object as a side-effect.

        Arguments
        ----------
        eps: positive float, optional
            The desired optimality tolerance.
            The default tolerance is 1e-8.
        timeout: float, optional
            The number of minutes solve will run before terminating.
            The default timeout is after 60 minutes.
        fixed_vars: List[int], optional
            Variable elements known to be fixed in (i.e. x[i] = 1 forall i in fixed_vars).
        branch_strategy: str, optional
            The method for choosing subproblems in the tree.
            Defaults to depth first search (dfs). Any other input
            will result in a lower bound shrinking strategy.
        
        Returns
        -------
        bool: Whether or not the problem was solved to global optimality.
        
        Raises
        ------
        AssertionError
            Raised if epsilon or timeout are not Numbers.
            Raised if var_scores doesn't contain enough scores.
        ValueError
            Raised if a fixed variable index is negative or >= n
        """
        
        ######### SETUP START #########
        
        start_time = time.time()
        loop_time = time.time() - start_time

        assert isinstance(eps, Number), "eps must be a Number"
        assert eps > 0, "eps must be positive."
        assert isinstance(timeout, Number), "timeout must be a Number"

        # keep track of these for metric purposes
        self.eps = eps
        self.timeout = timeout

        if greedy_branch_strategy != "max_out":
            self.greedy_branch_strategy = GreedyBranchStrategy.MIN_FIXED_IN

        if self.initial_upper_bounder != None:
            self.initial_UB, initial_ub_time, ub_fixed_in = self.initial_upper_bounder()
            self.UB = self.initial_UB
            initial_ub_node: Node = Node(ub_fixed_in, [])
            initial_ub_node.lb = self.initial_UB
            self.best_feasible_node = initial_ub_node
            self.number_feasible_nodes_explored += 1
            self.ub_bound_time += initial_ub_time
            print("Checking initial upper bound is feasible:", initial_ub_node.is_terminal_leaf)
            

        # check if there are fixed variables
        # create root node, gap, LB accordingly
        fixed_in_0, fixed_out_0 = [], []
        if fixed_in_vars != None:
            for i in range(len(fixed_in_vars)):
                assert isinstance(fixed_in_vars[i], int), "the fixed_in_vars list must contain integers."
            fixed_in_0 = deepcopy(fixed_in_vars)

        if fixed_out_vars != None:
            for i in range(len(fixed_out_vars)):
                assert isinstance(fixed_out_vars[i], int), "the fixed_out_vars list must contain integers."
            fixed_out_0 = deepcopy(fixed_out_vars)

        self._create_root_node(fixed_in_0, fixed_out_0)

        print(f"Setup complete | UB = {self.UB:.4f} | gap = {self.gap:.4f} | Open Subproblems: {len(self.unexplored_internal_nodes)}"
                + f" | Tree Remaining: {self.remaining_tree_size:,} | Running Time: {loop_time:.2f} seconds") 
        
        if self.test_logger != None:
            self.test_logger.log(0, time.time() - start_time, self.UB, self.LB, len(self.unexplored_internal_nodes), self.remaining_tree_size)

        ######### SETUP END #########

        
        ######### MAIN #########
        
        print("Gap greater than epsilon:", self.gap > eps)
        print("Timeout greater than loop time:", timeout > (loop_time / 60))

        while (self.gap > eps and timeout > (loop_time / 60)):
                       
            node: Node = self._choose_subproblem()

            # split problem handles updating UB and LB (if possible)
            # and handles adding the new subproblems to nodes and feasible_leaves
            self._dfs_problem(node)

            # Pruning
            unexplored_nodes = []
            for x in self.unexplored_internal_nodes:
                if ((self.UB - x.lb)  / x.lb) > eps:
                    unexplored_nodes.append(x)
                else:
                    self.remaining_tree_size -= self._subtree_size(len(x.fixed_in), len(x.fixed_out))
                    # print(len(x.fixed_in), len(x.fixed_out), self._subtree_size(len(x.fixed_in), len(x.fixed_out)))
            self.unexplored_internal_nodes = unexplored_nodes


            loop_time = time.time() - start_time

            if self.unexplored_internal_nodes:
                self.LB = min(min(self.unexplored_internal_nodes).lb, self.UB)
            else:
                self.LB = self.UB
            self.LB_update_iterations.append(self.num_iter)

            if (self.gap > eps and len(self.unexplored_internal_nodes) == 0):
                raise Exception("Node list is empty but GAP is unsatisfactory.")
            
            print(f"\033[KIteration {self.num_iter} | UB = {self.UB:.4f} | gap = {self.gap:.4f} | Open Subproblems: {len(self.unexplored_internal_nodes)}"
                + f" | Tree Remaining: {self.remaining_tree_size:,} | Running Time: {loop_time:.2f} seconds", end = "\r") 
            if self.test_logger != None:
                self.test_logger.log(self.num_iter, loop_time, self.UB, self.LB, len(self.unexplored_internal_nodes), self.remaining_tree_size)
        
        if (timeout < loop_time / 60):            
            self._status = "solve timed out."
            print("\nSolve timed out. Runtime:", time.time() - start_time)
            return False
        
        self._status = "global optimal found."
        self._value = self.UB
        self._solution = self.best_feasible_node
        self.solve_time = time.time() - start_time
        print("\nFound global optimal. Runtime:", self.solve_time)
        return True
    

    def _create_root_node(self, fixed_in, fixed_out):
        root_node: Node = Node(fixed_in, fixed_out)
        
        if root_node.is_terminal_leaf:
            root_node.lb, root_obj_time = self.phi(root_node)
            self.UB = root_node.lb
            self.ub_bound_time += root_obj_time
            self.UB_update_iterations.append(self.num_iter)
            self.best_feasible_node = root_node
            self.number_feasible_nodes_explored += 1
            self.remaining_tree_size = 0
        else:
            root_node.lb, root_obj_time = self.phi(root_node)
            self.unexplored_internal_nodes.append(root_node)

        self.LB = root_node.lb
        self.lb_bound_time += root_obj_time
        self.LB_update_iterations.append(self.num_iter)

        self.initial_gap = self.UB - self.LB

    
    def _choose_subproblem(self) -> Node:
        if self.branch_strategy == BranchStrategy.SHRINK:
            return self.unexplored_internal_nodes.pop(argmin(self.unexplored_internal_nodes))
        else:
            return self.unexplored_internal_nodes.pop()

    def _dfs_problem(self, node: Node):
        """
        For visualization purposes, assume "left" subproblem corresponds to selecting
        a variable while a "right" subproblem corresponds to discarding a variable.

        Don't need to handle if node is leaf since all leaves are passed into
        feasible solutions only.

        when adding see if pruning should take place
        or if leaf node conditions hold.

        For a chosen variable there are six possible conditions to consider
        WLOG, for x variable
        1. 
        2. fixed_in_full -> create right problem (make a terminal leaf)
        3. fixed_out_full -> create left subproblem (make a terminal leaf)
        4. internal node -> create two subproblems

        Note that the following conditions are handled prior to the if statements:
        1. WLOG node is_x_terminal_leaf -> ______________
        2. node is a terminal leaf node -> doesn't get added to L_k to begin with.
        
        """
        infeasible_node_list = []
        if Node.n - Node.k - len(node.fixed_out) > 1:
            for i in setdiff1d(list(range(self.n)), node.fixed_in + node.fixed_out):
                self.num_iter += 1
                new_node = Node(node.fixed_in, node.fixed_out + [i])
                new_node.lb, _ = self.f0(new_node)
                infeasible_node_list.append((new_node,i))
            additional_fixed_in = deepcopy(node.fixed_in)
            infeasible_node_list = sorted(infeasible_node_list, key=lambda tup: tup[0], reverse=True)[:Node.k - len(node.fixed_in)]
            for new_tup in infeasible_node_list:
                new_tup[0].fixed_in = deepcopy(additional_fixed_in)
                additional_fixed_in.append(new_tup[1])
            self.remaining_tree_size -= len(infeasible_node_list)
            feasible_node = Node(additional_fixed_in, [])
            if not feasible_node.is_terminal_leaf:
                raise ValueError(f"feasible node with fixed in length: {len(feasible_node.fixed_in)} is being called infeasible")
            feasible_node.lb, _ = self.f0(feasible_node)
            if self.UB > feasible_node.lb:
                self.UB = feasible_node.lb
                self.best_feasible_node = feasible_node
            self.remaining_tree_size -= 1
        elif Node.k - len(node.fixed_in) == 1:
            self.num_iter += 1
            for i in setdiff1d(list(range(self.n)), node.fixed_in + node.fixed_out):
                self.remaining_tree_size -= 1
                new_node = Node(node.fixed_in, node.fixed_out + [i])
                new_node.lb,_ = self.f0(new_node)
                if self.UB > feasible_node.lb:
                    self.UB = feasible_node.lb
                    self.best_feasible_node = feasible_node
        else:
            best_node = None
            best_var = None
            for i in setdiff1d(list(range(self.n)), node.fixed_in + node.fixed_out):
                new_node = Node(node.fixed_in, node.fixed_out + [i])
                new_node.lb,_ = self.f0(new_node)
                if best_node is None or new_node < best_node:
                    best_var = i
                    best_node = new_node
            if self.UB > best_node.lb:
                self.UB = best_node.lb
                self.best_feasible_node = best_node
            infeasible_node = Node(node.fixed_in +[best_var], node.fixed_out)
            infeasible_node.lb,_ = self.f0(infeasible_node)
            infeasible_node_list.append((infeasible_node,))
            self.num_iter += 1
            self.remaining_tree_size -= 2

        for tup in infeasible_node_list:
            if (self.UB - tup[0].lb) / tup[0].lb > self.eps:
                self.unexplored_internal_nodes.append(tup[0])
            else:
                self.remaining_tree_size -= self._subtree_size(len(tup[0].fixed_in), len(tup[0].fixed_out))
            

 

    def _subtree_size(self, fixed_in_len, fixed_out_len):
        # Returns the size of the subtree starting at the root node specified
        return 2 * math.comb(self.n - fixed_in_len - fixed_out_len, self.k - fixed_in_len) - 1