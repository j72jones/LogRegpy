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
    

class Tree:

    def __init__(
            self,
            n: int,
            k: int,
            obj: Objective,
            lower_bounder: Bounder,
            var_select: VariableChooser,
            branch_strategy: BranchStrategy = BranchStrategy.SHRINK,
            initial_upper_bound_strategy: UpperBounder = None,
            test_logger: TestLogger = None
            ) -> None:
        self.k: int = k
        self.n: int = n
        Node.k = k
        Node.n = n
        
        self.f0: Objective = obj
        self.phi: Bounder = lower_bounder
        self.next_var = var_select
        self.initial_upper_bounder = initial_upper_bound_strategy
        
        self.LB: float = -math.inf
        self.UB: float = math.inf
        self.unexplored_internal_nodes: List[Node] = []
        self.explored_internal_nodes: List[Node] = []
        self.feasible_leaves: List[Node] = []

        self.branch_strategy: BranchStrategy = branch_strategy
        
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

    @property
    def gap(self):
        return self.UB - self.LB
    
    @property
    def bound_time(self) -> float:
        return self.lb_bound_time + self.ub_bound_time
    
    def solve(self,
              eps: Number = 1e-8,
              timeout: Number = 60,
              fixed_in_vars: List[int] = None,
              fixed_out_vars: List[int] = None,
              #var_scores: List[float] = None,
              branch_strategy: str = "shrink"
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

        if branch_strategy == "dfs":
            self.branch_strategy: BranchStrategy = BranchStrategy.DFS

        assert isinstance(eps, Number), "eps must be a Number"
        assert eps > 0, "eps must be positive."
        assert isinstance(timeout, Number), "timeout must be a Number"

        # keep track of these for metric purposes
        self.eps = eps
        self.timeout = timeout

        # # check if there are variable scores
        # if var_scores != None:
        #     assert len(var_scores) == self.n, "there must be variable scores for all variables"
        #     self.variable_scores = deepcopy(var_scores)

        if self.initial_upper_bounder != None:
            self.initial_UB, initial_ub_time, ub_fixed_in = self.initial_upper_bounder()
            self.UB = self.initial_UB
            initial_ub_node: Node = Node(ub_fixed_in, [])
            initial_ub_node.lb = self.initial_UB
            self.feasible_leaves.append(initial_ub_node)
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

        print(f"Setup complete | current UB = {self.UB} | current gap = {self.gap} | Number of Open Subproblems = {len(self.unexplored_internal_nodes)}"
                + f" | Total Running Time = {(time.time() - start_time):.3f} seconds") 
        
        if self.test_logger != None:
            self.test_logger.log(0, time.time() - start_time, self.UB, self.LB, len(self.unexplored_internal_nodes))

        ######### SETUP END #########

        
        ######### MAIN #########
        
        print("Gap greater than epsilon:", self.gap > eps)
        print("Timeout greater than loop time:", timeout > (loop_time / 60))

        while (self.gap > eps and timeout > (loop_time / 60)):
            # Pruning
            if self.UB_update_iterations and self.UB_update_iterations[-1] == self.num_iter:
                self.unexplored_internal_nodes = [x for x in self.unexplored_internal_nodes if x.lb < self.UB]

            self.num_iter += 1
            node: Node = self._choose_subproblem()

            # split problem handles updating UB and LB (if possible)
            # and handles adding the new subproblems to nodes and feasible_leaves
            self._split_problem(node)

            loop_time = time.time() - start_time

            if (self.gap > eps and len(self.unexplored_internal_nodes) == 0):
                raise Exception("Node list is empty but GAP is unsatisfactory.")
            
            print(f"\033[KIteration {self.num_iter} | current UB = {self.UB} | current gap = {self.gap} | Number of Open Subproblems = {len(self.unexplored_internal_nodes)}"
                + f" | Total Running Time = {loop_time} seconds", end = "\r") 
            if self.test_logger != None:
                self.test_logger.log(self.num_iter, loop_time, self.UB, self.LB, len(self.unexplored_internal_nodes))
        
        if (timeout < loop_time / 60):            
            self._status = "solve timed out."
            print("\nSolve timed out. Runtime:", time.time() - start_time)
            return False
        
        self._status = "global optimal found."
        self._value = self.UB
        self._solution = min(self.feasible_leaves)
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
            self.feasible_leaves.append(root_node)
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

    def _split_problem(self, node: Node):
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

        chosen_var, var_choice_time = self.next_var(node)
        self.var_selection_time += var_choice_time

        if chosen_var is not None:
            self._create_left_subproblem(node, chosen_var)
            self._create_right_subproblem(node, chosen_var)
        else:
            raise Exception("Branching code ran into an unexpected case") # TODO: add information here, i.e. print something
        
        # lb_node = min(self.unexplored_internal_nodes + self.feasible_leaves)
        # if lb_node.lb < self.LB:
        #     print("\n\nPROBLEM:", lb_node.fixed_out, self.lb_node.fixed_out)
        
        # self.lb_node = lb_node

        # self.LB = self.lb_node.lb

        self.LB = min(self.unexplored_internal_nodes + self.feasible_leaves).lb
        self.LB_update_iterations.append(self.num_iter)


    def _create_left_subproblem(self, node: Node, branch_idx: int) -> None:
        """
        fixes in:
        - adds the new index to fixed_in
        - creates corresponding node
        """
        new_fixed_in = deepcopy(node.fixed_in) + [branch_idx]
        new_subproblem: Node = Node(new_fixed_in, node.fixed_out)
        
        self._evaluate_node(new_subproblem)
    

    def _create_right_subproblem(self, node: Node, branch_idx: int) -> None:
        """
        fixes out:
        - adds the new index to fixed_out
        - creates corresponding node
        """
        new_fixed_out = deepcopy(node.fixed_out) + [branch_idx]
        new_subproblem: Node = Node(node.fixed_in, new_fixed_out)
        
        self._evaluate_node(new_subproblem)

    def _evaluate_node(self, node: Node) -> None:
        # note that if you want generalilzation of bounding/obj functions then they need
        # to return bound val, bounding time respectively.

        if node.is_terminal_leaf:
            node.lb, bound_time = self.f0(node)
            self.ub_bound_time += bound_time
            self.feasible_leaves.append(node)
            if node.lb < self.UB:
                self.UB = node.lb
                self.UB_update_iterations.append(self.num_iter)
        else:
            node.lb, bound_time = self.phi(node)
            self.lb_bound_time += bound_time
            if node.lb < self.UB:
                self.unexplored_internal_nodes.append(node)