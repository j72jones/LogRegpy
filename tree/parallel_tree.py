from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from enum import Enum
from typing import List, Optional
from numbers import Number
from LogRegpy.utilities.bounding_func import Bounder
from LogRegpy.utilities.objective_func import Objective
from LogRegpy.utilities.variable_chooser import VariableChooser
from LogRegpy.utilities.upper_bounding_func import UpperBounder
from LogRegpy.tests.test_logger import TestLogger
from LogRegpy.tree.node import Node
from sortedcontainers import SortedList
from copy import deepcopy
from numpy import argmax, argmin
import math
import time
from numpy import (setdiff1d)

class BranchStrategy(Enum):
    SHRINK = 1
    DFS = 2
    
def node_lb_key(node):
    return node.lb


class ParallelTree:

    def __init__(
            self,
            n: int,
            k: int,
            obj: Objective,
            lower_bounder: Bounder,
            var_select: VariableChooser,
            number_of_branchers: int,
            lower_bounder_fixed_in_agnostic = False,
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
        self.phi_fixed_in_agnostic = lower_bounder_fixed_in_agnostic
        self.next_var = var_select
        self.initial_upper_bounder = initial_upper_bound_strategy
        self.brancher_count = number_of_branchers
        
        self.LB: float = -math.inf
        self.UB: float = math.inf
        self.unexplored_internal_nodes: SortedList[Node] = []
        self.best_infeasible_nodes: dict[int: Node] = {}
        self.number_infeasible_nodes_explored: int = 0
        self.best_feasible_node: Node = None
        self.number_feasible_nodes_explored: int = 0

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

        assert isinstance(eps, Number), "eps must be a Number"
        assert eps > 0, "eps must be positive."
        assert isinstance(timeout, Number), "timeout must be a Number"

        # keep track of these for metric purposes
        self.eps = eps
        self.timeout = timeout

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

        print(f"Setup complete | current UB = {self.UB} | current gap = {self.gap} | Number of Open Subproblems = {len(self.unexplored_internal_nodes)}"
                + f" | Total Running Time = {(time.time() - start_time):.3f} seconds") 
        
        if self.test_logger != None:
            self.test_logger.log(0, time.time() - start_time, self.UB, self.LB, len(self.unexplored_internal_nodes))

        ######### SETUP END #########

        
        ######### MAIN #########
        
        print("Gap greater than epsilon:", self.gap > eps)
        print("Timeout greater than loop time:", timeout > (loop_time / 60))


        with ProcessPoolExecutor(max_workers=self.brancher_count) as executor:
            futures = {}
            for _ in range(self.brancher_count):
                if len(self.unexplored_internal_nodes) == 0:
                    break
                node = self._choose_subproblem()
                futures[executor.submit(self._split_problem, node)] = node

            while len(futures) != 0:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    left_node, right_node = future.result()

                    if left_node.is_terminal_leaf:
                        if left_node.lb < self.UB:
                            self.best_feasible_node = left_node
                            self.UB = left_node.lb
                            self.UB_update_iterations.append(self.num_iter)
                    else:
                        if left_node.lb < self.UB:
                            self.unexplored_internal_nodes.append(left_node)

                    if right_node.is_terminal_leaf:
                        if right_node.lb < self.UB:
                            self.best_feasible_node = right_node
                            self.UB = right_node.lb
                            self.UB_update_iterations.append(self.num_iter)
                    else:
                        if right_node.lb < self.UB:
                            self.unexplored_internal_nodes.append(right_node)

                    if len(self.unexplored_internal_nodes) != 0:
                        node = self._choose_subproblem()
                        futures[executor.submit(self._split_problem, node)] = node

                    if self.UB_update_iterations and self.UB_update_iterations[-1] == self.num_iter:
                        self.unexplored_internal_nodes = [n for n in self.unexplored_internal_nodes if n.lb < self.UB]  # O(i)

                    del futures[future]

                    lb_list = [self.UB] + [node.lb for node in futures.values()]
                    if self.unexplored_internal_nodes:
                        lb_list.append(min(self.unexplored_internal_nodes).lb)
                    self.LB = min(lb_list)

                    loop_time = time.time() - start_time
                
                    print(f"\033[KIteration {self.num_iter} | current UB = {self.UB} | current gap = {self.gap} | Number of Open Subproblems = {len(self.unexplored_internal_nodes)}"
                        + f" | Total Running Time = {loop_time} seconds", end = "\r") 
                    if self.test_logger != None:
                        self.test_logger.log(self.num_iter, loop_time, self.UB, self.LB, len(self.unexplored_internal_nodes))

                    if (timeout < loop_time / 60):            
                        self._status = "solve timed out."
                        print("\nSolve timed out. Runtime:", time.time() - start_time)
                        return False

                    if self.gap <= eps:
                        self._status = "global optimal found."
                        self._value = self.UB
                        self._solution = self.best_feasible_node
                        self.solve_time = time.time() - start_time
                        print("\nFound global optimal. Runtime:", self.solve_time)
                        return True
                    
                    self.num_iter += 1


        if (self.gap > eps and len(self.unexplored_internal_nodes) == 0):
            raise Exception("Node list is empty but GAP is unsatisfactory.")
    

    def _create_root_node(self, fixed_in, fixed_out):
        root_node: Node = Node(fixed_in, fixed_out)
        
        if root_node.is_terminal_leaf:
            root_node.lb, root_obj_time = self.phi(root_node)
            self.UB = root_node.lb
            self.ub_bound_time += root_obj_time
            self.UB_update_iterations.append(self.num_iter)
            self.best_feasible_node = root_node
            self.number_feasible_nodes_explored += 1
        else:
            root_node.lb, root_obj_time = self.phi(root_node)
            self.unexplored_internal_nodes.append(root_node)
            self.best_infeasible_nodes[len(fixed_out)] = root_node

        self.LB = root_node.lb
        self.lb_bound_time += root_obj_time
        self.LB_update_iterations.append(self.num_iter)

        self.initial_gap = self.UB - self.LB

    
    def _choose_subproblem(self) -> Node:
        if self.branch_strategy == BranchStrategy.SHRINK:
            return self.unexplored_internal_nodes.pop(argmin(self.unexplored_internal_nodes))
        else:
            return self.unexplored_internal_nodes.pop()

